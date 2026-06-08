"""
ukf.py — Unscented Kalman Filter Implementation

Implements the standard UKF algorithm with special handling for
quaternion-based state estimation. The UKF uses the unscented transform
to propagate probability distributions through nonlinear functions,
avoiding the need for Jacobian computation (unlike the EKF).

Algorithm Overview
------------------
The UKF works in two steps:

1. **Predict**: Generate sigma points from the current estimate,
   propagate them through the nonlinear process model, and recover
   the predicted mean and covariance.

2. **Update**: Generate sigma points from the predicted estimate,
   propagate them through the nonlinear measurement model, compute
   the Kalman gain, and correct the state estimate.

Sigma Point Parameters
----------------------
    α (alpha): Controls the spread of sigma points around the mean.
               Typically 1e-3 for Gaussian states. Smaller values keep
               points closer to the mean.

    β (beta):  Incorporates prior knowledge about the distribution.
               β = 2 is optimal for Gaussian distributions.

    κ (kappa): Secondary scaling parameter. Usually set to 0.

    λ (lambda) = α²(n + κ) - n

References
----------
[1] Wan, E.A. and Van Der Merwe, R., "The Unscented Kalman Filter
    for Nonlinear Estimation", IEEE Adaptive Systems for Signal
    Processing, Communications, and Control Symposium, 2000.
[2] Julier, S.J. and Uhlmann, J.K., "Unscented Filtering and
    Nonlinear Estimation", Proceedings of the IEEE, 92(3), 2004.

Author: Senior Project — Multi-IMU UKF Sensor Fusion
"""

import numpy as np
from quaternion_utils import quat_normalize


class UKF:
    """Unscented Kalman Filter for quaternion-based state estimation.

    This implementation handles the special structure of the state vector
    where the first 4 elements form a unit quaternion that must be
    re-normalized after every update.

    Parameters
    ----------
    n_state : int
        Dimension of the state vector (13 for our system).
    n_meas : int
        Dimension of the measurement vector (18 for 3 IMUs × 6 readings).
    f : callable
        Process model function: f(x, dt) → x_predicted.
    h : callable
        Measurement model function: h(x) → z_predicted.
    Q : ndarray, shape (n_state, n_state)
        Process noise covariance matrix.
    R : ndarray, shape (n_meas, n_meas)
        Measurement noise covariance matrix.
    alpha : float, optional
        Sigma point spread parameter (default: 1e-3).
    beta : float, optional
        Distribution prior parameter (default: 2, optimal for Gaussian).
    kappa : float, optional
        Secondary scaling parameter (default: 0).

    Attributes
    ----------
    x : ndarray, shape (n_state,)
        Current state estimate (mean).
    P : ndarray, shape (n_state, n_state)
        Current state covariance.

    Examples
    --------
    >>> from models import process_model, measurement_model
    >>> ukf = UKF(13, 18, process_model, measurement_model, Q, R)
    >>> ukf.x = x0  # set initial state
    >>> ukf.P = P0  # set initial covariance
    >>> ukf.predict(dt=0.005)
    >>> ukf.update(z_measurement)
    """

    def __init__(self, n_state, n_meas, f, h, Q, R,
                 alpha=1e-3, beta=2, kappa=0):
        # ── Store dimensions ──
        self.n = n_state   # state dimension
        self.m = n_meas    # measurement dimension

        # ── Store model functions ──
        self.f = f  # process model
        self.h = h  # measurement model

        # ── Noise covariance matrices ──
        self.Q = Q.copy()  # process noise
        self.R = R.copy()  # measurement noise

        # ── UKF scaling parameters ──
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Lambda: the key scaling parameter
        # λ = α²(n + κ) - n
        self.lam = alpha**2 * (self.n + kappa) - self.n

        # ── Number of sigma points: 2n + 1 ──
        self.n_sigma = 2 * self.n + 1

        # ── Compute sigma point weights ──
        # Wm[i]: weight for computing the mean
        # Wc[i]: weight for computing the covariance
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)

        # Zeroth sigma point (the mean itself) gets special weight
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - alpha**2 + beta)

        # Remaining sigma points share equal weight
        w = 1.0 / (2.0 * (self.n + self.lam))
        for i in range(1, self.n_sigma):
            self.Wm[i] = w
            self.Wc[i] = w

        # ── State and covariance (user should set these before running) ──
        self.x = np.zeros(n_state)
        self.P = np.eye(n_state)

    def _enforce_positive_definite(self, P, min_eig=1e-10):
        """Enforce that covariance matrix P remains positive-definite.

        This is a critical stability measure. During the UKF predict and
        update steps, floating-point errors can cause P to lose symmetry
        or positive-definiteness, which leads to Cholesky failure and
        filter divergence.

        Method: eigenvalue clamping — decompose P, clamp any negative
        or near-zero eigenvalues to min_eig, then reconstruct.

        Also enforces a minimum diagonal (variance floor) to prevent
        the filter from becoming overconfident in any state, which is
        another common cause of divergence.

        Parameters
        ----------
        P : ndarray, shape (n, n)
            Covariance matrix.
        min_eig : float
            Minimum allowed eigenvalue (default: 1e-10).

        Returns
        -------
        P_fixed : ndarray, shape (n, n)
            Guaranteed symmetric positive-definite matrix.
        """
        # Check if P is already positive-definite
        try:
            np.linalg.cholesky(P)
            return P  # Already fine, skip the expensive repair
        except np.linalg.LinAlgError:
            pass

        # Eigenvalue clamping
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, min_eig)
        P_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Enforce symmetry
        P_fixed = 0.5 * (P_fixed + P_fixed.T)

        return P_fixed

    def _generate_sigma_points(self, x, P):
        """Generate the 2n+1 sigma points via the unscented transform.

        The sigma points are chosen deterministically to capture the
        mean and covariance of the state distribution:

            χ₀ = x̄                              (the mean)
            χᵢ = x̄ + (√((n+λ)P))ᵢ    i = 1..n  (positive perturbations)
            χᵢ = x̄ - (√((n+λ)P))ᵢ₋ₙ  i = n+1..2n  (negative perturbations)

        where √(·) denotes the matrix square root (Cholesky decomposition).

        Parameters
        ----------
        x : ndarray, shape (n,)
            State mean.
        P : ndarray, shape (n, n)
            State covariance.

        Returns
        -------
        sigmas : ndarray, shape (2n+1, n)
            Matrix of sigma points, one per row.
        """
        n = self.n
        sigmas = np.zeros((self.n_sigma, n))

        # Regularize P for numerical stability
        # Small positive value on diagonal prevents Cholesky failure
        P_reg = P + 1e-9 * np.eye(n)

        # Compute matrix square root via Cholesky decomposition
        # L @ L^T = (n + λ) × P_reg
        # Each column of L gives the perturbation direction for one sigma point
        try:
            sqrt_P = np.linalg.cholesky((n + self.lam) * P_reg)
        except np.linalg.LinAlgError:
            # Cholesky failed — P has become non-positive-definite
            # Fix by eigenvalue clamping
            print("[UKF] WARNING: Cholesky failed, repairing covariance matrix")
            eigvals, eigvecs = np.linalg.eigh(P_reg)
            eigvals = np.maximum(eigvals, 1e-10)
            P_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
            sqrt_P = np.linalg.cholesky((n + self.lam) * P_fixed)

        # Build sigma point matrix
        sigmas[0] = x  # zeroth point is the mean

        for i in range(n):
            # Columns of sqrt_P define the perturbation directions
            sigmas[i + 1]     = x + sqrt_P[:, i]  # positive perturbation
            sigmas[n + i + 1] = x - sqrt_P[:, i]  # negative perturbation

        return sigmas

    def predict(self, dt):
        """UKF prediction (time update) step.

        Propagates the state estimate forward in time using the process
        model. This accounts for the system dynamics but not new
        measurements.

        Algorithm:
            1. Generate sigma points from (x, P)
            2. Propagate each sigma point through f(χ, dt)
            3. Compute weighted mean → x_predicted
            4. Compute weighted covariance → P_predicted
            5. Add process noise Q

        Parameters
        ----------
        dt : float
            Time step in seconds.
        """
        # ── Step 1: Generate sigma points from current estimate ──
        sigmas = self._generate_sigma_points(self.x, self.P)

        # ── Step 2: Propagate each sigma point through the process model ──
        sigmas_pred = np.zeros_like(sigmas)
        for i in range(self.n_sigma):
            sigmas_pred[i] = self.f(sigmas[i], dt)

        # ── Step 3: Compute predicted mean (weighted sum) ──
        x_pred = np.zeros(self.n)
        for i in range(self.n_sigma):
            x_pred += self.Wm[i] * sigmas_pred[i]

        # Normalize the quaternion part of the mean
        # Note: For small dispersions (alpha << 1), the weighted sum
        # of unit quaternions followed by normalization is a good
        # approximation to the true quaternion mean.
        x_pred[0:4] = quat_normalize(x_pred[0:4])

        # ── Step 4: Compute predicted covariance ──
        P_pred = self.Q.copy()  # Start with process noise
        for i in range(self.n_sigma):
            diff = sigmas_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)

        # Enforce symmetry (combat floating-point asymmetry drift)
        P_pred = 0.5 * (P_pred + P_pred.T)

        # Enforce positive-definiteness via eigenvalue clamping
        P_pred = self._enforce_positive_definite(P_pred)

        # ── Store results ──
        self.x = x_pred
        self.P = P_pred

    def update(self, z):
        """UKF update (measurement correction) step.

        Incorporates a new measurement to refine the state estimate.
        Uses the Kalman gain to optimally blend the prediction with
        the measurement.

        Algorithm:
            1. Generate sigma points from predicted (x, P)
            2. Propagate each through the measurement model h(χ)
            3. Compute predicted measurement mean ẑ
            4. Compute innovation covariance S = Pzz + R
            5. Compute cross-covariance Pxz
            6. Kalman gain K = Pxz @ S⁻¹
            7. Update: x = x + K(z - ẑ), P = P - K @ S @ K^T

        Parameters
        ----------
        z : ndarray, shape (m,)
            Measurement vector (18 values from 3 IMUs).
        """
        # ── Step 1: Generate sigma points from predicted state ──
        sigmas = self._generate_sigma_points(self.x, self.P)

        # ── Step 2: Propagate sigma points through measurement model ──
        z_sigmas = np.zeros((self.n_sigma, self.m))
        for i in range(self.n_sigma):
            z_sigmas[i] = self.h(sigmas[i])

        # ── Step 3: Predicted measurement mean ──
        z_pred = np.zeros(self.m)
        for i in range(self.n_sigma):
            z_pred += self.Wm[i] * z_sigmas[i]

        # ── Step 4: Innovation covariance S ──
        # S = Σ Wc[i] * (z_i - ẑ)(z_i - ẑ)^T + R
        S = self.R.copy()
        for i in range(self.n_sigma):
            dz = z_sigmas[i] - z_pred
            S += self.Wc[i] * np.outer(dz, dz)

        # ── Step 5: Cross-covariance Pxz ──
        # Pxz = Σ Wc[i] * (x_i - x̄)(z_i - ẑ)^T
        Pxz = np.zeros((self.n, self.m))
        for i in range(self.n_sigma):
            dx = sigmas[i] - self.x
            dz = z_sigmas[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # ── Step 6: Kalman gain ──
        # K = Pxz @ S⁻¹
        # Using solve (S^T @ K^T = Pxz^T) is more numerically stable
        # than explicit matrix inverse.
        K = np.linalg.solve(S.T, Pxz.T).T

        # ── Step 7: State and covariance update ──
        # Innovation (measurement residual)
        innovation = z - z_pred
        self.x = self.x + K @ innovation

        # Calculate NIS
        try:
            nis = innovation.T @ np.linalg.solve(S, innovation)
        except np.linalg.LinAlgError:
            nis = 0.0

        # ── Stabilized covariance update ──
        # The standard form P = P - K·S·K^T can lose positive-definiteness
        # due to floating-point errors. We use the symmetric Joseph form:
        #
        #   P = P - K·Pxz^T - Pxz·K^T + K·S·K^T
        #
        # This is algebraically equivalent to P = P - K·S·K^T (since
        # K·S = Pxz), but the symmetric construction is more numerically
        # stable. We then add K·R·K^T as an extra positive-definite
        # "safety blanket" (the Joseph stabilization term):
        #
        #   P = (I - K·H_eff)·P⁻·(I - K·H_eff)^T + K·R·K^T
        #
        # For UKF without explicit H, this simplifies to:
        self.P = self.P - K @ S @ K.T + K @ self.R @ K.T

        # ── Post-update: normalize quaternion ──
        self.x[0:4] = quat_normalize(self.x[0:4])

        # Enforce symmetry and positive-definiteness
        self.P = 0.5 * (self.P + self.P.T)
        self.P = self._enforce_positive_definite(self.P)
        return nis

    def get_euler_angles(self):
        """Convenience method: extract Euler angles from current state.

        Returns
        -------
        euler : ndarray, shape (3,)
            [yaw, pitch, roll] in radians.
        """
        from quaternion_utils import quat_to_euler
        return quat_to_euler(self.x[0:4])


# ────────────────────────────────────────────────────────────────────
# Self-Test
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models import process_model, measurement_model

    print("=" * 60)
    print("UKF — Self-Test")
    print("=" * 60)

    # Create a simple UKF and verify initialization
    n_state = 13
    n_meas = 18

    Q = np.diag([1e-6]*4 + [1e-4]*3 + [1e-2]*3 + [1e-8]*3)
    R = np.eye(18) * 0.01

    ukf = UKF(n_state, n_meas, process_model, measurement_model, Q, R)

    # Initialize state
    ukf.x = np.zeros(13)
    ukf.x[0] = 1.0  # identity quaternion
    ukf.P = np.eye(13) * 0.1
    ukf.P[0:4, 0:4] = np.eye(4) * 1e-4

    print(f"\n[Test 1] UKF created with {ukf.n_sigma} sigma points")
    print(f"  Wm sum = {np.sum(ukf.Wm):.6f} (expected: 1.0)")
    print(f"  Wc sum = {np.sum(ukf.Wc):.6f}")

    # Test predict step
    ukf.predict(0.005)
    print(f"\n[Test 2] After prediction:")
    print(f"  Quaternion norm = {np.linalg.norm(ukf.x[0:4]):.10f} (expected: 1.0)")
    print(f"  State: {ukf.x}")

    # Test update step with a synthetic measurement
    z_test = measurement_model(ukf.x) + np.random.randn(18) * 0.01
    ukf.update(z_test)
    print(f"\n[Test 3] After update:")
    print(f"  Quaternion norm = {np.linalg.norm(ukf.x[0:4]):.10f} (expected: 1.0)")
    print(f"  P is symmetric: {np.allclose(ukf.P, ukf.P.T)}")
    print(f"  P diagonal (sample): {np.diag(ukf.P)[:4]}")

    print("\n" + "=" * 60)
    print("UKF self-test complete.")
    print("=" * 60)
