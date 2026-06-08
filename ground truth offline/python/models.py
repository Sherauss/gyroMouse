"""
models.py — Physical System Models for Multi-IMU Sensor Fusion

Defines the process model (state propagation) and measurement model
(predicted IMU readings) for a rigid body instrumented with 3 IMUs
at known lever-arm positions.

System Overview
---------------
We model a rigid body (e.g., a beam or drone) with 3 MPU6050 IMUs
mounted at different locations along the body x-axis:

    [Head IMU]----[Mid IMU]----[Butt IMU]
     +0.08 m       0.00 m      -0.08 m

Each IMU measures 3-axis acceleration and 3-axis angular velocity,
giving 18 total measurements per timestep.

State Vector (13 elements)
--------------------------
    x = [q0, q1, q2, q3,  ωx, ωy, ωz,  αx, αy, αz,  bx, by, bz]
         ├─── quaternion ──┤  ├── ang vel ──┤  ├── ang acc ──┤  ├── gyro bias ──┤
         (scalar-first)       (rad/s)         (rad/s²)        (rad/s)

    - Quaternion [q0..q3]: Orientation from body to world frame
    - Angular velocity [ωx, ωy, ωz]: Body-frame angular rates
    - Angular acceleration [αx, αy, αz]: Body-frame angular accelerations
    - Gyroscope bias [bx, by, bz]: Slowly-varying sensor bias

Physical Constants
------------------
These MUST match the MATLAB synthetic data generator exactly.

Author: Senior Project — Multi-IMU UKF Sensor Fusion
"""

import numpy as np
from quaternion_utils import (
    quat_multiply,
    quat_normalize,
    quat_from_axis_angle,
    quat_to_rotation_matrix
)


# ────────────────────────────────────────────────────────────────────
# Physical Constants (must match MATLAB exactly!)
# ────────────────────────────────────────────────────────────────────

G = 9.81  # Gravitational acceleration [m/s²]

# Lever-arm positions from the Center of Mass (COM) in body frame [m]
# These define where each IMU is mounted on the rigid body.
R_HEAD = np.array([0.08, 0.0, 0.0])   # Head IMU: +8 cm along x-axis
R_MID  = np.array([0.0,  0.0, 0.0])   # Mid  IMU: at the COM
R_BUTT = np.array([-0.08, 0.0, 0.0])  # Butt IMU: -8 cm along x-axis

# List of all lever arms (ordered: head, mid, butt)
LEVER_ARMS = [R_HEAD, R_MID, R_BUTT]

# Sampling parameters
FS = 200     # Sampling frequency [Hz]
DT = 1 / FS  # Time step [s] = 0.005

# MPU6050 noise parameters
SIGMA_GYRO  = 0.0012  # Gyroscope noise std [rad/s]
SIGMA_ACCEL = 0.056   # Accelerometer noise std [m/s²]


# ────────────────────────────────────────────────────────────────────
# Process Model
# ────────────────────────────────────────────────────────────────────

def process_model(x, dt):
    """Propagate the state vector forward by one time step.

    This implements the discrete-time state transition:

    1. **Quaternion update**: Integrate angular velocity using the
       axis-angle representation:
           θ = ||ω|| × dt
           axis = ω / ||ω||
           dq = quat_from_axis_angle(axis, θ)
           q_new = q ⊗ dq

    2. **Angular velocity update**: First-order Euler integration:
           ω_new = ω + α × dt

    3. **Angular acceleration**: Modeled as constant (random walk).
           α_new = α

    4. **Gyroscope bias**: Modeled as constant (slow random walk).
           b_new = b

    Parameters
    ----------
    x : ndarray, shape (13,)
        Current state vector:
        [q0, q1, q2, q3, ωx, ωy, ωz, αx, αy, αz, bx, by, bz]
    dt : float
        Time step in seconds.

    Returns
    -------
    x_new : ndarray, shape (13,)
        Predicted state at t + dt.

    Notes
    -----
    The quaternion integration uses the exact axis-angle method rather
    than the linearized approximation q + 0.5*Ω*q*dt, which provides
    better accuracy for larger angular rates.
    """
    # Unpack state components
    q     = x[0:4]    # quaternion [w, x, y, z]
    omega = x[4:7]    # angular velocity [rad/s]
    alpha = x[7:10]   # angular acceleration [rad/s²]
    bias  = x[10:13]  # gyroscope bias [rad/s]

    # ── Step 1: Quaternion update via axis-angle integration ──
    # The rotation angle over the timestep
    omega_norm = np.linalg.norm(omega)
    angle = omega_norm * dt

    if angle > 1e-10:
        # Non-trivial rotation: compute incremental quaternion
        axis = omega / omega_norm
        dq = quat_from_axis_angle(axis, angle)
        # Body-frame integration: q_new = q ⊗ dq
        # (dq is in body frame, so it's applied on the right)
        q_new = quat_multiply(q, dq)
    else:
        # Angular velocity is essentially zero — no rotation
        q_new = q.copy()

    # Re-normalize to maintain unit quaternion constraint
    q_new = quat_normalize(q_new)

    # ── Step 2: Angular velocity update (Euler integration) ──
    omega_new = omega + alpha * dt

    # ── Step 3 & 4: Constant models (random walk) ──
    alpha_new = alpha.copy()
    bias_new  = bias.copy()

    return np.concatenate([q_new, omega_new, alpha_new, bias_new])


# ────────────────────────────────────────────────────────────────────
# Measurement Model
# ────────────────────────────────────────────────────────────────────

def measurement_model(x):
    """Predict what all 3 IMUs would measure given the current state.

    Each IMU at lever-arm position r_i from the COM measures:

    **Accelerometer** (in body frame):
        a_i = R^T × g_world + α × r_i + ω × (ω × r_i)
              ├─ gravity ─┤   ├─ Euler ─┤   ├── centripetal ──┤

        where:
        - R^T × g_world projects world-frame gravity into body frame
        - α × r_i is the tangential (Euler) acceleration from lever arm
        - ω × (ω × r_i) is the centripetal acceleration from lever arm

    **Gyroscope** (in body frame):
        g_i = ω + b
        All IMUs measure the same angular velocity (plus bias), since
        the body is rigid.

    Parameters
    ----------
    x : ndarray, shape (13,)
        State vector:
        [q0, q1, q2, q3, ωx, ωy, ωz, αx, αy, αz, bx, by, bz]

    Returns
    -------
    z : ndarray, shape (18,)
        Predicted measurements, ordered as:
        [ax1, ay1, az1, gx1, gy1, gz1,   ← Head IMU
         ax2, ay2, az2, gx2, gy2, gz2,   ← Mid  IMU
         ax3, ay3, az3, gx3, gy3, gz3]   ← Butt IMU

    Notes
    -----
    For the mid IMU (r = [0,0,0]), the lever-arm terms vanish and
    the accelerometer reads only the gravitational component.
    """
    # Unpack state components
    q     = x[0:4]
    omega = x[4:7]
    alpha = x[7:10]
    bias  = x[10:13]

    # Rotation matrix: body → world (and R^T: world → body)
    R = quat_to_rotation_matrix(q)

    # Gravity vector in world frame: [0, 0, +g]
    # (Z-axis points up in world frame)
    g_world = np.array([0.0, 0.0, G])

    # Project gravity into body frame
    gravity_body = R.T @ g_world

    # Build the 18-element measurement vector
    z = np.zeros(18)

    for i, r in enumerate(LEVER_ARMS):
        # ── Accelerometer prediction ──
        # Tangential (Euler) acceleration: α × r
        a_euler = np.cross(alpha, r)

        # Centripetal acceleration: ω × (ω × r)
        a_centripetal = np.cross(omega, np.cross(omega, r))

        # Total acceleration at this IMU location (body frame)
        accel = gravity_body + a_euler + a_centripetal

        # ── Gyroscope prediction ──
        # All IMUs measure the same angular velocity (rigid body)
        # plus the common gyroscope bias
        gyro = omega + bias

        # Pack into output vector
        z[i*6 : i*6 + 3] = accel
        z[i*6 + 3 : i*6 + 6] = gyro

    return z


# ────────────────────────────────────────────────────────────────────
# Self-Test
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Physical Models — Self-Test")
    print("=" * 60)

    # Test with identity orientation, zero angular velocity
    x_test = np.zeros(13)
    x_test[0] = 1.0  # identity quaternion

    z = measurement_model(x_test)
    print(f"\n[Test 1] Static, upright orientation:")
    print(f"  Head IMU accel: {z[0:3]}  (expected: [0, 0, 9.81])")
    print(f"  Mid  IMU accel: {z[6:9]}  (expected: [0, 0, 9.81])")
    print(f"  Butt IMU accel: {z[12:15]} (expected: [0, 0, 9.81])")
    print(f"  All gyros zero: {np.allclose(z[[3,4,5,9,10,11,15,16,17]], 0)}")

    # Test process model — state should not change much with zero angular velocity
    x_next = process_model(x_test, DT)
    print(f"\n[Test 2] Process model with zero angular velocity:")
    print(f"  Quaternion unchanged: {np.allclose(x_next[0:4], [1, 0, 0, 0])}")
    print(f"  State unchanged: {np.allclose(x_next, x_test)}")

    # Test with angular velocity about z-axis
    x_spin = np.zeros(13)
    x_spin[0] = 1.0
    x_spin[6] = 1.0  # ωz = 1 rad/s
    x_next = process_model(x_spin, DT)
    print(f"\n[Test 3] Spinning about Z at 1 rad/s:")
    print(f"  New quaternion: {x_next[0:4]}")
    print(f"  Angular velocity preserved: {np.allclose(x_next[4:7], [0, 0, 1])}")

    print("\n" + "=" * 60)
    print("Models self-test complete.")
    print("=" * 60)
