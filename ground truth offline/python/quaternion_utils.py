"""
quaternion_utils.py — Quaternion Mathematics Library

Pure NumPy implementations of quaternion operations for rigid-body
orientation estimation. All quaternions use the scalar-first convention:

    q = [q0, q1, q2, q3] = [w, x, y, z]

where q0 (w) is the scalar part and [q1, q2, q3] (x, y, z) is the
vector part. The Hamilton product convention is used throughout.

Convention Notes
----------------
- A unit quaternion q represents a rotation from frame A to frame B.
- The rotation matrix R = quat_to_rotation_matrix(q) rotates vectors
  from body frame to world frame: v_world = R @ v_body.
- Hamilton product: q1 * q2 means "first apply q1, then apply q2"
  when rotating vectors via q * v * q_conj.

References
----------
[1] Kuipers, J.B., "Quaternions and Rotation Sequences", Princeton, 1999.
[2] Diebel, J., "Representing Attitude: Euler Angles, Unit Quaternions,
    and Rotation Vectors", Stanford University, 2006.

Author: Senior Project — Multi-IMU UKF Sensor Fusion
"""

import numpy as np


# ────────────────────────────────────────────────────────────────────
# Core Quaternion Operations
# ────────────────────────────────────────────────────────────────────

def quat_multiply(q1, q2):
    """Hamilton product of two quaternions (scalar-first convention).

    Given two quaternions:
        q1 = [a, b, c, d]  (a is scalar)
        q2 = [e, f, g, h]  (e is scalar)

    The Hamilton product is:
        q1 * q2 = [ a*e - b*f - c*g - d*h,
                    a*f + b*e + c*h - d*g,
                    a*g - b*h + c*e + d*f,
                    a*h + b*g - c*f + d*e ]

    Parameters
    ----------
    q1 : array_like, shape (4,)
        First quaternion [w, x, y, z].
    q2 : array_like, shape (4,)
        Second quaternion [w, x, y, z].

    Returns
    -------
    q_product : ndarray, shape (4,)
        The Hamilton product q1 ⊗ q2.
    """
    a, b, c, d = q1
    e, f, g, h = q2

    return np.array([
        a*e - b*f - c*g - d*h,   # scalar part
        a*f + b*e + c*h - d*g,   # i component
        a*g - b*h + c*e + d*f,   # j component
        a*h + b*g - c*f + d*e    # k component
    ])


def quat_conjugate(q):
    """Conjugate (inverse for unit quaternions) of a quaternion.

    For a unit quaternion, the conjugate equals the inverse:
        q* = [q0, -q1, -q2, -q3]
        q * q* = [1, 0, 0, 0]

    Parameters
    ----------
    q : array_like, shape (4,)
        Input quaternion [w, x, y, z].

    Returns
    -------
    q_conj : ndarray, shape (4,)
        Conjugate quaternion [w, -x, -y, -z].
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    """Normalize a quaternion to unit norm.

    A unit quaternion satisfies ||q|| = 1. Due to floating-point
    accumulation, quaternions drift from unit norm and must be
    periodically re-normalized.

    Also enforces the convention that q0 >= 0 to avoid the double-cover
    ambiguity (q and -q represent the same rotation).

    Parameters
    ----------
    q : array_like, shape (4,)
        Input quaternion [w, x, y, z].

    Returns
    -------
    q_normalized : ndarray, shape (4,)
        Unit quaternion with q[0] >= 0.
    """
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        # Degenerate case — return identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])
    q_norm = q / norm
    # Enforce positive scalar part (avoid double-cover ambiguity)
    if q_norm[0] < 0:
        q_norm = -q_norm
    return q_norm


def quat_to_rotation_matrix(q):
    """Convert a unit quaternion to a 3×3 rotation matrix.

    The rotation matrix R transforms vectors from body frame to world
    frame: v_world = R @ v_body.

    Given q = [w, x, y, z], the rotation matrix is:

        R = [ 1-2(y²+z²),  2(xy-wz),    2(xz+wy)   ]
            [ 2(xy+wz),    1-2(x²+z²),  2(yz-wx)   ]
            [ 2(xz-wy),    2(yz+wx),    1-2(x²+y²) ]

    Parameters
    ----------
    q : array_like, shape (4,)
        Unit quaternion [w, x, y, z].

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix (orthogonal, det = +1).
    """
    # Ensure unit quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    # Pre-compute repeated products (more efficient than computing each
    # matrix element independently)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1.0 - 2.0*(yy + zz),  2.0*(xy - wz),        2.0*(xz + wy)      ],
        [2.0*(xy + wz),        1.0 - 2.0*(xx + zz),   2.0*(yz - wx)      ],
        [2.0*(xz - wy),        2.0*(yz + wx),         1.0 - 2.0*(xx + yy)]
    ])
    return R


def quat_from_axis_angle(axis, angle):
    """Create a quaternion from a rotation axis and angle.

    Using Euler's rotation theorem, any rotation can be expressed as a
    single rotation of angle θ about a unit axis n̂:

        q = [cos(θ/2),  sin(θ/2) * n̂]

    Parameters
    ----------
    axis : array_like, shape (3,)
        Unit rotation axis [nx, ny, nz]. Will be normalized internally.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    q : ndarray, shape (4,)
        Unit quaternion representing the rotation.
    """
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        # Zero rotation axis → identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm

    half_angle = angle / 2.0
    q = np.array([
        np.cos(half_angle),
        np.sin(half_angle) * axis[0],
        np.sin(half_angle) * axis[1],
        np.sin(half_angle) * axis[2]
    ])
    return q


def quat_to_euler(q):
    """Convert a unit quaternion to ZYX Euler angles (aerospace convention).

    Decomposes the rotation into three successive rotations:
        1. Yaw   (ψ) about Z-axis
        2. Pitch (θ) about Y-axis
        3. Roll  (φ) about X-axis

    This is equivalent to the rotation sequence R = Rz(ψ) @ Ry(θ) @ Rx(φ).

    Parameters
    ----------
    q : array_like, shape (4,)
        Unit quaternion [w, x, y, z].

    Returns
    -------
    euler : ndarray, shape (3,)
        Euler angles [yaw, pitch, roll] in radians.
        - yaw   (ψ): rotation about Z, range [-π, π]
        - pitch (θ): rotation about Y, range [-π/2, π/2]
        - roll  (φ): rotation about X, range [-π, π]

    Notes
    -----
    Gimbal lock occurs when pitch = ±90°. This implementation handles
    the singularity by clamping the sin(pitch) value.
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    # Roll (φ) — rotation about X-axis
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (θ) — rotation about Y-axis
    sinp = 2.0 * (w*y - z*x)
    # Clamp to [-1, 1] to handle numerical errors near gimbal lock
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (ψ) — rotation about Z-axis
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([yaw, pitch, roll])


def rotate_vector(q, v):
    """Rotate a 3D vector by a unit quaternion.

    Uses the quaternion sandwich product:
        v' = q ⊗ [0, v] ⊗ q*

    This is equivalent to v' = R @ v where R is the rotation matrix
    corresponding to q.

    Parameters
    ----------
    q : array_like, shape (4,)
        Unit quaternion [w, x, y, z].
    v : array_like, shape (3,)
        3D vector to rotate.

    Returns
    -------
    v_rotated : ndarray, shape (3,)
        The rotated vector.
    """
    # Represent v as a pure quaternion [0, vx, vy, vz]
    v_quat = np.array([0.0, v[0], v[1], v[2]])

    # Sandwich product: q ⊗ v_quat ⊗ q*
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)

    # Extract vector part (indices 1, 2, 3)
    return result[1:4]


# ────────────────────────────────────────────────────────────────────
# Self-Test (run this file directly to verify correctness)
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Quaternion Utilities — Self-Test")
    print("=" * 60)

    # Test 1: Identity quaternion
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R_id = quat_to_rotation_matrix(q_id)
    print("\n[Test 1] Identity quaternion → rotation matrix:")
    print(f"  R = I₃? {np.allclose(R_id, np.eye(3))}")

    # Test 2: 90° rotation about Z-axis
    q_z90 = quat_from_axis_angle([0, 0, 1], np.pi / 2)
    v = np.array([1.0, 0.0, 0.0])
    v_rot = rotate_vector(q_z90, v)
    print(f"\n[Test 2] 90° about Z: [1,0,0] → {v_rot}")
    print(f"  Expected: [0, 1, 0], Match: {np.allclose(v_rot, [0, 1, 0], atol=1e-10)}")

    # Test 3: Quaternion multiply inverse = identity
    q_test = quat_normalize(np.array([0.5, 0.3, 0.1, 0.7]))
    q_inv = quat_conjugate(q_test)
    q_prod = quat_multiply(q_test, q_inv)
    print(f"\n[Test 3] q * q_conj = identity?")
    print(f"  Product: {q_prod}")
    print(f"  Match: {np.allclose(q_prod, [1, 0, 0, 0], atol=1e-10)}")

    # Test 4: Euler angles round-trip
    euler_in = np.array([0.3, 0.2, 0.1])  # yaw, pitch, roll
    # Build quaternion from Euler angles manually
    q_yaw = quat_from_axis_angle([0, 0, 1], euler_in[0])
    q_pitch = quat_from_axis_angle([0, 1, 0], euler_in[1])
    q_roll = quat_from_axis_angle([1, 0, 0], euler_in[2])
    q_euler = quat_multiply(quat_multiply(q_yaw, q_pitch), q_roll)
    euler_out = quat_to_euler(q_euler)
    print(f"\n[Test 4] Euler round-trip:")
    print(f"  In:  {np.degrees(euler_in)} deg")
    print(f"  Out: {np.degrees(euler_out)} deg")
    print(f"  Match: {np.allclose(euler_in, euler_out, atol=1e-10)}")

    print("\n" + "=" * 60)
    print("All tests passed!" if all([
        np.allclose(R_id, np.eye(3)),
        np.allclose(v_rot, [0, 1, 0], atol=1e-10),
        np.allclose(q_prod, [1, 0, 0, 0], atol=1e-10),
        np.allclose(euler_in, euler_out, atol=1e-10)
    ]) else "Some tests FAILED!")
    print("=" * 60)
