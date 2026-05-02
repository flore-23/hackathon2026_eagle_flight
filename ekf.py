"""
ekf.py — 15-state error-state Kalman filter (ESEKF) for INS/GPS fusion,
with the quaternion / rotation / NED helpers it needs.

Conventions (Hamilton quaternions, w-x-y-z):
  - Quaternion `q` represents a body→world rotation.
  - World frame: NED (north, east, down).
  - Body  frame: x=forward, y=right, z=down.
  - Gravity in NED: g = [0, 0, +9.81].
"""

import numpy as np


# ============================================================
#  Constants
# ============================================================

EARTH_RADIUS_M = 6378137.0
G_NED = np.array([0.0, 0.0, 9.81])


# ============================================================
#  Quaternion / rotation helpers
# ============================================================

def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic Euler (yaw → pitch → roll), all in radians."""
    cy, sy = np.cos(yaw   / 2), np.sin(yaw   / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll  / 2), np.sin(roll  / 2)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_to_rpy(q: np.ndarray):
    """Extract ZYX intrinsic Euler angles (roll, pitch, yaw) in radians.

    Inverse of `quat_from_rpy`. Returns (roll, pitch, yaw).
    """
    w, x, y, z = q
    # roll (x)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y) — guard against gimbal lock
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0, 0, 0])


def quat_from_axis_angle(theta_vec: np.ndarray) -> np.ndarray:
    """Build a quaternion from an axis-angle vector (rotation vector, radians)."""
    n = np.linalg.norm(theta_vec)
    if n < 1e-9:
        return np.array([1.0, theta_vec[0] / 2, theta_vec[1] / 2, theta_vec[2] / 2])
    half = n / 2
    s = np.sin(half) / n
    return np.array([np.cos(half), theta_vec[0] * s, theta_vec[1] * s, theta_vec[2] * s])


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])


def omega_from_quat_pair(q_prev: np.ndarray, q_next: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity from two consecutive attitude quaternions.

    For small Δq = q_prev⁻¹ ⊗ q_next ≈ [1, ω·dt/2] in body frame.
    """
    if dt <= 0:
        return np.zeros(3)
    dq = quat_mult(quat_conj(q_prev), q_next)
    if dq[0] < 0:
        dq = -dq  # ensure short-way rotation
    return 2.0 * dq[1:] / dt


# ============================================================
#  Local NED tangent-plane (good for distances < ~100 km)
# ============================================================

def lla_to_ned(lat, lon, alt, lat0, lon0, alt0):
    """(lat, lon, alt) [deg, deg, m] → (north, east, down) [m] relative to (lat0, lon0, alt0)."""
    lat0_rad = np.radians(lat0)
    n = np.radians(lat - lat0) * EARTH_RADIUS_M
    e = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(lat0_rad)
    d = alt0 - alt
    return np.array([n, e, d])


def ned_to_lla(n, e, d, lat0, lon0, alt0):
    """Inverse of `lla_to_ned`."""
    lat0_rad = np.radians(lat0)
    lat = lat0 + np.degrees(n / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(e / (EARTH_RADIUS_M * np.cos(lat0_rad)))
    alt = alt0 - d
    return lat, lon, alt


# ============================================================
#  15-state Error-State EKF (ESEKF) for INS/GPS
#
#  Nominal state: p (NED, m), v (NED, m/s), q (body→world), b_g (rad/s), b_a (m/s²)
#  Error  state : δp, δv, δθ (axis-angle), δb_g, δb_a   — 15-dim
#  IMU input    : ω_meas (body, rad/s), f_meas (body, m/s², specific force)
#  Update       : GPS position in local NED
# ============================================================

class StrapdownEKF:
    def __init__(self,
                 q0: np.ndarray,
                 p0: np.ndarray = None,
                 v0: np.ndarray = None,
                 sigma_a: float = 1.0,        # accel meas noise (m/s²)
                 sigma_g: float = 0.1,        # gyro  meas noise (rad/s)
                 sigma_bg: float = 1e-3,      # gyro  bias random walk
                 sigma_ba: float = 1e-2,      # accel bias random walk
                 sigma_gps_h: float = 5.0,    # GPS horizontal std (m)
                 sigma_gps_v: float = 10.0):  # GPS vertical   std (m)
        self.p = np.zeros(3) if p0 is None else np.array(p0, dtype=float)
        self.v = np.zeros(3) if v0 is None else np.array(v0, dtype=float)
        self.q = quat_normalize(np.array(q0, dtype=float))
        self.b_g = np.zeros(3)
        self.b_a = np.zeros(3)

        self.P = np.zeros((15, 15))
        self.P[0:3,   0:3]   = np.eye(3) * 25.0
        self.P[3:6,   3:6]   = np.eye(3) * 4.0
        self.P[6:9,   6:9]   = np.eye(3) * (np.radians(5.0) ** 2)
        self.P[9:12,  9:12]  = np.eye(3) * (1e-2 ** 2)
        self.P[12:15, 12:15] = np.eye(3) * (1e-1 ** 2)

        self.sigma_a, self.sigma_g = sigma_a, sigma_g
        self.sigma_bg, self.sigma_ba = sigma_bg, sigma_ba
        self.R_gps = np.diag([sigma_gps_h ** 2, sigma_gps_h ** 2, sigma_gps_v ** 2])

    def predict(self, omega_meas: np.ndarray, f_meas: np.ndarray, dt: float):
        """Propagate nominal state and covariance forward by dt."""
        if dt <= 0:
            return
        omega = omega_meas - self.b_g
        f     = f_meas    - self.b_a
        R = quat_to_rotmat(self.q)
        a_world = R @ f + G_NED

        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = quat_normalize(quat_mult(self.q, quat_from_axis_angle(omega * dt)))

        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -R @ skew(f)
        F[3:6, 12:15] = -R
        F[6:9, 6:9] = -skew(omega)
        F[6:9, 9:12] = -np.eye(3)
        Fd = np.eye(15) + F * dt

        Q = np.zeros((15, 15))
        Q[3:6,   3:6]   = np.eye(3) * (self.sigma_a ** 2) * dt
        Q[6:9,   6:9]   = np.eye(3) * (self.sigma_g ** 2) * dt
        Q[9:12,  9:12]  = np.eye(3) * (self.sigma_bg ** 2) * dt
        Q[12:15, 12:15] = np.eye(3) * (self.sigma_ba ** 2) * dt

        self.P = Fd @ self.P @ Fd.T + Q

    def update_gps(self, p_obs: np.ndarray):
        """GPS position update in NED. Injects error into nominal state and resets."""
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        y = p_obs - self.p
        dx = K @ y

        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_normalize(quat_mult(self.q, quat_from_axis_angle(dx[6:9])))
        self.b_g = self.b_g + dx[9:12]
        self.b_a = self.b_a + dx[12:15]

        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_gps @ K.T  # Joseph form

    def update_velocity(self, v_obs: np.ndarray, sigma: np.ndarray):
        """Velocity observation in NED (m/s). `sigma` is the per-axis std,
        a length-3 array. Use a very large sigma on axes you don't want to
        constrain (e.g., vertical when only GPS Doppler horizontal is reliable).
        """
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        Rm = np.diag(np.asarray(sigma, dtype=float) ** 2)
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        y = np.asarray(v_obs, dtype=float) - self.v
        dx = K @ y

        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_normalize(quat_mult(self.q, quat_from_axis_angle(dx[6:9])))
        self.b_g = self.b_g + dx[9:12]
        self.b_a = self.b_a + dx[12:15]

        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T   # Joseph form

    def update_course(self, course_rad: float, sigma_course: float = 0.2):
        """GPS course observation: pulls filter yaw toward the compass direction
        of the body's x-axis. Assumes no sideslip (body x ≈ velocity vector).

        Innovation uses the compass heading of R(q)[:,0], wrapped to (-π, π].
        Jacobian (1×15) is non-zero only in the body-frame attitude block:
            ∂ψ/∂δθ_body = [0, R[2,1]/D, R[2,2]/D]   with D = R[0,0]² + R[1,0]²
        Skips the update near gimbal lock (D ≈ 0, i.e. body x near vertical).
        """
        R = quat_to_rotmat(self.q)
        D = R[0, 0] ** 2 + R[1, 0] ** 2
        if D < 1e-6:
            return
        psi_pred = np.arctan2(R[1, 0], R[0, 0])
        innov = course_rad - psi_pred
        # wrap to (-π, π]
        while innov >  np.pi: innov -= 2.0 * np.pi
        while innov < -np.pi: innov += 2.0 * np.pi

        H = np.zeros((1, 15))
        H[0, 7] = R[2, 1] / D
        H[0, 8] = R[2, 2] / D

        Rm = np.array([[sigma_course ** 2]])
        S = float(H @ self.P @ H.T + Rm)
        K = (self.P @ H.T) / S            # (15,1)
        dx = (K * innov).flatten()

        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_normalize(quat_mult(self.q, quat_from_axis_angle(dx[6:9])))
        self.b_g = self.b_g + dx[9:12]
        self.b_a = self.b_a + dx[12:15]

        I = np.eye(15)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T   # Joseph form
