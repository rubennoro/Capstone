"""Synthesise an IMU stream (gyro + specific force in the IMU frame) from a pose trajectory (P1 §4.8 step 6).

Model:
- position: per-axis cubic smoothing spline (``scipy.interpolate.make_smoothing_spline``, GCV smoothing) so the
  ms-quantised iPhone timestamps do not explode in the second derivative; ``a_W = p''(t)``;
- rotation: ``RotationSpline`` (C2); gyro = body-frame central difference ``rotvec(R(t-h/2)^-1 R(t+h/2)) / h``;
- accelerometer (specific force): ``f_S = R_W_S(t)^T (a_W - g_W)`` with ``g_W = (0, 0, -9.80665)`` (z-up world), so
  at rest it reads ``R^T (0, 0, +g)``;
- optional noise: white noise ``density * sqrt(rate)`` plus a bias random walk ``b += walk * sqrt(dt) * N(0,1)``
  starting from a constant (seeded) bias.

``noise`` is a dict with the Kalibr names (``gyroscope_noise_density``, ``accelerometer_noise_density``,
``gyroscope_random_walk``, ``accelerometer_random_walk``; optional ``gyro_bias``/``accel_bias`` 3-vectors or scalars
for the initial bias std), or a ``rig_config.ImuNoise``, or None (noise-free).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.spatial.transform import Rotation, RotationSpline

GRAVITY = 9.80665
G_WORLD = np.array([0.0, 0.0, -GRAVITY])
DEFAULT_GYRO_BIAS = 2e-3   # rad/s, std of the initial constant bias per axis
DEFAULT_ACCEL_BIAS = 0.05  # m/s^2


def _noise_dict(noise) -> dict | None:
    if noise is None:
        return None
    if hasattr(noise, "kalibr_dict"):
        return dict(noise.kalibr_dict())
    return dict(noise)


def imu_times(t_first_ns: int, t_last_ns: int, rate_hz: float, jitter_ns: int = 0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample times ``t_first + round(k/rate)`` within [t_first, t_last], each with uniform ±jitter (kept increasing)."""
    period = 1e9 / rate_hz
    n = int(np.floor((int(t_last_ns) - int(t_first_ns)) / period)) + 1
    t = int(t_first_ns) + np.round(np.arange(n) * period).astype(np.int64)
    if jitter_ns:
        rng = rng or np.random.default_rng(0)
        j = rng.integers(-int(jitter_ns), int(jitter_ns) + 1, size=n)
        j[0] = max(j[0], 0)
        j[-1] = min(j[-1], 0)
        t = t + j
        t = np.clip(t, int(t_first_ns), int(t_last_ns))
        t = np.maximum.accumulate(t)
        bad = np.flatnonzero(np.diff(t) <= 0)
        if len(bad):  # extremely unlikely with |jitter| << period; drop collisions
            t = np.delete(t, bad + 1)
    return t


def add_noise(t_ns: np.ndarray, gyro: np.ndarray, accel: np.ndarray, noise, seed: int = 0):
    """Return noisy copies of ``gyro``/``accel`` (white noise + bias random walk from a seeded constant bias)."""
    nd = _noise_dict(noise)
    if nd is None:
        return gyro.copy(), accel.copy()
    rng = np.random.default_rng(seed)
    n = len(t_ns)
    if n == 0:
        return gyro.copy(), accel.copy()
    dt = np.diff(np.asarray(t_ns, dtype=np.int64)) * 1e-9
    rate = 1.0 / float(np.median(dt)) if n > 1 else 1000.0
    out = []
    for x, dens_k, walk_k, bias_k, bias_def in (
            (gyro, "gyroscope_noise_density", "gyroscope_random_walk", "gyro_bias", DEFAULT_GYRO_BIAS),
            (accel, "accelerometer_noise_density", "accelerometer_random_walk", "accel_bias", DEFAULT_ACCEL_BIAS)):
        sigma = float(nd.get(dens_k, 0.0)) * np.sqrt(rate)
        walk = float(nd.get(walk_k, 0.0))
        b0 = rng.normal(0.0, 1.0, 3) * np.broadcast_to(np.asarray(nd.get(bias_k, bias_def), dtype=np.float64), 3)
        steps = rng.normal(0.0, 1.0, (n, 3)) * (walk * np.sqrt(np.concatenate([[0.0], dt])))[:, None]
        bias = b0 + np.cumsum(steps, axis=0)
        out.append(x + bias + rng.normal(0.0, sigma, (n, 3)))
    return out[0], out[1]


def synth_imu(t_pose_ns, T_W_S, rate_hz: float, noise=None, seed: int = 0, jitter_ns: int = 0,
              h_s: float = 1e-3, smooth: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IMU samples over ``[t_pose[0], t_pose[-1]]`` at ``rate_hz`` from poses ``T_W_S`` (IMU frame S, z-up world).

    Returns ``(t_ns (M,) int64, gyro (M,3) rad/s, accel (M,3) m/s^2)``, both in frame S. ``smooth=False`` uses an
    interpolating spline (lam=0) instead of GCV smoothing (exact analytic inputs).
    """
    t_pose = np.asarray(t_pose_ns, dtype=np.int64)
    T = np.asarray(T_W_S, dtype=np.float64)
    if len(t_pose) < 5 or len(T) != len(t_pose):
        raise ValueError("need >= 5 poses with matching times")
    if np.any(np.diff(t_pose) <= 0):
        raise ValueError("pose times must be strictly increasing")
    rng = np.random.default_rng(seed + 7919)
    t_imu = imu_times(int(t_pose[0]), int(t_pose[-1]), rate_hz, jitter_ns, rng)

    t0 = int(t_pose[0])
    ts = (t_pose - t0) * 1e-9
    tq = (t_imu - t0) * 1e-9

    # position -> world acceleration
    a_W = np.empty((len(tq), 3))
    for ax in range(3):
        spl = make_smoothing_spline(ts, T[:, ax, 3], lam=None if smooth else 0.0)
        a_W[:, ax] = spl.derivative(2)(tq)

    # rotation -> body rates (central difference on the C2 rotation spline)
    rs = RotationSpline(ts, Rotation.from_matrix(T[:, :3, :3]))
    lo = np.clip(tq - h_s / 2, ts[0], ts[-1])
    hi = np.clip(tq + h_s / 2, ts[0], ts[-1])
    gyro = (rs(lo).inv() * rs(hi)).as_rotvec() / np.maximum(hi - lo, 1e-12)[:, None]
    R = rs(tq).as_matrix()
    accel = np.einsum("nji,nj->ni", R, a_W - G_WORLD)

    gyro, accel = add_noise(t_imu, gyro, accel, noise, seed)
    return t_imu, gyro, accel


def synth_static_imu(t_first_ns: int, t_last_ns: int, rate_hz: float, R_W_S: np.ndarray | None = None, noise=None,
                     seed: int = 0, jitter_ns: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IMU at rest with orientation ``R_W_S`` (default identity): gyro 0, accel ``R^T (0,0,+g)``, plus noise."""
    rng = np.random.default_rng(seed + 7919)
    t = imu_times(t_first_ns, t_last_ns, rate_hz, jitter_ns, rng)
    R = np.eye(3) if R_W_S is None else np.asarray(R_W_S, dtype=np.float64)
    gyro = np.zeros((len(t), 3))
    accel = np.tile(R.T @ -G_WORLD, (len(t), 1))
    gyro, accel = add_noise(t, gyro, accel, noise, seed)
    return t, gyro, accel
