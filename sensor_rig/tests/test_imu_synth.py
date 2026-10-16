"""T9: IMU synthesis against closed-form kinematics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from sensor_rig.sim.imu_synth import GRAVITY, synth_imu, synth_static_imu

R_CIRC, F_CIRC = 0.3, 0.5      # circle radius (m), frequency (Hz)
YAW_RATE = 0.7                 # rad/s
PITCH_A, PITCH_F = 0.2, 0.4    # rad, Hz
Z_A, Z_F = 0.05, 0.3


def _truth(t):
    w = 2 * np.pi * F_CIRC
    p = np.stack([R_CIRC * np.cos(w * t), R_CIRC * np.sin(w * t), 1.0 + Z_A * np.sin(2 * np.pi * Z_F * t)], 1)
    a = np.stack([-R_CIRC * w**2 * np.cos(w * t), -R_CIRC * w**2 * np.sin(w * t),
                  -Z_A * (2 * np.pi * Z_F) ** 2 * np.sin(2 * np.pi * Z_F * t)], 1)
    psi = YAW_RATE * t + 0.3
    wp = 2 * np.pi * PITCH_F
    th = PITCH_A * np.sin(wp * t)
    dth = PITCH_A * wp * np.cos(wp * t)
    R = Rotation.from_euler("ZY", np.stack([psi, th], 1)).as_matrix()  # Rz(psi) @ Ry(theta)
    # body rates of R = Rz(psi) Ry(th): Ry(th)^T (0,0,dpsi) + (0,dth,0)
    Ry = Rotation.from_euler("Y", th).as_matrix()
    omega = np.einsum("nji,j->ni", Ry, np.array([0.0, 0.0, YAW_RATE])) + np.stack(
        [np.zeros_like(t), dth, np.zeros_like(t)], 1)
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    return T, a, omega


def test_T9_analytic_trajectory():
    t0 = 1_000_000_000_000
    t_pose = t0 + np.round(np.arange(0, 6.0, 1 / 60) * 1e9).astype(np.int64)
    T, _, _ = _truth((t_pose - t0) * 1e-9)
    t_imu, gyro, acc = synth_imu(t_pose, T, 400.0, noise=None)
    ts = (t_imu - t0) * 1e-9
    T_q, a_W, omega = _truth(ts)
    f = np.einsum("nji,nj->ni", T_q[:, :3, :3], a_W - np.array([0, 0, -GRAVITY]))
    inner = (ts > ts[0] + 0.2) & (ts < ts[-1] - 0.2)
    assert np.abs(gyro[inner] - omega[inner]).max() < 2e-3
    assert np.abs(acc[inner] - f[inner]).max() < 2e-2
    assert t_imu[0] == t_pose[0] and t_imu[-1] <= t_pose[-1] and np.all(np.diff(t_imu) > 0)
    assert abs(np.median(np.diff(t_imu)) - 2.5e6) < 1


def test_T9_ms_quantised_times_still_reasonable():
    """iPhone times are floored to ms; the smoothing spline must keep the accelerometer usable."""
    t0 = 1_000_000_000_000
    t_true = t0 + np.round(np.arange(0, 6.0, 1 / 60) * 1e9).astype(np.int64)
    T, _, _ = _truth((t_true - t0) * 1e-9)
    t_q = (t_true // 1_000_000) * 1_000_000
    t_imu, gyro, acc = synth_imu(t_q, T, 400.0)
    T_q, a_W, omega = _truth((t_imu - t0) * 1e-9)
    f = np.einsum("nji,nj->ni", T_q[:, :3, :3], a_W - np.array([0, 0, -GRAVITY]))
    inner = slice(200, -200)
    assert np.sqrt(np.mean((acc[inner] - f[inner]) ** 2)) < 0.5
    assert np.sqrt(np.mean((gyro[inner] - omega[inner]) ** 2)) < 0.1


def test_T9_static_and_noise():
    R = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    t, g, a = synth_static_imu(0, 2_000_000_000, 400.0, R)
    assert np.allclose(a, R.T @ np.array([0, 0, GRAVITY]), atol=1e-12) and np.all(g == 0)
    noise = {"gyroscope_noise_density": 2e-4, "accelerometer_noise_density": 3e-3,
             "gyroscope_random_walk": 0.0, "accelerometer_random_walk": 0.0, "gyro_bias": 0.0, "accel_bias": 0.0}
    t, g, a = synth_static_imu(0, 20_000_000_000, 400.0, R, noise=noise, seed=1)
    assert abs(g.std() / (2e-4 * np.sqrt(400)) - 1) < 0.05
    assert abs((a - R.T @ [0, 0, GRAVITY]).std() / (3e-3 * np.sqrt(400)) - 1) < 0.05
    t2, g2, _ = synth_static_imu(0, 20_000_000_000, 400.0, R, noise=noise, seed=1)
    assert np.array_equal(g, g2)  # seeded
