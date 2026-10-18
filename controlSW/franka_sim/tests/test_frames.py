"""A2 frames.py: ARKit y-up -> z-up, anchor placement, rot6d / pose7 / rotvec conventions. Pure numpy (RE and BP).

Run: cd controlSW/franka_sim && env -u PYTHONPATH $BP -m pytest -q tests/test_frames.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from franka_sim import frames as F

REPO = Path(__file__).resolve().parents[3]
rng = np.random.default_rng(0)


def rand_R(n=None):
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    return Q * np.sign(np.linalg.det(Q))


def rand_T():
    return F.make_T(rand_R(), rng.uniform(-1, 1, 3))


def test_no_mujoco_import():
    code = "import sys; import franka_sim.frames; assert not [m for m in sys.modules if m.startswith('mujoco')]"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PATH": ""})
    assert r.returncode == 0, r.stderr


def test_T_YUP_ZUP_literal_and_sensor_rig():
    lit = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    assert np.array_equal(F.T_YUP_ZUP, lit)
    sys.path.append(str(REPO / "sensor_rig"))
    tr = pytest.importorskip("sensor_rig.transforms")
    assert np.array_equal(F.T_YUP_ZUP, tr.T_YUP_ZUP)


def test_arkit_up_maps_to_z_up():
    up = np.array([0, 1.0, 0, 0])                                    # ARKit +y (gravity-opposed direction)
    assert np.allclose(F.arkit_to_zup(np.eye(4)) @ up, [0, 0, 1, 0])
    T = F.make_T(np.eye(3), [0.1, 1.5, -0.3])                        # 1.5 m above the ARKit origin, 0.3 m "forward"
    assert np.allclose(F.arkit_to_zup(T)[:3, 3], [0.1, 0.3, 1.5])     # z-up: x, -z_arkit, y_arkit
    # it is the inverse of the sensor_rig z-up -> y-up map
    assert np.allclose(F.T_YUP_ZUP @ F.arkit_to_zup(T), T)
    # a rigid pose keeps its rotation angle
    R = F.arkit_to_zup(F.make_T(rand_R()))[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)


def test_anchor_placement_maps_grasp_and_keeps_gravity():
    for _ in range(20):
        Tg = F.arkit_to_zup(rand_T())
        g = rng.uniform(-0.5, 0.5, 3)
        yaw = rng.uniform(-np.pi, np.pi)
        Tb = F.anchor_placement(Tg, g, yaw)
        assert np.allclose((Tb @ Tg)[:3, 3], g, atol=1e-12)
        assert np.allclose(Tb[:3, 2], [0, 0, 1]) and np.allclose(Tb[2, :3], [0, 0, 1])   # base z axis preserved
        assert np.isclose(np.arctan2(Tb[1, 0], Tb[0, 0]), yaw) or np.isclose(abs(yaw), np.pi)
        X = rand_T()
        # relative poses are untouched
        assert np.allclose(F.relative(Tb @ Tg, Tb @ X), F.relative(Tg, X), atol=1e-12)


def test_yaw_about_z():
    T = F.yaw_about_z(F.make_T(None, [1, 0, 0.5]), np.pi / 2)
    assert np.allclose(T[:3, 3], [0, 1, 0.5])


def test_rot6d_round_trip_and_report_agreement():
    for _ in range(50):
        R = rand_R()
        r6 = F.R_to_rot6d(R)
        assert r6.shape == (6,) and np.allclose(r6[:3], R[0]) and np.allclose(r6[3:], R[1])   # first two ROWS
        assert np.allclose(F.rot6d_to_R(r6), R, atol=1e-12)
    Rs = np.stack([rand_R() for _ in range(7)])
    assert np.allclose(F.rot6d_to_R(F.R_to_rot6d(Rs)), Rs, atol=1e-12)
    sys.path.append(str(REPO / "controlSW" / "franka_deploy"))
    report = pytest.importorskip("franka_deploy.eval.report")
    noisy = rng.normal(size=(5, 6))
    assert np.allclose(F.rot6d_to_R(noisy), report.rot6d_to_R(noisy.copy()), atol=1e-12)


def test_pose7_round_trip():
    for _ in range(50):
        T = rand_T()
        p = F.T_to_pose7(T)
        assert p.shape == (7,) and p[3] >= 0 and np.isclose(np.linalg.norm(p[3:]), 1)
        assert np.allclose(F.pose7_to_T(p), T, atol=1e-12)
    # 180 degree rotations (the Shepperd branch) and identity
    for R in (np.diag([1, -1, -1.0]), np.diag([-1, 1, -1.0]), np.diag([-1, -1, 1.0]), np.eye(3)):
        assert np.allclose(F.pose7_to_T(F.T_to_pose7(F.make_T(R, [1, 2, 3]))), F.make_T(R, [1, 2, 3]), atol=1e-12)
    assert np.allclose(F.pose7_to_T([0, 0, 0, np.sqrt(.5), 0, 0, np.sqrt(.5)])[:3, :3],
                       [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


def test_rotvec_pos_to_T_matches_scipy_when_available():
    Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    for rv in [rng.normal(size=3) for _ in range(10)] + [np.zeros(3), np.array([1e-10, 0, 0]), np.array([0, 0, np.pi])]:
        T = F.rotvec_pos_to_T([1, 2, 3], rv)
        assert np.allclose(T[:3, :3], Rotation.from_rotvec(rv).as_matrix(), atol=1e-9)
        assert np.allclose(T[:3, 3], [1, 2, 3])
