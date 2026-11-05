import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataproc.geometry.trajectory import Trajectory
from dataproc.geometry.transforms import inv_T, make_T, rot_x, rot_z
from dataproc.stages.pose import MIN_ROTVEC_NORM, T_imu_tcp, tcp_poses
from test_geometry_common import make_rig

T0 = 1_000_000_000_000


@pytest.fixture(scope="module")
def rig():
    return make_rig()        # rig_v1: uw T_imu_cam = [I | (-0.02, 0, 0.01)], T_cam_tcp = [Rx(15) | (0, 0.06, 0.2)],
                             # world_offset = [Rz(30) | (0.1, 0.2, 0.3)]


def test_identity_imu_pose_hand_computed(rig):
    tr = Trajectory.from_T([T0, T0 + 10_000_000], np.stack([np.eye(4)] * 2))
    pos, rv, ok = tcp_poses(tr, [T0, T0 + 5_000_000], rig, max_gap_ns=20_000_000)
    assert pos.dtype == np.float32 and rv.dtype == np.float32 and ok.dtype == bool and ok.all()
    # p = (0.1, 0.2, 0.3) + Rz(30) (-0.02, 0.06, 0.21)
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    expect_p = [0.1 + (-0.02 * c - 0.06 * s), 0.2 + (-0.02 * s + 0.06 * c), 0.3 + 0.21]
    assert np.allclose(pos, [expect_p] * 2, atol=1e-6)
    assert np.allclose(expect_p, [0.052679492, 0.241961524, 0.51], atol=1e-9)
    # R = Rz(30) Rx(15): extrinsic x then z
    expect_rv = Rotation.from_euler("xz", [15, 30], degrees=True).as_rotvec()
    assert np.allclose(rv, [expect_rv] * 2, atol=1e-6)


def test_chain_order_with_moving_imu(rig):
    T_W_imu = make_T(rot_z(90) @ rot_x(20), [1.0, 2.0, 0.5])
    tr = Trajectory.from_T([T0, T0 + 1], np.stack([T_W_imu] * 2))
    pos, rv, ok = tcp_poses(tr, [T0], rig, max_gap_ns=10)
    T = rig.T_world_offset @ T_W_imu @ rig.cameras["ultrawide"].T_imu_cam @ rig.T_cam_tcp
    assert np.allclose(pos[0], T[:3, 3], atol=1e-6)
    assert np.allclose(Rotation.from_rotvec(rv[0]).as_matrix(), T[:3, :3], atol=1e-6)
    assert np.allclose(T_imu_tcp(rig), rig.cameras["ultrawide"].T_imu_cam @ rig.T_cam_tcp)


def test_interpolation_and_validity(rig):
    t = T0 + np.array([0, 100, 200, 600], dtype=np.int64) * 1_000_000
    T = np.stack([make_T(t=[x, 0, 0]) for x in (0.0, 0.1, 0.2, 0.6)])
    tr = Trajectory.from_T(t, T)
    grid = T0 + np.array([-10, 50, 150, 400, 600, 610], dtype=np.int64) * 1_000_000
    pos, rv, ok, why = tcp_poses(tr, grid, rig, max_gap_ns=150_000_000, return_reasons=True)
    assert ok.tolist() == [False, True, True, False, True, False]
    assert why.tolist() == ["outside_trajectory", "", "", "gap", "", "outside_trajectory"]
    p0, _, _ = tcp_poses(Trajectory.from_T([T0, T0 + 1], np.stack([np.eye(4)] * 2)), [T0], rig, 10)
    # a pure IMU translation dx moves the TCP by Rz(30) dx in the offset world
    assert np.allclose(pos[1] - p0[0], rot_z(30) @ [0.05, 0, 0], atol=1e-6)
    assert np.allclose(pos[2] - p0[0], rot_z(30) @ [0.15, 0, 0], atol=1e-6)
    assert np.isfinite(pos).all() and np.isfinite(rv).all()


def test_rotvec_near_zero_flagged(rig):
    # an IMU pose that makes the TCP rotation exactly the identity in the offset world
    R_bad = inv_T(rig.T_world_offset)[:3, :3] @ inv_T(T_imu_tcp(rig))[:3, :3]
    tr = Trajectory.from_T([T0, T0 + 10], np.stack([make_T(R_bad)] * 2))
    pos, rv, ok, why = tcp_poses(tr, [T0, T0 + 5], rig, max_gap_ns=100, return_reasons=True)
    assert np.linalg.norm(rv, axis=1).max() <= MIN_ROTVEC_NORM
    assert not ok.any() and set(why) == {"rotvec_near_zero"}


def test_rejects_non_imu_trajectory(rig):
    tr = Trajectory.from_T([T0, T0 + 1], np.stack([np.eye(4)] * 2), frame="ultrawide")
    with pytest.raises(ValueError, match="imu-frame"):
        tcp_poses(tr, [T0], rig, 10)


def test_empty_grid(rig):
    tr = Trajectory.from_T([T0, T0 + 1], np.stack([np.eye(4)] * 2))
    pos, rv, ok = tcp_poses(tr, np.zeros(0, np.int64), rig, 10)
    assert pos.shape == (0, 3) and rv.shape == (0, 3) and ok.shape == (0,)
