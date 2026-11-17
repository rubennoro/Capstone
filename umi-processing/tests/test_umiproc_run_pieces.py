"""The table frame and the pose checks of ``umiproc run`` (``umiproc.run.pieces``) on a synthetic handheld clip in which
the camera poses, the IMU and the table tag agree exactly (``umiproc_testlib.rig_motion``): a known ``tx_slam_tag``
gives back the poses in the table frame, a clean piece passes, a scaled one, a jump and a short one fail, and the
orientation check is judged only on a long enough demo."""

from types import SimpleNamespace

import numpy as np
import pytest

from umiproc.dp.config import load_rig
from umiproc.paths import repo_config
from umiproc.run import pieces as P
from umiproc.run.rules import Piece, apply_rules
from umiproc_testlib import fork_trajectory, rig_motion


@pytest.fixture(scope="module")
def rig():
    return load_rig(repo_config("rigs", "rig_pi_v3.yaml"))


@pytest.fixture(scope="module")
def clip(rig):
    m = rig_motion(rig.cameras["ultrawide"].T_imu_cam, duration_s=14.0)
    ep = SimpleNamespace(imu=SimpleNamespace(t_ns=m["imu_t_ns"], gyro=m["gyro_imu"], accel=m["acc_imu"]))
    return m, ep


def _grid(t):
    return np.arange(int(t[0]), int(t[-1]) + 1, 16_666_667, dtype=np.int64)


def _piece(t, a, b, k=1):
    return Piece(k, a, b, int(t[a]), int(t[b - 1]))


def test_poses_in_the_table_frame(rig, clip):
    m, _ = clip
    traj = fork_trajectory(m["t_cam_ns"], m["T_W_cam"])
    # the tag at the origin of the gravity frame G: tx_slam_tag = T_W_G, so the table frame is G
    table = P.table_imu_trajectory(traj, np.arange(len(traj.t_ns)), m["T_W_G"], rig.cameras["ultrawide"].T_imu_cam)
    T_G_imu = np.linalg.inv(m["T_W_G"]) @ m["T_W_imu"]
    assert table.world == "table_tag" and table.frame == "imu"
    np.testing.assert_allclose(table.T(), T_G_imu, atol=1e-9)
    assert np.allclose(table.p[:, 2].mean(), 0.45, atol=0.01)           # 45 cm above the table, z up
    # only tracked rows have a pose
    traj.tracked[3] = False
    with pytest.raises(ValueError, match="tracked"):
        P.slam_imu_trajectory(traj, np.arange(10), rig.cameras["ultrawide"].T_imu_cam)


def test_a_clean_piece_passes_its_pose_checks(rig, clip):
    m, ep = clip
    t = m["t_cam_ns"]
    traj = fork_trajectory(t, m["T_W_cam"])
    r = P.pose_checks(_piece(t, 0, len(t)), traj, ep, rig, m["T_W_G"], _grid(t))
    assert r["ok"], r
    v = r["values"]
    assert v["gravity_ok"] and v["g_norm"] == pytest.approx(9.807, abs=0.05)
    assert v["scale_status"] == "ok" and v["scale"] == pytest.approx(1.0, abs=0.03)
    assert 0.0 < v["max_speed_mps"] < 1.0 and v["pose_valid_rows"] == v["grid_rows"]


def test_a_scaled_or_jumping_or_short_piece_fails(rig, clip):
    m, ep = clip
    t = m["t_cam_ns"]
    T = m["T_W_cam"].copy()
    T[:, :3, 3] *= 1.3                                                  # a map with the wrong scale
    r = P.pose_checks(_piece(t, 0, len(t)), fork_trajectory(t, T), ep, rig, m["T_W_G"], _grid(t))
    assert not r["ok"] and r["reasons"] == ["scale_mismatch"] and r["values"]["scale"] == pytest.approx(1 / 1.3,
                                                                                                         abs=0.03)
    T = m["T_W_cam"].copy()
    T[200:, :3, 3] += [0.3, 0.0, 0.0]                                    # a 30 cm SLAM jump between two frames
    r = P.pose_checks(_piece(t, 150, 260), fork_trajectory(t, T), ep, rig, m["T_W_G"], _grid(t))
    assert "speed" in r["reasons"] and "speed_extreme" in r["reasons"] and r["values"]["max_speed_mps"] > 7
    # a 12-frame piece (0.4 s): the copied scale check cannot measure it
    r = P.pose_checks(_piece(t, 100, 112), fork_trajectory(t, m["T_W_cam"]), ep, rig, m["T_W_G"], _grid(t))
    assert "scale_unobservable" in r["reasons"]
    # outside the button span (no grid rows)
    r = P.pose_checks(_piece(t, 100, 160), fork_trajectory(t, m["T_W_cam"]), ep, rig, m["T_W_G"],
                      _grid(t)[:5])
    assert "no_grid_rows" in r["reasons"]


def test_orientation_is_judged_only_on_a_long_enough_demo(rig, clip):
    m, ep = clip
    t = m["t_cam_ns"]
    o = P.orientation_check(fork_trajectory(t, m["T_W_cam"]), ep, rig)
    assert o["judged"] and o["ok"] and o["windows"] >= 4 and o["max_deg"] < 1.0
    short = fork_trajectory(t[:240], m["T_W_cam"][:240])                # 8 s: 3 windows of 2.5 s
    o = P.orientation_check(short, ep, rig)
    assert not o["judged"] and o["ok"] and o["windows"] == 3
    # a camera-IMU rotation error of 6 deg makes gravity turn with the rig: judged and failed
    from scipy.spatial.transform import Rotation

    T_imu_cam = np.asarray(rig.cameras["ultrawide"].T_imu_cam).copy()
    T_imu_cam[:3, :3] = Rotation.from_euler("x", 6, degrees=True).as_matrix() @ T_imu_cam[:3, :3]
    wrong = SimpleNamespace(cameras={"ultrawide": SimpleNamespace(T_imu_cam=T_imu_cam)},
                            imu_time_ns=rig.imu_time_ns)
    o = P.orientation_check(fork_trajectory(t, m["T_W_cam"]), ep, wrong)
    assert o["judged"] and not o["ok"] and o["max_deg"] > 3.0


def test_pieces_from_the_rules_cover_the_right_frames(rig, clip):
    m, _ = clip
    t = m["t_cam_ns"]
    tr = np.ones(len(t), bool)
    tr[[100, 101]] = False
    r = apply_rules(tr, t, stopped=False)
    assert [(p.start, p.stop) for p in r.pieces] == [(0, 100), (102, len(t))]
    g = P.grid_rows(_grid(t), r.pieces[1])
    assert g[0] >= t[102] and g[-1] <= t[-1]
    assert P.frames_covered(t, r.pieces[1], int(g[0]), int(g[-1])) == len(t) - 102
    assert P.piece_zarr_name("pi-x", 2, 2) == "pi-x_piece2" and P.piece_zarr_name("pi-x", 1, 1) == "pi-x"

