"""The table frame and the pose checks of ``umiproc run`` (``umiproc.run.pieces``) on a synthetic handheld clip in which
the camera poses, the IMU and the table tag agree exactly (``umiproc_testlib.rig_motion``): a known ``tx_slam_tag``
gives back the poses in the table frame; gravity and scale are checked once per demo, so a demo split into 3.0 s +
0.8 s keeps both pieces and a 1 s demo is never dropped for an unobservable scale, while a scaled demo fails; a jump
fails its piece; the orientation check is judged only on a long enough demo; the trimming report."""

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


def _lose(traj, frames):
    for k in frames:
        traj.tracked[k], traj.is_lost[k], traj.posed[k] = False, True, False
    return traj


def test_a_clean_demo_passes_its_pose_checks(rig, clip):
    m, ep = clip
    t = m["t_cam_ns"]
    traj = fork_trajectory(t, m["T_W_cam"])
    d = P.demo_pose_checks(traj, ep, rig)
    assert d["ok"] and d["reasons"] == [] and d["reported"] == [], d
    v = d["values"]
    assert v["gravity_ok"] and v["g_norm"] == pytest.approx(9.807, abs=0.05) and v["n_tracked"] == len(t)
    assert v["scale_status"] == "ok" and v["scale"] == pytest.approx(1.0, abs=0.03)
    r = P.pose_checks(_piece(t, 0, len(t)), traj, rig, m["T_W_G"], _grid(t))
    assert r["ok"], r
    assert 0.0 < r["values"]["max_speed_mps"] < 1.0 and r["values"]["pose_valid_rows"] == r["values"]["grid_rows"]


def test_a_demo_split_into_3s_and_0_8s_keeps_both_pieces(rig, clip):
    """UMI keeps every piece of >= 0.4 s: gravity and scale are judged on the whole demo, never on a short piece
    (the copied scale check alone cannot measure 0.8 s, and |g| over 0.8 s includes the hand's mean acceleration)."""
    from umiproc.dp.slam.base import finalize_trajectory

    m, ep = clip
    t, T = m["t_cam_ns"][30:146], m["T_W_cam"][30:146]                 # 90 + 2 lost + 24 frames
    traj = _lose(fork_trajectory(t, T), [90, 91])
    rules = apply_rules(traj.tracked, traj.t_ns, stopped=False)
    assert rules.ok and [(p.start, p.stop) for p in rules.pieces] == [(0, 90), (92, 116)]
    d = P.demo_pose_checks(traj, ep, rig)
    assert d["ok"] and d["values"]["scale_status"] == "ok" and d["values"]["n_tracked"] == 114
    assert d["values"]["scale"] == pytest.approx(1.0, abs=0.03)
    for p in rules.pieces:
        assert P.pose_checks(p, traj, rig, m["T_W_G"], _grid(t))["ok"]
    # what judging the 0.8 s piece by itself gave (the reviewed bug): no measurable scale
    sel = np.arange(92, 116)
    fin = finalize_trajectory(P.slam_imu_trajectory(traj, sel, rig.cameras["ultrawide"].T_imu_cam), ep, rig,
                              traj.t_ns[sel])
    assert fin.diagnostics["scale_check"]["status"] == "scale_unobservable"


@pytest.mark.parametrize("a, n", [(40, 30), (120, 33), (40, 36)])
def test_a_demo_of_30_to_36_tracked_frames_is_not_dropped_for_scale(rig, clip, a, n):
    m, ep = clip
    traj = fork_trajectory(m["t_cam_ns"][a:a + n], m["T_W_cam"][a:a + n])
    assert apply_rules(traj.tracked, traj.t_ns, stopped=False).ok           # it passes UMI's rules
    d = P.demo_pose_checks(traj, ep, rig)
    assert d["ok"] and d["reasons"] == [], d
    if d["values"]["scale_status"] != "ok":
        assert d["reported"] == ["scale_unobservable"]                       # reported, never a failure
    if n == 30:
        assert d["reported"] == ["scale_unobservable"]


def test_an_observable_scale_mismatch_fails_the_demo_and_a_jump_fails_its_piece(rig, clip):
    m, ep = clip
    t = m["t_cam_ns"]
    T = m["T_W_cam"].copy()
    T[:, :3, 3] *= 1.3                                                  # a map with the wrong scale
    d = P.demo_pose_checks(fork_trajectory(t, T), ep, rig)
    assert not d["ok"] and d["reasons"] == ["scale_mismatch"]
    assert d["values"]["scale"] == pytest.approx(1 / 1.3, abs=0.03)
    assert "scale 0.77" in P.describe_demo_checks(d, {"judged": False, "windows": 3})
    T = m["T_W_cam"].copy()
    T[200:, :3, 3] += [0.3, 0.0, 0.0]                                    # a 30 cm SLAM jump between two frames
    r = P.pose_checks(_piece(t, 150, 260), fork_trajectory(t, T), rig, m["T_W_G"], _grid(t))
    assert "speed" in r["reasons"] and "speed_extreme" in r["reasons"] and r["values"]["max_speed_mps"] > 7
    # outside the button span (no grid rows)
    r = P.pose_checks(_piece(t, 100, 160), fork_trajectory(t, m["T_W_cam"]), rig, m["T_W_G"], _grid(t)[:5])
    assert "no_grid_rows" in r["reasons"]


def test_the_tcp_is_in_the_table_frame_without_the_rigs_world_offset(rig, clip):
    from umiproc.dp.stages.pose import T_imu_tcp, tcp_poses

    m, _ = clip
    assert not np.allclose(rig.T_world_offset, np.eye(4))                # rig_v1's offset is in the rig file
    t = m["t_cam_ns"][:60]
    traj = fork_trajectory(t, m["T_W_cam"][:60])
    table = P.table_imu_trajectory(traj, np.arange(60), m["T_W_G"], rig.cameras["ultrawide"].T_imu_cam)
    pos, rot, valid = tcp_poses(table, t[5:50], P.table_rig(rig), max_gap_ns=10**9)
    T_G_tcp = np.linalg.inv(m["T_W_G"]) @ m["T_W_cam"][5:50] @ np.linalg.inv(rig.cameras["ultrawide"].T_imu_cam) \
        @ T_imu_tcp(rig)
    np.testing.assert_allclose(pos, T_G_tcp[:, :3, 3], atol=1e-5)
    # looking down at the table: about 180 deg from the tag frame, far from the zero rotation vector the copied
    # export refuses
    assert valid.all() and np.all(np.linalg.norm(rot, axis=1) > 2.5)
    np.testing.assert_allclose(P.table_rig(rig).T_world_offset, np.eye(4))
    assert not np.allclose(rig.T_world_offset, np.eye(4))                # the rig itself is left as it was


def test_the_trimming_report_says_what_the_kept_span_left_out():
    rows = np.arange(10, dtype=np.int64) * 16_666_667
    pose = np.ones(10, bool)
    width = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1, 1], bool)
    main = np.ones(10, bool)
    main[9] = False
    kept = {"i0": 2, "i1": 7, "n": 5}
    tr = P.trim_report(rows, kept, pose, width, main, n_frames=6, frames_kept=4)
    assert (tr["rows"], tr["of_rows"], tr["start_rows"], tr["end_rows"], tr["frames"]) == (5, 10, 2, 3, 2)
    assert tr["seconds"] == pytest.approx(5 / 60, abs=1e-3)
    assert tr["causes"] == {"no_width": 3, "no_main_frame": 1, "no_pose": 0, "outside_longest": 1}
    text = P.describe_trim(tr)
    assert text.startswith("trimmed 0.08 s (5 of 10 rows, 2 of 6 frames)") and "no finger width 3 rows" in text
    assert "no pose" not in text
    assert P.describe_trim(P.trim_report(rows, {"i0": 0, "i1": 10, "n": 10}, pose, pose, pose, n_frames=6,
                                         frames_kept=6)) == ""


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

