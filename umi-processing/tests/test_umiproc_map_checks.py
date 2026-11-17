"""umiproc.map.checks: M1-M5 at their boundaries, on the recorded SLAM logs (tests/fixtures/slam) and on synthetic
trajectories, plus the speed, lost-stretch and pacing helpers, M4's orientation sub-check on synthetic handheld motion
(consistent, a 5 deg camera-IMU rotation error, slow drift, an accelerometer bias, too short), and the M1/M2 hint after
many map resets right after initialisation (the real mapping log of clip 128, trimmed)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from umiproc.map import checks as C
from umiproc.map.tag import SightingStats, Sighting
from umiproc.slam.log import parse_log, read_log
from umiproc.slam.trajectory_csv import ForkTrajectory
from umiproc_testlib import fork_trajectory, rig_motion

FIX = Path(__file__).parent / "fixtures" / "slam"
DT = 33_333_333
T0 = 1_000_000_000


def _summary_log(maps, rc=0, extra=""):
    import json

    s = {"format": "umiproc.slam_summary.v2", "exit_code": rc, "error": None, "frames_total": 10, "frames_fed": 10,
         "lost_frames": 0, "not_tracking_frames": 0, "stopped_early": False, "stop_reason": None,
         "state_counts": {"2": 10}, "maps": maps, "t0_ns": 0, "map_removed": False, "options": {},
         "pacing": {"track_ms": {"median": 20.0, "p95": 25.0, "max": 30.0}, "frames_over_period": 0,
                    "feed_rate_hz": 29.9, "recording_rate_hz": 30.0, "feed_wall_s": 1.0}}
    return parse_log(extra + "UMIPROC_SUMMARY " + json.dumps(s) + "\n")


def _map(i=0, kf=10, imu=True, b1=True, b2=True):
    return {"id": i, "keyframes": kf, "map_points": 100, "imu_initialized": imu, "inertial_ba1": b1,
            "inertial_ba2": b2}


# --------------------------------------------------------------------------------------------- M1, M2

def test_m1_on_the_recorded_mapping_runs():
    lg = read_log(FIX / "map_run.log")                    # one map, 34 keyframes
    c = C.check_m1(0, False, lg.summary, lg, map_saved=True)
    assert c.ok and c.values["maps_with_keyframes"] == 1 and "34 keyframes" in c.message
    lg = read_log(FIX / "map_run_reset_at_end.log")       # reset at the end: no map left
    c = C.check_m1(0, False, lg.summary, lg, map_saved=True)
    assert c.passed is False and c.values["maps_with_keyframes"] == 0 and c.values["map_resets"] >= 1
    assert "no map with keyframes" in c.message and c.hints


@pytest.mark.parametrize("maps,saved,ok,text", [
    ([_map()], True, True, "1 map"),
    ([_map(), _map(1, 0)], True, True, "1 map"),                 # an empty map does not count
    ([_map(), _map(1, 5)], True, False, "2 maps with keyframes"),
    ([], True, False, "no map with keyframes"),
    ([_map()], False, False, "not saved"),
])
def test_m1_boundaries(maps, saved, ok, text):
    lg = _summary_log(maps)
    c = C.check_m1(0, False, lg.summary, lg, saved)
    assert c.ok is ok and text in c.message
    assert bool(c.hints) is not ok


def test_m1_never_initialised_vs_lost_hint():
    lg = _summary_log([])
    assert "never initialised" in C.check_m1(0, False, lg.summary, lg, False).hints[0]
    lg = _summary_log([], extra="init_success=1\nNew Map created with 100 points\n"
                                "IMU is not or recently initialized. Reseting active map...\n"
                                "SYSTEM-> Reseting active map in monocular case\n")
    h = C.check_m1(0, False, lg.summary, lg, False).hints[0]
    assert "lost every map" in h and "before the IMU was fully initialised" in h


@pytest.mark.parametrize("rc,timed_out,summary", [(1, False, True), (None, True, False), (-11, False, False)])
def test_m1_run_that_did_not_finish(rc, timed_out, summary):
    lg = _summary_log([_map()], rc=rc or 0) if summary else parse_log("")
    c = C.check_m1(rc, timed_out, lg.summary, lg, True)
    assert c.passed is False and "did not finish" in c.message and c.values["finished"] is False
    assert "not a recording problem" in c.hints[0] and "mapping/umiproc_slam.log" in c.hints[0]
    if rc == -11:
        assert "crashed (SIGSEGV, exit -11)" in c.hints[0] and "exit -11" in c.message
    else:
        assert "crashed" not in c.hints[0]


# --------------------------------------------------------------------------------------------- many resets after init

def _clip128():
    """The mapping run of real clip 128 (nominal lens), trimmed to its first 12 map resets: its log and the states of
    its trajectory CSV up to the 13th map."""
    lg = read_log(FIX / "map_run_init_resets.log")
    states = np.loadtxt(FIX / "map_run_init_resets_states.csv", delimiter=",", skiprows=1, usecols=2, dtype=np.int64)
    return lg, states


def test_init_resets_on_the_real_clip_128():
    lg, states = _clip128()
    assert len(lg.resets) == 12 and len(lg.of_kind("local_map_failed")) == 12
    assert C.tracked_frames_before_restart(states) == [5, 7, 8, 196, 4, 160, 5, 197, 72, 2, 2, 4]
    e = C.init_resets(lg, states)
    assert (e["resets"], e["dropped_maps"], e["early"], e["suspect"]) == (12, 12, 8, True)
    assert e["source"] == "mapping trajectory" and e["local_map_failed_right_after_init"] == 12
    e = C.init_resets(lg)                                  # without the CSV: the log pattern
    assert (e["source"], e["early"], e["suspect"]) == ("mapping log", 12, True)


def test_m1_m2_hints_point_at_the_camera_model_after_many_early_resets():
    lg, states = _clip128()
    e = C.init_resets(lg, states)
    m1 = C.check_m1(0, False, lg.summary, lg, True, e)
    assert m1.ok and m1.values["init_resets"]["suspect"] is True       # clip 128: one map with 15 keyframes
    m2 = C.check_m2(lg.summary, e)
    assert m2.passed is False and len(m2.hints) == 1
    h = m2.hints[0]
    assert "camera model" in h and "lens" in h and "T_imu_cam" in h and "8 of the 12 new maps" in h
    assert "record longer" not in h.lower() and "Record longer" not in h
    # M1 failing the same way (no map left, or two maps): the same hint, not the recording hints
    for maps in ([], [_map(), _map(1, 5)]):
        s = _summary_log(maps).summary
        c = C.check_m1(0, False, s, lg, True, e)
        assert c.passed is False and "camera model" in c.hints[0] and "record longer" not in c.hints[0].lower()


def test_few_or_late_resets_keep_the_recording_hints():
    lg, _ = _clip128()
    late = np.concatenate([np.ones(5, int)] + [np.r_[np.full(100, 2), 3, np.ones(4, int)] for _ in range(6)]
                          + [np.full(50, 2)])
    e = C.init_resets(lg, late)                            # 12 resets, but every dropped map tracked 100 frames
    assert e["early"] == 0 and e["suspect"] is False
    assert "Record longer" in C.check_m2(lg.summary, e).hints[0]
    lg4 = _summary_log([], extra="init_success=1\nNew Map created with 100 points\nFail to track local map!\n"
                                 "IMU is not or recently initialized. Reseting active map...\n"
                                 "SYSTEM-> Reseting active map in monocular case\n")
    e = C.init_resets(lg4)
    assert (e["resets"], e["early"], e["suspect"]) == (1, 1, False)    # one reset: below the 5 resets
    assert "lost every map" in C.check_m1(0, False, lg4.summary, lg4, False, e).hints[0]


def test_m2_on_the_recorded_runs():
    assert C.check_m2(read_log(FIX / "map_run.log").summary).message == "not finished: VIBA 1, VIBA 2"
    assert C.check_m2(read_log(FIX / "localize_run.log").summary).ok


@pytest.mark.parametrize("m,ok", [(_map(), True), (_map(imu=False, b1=False, b2=False), False),
                                  (_map(b2=False), False), (_map(b1=False), False)])
def test_m2_boundaries(m, ok):
    c = C.check_m2(_summary_log([m]).summary)
    assert c.ok is ok and bool(c.hints) is not ok


def test_m2_judges_the_largest_map_and_needs_one():
    c = C.check_m2(_summary_log([_map(0, 5, b2=False), _map(1, 20)]).summary)
    assert c.ok and c.values["map_id"] == 1
    c = C.check_m2(_summary_log([]).summary)
    assert c.passed is None and not c.ok and "not evaluated" in c.message


# --------------------------------------------------------------------------------------------- M3

def _fork(tracked, n_total=None, keyframes=(), x=None):
    tracked = np.asarray(tracked, bool)
    n = len(tracked)
    t = T0 + DT * np.arange(n, dtype=np.int64)
    Tw = np.tile(np.eye(4), (n, 1, 1))
    if x is not None:
        Tw[:, 0, 3] = x
    Tw[~tracked] = np.nan
    kf = np.zeros(n, bool)
    kf[list(keyframes)] = True
    return ForkTrajectory(path=Path("x.csv"), t0_ns=T0, t_ns=t, state=np.where(tracked, 2, 6), is_lost=~tracked,
                          is_keyframe=kf, posed=tracked, tracked=tracked, T_W_cam=Tw,
                          n_frames_total=n if n_total is None else n_total, anomalies={"stale": 0, "bad_pose": 0})


def _times(n):
    return T0 + DT * np.arange(n, dtype=np.int64)


def _m3(tracked, n_total=None, stopped=False):
    n_total = len(tracked) if n_total is None else n_total
    lg = _summary_log([_map()], extra="INIT_RELOCALIZE success!\npKFCur->mTimeStamp0.1\n")
    s = lg.summary
    if stopped:
        from dataclasses import replace

        s = replace(s, stopped_early=True, stop_reason="max_lost_frames", frames_fed=len(tracked))
    return C.check_m3(0 if not stopped else 3, False, s, lg, _fork(tracked, n_total), _times(n_total))


def test_m3_relocalized_within_half_a_second():
    n = 200
    first_ok = round(0.5e9 / DT)                          # frame 15 is at exactly 0.5 s (15 * 33.33 ms)
    tr = np.ones(n, bool)
    tr[:first_ok] = False
    c = _m3(tr, n)
    assert c.values["first_tracked_s"] == pytest.approx(0.5, abs=1e-6)
    assert c.values["tracked_fraction"] == pytest.approx(185 / 200)
    assert c.passed is False                               # 92.5 % tracked: fails on the fraction only
    assert len(c.hints) == 1 and "lost against the map" in c.hints[0]
    n = 400
    tr = np.ones(n, bool)
    tr[:first_ok] = False                                  # 385/400 = 96.25 %: passes
    assert _m3(tr, n).ok
    tr = np.ones(n, bool)
    tr[:first_ok + 1] = False                              # first tracked at 0.533 s
    c = _m3(tr, n)
    assert c.passed is False and "first half second" in c.hints[0]


def test_m3_fraction_boundary_and_all_frames_count():
    n = 200
    tr = np.ones(n, bool)
    tr[50:60] = False                                      # 190/200 = 95.0 %: passes
    c = _m3(tr)
    assert c.ok and c.values["tracked_fraction"] == pytest.approx(0.95)
    assert c.values["longest_lost"][0]["frames"] == 10
    assert c.values["longest_lost"][0]["start_s"] == pytest.approx(50 * DT / 1e9)
    tr[60] = False                                         # 189/200 = 94.5 %
    assert _m3(tr).passed is False
    c = _m3(np.ones(150, bool), n_total=200, stopped=True)  # stopped early: the 50 frames never fed are lost
    assert c.passed is False and c.values["tracked_fraction"] == pytest.approx(0.75)
    assert "stopped after 150 frames" in c.message


def test_m3_never_tracked_and_unreadable():
    c = _m3(np.zeros(50, bool))
    assert c.passed is False and c.values["first_tracked_s"] is None and "never" in c.message
    lg = _summary_log([_map()])
    c = C.check_m3(0, False, lg.summary, lg, None, _times(10))
    assert c.passed is False and "could not be read" in c.message


# --------------------------------------------------------------------------------------------- M4

ORIENT_OK = {"ok": True, "windows": 11, "max_deg": 0.5, "median_deg": 0.1, "max_at_s": 3.0, "reason": None}
ORIENT_BAD = {"ok": False, "windows": 11, "max_deg": 4.2, "median_deg": 2.0, "max_at_s": 12.5, "reason": None}
ORIENT_SHORT = {"ok": False, "windows": 3, "max_deg": 0.2, "median_deg": 0.1, "max_at_s": 0.0,
                "reason": "not enough tracked span: 3 window(s) of 2.5 s (need 4, about 10 s tracked)"}


def _fin(g_ok=True, scale="ok", s=1.0, mean_world=(0.0, 0.0, 9.81)):
    return SimpleNamespace(diagnostics={
        "gravity_check": {"ok": g_ok, "g_norm": 9.8, "angle_to_up_deg": 0.0},
        "gravity_alignment": {"applied": True, "angle_deg": 12.0, "mean_world_before": list(mean_world)},
        "scale_check": {"status": scale, "scale": s, "std_err": 0.01, "excitation_rms": 0.5,
                        "problems": [] if scale == "ok" else [f"{scale}: x"]}})


@pytest.mark.parametrize("fin,speed,orient,ok,hint", [
    (_fin(), (3.0, 1.0), ORIENT_OK, True, None),
    (_fin(), (3.0001, 1.0), ORIENT_OK, False, "move slower"),
    (_fin(g_ok=False), (1.0, 1.0), ORIENT_OK, False, "gravity"),
    (_fin(scale="scale_mismatch", s=1.2), (1.0, 1.0), ORIENT_OK, False, "off by 20 %"),
    (_fin(scale="scale_unobservable", s=None), (1.0, 1.0), ORIENT_OK, False, "could not be measured"),
    (_fin(), (None, None), ORIENT_OK, False, "move slower"),
    (_fin(), (1.0, 1.0), ORIENT_BAD, False, "disagrees with gravity by up to 4.2 deg"),
    (_fin(), (1.0, 1.0), ORIENT_SHORT, False, "not enough tracked span"),
    (_fin(), (1.0, 1.0), None, False, "could not be compared"),
])
def test_m4_boundaries(fin, speed, orient, ok, hint):
    c = C.check_m4(fin, None, speed, orient)
    assert c.ok is ok
    assert "orientation vs gravity" in c.message
    if hint:
        assert any(hint in h for h in c.hints)


def test_m4_without_a_trajectory():
    c = C.check_m4(None, "backend produced 1 poses (need >= 2)", (None, None), None)
    assert c.passed is False and "need >= 2" in c.message


# --------------------------------------------------------------------------------------------- M4: orientation

@pytest.fixture(scope="module")
def T_imu_cam(tmp_path_factory):
    from umiproc.sensors import load_sensors

    return load_sensors(tmp_path_factory.mktemp("root"), "rig_pi_v3", log=lambda *a, **k: None) \
        .rig.cameras["ultrawide"].T_imu_cam


def _orient(m, T_W_cam, T_imu_cam_used):
    traj = C.imu_trajectory(fork_trajectory(m["t_cam_ns"], T_W_cam), T_imu_cam_used)
    return C.orientation_consistency(traj, m["imu_t_ns"], m["acc_imu"], t_ref_ns=int(m["t_cam_ns"][0]))


def test_orientation_consistent_motion_passes(T_imu_cam):
    m = rig_motion(T_imu_cam)                              # 30 s, yaw swinging +-92 deg
    o = _orient(m, m["T_W_cam"], T_imu_cam)
    assert o["ok"] and o["windows"] == 11 and o["windows_skipped"] == 0
    assert o["max_deg"] < 1.0 and o["median_deg"] < 0.5 and len(o["window_deg"]) == 11
    assert np.degrees(np.arccos(np.clip(np.dot(o["up_world"], m["up_W"]), -1, 1))) < 0.1


def test_orientation_tolerates_a_20_mg_accelerometer_bias(T_imu_cam):
    for bias in ((0.196, 0.0, 0.0), (0.139, 0.139, 0.0)):  # 20 mg (the LSM6DSOX's typical zero-g offset)
        m = rig_motion(T_imu_cam, accel_bias=bias)
        o = _orient(m, m["T_W_cam"], T_imu_cam)
        assert o["ok"] and o["max_deg"] < 2.5


@pytest.mark.parametrize("axis", ["x", "y"])
def test_orientation_a_5_deg_camera_imu_rotation_error_fails(T_imu_cam, axis):
    from scipy.spatial.transform import Rotation

    m = rig_motion(T_imu_cam)
    wrong = np.array(T_imu_cam, float)
    wrong[:3, :3] = wrong[:3, :3] @ Rotation.from_euler(axis, 5.0, degrees=True).as_matrix()
    o = _orient(m, m["T_W_cam"], wrong)                    # the IMU was generated with the true T_imu_cam
    assert o["ok"] is False and o["windows"] == 11 and o["max_deg"] > 4.0
    c = C.check_m4(_fin(), None, (1.0, 1.0), o)
    assert c.passed is False and any("T_imu_cam" in h for h in c.hints)
    m0 = rig_motion(T_imu_cam, yaw_amp=0.0)                # without turning, the same error stays invisible
    assert _orient(m0, m0["T_W_cam"], wrong)["max_deg"] < 1.0


def test_orientation_slow_drift_fails(T_imu_cam):
    from scipy.spatial.transform import Rotation

    m = rig_motion(T_imu_cam)
    t = (m["t_cam_ns"] - m["t_cam_ns"].mean()) * 1e-9
    D = Rotation.from_rotvec(np.outer(t * np.radians(0.3), [1.0, 0.0, 0.0])).as_matrix()   # 0.3 deg/s, 9 deg in all
    T = m["T_W_cam"].copy()
    T[:, :3, :3] = np.einsum("nij,njk->nik", D, T[:, :3, :3])
    o = _orient(m, T, T_imu_cam)
    assert o["ok"] is False and o["max_deg"] > 3.0
    assert o["max_at_s"] in (0.0, 25.0)                    # the first or the last window is the farthest


def test_orientation_needs_four_windows(T_imu_cam):
    m = rig_motion(T_imu_cam, duration_s=9.0)
    o = _orient(m, m["T_W_cam"], T_imu_cam)
    assert o["ok"] is False and o["windows"] == 3 and "not enough tracked span" in o["reason"]
    m = rig_motion(T_imu_cam, duration_s=10.1)
    assert _orient(m, m["T_W_cam"], T_imu_cam)["windows"] == 4
    # a long loss of tracking inside the span: its windows have no samples and are skipped
    m = rig_motion(T_imu_cam, duration_s=20.0)
    f = fork_trajectory(m["t_cam_ns"], m["T_W_cam"])
    keep = np.ones(len(m["t_cam_ns"]), bool)
    keep[150:450] = False                                  # 10 s lost: 0.2 s pose gap rule drops those samples
    from umiproc.dp.geometry.trajectory import Trajectory

    full = C.imu_trajectory(f, T_imu_cam)
    part = Trajectory.from_T(full.t_ns[keep], full.T()[keep], frame="imu", world="slam_first_keyframe")
    o = C.orientation_consistency(part, m["imu_t_ns"], m["acc_imu"])
    assert o["windows"] + o["windows_skipped"] == 7 and o["windows_skipped"] >= 3 and o["windows"] == 3


def test_max_speed_uses_consecutive_tracked_frames_only():
    x = np.arange(10) * 0.1                                # 0.1 m per frame = 3.0 m/s at 30 fps
    tr = np.ones(10, bool)
    v, at = C.max_speed(_fork(tr, x=x))
    assert v == pytest.approx(0.1 / (DT * 1e-9)) and v == pytest.approx(3.0, rel=1e-6)
    x2 = x.copy()
    x2[5:] += 1.0                                          # a 1 m jump between frames 4 and 5 ...
    tr[5] = False                                          # ... across a lost frame: not a pair
    x2[5] = np.nan
    v, at = C.max_speed(_fork(tr, x=x2))
    assert v == pytest.approx(3.0, rel=1e-6)
    tr[5] = True
    x2[5] = x[5] + 1.0
    v, at = C.max_speed(_fork(tr, x=x2))
    assert v > 30 and at == pytest.approx(4 * DT / 1e9)
    assert C.max_speed(_fork(np.zeros(5, bool))) == (None, None)


def test_imu_trajectory_is_tracked_rows_in_the_imu_frame():
    tr = np.array([1, 0, 1, 1], bool)
    f = _fork(tr, x=np.arange(4) * 0.01)
    T_imu_cam = np.eye(4)
    T_imu_cam[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    T_imu_cam[:3, 3] = [0.01, 0.02, 0.03]
    t = C.imu_trajectory(f, T_imu_cam)
    assert t.frame == "imu" and list(t.t_ns) == list(f.t_ns[tr])
    np.testing.assert_allclose(t.T(), f.T_W_cam[tr] @ np.linalg.inv(T_imu_cam), atol=1e-12)


# --------------------------------------------------------------------------------------------- M5

def _stats(n):
    st = SightingStats(keyframes=n, keyframes_with_tag=n)
    st.sightings = [Sighting(k, k, np.eye(4), 0.5, 0.1, 2) for k in range(n)]
    return st


def _tilted(deg):
    R = np.eye(4)
    a = np.radians(deg)
    R[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    return R


def _agg(std, normal_deg=0.0, chosen_deg=None):
    R = _tilted(normal_deg if chosen_deg is None else chosen_deg)
    return {"T_world_tag": R, "kept": np.ones(20, bool), "std_m": np.asarray(std, float), "threshold_m": 0.01,
            "index": 0, "mean": np.zeros(3), "normal_median": _tilted(normal_deg)[:3, 2], "normal_chosen": R[:3, 2]}


UP = np.array([0.0, 0.0, 1.0])


@pytest.mark.parametrize("n,std,normal,up,ok", [
    (20, [0.01, 0, 0], 5.0, UP, True),                     # every limit exactly
    (19, [0.0, 0, 0], 0.0, UP, False),
    (20, [0.0101, 0, 0], 0.0, UP, False),
    (20, [0, 0, 0.0101], 0.0, UP, False),
    (20, [0.006, 0.006, 0.006], 0.0, UP, True),            # per axis 0.6 cm: passes (the 3-D 1.04 cm is reported)
    (20, [0.01, 0.01, 0.01], 0.0, UP, True),               # each axis exactly 1 cm
    (20, [0.0, 0, 0], 5.01, UP, False),
    (20, [0.0, 0, 0], 0.0, None, False),                   # gravity unknown
])
def test_m5_boundaries(n, std, normal, up, ok):
    c = C.check_m5(_stats(n), _agg(std, normal), up, frames_with_tag=100, frames_total=300)
    assert c.ok is ok
    assert bool(c.hints) is not ok
    if c.values.get("normal_to_up_deg") is not None:
        assert c.values["normal_to_up_deg"] == pytest.approx(normal, abs=1e-6)
    assert c.values["spread_3d_m"] == pytest.approx(np.linalg.norm(std))
    assert c.values["spread_max_axis_m"] == pytest.approx(max(std))
    assert "spread x/y/z" in c.message and "3-D" in c.message


def test_m5_judges_the_median_normal_and_reports_the_chosen_one():
    c = C.check_m5(_stats(25), _agg([0, 0, 0], normal_deg=1.0, chosen_deg=7.0), UP, 100, 300)
    assert c.ok and c.values["normal_to_up_deg"] == pytest.approx(1.0, abs=1e-6)
    assert c.values["chosen_normal_to_up_deg"] == pytest.approx(7.0, abs=1e-6)
    assert "the chosen sighting's 7.0 deg" in c.message
    c = C.check_m5(_stats(25), _agg([0, 0, 0], normal_deg=6.0, chosen_deg=0.0), UP, 100, 300)
    assert c.passed is False and "tilted 6.0 deg" in c.hints[0]


def test_m5_median_normal_ignores_a_few_bad_sightings():
    from umiproc.map.tag import aggregate_sightings

    T = np.stack([_tilted(a) for a in np.r_[np.full(17, 0.5), [12.0, 15.0, 20.0]]])
    agg = aggregate_sightings(T)                           # identical positions: every sighting is kept
    assert agg["kept"].all()
    nm = np.degrees(np.arccos(np.clip(agg["normal_median"] @ UP, -1, 1)))
    mean_z = T[agg["kept"], :3, 2].mean(0)
    assert nm < 1.0 < np.degrees(np.arccos(mean_z[2] / np.linalg.norm(mean_z)))
    assert np.isclose(np.linalg.norm(agg["normal_median"]), 1.0)
    np.testing.assert_allclose(agg["normal_chosen"], agg["T_world_tag"][:3, 2])


def test_m5_upside_down_tag_fails_and_hints():
    c = C.check_m5(_stats(25), _agg([0, 0, 0], 180.0), UP, 100, 300)
    assert c.passed is False and c.values["normal_to_up_deg"] == pytest.approx(180.0)
    c = C.check_m5(SightingStats(), None, UP, 0, 300)
    assert c.passed is False and "never detected" in c.hints[0]
    st = SightingStats(keyframes=40, keyframes_with_tag=30, rejected_distance=6, rejected_centre=10)
    st.sightings = [Sighting(k, k, np.eye(4), 0.5, 0.1, 2) for k in range(14)]
    c = C.check_m5(st, _agg([0, 0, 0]), UP, 120, 300)
    assert c.passed is False and "farther than 2 m" in c.hints[0] and "image centre" in c.hints[0]


def test_up_direction():
    assert np.allclose(C.up_direction(_fin(mean_world=(0, 0, 2.0))), UP)
    assert C.up_direction(None) is None
    assert C.up_direction(SimpleNamespace(diagnostics={})) is None


# --------------------------------------------------------------------------------------------- diagnostics

def test_pacing_warning_below_90_percent_of_the_recording_rate():
    def summ(rate):
        lg = _summary_log([_map()])
        from dataclasses import replace

        return replace(lg.summary, pacing=replace(lg.summary.pacing, feed_rate_hz=rate, recording_rate_hz=30.0))

    assert C.pacing_warnings("mapping run", summ(27.0)) == []
    w = C.pacing_warnings("mapping run", summ(26.99))
    assert len(w) == 1 and "slower than real time" in w[0] and "fed 26.99 of 30.00" in w[0]
    assert C.pacing_warnings("x", None) == []


def test_lost_stretches():
    tr = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0], bool)
    s = C.lost_stretches(tr, _times(10), T0)
    assert [x["frames"] for x in s] == [3, 2, 1]
    assert s[0]["start_s"] == pytest.approx(7 * DT / 1e9)


def test_every_check_reports_json():
    import json

    for c in (C.check_m2(_summary_log([_map()]).summary), C.not_evaluated("M5", "x"),
              C.check_m5(_stats(20), _agg([0, 0, 0]), UP, 1, 1)):
        d = c.to_json()
        json.dumps(d)
        assert set(d) == {"id", "title", "passed", "values", "limits", "message", "hints"}
