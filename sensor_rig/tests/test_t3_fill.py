"""T3: fill mode made align-exact (F2, checked with the REAL iPhUMI align functions), gap aborts (F6), VIO motion
checks (F5), the rotation-vector guard (F9), locked configs (F7), --mock-replay time offsets, and the calibration tag
check (both cv2.aruco APIs)."""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sensor_rig.export.exporter import ExportAbort, ExportOptions, _gap_abort, choose_world_yaw, export_episode
from sensor_rig.export.grid import fill_slots, lag_stats
from sensor_rig.pose.stage import TrajectoryImportError, import_trajectory
from sensor_rig.pose.trajectory import Trajectory, motion_consistency_check, save_tum
from sensor_rig.raw_episode import RawEpisode
from sensor_rig.record import tagcheck
from sensor_rig.timefmt import iso_ms, parse_iso
from fixtures_sim import *  # noqa: F401,F403
from wpa_helpers import MONO_BASE, make_raw_episode, synth_imu, truth_pose

REPO = Path(__file__).resolve().parents[2]
IPHUMI = REPO / "iPhUMI" / "python_package"


@pytest.fixture(scope="module")
def align():
    """The real iPhUMI align module (no copy of its logic)."""
    if str(IPHUMI) not in sys.path:
        sys.path.insert(0, str(IPHUMI))
    pytest.importorskip("omegaconf")
    from iphumi.common.timecode_util import datetime_fromisoformat
    from iphumi.demonstration_processing.process_stages import align as A

    return A, datetime_fromisoformat


def align_indices(align, pose_strings: list[str], rgb_strings: list[str]):
    """What align_multi_iphone_data does for one side: sample_times from the pose strings, then rgb indices."""
    A, fromiso = align
    pose_times = [fromiso(x).timestamp() for x in pose_strings]
    sample_times = np.arange(pose_times[0], pose_times[-1], 1.0 / 60, dtype=np.float64)
    return sample_times, A._compute_frame_indices(rgb_strings, sample_times)


def _stream(n, fps=30.0, jitter_ns=0, ppm=0.0, seed=0, t0=MONO_BASE + 7_654_321):
    rng = np.random.default_rng(seed)
    t = t0 + np.round(np.arange(n) * 1e9 / fps * (1 + ppm * 1e-6)).astype(np.int64)
    return t + (rng.normal(0, jitter_ns, n).astype(np.int64) if jitter_ns else 0)


# ------------------------------------------------------------------------------------------------ F2

@pytest.mark.parametrize("wall", [1_792_000_000_123_456_789, 1_792_000_000_000_000_000, 1_792_000_000_999_999_999])
@pytest.mark.parametrize("jitter", [0, 300_000])
def test_fill_strings_are_align_exact(align, wall, jitter):
    t = _stream(330, jitter_ns=jitter, seed=jitter)
    fs = fill_slots(t, int(t[0]), int(t[-1]), wall)
    strings = [iso_ms(int(x)) for x in fs.string_unix_ns]
    assert strings == [iso_ms(int(x) + wall) for x in fs.string_ns]
    samples, idx = align_indices(align, strings, strings)
    assert len(samples) == len(fs) - 1                          # align drops the last string (end of its range)
    assert np.array_equal(idx, np.arange(len(samples)))         # rgb_frame_idx == k
    # lag of the image align pairs with each sample (sample times from align itself)
    sample_rig = np.round(samples * 1e9).astype(np.int64) - wall
    assert np.abs(sample_rig - fs.sample_ns[:-1]).max() < 2_000     # align's float samples = our sample times
    lag = (t[fs.src_idx[idx]] - sample_rig) / 1e6
    # zero mean needs both neighbours at near-midway slots, so the worst |lag| is half a frame interval plus the
    # slot's distance from midway: the residual grid phase (<= 1/3 ms: the grid start is a ms boundary and
    # 1/60 s = 16 2/3 ms) -> <= 17.0 ms on a clean stream; + ~1 ms with 0.3 ms rms stamp jitter
    bound = 17.0 + (1.0 if jitter else 0.0)
    assert abs(lag.mean()) < 2.0 and np.abs(lag).max() <= bound, (lag.mean(), np.abs(lag).max())
    st = lag_stats(t, fs)
    assert st["obs_pairs_same_frame"] == 0 and abs(st["lag_ms_mean"]) < 2.0


def test_fill_drifting_clock_stays_zero_mean(align):
    # 360 ppm drift (what the iPhone-derived 30 fps sims show): the frames walk ~4 ms across the 60 Hz grid
    t = _stream(330, ppm=360.0)
    wall = 1_792_000_000_456_000_000
    fs = fill_slots(t, int(t[0]), int(t[-1]), wall)
    strings = [iso_ms(int(x)) for x in fs.string_unix_ns]
    _, idx = align_indices(align, strings, strings)
    assert np.array_equal(idx, np.arange(len(fs) - 1))
    st = lag_stats(t, fs)
    assert abs(st["lag_ms_mean"]) < 2.0 and st["obs_pairs_same_frame"] == 0
    # zero mean costs at most half a frame interval + the phase excursion (~2 ms) + the 2 ms allowance
    assert st["lag_ms_max_abs"] <= 1e3 / 30 / 2 + 2.5


def test_fill_export_through_real_align(align, sim_export):
    se = sim_export("main30")
    assert se.result.status == "exported" and se.result.grid_mode == "fill"
    j = se.json
    _, idx = align_indices(align, j["poseTimes"], j["rgbTimes"])
    assert np.array_equal(idx, np.arange(len(idx)))
    uw = [s for s in j["ultrawideRGBTimes"] if s]
    assert set(uw) <= set(j["rgbTimes"])
    lag = se.result.lag
    assert abs(lag["lag_ms_mean"]) < 2.0 and lag["obs_pairs_same_frame"] == 0


def test_native_strings_unchanged(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, main_fps=60)
    r = export_episode(raw, tmp_path / "out")
    assert r.grid_mode == "native" and r.lag == {}


# ------------------------------------------------------------------------------------------------ F6

def test_gap_abort_names_gap_and_time():
    t = MONO_BASE + np.round(np.arange(60) * 1e9 / 30).astype(np.int64)
    _gap_abort(t, int(t[0]), 0.10, "main")                                   # healthy: no abort
    t2 = np.concatenate([t[:30], t[34:]])                                    # 4 frames lost: 0.167 s gap at 0.97 s
    with pytest.raises(ExportAbort) as e:
        _gap_abort(t2, int(t[0]), 0.10, "main")
    assert e.value.reason == "main_gap" and "0.167 s" in e.value.detail and "0.97 s" in e.value.detail
    _gap_abort(t2, int(t[0]), None, "main")                                  # disabled


def test_export_aborts_on_main_gap(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, main_fps=30)
    ep = RawEpisode(raw)
    lines = (raw / "cam_main" / "data.csv").read_text().splitlines()
    (raw / "cam_main" / "data.csv").write_text("\n".join(lines[:30] + lines[35:]) + "\n")   # 5 frames missing
    r = export_episode(raw, tmp_path / "out")
    assert r.status == "aborted" and r.reason == "main_gap", r
    assert any("gap between main frames" in w for w in r.warnings)
    r2 = export_episode(raw, tmp_path / "out2", ExportOptions(max_main_gap_s=None))
    assert r2.status == "exported"
    del ep


# ------------------------------------------------------------------------------------------------ F5

def _rig_imu(test_rig):
    t = np.arange(0, 3.0, 1 / 400)
    g, a = synth_imu(t)
    return (MONO_BASE + np.round(t * 1e9)).astype(np.int64), g, a


def test_motion_check_passes_consistent_trajectory(test_rig):
    ti, g, _ = _rig_imu(test_rig)
    tt = MONO_BASE + np.round(np.arange(0.2, 2.8, 1 / 30) * 1e9).astype(np.int64)
    traj = Trajectory.from_T(tt, truth_pose((tt - MONO_BASE) * 1e-9), frame="imu")
    m = motion_consistency_check(traj, ti, g, test_rig)
    assert m["ok"] and m["gyro_resid_rms"] < 0.02, m


def test_motion_check_catches_rotation_and_translation_jumps(test_rig):
    ti, g, _ = _rig_imu(test_rig)
    tt = MONO_BASE + np.round(np.arange(0.2, 2.8, 1 / 30) * 1e9).astype(np.int64)
    T = truth_pose((tt - MONO_BASE) * 1e-9)
    Tr = T.copy()
    Tr[40:, :3, :3] = Rotation.from_euler("z", 8, degrees=True).as_matrix() @ Tr[40:, :3, :3]   # relocalisation
    m = motion_consistency_check(Trajectory.from_T(tt, Tr, frame="imu"), ti, g, test_rig)
    assert not m["ok"] and any("residual max" in p for p in m["problems"]), m
    Tt = T.copy()
    Tt[50:, :3, 3] += [0.3, 0, 0]                                                                 # 30 cm jump
    m = motion_consistency_check(Trajectory.from_T(tt, Tt, frame="imu"), ti, g, test_rig)
    assert not m["ok"] and any("translation jump" in p for p in m["problems"]), m


def test_pose_import_refuses_jump_unless_forced(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, main_fps=30)
    ep = RawEpisode(raw)
    tt = ep.camera_times("main") + int(test_rig.capture_time_ns("main", 0))
    T = truth_pose((tt - (MONO_BASE + 20_000_000)) * 1e-9)
    T[40:, :3, 3] += [0.4, 0.0, 0.0]
    f = tmp_path / "jump.tum"
    save_tum(Trajectory.from_T(tt, T, frame="imu"), f)
    with pytest.raises(TrajectoryImportError, match="translation jump"):
        import_trajectory(ep, f, frame="imu")
    info = import_trajectory(ep, f, frame="imu", force=True)
    assert info["forced"] and not info["motion_check"]["ok"]
    ok = tmp_path / "ok.tum"
    save_tum(Trajectory.from_T(tt, truth_pose((tt - (MONO_BASE + 20_000_000)) * 1e-9), frame="imu"), ok)
    info = import_trajectory(ep, ok, frame="imu")
    assert info["motion_check"]["ok"] and info["time_offset_applied_ns"] == 0


def test_pose_import_time_offset(tmp_path, test_rig):
    """A trajectory on another clock (the replay source) lines up once the offset is applied."""
    raw = make_raw_episode(tmp_path / "raw", test_rig, main_fps=30)
    ep = RawEpisode(raw)
    tt = ep.camera_times("main") + int(test_rig.capture_time_ns("main", 0))
    off = 123_456_789_000
    src = tmp_path / "src.tum"
    save_tum(Trajectory.from_T(tt - off, truth_pose((tt - (MONO_BASE + 20_000_000)) * 1e-9), frame="imu"), src)
    with pytest.raises(TrajectoryImportError, match="overlaps"):
        import_trajectory(ep, src, frame="imu")
    info = import_trajectory(ep, src, frame="imu", time_offset_ns=off)
    assert info["main_overlap_fraction"] > 0.95 and info["time_offset_source"] == "argument"
    meta = json.loads((raw / "meta.json").read_text())
    meta["sim"] = {"replay_offset_ns": off}
    (raw / "meta.json").write_text(json.dumps(meta))
    info = import_trajectory(RawEpisode(raw), src, frame="imu")                   # auto: from meta.sim
    assert info["time_offset_applied_ns"] == off and info["time_offset_source"] == "meta.sim.replay_offset_ns"


# ------------------------------------------------------------------------------------------------ F9

def test_choose_world_yaw_keeps_angles_small():
    R = Rotation.from_euler("y", np.linspace(150, 179.5, 50), degrees=True) * Rotation.from_euler("x", 10, degrees=True)
    T = np.tile(np.eye(4), (50, 1, 1))
    T[:, :3, :3] = R.as_matrix()
    assert choose_world_yaw(T[:10], 170.0) == 0.0          # first poses only reach ~156 deg: nothing to do
    yaw = choose_world_yaw(T, 170.0)
    assert yaw != 0.0
    T2 = Rotation.from_euler("y", yaw, degrees=True).as_matrix() @ T[:, :3, :3]
    assert np.degrees(Rotation.from_matrix(T2).magnitude()).max() < 170.0


def test_export_world_yaw_warn_and_fix(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, main_fps=60)        # this trajectory reaches 179.7 deg
    r = export_episode(raw, tmp_path / "a")
    assert r.status == "exported" and r.world_yaw_extra_deg == 0.0
    assert any("fix-world-yaw" in w for w in r.warnings)
    r2 = export_episode(raw, tmp_path / "b", ExportOptions(fix_world_yaw=True))
    assert r2.status == "exported" and r2.world_yaw_extra_deg > 0
    Ta = np.asarray(json.loads((r.out_dir / f"{r.name}.json").read_text())["poseTransforms"], float)
    Tb = np.asarray(json.loads((r2.out_dir / f"{r2.name}.json").read_text())["poseTransforms"], float)
    assert np.degrees(Rotation.from_matrix(Tb[:, :3, :3]).magnitude()).max() < 170.0
    rel_a = np.linalg.inv(Ta[:-5]) @ Ta[5:]                                  # relative motion is unchanged
    rel_b = np.linalg.inv(Tb[:-5]) @ Tb[5:]
    assert np.abs(rel_a - rel_b).max() < 1e-5


# ------------------------------------------------------------------------------------------------ F7

@pytest.mark.parametrize("cfg", ["rig_default.yaml", "rig_jetson_imx219_2p1_1p55.yaml"])
def test_configs_locked_and_covered(cfg, tmp_path):
    from sensor_rig import doctor as D
    from sensor_rig.rig_config import load_rig

    rig = load_rig(Path(D.__file__).parent / "configs" / cfg)
    for c in rig.cameras.values():
        a = c.argus
        assert a["aelock"] and a["awblock"]
        lo, hi = a["exposure_time_range_ns"]
        assert lo == hi <= 4_000_000 and a["gain_range"][0] == a["gain_range"][1]
    rep = D.run_doctor(cfg, D.DoctorOptions(mock=True, quick=True, sensors=False), rig=rig)
    assert {r.status for r in rep.results if r.name in ("exposure/WB lock", "exposure time")} == {D.PASS}
    assert all(v >= 0.99 for v in rig.coverage_report().values())


# ------------------------------------------------------------------------------------------------ replay offsets

def test_playlist_replay_offset_and_pose_alignment(tmp_path, test_rig):
    """collect --mock-replay: one constant source->live offset per episode, stored in meta.sim; the source
    trajectory imports onto the re-recorded episode with it, and the frames are the source frames re-stamped."""
    from sensor_rig.drivers.replay_playlist import PlaylistCamera, PlaylistImu, ReplayPlaylist
    from sensor_rig.record.collect import CollectOptions, CollectSession, QueueInput
    from sensor_rig.record.monitor import LiveAbortConfig
    from sensor_rig.record.recorder import Recorder

    rig = test_rig.replace(cameras={**test_rig.cameras,
                                    "main": dataclasses.replace(test_rig.cameras["main"], fps=30.0)})
    src_root = tmp_path / "src"
    cal = make_raw_episode(src_root, rig, kind="gripper_calibration", main_fps=30, duration_s=3.5, rid="11111")
    dem = make_raw_episode(src_root, rig, main_fps=30, duration_s=2.5, rid="22222")
    pl = ReplayPlaylist([cal, dem])
    cams = {c: PlaylistCamera(pl, c) for c in ("main", "ultrawide")}
    rec = Recorder(rig, tmp_path / "raw", cameras=cams, imu=PlaylistImu(pl), warmup_s=1.0, keep_aborted=True,
                   live_abort=LiveAbortConfig(all_cameras_fps=True, max_frame_gap_s=0.25))
    inp = QueueInput()
    lines = []

    class Driver(QueueInput):
        def get(self, timeout_s):
            # press Enter whenever collect waits idle; quit after cal + 1 demo
            done = sum(1 for ln in lines if " KEPT" in ln or "REJECTED" in ln)
            if sess.active is None:
                return "quit" if done >= 2 or len(sess.episodes) > 4 else ("toggle" if not lines or lines[-1].startswith("->") else None)
            import time as _t
            _t.sleep(timeout_s)
            return None

    sess = CollectSession(rig, tmp_path / "raw", CollectOptions(session="rplay", tag_check=False), rec, Driver(),
                          out=lines.append, replay=pl)
    assert sess.run() == 0, "\n".join(lines)
    assert sum("replayed source finished" in ln for ln in lines) == 2
    eps = sorted((tmp_path / "raw").glob("20*/*"))
    kinds = [RawEpisode(p).kind for p in eps]
    assert sorted(kinds) == ["demonstration", "gripper_calibration"], "\n".join(lines)
    for p in eps:
        ep = RawEpisode(p)
        sim = ep.meta["sim"]
        src = RawEpisode(Path(sim["replay_source_dir"]))
        assert src.kind == ep.kind
        off = int(sim["replay_offset_ns"])
        for cam in ("main", "ultrawide"):                         # every source frame, re-stamped by one offset
            assert np.array_equal(ep.camera_times(cam), src.camera_times(cam) + off)
        if ep.kind == "demonstration":
            info = import_trajectory(ep, src.trajectory_path(), frame="imu")
            assert info["time_offset_applied_ns"] == off and info["main_overlap_fraction"] > 0.95
            assert info["motion_check"]["ok"] and info["gravity_check"]["ok"]
    del inp


# ------------------------------------------------------------------------------------------------ tag check

def _marker_image(ids=(0, 1)):
    ar = cv2.aruco
    d = ar.getPredefinedDictionary(ar.DICT_4X4_50)
    img = np.full((480, 640), 255, np.uint8)
    for k, i in enumerate(ids):
        m = ar.generateImageMarker(d, i, 120) if hasattr(ar, "generateImageMarker") else ar.drawMarker(d, i, 120)
        img[180:300, 100 + 260 * k: 220 + 260 * k] = m
    return img


def test_detect_markers_real_cv2():
    assert tagcheck.detect_marker_ids(_marker_image((0, 1))) == [0, 1]
    assert tagcheck.both_tags([0, 1]) and tagcheck.both_tags([6, 7, 2]) and not tagcheck.both_tags([1, 2])


class _FakeAruco:
    """Records which API path was used; returns ids 0 and 1."""

    DICT_4X4_50 = 0

    def __init__(self, new_api: bool):
        self.calls = []
        if new_api:
            self.ArucoDetector = self._detector
            self.DetectorParameters = lambda: "params-new"
        else:
            self.DetectorParameters_create = lambda: "params-old"
            self.detectMarkers = self._detect_old
            self.Dictionary_get = lambda i: f"dict{i}"

    def getPredefinedDictionary(self, i):
        return f"dict{i}"

    def _detector(self, d, p):
        calls = self.calls

        class Det:
            def detectMarkers(self, img):
                calls.append(("new", d, p))
                return [], np.array([[0], [1]]), []
        return Det()

    def _detect_old(self, img, d, parameters=None):
        self.calls.append(("old", d, parameters))
        return [], np.array([[1], [0]]), []


@pytest.mark.parametrize("new_api", [True, False])
def test_detect_markers_both_cv2_apis(new_api):
    fa = _FakeAruco(new_api)
    assert tagcheck.detect_marker_ids(np.zeros((10, 10), np.uint8), aruco=fa) == [0, 1]
    assert fa.calls[0][0] == ("new" if new_api else "old")
    assert fa.calls[0][2] == ("params-new" if new_api else "params-old")


def test_calibration_tag_check_on_sim_calibration(sim_export):
    se = sim_export("calib")                                       # simulated from a real iPhone calibration
    r = tagcheck.calibration_tag_check(se.raw.path)
    assert r["ok"] and r["both"] >= 1 and {0, 1} <= set(r["ids_seen"]), r


def test_calibration_tag_check_rejects_tagless(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, kind="gripper_calibration")
    r = tagcheck.calibration_tag_check(raw)
    assert not r["ok"] and r["both"] == 0 and r["frames"] == 10
