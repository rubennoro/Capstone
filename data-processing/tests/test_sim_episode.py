"""dataproc.sim.episode / render: contract validity, geometry, IMU, time stamps, ArUco widths, determinism, faults,
and the ``dataproc sim`` CLI. Uses the ``--fast`` rig (main 320x180, ultrawide 410x308) and 4 s episodes."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataproc import cli
from dataproc.config import load_rig
from dataproc.geometry.camera_model import VirtualPinhole, build_remap
from dataproc.geometry.transforms import make_T
from dataproc.raw_episode import VERIFIED, RawEpisode
from dataproc.sim.episode import (EXPOSURE_NS, TRUTH_FILE, TRUTH_VERSION, fast_rig, generate_episode,
                                  next_episode_number, read_truth)
from dataproc.sim.imu_synth import G_WORLD
from dataproc.sim.render import CameraRenderer, Quad, build_scene, camera_model, quad_half_extent
from dataproc.sim.trajectory import camera_base_rotation, make_trajectory

RIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"
DUR, PRE = 4.0, 2.0


@pytest.fixture(scope="module")
def rig():
    return fast_rig(RIG_PATH)[1]


@pytest.fixture(scope="module")
def demo(rig, tmp_path_factory):
    out = tmp_path_factory.mktemp("sim") / "s1_episode_000007"
    generate_episode(out, rig, "demonstration", seed=11, duration_s=DUR, preroll_s=PRE, session_id="s1",
                     episode_number=7)
    return out


def _detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- format

def test_episode_passes_the_contract_validator(demo, rig):
    ep = RawEpisode.open(demo)
    rep = ep.validate(decode_videos=True)
    assert rep.problems == (), [p.to_dict() for p in rep.problems]
    m = ep.meta
    assert (m.session_id, m.episode_number, m.rig_id, m.clock) == ("s1", 7, rig.rig_id, "CLOCK_MONOTONIC")
    assert m.cameras["cam0"].resolution == (320, 180) and m.cameras["cam1"].resolution == (410, 308)
    assert m.cameras["cam0"].frame_count == 240 and m.cameras["cam1"].frame_count == 120
    assert ep.is_verified and ep.is_complete
    # button interval: after the pre-roll, before the end, covered by every stream
    t0 = int(ep.imu.t_ns[0])
    assert m.button_start_ns == t0 + int(PRE * 1e9) and m.button_stop_ns == t0 + int((DUR - 0.5) * 1e9)
    for cam in ("cam0", "cam1"):
        ft = ep.frames(cam)
        assert ft.t_ns[0] < m.button_start_ns and ft.t_ns[-1] > m.button_stop_ns
    assert TRUTH_FILE in (demo / "MANIFEST.sha256").read_text()


def test_truth_file(demo, rig):
    tr = read_truth(demo)
    assert tr["version"] == TRUTH_VERSION and tr["rig_id"] == rig.rig_id and tr["seed"] == 11
    t = tr["t_ns"]
    assert t.dtype == np.int64 and np.all(np.diff(t) == 1_000_000)
    n = len(t)
    assert tr["T_W_imu"].shape == (n, 4, 4) and tr["width_m"].shape == (n,)
    ep = RawEpisode.open(demo)
    assert t[0] <= ep.imu.t_ns[0] and t[-1] >= ep.imu.t_ns[-1]
    for cam in ("cam0", "cam1"):
        assert t[0] <= tr[f"{cam}_t_ns"][0] and tr[f"{cam}_t_ns"][-1] <= t[-1]
    # the trajectory function reproduces the stored truth exactly
    traj = make_trajectory(rig, "demonstration", 11, DUR, PRE)
    assert np.allclose(traj.T_W_imu((t - t[0]) * 1e-9), tr["T_W_imu"], atol=1e-12)


# --------------------------------------------------------------------------------------------- time stamps

def test_camera_and_imu_time_stamps_give_true_times(demo, rig):
    ep, tr = RawEpisode.open(demo), read_truth(demo)
    for cam, role in (("cam0", "main"), ("cam1", "ultrawide")):
        ft = ep.frames(cam)
        assert np.all(ft.exposure_ns == EXPOSURE_NS[role])
        assert np.array_equal(rig.camera_time_ns(role, ft.t_ns, ft.exposure_ns), tr[f"{cam}_t_ns"])
    assert np.array_equal(rig.imu_time_ns(ep.imu.t_ns), tr["imu_t_ns"])


def test_time_model_with_other_timestamp_ref_and_offsets(tmp_path):
    """start_of_exposure + non-zero offsets: CSV t_ns still maps to the true mid-exposure / sample times."""
    d, _ = fast_rig(RIG_PATH)
    d["cameras"]["main"]["timestamp_ref"] = "start_of_exposure"
    d["cameras"]["main"]["time_offset_ns"] = 1_234_567
    d["cameras"]["ultrawide"]["time_offset_ns"] = -2_000_000
    d["imu"]["time_offset_ns"] = 3_000_000
    from dataproc.config import parse_rig

    rig2 = parse_rig(d)
    generate_episode(tmp_path / "ep", rig2, duration_s=3.5, preroll_s=2.0, noise_sigma=0, ss=1)
    ep, tr = RawEpisode.open(tmp_path / "ep"), read_truth(tmp_path / "ep")
    for cam, role in (("cam0", "main"), ("cam1", "ultrawide")):
        ft = ep.frames(cam)
        assert np.array_equal(rig2.camera_time_ns(role, ft.t_ns, ft.exposure_ns), tr[f"{cam}_t_ns"])
    assert np.array_equal(rig2.imu_time_ns(ep.imu.t_ns), tr["imu_t_ns"])
    assert ep.validate().ok


def test_video_frames_show_the_pose_at_their_time(demo, rig):
    """Re-render frame k at the pose of camera_time_ns(row k): it matches the decoded frame; +-1 frame does not."""
    ep, tr = RawEpisode.open(demo), read_truth(demo)
    traj = make_trajectory(rig, "demonstration", 11, DUR, PRE)
    t0 = int(tr["t_ns"][0])
    scene = build_scene(0)
    for cam, role, k in (("cam1", "ultrawide", 30), ("cam0", "main", 70)):    # in the pre-roll excitation
        c = rig.cameras[role]
        rend = CameraRenderer(camera_model(c), 2)
        ft = ep.frames(cam)
        t_true = rig.camera_time_ns(role, ft.t_ns[k], ft.exposure_ns[k])
        gray = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(float)    # noqa: E731 (luma: no 4:2:0)
        frame = gray(dict(ep.read_video_frames(cam, [k]))[k])
        err = {}
        for dt in (-1e9 / c.fps, 0.0, 1e9 / c.fps):
            T = traj.T_W_imu([(t_true + dt - t0) * 1e-9])[0] @ c.T_imu_cam
            img = gray(rend.render(scene, T))
            m = np.ones(img.shape, bool)
            m[int(0.6 * m.shape[0]):] = False                                   # skip the finger tags
            err[dt] = np.abs(img - frame)[m].mean()
        assert err[0.0] < 3.0, err
        assert min(err[-1e9 / c.fps], err[1e9 / c.fps]) > 3 * err[0.0], err


# --------------------------------------------------------------------------------------------- geometry

@pytest.fixture(scope="module")
def full_renderers():
    full = load_rig(RIG_PATH)                                    # full resolution: sharper corners
    return {role: CameraRenderer(camera_model(full.cameras[role]), 2) for role in ("main", "ultrawide")}


@pytest.mark.parametrize("role", ["main", "ultrawide"])
@pytest.mark.parametrize("offset", [(0.0, 0.0), (0.12, -0.06), (-0.15, 0.08)])
def test_projection_consistency(full_renderers, role, offset):
    """A world-fixed ArUco marker: detected corners == the camera model's projection of its 3D corners (< 1 px)."""
    rend = full_renderers[role]
    model = rend.model
    scene = build_scene(0)
    T_W_cam = make_T(camera_base_rotation(np.deg2rad(5.0)), [-0.3, 0.0, 1.3])
    size = 0.08
    R_W_m = np.stack([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], 1)   # facing the camera (-x)
    R_W_m = R_W_m @ Rotation.from_euler("xy", [10, -20], degrees=True).as_matrix()
    centre = T_W_cam[:3, :3] @ np.array([offset[0], offset[1], 0.0]) * 0.6 + T_W_cam[:3, 3] + [0.55, 0, 0]
    T_W_m = make_T(R_W_m, centre)
    q = Quad(T_W_m, quad_half_extent("DICT_4X4_50", size), scene.marker_rects[5])
    img = rend.render(scene, T_W_cam, [q])
    corners, ids, _ = _detector().detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    assert ids is not None and 5 in ids.ravel()
    det = corners[list(ids.ravel()).index(5)].reshape(4, 2)
    s = size / 2
    obj = np.array([[-s, s, 0, 1], [s, s, 0, 1], [s, -s, 0, 1], [-s, -s, 0, 1]])
    P_cam = (np.linalg.inv(T_W_cam) @ T_W_m @ obj.T).T[:, :3]
    uv, ok = model.project(P_cam)
    assert ok.all()
    assert np.abs(det - uv).max() < 1.0, (det, uv)


def test_aruco_widths_on_the_virtual_ultrawide(demo, rig):
    """iPhUMI-style detection (subpixel corners, zero-distortion PnP) on the virtual pinhole: both tags in nearly
    every frame, tag-centre x distance == truth width within 2 mm, depth ~ aruco.nominal_z_m."""
    ep, tr = RawEpisode.open(demo), read_truth(demo)
    vu = rig.virtual_ultrawide
    virt = VirtualPinhole(*vu.resolution, *vu.intrinsics)
    remap = build_remap(camera_model(rig.cameras["ultrawide"]), virt, vu.R_phys_virt)
    s = rig.aruco.marker_size_m / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    ft = ep.frames("cam1")
    w_true = np.interp(rig.camera_time_ns("ultrawide", ft.t_ns, ft.exposure_ns), tr["t_ns"], tr["width_m"])
    det = _detector()
    errs, zs = [], []
    for i, img in ep.read_video_frames("cam1"):
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(remap.apply(np.ascontiguousarray(img)), cv2.COLOR_RGB2GRAY))
        tv = {}
        for cc, tid in zip(corners, [] if ids is None else ids.ravel()):
            ok, _, t = cv2.solvePnP(obj, cc.reshape(4, 2), virt.K, np.zeros(5))
            tv[int(tid)] = t.ravel()
        if rig.aruco.left_id in tv and rig.aruco.right_id in tv:
            errs.append(tv[rig.aruco.right_id][0] - tv[rig.aruco.left_id][0] - w_true[i])
            zs += [tv[rig.aruco.left_id][2], tv[rig.aruco.right_id][2]]
    assert len(errs) >= 0.95 * len(ft)
    assert np.abs(errs).max() < 0.002
    assert np.abs(np.array(zs) - rig.aruco.nominal_z_m).max() < 0.005
    assert np.ptp(w_true) > 0.03                                   # the gripper really opened and closed


# --------------------------------------------------------------------------------------------- IMU

def test_imu_consistent_with_truth(demo, rig):
    """Gyro integrates to the truth rotation; accel matches the truth specific force (from finite differences of
    the truth poses) up to the simulated bias + white noise."""
    ep, tr = RawEpisode.open(demo), read_truth(demo)
    imu = ep.imu
    t_imu = rig.imu_time_ns(imu.t_ns)
    tt = tr["t_ns"]
    T = tr["T_W_imu"]
    assert abs(np.median(np.diff(t_imu)) - 1e9 / ep.meta.imu.rate_hz) < 2
    # specific force from the truth (central differences at 1 kHz), at the IMU sample times
    p = T[:, :3, 3]
    a_W = np.zeros_like(p)
    a_W[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / 1e-6
    i = np.searchsorted(tt, t_imu)
    inner = (i > 2) & (i < len(tt) - 3) & (tt[np.clip(i, 0, len(tt) - 1)] == t_imu)
    f_true = np.einsum("nji,nj->ni", T[i[inner], :3, :3], a_W[i[inner]] - G_WORLD)
    d = imu.accel[inner] - f_true
    assert np.abs(np.median(d, 0)).max() < 0.25                    # bias (std 0.05 m/s^2 per axis)
    assert (d - np.median(d, 0)).std(0).max() < 0.05               # white noise 7e-4*sqrt(416) = 0.014
    # integrate the gyro (minus its mean bias estimate over the static start) from the first sample
    R = Rotation.from_matrix(T[np.searchsorted(tt, t_imu[0]), :3, :3])
    static = t_imu < t_imu[0] + 0.25e9
    gyro = imu.gyro - imu.gyro[static].mean(0)
    dt = np.diff(t_imu) * 1e-9
    for k in range(len(dt)):
        R = R * Rotation.from_rotvec(0.5 * (gyro[k] + gyro[k + 1]) * dt[k])
    R_end = Rotation.from_matrix(T[np.searchsorted(tt, t_imu[-1]), :3, :3])
    assert np.rad2deg((R.inv() * R_end).magnitude()) < 0.5


# --------------------------------------------------------------------------------------------- determinism, faults

def test_deterministic_with_the_same_seed(demo, rig, tmp_path):
    out = tmp_path / "again"
    generate_episode(out, rig, "demonstration", seed=11, duration_s=DUR, preroll_s=PRE, session_id="s1",
                     episode_number=7)
    for f in ("cam0_frames.csv", "cam1_frames.csv", "imu.csv", "meta.json", TRUTH_FILE, "MANIFEST.sha256"):
        assert _sha(out / f) == _sha(demo / f), f
    a = RawEpisode.open(out)
    b = RawEpisode.open(demo)
    for (_, x), (_, y) in zip(a.read_video_frames("cam1"), b.read_video_frames("cam1")):
        assert np.array_equal(x, y)


def test_faults(rig, tmp_path):
    ep_dir = tmp_path / "f"
    generate_episode(ep_dir, rig, duration_s=3.5, preroll_s=2.0, faults={
        "drop_frames": {"cam1": [10, 11, 40]}, "imu_gaps": [(2.5, 0.05)], "count_mismatch": ["cam0"]}, ss=1)
    ep = RawEpisode.open(ep_dir)
    rep = ep.validate(decode_videos=True)
    errors = {(p.code, p.file) for p in rep.errors}
    warnings = {(p.code, p.file) for p in rep.warnings}
    assert errors == {("count_mismatch", "cam0_frames.csv"), ("decoded_count_mismatch", "cam0.mp4")}
    assert warnings == {("sensor_seq_gap", "cam1_frames.csv"), ("imu_sample_gap", "imu.csv")}
    gap = next(p for p in rep.warnings if p.code == "sensor_seq_gap").details
    assert gap["n_gaps"] == 2 and gap["total_missing"] == 3
    igap = next(p for p in rep.warnings if p.code == "imu_sample_gap").details
    assert 20 <= igap["total_missing"] <= 22 and 50 <= igap["max_dt_ms"] <= 53
    assert ep.meta.cameras["cam1"].dropped_frames == 3 and ep.meta.imu.dropped_samples == igap["total_missing"]
    # the CSV of the dropped camera still describes exactly the encoded frames
    tr = read_truth(ep_dir)
    ft = ep.frames("cam1")
    assert len(ft) == ep.meta.cameras["cam1"].frame_count == len(tr["cam1_t_ns"])
    assert np.array_equal(rig.camera_time_ns("ultrawide", ft.t_ns, ft.exposure_ns), tr["cam1_t_ns"])


@pytest.mark.parametrize("faults, exc", [({"drop_frames": {"cam2": [1]}}, ValueError),
                                         ({"bogus": 1}, ValueError), ({"drop_frames": {"cam0": [10**6]}}, ValueError),
                                         ("x", TypeError)])
def test_bad_faults(rig, tmp_path, faults, exc):
    with pytest.raises(exc):
        generate_episode(tmp_path / "e", rig, duration_s=3.5, preroll_s=2.0, faults=faults)


def test_refuses_non_empty_output(rig, tmp_path):
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "x").touch()
    with pytest.raises(FileExistsError):
        generate_episode(tmp_path / "e", rig, duration_s=3.5, preroll_s=2.0)


# --------------------------------------------------------------------------------------------- fast rig + CLI

def test_fast_rig_scales_intrinsics():
    d, r = fast_rig(RIG_PATH)
    full = load_rig(RIG_PATH)
    assert r.rig_id == "rig_v1_fast" and d["rig_id"] == "rig_v1_fast"
    for role, s in (("main", 0.25), ("ultrawide", 0.5)):
        fx, fy, cx, cy = full.cameras[role].model.intrinsics
        assert np.allclose(r.cameras[role].model.intrinsics, [s * fx, s * fy, s * (cx + 0.5) - 0.5,
                                                              s * (cy + 0.5) - 0.5])
        assert r.cameras[role].model.distortion == full.cameras[role].model.distortion
    assert np.array_equal(r.T_cam_tcp, full.T_cam_tcp) and r.virtual_ultrawide.resolution == (640, 480)


def test_cli_sim_contract(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path))
    inc = tmp_path / "incoming"
    assert cli.main(["sim", "--out", str(inc), "--episodes", "1", "--fast", "--duration", "3.5", "--session",
                     "t1", "--calibration", "--seed", "4", "--contract"]) == 0
    (inc / "other_episode_000004").mkdir()                     # numbering continues after the largest number
    assert next_episode_number(inc) == 5
    assert cli.main(["sim", "--out", str(inc), "--episodes", "1", "--fast", "--duration", "3.5", "--session", "t2",
                     "--unverified", "--contract"]) == 0
    names = sorted(p.name for p in inc.iterdir())
    assert names == ["other_episode_000004", "t1_episode_000000", "t2_episode_000005"]
    cal = RawEpisode.open(inc / "t1_episode_000000")
    assert cal.validate().ok and cal.is_verified
    meta = json.loads((inc / "t1_episode_000000" / "meta.json").read_text())
    assert meta["sim"]["kind"] == "calibration" and meta["rig_id"] == "rig_v1_fast"
    assert np.ptp(read_truth(inc / "t1_episode_000000")["width_m"]) > 0.05
    assert not (inc / "t2_episode_000005" / VERIFIED).exists()
    rig_file = tmp_path / "rigs" / "rig_v1_fast.yaml"
    assert load_rig(rig_file).rig_id == "rig_v1_fast"           # processing can load the rig it was made with
    out = capsys.readouterr().out
    assert "t1_episode_000000" in out and str(rig_file) in out


def test_cli_sim_pi_format(tmp_path, monkeypatch, capsys):
    """Default output: the Pi's format as ``episode_NNNNNN`` (Pi meta ``episode`` = NNNNNN), numbering shared by
    calibration and demonstration runs into one --out; ``truth.npz`` carried and listed in the manifest; the
    adapter takes it without problems."""
    from dataproc.pi_adapter import convert_episode, inspect_source
    from dataproc.raw_episode import COMPLETE, MANIFEST

    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "root"))
    src = tmp_path / "src"
    assert cli.main(["sim", "--out", str(src), "--fast", "--duration", "3.5", "--calibration"]) == 0
    assert cli.main(["sim", "--out", str(src), "--fast", "--duration", "3.5", "--episodes", "2"]) == 0
    assert sorted(p.name for p in src.iterdir()) == ["episode_000000", "episode_000001", "episode_000002"]
    out = capsys.readouterr().out
    assert "(Pi format)" in out and "pass --rig rig_v1_fast to dataproc calibrate" in out
    d = src / "episode_000002"
    assert sorted(p.name for p in d.iterdir()) == sorted(["cam0.mp4", "cam0_frames.csv", "cam1.mp4",
                                                          "cam1_frames.csv", "imu.csv", "meta.json", TRUTH_FILE,
                                                          MANIFEST, COMPLETE])
    meta = json.loads((d / "meta.json").read_text())
    assert meta["episode"] == 2 and meta["valid"] is True
    assert {c["model"]: c["file"] for c in meta["cameras"]} == {"imx219": "cam0.mp4", "imx519": "cam1.mp4"}
    assert TRUTH_FILE in (d / MANIFEST).read_text()
    s = inspect_source(d)
    assert s.ok, s.problem
    convert_episode(s, tmp_path / "conv", "rig_v1_fast")
    assert (tmp_path / "conv" / TRUTH_FILE).is_file()
    assert read_truth(tmp_path / "conv")["kind"] == "demonstration"


def test_cli_sim_usage_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        cli.main(["sim", "--out", str(tmp_path), "--duration", "-1"])
    assert e.value.code == 2
    assert cli.main(["sim", "--out", str(tmp_path / "x"), "--fast", "--duration", "2"]) == 1   # too short
    assert "duration_s" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli.main(["sim", "--out", str(tmp_path / "y"), "--unverified"])                    # Pi format: no .verified
    assert e.value.code == 2 and "--unverified needs --contract" in capsys.readouterr().err
    assert cli.main(["sim", "--out", str(tmp_path / "z"), "--fast", "--imu-gap", "4"]) == 1     # > duration / 2
    assert "imu gap" in capsys.readouterr().err
    assert not (tmp_path / "y").exists()
