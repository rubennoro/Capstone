import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from umiproc.dp.geometry.transforms import inv_T
from umiproc.dp.raw_episode import RawEpisode
from umiproc.dp.slam import SlamError, get_backend
from umiproc.dp.geometry.trajectory import Trajectory
from umiproc.dp.slam.base import finalize_trajectory, ultrawide_times_ns
from umiproc.dp.slam.oracle import OracleBackend, load_truth
from umiproc.dp.slam.orbslam3 import (
    EUROC_IMU_HEADER, IMU_INIT_STRINGS, ORB_BIN, ORB_SLAM3_DIR_ENV, ORB_VOC, Orbslam3Backend, count_map_events,
    count_map_inits, default_orb_slam3_dir, imu_init_events, import_trajectory, no_keyframes_message,
    orbslam3_command, subprocess_runner, write_euroc,
)
from test_geometry_common import EXPOSURE_NS, IMU_HZ, analytic_episode, make_rig, truth_poses, yaw_residual

UW_ROT = {"rotation": {"euler_xyz_deg": [10.0, -5.0, 90.0]}, "translation_m": [-0.020, 0.004, 0.010]}


@pytest.fixture(scope="module")
def rig():
    return make_rig(**{"cameras.ultrawide.T_imu_cam": UW_ROT})


@pytest.fixture(scope="module")
def episode(tmp_path_factory, rig):
    return RawEpisode.open(analytic_episode(tmp_path_factory.mktemp("slam"), rig))


def test_registry():
    assert isinstance(get_backend("oracle"), OracleBackend)
    assert isinstance(get_backend("orbslam3", scale=0.5), Orbslam3Backend)
    with pytest.raises(ValueError, match="unknown SLAM backend"):
        get_backend("vins")
    with pytest.raises(TypeError):
        get_backend("oracle", bogus=1)


def test_ultrawide_times_are_mid_exposure(episode, rig):
    fr = episode.frames("cam1")
    t = ultrawide_times_ns(episode, rig)
    assert np.array_equal(t, fr.t_ns - EXPOSURE_NS // 2)          # end_of_exposure stamps, offset 0


# --------------------------------------------------------------------------------------------- oracle

def test_oracle_noise_free_exact(episode, rig):
    res = get_backend("oracle", gravity_align=False).run(episode, rig, None)
    t_uw = ultrawide_times_ns(episode, rig)
    assert res.fraction_tracked == 1.0 and np.array_equal(res.trajectory.t_ns, t_uw)
    assert res.trajectory.frame == "imu" and res.trajectory.world == "z_up_gravity"
    assert np.allclose(res.trajectory.T(), truth_poses(t_uw), atol=1e-6)
    assert res.problems == [] and res.diagnostics["gravity_check"]["ok"]


def test_oracle_noise_within_sigma(episode, rig):
    sp, sr = 0.002, 0.5
    res = get_backend("oracle", noise_pos_m=sp, noise_rot_deg=sr, gravity_align=False, seed=3).run(episode, rig)
    T_true = truth_poses(res.trajectory.t_ns)
    T = res.trajectory.T()
    dp = T[:, :3, 3] - T_true[:, :3, 3]
    assert np.std(dp) == pytest.approx(sp, rel=0.25) and abs(dp.mean()) < sp / 2
    ang = np.degrees(Rotation.from_matrix(np.einsum("nji,njk->nik", T_true[:, :3, :3], T[:, :3, :3])).magnitude())
    assert np.sqrt(np.mean(ang ** 2)) == pytest.approx(sr * np.sqrt(3), rel=0.25)
    again = get_backend("oracle", noise_pos_m=sp, noise_rot_deg=sr, gravity_align=False, seed=3).run(episode, rig)
    assert np.array_equal(again.trajectory.p, res.trajectory.p)          # seeded


def test_oracle_drift_grows(episode, rig):
    res = get_backend("oracle", drift_pos_m_per_sqrt_s=0.01, gravity_align=False, seed=1).run(episode, rig)
    err = np.linalg.norm(res.trajectory.p - truth_poses(res.trajectory.t_ns)[:, :3, 3], axis=1)
    assert err[0] < 1e-3 and 0.002 < err[-1] < 0.08                      # ~0.01 * sqrt(4 s) * sqrt(3)


def test_oracle_world_change_and_gravity_align(episode, rig):
    be = get_backend("oracle", world_yaw_deg=40, world_tilt_deg=12, world_tilt_azimuth_deg=70,
                     world_translation_m=[1, -2, 0.5])
    res = be.run(episode, rig, None)
    info = res.diagnostics["gravity_alignment"]
    assert info["applied"] and info["angle_deg"] == pytest.approx(12.0, abs=0.1)
    T_true = truth_poses(res.trajectory.t_ns)
    R0, tilt, spread = yaw_residual(res.trajectory.T(), T_true)
    assert tilt < 0.1 and spread < 1e-4                                     # yaw-only difference left
    # translation: p_est = R0 p_true + c (constant c)
    c = res.trajectory.p - T_true[:, :3, 3] @ R0.T
    assert np.ptp(c, axis=0).max() < 1e-3
    assert res.diagnostics["gravity_check"]["ok"]
    # without the alignment the tilt stays and the gravity check fails
    raw = get_backend("oracle", world_tilt_deg=12, gravity_align=False).run(episode, rig)
    assert not raw.diagnostics["gravity_check"]["ok"] and any("gravity" in p for p in raw.problems)


def test_oracle_truth_errors(tmp_path, rig, episode):
    with pytest.raises(SlamError, match="no ground truth"):
        OracleBackend(truth_file="nope.npz").run(episode, rig)
    other = make_rig(rig_id="rig_v2")
    with pytest.raises(SlamError, match="rig_id"):
        OracleBackend().run(episode, other)
    bad = tmp_path / "t.npz"
    np.savez(bad, t_ns=np.array([2, 1]), T_W_imu=np.stack([np.eye(4)] * 2), width_m=np.zeros(2), rig_id="rig_v1",
             seed=0, version="x")
    with pytest.raises(SlamError, match="strictly increasing"):
        load_truth(bad)
    with pytest.raises(ValueError):
        OracleBackend(noise_pos_m=-1)


def test_oracle_untracked_outside_truth(tmp_path, rig):
    ep = analytic_episode(tmp_path, rig)
    z = dict(np.load(ep / "truth.npz"))
    keep = z["t_ns"] <= z["t_ns"][0] + 2_000_000_000
    np.savez(ep / "truth.npz", **{k: (v[keep] if k in ("t_ns", "T_W_imu", "width_m") else v) for k, v in z.items()})
    res = OracleBackend(gravity_align=False).run(RawEpisode.open(ep), rig)
    assert 0.45 < res.fraction_tracked < 0.55
    assert np.array_equal(res.trajectory.t_ns, res.t_frames_ns[res.tracked])


# --------------------------------------------------------------------------------------------- orbslam3

def _settings(path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.]+):\s*(.*)$", ln)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def _opencv_matrix(text: str, name: str) -> np.ndarray:
    m = re.search(rf"{re.escape(name)}: !!opencv-matrix\s+rows: (\d+)\s+cols: (\d+)\s+dt: f\s+data: \[([^\]]*)\]", text)
    return np.array([float(x) for x in m.group(3).split(",")]).reshape(int(m.group(1)), int(m.group(2)))


class FakeOrbslam:
    """Stands in for mono_inertial_euroc: checks the cwd and paths, writes an ORB-SLAM3-style f_episode.txt and a
    log."""

    def __init__(self, skip_first=5, rc=0, write=True, log_text="Creation of new map with id: 0\n"):
        self.skip_first, self.rc, self.write, self.log_text = skip_first, rc, write, log_text
        self.calls = []

    def __call__(self, cmd, log_path, timeout_s, cwd):
        self.calls.append(cmd)
        work = log_path.parent
        assert cwd == work and cmd[3] == str(work) and cmd[2] == str(work / "orbslam3_settings.yaml")
        with open(log_path, "a") as f:
            f.write(self.log_text)
        if self.write:
            t = np.array([int(x) for x in (work / "times.txt").read_text().split()])[self.skip_first:]
            T = truth_poses(t)
            T_b0_imu = inv_T(T[0]) @ T                                 # world re-based on the first KF IMU frame
            q = Rotation.from_matrix(T_b0_imu[:, :3, :3]).as_quat()
            lines = []
            for ti, Ti, qi in zip(t, T_b0_imu, q):
                ts = float(ti) * 1e-9                                  # ORB-SLAM3 keeps seconds in a double
                lines.append(f"{1e9 * ts:.6f} " + " ".join(f"{v:.9f}" for v in (*Ti[:3, 3], *qi)))
            (work / "f_episode.txt").write_text("\n".join(lines) + "\n")
        return self.rc


def test_command_layout(tmp_path):
    orb, work = tmp_path / "ORB_SLAM3", tmp_path / "work"
    cmd = orbslam3_command(orb, work)
    assert cmd == [str(orb / ORB_BIN), str(orb / ORB_VOC), str(work / "orbslam3_settings.yaml"), str(work),
                   str(work / "times.txt"), "episode"]


def test_orb_slam3_dir_default(monkeypatch, tmp_path):
    monkeypatch.delenv(ORB_SLAM3_DIR_ENV, raising=False)
    assert default_orb_slam3_dir() == Path.home() / "ORB_SLAM3"
    monkeypatch.setenv(ORB_SLAM3_DIR_ENV, str(tmp_path))
    assert default_orb_slam3_dir() == tmp_path
    assert Orbslam3Backend().orb_dir == tmp_path
    assert Orbslam3Backend(orb_slam3_dir=tmp_path / "x").orb_dir == tmp_path / "x"


def test_missing_binary_is_a_slam_error(tmp_path, rig, episode):
    with pytest.raises(SlamError, match="not found or not executable"):
        subprocess_runner(orbslam3_command(tmp_path / "nope", tmp_path), tmp_path / "log", 1.0, tmp_path)
    with pytest.raises(SlamError, match="ORBSLAM3.md"):
        Orbslam3Backend(orb_slam3_dir=tmp_path / "nope", scale=0.25).run(episode, rig, tmp_path / "w")


def test_orbslam3_fake_end_to_end(tmp_path, rig):
    ep = RawEpisode.open(analytic_episode(tmp_path, rig, imu_late_frames=3))
    fake = FakeOrbslam(log_text="Creation of new map with id: 0\nSYSTEM-> Reseting active map in monocular case\n"
                               "Creation of new map with id: 1\n")
    work = tmp_path / "work"
    res = get_backend("orbslam3", scale=0.5, runner=fake).run(ep, rig, work)
    assert len(fake.calls) == 1

    # EuRoC layout
    t_uw = ultrawide_times_ns(ep, rig)
    imu_t = ep.imu.t_ns
    expect = t_uw[(t_uw >= imu_t[0]) & (t_uw < imu_t[-1])]
    assert len(expect) == len(t_uw) - 4                              # 3 before the IMU, the last at its end
    times = [int(x) for x in (work / "times.txt").read_text().split()]
    assert times == expect.tolist()
    rows = (work / "mav0/cam0/data.csv").read_text().splitlines()
    assert rows[0] == "#timestamp [ns],filename" and rows[1] == f"{times[0]},{times[0]}.png"
    img = cv2.imread(str(work / f"mav0/cam0/data/{times[5]}.png"), cv2.IMREAD_UNCHANGED)
    assert img.shape == (308, 410) and img.dtype == np.uint8         # gray, 820x616 * 0.5
    imu_rows = (work / "mav0/imu0/data.csv").read_text().splitlines()
    assert imu_rows[0] == EUROC_IMU_HEADER and len(imu_rows) == len(imu_t) + 1
    first = imu_rows[1].split(",")
    assert int(first[0]) == imu_t[0] and np.allclose([float(x) for x in first[1:]],
                                                     [*ep.imu.gyro[0], *ep.imu.accel[0]], atol=1e-6)

    # settings
    text = (work / "orbslam3_settings.yaml").read_text()
    s = _settings(work / "orbslam3_settings.yaml")
    assert text.startswith("%YAML:1.0\n---\n") and s["File.version"] == "1.0"
    assert s["Camera.type"] == "KannalaBrandt8"
    fx, fy, cx, cy = rig.cameras["ultrawide"].model.intrinsics
    assert float(s["Camera1.fx"]) == pytest.approx(fx * 0.5) and float(s["Camera1.fy"]) == pytest.approx(fy * 0.5)
    assert float(s["Camera1.cx"]) == pytest.approx(0.5 * (cx + 0.5) - 0.5)
    assert float(s["Camera1.cy"]) == pytest.approx(0.5 * (cy + 0.5) - 0.5)
    for k, v in zip(("k1", "k2", "k3", "k4"), rig.cameras["ultrawide"].model.distortion):
        assert float(s[f"Camera1.{k}"]) == pytest.approx(v, abs=1e-12)
    assert (s["Camera.width"], s["Camera.height"], s["Camera.fps"], s["Camera.RGB"]) == ("410", "308", "30", "0")
    assert np.allclose(_opencv_matrix(text, "IMU.T_b_c1"), rig.cameras["ultrawide"].T_imu_cam, atol=1e-9)
    n = rig.imu.noise.for_vio()
    assert float(s["IMU.NoiseGyro"]) == pytest.approx(n.gyroscope_noise_density)
    assert float(s["IMU.NoiseAcc"]) == pytest.approx(n.accelerometer_noise_density)
    assert float(s["IMU.GyroWalk"]) == pytest.approx(n.gyroscope_random_walk)
    assert float(s["IMU.AccWalk"]) == pytest.approx(n.accelerometer_random_walk)
    assert float(s["IMU.Frequency"]) == pytest.approx(IMU_HZ, abs=0.1)

    # import: times snapped to the frames, T_W_imu recovered up to yaw + translation, gravity-aligned
    tr = res.trajectory
    assert tr.frame == "imu" and tr.world == "z_up_gravity"
    assert np.array_equal(tr.t_ns, expect[5:])
    assert res.tracked.sum() == len(expect) - 5 and len(res.tracked) == len(t_uw)
    assert res.fraction_tracked == pytest.approx((len(expect) - 5) / len(t_uw))
    assert (res.n_maps, res.n_resets) == (2, 1)
    R0, tilt, spread = yaw_residual(tr.T(), truth_poses(tr.t_ns))
    assert tilt < 0.1 and spread < 1e-5
    assert res.diagnostics["gravity_alignment"]["applied"]
    assert res.diagnostics["import"]["n_unmatched"] == 0
    assert res.diagnostics["gravity_check"]["ok"] and res.diagnostics["motion_check"]["ok"]
    assert res.diagnostics["scale_check"]["status"] == "ok"
    assert res.diagnostics["imu_init"] == dict.fromkeys(IMU_INIT_STRINGS, 0)


def test_orbslam3_failures(tmp_path, rig, episode):
    with pytest.raises(SlamError, match="rc=1"):
        Orbslam3Backend(runner=FakeOrbslam(rc=1), scale=0.25).run(episode, rig, tmp_path / "a")
    with pytest.raises(SlamError, match="no trajectory"):
        Orbslam3Backend(runner=FakeOrbslam(write=False), scale=0.25).run(episode, rig, tmp_path / "b")
    # the real crash on a static recording: no keyframe ever made, then a segfault while saving the empty map
    empty = FakeOrbslam(rc=-11, write=False, log_text="Shutdown\nThere are 1 maps in the atlas\n  Map 0 has 0 KFs\n")
    with pytest.raises(SlamError, match="never initialised"):
        Orbslam3Backend(runner=empty, scale=0.25).run(episode, rig, tmp_path / "c")


def test_none_and_auto_backends(monkeypatch, tmp_path):
    from umiproc.dp.slam import NoSlamBackend, orbslam3_available, resolve_backend_name

    assert isinstance(get_backend("none"), NoSlamBackend)
    monkeypatch.setenv("ORB_SLAM3_DIR", str(tmp_path / "missing"))
    assert not orbslam3_available() and resolve_backend_name("auto") == "none"
    exe = tmp_path / "orb" / "Examples" / "Monocular-Inertial" / "mono_inertial_euroc"
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    (tmp_path / "orb" / "Vocabulary").mkdir()
    (tmp_path / "orb" / "Vocabulary" / "ORBvoc.txt").write_text("")
    monkeypatch.setenv("ORB_SLAM3_DIR", str(tmp_path / "orb"))
    assert orbslam3_available() and resolve_backend_name("auto") == "orbslam3"
    assert resolve_backend_name("oracle") == "oracle"


def test_import_snapping_and_map_events(tmp_path):
    frames = [5_000_000_000_000, 5_000_033_333_333, 5_000_066_666_667]
    (tmp_path / "f.txt").write_text(
        "5000000000000.000977 0 0 0 0 0 0 1\n"           # double round-off: snapped
        "5000033333333.000244 0.1 0 0 0 0 0 1\n"
        "5000050000000.000000 0.2 0 0 0 0 0 1\n")         # between frames (> 1 ms away): dropped
    tr, info = import_trajectory(tmp_path / "f.txt", frames)
    assert tr.t_ns.tolist() == frames[:2] and info["n_unmatched"] == 1
    assert count_map_events("") == (1, 0)
    assert count_map_events("Creation of new map with id: 0\nCreation of new map with id: 1\n"
                            "SYSTEM-> Reseting active map in monocular case\n") == (2, 1)


def test_reset_counted_once_per_executed_reset():
    """Tracking.cc prints its own "...Reseting active map..." before System::ResetActiveMap, which prints the
    "SYSTEM->" line: one reset, counted once. LocalMapping's "Not enough motion" line is a separate diagnostic."""
    log = ("Creation of new map with id: 0\n"
           "IMU is not or recently initialized. Reseting active map...\n"
           "SYSTEM-> Reseting active map in monocular case\n"
           "Not enough motion for initializing. Reseting...\n"
           "LM: Reseting current map in Local Mapping...\n"
           "TRACK: Reset map because local mapper set the bad imu flag \n"
           "SYSTEM-> Reseting active map in monocular case\n"
           "Creation of new map with id: 1\n")
    assert count_map_events(log) == (2, 2)
    assert imu_init_events(log)["not_enough_motion_reset"] == 1
    assert count_map_events("IMU is not or recently initialized. Reseting active map...\n") == (1, 0)
    assert count_map_events("end VIBA 1SYSTEM-> Reseting active map in monocular case\n") == (1, 1)  # interleaved


def test_orbslam3_reports_not_enough_motion(tmp_path, rig, episode):
    log = "Not enough motion for initializing. Reseting...\nSYSTEM-> Reseting active map in monocular case\n"
    res = Orbslam3Backend(runner=FakeOrbslam(log_text=log), scale=0.25).run(episode, rig, tmp_path / "w")
    assert res.n_resets == 1 and res.diagnostics["n_not_enough_motion_resets"] == 1


# --------------------------------------------------------------------------------------------- slam_mask

STRIP_MASK = {"polygons": [[[0, 462], [820, 462], [820, 616], [0, 616]]], "source": "test: bottom strip"}


@pytest.fixture(scope="module")
def masked_rig():
    return make_rig(**{"cameras.ultrawide.T_imu_cam": UW_ROT, "cameras.ultrawide.slam_mask": STRIP_MASK})


def _png(work, t) -> np.ndarray:
    return cv2.imread(str(work / f"mav0/cam0/data/{t}.png"), cv2.IMREAD_UNCHANGED)


def test_orbslam3_fake_end_to_end_masked(tmp_path, rig, masked_rig, episode):
    """Same episode with and without the mask at scale 0.5: the strip (rows 231..307 of 410x308) is black, the rest
    of each image and the settings are unchanged, and only the masked run reports the mask."""
    runs = {name: get_backend("orbslam3", scale=0.5, runner=FakeOrbslam()).run(episode, r, tmp_path / name)
            for name, r in (("plain", rig), ("masked", masked_rig))}
    wp, wm = tmp_path / "plain", tmp_path / "masked"
    assert "slam_mask" not in runs["plain"].diagnostics["prep"]
    assert runs["masked"].diagnostics["prep"]["slam_mask"] == {"n_polygons": 1, "masked_fraction": 0.25}
    assert (wm / "orbslam3_settings.yaml").read_bytes() == (wp / "orbslam3_settings.yaml").read_bytes()
    times = (wm / "times.txt").read_text().split()
    assert times == (wp / "times.txt").read_text().split() and len(times) > 100
    for t in times[::20]:
        a, b = _png(wp, t), _png(wm, t)
        assert a.shape == b.shape == (308, 410)
        assert (b[231:] == 0).all() and a[231:].any()
        # the episode's 64x48 frames get their own mask before the upscale; its edge is a source pixel boundary
        # (462 / 616 * 48 = 36), so nothing above the strip changes
        assert np.array_equal(b[:231], a[:231])


@pytest.mark.parametrize("frame_size", [(820, 616), (1640, 1232), (64, 48)])
def test_write_euroc_frames_hook_masked(tmp_path, masked_rig, episode, frame_size):
    """White frames through the frames= hook, at the rig's resolution and at other sizes: written at scale 0.5,
    the strip is exactly 0 and the rest stays white."""
    white = np.full((frame_size[1], frame_size[0], 3), 255, np.uint8)
    n = len(ultrawide_times_ns(episode, masked_rig))
    info = write_euroc(episode, masked_rig, tmp_path, scale=0.5, frames=((i, white) for i in range(n)))
    assert info["slam_mask"] == {"n_polygons": 1, "masked_fraction": 0.25} and info["size"] == [410, 308]
    for t in info["t_cam_ns"][::15]:
        img = _png(tmp_path, t)
        assert img.shape == (308, 410) and (img[231:] == 0).all() and (img[:231] == 255).all()


# real mono_inertial_euroc output (ORB-SLAM3 v1.0 on the Pi recordings, 2026-11-15)
LOG_START = ("Initialization of Atlas from scratch \nCreation of new map with id: 0\n"
             "Creation of new map with last KF id: 0\nSeq. Name: \nThere are 1 cameras in the atlas\n"
             "Camera 0 is fisheye\n")
LOG_LOST = ("First KF:0; Map init KF:0\nNew Map created with 111 points\nFail to track local map!\n"
            "IMU is not or recently initialized. Reseting active map...\n"
            "SYSTEM-> Reseting active map in monocular case\nLM: Active map reset recieved\n"
            "LM: Active map reset, waiting...\nLM: Reseting current map in Local Mapping...\n"
            "LM: End reseting Local Mapping...\nLM: Reset free the mutex\nLM: Active map reset, Done!!!\n"
            "mnFirstFrameId = 242\nmnInitialFrameId = 0\n72 Frames set to lost\n")
LOG_NO_MOTION = ("First KF:12; Map init KF:0\nNew Map created with 79 points\n"
                 "Not enough motion for initializing. Reseting...\n"
                 "TRACK: Reset map because local mapper set the bad imu flag \n"
                 "SYSTEM-> Reseting active map in monocular case\n174 Frames set to lost\n")
LOG_EMPTY_SAVE = ("Shutdown\n\nSaving trajectory to f_episode.txt ...\nThere are 1 maps in the atlas\n"
                  "  Map 0 has 0 KFs\n")


def test_orbslam3_zero_keyframe_messages(tmp_path, rig, masked_rig, episode):
    """ORB-SLAM3 crashed saving an empty map: "never initialised" only when the log shows no initialisation."""
    def fail(r, log_text, name):
        with pytest.raises(SlamError) as e:
            Orbslam3Backend(runner=FakeOrbslam(rc=-11, write=False, log_text=log_text), scale=0.25).run(
                episode, r, tmp_path / name)
        return str(e.value)

    log_a = (tmp_path / "a").resolve() / "orbslam3.log"
    assert fail(rig, LOG_START + LOG_EMPTY_SAVE, "a") == (
        "orbslam3: never initialised (0 keyframes; mono-inertial needs the camera to move with parallax: a static "
        "recording cannot work; the gripper itself may be blocking SLAM: set cameras.ultrawide.slam_mask in the "
        f"rig), see {log_a}")
    msg = fail(masked_rig, LOG_START + LOG_EMPTY_SAVE, "b")
    assert "never initialised" in msg and "check that cameras.ultrawide.slam_mask still covers the gripper" in msg
    log_c = (tmp_path / "c").resolve() / "orbslam3.log"
    assert fail(masked_rig, LOG_START + 2 * LOG_LOST + LOG_EMPTY_SAVE, "c") == (
        "orbslam3: initialised 2 times but lost tracking before the IMU finished initialising (2 map resets, no "
        "keyframes kept): the recording needs more texture in view and side-to-side motion at the start, "
        f"see {log_c}")
    msg = fail(rig, LOG_START + LOG_LOST + LOG_NO_MOTION + LOG_EMPTY_SAVE, "d")
    assert msg.startswith("orbslam3: initialised 2 times but lost tracking before the IMU finished initialising "
                          "(2 map resets, 1 of them for not enough motion during the IMU initialisation, no "
                          "keyframes kept): ")
    assert "; the gripper itself may also be blocking SLAM: set cameras.ultrawide.slam_mask in the rig, see" in msg
    # a crash with a non-empty map is not a 0-keyframe run
    two_maps = "There are 2 maps in the atlas\n  Map 0 has 0 KFs\n  Map 1 has 23 KFs\n"
    assert "rc=-11" in fail(rig, LOG_START + LOG_LOST + two_maps, "e")


def test_no_keyframes_message_counts():
    assert count_map_inits(LOG_START + 2 * LOG_LOST + LOG_NO_MOTION) == 3
    wrong = "New Map created with 60 points\nWrong initialization, reseting...\n"     # rejected at once: not counted
    assert count_map_inits(LOG_START + wrong) == 0 and count_map_inits(wrong + LOG_LOST) == 1
    assert "never initialised" in no_keyframes_message(LOG_START + wrong + LOG_EMPTY_SAVE, "x", masked=True)
    one = no_keyframes_message(LOG_START + LOG_LOST + LOG_EMPTY_SAVE, "x", masked=True)
    assert one.startswith("orbslam3: initialised 1 time but") and "(1 map reset, no keyframes kept)" in one
    assert no_keyframes_message(LOG_START + LOG_LOST, "x", masked=True) is None          # no save listing


# --------------------------------------------------------------------------------------------- scale check

def test_finalize_flags_wrong_scale(episode, rig):
    """Reviewer repro: a trajectory at half the metric scale (mono map never IMU-initialised) must be flagged."""
    res = OracleBackend().run(episode, rig)
    sc = res.diagnostics["scale_check"]
    assert sc["status"] == "ok" and sc["scale"] == pytest.approx(1.0, abs=0.01)
    T = res.trajectory.T()
    T[:, :3, 3] *= 0.5
    r2 = finalize_trajectory(Trajectory.from_T(res.trajectory.t_ns, T), episode, rig, res.t_frames_ns)
    assert r2.diagnostics["scale_check"]["status"] == "scale_mismatch"
    assert any(p.startswith("scale_mismatch") for p in r2.problems)


def test_scale_check_with_oracle_noise(episode, rig):
    res = OracleBackend(noise_pos_m=0.001, noise_rot_deg=0.2, drift_pos_m_per_sqrt_s=0.002, world_tilt_deg=8,
                        world_yaw_deg=-60, seed=4).run(episode, rig)
    sc = res.diagnostics["scale_check"]
    assert sc["status"] == "ok" and sc["scale"] == pytest.approx(1.0, abs=0.05)
    assert not any(p.startswith("scale_") for p in res.problems)


def test_imu_init_events():
    log = ("start VIBA 1\nend VIBA 1\nNot enough motion for initializing. Reseting...\nscale too small\n"
           "start VIBA 2\nend VIBA 2\n")
    assert imu_init_events(log) == {"viba1_done": 1, "viba2_done": 1, "not_enough_motion_reset": 1,
                                    "scale_too_small": 1}
    assert imu_init_events("") == dict.fromkeys(IMU_INIT_STRINGS, 0)
