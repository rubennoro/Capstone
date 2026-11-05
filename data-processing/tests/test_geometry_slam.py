import re

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataproc.geometry.transforms import inv_T
from dataproc.raw_episode import RawEpisode
from dataproc.slam import SlamError, get_backend
from dataproc.geometry.trajectory import Trajectory
from dataproc.slam.base import finalize_trajectory, ultrawide_times_ns
from dataproc.slam.oracle import OracleBackend, load_truth
from dataproc.slam.orbslam3 import (
    DEFAULT_IMAGE, EUROC_IMU_HEADER, IMU_INIT_STRINGS, ORB_BIN, ORB_VOC, Orbslam3Backend, count_map_events,
    imu_init_events, import_trajectory, orbslam3_docker_command,
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


class FakeDocker:
    """Stands in for `docker run`: checks the mount, writes an ORB-SLAM3-style f_episode.txt and a log."""

    def __init__(self, skip_first=5, rc=0, write=True, log_text="Creation of new map with id: 0\n"):
        self.skip_first, self.rc, self.write, self.log_text = skip_first, rc, write, log_text
        self.calls = []

    def __call__(self, cmd, log_path, timeout_s):
        self.calls.append(cmd)
        work = log_path.parent
        assert cmd[cmd.index("-v") + 1] == f"{work}:/data"
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


def test_docker_command_layout(tmp_path):
    cmd = orbslam3_docker_command(DEFAULT_IMAGE, tmp_path, user="1000:1000")
    assert cmd == ["docker", "run", "--rm", "--user", "1000:1000", "-v", f"{tmp_path.resolve()}:/data", "-w", "/data",
                   "dataproc/orbslam3:v1.0", ORB_BIN, ORB_VOC, "/data/orbslam3_settings.yaml", "/data",
                   "/data/times.txt", "episode"]


def test_orbslam3_fake_docker_end_to_end(tmp_path, rig):
    ep = RawEpisode.open(analytic_episode(tmp_path, rig, imu_late_frames=3))
    fake = FakeDocker(log_text="Creation of new map with id: 0\nSYSTEM-> Reseting active map in monocular case\n"
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
        Orbslam3Backend(runner=FakeDocker(rc=1), scale=0.25).run(episode, rig, tmp_path / "a")
    with pytest.raises(SlamError, match="no trajectory"):
        Orbslam3Backend(runner=FakeDocker(write=False), scale=0.25).run(episode, rig, tmp_path / "b")


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
