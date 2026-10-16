"""T10: trajectory I/O, EuRoC/ORB-SLAM3/OpenVINS preparation, gravity and time-offset checks, pose-import."""

import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sensor_rig.pose.euroc import scaled_size, write_euroc
from sensor_rig.pose.frames import hom
from sensor_rig.pose.orbslam3 import write_orbslam3_settings
from sensor_rig.pose.stage import TrajectoryImportError, import_trajectory, prepare
from sensor_rig.pose.trajectory import (
    Trajectory,
    detect_time_unit,
    gravity_check,
    load_orbslam3_euroc,
    load_tum,
    save_tum,
    time_offset_check,
)
from sensor_rig.raw_episode import EUROC_IMU_HEADER, RawEpisode
from wpa_helpers import MONO_BASE, make_raw_episode, synth_imu, truth_pose

T0 = MONO_BASE + 20_000_000


@pytest.fixture(scope="module")
def raw(tmp_path_factory, test_rig):
    return RawEpisode(make_raw_episode(tmp_path_factory.mktemp("pose") / "raw", test_rig, duration_s=4.0))


def test_tum_round_trip_exact_ns(tmp_path):
    t = MONO_BASE + np.array([0, 16_666_667, 33_333_333, 50_000_001])
    T = truth_pose(np.arange(4) * 0.0167)
    tr = Trajectory.from_T(t, T, frame="imu0")
    assert tr.frame == "imu"
    save_tum(tr, tmp_path / "a.tum")
    back = load_tum(tmp_path / "a.tum")
    assert np.array_equal(back.t_ns, t)  # exact despite ~1e15 ns magnitudes
    assert np.abs(back.T() - T).max() < 1e-8


def test_unit_detection_and_orbslam3_format(tmp_path):
    assert detect_time_unit(np.array([1234567.89, 1234568.0])) == "s"
    assert detect_time_unit(np.array([1.23456789e15])) == "ns"
    # ORB-SLAM3 SaveTrajectoryEuRoC: fixed-point ns, setprecision(6)
    (tmp_path / "f.txt").write_text(
        "1234567890000000.000000 0.1 0.2 0.3 0 0 0 1\n1234567906666667.000000 0.1 0.2 0.3 0 0 0.0087265 0.9999619\n")
    tr = load_orbslam3_euroc(tmp_path / "f.txt")
    assert tr.t_ns.tolist() == [1234567890000000, 1234567906666667]
    assert load_tum(tmp_path / "f.txt").t_ns.tolist() == tr.t_ns.tolist()  # auto unit
    (tmp_path / "s.txt").write_text("# comment\n1234567.890000000 0 0 0 0 0 0 1\n1234567.906666667 0 0 0 0 0 0 1\n")
    assert load_tum(tmp_path / "s.txt").t_ns.tolist() == [1234567890000000, 1234567906666667]


def test_sample_validity():
    t = MONO_BASE + np.array([0, 100, 200, 600], dtype=np.int64) * 1_000_000
    tr = Trajectory.from_T(t, truth_pose(np.array([0, 0.1, 0.2, 0.6])))
    q = MONO_BASE + np.array([-1, 0, 50, 200, 300, 600, 601], dtype=np.int64) * 1_000_000
    T, valid = tr.sample(q, max_gap_ns=150_000_000)
    assert valid.tolist() == [False, True, True, True, False, True, False]
    assert np.allclose(T[1], tr.T()[0]) and np.allclose(T[5], tr.T()[3])
    assert np.allclose(T[2][:3, 3], (tr.p[0] + tr.p[1]) / 2)


def test_scale_snaps_to_exact_uniform_scale(test_rig, tmp_path):
    assert scaled_size((1640, 1232), 0.4) == (615, 462)  # 656x493 would not be a uniform scale
    assert scaled_size((1640, 1232), 0.5) == (820, 616)
    assert scaled_size((640, 480), 1.0) == (640, 480)
    with pytest.raises(ValueError):
        scaled_size((640, 480), 1.5)
    p = write_orbslam3_settings(test_rig, "ultrawide", 0.37, tmp_path / "s.yaml")  # 640x480 -> 236x177 (59 x 4:3)
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    assert fs.getNode("Camera.width").real() == 236 and fs.getNode("Camera.height").real() == 177


def test_euroc_layout_and_corrected_times(raw, test_rig, tmp_path):
    info = write_euroc(raw, tmp_path / "seq", camera="ultrawide", scale=0.5, imu_rate_hz=200)
    cam = (tmp_path / "seq" / "mav0" / "cam0" / "data.csv").read_text().splitlines()
    assert cam[0] == "#timestamp [ns],filename" and len(cam) - 1 == info["n_images"]
    t_first = int(cam[1].split(",")[0])
    assert t_first == int(raw.camera_times("ultrawide")[0]) + test_rig.cameras["ultrawide"].time.total_offset_ns
    img = cv2.imread(str(tmp_path / "seq" / "mav0" / "cam0" / "data" / cam[1].split(",")[1]), cv2.IMREAD_UNCHANGED)
    assert img.ndim == 2 and img.shape == (240, 320)
    imu = (tmp_path / "seq" / "mav0" / "imu0" / "data.csv").read_text().splitlines()
    assert imu[0] == EUROC_IMU_HEADER and info["imu_decimation"] == 2 and abs(info["imu_rate_hz"] - 200) < 1
    times = (tmp_path / "seq" / "times.txt").read_text().split()
    assert int(times[0]) == t_first


def test_orbslam3_settings_parse(test_rig, tmp_path):
    p = write_orbslam3_settings(test_rig, "ultrawide", 0.5, tmp_path / "s.yaml")
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    assert fs.getNode("Camera.type").string() == "KannalaBrandt8"
    assert fs.getNode("Camera.width").real() == 320
    assert fs.getNode("Camera1.fx").real() == pytest.approx(287.0 * 0.5)
    for k in ("Camera1.k1", "Camera1.k4", "IMU.NoiseGyro", "IMU.NoiseAcc", "IMU.GyroWalk", "IMU.AccWalk",
              "IMU.Frequency", "ORBextractor.nFeatures"):
        assert not fs.getNode(k).empty(), k
    Tbc = fs.getNode("IMU.T_b_c1").mat()
    assert np.allclose(Tbc, test_rig.T("imu", "ultrawide"), atol=1e-6)
    p2 = write_orbslam3_settings(test_rig, "main", 1.0, tmp_path / "m.yaml")
    fs2 = cv2.FileStorage(str(p2), cv2.FILE_STORAGE_READ)
    assert fs2.getNode("Camera.type").string() == "PinHole" and not fs2.getNode("Camera1.p1").empty()


def test_prepare_targets(raw, test_rig, tmp_path):
    info = prepare(raw, tmp_path, target="orbslam3", scale=0.5, imu_rate_hz=200)
    assert (tmp_path / raw.name / "run_orbslam3.sh").is_file() and info["n_images"] > 0
    info = prepare(raw, tmp_path / "ov", target="openvins")
    fs = cv2.FileStorage(info["kalibr"]["imu"], cv2.FILE_STORAGE_READ)
    assert not fs.getNode("imu0").empty()


def _imu(t_ns):
    g, a = synth_imu((t_ns - T0) * 1e-9)
    return g, a


def test_gravity_check(test_rig):
    t = T0 + np.arange(0, 3_000_000_000, 10_000_000, dtype=np.int64)
    tr = Trajectory.from_T(t, truth_pose((t - T0) * 1e-9), frame="imu")
    ti = T0 + np.arange(0, 3_000_000_000, 2_500_000, dtype=np.int64)
    _, acc = _imu(ti)
    res = gravity_check(tr, ti, acc, test_rig)
    assert res["ok"] and res["angle_to_up_deg"] < 1.0, res
    tilt = hom(Rotation.from_euler("x", 20, degrees=True).as_matrix())
    tr_bad = Trajectory.from_T(t, tilt @ tr.T(), frame="imu")
    bad = gravity_check(tr_bad, ti, acc, test_rig)
    assert not bad["ok"] and abs(bad["angle_to_up_deg"] - 20) < 1.5
    # the same motion expressed as the ultrawide's pose gives the same answer
    tr_u = Trajectory.from_T(t, tr.T() @ test_rig.T("imu", "ultrawide"), frame="ultrawide")
    assert gravity_check(tr_u, ti, acc, test_rig)["ok"]


def test_time_offset_check():
    t = T0 + np.arange(0, 6_000_000_000, 10_000_000, dtype=np.int64)
    ti = T0 + np.arange(-200_000_000, 6_200_000_000, 1_000_000, dtype=np.int64)
    gyro, _ = _imu(ti)
    tr = Trajectory.from_T(t + 12_000_000, truth_pose((t - T0) * 1e-9))  # trajectory stamps 12 ms late
    res = time_offset_check(tr, ti, gyro)
    assert abs(res["traj_delay_ms"] - 12.0) < 2.0 and res["corr"] > 0.9 and not res["ok"], res
    tr0 = Trajectory.from_T(t, truth_pose((t - T0) * 1e-9))
    res0 = time_offset_check(tr0, ti, gyro)
    assert abs(res0["traj_delay_ms"]) < 2.0 and res0["ok"], res0


def test_pose_import(raw, test_rig, tmp_path):
    t = raw.camera_times("main") + test_rig.cameras["main"].time.total_offset_ns
    tr = Trajectory.from_T(t, truth_pose((t - T0) * 1e-9))
    save_tum(tr, tmp_path / "vio.tum")
    info = import_trajectory(raw, tmp_path / "vio.tum", frame="imu0")
    assert info["gravity_check"]["ok"] and info["main_overlap_fraction"] > 0.99
    assert json.loads((raw.path / "trajectory" / "trajectory.json").read_text())["frame"] == "imu"
    assert np.array_equal(load_tum(raw.trajectory_path()).t_ns, t)
    assert not info["gravity_alignment"]["applied"] and info["gravity_alignment"]["angle_deg"] < 1.0  # level: kept
    # a tilted world is refused when gravity alignment is switched off (and not forced)
    tilt = hom(Rotation.from_euler("y", 25, degrees=True).as_matrix())
    save_tum(Trajectory.from_T(t, tilt @ tr.T()), tmp_path / "bad.tum")
    with pytest.raises(TrajectoryImportError):
        import_trajectory(raw, tmp_path / "bad.tum", gravity_align=False)
    # a trajectory on the wrong clock (no overlap) is refused
    save_tum(Trajectory.from_T(t - 10**12, tr.T()), tmp_path / "far.tum")
    with pytest.raises(TrajectoryImportError, match="overlaps"):
        import_trajectory(raw, tmp_path / "far.tum")
    # restore the good trajectory for other tests sharing the episode
    import_trajectory(raw, tmp_path / "vio.tum")


def test_tilted_world_is_gravity_aligned(tmp_path, test_rig):
    """F2: a VIO world tilted 30 deg (like ORB-SLAM3's first-keyframe frame) imports and exports the same poses as the
    gravity-aligned one, up to yaw/translation: identical relative poses and identical up-vector in the camera."""
    from sensor_rig.export.exporter import ExportOptions, export_episode

    ep = RawEpisode(make_raw_episode(tmp_path / "raw", test_rig, duration_s=3.0))
    t = ep.camera_times("main") + test_rig.cameras["main"].time.total_offset_ns
    T = truth_pose((t - T0) * 1e-9)
    tilt = hom(Rotation.from_euler("xz", [30, 50], degrees=True).as_matrix(), [0.3, -2.0, 1.0])
    out = {}
    for label, poses in (("level", T), ("tilted", tilt @ T)):
        save_tum(Trajectory.from_T(t, poses), tmp_path / f"{label}.tum")
        info = import_trajectory(ep, tmp_path / f"{label}.tum")
        assert info["gravity_check"]["ok"], info
        r = export_episode(ep.path, tmp_path / label, ExportOptions())
        assert r.status == "exported", r
        out[label] = (np.asarray(json.loads((r.out_dir / f"{r.name}.json").read_text())["poseTransforms"]), info)
    ga = out["tilted"][1]["gravity_alignment"]
    assert ga["applied"] and abs(ga["angle_deg"] - 30) < 1.0
    A, B = out["level"][0], out["tilted"][0]
    # world-up seen from the camera agrees to within the mean-acceleration estimate (< ~1 deg)
    assert np.abs(A[:, 1, :3] - B[:, 1, :3]).max() < 0.02
    relA = np.linalg.inv(A[0]) @ A
    relB = np.linalg.inv(B[0]) @ B
    assert np.abs(relA - relB).max() < 1e-4
