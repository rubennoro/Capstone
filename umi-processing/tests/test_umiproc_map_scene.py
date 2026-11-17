"""The conventions M4 and M5 rely on, locked on one synthetic scene in which everything agrees: IMU samples in the IMU
frame (specific force ``R_imu_G (a + g up)``), camera poses ``T_W_cam = T_W_imu @ T_imu_cam`` in a SLAM world that is
the gravity frame rotated and shifted arbitrarily, and table tag 13 lying flat on the table, rendered through the
fisheye model at the keyframes and detected as ``umiproc map`` does. Then M4 passes (copied gravity alignment and
checks, orientation sub-check), ``up_direction`` is the true up, and M5's median normal and spread are ~0."""

from types import SimpleNamespace

import numpy as np
import pytest

from umiproc.dp.slam.base import finalize_trajectory
from umiproc.map import checks as C
from umiproc.map.tag import TableTagDetector, aggregate_sightings, collect_sightings
from umiproc.sensors import load_sensors
from umiproc_testlib import fork_trajectory, render_tag, rig_motion


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    sensors = load_sensors(tmp_path_factory.mktemp("root"), "rig_pi_v3", log=lambda *a, **k: None)
    T_imu_cam = sensors.rig.cameras["ultrawide"].T_imu_cam
    m = rig_motion(T_imu_cam)                                  # 30 s at 30 fps, IMU at 104 Hz
    fork = fork_trajectory(m["t_cam_ns"], m["T_W_cam"], keyframe_every=36)
    det = TableTagDetector(sensors)
    T_G_tag = np.eye(4)                                        # flat on the table: its z axis is up
    for k in np.flatnonzero(fork.is_keyframe):
        T_cam_tag = np.linalg.inv(m["T_G_cam"][k]) @ T_G_tag
        det.observe(int(k), render_tag(det.K, det.D, T_cam_tag, det.size_m))
    return sensors, m, fork, det


def test_m4_passes_and_up_is_the_true_up(scene):
    sensors, m, fork, _ = scene
    rig = sensors.rig
    imu_traj = C.imu_trajectory(fork, rig.cameras["ultrawide"].T_imu_cam)
    np.testing.assert_allclose(imu_traj.T(), m["T_W_imu"], atol=1e-9)
    episode = SimpleNamespace(imu=SimpleNamespace(t_ns=m["imu_t_ns"], gyro=m["gyro_imu"], accel=m["acc_imu"]))
    fin = finalize_trajectory(imu_traj, episode, rig, m["t_cam_ns"], align=True, align_min_deg=0.0)
    orient = C.orientation_consistency(imu_traj, m["imu_t_ns"], m["acc_imu"], t_ref_ns=int(m["t_cam_ns"][0]))
    c = C.check_m4(fin, None, C.max_speed(fork), orient)
    assert c.ok, c.message
    assert c.values["g_norm"] == pytest.approx(9.80665, abs=0.02) and c.values["scale"] == pytest.approx(1.0, abs=0.02)
    assert c.values["orientation_windows"] == 11 and c.values["orientation_max_deg"] < 1.0
    up = C.up_direction(fin)
    assert np.degrees(np.arccos(np.clip(up @ m["up_W"], -1, 1))) < 0.1
    assert np.degrees(np.arccos(np.clip(np.dot(orient["up_world"], m["up_W"]), -1, 1))) < 0.1


def test_m5_normal_and_spread_are_zero(scene):
    sensors, m, fork, det = scene
    n = len(m["t_cam_ns"])
    stats = collect_sightings(fork, np.arange(n), det.detections, sensors.ultrawide.resolution)
    assert len(det.detections) == int(fork.is_keyframe.sum()) == 25
    assert len(stats.sightings) >= C.M5_MIN_SIGHTINGS
    agg = aggregate_sightings(np.stack([s.T_world_tag for s in stats.sightings]))
    true_tag = m["T_W_G"]                                      # T_W_tag = T_W_G @ identity
    assert np.linalg.norm(agg["mean"] - true_tag[:3, 3]) < 0.003
    c = C.check_m5(stats, agg, m["up_W"], len(det.detections), n)
    assert c.ok, c.message
    assert c.values["normal_to_up_deg"] < 0.5 and c.values["chosen_normal_to_up_deg"] < 1.0
    assert c.values["spread_max_axis_m"] < 0.002
