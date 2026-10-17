"""Picamera2Camera with a fake ``picamera2`` module (``fake_picamera2.py``), and a whole Pi-path episode: recorder with
``backends: picamera2`` + mock IMU -> pose-import of a synthetic trajectory -> export -> validate.

Hardware-only (NOT covered here, see HARDWARE_CHECKLIST.md "Raspberry Pi"): real libcamera mode lists and ISP sizes,
SensorTimestamp clock domain on the actual kernel, focus / exposure locks, two cameras + SPI under load, thermals.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
import yaml

import fake_picamera2 as fake
from sensor_rig.drivers.base import BackendError, SimClock, make_camera
from sensor_rig.drivers.mock_imu import MockImu
from sensor_rig.drivers.picamera2_camera import Picamera2Camera
from sensor_rig.raw_episode import RawEpisode
from sensor_rig.rig_config import load_rig

FPS = 30.0
PERIOD = 1e9 / FPS


@pytest.fixture
def pi_test_rig(test_rig_path):
    """The WP-A test rig (640x480 @ 30 fps cameras) recorded through the picamera2 backend."""
    raw = yaml.safe_load(test_rig_path.read_text())
    for i, cam in enumerate(("main", "ultrawide")):
        c = raw["cameras"][cam]
        c["sensor"], c["sensor_mode"], c["argus"] = "fakecam", None, {}
        c["picamera2"] = {"camera_num": i, "model": "fakecam", "exposure_us": 3000, "analogue_gain": 2.0,
                          "colour_gains": [1.9, 1.6], "buffer_count": 4}
    raw["cameras"]["main"]["picamera2"]["lens_position"] = 2.5
    raw["capture"] = {"backends": {"main": "picamera2", "ultrawide": "picamera2", "imu": "mock"},
                      "jpeg": {"encoder": "opencv", "quality": 90, "workers_per_camera": 2}}
    return load_rig(raw)


def _specs(**main_kw):
    return [fake.FakeCameraSpec(autofocus=True, **main_kw), fake.FakeCameraSpec()]


def _read(cam, n, timeout_s=2.0):
    out = []
    for _ in range(n):
        f = cam.read(timeout_s=timeout_s)
        assert f is not None, f"timed out after {len(out)} frames"
        out.append(f)
    return out


def test_frames_order_stamps_bgr_and_controls(monkeypatch, pi_test_rig):
    clock = SimClock(speed=4.0)
    state = fake.install(monkeypatch, _specs(), clock)
    cam = make_camera(pi_test_rig, "main", clock=clock)
    assert isinstance(cam, Picamera2Camera)
    with cam:
        frames = _read(cam, 30)
        recent = cam.read_recent(8)
        info = cam.info()
    picam = state.instances[0]
    seq = [f.seq for f in frames]
    assert seq == list(range(30))
    t = np.array([f.t_ns for f in frames])
    np.testing.assert_allclose(np.diff(t), PERIOD, atol=1000)   # frame duration is integer us (33333)
    # t_ns IS the SensorTimestamp (Pi 5: CLOCK_MONOTONIC start of frame), no offsets applied in the backend
    assert all(f.t_ns == f.meta["sensor_ts_raw_ns"] for f in frames)
    lat = np.array([f.t_arrival_ns - f.t_ns for f in frames])
    assert lat.min() >= 8_000_000 and np.median(lat) < 30_000_000
    assert info["latency_negative"] == 0 and 8.0 <= info["latency_ms_median"] < 30.0
    # BGR: the fake scene's red square is [B, G, R] = (40, 40, 230) in memory, used as-is
    img = frames[0].image
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8 and img.flags["C_CONTIGUOUS"]
    red = np.all(img == fake.RED_BGR, axis=2)
    assert red.sum() > 1000 and np.all(img[0, 0] == fake.BACKGROUND_BGR)
    assert frames[0].meta["exposure_ns"] == 3_000_000 and frames[0].meta["gain"] == 2.0
    # ring: newest last, at least 8 deep
    assert len(recent) == 8 and [f.seq for f in recent] == sorted(f.seq for f in recent)
    # configuration and locked controls
    cfg = picam.camera_config
    assert cfg["main"] == {"size": (640, 480), "format": "RGB888"} and cfg["sensor"]["output_size"] == (640, 480)
    assert cfg["buffer_count"] == 4 and cfg["queue"] is False and cfg["transform"] is None
    c = picam.controls_set
    assert c["FrameDurationLimits"] == (33333, 33333) and c["AeEnable"] is False and c["ExposureTime"] == 3000
    assert c["AwbEnable"] is False and c["ColourGains"] == (1.9, 1.6)
    assert c["AfMode"] == 0 and c["LensPosition"] == 2.5 and c["ScalerCrop"] == (0, 0, 1280, 960)
    assert info["sensor_mode"] == "640x480" and info["camera_num"] == 0 and info["model"] == "fakecam"
    assert info["sensor_crop"] == [0, 0, 1280, 960] and info["exposure_us"] == 3000
    # clean stop
    assert picam.closed and not picam.started and cam._picam2 is None and not cam.running
    assert cam.read(timeout_s=0.05) is None


def test_sensor_gaps_advance_seq(monkeypatch, pi_test_rig):
    clock = SimClock(speed=4.0)
    fake.install(monkeypatch, _specs(drop_frames={5, 6, 20}), clock)
    with make_camera(pi_test_rig, "main", clock=clock) as cam:
        frames = _read(cam, 25)
    seq = [f.seq for f in frames]
    # source frames 5, 6 and 20 never reached userspace: seq jumps exactly there (the recorder counts sensor_gap)
    assert seq[:5] == [0, 1, 2, 3, 4] and seq[5] == 7 and 20 not in seq and seq[18] == 21
    assert cam.stats["sensor_gaps"] == 2 and cam.stats["sensor_missing"] == 3
    t = np.array([f.t_ns for f in frames])
    np.testing.assert_allclose((t - t[0]) / PERIOD, np.array(seq) - seq[0], atol=1e-3)


def test_internal_drops_when_consumer_stalls(monkeypatch, pi_test_rig):
    clock = SimClock(speed=8.0)
    fake.install(monkeypatch, _specs(), clock)
    cam = make_camera(pi_test_rig, "main", clock=clock, queue_size=3)
    with cam:
        clock.sleep(1.0)                  # ~30 frames produced, nobody reads
        latest = cam.latest()
        got = []
        while (f := cam.read(timeout_s=0.0)) is not None:
            got.append(f)
    assert cam.stats["internal_drops"] > 10 and len(got) <= 3
    assert [f.seq for f in got] == sorted(f.seq for f in got) and got[-1].seq <= latest.seq
    assert cam.stats["frames"] >= cam.stats["internal_drops"] + len(got)   # every frame queued is read or dropped


def test_boottime_conversion_and_stride(monkeypatch, pi_test_rig):
    clock = SimClock(speed=4.0)
    fake.install(monkeypatch, _specs(ts_offset_ns=7_000_000_000, stride_pad=16), clock)
    monkeypatch.setattr("sensor_rig.drivers.picamera2_camera.boot_minus_mono_ns", lambda: 7_000_000_000)
    cam = make_camera(pi_test_rig, "main", clock=clock, settings_overrides={"timestamp_clock": "boottime"})
    with cam:
        f = _read(cam, 3)[-1]
    assert f.meta["sensor_ts_raw_ns"] - f.t_ns == 7_000_000_000
    assert 0 < f.t_arrival_ns - f.t_ns < 40_000_000          # back on the rig clock
    assert f.image.shape == (480, 640, 3)                     # stride padding cut off


def test_flips_and_model_lookup(monkeypatch, pi_test_rig):
    clock = SimClock(speed=4.0)
    specs = [fake.FakeCameraSpec(model="othercam"), fake.FakeCameraSpec(autofocus=True)]
    state = fake.install(monkeypatch, specs, clock)
    # main says camera_num 0 / model fakecam, but camera 0 is now "othercam": the unique fakecam (1) is used
    cam = make_camera(pi_test_rig, "main", clock=clock, settings_overrides={"hflip": True, "vflip": True})
    with cam:
        _read(cam, 2)
        assert cam.info()["camera_num"] == 1
    tr = state.instances[0].camera_config["transform"]
    assert isinstance(tr, fake.Transform) and (tr.hflip, tr.vflip) == (1, 1)


@pytest.mark.parametrize("specs, overrides, match", [
    ([fake.FakeCameraSpec(), fake.FakeCameraSpec()], {}, "does not support"),                 # lens_position on fixed focus
    ([fake.FakeCameraSpec(autofocus=True), fake.FakeCameraSpec()], {"lens_position": None}, "autofocus lens"),
    ([fake.FakeCameraSpec(autofocus=True, adjust_main_size=(640, 482)), fake.FakeCameraSpec()], {},
     "calibration resolution"),
    ([fake.FakeCameraSpec(autofocus=True, modes=[(640, 480, (320, 240, 640, 480), 60.0, 10)]),
      fake.FakeCameraSpec()], {}, "crops the sensor"),
    ([fake.FakeCameraSpec(model="imx477")], {}, "no unique match"),
])
def test_start_refuses_bad_setups(monkeypatch, pi_test_rig, specs, overrides, match):
    clock = SimClock(speed=4.0)
    state = fake.install(monkeypatch, specs, clock)
    cam = make_camera(pi_test_rig, "main", clock=clock, settings_overrides=overrides)
    with pytest.raises(BackendError, match=match):
        cam.start()
    assert all(p.closed or not p.started for p in state.instances)


def test_no_frames_raises(monkeypatch, pi_test_rig):
    clock = SimClock(speed=4.0)
    state = fake.install(monkeypatch, _specs(), clock)
    cam = make_camera(pi_test_rig, "main", clock=clock, start_timeout_s=0.3)
    cam.start()
    state.instances[0].post_callback = None          # frames go nowhere
    try:
        deadline = time.monotonic() + 3.0
        with pytest.raises(BackendError, match="no frame"):
            while time.monotonic() < deadline:
                cam.read(timeout_s=0.1)
    finally:
        cam.stop()


# ---------------------------------------------------------------------------------------------- whole episode
def _record(rec, seconds, **kw):
    kw.setdefault("session", "pi-test")
    kw.setdefault("side", "left")
    name = rec.start_episode(**kw)
    rec.clock.sleep_until(rec.clock.now_ns() + int(seconds * 1e9))
    return name, rec.stop_episode()


def test_pi_episode_records_and_exports(monkeypatch, pi_test_rig, tmp_path):
    """An episode collected on the Pi path (Picamera2 backends from rig.capture + mock IMU, real recorder) gets a
    trajectory via pose-import and exports to a valid iPhone-format episode."""
    from sensor_rig.cli import main as cli_main
    from sensor_rig.export.exporter import export_episode
    from sensor_rig.export.validate import validate_iphone_episode
    from sensor_rig.pose.trajectory import Trajectory, save_tum
    from sensor_rig.record.recorder import Recorder

    clock = SimClock(speed=4.0)
    state = fake.install(monkeypatch, _specs(), clock)
    raw_root = tmp_path / "raw"
    rec = Recorder(pi_test_rig, raw_root, imu=MockImu.static(rate_hz=1000, clock=clock), clock=clock,
                   warmup_s=1.2, monitor_period_s=0.5)
    assert {type(c).__name__ for c in rec.cameras.values()} == {"Picamera2Camera"}
    with rec:
        cal_name, cal_path = _record(rec, 1.5, kind="gripper_calibration")
        name, path = _record(rec, 3.0, kind="demonstration")
    assert all(p.closed for p in state.instances)

    meta = json.loads((path / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["gripper_calibration_run_name"] == cal_name
    ep = RawEpisode(path)
    for cam in ("main", "ultrawide"):
        s = meta["sensors"][cam]
        assert s["backend"] == "picamera2" and s["model"] == "fakecam" and s["sensor_mode"] == "640x480"
        assert s["fps_measured"] == pytest.approx(FPS, rel=0.03) and s["latency_negative"] == 0
        assert (path / f"cam_{cam}" / "drops.csv").read_text().count("\n") == 1      # header only
        t = ep.camera_times(cam)
        np.testing.assert_allclose(np.diff(t), PERIOD, atol=1000)   # frame duration is integer us (33333)
        img = ep.read_image(cam, 0)                       # JPEG round trip keeps BGR: the square is still red
        b, g, r = (img[..., i].astype(int) for i in range(3))
        assert ((r > 180) & (b < 90)).sum() > 1000
    # a smooth synthetic trajectory (IMU frame, gravity-aligned: yaw-only rotation) on the corrected clock
    t_main = pi_test_rig.capture_time_ns("main", ep.camera_times("main"))
    tt = np.arange(t_main[0] - 100_000_000, t_main[-1] + 100_000_000, 10_000_000, dtype=np.int64)
    s = (tt - tt[0]) * 1e-9
    T = np.tile(np.eye(4), (len(tt), 1, 1))
    yaw = 0.3 + 0.2 * np.sin(0.7 * s)
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(yaw), -np.sin(yaw), np.sin(yaw), np.cos(yaw)
    T[:, :3, 3] = np.stack([0.1 * np.sin(s), 0.05 * s, 0.3 + 0.02 * np.cos(2 * s)], axis=1)
    traj_file = tmp_path / "traj.tum"
    save_tum(Trajectory.from_T(tt, T, frame="imu"), traj_file)
    assert cli_main(["pose-import", "--raw", str(path), "--traj", str(traj_file), "--format", "tum",
                     "--frame", "imu"]) == 0
    assert (path / "trajectory" / "trajectory.tum").is_file()

    out = tmp_path / "export"
    rc = export_episode(cal_path, out)
    r = export_episode(path, out)
    assert rc.status == "exported" and r.status == "exported", (rc, r)
    j = json.loads((r.out_dir / f"{r.out_dir.name}.json").read_text())
    assert j["gripperCalibrationRunName"] == cal_name and len(j["poseTimes"]) >= 150
    assert validate_iphone_episode(r.out_dir, decode=True) == []
    assert cli_main(["validate", str(out)]) == 0
    frame = cv2.imread(str(next((path / "cam_main" / "data").iterdir())))
    assert frame.shape == (480, 640, 3)
