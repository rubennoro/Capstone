"""Backend interface + mocks: SimClock, MockCamera (synthetic / arrays, read / latest / read_recent, internal drops,
sensor drops), MockImu (static), factories."""

from __future__ import annotations

import time
import types

import numpy as np
import pytest

from sensor_rig.drivers.base import BackendError, ImuBatch, SimClock, make_camera, make_imu
from sensor_rig.drivers.mock_camera import MockCamera
from sensor_rig.drivers.mock_imu import MockImu


def test_simclock_scaled():
    c = SimClock(speed=50.0)
    t0 = c.now_ns()
    c.sleep_until(t0 + 500_000_000)        # 0.5 s sim = 10 ms real
    assert c.now_ns() >= t0 + 500_000_000
    s = c.sample()
    assert s.monotonic_ns == s.monotonic_raw_ns and s.realtime_ns - s.monotonic_ns == c.realtime_offset_ns
    with pytest.raises(ValueError):
        SimClock(speed=0)


def test_mock_camera_synthetic_read_latest_recent():
    clk = SimClock(speed=20.0)
    cam = MockCamera.synthetic("main", 64, 48, 60, clock=clk)
    with cam:
        frames = [cam.read(timeout_s=1.0) for _ in range(30)]
    assert all(f is not None for f in frames)
    t = np.array([f.t_ns for f in frames])
    np.testing.assert_allclose(np.diff(t), 1e9 / 60, atol=1)
    assert [f.seq for f in frames] == list(range(30))
    assert frames[0].image.shape == (48, 64, 3) and frames[0].image.dtype == np.uint8
    assert all(f.t_arrival_ns >= f.t_ns for f in frames)
    assert not np.array_equal(frames[0].image, frames[1].image)   # pattern moves
    rec = cam.read_recent(5)
    assert len(rec) == 5 and all(a.t_ns < b.t_ns for a, b in zip(rec, rec[1:]))
    assert cam.latest() is rec[-1]
    assert len(cam.read_recent(100)) >= 8


def test_mock_camera_internal_drops_and_sensor_drops():
    clk = SimClock(speed=50.0)
    cam = MockCamera.synthetic("main", 32, 24, 100, clock=clk, queue_size=4, drop_seqs={3, 4}, draw_number=False)
    cam.start()
    time.sleep(0.05)                       # ~250 frames of sim time without reading
    f = cam.read()
    cam.stop()
    assert cam.stats["internal_drops"] > 0
    assert f is not None and f.seq > 4
    cam2 = MockCamera.synthetic("x", 32, 24, 100, clock=SimClock(speed=20.0), drop_seqs={3, 4}, n_frames=8,
                                draw_number=False)
    with cam2:
        seqs = []
        while True:
            fr = cam2.read(timeout_s=1.0)
            if fr is None:
                break
            seqs.append(fr.seq)
    assert seqs == [0, 1, 2, 5, 6, 7] and cam2.exhausted


def test_mock_camera_from_arrays_offline():
    imgs = [np.full((10, 12, 3), i, np.uint8) for i in range(20)]
    t = 10**12 + np.arange(20) * 33_333_333
    cam = MockCamera.from_arrays(imgs, t, name="ultrawide", queue_size=3)   # realtime=False: producer blocks
    with cam:
        got = []
        while (f := cam.read(timeout_s=1.0)) is not None:
            got.append(f)
    assert [f.t_ns for f in got] == list(t) and cam.stats["internal_drops"] == 0
    assert got[7].image[0, 0, 0] == 7 and cam.fps == pytest.approx(30.0, rel=1e-6)
    cam2 = MockCamera.from_arrays(imgs, t, preserve_timestamps=False, clock=SimClock(speed=100.0), realtime=True)
    with cam2:
        f0 = cam2.read()
        f1 = cam2.read()
    assert f1.t_ns - f0.t_ns == 33_333_333 and f0.t_ns != t[0]


def test_camera_backend_error_surfaces():
    def bad_frame(k):
        raise IOError("sensor unplugged")

    cam = MockCamera("main", 8, 8, 30, bad_frame, lambda k, t0: t0 + k, realtime=False)
    cam.start()
    with pytest.raises(BackendError, match="unplugged"):
        cam.read(timeout_s=1.0)
    cam.stop()


def test_mock_imu_static():
    clk = SimClock(speed=20.0)
    imu = MockImu.static(rate_hz=1000, clock=clk, seed=1)
    with imu:
        clk.sleep(0.5)
        b = imu.read_batch(timeout_s=1.0)
    assert b is not None and len(b) > 300
    assert np.all(np.diff(b.t_ns) > 0)
    np.testing.assert_allclose(np.diff(b.t_ns), 1e6, atol=1)
    np.testing.assert_allclose(b.accel.mean(axis=0), [0, 0, 9.80665], atol=0.01)
    assert np.abs(b.gyro.mean(axis=0)).max() < 1e-3
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)    # IMU z axis horizontal
    imu2 = MockImu.static(rate_hz=200, clock=SimClock(speed=20.0), R_world_imu=R, gyro_noise_density=0,
                          accel_noise_density=0)
    with imu2:
        imu2.clock.sleep(0.2)
        b2 = imu2.read_batch(timeout_s=1.0)
    np.testing.assert_allclose(b2.accel[0], R.T @ [0, 0, 9.80665], atol=1e-12)


def test_imu_batch_concat_select():
    a = ImuBatch(np.array([1, 2]), np.zeros((2, 3)), np.ones((2, 3)), {"chip_us": np.array([10, 20])})
    b = ImuBatch(np.array([3]), np.zeros((1, 3)), np.ones((1, 3)), {"chip_us": np.array([30])})
    c = ImuBatch.concat([a, b])
    assert list(c.t_ns) == [1, 2, 3] and list(c.aux["chip_us"]) == [10, 20, 30]
    assert list(c.select(c.t_ns > 1).t_ns) == [2, 3]


def _stub_rig(backends=None):
    cams = {n: types.SimpleNamespace(resolution=(80, 60), fps=f, sensor_id=i, sensor_mode=3, argus={})
            for i, (n, f) in enumerate((("main", 60.0), ("ultrawide", 30.0)))}
    imu = types.SimpleNamespace(odr_hz=1000, output_rate_hz=500)
    return types.SimpleNamespace(cameras=cams, imu=imu, capture={"backends": backends or {}})


def test_factories_mock():
    rig = _stub_rig({"main": "mock", "ultrawide": "mock", "imu": "mock"})
    cam = make_camera(rig, "ultrawide", clock=SimClock())
    assert isinstance(cam, MockCamera) and (cam.width, cam.height, cam.fps) == (80, 60, 30.0)
    imu = make_imu(rig, clock=SimClock())
    assert isinstance(imu, MockImu) and imu.rate_hz == 500
    with pytest.raises(ValueError):
        make_camera(rig, "main", backend="v4l2")
    # default backend for a rig without capture: argus (constructing it does not import GStreamer)
    argus = make_camera(_stub_rig(), "main")
    assert type(argus).__name__ == "ArgusCamera" and "sensor-id=0" in argus.pipeline_str


def test_factories_with_real_rig(test_rig):
    cam = make_camera(test_rig, "main", backend="mock", clock=SimClock())
    assert (cam.width, cam.height) == tuple(test_rig.cameras["main"].resolution)
    assert make_camera(test_rig, "main").__class__.__name__ == "ArgusCamera"
    spi = make_imu(test_rig)            # icm42688_spi by default; spidev only imported on start()
    assert spi.__class__.__name__ == "Icm42688Spi" and spi.packet_size == 20
