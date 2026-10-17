"""T5b: the argus_sof backend (libargus start-of-frame stamps) against a fake helper process, and T5a stamp-offset."""

from __future__ import annotations

import dataclasses
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sensor_rig.clock import ClockSample
from sensor_rig.drivers.argus_sof_camera import ArgusSofCamera
from sensor_rig.drivers.argus_sof_logic import FrameMsg, ProtocolError, Ready, helper_args, parse_line
from sensor_rig.drivers.base import BackendError, make_camera

FAKE = Path(__file__).resolve().parent / "fake_argus_sof.py"
W, H = 64, 48


@pytest.fixture
def helper(tmp_path, monkeypatch):
    exe = tmp_path / "argus_sof"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setenv("FAKE_SOF_SHM_DIR", str(shm))
    return exe, shm


def cam_for(helper, **kw):
    exe, shm = helper
    params = {"sensor_id": 0, "sensor_mode": 3, "width": W, "height": H, "fps": 30.0,
              "exposure_time_range_ns": (3_000_000, 3_000_000), "gain_range": (4.0, 4.0), "aelock": True,
              "awblock": True}
    kw.setdefault("startup_timeout_s", 3.0)
    return ArgusSofCamera("main", [str(exe), *helper_args(params, f"t{os.getpid()}_{time.monotonic_ns()}")], W, H,
                          30.0, shm_dir=str(shm), **kw)


def read_n(cam, n, timeout=5.0):
    out, t0 = [], time.monotonic()
    while len(out) < n and time.monotonic() - t0 < timeout:
        f = cam.read(timeout_s=0.5)
        if f is not None:
            out.append(f)
    return out


def test_parse_line():
    r = parse_line(f"READY {W} {H} NV12 8 {16 + W * H * 3 // 2} shm1\n")
    assert r == Ready(W, H, "NV12", 8, 16 + W * H * 3 // 2, "shm1")
    assert parse_line("F 12 1000000000123 3000000 4.0000 4") == FrameMsg(12, 1000000000123, 3000000, 4.0, 4)
    assert parse_line("E camera busy") == ("error", "camera busy")
    assert parse_line("") is None and parse_line("hello") is None
    with pytest.raises(ProtocolError):
        parse_line("READY 64 48 NV12 8 99 shm")             # slot size does not match
    with pytest.raises(ProtocolError):
        parse_line("F 1 x 2 3 4")


def test_helper_args_from_rig_params():
    a = helper_args({"sensor_id": 1, "sensor_mode": None, "width": 1640, "height": 1232, "fps": 30.0,
                     "exposure_time_range_ns": (3000000, 3000000), "gain_range": (4.0, 4.0), "aelock": True}, "s")
    d = dict(zip(a[::2], a[1::2]))
    assert d["--sensor-id"] == "1" and d["--mode"] == "-1" and d["--exposure-ns"] == "3000000,3000000"
    assert d["--ae-lock"] == "1" and d["--awb-lock"] == "0" and d["--shm-name"] == "s"


def test_frames_pixels_and_clock_conversion(helper):
    raw_minus_mono = 5_000_000_000

    def sampler():                                   # fake clocks: MONOTONIC_RAW = MONOTONIC + 5 s
        m = time.monotonic_ns()
        return ClockSample(m, m + raw_minus_mono, time.time_ns(), 0)

    cam = cam_for(helper, clock_sampler=sampler)
    cam.start()
    try:
        frames = read_n(cam, 10)
    finally:
        cam.stop()
    assert len(frames) == 10
    for f in frames:
        assert f.t_ns == f.meta["sensor_ts_raw_ns"] - raw_minus_mono       # RAW -> MONOTONIC
        assert f.meta["sensor_ts_raw_ns"] == 1_000_000_000_000 + f.seq * 33_333_333
        assert f.image.shape == (H, W, 3) and f.meta["exposure_ns"] == 3_000_000
    gray = int(np.median(frames[3].image))
    y = (frames[3].seq * 7) % 256
    assert abs(gray - np.clip((y - 16) * 255 / 219, 0, 255)) <= 2      # NV12 luma (BT.601 video range) survived
    assert cam.info()["backend"] == "argus_sof" and cam.stats["sensor_gaps"] == 0
    assert not list(Path(helper[1]).iterdir())                          # ring file removed on stop


def test_seq_gap_counted(helper, monkeypatch):
    monkeypatch.setenv("FAKE_SOF_GAP_AT", "4")
    cam = cam_for(helper)
    cam.start()
    try:
        frames = read_n(cam, 9)
    finally:
        cam.stop()
    assert [f.seq for f in frames][:6] == [0, 1, 2, 3, 5, 6] and cam.stats["sensor_gaps"] == 1


def test_helper_crash_raises_backend_error(helper, monkeypatch):
    monkeypatch.setenv("FAKE_SOF_CRASH_AT", "2")
    cam = cam_for(helper)
    cam.start()
    try:
        with pytest.raises(BackendError, match=r"exited \(rc=3\).*NvBufSurfaceMap failed"):
            for _ in range(20):
                cam.read(timeout_s=0.5)
    finally:
        cam.stop()


def test_startup_timeout_is_bounded(helper, monkeypatch):
    monkeypatch.setenv("FAKE_SOF_NO_READY", "1")
    cam = cam_for(helper, startup_timeout_s=0.8)
    t0 = time.monotonic()
    with pytest.raises(BackendError, match="no READY within 0.8 s"):
        cam.start()
    assert time.monotonic() - t0 < 4.0
    cam.stop()


def test_helper_error_and_missing_binary(helper, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_SOF_ERROR", "cannot create a capture session (camera busy?)")
    cam = cam_for(helper)
    with pytest.raises(BackendError, match="camera busy"):
        cam.start()
    cam.stop()
    missing = ArgusSofCamera("main", [str(tmp_path / "nope")], W, H, 30.0)
    with pytest.raises(BackendError, match="build it"):
        missing.start()


def test_factory_and_doctor_know_argus_sof(test_rig, helper, tmp_path):
    from sensor_rig import doctor as D

    cap = {**test_rig.capture, "backends": {"main": "argus_sof", "ultrawide": "argus_sof", "imu": "mock"},
           "argus_sof": {"helper": str(helper[0])}}
    rig = test_rig.replace(capture=cap)
    cam = make_camera(rig, "main")
    assert isinstance(cam, ArgusSofCamera) and cam.cmd[0] == str(helper[0])
    from test_doctor import FakeProbes

    probes = FakeProbes()
    probes.paths = probes.paths | {str(helper[0])}
    rep = D.run_doctor("rig.yaml", D.DoctorOptions(out=str(tmp_path), capture_s=0.3, hw_timeout_s=1.0, sensors=False),
                       probes, rig=rig)
    st = {r.name: r.status for r in rep.results}
    assert st["argus_sof helper"] == D.PASS and st["gstreamer"] == D.SKIP
    rig2 = test_rig.replace(capture={**cap, "argus_sof": {"helper": str(tmp_path / "missing")}})
    rep = D.run_doctor("rig.yaml", D.DoctorOptions(out=str(tmp_path), sensors=False), FakeProbes(), rig=rig2)
    r = [x for x in rep.results if x.name == "argus_sof helper"][0]
    assert r.status == D.FAIL and "build.sh" in r.fix


def test_native_sources_present_and_marked_unbuilt():
    d = Path(__file__).resolve().parents[1] / "native" / "argus_sof"
    cpp = (d / "argus_sof.cpp").read_text()
    assert "getSensorTimestamp" in cpp and "NOT BUILT" in cpp and "userAutoExposure" in cpp
    cm = (d / "CMakeLists.txt").read_text()
    assert "FATAL_ERROR" in cm and "Jetson Multimedia API not found" in cm


# ------------------------------------------------------------------------------------------------ T5a stamp-offset

from sensor_rig.calib import stamp_offset as so  # noqa: E402


def _sim(off_ns, jit_ns, n=1800, fps=30.0, seed=0):
    rng = np.random.default_rng(seed)
    tt = (np.arange(n) * 1e9 / fps + 1e12).astype(np.int64)
    ev, t, on = [], int(1e12 + 3e8), False
    while t < tt[-1]:
        on = not on
        ev.append((t, on))
        t += int(rng.uniform(*((0.1, 0.2) if on else (0.15, 0.35))) * 1e9)
    evt = np.array([e[0] for e in ev])
    evo = np.array([e[1] for e in ev])
    idx = np.searchsorted(evt, tt, side="right") - 1
    states = np.where(idx >= 0, evo[np.maximum(idx, 0)], False)
    stamps = (tt - off_ns + (rng.normal(0, jit_ns, n) if jit_ns else 0)).astype(np.int64)
    return stamps, states, ev


@pytest.mark.parametrize("seed", range(5))
def test_estimate_offset_recovers_offset_and_jitter(seed):
    s, on, ev = _sim(-30_000_000, 500_000, seed=seed)
    r = so.estimate_offset(s, on, ev)
    assert abs(r["offset_ms"] + 30.0) < 1.0, r
    assert abs(r["jitter_ms_rms"] - 0.5) < 0.1, r                       # within 20 %
    assert r["n_edges"] > 200


def test_estimate_offset_no_led():
    s, on, ev = _sim(-30_000_000, 0)
    with pytest.raises(so.NoLedError, match="no LED"):
        so.estimate_offset(s, np.zeros_like(on), ev)
    with pytest.raises(so.NoLedError, match="no LED"):
        so.led_states(np.full(100, 42.0) + np.random.default_rng(0).normal(0, 1, 100))


def test_blinker_with_fake_led():
    from sensor_rig.drivers.base import SimClock

    clock = SimClock(speed=20.0)
    led = so.FakeLed(clock)
    b = so.Blinker(led, clock)
    b.start()
    clock.sleep(5.0)
    b.stop()
    assert len(b.events) >= 15 and [e[1] for e in b.events[:4]] == [True, False, True, False]
    for t, state in b.events:
        assert led.state_at(t + 1_000_000) == state                    # logged time ~ the actual write


def test_stamp_offset_end_to_end_mock(tmp_path, test_rig):
    """Full load (recorder, JPEG workers, IMU) with mock cameras that draw the LED with a known offset + jitter."""
    from sensor_rig.drivers.base import SimClock
    from sensor_rig.drivers.mock_imu import MockImu

    clock = SimClock(speed=3.0)
    led = so.FakeLed(clock)
    offs = {"main": -30_000_000, "ultrawide": -27_000_000}
    cams = {c: so.LedMockCamera.make(c, 320, 240, 30.0, led, offs[c], 500_000, clock, seed=i)
            for i, c in enumerate(("main", "ultrawide"))}
    rig = test_rig.replace(cameras={c: dataclasses.replace(test_rig.cameras[c], resolution=(320, 240), fps=30.0)
                                    for c in test_rig.cameras})
    res = so.run_stamp_offset(rig, tmp_path, led, cams, MockImu.static(rate_hz=400, clock=clock), clock,
                              duration_s=60.0, warmup_s=1.2, log=lambda m: None)
    assert res.ok, res.cameras
    for c in ("main", "ultrawide"):
        r = res.cameras[c]
        assert abs(r["offset_ns"] - offs[c]) < 1_000_000, (c, r)
        assert abs(r["jitter_ms_rms"] - 0.5) < 0.1, (c, r)
        assert res.patch["cameras"][c]["time"]["offset_prior_ns"] == r["offset_ns"]
    assert not (tmp_path / "stamp_offset_raw").exists()               # temporary episode removed


def test_stamp_offset_no_led_detected(tmp_path, test_rig):
    from sensor_rig.drivers.base import SimClock
    from sensor_rig.drivers.mock_imu import MockImu

    clock = SimClock(speed=8.0)
    led = so.FakeLed(clock)
    cams = {c: so.LedMockCamera.make(c, 160, 120, 30.0, None, 0, 0, clock, seed=i)   # LED never drawn
            for i, c in enumerate(("main", "ultrawide"))}
    rig = test_rig.replace(cameras={c: dataclasses.replace(test_rig.cameras[c], resolution=(160, 120), fps=30.0)
                                    for c in test_rig.cameras})
    res = so.run_stamp_offset(rig, tmp_path, led, cams, MockImu.static(rate_hz=400, clock=clock), clock,
                              duration_s=10.0, warmup_s=1.2, log=lambda m: None)
    assert not res.ok and all("no LED" in r["error"] for r in res.cameras.values())
    assert res.patch["cameras"] == {}


def test_stamp_offset_cli_requires_led(capsys):
    from sensor_rig.cli import main

    assert main(["stamp-offset"]) == 2 and "--led gpio:PIN" in capsys.readouterr().err


def test_per_camera_values():
    assert so.per_camera("-12.3") == {"main": -12.3, "ultrawide": -12.3}
    assert so.per_camera("main=-30,ultrawide=-27") == {"main": -30.0, "ultrawide": -27.0}
    assert so.per_camera("ultrawide=1.5") == {"main": 0.0, "ultrawide": 1.5}
    for bad in ("abc", "side=3", "main=x"):
        with pytest.raises(ValueError, match="expected a number or cam=value"):
            so.per_camera(bad)


def test_cli_mock_bare_values_and_short_run_drift(tmp_path, capsys):
    from sensor_rig.cli import main

    out = tmp_path / "r.json"
    rc = main(["stamp-offset", "--mock", "--mock-offset-ms", "-12.3", "--mock-jitter-ms", "main=0.3,ultrawide=0.6",
               "--duration", "20", "--mock-speed", "8", "--out", str(tmp_path), "--json", str(out)])
    cap = capsys.readouterr()
    assert rc == 0, cap.out + cap.err
    assert "drift estimate will be unreliable" in cap.err and "unreliable" in cap.out
    import json as _json

    r = _json.loads(out.read_text())["cameras"]
    for c in ("main", "ultrawide"):
        assert abs(r[c]["offset_ms"] + 12.3) < 1.5 and r[c]["drift_reliable"] is False
    assert r["ultrawide"]["jitter_ms_rms"] > r["main"]["jitter_ms_rms"]
    assert main(["stamp-offset", "--mock", "--mock-offset-ms", "lens=3", "--out", str(tmp_path)]) == 2
    assert "expected a number or cam=value" in capsys.readouterr().err


def test_drift_reliable_flag_by_run_length():
    s, on, ev = _sim(-30_000_000, 500_000, n=1830)                     # 61 s
    assert so.estimate_offset(s, on, ev)["drift_reliable"] is True
    s, on, ev = _sim(-30_000_000, 500_000, n=600)                      # 20 s
    r = so.estimate_offset(s, on, ev)
    assert r["drift_reliable"] is False and r["run_s"] < 60
