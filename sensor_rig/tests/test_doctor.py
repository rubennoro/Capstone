"""T1: ``sensor-rig doctor`` with fake probes: one test per PASS/WARN/FAIL branch, plus bounded time on hangs."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from sensor_rig import doctor as D
from sensor_rig.drivers.base import CameraBackend, ImuBackend, ImuBatch
from sensor_rig.rig_config import load_rig

RIG_DEFAULT = Path(D.__file__).resolve().parent / "configs" / "rig_default.yaml"
DiskUsage = namedtuple("DiskUsage", "total used free")

ARGUS_TEXT = """Setting pipeline to PAUSED ...
GST_ARGUS: Creating output stream
CONSUMER: Waiting until producer is connected...
GST_ARGUS: Available Sensor modes :
GST_ARGUS: 3280 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 3280 x 1848 FR = 28.000001 fps Duration = 35714284 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 1920 x 1080 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 1640 x 1232 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: 1280 x 720 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
GST_ARGUS: Running with following settings:
   Camera index = 0
   Camera mode  = 4
   Output Stream W = 1280 H = 720
"""


class HangingCamera(CameraBackend):
    """start() never returns (a wedged nvargus-daemon)."""

    def __init__(self, name, w, h, fps):
        super().__init__(name, w, h, fps)
        self.release = threading.Event()

    def start(self):
        self.release.wait(30)

    def _run(self):
        pass

    def stop(self):
        self.release.set()


class SilentCamera(CameraBackend):
    """Starts, but never delivers a frame."""

    def _run(self):
        while not self._stopping.is_set():
            time.sleep(0.01)


class ScaledImu(ImuBackend):
    """Static IMU whose accel reads a wrong scale (e.g. 4x too small)."""

    def __init__(self, rate_hz=1000.0, scale=0.25):
        super().__init__(rate_hz)
        self.scale = scale

    def _run(self):
        t = time.monotonic_ns()
        while not self._stopping.is_set():
            n = 5
            ts = t + np.arange(1, n + 1, dtype=np.int64) * 1_000_000
            t = int(ts[-1])
            self._push(ImuBatch(ts, np.zeros((n, 3)), np.tile([0, 0, D.G * self.scale], (n, 1))))
            time.sleep(0.005)


class FakeProbes(D.Probes):
    """Every probe answers 'healthy Jetson' unless a test overrides an attribute."""

    def __init__(self, **kw):
        self.missing_modules: set[str] = set()
        self.gst_error: Exception | None = None
        self.has_argus_element = True
        self.daemon = True
        self.paths = {"/dev/video0", "/dev/video1", "/dev/spidev0.0"}
        self.denied: dict[str, str] = {}
        self.video_names = {"/dev/video0": "vi-output, imx219 9-0010", "/dev/video1": "vi-output, imx219 10-0010"}
        self.argus_text = ARGUS_TEXT
        self.argus_delay_s = 0.0
        self.who = 0x47
        self.camera_factory = None       # (rig, name) -> backend; None = mock backends
        self.imu_factory = None
        self.free_bytes = 500e9
        self.mb_s = 500.0
        self.now = time.mktime((2026, 10, 20, 12, 0, 0, 0, 0, 0))
        self.ntp = True
        self.sensor_rig_exe = "/venv/bin/sensor-rig"
        self.argv = "/venv/bin/sensor-rig"
        for k, v in kw.items():
            setattr(self, k, v)

    def import_module(self, name):
        if name in self.missing_modules:
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        import importlib

        return importlib.import_module(name) if name != "spidev" else object()

    def import_gst(self):
        if self.gst_error:
            raise self.gst_error
        return "GStreamer 1.20.3"

    def gst_has_element(self, name):
        return self.has_argus_element

    def process_running(self, name):
        return self.daemon

    def path_exists(self, path):
        return path in self.paths

    def access_rw(self, path):
        return self.denied.get(path)

    def video_name(self, dev):
        return self.video_names.get(dev)

    def argus_mode_text(self, sensor_id, timeout_s):
        if self.argus_delay_s:
            time.sleep(self.argus_delay_s)
        return self.argus_text

    def who_am_i(self, device, speed_hz, mode):
        return self.who

    def make_camera(self, rig, name, mock):
        if self.camera_factory:
            return self.camera_factory(rig, name)
        return super().make_camera(rig, name, True)

    def make_imu(self, rig, mock):
        if self.imu_factory:
            return self.imu_factory(rig)
        return super().make_imu(rig, True)

    def disk_usage(self, path):
        return DiskUsage(1e12, 1e12 - self.free_bytes, self.free_bytes)

    def write_speed_mb_s(self, directory, mb, max_s):
        return self.mb_s

    def wall_time_s(self):
        return self.now

    def ntp_synced(self):
        return self.ntp

    def which(self, name):
        return self.sensor_rig_exe

    def argv0(self):
        return self.argv


@pytest.fixture(scope="module")
def rig():
    return load_rig(RIG_DEFAULT)


def locked_rig(rig):
    cams = {n: dataclasses.replace(c, placeholder=False, argus={**c.argus, "aelock": True, "awblock": True})
            for n, c in rig.cameras.items()}
    return rig.replace(cameras=cams, imu=dataclasses.replace(rig.imu, placeholder=False))


def run(rig, probes=None, tmp_path=None, **opts):
    o = D.DoctorOptions(capture_s=0.4, imu_s=0.3, hw_timeout_s=1.0, out=str(tmp_path) if tmp_path else None, **opts)
    return D.run_doctor("rig.yaml", o, probes or FakeProbes(), rig=rig)


def by_name(rep, prefix):
    hits = [r for r in rep.results if r.name.startswith(prefix)]
    assert hits, f"no check named {prefix!r}: {[r.name for r in rep.results]}"
    return hits


def status(rep, prefix):
    return {r.status for r in by_name(rep, prefix)}


# ------------------------------------------------------------------------------------------------ pure helpers

def test_parse_argus_modes():
    modes = D.parse_argus_modes(ARGUS_TEXT)
    assert [(m["width"], m["height"]) for m in modes] == [(3280, 2464), (3280, 1848), (1920, 1080), (1640, 1232),
                                                          (1280, 720)]
    assert modes[3]["fps"] == pytest.approx(30.0, abs=1e-5)
    assert modes[3]["exposure_ns"] == [13000, 683709000]
    assert D.parse_argus_modes("nothing here") == []


@pytest.mark.parametrize("mode,res,fps,expect", [
    (3, (1640, 1232), 30, D.PASS),
    (4, (1640, 1232), 30, D.FAIL),     # wrong resolution for that mode
    (9, (1640, 1232), 30, D.FAIL),     # mode does not exist
    (3, (1640, 1232), 60, D.FAIL),     # too fast for the mode
    (None, (1640, 1232), 30, D.WARN),  # Argus would pick
])
def test_check_argus_mode(mode, res, fps, expect):
    st, det, fix = D.check_argus_mode(D.parse_argus_modes(ARGUS_TEXT), mode, res, fps)
    assert st == expect, det
    if st != D.PASS:
        assert fix


def test_check_argus_mode_no_modes_and_exposure_range():
    assert D.check_argus_mode([], 3, (1640, 1232), 30)[0] == D.FAIL
    st, det, _ = D.check_argus_mode(D.parse_argus_modes(ARGUS_TEXT), 3, (1640, 1232), 30, (1000, 4_000_000))
    assert st == D.WARN and "exposure" in det


def test_judge_capture_branches():
    t = [i * 33_333_333 for i in range(60)]
    ok = D.summarize_frames(t, [x + 5_000_000 for x in t], list(range(60)))
    assert D.judge_capture("main", ok, 30.0, 0).status == D.PASS
    neg = D.summarize_frames(t, [x - 1 for x in t], list(range(60)))
    r = D.judge_capture("main", neg, 30.0, 0)
    assert r.status == D.FAIL and "MONOTONIC" in r.fix
    assert D.judge_capture("main", ok, 30.0, 3).status == D.WARN
    slow = D.summarize_frames(t[::2], [x + 1 for x in t[::2]], list(range(30)))
    assert D.judge_capture("main", slow, 30.0, 0).status == D.FAIL
    assert D.judge_capture("main", D.summarize_frames([], [], []), 30.0, 0).status == D.FAIL


def test_judge_imu_branches():
    assert [r.status for r in D.judge_imu(1000, 1000, 9.81, 0.001, 0)] == [D.PASS, D.PASS]
    assert D.judge_imu(None, 1000, None, None, 0)[0].status == D.FAIL
    assert D.judge_imu(800, 1000, 9.81, 0.0, 0)[0].status == D.FAIL
    assert D.judge_imu(1000, 1000, 9.81, 0.0, 3)[0].status == D.WARN
    assert D.judge_imu(1000, 1000, 2.45, 0.0, 0)[1].status == D.FAIL
    assert D.judge_imu(1000, 1000, 9.81, 0.3, 0)[1].status == D.WARN


def test_run_bounded():
    assert D.run_bounded(lambda: 7, 1.0) == (True, 7)
    t0 = time.monotonic()
    assert D.run_bounded(lambda: time.sleep(5), 0.2) == (False, None)
    assert time.monotonic() - t0 < 1.0
    with pytest.raises(ZeroDivisionError):
        D.run_bounded(lambda: 1 / 0, 1.0)


# ------------------------------------------------------------------------------------------------ full runs

def test_healthy_jetson_all_pass(rig, tmp_path):
    rep = run(locked_rig(rig), tmp_path=tmp_path)
    bad = [r for r in rep.results if r.status not in (D.PASS,)]
    assert rep.ok and not bad, rep.table()
    names = [r.name for r in rep.results]
    for n in ("python deps", "rig.yaml", "wall clock", "disk free", "disk write", "gstreamer", "nvargus-daemon",
              "/dev/video0 (main)", "/dev/video1 (ultrawide)", "argus mode main", "argus mode ultrawide",
              "spi /dev/spidev0.0", "imu WHO_AM_I", "capture main", "capture ultrawide", "imu rate", "imu gravity"):
        assert n in names


def test_default_rig_warns_but_passes(rig, tmp_path):
    rep = run(rig, tmp_path=tmp_path)
    assert rep.ok
    assert status(rep, "rig placeholders") == {D.WARN}
    assert status(rep, "exposure/WB lock") == {D.PASS}          # T3/F7: rig_default.yaml is locked now


def test_unlocked_exposure_warns(rig, tmp_path):
    cams = {n: dataclasses.replace(c, argus={**c.argus, "aelock": False, "awblock": False,
                                             "exposure_time_range_ns": [100000, 8000000]})
            for n, c in rig.cameras.items()}
    rep = run(rig.replace(cameras=cams), tmp_path=tmp_path)
    assert rep.ok and status(rep, "exposure/WB lock") == {D.WARN} and status(rep, "exposure time") == {D.WARN}


def test_missing_python_dep_stops_early(rig):
    rep = run(rig, FakeProbes(missing_modules={"scipy"}))
    assert not rep.ok and len(rep.results) == 1
    assert "python3-scipy" in rep.results[0].fix


def test_missing_camera_device(rig, tmp_path):
    rep = run(rig, FakeProbes(paths={"/dev/video0", "/dev/spidev0.0"}), tmp_path)
    r = by_name(rep, "/dev/video1")[0]
    assert r.status == D.FAIL and "jetson-io" in r.fix
    assert status(rep, "capture") == {D.SKIP}
    assert status(rep, "imu WHO_AM_I") == {D.PASS}       # the IMU is still checked


def test_webcam_on_video_node_warns(rig, tmp_path):
    names = {"/dev/video0": "Integrated_Webcam_HD: Integrate", "/dev/video1": "vi-output, imx219 10-0010"}
    rep = run(rig, FakeProbes(video_names=names), tmp_path)
    assert by_name(rep, "/dev/video0")[0].status == D.WARN and "webcam" in by_name(rep, "/dev/video0")[0].fix
    assert by_name(rep, "/dev/video1")[0].status == D.PASS
    assert D.looks_like_csi("vi-output, imx219 9-0010") and not D.looks_like_csi("HD Webcam")


def test_sensors_false_skips_capture(rig, tmp_path):
    rep = run(rig, FakeProbes(), tmp_path, sensors=False, quick=True)
    assert status(rep, "capture + imu stream") == {D.SKIP}
    assert "capture main" not in [r.name for r in rep.results]
    assert status(rep, "imu WHO_AM_I") == {D.PASS}


def test_camera_permission_denied(rig, tmp_path):
    rep = run(rig, FakeProbes(denied={"/dev/video0": "Permission denied"}), tmp_path)
    r = by_name(rep, "/dev/video0")[0]
    assert r.status == D.FAIL and "video" in r.fix and not rep.ok


def test_spi_missing_and_permission_denied(rig, tmp_path):
    rep = run(rig, FakeProbes(paths={"/dev/video0", "/dev/video1"}), tmp_path)
    assert status(rep, "spi") == {D.FAIL} and "jetson-io" in by_name(rep, "spi")[0].fix
    assert status(rep, "imu stream") == {D.SKIP}
    rep = run(rig, FakeProbes(denied={"/dev/spidev0.0": "Permission denied"}), tmp_path)
    r = by_name(rep, "spi")[0]
    assert r.status == D.FAIL and "udev" in r.fix


def test_wrong_who_am_i(rig, tmp_path):
    rep = run(rig, FakeProbes(who=0xFF), tmp_path)
    r = by_name(rep, "imu WHO_AM_I")[0]
    assert r.status == D.FAIL and "0xff" in r.detail and "wiring" in r.fix
    assert status(rep, "imu stream") == {D.SKIP}


def test_spidev_module_missing(rig, tmp_path):
    rep = run(rig, FakeProbes(missing_modules={"spidev"}), tmp_path)
    assert status(rep, "imu WHO_AM_I") == {D.FAIL}


def test_wrong_sensor_mode(rig, tmp_path):
    cams = dict(rig.cameras)
    cams["main"] = dataclasses.replace(cams["main"], sensor_mode=4)
    rep = run(rig.replace(cameras=cams), tmp_path=tmp_path)
    r = by_name(rep, "argus mode main")[0]
    assert r.status == D.FAIL and "1280x720" in r.detail
    assert status(rep, "argus mode ultrawide") == {D.PASS}
    assert status(rep, "capture") == {D.SKIP}


def test_mode_listing_timeout_is_bounded(rig, tmp_path):
    t0 = time.monotonic()
    rep = run(rig, FakeProbes(argus_delay_s=10), tmp_path, quick=False)
    assert time.monotonic() - t0 < 12      # 2 cameras x (hw_timeout 1 s + 2 s grace) + rest
    assert {r.status for r in by_name(rep, "argus mode")} == {D.FAIL}


def test_camera_start_hang_is_bounded(rig, tmp_path):
    def factory(rig_, name):
        c = rig_.cameras[name]
        return HangingCamera(name, *c.resolution, c.fps)

    t0 = time.monotonic()
    rep = run(rig, FakeProbes(camera_factory=factory), tmp_path)
    assert time.monotonic() - t0 < 10
    caps = by_name(rep, "capture ")
    assert {r.status for r in caps} == {D.FAIL} and "timed out" in caps[0].detail
    assert not rep.ok


def test_camera_without_frames(rig, tmp_path):
    def factory(rig_, name):
        c = rig_.cameras[name]
        return SilentCamera(name, *c.resolution, c.fps)

    rep = run(rig, FakeProbes(camera_factory=factory), tmp_path)
    assert {r.status for r in by_name(rep, "capture ")} == {D.FAIL}


def test_imu_wrong_scale(rig, tmp_path):
    rep = run(rig, FakeProbes(imu_factory=lambda r: ScaledImu()), tmp_path)
    r = by_name(rep, "imu gravity")[0]
    assert r.status == D.FAIL and "16byte" in r.fix


def test_gstreamer_and_daemon(rig, tmp_path):
    rep = run(rig, FakeProbes(gst_error=ImportError("No module named 'gi'"), daemon=False), tmp_path)
    assert status(rep, "gstreamer") == {D.FAIL} and "system-site-packages" in by_name(rep, "gstreamer")[0].fix
    assert status(rep, "nvargus-daemon") == {D.FAIL}
    assert status(rep, "capture") == {D.SKIP}
    rep = run(rig, FakeProbes(has_argus_element=False), tmp_path)
    assert status(rep, "gstreamer") == {D.FAIL}


def test_clock_1970_fails(rig, tmp_path):
    rep = run(rig, FakeProbes(now=0.0), tmp_path)
    r = by_name(rep, "wall clock")[0]
    assert r.status == D.FAIL and "1970" in r.detail and "timedatectl" in r.fix and not rep.ok


def test_clock_not_synced_warns(rig, tmp_path):
    assert status(run(rig, FakeProbes(ntp=False), tmp_path), "wall clock") == {D.WARN}


def test_disk_branches(rig, tmp_path):
    assert status(run(rig, FakeProbes(free_bytes=2.5e9), tmp_path), "disk free") == {D.FAIL}
    assert status(run(rig, FakeProbes(free_bytes=40e9), tmp_path), "disk free") == {D.WARN}   # ~22 min
    assert status(run(rig, FakeProbes(mb_s=20), tmp_path), "disk write") == {D.FAIL}
    assert status(run(rig, FakeProbes(mb_s=45), tmp_path), "disk write") == {D.WARN}
    assert status(run(rig, FakeProbes()), "disk") == {D.SKIP}                                   # no --out
    rep = run(rig, FakeProbes(), tmp_path, quick=True)
    assert "disk write" not in [r.name for r in rep.results]


def test_real_write_speed_probe_cleans_up(tmp_path):
    mbs = D.Probes().write_speed_mb_s(str(tmp_path), 4, max_s=5)
    assert mbs > 0 and list(tmp_path.iterdir()) == []


def test_entry_point(rig, tmp_path):
    assert status(run(rig, FakeProbes(sensor_rig_exe=None), tmp_path), "entry point") == {D.WARN}
    r = by_name(run(rig, FakeProbes(argv="/x/sensor_rig/__main__.py"), tmp_path), "entry point")[0]
    assert r.status == D.PASS and "sensor-rig" in r.detail


def test_coverage_fail_on_calibrated_rig(tmp_path):
    text = RIG_DEFAULT.read_text().replace("intrinsics: [1150.0, 1150.0, 819.5, 615.5]",
                                           "intrinsics: [1600.0, 1600.0, 819.5, 615.5]")   # long lens: < 0.99
    p = tmp_path / "narrow.yaml"
    p.write_text(text)
    narrow = load_rig(p)
    assert status(run(narrow, tmp_path=tmp_path), "coverage") == {D.WARN}           # placeholder lens: warn
    assert status(run(locked_rig(narrow), tmp_path=tmp_path), "coverage") == {D.FAIL}


def test_mock_mode_skips_hardware_and_passes(rig, tmp_path):
    rep = run(rig, D.Probes(), tmp_path, mock=True)
    assert rep.ok, rep.table()
    assert {r.status for r in rep.results if r.name in ("gstreamer", "nvargus-daemon", "camera devices",
                                                        "spi + WHO_AM_I")} == {D.SKIP}
    assert status(rep, "capture main") == {D.PASS}
    assert status(rep, "imu gravity") == {D.PASS}


def test_bad_rig_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("schema_version: 1\n")
    rep = D.run_doctor(p, D.DoctorOptions(), FakeProbes())
    assert not rep.ok and by_name(rep, "rig.yaml")[0].status == D.FAIL


def test_cli_json_and_exit_code(tmp_path, monkeypatch):
    from sensor_rig.cli import main

    out = tmp_path / "r.json"
    rc = main(["doctor", "--mock", "--quick", "--out", str(tmp_path / "raw"), "--json", str(out), "--capture-s",
               "0.4"])
    d = json.loads(out.read_text())
    assert rc == (0 if d["ok"] else 1)
    assert d["mock"] is True and any(r["name"] == "capture main" for r in d["results"])


def test_cli_missing_module_message(monkeypatch, capsys):
    from sensor_rig import cli

    def boom(args):
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr(D, "cmd_doctor", boom)
    rc = cli.main(["doctor"])
    err = capsys.readouterr().err
    assert rc == 1 and "python3-scipy" in err and "jetson_setup.sh" in err
