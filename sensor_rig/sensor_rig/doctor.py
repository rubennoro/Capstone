"""``sensor-rig doctor``: preflight checks before recording on the Jetson (or a laptop with ``--mock``).

    sensor-rig doctor --rig rig.yaml [--out /mnt/nvme/raw] [--mock] [--json report.json] [--quick]

Every check returns ``CheckResult(name, status, detail, fix)`` with status PASS / WARN / FAIL / SKIP. Exit code 0 when
nothing FAILs (WARN is allowed), 1 otherwise. ``run_doctor()`` is the reusable entry point (``collect`` calls it with
``quick=True``: no Argus mode listing, no write-speed test, shorter capture).

All system access goes through a ``Probes`` object, so every branch is unit-tested with fakes
(``tests/test_doctor.py``). Hardware calls (GStreamer start, SPI) run in daemon threads with a deadline, so a hung
camera or daemon turns into a FAIL instead of hanging the doctor.

Checks (in order):
  python deps, entry point, rig.yaml (load / placeholders / AE-AWB locks / exposure / coverage), wall clock + NTP,
  disk free + write speed, GStreamer bindings, nvargus-daemon, /dev/videoN, Argus sensor mode, SPI device access,
  WHO_AM_I, simultaneous capture on both cameras, IMU rate + gravity at rest.
With ``--mock`` the hardware-only checks are SKIPped and the capture/IMU checks run on the mock backends.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
DEFAULT_RIG = Path(__file__).resolve().parent / "configs" / "rig_default.yaml"
RECORD_MB_S = 30.0          # ~2 x 1640x1232 @ 30 fps JPEG Q90 (~20 MB/s) plus headroom
MIN_YEAR = 2026             # the Jetson has no RTC battery by default: a 1970/2024 clock means NTP never ran
G = 9.80665

APT_HINT = {
    "numpy": "sudo apt install python3-numpy",
    "cv2": "sudo apt install python3-opencv (JetPack's build; never pip-install opencv into the venv)",
    "yaml": "sudo apt install python3-yaml",
    "scipy": "sudo apt install python3-scipy",
    "spidev": "sudo apt install python3-spidev, or pip install spidev in the venv",
    "gi": "sudo apt install python3-gi gir1.2-gstreamer-1.0 and create the venv with --system-site-packages",
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    fix: str = ""


@dataclass
class DoctorReport:
    results: list[CheckResult] = field(default_factory=list)
    rig_path: str | None = None
    mock: bool = False

    @property
    def ok(self) -> bool:
        return not any(r.status == FAIL for r in self.results)

    def add(self, r: CheckResult) -> CheckResult:
        self.results.append(r)
        return r

    def to_dict(self) -> dict:
        return {"ok": self.ok, "rig": self.rig_path, "mock": self.mock, "results": [asdict(r) for r in self.results]}

    def table(self) -> str:
        w = max([len(r.name) for r in self.results] + [5])
        lines = []
        for r in self.results:
            lines.append(f"{r.status:4s}  {r.name:{w}s}  {r.detail}")
            if r.fix and r.status in (FAIL, WARN):
                lines.append(f"{'':4s}  {'':{w}s}  -> {r.fix}")
        n = {s: sum(r.status == s for r in self.results) for s in (PASS, WARN, FAIL, SKIP)}
        lines.append(f"doctor: {n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail, {n[SKIP]} skipped -> "
                     f"{'OK' if self.ok else 'NOT READY'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------------------------------------------------
_MODE_RE = re.compile(r"(\d+)\s*x\s*(\d+)\s+FR\s*=\s*([\d.]+)\s*fps(?:.*?Exposure Range min\s*(\d+),\s*max\s*(\d+))?")


def parse_argus_modes(text: str) -> list[dict]:
    """Sensor modes from nvarguscamerasrc's stderr (``GST_ARGUS: 1640 x 1232 FR = 29.999999 fps ...``), in mode
    order. Only lines between "Available Sensor modes" and "Running with following settings" are used when present."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Available Sensor modes" in ln), -1)
    out = []
    for ln in lines[start + 1:]:
        if "Running with following settings" in ln:
            break
        m = _MODE_RE.search(ln)
        if m:
            d = {"width": int(m.group(1)), "height": int(m.group(2)), "fps": float(m.group(3))}
            if m.group(4):
                d["exposure_ns"] = [int(m.group(4)), int(m.group(5))]
            out.append(d)
    return out


def check_argus_mode(modes: list[dict], sensor_mode: int | None, resolution, fps: float,
                     exposure_range_ns=None) -> tuple[str, str, str]:
    """(status, detail, fix) for the rig.yaml camera vs the listed sensor modes."""
    w, h = (int(v) for v in resolution)
    listing = ", ".join(f"{i}: {m['width']}x{m['height']}@{m['fps']:.0f}" for i, m in enumerate(modes))
    fix_modes = f"set sensor_mode/resolution/fps in rig.yaml to one of: {listing}"
    if not modes:
        return FAIL, "no sensor modes listed (camera not detected by Argus)", \
            "check the CSI cable and jetson-io 'Camera IMX219 Dual', then sudo systemctl restart nvargus-daemon"
    if sensor_mode is None or int(sensor_mode) < 0:
        return WARN, f"sensor_mode not set; Argus picks one ({listing})", fix_modes
    i = int(sensor_mode)
    if i >= len(modes):
        return FAIL, f"sensor_mode {i} does not exist ({listing})", fix_modes
    m = modes[i]
    if (m["width"], m["height"]) != (w, h):
        return FAIL, f"mode {i} is {m['width']}x{m['height']}, rig.yaml says {w}x{h}", fix_modes
    if m["fps"] + 0.5 < float(fps):
        return FAIL, f"mode {i} runs at most {m['fps']:.2f} fps, rig.yaml wants {fps:g}", fix_modes
    if exposure_range_ns is not None and "exposure_ns" in m:
        lo, hi = exposure_range_ns
        if lo < m["exposure_ns"][0] or hi > m["exposure_ns"][1]:
            return WARN, f"mode {i} OK, but exposure range {lo}-{hi} ns is outside the sensor's " \
                         f"{m['exposure_ns'][0]}-{m['exposure_ns'][1]} ns", "adjust argus.exposure_time_range_ns"
    return PASS, f"mode {i} = {m['width']}x{m['height']}@{m['fps']:.2f}", ""


CSI_NAME_HINTS = ("vi-output", "imx", "nvcsi", "tegra-capture")   # not bare "tegra": "Integrated Webcam" contains it


def looks_like_csi(v4l2_name: str) -> bool:
    """True for the V4L2 names Jetson CSI sensors get ('vi-output, imx219 9-0010'); False for e.g. a USB webcam."""
    n = v4l2_name.lower()
    return any(h in n for h in CSI_NAME_HINTS)


def summarize_frames(t_ns: list[int], t_arrival_ns: list[int], seqs: list[int]) -> dict:
    n = len(t_ns)
    d: dict[str, Any] = {"frames": n, "fps": None, "latency_ms_median": None, "negative_latency": 0, "seq_gaps": 0,
                         "nonmonotonic": 0}
    if n >= 2 and t_ns[-1] > t_ns[0]:
        d["fps"] = (n - 1) / ((t_ns[-1] - t_ns[0]) / 1e9)
    if n:
        lat = sorted(a - t for a, t in zip(t_arrival_ns, t_ns))
        d["latency_ms_median"] = lat[len(lat) // 2] / 1e6
        d["negative_latency"] = sum(x < 0 for x in lat)
    d["seq_gaps"] = sum(max(0, b - a - 1) for a, b in zip(seqs, seqs[1:]))
    d["nonmonotonic"] = sum(b <= a for a, b in zip(t_ns, t_ns[1:]))
    return d


def judge_capture(name: str, s: dict, fps_nominal: float, internal_drops: int) -> CheckResult:
    cname = f"capture {name}"
    if s["frames"] < 2 or s["fps"] is None:
        return CheckResult(cname, FAIL, f"{s['frames']} frames", "camera delivers no frames: check cable/mode, "
                           "sudo systemctl restart nvargus-daemon, and run the gst-launch test in HARDWARE_CHECKLIST §1")
    detail = (f"{s['fps']:.1f}/{fps_nominal:g} fps, latency {s['latency_ms_median']:.1f} ms, "
              f"drops {internal_drops}, seq gaps {s['seq_gaps']}")
    if s["negative_latency"] or s["nonmonotonic"]:
        return CheckResult(cname, FAIL, detail + f", {s['negative_latency']} negative latencies, "
                           f"{s['nonmonotonic']} non-increasing stamps",
                           "stamps are not CLOCK_MONOTONIC: check use_clock() in drivers/argus_camera.py")
    ratio = s["fps"] / fps_nominal
    if ratio < 0.85:
        return CheckResult(cname, FAIL, detail, "fps far below nominal: check sensor_mode, nvpmodel (sudo nvpmodel -m 0;"
                           " sudo jetson_clocks), CPU load")
    if ratio < 0.97 or internal_drops or s["seq_gaps"]:
        return CheckResult(cname, WARN, detail, "frames lost: check CPU load / jetson_clocks; rerun doctor")
    return CheckResult(cname, PASS, detail)


def judge_imu(rate: float | None, rate_nominal: float, accel_norm: float | None, gyro_norm: float | None,
              overflows: int) -> list[CheckResult]:
    out = []
    if rate is None:
        return [CheckResult("imu rate", FAIL, "no samples", "check wiring / spi_mode / spi_speed_hz (try 1 MHz)")]
    r = rate / rate_nominal
    det = f"{rate:.0f}/{rate_nominal:g} Hz, fifo overflows {overflows}"
    if r < 0.9 or r > 1.1:
        out.append(CheckResult("imu rate", FAIL, det, "raise imu.fifo.poll_period_ms or reduce CPU load"))
    elif r < 0.98 or r > 1.02 or overflows:
        out.append(CheckResult("imu rate", WARN, det, "rate off by > 2 % or overflows: check CPU load"))
    else:
        out.append(CheckResult("imu rate", PASS, det))
    if accel_norm is None:
        return out
    det = f"|a| = {accel_norm:.2f} m/s^2, |w| = {gyro_norm:.3f} rad/s (rig must be at rest)"
    if abs(accel_norm - G) > 1.0:
        out.append(CheckResult("imu gravity", FAIL, det, "accel scale wrong (2.45 or 39.2 -> fix HIRES_* in "
                               "icm42688_proto.py or set imu.fifo.packet: 16byte)"))
    elif abs(accel_norm - G) > 0.3 or (gyro_norm or 0) > 0.05:
        out.append(CheckResult("imu gravity", WARN, det, "keep the rig still during doctor and rerun"))
    else:
        out.append(CheckResult("imu gravity", PASS, det))
    return out


def run_bounded(fn: Callable[[], Any], timeout_s: float) -> tuple[bool, Any]:
    """Run ``fn`` in a daemon thread. ``(True, result)``; ``(False, None)`` on timeout; exceptions are re-raised."""
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e

    th = threading.Thread(target=target, name="doctor-bounded", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return False, None
    if "error" in box:
        raise box["error"]
    return True, box.get("result")


# --------------------------------------------------------------------------------------------------------------------
# Probes: every bit of system access (fakes in tests)
# --------------------------------------------------------------------------------------------------------------------
class Probes:
    def import_module(self, name: str):
        return importlib.import_module(name)

    def import_gst(self) -> str:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return Gst.version_string()

    def gst_has_element(self, name: str) -> bool:
        from gi.repository import Gst

        return Gst.ElementFactory.find(name) is not None

    def process_running(self, name: str) -> bool:
        for p in Path("/proc").glob("[0-9]*/comm"):
            try:
                if p.read_text().strip() == name[:15]:
                    return True
            except OSError:
                continue
        return False

    def path_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def is_executable(self, path: str) -> bool:
        return os.access(path, os.X_OK)

    def access_rw(self, path: str) -> str | None:
        """None if the file can be opened read/write, else the error (e.g. 'Permission denied')."""
        try:
            fd = os.open(path, os.O_RDWR)
            os.close(fd)
            return None
        except OSError as e:
            return e.strerror or str(e)

    def video_name(self, dev: str) -> str | None:
        """V4L2 device name from sysfs, e.g. 'vi-output, imx219 9-0010' for a Jetson CSI camera."""
        try:
            return Path(f"/sys/class/video4linux/{Path(dev).name}/name").read_text().strip()
        except OSError:
            return None

    def argus_mode_text(self, sensor_id: int, timeout_s: float) -> str:
        cmd = ["gst-launch-1.0", "nvarguscamerasrc", f"sensor-id={int(sensor_id)}", "num-buffers=1", "!",
               "fakesink"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return (p.stdout or "") + (p.stderr or "")

    def who_am_i(self, device: str, speed_hz: int, mode: int) -> int:
        import spidev

        from sensor_rig.drivers.icm42688_proto import REG_WHO_AM_I, spi_read_cmd
        from sensor_rig.drivers.icm42688_spi import parse_spi_device

        bus, dev = parse_spi_device(device)
        spi = spidev.SpiDev()
        spi.open(bus, dev)
        try:
            spi.max_speed_hz = int(speed_hz)
            spi.mode = int(mode)
            return int(spi.xfer2(spi_read_cmd(REG_WHO_AM_I, 1))[1])
        finally:
            spi.close()

    def make_camera(self, rig, name: str, mock: bool):
        from sensor_rig.drivers.base import make_camera

        return make_camera(rig, name, backend="mock" if mock else None)

    def make_imu(self, rig, mock: bool):
        from sensor_rig.drivers.base import make_imu

        return make_imu(rig, backend="mock" if mock else None)

    def disk_usage(self, path: str):
        return shutil.disk_usage(path)

    def write_speed_mb_s(self, directory: str, mb: int, max_s: float) -> float:
        """Sequential write + fsync of up to ``mb`` MB (stops after ``max_s``); the test file is removed."""
        path = Path(directory) / f".sensor_rig_doctor_{os.getpid()}.tmp"
        chunk = os.urandom(4 << 20)
        written = 0
        t0 = time.monotonic()
        try:
            with open(path, "wb") as f:
                while written < mb << 20 and time.monotonic() - t0 < max_s:
                    f.write(chunk)
                    written += len(chunk)
                f.flush()
                os.fsync(f.fileno())
            dt = time.monotonic() - t0
        finally:
            try:
                path.unlink()
            except OSError:
                pass
        return written / (1 << 20) / max(dt, 1e-6)

    def wall_time_s(self) -> float:
        return time.time()

    def ntp_synced(self) -> bool | None:
        try:
            p = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], capture_output=True,
                               text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            return None
        v = p.stdout.strip().lower()
        return True if v == "yes" else False if v == "no" else None

    def argv0(self) -> str:
        return sys.argv[0] if sys.argv else ""

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def sleep(self, s: float) -> None:
        time.sleep(s)


# --------------------------------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class DoctorOptions:
    mock: bool = False
    quick: bool = False
    out: str | None = None
    capture_s: float = 2.0
    imu_s: float = 1.0
    write_test_mb: int = 128
    hw_timeout_s: float = 10.0        # deadline for every hardware call (camera start, mode listing, SPI)
    sensors: bool = True              # False: skip the capture + IMU stream checks (collect: the recorder's warm-up
                                      # checks the rates right after, and the sensors are opened only once)


def _deps(p: Probes, o: DoctorOptions, rep: DoctorReport) -> bool:
    need = ["numpy", "cv2", "yaml", "scipy"]
    missing, versions = [], []
    for m in need:
        try:
            mod = p.import_module(m)
            versions.append(f"{m} {getattr(mod, '__version__', '?')}")
        except ImportError as e:
            missing.append((m, str(e)))
    if missing:
        rep.add(CheckResult("python deps", FAIL, "missing: " + ", ".join(m for m, _ in missing),
                            "; ".join(APT_HINT[m] for m, _ in missing) + " (or rerun scripts/jetson_setup.sh)"))
        return False
    detail = ", ".join(versions)
    try:
        cv2 = p.import_module("cv2")
        venv = os.environ.get("VIRTUAL_ENV")
        if not o.mock and venv and str(getattr(cv2, "__file__", "")).startswith(venv):
            rep.add(CheckResult("python deps", WARN, detail + f"; cv2 comes from the venv ({cv2.__file__})",
                                "pip uninstall opencv-python opencv-python-headless (use JetPack's python3-opencv)"))
            return True
    except ImportError:
        pass
    rep.add(CheckResult("python deps", PASS, detail))
    return True


def _entry_point(p: Probes, o: DoctorOptions, rep: DoctorReport) -> None:
    exe = p.which("sensor-rig")
    if exe is None:
        rep.add(CheckResult("entry point", WARN, "sensor-rig is not on PATH",
                            "pip install --no-deps -e sensor_rig in the venv and activate it; "
                            "`python -m sensor_rig` fails when run from the repo root"))
    elif Path(p.argv0()).name not in ("sensor-rig",):
        rep.add(CheckResult("entry point", PASS, f"{exe} (tip: use `sensor-rig`, `python -m sensor_rig` breaks in "
                                                 "the repo root)"))
    else:
        rep.add(CheckResult("entry point", PASS, exe))


def _rig_checks(rig, rep: DoctorReport) -> None:
    ph = [f"cameras.{n}" for n, c in rig.cameras.items() if c.placeholder] + (["imu"] if rig.imu.placeholder else [])
    if ph:
        rep.add(CheckResult("rig placeholders", WARN, "placeholder sections: " + ", ".join(ph),
                            "fine for bring-up and recording (re-export later with --rig <calibrated>.yaml); "
                            "calibrate with docs/calibration.md before training"))
    else:
        rep.add(CheckResult("rig placeholders", PASS, "all sections calibrated"))
    backends = (rig.capture.get("backends") or {}) if isinstance(rig.capture, dict) else {}
    unlocked, long_exp = [], []
    for n, c in rig.cameras.items():
        b = backends.get(n, "argus")
        if b == "argus":
            a = c.argus or {}
            if not a.get("aelock") or not a.get("awblock"):
                unlocked.append(f"{n} (aelock={bool(a.get('aelock'))}, awblock={bool(a.get('awblock'))})")
            rng = a.get("exposure_time_range_ns")
            if rng and rng[1] > 4_000_000:
                long_exp.append(f"{n} up to {rng[1] / 1e6:g} ms")
        elif b == "picamera2":
            pc = getattr(c, "picamera2", None) or {}
            if pc.get("exposure_us") is None or pc.get("colour_gains") is None:
                unlocked.append(f"{n} (picamera2 auto exposure/white balance)")
    if unlocked:
        rep.add(CheckResult("exposure/WB lock", WARN, "unlocked: " + "; ".join(unlocked),
                            "set argus.aelock/awblock: true and a fixed exposure_time_range_ns [e, e] (<= 4 ms) and "
                            "gain_range for recording and calibration"))
    else:
        rep.add(CheckResult("exposure/WB lock", PASS, "AE and AWB locked"))
    if long_exp:
        rep.add(CheckResult("exposure time", WARN, "; ".join(long_exp) + " (> 4 ms blurs at hand speed)",
                            "cap exposure_time_range_ns at 4000000"))
    cov = rig.coverage_report()
    bad = {n: v for n, v in cov.items() if v < rig.virtual[n].min_coverage}
    det = ", ".join(f"{n} {v:.3f}" for n, v in cov.items())
    if not bad:
        rep.add(CheckResult("coverage", PASS, det))
    else:
        srcs_placeholder = all(rig.cameras[rig.virtual[n].source].placeholder for n in bad)
        rep.add(CheckResult("coverage", WARN if srcs_placeholder else FAIL, det + " (below min_coverage)",
                            "the lens does not see the whole virtual iPhone view: wider lens or re-run calib-align"))


def _clock(p: Probes, rep: DoctorReport) -> None:
    t = p.wall_time_s()
    year = time.gmtime(t).tm_year
    if year < MIN_YEAR:
        rep.add(CheckResult("wall clock", FAIL, f"system time is {time.strftime('%Y-%m-%d %H:%M', time.gmtime(t))} UTC",
                            "no RTC battery: connect to a network and `sudo timedatectl set-ntp true`, or set it by "
                            "hand with `sudo date -u -s 'YYYY-MM-DD HH:MM:SS'` (episode names use the wall clock)"))
        return
    ntp = p.ntp_synced()
    det = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t))
    if ntp is False:
        rep.add(CheckResult("wall clock", WARN, det + ", NTP not synchronized",
                            "sudo timedatectl set-ntp true (needs network); check the time is right"))
    else:
        rep.add(CheckResult("wall clock", PASS, det + ("" if ntp else ", NTP status unknown")))


def _disk(p: Probes, o: DoctorOptions, rig, rep: DoctorReport) -> None:
    if not o.out:
        rep.add(CheckResult("disk", SKIP, "no --out given"))
        return
    out = Path(o.out)
    probe_dir = out
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent
    try:
        du = p.disk_usage(str(probe_dir))
    except OSError as e:
        rep.add(CheckResult("disk", FAIL, f"{out}: {e}", "create the raw root on the NVMe"))
        return
    free_gb = du.free / 1e9
    minutes = du.free / 1e6 / RECORD_MB_S / 60
    la = (rig.capture.get("live_abort") or {}) if isinstance(rig.capture, dict) else {}
    min_gb = float(la.get("min_disk_free_gb", 2.0))
    det = f"{free_gb:.1f} GB free on {probe_dir} (~{minutes:.0f} min at {RECORD_MB_S:g} MB/s)"
    if free_gb < min_gb + 1.0:
        rep.add(CheckResult("disk free", FAIL, det, f"free space (the recorder aborts below {min_gb:g} GB): move old "
                            "sessions off the Jetson"))
    elif minutes < 30:
        rep.add(CheckResult("disk free", WARN, det, "less than 30 min of recording left"))
    else:
        rep.add(CheckResult("disk free", PASS, det))
    if o.quick:
        return
    if not out.exists():
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            rep.add(CheckResult("disk write", FAIL, f"cannot create {out}: {e}", "fix permissions on the raw root"))
            return
    try:
        mbs = p.write_speed_mb_s(str(out), o.write_test_mb, max_s=10.0)
    except OSError as e:
        rep.add(CheckResult("disk write", FAIL, f"{out}: {e}", "fix permissions / mount of the raw root"))
        return
    det = f"{mbs:.0f} MB/s sequential write + fsync"
    if mbs < RECORD_MB_S:
        rep.add(CheckResult("disk write", FAIL, det, "too slow for 2 cameras: record to the NVMe, not the SD card"))
    elif mbs < 2 * RECORD_MB_S:
        rep.add(CheckResult("disk write", WARN, det, "little headroom; prefer the NVMe"))
    else:
        rep.add(CheckResult("disk write", PASS, det))


def _argus_checks(p: Probes, o: DoctorOptions, rig, cams: list[str], rep: DoctorReport, sof: set | None = None) -> bool:
    """GStreamer bindings (argus backend), argus_sof helper (argus_sof backend), daemon, /dev/videoN, sensor modes.
    Returns False if capture can't work."""
    ok = True
    sof = sof or set()
    if sof:
        from sensor_rig.drivers.argus_sof_camera import helper_path

        exe = str(helper_path(rig))
        if p.path_exists(exe) and p.is_executable(exe):
            rep.add(CheckResult("argus_sof helper", PASS, f"{exe} (cameras: {', '.join(sorted(sof))})"))
        else:
            rep.add(CheckResult("argus_sof helper", FAIL, f"{exe} missing or not executable",
                                "build it on the Jetson: sensor_rig/native/argus_sof/build.sh (or set "
                                "capture.argus_sof.helper / switch the camera back to the argus backend)"))
            ok = False
    if not set(cams) - sof:
        rep.add(CheckResult("gstreamer", SKIP, "only argus_sof cameras (no GStreamer needed)"))
    else:
        ok &= _gst_check(p, rep)
    ok &= _daemon_and_devices(p, o, rig, cams, rep)
    if o.quick or not ok:
        return ok
    for n in sorted(sof):
        rep.add(CheckResult(f"argus mode {n}", SKIP, "argus_sof: the helper selects and checks the sensor mode at start"))
    return _mode_checks(p, o, rig, [c for c in cams if c not in sof], rep) and ok


def _gst_check(p: Probes, rep: DoctorReport) -> bool:
    ok = True
    try:
        v = p.import_gst()
        if not p.gst_has_element("nvarguscamerasrc"):
            rep.add(CheckResult("gstreamer", FAIL, f"{v}, but no nvarguscamerasrc element",
                                "install JetPack's nvidia-l4t-gstreamer (sudo apt install nvidia-l4t-gstreamer)"))
            ok = False
        else:
            rep.add(CheckResult("gstreamer", PASS, f"{v}, nvarguscamerasrc found"))
    except (ImportError, ValueError) as e:
        rep.add(CheckResult("gstreamer", FAIL, f"python GStreamer bindings: {e}", APT_HINT["gi"]))
        ok = False
    return ok


def _daemon_and_devices(p: Probes, o: DoctorOptions, rig, cams: list[str], rep: DoctorReport) -> bool:
    ok = True
    if p.process_running("nvargus-daemon"):
        rep.add(CheckResult("nvargus-daemon", PASS, "running"))
    else:
        rep.add(CheckResult("nvargus-daemon", FAIL, "not running", "sudo systemctl restart nvargus-daemon"))
        ok = False
    for n in cams:
        dev = f"/dev/video{int(rig.cameras[n].sensor_id)}"
        if not p.path_exists(dev):
            rep.add(CheckResult(f"{dev} ({n})", FAIL, "missing", "enable 'Camera IMX219 Dual' in "
                                "sudo /opt/nvidia/jetson-io/jetson-io.py, reboot, check the CSI cable orientation"))
            ok = False
            continue
        err = p.access_rw(dev)
        if err:
            rep.add(CheckResult(f"{dev} ({n})", FAIL, err, "sudo usermod -aG video $USER, then log out and in"))
            ok = False
        else:
            vname = p.video_name(dev)
            if vname is not None and not looks_like_csi(vname):
                rep.add(CheckResult(f"{dev} ({n})", WARN, f"present, but it is {vname!r}, not a Jetson CSI camera",
                                    "a USB webcam took this number, or the CSI camera is not detected: unplug USB "
                                    "cameras, check jetson-io 'Camera IMX219 Dual' and the CSI cable"))
            else:
                rep.add(CheckResult(f"{dev} ({n})", PASS, f"present, accessible ({vname or 'name unknown'})"))
    return ok


def _mode_checks(p: Probes, o: DoctorOptions, rig, cams: list[str], rep: DoctorReport) -> bool:
    ok = True
    for n in cams:
        c = rig.cameras[n]
        name = f"argus mode {n}"
        try:
            done, text = run_bounded(lambda c=c: p.argus_mode_text(int(c.sensor_id), o.hw_timeout_s),
                                     o.hw_timeout_s + 2)
        except (OSError, subprocess.SubprocessError) as e:
            rep.add(CheckResult(name, FAIL, f"gst-launch-1.0 failed: {e}", "sudo systemctl restart nvargus-daemon"))
            ok = False
            continue
        if not done:
            rep.add(CheckResult(name, FAIL, f"mode listing timed out after {o.hw_timeout_s:g} s",
                                "sudo systemctl restart nvargus-daemon; check the camera is not in use"))
            ok = False
            continue
        rng = (c.argus or {}).get("exposure_time_range_ns")
        st, det, fix = check_argus_mode(parse_argus_modes(text), c.sensor_mode, c.resolution, c.fps, rng)
        rep.add(CheckResult(name, st, det, fix))
        ok &= st != FAIL
    return ok


def _spi_checks(p: Probes, o: DoctorOptions, rig, rep: DoctorReport) -> bool:
    imu = rig.imu
    dev = imu.spi_device
    if not p.path_exists(dev):
        rep.add(CheckResult(f"spi {dev}", FAIL, "missing", "enable SPI1 (pins 19/21/23/24) in sudo "
                            "/opt/nvidia/jetson-io/jetson-io.py and reboot (or fix imu.spi.device)"))
        return False
    err = p.access_rw(dev)
    if err:
        rep.add(CheckResult(f"spi {dev}", FAIL, err, "install the udev rule from scripts/jetson_setup.sh "
                            "(group 'spi'), add yourself to the group, log out and in"))
        return False
    rep.add(CheckResult(f"spi {dev}", PASS, "present, read/write"))
    try:
        p.import_module("spidev")
    except ImportError:
        rep.add(CheckResult("imu WHO_AM_I", FAIL, "python module spidev missing", APT_HINT["spidev"]))
        return False
    try:
        done, who = run_bounded(lambda: p.who_am_i(dev, imu.spi_speed_hz, imu.spi_mode), o.hw_timeout_s)
    except OSError as e:
        rep.add(CheckResult("imu WHO_AM_I", FAIL, f"SPI transfer failed: {e}", "check the SPI overlay / device"))
        return False
    if not done:
        rep.add(CheckResult("imu WHO_AM_I", FAIL, "SPI transfer timed out", "check the SPI overlay / device"))
        return False
    if who != 0x47:
        rep.add(CheckResult("imu WHO_AM_I", FAIL, f"0x{who:02x}, expected 0x47",
                            "0x00/0xff: check wiring (MISO/MOSI swapped?), CS0, 3.3 V, imu.spi.mode (0 or 3), "
                            "and lower imu.spi.speed_hz to 1000000"))
        return False
    rep.add(CheckResult("imu WHO_AM_I", PASS, "0x47 (ICM-42688-P)"))
    return True


def _capture(p: Probes, o: DoctorOptions, rig, cams: list[str], rep: DoctorReport) -> None:
    """Start every camera at once, read for capture_s, judge each. Bounded by hw_timeout_s + capture_s."""
    backends: dict[str, Any] = {}
    errors: dict[str, str] = {}
    data = {n: ([], [], []) for n in cams}

    cancelled = threading.Event()

    def start_all():
        for n in cams:
            if cancelled.is_set():      # the doctor gave up on a hung start: don't open more cameras
                return
            try:
                cam = p.make_camera(rig, n, o.mock)
                backends[n] = cam
                cam.start()
            except Exception as e:  # noqa: BLE001 - reported per camera
                errors[n] = f"{type(e).__name__}: {e}"

    def read_all():
        t_end = time.monotonic() + o.capture_s
        live = [n for n in cams if n in backends and n not in errors]
        threads = []

        def reader(n):
            cam = backends[n]
            try:
                while time.monotonic() < t_end:
                    f = cam.read(timeout_s=0.2)
                    if f is None:
                        if cam.exhausted:
                            break
                        continue
                    data[n][0].append(int(f.t_ns))
                    data[n][1].append(int(f.t_arrival_ns))
                    data[n][2].append(int(f.seq))
            except Exception as e:  # noqa: BLE001
                errors[n] = f"{type(e).__name__}: {e}"

        for n in live:
            th = threading.Thread(target=reader, args=(n,), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

    try:
        done, _ = run_bounded(start_all, o.hw_timeout_s)
        if not done:
            for n in cams:
                rep.add(CheckResult(f"capture {n}", FAIL, f"camera start timed out after {o.hw_timeout_s:g} s",
                                    "sudo systemctl restart nvargus-daemon; check no other process uses the camera"))
            return
        done, _ = run_bounded(read_all, o.capture_s + o.hw_timeout_s)
        for n in cams:
            if n in errors:
                rep.add(CheckResult(f"capture {n}", FAIL, errors[n], "see HARDWARE_CHECKLIST §1; "
                                    "sudo systemctl restart nvargus-daemon"))
                continue
            if not done:
                rep.add(CheckResult(f"capture {n}", FAIL, "reading frames timed out", "camera hung: restart "
                                    "nvargus-daemon"))
                continue
            cam = backends[n]
            exp = tuple(int(v) for v in rig.cameras[n].resolution)
            if (cam.width, cam.height) != exp:
                rep.add(CheckResult(f"capture {n}", FAIL, f"backend {cam.width}x{cam.height} != rig.yaml {exp}",
                                    "fix resolution in rig.yaml"))
                continue
            s = summarize_frames(*data[n])
            rep.add(judge_capture(n, s, float(cam.fps), int(cam.stats.get("internal_drops", 0))))
    finally:
        cancelled.set()
        for cam in list(backends.values()):
            run_bounded(cam.stop, 5.0)


def _imu_stream(p: Probes, o: DoctorOptions, rig, rep: DoctorReport) -> None:
    import numpy as np

    box: dict = {}

    def run():
        imu = p.make_imu(rig, o.mock)
        box["imu"] = imu
        imu.start()
        t0 = time.monotonic()
        settle = min(0.2, o.imu_s / 4)
        batches = []
        while time.monotonic() - t0 < settle + o.imu_s:
            b = imu.read_batch(timeout_s=0.05)
            if b is not None and len(b) and time.monotonic() - t0 >= settle:
                batches.append(b)
        return imu, batches

    try:
        done, res = run_bounded(run, o.imu_s + o.hw_timeout_s)
    except Exception as e:  # noqa: BLE001
        rep.add(CheckResult("imu rate", FAIL, f"{type(e).__name__}: {e}", "see HARDWARE_CHECKLIST §2"))
        if "imu" in box:
            run_bounded(box["imu"].stop, 5.0)
        return
    if not done:
        rep.add(CheckResult("imu rate", FAIL, "IMU start/read timed out", "check SPI wiring / overlay"))
        if "imu" in box:
            run_bounded(box["imu"].stop, 5.0)
        return
    imu, batches = res
    try:
        t = np.concatenate([b.t_ns for b in batches]) if batches else np.zeros(0, np.int64)
        rate = (len(t) - 1) / ((t[-1] - t[0]) / 1e9) if len(t) >= 2 and t[-1] > t[0] else None
        a = np.concatenate([b.accel for b in batches]) if batches else None
        w = np.concatenate([b.gyro for b in batches]) if batches else None
        an = float(np.linalg.norm(a.mean(axis=0))) if a is not None and len(a) else None
        wn = float(np.linalg.norm(w.mean(axis=0))) if w is not None and len(w) else None
        for r in judge_imu(rate, float(imu.rate_hz), an, wn, int(imu.stats.get("fifo_overflows", 0))):
            rep.add(r)
    finally:
        run_bounded(imu.stop, 5.0)


def run_doctor(rig_path: str | os.PathLike | None = None, opts: DoctorOptions | None = None,
               probes: Probes | None = None, rig=None) -> DoctorReport:
    """Run every check and return the report (never raises for a failed check)."""
    o = opts or DoctorOptions()
    p = probes or Probes()
    rep = DoctorReport(rig_path=str(rig_path) if rig_path else None, mock=o.mock)
    if not _deps(p, o, rep):
        return rep
    _entry_point(p, o, rep)
    if rig is None:
        try:
            from sensor_rig.rig_config import load_rig

            rig = load_rig(rig_path or DEFAULT_RIG)
        except Exception as e:  # noqa: BLE001 - any load/validation error is a FAIL
            rep.add(CheckResult("rig.yaml", FAIL, f"{type(e).__name__}: {e}", "fix the rig.yaml (sensor-rig rig-validate)"))
            return rep
    rep.add(CheckResult("rig.yaml", PASS, f"{rig.rig_name} ({rig_path or DEFAULT_RIG})"))
    _rig_checks(rig, rep)
    _clock(p, rep)
    _disk(p, o, rig, rep)

    backends = (rig.capture.get("backends") or {}) if isinstance(rig.capture, dict) else {}
    cams = list(rig.cameras)
    argus_cams = [n for n in cams if backends.get(n, "argus") in ("argus", "argus_sof")]
    sof_cams = [n for n in argus_cams if backends.get(n) == "argus_sof"]
    imu_backend = backends.get("imu", "icm42688_spi")
    cams_ok = imu_ok = True
    if o.mock:
        for name in ("gstreamer", "nvargus-daemon", "camera devices", "spi + WHO_AM_I"):
            rep.add(CheckResult(name, SKIP, "--mock"))
    else:
        if argus_cams:
            cams_ok = _argus_checks(p, o, rig, argus_cams, rep, sof=set(sof_cams))
        if imu_backend == "icm42688_spi":
            imu_ok = _spi_checks(p, o, rig, rep)
    if not o.sensors:
        rep.add(CheckResult("capture + imu stream", SKIP, "checked by the recorder warm-up"))
        return rep
    if cams_ok:
        _capture(p, o, rig, cams, rep)
    else:
        rep.add(CheckResult("capture", SKIP, "camera checks failed above"))
    if imu_ok:
        _imu_stream(p, o, rig, rep)
    else:
        rep.add(CheckResult("imu stream", SKIP, "SPI checks failed above"))
    return rep


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def register(subparsers) -> None:
    d = subparsers.add_parser("doctor", help="preflight checks before recording (devices, IMU, clock, disk, rig.yaml)",
                              description=__doc__.splitlines()[0])
    d.add_argument("--rig", default=str(DEFAULT_RIG), help="rig.yaml (default: configs/rig_default.yaml)")
    d.add_argument("--out", default=None, help="raw root to check (free space, write speed)")
    d.add_argument("--mock", action="store_true", help="mock cameras + IMU; skip hardware-only checks (laptop)")
    d.add_argument("--quick", action="store_true", help="skip the Argus mode listing and the write-speed test")
    d.add_argument("--json", default=None, help="also write the report as JSON to this path")
    d.add_argument("--capture-s", type=float, default=2.0)
    d.add_argument("--timeout", type=float, default=10.0, help="deadline per hardware call (s)")
    d.set_defaults(func=cmd_doctor)


def cmd_doctor(args) -> int:
    opts = DoctorOptions(mock=args.mock, quick=args.quick, out=args.out, capture_s=args.capture_s,
                         hw_timeout_s=args.timeout)
    rep = run_doctor(args.rig, opts)
    print(rep.table())
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rep.to_dict(), f, indent=1)
            f.write("\n")
    return 0 if rep.ok else 1
