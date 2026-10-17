"""``sensor-rig stamp-offset``: measure each camera's stamp -> capture-time offset with a GPIO-driven LED (T5a).

    sensor-rig stamp-offset --rig rig.yaml --led gpio:PIN [--duration 60] [--roi main=x,y,w,h ...] [--json out.json]
    sensor-rig stamp-offset --mock [--mock-offset-ms main=-30,ultrawide=-27] [--mock-jitter-ms 0.5]

Procedure: point both cameras at an LED wired to a header pin (BOARD numbering, 330 Ohm series resistor to GND).
The recorder runs exactly as in a session (both cameras, JPEG workers, IMU: full load) while a thread blinks the
LED at random phases (on 100-200 ms, off 150-350 ms: ~5 edges/s) and logs CLOCK_MONOTONIC right after each GPIO write.

Estimate per camera (``estimate_offset``, pure): the LED brightness in an ROI (given, or auto: the pixel with the
largest temporal variance) is thresholded (midpoint of the two brightness clusters). For every rising/falling edge
between frames k-1 and k, the LED transition time t_led satisfies ``stamp[k-1] + off < t_led <= stamp[k] + off``
(rig convention ``t_capture = t_stamp + offset``; the edge frame is the first whose mid-exposure saw the new
state), i.e. ``off`` lies in ``[t_led - stamp[k], t_led - stamp[k-1])``. With random LED phases the intersection of
these intervals shrinks to the true offset (gap ~ frame period / number of edges); a soft intersection (minimum
summed violation) is used so stamp jitter cannot empty it. 60 s gives ~300 edges: offset error <= ~0.5 ms with
0.5 ms rms stamp jitter (simulated, 40 runs). Jitter = robust std of the frame intervals / sqrt(2) (the
stamp noise of consecutive frames); drift = offset in the second half of the run minus the first half, per minute
(resolution ~2 ms/min for a 60 s run: smaller values are noise; runs shorter than 60 s mark it unreliable and the CLI
does not print a value).

Output: per camera offset (median of the interval midpoint), jitter RMS, drift, number of edges, the ROI, and the
suggested rig.yaml patch ``cameras.<c>.time.offset_prior_ns`` (negative = stamps are later than mid-exposure). Set
``kalibr_timeshift_cam_imu_s`` to 0 when you paste it, then let Kalibr refine it.
The GPIO LED (``GpioLed``, Jetson.GPIO) is NOT RUN ON HARDWARE YET; ``--mock`` and the tests use ``FakeLed`` and a
camera that draws the LED with a known offset and jitter.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------ LEDs


class Led:
    """Output pin interface (like ``record.button.Button`` for inputs)."""

    def open(self) -> None:
        pass

    def set(self, on: bool) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeLed(Led):
    """Remembers its state history ``[(t_ns, on), ...]`` on ``clock`` (the mock camera draws from it)."""

    def __init__(self, clock):
        self.clock = clock
        self.history: list[tuple[int, bool]] = [(0, False)]
        self._lock = threading.Lock()

    def set(self, on: bool) -> None:
        with self._lock:
            self.history.append((self.clock.now_ns(), bool(on)))

    def state_at(self, t_ns: int) -> bool:
        with self._lock:
            h = list(self.history)
        i = int(np.searchsorted([t for t, _ in h], t_ns, side="right")) - 1
        return h[max(i, 0)][1]


class GpioLed(Led):  # pragma: no cover - needs a Jetson
    def __init__(self, pin: int):
        self.pin = int(pin)
        self._gpio = None

    def open(self) -> None:
        try:
            import Jetson.GPIO as GPIO
        except ImportError as e:
            raise RuntimeError("the LED needs Jetson.GPIO (sudo apt install python3-jetson-gpio; user in group "
                               "'gpio')") from e
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._gpio = GPIO

    def set(self, on: bool) -> None:
        self._gpio.output(self.pin, self._gpio.HIGH if on else self._gpio.LOW)

    def close(self) -> None:
        if self._gpio is not None:
            self._gpio.output(self.pin, self._gpio.LOW)
            self._gpio.cleanup(self.pin)


class Blinker:
    """Blinks ``led`` at random phases; ``events`` = [(t_ns logged right after the write, on)]."""

    def __init__(self, led: Led, clock, on_s=(0.10, 0.20), off_s=(0.15, 0.35), seed: int = 0):
        self.led, self.clock = led, clock
        self.on_s, self.off_s = on_s, off_s
        self.rng = np.random.default_rng(seed)
        self.events: list[tuple[int, bool]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="blinker", daemon=True)

    def _loop(self) -> None:
        on = False
        while not self._stop.is_set():
            self.clock.sleep(float(self.rng.uniform(*(self.off_s if not on else self.on_s))))
            if self._stop.is_set():
                break
            on = not on
            t0 = self.clock.now_ns()
            self.led.set(on)
            t1 = self.clock.now_ns()
            self.events.append(((t0 + t1) // 2, on))      # a GPIO write takes microseconds
        self.led.set(False)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


# ------------------------------------------------------------------------------------------------ analysis


MIN_DRIFT_RUN_S = 60.0   # shorter runs: the half-vs-half drift estimate is noise (e.g. -9.9 ms/min at 20 s)


class NoLedError(RuntimeError):
    pass


def auto_roi(stack: np.ndarray, half: int = 2) -> tuple[int, int, int, int]:
    """(x, y, w, h) around the pixel with the largest temporal variance of a (N, H, W) brightness stack."""
    var = stack.astype(np.float32).var(axis=0)
    y, x = np.unravel_index(int(np.argmax(var)), var.shape)
    y0, x0 = max(0, y - half), max(0, x - half)
    return int(x0), int(y0), int(min(var.shape[1], x + half + 1) - x0), int(min(var.shape[0], y + half + 1) - y0)


def led_states(brightness: np.ndarray, min_contrast: float = 15.0) -> np.ndarray:
    """Threshold at the midpoint of the two brightness clusters (2-means). Raises NoLedError on low contrast."""
    b = np.asarray(brightness, np.float64)
    lo, hi = np.percentile(b, 5), np.percentile(b, 95)
    if hi - lo < min_contrast:
        raise NoLedError(f"no LED: brightness range {hi - lo:.1f} < {min_contrast} gray levels in the ROI")
    thr = (lo + hi) / 2
    for _ in range(10):
        a, c = b[b <= thr], b[b > thr]
        if not len(a) or not len(c):
            break
        thr = (a.mean() + c.mean()) / 2
    return b > thr


def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def estimate_offset(stamps_ns, on, led_events, min_edges: int = 8, max_abs_offset_ns: int = 200_000_000) -> dict:
    """Offset (t_capture - t_stamp) from frame stamps, per-frame LED states and the logged LED transitions."""
    t = np.asarray(stamps_ns, np.int64)
    on = np.asarray(on, bool)
    ev_t = np.array([e[0] for e in led_events], np.int64)
    ev_on = np.array([e[1] for e in led_events], bool)
    dt = np.diff(t)
    period = float(np.median(dt))
    edges = [k for k in range(1, len(t)) if on[k] != on[k - 1] and dt[k - 1] < 1.5 * period]
    if len(edges) < min_edges or not len(ev_t):
        raise NoLedError(f"no LED: {len(edges)} brightness edges (need >= {min_edges}) and {len(ev_t)} LED events")
    lows, highs, times = [], [], []
    for k in edges:
        cand = ev_t[ev_on == on[k]]                      # transitions to the state frame k shows
        if not len(cand):
            continue
        j = int(np.argmin(np.abs(cand - t[k])))
        if abs(int(cand[j]) - int(t[k])) > max_abs_offset_ns:
            continue
        lows.append(int(cand[j]) - int(t[k]))
        highs.append(int(cand[j]) - int(t[k - 1]))
        times.append(int(t[k]))
    if len(lows) < min_edges:
        raise NoLedError(f"no LED: only {len(lows)} edges match a logged LED transition within "
                         f"{max_abs_offset_ns / 1e6:.0f} ms")
    lows, highs, times = map(np.asarray, (lows, highs, times))

    def est(lo, hi):
        """x minimising the summed interval violations sum(max(0, lo - x) + max(0, x - hi)) (a soft interval
        intersection that tolerates stamp jitter); the centre of the flat optimum."""
        grid = np.arange(np.percentile(lo, 50), np.percentile(hi, 50) + 1, 20_000.0)    # 20 us steps
        if not len(grid):
            return 0.5 * (np.median(lo) + np.median(hi))
        cost = (np.maximum(0, lo[None, :] - grid[:, None]).sum(1) + np.maximum(0, grid[:, None] - hi[None, :]).sum(1))
        best = np.flatnonzero(cost <= cost.min() * (1 + 1e-9) + 1.0)
        return float(grid[best].mean())

    off = est(lows, highs)
    half = len(lows) // 2
    drift = None
    if half >= 4:
        span_min = (np.median(times[half:]) - np.median(times[:half])) / 6e10
        if span_min > 0:
            drift = float((est(lows[half:], highs[half:]) - est(lows[:half], highs[:half])) / 1e6 / span_min)
    good = dt[dt < 1.5 * period]
    jitter = robust_std(good) / np.sqrt(2)
    run_s = (int(t[-1]) - int(t[0])) / 1e9
    drift_reliable = drift is not None and run_s >= MIN_DRIFT_RUN_S
    return {"offset_ns": int(round(off)), "offset_ms": off / 1e6, "jitter_ms_rms": jitter / 1e6,
            "drift_ms_per_min": drift, "drift_reliable": drift_reliable, "run_s": run_s,
            "n_edges": int(len(lows)), "interval_width_ms":
            float((np.percentile(highs, 10) - np.percentile(lows, 90)) / 1e6), "frame_period_ms": period / 1e6}


def analyse_episode(ep_dir, led_events, rois: dict | None = None, scale: int = 8) -> dict:
    """Per camera: LED brightness series from the recorded frames -> ``estimate_offset``."""
    import cv2

    from sensor_rig.raw_episode import RawEpisode

    ep = RawEpisode(ep_dir)
    out = {}
    for cam in ("main", "ultrawide"):
        t = ep.camera_times(cam)
        small = []
        for i in range(len(t)):
            g = cv2.cvtColor(ep.read_image(cam, i), cv2.COLOR_BGR2GRAY)
            small.append(cv2.resize(g, (g.shape[1] // scale, g.shape[0] // scale), interpolation=cv2.INTER_AREA))
        stack = np.stack(small)
        if rois and cam in rois:
            x, y, w, h = (int(v) // scale for v in rois[cam])
            roi = (x, y, max(1, w), max(1, h))
        else:
            roi = auto_roi(stack)
        x, y, w, h = roi
        b = stack[:, y:y + h, x:x + w].reshape(len(stack), -1).mean(axis=1)
        try:
            r = estimate_offset(t, led_states(b), led_events)
            r["roi_px"] = [x * scale, y * scale, w * scale, h * scale]
            r["ok"] = True
        except NoLedError as e:
            r = {"ok": False, "error": str(e), "roi_px": [x * scale, y * scale, w * scale, h * scale]}
        out[cam] = r
    return out


# ------------------------------------------------------------------------------------------------ mock camera


class LedMockCamera:
    """Factory for a CameraBackend that draws the FakeLed (bright square) at the TRUE capture time and stamps each
    frame ``t_stamp = t_true - offset + N(0, jitter)``, i.e. ``offset`` is exactly what stamp-offset must find."""

    @staticmethod
    def make(name, w, h, fps, led: FakeLed, offset_ns: int, jitter_ns: int, clock, seed: int = 0,
             led_box=(0.6, 0.3, 0.08)):
        from sensor_rig.drivers.base import CameraBackend, Frame

        rng = np.random.default_rng(seed)
        base = np.full((h, w, 3), 40, np.uint8)
        noise = rng.integers(0, 8, size=(h, w, 1), dtype=np.uint8)
        base = base + noise
        bx, by, bs = int(led_box[0] * w), int(led_box[1] * h), max(4, int(led_box[2] * w))

        class _Cam(CameraBackend):
            def _run(self):
                period = int(1e9 / fps)
                t = self.clock.now_ns()
                seq = 0
                while not self._stopping.is_set():
                    t += period
                    self.clock.sleep_until(t)
                    img = base.copy()
                    if led is not None and led.state_at(t):
                        img[by:by + bs, bx:bx + bs] = 250
                    stamp = t - int(offset_ns) + int(rng.normal(0, jitter_ns)) if jitter_ns else t - int(offset_ns)
                    self._push(Frame(img, stamp, seq, self.clock.now_ns()))
                    seq += 1

        return _Cam(name, w, h, fps, clock=clock)


# ------------------------------------------------------------------------------------------------ run + CLI

@dataclass
class StampOffsetResult:
    cameras: dict
    patch: dict
    episode: str | None

    @property
    def ok(self) -> bool:
        return all(c.get("ok") for c in self.cameras.values())


def run_stamp_offset(rig, out_dir, led: Led, cameras=None, imu=None, clock=None, duration_s: float = 60.0,
                     rois: dict | None = None, warmup_s: float = 2.0, seed: int = 0, keep: bool = False,
                     log=print) -> StampOffsetResult:
    import shutil

    from sensor_rig.record.recorder import Recorder

    out_dir = Path(out_dir)
    raw = out_dir / "stamp_offset_raw"
    rec = Recorder(rig, raw, cameras=cameras, imu=imu, clock=clock, warmup_s=warmup_s, log=lambda m: None)
    clk = rec.clock
    led.open()
    blink = Blinker(led, clk, seed=seed)
    try:
        rec.open()
        try:
            rec.start_episode("demonstration", session="rig-stampoffset", side="left", allow_no_calibration=True,
                              note="stamp-offset LED run")
            blink.start()
            log(f"recording {duration_s:g} s with the LED blinking (both cameras + JPEG + IMU = full load) ...")
            clk.sleep(duration_s)
            blink.stop()
            clk.sleep(0.3)
            path = rec.stop_episode()
        finally:
            rec.close()
    finally:
        led.close()
    cams = analyse_episode(path, blink.events, rois)
    patch = {c: {"time": {"offset_prior_ns": r["offset_ns"], "kalibr_timeshift_cam_imu_s": 0.0}}
             for c, r in cams.items() if r.get("ok")}
    if not keep:
        shutil.rmtree(raw, ignore_errors=True)
    return StampOffsetResult(cams, {"cameras": patch}, str(path) if keep else None)


def register(subparsers) -> None:
    p = subparsers.add_parser("stamp-offset", help="measure camera stamp offsets with a GPIO LED (under full load)",
                              description=__doc__.splitlines()[0])
    p.add_argument("--rig", default=None, help="rig.yaml (default: configs/rig_default.yaml)")
    p.add_argument("--led", default=None, metavar="gpio:PIN", help="LED on this BOARD pin (required unless --mock)")
    p.add_argument("--out", default=".", help="work dir (the temporary raw episode goes here)")
    p.add_argument("--duration", type=float, default=60.0, help="seconds of blinking (60 s ~ 300 edges)")
    p.add_argument("--roi", action="append", default=[], metavar="CAM=x,y,w,h", help="LED region per camera")
    p.add_argument("--json", default=None, help="write the result (incl. the rig.yaml patch) here")
    p.add_argument("--keep", action="store_true", help="keep the recorded episode")
    p.add_argument("--mock", action="store_true", help="mock cameras drawing a fake LED with known offsets")
    p.add_argument("--mock-offset-ms", default="main=-30,ultrawide=-27",
                   help="injected offset: a number for all cameras or cam=value[,cam=value]")
    p.add_argument("--mock-jitter-ms", default="0.5", help="injected stamp jitter (rms): number or cam=value,...")
    p.add_argument("--mock-speed", type=float, default=4.0, help="mock: sim clock speed-up")
    p.set_defaults(func=cmd_stamp_offset)


CAMS = ("main", "ultrawide")


def per_camera(spec, cams=CAMS) -> dict:
    """``-12.3`` (all cameras) or ``main=-30,ultrawide=-27`` (unlisted cameras: 0) -> {cam: float}.
    Raises ValueError with a readable message."""
    spec = str(spec).strip()
    try:
        if "=" not in spec:
            v = float(spec)
            return {c: v for c in cams}
        out = {c: 0.0 for c in cams}
        for part in spec.split(","):
            if part.strip():
                k, v = part.split("=")
                k = k.strip()
                if k not in cams:
                    raise ValueError(f"unknown camera {k!r} (cameras: {', '.join(cams)})")
                out[k] = float(v)
        return out
    except ValueError as e:
        raise ValueError(f"expected a number or cam=value[,cam=value], got {spec!r}: {e}") from None


def cmd_stamp_offset(args) -> int:
    import yaml

    from sensor_rig.doctor import DEFAULT_RIG
    from sensor_rig.rig_config import load_rig_arg

    rig = load_rig_arg(args.rig or DEFAULT_RIG)
    rois = {}
    for r in args.roi:
        cam, vals = r.split("=", 1)
        rois[cam] = [int(v) for v in vals.split(",")]
    cameras = imu = clock = None
    if args.mock:
        from sensor_rig.drivers.base import SimClock
        from sensor_rig.drivers.mock_imu import MockImu

        clock = SimClock(speed=args.mock_speed)
        led = FakeLed(clock)
        try:
            offs = per_camera(args.mock_offset_ms)
            jits = per_camera(args.mock_jitter_ms)
        except ValueError as e:
            print(f"stamp-offset: {e}", file=sys.stderr)
            return 2
        cameras = {}
        for i, c in enumerate(("main", "ultrawide")):
            w, h = (int(v) for v in rig.cameras[c].resolution)
            w, h = min(w, 640), min(h, 480)
            cameras[c] = LedMockCamera.make(c, w, h, float(rig.cameras[c].fps), led, int(offs.get(c, 0) * 1e6),
                                            int(jits[c] * 1e6), clock, seed=i)
        rig = rig.replace(cameras={c: _dc_replace(rig.cameras[c], resolution=(cameras[c].width, cameras[c].height))
                                   for c in rig.cameras})
        imu = MockImu.static(rate_hz=float(rig.imu.output_rate_hz), clock=clock)
    else:
        if not args.led or not args.led.startswith("gpio:"):
            print("stamp-offset: --led gpio:PIN is required (or --mock)", file=sys.stderr)
            return 2
        led = GpioLed(int(args.led.split(":", 1)[1]))
    try:
        if args.duration < MIN_DRIFT_RUN_S:
            print(f"warning: --duration {args.duration:g} s < {MIN_DRIFT_RUN_S:.0f} s: the drift estimate will be "
                  "unreliable (offset and jitter are still measured)", file=sys.stderr)
        res = run_stamp_offset(rig, args.out, led, cameras, imu, clock, args.duration, rois, keep=args.keep)
    except RuntimeError as e:
        print(f"stamp-offset: {e}", file=sys.stderr)
        return 1
    for cam, r in res.cameras.items():
        if r.get("ok"):
            if r["drift_ms_per_min"] is None:
                drift = "-"
            elif not r.get("drift_reliable"):
                drift = f"n/a (run {r['run_s']:.0f} s < {MIN_DRIFT_RUN_S:.0f} s: unreliable)"
            else:
                drift = f"{r['drift_ms_per_min']:+.2f}"
            print(f"{cam:9s} offset {r['offset_ms']:+8.2f} ms   jitter {r['jitter_ms_rms']:.2f} ms rms   drift "
                  f"{drift} ms/min   edges {r['n_edges']}   roi {r['roi_px']}")
        else:
            print(f"{cam:9s} FAILED: {r['error']} (roi {r['roi_px']}): check the LED is in view and lit, or pass --roi")
    if res.patch["cameras"]:
        print("\nrig.yaml patch (t_capture = t_stamp + offset; negative = stamps are later than mid-exposure):")
        print(yaml.safe_dump(res.patch, sort_keys=False).rstrip())
    if args.json:
        Path(args.json).write_text(json.dumps({"cameras": res.cameras, "patch": res.patch}, indent=1) + "\n")
    return 0 if res.ok else 1


def _dc_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)
