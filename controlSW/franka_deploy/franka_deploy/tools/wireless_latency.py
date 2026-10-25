"""wireless-latency: measure the capture-to-arrival latency of a wireless camera stream with a QR clock.

Why: `sensor.clock.latency_s` = 0.08 (iPhone app, protocol v1) is a placeholder (iphone16pro_wifi.yaml provenance
`clock: placeholder`), and the rig's `residual_latency_s` = 0 is unmeasured. The receiver (WirelessCamera, B5) dates
every v1 frame as arrival - latency_s, so a wrong latency shifts every image against the robot state.

How: ONE process, ONE clock. A window on the policy host's own screen shows a QR code of "{t_mono_ns} {counter}",
redrawn at --display-hz. The phone (or the rig) films the screen. The B5 receiver runs in this process with the
station's camera_config (a subclass hooks `_store`; the receiver is not edited). Each received frame is decoded off the
socket thread (cv2.QRCodeDetector, a downscaled grey copy first) and compared with the time it shows.
- v1 (iPhone app, no stamps): latency = t_arrival - t_qr. This is what `clock.latency_s` must equal.
- v2 (rig / sim, sender stamps mapped by ping sync): error = t_mapped - t_qr. The new residual_latency_s is the old
  one plus the p50 of this error (the mapped time is t_sender - theta - residual).
- Also reported for both: arrival - t_qr, which is the wait the task needs after a pause (`sensor.task.camera_latency_s`,
  franka_task.py: pause target = last action time + camera_latency_s). That value is NOT a stamp correction.

Bias: t_qr is the moment the code was drawn, not the moment its photons left the panel, so every number over-estimates
the latency by the display lag (up to one refresh plus panel processing: roughly 10-25 ms at 60 Hz, half of that at
120 Hz). The result prints this bias with the display rate. Use a 120 Hz or faster screen when you can.

The tool never edits station YAML. It prints a `--set` snippet and a provenance line for you to apply.

    python -m franka_deploy wireless-latency measure --config deploy.yaml --profile real-iphone|real-rig [--set k=v ...]
        [--duration-s 60] [--display-hz 60] [--fullscreen] --out DIR
    python -m franka_deploy wireless-latency analyze --frames DIR/frames.jsonl --out DIR
    python -m franka_deploy wireless-latency show [--display-hz 60] [--fullscreen] [--seconds 30]    (aim the phone first)
    python -m franka_deploy wireless-latency synth --protocol v1|v2 --latency-s 0.08 --out DIR      (self-test, no device)

Pass numbers: decode rate >= 0.8; p95 - p50 <= 30 ms (v1) / 5 ms (v2) (= criteria.stage2.cam_stamp_err_ms_p95_max);
the latency is stable over the run within 5 ms (first vs last window).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PASS = {"decode_rate_min": 0.8, "spread_ms_max": {"v1": 30.0, "v2": 5.0}, "stable_ms_max": 5.0}
STREAMS = ("main", "ultrawide")


# ------------------------------------------------------------------------------------------------------ QR clock
def qr_text(t_ns: int, counter: int) -> str:
    return f"{int(t_ns)} {int(counter)}"


def parse_qr(text: str):
    """(t_ns, counter) from a decoded QR string, or None when it is not ours."""
    try:
        a, b = text.split(" ")
        return int(a), int(b)
    except Exception:
        return None


def make_qr(text: str, side_px: int = 480) -> np.ndarray:
    """Grey image side_px x side_px: the QR code (nearest-neighbour scaled, white quiet zone)."""
    enc = cv2.QRCodeEncoder.create()
    m = enc.encode(text)                          # 1 px per module, includes a small border
    if m is None or m.size == 0:
        raise RuntimeError("QR encode failed")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    pad = 4
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    k = max(1, side_px // m.shape[0])
    m = cv2.resize(m, (m.shape[1] * k, m.shape[0] * k), interpolation=cv2.INTER_NEAREST)
    out = np.full((side_px, side_px), 255, np.uint8)
    y0, x0 = max(0, (side_px - m.shape[0]) // 2), max(0, (side_px - m.shape[1]) // 2)
    out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m[:side_px, :side_px]
    return out


def qr_frame(text: str, w: int = 960, h: int = 720, side_px: int | None = None) -> np.ndarray:
    """BGR frame w x h with the QR centred (what the synth sender 'films')."""
    side = side_px or int(min(w, h) * 0.8)
    q = make_qr(text, side)
    img = np.full((h, w), 200, np.uint8)
    y0, x0 = (h - q.shape[0]) // 2, (w - q.shape[1]) // 2
    img[y0:y0 + q.shape[0], x0:x0 + q.shape[1]] = q
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def decode_qr(img_bgr: np.ndarray, det=None):
    """(t_ns, counter) from a camera frame or None. Tries a 0.5x grey copy first (fast), then full size."""
    det = det or cv2.QRCodeDetector()
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    for scale in (0.5, 1.0):
        g = grey if scale == 1.0 else cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        try:
            text, _, _ = det.detectAndDecode(g)
        except cv2.error:
            text = ""
        if text:
            r = parse_qr(text)
            if r is not None:
                return r
    return None


# ------------------------------------------------------------------------------------------------------ analysis
def _pct(x, q):
    return float(np.percentile(np.asarray(x, np.float64), q)) if len(x) else None


def analyze_records(records: list[dict], n_frames: dict, protocol: str, meta: dict | None = None) -> dict:
    """Per-stream statistics + verdicts from decoded frames.

    records: {stream, t_arr, t_map, t_qr, counter} (host monotonic seconds). n_frames: frames received per stream
    (decoded or not), the denominator of the decode rate."""
    meta = meta or {}
    out = {"protocol": protocol, "display_hz": meta.get("display_hz"), "streams": {}, "pass": {}}
    for s in STREAMS:
        rs = [r for r in records if r["stream"] == s]
        n_rx = int(n_frames.get(s, 0))
        if not rs and not n_rx:
            continue
        arr_lat = np.array([r["t_arr"] - r["t_qr"] for r in rs])
        map_err = np.array([r["t_map"] - r["t_qr"] for r in rs])
        primary = arr_lat if protocol == "v1" else map_err
        t = np.array([r["t_arr"] for r in rs])
        d = {"n_received": n_rx, "n_decoded": len(rs), "decode_rate": (len(rs) / n_rx) if n_rx else 0.0}
        if len(rs) >= 3:
            ms = primary * 1e3
            d["primary"] = "arrival_minus_qr_ms" if protocol == "v1" else "mapped_minus_qr_ms"
            d["ms"] = {"p05": _pct(ms, 5), "p50": _pct(ms, 50), "p95": _pct(ms, 95), "max": float(ms.max()),
                       "min": float(ms.min())}
            d["spread_p95_minus_p50_ms"] = d["ms"]["p95"] - d["ms"]["p50"]
            d["arrival_minus_qr_ms"] = {"p50": _pct(arr_lat * 1e3, 50), "p95": _pct(arr_lat * 1e3, 95)}
            if np.ptp(t) > 1.0:
                d["drift_ms_per_min"] = float(np.polyfit(t - t[0], ms, 1)[0] * 60.0)
                w = min(10.0, np.ptp(t) / 3.0)
                a, b = ms[t - t[0] < w], ms[t - t[0] > np.ptp(t) - w]
                d["stable_first_last_ms"] = float(abs(np.median(b) - np.median(a))) if len(a) >= 5 and len(b) >= 5 else None
            else:
                d["drift_ms_per_min"], d["stable_first_last_ms"] = None, None
            d["counter_gaps"] = int(sum(1 for a, b in zip([r["counter"] for r in rs], [r["counter"] for r in rs][1:]) if b < a))
        out["streams"][s] = d
        ok_dec = d["decode_rate"] >= PASS["decode_rate_min"]
        ok_spread = ("spread_p95_minus_p50_ms" in d) and d["spread_p95_minus_p50_ms"] <= PASS["spread_ms_max"][protocol]
        stable = d.get("stable_first_last_ms")
        ok_stable = stable is not None and stable <= PASS["stable_ms_max"]
        out["pass"][s] = {"decode_rate": bool(ok_dec), "spread": bool(ok_spread), "stable": bool(ok_stable),
                          "stable_note": None if stable is not None else "run too short to compare windows (use >= 30 s)"}
    hz = meta.get("display_hz") or 60.0
    out["bias"] = {"display_hz": hz, "note": ("t_qr is the draw time, so every value over-estimates the latency by the display lag: "
                                              f"up to 1 refresh ({1e3 / hz:.1f} ms at {hz:g} Hz) plus panel processing, "
                                              "about 10-25 ms at 60 Hz; use a >= 120 Hz screen to halve it"),
                   "refresh_ms": 1e3 / hz}
    return out


def snippet(result: dict, meta: dict | None = None, today: str | None = None, out_dir: str = "<out>") -> dict:
    """The `--set` lines and the provenance line for the station YAML (the tool never writes them itself)."""
    meta = meta or {}
    proto = result["protocol"]
    today = today or time.strftime("%Y-%m-%d")
    sets, notes = [], []
    st = result["streams"]
    if proto == "v1":
        for s, d in st.items():
            if "ms" in d:
                sets.append(f"sensor.clock.latency_s.{s}={d['ms']['p50'] / 1e3:.4f}")
        prov = f'clock: {{status: measured, source: "wireless-latency {today} {out_dir}/latency.json"}}'
    else:
        cur = meta.get("residual_latency_s") or {}
        for s, d in st.items():
            if "ms" in d:
                sets.append(f"sensor.clock.residual_latency_s.{s}={float(cur.get(s, 0.0)) + d['ms']['p50'] / 1e3:.4f}")
        prov = f'clock: {{status: measured, source: "wireless-latency {today} {out_dir}/latency.json"}}'
    arr95 = [d["arrival_minus_qr_ms"]["p95"] for d in st.values() if "arrival_minus_qr_ms" in d]
    task = None
    if arr95:
        task = float(np.ceil(max(arr95))) / 1e3
        sets.append(f"sensor.task.camera_latency_s={task:.3f}")
        notes.append("sensor.task.camera_latency_s is the wait after a pause until the last stamped frame has arrived "
                     "(franka_task.py: pause target = last action time + camera_latency_s), not a stamp correction. "
                     "Suggested: the p95 arrival-minus-QR time of the slowest stream, rounded up to 1 ms. "
                     "The v1 stamp correction is clock.latency_s above.")
    notes.append(result["bias"]["note"])
    return {"set": sets, "provenance_line": prov, "task_camera_latency_s": task, "notes": notes}


# ------------------------------------------------------------------------------------------------------ receiver hook
def _make_camera_class():
    """LatencyCamera = WirelessCamera that also queues every stored frame for QR decoding (imports real_env lazily)."""
    from franka_deploy.peripherals.wireless_camera import WirelessCamera

    class LatencyCamera(WirelessCamera):
        def __init__(self, *a, **kw):
            self.frame_q: queue.Queue = queue.Queue(maxsize=4096)
            self.n_rx = {s: 0 for s in STREAMS}
            super().__init__(*a, **kw)

        def _store(self, stream, bgr, t_arr, t_sender_ns, seq):
            super()._store(stream, bgr, t_arr, t_sender_ns, seq)    # clock.map exactly once, as in a real run
            self.n_rx[stream] += 1
            with self._lock:
                fr = self._frames.get(stream)
            t_map = fr[1] if fr is not None and fr[3] == seq else None
            if t_map is None:
                return                                              # dropped before the first ping (counted by the receiver)
            try:
                self.frame_q.put_nowait((stream, bgr, t_arr, t_map, seq))
            except queue.Full:
                pass

    return LatencyCamera


def cleanup_own_shm() -> None:
    """The in-process receiver's robotmq segments (/dev/shm/rmq_*_<pid>_*) outlive the process; a later launcher's
    preflight shm_clean would fail on them. Remove ours when the tool is done."""
    import getpass
    import glob

    for f in glob.glob(f"/dev/shm/rmq_{getpass.getuser()}_{os.getpid()}_*"):   # exact: user and pid
        try:
            os.remove(f)
        except OSError:
            pass


class Collector:
    """Decode queued frames on worker threads; keeps records and per-stream received counts."""

    def __init__(self, cam, n_workers: int = 2):
        self.cam, self.records, self.stop = cam, [], threading.Event()
        self.lock = threading.Lock()
        self.threads = [threading.Thread(target=self._work, daemon=True, name=f"qr-decode-{i}") for i in range(n_workers)]
        for t in self.threads:
            t.start()

    def _work(self):
        det = cv2.QRCodeDetector()
        while not self.stop.is_set() or not self.cam.frame_q.empty():
            try:
                stream, bgr, t_arr, t_map, seq = self.cam.frame_q.get(timeout=0.2)
            except queue.Empty:
                continue
            r = decode_qr(bgr, det)
            if r is not None:
                with self.lock:
                    self.records.append({"stream": stream, "t_arr": t_arr, "t_map": t_map, "t_qr": r[0] * 1e-9,
                                         "counter": r[1], "seq": seq})

    def finish(self, timeout_s: float = 30.0):
        end = time.monotonic() + timeout_s
        while not self.cam.frame_q.empty() and time.monotonic() < end:
            time.sleep(0.05)
        self.stop.set()
        for t in self.threads:
            t.join(timeout=5.0)
        with self.lock:
            return sorted(self.records, key=lambda r: r["t_arr"])


def write_outputs(out: str, records, n_frames, protocol: str, meta: dict) -> dict:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "frames.jsonl", "w") as f:
        f.write(json.dumps({"kind": "meta", "protocol": protocol, "n_frames": n_frames, **meta}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    return finish_analysis(d, records, n_frames, protocol, meta)


def finish_analysis(d: Path, records, n_frames, protocol, meta) -> dict:
    res = analyze_records(records, n_frames, protocol, meta)
    snip = snippet(res, meta, out_dir=str(d))
    res["snippet"] = snip
    (d / "latency.json").write_text(json.dumps(res, indent=1))
    (d / "snippet.txt").write_text("\n".join(["# apply to the station YAML by hand (the tool never edits it)"]
                                             + [f"--set {s}" for s in snip["set"]] + ["# " + snip["provenance_line"]]
                                             + ["# " + n for n in snip["notes"]]) + "\n")
    return res


def print_report(res: dict) -> bool:
    ok_all = True
    for s, d in res["streams"].items():
        p = res["pass"].get(s, {})
        line = f"[{s}] received {d['n_received']}, decoded {d['n_decoded']} (rate {d['decode_rate']:.2f})"
        if "ms" in d:
            m = d["ms"]
            line += (f"  {d['primary']}: p05 {m['p05']:.1f} p50 {m['p50']:.1f} p95 {m['p95']:.1f} max {m['max']:.1f} ms"
                     f"  spread {d['spread_p95_minus_p50_ms']:.1f} ms  drift {d['drift_ms_per_min']} ms/min")
        print(line)
        print(f"      pass: decode>=0.8 {p.get('decode_rate')}, spread {p.get('spread')}, stable {p.get('stable')}"
              + (f" ({p['stable_note']})" if p.get("stable_note") else ""))
        ok_all &= all(p.get(k) for k in ("decode_rate", "spread", "stable"))
    sn = res.get("snippet", {})
    print("snippet (apply by hand):")
    for s in sn.get("set", []):
        print(f"  --set {s}")
    print(f"  provenance  {sn.get('provenance_line')}")
    for n in sn.get("notes", []):
        print(f"  note: {n}")
    return ok_all


# ------------------------------------------------------------------------------------------------------ display
def run_display(seconds: float, redraw_hz: float, fullscreen: bool, on_frame=None, stop: threading.Event | None = None):
    """Draw the QR clock on this host's screen for `seconds`. Returns the number of frames drawn."""
    win = "wireless-latency QR clock"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, 800, 800)
    period, t_end, n = 1.0 / redraw_hz, time.monotonic() + seconds, 0
    nxt = time.monotonic()
    while time.monotonic() < t_end and not (stop is not None and stop.is_set()):
        t_ns = time.monotonic_ns()
        img = make_qr(qr_text(t_ns, n), 800)
        cv2.imshow(win, img)
        n += 1
        if on_frame:
            on_frame(n)
        nxt += period
        k = cv2.waitKey(max(1, int((nxt - time.monotonic()) * 1e3))) & 0xFF
        if k in (27, ord("q")):
            break
    cv2.destroyWindow(win)
    return n


# ------------------------------------------------------------------------------------------------------ measure
def _camera_kwargs(a):
    from franka_deploy.launch.launcher import load_config
    from franka_deploy.peripherals.wireless_camera import kwargs_from_camera_config
    from franka_deploy.station import station_from_deploy_cfg

    cfg = load_config(a.config, a.profile)
    cfg["station_set"] = list(cfg.get("station_set", [])) + list(a.set or [])
    st = station_from_deploy_cfg(cfg)
    if st is None or "sensor" not in st.groups:
        raise SystemExit(f"profile {a.profile!r} of {a.config} has no station sensor group")
    cam = cfg["camera"]
    cc = st.camera_config(a.camera_endpoint or cam["endpoint"], a.logger_endpoint or cam["logger_endpoint"])
    cc["on_size_mismatch"] = "count"          # report a wrong resolution instead of dying on it
    cc["health"] = {**cc["health"], "stats_period_s": 5.0}
    return kwargs_from_camera_config(cc), cc, st


def measure(a) -> int:
    kw, cc, st = _camera_kwargs(a)
    proto = "v1" if "v1" in cc["protocol"] else "v2"
    cam = _make_camera_class()(**kw)
    col = Collector(cam)
    meta = {"display_hz": a.display_hz, "profile": a.profile, "duration_s": a.duration_s,
            "latency_s": cc["clock"].get("latency_s"), "residual_latency_s": cc["clock"].get("residual_latency_s"),
            "clock_mode": cc["clock"].get("mode")}
    print(f"[wireless-latency] receiver up ({proto}, ports {cc['ports']}); show the QR window to the camera for "
          f"{a.duration_s:g} s. Make sure the app/sender is connected and the QR fills a good part of the view.")
    t0 = time.monotonic()
    try:
        if a.no_window:
            time.sleep(a.duration_s)
        else:
            run_display(a.duration_s, a.redraw_hz, a.fullscreen,
                        on_frame=lambda n: print(f"\r  t={time.monotonic() - t0:5.1f}s  rx main/uw {cam.n_rx['main']}/"
                                                 f"{cam.n_rx['ultrawide']}  decoded {len(col.records)}", end="", flush=True)
                        if n % 30 == 0 else None)
    finally:
        records = col.finish()
        cam.close()
        cleanup_own_shm()
    print()
    res = write_outputs(a.out, records, dict(cam.n_rx), proto, meta)
    if cam._fatal:
        print(f"receiver error: {cam._fatal}")
    ok = print_report(res)
    print(f"wrote {a.out}/latency.json, frames.jsonl, snippet.txt")
    return 0 if ok and not cam._fatal else 2


# ------------------------------------------------------------------------------------------------------ synth
def synth(a) -> dict:
    """Self-test without a device: a StreamClient sender (the B5 wire format, over LinkConfig clock skew for v2) streams
    QR frames, captured `--latency-s` before they are sent, to a LatencyCamera in this process.

    NOTE v1 is an encode/decode/statistics self-check, NOT a latency-accuracy test: `expected` is built from the same
    sender timestamps that produce `recovered`. v2 (offset + drift through ping sync) tests accuracy.
    v1: recovered = arrival - t_qr; expected = latency_s + the delay of the same frames between the instant the sender
    minted the QR time and their arrival (render + JPEG + loopback, ~4-6 ms, per frame from the sender's own record), so
    the check isolates the QR / JPEG / decode / time-parse / statistics chain.
    v2: the sender clock is offset by --offset-s and drifts by --drift-ppm; recovered = mapped - t_qr, expected 0."""
    from franka_deploy.station import PROFILES, load_station
    from franka_deploy.stream.client import StreamClient
    from franka_deploy.stream.link import LinkConfig

    proto = a.protocol
    prof = "sim-iphone" if proto == "v1" else "sim-rig"
    st = load_station(PROFILES[prof])
    cc = st.camera_config(f"tcp://localhost:{a.camera_port}", f"tcp://localhost:{a.camera_port + 1}")
    port0 = a.port
    cc["ports"] = {s: port0 + i for i, s in enumerate(cc["ports"])}
    cc["bind_host"] = "127.0.0.1"
    cc["on_size_mismatch"] = "count"
    cc["health"] = {**cc["health"], "stats_period_s": 5.0}
    if proto == "v1":
        cc["clock"] = {**cc["clock"], "latency_s": {"main": 0.0, "ultrawide": 0.0}}   # raw arrival is analysed anyway
    from franka_deploy.peripherals.wireless_camera import kwargs_from_camera_config

    cam = _make_camera_class()(**kwargs_from_camera_config(cc))
    col = Collector(cam)
    ports = cc["ports"]
    link = None if proto == "v1" else LinkConfig(clock_offset_s=a.offset_s, clock_drift_ppm=a.drift_ppm)
    meta_v2 = {"fps_hz": 30.0, "streams": {s: {"w": 960, "h": 720} for s in STREAMS}, "K": {s: cc["streams"][s]["K"] for s in STREAMS}}
    cli = StreamClient("127.0.0.1", ports, proto, link=link, sender="wireless_latency_synth", v2_meta=meta_v2)
    cli.connect(20.0)
    if proto == "v2":
        time.sleep(1.0)      # hello + ping burst
    send_start: dict[tuple, float] = {}
    n = int(a.seconds * 30)
    t0 = time.monotonic() + 0.3
    for i in range(n):
        dt = t0 + i / 30.0 - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        t_send = time.monotonic()
        t_cap = t_send - a.latency_s                    # the camera captured the QR `latency` ago
        t_mint = t_send
        img = qr_frame(qr_text(int(t_cap * 1e9), i))
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        if proto == "v1":
            for s in STREAMS:
                send_start[(s, i)] = t_mint
                cli.send_v1(s, jpg, t_cap, i)
        else:
            send_start[("main", i)] = send_start[("ultrawide", i)] = t_mint
            cli.send_v2(jpg, jpg, t_cap, t_cap, i, w=960, h=720)
    cli.flush(5.0)
    time.sleep(0.5)
    records = col.finish()
    cam.close()
    cli.close()
    cleanup_own_shm()
    meta = {"display_hz": a.display_hz, "synthetic": True, "latency_s": a.latency_s, "offset_s": a.offset_s,
            "drift_ppm": a.drift_ppm, "residual_latency_s": {"main": 0.0, "ultrawide": 0.0}}
    res = write_outputs(a.out, records, dict(cam.n_rx), proto, meta)
    # delay of the same frames from minting the QR time to the receiver's arrival; sender and QR share this host's clock
    loop = {s: [r["t_arr"] - send_start[(s, r["counter"])] for r in records if r["stream"] == s and (s, r["counter"]) in send_start]
            for s in STREAMS}
    out = {"protocol": proto, "streams": {}, "truth": {"latency_s": a.latency_s, "offset_s": a.offset_s, "drift_ppm": a.drift_ppm}}
    for s, d in res["streams"].items():
        if "ms" not in d:
            continue
        lp = float(np.median(loop[s]) * 1e3) if loop[s] else 0.0
        expected = (a.latency_s * 1e3 + lp) if proto == "v1" else 0.0
        out["streams"][s] = {"recovered_p50_ms": d["ms"]["p50"], "expected_ms": expected, "send_delay_p50_ms": lp,
                             "error_ms": d["ms"]["p50"] - expected, "decode_rate": d["decode_rate"],
                             "arrival_p50_ms": d["arrival_minus_qr_ms"]["p50"]}
    (Path(a.out) / "synth.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------------ analyze
def analyze_file(a) -> int:
    lines = [json.loads(x) for x in Path(a.frames).read_text().splitlines() if x.strip()]
    meta = next((x for x in lines if x.get("kind") == "meta"), {})
    recs = [x for x in lines if x.get("kind") != "meta"]
    res = write_outputs(a.out, recs, meta.get("n_frames", {}), meta.get("protocol", "v1"),
                        {k: v for k, v in meta.items() if k not in ("kind", "protocol", "n_frames")})
    return 0 if print_report(res) else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_deploy wireless-latency", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="receive the stream in this process while the QR clock is on screen")
    m.add_argument("--config", required=True)
    m.add_argument("--profile", required=True)
    m.add_argument("--set", action="append", default=[], help="station override group.path=value (repeatable)")
    m.add_argument("--duration-s", type=float, default=60.0)
    m.add_argument("--display-hz", type=float, default=60.0, help="the screen's REFRESH rate (only sets the stated display-lag bias)")
    m.add_argument("--redraw-hz", type=float, default=60.0, help="how often the QR code is redrawn; keep it <= the refresh rate, "
                   "or redraws alias on the panel (default 60)")
    m.add_argument("--fullscreen", action="store_true")
    m.add_argument("--camera-endpoint", default=None, help="override camera.endpoint (tests)")
    m.add_argument("--logger-endpoint", default=None, help="override camera.logger_endpoint (tests)")
    m.add_argument("--no-window", action="store_true", help="no QR window (tests)")
    m.add_argument("--out", required=True)
    an = sub.add_parser("analyze", help="re-run the statistics on a frames.jsonl")
    an.add_argument("--frames", required=True)
    an.add_argument("--out", required=True)
    sh = sub.add_parser("show", help="only show the QR clock (aim the phone, no receiver)")
    sh.add_argument("--redraw-hz", type=float, default=60.0, help="QR redraw rate (keep <= the screen's refresh rate)")
    sh.add_argument("--fullscreen", action="store_true")
    sh.add_argument("--seconds", type=float, default=30.0)
    sy = sub.add_parser("synth", help="self-test without a device (a StreamClient sender + the real receiver)")
    sy.add_argument("--protocol", choices=["v1", "v2"], required=True)
    sy.add_argument("--latency-s", type=float, default=0.08)
    sy.add_argument("--offset-s", type=float, default=12.345, help="v2: sender clock offset")
    sy.add_argument("--drift-ppm", type=float, default=30.0, help="v2: sender clock drift")
    sy.add_argument("--seconds", type=float, default=10.0)
    sy.add_argument("--display-hz", type=float, default=60.0)
    sy.add_argument("--port", type=int, default=5555, help="first receiver port (v1 uses port, port+1)")
    sy.add_argument("--camera-port", type=int, default=16702)
    sy.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "analyze":
        return analyze_file(a)
    if a.cmd == "show":
        run_display(a.seconds, a.redraw_hz, a.fullscreen)
        return 0
    if a.cmd == "synth":
        r = synth(a)
        print(json.dumps(r, indent=1))
        ok = all(abs(d["error_ms"]) <= 2.0 and d["decode_rate"] >= PASS["decode_rate_min"] for d in r["streams"].values()) and r["streams"]
        print("synth " + ("PASS (|error| <= 2 ms)" if ok else "FAIL")
              + (": v1 checks the QR encode / JPEG / decode / time-parse / statistics chain only (expected comes from the same "
                 "timestamps); it does not test latency accuracy. v2 (clock offset + drift through ping sync) is the real accuracy test."
                 if a.protocol == "v1" else ""))
        return 0 if ok else 2
    return measure(a)


if __name__ == "__main__":
    sys.exit(main())
