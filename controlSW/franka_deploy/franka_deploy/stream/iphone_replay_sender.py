"""Replay a recorded iPhone episode over the wire exactly as the iPhUMI app would stream it (v1) or as a rig (v2).

Why: lets the policy host run its real receive path (B5/B6) without the phone: same events, same base64 JPEG
payloads, same 960x720 frames at 30 fps, optionally over an emulated Wi-Fi link.

What the app really streams (iPhUMI/ios_app/iPhUMI/Deployment/DeploymentViewController.swift, details in
sensor_rig/stream/protocol.py): BOTH cameras at the same capture size and fps (High: 1920x1440, 30 fps) scaled x0.5 to
960x720, JPEG q=0.9, one "rgb" event per sample buffer, wide on port 5555 and ultrawide on 5556 (:122-137, :293-307,
:512-575). Training data instead has UW from the 640x480 ARKit video at ~10 Hz. DECISION (emulate the app): the replay
sends UW at the same ticks as main (30 Hz), each tick carrying the latest recorded UW frame (zero-order hold, ~3
repeats) upscaled x1.5 to 960x720. The truth file records, for the UW stream, the capture time of the frame content
(the held frame's slot time), so receivers can measure staleness. This decision is logged on stderr at start.

Resampling (half-pixel convention, as cv2 and the K scaling): main 1920x1440 -> output uses INTER_AREA (downscale);
UW 640x480 -> output uses INTER_LINEAR (upscale). K is scaled with the same convention:
f' = f*s, c' = (c + 0.5)*s - 0.5 (per axis).

Timing: frame i of the episode is sent at start + (t_i - t_0) with t_i from IphoneEpisode.main_times_unix_ns();
`start` = --start-now-ns (host monotonic ns) or now + 0.5 s. With --fps 30 every 2nd main slot is used (stride =
round(source_hz / fps)). Decode+encode runs in a producer thread so pacing is not disturbed.

Usage:
  python -m franka_deploy.stream.iphone_replay_sender --episode data/raw/2026-10-01/<ep> --host 127.0.0.1 \\
      [--ports 5555,5556] [--protocol v1|v2] [--fps 30] [--size 960x720] [--jpeg-quality 90] \\
      [--link run/link/wifi_good.yaml] [--truth T.jsonl] [--ready-file F] [--loop] [--start-now-ns N] [--max-s S]
Ready file (JSON, atomic): {start_now_ns, first_tick_ns, first_slot, fps, protocol, episode, size, K_main, K_uw, loop_span_ns}.
loop_span_ns = the loop period (host ns per lap, B8 replay_gt).
start_now_ns = host monotonic ns of episode slot 0 (what --start-now-ns sets); first_tick_ns = host ns of the first
frame actually sent (slot first_slot, usually 2: no tick before the first UW frame). Align replay_gt with start_now_ns
(the episode's own time origin), not with the first tick.
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

from sensor_rig.sim.iphone_reader import IphoneEpisode
from sensor_rig.stream import protocol as P

from .client import StreamClient
from .link import LinkConfig


def scale_K(K, src_wh, dst_wh) -> list:
    """Intrinsics for a resized image, half-pixel convention."""
    K = np.asarray(K, dtype=np.float64).copy()
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K.tolist()


def resize(img: np.ndarray, size) -> np.ndarray:
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def _jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def plan_ticks(ep: IphoneEpisode, fps: float, max_s: float | None = None):
    """(slot indices sent, their times in s from the first slot, uw frame index held at each tick, uw content times)."""
    t = (ep.main_times_unix_ns() - ep.main_times_unix_ns()[0]) / 1e9
    n = len(t)
    src_hz = (n - 1) / t[-1] if t[-1] > 0 else 60.0
    stride = max(1, int(round(src_hz / fps)))
    uw_slots = ep.uw_slots()
    slots = np.arange(0, n, stride)
    slots = slots[slots >= uw_slots[0]]   # no tick before the first UW frame exists (nothing to hold, never a future frame)
    if max_s is not None:
        slots = slots[t[slots] <= max_s]
    k = np.clip(np.searchsorted(uw_slots, slots, side="right") - 1, 0, len(uw_slots) - 1)
    return slots, t[slots], k, t[uw_slots[k]], stride, src_hz


def _produce(ep, slots, k, size, quality, q, stop):
    """Decode both videos once, push (tick, main_jpeg, uw_jpeg, uw_idx) in order."""
    need = set(slots.tolist())
    uw_jpegs: dict[int, bytes] = {}
    uw_iter = ep.iter_uw_frames()
    uw_next = -1
    uw_cur = None
    last_main_slot = int(slots[-1])
    ti = 0
    for i, frame in enumerate(ep.iter_main_frames()):
        if stop.is_set() or i > last_main_slot:
            break
        if i not in need:
            continue
        kk = int(k[ti])
        while uw_next < kk:
            uw_cur = next(uw_iter)
            uw_next += 1
        if kk not in uw_jpegs:
            uw_jpegs.clear()  # frames are held forward only; keep one
            uw_jpegs[kk] = _jpeg(resize(uw_cur, size), quality)
        item = (ti, _jpeg(resize(frame, size), quality), uw_jpegs[kk], kk)
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                break
            except queue.Full:
                pass
        ti += 1
    q.put(None)


def run(args) -> dict:
    ep = IphoneEpisode(args.episode)
    size = tuple(int(x) for x in args.size.lower().split("x"))
    ports = [int(p) for p in args.ports.split(",")]
    slots, tt, k, uw_t, stride, src_hz = plan_ticks(ep, args.fps, args.max_s)
    first = next(ep.iter_main_frames())
    first_uw = next(ep.iter_uw_frames())
    K_main = scale_K(ep.K_main, (first.shape[1], first.shape[0]), size)
    K_uw = scale_K(ep.K_uw, (first_uw.shape[1], first_uw.shape[0]), size)
    print(f"[replay] {ep.name}: {len(slots)} ticks, source {src_hz:.1f} Hz stride {stride} -> {args.fps} fps, "
          f"size {size}; UW emulates the app: held at main ticks from {len(ep.uw_slots())} recorded UW frames "
          f"(app streams real UW at camera fps)", file=sys.stderr)
    link = LinkConfig.from_yaml(args.link) if args.link else None
    lat_main = float(args.capture_latency_s)
    lat_uw = lat_main if args.capture_latency_uw_s is None else float(args.capture_latency_uw_s)
    stream_ports = {"main": ports[0], "ultrawide": ports[1] if len(ports) > 1 else ports[0] + 1}
    meta = {"fps_hz": args.fps, "streams": {"main": {"w": size[0], "h": size[1]}, "ultrawide": {"w": size[0], "h": size[1]}},
            "K": {"main": K_main, "ultrawide": K_uw}}
    client = StreamClient(args.host, stream_ports if args.protocol == "v1" else {"main": ports[0]}, args.protocol,
                          link=link, truth_path=args.truth, sender="iphone_replay_sender", v2_meta=meta)
    client.connect(args.connect_timeout)
    start_ns = args.start_now_ns if args.start_now_ns is not None else time.monotonic_ns() + 500_000_000
    if args.ready_file:
        # start_now_ns is the host time of episode slot 0; the first tick sent is slot `slots[0]` (later than slot 0
        # when the first UW frame comes after it, typically slot 2): first_tick_ns is that tick's host time.
        d = {"start_now_ns": int(start_ns), "first_tick_ns": int(start_ns + round(float(tt[0]) * 1e9)) if len(tt) else None,
             "first_slot": int(slots[0]) if len(slots) else None, "fps": args.fps, "protocol": args.protocol,
             "episode": ep.name, "size": list(size), "K_main": K_main, "K_uw": K_uw,
             "loop_span_ns": int(round((float(tt[-1]) + 1.0 / args.fps) * 1e9)) if len(tt) else None}   # B8: replay_gt
        tmp = args.ready_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, args.ready_file)
    period = float(tt[-1] + 1.0 / args.fps) if len(tt) else 0.0
    sent = 0
    late_max = 0.0
    stop = threading.Event()
    try:
        loop = 0
        t_first_sent = t_last_sent = None
        while True:
            q: queue.Queue = queue.Queue(maxsize=32)
            th = threading.Thread(target=_produce, args=(ep, slots, k, size, args.jpeg_quality, q, stop), daemon=True)
            th.start()
            while True:
                item = q.get()
                if item is None:
                    break
                ti, mj, uj, kk = item
                t_target = start_ns / 1e9 + loop * period + float(tt[ti])
                seq = sent
                t_uw_true = t_target - float(tt[ti] - uw_t[ti])
                if args.protocol == "v1":
                    # v1 has no stamps: the receiver dates a frame by arrival minus the app's capture latency. Model that
                    # pipeline here: a frame captured at t_target (truth) leaves the phone capture_latency later.
                    for lat, stream, jpg, t_true in sorted(
                            [(lat_main, "main", mj, t_target), (lat_uw, "ultrawide", uj, t_uw_true)], key=lambda x: x[0]):
                        dt = t_target + lat - time.monotonic()
                        if dt > 0:
                            time.sleep(dt)
                        else:
                            late_max = max(late_max, -dt)
                        client.send_v1(stream, jpg, t_true, seq)
                else:
                    dt = t_target - time.monotonic()
                    if dt > 0:
                        time.sleep(dt)
                    else:
                        late_max = max(late_max, -dt)
                    client.send_v2(mj, uj, t_target, t_uw_true, seq, w=size[0], h=size[1])
                sent += 1
                t_first_sent = t_first_sent or t_target
                t_last_sent = t_target
            th.join()
            if not args.loop:
                break
            loop += 1
        client.flush(5.0)
    finally:
        stop.set()
        st = client.stats()
        client.close()
    return {"sent": sent, "late_max_s": late_max, "span_s": (t_last_sent - t_first_sent) if t_first_sent else 0.0, **st}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episode", required=True, help="raw iPhone episode directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="5555,5556")
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v1")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default="960x720")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--link", default=None, help="run/link/*.yaml (default: no emulation)")
    ap.add_argument("--capture-latency-s", type=float, default=0.0,
                    help="v1 only: the app's capture->send pipeline (main stream): frames leave this long after their capture time "
                         "(truth keeps the capture time). Set from the sensor's clock.latency_s when its clock mode is "
                         "arrival_minus_latency; the link's own latency_s stays the Wi-Fi hop.")
    ap.add_argument("--capture-latency-uw-s", type=float, default=None, help="same for the ultrawide stream (default: the main value)")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--start-now-ns", type=int, default=None, help="host monotonic ns of the first frame")
    ap.add_argument("--max-s", type=float, default=None, help="stop after S seconds of episode time")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args(argv)
    res = run(args)
    print("[replay] done " + json.dumps(res), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
