"""render_sender (A7): the ``run.sim.image_source: render`` sender. It follows the sim state PUB (A6, I-SIMSTATE), renders
what the station's sensor would see (``franka_sim.render.Renderer``) and streams it like the device over the run's
emulated link: the iPhUMI app (v1) or the rig (v2).

Why: Stage 3 closes the loop through images that follow the simulated arm, fingers and eraser. The policy host's
receive path (B5 WirelessCamera, ClockMap, BaseCamera) runs unchanged on these frames.

Each tick at --fps: the newest sim state (a SUB with CONFLATE: newest wins); a state older than --stale-s (0.2 s) is
skipped and counted (``stale_states``); else its full ``qpos`` is rendered and JPEG-encoded at
``streams.<s>.jpeg_quality`` of --config (0-1 -> x100).
- v1 (no stamps on the wire): each stream is sent at ``state.t_mono + clock.latency_s[stream]`` (--config), so the
  receiver's ``arrival - latency`` recovers the state time, as iphone_replay_sender models the app's capture
  pipeline (the launcher passes no --capture-latency-s for render). A frame that is ready later is sent at once
  (counted ``late``).
- v2: ``send_v2(main, uw, t_state, t_state, seq, w, h)`` immediately: the capture stamp is the state time; the hello
  K is the station sensor K (the rig's virtual-iPhone K).
Truth lines (--truth, via StreamClient) hold the state time. The ready file (--ready-file, atomic) is written after
the first frame was sent: {sender, protocol, fps, size, K_main, K_uw, state_endpoint, start_now_ns, first_tick_ns,
render_ms_p95, uw_internal_size}. ``render_ms_p95`` = render + JPEG of both streams over a 10-frame warm-up.
``--loop`` is accepted for the launcher (the sender is the device and runs until killed). SIGINT / SIGTERM -> exit 0;
any other exit is non-zero (the launcher's ``sender_died``).

    python -m franka_sim.senders.render_sender --config sender.yaml --state-endpoint tcp://127.0.0.1:5570 \\
        --station station_resolved.yaml --host 127.0.0.1 --ports 5555,5556 --protocol v1 --fps 30 [--link link.yaml] \\
        --truth sender_truth.jsonl --ready-file sender_ready.json [--loop] [--max-s S] [--connect-timeout 15]
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

STALE_S = 0.2


def _quality(q) -> int:
    q = float(q)
    return int(round(100 * q)) if q <= 1.0 else int(q)


class LatestState:
    """SUB on the sim-state PUB, subscribe "", CONFLATE (only the newest message is kept)."""

    def __init__(self, endpoint: str):
        import zmq

        self._zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(endpoint)
        self.last = None
        self.n = 0

    def poll(self, timeout_s: float = 0.0):
        from robotmq import deserialize

        if self.sock.poll(int(max(0, timeout_s * 1000)), self._zmq.POLLIN):
            self.last = deserialize(self.sock.recv())
            self.n += 1
        return self.last

    def close(self):
        self.sock.close(linger=0)


class _Releaser(threading.Thread):
    """One thread owns the v1 sends: items ``(t_release, stream, jpg, t_state, seq)`` go out in t_release order, each
    not before its release time (a frame already past it is sent at once and counted ``late``). The render tick never waits."""

    def __init__(self, client, stop):
        super().__init__(name="render-release", daemon=True)
        self.client, self.stop_ev = client, stop
        self.heap: list = []
        self.cv = threading.Condition()
        self.closing = False
        self.late = 0
        self.late_max = 0.0
        self._n = 0

    def put(self, t_release, stream, jpg, t_state, seq):
        with self.cv:
            self._n += 1
            heapq.heappush(self.heap, (t_release, self._n, stream, jpg, t_state, seq))
            self.cv.notify()

    def finish(self, drain_s: float = 1.0):
        end = time.monotonic() + drain_s
        with self.cv:
            while self.heap and time.monotonic() < end:
                self.cv.wait(0.05)
            self.closing = True
            self.cv.notify()
        self.join(5.0)

    def run(self):
        while True:
            with self.cv:
                while not self.heap and not self.closing:
                    self.cv.wait()
                if self.closing:
                    return
                t_rel = self.heap[0][0]
                dt = t_rel - time.monotonic()
                if dt > 0:
                    self.cv.wait(dt)            # a new, earlier item may arrive meanwhile: re-evaluate
                    continue
                _, _, stream, jpg, t_state, seq = heapq.heappop(self.heap)
                self.cv.notify_all()
            if -dt > 0.002:
                self.late += 1
                self.late_max = max(self.late_max, -dt)
            self.client.send_v1(stream, jpg, t_state, seq)


def _atomic_json(path: str, d: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


def run(args) -> dict:
    import cv2
    import yaml

    from franka_deploy.stream.client import StreamClient
    from franka_deploy.stream.link import LinkConfig
    from franka_sim.backend import load_station_spec
    from franka_sim.model import build_model, forward_at, sim_config
    from franka_sim.render import Renderer

    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    station = load_station_spec(args.station)
    sm = build_model(station, sim_config(station, None))
    rend = Renderer(sm, station)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    sub = client = None
    try:
        streams = ("main", "ultrawide")
        size = rend.size["main"]
        if rend.size["ultrawide"] != size:
            raise SystemExit(f"render_sender: main {size} and ultrawide {rend.size['ultrawide']} sizes differ")
        scfg = (cfg.get("streams") or {})
        quality = {s: _quality((scfg.get(s) or {}).get("jpeg_quality", station.get(f"sensor.streams.{s}.jpeg_quality", 0.9)))
                   for s in streams}
        lat_cfg = ((cfg.get("clock") or {}).get("latency_s")) or {}
        latency = {s: float(lat_cfg.get(s, lat_cfg.get("main", 0.0))) for s in streams}
        K = {s: np.round(rend.K[s], 6).tolist() for s in streams}
        # warm-up (EGL + JPEG) on the home pose: render_ms_p95
        q_home = forward_at(sm).qpos.copy()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            imgs = rend.render(q_home)
            for s in streams:
                cv2.imencode(".jpg", imgs[s], [cv2.IMWRITE_JPEG_QUALITY, quality[s]])
            ts.append((time.perf_counter() - t0) * 1e3)
        render_p95 = float(np.percentile(ts[2:], 95))
        link = LinkConfig.from_yaml(args.link) if args.link else (LinkConfig.from_yaml(cfg["link"]) if cfg.get("link") else None)
        ports = [int(p) for p in args.ports.split(",")]
        stream_ports = {"main": ports[0], "ultrawide": ports[1] if len(ports) > 1 else ports[0] + 1}
        meta = {"fps_hz": args.fps, "streams": {s: {"w": size[0], "h": size[1]} for s in streams}, "K": K}
        client = StreamClient(args.host, stream_ports if args.protocol == "v1" else {"main": ports[0]}, args.protocol,
                              link=link, truth_path=args.truth, sender="render_sender", v2_meta=meta)
        client.connect(args.connect_timeout)
        sub = LatestState(args.state_endpoint)
        print(f"[render_sender] {args.protocol} {size[0]}x{size[1]} @ {args.fps:g} fps, state {args.state_endpoint}, "
              f"jpeg {quality}, v1 latency {latency}, render+jpeg p95 {render_p95:.1f} ms (warm-up)", file=sys.stderr, flush=True)
        period = 1.0 / float(args.fps)
        start_ns = time.monotonic_ns()
        t_next = start_ns / 1e9
        t_end = None if args.max_s is None else t_next + float(args.max_s)
        seq = sent = stale = 0
        first_tick_ns = None
        rt: list = []
        rel = _Releaser(client, stop) if args.protocol == "v1" else None
        if rel is not None:
            rel.start()
        while not stop.is_set():
            now = time.monotonic()
            if t_end is not None and now >= t_end:
                break
            if t_next > now:
                stop.wait(t_next - now)
                if stop.is_set():
                    break
            t_next += period
            if time.monotonic() > t_next:                 # fell behind by more than a tick: resync, never burst
                t_next = time.monotonic() + period
            s = sub.poll(0.0)
            if s is None or time.monotonic() - float(s["t_mono"]) > args.stale_s:
                stale += 1
                continue
            t_state = float(s["t_mono"])
            t0 = time.perf_counter()
            imgs = rend.render(np.asarray(s["qpos"], float))
            jpg = {k: cv2.imencode(".jpg", imgs[k], [cv2.IMWRITE_JPEG_QUALITY, quality[k]])[1].tobytes() for k in streams}
            rt.append((time.perf_counter() - t0) * 1e3)
            if args.protocol == "v1":
                for k in streams:
                    rel.put(t_state + latency[k], k, jpg[k], t_state, seq)
            else:
                client.send_v2(jpg["main"], jpg["ultrawide"], t_state, t_state, seq, w=size[0], h=size[1])
            seq += 1
            sent += 1
            if first_tick_ns is None:
                first_tick_ns = time.monotonic_ns()
                if args.ready_file:
                    _atomic_json(args.ready_file, {
                        "sender": "render_sender", "protocol": args.protocol, "fps": args.fps, "size": list(size),
                        "K_main": K["main"], "K_uw": K["ultrawide"], "state_endpoint": args.state_endpoint,
                        "start_now_ns": int(start_ns), "first_tick_ns": int(first_tick_ns), "render_ms_p95": round(render_p95, 2),
                        "uw_internal_size": None, "latency_s": latency if args.protocol == "v1" else None})
        if rel is not None:
            rel.finish()
            late, late_max = rel.late, rel.late_max
        else:
            late = 0
            late_max = 0.0
        client.flush(2.0)
        st = client.stats()
    finally:
        if client is not None:
            client.close()
        if sub is not None:
            sub.close()
        rend.close()
    return {"sent": sent, "stale_states": stale, "late": late, "late_max_s": round(late_max, 4),
            "render_ms_p50": float(np.percentile(rt, 50)) if rt else None,
            "render_ms_p95": float(np.percentile(rt, 95)) if rt else None, **st}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="sender.yaml (launcher: <session>/config/sender.yaml)")
    ap.add_argument("--state-endpoint", required=True, help="the MuJoCoBackend sim-state PUB")
    ap.add_argument("--station", required=True, help="station_resolved.yaml (or a profile name)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="5555,5556")
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v1")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--link", default=None, help="run/link/*.yaml (default: the config's link, else no emulation)")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--loop", action="store_true", help="accepted for the launcher; the renderer runs until killed")
    ap.add_argument("--max-s", type=float, default=None, help="stop after S seconds")
    ap.add_argument("--stale-s", type=float, default=STALE_S, help="skip a state older than this")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    res = run(args)
    print("[render_sender] done " + json.dumps(res), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
