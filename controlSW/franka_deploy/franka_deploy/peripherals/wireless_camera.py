"""WirelessCamera: the policy-host receiver of the wireless sensor path, as a real-env BaseCamera.

Why: the iPhone app (protocol v1) and the rig/sim senders (v2) stream JPEG frames over Socket.IO instead of a cable.
This peripheral receives them, maps each frame to a capture time on this host's time.monotonic() clock exactly once
(ClockMap, clock_map.py), and republishes {"main", "ultrawide"} at a fixed 60 Hz through BaseCamera's robotmq shared
memory, so the task reads it exactly like RigCamera (rig_camera.py) and nothing downstream changes.

What it relies on in BaseCamera (behavior_prompting/deps/real-env/real_env/peripherals/base_camera.py, read-only):
- constructor args name/server_endpoint/logger_endpoint/camera_latency_s/camera_configs/display_resolution_WH/
  config_str (:40-48); config_str must be JSON with "camera_configs" (BaseCameraClient.get_config, :324-332);
- run() calls get_images_bgr_hwc_and_timestamp() in a loop (:184) and skips publishing on an empty dict (:196), so
  "no fresh frame" is returned as ({}, 0.0);
- shapes must equal camera_configs (height, width, 3) (:214-217) and the timestamp must be > 0 (:218-220);
- corrected = timestamp - camera_latency_s (:221): we force camera_latency_s = 0, ClockMap applies the latency;
- clients assert non-decreasing timestamps (:393-395): ClockMap clamps each stream monotonic, and a repeated frame
  keeps its stamp (as RigCamera fixed_rate does, rig_camera.py get_images_bgr_hwc_and_timestamp).

Wire behaviour (sensor_rig/stream/protocol.py has the Swift file:line facts):
- one Flask-SocketIO server per configured port (v1: main 5555, ultrawide 5556; v2: main only), threading mode,
  in daemon threads, bound to transport.bind_host. async_handlers=False: events of one connection are handled in
  arrival order in its own thread (decode happens there), so "newest frame wins" is never reordered.
- v1: "rgb" (base64 JPEG str -> cv2.imdecode BGR), "validate" (returns transport.validate_ack; an empty ack makes the
  app reconnect, SocketClient.swift:93-98), "depth" (counted, ignored).
- v2: "hello" (checked against the sensor yaml: protocol, size, fps, K within 1e-3; problems are logged and kept in
  the health JSON), "packet" (binary JPEGs + sender stamps + seq); a thread calls "clock_ping" at clock.ping_hz
  (plus a short burst after hello) and feeds ClockMap. Frames are dropped (counted) until the first ping.
- require_websocket: polling is NOT refused, because the real app has no forceWebsockets and starts on long-polling,
  then upgrades (protocol.py v1 facts). The current engineio transport per stream is reported in health instead,
  and launch preflight fails on "polling".

Health: every health.stats_period_s the stats JSON (stats_path) is atomically replaced and one line is printed. The
truth log (sim only, truth_path) has one JSON line {seq, stream, t_est, t_arrival} per frame; v1 seq is the per-stream
arrival index (the app sends none), which equals the sim sender's seq when the receiver starts first.

Usage:
    python -m franka_deploy.peripherals.wireless_camera --config camera.yaml [--stats-path S] [--truth-path T]
    # camera.yaml = Station.camera_config(endpoint, logger_endpoint) (station.py), e.g. written by the launcher
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import yaml

from real_env.peripherals.base_camera import BaseCamera

from .clock_map import clock_map_from_config

STREAMS = ("main", "ultrawide")
_V1 = ("v1", "iphumi_socketio_v1")
_V2 = ("v2", "iphumi_socketio_v2")
LOW_RES_HINT = "iPhone app is on Low (320x240): set Stream resolution = High"


def _protocol(p: str) -> str:
    if p in _V1:
        return "v1"
    if p in _V2:
        return "v2"
    raise ValueError(f"transport.protocol must be one of {_V1 + _V2}, got {p!r}")


def _pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=np.float64), q)) if len(x) else None


def check_hello(hello: dict, protocol: str, streams: dict) -> list[str]:
    """Problems of a v2 hello against the sensor yaml streams (empty list = consistent)."""
    from sensor_rig.stream import protocol as P

    errs = list(P.check_hello(hello))
    if errs:
        return errs
    if hello.get("protocol") not in ("v2", protocol):
        errs.append(f"protocol {hello.get('protocol')!r} != {protocol!r}")
    for s, cfg in streams.items():
        hs = (hello.get("streams") or {}).get(s)
        if hs is not None:
            wh = (hs.get("w", hs.get("width_px")), hs.get("h", hs.get("height_px")))
            if wh != (cfg["width_px"], cfg["height_px"]):
                errs.append(f"{s}: size {wh} != yaml {(cfg['width_px'], cfg['height_px'])}")
        fps = cfg.get("physical_fps_hz")
        if fps and hello.get("fps_hz") and abs(float(hello["fps_hz"]) - float(fps)) > 1e-6:
            errs.append(f"{s}: fps {hello['fps_hz']} != yaml {fps}")
        K = (hello.get("K") or {}).get(s)
        if K is not None and cfg.get("K") is not None:
            dK = float(np.max(np.abs(np.asarray(K, float) - np.asarray(cfg["K"], float))))
            if dK > 1e-3:
                errs.append(f"{s}: K differs from yaml by {dK:.4g} (> 1e-3)")
    return errs


class _StreamStats:
    def __init__(self):
        self.arrivals: deque = deque(maxlen=600)
        self.decode_ms: deque = deque(maxlen=600)
        self.age_s: deque = deque(maxlen=600)
        self.n_frames = 0
        self.n_frames_at_last = 0
        self.connects = 0
        self.disconnects = 0
        self.validates = 0
        self.depth = 0
        self.size_mismatches = 0
        self.seq_gaps = 0
        self.unsynced_dropped = 0
        self.transport = None
        self.sids: set = set()


class WirelessCamera(BaseCamera):
    def __init__(self, transport: dict, clock: dict, health: dict, publish: dict | None = None,
                 stats_path: str | None = None, truth_path: str | None = None, on_size_mismatch: str = "error",
                 streams: dict | None = None, video_codec: str | None = "libx264", **base_kwargs):
        """transport/clock/health/publish/streams: the blocks of Station.camera_config() (sensor yaml).
        streams adds K and physical_fps_hz (dejitter grid, hello check); it is an addition to the spec signature.
        base_kwargs go to BaseCamera; camera_latency_s must be 0 (ClockMap applies the latency once)."""
        lat = float(base_kwargs.setdefault("camera_latency_s", 0.0))
        if lat != 0.0:
            raise ValueError(f"WirelessCamera: camera_latency_s must be 0 (got {lat}); put the latency in clock.latency_s "
                             "so it is applied exactly once")
        base_kwargs.setdefault("display_resolution_WH", None)
        super().__init__(**base_kwargs)
        if set(self.camera_configs) != set(STREAMS):
            raise ValueError(f"camera_configs must be exactly {STREAMS}, got {sorted(self.camera_configs)}")
        if on_size_mismatch not in ("error", "count"):
            raise ValueError(f"on_size_mismatch must be error or count, got {on_size_mismatch!r}")
        self.transport = dict(transport)
        self.protocol = _protocol(self.transport["protocol"])
        self.validate_ack = self.transport.get("validate_ack")
        if not self.validate_ack:
            raise ValueError("transport.validate_ack must be non-empty (an empty ack makes the iPhone app reconnect)")
        self.clock_cfg = dict(clock)
        self.health_cfg = dict(health)
        self.publish_cfg = dict(publish or {})
        self.streams_cfg = {s: dict(v) for s, v in (streams or {}).items()}
        for s in STREAMS:
            self.streams_cfg.setdefault(s, {"width_px": self.camera_configs[s]["width"],
                                            "height_px": self.camera_configs[s]["height"]})
        self.on_size_mismatch = on_size_mismatch
        self.stats_path, self.truth_path = stats_path, truth_path
        if video_codec:
            self.video_logger.codec = video_codec       # as RigCamera: no NVENC on the Jetson

        self.clock = clock_map_from_config(self.clock_cfg, self.streams_cfg)
        self._clock_lock = threading.Lock()
        self.max_age_s = float(self.health_cfg.get("max_age_s", 0.3))
        self.stats_period_s = float(self.health_cfg.get("stats_period_s", 1.0))
        self.publish_fps = float(self.camera_configs["main"]["fps"])

        self._lock = threading.Lock()
        self._frames: dict[str, tuple] = {}            # stream -> (bgr, t_capture_mapped, t_arrival, seq)
        self._st = {s: _StreamStats() for s in STREAMS}
        self._seq_rx = {s: 0 for s in STREAMS}         # v1 per-stream arrival index
        self._last_seq: int | None = None              # v2
        self.hello_problems: list[str] = []
        self.hellos = 0
        self.packet_errors = 0
        self.ping_errors = 0
        self.stale_ticks = 0
        self.stale_uw_ticks = 0
        self.published = 0
        self._published_at_last = 0
        self._skew: deque = deque(maxlen=600)
        self._fatal: str | None = None
        self._next_tick: float | None = None
        self._truth = open(truth_path, "a", buffering=1) if truth_path else None
        self._truth_lock = threading.Lock()
        self._stop = threading.Event()
        self._t_start = time.monotonic()
        self._last_stats_t = self._t_start

        self._servers = []
        self._sio: dict[str, Any] = {}
        ports = self.transport["ports"]
        if self.protocol == "v1":
            for s in STREAMS:
                self._start_server(s, int(ports[s]))
        else:
            self._start_server("main", int(ports["main"]))
            threading.Thread(target=self._ping_loop, name="wl-clock-ping", daemon=True).start()
        threading.Thread(target=self._stats_loop, name="wl-stats", daemon=True).start()
        print(f"[{self.name}] WirelessCamera {self.protocol} on {self.transport.get('bind_host')}:"
              f"{[int(p) for p in ports.values()]} clock={self.clock.mode} publish {self.publish_fps:g} Hz "
              f"max_age {self.max_age_s} s")

    # ------------------------------------------------------------------------------------------ socket servers
    def _start_server(self, port_stream: str, port: int) -> None:
        import logging

        from flask import Flask, request
        from flask_socketio import SocketIO
        from werkzeug.serving import make_server

        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app = Flask(f"wireless_{port_stream}_{port}")
        sio = SocketIO(app, async_mode="threading", async_handlers=False,
                       max_http_buffer_size=int(self.transport.get("max_http_buffer_bytes", 10 * 1024 * 1024)),
                       cors_allowed_origins="*")
        self._big_ws_reads(sio)
        st = self._st[port_stream]

        @sio.on("connect")
        def _connect(*_):
            with self._lock:
                st.connects += 1
                st.sids.add(request.sid)
                try:
                    st.transport = sio.server.transport(request.sid)
                except Exception:
                    pass
            print(f"\n[{self.name}] {port_stream}: connect #{st.connects} transport={st.transport}")

        @sio.on("disconnect")
        def _disconnect(*_):
            with self._lock:
                st.disconnects += 1
                st.sids.discard(request.sid)
            print(f"\n[{self.name}] {port_stream}: disconnect")

        @sio.on("validate")
        def _validate(*_):
            st.validates += 1
            return self.validate_ack

        if self.protocol == "v1":
            @sio.on("rgb")
            def _rgb(data):
                t_arr = time.monotonic()
                self._on_v1(port_stream, data, t_arr)

            @sio.on("depth")
            def _depth(*_):
                st.depth += 1
        else:
            @sio.on("hello")
            def _hello(d):
                self._on_hello(d)
                sid = request.sid
                threading.Thread(target=self._ping_burst, args=(sio, sid), daemon=True).start()

            @sio.on("packet")
            def _packet(d):
                t_arr = time.monotonic()
                self._on_packet(d, t_arr)

        srv = make_server(self.transport.get("bind_host", "0.0.0.0"), port, app, threaded=True)  # raises if busy
        threading.Thread(target=srv.serve_forever, name=f"wl-server-{port}", daemon=True).start()
        self._servers.append(srv)
        self._sio[port_stream] = sio

    @staticmethod
    def _big_ws_reads(sio) -> None:
        """Read websocket frames in 1 MiB recv() calls instead of simple-websocket's 4 KiB default (engineio passes
        no options to simple_websocket.Server, engineio/socket.py:151). A ~180 KB frame otherwise takes ~45 recv +
        wsproto rounds, each a GIL handover: measured on localhost 4.8 -> 3.3 ms one-way p50 for a 960x720 frame.
        Per-server copy of the driver table. The signatures are checked HERE (simple_websocket.Server takes
        receive_bytes, SimpleWebSocketWSGI forwards **kwargs, the driver table has "websocket"), so an incompatible
        library version skips the patch with a startup warning instead of failing on every connection (which would
        leave clients on polling). Checked with simple-websocket 1.1.0 / python-engineio 4.14.0."""
        import inspect

        try:
            import simple_websocket
            from engineio.async_drivers._websocket_wsgi import SimpleWebSocketWSGI

            eio = sio.server.eio
            ws_params = inspect.signature(simple_websocket.Server.__init__).parameters
            wsgi_params = inspect.signature(SimpleWebSocketWSGI.__init__).parameters
            problems = []
            if "receive_bytes" not in ws_params:
                problems.append("simple_websocket.Server has no receive_bytes")
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in wsgi_params.values()):
                problems.append("SimpleWebSocketWSGI does not forward **kwargs")
            if eio._async.get("websocket") is not SimpleWebSocketWSGI:
                problems.append(f"engineio websocket driver is {eio._async.get('websocket')!r}")
            if problems:
                print(f"[wireless_camera] WARNING: websocket read-size patch skipped ({'; '.join(problems)}); "
                      "frames still work, ~1.5 ms more transport delay")
                return
            eio._async = dict(eio._async)
            eio._async["websocket"] = lambda handler, server: SimpleWebSocketWSGI(handler, server, receive_bytes=1 << 20)
        except Exception as e:  # pragma: no cover
            print(f"[wireless_camera] WARNING: websocket read-size patch skipped ({e!r})")

    # ------------------------------------------------------------------------------------------ frames
    def _decode(self, stream: str, jpeg: bytes):
        t0 = time.perf_counter()
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self._st[stream].decode_ms.append((time.perf_counter() - t0) * 1e3)
        if bgr is None:
            self.packet_errors += 1
            return None
        cc = self.camera_configs[stream]
        if bgr.shape[:2] != (cc["height"], cc["width"]):
            self._st[stream].size_mismatches += 1
            h, w = bgr.shape[:2]
            msg = (f"{stream}: received {w}x{h} frames, expected {cc['width']}x{cc['height']}"
                   + (f". {LOW_RES_HINT}" if (w, h) == (320, 240) else ""))
            if self.on_size_mismatch == "error":
                self._fatal = msg
            elif self._st[stream].size_mismatches == 1:
                print(f"\n[{self.name}] WARNING (counted, frame dropped): {msg}")
            return None
        return bgr

    def _store(self, stream: str, bgr, t_arr: float, t_sender_ns: int | None, seq: int) -> None:
        st = self._st[stream]
        with self._clock_lock:
            if not self.clock.synced:
                st.unsynced_dropped += 1
                return
            t_est = self.clock.map(stream, t_arr, t_sender_ns, seq)
        with self._lock:
            self._frames[stream] = (bgr, t_est, t_arr, seq)
            st.arrivals.append(t_arr)
            st.n_frames += 1
        if self._truth is not None:
            with self._truth_lock:
                self._truth.write(json.dumps({"seq": seq, "stream": stream, "t_est": t_est, "t_arrival": t_arr}) + "\n")

    def _on_v1(self, stream: str, data, t_arr: float) -> None:
        seq = self._seq_rx[stream]
        self._seq_rx[stream] += 1
        try:
            jpeg = base64.b64decode(data, validate=True) if isinstance(data, str) else bytes(data)
        except Exception:
            self.packet_errors += 1
            return
        bgr = self._decode(stream, jpeg)
        if bgr is not None:
            self._store(stream, bgr, t_arr, None, seq)

    def _on_hello(self, d) -> None:
        self.hellos += 1
        probs = check_hello(d, self.transport["protocol"], self.streams_cfg)
        self.hello_problems = probs
        print(f"\n[{self.name}] hello from {d.get('sender') if isinstance(d, dict) else d!r}: "
              + ("consistent with the sensor yaml" if not probs else "PROBLEMS: " + "; ".join(probs)))

    def _on_packet(self, d, t_arr: float) -> None:
        from sensor_rig.stream import protocol as P

        errs = P.check_packet(d)
        if errs:
            self.packet_errors += 1
            if self.packet_errors <= 3:
                print(f"\n[{self.name}] bad packet: {errs}")
            return
        seq = d["seq"]
        if self._last_seq is not None and seq > self._last_seq + 1:
            self._st["main"].seq_gaps += seq - self._last_seq - 1
        self._last_seq = seq
        if d["clock"] == "sender_realtime" and self.clock.mode == "sender_stamp_ping":
            self.packet_errors += 1     # realtime stamps cannot be mapped with monotonic ping sync
            return
        for stream, jk, tk in (("main", "main_jpeg", "t_main_ns"), ("ultrawide", "uw_jpeg", "t_uw_ns")):
            bgr = self._decode(stream, d[jk])
            if bgr is not None:
                self._store(stream, bgr, t_arr, d[tk], seq)

    # ------------------------------------------------------------------------------------------ clock sync (v2)
    def _ping_once(self, sio, sid) -> bool:
        # emit + ack callback, not sio.call(): call() is refused with async_handlers=False (python-socketio). The ack
        # is delivered by the connection's read thread; t3 is taken there, as soon as the reply is parsed.
        try:
            done = threading.Event()
            box = {}

            def ack(r=None, *_):
                box["t3"] = time.monotonic_ns()
                box["r"] = r
                done.set()

            t0 = time.monotonic_ns()
            sio.emit("clock_ping", {"t0_ns": t0}, to=sid, callback=ack)
            if not done.wait(1.0):
                raise TimeoutError("no clock_ping reply within 1 s")
            r = box["r"]
            with self._clock_lock:
                self.clock.add_ping(t0, int(r["t1_ns"]), int(r["t2_ns"]), box["t3"])
            return True
        except Exception as e:
            self.ping_errors += 1
            if self.ping_errors <= 3:
                print(f"\n[{self.name}] clock_ping failed: {e!r}")
            return False

    def _ping_burst(self, sio, sid, n: int = 5) -> None:
        for _ in range(n):
            if self._stop.is_set() or not self._ping_once(sio, sid):
                return
            time.sleep(0.05)

    def _ping_loop(self) -> None:
        hz = float(self.clock_cfg.get("ping_hz", 2.0))
        if self.clock.mode != "sender_stamp_ping" or hz <= 0:
            return
        sio = self._sio["main"]
        while not self._stop.wait(1.0 / hz):
            with self._lock:
                sids = list(self._st["main"].sids)
            for sid in sids:
                self._ping_once(sio, sid)

    # ------------------------------------------------------------------------------------------ publish
    def get_images_bgr_hwc_and_timestamp(self):
        if self._fatal:
            raise RuntimeError(f"[{self.name}] {self._fatal}")
        period = 1.0 / self.publish_fps                 # fixed-rate tick, as RigCamera fixed_rate
        now = time.monotonic()
        if self._next_tick is None or now - self._next_tick > period:
            self._next_tick = now
        else:
            time.sleep(max(0.0, self._next_tick - now))
        self._next_tick += period
        now = time.monotonic()
        with self._lock:
            fm, fu = self._frames.get("main"), self._frames.get("ultrawide")
        if fm is None or fu is None or now - fm[1] > self.max_age_s:
            self.stale_ticks += 1
            return {}, 0.0
        if now - fu[1] > self.max_age_s:
            # v1 ultrawide is its own socket (port 5556): if it stalls while main flows, publishing a frozen UW
            # image would feed the policy stale data silently. Counted separately so health shows which stream.
            self.stale_ticks += 1
            self.stale_uw_ticks += 1
            return {}, 0.0
        self._st["main"].age_s.append(now - fm[1])
        self._st["ultrawide"].age_s.append(now - fu[1])
        self._skew.append(fm[1] - fu[1])
        self.published += 1
        return {"main": fm[0], "ultrawide": fu[0]}, fm[1]

    # ------------------------------------------------------------------------------------------ health
    def stats(self) -> dict:
        now = time.monotonic()
        dt = max(now - self._last_stats_t, 1e-6)
        out_streams = {}
        for s, st in self._st.items():
            arr = list(st.arrivals)
            ia = np.diff(arr) if len(arr) > 1 else []
            with self._lock:
                for sid in list(st.sids):
                    try:
                        st.transport = self._sio[s].server.transport(sid) if s in self._sio else st.transport
                    except Exception:
                        pass
            out_streams[s] = {
                "fps_in": (st.n_frames - st.n_frames_at_last) / dt, "frames": st.n_frames,
                "inter_arrival_ms": {"p50": _ms(_pct(ia, 50)), "p95": _ms(_pct(ia, 95)),
                                     "max": _ms(float(np.max(ia))) if len(ia) else None},
                "age_s": {"p50": _pct(st.age_s, 50), "p95": _pct(st.age_s, 95)},
                "decode_ms_p95": _pct(st.decode_ms, 95), "connects": st.connects,
                "reconnects": max(0, st.connects - 1), "disconnects": st.disconnects, "connected": len(st.sids),
                "transport": st.transport, "validates": st.validates, "depth_ignored": st.depth,
                "size_mismatches": st.size_mismatches, "seq_gaps": st.seq_gaps, "unsynced_dropped": st.unsynced_dropped,
            }
        with self._clock_lock:
            cs = self.clock.stats()
        return {"t": now, "name": self.name, "protocol": self.protocol, "uptime_s": now - self._t_start,
                "publish_fps": (self.published - self._published_at_last) / dt, "published": self.published,
                "stale_ticks": self.stale_ticks, "stale_uw_ticks": self.stale_uw_ticks, "skew_main_uw_ms": {"p50": _ms(_pct(self._skew, 50)),
                                                                     "p95": _ms(_pct(self._skew, 95))},
                "hellos": self.hellos, "hello_problems": self.hello_problems, "packet_errors": self.packet_errors,
                "ping_errors": self.ping_errors,
                "fatal": self._fatal, "streams": out_streams, "clock": cs}

    def _stats_loop(self) -> None:
        while not self._stop.wait(self.stats_period_s):
            try:
                d = self.stats()
                self._last_stats_t = d["t"]
                self._published_at_last = d["published"]
                for s, st in self._st.items():
                    st.n_frames_at_last = d["streams"][s]["frames"]
                if self.stats_path:
                    tmp = f"{self.stats_path}.tmp"
                    with open(tmp, "w") as f:
                        json.dump(d, f, default=str)
                    os.replace(tmp, self.stats_path)
                m, u = d["streams"]["main"], d["streams"]["ultrawide"]
                print(f"\n[{self.name}] in {m['fps_in']:.1f}/{u['fps_in']:.1f} fps, pub {d['publish_fps']:.1f} Hz, "
                      f"age p95 {_fmt(m['age_s']['p95'])} s, stale {d['stale_ticks']}, decode p95 "
                      f"{_fmt(m['decode_ms_p95'])} ms, reconnects {m['reconnects']}/{u['reconnects']}, "
                      f"transport {m['transport']}/{u['transport']}, clock {d['clock']['mode']}")
            except Exception as e:  # health must never kill the camera
                print(f"\n[{self.name}] stats error: {e!r}")

    def close(self) -> None:
        self._stop.set()
        for srv in self._servers:
            try:
                srv.shutdown()
            except Exception:
                pass
        if self._truth is not None:
            with self._truth_lock:
                self._truth.close()
                self._truth = None

    def run(self):
        try:
            super().run()
        finally:
            self.close()


def _ms(x):
    return None if x is None else x * 1e3


def _fmt(x):
    return "-" if x is None else f"{x:.3f}"


def kwargs_from_camera_config(cfg: dict) -> dict:
    """Station.camera_config() dict (camera.yaml) -> WirelessCamera constructor kwargs."""
    streams = cfg["streams"]
    camera_configs = {s: {"type": "rgb", "width": int(v["width_px"]), "height": int(v["height_px"]),
                          "fps": v.get("fps_hz", cfg.get("publish", {}).get("fps_hz", 60))} for s, v in streams.items()}
    lat = float(cfg.get("camera_latency_s", 0.0))
    if lat != 0.0:
        raise ValueError(f"camera.yaml camera_latency_s must be 0 for the wireless receiver (got {lat}); "
                         "the latency belongs in clock.latency_s")
    transport = {k: cfg[k] for k in ("protocol", "bind_host", "ports", "validate_ack", "require_websocket",
                                     "max_http_buffer_bytes") if k in cfg}
    return {
        "transport": transport, "clock": cfg["clock"], "health": cfg.get("health", {}), "publish": cfg.get("publish"),
        "on_size_mismatch": cfg.get("on_size_mismatch", "error"), "streams": streams,
        "video_codec": cfg.get("video_codec", "libx264"), "stats_path": cfg.get("stats_path"),
        "truth_path": cfg.get("truth_path"),
        "name": cfg["name"], "server_endpoint": cfg["endpoint"], "logger_endpoint": cfg["logger_endpoint"],
        "camera_latency_s": 0.0, "camera_configs": camera_configs, "display_resolution_WH": None,
        "config_str": json.dumps({**cfg, "camera_configs": camera_configs}, default=str),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="camera.yaml (Station.camera_config output)")
    ap.add_argument("--stats-path", default=None, help="health JSON (atomically replaced every stats_period_s)")
    ap.add_argument("--truth-path", default=None, help="sim only: per-frame {seq, stream, t_est, t_arrival} JSON lines")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.stats_path:
        cfg["stats_path"] = args.stats_path
    if args.truth_path:
        cfg["truth_path"] = args.truth_path
    # GIL hand-over: the default 5 ms switch interval lets the publish loop (robotmq put_data, BaseCamera.run) keep
    # the socket threads waiting up to 5 ms per handover while they read a ~180 KB frame in small chunks.
    sys.setswitchinterval(float(cfg.get("gil_switch_interval_s", 0.0005)))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # SystemExit runs the finally blocks (servers, truth file)
    cam = WirelessCamera(**kwargs_from_camera_config(cfg))
    cam.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
