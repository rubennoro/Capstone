"""RigStreamSender: the rig's half of the wireless path (v2 protocol), sensor side.

Why: the policy host receives iPhone-equivalent frames over Wi-Fi. On the rig, RigFrameSource already produces them
(both cameras remapped to the virtual iPhone cameras at 960x720, true capture stamps). This module JPEG-encodes each
packet and sends it as a v2 "packet" event whose stamps are the rig's real capture times in the sender's own clock
(CLOCK_MONOTONIC), so the receiver (franka_deploy ClockMap) can map them to host time with `clock_ping`. Unlike the
iPhone app (whose backlog sits in the socket buffer and grows latency), the rig has real backpressure: a bounded queue
between capture and send that drops the oldest packet, so latency never builds up and every drop is counted.

Threads: capture (RigFrameSource.next_packet -> queue) and the caller's thread (queue -> JPEG -> transport). The queue
holds raw packets (not JPEGs), so a dropped packet costs no encode time.

Transport: any object with connect(hello, timeout_s) / send_packet(pkt_dict, size_bytes) / answer_ping() / stats() /
close(). The default (`SocketioTransport`) is a small python-socketio websocket client; the sim replay sender injects
one that routes through the Wi-Fi LinkEmulator (franka_deploy), which is why sensor_rig never imports franka_deploy.

Usage:
    sender = RigStreamSender(rig, "192.168.1.20", 5555, backend="argus")
    stats = sender.run(duration_s=60)          # Ctrl-C / sender.stop() ends it; CLI: sensor-rig stream-send
"""

from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np

from sensor_rig.stream import protocol as P

log = logging.getLogger("sensor_rig.stream")

DROP_POLICIES = ("drop_oldest", "drop_newest")
CAMERAS = ("main", "ultrawide")


def _pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=np.float64), q)) if len(v) else None


class SocketioTransport:
    """python-socketio websocket client: hello on every (re)connect, `clock_ping` answered in the sender clock.

    send_packet() blocks until the engine.io write queue has drained (the real backpressure point: a slow link makes
    the sender thread wait, the bounded capture queue then drops the oldest packets), and skips (counted) while
    disconnected. Reconnection is automatic (Wi-Fi roaming)."""

    def __init__(self, host: str, port: int, clock_ns=time.monotonic_ns, drain_timeout_s: float = 1.0):
        import socketio

        self.url = f"http://{host}:{int(port)}"
        self.clock_ns = clock_ns
        self.drain_timeout_s = float(drain_timeout_s)
        self._hello: dict | None = None
        self.skipped_disconnected = 0
        self.connects = 0
        self.sio = socketio.Client(reconnection=True, reconnection_delay=0.2, reconnection_delay_max=2.0, logger=False,
                                   engineio_logger=False)

        @self.sio.event
        def connect():
            self.connects += 1
            if self._hello is not None:
                self.sio.emit(P.EVENT_HELLO, self._hello)

        @self.sio.on(P.EVENT_CLOCK_PING)
        def clock_ping(data):
            return self.answer_ping(data)

    def connect(self, hello: dict, timeout_s: float = 10.0) -> None:
        self._hello = hello
        end = time.monotonic() + timeout_s
        while True:
            try:
                self.sio.connect(self.url, transports=["websocket"],
                                 wait_timeout=max(0.5, min(2.0, end - time.monotonic())))
                return
            except Exception as e:
                if time.monotonic() >= end:
                    raise ConnectionError(f"policy host receiver at {self.url} not reachable: {e}") from e
                time.sleep(0.2)

    def answer_ping(self, data=None) -> dict:
        t1 = int(self.clock_ns())
        return {"t1_ns": t1, "t2_ns": int(self.clock_ns())}

    def send_packet(self, pkt: dict, size: int) -> None:
        if not self.sio.connected:
            self.skipped_disconnected += 1
            return
        q = getattr(getattr(self.sio, "eio", None), "queue", None)
        if q is not None:
            end = time.monotonic() + self.drain_timeout_s
            while q.qsize() > 0 and time.monotonic() < end:
                time.sleep(0.001)
        self.sio.emit(P.EVENT_PACKET, pkt)

    def stats(self) -> dict:
        return {"connects": self.connects, "skipped_disconnected": self.skipped_disconnected,
                "connected": bool(self.sio.connected)}

    def close(self) -> None:
        try:
            self.sio.disconnect()
        except Exception:
            pass


class RigStreamSender:
    def __init__(self, rig, host: str, port: int, *, fps_hz: float = 30.0, jpeg_quality: int = 90, backend: str = "argus",
                 backend_opts: dict | None = None, pairing: dict | None = None, max_queue: int = 4,
                 drop_policy: str = "drop_oldest", output_size=(960, 720), clock: str = "sender_monotonic",
                 transport=None, sources: dict | None = None, sender_name: str = "sensor-rig stream-send",
                 sync: str = "live", startup_timeout_s: float = 5.0, connect_timeout_s: float = 15.0):
        if drop_policy not in DROP_POLICIES:
            raise ValueError(f"drop_policy must be one of {DROP_POLICIES}, got {drop_policy!r}")
        if clock not in P.CLOCKS:
            raise ValueError(f"clock must be one of {P.CLOCKS}, got {clock!r}")
        if int(max_queue) < 1:
            raise ValueError("max_queue must be >= 1")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be 1..100")
        if fps_hz <= 0:
            raise ValueError("fps_hz must be > 0")
        self.rig, self.host, self.port = rig, host, int(port)
        self.fps_hz, self.jpeg_quality = float(fps_hz), int(jpeg_quality)
        self.backend, self.backend_opts, self.pairing = backend, dict(backend_opts or {}), pairing
        self.max_queue, self.drop_policy = int(max_queue), drop_policy
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.clock, self.sender_name = clock, sender_name
        self.sources, self.sync = sources, sync
        self.startup_timeout_s, self.connect_timeout_s = float(startup_timeout_s), float(connect_timeout_s)
        # realtime stamps = monotonic + the (constant) realtime-monotonic offset taken now
        self._rt_offset_ns = time.time_ns() - time.monotonic_ns() if clock == "sender_realtime" else 0
        self.transport = transport if transport is not None else SocketioTransport(host, port, clock_ns=self.clock_ns)
        self.frame_source = None
        self._q: collections.deque = collections.deque()      # (t_enqueued_monotonic, seq, Packet)
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._capture_done = False
        self._capture_error: BaseException | None = None
        self.reset_stats()

    # ------------------------------------------------------------------------------------------------ clock
    def clock_ns(self) -> int:
        """The sender clock the stamps and clock_ping answers use (monotonic, or realtime when clock=sender_realtime)."""
        return time.monotonic_ns() + self._rt_offset_ns

    def stamp_ns(self, t_capture_mono_ns: int) -> int:
        return int(t_capture_mono_ns) + self._rt_offset_ns

    # ------------------------------------------------------------------------------------------------ stats
    def reset_stats(self) -> None:
        self.n_captured = self.n_sent = self.n_dropped = self.n_repeated = self.n_decimated = 0
        self.bytes_sent = 0
        self._wait_ms: collections.deque = collections.deque(maxlen=2000)
        self._enc_ms: collections.deque = collections.deque(maxlen=2000)
        self._send_ms: collections.deque = collections.deque(maxlen=2000)
        self.max_queue_len = 0
        self._t_first = self._t_last = None

    def stats(self) -> dict:
        span = (self._t_last - self._t_first) if self._t_first is not None and self._t_last is not None else 0.0
        return {"captured": self.n_captured, "sent": self.n_sent, "dropped": self.n_dropped,
                "repeats_skipped": self.n_repeated, "decimated": self.n_decimated, "bytes": self.bytes_sent,
                "max_queue_len": self.max_queue_len, "max_queue": self.max_queue, "drop_policy": self.drop_policy,
                "send_fps": (self.n_sent - 1) / span if span > 0 and self.n_sent > 1 else None,
                "queue_wait_ms": {"p50": _pct(self._wait_ms, 50), "p95": _pct(self._wait_ms, 95),
                                  "max": max(self._wait_ms) if self._wait_ms else None},
                "encode_ms": {"p50": _pct(self._enc_ms, 50), "p95": _pct(self._enc_ms, 95)},
                "transport_ms": {"p50": _pct(self._send_ms, 50), "p95": _pct(self._send_ms, 95)},
                "transport": self.transport.stats() if hasattr(self.transport, "stats") else {},
                "source": self.frame_source.stats() if self.frame_source is not None else None}

    # ------------------------------------------------------------------------------------------------ hello
    def hello(self) -> dict:
        w, h = self.output_size
        K = {n: np.asarray(self.rig.virtual[n].camera((w, h)).K).tolist() for n in CAMERAS}
        return P.make_hello(self.sender_name, P.V2, {n: {"w": w, "h": h} for n in CAMERAS}, self.fps_hz, K)

    # ------------------------------------------------------------------------------------------------ capture
    def _enqueue(self, seq: int, packet) -> None:
        with self._cv:
            if len(self._q) >= self.max_queue:
                if self.drop_policy == "drop_oldest":
                    self._q.popleft()
                    self._q.append((time.monotonic(), seq, packet))
                self.n_dropped += 1                     # drop_newest: the incoming packet is the one lost
            else:
                self._q.append((time.monotonic(), seq, packet))
            self.max_queue_len = max(self.max_queue_len, len(self._q))
            self._cv.notify()

    def _capture_loop(self) -> None:
        period_ns = 1e9 / self.fps_hz
        last_t = None
        seq = 0
        try:
            while not self._stop.is_set():
                pkt = self.frame_source.next_packet()
                if pkt is None:                         # finite replay exhausted
                    break
                if pkt.repeated:                        # same stamp as before: re-sending it would only add latency
                    self.n_repeated += 1
                    continue
                t = pkt.t_capture_ns["main"]
                if last_t is not None and t - last_t < 0.75 * period_ns:   # camera faster than the target rate
                    self.n_decimated += 1
                    continue
                last_t = t
                self.n_captured += 1
                self._enqueue(seq, pkt)
                seq += 1
        except BaseException as e:                      # RuntimeError from RigFrameSource after repeated timeouts
            self._capture_error = e
            log.error("capture stopped: %r", e)
        finally:
            with self._cv:
                self._capture_done = True
                self._cv.notify_all()

    # ------------------------------------------------------------------------------------------------ send
    def _jpeg(self, img) -> bytes:
        import cv2

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def _build(self, seq: int, pkt) -> tuple[dict, int]:
        t0 = time.perf_counter()
        mj, uj = self._jpeg(pkt.images["main"]), self._jpeg(pkt.images["ultrawide"])
        self._enc_ms.append((time.perf_counter() - t0) * 1e3)
        w, h = self.output_size
        d = P.make_packet(seq, mj, uj, self.stamp_ns(pkt.t_capture_ns["main"]), self.stamp_ns(pkt.t_capture_ns["ultrawide"]),
                          self.clock, w, h)
        return d, len(mj) + len(uj)

    def _make_source(self):
        from sensor_rig.deploy.frame_source import RigFrameSource

        w, h = self.output_size
        return RigFrameSource(self.rig, output_sizes={n: (w, h) for n in CAMERAS}, backend=self.backend,
                              backend_opts=self.backend_opts, pairing=self.pairing, sources=self.sources, sync=self.sync,
                              startup_timeout_s=self.startup_timeout_s)

    def run(self, duration_s: float | None = None) -> dict:
        """Connect, capture and send until `duration_s`, stop(), or a finite source is exhausted. Returns stats()."""
        self._stop.clear()
        self._capture_done, self._capture_error = False, None
        self.frame_source = self._make_source()
        self.frame_source.start()
        th = None
        try:
            self.transport.connect(self.hello(), self.connect_timeout_s)
            th = threading.Thread(target=self._capture_loop, name="rig-capture", daemon=True)
            th.start()
            t_end = time.monotonic() + duration_s if duration_s is not None else None
            while not self._stop.is_set():
                if t_end is not None and time.monotonic() >= t_end:
                    break
                with self._cv:
                    if not self._q:
                        if self._capture_done:
                            break
                        self._cv.wait(0.1)
                        continue
                    t_enq, seq, pkt = self._q.popleft()
                self._wait_ms.append((time.monotonic() - t_enq) * 1e3)
                d, size = self._build(seq, pkt)
                t0 = time.perf_counter()
                self.transport.send_packet(d, size)
                self._send_ms.append((time.perf_counter() - t0) * 1e3)
                self.n_sent += 1
                self.bytes_sent += size
                now = time.monotonic()
                self._t_first = self._t_first if self._t_first is not None else now
                self._t_last = now
        finally:
            self._stop.set()
            if th is not None:
                th.join(timeout=2.0)
            self.frame_source.stop()
        stats = self.stats()
        if self._capture_error is not None:
            stats["capture_error"] = repr(self._capture_error)
        return stats

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def close(self) -> None:
        self.stop()
        if hasattr(self.transport, "close"):
            self.transport.close()
