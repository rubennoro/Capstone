"""StreamClient: Socket.IO sender that reproduces the iPhUMI app's wire behaviour (v1) or the rig's (v2).

Why: the policy host's receiver (B5) must be exercised with exactly what the phone puts on the wire, over a link with
Wi-Fi-like delays. This client is used by the iPhone replay sender, the sim camera sender and tests. Protocol facts
(events, payloads, validate/reconnect rules, with Swift file:line) live in sensor_rig/stream/protocol.py.

Behaviour copied from the app (iPhUMI/ios_app/iPhUMI/Deployment/):
- v1: one python-socketio Client per stream port; "rgb" events carry base64 JPEG str (DeploymentViewController.swift:531,573),
  emitted only while that socket is connected (:530,572), fire-and-forget (SocketClient.swift:78-85).
- Every validate_period_s (app: 1 s Timer, DeploymentViewController.swift:192) each stream is handled like
  connectToEthernet() (:696-713): connected -> emit "validate" with ack and 1 s timeout; not connected -> connect.
  An empty/missing ack -> disconnect + connect (SocketClient.swift:93-98). Library auto-reconnect is OFF (the app's
  own rules are the only reconnect path).
- The validate request goes through the same link FIFO as the frames (one TCP connection per stream in reality), so a
  long stall or backlog makes the ack time out and the client reconnects, as the app would.
- websocket transport only (the app would start with long-polling and upgrade; message content is identical).
v2 (rig/sim): "hello" after every connect, "packet" with binary JPEGs and SENDER-clock stamps, and a "clock_ping"
handler answering {t1_ns, t2_ns} in the sender clock. The clock_ping reply is not delayed by the link (the receiver's
RTT filter has to cope with real asymmetry; add it if a test needs it).

Truth file: one JSON line per frame, {"seq", "stream", "t_true"} (t_true = capture time in THIS host's monotonic
clock, seconds). The wire stamps are skewed by the link's clock offset/drift, so the truth is the oracle for sync
tests.

Usage:
    c = StreamClient("127.0.0.1", {"main": 5555, "ultrawide": 5556}, "v1", link=LinkConfig.from_yaml("run/link/wifi_good.yaml"))
    c.connect(); c.send_v1("main", jpeg_bytes, t_true_capture=time.monotonic(), seq=0); ...; c.close()
"""

from __future__ import annotations

import json
import threading
import time

import socketio

from sensor_rig.stream import protocol as P

from .link import LinkConfig, LinkEmulator


class StreamClient:
    def __init__(self, host: str, ports: dict, protocol: str, link: LinkConfig | None = None,
                 truth_path: str | None = None, validate_period_s: float = 1.0,
                 sender: str = "franka_deploy.stream", v2_meta: dict | None = None):
        if protocol not in (P.V1, P.V2):
            raise ValueError(f"protocol must be 'v1' or 'v2', got {protocol!r}")
        self.host, self.ports, self.protocol = host, dict(ports), protocol
        self.validate_period_s = validate_period_s
        self.sender, self.v2_meta = sender, v2_meta or {}
        self._cfg = link
        self._link = LinkEmulator(link, self._deliver) if link is not None else None
        self._streams = list(self.ports) if protocol == P.V1 else ["main"]
        self._sio: dict[str, socketio.Client] = {}
        for s in self._streams:
            self._sio[s] = self._make_client(s)
        self._truth = open(truth_path, "w", buffering=1) if truth_path else None
        self._truth_lock = threading.Lock()
        self._lock = threading.Lock()
        self._acks: dict[int, threading.Event] = {}
        self._ack_id = 0
        self._stop = threading.Event()
        self._vthread: threading.Thread | None = None
        self.n_sent = 0
        self.n_delivered = 0
        self.n_skipped = 0
        self.reconnects = 0
        self.validate_ok = 0
        self.validate_timeouts = 0

    # ---- connection
    def _make_client(self, stream: str) -> socketio.Client:
        sio = socketio.Client(reconnection=False, logger=False, engineio_logger=False)
        if self.protocol == P.V2:
            @sio.event
            def connect():
                self._emit_hello(sio)

            @sio.on(P.EVENT_CLOCK_PING)
            def clock_ping(data):
                t1 = self.sender_clock_ns()
                return {"t1_ns": t1, "t2_ns": self.sender_clock_ns()}
        return sio

    def _url(self, stream: str) -> str:
        return f"http://{self.host}:{self.ports[stream]}"

    def _connect_one(self, stream: str, wait_timeout: float) -> None:
        self._sio[stream].connect(self._url(stream), transports=["websocket"], wait_timeout=wait_timeout)

    def connect(self, timeout_s: float = 10.0) -> None:
        end = time.monotonic() + timeout_s
        for s in self._streams:
            while True:
                try:
                    self._connect_one(s, wait_timeout=max(0.5, min(2.0, end - time.monotonic())))
                    break
                except Exception as e:
                    if time.monotonic() >= end:
                        raise ConnectionError(f"stream {s} at {self._url(s)} not reachable: {e}") from e
                    time.sleep(0.1)
        if self.protocol == P.V1 and self.validate_period_s > 0 and self._vthread is None:
            self._vthread = threading.Thread(target=self._validate_loop, name="stream-validate", daemon=True)
            self._vthread.start()

    def close(self) -> None:
        self._stop.set()
        if self._vthread is not None:
            self._vthread.join(timeout=3.0)
        if self._link is not None:
            self._link.close(drain=False)
        for sio in self._sio.values():
            try:
                sio.disconnect()
            except Exception:
                pass
        if self._truth:
            self._truth.close()
            self._truth = None

    def flush(self, timeout_s: float = 10.0) -> bool:
        """Wait until the link has delivered everything submitted (no-op without a link)."""
        return self._link.flush(timeout_s) if self._link is not None else True

    def connected(self, stream: str = "main") -> bool:
        return self._sio[stream].connected

    # ---- validate loop (v1)
    def _validate_loop(self):
        nxt = time.monotonic() + self.validate_period_s
        while not self._stop.wait(max(0.0, nxt - time.monotonic())):
            nxt += self.validate_period_s
            for s in self._streams:
                if self._stop.is_set():
                    return
                try:
                    if self._sio[s].connected:
                        self._validate(s)
                    else:
                        self._connect_one(s, wait_timeout=2.0)
                except Exception:
                    pass  # "Could not connect": stay down, the next tick retries (SocketClient.swift:42-47)

    def _validate(self, stream: str):
        ev = threading.Event()
        with self._lock:
            self._ack_id += 1
            aid = self._ack_id
            self._acks[aid] = ev
        t0 = time.monotonic()
        self._route(("validate", stream, aid), 16)
        ok = ev.wait(max(0.0, P.VALIDATE_TIMEOUT_S - (time.monotonic() - t0)))
        with self._lock:
            self._acks.pop(aid, None)
        if ok:
            self.validate_ok += 1
            return
        self.validate_timeouts += 1
        self.reconnects += 1
        sio = self._sio[stream]
        try:
            sio.disconnect()
        finally:
            self._connect_one(stream, wait_timeout=2.0)

    # ---- sending
    def _route(self, item, size: int):
        if self._link is not None:
            self._link.submit(item, size)
        else:
            self._deliver(item)

    def _deliver(self, item):
        kind, stream = item[0], item[1]
        sio = self._sio[stream]
        if not sio.connected:           # the app only emits while connected
            if kind != "validate":
                self.n_skipped += 1
            return
        if kind == "rgb":
            sio.emit(P.EVENT_RGB, item[2])
        elif kind == "packet":
            sio.emit(P.EVENT_PACKET, item[2])
        elif kind == "validate":
            aid = item[2]

            def cb(*ack):
                if ack:                 # non-empty ack list = success (SocketClient.swift:93)
                    with self._lock:
                        ev = self._acks.get(aid)
                    if ev is not None:
                        ev.set()

            sio.emit(P.EVENT_VALIDATE, P.VALIDATE_PAYLOAD, callback=cb)
        self.n_delivered += 1

    def _truth_line(self, seq, stream, t_true):
        if self._truth:
            with self._truth_lock:
                self._truth.write(json.dumps({"seq": seq, "stream": stream, "t_true": t_true}) + "\n")

    def send_v1(self, stream: str, jpeg: bytes, t_true_capture: float, seq: int) -> None:
        if self.protocol != P.V1:
            raise RuntimeError("send_v1 on a v2 client")
        payload = P.encode_v1(jpeg)
        self._truth_line(seq, stream, t_true_capture)
        self.n_sent += 1
        self._route(("rgb", stream, payload), len(payload))

    def send_v2(self, main_jpeg: bytes, uw_jpeg: bytes, t_main_true: float, t_uw_true: float, seq: int,
                w: int = 960, h: int = 720) -> None:
        if self.protocol != P.V2:
            raise RuntimeError("send_v2 on a v1 client")
        tm = self._to_sender_ns(t_main_true)
        tu = self._to_sender_ns(t_uw_true)
        pkt = P.make_packet(seq, main_jpeg, uw_jpeg, tm, tu, "sender_monotonic", w, h)
        self._truth_line(seq, "main", t_main_true)
        self._truth_line(seq, "ultrawide", t_uw_true)
        self.n_sent += 1
        self._route(("packet", "main", pkt), len(main_jpeg) + len(uw_jpeg))

    def _emit_hello(self, sio):
        m = self.v2_meta
        sio.emit(P.EVENT_HELLO, P.make_hello(self.sender, "v2", m.get("streams", {}), m.get("fps_hz", 0.0), m.get("K", {})))

    # ---- clocks
    def _to_sender_ns(self, t_mono_s: float) -> int:
        return self._link.to_sender_ns(t_mono_s) if self._link is not None else int(round(t_mono_s * 1e9))

    def sender_clock_ns(self) -> int:
        return self._link.sender_clock_ns() if self._link is not None else time.monotonic_ns()

    def stats(self) -> dict:
        ls = self._link.stats() if self._link is not None else {"delivered": self.n_delivered, "backlog_s": 0.0}
        t = None
        try:
            t = self._sio[self._streams[0]].transport()
        except Exception:
            pass
        return {"sent": self.n_sent, "delivered": ls["delivered"], "backlog_s": ls["backlog_s"],
                "reconnects": self.reconnects, "transport": t, "skipped_disconnected": self.n_skipped,
                "validate_ok": self.validate_ok, "validate_timeouts": self.validate_timeouts}
