"""Wire protocol of the wireless sensor path: v1 (the iPhUMI iPhone app, verbatim) and v2 (rig and sim senders).

Why: the policy host receives camera frames over Socket.IO. The iPhone app's behaviour is fixed (we cannot change the
app), so v1 is documented here from its Swift source and reproduced exactly by the sim/replay sender. v2 is ours: it
adds sequence numbers and sender-clock stamps so the receiver can map frames to host time. This module is pure stdlib
(the rig must not depend on franka_deploy or numpy) and only builds/checks messages; sockets live in
franka_deploy/stream/client.py.

Usage:
    from sensor_rig.stream import protocol as P
    s = P.encode_v1(jpeg_bytes); jpeg = P.decode_v1(s)
    pkt = P.make_packet(seq=3, main_jpeg=a, uw_jpeg=b, t_main_ns=1, t_uw_ns=2, clock="sender_monotonic", w=960, h=720)
    assert P.check_packet(pkt) == []

v1 facts (iPhone app, all paths under iPhUMI/ios_app/iPhUMI/Deployment/; verified against the Swift code)
- One Socket.IO server per stream: main (wide) 5555, ultrawide 5556, depth 5557 (only if depth enabled)
  [DeploymentViewController.swift:105-108]. URL is http://<hostIP>:<port> [SocketClient.swift:23]. Config:
  `.log(false), .compress` only [SocketClient.swift:23]: there is NO forceWebsockets, so the real app starts on HTTP
  long-polling and upgrades; our client is websocket-only by design (the receiver's `require_websocket`), a documented
  simplification that does not change message content.
- Frames: event "rgb", payload a single Swift String = base64 of a JPEG at quality 0.9, no line breaks
  [DeploymentViewController.swift:531 (ultrawide), :573 (main); RgbSocketPacket.toEncodedString :45-56;
  SocketUtil.swift:30-40]. Emitted with fire-and-forget `emit` (no ack) [SocketClient.swift:78-85], only when
  `isConnected()` [DeploymentViewController.swift:530, 572]. Frame size = capture size x 0.5 (cameraResizeScale
  :134), i.e. 960x720 in High mode (1920x1440 capture, :122-123). No stamps, no sequence numbers.
- Both cameras run at the same `cameraFPS` (30 or 60, :137, :303-307) and the same capture resolution
  [:293-299, :390, :446]; each sample buffer is sent as it arrives (:512, :555). So the app sends the ultrawide at
  camera fps with REAL 960x720 frames (not the 10 Hz / 640x480 ARKit video used in training).
- "depth" event: payload float32 base64, ignored by the host here (iphone_camera.py:289-293).
- Validate: a 1 s Timer [DeploymentViewController.swift:192] calls connectToEthernet() [:778-781]; for each socket,
  if connected -> `validateConnection()` else -> `connect()` [:696-713]. validateConnection does
  `emitWithAck("validate", "ping").timingOut(after: 1)` [SocketClient.swift:88-103]. The server handler returns a value
  (reference: "server got it!", iphone_camera.py:295-297); any non-empty ack list is success. An EMPTY ack list (which is
  what a timeout produces) -> disconnect() + connect() on the same host/port [SocketClient.swift:93-98].
- Other reconnect triggers: Socket.IO "error" client event with a String message: "Could not connect to the server."
  -> disconnect only (the 1 s timer reconnects); any other String -> disconnect() + connect() immediately
  [SocketClient.swift:42-54]. Connect/disconnect events only flip `ready`/`socketOpened` [:30-40].
- The sender never drops or throttles frames on its own: backlog sits in the socket (TCP) buffer.

v2 (rig and sim senders; main port only)
- "hello" {v:2, sender, protocol, streams, fps_hz, K} once after connect.
- "packet" {v:2, seq, main_jpeg: bytes, uw_jpeg: bytes, t_main_ns, t_uw_ns, clock, w, h} (binary attachments);
  stamps are in the SENDER's clock; `clock` is "sender_monotonic" or "sender_realtime".
- server -> client call "clock_ping" {t0_ns} -> {t1_ns, t2_ns}, both in the client's clock (t1 receive, t2 reply).
"""

from __future__ import annotations

import base64

EVENT_RGB = "rgb"
EVENT_DEPTH = "depth"
EVENT_VALIDATE = "validate"
EVENT_HELLO = "hello"
EVENT_PACKET = "packet"
EVENT_CLOCK_PING = "clock_ping"

V1 = "v1"
V2 = "v2"
VALIDATE_PAYLOAD = "ping"          # SocketClient.swift:89
VALIDATE_ACK = "server got it!"    # iphone_camera.py reference server
VALIDATE_TIMEOUT_S = 1.0           # SocketClient.swift:88 default
VALIDATE_PERIOD_S = 1.0            # DeploymentViewController.swift:192
V1_PORTS = {"main": 5555, "ultrawide": 5556}
CLOCKS = ("sender_monotonic", "sender_realtime")


def encode_v1(jpeg_bytes: bytes) -> str:
    """JPEG bytes -> the app's "rgb" payload (base64 str, no newlines)."""
    return base64.b64encode(bytes(jpeg_bytes)).decode("ascii")


def decode_v1(s: str) -> bytes:
    if not isinstance(s, str):
        raise TypeError(f"v1 rgb payload must be str, got {type(s).__name__}")
    return base64.b64decode(s, validate=True)


def make_hello(sender: str, protocol: str, streams: dict, fps_hz: float, K: dict) -> dict:
    return {"v": 2, "sender": sender, "protocol": protocol, "streams": streams, "fps_hz": fps_hz, "K": K}


def make_packet(seq: int, main_jpeg: bytes, uw_jpeg: bytes, t_main_ns: int, t_uw_ns: int, clock: str, w: int, h: int) -> dict:
    return {"v": 2, "seq": int(seq), "main_jpeg": bytes(main_jpeg), "uw_jpeg": bytes(uw_jpeg),
            "t_main_ns": int(t_main_ns), "t_uw_ns": int(t_uw_ns), "clock": clock, "w": int(w), "h": int(h)}


_PACKET_TYPES = {"v": int, "seq": int, "main_jpeg": (bytes, bytearray), "uw_jpeg": (bytes, bytearray),
                 "t_main_ns": int, "t_uw_ns": int, "clock": str, "w": int, "h": int}
_HELLO_KEYS = {"v", "sender", "protocol", "streams", "fps_hz", "K"}


def check_packet(d) -> list[str]:
    """Problems with a v2 packet (empty list = valid): missing/extra fields, wrong types, bad values."""
    if not isinstance(d, dict):
        return [f"packet must be a dict, got {type(d).__name__}"]
    errs = [f"missing field: {k}" for k in _PACKET_TYPES if k not in d]
    errs += [f"unexpected field: {k}" for k in d if k not in _PACKET_TYPES]
    for k, t in _PACKET_TYPES.items():
        if k in d:
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, t):
                errs.append(f"field {k}: expected {t}, got {type(v).__name__}")
    if not errs:
        if d["v"] != 2:
            errs.append(f"v must be 2, got {d['v']}")
        if d["clock"] not in CLOCKS:
            errs.append(f"clock must be one of {CLOCKS}, got {d['clock']!r}")
        if d["w"] <= 0 or d["h"] <= 0:
            errs.append("w and h must be positive")
        if d["seq"] < 0:
            errs.append("seq must be >= 0")
    return errs


def check_hello(d) -> list[str]:
    if not isinstance(d, dict):
        return [f"hello must be a dict, got {type(d).__name__}"]
    errs = [f"missing field: {k}" for k in sorted(_HELLO_KEYS - set(d))]
    errs += [f"unexpected field: {k}" for k in sorted(set(d) - _HELLO_KEYS)]
    if d.get("v") != 2:
        errs.append(f"v must be 2, got {d.get('v')}")
    return errs
