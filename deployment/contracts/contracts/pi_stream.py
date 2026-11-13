"""C0 as the Pi sends it: the ``pi_stream/v1`` wire format (PI_PAYLOAD.md section 0). Pi 5 -> process_frames.

One ZeroMQ multipart message of three parts per main frame, published by the Pi (PUB) to every connected receiver:

    part 0  header, JSON object (below)
    part 1  main camera (imx519) frame, JPEG
    part 2  ultrawide camera (imx219) frame, JPEG

Header keys, all required, no others: ``version`` (1), ``packet_no``, ``t_send_ns``, ``clock_offset_ns``, ``main``
and ``ultrawide`` (each ``{frame_no, t_sensor_ns, exposure_us, t_mid_ns}``), ``skew_ns``, ``uw_reused``. Integers are
JSON integers (never booleans), ``uw_reused`` a boolean. Nothing is sent back to the Pi.

process_frames converts each packet to the internal C0 form (``contracts.c0_frames.Packet``); this module only parses
and checks the wire format, and builds it (for the stand-in Pi).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from contracts.errors import ContractError

SCHEMA = "pi_stream/v1"
VERSION = 1
N_PARTS = 3
MAIN_SENSOR = "imx519"
ULTRAWIDE_SENSOR = "imx219"


def _fail(field: str, problem: str, detail: str) -> ContractError:
    return ContractError(SCHEMA, field, problem, detail)


def _keys(where: str, d: dict, expected: Sequence[str]) -> None:
    for k in expected:
        if k not in d:
            raise _fail(f"{where}{k}", "missing", "required key")
    for k in d:
        if k not in expected:
            raise _fail(f"{where}{k}", "unknown_field", f"not a {SCHEMA} key (expected {list(expected)})")


def _int(where: str, d: dict, key: str, min_value: int | None = None) -> int:
    v = d[key]
    if not isinstance(v, int) or isinstance(v, bool):
        raise _fail(f"{where}{key}", "type", f"expected an integer, got {type(v).__name__} {v!r}")
    if min_value is not None and v < min_value:
        raise _fail(f"{where}{key}", "value", f"expected >= {min_value}, got {v}")
    return v


@dataclass(frozen=True)
class PiFrameStamp:
    """The per-camera map of the header: the Pi's own numbers for one frame."""

    frame_no: int        # counts this camera's frames on the Pi
    t_sensor_ns: int     # libcamera SensorTimestamp (end of the first row's exposure), Pi CLOCK_BOOTTIME, ns
    exposure_us: int     # libcamera ExposureTime, us
    t_mid_ns: int        # mid-exposure: t_sensor_ns - exposure_us * 1000 / 2

    KEYS = ("frame_no", "t_sensor_ns", "exposure_us", "t_mid_ns")

    @classmethod
    def from_dict(cls, where: str, d: Any) -> "PiFrameStamp":
        if not isinstance(d, dict):
            raise _fail(where.rstrip("."), "not_a_map", f"expected an object, got {type(d).__name__}")
        _keys(where, d, cls.KEYS)
        return cls(frame_no=_int(where, d, "frame_no", 0), t_sensor_ns=_int(where, d, "t_sensor_ns"),
                   exposure_us=_int(where, d, "exposure_us", 0), t_mid_ns=_int(where, d, "t_mid_ns"))


@dataclass(frozen=True)
class PiStreamHeader:
    version: int
    packet_no: int           # counts packets (main frames) since the Pi sender started
    t_send_ns: int           # the Pi's clock (as t_sensor_ns) just before sending
    clock_offset_ns: int     # as sent by the Pi; not used by process_frames
    main: PiFrameStamp
    ultrawide: PiFrameStamp
    skew_ns: int             # as sent by the Pi (main.t_mid_ns - ultrawide.t_mid_ns); not used by process_frames
    uw_reused: bool          # part 2 repeats the ultrawide frame of an earlier packet (same frame_no)

    KEYS = ("version", "packet_no", "t_send_ns", "clock_offset_ns", "main", "ultrawide", "skew_ns", "uw_reused")

    @classmethod
    def from_json(cls, raw: bytes) -> "PiStreamHeader":
        try:
            d = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise _fail("parts[0]", "value", f"header is not JSON ({e}); first bytes {bytes(raw[:32])!r}") from None
        if not isinstance(d, dict):
            raise _fail("parts[0]", "not_a_map", f"header is JSON {type(d).__name__}, expected an object")
        _keys("", d, cls.KEYS)
        version = _int("", d, "version")
        if version != VERSION:
            raise _fail("version", "schema_version", f"got {version}, this receiver reads version {VERSION}")
        if not isinstance(d["uw_reused"], bool):
            raise _fail("uw_reused", "type", f"expected a boolean, got {type(d['uw_reused']).__name__}")
        return cls(version=version, packet_no=_int("", d, "packet_no", 0), t_send_ns=_int("", d, "t_send_ns"),
                   clock_offset_ns=_int("", d, "clock_offset_ns"),
                   main=PiFrameStamp.from_dict("main.", d["main"]),
                   ultrawide=PiFrameStamp.from_dict("ultrawide.", d["ultrawide"]),
                   skew_ns=_int("", d, "skew_ns"), uw_reused=d["uw_reused"])

    def to_json(self) -> bytes:
        return json.dumps({k: asdict(v) if isinstance(v, PiFrameStamp) else v
                           for k, v in ((k, getattr(self, k)) for k in self.KEYS)}).encode()


@dataclass(frozen=True)
class PiStreamPacket:
    """One received (or built) message. An object that exists is valid."""

    header: PiStreamHeader
    main_jpeg: bytes
    ultrawide_jpeg: bytes

    @classmethod
    def from_parts(cls, parts: Sequence[bytes]) -> "PiStreamPacket":
        """Parse and check one multipart message. ContractError (schema pi_stream/v1) if it breaks the format."""
        parts = list(parts)
        if len(parts) != N_PARTS:
            raise _fail("parts", "value", f"{len(parts)} parts, expected {N_PARTS} (header, main JPEG, ultrawide JPEG)")
        header = PiStreamHeader.from_json(parts[0])
        for i, name in ((1, "main"), (2, "ultrawide")):
            if not parts[i]:
                raise _fail(f"parts[{i}]", "value", f"empty {name} frame")
        return cls(header, bytes(parts[1]), bytes(parts[2]))

    def to_parts(self) -> list[bytes]:
        return [self.header.to_json(), self.main_jpeg, self.ultrawide_jpeg]


def jpeg_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a JPEG's frame header (SOFn marker), without decoding. ContractError if not found."""
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise _fail("jpeg", "value", "not a JPEG (no SOI marker)")
    i = 2
    while i + 3 < n:
        if data[i] != 0xFF:
            raise _fail("jpeg", "value", f"bad JPEG marker at byte {i}")
        marker = data[i + 1]
        if marker == 0xFF:                           # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:      # no length field
            i += 2
            continue
        length = (data[i + 2] << 8) | data[i + 3]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > n:
                break
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return width, height
        if marker == 0xDA:                           # start of scan before any frame header
            break
        i += 2 + length
    raise _fail("jpeg", "value", "no JPEG frame header (SOF) found")
