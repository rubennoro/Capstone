"""Pure parts of the ``argus_sof`` backend (unit-tested): the helper's line protocol and the shared-memory ring.

Protocol and layout: see ``sensor_rig/native/argus_sof/argus_sof.cpp`` (header comment).
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"ARGSOF01"
HEADER = 32
SLOT_HDR = 16


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Ready:
    width: int
    height: int
    fmt: str
    slots: int
    slot_bytes: int
    shm_name: str


@dataclass(frozen=True)
class FrameMsg:
    seq: int
    sensor_ts_ns: int
    exposure_ns: int
    gain: float
    slot: int


def parse_line(line: str):
    """-> Ready | FrameMsg | ("error", msg) | None (blank / unknown lines are ignored). Raises ProtocolError."""
    s = line.strip()
    if not s:
        return None
    parts = s.split()
    try:
        if parts[0] == "READY":
            if len(parts) != 7:
                raise ProtocolError(f"bad READY line: {s!r}")
            w, h, fmt, slots, sb, name = int(parts[1]), int(parts[2]), parts[3], int(parts[4]), int(parts[5]), parts[6]
            if fmt != "NV12" or w <= 0 or h <= 0 or slots <= 0 or sb != SLOT_HDR + w * h * 3 // 2:
                raise ProtocolError(f"unsupported READY: {s!r}")
            return Ready(w, h, fmt, slots, sb, name)
        if parts[0] == "F":
            if len(parts) != 6:
                raise ProtocolError(f"bad frame line: {s!r}")
            return FrameMsg(int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4]), int(parts[5]))
        if parts[0] == "E":
            return ("error", s[2:].strip())
    except ValueError as e:
        if isinstance(e, ProtocolError):
            raise
        raise ProtocolError(f"cannot parse {s!r}: {e}") from e
    return None


class ShmRing:
    """Read side of the helper's ring (``/dev/shm/<name>``, mapped read-only; no Python resource tracker)."""

    def __init__(self, ready: Ready, shm_dir: str = "/dev/shm"):
        self.ready = ready
        path = Path(shm_dir) / ready.shm_name
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            need = HEADER + ready.slots * ready.slot_bytes
            if size < need:
                raise ProtocolError(f"{path}: {size} bytes < {need}")
            self.mm = mmap.mmap(fd, need, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        if self.mm[:8] != MAGIC:
            raise ProtocolError(f"{path}: bad magic {self.mm[:8]!r}")
        slots, sb, w, h = struct.unpack_from("<4I", self.mm, 8)
        if (slots, sb, w, h) != (ready.slots, ready.slot_bytes, ready.width, ready.height):
            raise ProtocolError(f"{path}: header {(slots, sb, w, h)} does not match READY")
        self.payload = ready.width * ready.height * 3 // 2

    def read(self, slot: int, seq: int) -> bytes | None:
        """Payload of ``slot`` if it still holds frame ``seq`` (seqlock check), else None (overwritten)."""
        if not 0 <= slot < self.ready.slots:
            raise ProtocolError(f"slot {slot} out of range")
        off = HEADER + slot * self.ready.slot_bytes
        end_seq = struct.unpack_from("<Q", self.mm, off + 8)[0]
        data = self.mm[off + SLOT_HDR: off + SLOT_HDR + self.payload]
        begin_seq = struct.unpack_from("<Q", self.mm, off)[0]
        if begin_seq != seq or end_seq != seq:
            return None
        return data

    def close(self) -> None:
        self.mm.close()


def nv12_to_bgr(buf: bytes, width: int, height: int) -> np.ndarray:
    import cv2

    yuv = np.frombuffer(buf, np.uint8).reshape(height * 3 // 2, width)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def helper_args(params: dict, shm_name: str, slots: int = 8) -> list[str]:
    """Command-line options for the helper from ``argus_logic.argus_params_from_camera`` output."""
    exp = params.get("exposure_time_range_ns") or (100000, 4000000)
    gain = params.get("gain_range") or (1.0, 8.0)
    mode = params.get("sensor_mode")
    return ["--sensor-id", str(int(params["sensor_id"])), "--mode", str(-1 if mode is None else int(mode)),
            "--width", str(int(params["width"])), "--height", str(int(params["height"])),
            "--fps", f"{float(params['fps']):g}", "--exposure-ns", f"{int(exp[0])},{int(exp[1])}",
            "--gain", f"{float(gain[0]):g},{float(gain[1]):g}", "--ae-lock", "1" if params.get("aelock") else "0",
            "--awb-lock", "1" if params.get("awblock") else "0", "--slots", str(int(slots)), "--shm-name", shm_name]
