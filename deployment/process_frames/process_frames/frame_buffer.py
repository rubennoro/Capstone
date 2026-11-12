"""FrameBuffer and BufferedFrame: the last ~1 s of frames of one camera (STRUCTURE.md §5.8)."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BufferedFrame:
    camera: str
    seq: int
    t_pc_ns: int             # PC mid-exposure time (what C1 reports)
    t_pi_mid_ns: int         # Pi mid-exposure time (kept for logs and tests)
    arrival_ns: int          # PC time the C0 message was taken from the socket
    rgb: np.ndarray          # (H, W, 3) uint8 at the rig resolution, read-only

    def __post_init__(self) -> None:
        self.rgb.flags.writeable = False          # immutable once buffered


class FrameBuffer:
    def __init__(self, camera: str, keep_s: float) -> None:
        self.camera = camera
        self._keep_ns = int(keep_s * 1e9)
        self._frames: deque[BufferedFrame] = deque()
        self._lock = threading.Lock()

    def push(self, f: BufferedFrame) -> None:
        """Append; then drop frames older than keep_s before it. ValueError if its time is not after the newest."""
        with self._lock:
            if self._frames and f.t_pc_ns <= self._frames[-1].t_pc_ns:
                raise ValueError(f"{self.camera}: frame time {f.t_pc_ns} is not after the newest "
                                 f"{self._frames[-1].t_pc_ns}")
            self._frames.append(f)
            limit = f.t_pc_ns - self._keep_ns
            while self._frames[0].t_pc_ns < limit:
                self._frames.popleft()

    def snapshot(self) -> tuple[BufferedFrame, ...]:
        """Oldest first; a copy of the list of references, taken under the lock."""
        with self._lock:
            return tuple(self._frames)

    def newest(self) -> BufferedFrame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
