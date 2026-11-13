"""PacketBuffer and BufferedPacket: the last ~1 s of decoded packets (STRUCTURE.md §5.9)."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BufferedPacket:
    packet_seq: int
    t_pc_ns: int                        # the packet time: main mid-exposure on the PC clock (C1 reports it for both)
    t_pi_mid_ns: int                    # the same on the Pi clock (logs and tests)
    arrival_ns: int                     # PC time the packet was taken from the socket
    main_seq: int                       # the main frame's frame_seq
    main_t_pi_ns: int                   # the main frame's SensorTimestamp, as sent
    main_rgb: np.ndarray | None         # (H, W, 3) uint8, read-only; None: unusable (waiting for a keyframe)
    pair_seq: int                       # the header's pair_seq; -1: no pair
    pair_t_pi_ns: int | None            # the pair's own SensorTimestamp (logs, pair-gap statistic); None: not found
    ultrawide_rgb: np.ndarray | None    # the pair's image, read-only; None: no pair, or it is unusable or not received

    def __post_init__(self) -> None:
        for rgb in (self.main_rgb, self.ultrawide_rgb):
            if rgb is not None:
                rgb.flags.writeable = False          # immutable once buffered

    def image(self, camera: str) -> np.ndarray | None:
        """The usable image of that camera, or None."""
        if camera == "main":
            return self.main_rgb
        if camera == "ultrawide":
            return self.ultrawide_rgb
        raise ValueError(f"unknown camera {camera!r}")

    @property
    def pair_gap_ns(self) -> int | None:
        """Main capture minus pair capture (SensorTimestamps, as the pairing rule compares them); None: no pair."""
        return None if self.pair_t_pi_ns is None else self.main_t_pi_ns - self.pair_t_pi_ns


class PacketBuffer:
    def __init__(self, keep_s: float) -> None:
        self._keep_ns = int(keep_s * 1e9)
        self._packets: deque[BufferedPacket] = deque()
        self._lock = threading.Lock()

    def push(self, p: BufferedPacket) -> None:
        """Append; then drop packets older than keep_s before it. ValueError if its time is not after the newest."""
        with self._lock:
            if self._packets and p.t_pc_ns <= self._packets[-1].t_pc_ns:
                raise ValueError(f"packet time {p.t_pc_ns} is not after the newest {self._packets[-1].t_pc_ns}")
            self._packets.append(p)
            limit = p.t_pc_ns - self._keep_ns
            while self._packets[0].t_pc_ns < limit:
                self._packets.popleft()

    def snapshot(self) -> tuple[BufferedPacket, ...]:
        """Oldest first; a copy of the list of references, taken under the lock."""
        with self._lock:
            return tuple(self._packets)

    def newest(self) -> BufferedPacket | None:
        with self._lock:
            return self._packets[-1] if self._packets else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._packets)
