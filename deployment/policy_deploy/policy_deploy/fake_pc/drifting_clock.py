"""DriftingClock: the fake PC's CLOCK_MONOTONIC (STRUCTURE.md 10.1, TOP 6.3).

t_pc = laptop_monotonic x (1 + drift_ppm x 1e-6) + offset_s. Used for C2 t_ns, hello t_pc_ns and the fake gate's
"now". shift() moves the offset (a test hook, to provoke a clock mismatch).
"""
from __future__ import annotations

import threading
import time
from typing import Callable


class DriftingClock:
    def __init__(self, offset_s: float, drift_ppm: float, laptop_clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._offset_ns = int(round(offset_s * 1e9))
        self._factor = 1.0 + drift_ppm * 1e-6
        self._laptop = laptop_clock
        self._lock = threading.Lock()

    def __call__(self) -> int:
        return self.at(self._laptop())

    def at(self, laptop_ns: int) -> int:
        """The fake PC time at a laptop CLOCK_MONOTONIC time."""
        with self._lock:
            offset = self._offset_ns
        return int(round(laptop_ns * self._factor)) + offset

    def to_laptop(self, t_pc_ns: int) -> int:
        """The inverse of at() (tests)."""
        with self._lock:
            offset = self._offset_ns
        return int(round((t_pc_ns - offset) / self._factor))

    def shift(self, seconds: float) -> None:
        with self._lock:
            self._offset_ns += int(round(seconds * 1e9))
