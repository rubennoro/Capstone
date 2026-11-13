"""CameraHealth: ok / stale / down per camera (STRUCTURE.md §5.11, §8). Written by the receiver; its own lock."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

from contracts import CameraHealthReport

log = logging.getLogger("process_frames.camera_health")

OK, STALE, DOWN = "ok", "stale", "down"


class CameraHealth:
    def __init__(self, camera: str, stale_s: float, down_s: float, fps_window_s: float,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.camera = camera
        self._stale_ns = int(stale_s * 1e9)
        self._down_ns = int(down_s * 1e9)
        self._window_ns = int(fps_window_s * 1e9)
        self._clock = clock
        self._start_ns = clock()
        self._times: deque[int] = deque()        # t_pc of buffered frames, newest last
        self._newest: int | None = None
        self._latched: str | None = None
        self._last_status: str | None = None
        self._lock = threading.Lock()
        self.dropped = 0

    def record_frame(self, t_pc_ns: int) -> None:
        """A usable frame was buffered."""
        with self._lock:
            self._newest = int(t_pc_ns)
            self._times.append(int(t_pc_ns))
            while self._times and self._times[0] < t_pc_ns - self._window_ns:
                self._times.popleft()

    def record_dropped(self, n: int) -> None:
        """n frames lost (a seq gap)."""
        with self._lock:
            self.dropped += int(n)

    def mark_down(self, reason: str) -> None:
        """Latched ``down`` for the rest of the run (a Pi stream restart)."""
        with self._lock:
            first = self._latched is None
            if first:
                self._latched = reason
        if first:
            log.warning("%s: latched down: %s", self.camera, reason)

    def _age_ns(self, now: int) -> int:
        ref = self._newest if self._newest is not None else self._start_ns
        return max(0, now - ref)

    def _status_locked(self, now: int) -> tuple[str, str]:
        if self._latched is not None:
            return DOWN, self._latched
        age = self._age_ns(now)
        if self._newest is None:                   # never ok before the first frame
            status = STALE if age <= self._down_ns else DOWN
            return status, f"no frame yet after {age / 1e6:.0f} ms"
        if age <= self._stale_ns:
            return OK, f"newest frame {age / 1e6:.0f} ms old"
        return (STALE if age <= self._down_ns else DOWN), f"newest frame {age / 1e6:.0f} ms old"

    def status(self, now_ns: int | None = None) -> str:
        """``ok`` | ``stale`` | ``down``. Each change is logged once."""
        now = self._clock() if now_ns is None else now_ns
        with self._lock:
            status, why = self._status_locked(now)
            old, self._last_status = self._last_status, status
        if old is not None and old != status:
            log.info("%s: %s -> %s (%s)", self.camera, old, status, why)
        elif old is None and status == OK:
            log.info("%s: stale -> ok (%s)", self.camera, why)
        return status

    def fps(self, now_ns: int | None = None) -> float:
        """Measured rate over the buffered frames of the last fps_window_s; 0 with fewer than 2."""
        now = self._clock() if now_ns is None else now_ns
        with self._lock:
            times = [t for t in self._times if t >= now - self._window_ns]
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / ((times[-1] - times[0]) / 1e9)

    def report(self, now_ns: int | None = None) -> CameraHealthReport:
        now = self._clock() if now_ns is None else now_ns
        status = self.status(now)
        with self._lock:
            age_ms = self._age_ns(now) / 1e6
            dropped = self.dropped
        return CameraHealthReport(status=status, last_frame_age_ms=float(age_ms), fps=float(self.fps(now)),
                                  dropped=int(dropped))
