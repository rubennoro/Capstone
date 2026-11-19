# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/link_stats.py @ 2ffe641
# changes: imports only
"""LinkStats: what the link to the PC does (STRUCTURE.md 6.2): C3 round trips (p50/p95/max over a window), the C2
rate, C2 seq gaps and the newest C2 arrival. For the status line, link.jsonl and pc_probe. One lock."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LinkSnapshot:
    rtt_n: int                       # C3 round trips in the window
    rtt_p50_ms: float | None
    rtt_p95_ms: float | None
    rtt_max_ms: float | None
    c2_rate_hz: float | None         # C2 messages per second over the window
    c2_received: int                 # since the start
    seq_gaps: int                    # since the start
    newest_arrival_ns: int | None    # laptop clock


class LinkStats:
    def __init__(self, window_s: float, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._window_ns = int(window_s * 1e9)
        self._clock = clock
        self._lock = threading.Lock()
        self._rtts: deque[tuple[int, int]] = deque()       # (laptop ns, rtt ns)
        self._arrivals: deque[int] = deque()
        self._received = 0
        self._gaps = 0
        self._last_seq: int | None = None
        self._newest: int | None = None

    def rtt(self, request: str, rtt_ns: int) -> None:
        now = self._clock()
        with self._lock:
            self._rtts.append((now, int(rtt_ns)))
            self._trim(self._rtts, now, key=lambda x: x[0])

    def c2(self, arrival_ns: int, seq: int) -> None:
        with self._lock:
            if self._last_seq is not None and seq > self._last_seq + 1:
                self._gaps += 1
            self._last_seq = int(seq)
            self._received += 1
            self._newest = int(arrival_ns)
            self._arrivals.append(int(arrival_ns))
            self._trim(self._arrivals, int(arrival_ns), key=lambda x: x)

    def _trim(self, q: deque, now: int, key) -> None:
        while q and key(q[0]) < now - self._window_ns:
            q.popleft()

    def newest_arrival_ns(self) -> int | None:
        with self._lock:
            return self._newest

    def snapshot(self, now_ns: int | None = None) -> LinkSnapshot:
        now = self._clock() if now_ns is None else now_ns
        with self._lock:
            self._trim(self._rtts, now, key=lambda x: x[0])
            self._trim(self._arrivals, now, key=lambda x: x)
            rtts = np.array([r for _, r in self._rtts], dtype=np.float64) / 1e6
            arrivals = list(self._arrivals)
            received, gaps, newest = self._received, self._gaps, self._newest
        rate = None
        if len(arrivals) >= 2 and arrivals[-1] > arrivals[0]:
            rate = (len(arrivals) - 1) / ((arrivals[-1] - arrivals[0]) / 1e9)
        if rtts.size:
            p50, p95, mx = (float(np.percentile(rtts, 50)), float(np.percentile(rtts, 95)), float(rtts.max()))
        else:
            p50 = p95 = mx = None
        return LinkSnapshot(rtt_n=int(rtts.size), rtt_p50_ms=p50, rtt_p95_ms=p95, rtt_max_ms=mx, c2_rate_hz=rate,
                            c2_received=received, seq_gaps=gaps, newest_arrival_ns=newest)
