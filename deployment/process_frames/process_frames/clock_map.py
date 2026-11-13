# adapted from rnoro/main-sim:controlSW/franka_deploy/franka_deploy/peripherals/clock_map.py @ 935c0b3
# kept: the sender_stamp_ping idea (shortest-round-trip exchanges of a window, max-drift fallback); differs: plain
# least-squares line (not Theil-Sen), integer ns, one Pi stamp per reply, minimum span before drift, freshness, lock
"""ClockMap: Pi time -> PC time from the time_sync samples (STRUCTURE.md §5.6, §7). Pure arithmetic plus a lock.

OneWayClockMap: the same from one-way stamps (stream pi_v1: the Pi's t_send_ns against the PC arrival time), for a
Pi that answers no time_sync requests (STRUCTURE.md §5.6a)."""

from __future__ import annotations

import dataclasses
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClockSample:
    t_send_ns: int         # PC monotonic, just before the request was sent
    t_pi_ns: int           # the Pi's CLOCK_BOOTTIME from the reply
    t_recv_ns: int         # PC monotonic, just after the reply arrived

    @property
    def rtt_ns(self) -> int:
        return self.t_recv_ns - self.t_send_ns

    @property
    def mid_ns(self) -> int:
        return (self.t_send_ns + self.t_recv_ns) // 2

    @property
    def offset_ns(self) -> int:
        """Pi minus PC, assuming the Pi read its clock in the middle of the round trip."""
        return self.t_pi_ns - self.mid_ns


@dataclass(frozen=True)
class ClockStats:
    synced: bool
    samples: int
    used: int
    offset_ns: int | None
    drift_ppm: float | None
    rtt_min_ms: float | None
    last_sync_age_s: float | None


class NotSynced(RuntimeError):
    """No time_sync sample yet."""


@dataclass(frozen=True)
class _Fit:
    ref_pc_ns: int        # the line is offset(t) = ref_offset_ns + slope * (t - ref_pc_ns)
    ref_offset_ns: int
    slope: float          # ns of offset per ns of PC time (drift_ppm * 1e-6)
    used: int
    rtt_min_ns: int


class ClockMap:
    def __init__(self, window_s: float, best_fraction: float, drift_min_span_s: float, max_drift_ppm: float,
                 max_sync_age_s: float) -> None:
        self._window_ns = int(window_s * 1e9)
        self._best_fraction = best_fraction
        self._min_span_ns = int(drift_min_span_s * 1e9)
        self._max_drift_ppm = max_drift_ppm
        self._max_age_ns = int(max_sync_age_s * 1e9)
        self._samples: deque[ClockSample] = deque()
        self._fit: _Fit | None = None
        self._lock = threading.Lock()

    def add_sample(self, s: ClockSample) -> bool:
        """Add one exchange; False (ignored) for an impossible sample (t_recv < t_send)."""
        if s.t_recv_ns < s.t_send_ns:
            return False
        with self._lock:
            self._samples.append(s)
            while self._samples and self._samples[0].t_recv_ns < s.t_recv_ns - self._window_ns:
                self._samples.popleft()
            self._fit = self._compute(list(self._samples))
        return True

    def _best(self, samples: list[ClockSample], n_best: int) -> list[ClockSample]:
        """The samples the fit uses: the shortest round trips."""
        return sorted(samples, key=lambda x: x.rtt_ns)[:n_best]

    def _compute(self, samples: list[ClockSample]) -> _Fit:
        n_best = max(1, int(len(samples) * self._best_fraction))
        best = self._best(samples, n_best)
        mids = np.array([b.mid_ns for b in best], dtype=np.int64)
        offs = np.array([b.offset_ns for b in best], dtype=np.int64)
        ref_pc = int(mids[0])
        ref_off = int(offs[0])
        x = (mids - ref_pc).astype(np.float64)           # small numbers: no precision lost
        y = (offs - ref_off).astype(np.float64)
        mean_x, mean_y = float(x.mean()), float(y.mean())
        slope = 0.0
        if int(mids.max() - mids.min()) >= self._min_span_ns and len(best) >= 2:
            dx = x - mean_x
            slope = float((dx * (y - mean_y)).sum() / (dx * dx).sum())
            if abs(slope) * 1e6 > self._max_drift_ppm:
                slope = 0.0                               # noise: real crystals are under 100 ppm
        # the line through the mean point; with slope 0 this is the mean offset
        return _Fit(ref_pc_ns=ref_pc + int(round(mean_x)), ref_offset_ns=ref_off + int(round(mean_y)),
                    slope=slope, used=len(best), rtt_min_ns=int(best[0].rtt_ns))

    def is_fresh(self, now_ns: int) -> bool:
        """At least one sample, and the newest is at most max_sync_age_s old."""
        with self._lock:
            return bool(self._samples) and now_ns - self._samples[-1].t_recv_ns <= self._max_age_ns

    def offset_ns(self, at_pc_ns: int) -> int:
        """Estimated (Pi - PC) at that PC time; NotSynced if there is no sample."""
        with self._lock:
            fit = self._fit
        if fit is None:
            raise NotSynced("no time_sync sample yet")
        return fit.ref_offset_ns + int(round(fit.slope * (at_pc_ns - fit.ref_pc_ns)))

    def to_pc_ns(self, t_pi_ns: int, at_pc_ns: int) -> int:
        return int(t_pi_ns) - self.offset_ns(at_pc_ns)

    def stats(self, now_ns: int) -> ClockStats:
        with self._lock:
            fit = self._fit
            n = len(self._samples)
            newest = self._samples[-1].t_recv_ns if self._samples else None
        if fit is None:
            return ClockStats(False, 0, 0, None, None, None, None)
        return ClockStats(synced=self.is_fresh(now_ns), samples=n, used=fit.used, offset_ns=self.offset_ns(now_ns),
                          drift_ppm=fit.slope * 1e6, rtt_min_ms=fit.rtt_min_ns / 1e6,
                          last_sync_age_s=(now_ns - newest) / 1e9)


class OneWayClockMap(ClockMap):
    """Pi -> PC from one-way stamps: ``add_one_way(t_pi_send_ns, t_pc_arrival_ns)`` per received packet.

    Each stamp gives ``t_pi_send - t_pc_arrival = offset - latency`` (latency >= 0, unknown). The least-delayed
    stamps (the largest values: the best_fraction of each 1 s slice of the window, so that they span the window and
    drift is still fitted) are fitted as ClockMap fits the shortest round trips, so the offset is ``true offset -
    minimum latency``: PC times come out late by the minimum one-way latency (a few ms on the Wi-Fi). Fresh while
    packets keep arriving (max_sync_age_s).
    """

    SLICE_NS = 1_000_000_000
    REFIT_NS = 250_000_000        # refit at most this often (a fit of a full window costs ~2 ms on the Jetson)

    def add_one_way(self, t_pi_send_ns: int, t_pc_arrival_ns: int) -> bool:
        s = ClockSample(t_send_ns=int(t_pc_arrival_ns), t_pi_ns=int(t_pi_send_ns), t_recv_ns=int(t_pc_arrival_ns))
        with self._lock:
            self._samples.append(s)
            while self._samples and self._samples[0].t_recv_ns < s.t_recv_ns - self._window_ns:
                self._samples.popleft()
            last = getattr(self, "_last_fit_ns", None)
            if self._fit is None or last is None or s.t_recv_ns - last >= self.REFIT_NS:
                self._fit = self._compute(list(self._samples))
                self._last_fit_ns = s.t_recv_ns
        return True

    def _best(self, samples: list[ClockSample], n_best: int) -> list[ClockSample]:
        """The least-delayed stamps of each 1 s slice: the largest (t_pi_send - t_pc_arrival), oldest slice first."""
        slices: dict[int, list[ClockSample]] = {}
        for x in samples:
            slices.setdefault(x.t_recv_ns // self.SLICE_NS, []).append(x)
        best: list[ClockSample] = []
        for key in sorted(slices):
            group = sorted(slices[key], key=lambda x: -x.offset_ns)
            best += group[:max(1, int(len(group) * self._best_fraction))]
        return best

    def stats(self, now_ns: int) -> ClockStats:
        st = super().stats(now_ns)
        return dataclasses.replace(st, rtt_min_ms=None)          # no round trips here
