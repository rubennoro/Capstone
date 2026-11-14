# moved from rnoro/deployment:deployment/process_frames/process_frames/clock_map.py @ 91b178b
# changes: names say local and remote instead of PC and Pi (ClockSample.t_remote_ns, offset_ns = remote - local,
# to_local_ns(t_remote_ns, at_local_ns), new to_remote_ns(t_local_ns)); the window, best fraction, minimum drift span,
# drift cap, freshness and (one-way) refit interval are constructor arguments; the fit is a public frozen ClockFit
# (snapshot()); stats() adds the sample span; NotSynced keeps its name. The math is unchanged.
"""The clock fit shared by frames (Pi -> laptop) and the manager (laptop <-> PC) (STRUCTURE.md 3.4).

Pure arithmetic plus one lock. `local` is the clock of this process (laptop CLOCK_MONOTONIC), `remote` the other
machine's (the Pi's, or the PC's). The fit is a line `offset(t) = remote - local` in local time.

ClockMap: from round trips (`add_sample`): the shortest round trips of the window, each assuming the remote stamp sits
at the midpoint of its round trip.

OneWayClockMap: from one-way stamps (`add_one_way(t_remote_send_ns, t_local_arrival_ns)`): each gives
`offset - latency` (latency >= 0, unknown); the least-delayed stamps of each 1 s slice are fitted, so the offset comes
out short by the minimum one-way latency (local times late by it, remote times early by it).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClockSample:
    t_send_ns: int         # local, just before the request was sent (one-way: the arrival time)
    t_remote_ns: int       # the remote clock from the reply (one-way: the remote send stamp)
    t_recv_ns: int         # local, just after the reply arrived (one-way: the arrival time)

    @property
    def rtt_ns(self) -> int:
        return self.t_recv_ns - self.t_send_ns

    @property
    def mid_ns(self) -> int:
        return (self.t_send_ns + self.t_recv_ns) // 2

    @property
    def offset_ns(self) -> int:
        """Remote minus local, assuming the remote side read its clock in the middle of the round trip."""
        return self.t_remote_ns - self.mid_ns


@dataclass(frozen=True)
class ClockStats:
    synced: bool                    # fresh: at least one sample, the newest at most max_sync_age_s old
    samples: int                    # samples in the window
    used: int                       # samples the fit uses
    offset_ns: int | None           # remote - local at "now"
    drift_ppm: float | None
    rtt_min_ms: float | None        # shortest round trip used (None for one-way)
    last_sync_age_s: float | None   # age of the newest sample
    span_s: float | None            # newest - oldest sample time in the window

    @property
    def newest_age_s(self) -> float | None:
        return self.last_sync_age_s


class NotSynced(RuntimeError):
    """No sample yet."""


@dataclass(frozen=True)
class ClockFit:
    """One frozen fit: offset(t) = ref_offset_ns + slope * (t - ref_local_ns), t in local time."""

    ref_local_ns: int
    ref_offset_ns: int
    slope: float          # ns of offset per ns of local time (drift_ppm * 1e-6)
    used: int
    rtt_min_ns: int

    @property
    def drift_ppm(self) -> float:
        return self.slope * 1e6

    def offset_ns(self, at_local_ns: int) -> int:
        """Estimated (remote - local) at that local time."""
        return self.ref_offset_ns + int(round(self.slope * (int(at_local_ns) - self.ref_local_ns)))

    def to_local_ns(self, t_remote_ns: int, at_local_ns: int) -> int:
        return int(t_remote_ns) - self.offset_ns(at_local_ns)

    def to_remote_ns(self, t_local_ns: int) -> int:
        return int(t_local_ns) + self.offset_ns(t_local_ns)


class ClockMap:
    """Remote -> local from round trips (the shortest best_fraction of the window); refits on every sample."""

    def __init__(self, window_s: float, best_fraction: float, drift_min_span_s: float, max_drift_ppm: float,
                 max_sync_age_s: float) -> None:
        self._window_ns = int(window_s * 1e9)
        self._best_fraction = best_fraction
        self._min_span_ns = int(drift_min_span_s * 1e9)
        self._max_drift_ppm = max_drift_ppm
        self._max_age_ns = int(max_sync_age_s * 1e9)
        self._samples: deque[ClockSample] = deque()
        self._fit: ClockFit | None = None
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

    def _compute(self, samples: list[ClockSample]) -> ClockFit:
        n_best = max(1, int(len(samples) * self._best_fraction))
        best = self._best(samples, n_best)
        mids = np.array([b.mid_ns for b in best], dtype=np.int64)
        offs = np.array([b.offset_ns for b in best], dtype=np.int64)
        ref_local = int(mids[0])
        ref_off = int(offs[0])
        x = (mids - ref_local).astype(np.float64)        # small numbers: no precision lost
        y = (offs - ref_off).astype(np.float64)
        mean_x, mean_y = float(x.mean()), float(y.mean())
        slope = 0.0
        if int(mids.max() - mids.min()) >= self._min_span_ns and len(best) >= 2:
            dx = x - mean_x
            slope = float((dx * (y - mean_y)).sum() / (dx * dx).sum())
            if abs(slope) * 1e6 > self._max_drift_ppm:
                slope = 0.0                               # noise: real crystals are under 100 ppm
        # the line through the mean point; with slope 0 this is the mean offset
        return ClockFit(ref_local_ns=ref_local + int(round(mean_x)), ref_offset_ns=ref_off + int(round(mean_y)),
                        slope=slope, used=len(best), rtt_min_ns=int(min(b.rtt_ns for b in best)))

    def is_fresh(self, now_ns: int) -> bool:
        """At least one sample, and the newest is at most max_sync_age_s old."""
        with self._lock:
            return bool(self._samples) and now_ns - self._samples[-1].t_recv_ns <= self._max_age_ns

    def snapshot(self) -> ClockFit:
        """The current fit, frozen (a refit later does not change it); NotSynced if there is no sample."""
        with self._lock:
            fit = self._fit
        if fit is None:
            raise NotSynced("no clock sample yet")
        return fit

    def offset_ns(self, at_local_ns: int) -> int:
        """Estimated (remote - local) at that local time; NotSynced if there is no sample."""
        return self.snapshot().offset_ns(at_local_ns)

    def to_local_ns(self, t_remote_ns: int, at_local_ns: int) -> int:
        return self.snapshot().to_local_ns(t_remote_ns, at_local_ns)

    def to_remote_ns(self, t_local_ns: int) -> int:
        return self.snapshot().to_remote_ns(t_local_ns)

    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    def sample_span_ns(self) -> int:
        """Newest minus oldest sample (local receive time) in the window; 0 with fewer than two."""
        with self._lock:
            if len(self._samples) < 2:
                return 0
            return self._samples[-1].t_recv_ns - self._samples[0].t_recv_ns

    def newest_ns(self) -> int | None:
        """Local receive time of the newest sample, or None."""
        with self._lock:
            return self._samples[-1].t_recv_ns if self._samples else None

    def stats(self, now_ns: int) -> ClockStats:
        with self._lock:
            fit = self._fit
            n = len(self._samples)
            newest = self._samples[-1].t_recv_ns if self._samples else None
            span = (self._samples[-1].t_recv_ns - self._samples[0].t_recv_ns) if self._samples else None
        if fit is None:
            return ClockStats(False, 0, 0, None, None, None, None, None)
        return ClockStats(synced=self.is_fresh(now_ns), samples=n, used=fit.used, offset_ns=fit.offset_ns(now_ns),
                          drift_ppm=fit.drift_ppm, rtt_min_ms=self._rtt_min_ms(fit),
                          last_sync_age_s=(now_ns - newest) / 1e9, span_s=span / 1e9)

    def _rtt_min_ms(self, fit: ClockFit) -> float | None:
        return fit.rtt_min_ns / 1e6


class OneWayClockMap(ClockMap):
    """Remote -> local from one-way stamps: ``add_one_way(t_remote_send_ns, t_local_arrival_ns)`` per message.

    Each stamp gives ``t_remote_send - t_local_arrival = offset - latency`` (latency >= 0, unknown). The least-delayed
    stamps (the largest values: the best_fraction of each 1 s slice of the window, so that they span the window and
    drift is still fitted) are fitted as ClockMap fits the shortest round trips, so the offset is ``true offset -
    minimum latency``: local times come out late by the minimum one-way latency. Refits at most every refit_s.
    """

    SLICE_NS = 1_000_000_000

    def __init__(self, window_s: float, best_fraction: float, drift_min_span_s: float, max_drift_ppm: float,
                 max_sync_age_s: float, refit_s: float = 0.25) -> None:
        super().__init__(window_s, best_fraction, drift_min_span_s, max_drift_ppm, max_sync_age_s)
        self._refit_ns = int(refit_s * 1e9)
        self._last_fit_ns: int | None = None

    def add_one_way(self, t_remote_send_ns: int, t_local_arrival_ns: int) -> bool:
        s = ClockSample(t_send_ns=int(t_local_arrival_ns), t_remote_ns=int(t_remote_send_ns),
                        t_recv_ns=int(t_local_arrival_ns))
        with self._lock:
            self._samples.append(s)
            while self._samples and self._samples[0].t_recv_ns < s.t_recv_ns - self._window_ns:
                self._samples.popleft()
            last = self._last_fit_ns
            if self._fit is None or last is None or s.t_recv_ns - last >= self._refit_ns:
                self._fit = self._compute(list(self._samples))
                self._last_fit_ns = s.t_recv_ns
        return True

    def _best(self, samples: list[ClockSample], n_best: int) -> list[ClockSample]:
        """The least-delayed stamps of each 1 s slice: the largest (t_remote_send - t_local_arrival)."""
        slices: dict[int, list[ClockSample]] = {}
        for x in samples:
            slices.setdefault(x.t_recv_ns // self.SLICE_NS, []).append(x)
        best: list[ClockSample] = []
        for key in sorted(slices):
            group = sorted(slices[key], key=lambda x: -x.offset_ns)
            best += group[:max(1, int(len(group) * self._best_fraction))]
        return best

    def _rtt_min_ms(self, fit: ClockFit) -> float | None:
        return None                                       # no round trips here
