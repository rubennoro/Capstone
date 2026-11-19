# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/pc_clock.py @ 2ffe641
# changes: imports only
"""The laptop <-> PC clock map behind one interface (STRUCTURE.md 8.3; PLAN 5.6; TOP 6.3).

PcClock estimates `PC CLOCK_MONOTONIC - laptop CLOCK_MONOTONIC` as a line in laptop time, over
teleop_collect.contracts.clock_fit (local = laptop, remote = PC):

- RoundTripPcClock (`round_trip`, the default): a ClockMap fed by hello samples (t_pc_ns against the midpoint of the
  round trip, the shortest round trips of the window). add_state is ignored.
- OneWayPcClock (`one_way`, the backup): a OneWayClockMap fed by C2 (t_ns against the laptop arrival, only when t_ns
  advanced). Hellos are kept for the start-up agreement check, never as samples.

Guards: ready() needs min_samples_s of span, min_samples samples and a newest sample at most max_sample_age_s old
(guard 1); agreement_ms() is the start-up check against a hello (guard 2); every refit that moves "now" by more than
jump_log_ms is recorded (guard 3, read with take_jumps()). One lock; snapshot() is a frozen fit, so a refit between two
lookups cannot shift one row against the other.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable

from teleop_collect.contracts import ClockFit, ClockMap, ClockSample, NotSynced, OneWayClockMap
from teleop_collect.teleop.settings import ClockSettings

ESTIMATORS = ("round_trip", "one_way")
HELLOS_KEPT = 500


@dataclass(frozen=True)
class HelloSample:
    """One hello round trip: laptop stamps just before the request and just after the reply, and the PC's stamp."""

    t_send_ns: int
    t_pc_ns: int
    t_recv_ns: int

    @property
    def rtt_ns(self) -> int:
        return self.t_recv_ns - self.t_send_ns

    @property
    def mid_ns(self) -> int:
        return (self.t_send_ns + self.t_recv_ns) // 2


@dataclass(frozen=True)
class PcClockFit:
    """A frozen fit: laptop -> PC."""

    fit: ClockFit

    def to_pc_ns(self, t_ns: int) -> int:
        """A laptop time on the PC clock."""
        return self.fit.to_remote_ns(t_ns)

    def to_laptop_ns(self, t_pc_ns: int, at_ns: int) -> int:
        return self.fit.to_local_ns(t_pc_ns, at_ns)

    def offset_ns(self, at_ns: int) -> int:
        """PC - laptop at that laptop time."""
        return self.fit.offset_ns(at_ns)

    @property
    def drift_ppm(self) -> float:
        return self.fit.drift_ppm

    @property
    def samples(self) -> int:
        return self.fit.used


@dataclass(frozen=True)
class PcClockStats:
    estimator: str
    ready: bool
    samples: int
    used: int
    offset_ns: int | None            # PC - laptop at "now"
    drift_ppm: float | None
    rtt_min_ms: float | None         # round_trip: the shortest round trip used
    newest_age_s: float | None
    span_s: float | None


@dataclass(frozen=True)
class ClockJump:
    t_ns: int                        # laptop time of the refit
    jump_ms: float                   # how far the refit moved "now" (PC time)


class PcClock(ABC):
    NAME = ""

    def __init__(self, settings: ClockSettings, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._s = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._hellos: deque[HelloSample] = deque(maxlen=HELLOS_KEPT)
        self._jumps: list[ClockJump] = []
        self._last_t_pc_ns: int | None = None
        self._map = self._make_map()

    @abstractmethod
    def _make_map(self) -> ClockMap: ...

    # ---- samples

    def add_hello(self, s: HelloSample) -> None:
        """Every hello: both estimators keep it for the checks; round_trip also fits it."""
        with self._lock:
            self._hellos.append(s)
        self._on_hello(s)

    def add_state(self, t_pc_ns: int, arrival_ns: int) -> None:
        """A C2 message: one_way fits it when t_ns advanced (a repeated t_ns is never a sample)."""
        with self._lock:
            if self._last_t_pc_ns is not None and t_pc_ns <= self._last_t_pc_ns:
                return
            self._last_t_pc_ns = int(t_pc_ns)
        self._on_state(int(t_pc_ns), int(arrival_ns))

    def _on_hello(self, s: HelloSample) -> None:
        pass

    def _on_state(self, t_pc_ns: int, arrival_ns: int) -> None:
        pass

    def _refit(self, add: Callable[[], object]) -> None:
        """Run one add on the map and record a jump of "now" above jump_log_ms."""
        before = self._try_snapshot()
        add()
        after = self._try_snapshot()
        if before is not None and after is not None and after is not before:
            now = self._clock()
            jump_ms = (after.to_remote_ns(now) - before.to_remote_ns(now)) / 1e6
            if abs(jump_ms) > self._s.jump_log_ms:
                with self._lock:
                    self._jumps.append(ClockJump(now, jump_ms))

    def _try_snapshot(self) -> ClockFit | None:
        try:
            return self._map.snapshot()
        except NotSynced:
            return None

    # ---- reading

    def has_fit(self) -> bool:
        return self._try_snapshot() is not None

    def ready(self, now_ns: int) -> bool:
        """Guard 1: span >= min_samples_s, count >= min_samples, newest <= max_sample_age_s old."""
        newest = self._map.newest_ns()
        if newest is None or now_ns - newest > self._s.max_sample_age_s * 1e9:
            return False
        return (self._map.sample_count() >= self._s.min_samples
                and self._map.sample_span_ns() >= self._s.min_samples_s * 1e9)

    def not_ready_reason(self, now_ns: int) -> str:
        newest = self._map.newest_ns()
        if newest is None:
            return "no clock sample yet"
        age = (now_ns - newest) / 1e9
        if age > self._s.max_sample_age_s:
            return f"newest clock sample {age:.1f} s old"
        return (f"{self._map.sample_count()} clock samples over {self._map.sample_span_ns() / 1e9:.2f} s, "
                f"need {self._s.min_samples} over {self._s.min_samples_s:g} s")

    def snapshot(self) -> PcClockFit:
        """The current fit, frozen. NotSynced if there is no sample yet."""
        return PcClockFit(self._map.snapshot())

    def now_pc_ns(self) -> int:
        return self.snapshot().to_pc_ns(self._clock())

    def agreement_ms(self, s: HelloSample) -> float:
        """t_pc_ns minus the map's PC time at the hello's midpoint, ms (guard 2)."""
        return (s.t_pc_ns - self.snapshot().to_pc_ns(s.mid_ns)) / 1e6

    def hellos(self) -> list[HelloSample]:
        with self._lock:
            return list(self._hellos)

    def take_jumps(self) -> list[ClockJump]:
        with self._lock:
            out, self._jumps = self._jumps, []
        return out

    def stats(self, now_ns: int) -> PcClockStats:
        st = self._map.stats(now_ns)
        return PcClockStats(estimator=self.NAME, ready=self.ready(now_ns), samples=st.samples, used=st.used,
                            offset_ns=st.offset_ns, drift_ppm=st.drift_ppm, rtt_min_ms=st.rtt_min_ms,
                            newest_age_s=st.newest_age_s, span_s=st.span_s)


class RoundTripPcClock(PcClock):
    NAME = "round_trip"

    def _make_map(self) -> ClockMap:
        s = self._s
        return ClockMap(window_s=s.round_trip_window_s, best_fraction=s.round_trip_best_fraction,
                        drift_min_span_s=s.drift_min_span_s, max_drift_ppm=s.max_drift_ppm,
                        max_sync_age_s=s.max_sample_age_s)

    def _on_hello(self, s: HelloSample) -> None:
        self._refit(lambda: self._map.add_sample(ClockSample(s.t_send_ns, s.t_pc_ns, s.t_recv_ns)))


class OneWayPcClock(PcClock):
    NAME = "one_way"

    def _make_map(self) -> ClockMap:
        s = self._s
        return OneWayClockMap(window_s=s.one_way_window_s, best_fraction=s.one_way_best_fraction,
                              drift_min_span_s=s.drift_min_span_s, max_drift_ppm=s.max_drift_ppm,
                              max_sync_age_s=s.max_sample_age_s)

    def _on_state(self, t_pc_ns: int, arrival_ns: int) -> None:
        m = self._map
        assert isinstance(m, OneWayClockMap)
        self._refit(lambda: m.add_one_way(t_pc_ns, arrival_ns))


def make_pc_clock(estimator: str, settings: ClockSettings, clock: Callable[[], int] = time.monotonic_ns) -> PcClock:
    """link.yaml clock.estimator -> the PcClock."""
    if estimator == "round_trip":
        return RoundTripPcClock(settings, clock)
    if estimator == "one_way":
        return OneWayPcClock(settings, clock)
    raise ValueError(f"clock estimator must be one of {', '.join(ESTIMATORS)}, got {estimator!r}")
