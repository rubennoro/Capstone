"""The rig clock (brief D5).

Every timestamp on the Jetson is CLOCK_MONOTONIC in integer nanoseconds (the clock of ``time.monotonic_ns()``).
Argus sensor-of-frame stamps are probably CLOCK_MONOTONIC_RAW; ``RawToMono`` converts them using paired clock samples.
libcamera documents ``SensorTimestamp`` as CLOCK_BOOTTIME (on a Pi 5 it is in fact CLOCK_MONOTONIC);
``boot_minus_mono_ns`` gives the offset for the optional conversion (zero unless the system was suspended).
Wall-clock (UNIX) time is only needed for the iPhone-format time strings and is derived from one ``ClockSample`` taken
at record start (``realtime_ns - monotonic_ns``).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

_HAS_RAW = hasattr(time, "CLOCK_MONOTONIC_RAW")


def now_ns() -> int:
    """CLOCK_MONOTONIC in integer ns (the rig clock)."""
    return time.monotonic_ns()


def now_raw_ns() -> int:
    """CLOCK_MONOTONIC_RAW in integer ns (falls back to CLOCK_MONOTONIC where unavailable, e.g. macOS)."""
    if _HAS_RAW:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return time.monotonic_ns()


def boot_minus_mono_ns(n: int = 5) -> int:
    """CLOCK_BOOTTIME - CLOCK_MONOTONIC in ns (time spent suspended; 0 where BOOTTIME is unavailable).

    Both clocks tick at the same (NTP-slewed) rate, so one offset converts between them until the next suspend.
    """
    if not hasattr(time, "CLOCK_BOOTTIME"):
        return 0
    best = None
    for _ in range(max(1, n)):
        m1 = time.monotonic_ns()
        b = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        m2 = time.monotonic_ns()
        if best is None or m2 - m1 < best[0]:
            best = (m2 - m1, b - (m1 + m2) // 2)
    return int(best[1])


@dataclass(frozen=True)
class ClockSample:
    monotonic_ns: int
    monotonic_raw_ns: int
    realtime_ns: int
    uncertainty_ns: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClockSample":
        return cls(int(d["monotonic_ns"]), int(d["monotonic_raw_ns"]), int(d["realtime_ns"]), int(d["uncertainty_ns"]))


def sample_clocks(n: int = 15) -> ClockSample:
    """Read MONOTONIC, MONOTONIC_RAW and REALTIME as close together as possible.

    Each try reads mono, raw, realtime, mono2. The try with the smallest (mono2 - mono) wins; mono is reported at the
    midpoint of that bracket and the uncertainty is half its width.
    """
    best = None
    for _ in range(max(1, n)):
        m1 = time.monotonic_ns()
        raw = now_raw_ns()
        rt = time.time_ns()
        m2 = time.monotonic_ns()
        width = m2 - m1
        if best is None or width < best[0]:
            best = (width, m1, raw, rt, m2)
    width, m1, raw, rt, m2 = best
    return ClockSample(
        monotonic_ns=(m1 + m2) // 2, monotonic_raw_ns=raw, realtime_ns=rt, uncertainty_ns=width // 2
    )


class RawToMono:
    """Linear map CLOCK_MONOTONIC_RAW -> CLOCK_MONOTONIC.

    With one sample it is a pure offset. With two samples (e.g. episode start and stop) it also corrects the slope
    (NTP slews MONOTONIC but not MONOTONIC_RAW).
    """

    def __init__(self, a: ClockSample, b: ClockSample | None = None):
        self.raw0 = int(a.monotonic_raw_ns)
        self.mono0 = int(a.monotonic_ns)
        if b is None or b.monotonic_raw_ns == a.monotonic_raw_ns:
            self.slope = 1.0
        else:
            self.slope = (b.monotonic_ns - a.monotonic_ns) / (b.monotonic_raw_ns - a.monotonic_raw_ns)

    def __call__(self, raw_ns):
        if isinstance(raw_ns, np.ndarray):
            d = raw_ns.astype(np.int64) - self.raw0
            return self.mono0 + np.round(d * self.slope).astype(np.int64)
        return self.mono0 + int(round((int(raw_ns) - self.raw0) * self.slope))
