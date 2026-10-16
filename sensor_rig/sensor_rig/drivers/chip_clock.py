"""Map the IMU chip's timestamp clock (microseconds) to host CLOCK_MONOTONIC (ns). Pure; unit-tested.

The driver feeds one point per FIFO read: ``(chip_us of the newest packet counted, host_ns right after reading
FIFO_COUNT)``. That packet was latched before the count was read, so ``host >= a + b * chip`` always holds (up to the
true clock relation), and the excess is a positive, random delay (scheduling, SPI latency, time since the sample).
The model ``host = a + b * chip`` is therefore fitted to the **lower envelope** of the points in a sliding window:

    maximise  a + b * mean(chip)   subject to   a + b * chip_i <= host_i  for all i

(equivalently: minimise the summed vertical gap to the points while staying below all of them). The optimum is the
lower-convex-hull edge spanning ``mean(chip)``, found exactly in O(n log n). ``b`` (ns per chip us, nominal 1000) is
clamped to ``1000 * (1 +- max_ppm * 1e-6)``. The chip's internal clock can be off by up to +-1 % with the PLL (DS-000347
CLKIN section), hence the default clamp of +-2 % rather than ppm-level (P1 §2.2 said +-500 ppm, which would be wrong).

The fit refreshes once per ``refit_period_s`` of host time (and on every point until ``min_points`` are collected).
The residual constant bias (the minimum delay, tens of us) is absorbed by the IMU/camera time offset calibration.
"""

from __future__ import annotations

import collections

import numpy as np

NOMINAL_NS_PER_US = 1000.0


def _lower_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the lower convex hull of points sorted by x (Andrew's monotone chain)."""
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            # cross product <= 0 -> i2 is not below the segment i1-i  => pop
            if (x[i2] - x[i1]) * (y[i] - y[i1]) - (y[i2] - y[i1]) * (x[i] - x[i1]) <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.asarray(hull, dtype=np.int64)


def fit_lower_envelope(chip_us: np.ndarray, host_ns: np.ndarray, slope_min: float, slope_max: float
                       ) -> tuple[int, float, float, int]:
    """Fit ``host = a + b * chip`` under all points. Returns ``(y0_ns, a_rel_ns, b_ns_per_us, x0_us)`` with the line
    written as ``host = y0 + a_rel + b * (chip - x0)`` (integer y0/x0 offsets keep float64 precision)."""
    ci = np.asarray(chip_us, dtype=np.int64)
    hi = np.asarray(host_ns, dtype=np.int64)
    x0 = int(ci.min())
    y0 = int(hi.min())
    order = np.argsort(ci, kind="stable")
    x = (ci[order] - x0).astype(np.float64)
    y = (hi[order] - y0).astype(np.float64)
    b = NOMINAL_NS_PER_US
    if len(x) >= 2 and x[-1] > x[0]:
        hull = _lower_hull(x, y)
        xm = float(np.mean(x))
        # hull edge spanning the mean
        k = int(np.searchsorted(x[hull], xm, side="right")) - 1
        k = min(max(k, 0), len(hull) - 2)
        i, j = hull[k], hull[k + 1]
        if x[j] > x[i]:
            b = float((y[j] - y[i]) / (x[j] - x[i]))
    b = float(np.clip(b, slope_min, slope_max))
    a = float(np.min(y - b * x))   # highest line with slope b that stays below every point
    return y0, a, b, x0


class ChipClockMapper:
    def __init__(self, window_s: float = 10.0, refit_period_s: float = 1.0, max_ppm: float = 20000.0,
                 min_points: int = 20):
        self.window_ns = int(window_s * 1e9)
        self.refit_period_ns = int(refit_period_s * 1e9)
        self.slope_min = NOMINAL_NS_PER_US * (1 - max_ppm * 1e-6)
        self.slope_max = NOMINAL_NS_PER_US * (1 + max_ppm * 1e-6)
        self.min_points = int(min_points)
        self._pts: collections.deque[tuple[int, int]] = collections.deque()
        self._fit: tuple[int, float, float, int] | None = None
        self._last_fit_host: int | None = None
        self.n_fits = 0

    def reset(self) -> None:
        self._pts.clear()
        self._fit = None
        self._last_fit_host = None

    def add(self, chip_us: int, host_ns: int) -> None:
        chip_us, host_ns = int(chip_us), int(host_ns)
        self._pts.append((chip_us, host_ns))
        while self._pts and host_ns - self._pts[0][1] > self.window_ns:
            self._pts.popleft()
        if (self._fit is None or len(self._pts) < self.min_points
                or host_ns - self._last_fit_host >= self.refit_period_ns):
            self.refit()

    def refit(self) -> None:
        if not self._pts:
            return
        arr = np.array(self._pts, dtype=np.int64)
        self._fit = fit_lower_envelope(arr[:, 0], arr[:, 1], self.slope_min, self.slope_max)
        self._last_fit_host = int(arr[-1, 1])
        self.n_fits += 1

    @property
    def ready(self) -> bool:
        return self._fit is not None

    @property
    def slope_ppm(self) -> float | None:
        """(b / 1000 - 1) * 1e6: positive if the chip's microsecond is longer than a host microsecond."""
        return None if self._fit is None else (self._fit[2] / NOMINAL_NS_PER_US - 1.0) * 1e6

    def map(self, chip_us) -> np.ndarray:
        """chip microseconds (array or scalar) -> host CLOCK_MONOTONIC ns (int64)."""
        if self._fit is None:
            raise RuntimeError("ChipClockMapper has no points yet")
        y0, a, b, x0 = self._fit
        c = (np.asarray(chip_us, dtype=np.int64) - x0).astype(np.float64)
        return y0 + np.round(a + b * c).astype(np.int64)

    def chip_from_host(self, host_ns: int) -> int:
        """Inverse map (for resynchronising the timestamp unwrapper after a FIFO overflow)."""
        if self._fit is None:
            raise RuntimeError("ChipClockMapper has no points yet")
        y0, a, b, x0 = self._fit
        return x0 + int(round((int(host_ns) - y0 - a) / b))

    def residuals_us(self) -> np.ndarray:
        """Point heights above the fitted line (us) in the current window (>= 0 up to refit lag)."""
        if self._fit is None or not self._pts:
            return np.zeros(0)
        arr = np.array(self._pts, dtype=np.int64)
        return (arr[:, 1] - self.map(arr[:, 0])) / 1e3

    def summary(self) -> dict:
        r = self.residuals_us()
        return {"slope_ppm": self.slope_ppm, "resid_us_p99": float(np.percentile(r, 99)) if len(r) else None}
