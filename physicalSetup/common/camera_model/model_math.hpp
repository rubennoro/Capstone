// SPDX-License-Identifier: MIT
// Private numeric helpers of the camera models (T4). Scalar ports of sensor_rig/sensor_rig/camera_model.py.
#pragma once

#include <algorithm>
#include <cmath>

namespace rig::camera_model_detail {

/// Smallest x in (0, hi] with g(x) <= 0 (dense scan of n samples + 100 bisection steps); `hi` if there is none.
/// Port of camera_model.py:73-86 (`_first_nonpositive`), including numpy.linspace's sample positions
/// (x_i = i * (hi / (n - 1)), last sample exactly hi).
template <class G>
double first_nonpositive(G g, double hi, int n = 20001) {
  const double step = hi / static_cast<double>(n - 1);
  int bad = -1;
  double x_prev = 0.0, x_bad = 0.0;
  for (int i = 0; i < n; ++i) {
    const double x = (i == n - 1) ? hi : static_cast<double>(i) * step;
    if (g(x) <= 0.0) {
      bad = i;
      x_bad = x;
      break;
    }
    x_prev = x;
  }
  if (bad < 0) return hi;
  // camera_model.py:79: lo, up = x[bad[0] - 1], x[bad[0]] (bad[0] >= 1 because g(0) = 1 > 0 for every model).
  double lo = x_prev, up = x_bad;
  for (int k = 0; k < 100; ++k) {
    const double mid = 0.5 * (lo + up);
    if (g(mid) > 0.0) {
      lo = mid;
    } else {
      up = mid;
    }
  }
  return lo;
}

/// Solves f(x) = target for x in [0, hi], f increasing with derivative df (safeguarded Newton).
/// Port of camera_model.py:89-109 (`_monotone_inverse`) for one element. Returns `solvable` (false where
/// target > f(hi), or target is NaN); *x is the iterate either way.
template <class F, class DF>
bool monotone_inverse(F f, DF df, double target, double hi, double* x_out, int iters = 100) {
  const bool solvable = target <= f(hi) * (1.0 + 1e-12);
  double lo = 0.0, up = hi;
  double x = std::clamp(target, 0.0, hi);
  if (!(target == target)) x = 0.0;  // NaN guard (np.clip would propagate NaN; the result is invalid anyway)
  if (solvable) {
    const double tol = 4e-16 * std::max(1.0, hi);
    for (int i = 0; i < iters; ++i) {
      const double e = f(x) - target;
      if (std::abs(e) <= tol) break;
      if (e < 0) {
        lo = x;
      } else {
        up = x;
      }
      const double d = df(x);
      const double xn = x - e / d;
      x = (!std::isfinite(xn) || xn < lo || xn > up) ? 0.5 * (lo + up) : xn;
    }
  }
  *x_out = x;
  return solvable;
}

constexpr double kPi = 3.14159265358979323846;
inline double deg2rad(double d) { return d * (kPi / 180.0); }

}  // namespace rig::camera_model_detail
