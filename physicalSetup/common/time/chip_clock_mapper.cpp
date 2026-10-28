// SPDX-License-Identifier: MIT
// ChipClockMapper (rig/time/clock.hpp): maps an unwrapped device counter to CLOCK_MONOTONIC_RAW from bracketed
// register reads. Implemented by T1 (STATUS.md contract note); generic, no LSM6DSOX knowledge.
//
// Model. A bracket (before, ticks, after) says the counter was latched at some host time in [before, after]
// (`before`/`after` = now_raw_ns() around the bus transaction). The relation is affine over a sliding window:
//   t_host(ticks) = y0 + a + b * (ticks - ref)            (y0, ref = integer anchors of the window: precision)
//
// Slope b (both methods): weighted least squares on bracket midpoints, weights 1 / (width^2/12 + tick^2/12) (the
// latch is uniform inside the bracket, plus counter quantization), made robust with Huber IRLS on residuals
// normalized by a MAD scale. A long window (30 s) makes b precise to well below 1 ppm with ~100 us jitter.
// b is clamped to nominal * (1 +- max_slope_deviation).
//
// Offset a:
// - kRobustLinear: the IRLS intercept.
// - kLowerEnvelope (default): interval intersection. Every inlier bracket bounds the line from both sides:
//   before_i <= latch_i <= after_i, so  a_lo = max(before_i - b x_i)  and  a_hi = min(after_i - b x_i), and
//   a = (a_lo + a_hi) / 2. The extremes come from the fastest, least-delayed transactions: a read that was
//   preempted (late `after`, or late latch after `before`) never tightens the bound, so scheduling delays do not
//   bias the result. (The PIPELINE 5 "lower envelope" idea of sensor_rig chip_clock.py, written for two-sided
//   brackets: the line lies under every `after` point and over every `before` point.)
//
// Outliers. With >= 10 brackets in the fit, a bracket whose interval is farther than outlier_gate_ns from the
// prediction is rejected (a torn multi-byte counter read would otherwise move a_lo/a_hi); 5 in a row restart
// the window. Inside the fit, brackets with |normalized residual| > 4 are left out of the envelope.
#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <vector>

#include "rig/time/clock.hpp"

namespace rig {

namespace {

constexpr int kGateMinBrackets = 10;
constexpr int kGateRestartCount = 5;
constexpr int kIrlsIterations = 8;
constexpr double kHuberK = 1.5;
constexpr double kInlierLimit = 4.0;

struct Bracket {
  TimeNs before;
  int64_t ticks;
  TimeNs after;
};

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  const std::size_t m = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
  return v[m];
}

SummaryStats summarize(std::vector<double> v) {
  SummaryStats s;
  s.count = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.max = v.back();
  double sum = 0.0;
  for (double x : v) sum += x;
  s.mean = sum / static_cast<double>(v.size());
  auto rank = [&](double q) {
    // Nearest-rank percentile (exact over the window).
    std::size_t k = static_cast<std::size_t>(std::ceil(q * static_cast<double>(v.size())));
    k = std::max<std::size_t>(k, 1);
    return v[std::min(k, v.size()) - 1];
  };
  s.p50 = rank(0.50);
  s.p99 = rank(0.99);
  return s;
}

}  // namespace

struct ChipClockMapper::Impl {
  ChipClockConfig cfg;
  std::deque<Bracket> win;
  uint64_t added = 0;
  uint64_t rejected = 0;
  uint64_t clamp_events = 0;
  int consecutive_outliers = 0;

  // Current fit.
  bool valid = false;
  bool clamped = false;
  int64_t ref = 0;    // ticks anchor
  TimeNs y0 = 0;      // host anchor
  double a = 0.0;     // offset relative to y0 at ref [ns]
  double b = 0.0;     // slope [ns/tick]

  explicit Impl(ChipClockConfig c) : cfg(c) {}

  double fit_at(int64_t ticks) const { return a + b * static_cast<double>(ticks - ref); }

  void refit() {
    valid = false;
    clamped = false;
    if (win.empty()) return;
    ref = win.front().ticks;
    y0 = win.front().before;
    const double b0 = cfg.nominal_tick_ns;
    const std::size_t n = win.size();
    std::vector<double> x(n), mid(n), lo(n), hi(n), w0(n);
    for (std::size_t i = 0; i < n; ++i) {
      const Bracket& k = win[i];
      x[i] = static_cast<double>(k.ticks - ref);
      lo[i] = static_cast<double>(k.before - y0);
      hi[i] = static_cast<double>(k.after - y0);
      mid[i] = 0.5 * (lo[i] + hi[i]);
      const double wd = hi[i] - lo[i];
      w0[i] = 1.0 / (wd * wd / 12.0 + b0 * b0 / 12.0);
    }
    const bool span = x.back() > x.front();
    if (n < 2 || !span) {
      // Nominal slope through the newest midpoint (contract: one bracket -> nominal slope through it).
      b = b0;
      a = mid.back() - b * x.back();
      return;
    }

    std::vector<double> w = w0, h(n, 1.0), u(n, 0.0);
    double fa = 0.0, fb = b0;
    for (int it = 0; it < kIrlsIterations; ++it) {
      double sw = 0, sx = 0, sy = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const double wi = w0[i] * h[i];
        sw += wi;
        sx += wi * x[i];
        sy += wi * mid[i];
      }
      const double xm = sx / sw, ym = sy / sw;
      double sxx = 0, sxy = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const double wi = w0[i] * h[i];
        sxx += wi * (x[i] - xm) * (x[i] - xm);
        sxy += wi * (x[i] - xm) * (mid[i] - ym);
      }
      fb = sxx > 0 ? sxy / sxx : b0;
      fa = ym - fb * xm;
      // Normalized residuals and a MAD scale (floored at 1: never down-weight below the bracket model).
      std::vector<double> abs_u(n);
      for (std::size_t i = 0; i < n; ++i) {
        u[i] = (mid[i] - (fa + fb * x[i])) * std::sqrt(w0[i]);
        abs_u[i] = std::abs(u[i]);
      }
      const double s = std::max(1.0, 1.4826 * median_of(abs_u));
      for (std::size_t i = 0; i < n; ++i) {
        u[i] /= s;
        const double au = std::abs(u[i]);
        h[i] = au <= kHuberK ? 1.0 : kHuberK / au;
      }
    }

    const double bmin = b0 * (1.0 - cfg.max_slope_deviation), bmax = b0 * (1.0 + cfg.max_slope_deviation);
    if (!(fb >= bmin && fb <= bmax)) {
      clamped = true;
      ++clamp_events;
      fb = std::min(std::max(fb, bmin), bmax);
      double sw = 0, sr = 0;
      for (std::size_t i = 0; i < n; ++i) {
        sw += w0[i] * h[i];
        sr += w0[i] * h[i] * (mid[i] - fb * x[i]);
      }
      fa = sr / sw;
    }
    b = fb;
    a = fa;
    if (cfg.method == ClockFitMethod::kLowerEnvelope) {
      double alo = -INFINITY, ahi = INFINITY;
      for (std::size_t i = 0; i < n; ++i) {
        if (!clamped && std::abs(u[i]) > kInlierLimit) continue;
        alo = std::max(alo, lo[i] - b * x[i]);
        ahi = std::min(ahi, hi[i] - b * x[i]);
      }
      if (std::isfinite(alo) && std::isfinite(ahi)) a = 0.5 * (alo + ahi);
    }
    valid = true;
  }
};

ChipClockMapper::ChipClockMapper(ChipClockConfig config) : impl_(std::make_unique<Impl>(config)) {}
ChipClockMapper::~ChipClockMapper() = default;
ChipClockMapper::ChipClockMapper(ChipClockMapper&&) noexcept = default;
ChipClockMapper& ChipClockMapper::operator=(ChipClockMapper&&) noexcept = default;

Status ChipClockMapper::add_bracket(TimeNs t_host_before_ns, int64_t chip_ticks, TimeNs t_host_after_ns) {
  Impl& m = *impl_;
  if (t_host_before_ns == kInvalidTime || t_host_after_ns == kInvalidTime || t_host_after_ns < t_host_before_ns) {
    ++m.rejected;
    return invalid_argument("ChipClockMapper: bracket after < before");
  }
  if (t_host_after_ns - t_host_before_ns > m.cfg.max_bracket_width_ns) {
    ++m.rejected;
    return invalid_argument("ChipClockMapper: bracket width " + std::to_string(t_host_after_ns - t_host_before_ns) +
                            " ns > max " + std::to_string(m.cfg.max_bracket_width_ns));
  }
  if (!m.win.empty()) {
    const Bracket& last = m.win.back();
    if (chip_ticks < last.ticks || t_host_before_ns < last.before) {
      ++m.rejected;
      return invalid_argument("ChipClockMapper: ticks or host time went backwards (ticks " +
                              std::to_string(chip_ticks) + " < " + std::to_string(last.ticks) + " or host)");
    }
  }
  if (m.cfg.outlier_gate_ns > 0 && m.valid && m.win.size() >= static_cast<std::size_t>(kGateMinBrackets)) {
    const double pred = static_cast<double>(m.y0) + m.fit_at(chip_ticks);
    const double gate = static_cast<double>(m.cfg.outlier_gate_ns);
    if (pred < static_cast<double>(t_host_before_ns) - gate || pred > static_cast<double>(t_host_after_ns) + gate) {
      if (++m.consecutive_outliers < kGateRestartCount) {
        ++m.rejected;
        return out_of_range("ChipClockMapper: bracket is an outlier versus the fit");
      }
      m.win.clear();  // the relation really changed: restart from this bracket
    }
  }
  m.consecutive_outliers = 0;
  m.win.push_back({t_host_before_ns, chip_ticks, t_host_after_ns});
  const std::size_t cap = static_cast<std::size_t>(std::max(2, m.cfg.window_brackets));
  while (m.win.size() > cap) m.win.pop_front();
  ++m.added;
  m.refit();
  return Status::Ok();
}

TimeNs ChipClockMapper::map(int64_t chip_ticks) const {
  const Impl& m = *impl_;
  if (m.win.empty()) return kInvalidTime;
  return m.y0 + static_cast<TimeNs>(std::llround(m.fit_at(chip_ticks)));
}

bool ChipClockMapper::ready() const { return impl_->valid; }

ClockFit ChipClockMapper::fit() const {
  const Impl& m = *impl_;
  ClockFit f;
  f.valid = m.valid;
  if (m.win.empty()) return f;
  f.slope_ns_per_tick = m.b;
  f.offset_ns = static_cast<double>(m.y0) + m.a;
  f.ref_ticks = m.ref;
  f.slope_clamped = m.clamped;
  return f;
}

ClockFitStats ChipClockMapper::stats() const {
  const Impl& m = *impl_;
  ClockFitStats s;
  s.brackets_added = m.added;
  s.brackets_rejected = m.rejected;
  s.slope_clamp_events = m.clamp_events;
  std::vector<double> res, wid;
  res.reserve(m.win.size());
  wid.reserve(m.win.size());
  for (const Bracket& k : m.win) {
    const double mid = 0.5 * (static_cast<double>(k.before - m.y0) + static_cast<double>(k.after - m.y0));
    res.push_back(mid - m.fit_at(k.ticks));
    wid.push_back(static_cast<double>(k.after - k.before));
  }
  s.residual_ns = summarize(std::move(res));
  s.bracket_width_ns = summarize(std::move(wid));
  s.fit = fit();
  return s;
}

const ChipClockConfig& ChipClockMapper::config() const { return impl_->cfg; }

void ChipClockMapper::reset() {
  Impl& m = *impl_;
  m.win.clear();
  m.valid = false;
  m.clamped = false;
  m.consecutive_outliers = 0;
}

}  // namespace rig
