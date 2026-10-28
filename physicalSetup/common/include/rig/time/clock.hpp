// SPDX-License-Identifier: MIT
/// @file rig/time/clock.hpp
/// @brief Host clocks and the chip-clock -> host-clock mapper. FROZEN CONTRACT (T0); implemented by T3
/// (`now_*_ns` and `sample_clock_anchor` already implemented in T0, common/time/clock.cpp).
///
/// The rig's single time base is CLOCK_MONOTONIC_RAW (PIPELINE 5 "One clock"): not slewed by NTP, the domain of
/// Argus sensor timestamps [I: verified on day H, BRINGUP step 13], and the domain every IMU sample is mapped into.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

/// clock_gettime(CLOCK_MONOTONIC_RAW) in ns. Thread-safe, async-signal-safe, never fails on Linux.
TimeNs now_raw_ns() noexcept;
/// clock_gettime(CLOCK_MONOTONIC) in ns (slewed by NTP; used only for anchors and sleeps).
TimeNs now_mono_ns() noexcept;
/// clock_gettime(CLOCK_REALTIME) in ns since the Unix epoch (wall clock; used for names and manifests only).
TimeNs now_unix_ns() noexcept;

/// One simultaneous reading of the three host clocks, written to session/package metadata so that wall-clock
/// times can be related to MONOTONIC_RAW afterwards.
struct ClockAnchor {
  TimeNs raw_ns = kInvalidTime;   ///< CLOCK_MONOTONIC_RAW at the middle of the bracket.
  TimeNs mono_ns = kInvalidTime;  ///< CLOCK_MONOTONIC read inside the bracket.
  TimeNs unix_ns = kInvalidTime;  ///< CLOCK_REALTIME read inside the bracket.
  TimeNs bracket_ns = -1;         ///< Width of the raw-clock bracket around the two reads [ns] (uncertainty).
};

/// Samples the three clocks `rounds` times (raw, mono, unix, raw) and returns the narrowest bracket.
/// Thread-safe. @param rounds >= 1 (values < 1 are treated as 1).
ClockAnchor sample_clock_anchor(int rounds = 9) noexcept;

/// One bracketed read of a device's free-running counter: host clock read immediately before the bus transaction,
/// the counter value, host clock immediately after. The true host time of the latch lies in [before, after].
struct ClockBracket {
  TimeNs t_host_before_ns = kInvalidTime;  ///< now_raw_ns() before the I2C transaction started.
  int64_t chip_ticks = -1;                 ///< Unwrapped (monotonic 64-bit) chip counter value.
  TimeNs t_host_after_ns = kInvalidTime;   ///< now_raw_ns() after the I2C transaction completed.
};

/// How ChipClockMapper fits host time against chip ticks.
enum class ClockFitMethod : uint8_t {
  kLowerEnvelope = 0,  ///< ("lower_envelope") line under the bracket `before` points: rejects late (preempted)
                       ///< reads; default.
  kRobustLinear = 1,   ///< ("robust_linear") Huber/IRLS fit on bracket midpoints.
};

/// ChipClockMapper parameters. Mirrors rig.yaml `imu.driver.clock`.
struct ChipClockConfig {
  /// Nominal tick period [ns]. LSM6DSOX: 25 us/LSB nominal (DS13345), corrected by the driver with the chip's
  /// INTERNAL_FREQ_FINE trim (AN5272) before the mapper is constructed.
  double nominal_tick_ns = 25000.0;
  /// The fitted slope is clamped to nominal_tick_ns * (1 +/- max_slope_deviation). 0.03 = +/-3 %.
  double max_slope_deviation = 0.03;
  /// Sliding window: number of most recent brackets used in the fit (100 ms period -> 300 = 30 s).
  int window_brackets = 300;
  /// Brackets wider than this [ns] are rejected (a preempted or retried transaction carries no information).
  int64_t max_bracket_width_ns = 2000000;
  /// Fit method.
  ClockFitMethod method = ClockFitMethod::kLowerEnvelope;
};

/// Current fit: t_host(ticks) = offset_ns + slope_ns_per_tick * (ticks - ref_ticks).
struct ClockFit {
  bool valid = false;               ///< At least 2 accepted brackets spanning > 0 ticks.
  double slope_ns_per_tick = kNaN;  ///< [ns/tick], within the clamp.
  double offset_ns = kNaN;          ///< Host time at ref_ticks [ns, MONOTONIC_RAW].
  int64_t ref_ticks = 0;            ///< Reference tick (keeps the arithmetic well-conditioned).
  bool slope_clamped = false;       ///< The unclamped estimate was outside the allowed range.
};

/// Diagnostic statistics of the mapper (exported to clock_fit.json and ImuStats).
struct ClockFitStats {
  uint64_t brackets_added = 0;       ///< add_bracket() calls that were accepted.
  uint64_t brackets_rejected = 0;    ///< Rejected: too wide, non-monotonic ticks or host time.
  uint64_t slope_clamp_events = 0;   ///< Fits whose slope hit the clamp.
  SummaryStats residual_ns;          ///< Signed residual (bracket midpoint - fit) over the window [ns].
  SummaryStats bracket_width_ns;     ///< after - before over the window [ns].
  ClockFit fit;                      ///< Fit at the time stats() was called.
};

/// Maps an unwrapped chip counter to CLOCK_MONOTONIC_RAW from bracketed reads (PLAN T1 "Clock", T3 "time").
///
/// Threading: NOT thread-safe; owned and used by one thread (the IMU driver thread). Copy stats() out for others.
/// Behaviour before the fit is valid: with exactly one accepted bracket, map() uses the nominal slope through that
/// bracket's midpoint; with none, map() returns kInvalidTime.
class ChipClockMapper {
 public:
  explicit ChipClockMapper(ChipClockConfig config = {});
  ~ChipClockMapper();
  ChipClockMapper(ChipClockMapper&&) noexcept;
  ChipClockMapper& operator=(ChipClockMapper&&) noexcept;
  ChipClockMapper(const ChipClockMapper&) = delete;
  ChipClockMapper& operator=(const ChipClockMapper&) = delete;

  /// Adds one bracket. Returns kInvalidArgument (and counts a rejection) if after < before, if the bracket is
  /// wider than max_bracket_width_ns, or if ticks / host time go backwards versus the previous accepted bracket.
  Status add_bracket(TimeNs t_host_before_ns, int64_t chip_ticks, TimeNs t_host_after_ns);
  Status add_bracket(const ClockBracket& b) {
    return add_bracket(b.t_host_before_ns, b.chip_ticks, b.t_host_after_ns);
  }

  /// Host time [ns, MONOTONIC_RAW] of `chip_ticks` under the current fit (extrapolates outside the window).
  TimeNs map(int64_t chip_ticks) const;

  /// True once the fit is valid (>= 2 accepted brackets).
  bool ready() const;
  ClockFit fit() const;
  ClockFitStats stats() const;
  const ChipClockConfig& config() const;
  /// Drops all brackets (e.g. after a chip reset, when the counter restarts).
  void reset();

 private:
  struct Impl;  // defined by T3 in common/time/
  std::unique_ptr<Impl> impl_;
};

}  // namespace rig
