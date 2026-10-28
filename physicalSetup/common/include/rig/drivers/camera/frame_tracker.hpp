// SPDX-License-Identifier: MIT
/// @file rig/drivers/camera/frame_tracker.hpp
/// @brief Per-camera drop detection and timing statistics shared by every camera backend. Owned by T2.
///
/// Drop rules (CameraStats in rig/drivers/camera_backend.hpp):
/// - seq gap: the driver's frame counter skipped `n` numbers -> `drops_seq_gap += n` (frames lost before userspace,
///   or by the EGLStream / backend).
/// - time gap: SOF interval dt > drop_gap_factor / fps -> `missing = max(1, round(dt * fps) - 1)` frame slots.
///   Slots not already explained by a seq gap are counted in `drops_time_gap` (a sensor/driver stall the counter did
///   not see). Total missing frames = max(seq gap, time gap).
/// - non-monotonic SOF or seq: the frame is rejected (never delivered: downstream requires strictly increasing image
///   times) and counted as `drops_error` + a timestamp anomaly.
/// Threading: NOT thread-safe; the owning backend serializes access.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "rig/drivers/camera_backend.hpp"
#include "rig/types.hpp"

namespace rig {

/// What observe() found about the gap before one frame.
struct DropEvent {
  uint64_t seq_missing = 0;   ///< Missing driver sequence numbers.
  uint64_t time_missing = 0;  ///< Missing frame slots from the SOF gap not explained by the seq gap.
  int64_t gap_ns = 0;         ///< SOF interval to the previous frame [ns] (0 for the first frame).
  bool rejected = false;      ///< The frame was rejected (non-monotonic time or seq).
  std::string reason;         ///< Human-readable explanation; empty when nothing happened.

  bool any() const noexcept { return seq_missing > 0 || time_missing > 0 || rejected; }
};

/// Exact summary (sorts a copy; linear-interpolated percentiles like numpy's default). NaN fields when empty.
SummaryStats summarize(std::vector<double> values);

class FrameTracker {
 public:
  /// @param fps nominal frame rate [Hz] (> 0).  @param gap_factor  CameraConfig::drop_gap_factor (> 1).
  /// @param window_frames frames kept for inter_frame_ns / latency_ns; 0 = max(32, 2 s of frames).
  FrameTracker(double fps, double gap_factor, std::size_t window_frames = 0);

  /// Observes one frame in capture order. `driver_seq` = the driver's counter (std::nullopt if it has none).
  /// Returns false if the frame must be rejected (see file comment); `ev` (optional) receives the details.
  bool observe(std::optional<uint64_t> driver_seq, TimeNs t_sof_ns, TimeNs t_arrival_ns, DropEvent* ev = nullptr);

  /// Writes drops_seq_gap, drops_time_gap, fps_estimate, inter_frame_ns, latency_ns and last_t_sof_ns into `s`
  /// (the backend adds rejected() to drops_error and timestamp_anomalies).
  void fill(CameraStats& s) const;

  uint64_t drops_seq_gap() const noexcept { return drops_seq_; }
  uint64_t drops_time_gap() const noexcept { return drops_time_; }
  uint64_t rejected() const noexcept { return rejected_; }
  uint64_t accepted() const noexcept { return accepted_; }
  double period_ns() const noexcept { return period_ns_; }
  void reset();

 private:
  double period_ns_;
  double gap_factor_;
  std::size_t window_;
  std::optional<uint64_t> last_seq_;
  TimeNs last_sof_ = kInvalidTime;
  uint64_t drops_seq_ = 0;
  uint64_t drops_time_ = 0;
  uint64_t rejected_ = 0;
  uint64_t accepted_ = 0;
  std::deque<double> intervals_;
  std::deque<double> latencies_;
  std::deque<TimeNs> sof_1s_;  ///< SOF times within 1 s of the newest (fps_estimate).
};

}  // namespace rig
