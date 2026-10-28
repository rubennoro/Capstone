// SPDX-License-Identifier: MIT
// Drop detection and timing windows (rig/drivers/camera/frame_tracker.hpp).
#include "rig/drivers/camera/frame_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rig {

SummaryStats summarize(std::vector<double> v) {
  SummaryStats s;
  s.count = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0.0;
  for (double x : v) sum += x;
  s.min = v.front();
  s.max = v.back();
  s.mean = sum / static_cast<double>(v.size());
  auto pct = [&](double p) {
    const double pos = p * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return v[lo] + (v[hi] - v[lo]) * frac;
  };
  s.p50 = pct(0.50);
  s.p99 = pct(0.99);
  return s;
}

FrameTracker::FrameTracker(double fps, double gap_factor, std::size_t window_frames)
    : period_ns_(fps > 0.0 ? 1e9 / fps : 0.0),
      gap_factor_(gap_factor > 1.0 ? gap_factor : 1.5),
      window_(window_frames > 0 ? window_frames
                                : std::max<std::size_t>(32, static_cast<std::size_t>(std::ceil(2.0 * std::max(fps, 0.0))))) {}

void FrameTracker::reset() {
  last_seq_.reset();
  last_sof_ = kInvalidTime;
  drops_seq_ = drops_time_ = rejected_ = accepted_ = 0;
  intervals_.clear();
  latencies_.clear();
  sof_1s_.clear();
}

bool FrameTracker::observe(std::optional<uint64_t> driver_seq, TimeNs t_sof, TimeNs t_arrival, DropEvent* ev) {
  DropEvent local;
  DropEvent& e = ev != nullptr ? *ev : local;
  e = DropEvent{};
  char buf[200];

  if (t_sof == kInvalidTime) {
    ++rejected_;
    e.rejected = true;
    e.reason = "frame without a timestamp";
    return false;
  }
  if (last_sof_ != kInvalidTime && t_sof <= last_sof_) {
    ++rejected_;
    e.rejected = true;
    std::snprintf(buf, sizeof(buf), "non-monotonic SOF: %lld ns <= previous %lld ns (frame rejected)",
                  static_cast<long long>(t_sof), static_cast<long long>(last_sof_));
    e.reason = buf;
    return false;
  }
  if (driver_seq && last_seq_ && *driver_seq <= *last_seq_) {
    ++rejected_;
    e.rejected = true;
    std::snprintf(buf, sizeof(buf), "non-monotonic driver seq: %llu <= previous %llu (frame rejected)",
                  static_cast<unsigned long long>(*driver_seq), static_cast<unsigned long long>(*last_seq_));
    e.reason = buf;
    return false;
  }

  if (last_sof_ != kInvalidTime) {
    const int64_t dt = t_sof - last_sof_;
    e.gap_ns = dt;
    uint64_t seq_missing = 0;
    if (driver_seq && last_seq_) seq_missing = *driver_seq - *last_seq_ - 1;
    uint64_t slots_missing = 0;
    if (period_ns_ > 0.0 && static_cast<double>(dt) > gap_factor_ * period_ns_) {
      const long long n = std::llround(static_cast<double>(dt) / period_ns_) - 1;
      slots_missing = static_cast<uint64_t>(std::max<long long>(1, n));
    }
    e.seq_missing = seq_missing;
    e.time_missing = slots_missing > seq_missing ? slots_missing - seq_missing : 0;
    drops_seq_ += e.seq_missing;
    drops_time_ += e.time_missing;
    if (e.seq_missing > 0 || e.time_missing > 0) {
      std::snprintf(buf, sizeof(buf), "gap %.3f ms (%.2f periods): %llu missing seq, %llu unexplained time slots",
                    static_cast<double>(dt) * 1e-6, period_ns_ > 0 ? static_cast<double>(dt) / period_ns_ : 0.0,
                    static_cast<unsigned long long>(e.seq_missing), static_cast<unsigned long long>(e.time_missing));
      e.reason = buf;
    }
    intervals_.push_back(static_cast<double>(dt));
    if (intervals_.size() > window_) intervals_.pop_front();
  }
  if (t_arrival != kInvalidTime) {
    latencies_.push_back(static_cast<double>(t_arrival - t_sof));
    if (latencies_.size() > window_) latencies_.pop_front();
  }
  sof_1s_.push_back(t_sof);
  while (!sof_1s_.empty() && sof_1s_.front() < t_sof - 1'000'000'000) sof_1s_.pop_front();

  last_sof_ = t_sof;
  if (driver_seq) last_seq_ = driver_seq;
  ++accepted_;
  return true;
}

void FrameTracker::fill(CameraStats& s) const {
  s.drops_seq_gap = drops_seq_;
  s.drops_time_gap = drops_time_;
  s.inter_frame_ns = summarize(std::vector<double>(intervals_.begin(), intervals_.end()));
  s.latency_ns = summarize(std::vector<double>(latencies_.begin(), latencies_.end()));
  s.last_t_sof_ns = last_sof_;
  if (sof_1s_.size() >= 2 && sof_1s_.back() > sof_1s_.front()) {
    s.fps_estimate = static_cast<double>(sof_1s_.size() - 1) * 1e9 /
                     static_cast<double>(sof_1s_.back() - sof_1s_.front());
  } else {
    s.fps_estimate = kNaN;
  }
}

}  // namespace rig
