// SPDX-License-Identifier: MIT
// MockCamera: deterministic synthetic frames for tests (PLAN T2; drives T5's tests).
//
// Timing model (all CLOCK_MONOTONIC_RAW):
// - frame k (k = 0, 1, ...) has the ideal start of exposure t_sof(k) = t0 + round(k * P) + jitter(k) [+ stall_ns if
//   k >= stall_after_frames], with P = 1/fps, t0 = now_raw_ns() at start() + P, jitter(k) uniform in
//   [-jitter_ns, +jitter_ns] from a hash of (seed, k) (independent of how read() is called);
// - frame k is dropped (never delivered; its seq number is skipped, like a real drop) when drop_every_n > 0 and
//   (k + 1) % drop_every_n == 0;
// - after stall_after_frames frame indices, frames stop for stall_ns (a time gap without a seq gap, i.e. a driver
//   stall); stall_ns == 0 stops forever;
// - read() returns frame k when now >= t_arrival_target(k) = t_sof(k) + exposure + readout + 1 ms; t_arrival_ns is
//   the actual now_raw_ns() at return. seq = k.
// - a consumer more than buffer_count periods behind loses the oldest frames (driver FIFO overflow; seq gaps).
// Pixels: pattern checker | gradient | noise | counter (frame index as text, and as 8 little-endian bytes in the first
// 8 pixels of row 0 of the luma/gray plane, for machine checks), rendered in gray and converted to pixel_format.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <random>
#include <string>

#include <opencv2/imgproc.hpp>

#include "backends.hpp"
#include "camera_internal.hpp"
#include "rig/drivers/camera/capture_plan.hpp"
#include "rig/drivers/camera/sensor_modes.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace cam {
namespace {

uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

class MockCamera final : public CameraBase {
 public:
  explicit MockCamera(const CameraConfig& c) : CameraBase(c) {}
  ~MockCamera() override { (void)MockCamera::stop(); }

  Result<std::vector<SensorMode>> modes() const override {
    if (is_known_sensor(config_.sensor)) return known_sensor_modes(config_.sensor);
    SensorMode m;
    m.index = 0;
    m.width = config_.width;
    m.height = config_.height;
    m.min_fps = 1.0;
    m.max_fps = std::max(config_.fps, 1.0);
    m.min_exposure_ns = 13000;
    m.max_exposure_ns = 1'000'000'000;
    m.min_analog_gain = 1.0;
    m.max_analog_gain = 16.0;
    m.bit_depth = 10;
    m.full_fov = true;
    m.description = "mock synthetic mode";
    return std::vector<SensorMode>{m};
  }

  Status start() override {
    if (running_.load()) return failed_precondition("camera " + config_.id + ": already started");
    const std::string ctx = "cameras." + config_.id + ".driver";
    const auto& mk = config_.mock;
    if (!(config_.fps > 0.0)) return invalid_argument(ctx + ".fps: must be > 0");
    if (packed_frame_bytes(config_.pixel_format, config_.width, config_.height) == 0)
      return invalid_argument(ctx + ": unsupported size/format " + std::to_string(config_.width) + "x" +
                              std::to_string(config_.height) + " " + to_string(config_.pixel_format) +
                              " (NV12 needs even sizes; jpeg is not a capture format)");
    period_ = 1e9 / config_.fps;
    if (mk.jitter_ns < 0 || static_cast<double>(mk.jitter_ns) >= period_ / 2)
      return invalid_argument(ctx + ".mock.jitter_ns: must be in [0, period/2) to keep SOF monotonic");
    if (mk.drop_every_n < 0 || mk.drop_every_n == 1)
      return invalid_argument(ctx + ".mock.drop_every_n: 0 (never) or >= 2");
    if (mk.stall_ns < 0) return invalid_argument(ctx + ".mock.stall_ns: must be >= 0");
    if (mk.pattern != "checker" && mk.pattern != "gradient" && mk.pattern != "noise" && mk.pattern != "counter")
      return invalid_argument(ctx + ".mock.pattern: unknown '" + mk.pattern + "' (checker | gradient | noise | counter)");
    RIG_ASSIGN_OR_RETURN(std::vector<SensorMode> ms, modes());
    RIG_ASSIGN_OR_RETURN(SensorMode mode, select_sensor_mode(ms, config_));
    RIG_ASSIGN_OR_RETURN(CapturePlan plan, build_capture_plan(config_, mode));
    set_resolved_mode(mode.index);
    exposure_ns_ = plan.exposure_locked ? config_.exposure_ns
                                        : std::min<int64_t>(static_cast<int64_t>(period_ / 2), 10'000'000);
    gain_ = plan.gain_locked ? config_.analog_gain : 1.0;
    readout_ns_ = static_cast<int64_t>(period_ * 0.5);  // synthetic rolling-shutter readout
    mode_index_ = mode.index;
    begin_stream(config_.fps, mode.index, "mock");
    if (config_.focus.mode == FocusMode::kFixed)
      set_focus(FocusResult{FocusStatus::kUnknown, config_.focus.position, false,
                            "mock: no lens (configured position recorded, status unknown)"});
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = false;
    }
    k_ = 0;
    pending_.reset();
    t0_ = now_raw_ns() + static_cast<TimeNs>(period_);
    running_.store(true);
    return Status::Ok();
  }

  Status stop() override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    running_.store(false);
    return Status::Ok();
  }

  std::optional<ImageFrame> read(std::chrono::nanoseconds timeout) override {
    if (!running_.load()) return std::nullopt;
    const TimeNs deadline = now_raw_ns() + timeout.count();
    const auto& mk = config_.mock;
    for (;;) {
      while (mk.drop_every_n > 0 && (k_ + 1) % static_cast<uint64_t>(mk.drop_every_n) == 0) ++k_;
      if (mk.stall_after_frames >= 0 && mk.stall_ns == 0 && k_ >= static_cast<uint64_t>(mk.stall_after_frames)) {
        sleep_until(deadline);  // stalled forever
        return std::nullopt;
      }
      // Behind real time (R1 F12): the simulated sensor keeps running and its driver FIFO holds buffer_count frames
      // (like the EGLStream FIFO / appsink queue). Frames older than that are lost: skipped here, so their seq
      // numbers are missing and the tracker counts them (drops_seq_gap). Frames within the FIFO window are still
      // delivered with their true stamps, just late (latency grows), as on hardware.
      const TimeNs fifo_window = static_cast<TimeNs>(period_ * std::max(1, config_.buffer_count));
      if (arrival_target(k_) + fifo_window < now_raw_ns()) {
        ++k_;
        pending_.reset();
        continue;
      }
      if (!pending_) pending_ = render(k_);
      const TimeNs target = arrival_target(k_);
      if (target > deadline) {
        sleep_until(deadline);
        return std::nullopt;
      }
      if (!sleep_until(target)) return std::nullopt;  // stopped
      ImageFrame f = std::move(*pending_);
      pending_.reset();
      f.t_arrival_ns = now_raw_ns();
      const uint64_t k = k_++;
      if (!track(f, k)) continue;  // cannot happen with valid jitter; keep the loop honest
      deliver(f);
      return f;
    }
  }

 private:
  TimeNs sof(uint64_t k) const {
    const auto& mk = config_.mock;
    TimeNs t = t0_ + static_cast<TimeNs>(std::llround(static_cast<double>(k) * period_));
    if (mk.stall_after_frames >= 0 && mk.stall_ns > 0 && k >= static_cast<uint64_t>(mk.stall_after_frames))
      t += mk.stall_ns;
    if (mk.jitter_ns > 0) {
      const uint64_t h = splitmix64(mk.seed * 0x632BE59BD9B4E019ULL ^ k);
      t += static_cast<TimeNs>(h % static_cast<uint64_t>(2 * mk.jitter_ns + 1)) - mk.jitter_ns;
    }
    return t;
  }

  /// Host time at which frame k is available to read() (sensor exposure + readout + 1 ms pipeline).
  TimeNs arrival_target(uint64_t k) const { return sof(k) + exposure_ns_ + readout_ns_ + 1'000'000; }

  /// Sleeps until `t` (MONOTONIC_RAW). Returns false if stop() was called.
  bool sleep_until(TimeNs t) {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      if (stop_) return false;
      const TimeNs now = now_raw_ns();
      if (now >= t) return true;
      cv_.wait_for(lk, std::chrono::nanoseconds(t - now));
    }
  }

  std::optional<ImageFrame> render(uint64_t k) {
    ImageFrame f;
    uint8_t* dst = allocate_frame(pool_, config_.pixel_format, config_.width, config_.height, f);
    const int w = config_.width, h = config_.height;
    gray_.create(h, w, CV_8UC1);
    const std::string& pat = config_.mock.pattern;
    if (pat == "checker") {
      const int s = std::max(8, w / 40);
      const int shift = static_cast<int>((k * 2) % static_cast<uint64_t>(2 * s));
      for (int y = 0; y < h; ++y) {
        uint8_t* row = gray_.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) row[x] = ((((x + shift) / s) + (y / s)) & 1) ? 230 : 25;
      }
    } else if (pat == "gradient") {
      for (int y = 0; y < h; ++y) {
        uint8_t* row = gray_.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) row[x] = static_cast<uint8_t>((x + y + k) & 0xFF);
      }
    } else if (pat == "noise") {
      std::mt19937_64 rng(splitmix64(config_.mock.seed) ^ splitmix64(k + 1));
      uint8_t* p = gray_.data;
      const std::size_t n = gray_.total();
      std::size_t i = 0;
      while (i < n) {
        uint64_t r = rng();
        for (int b = 0; b < 8 && i < n; ++b, ++i, r >>= 8) p[i] = static_cast<uint8_t>(r & 0xFF);
      }
    } else {  // counter
      gray_.setTo(cv::Scalar(128));
      const double scale = std::max(0.5, h / 120.0);
      cv::putText(gray_, std::to_string(k), cv::Point(w / 10, h / 2), cv::FONT_HERSHEY_SIMPLEX, scale,
                  cv::Scalar(255), std::max(1, static_cast<int>(scale * 2)));
      if (w >= 8) {
        uint8_t* row0 = gray_.ptr<uint8_t>(0);
        for (int b = 0; b < 8; ++b) row0[b] = static_cast<uint8_t>((k >> (8 * b)) & 0xFF);
      }
    }
    convert_into_frame(gray_.data, 1, gray_.step, f, dst);
    f.t_sof_ns = sof(k);
    f.meta.exposure_ns = exposure_ns_;
    f.meta.analog_gain = gain_;
    f.meta.digital_gain = config_.isp_digital_gain;
    f.meta.readout_ns = readout_ns_;
    f.meta.sensor_mode = mode_index_;
    f.t_mid_ns = mid_exposure_time(f.t_sof_ns, exposure_ns_, readout_ns_);
    return f;
  }

  BufferPool pool_;
  cv::Mat gray_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::atomic<bool> running_{false};
  double period_ = 0.0;
  TimeNs t0_ = 0;
  uint64_t k_ = 0;
  int64_t exposure_ns_ = 0;
  int64_t readout_ns_ = 0;
  double gain_ = 1.0;
  int mode_index_ = -1;
  std::optional<ImageFrame> pending_;
};

}  // namespace

std::unique_ptr<CameraBackend> make_mock_camera(const CameraConfig& config) {
  return std::make_unique<MockCamera>(config);
}

}  // namespace cam
}  // namespace rig
