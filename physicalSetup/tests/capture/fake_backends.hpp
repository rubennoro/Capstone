// SPDX-License-Identifier: MIT
// Test-only camera/IMU backends for CaptureService (T3). They run on a VIRTUAL clock that advances `accel` times
// faster than real time: frame k has t_sof = t0 + k / fps (virtual), delivered when real time reaches
// t0_real + k / (fps * accel). Late frames are delivered immediately (never dropped), so counts are exact even
// under sanitizers. Injectable failures: start error, fatal error after N frames/samples.
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "rig/drivers/camera_backend.hpp"
#include "rig/drivers/imu_backend.hpp"

namespace rig {
namespace testing {

struct FakeCameraOptions {
  std::string id = "main";
  double fps = 60.0;
  int width = 64;
  int height = 48;
  double accel = 1.0;
  Status start_error;            ///< returned by start() if not OK
  int64_t fatal_after = -1;      ///< status() turns fatal after this many frames
  int64_t stall_at = -1;         ///< no frames from this index on (until stop)
  int64_t drop_every = 0;        ///< skip one frame (seq + time gap) every N
};

class FakeCamera final : public CameraBackend {
 public:
  explicit FakeCamera(FakeCameraOptions o);
  const CameraConfig& config() const override { return cfg_; }
  Status start() override;
  Status stop() override;
  std::optional<ImageFrame> read(std::chrono::nanoseconds timeout) override;
  std::optional<ImageFrame> latest() const override;
  CameraStats stats() const override;
  Status status() const override;
  Result<std::vector<SensorMode>> modes() const override { return std::vector<SensorMode>{}; }
  std::atomic<int> start_calls{0}, stop_calls{0};
  /// Placement observed inside start() on the calling thread and on a thread start() creates (like a backend's
  /// internal acquire/poll thread). Written in start() before it returns; read by the test after start().
  std::vector<int> start_affinity, child_affinity;
  int start_nice = 0, child_nice = 0;

 private:
  FakeCameraOptions o_;
  CameraConfig cfg_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool running_ = false;
  int64_t k_ = 0;  // next frame index
  TimeNs t0_virtual_ = 0, t0_real_ = 0;
  CameraStats stats_;
  Status status_;
  std::optional<ImageFrame> latest_;
};

struct FakeImuOptions {
  std::string id = "imu0";
  double odr_hz = 833.0;
  double accel = 1.0;
  int bracket_period_ms = 100;  ///< virtual
  Status start_error;
  int64_t fatal_after = -1;
  int64_t max_samples = -1;  ///< stop producing after this many (until stop)
};

class FakeImu final : public ImuBackend {
 public:
  explicit FakeImu(FakeImuOptions o);
  const ImuConfig& config() const override { return cfg_; }
  Status start() override;
  Status stop() override;
  std::size_t read(std::vector<ImuSample>& out, std::chrono::nanoseconds timeout) override;
  std::size_t read_brackets(std::vector<ClockBracket>& out) override;
  ImuStats stats() const override;
  Status status() const override;
  std::atomic<int> start_calls{0}, stop_calls{0};

 private:
  FakeImuOptions o_;
  ImuConfig cfg_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool running_ = false;
  int64_t k_ = 0, next_bracket_ = 0;
  TimeNs t0_virtual_ = 0, t0_real_ = 0;
  ImuStats stats_;
  Status status_;
};

}  // namespace testing
}  // namespace rig
