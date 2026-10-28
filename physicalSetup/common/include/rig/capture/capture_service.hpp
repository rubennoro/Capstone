// SPDX-License-Identifier: MIT
/// @file rig/capture/capture_service.hpp
/// @brief Owns the camera and IMU backends, runs one capture thread per sensor and fans out to StreamHubs
/// (PIPELINE 5). FROZEN CONTRACT (T0); implemented by T3 in common/capture/. Shared with deployment.
///
/// Lifecycle: create() -> subscribe on the hubs -> start() -> [wait_steady()] -> stop(). Not restartable: create a
/// new service instead.
/// Threading: create/start/stop from the owning thread. Hub accessors return pointers valid for the service's
/// lifetime. is_steady(), status(), health(), *_stats() are thread-safe.
/// Each capture thread: loops backend->read(timeout) -> hub->publish(); applies its ThreadConfig (affinity,
/// priority). Failing to apply it (no CAP_SYS_NICE on the laptop) is logged and counted in
/// CaptureStatus::thread_config_failures, never fatal.
#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rig/capture/stream_hub.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/drivers/camera_backend.hpp"
#include "rig/drivers/imu_backend.hpp"
#include "rig/shm/status_channel.hpp"
#include "rig/status.hpp"

namespace rig {

/// What CaptureService needs besides the backends (normally derived from RigConfig).
struct CaptureOptions {
  CaptureConfig capture;                       ///< steady-state + read timeout.
  std::map<std::string, ThreadConfig> camera_threads;  ///< Per camera id; missing = no pinning.
  ThreadConfig imu_thread;
};

class CaptureService {
 public:
  /// Builds backends with make_camera()/make_imu() from `config.cameras` / `config.imu`.
  static Result<std::unique_ptr<CaptureService>> create(const RigConfig& config);
  /// Test/deployment injection: takes ownership of ready-made (not started) backends. `imu` may be null (cameras
  /// only, e.g. deployment).
  static Result<std::unique_ptr<CaptureService>> create(CaptureOptions options,
                                                       std::vector<std::unique_ptr<CameraBackend>> cameras,
                                                       std::unique_ptr<ImuBackend> imu);
  virtual ~CaptureService() = default;

  /// Starts every backend and capture thread. On failure, stops what was started and returns the first error.
  virtual Status start() = 0;
  /// Stops threads and backends, then closes all hubs (consumers drain and see closed()). Idempotent.
  virtual Status stop() = 0;

  /// True once every stream reached steady state (SteadyStateConfig) since start(). Never goes back to false
  /// (later rate problems are the live checks' job).
  virtual bool is_steady() const = 0;
  /// Waits up to `timeout` for is_steady(). kTimeout naming the streams that are not steady; the first fatal
  /// backend error if one occurs meanwhile.
  virtual Status wait_steady(std::chrono::nanoseconds timeout) = 0;

  /// Camera ids in config order.
  virtual std::vector<std::string> camera_ids() const = 0;
  /// Frame hub of a camera; nullptr for an unknown id.
  virtual StreamHub<ImageFrame>* camera_hub(const std::string& camera_id) = 0;
  /// IMU sample hub; nullptr without an IMU.
  virtual StreamHub<ImuSample>* imu_hub() = 0;
  /// IMU clock-bracket hub (RawWriter -> chip.csv); nullptr without an IMU.
  virtual StreamHub<ClockBracket>* imu_bracket_hub() = 0;

  virtual Result<CameraStats> camera_stats(const std::string& camera_id) const = 0;
  virtual Result<ImuStats> imu_stats() const = 0;
  /// Snapshot for the shm status channel (filled at call time).
  virtual CaptureStatus status() const = 0;
  /// OK, or the first fatal error from any backend / capture thread.
  virtual Status health() const = 0;
};

}  // namespace rig
