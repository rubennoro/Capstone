// SPDX-License-Identifier: MIT
/// @file rig/slam/slam_backend.hpp
/// @brief Live SLAM backend interface used by ec-slam-live (PIPELINE 5, 6.3). FROZEN CONTRACT (T0);
/// implemented by T6 in common/slam/ (MockSlam now; OrbSlam3Backend in common/slam/orbslam3/, built only with
/// RIG_WITH_ORBSLAM3).
///
/// Threading: all methods from one thread (the ec-slam-live tracking thread). Not thread-safe.
/// Live SLAM output is for live checks and display only, never training data (PIPELINE D1).
#pragma once

#include <memory>
#include <string>

#include "rig/config/rig_config.hpp"
#include "rig/shm/status_channel.hpp"
#include "rig/span.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

/// Everything a backend needs, assembled by ec-slam-live from the session's rig.yaml.
struct SlamSettings {
  SlamLiveConfig live;      ///< `slam_live`.
  RigCameraConfig camera;   ///< The input camera (`slam_live.camera`), model at capture size.
  RigImuConfig imu;         ///< Noise + rate for the IMU factor.
  std::string work_dir;     ///< Scratch dir for generated settings files (ORB-SLAM3 YAML) and logs.
};

class SlamBackend {
 public:
  virtual ~SlamBackend() = default;

  /// "mock" or "orbslam3".
  virtual std::string name() const = 0;

  /// Loads vocabulary / writes settings / starts the viewer. Blocking (ORB-SLAM3 vocabulary load takes seconds).
  virtual Status init(const SlamSettings& settings) = 0;

  /// Tracks one frame. @param frame input image (GRAY8 or NV12, any size; the backend downscales to
  /// live.input_width x input_height). Time = frame.t_mid_ns.
  /// @param imu all IMU samples with t_ns in (previous frame t_mid, this frame t_mid], in time order (pointer +
  /// count; may be empty for the first frame).
  /// Returns the status for this frame with t_ns, tracking_state, map_id, pose, frames_tracked, track_ms filled;
  /// heartbeat_seq and t_publish_ns are filled by the publisher. An error = backend failure (ec-slam-live exits;
  /// ec-record aborts with slam_crashed).
  virtual Result<SlamStatus> track(const ImageFrame& frame, Span<const ImuSample> imu) = 0;

  /// Stops threads, closes the viewer. Idempotent.
  virtual Status shutdown() = 0;
};

/// Creates a backend of `kind` (not initialized). kUnimplemented for orbslam3 without RIG_WITH_ORBSLAM3.
/// Implemented by T6 (common/slam/slam_factory.cpp; T0 ships a stub).
Result<std::unique_ptr<SlamBackend>> make_slam_backend(SlamBackendKind kind);

}  // namespace rig
