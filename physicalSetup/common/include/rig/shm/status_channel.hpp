// SPDX-License-Identifier: MIT
/// @file rig/shm/status_channel.hpp
/// @brief Small fixed POD status structs exchanged between ec-record, ec-slam-live, ec-viz and deployment, and a
/// typed latest-value channel over the shm ring. FROZEN CONTRACT (T0), including struct layouts (cross-process ABI;
/// bump the struct's kVersion and kShmRingVersion on change).
///
/// The channel templates below are implemented here (inline) on top of ShmRingWriter/ShmRingReader (T3).
/// Threading: one writer thread per channel; each reader object used by one thread.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "rig/shm/ring.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

// ---------------------------------------------------------------------------------------------------------------
// ec-record state
// ---------------------------------------------------------------------------------------------------------------

/// ec-record state machine (PLAN T5).
enum class RecorderState : uint8_t {
  kInit = 0,
  kArming = 1,     ///< Pre-roll recording; waiting for SLAM tracking_ok and steady streams.
  kRecording = 2,
  kStopping = 3,
  kDone = 4,
  kAborted = 5,
};

constexpr const char* to_string(RecorderState s) noexcept {
  switch (s) {
    case RecorderState::kInit: return "init";
    case RecorderState::kArming: return "arming";
    case RecorderState::kRecording: return "recording";
    case RecorderState::kStopping: return "stopping";
    case RecorderState::kDone: return "done";
    case RecorderState::kAborted: return "aborted";
  }
  return "unknown";
}

/// Why an episode was aborted (PIPELINE 6.2; one value per abort rule). Strings are used in events.csv and
/// live_checks.json.
enum class AbortReason : uint16_t {
  kNone = 0,
  kMainFpsLow = 1,       ///< "main_fps_low" (iPhone parity, ViewController.swift:1819).
  kPoseJump = 2,         ///< "pose_jump" (> 7 m/s, ViewController.swift:1829-1843).
  kUwNoFrames = 3,       ///< "uw_no_frames" (DemonstrationData.swift:574).
  kUwFpsLow = 4,         ///< "uw_fps_low".
  kSlamLost = 5,         ///< "slam_lost".
  kSlamReset = 6,        ///< "slam_reset".
  kSlamMapChanged = 7,   ///< "slam_map_changed".
  kSlamHeartbeat = 8,    ///< "slam_heartbeat" (stalled > 0.5 s).
  kSlamCrashed = 9,      ///< "slam_crashed" (child process exited).
  kImuGap = 10,          ///< "imu_gap".
  kImuOverrun = 11,      ///< "imu_overrun".
  kImuRateLow = 12,      ///< "imu_rate_low".
  kWriterOverflow = 13,  ///< "writer_overflow" (lossless queue overflow).
  kDiskLow = 14,         ///< "disk_low".
  kThermalThrottle = 15, ///< "thermal_throttle".
  kArmingTimeout = 16,   ///< "arming_timeout".
  kUserCancelled = 17,   ///< "user_cancelled" (ended before recording started).
  kBackendError = 18,    ///< "backend_error" (a sensor backend failed).
  kInternal = 19,        ///< "internal".
};

constexpr const char* to_string(AbortReason r) noexcept {
  switch (r) {
    case AbortReason::kNone: return "none";
    case AbortReason::kMainFpsLow: return "main_fps_low";
    case AbortReason::kPoseJump: return "pose_jump";
    case AbortReason::kUwNoFrames: return "uw_no_frames";
    case AbortReason::kUwFpsLow: return "uw_fps_low";
    case AbortReason::kSlamLost: return "slam_lost";
    case AbortReason::kSlamReset: return "slam_reset";
    case AbortReason::kSlamMapChanged: return "slam_map_changed";
    case AbortReason::kSlamHeartbeat: return "slam_heartbeat";
    case AbortReason::kSlamCrashed: return "slam_crashed";
    case AbortReason::kImuGap: return "imu_gap";
    case AbortReason::kImuOverrun: return "imu_overrun";
    case AbortReason::kImuRateLow: return "imu_rate_low";
    case AbortReason::kWriterOverflow: return "writer_overflow";
    case AbortReason::kDiskLow: return "disk_low";
    case AbortReason::kThermalThrottle: return "thermal_throttle";
    case AbortReason::kArmingTimeout: return "arming_timeout";
    case AbortReason::kUserCancelled: return "user_cancelled";
    case AbortReason::kBackendError: return "backend_error";
    case AbortReason::kInternal: return "internal";
  }
  return "unknown";
}

/// Published by ec-record (~10 Hz) for ec-viz. Channel name "<prefix>.recorder".
struct RecorderStatus {
  static constexpr uint32_t kTypeTag = 0x52454331;  ///< "REC1"
  static constexpr uint32_t kVersion = 1;
  TimeNs t_ns;                 ///< now_raw_ns() at publish.
  uint64_t heartbeat_seq;      ///< +1 per publish.
  uint8_t state;               ///< RecorderState.
  uint8_t reserved0;
  uint16_t abort_reason;       ///< AbortReason (kNone unless state == kAborted).
  uint32_t warnings;           ///< Bit i set = warning i active (bit 0 thermal, 1 tags unseen, 2 disk low-ish).
  TimeNs t_arm_ns;             ///< kInvalidTime until armed.
  TimeNs t_rec_start_ns;       ///< kInvalidTime until recording.
  double max_pose_speed_mps;   ///< Max live ||dp||/dt during recording (the iPhone UI number).
  double disk_free_gb;
  float soc_temp_c;            ///< Hottest watched thermal zone.
  uint32_t reserved1;
  char episode_name[128];      ///< NUL-terminated package name, "" before creation.
};
static_assert(std::is_trivially_copyable_v<RecorderStatus> && std::is_standard_layout_v<RecorderStatus>);

/// One stream's line in CaptureStatus.
struct CaptureStreamStatus {
  char id[16];                ///< NUL-terminated camera id ("" = unused entry).
  double fps;                 ///< Over the last 1 s [Hz].
  uint64_t frames;
  uint64_t drops;             ///< CameraStats::drops_total().
  uint64_t lossless_overflows;  ///< Sum over this stream's kLossless subscribers.
  TimeNs last_t_ns;           ///< Last frame t_mid.
};
static_assert(std::is_trivially_copyable_v<CaptureStreamStatus>);

inline constexpr int kMaxStatusCameras = 4;

/// Published by the process owning CaptureService (~10 Hz). Channel name "<prefix>.capture".
struct CaptureStatus {
  static constexpr uint32_t kTypeTag = 0x43415031;  ///< "CAP1"
  static constexpr uint32_t kVersion = 1;
  TimeNs t_ns;
  uint64_t heartbeat_seq;
  uint32_t n_cameras;
  uint8_t steady;              ///< CaptureService::is_steady().
  uint8_t healthy;             ///< CaptureService::health().ok().
  uint16_t thread_config_failures;  ///< Threads whose affinity/priority could not be applied.
  CaptureStreamStatus cameras[kMaxStatusCameras];
  double imu_rate_hz;
  uint64_t imu_samples;
  uint64_t imu_fifo_overruns;
  uint64_t imu_queue_overruns;
  uint64_t imu_gaps;
  int64_t imu_max_gap_ns;
  int32_t imu_fifo_level_max;
  uint32_t reserved0;
  TimeNs imu_last_t_ns;
};
static_assert(std::is_trivially_copyable_v<CaptureStatus> && std::is_standard_layout_v<CaptureStatus>);

// ---------------------------------------------------------------------------------------------------------------
// Live SLAM
// ---------------------------------------------------------------------------------------------------------------

enum class TrackingState : uint8_t {
  kNotStarted = 0,    ///< Backend not initialized / no frames yet.
  kInitializing = 1,  ///< Waiting for enough motion (ORB-SLAM3 NOT_INITIALIZED / visual-inertial init).
  kTrackingOk = 2,    ///< Pose valid.
  kLost = 3,          ///< Tracking lost (ORB-SLAM3 LOST / RECENTLY_LOST).
  kReset = 4,         ///< The backend reset itself (map discarded).
};

constexpr const char* to_string(TrackingState s) noexcept {
  switch (s) {
    case TrackingState::kNotStarted: return "not_started";
    case TrackingState::kInitializing: return "initializing";
    case TrackingState::kTrackingOk: return "tracking_ok";
    case TrackingState::kLost: return "lost";
    case TrackingState::kReset: return "reset";
  }
  return "unknown";
}

/// Live SLAM output, one per tracked frame (PLAN T0 `{t_ns, tracking_state, map_id, pose[7], heartbeat_seq}`).
/// Channel name "<prefix>.slam". Diagnostic/live-check use only; never training data (PIPELINE D1).
/// Pose: W_T_C of the SLAM input camera (uw optical frame) in the SLAM world frame; p [m], q unit quaternion
/// (w, x, y, z). Valid only when tracking_state == kTrackingOk.
struct SlamStatus {
  static constexpr uint32_t kTypeTag = 0x534C4D31;  ///< "SLM1"
  static constexpr uint32_t kVersion = 1;
  TimeNs t_ns;                 ///< t_mid_ns of the frame this pose is for [ns, MONOTONIC_RAW].
  uint8_t tracking_state;      ///< TrackingState.
  uint8_t reserved0[3];
  uint32_t map_id;             ///< Atlas map id; a change = new map (abort rule).
  double p[3];                 ///< Position [m].
  double q[4];                 ///< Orientation (w, x, y, z).
  uint64_t heartbeat_seq;      ///< +1 per published status (also while not tracking).
  uint64_t frames_tracked;     ///< Frames passed to track() so far.
  TimeNs t_publish_ns;         ///< now_raw_ns() at publish (heartbeat age = now - t_publish_ns).
  float track_ms;              ///< Wall time of the last track() call [ms].
  uint32_t n_tracked_points;   ///< Map points matched in this frame (0 if unknown).
};
static_assert(std::is_trivially_copyable_v<SlamStatus> && std::is_standard_layout_v<SlamStatus>);

/// Fresh-initialized status (state not started, identity pose).
inline SlamStatus make_slam_status() noexcept {
  SlamStatus s{};
  s.t_ns = kInvalidTime;
  s.tracking_state = static_cast<uint8_t>(TrackingState::kNotStarted);
  s.q[0] = 1.0;
  s.t_publish_ns = kInvalidTime;
  return s;
}

// ---------------------------------------------------------------------------------------------------------------
// Typed latest-value channel
// ---------------------------------------------------------------------------------------------------------------

/// Number of ring slots behind a status channel (readers only want the newest value).
inline constexpr std::size_t kStatusChannelSlots = 4;

/// Writer side. T must be trivially copyable and declare `static constexpr uint32_t kTypeTag`.
template <class T>
class StatusChannelWriter {
  static_assert(std::is_trivially_copyable_v<T>, "status structs must be trivially copyable");

 public:
  /// @param name POSIX shm name, e.g. "/rig.slam".
  static Result<StatusChannelWriter> create(const std::string& name) {
    ShmRingOptions o;
    o.name = name;
    o.kind = StreamKind::kStatus;
    o.slot_size = sizeof(T);
    o.n_slots = kStatusChannelSlots;
    o.type_tag = T::kTypeTag;
    o.stream_name = name;
    auto ring = ShmRingWriter::create(o);
    if (!ring.ok()) return ring.status();
    return StatusChannelWriter(std::move(ring).value());
  }

  /// Publishes a copy of `value` with timestamp `t_ns`. Never blocks.
  Status publish(const T& value, TimeNs t_ns) {
    auto r = ring_->write(Span<const uint8_t>(reinterpret_cast<const uint8_t*>(&value), sizeof(T)), t_ns);
    return r.ok() ? Status::Ok() : r.status();
  }

  uint64_t published() const { return ring_->write_seq(); }

 private:
  explicit StatusChannelWriter(std::unique_ptr<ShmRingWriter> ring) : ring_(std::move(ring)) {}
  std::unique_ptr<ShmRingWriter> ring_;
};

/// Reader side.
template <class T>
class StatusChannelReader {
  static_assert(std::is_trivially_copyable_v<T>, "status structs must be trivially copyable");

 public:
  static Result<StatusChannelReader> open(const std::string& name) {
    auto ring = ShmRingReader::open(name, StreamKind::kStatus, T::kTypeTag, /*start_at_latest=*/true);
    if (!ring.ok()) return ring.status();
    return StatusChannelReader(std::move(ring).value());
  }

  /// Newest consistent value, or std::nullopt if nothing was published since the last call (or ever).
  Result<std::optional<T>> read_latest() {
    T value{};
    auto r = ring_->read_latest(Span<uint8_t>(reinterpret_cast<uint8_t*>(&value), sizeof(T)));
    if (!r.ok()) return r.status();
    if (!r.value().has_value()) return std::optional<T>();
    return std::optional<T>(value);
  }

  /// See ShmRingReader::writer_alive().
  bool writer_alive() const { return ring_->writer_alive(); }
  uint64_t writer_seq() const { return ring_->writer_seq(); }

 private:
  explicit StatusChannelReader(std::unique_ptr<ShmRingReader> ring) : ring_(std::move(ring)) {}
  std::unique_ptr<ShmRingReader> ring_;
};

}  // namespace rig
