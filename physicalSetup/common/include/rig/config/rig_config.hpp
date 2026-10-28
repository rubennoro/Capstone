// SPDX-License-Identifier: MIT
/// @file rig/config/rig_config.hpp
/// @brief In-memory form of `rig.yaml`. FROZEN CONTRACT (T0); loader/validator/hash implemented by T3 in
/// common/config/.
///
/// Mapping rules (1:1 with episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml):
/// - Every struct below is one YAML mapping; every field is one key with the same name (snake_case).
///   Exceptions, filled by the loader rather than read: `CameraConfig::id` (= the key under `cameras:`),
///   `RigConfig::source_path`.
/// - Enums are YAML strings (spellings in each enum's comment). Eigen 3x3 matrices are YAML lists of 3 rows, vectors
///   lists of 3 numbers. Integers may be written in hex (`0x6A`).
/// - Unknown keys are errors. Missing keys take the defaults written here; required keys are marked REQUIRED.
/// - Errors are kInvalidArgument with the YAML path first, e.g.
///   `cameras.uw.driver: 1280x720@60 is a cropped mode on imx219; full-FOV modes are 1640x1232@30, 3280x2464@21`.
/// Threading: plain values.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "rig/camera_model/model_params.hpp"
#include "rig/drivers/camera_backend.hpp"
#include "rig/drivers/imu_backend.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

/// Current rig.yaml schema version. Bumped on any incompatible key change.
inline constexpr int kRigConfigSchemaVersion = 1;

/// Thread placement. Mirrors `*.thread`.
struct ThreadConfig {
  std::vector<int> cpus;  ///< CPU affinity set (0..5 on the Orin Nano); empty = no pinning.
  int priority = 0;       ///< 0 = SCHED_OTHER; 1..99 = SCHED_FIFO priority (needs CAP_SYS_NICE / rtprio limit).
  int nice = 0;           ///< Nice value for SCHED_OTHER threads (-20..19).
};

/// Rigid transform T_a_b: maps points in frame b to frame a (pose of b in a). Mirrors e.g. `cameras.<id>.T_imu_cam`.
struct Extrinsics {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();  ///< R_a_b, YAML 3 rows; must be orthonormal, det +1.
  Eigen::Vector3d translation_m = Eigen::Vector3d::Zero();  ///< t_a_b [m].
  Provenance provenance = Provenance::kNominal;
  std::string note;  ///< Source of the numbers (CAD file, Kalibr run id, ...).
};

/// Per-camera time correction applied only when consuming (offline stages, deployment), never baked into raw
/// files (PIPELINE 5). Mirrors `cameras.<id>.time`.
struct CameraTimeConfig {
  int64_t offset_ns = 0;  ///< t_true = t_mid_ns + offset_ns [ns] (Kalibr timeshift cam->imu, LED test).
  Provenance provenance = Provenance::kNominal;
};

/// What RawWriter stores for a camera (PIPELINE 6.1 "Stored resolutions"). Mirrors `cameras.<id>.storage`.
struct CameraStorageConfig {
  int width = 0;                  ///< Stored JPEG width [px]; 0 = native capture size.
  int height = 0;                 ///< Stored JPEG height [px]; 0 = native. Must keep the capture aspect ratio.
  int jpeg_quality = 90;          ///< libjpeg quality 1..100.
  int encoder_threads = 2;        ///< JPEG worker threads for this camera.
  std::size_t queue_frames = 120; ///< Lossless RawWriter queue capacity [frames] (overflow = abort, PIPELINE 6.2).
};

/// One physical camera. Mirrors `cameras.<id>`.
struct RigCameraConfig {
  std::string role;               ///< "main" | "ultrawide" (REQUIRED). Exactly one of each.
  CameraConfig driver;            ///< `driver` (REQUIRED).
  CameraModelParams model;        ///< `model` (REQUIRED). Size must equal driver width x height.
  Extrinsics T_imu_cam;           ///< `T_imu_cam`: camera pose in the IMU body frame.
  CameraTimeConfig time;          ///< `time`.
  CameraStorageConfig storage;    ///< `storage`.
  ThreadConfig thread;            ///< `thread`: capture thread placement.
};

/// IMU noise for SLAM / Kalibr (continuous-time densities). Mirrors `imu.noise`.
struct ImuNoiseConfig {
  double accel_noise_density = kNaN;  ///< [m/s^2/sqrt(Hz)].
  double accel_random_walk = kNaN;    ///< [m/s^3/sqrt(Hz)].
  double gyro_noise_density = kNaN;   ///< [rad/s/sqrt(Hz)].
  double gyro_random_walk = kNaN;     ///< [rad/s^2/sqrt(Hz)].
  Provenance provenance = Provenance::kNominal;
  std::string note;
};

/// Mirrors `imu`.
struct RigImuConfig {
  ImuConfig driver;                   ///< `driver` (REQUIRED).
  ImuNoiseConfig noise;               ///< `noise`.
  ThreadConfig thread;                ///< `thread`: IMU capture/poll thread placement.
  std::size_t queue_samples = 16384;  ///< Lossless RawWriter queue capacity [samples].
};

/// Virtual target camera for a physical one (PIPELINE 2: uw -> virtual iPhone ultrawide). Mirrors
/// `virtual_cameras.<id>`; the key is the physical camera id.
struct VirtualCameraConfig {
  CameraModelParams model;            ///< kind must be "pinhole" (REQUIRED).
  Eigen::Matrix3d R_phys_virt = Eigen::Matrix3d::Identity();  ///< Virtual camera orientation in the physical one.
  double min_coverage = 1.0;          ///< ImagePrep::create fails below this remap coverage [0..1].
};

/// Mirrors `imageprep`.
struct ImagePrepConfig {
  int out_size = 224;  ///< Policy image side [px] (iPhUMI gen_replay_buffer_zarr.py:430).
};

/// Steady-state detection before recording may start (PLAN T3 "capture"). Mirrors `capture.steady_state`.
struct SteadyStateConfig {
  int min_frames = 30;           ///< Consecutive frames per camera with interval within tolerance.
  double rate_tolerance = 0.10;  ///< |interval * fps - 1| <= tolerance for each of those frames.
  double imu_window_s = 1.0;     ///< IMU rate within tolerance over this window [s].
};

/// Mirrors `capture`.
struct CaptureConfig {
  SteadyStateConfig steady_state;
  int64_t read_timeout_ms = 200;  ///< Per-read timeout of capture threads [ms] (shutdown latency bound).
};

/// Live abort/warn rules (PIPELINE 6.2). Defaults are the iPhone-parity values and our own thresholds.
/// Mirrors `live_checks`.
struct LiveChecksConfig {
  // --- iPhone parity (exact) ---
  std::string main_camera = "main";  ///< Camera the fps rule watches.
  double main_fps_min = 31.0;        ///< Abort if per-frame 1/dt < this (ViewController.swift:1819 `fps < 31`).
  double main_fps_grace_s = 1.0;     ///< ...only after this long into recording (ViewController.swift:1819 `> 1`).
  double pose_speed_max_mps = 7.0;   ///< Abort if live ||dp||/dt > this (ViewController.swift:305, 1829-1843).
  bool uw_require_frames = true;     ///< Abort if the ultrawide produced no frames (DemonstrationData.swift:574).
  // --- ours ---
  std::string uw_camera = "uw";      ///< Camera the uw rules watch.
  double uw_fps_min = 27.0;          ///< Abort if uw fps over uw_fps_window_s < this [Hz].
  double uw_fps_window_s = 1.0;      ///< [s].
  bool slam_abort_on_lost = true;    ///< Abort when live SLAM reports tracking lost.
  bool slam_abort_on_reset = true;   ///< Abort when live SLAM resets.
  bool slam_abort_on_map_change = true;  ///< Abort when the live SLAM map id changes.
  double slam_heartbeat_timeout_s = 0.5;  ///< Abort if no SLAM heartbeat for this long [s].
  double imu_gap_max_ms = 20.0;      ///< Abort on an IMU sample gap > this [ms].
  double imu_rate_min_ratio = 0.95;  ///< Abort if IMU rate over imu_rate_window_s < ratio * odr_hz.
  double imu_rate_window_s = 1.0;    ///< [s].
  bool imu_abort_on_overrun = true;  ///< Abort on a FIFO or host queue overrun.
  bool writer_abort_on_overflow = true;  ///< Abort on any lossless StreamHub queue overflow.
  double disk_min_free_gb = 5.0;     ///< Refuse to arm / abort below this free space on storage.data_root [GB].
  double thermal_warn_c = 85.0;      ///< Warn when any watched thermal zone exceeds this [deg C].
  bool thermal_abort_on_throttle = true;  ///< Abort if clock throttling is detected.
  std::vector<std::string> thermal_zones;  ///< /sys/class/thermal zone `type` names to watch; empty = all.
  double tags_unseen_warn_s = 0.5;   ///< Warn if the finger tags are not seen for this long [s] (warn only).
};

/// Mirrors `record` (ec-record state machine, PLAN T5).
struct RecordConfig {
  double arming_timeout_s = 30.0;          ///< ARMING -> clean exit if not recording by then [s].
  bool require_gripper_calibration = true; ///< Refuse to arm without a PASSing gripper_range.json.
  double stop_flush_timeout_s = 10.0;      ///< Max time to drain writer queues on stop [s].
};

/// Live SLAM backend kind. YAML strings in parentheses.
enum class SlamBackendKind : uint8_t {
  kOrbSlam3 = 0,  ///< ("orbslam3") needs RIG_WITH_ORBSLAM3.
  kMock = 1,      ///< ("mock") replays a TUM trajectory with injectable failures (T6).
};

/// MockSlam options. Mirrors `slam_live.mock`. Additive fields allowed (T6).
struct MockSlamConfig {
  std::string trajectory_tum;    ///< TUM file "t x y z qx qy qz qw" (t in s); empty = a built-in slow circle.
  double init_delay_s = 2.0;     ///< Report kInitializing for this long before kTrackingOk [s].
};

/// Mirrors `slam_live` (PIPELINE 6.3).
struct SlamLiveConfig {
  bool enabled = true;
  SlamBackendKind backend = SlamBackendKind::kOrbSlam3;
  std::string camera = "uw";          ///< Input camera id.
  std::string vocabulary_path;        ///< ORBvoc.txt (or .bin) path on the Jetson.
  int input_width = 800;              ///< Downscaled grey input [px] (about 800x600).
  int input_height = 600;
  int n_features = 1000;              ///< ORB features per frame.
  int frame_decimation = 1;           ///< Track every Nth uw frame (2 = 15 Hz fallback, PIPELINE 6.3).
  bool viewer = true;                 ///< Pangolin viewer (only when viz.mode is slam or both).
  double viewer_refresh_hz = 10.0;
  ThreadConfig thread;                ///< ec-slam-live tracking thread placement.
  MockSlamConfig mock;
};

/// Viewer selection. YAML strings in parentheses.
enum class VizMode : uint8_t {
  kNone = 0,     ///< ("none")
  kCameras = 1,  ///< ("cameras") ec-viz mosaic only.
  kSlam = 2,     ///< ("slam") Pangolin only.
  kBoth = 3,     ///< ("both") default (PIPELINE 4.2).
};

/// Mirrors `viz`.
struct VizConfig {
  VizMode mode = VizMode::kBoth;
  double refresh_hz = 10.0;    ///< Mosaic refresh [Hz].
  int mosaic_width = 1280;     ///< Window width [px].
  bool tag_overlay = true;     ///< Low-rate finger-tag overlay (T7 detector).
  double tag_detect_hz = 2.0;  ///< [Hz].
};

/// Mirrors `shm` (POSIX shared-memory bus, PIPELINE 5).
struct ShmConfig {
  std::string prefix = "/rig";      ///< Segment name prefix: "<prefix>.<stream>" (e.g. /rig.cam_uw).
  std::size_t frame_ring_slots = 8; ///< Slots per camera frame ring.
  int viz_frame_width = 640;        ///< Width of the downscaled frames published for ec-viz [px]; 0 = native.
  std::size_t imu_ring_slots = 8192;  ///< Slots (one sample each) in the IMU ring.
};

/// Mirrors `storage`.
struct StorageConfig {
  std::string data_root = "/data/rig";  ///< Sessions live in <data_root>/<session>/ (PIPELINE 6.1); NVMe on the Jetson.
  bool fsync_on_stop = true;            ///< fsync every written file + directories on stop.
};

/// Rig-level calibration bookkeeping. Mirrors `calibration`.
struct CalibrationConfig {
  Provenance provenance = Provenance::kNominal;  ///< Worst provenance of the camera/IMU calibration in this file.
  std::string source;                 ///< Free text (e.g. "lens spec sheets + mount CAD rev B").
  std::string date;                   ///< ISO date of the calibration, empty if never calibrated.
  Extrinsics T_uw_tcp;                ///< TCP (gripper tip centre) pose in the uw camera frame (D6: CAD first).
};

/// The whole rig.yaml.
struct RigConfig {
  int schema_version = kRigConfigSchemaVersion;  ///< REQUIRED; must equal kRigConfigSchemaVersion.
  std::string rig_name;                          ///< REQUIRED; free identifier.
  std::string notes;
  std::map<std::string, RigCameraConfig> cameras;  ///< REQUIRED; ids match [a-z0-9_]+; one main + one ultrawide.
  RigImuConfig imu;                                ///< REQUIRED.
  std::map<std::string, VirtualCameraConfig> virtual_cameras;  ///< Keys are physical camera ids.
  ImagePrepConfig imageprep;
  CaptureConfig capture;
  LiveChecksConfig live_checks;
  RecordConfig record;
  SlamLiveConfig slam_live;
  VizConfig viz;
  ShmConfig shm;
  StorageConfig storage;
  CalibrationConfig calibration;

  std::string source_path;  ///< File it was loaded from (not a YAML key).
};

/// Loads and fully validates a rig.yaml (every key typed and range-checked, unknown keys rejected, cross-field
/// checks such as model size == capture size and mode legality). Never throws.
Result<RigConfig> load_rig_config(const std::string& path);

/// Same, from YAML text; `source_name` is used in error messages.
Result<RigConfig> parse_rig_config(const std::string& yaml_text, const std::string& source_name = "<string>");

/// Canonical YAML of `config` (all keys, fixed order, no comments). parse_rig_config(to_yaml(c)) == c.
std::string to_yaml(const RigConfig& config);

/// Stable content hash: lowercase hex SHA-256 of to_yaml(config) (comments and key order in the source file do
/// not change it). Recorded in session.json and manifests.
std::string content_hash(const RigConfig& config);

}  // namespace rig
