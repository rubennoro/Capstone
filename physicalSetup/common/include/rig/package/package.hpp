// SPDX-License-Identifier: MIT
/// @file rig/package/package.hpp
/// @brief Session and package directories, iPhUMI-identical naming, manifest with stage records, event log
/// (PIPELINE 6.1, 7.1). FROZEN CONTRACT (T0); implemented by T3 in common/package/.
///
/// Package naming (exact iPhUMI parity):
///   `<ISO>_<5 digits>_<session>_<kind>_<side>`, e.g. `2026-10-01T21-13-57.489Z_75494_firstreal_demonstration_left`
/// - ISO: ISO8601DateFormatter with .withInternetDateTime + .withFractionalSeconds (UTC, millisecond fraction, 'Z')
///   (iPhUMI/ios_app/iPhUMI/Util/DateManager.swift:13-17; ViewController.swift:1092), then every ':' replaced by '-'
///   (ViewController.swift:1336).
/// - 5 digits: each digit uniform in 1...9 (no zeros) (ViewController.swift:1097-1102).
/// - session: the session name (ViewController.swift:1339-1341).
/// - kind: "grippercalibration" | "demonstration" (ViewController.swift:1344-1348); ours adds "rigcalibration".
/// - side: "left" | "right" | "head" (ViewController.swift:73-81; ViewController.swift:1351).
/// iPhUMI parses names with `name.split('_')` and requires exactly 5 parts
/// (python_package/iphumi/demonstration_processing/process_stages/group.py:36-38); the grouped 4-part title is
/// split again by utils/generic_util.py:306-308 (kind = last part) and :368-370. Therefore NO component may
/// contain '_': session names
/// are validated against [A-Za-z0-9-]+ and the rig calibration kind is spelled "rigcalibration".
///
/// Time fields in session.json / manifest.json: `*_unix_ns` = CLOCK_REALTIME ns (wall clock, for humans and
/// names); stream data always uses MONOTONIC_RAW. JSON writes are atomic: temp file + fsync + rename + fsync(dir).
/// Threading: Session/Package objects are not thread-safe (one owner thread). EventLog::append is thread-safe.
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

/// Package kind = the `<kind>` name component.
enum class PackageKind : uint8_t {
  kDemonstration = 0,      ///< "demonstration"
  kGripperCalibration = 1, ///< "grippercalibration"
  kRigCalibration = 2,     ///< "rigcalibration" (ec-calibrate --rig; not an iPhUMI kind)
};
constexpr const char* to_string(PackageKind k) noexcept {
  switch (k) {
    case PackageKind::kDemonstration: return "demonstration";
    case PackageKind::kGripperCalibration: return "grippercalibration";
    case PackageKind::kRigCalibration: return "rigcalibration";
  }
  return "unknown";
}

/// Gripper side = the `<side>` name component.
enum class Side : uint8_t {
  kLeft = 0,   ///< "left"
  kRight = 1,  ///< "right" (the rig default; iPhUMI single mode canonical side, generic_util.py:21)
  kHead = 2,   ///< "head"
};
constexpr const char* to_string(Side s) noexcept {
  switch (s) {
    case Side::kLeft: return "left";
    case Side::kRight: return "right";
    case Side::kHead: return "head";
  }
  return "unknown";
}

/// Parsed / constructed package name.
struct PackageName {
  std::string iso;         ///< e.g. "2026-10-01T21-13-57.489Z" (colons already replaced).
  std::string randomizer;  ///< 5 chars in [1-9].
  std::string session;     ///< [A-Za-z0-9-]+.
  PackageKind kind = PackageKind::kDemonstration;
  Side side = Side::kRight;

  /// "<iso>_<randomizer>_<session>_<kind>_<side>".
  std::string str() const {
    return iso + "_" + randomizer + "_" + session + "_" + to_string(kind) + "_" + to_string(side);
  }
};

/// kInvalidArgument unless `session` matches [A-Za-z0-9-]+ (1..64 chars).
Status validate_session_name(std::string_view session);
/// "YYYY-MM-DDTHH-MM-SS.mmmZ" (UTC, milliseconds truncated) for a CLOCK_REALTIME time.
std::string iso_name_timestamp(TimeNs unix_ns);
/// Five digits each in 1..9 drawn from `seed` (deterministic for tests; pass a random seed in production).
std::string random_five_digits(uint64_t seed);
/// Builds a name; validates the session.
Result<PackageName> make_package_name(std::string_view session, PackageKind kind, Side side, TimeNs unix_ns,
                                      uint64_t seed);
/// Strict parse (exactly 5 '_'-separated parts, each validated).
Result<PackageName> parse_package_name(std::string_view name);

// ---------------------------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------------------------

inline constexpr int kManifestSchemaVersion = 1;

/// Outcome of one offline stage (PIPELINE 7.1).
enum class StageStatus : uint8_t {
  kDone = 0,     ///< "done"
  kFailed = 1,   ///< "failed"
  kSkipped = 2,  ///< "skipped"
};
constexpr const char* to_string(StageStatus s) noexcept {
  switch (s) {
    case StageStatus::kDone: return "done";
    case StageStatus::kFailed: return "failed";
    case StageStatus::kSkipped: return "skipped";
  }
  return "unknown";
}

/// manifest.json `stages.<name>` (PIPELINE 7.1).
struct StageRecord {
  StageStatus status = StageStatus::kDone;
  std::string impl;            ///< Algorithm that ran, e.g. "orbslam3_mi", "imu_preint", "external_tum".
  std::string inputs_hash;     ///< Hex SHA-256 over input files + relevant config/calibration keys.
  std::vector<std::string> outputs;  ///< Paths relative to the package dir.
  std::string tool_version;    ///< e.g. "rig 0.1.0 (git abc1234)".
  TimeNs t_start_unix_ns = kInvalidTime;
  TimeNs t_end_unix_ns = kInvalidTime;
  std::string log;             ///< Log path relative to the package dir.
  std::string message;         ///< Failure reason / summary (empty when done).
};

/// Derived package state (PIPELINE 7.1), plus the live states.
enum class PackageState : uint8_t {
  kRecording = 0,   ///< "recording": created by ec-record, not finished.
  kRecorded = 1,    ///< "recorded"
  kRawOk = 2,       ///< "raw_ok"
  kSlamDone = 3,    ///< "slam_done"
  kTrajDone = 4,    ///< "traj_done"
  kPosed = 5,       ///< "posed"
  kWidthDone = 6,   ///< "width_done"
  kValid = 7,       ///< "valid"
  kInvalid = 8,     ///< "invalid"
  kExported = 9,    ///< "exported"
};
constexpr const char* to_string(PackageState s) noexcept {
  switch (s) {
    case PackageState::kRecording: return "recording";
    case PackageState::kRecorded: return "recorded";
    case PackageState::kRawOk: return "raw_ok";
    case PackageState::kSlamDone: return "slam_done";
    case PackageState::kTrajDone: return "traj_done";
    case PackageState::kPosed: return "posed";
    case PackageState::kWidthDone: return "width_done";
    case PackageState::kValid: return "valid";
    case PackageState::kInvalid: return "invalid";
    case PackageState::kExported: return "exported";
  }
  return "unknown";
}

/// manifest.json.
struct Manifest {
  int schema_version = kManifestSchemaVersion;
  std::string name;               ///< PackageName::str().
  PackageKind kind = PackageKind::kDemonstration;
  Side side = Side::kRight;
  std::string session;
  std::string task;               ///< Task label (whole demo = task; PIPELINE 7.2 stage 7).
  bool recorded = false;          ///< Set by ec-record after the final fsync.
  TimeNs created_unix_ns = kInvalidTime;
  TimeNs recorded_unix_ns = kInvalidTime;
  std::string rig_config_hash;    ///< content_hash() of the session's frozen rig.yaml.
  std::string software_version;
  std::map<std::string, StageRecord> stages;  ///< Keyed by stage name ("validate_raw", "slam", ...).
};

/// State implied by `recorded` and the stage records (PIPELINE 7.1 order).
PackageState derive_state(const Manifest& manifest);

Result<Manifest> read_manifest(const std::filesystem::path& path);
/// Atomic write (temp + fsync + rename + dir fsync).
Status write_manifest_atomic(const std::filesystem::path& path, const Manifest& manifest);

// ---------------------------------------------------------------------------------------------------------------
// Event log
// ---------------------------------------------------------------------------------------------------------------

/// Standard event types in events.csv (PIPELINE 6.1).
namespace event_type {
inline constexpr const char* kArm = "arm";
inline constexpr const char* kRecStart = "rec_start";
inline constexpr const char* kRecStop = "rec_stop";
inline constexpr const char* kWarning = "warning";
inline constexpr const char* kAbort = "abort";
inline constexpr const char* kDrop = "drop";
inline constexpr const char* kInfo = "info";
}  // namespace event_type

/// events.csv writer: header `t_ns,type,detail`; t_ns in MONOTONIC_RAW; detail CSV-quoted when needed.
class EventLog {
 public:
  /// Creates (writing the header) or appends to `path`.
  static Result<std::unique_ptr<EventLog>> open(const std::filesystem::path& path);
  virtual ~EventLog() = default;
  /// Thread-safe. Line-buffered (flushed per event).
  virtual Status append(TimeNs t_ns, std::string_view type, std::string_view detail) = 0;
  /// fsync.
  virtual Status sync() = 0;
  virtual const std::filesystem::path& path() const = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// Session and package
// ---------------------------------------------------------------------------------------------------------------

/// session.json.
struct SessionInfo {
  std::string session;            ///< Session name ([A-Za-z0-9-]+); also the directory name.
  std::string task;
  std::string operator_name;
  TimeNs created_unix_ns = kInvalidTime;
  std::string rig_config_hash;    ///< content_hash() of rig.yaml.
  std::string software_version;
  TimeNs anchor_raw_ns = kInvalidTime;   ///< ClockAnchor at creation (relate wall clock to MONOTONIC_RAW).
  TimeNs anchor_unix_ns = kInvalidTime;
};

/// One package directory (layout PIPELINE 6.1).
class Package {
 public:
  /// Opens an existing package (reads manifest.json).
  static Result<Package> open(const std::filesystem::path& dir);

  const std::filesystem::path& dir() const { return dir_; }
  const PackageName& name() const { return name_; }
  Manifest& manifest() { return manifest_; }
  const Manifest& manifest() const { return manifest_; }
  /// Atomically rewrites manifest.json.
  Status save_manifest() const;

  std::filesystem::path raw_dir() const { return dir_ / "raw"; }
  /// raw/cam_<camera_id>.
  std::filesystem::path camera_dir(std::string_view camera_id) const { return raw_dir() / ("cam_" + std::string(camera_id)); }
  /// raw/<imu_id> (imu0).
  std::filesystem::path imu_dir(std::string_view imu_id = "imu0") const { return raw_dir() / std::string(imu_id); }
  std::filesystem::path events_path() const { return raw_dir() / "events.csv"; }
  std::filesystem::path live_checks_path() const { return raw_dir() / "live_checks.json"; }
  std::filesystem::path live_slam_path() const { return raw_dir() / "live_slam.csv"; }
  std::filesystem::path stage_dir(std::string_view stage) const { return dir_ / "stages" / std::string(stage); }
  std::filesystem::path export_dir() const { return dir_ / "export"; }

  /// Moves events.csv + live_checks.json into `<aborted_dir>/<name>/` and deletes the package directory.
  Status discard(const std::filesystem::path& aborted_dir);
  /// Checks the PIPELINE 6.1 layout for this kind (required files/dirs present, CSV headers). Returns the first
  /// problem as kDataLoss / kNotFound with the relative path.
  Status validate_layout() const;

 private:
  friend class Session;
  Package() = default;
  std::filesystem::path dir_;
  PackageName name_;
  Manifest manifest_;
};

/// A session directory `<data_root>/<session>/` (PIPELINE 6.1).
class Session {
 public:
  /// Creates the directory tree, writes session.json, and copies `rig_yaml_path` verbatim to `<dir>/rig.yaml`
  /// (the frozen config). kAlreadyExists if the session exists.
  static Result<Session> create(const std::filesystem::path& data_root, const SessionInfo& info,
                                const std::filesystem::path& rig_yaml_path);
  /// Opens an existing session (reads session.json).
  static Result<Session> open(const std::filesystem::path& session_dir);

  const std::filesystem::path& dir() const { return dir_; }
  const SessionInfo& info() const { return info_; }
  std::filesystem::path rig_yaml_path() const { return dir_ / "rig.yaml"; }
  std::filesystem::path calibration_dir() const { return dir_ / "calibration"; }
  std::filesystem::path episodes_dir() const { return dir_ / "episodes"; }
  std::filesystem::path aborted_dir() const { return dir_ / "aborted"; }
  std::filesystem::path gripper_range_path() const { return calibration_dir() / "gripper_range.json"; }

  /// Creates a new package (calibration kinds under calibration/, demonstrations under episodes/) with raw/
  /// subdirectories and an initial manifest (recorded = false). @param unix_ns name time (now_unix_ns()).
  Result<Package> create_package(PackageKind kind, Side side, TimeNs unix_ns, uint64_t seed);
  /// Package directories of `kind` in name order.
  Result<std::vector<std::filesystem::path>> list_packages(PackageKind kind) const;

 private:
  Session() = default;
  std::filesystem::path dir_;
  SessionInfo info_;
};

}  // namespace rig
