// SPDX-License-Identifier: MIT
/// @file rig/package/layout.hpp
/// @brief File names and exact CSV header lines of the package layout (PIPELINE 6.1). Writers (T5 RawWriter,
/// T7) and validate_layout() / offline stage 0 use these constants so they cannot drift. T3 addition.
#pragma once

namespace rig {
namespace layout {

inline constexpr const char* kSessionJson = "session.json";
inline constexpr const char* kRigYaml = "rig.yaml";
inline constexpr const char* kManifest = "manifest.json";
inline constexpr const char* kRawDir = "raw";
inline constexpr const char* kStagesDir = "stages";
inline constexpr const char* kExportDir = "export";
inline constexpr const char* kCameraDirPrefix = "cam_";  ///< raw/cam_<camera id>/

// raw/cam_<id>/
/// data/<t_ns>.jpg (t_ns = t_mid_ns): the EuRoC cam0/data/ folder name, so EuRoC readers (ORB-SLAM3 examples,
/// Kalibr bagcreater) use raw/cam_<id>/ directly (R1 F14).
inline constexpr const char* kFramesDir = "data";
inline constexpr const char* kCamData = "data.csv";
/// EuRoC cam0/data.csv header, verbatim, so ORB-SLAM3 / Kalibr tooling reads the folder directly (PIPELINE 6.1).
/// Rows: <t_mid_ns>,<t_mid_ns>.jpg
inline constexpr const char* kCamDataHeader = "#timestamp [ns],filename";
inline constexpr const char* kFrameMeta = "frame_meta.csv";
/// One row per stored frame; t_ns joins data.csv. focus_status = to_string(FocusStatus).
inline constexpr const char* kFrameMetaHeader =
    "t_ns,seq,t_sof_ns,t_arrival_ns,exposure_ns,analog_gain,digital_gain,readout_ns,sensor_mode,focus_position,"
    "focus_status";
inline constexpr const char* kDrops = "drops.csv";
/// One row per gap: t_ns of the first frame after it, its seq, frames missing, reason
/// (seq_gap | time_gap | backend_queue | error | writer).
inline constexpr const char* kDropsHeader = "t_ns,seq,missing,reason";

// raw/<imu id>/
inline constexpr const char* kImuData = "data.csv";
/// EuRoC imu0/data.csv header, verbatim. Rows: t_ns, gyro xyz [rad/s], accel xyz [m/s^2].
inline constexpr const char* kImuDataHeader =
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],"
    "a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]";
inline constexpr const char* kChip = "chip.csv";
/// The raw clock brackets (ImuBackend::read_brackets, T0 contract) for an offline re-fit.
inline constexpr const char* kChipHeader = "t_host_before_ns,chip_ticks,t_host_after_ns";
inline constexpr const char* kSampleMeta = "sample_meta.csv";  ///< OPTIONAL per-sample chip ticks/temp/flags.
inline constexpr const char* kSampleMetaHeader = "t_ns,chip_ticks,temp_c,flags";
inline constexpr const char* kClockFit = "clock_fit.json";

// raw/
inline constexpr const char* kEvents = "events.csv";
inline constexpr const char* kEventsHeader = "t_ns,type,detail";
inline constexpr const char* kLiveChecks = "live_checks.json";
inline constexpr const char* kLiveSlam = "live_slam.csv";
/// Pose = SlamStatus (W_T_C of the uw camera; q = w,x,y,z). Diagnostic only.
inline constexpr const char* kLiveSlamHeader =
    "t_ns,tracking_state,map_id,px,py,pz,qw,qx,qy,qz,heartbeat_seq,n_tracked_points";

}  // namespace layout
}  // namespace rig
