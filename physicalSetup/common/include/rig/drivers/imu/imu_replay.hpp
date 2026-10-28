// SPDX-License-Identifier: MIT
/// @file rig/drivers/imu/imu_replay.hpp
/// @brief IMU replay backend ("replay", T1): EuRoC imu0/data.csv or a package raw/imu0/data.csv.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rig/drivers/imu_backend.hpp"

namespace rig {

/// Parses `path` (a data.csv, or a directory containing one): t_ns, w_xyz [rad/s], a_xyz [m/s^2]; '#' comments.
Result<std::vector<ImuSample>> load_euroc_imu_csv(const std::string& path);

/// Replay backend over config.replay (path, realtime pacing, retime to now_raw_ns() at start()).
Result<std::unique_ptr<ImuBackend>> make_imu_replay(const ImuConfig& config);

}  // namespace rig
