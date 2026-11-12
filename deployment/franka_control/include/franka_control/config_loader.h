// Loads config.yaml, merges station.yaml's franka_control section over it (deploy STRUCTURE.md 6.3), reads the
// station's endpoints and session_root and the contract limits, and checks every value (STRUCTURE.md section 10).
#pragma once

#include <filesystem>
#include <string>

#include "franka_control/config.h"

namespace franka_control {

class ConfigLoader {
 public:
  // The plan capacity compiled in (WaypointTrajectory::kMaxWaypoints); limits.yaml's max_waypoints must fit.
  static Config load(const std::filesystem::path& station_file, bool mock, const std::filesystem::path& config_file,
                     const std::filesystem::path& limits_file);   // throws ConfigError listing every problem
  static std::string toYaml(const Config& c);                      // config_effective.yaml, --check-config output
};

}  // namespace franka_control
