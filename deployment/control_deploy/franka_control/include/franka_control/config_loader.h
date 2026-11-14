// from rnoro/deployment:deployment/franka_control/include/franka_control/config_loader.h @ 91b178b
// changes: load() takes --local and the three contract files; F_T_TCP from frames.yaml, the endpoints and the watchdog
// limit from link.yaml; a station `endpoints:` section is refused (control_deploy/STRUCTURE.md 3.1).
// Loads config.yaml, merges the station's franka_control section over it, reads the station's session_root and the
// contract files (limits.yaml, frames.yaml, link.yaml), and checks every value (control_deploy/STRUCTURE.md 7).
#pragma once

#include <filesystem>
#include <string>

#include "franka_control/config.h"

namespace franka_control {

// The three contract files franka_control reads at run time (baked-in paths, or a test's temporary copies).
struct ContractFiles {
  std::filesystem::path limits;
  std::filesystem::path frames;
  std::filesystem::path link;
};

// The baked-in paths (CONTRACTS_LIMITS_FILE, CONTRACTS_FRAMES_FILE, CONTRACTS_LINK_FILE).
ContractFiles bakedContractFiles();

class ConfigLoader {
 public:
  // Throws ConfigError listing every problem. limits.yaml's max_waypoints must fit the compiled plan capacity
  // (WaypointTrajectory::kMaxWaypoints).
  static Config load(const std::filesystem::path& station_file, bool mock, bool local,
                     const std::filesystem::path& config_file, const ContractFiles& files);
  static std::string toYaml(const Config& c);                      // config_effective.yaml, --check-config output
};

}  // namespace franka_control
