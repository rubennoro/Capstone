// from rnoro/deployment:deployment/franka_control/include/franka_control/contract_limits.h @ 91b178b
// changes: the comment naming the old contracts/deploy paths.
// (`STRUCTURE.md n` below: rnoro/deployment's franka_control/STRUCTURE.md, old-S.)
// The contract numbers C++ needs, read from deployment/contracts/limits.yaml (deployment/STRUCTURE.md 4.1, 4.4).
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace franka_control {

struct ContractLimits {
  uint64_t max_waypoints = 0;
  double rotation_tol = 0.0;
  double gripper_width_max_m = 0.0;

  // Throws ConfigError listing every problem (missing file, missing key, wrong type, not > 0).
  static ContractLimits load(const std::filesystem::path& limits_file);
  // Same, but appends the problems to `problems` instead of throwing.
  static ContractLimits load(const std::filesystem::path& limits_file, std::vector<std::string>& problems);
};

}  // namespace franka_control
