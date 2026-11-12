// The contract numbers C++ needs, read from contracts/contracts/limits.yaml (contracts STRUCTURE.md 4.7, 8).
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
