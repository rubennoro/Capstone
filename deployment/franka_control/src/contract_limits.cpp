#include "franka_control/contract_limits.h"

#include <cmath>

#include <yaml-cpp/yaml.h>

#include "franka_control/config.h"

namespace franka_control {

ContractLimits ContractLimits::load(const std::filesystem::path& limits_file, std::vector<std::string>& problems) {
  ContractLimits limits;
  const std::string where = "limits.yaml (" + limits_file.string() + ")";
  YAML::Node root;
  try {
    root = YAML::LoadFile(limits_file.string());
  } catch (const YAML::Exception& e) {
    problems.push_back(where + ": cannot read: " + e.what());
    return limits;
  }
  if (!root.IsMap()) {
    problems.push_back(where + ": not a map of limits");
    return limits;
  }
  auto number = [&](const char* key, double& out) {
    const YAML::Node n = root[key];
    if (!n) {
      problems.push_back(where + ": " + key + ": missing");
      return false;
    }
    try {
      out = n.as<double>();
    } catch (const YAML::Exception&) {
      problems.push_back(where + ": " + key + ": expected a number");
      return false;
    }
    if (!std::isfinite(out) || out <= 0.0) {
      problems.push_back(where + ": " + key + ": must be > 0");
      return false;
    }
    return true;
  };
  double max_waypoints = 0.0;
  if (number("max_waypoints", max_waypoints)) {
    if (max_waypoints != std::floor(max_waypoints)) {
      problems.push_back(where + ": max_waypoints: expected an integer");
    } else {
      limits.max_waypoints = static_cast<uint64_t>(max_waypoints);
    }
  }
  number("rotation_tol", limits.rotation_tol);
  number("gripper_width_max_m", limits.gripper_width_max_m);
  return limits;
}

ContractLimits ContractLimits::load(const std::filesystem::path& limits_file) {
  std::vector<std::string> problems;
  ContractLimits limits = load(limits_file, problems);
  if (!problems.empty()) throw ConfigError(problems);
  return limits;
}

}  // namespace franka_control
