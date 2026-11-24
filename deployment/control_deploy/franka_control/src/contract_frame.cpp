// New (control_deploy/STRUCTURE.md 3.1). The rigid check and the re-normalisation are those franka_control applied to
// config.yaml's tcp.F_T_TCP at rnoro/deployment@91b178b (src/config_loader.cpp:402-408).
#include "franka_control/contract_frame.h"

#include <array>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include "franka_control/pose_math.h"

namespace franka_control {
namespace {

bool toNumber(const YAML::Node& n, double& out) {
  if (!n.IsScalar()) return false;
  const std::string s = n.Scalar();
  if (s == "true" || s == "false" || n.Tag() == "!") return false;   // a quoted string or a bool is not a number
  try {
    out = n.as<double>();
  } catch (const YAML::Exception&) {
    return false;
  }
  return std::isfinite(out);
}

}  // namespace

Pose ContractFrame::load(const std::filesystem::path& path, double rotation_tol, std::vector<std::string>& problems) {
  const std::string where = "frames.yaml (" + path.string() + ")";
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception& e) {
    problems.push_back(where + ": cannot read: " + e.what());
    return Pose::Identity();
  }
  if (!root.IsMap()) {
    problems.push_back(where + ": not a map");
    return Pose::Identity();
  }
  const YAML::Node m = root["F_T_TCP"];
  if (!m) {
    problems.push_back("frames.yaml: F_T_TCP: missing");
    return Pose::Identity();
  }
  const std::string expected = "frames.yaml: F_T_TCP: expected 4 rows of 4 finite numbers (row-major)";
  if (!m.IsSequence() || m.size() != 4) {
    problems.push_back(expected);
    return Pose::Identity();
  }
  std::array<double, 16> v{};
  for (size_t r = 0; r < 4; ++r) {
    if (!m[r].IsSequence() || m[r].size() != 4) {
      problems.push_back(expected);
      return Pose::Identity();
    }
    for (size_t c = 0; c < 4; ++c) {
      if (!toNumber(m[r][c], v[4 * r + c])) {
        problems.push_back(expected);
        return Pose::Identity();
      }
    }
  }
  if (!(rotation_tol > 0.0)) return Pose::Identity();   // limits.yaml is broken: its own problem is listed
  if (!isRigidRowMajor(v, rotation_tol)) {
    problems.push_back("frames.yaml: F_T_TCP: not a rigid transform (bottom row [0, 0, 0, 1], orthonormal rotation "
                       "with det > 0)");
    return Pose::Identity();
  }
  return normalized(poseFromRowMajor(v));
}

}  // namespace franka_control
