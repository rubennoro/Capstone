// New (control_deploy/STRUCTURE.md 3.1): F_T_TCP from contracts/frames.yaml (deployment/STRUCTURE.md 4.2, 4.4).
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "franka_control/types.h"

namespace franka_control {

class ContractFrame {
 public:
  // F_T_TCP: 4 rows of 4 finite numbers, rigid within rotation_tol, returned re-normalised through a unit quaternion.
  // `settled` and any other key are ignored. Problems are appended ("frames.yaml: F_T_TCP: ..."); Identity then.
  static Pose load(const std::filesystem::path& path, double rotation_tol, std::vector<std::string>& problems);
};

}  // namespace franka_control
