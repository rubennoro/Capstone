// The robot's mode and error flags, and the port of main-sim's derive_panda_error (STRUCTURE.md 9.2).
#pragma once

#include <string>

namespace franka_control {

// libfranka's franka::RobotMode, without depending on libfranka.
enum class RobotMode { kOther, kIdle, kMove, kGuiding, kReflex, kUserStopped, kAutomaticErrorRecovery };

struct ArmStatusFlags {
  RobotMode mode = RobotMode::kIdle;
  std::string current_errors;   // libfranka's std::string(Errors): "[]" (or "") when there are none
};

std::string robotModeName(RobotMode mode);

// One line describing why the arm can't be used, or "" when it can.
// Reflex mode wins over error flags; then current_errors; then any mode other than Idle.
std::string describeArmProblem(const ArmStatusFlags& flags);

}  // namespace franka_control
