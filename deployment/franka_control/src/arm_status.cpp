// Port of the still-relevant part of main-sim's derive_panda_error
// (rnoro/main-sim:controlSW/franka_deploy/franka_deploy/bridge/backends.py @ 4be9982): robot mode and current_errors.
// The last_motion_errors workaround is not ported: errors now arrive as libfranka exceptions.
#include "franka_control/arm_status.h"

namespace franka_control {
namespace {

std::string errorsText(const std::string& errs) {
  std::string t = errs;
  const auto b = t.find_first_not_of(" \t");
  const auto e = t.find_last_not_of(" \t");
  t = b == std::string::npos ? "" : t.substr(b, e - b + 1);
  if (t == "[]" || t == "None") return "";
  return t;
}

}  // namespace

std::string robotModeName(RobotMode mode) {
  switch (mode) {
    case RobotMode::kOther: return "Other";
    case RobotMode::kIdle: return "Idle";
    case RobotMode::kMove: return "Move";
    case RobotMode::kGuiding: return "Guiding";
    case RobotMode::kReflex: return "Reflex";
    case RobotMode::kUserStopped: return "UserStopped";
    case RobotMode::kAutomaticErrorRecovery: return "AutomaticErrorRecovery";
  }
  return "Other";
}

std::string describeArmProblem(const ArmStatusFlags& flags) {
  const std::string cur = errorsText(flags.current_errors);
  if (flags.mode == RobotMode::kReflex) {
    return "reflex (robot_mode Reflex, current_errors: " + (cur.empty() ? std::string("no error flags") : cur) + ")";
  }
  if (!cur.empty()) return "error flags (current_errors: " + cur + ")";
  switch (flags.mode) {
    case RobotMode::kIdle: return "";
    case RobotMode::kUserStopped: return "user stop pressed (robot_mode UserStopped)";
    case RobotMode::kMove: return "another motion is running (robot_mode Move)";
    case RobotMode::kGuiding: return "hand-guiding is active (robot_mode Guiding)";
    case RobotMode::kAutomaticErrorRecovery:
      return "error recovery is running (robot_mode AutomaticErrorRecovery)";
    case RobotMode::kOther:
    default: return "not ready (robot_mode Other: brakes locked or FCI not active?)";
  }
}

}  // namespace franka_control
