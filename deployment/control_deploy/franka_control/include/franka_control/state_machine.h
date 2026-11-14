// from rnoro/deployment:deployment/franka_control/include/franka_control/state_machine.h @ 91b178b
// changes: + kHomingAborted (homing -> stopped); kHome is used for both kinds of homing (arm and hand); kStop's comment
// (control_deploy/STRUCTURE.md 3.3, 5.2).
// idle / running / holding / stopped / homing / fault and the allowed moves between them.
// Messaging thread only; pure (no clock, no I/O). A fault is never cleared.
#pragma once

#include <string>

namespace franka_control {

enum class Mode { kIdle, kRunning, kHolding, kStopped, kHoming, kFault };

enum class Event {
  // C3 requests
  kHeartbeat,
  kGetState,
  kStart,
  kStop,
  kHome,                    // C3 home or the PC's h (arm), the PC's g (hand): the same rules
  kSetWaypoints,            // the request itself: allowed in holding/running only (4.3)
  // outcomes and events
  kWaypointsAccepted,       // >= 1 accepted -> running
  kWaypointsAllSkipped,     // accepted 0, skipped N -> holding
  kWaypointsRejected,       // the gate rejected one -> fault
  kPlanEnded,               // now >= the plan's last time -> holding
  kWatchdog,                // no C3 request for limit_s (or the loop's backstop) -> stopped
  kHomingDone,              // homing -> idle
  kHomingFailed,            // homing -> fault (move failed, timeout, home TCP outside the box)
  kHomingAborted,           // homing -> stopped (a C3 stop or the PC's s stopped the arm's move or the hand's homing)
  kLoopFault,               // libfranka exception, non-finite torque or state -> fault
  kGripperFault,            // -> fault
};

struct Transition {
  bool allowed = false;     // false: refused, nothing changes (reason says why)
  Mode from = Mode::kIdle;
  Mode to = Mode::kIdle;
  std::string reason;       // refusals: one line, e.g. "refused: mode is running: send stop first"
};

std::string modeName(Mode mode);

class StateMachine {
 public:
  Mode mode() const { return mode_; }
  // The text that explains the mode: the fault text in fault, the cause in stopped ("stop", "watchdog: ..."), else "".
  const std::string& detail() const { return detail_; }
  Transition check(Event e) const;                            // what apply() would do; changes nothing
  Transition apply(Event e, const std::string& detail = "");  // detail: for fault and stopped

 private:
  Mode mode_ = Mode::kIdle;
  std::string detail_;
};

}  // namespace franka_control
