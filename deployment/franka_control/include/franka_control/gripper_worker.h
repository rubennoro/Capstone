// The gripper thread (STRUCTURE.md 9.3): follows gripper_target_m with a deadband, reports the measured width.
#pragma once

#include <functional>

#include "franka_control/clock.h"
#include "franka_control/config.h"
#include "franka_control/gripper.h"
#include "franka_control/shared_state.h"

namespace franka_control {

class GripperWorker {                              // gripper thread
 public:
  GripperWorker(Gripper& g, SharedState& shared, const GripperConfig& cfg, Clock clock);
  void run();                                      // until shutdown or an error
  bool stepOnce();                                 // one iteration (tests); false after a GripperError

 private:
  Gripper& gripper_;
  SharedState& shared_;
  GripperConfig cfg_;
  Clock clock_;
  double last_sent_m_;
};

}  // namespace franka_control
