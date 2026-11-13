// The gripper threads (STRUCTURE.md 9.3): run() follows gripper_target_m with a deadband (the gripper thread);
// runReader() reports the measured width at gripper.read_rate_hz (the gripper-reader thread), also during a move.
#pragma once

#include <atomic>
#include <functional>

#include "franka_control/clock.h"
#include "franka_control/config.h"
#include "franka_control/gripper.h"
#include "franka_control/shared_state.h"

namespace franka_control {

class GripperWorker {
 public:
  GripperWorker(Gripper& g, SharedState& shared, const GripperConfig& cfg, Clock clock);
  void run();                                      // gripper thread: until shutdown or an error (either loop's)
  void runReader();                                // gripper-reader thread: until shutdown or an error
  bool stepOnce();                                 // one command iteration (tests); false after a GripperError
  bool readStep();                                 // one width read into SharedState (tests); false after one

 private:
  bool fail(const GripperError& e);                // sets the gripper fault, ends both loops; returns false

  Gripper& gripper_;
  SharedState& shared_;
  GripperConfig cfg_;
  Clock clock_;
  double last_sent_m_;                             // gripper thread only
  std::atomic<bool> failed_{false};
};

}  // namespace franka_control
