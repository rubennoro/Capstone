// from rnoro/deployment:deployment/franka_control/include/franka_control/gripper_worker.h @ 91b178b
// changes: run() first serves a new hand-homing request (gripper.home() on this thread, the one that owns finger
// moves), reports done or aborted, then re-sends the current target even inside the deadband (control_deploy/
// STRUCTURE.md 3.3, 4.4).
// The gripper threads: run() follows gripper_target_m with a deadband (the gripper thread); runReader() reports the
// measured width at gripper.read_rate_hz (the gripper-reader thread), also during a move or the hand's homing.
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
  bool serveHandHoming();                          // a new hand-homing request, if any (tests); false after an error

 private:
  bool fail(const GripperError& e);                // sets the gripper fault, ends both loops; returns false

  Gripper& gripper_;
  SharedState& shared_;
  GripperConfig cfg_;
  Clock clock_;
  double last_sent_m_;                             // gripper thread only
  bool resend_ = false;                            // gripper thread only: send the target even inside the deadband
  uint64_t served_hand_generation_ = 0;            // gripper thread only
  std::atomic<bool> failed_{false};
};

}  // namespace franka_control
