// from rnoro/deployment:deployment/franka_control/include/franka_control/control_loop.h @ 91b178b
// changes: run() only: a homing abort before the move skips it; a stop-requested end of the move with the abort flag
// set waits for the robot to leave its move mode and starts a new torque session instead of ending the thread
// (control_deploy/STRUCTURE.md 3.3, 4.4); the homing move is marked in SharedState (begin/endHomingMove) instead of
// in_homing_ (fix wave). step() and observeHoming() are unchanged.
// The control thread: torque sessions, the 1 kHz step, homing moves between sessions.
// step() never blocks, allocates, locks, logs or throws on purpose; it talks to the other threads only through the
// wait-free parts of SharedState.
#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

#include "franka_control/arm.h"
#include "franka_control/clock.h"
#include "franka_control/config.h"
#include "franka_control/impedance_controller.h"
#include "franka_control/shared_state.h"
#include "franka_control/watchdog.h"

namespace franka_control {

class ControlLoop {
 public:
  ControlLoop(Arm& arm, SharedState& shared, const Config& cfg, Clock clock);

  void run();                                      // the control thread's body; returns when the loop has ended
  void beginSession();                             // the next step() is the first cycle of a torque session
  TorqueCommand step(const ArmReading& r);         // one 1 kHz cycle (called by Arm::runTorqueControl)
  void observeHoming(const ArmReading& r);         // one cycle of a homing move (keeps the snapshot fresh)

  bool done() const { return done_.load(); }       // run() has returned
  std::optional<ArmError::Kind> failureKind() const;   // why run() ended, if an ArmError ended it

 private:
  enum class Follow { kHold, kPlan };
  enum class Ending { kNone, kHome, kShutdown };

  void holdCurrent();
  void setError(const char* text);
  void publish();
  void endWith(ArmError::Kind kind, const char* what);
  std::string waitForArmAfterStop();                // "" when the robot left its move mode and reports no problem
  void abortedHoming();                             // the homing is over without reaching home: a new session holds

  Arm& arm_;
  SharedState& shared_;
  Clock clock_;
  Pose F_T_TCP_;
  ImpedanceController controller_;
  Watchdog backstop_;
  HomingConfig homing_;
  int64_t late_period_ns_;
  int64_t window_ns_;
  double settle_speed_;
  int64_t settle_timeout_ns_;

  // loop state (control thread only)
  bool first_cycle_ = true;
  Follow follow_ = Follow::kHold;
  const WaypointTrajectory* plan_ = nullptr;   // points into the command read slot; valid until the next refresh
  Target held_;
  Target last_target_;
  Vec7 last_q_ = Vec7::Zero();
  uint64_t applied_generation_ = 0;
  uint64_t watchdog_trip_generation_ = 0;
  uint64_t home_generation_ = 0;
  uint64_t homed_generation_ = 0;
  uint64_t homing_aborted_generation_ = 0;
  Ending ending_ = Ending::kNone;
  int64_t ending_start_ns_ = 0;
  bool have_prev_ = false;
  int64_t prev_t_ns_ = 0;
  uint64_t cycle_ = 0;
  uint64_t session_ = 0;
  uint64_t late_total_ = 0;
  bool fault_ = false;
  LoopStats window_;
  LoopStats last_window_;
  int64_t window_start_ns_ = 0;
  bool window_started_ = false;
  LoopSnapshot cur_;                               // filled every cycle, copied into the snapshot slot

  std::atomic<bool> done_{false};
  std::atomic<int> failure_kind_{-1};
};

}  // namespace franka_control
