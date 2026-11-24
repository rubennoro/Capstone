// from rnoro/deployment:deployment/franka_control/include/franka_control/supervisor.h @ 91b178b
// changes: onHome(Source), onHandHome(), onStop(Source) (aborts a homing), onKey(Key); the homing kind (arm or hand)
// and its source (C3 or the PC keyboard); tick() sends the arm stop once the move runs and resolves the outcomes
// (aborted, hand homing done, hand homing timeout); new last constructor parameter `Gripper* gripper = nullptr`
// (control_deploy/STRUCTURE.md 3.3, 4.4). Fix wave: the arm stop (abort or homing timeout) goes through
// SharedState::stopHomingMove and is re-sent every 100 ms while that move runs and its outcome is unresolved; the
// hand's stop goes through SharedState::stopHandHoming (never a finger move).
// + key p, the start pose (onStartPose): the arm's homing path to homing.q_start_rad from home, the hand opened to
// gripper_width_max_m meanwhile, completion checked against the box and the cone; C3 start refused away from the start
// pose once homing.q_start_rad is set.
// The messaging thread's decisions: owns the StateMachine, the SafetyGate, the watchdog and the current plan; turns
// the loop's, the gripper's and the keyboard's events into state changes; builds every C3 reply and C2 message.
#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "franka_control/arm.h"
#include "franka_control/clock.h"
#include "franka_control/config.h"
#include "franka_control/gripper.h"
#include "franka_control/key_queue.h"
#include "franka_control/messages.h"
#include "franka_control/safety_gate.h"
#include "franka_control/shared_state.h"
#include "franka_control/state_machine.h"
#include "franka_control/watchdog.h"

namespace franka_control {

// The prefix of a C3 home reply whose homing was stopped (deployment/STRUCTURE.md 3.7.2): exactly these 8 characters,
// then one space and the text. No other home reply of this process starts with it.
inline constexpr const char* kHomeStoppedPrefix = "stopped:";

class Supervisor {                                 // messaging thread
 public:
  enum class Source { kC3, kKeyboard };            // who asked (a C3 request, or the PC keyboard)

  // gripper: for the hand's homing (key g); nullptr refuses g ("no gripper").
  Supervisor(SharedState& shared, Arm& arm, const Config& cfg, const ContractLimits& limits, Clock clock,
             Gripper* gripper = nullptr);
  void tick();                                     // loop/gripper events, watchdog, plan end, homing progress
  bool watchdogExpiredNow();                       // primary check; changes state if expired
  std::variant<msg::SetWaypointsReply, msg::ErrorReply> onSetWaypoints(const msg::SetWaypoints& m);
  msg::Ack onHeartbeat();
  msg::Ack onStart();
  msg::Ack onStop(Source source = Source::kC3);    // never refused; in homing it aborts the move (outcome in tick())
  // nullopt: homing started; a C3 home gets its reply later through pollHome(), a key homing only a log line.
  std::optional<msg::Ack> onHome(Source source = Source::kC3);
  std::optional<msg::Ack> onHandHome();            // the hand's own homing (key g only); nullopt: started
  // Key p only: from home, the arm's homing move to homing.q_start_rad (mode homing) and the hand opened; nullopt:
  // started. Refused while q_start_rad is unset, away from home (homing.at_pose_tol_rad) or outside idle/stopped.
  std::optional<msg::Ack> onStartPose(Source source = Source::kKeyboard);
  void onKey(Key key);                             // h, p, g, s from the PC keyboard: acts and logs one line
  std::optional<msg::Ack> pollHome();              // the deferred reply, once homing ended (ok, failed, timeout)
  msg::ArmState stateMessage(uint64_t seq) const;  // C2 content (also the get_state reply)
  Mode mode() const { return sm_.mode(); }
  const std::string& modeDetail() const { return sm_.detail(); }
  const msg::GateCounters& gateCounters() const { return gate_counts_; }

 private:
  uint64_t post(LoopCommand::Kind kind, const WaypointTrajectory* plan = nullptr,
                const std::optional<Vec7>& q_target = std::nullopt,
                std::optional<double> open_width_m = std::nullopt);
  void toFault(Event e, const std::string& text);
  bool started() const { return sm_.mode() == Mode::kHolding || sm_.mode() == Mode::kRunning; }
  TrajectoryPoint snapshotTarget(int64_t now_ns) const;
  void logStats(const LoopSnapshot& s);
  void setHomeReply(bool ok, const std::string& reason);   // only for a C3-started homing
  void tickHoming(const LoopSnapshot& s, int64_t now);
  void abortHoming(Source source);
  void stopArmMove(const LoopSnapshot& s, int64_t now);   // arm.requestStop() while this homing's move runs
  std::string keyRefusal(const std::string& reason) const;

  enum class HomingKind { kArm, kHand };
  enum class ArmGoal { kHome, kStartPose };          // h (and C3 home), or p

  // The arm's homing path shared by h and p: the timeout check, the bookkeeping, the kHome command, the log line.
  std::optional<msg::Ack> startArmMove(Source source, ArmGoal goal, const Vec7& target,
                                       std::optional<double> open_width_m);
  static double maxJointError(const Vec7& a, const Vec7& b) { return (a - b).cwiseAbs().maxCoeff(); }

  SharedState& shared_;
  Arm& arm_;
  Config cfg_;
  Clock clock_;
  StateMachine sm_;
  SafetyGate gate_;
  Watchdog watchdog_;
  WaypointTrajectory plan_;                        // the plan the loop follows (holding/running)
  std::unique_ptr<LoopCommand> scratch_;           // ~5 KB, built here then copied into the shared slot
  uint64_t next_generation_ = 1;
  uint64_t follow_generation_ = 0;
  LoopSnapshot snap_;                              // the newest snapshot that passed the finiteness check
  msg::GateCounters gate_counts_;
  uint64_t home_generation_ = 0;
  uint64_t home_session_ = 0;
  int64_t home_start_ns_ = 0;
  std::optional<msg::Ack> home_reply_;
  int64_t next_stats_ns_ = 0;
  Gripper* gripper_ = nullptr;
  HomingKind homing_kind_ = HomingKind::kArm;
  ArmGoal arm_goal_ = ArmGoal::kHome;
  double gripper_open_m_ = 0.0;                    // contracts/limits.yaml gripper_width_max_m (p opens to it)
  Source homing_source_ = Source::kC3;
  bool abort_pending_ = false;                     // a stop arrived during this homing
  Source abort_source_ = Source::kC3;
  std::optional<int64_t> arm_stop_sent_ns_;        // the last arm.requestStop() for this homing's move
  bool timeout_stop_pending_ = false;              // homing timed out (fault): stop the move until it is over
  uint64_t hand_generation_ = 0;                   // the newest hand-homing request
};

}  // namespace franka_control
