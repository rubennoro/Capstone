// from rnoro/deployment:deployment/franka_control/include/franka_control/supervisor.h @ 91b178b
// changes: onHome(Source), onHandHome(), onStop(Source) (aborts a homing), onKey(Key); the homing kind (arm or hand)
// and its source (C3 or the PC keyboard); tick() sends the arm stop once the move runs and resolves the outcomes
// (aborted, hand homing done, hand homing timeout); new last constructor parameter `Gripper* gripper = nullptr`
// (control_deploy/STRUCTURE.md 3.3, 4.4).
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
  void onKey(Key key);                             // h, g, s from the PC keyboard: acts and logs one line
  std::optional<msg::Ack> pollHome();              // the deferred reply, once homing ended (ok, failed, timeout)
  msg::ArmState stateMessage(uint64_t seq) const;  // C2 content (also the get_state reply)
  Mode mode() const { return sm_.mode(); }
  const std::string& modeDetail() const { return sm_.detail(); }
  const msg::GateCounters& gateCounters() const { return gate_counts_; }

 private:
  uint64_t post(LoopCommand::Kind kind, const WaypointTrajectory* plan = nullptr);
  void toFault(Event e, const std::string& text);
  bool started() const { return sm_.mode() == Mode::kHolding || sm_.mode() == Mode::kRunning; }
  TrajectoryPoint snapshotTarget(int64_t now_ns) const;
  void logStats(const LoopSnapshot& s);
  void setHomeReply(bool ok, const std::string& reason);   // only for a C3-started homing
  void tickHoming(const LoopSnapshot& s, int64_t now);
  void abortHoming(Source source);
  std::string keyRefusal(const std::string& reason) const;

  enum class HomingKind { kArm, kHand };

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
  Source homing_source_ = Source::kC3;
  bool abort_pending_ = false;                     // a stop arrived during this homing
  Source abort_source_ = Source::kC3;
  bool arm_stop_sent_ = false;                     // arm.requestStop() sent for this abort
  uint64_t hand_generation_ = 0;                   // the newest hand-homing request
};

}  // namespace franka_control
