// The messaging thread's decisions (STRUCTURE.md 8.2): owns the StateMachine, the SafetyGate, the watchdog and the
// current plan; turns the loop's and the gripper's events into state changes; builds every C3 reply and C2 message.
#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "franka_control/arm.h"
#include "franka_control/clock.h"
#include "franka_control/config.h"
#include "franka_control/messages.h"
#include "franka_control/safety_gate.h"
#include "franka_control/shared_state.h"
#include "franka_control/state_machine.h"
#include "franka_control/watchdog.h"

namespace franka_control {

class Supervisor {                                 // messaging thread
 public:
  Supervisor(SharedState& shared, Arm& arm, const Config& cfg, const ContractLimits& limits, Clock clock);
  void tick();                                     // loop/gripper events, watchdog, plan end, homing progress
  bool watchdogExpiredNow();                       // primary check; changes state if expired
  std::variant<msg::SetWaypointsReply, msg::ErrorReply> onSetWaypoints(const msg::SetWaypoints& m);
  msg::Ack onHeartbeat();
  msg::Ack onStart();
  msg::Ack onStop();
  std::optional<msg::Ack> onHome();                // nullopt: homing started, reply later
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
};

}  // namespace franka_control
