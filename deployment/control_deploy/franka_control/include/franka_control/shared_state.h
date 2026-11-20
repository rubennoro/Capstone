// from rnoro/deployment:deployment/franka_control/include/franka_control/shared_state.h @ 91b178b
// changes: + LoopSnapshot::homing_aborted_generation; + the homing-abort flag (messaging -> control), the hand-homing
// request (messaging -> gripper) and its report (gripper -> messaging); none is touched by ControlLoop::step
// (control_deploy/STRUCTURE.md 4.1). Fix wave: the arm's homing move and the hand's homing each have a small lock, so
// a stop reaches only the move it is meant for (STRUCTURE.md 4.4). + LoopCommand::q_target and open_width_m (kHome:
// the start pose of key p, and the hand opening during that move).
// Where the threads meet. The 1 kHz loop uses only the wait-free parts.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>

#include "franka_control/triple_buffer.h"
#include "franka_control/types.h"
#include "franka_control/waypoint_trajectory.h"

namespace franka_control {

struct LoopCommand {                      // messaging -> control
  enum class Kind { kHold, kFollow, kHome };
  uint64_t generation = 0;                // +1 per posted command; the loop acts on each generation once
  Kind kind = Kind::kHold;
  WaypointTrajectory plan;                // used with kFollow
  std::optional<Vec7> q_target;           // kHome: the joint move's goal (nullopt: homing.q_home_rad)
  std::optional<double> open_width_m;     // kHome: the fingers' target during the move (nullopt: unchanged)
};

struct LoopStats {                        // the last completed window, computed by the loop itself
  uint64_t cycles = 0;
  int64_t period_max_ns = 0;
  int64_t compute_max_ns = 0;
  uint64_t missed_packets = 0;
  uint64_t late_cycles = 0;
  double success_rate_min = 1.0;
};

struct LoopSnapshot {                     // control -> messaging
  enum class Phase { kStarting, kTorque, kHoming, kEnded };
  int64_t t_ns = 0;                       // CLOCK_MONOTONIC when this reading arrived (C2 t_ns)
  uint64_t cycle = 0;                     // cycle counter
  uint64_t session = 0;                   // torque session counter (+1 after each homing)
  Phase phase = Phase::kStarting;
  Vec7 q = Vec7::Zero();
  Vec7 dq = Vec7::Zero();
  Pose O_T_TCP = Pose::Identity();
  Pose O_T_TCP_target = Pose::Identity();
  double gripper_target_m = 0.0;
  uint64_t applied_generation = 0;        // the command the loop is executing
  uint64_t watchdog_trip_generation = 0;  // 0, or the generation the loop's backstop watchdog stopped
  uint64_t homed_generation = 0;          // the kHome generation whose move has finished
  uint64_t homing_aborted_generation = 0; // 0, or the kHome generation whose homing was aborted (no move, or stopped)
  bool fault = false;
  std::array<char, 256> error{};          // fixed size: no allocation in the loop
  uint64_t late_cycles_total = 0;         // late cycles since start, all sessions (C2 loop_late)
  LoopStats stats;
};

class SharedState {
 public:
  SharedState();
  void postCommand(const LoopCommand& c);            // messaging
  const LoopCommand* takeCommand();                  // control: newest command, or nullptr if nothing new
  LoopSnapshot& snapshotSlot();                      // control: fill in place, then
  void publishSnapshot();                            // control
  const LoopSnapshot& latestSnapshot();              // messaging: newest complete snapshot
  void markAlive(int64_t now_ns);                    // messaging
  int64_t lastAliveNs() const;                       // control, messaging
  void setGripperTarget(double w);                   // control
  double gripperTarget() const;                      // gripper
  void setGripperWidth(double w);                    // gripper (and main at start-up)
  double gripperWidth() const;                       // messaging, control
  void setGripperFault(std::string text);            // gripper
  std::optional<std::string> gripperFault() const;   // messaging
  void requestShutdown();                            // main
  bool shutdownRequested() const;                    // all

  // Stop during an arm homing: messaging sets it, the control thread reads and clears it.
  void requestHomingAbort();                         // messaging
  bool homingAbortRequested() const;                 // control, messaging
  void clearHomingAbort();                           // control

  // The hand's homing (key g): the newest request generation (0 = none), and the gripper thread's report.
  enum HandHomingResult : uint64_t { kHandHomingDone = 1, kHandHomingAborted = 2 };
  void requestHandHoming(uint64_t generation);       // messaging
  uint64_t handHomingRequested() const;              // gripper
  void reportHandHoming(uint64_t generation, uint64_t result);   // gripper: one atomic, generation * 4 + result
  uint64_t handHomingReport() const;                 // messaging: 0, or generation * 4 + result

  // A stop must never reach a finger move (libfranka aborts it with an exception: a gripper fault), so it reaches the
  // hand only while home() runs for that request. The gripper thread brackets home() with begin/end; the messaging
  // thread's stop runs under the same lock, so home() cannot end and a finger move start in between.
  enum class HandStop { kBeforeStart, kStopped, kAfterEnd };
  bool beginHandHoming(uint64_t generation);         // gripper: false if a stop for it came first (skip home())
  void endHandHoming(uint64_t generation);           // gripper: home() returned or threw
  // messaging: kStopped (stop() ran, home() is running), kBeforeStart (home() will be skipped) or kAfterEnd (nothing).
  HandStop stopHandHoming(uint64_t generation, const std::function<void()>& stop);

  // The arm's homing move: the control thread marks it running around moveToJoints (outside the 1 kHz callbacks).
  // stop() (arm.requestStop()) runs under the same lock, so it cannot land in the torque session that follows the
  // move. kAnySession: whatever homing move runs (the shutdown).
  static constexpr uint64_t kAnySession = ~uint64_t{0};
  void beginHomingMove(uint64_t session);            // control: just before moveToJoints
  void endHomingMove();                              // control: moveToJoints returned or threw
  bool inHomingMove() const;                         // any
  bool stopHomingMove(uint64_t session, const std::function<void()>& stop);   // messaging, main: true if stop() ran

 private:
  static_assert(std::atomic<double>::is_always_lock_free, "atomic<double> must not hide a lock");
  static_assert(std::atomic<int64_t>::is_always_lock_free, "atomic<int64_t> must not hide a lock");
  static_assert(std::atomic<bool>::is_always_lock_free, "atomic<bool> must not hide a lock");
  static_assert(std::atomic<uint64_t>::is_always_lock_free, "atomic<uint64_t> must not hide a lock");

  TripleBuffer<LoopCommand> commands_;
  TripleBuffer<LoopSnapshot> snapshots_;
  std::atomic<int64_t> last_alive_ns_{0};
  std::atomic<double> gripper_target_m_{0.0};
  std::atomic<double> gripper_width_m_{0.0};
  mutable std::mutex gripper_fault_mutex_;           // gripper <-> messaging only; never the loop
  std::optional<std::string> gripper_fault_;
  std::atomic<bool> shutdown_{false};
  std::atomic<bool> homing_abort_{false};
  std::atomic<uint64_t> hand_homing_request_{0};
  std::atomic<uint64_t> hand_homing_report_{0};
  std::mutex hand_homing_mutex_;                     // gripper <-> messaging only; never the loop
  uint64_t hand_abort_generation_ = 0;               // a stop came before home() began
  uint64_t hand_running_generation_ = 0;             // home() runs for this request (0: none)
  uint64_t hand_ended_generation_ = 0;
  mutable std::mutex homing_move_mutex_;             // control (outside step/observeHoming) <-> messaging, main
  bool homing_move_running_ = false;
  uint64_t homing_move_session_ = 0;
};

// A homing stop is re-sent this often while its move runs and its outcome is unresolved (control_deploy/STRUCTURE.md
// 4.4): a stop sent just before libfranka's move call records its stop epoch would be lost.
constexpr int64_t kHomingStopResendNs = 100000000;

}  // namespace franka_control
