// Where the threads meet (STRUCTURE.md 3.2). The 1 kHz loop uses only the wait-free parts.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
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

 private:
  static_assert(std::atomic<double>::is_always_lock_free, "atomic<double> must not hide a lock");
  static_assert(std::atomic<int64_t>::is_always_lock_free, "atomic<int64_t> must not hide a lock");
  static_assert(std::atomic<bool>::is_always_lock_free, "atomic<bool> must not hide a lock");

  TripleBuffer<LoopCommand> commands_;
  TripleBuffer<LoopSnapshot> snapshots_;
  std::atomic<int64_t> last_alive_ns_{0};
  std::atomic<double> gripper_target_m_{0.0};
  std::atomic<double> gripper_width_m_{0.0};
  mutable std::mutex gripper_fault_mutex_;           // gripper <-> messaging only; never the loop
  std::optional<std::string> gripper_fault_;
  std::atomic<bool> shutdown_{false};
};

}  // namespace franka_control
