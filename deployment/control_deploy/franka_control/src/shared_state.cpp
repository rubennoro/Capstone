// from rnoro/deployment:deployment/franka_control/src/shared_state.cpp @ 91b178b
// changes: + the homing-abort flag and the hand-homing request and report (control_deploy/STRUCTURE.md 4.1); + the
// hand-homing and homing-move locks for the stops (4.4).
#include "franka_control/shared_state.h"

namespace franka_control {

SharedState::SharedState() = default;

void SharedState::postCommand(const LoopCommand& c) {
  commands_.writeSlot() = c;
  commands_.publish();
}

const LoopCommand* SharedState::takeCommand() { return commands_.refresh() ? &commands_.readSlot() : nullptr; }

LoopSnapshot& SharedState::snapshotSlot() { return snapshots_.writeSlot(); }

void SharedState::publishSnapshot() { snapshots_.publish(); }

const LoopSnapshot& SharedState::latestSnapshot() {
  snapshots_.refresh();
  return snapshots_.readSlot();
}

void SharedState::markAlive(int64_t now_ns) { last_alive_ns_.store(now_ns, std::memory_order_release); }
int64_t SharedState::lastAliveNs() const { return last_alive_ns_.load(std::memory_order_acquire); }

void SharedState::setGripperTarget(double w) { gripper_target_m_.store(w, std::memory_order_relaxed); }
double SharedState::gripperTarget() const { return gripper_target_m_.load(std::memory_order_relaxed); }

void SharedState::setGripperWidth(double w) { gripper_width_m_.store(w, std::memory_order_relaxed); }
double SharedState::gripperWidth() const { return gripper_width_m_.load(std::memory_order_relaxed); }

void SharedState::setGripperFault(std::string text) {
  std::lock_guard<std::mutex> lock(gripper_fault_mutex_);
  if (!gripper_fault_) gripper_fault_ = std::move(text);
}

std::optional<std::string> SharedState::gripperFault() const {
  std::lock_guard<std::mutex> lock(gripper_fault_mutex_);
  return gripper_fault_;
}

void SharedState::requestShutdown() { shutdown_.store(true, std::memory_order_release); }
bool SharedState::shutdownRequested() const { return shutdown_.load(std::memory_order_acquire); }

void SharedState::requestHomingAbort() { homing_abort_.store(true, std::memory_order_release); }
bool SharedState::homingAbortRequested() const { return homing_abort_.load(std::memory_order_acquire); }
void SharedState::clearHomingAbort() { homing_abort_.store(false, std::memory_order_release); }

void SharedState::requestHandHoming(uint64_t generation) {
  hand_homing_request_.store(generation, std::memory_order_release);
}
uint64_t SharedState::handHomingRequested() const { return hand_homing_request_.load(std::memory_order_acquire); }
void SharedState::reportHandHoming(uint64_t generation, uint64_t result) {
  hand_homing_report_.store(generation * 4 + result, std::memory_order_release);
}
uint64_t SharedState::handHomingReport() const { return hand_homing_report_.load(std::memory_order_acquire); }

bool SharedState::beginHandHoming(uint64_t generation) {
  std::lock_guard<std::mutex> lock(hand_homing_mutex_);
  if (hand_abort_generation_ == generation) {
    hand_ended_generation_ = generation;
    return false;
  }
  hand_running_generation_ = generation;
  return true;
}

void SharedState::endHandHoming(uint64_t generation) {
  std::lock_guard<std::mutex> lock(hand_homing_mutex_);
  hand_running_generation_ = 0;
  hand_ended_generation_ = generation;
}

SharedState::HandStop SharedState::stopHandHoming(uint64_t generation, const std::function<void()>& stop) {
  std::lock_guard<std::mutex> lock(hand_homing_mutex_);
  if (hand_ended_generation_ == generation) return HandStop::kAfterEnd;
  if (hand_running_generation_ == generation) {
    stop();
    return HandStop::kStopped;
  }
  hand_abort_generation_ = generation;
  return HandStop::kBeforeStart;
}

void SharedState::beginHomingMove(uint64_t session) {
  std::lock_guard<std::mutex> lock(homing_move_mutex_);
  homing_move_running_ = true;
  homing_move_session_ = session;
}

void SharedState::endHomingMove() {
  std::lock_guard<std::mutex> lock(homing_move_mutex_);
  homing_move_running_ = false;
}

bool SharedState::inHomingMove() const {
  std::lock_guard<std::mutex> lock(homing_move_mutex_);
  return homing_move_running_;
}

bool SharedState::stopHomingMove(uint64_t session, const std::function<void()>& stop) {
  std::lock_guard<std::mutex> lock(homing_move_mutex_);
  if (!homing_move_running_ || (session != kAnySession && session != homing_move_session_)) return false;
  stop();
  return true;
}

}  // namespace franka_control
