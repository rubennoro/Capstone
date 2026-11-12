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

}  // namespace franka_control
