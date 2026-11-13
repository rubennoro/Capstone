// libfranka 0.9.2 franka::Gripper calls (include/franka/gripper.h, gripper_state.h) checked against that tag.
#include "franka_control/franka_hand_gripper.h"

#include <algorithm>

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/gripper_state.h>

#include "franka_control/log.h"

namespace franka_control {

FrankaHandGripper::FrankaHandGripper(const std::string& robot_ip, const GripperConfig& cfg, double width_max_m)
    : robot_ip_(robot_ip), cfg_(cfg), width_max_m_(width_max_m) {}

FrankaHandGripper::~FrankaHandGripper() = default;

void FrankaHandGripper::prepare() {
  try {
    gripper_ = std::make_unique<franka::Gripper>(robot_ip_);
    if (cfg_.homing_on_start && !gripper_->homing()) throw GripperError("Franka Hand homing failed");
  } catch (const franka::Exception& e) {
    throw GripperError(std::string("Franka Hand at ") + robot_ip_ + ": " + e.what());
  }
}

double FrankaHandGripper::widthM() {
  try {
    std::lock_guard<std::mutex> lock(read_mutex_);
    return gripper_->readOnce().width + cfg_.width_offset_m;
  } catch (const franka::Exception& e) {
    throw GripperError(std::string("Franka Hand: ") + e.what());
  }
}

void FrankaHandGripper::moveTo(double width_m) {
  try {
    const franka::GripperState s = [this] {
      std::lock_guard<std::mutex> lock(read_mutex_);   // released before the blocking grasp/move below
      return gripper_->readOnce();
    }();
    const double hand = std::clamp(width_m - cfg_.width_offset_m, 0.0, std::min(width_max_m_, s.max_width));
    if (hand < s.width - cfg_.deadband_m) {
      // closing: hold with grasp_force wherever an object stops the fingers
      if (!gripper_->grasp(hand, cfg_.speed_m_per_s, cfg_.grasp_force_n, hand, s.max_width)) {
        Log::warning("Franka Hand: grasp to " + std::to_string(hand) + " m reported no grasp");
      }
    } else if (!gripper_->move(hand, cfg_.speed_m_per_s)) {
      Log::warning("Franka Hand: move to " + std::to_string(hand) + " m reported failure");
    }
  } catch (const franka::Exception& e) {
    throw GripperError(std::string("Franka Hand: ") + e.what());
  }
}

void FrankaHandGripper::stop() {
  if (!gripper_) return;
  try {
    gripper_->stop();
  } catch (const franka::Exception& e) {
    Log::warning(std::string("Franka Hand stop(): ") + e.what());
  }
}

}  // namespace franka_control
