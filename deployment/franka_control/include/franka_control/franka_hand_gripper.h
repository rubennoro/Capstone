// The Franka Hand with our printed fingers, through franka::Gripper (STRUCTURE.md 9.3). Only built with libfranka.
// One connection, used by two threads: the gripper thread (moveTo, stop) and the gripper-reader thread (widthM).
// franka::Gripper's members are thread-safe (its header), except that readOnce throws if another readOnce is
// running, so read_mutex_ lets only one readOnce run at a time. It is never held during grasp, move or stop.
#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "franka_control/config.h"
#include "franka_control/gripper.h"

namespace franka {
class Gripper;
}  // namespace franka

namespace franka_control {

class FrankaHandGripper : public Gripper {
 public:
  FrankaHandGripper(const std::string& robot_ip, const GripperConfig& cfg, double width_max_m);
  ~FrankaHandGripper() override;
  void prepare() override;
  double widthM() override;
  void moveTo(double width_m) override;
  void stop() override;

 private:
  std::string robot_ip_;
  GripperConfig cfg_;
  double width_max_m_;
  std::unique_ptr<franka::Gripper> gripper_;
  std::mutex read_mutex_;                          // around every gripper_->readOnce()
};

}  // namespace franka_control
