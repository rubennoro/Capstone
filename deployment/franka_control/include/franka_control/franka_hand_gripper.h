// The Franka Hand with our printed fingers, through franka::Gripper (STRUCTURE.md 9.3). Only built with libfranka.
#pragma once

#include <memory>
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
};

}  // namespace franka_control
