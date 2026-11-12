// The real Panda through franka::Robot + franka::Model (STRUCTURE.md 5.2, 5.6, 7.3). Only built with libfranka.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "franka_control/arm.h"
#include "franka_control/config.h"

namespace franka {
class Robot;
class Model;
}  // namespace franka

namespace franka_control {

class LibfrankaArm : public Arm {
 public:
  // Connects, checks the arm is ready (describeArmProblem), sets collision behaviour, joint impedance and the load,
  // loads the model. Throws ArmError with a one-line "what failed: what to check" text.
  LibfrankaArm(const std::string& robot_ip, RealtimeMode mode, const ControlConfig& control, const LoadConfig& load);
  ~LibfrankaArm() override;

  ArmReading readOnce() override;
  ArmStatusFlags status() override;
  void runTorqueControl(const TorqueStep& step) override;
  void moveToJoints(const HomingMotion& m, const Observer& observe) override;
  void requestStop() override;
  std::string describe() const override;

 private:
  std::string robot_ip_;
  RealtimeMode mode_;
  ControlConfig control_;
  std::unique_ptr<franka::Robot> robot_;
  std::unique_ptr<franka::Model> model_;
  std::atomic<bool> stop_requested_{false};
};

}  // namespace franka_control
