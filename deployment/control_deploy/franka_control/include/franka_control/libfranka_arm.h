// from rnoro/deployment:deployment/franka_control/include/franka_control/libfranka_arm.h @ 91b178b
// changes: the sticky stop flag became a stop epoch (control_deploy/STRUCTURE.md 3.3; a change on the verified path,
// tested at bring-up, section 11.1). NOT COMPILED on the laptop (no libfranka): build on the PC.
// The real Panda through franka::Robot + franka::Model. Only built with libfranka.
#pragma once

#include <atomic>
#include <cstdint>
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
  // +1 per requestStop(). Each runTorqueControl/moveToJoints records it at its start and calls a ControlException
  // kStopRequested only if it changed during that call; otherwise it is a reflex (kReflex), as before.
  std::atomic<uint64_t> stop_epoch_{0};
};

}  // namespace franka_control
