// from rnoro/deployment:deployment/franka_control/include/franka_control/fake_arm.h @ 91b178b
// changes: the stop epoch rule of LibfrankaArm (a stop affects only the call running at that moment); test hook
// statusAfterStop (control_deploy/STRUCTURE.md 3.3).
// A simulated arm for --mock and tests: a floating rigid body the real controller pushes.
// J_F = [I6 | 0] (joints 1-6 move the flange along x, y, z and about x, y, z; joint 7 does nothing), zero Coriolis,
// tau_J_d = the last torque. Each 1 ms it integrates force tau[0..2] and moment tau[3..5] (semi-implicit Euler).
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <string>

#include "franka_control/arm.h"
#include "franka_control/clock.h"
#include "franka_control/config.h"

namespace franka_control {

class FakeArm : public Arm {
 public:
  FakeArm(const FakeArmConfig& cfg, const Pose& F_T_TCP, const Vec7& q_home, Clock clock);

  ArmReading readOnce() override;
  ArmStatusFlags status() override;
  void runTorqueControl(const TorqueStep& step) override;
  void moveToJoints(const HomingMotion& m, const Observer& observe) override;
  void requestStop() override;
  std::string describe() const override;

  void injectFault(const std::string& text);       // tests: the next cycle throws ArmError(kReflex, text)
  void stallHoming(bool on);                       // tests: moveToJoints never finishes (homing timeout)
  void ignoreFinish(bool on);                      // tests: a torque session ends only on requestStop (shutdown)
  void statusAfterStop(ArmStatusFlags flags);      // tests: what status() reports once a stop was requested

 private:
  ArmReading reading() const;
  void waitNextCycle(int64_t& next_ns);            // clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)
  void throwIfInterrupted(uint64_t epoch);         // epoch: stop_epoch_ when the running call started
  void integrate(const Vec7& tau, double dt_s);

  FakeArmConfig cfg_;
  Pose F_T_TCP_;
  Vec7 q_home_;
  Clock clock_;
  // The body: flange position/rotation, linear/angular speed. Touched by the thread running the motion only.
  Eigen::Vector3d p_F_;
  Eigen::Matrix3d R_F_;
  Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d w_ = Eigen::Vector3d::Zero();
  Vec7 q_;
  Vec7 dq_ = Vec7::Zero();
  Vec7 tau_last_ = Vec7::Zero();

  std::atomic<uint64_t> stop_epoch_{0};            // +1 per requestStop()
  mutable std::mutex status_mutex_;
  ArmStatusFlags status_after_stop_{RobotMode::kIdle, "[]"};
  std::atomic<bool> fault_pending_{false};
  std::atomic<bool> stall_homing_{false};
  std::atomic<bool> ignore_finish_{false};
  std::array<char, 256> fault_text_{};
};

}  // namespace franka_control
