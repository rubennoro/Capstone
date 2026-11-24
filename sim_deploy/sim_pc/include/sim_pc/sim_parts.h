// sim_deploy/sim_pc (E4): SimArm and SimGripper, franka_control's Arm and Gripper backed by one SimWorld, and the
// factory app.cpp calls under --sim (DESIGN.md section 8).
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "franka_control/arm.h"
#include "franka_control/config.h"
#include "franka_control/gripper.h"
#include "franka_control/key_queue.h"

namespace sim_pc {

class SimWorld;

// Arm: each call is served by the physics thread, one physics step per 1 ms cycle; libfranka's semantics:
// requestStop() ends only the call running at that moment (kStopRequested), a reflex ends it with kReflex and stays
// until restart (status() kReflex), a physics thread more than realtime_error_lag_ms behind ends it with kRealtime.
class SimArm : public franka_control::Arm {
 public:
  explicit SimArm(std::shared_ptr<SimWorld> world);
  franka_control::ArmReading readOnce() override;
  franka_control::ArmStatusFlags status() override;
  void runTorqueControl(const TorqueStep& step) override;
  void moveToJoints(const franka_control::HomingMotion& m, const Observer& observe) override;
  void requestStop() override;
  std::string describe() const override;

 private:
  std::shared_ptr<SimWorld> world_;
};

// Gripper: the Franka Hand's fingers with libfranka's move/grasp semantics and FrankaHandGripper's width mapping:
// widthM() = fingertip gap + gripper.width_offset_m; moveTo(w) moves the gap to clamp(w - offset, 0, min(width max,
// stroke)), as a grasp (grasp_force_n, held on the object) when closing by more than the deadband, else a move.
class SimGripper : public franka_control::Gripper {
 public:
  SimGripper(std::shared_ptr<SimWorld> world, const franka_control::GripperConfig& cfg, double width_max_m,
             double homing_speed_m_per_s, double move_force_n);
  void prepare() override;
  double widthM() override;
  void moveTo(double width_m) override;
  void stop() override;
  bool home() override;
  void clearStop() override;

 private:
  enum class Outcome { kDone, kStopped, kFailed };
  Outcome run(double gap_m, double speed, double force, bool grasp);

  std::shared_ptr<SimWorld> world_;
  franka_control::GripperConfig cfg_;
  double width_max_m_;
  double homing_speed_;
  double move_force_;
  std::atomic<bool> stop_{false};
};

struct SimParts {
  std::unique_ptr<franka_control::Arm> arm;
  std::unique_ptr<franka_control::Gripper> gripper;
  // the scene's {"cmd": "key"} pushes into this KeyQueue (app.cpp: the messaging thread's); nullptr detaches
  std::function<void(franka_control::KeyQueue*)> attach_keys;
};

// Loads the station's sim: block and the model, starts the physics and io threads. Throws std::runtime_error.
SimParts makeSim(const std::filesystem::path& station_file, const franka_control::Config& cfg);

}  // namespace sim_pc
