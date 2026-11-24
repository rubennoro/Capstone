// sim_deploy/sim_pc (E4): SimArm, SimGripper, makeSim (sim_parts.h).
#include "sim_pc/sim_parts.h"

#include <algorithm>
#include <chrono>
#include <thread>

#include "franka_control/log.h"
#include "sim_pc/sim_config.h"
#include "sim_pc/sim_world.h"

namespace sim_pc {

using franka_control::GripperError;
using franka_control::Log;

// ---- SimArm ----

SimArm::SimArm(std::shared_ptr<SimWorld> world) : world_(std::move(world)) {}

franka_control::ArmReading SimArm::readOnce() { return world_->latestReading(); }

franka_control::ArmStatusFlags SimArm::status() { return world_->status(); }

void SimArm::runTorqueControl(const TorqueStep& step) {
  ArmCall call;
  call.kind = ArmCall::Kind::kTorque;
  call.step = &step;
  call.epoch = world_->stopEpoch();
  world_->runCall(call);
}

void SimArm::moveToJoints(const franka_control::HomingMotion& m, const Observer& observe) {
  ArmCall call;
  call.kind = ArmCall::Kind::kJoints;
  call.motion = &m;
  call.observe = &observe;
  call.epoch = world_->stopEpoch();
  world_->runCall(call);
}

void SimArm::requestStop() { world_->requestStop(); }

std::string SimArm::describe() const {
  return "SimArm (MuJoCo, model sha256 " + world_->modelSha().substr(0, 16) + "...)";
}

// ---- SimGripper ----

SimGripper::SimGripper(std::shared_ptr<SimWorld> world, const franka_control::GripperConfig& cfg, double width_max_m,
                       double homing_speed_m_per_s, double move_force_n)
    : world_(std::move(world)), cfg_(cfg), width_max_m_(width_max_m), homing_speed_(homing_speed_m_per_s),
      move_force_(move_force_n) {}

void SimGripper::prepare() {
  if (cfg_.homing_on_start && !home()) throw GripperError("Franka Hand homing failed (sim)");
}

double SimGripper::widthM() { return world_->gapM() + cfg_.width_offset_m; }

SimGripper::Outcome SimGripper::run(double gap_m, double speed, double force, bool grasp) {
  const uint64_t id = world_->postGrip(gap_m, speed, force, grasp);
  for (;;) {
    if (stop_.load()) world_->abortGrip(id);
    switch (world_->gripResult(id)) {
      case SimWorld::GripResult::kPending:
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      case SimWorld::GripResult::kReached:
      case SimWorld::GripResult::kStalled:
        return Outcome::kDone;
      case SimWorld::GripResult::kAborted:
        return Outcome::kStopped;
      case SimWorld::GripResult::kTimeout:
        return Outcome::kFailed;
    }
  }
}

void SimGripper::moveTo(double width_m) {
  stop_.store(false);   // like the hand: stop() ends a running move only
  // FrankaHandGripper::moveTo: hand = clamp(width - width_offset_m, 0, min(width_max_m, max_width))
  const double hand = std::clamp(width_m - cfg_.width_offset_m, 0.0, std::min(width_max_m_, world_->maxGapM()));
  const double now = world_->gapM();
  Outcome o;
  if (hand < now - cfg_.deadband_m) {
    // grasp: held wherever an object stops it. W0 opt-in sim.gripper.grasp_closes_fully: like the Franka Hand's
    // force-controlled grasp (the real arm, 2026-11-20: the fingers closed flush with nothing between them), it closes
    // until the fingers stall, not to the commanded width
    o = run(world_->graspClosesFully() ? 0.0 : hand, cfg_.speed_m_per_s, cfg_.grasp_force_n, true);
  } else {
    o = run(hand, cfg_.speed_m_per_s, move_force_, false);
  }
  if (o == Outcome::kStopped) throw GripperError("SimGripper: finger move aborted by stop()");
  if (o == Outcome::kFailed) Log::warning("SimGripper: move to " + std::to_string(hand) + " m reported failure");
}

bool SimGripper::home() {
  if (stop_.load()) return false;   // stopped before it began: the fingers do not move
  for (const double gap : {0.0, world_->maxGapM()}) {   // the hand closes fully, then opens fully
    const Outcome o = run(gap, homing_speed_, move_force_, false);
    if (o == Outcome::kStopped) return false;
    if (o == Outcome::kFailed) throw GripperError("Franka Hand homing failed (sim: fingers did not reach the stop)");
  }
  return true;
}

void SimGripper::stop() { stop_.store(true); }

void SimGripper::clearStop() { stop_.store(false); }

// ---- factory ----

SimParts makeSim(const std::filesystem::path& station_file, const franka_control::Config& cfg) {
  const SimConfig sim = SimConfig::load(station_file);
  auto world = std::make_shared<SimWorld>(sim, cfg);
  world->start();
  SimParts parts;
  parts.arm = std::make_unique<SimArm>(world);
  parts.gripper = std::make_unique<SimGripper>(world, cfg.gripper, cfg.limits.gripper_width_max_m,
                                               sim.gripper_homing_speed_m_per_s, sim.gripper_move_force_n);
  std::weak_ptr<SimWorld> weak = world;   // the arm and the gripper own the world, not this hook
  parts.attach_keys = [weak](franka_control::KeyQueue* keys) {
    if (auto w = weak.lock()) w->attachKeys(keys);
  };
  return parts;
}

}  // namespace sim_pc
