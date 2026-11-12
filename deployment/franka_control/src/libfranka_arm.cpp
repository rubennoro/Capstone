// libfranka 0.9.2 calls used here were checked against that tag's headers (include/franka/robot.h, model.h,
// exception.h, control_types.h, robot_state.h, duration.h, lowpass_filter.h): see STRUCTURE.md section 15.
#include "franka_control/libfranka_arm.h"

#include <array>
#include <optional>
#include <stdexcept>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include "franka_control/log.h"
#include "franka_control/pose_math.h"

namespace franka_control {
namespace {

RobotMode toMode(franka::RobotMode m) {
  switch (m) {
    case franka::RobotMode::kIdle: return RobotMode::kIdle;
    case franka::RobotMode::kMove: return RobotMode::kMove;
    case franka::RobotMode::kGuiding: return RobotMode::kGuiding;
    case franka::RobotMode::kReflex: return RobotMode::kReflex;
    case franka::RobotMode::kUserStopped: return RobotMode::kUserStopped;
    case franka::RobotMode::kAutomaticErrorRecovery: return RobotMode::kAutomaticErrorRecovery;
    case franka::RobotMode::kOther:
    default: return RobotMode::kOther;
  }
}

// One cycle's ArmReading from libfranka's state and model (column-major arrays -> Eigen, once, here).
ArmReading fill(const franka::RobotState& s, const franka::Model& model, franka::Duration period) {
  ArmReading r;
  r.q = Eigen::Map<const Vec7>(s.q.data());
  r.dq = Eigen::Map<const Vec7>(s.dq.data());
  r.tau_J_d = Eigen::Map<const Vec7>(s.tau_J_d.data());
  const std::array<double, 7> coriolis = model.coriolis(s);
  r.coriolis = Eigen::Map<const Vec7>(coriolis.data());
  r.O_T_F = poseFromColumnMajor(model.pose(franka::Frame::kFlange, s));
  const std::array<double, 42> J = model.zeroJacobian(franka::Frame::kFlange, s);
  r.J_F = Eigen::Map<const Mat67>(J.data());   // column-major 6x7, rows [linear; angular]
  r.success_rate = s.control_command_success_rate;
  const uint64_t ms = period.toMSec();
  r.missed_packets = ms > 1 ? static_cast<uint32_t>(ms - 1) : 0;
  return r;
}

std::array<double, 7> toArray(const Vec7& v) {
  std::array<double, 7> a{};
  for (int i = 0; i < 7; ++i) a[static_cast<size_t>(i)] = v(i);
  return a;
}

}  // namespace

LibfrankaArm::LibfrankaArm(const std::string& robot_ip, RealtimeMode mode, const ControlConfig& control,
                           const LoadConfig& load)
    : robot_ip_(robot_ip), mode_(mode), control_(control) {
  const franka::RealtimeConfig rt =
      mode == RealtimeMode::kEnforce ? franka::RealtimeConfig::kEnforce : franka::RealtimeConfig::kIgnore;
  try {
    robot_ = std::make_unique<franka::Robot>(robot_ip, rt);
  } catch (const franka::IncompatibleVersionException& e) {
    throw ArmError(ArmError::Kind::kOther,
                   "libfranka " FRANKA_CONTROL_LIBFRANKA_VERSION
                   " does not match the robot's system version (" + std::string(e.what()) +
                       "): check Franka's compatibility table, set libfranka_version in config.yaml, rerun build.sh");
  } catch (const franka::NetworkException& e) {
    throw ArmError(ArmError::Kind::kNetwork, "cannot connect to the arm at " + robot_ip + " (libfranka: " +
                                                 e.what() + "): check the cable, the PC's static IP, FCI enabled in Desk");
  } catch (const franka::Exception& e) {
    throw ArmError(ArmError::Kind::kOther, std::string("libfranka: ") + e.what());
  }
  // franka_control never calls automaticErrorRecovery(): clearing a reflex is the operator's act.
  const std::string problem = describeArmProblem(status());
  if (!problem.empty()) {
    throw ArmError(ArmError::Kind::kReflex, "arm reports " + problem +
                                                ": press and release the user stop, check Desk, then rerun deploy run");
  }
  try {
    robot_->setCollisionBehavior(control.collision_torque_thresholds_nm, control.collision_torque_thresholds_nm,
                                 control.collision_force_thresholds_n, control.collision_force_thresholds_n);
    robot_->setJointImpedance(control.joint_impedance);
    if (load.mass_kg > 0.0) {
      // libfranka takes the inertia column-major; config.yaml's is symmetric, so row-major is the same numbers.
      robot_->setLoad(load.mass_kg, load.com_m, load.inertia_kg_m2);
    }
    model_ = std::make_unique<franka::Model>(robot_->loadModel());
  } catch (const franka::Exception& e) {
    throw ArmError(ArmError::Kind::kCommand, std::string("libfranka: arm set-up failed: ") + e.what());
  }
}

LibfrankaArm::~LibfrankaArm() = default;

std::string LibfrankaArm::describe() const {
  return "LibfrankaArm at " + robot_ip_ + ", realtime " + realtimeName(mode_);
}

ArmStatusFlags LibfrankaArm::status() {
  try {
    const franka::RobotState s = robot_->readOnce();
    return ArmStatusFlags{toMode(s.robot_mode), static_cast<std::string>(s.current_errors)};
  } catch (const franka::Exception& e) {
    throw ArmError(ArmError::Kind::kNetwork, std::string("libfranka: ") + e.what());
  }
}

ArmReading LibfrankaArm::readOnce() {
  try {
    const franka::RobotState s = robot_->readOnce();
    return fill(s, *model_, franka::Duration(0));
  } catch (const franka::Exception& e) {
    throw ArmError(ArmError::Kind::kNetwork, std::string("libfranka: ") + e.what());
  }
}

namespace {

[[noreturn]] void rethrow(const std::atomic<bool>& stop_requested) {
  try {
    throw;
  } catch (const franka::ControlException& e) {
    std::string text = std::string("libfranka: ") + e.what();
    if (!e.log.empty()) {
      const std::string last = static_cast<std::string>(e.log.back().state.last_motion_errors);
      if (last != "[]" && !last.empty()) text += " (last_motion_errors: " + last + ")";
    }
    throw ArmError(stop_requested.load() ? ArmError::Kind::kStopRequested : ArmError::Kind::kReflex, text);
  } catch (const franka::NetworkException& e) {
    throw ArmError(ArmError::Kind::kNetwork, std::string("libfranka: connection lost: ") + e.what());
  } catch (const franka::RealtimeException& e) {
    throw ArmError(ArmError::Kind::kRealtime, std::string("libfranka: ") + e.what());
  } catch (const franka::CommandException& e) {
    throw ArmError(ArmError::Kind::kCommand, std::string("libfranka: ") + e.what());
  } catch (const franka::Exception& e) {
    throw ArmError(ArmError::Kind::kOther, std::string("libfranka: ") + e.what());
  } catch (const std::invalid_argument& e) {   // non-finite command (the loop checks first; a guard)
    throw ArmError(ArmError::Kind::kOther, std::string("libfranka: ") + e.what());
  }
}

}  // namespace

void LibfrankaArm::runTorqueControl(const TorqueStep& step) {
  try {
    robot_->control(
        [&](const franka::RobotState& s, franka::Duration period) -> franka::Torques {
          const TorqueCommand c = step(fill(s, *model_, period));
          franka::Torques tau(toArray(c.tau));
          if (c.finished) return franka::MotionFinished(tau);
          return tau;
        },
        control_.libfranka_limit_rate, control_.libfranka_cutoff_frequency_hz);
  } catch (...) {
    rethrow(stop_requested_);
  }
}

void LibfrankaArm::moveToJoints(const HomingMotion& m, const Observer& observe) {
  // The profile restarts from the commanded position libfranka reports at the first cycle (q_d), so the joint
  // motion starts exactly where the robot is (libfranka rejects a jump at the start).
  std::optional<HomingMotion> motion;
  double t = 0.0;
  try {
    robot_->control(
        [&](const franka::RobotState& s, franka::Duration period) -> franka::JointPositions {
          t += period.toSec();
          if (!motion) {
            motion.emplace(Eigen::Map<const Vec7>(s.q_d.data()), m.goal(), m.maxJointSpeed(), m.minDurationS());
          }
          observe(fill(s, *model_, period));
          franka::JointPositions q(toArray(motion->positionAt(t)));
          if (t >= motion->durationS()) return franka::MotionFinished(q);
          return q;
        },
        franka::ControllerMode::kJointImpedance, control_.libfranka_limit_rate, control_.libfranka_cutoff_frequency_hz);
  } catch (...) {
    rethrow(stop_requested_);
  }
}

void LibfrankaArm::requestStop() {
  stop_requested_.store(true);
  try {
    robot_->stop();   // preempts a control loop running in another thread with a ControlException
  } catch (const franka::Exception& e) {
    Log::warning(std::string("libfranka stop(): ") + e.what());
  }
}

}  // namespace franka_control
