// Value types shared by the loop, the gate and the arm (STRUCTURE.md 9.1).
#pragma once

#include <cstdint>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace franka_control {

using Vec7 = Eigen::Matrix<double, 7, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat67 = Eigen::Matrix<double, 6, 7>;
using Pose = Eigen::Isometry3d;

// One point of a plan: reach position p and orientation q (TCP, base frame) and this gripper width at t_ns.
struct TrajectoryPoint {
  int64_t t_ns = 0;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  double width_m = 0.0;
};

struct Target {
  Pose O_T_TCP = Pose::Identity();
  double width_m = 0.0;
};

// One cycle's input, independent of libfranka.
struct ArmReading {
  Vec7 q = Vec7::Zero();
  Vec7 dq = Vec7::Zero();
  Vec7 coriolis = Vec7::Zero();
  Vec7 tau_J_d = Vec7::Zero();          // the last commanded torque
  Pose O_T_F = Pose::Identity();         // flange pose, base frame
  Mat67 J_F = Mat67::Zero();             // flange zero Jacobian, base frame ([linear; angular])
  double success_rate = 1.0;             // control_command_success_rate (FakeArm: 1.0)
  uint32_t missed_packets = 0;
};

struct TorqueCommand {
  Vec7 tau = Vec7::Zero();
  bool finished = false;
};

}  // namespace franka_control
