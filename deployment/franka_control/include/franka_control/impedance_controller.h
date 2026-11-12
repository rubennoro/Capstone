// Cartesian impedance at the TCP (STRUCTURE.md 5.4), from libfranka's examples/cartesian_impedance_control.cpp plus
// a nullspace term, error clipping and torque limits. Control thread; pure, fixed-size Eigen only (no allocation).
#pragma once

#include "franka_control/config.h"
#include "franka_control/types.h"

namespace franka_control {

struct TorqueResult {
  Vec7 tau = Vec7::Zero();
  Pose O_T_TCP = Pose::Identity();   // the measured TCP pose
  double position_error_m = 0.0;     // before clipping
  double rotation_error_rad = 0.0;   // before clipping
  bool finite = true;                // false: some input or output is not a finite number (tau is then zero)
};

class ImpedanceController {
 public:
  ImpedanceController(const ControlConfig& cfg, const Pose& F_T_TCP);
  void setNullspaceTarget(const Vec7& q_ns);      // at each session start
  TorqueResult compute(const ArmReading& r, const Pose& O_T_TCP_target) const;

 private:
  ControlConfig cfg_;
  Pose F_T_TCP_;
  Vec7 q_ns_ = Vec7::Zero();
  Vec6 K_, D_;
};

}  // namespace franka_control
