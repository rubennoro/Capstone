// The safety gate (STRUCTURE.md section 6): every waypoint of a set_waypoints is checked before anything reaches the
// loop. Accepted, shortened (speed), pulled back (box/cone within a margin) or rejected.
// Port of main-sim's safety.py (rnoro/main-sim:controlSW/franka_deploy/franka_deploy/bridge/safety.py @ 4be9982),
// changed: margins (pull back only within them, reject beyond), skipped past waypoints, the lead and width rules,
// and the bucket formula for timed waypoints (6.2).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "franka_control/config.h"
#include "franka_control/types.h"

namespace franka_control {

struct GateIssue {
  uint64_t index = 0;          // position in the request
  std::string reason;
  bool failed = true;          // rejected list only: false for "not used: waypoint <i> of this request was rejected"
};

struct GateResult {
  std::vector<TrajectoryPoint> accepted;   // in order, already pulled back / shortened; empty if any rejected
  uint64_t skipped = 0;
  std::vector<GateIssue> shortened, pulled_back, rejected;   // pulled_back: for the C2 counter and the log
  uint64_t failedCount() const;            // rejected entries that failed a rule themselves (C2 gate.rejected)
};

class SafetyGate {                         // messaging thread; pure, keeps no state between requests
 public:
  SafetyGate(const GateConfig& cfg, const Pose& F_T_TCP, double gripper_width_max_m);
  GateResult check(const TrajectoryPoint& start, const std::vector<TrajectoryPoint>& requested,
                   int64_t now_ns) const;
  bool insideBox(const Eigen::Vector3d& p_tcp) const;      // no margin; the after-homing check (7.2)
  // Tilt (deg) of the hand's approach axis (the flange z axis) from cone_axis, for a TCP orientation.
  double tiltDeg(const Eigen::Quaterniond& q_tcp) const;

 private:
  GateConfig cfg_;
  Eigen::Quaterniond q_F_TCP_;   // rotation part of F_T_TCP
  double width_max_m_;
};

}  // namespace franka_control
