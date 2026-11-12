// A minimum-jerk joint move to the home posture (STRUCTURE.md 7.2): q(s) = q0 + dq (10s^3 - 15s^4 + 6s^5).
#pragma once

#include "franka_control/types.h"

namespace franka_control {

class HomingMotion {                               // pure
 public:
  HomingMotion(const Vec7& q_start, const Vec7& q_goal, double max_joint_speed_rad_per_s, double min_duration_s);
  double durationS() const { return duration_s_; }
  Vec7 positionAt(double t_s) const;               // t clamped to [0, T]
  Vec7 velocityAt(double t_s) const;
  const Vec7& start() const { return q_start_; }
  const Vec7& goal() const { return q_goal_; }
  double maxJointSpeed() const { return max_speed_; }
  double minDurationS() const { return min_duration_s_; }

 private:
  Vec7 q_start_, q_goal_;
  double max_speed_, min_duration_s_, duration_s_;
};

}  // namespace franka_control
