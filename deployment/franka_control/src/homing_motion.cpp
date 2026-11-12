#include "franka_control/homing_motion.h"

#include <algorithm>

namespace franka_control {
namespace {
constexpr double kPeakSpeedFactor = 1.875;   // max of d/ds (10s^3 - 15s^4 + 6s^5), at s = 0.5
}

HomingMotion::HomingMotion(const Vec7& q_start, const Vec7& q_goal, double max_joint_speed_rad_per_s,
                           double min_duration_s)
    : q_start_(q_start),
      q_goal_(q_goal),
      max_speed_(max_joint_speed_rad_per_s),
      min_duration_s_(min_duration_s),
      duration_s_(std::max(min_duration_s,
                           kPeakSpeedFactor * (q_goal - q_start).cwiseAbs().maxCoeff() / max_joint_speed_rad_per_s)) {}

Vec7 HomingMotion::positionAt(double t_s) const {
  const double s = std::clamp(t_s / duration_s_, 0.0, 1.0);
  if (s >= 1.0) return q_goal_;
  const double s3 = s * s * s;
  return q_start_ + (q_goal_ - q_start_) * (10.0 * s3 - 15.0 * s3 * s + 6.0 * s3 * s * s);
}

Vec7 HomingMotion::velocityAt(double t_s) const {
  const double s = std::clamp(t_s / duration_s_, 0.0, 1.0);
  const double s2 = s * s;
  return (q_goal_ - q_start_) * ((30.0 * s2 - 60.0 * s2 * s + 30.0 * s2 * s2) / duration_s_);
}

}  // namespace franka_control
