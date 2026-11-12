#include "franka_control/waypoint_trajectory.h"

namespace franka_control {

bool WaypointTrajectory::append(const TrajectoryPoint& p) {
  if (n_ >= kCapacity) return false;
  if (n_ > 0 && p.t_ns <= points_[n_ - 1].t_ns) return false;
  points_[n_++] = p;
  return true;
}

TrajectoryPoint WaypointTrajectory::at(int64_t t_ns) const {
  if (t_ns <= points_[0].t_ns) {
    TrajectoryPoint out = points_[0];
    out.t_ns = t_ns;
    return out;
  }
  for (size_t i = 0; i + 1 < n_; ++i) {
    const TrajectoryPoint& a = points_[i];
    const TrajectoryPoint& b = points_[i + 1];
    if (t_ns < b.t_ns) {
      const double s = static_cast<double>(t_ns - a.t_ns) / static_cast<double>(b.t_ns - a.t_ns);
      TrajectoryPoint out;
      out.t_ns = t_ns;
      out.p = a.p + s * (b.p - a.p);
      out.q = a.q.slerp(s, b.q).normalized();   // Eigen's slerp takes the shorter way
      out.width_m = a.width_m + s * (b.width_m - a.width_m);
      return out;
    }
  }
  TrajectoryPoint out = points_[n_ - 1];
  out.t_ns = t_ns;
  return out;
}

}  // namespace franka_control
