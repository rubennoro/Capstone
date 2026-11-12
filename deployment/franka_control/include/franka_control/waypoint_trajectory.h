// A plan: the start point plus the accepted waypoints, and the target at any time (STRUCTURE.md 5.3).
// Fixed capacity, no allocation: the 1 kHz loop evaluates it in place.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "franka_control/types.h"

namespace franka_control {

class WaypointTrajectory {
 public:
  static constexpr size_t kMaxWaypoints = 64;              // must be >= limits.yaml max_waypoints
  static constexpr size_t kCapacity = kMaxWaypoints + 1;   // + the start point

  void clear() { n_ = 0; }
  // Appends a point; false (nothing changed) if full or its time is not after the last point's.
  bool append(const TrajectoryPoint& p);
  size_t size() const { return n_; }
  bool empty() const { return n_ == 0; }
  const TrajectoryPoint& point(size_t i) const { return points_[i]; }
  int64_t lastTimeNs() const { return points_[n_ - 1].t_ns; }

  // The target at t: the first point before it, linear position/width and slerp (the shorter way) between points,
  // the last point after it (holding). Requires size() >= 1.
  TrajectoryPoint at(int64_t t_ns) const;

 private:
  std::array<TrajectoryPoint, kCapacity> points_{};
  size_t n_ = 0;
};

}  // namespace franka_control
