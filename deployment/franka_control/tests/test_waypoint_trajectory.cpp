#include <gtest/gtest.h>

#include "franka_control/pose_math.h"
#include "franka_control/waypoint_trajectory.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

using test::point;
using test::rotvec;

WaypointTrajectory twoPoints(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1) {
  WaypointTrajectory t;
  EXPECT_TRUE(t.append(point(1000, {0.4, 0.0, 0.3}, q0, 0.02)));
  EXPECT_TRUE(t.append(point(2000, {0.5, 0.2, 0.1}, q1, 0.06)));
  return t;
}

TEST(WaypointTrajectory, MidpointLerpOfPositionAndWidth) {
  const auto t = twoPoints(Eigen::Quaterniond::Identity(), Eigen::Quaterniond::Identity());
  const TrajectoryPoint m = t.at(1500);
  EXPECT_TRUE(m.p.isApprox(Eigen::Vector3d(0.45, 0.1, 0.2), 1e-12));
  EXPECT_NEAR(m.width_m, 0.04, 1e-12);
  EXPECT_EQ(m.t_ns, 1500);
}

TEST(WaypointTrajectory, SlerpAtAQuarter) {
  const auto t = twoPoints(Eigen::Quaterniond::Identity(), rotvec({0, 0, 0.8}));
  EXPECT_NEAR(rotationAngle(t.at(1250).q, rotvec({0, 0, 0.2})), 0.0, 1e-9);
}

TEST(WaypointTrajectory, TakesTheShorterWayWithOppositeSigns) {
  Eigen::Quaterniond q1 = rotvec({0, 0, 0.8});
  q1.coeffs() = -q1.coeffs();   // same rotation, other hemisphere
  const auto t = twoPoints(Eigen::Quaterniond::Identity(), q1);
  EXPECT_NEAR(rotationAngle(t.at(1500).q, rotvec({0, 0, 0.4})), 0.0, 1e-9);
}

TEST(WaypointTrajectory, ExactAtPointsAndHoldsOutside) {
  const auto t = twoPoints(rotvec({0.1, 0, 0}), rotvec({0, 0.3, 0}));
  EXPECT_TRUE(t.at(1000).p.isApprox(t.point(0).p));
  EXPECT_TRUE(t.at(2000).p.isApprox(t.point(1).p));
  EXPECT_NEAR(rotationAngle(t.at(2000).q, t.point(1).q), 0.0, 1e-12);
  EXPECT_TRUE(t.at(0).p.isApprox(t.point(0).p));         // before the first point: the first point
  EXPECT_TRUE(t.at(99999).p.isApprox(t.point(1).p));     // after the last: holding it
  EXPECT_NEAR(t.at(99999).width_m, 0.06, 1e-15);
}

TEST(WaypointTrajectory, AppendRefusesNonIncreasingTimesAndOverCapacity) {
  WaypointTrajectory t;
  EXPECT_TRUE(t.append(point(10, {0, 0, 0}, Eigen::Quaterniond::Identity(), 0)));
  EXPECT_FALSE(t.append(point(10, {0, 0, 0}, Eigen::Quaterniond::Identity(), 0)));
  EXPECT_FALSE(t.append(point(5, {0, 0, 0}, Eigen::Quaterniond::Identity(), 0)));
  for (size_t i = 1; i < WaypointTrajectory::kCapacity; ++i) {
    EXPECT_TRUE(t.append(point(10 + static_cast<int64_t>(i), {0, 0, 0}, Eigen::Quaterniond::Identity(), 0)));
  }
  EXPECT_EQ(t.size(), WaypointTrajectory::kCapacity);
  EXPECT_FALSE(t.append(point(100000, {0, 0, 0}, Eigen::Quaterniond::Identity(), 0)));
}

}  // namespace
}  // namespace franka_control
