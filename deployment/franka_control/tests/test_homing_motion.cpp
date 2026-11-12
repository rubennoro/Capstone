#include <gtest/gtest.h>

#include "franka_control/homing_motion.h"

namespace franka_control {
namespace {

const Vec7 kStart = (Vec7() << 0.1, -0.5, 0.2, -2.0, 0.1, 1.2, 0.5).finished();
const Vec7 kGoal = (Vec7() << 0, -0.7853982, 0, -2.3561945, 0, 1.5707963, 0.7853982).finished();

TEST(HomingMotion, StartsAndEndsAtRestAndReachesTheGoalAtT) {
  const HomingMotion m(kStart, kGoal, 0.3, 2.0);
  EXPECT_TRUE(m.positionAt(0.0).isApprox(kStart));
  EXPECT_NEAR(m.velocityAt(0.0).norm(), 0.0, 1e-15);
  EXPECT_NEAR(m.velocityAt(m.durationS()).norm(), 0.0, 1e-12);
  EXPECT_EQ(m.positionAt(m.durationS()), kGoal);
  EXPECT_EQ(m.positionAt(m.durationS() + 5.0), kGoal);   // clamped
}

TEST(HomingMotion, PeakSpeedWithinTheLimitAndAllJointsTogether) {
  const HomingMotion m(kStart, kGoal, 0.3, 0.5);
  double peak = 0.0;
  for (int i = 0; i <= 1000; ++i) {
    const double t = m.durationS() * i / 1000.0;
    peak = std::max(peak, m.velocityAt(t).cwiseAbs().maxCoeff());
    // every joint is at the same fraction of its move
    const Vec7 frac = (m.positionAt(t) - kStart).cwiseQuotient(kGoal - kStart);
    EXPECT_NEAR(frac.maxCoeff() - frac.minCoeff(), 0.0, 1e-9);
  }
  EXPECT_LE(peak, 0.3 + 1e-9);
  EXPECT_GE(peak, 0.3 * 0.99);   // the slowest joint sets T, so the limit is reached
}

TEST(HomingMotion, DurationAtLeastTheMinimum) {
  const Vec7 near_goal = kGoal + Vec7::Constant(1e-3);
  EXPECT_DOUBLE_EQ(HomingMotion(near_goal, kGoal, 0.3, 2.0).durationS(), 2.0);
  EXPECT_DOUBLE_EQ(HomingMotion(kGoal, kGoal, 0.3, 2.0).durationS(), 2.0);
}

}  // namespace
}  // namespace franka_control
