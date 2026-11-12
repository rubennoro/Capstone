#include <gtest/gtest.h>

#include "franka_control/fake_gripper.h"
#include "franka_control/gripper_worker.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

struct Rig {
  explicit Rig(std::optional<double> object = std::nullopt) {
    FakeGripperConfig fc;   // start 0.08, 0.1 m/s
    fc.object_width_m = object;
    gripper = std::make_unique<FakeGripper>(fc, [this](int64_t ns) { slept_ns += ns; });
    shared.setGripperWidth(gripper->widthM());
    shared.setGripperTarget(gripper->widthM());
    worker = std::make_unique<GripperWorker>(*gripper, shared, GripperConfig{}, clock.clock());
  }
  test::FakeClock clock;
  SharedState shared;
  int64_t slept_ns = 0;
  std::unique_ptr<FakeGripper> gripper;
  std::unique_ptr<GripperWorker> worker;
};

TEST(GripperWorker, ChangesBelowTheDeadbandAreNotSent) {
  Rig r;
  r.shared.setGripperTarget(0.08 - 0.004);   // deadband 5 mm
  EXPECT_TRUE(r.worker->stepOnce());
  EXPECT_DOUBLE_EQ(r.gripper->widthM(), 0.08);
  EXPECT_EQ(r.slept_ns, 0);
}

TEST(GripperWorker, ClosingAndOpeningReachTheTargetAndTheWidthReachesSharedState) {
  Rig r;
  r.shared.setGripperTarget(0.02);
  EXPECT_TRUE(r.worker->stepOnce());
  EXPECT_DOUBLE_EQ(r.gripper->widthM(), 0.02);
  EXPECT_DOUBLE_EQ(r.shared.gripperWidth(), 0.02);
  EXPECT_EQ(r.slept_ns, secondsToNs(0.06 / 0.1));   // the move blocks for its duration
  r.shared.setGripperTarget(0.07);
  EXPECT_TRUE(r.worker->stepOnce());
  EXPECT_DOUBLE_EQ(r.shared.gripperWidth(), 0.07);
}

TEST(GripperWorker, FingersStopAtTheObject) {   // port of test_fake_gripper_stalls_at_object
  Rig r(0.045);
  r.shared.setGripperTarget(0.01);
  EXPECT_TRUE(r.worker->stepOnce());
  EXPECT_NEAR(r.shared.gripperWidth(), 0.045, 1e-12);
}

TEST(GripperWorker, AGripperErrorSetsTheGripperFault) {
  Rig r;
  r.gripper->failNextMove(true);
  r.shared.setGripperTarget(0.02);
  EXPECT_FALSE(r.worker->stepOnce());
  ASSERT_TRUE(r.shared.gripperFault());
  EXPECT_NE(r.shared.gripperFault()->find("injected gripper error"), std::string::npos);
}

}  // namespace
}  // namespace franka_control
