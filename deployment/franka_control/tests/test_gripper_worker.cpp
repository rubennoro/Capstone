#include <gtest/gtest.h>

#include <set>
#include <thread>

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
  EXPECT_EQ(r.slept_ns, secondsToNs(0.06 / 0.1));   // the move blocks for its duration
  EXPECT_TRUE(r.worker->readStep());
  EXPECT_DOUBLE_EQ(r.shared.gripperWidth(), 0.02);
  r.shared.setGripperTarget(0.07);
  EXPECT_TRUE(r.worker->stepOnce());
  EXPECT_TRUE(r.worker->readStep());
  EXPECT_DOUBLE_EQ(r.shared.gripperWidth(), 0.07);
}

TEST(GripperWorker, FingersStopAtTheObject) {   // port of test_fake_gripper_stalls_at_object
  Rig r(0.045);
  r.shared.setGripperTarget(0.01);
  EXPECT_TRUE(r.worker->stepOnce());
  EXPECT_TRUE(r.worker->readStep());
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

TEST(GripperWorker, AReadErrorSetsTheGripperFault) {
  Rig r;
  r.gripper->failNextRead(true);
  EXPECT_FALSE(r.worker->readStep());
  ASSERT_TRUE(r.shared.gripperFault());
  EXPECT_NE(r.shared.gripperFault()->find("injected gripper read error"), std::string::npos);
}

TEST(GripperWorker, FakeMoveStepsThroughIntermediateWidths) {
  FakeGripperConfig fc;   // start 0.08, 0.1 m/s
  std::vector<double> seen;
  FakeGripper* g = nullptr;
  FakeGripper gripper(fc, [&](int64_t) { seen.push_back(g->widthM()); });
  g = &gripper;
  gripper.moveTo(0.0);   // 0.8 s = 160 steps of 5 ms
  ASSERT_EQ(seen.size(), 160u);
  EXPECT_DOUBLE_EQ(seen.front(), 0.08);
  EXPECT_NEAR(seen[80], 0.04, 1e-9);
  EXPECT_DOUBLE_EQ(gripper.widthM(), 0.0);
}

// Both gripper threads on a FakeGripper whose moves take real time (0.1 m/s), as in the App.
struct ThreadRig {
  ThreadRig()
      : gripper(FakeGripperConfig{}, [](int64_t ns) { std::this_thread::sleep_for(std::chrono::nanoseconds(ns)); }) {
    shared.setGripperWidth(gripper.widthM());
    shared.setGripperTarget(gripper.widthM());
    worker = std::make_unique<GripperWorker>(gripper, shared, GripperConfig{}, monotonicNowNs);   // read at 50 Hz
    commands = std::thread([this] {
      worker->run();
      commands_done.store(true);
    });
    reader = std::thread([this] {
      worker->runReader();
      reader_done.store(true);
    });
  }
  ~ThreadRig() {
    shared.requestShutdown();
    gripper.stop();
    if (commands.joinable()) commands.join();
    if (reader.joinable()) reader.join();
  }
  SharedState shared;
  FakeGripper gripper;
  std::unique_ptr<GripperWorker> worker;
  std::atomic<bool> commands_done{false}, reader_done{false};
  std::thread commands, reader;
};

TEST(GripperWorkerThreads, TheWidthInSharedStateChangesDuringAFingerMove) {
  ThreadRig r;
  r.shared.setGripperTarget(0.0);   // one blocking move of 0.8 s
  std::set<double> mid_move;        // widths strictly between start and target: only seen while the move runs
  ASSERT_TRUE(test::waitUntil(
      [&] {
        const double w = r.shared.gripperWidth();
        if (w > 0.0 && w < 0.08) mid_move.insert(w);
        return w == 0.0;
      },
      5.0));
  // 0.8 s read at 50 Hz: ~40 reads during the move; at least 10 distinct widths shows it updates throughout
  EXPECT_GE(mid_move.size(), 10u);
  EXPECT_FALSE(r.shared.gripperFault());
}

TEST(GripperWorkerThreads, BothLoopsEndPromptlyAtShutdownMidMove) {
  ThreadRig r;
  r.shared.setGripperTarget(0.0);
  ASSERT_TRUE(test::waitUntil([&] { return r.shared.gripperWidth() < 0.07; }, 5.0));   // the move is running
  const int64_t t0 = monotonicNowNs();
  r.shared.requestShutdown();   // what App does: shutdown, then gripper.stop()
  r.gripper.stop();
  ASSERT_TRUE(test::waitUntil([&] { return r.commands_done.load() && r.reader_done.load(); }, 1.0));
  EXPECT_LT(nsToSeconds(monotonicNowNs() - t0), 0.1);   // one 5 ms fake step and one 20 ms read period
  const double stopped_at = r.gripper.widthM();
  EXPECT_GT(stopped_at, 0.0);   // the move was stopped, not finished
  EXPECT_FALSE(r.shared.gripperFault());
}

}  // namespace
}  // namespace franka_control
