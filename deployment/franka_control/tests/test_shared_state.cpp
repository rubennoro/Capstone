#include <gtest/gtest.h>

#include "franka_control/shared_state.h"

namespace franka_control {
namespace {

static_assert(std::atomic<double>::is_always_lock_free);
static_assert(std::atomic<int64_t>::is_always_lock_free);
static_assert(std::atomic<uint8_t>::is_always_lock_free);

TEST(SharedState, AtomicsAreLockFree) {
  std::atomic<double> d{0.0};
  std::atomic<int64_t> i{0};
  std::atomic<bool> b{false};
  EXPECT_TRUE(d.is_lock_free());
  EXPECT_TRUE(i.is_lock_free());
  EXPECT_TRUE(b.is_lock_free());
}

TEST(SharedState, GripperFaultSetAndReadKeepsTheFirst) {
  SharedState s;
  EXPECT_FALSE(s.gripperFault());
  s.setGripperFault("gripper: lost");
  s.setGripperFault("gripper: second");
  ASSERT_TRUE(s.gripperFault());
  EXPECT_EQ(*s.gripperFault(), "gripper: lost");
}

TEST(SharedState, CommandsAndSnapshotsPassThrough) {
  auto s = std::make_unique<SharedState>();
  EXPECT_EQ(s->takeCommand(), nullptr);
  LoopCommand c;
  c.generation = 3;
  c.kind = LoopCommand::Kind::kHome;
  s->postCommand(c);
  const LoopCommand* got = s->takeCommand();
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got->generation, 3u);
  EXPECT_EQ(s->takeCommand(), nullptr);
  s->snapshotSlot().cycle = 42;
  s->publishSnapshot();
  EXPECT_EQ(s->latestSnapshot().cycle, 42u);
  s->markAlive(123);
  EXPECT_EQ(s->lastAliveNs(), 123);
  s->setGripperTarget(0.03);
  s->setGripperWidth(0.04);
  EXPECT_DOUBLE_EQ(s->gripperTarget(), 0.03);
  EXPECT_DOUBLE_EQ(s->gripperWidth(), 0.04);
  EXPECT_FALSE(s->shutdownRequested());
  s->requestShutdown();
  EXPECT_TRUE(s->shutdownRequested());
}

}  // namespace
}  // namespace franka_control
