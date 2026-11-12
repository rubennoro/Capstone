// ControlLoop::step() with hand-made readings and a FakeClock; the test plays the messaging thread.
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#include "franka_control/control_loop.h"
#include "franka_control/pose_math.h"
#include "support/test_support.h"

// Counts heap allocations made while `g_counting` is set (the 1 kHz step must make none).
namespace {
std::atomic<bool> g_counting{false};
std::atomic<long> g_allocations{0};
}  // namespace

// Every form is replaced (all on malloc/free), so the sanitizers always see matching pairs.
namespace {
void* countedAlloc(std::size_t n, std::size_t align) noexcept {
  if (g_counting.load(std::memory_order_relaxed)) g_allocations.fetch_add(1);
  if (align <= alignof(std::max_align_t)) return std::malloc(n ? n : 1);
  return std::aligned_alloc(align, ((n ? n : 1) + align - 1) / align * align);
}
void* countedOrThrow(std::size_t n, std::size_t align) {
  if (void* p = countedAlloc(n, align)) return p;
  throw std::bad_alloc();
}
}  // namespace

void* operator new(std::size_t n) { return countedOrThrow(n, 0); }
void* operator new[](std::size_t n) { return countedOrThrow(n, 0); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return countedAlloc(n, 0); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return countedAlloc(n, 0); }
void* operator new(std::size_t n, std::align_val_t a) { return countedOrThrow(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return countedOrThrow(n, static_cast<std::size_t>(a)); }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return countedAlloc(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return countedAlloc(n, static_cast<std::size_t>(a));
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace franka_control {
namespace {

using test::point;

class NullArm : public Arm {
 public:
  ArmReading readOnce() override { return {}; }
  ArmStatusFlags status() override { return {}; }
  void runTorqueControl(const TorqueStep&) override {}
  void moveToJoints(const HomingMotion&, const Observer&) override {}
  void requestStop() override {}
  std::string describe() const override { return "NullArm"; }
};

constexpr int64_t kMs = 1000000;

struct Rig {
  Rig() : cfg(test::testConfig()), shared(std::make_unique<SharedState>()) {
    cfg.F_T_TCP = Pose::Identity();
    shared->setGripperWidth(0.05);
    loop = std::make_unique<ControlLoop>(arm, *shared, cfg, clock.clock());
    loop->beginSession();
  }
  ArmReading reading(const Eigen::Vector3d& p = {0.45, 0, 0.45}) const {
    ArmReading r;
    r.O_T_F = test::makePose(p, kDown);
    r.J_F.setZero();
    r.J_F.leftCols<6>().setIdentity();
    return r;
  }
  TorqueCommand step(const ArmReading& r, int64_t dt_ns = kMs) {
    clock.advance(dt_ns);
    return loop->step(r);
  }
  const LoopSnapshot& snap() { return shared->latestSnapshot(); }
  uint64_t post(LoopCommand::Kind kind, const WaypointTrajectory* plan = nullptr) {
    LoopCommand c;
    c.generation = ++generation;
    c.kind = kind;
    if (plan) c.plan = *plan;
    shared->postCommand(c);
    return c.generation;
  }

  const Eigen::Quaterniond kDown = test::rotvec({M_PI, 0, 0});
  Config cfg;
  test::FakeClock clock;
  NullArm arm;
  std::unique_ptr<SharedState> shared;
  std::unique_ptr<ControlLoop> loop;
  uint64_t generation = 0;
};

TEST(ControlLoop, HoldsTheMeasuredPoseAtSessionStart) {
  Rig r;
  const TorqueCommand c = r.step(r.reading({0.5, 0.1, 0.3}));
  EXPECT_FALSE(c.finished);
  EXPECT_TRUE(c.tau.isZero(1e-9));   // target = measured: nothing to push
  const LoopSnapshot& s = r.snap();
  EXPECT_TRUE(s.O_T_TCP_target.translation().isApprox(Eigen::Vector3d(0.5, 0.1, 0.3)));
  EXPECT_DOUBLE_EQ(s.gripper_target_m, 0.05);
  EXPECT_DOUBLE_EQ(r.shared->gripperTarget(), 0.05);
}

TEST(ControlLoop, FollowsAPlan) {
  Rig r;
  r.step(r.reading());
  r.shared->markAlive(r.clock.now());
  WaypointTrajectory plan;
  plan.append(point(r.clock.now(), {0.45, 0, 0.45}, r.kDown, 0.05));
  plan.append(point(r.clock.now() + 100 * kMs, {0.55, 0, 0.45}, r.kDown, 0.03));
  r.post(LoopCommand::Kind::kFollow, &plan);
  r.step(r.reading(), 50 * kMs);
  const LoopSnapshot& s = r.snap();
  EXPECT_NEAR(s.O_T_TCP_target.translation().x(), 0.50, 1e-9);
  EXPECT_NEAR(s.gripper_target_m, 0.04, 1e-12);
  EXPECT_EQ(s.applied_generation, 1u);
  r.step(r.reading(), 200 * kMs);   // past the end: holds the last point
  EXPECT_NEAR(r.snap().O_T_TCP_target.translation().x(), 0.55, 1e-12);
}

TEST(ControlLoop, LateCyclesAreCounted) {
  Rig r;
  r.step(r.reading());
  r.step(r.reading(), kMs);
  EXPECT_EQ(r.snap().late_cycles_total, 0u);
  r.step(r.reading(), 2 * kMs);   // > late_period_ms 1.5
  EXPECT_EQ(r.snap().late_cycles_total, 1u);
  ArmReading missed = r.reading();
  missed.missed_packets = 1;
  r.step(missed, kMs);
  EXPECT_EQ(r.snap().late_cycles_total, 2u);
  r.step(r.reading(), kMs);
  EXPECT_EQ(r.snap().late_cycles_total, 2u);
}

TEST(ControlLoop, ActsOnEachCommandGenerationOnce) {
  Rig r;
  r.step(r.reading());
  r.post(LoopCommand::Kind::kHome);
  EXPECT_TRUE(r.step(r.reading()).finished);   // still (dq = 0): the session ends at once
  r.loop->beginSession();                       // a new session after the homing move
  for (int i = 0; i < 5; ++i) EXPECT_FALSE(r.step(r.reading()).finished);   // kHome is not acted on again
  EXPECT_EQ(r.snap().applied_generation, 1u);
}

TEST(ControlLoop, TheBackstopHoldsAfterLimitPlusExtra) {
  Rig r;
  r.step(r.reading());
  r.shared->markAlive(r.clock.now());
  WaypointTrajectory plan;
  plan.append(point(r.clock.now(), {0.45, 0, 0.45}, r.kDown, 0.05));
  plan.append(point(r.clock.now() + 2000 * kMs, {0.65, 0, 0.45}, r.kDown, 0.05));
  const uint64_t gen = r.post(LoopCommand::Kind::kFollow, &plan);
  r.step(r.reading(), 599 * kMs);   // 0.599 s since the last request: still following
  EXPECT_EQ(r.snap().watchdog_trip_generation, 0u);
  const double x_before = r.snap().O_T_TCP_target.translation().x();
  r.step(r.reading(), 2 * kMs);     // 0.601 s > 0.5 + 0.1
  EXPECT_EQ(r.snap().watchdog_trip_generation, gen);
  const double x_hold = r.snap().O_T_TCP_target.translation().x();
  EXPECT_NEAR(x_hold, x_before, 0.001);
  r.step(r.reading(), 100 * kMs);
  EXPECT_DOUBLE_EQ(r.snap().O_T_TCP_target.translation().x(), x_hold);   // holding, not following
}

TEST(ControlLoop, HomeEndsTheSessionOnlyAfterSettling) {
  Rig r;
  r.step(r.reading());
  r.post(LoopCommand::Kind::kHome);
  ArmReading moving = r.reading();
  moving.dq(2) = 0.05;   // > settle_speed 0.02
  EXPECT_FALSE(r.step(moving).finished);
  EXPECT_FALSE(r.step(moving, 500 * kMs).finished);
  EXPECT_TRUE(r.step(moving, 501 * kMs).finished);   // settle_timeout_s 1.0
  Rig s;
  s.step(s.reading());
  s.post(LoopCommand::Kind::kHome);
  EXPECT_FALSE(s.step(moving).finished);
  ArmReading still = s.reading();
  still.dq(2) = 0.01;
  EXPECT_TRUE(s.step(still).finished);
}

TEST(ControlLoop, ANonFiniteTorqueFinishesWithAFault) {
  Rig r;
  r.step(r.reading());
  ArmReading bad = r.reading();
  bad.dq(0) = std::nan("");
  const TorqueCommand c = r.step(bad);
  EXPECT_TRUE(c.finished);
  EXPECT_TRUE(c.tau.isZero());
  EXPECT_TRUE(r.snap().fault);
  EXPECT_STREQ(r.snap().error.data(), "controller produced a non-finite torque");
}

TEST(ControlLoop, SnapshotFieldsAreFilled) {
  Rig r;
  ArmReading a = r.reading({0.5, 0.0, 0.3});
  a.q = Vec7::Constant(0.2);
  a.dq = Vec7::Constant(0.001);
  r.step(a);
  r.step(a);
  const LoopSnapshot& s = r.snap();
  EXPECT_EQ(s.t_ns, r.clock.now());
  EXPECT_EQ(s.cycle, 2u);
  EXPECT_EQ(s.session, 0u);
  EXPECT_EQ(s.phase, LoopSnapshot::Phase::kTorque);
  EXPECT_EQ(s.q, a.q);
  EXPECT_EQ(s.dq, a.dq);
  EXPECT_TRUE(s.O_T_TCP.translation().isApprox(Eigen::Vector3d(0.5, 0.0, 0.3)));
  EXPECT_FALSE(s.fault);
  EXPECT_EQ(s.homed_generation, 0u);
}

TEST(ControlLoop, StepNeverAllocates) {
  Rig r;
  WaypointTrajectory plan;
  plan.append(point(r.clock.now(), {0.45, 0, 0.45}, r.kDown, 0.05));
  for (int k = 1; k < 60; ++k) {
    plan.append(point(r.clock.now() + k * 20 * kMs, {0.45 + 0.001 * k, 0, 0.45}, r.kDown, 0.05 - 0.0005 * k));
  }
  const ArmReading a = r.reading();
  ArmReading missed = a;
  missed.missed_packets = 2;
  r.step(a);   // session start
  r.shared->markAlive(r.clock.now());
  r.post(LoopCommand::Kind::kFollow, &plan);
  g_allocations.store(0);
  g_counting.store(true);
  for (int i = 0; i < 500; ++i) r.step(i % 50 == 0 ? missed : a, i % 97 == 0 ? 3 * kMs : kMs);   // late cycles too
  r.step(a, 700 * kMs);                                                                          // the backstop
  for (int i = 0; i < 1100; ++i) r.step(a);                                                      // a stats window
  g_counting.store(false);
  EXPECT_EQ(g_allocations.load(), 0);
  g_counting.store(true);
  { const auto probe = std::make_unique<int>(1); }   // the counter itself works
  g_counting.store(false);
  EXPECT_EQ(g_allocations.load(), 1);
  EXPECT_NE(r.snap().watchdog_trip_generation, 0u);
  EXPECT_GT(r.snap().stats.cycles, 0u);
}

TEST(ControlLoop, ShutdownEndsTheSessionAfterSettling) {
  Rig r;
  r.step(r.reading());
  r.shared->requestShutdown();
  EXPECT_TRUE(r.step(r.reading()).finished);
}

}  // namespace
}  // namespace franka_control
