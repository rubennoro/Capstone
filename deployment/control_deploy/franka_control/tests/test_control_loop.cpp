// from rnoro/deployment:deployment/franka_control/tests/test_control_loop.cpp @ 91b178b
// changes: the backstop case uses link.yaml's 1.0 s; + run() with a scripted Arm (stop during homing) and the stop
// epoch with FakeArm (control_deploy/STRUCTURE.md 9.1). Old cases otherwise unchanged.
// ControlLoop::step() with hand-made readings and a FakeClock; the test plays the messaging thread.
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <optional>
#include <thread>

#include "franka_control/control_loop.h"
#include "franka_control/fake_arm.h"
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
  r.step(r.reading(), 1099 * kMs);  // 1.099 s since the last request: still following
  EXPECT_EQ(r.snap().watchdog_trip_generation, 0u);
  const double x_before = r.snap().O_T_TCP_target.translation().x();
  r.step(r.reading(), 2 * kMs);     // 1.101 s > 1.0 (link.yaml) + 0.1
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
  r.step(a, 1200 * kMs);                                                                         // the backstop
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

// ---- run(): stop during homing (control_deploy/STRUCTURE.md 4.4), with a scripted Arm and a FakeClock ----

class ScriptArm : public Arm {
 public:
  ScriptArm(SharedState& shared, test::FakeClock& clock) : shared_(shared), clock_(clock) {}
  ArmReading readOnce() override { return reading(); }
  ArmStatusFlags status() override {
    ++status_calls;
    clock_.advance(100 * kMs);   // so a robot that never leaves Move mode ends the wait by the timeout
    return after_stop;
  }
  // Session 1 ends for the posted kHome; session 2 requests shutdown at its start and ends after settling.
  void runTorqueControl(const TorqueStep& step) override {
    ++sessions;
    if (sessions >= 2) shared_.requestShutdown();
    for (int i = 0; i < 100; ++i) {
      clock_.advance(kMs);
      const TorqueCommand c = step(reading());
      if (i == 0) first_command.push_back(c);
      if (c.finished) return;
    }
    ADD_FAILURE() << "a torque session did not finish";
  }
  void moveToJoints(const HomingMotion&, const Observer& observe) override {
    ++moves;
    if (on_move) on_move(observe);
  }
  void requestStop() override {}
  std::string describe() const override { return "ScriptArm"; }

  ArmReading reading() const {
    ArmReading r;
    r.q = q;
    r.O_T_F = test::makePose(p, test::rotvec({M_PI, 0, 0}));
    r.J_F.setZero();
    r.J_F.leftCols<6>().setIdentity();
    return r;
  }

  Eigen::Vector3d p{0.45, 0.0, 0.45};
  Vec7 q = Vec7::Zero();
  ArmStatusFlags after_stop{RobotMode::kIdle, "[]"};
  std::function<void(const Observer&)> on_move;
  std::vector<TorqueCommand> first_command;   // the first command of each session
  int sessions = 0, moves = 0, status_calls = 0;

 private:
  SharedState& shared_;
  test::FakeClock& clock_;
};

struct RunRig {
  RunRig() : cfg(test::testConfig()), shared(std::make_unique<SharedState>()), arm(*shared, clock) {
    cfg.F_T_TCP = Pose::Identity();
    shared->setGripperWidth(0.05);
    loop = std::make_unique<ControlLoop>(arm, *shared, cfg, clock.clock());
    LoopCommand c;
    c.generation = home_gen;
    c.kind = LoopCommand::Kind::kHome;
    shared->postCommand(c);
  }
  // The move runs (one observed cycle at a new pose, as the real move would report), the messaging thread asks for
  // the abort meanwhile, and the move ends as libfranka's preempted move does.
  void stopTheMove(bool with_flag) {
    arm.on_move = [this, with_flag](const Arm::Observer& observe) {
      arm.p = {0.40, 0.05, 0.40};
      arm.q(6) = 0.5;   // the redundant joint moved too: the nullspace target must follow it
      observe(arm.reading());
      EXPECT_EQ(shared->latestSnapshot().phase, LoopSnapshot::Phase::kHoming);
      if (with_flag) shared->requestHomingAbort();
      throw ArmError(ArmError::Kind::kStopRequested, "libfranka: Move command preempted");
    };
  }
  const uint64_t home_gen = 7;
  Config cfg;
  test::FakeClock clock;
  std::unique_ptr<SharedState> shared;
  ScriptArm arm;
  std::unique_ptr<ControlLoop> loop;
};

TEST(ControlLoopRun, AnAbortBeforeTheMoveSkipsItAndANewSessionHolds) {
  RunRig r;
  r.shared->requestHomingAbort();
  r.loop->run();
  EXPECT_EQ(r.arm.moves, 0);
  EXPECT_EQ(r.arm.sessions, 2);
  EXPECT_FALSE(r.shared->homingAbortRequested());
  const LoopSnapshot& s = r.shared->latestSnapshot();
  EXPECT_EQ(s.homing_aborted_generation, r.home_gen);
  EXPECT_EQ(s.homed_generation, 0u);
  EXPECT_EQ(s.session, 1u);
  EXPECT_FALSE(s.fault);
}

TEST(ControlLoopRun, AStoppedMoveWithTheAbortFlagHoldsWhereTheArmStopped) {
  RunRig r;
  r.stopTheMove(true);
  r.loop->run();
  EXPECT_EQ(r.arm.moves, 1);
  EXPECT_GE(r.arm.status_calls, 1);   // waited for the robot to leave its move mode
  EXPECT_EQ(r.arm.sessions, 2);       // a new torque session, not the end of the thread
  EXPECT_FALSE(r.shared->homingAbortRequested());
  const LoopSnapshot& s = r.shared->latestSnapshot();
  EXPECT_FALSE(s.fault);
  EXPECT_FALSE(r.loop->failureKind());
  EXPECT_EQ(s.homing_aborted_generation, r.home_gen);
  EXPECT_EQ(s.session, 1u);
  EXPECT_TRUE(s.O_T_TCP_target.translation().isApprox(Eigen::Vector3d(0.40, 0.05, 0.40)));   // the measured pose
  ASSERT_EQ(r.arm.first_command.size(), 2u);
  EXPECT_TRUE(r.arm.first_command[1].tau.isZero(1e-9));   // target = measured, nullspace target = measured q
}

TEST(ControlLoopRun, AStoppedMoveThenAReflexFaultsWithTheProblem) {
  RunRig r;
  r.stopTheMove(true);
  r.arm.after_stop = ArmStatusFlags{RobotMode::kReflex, "[joint_reflex]"};
  r.loop->run();
  EXPECT_EQ(r.arm.sessions, 1);
  const LoopSnapshot& s = r.shared->latestSnapshot();
  EXPECT_TRUE(s.fault);
  EXPECT_EQ(s.phase, LoopSnapshot::Phase::kEnded);
  EXPECT_EQ(std::string(s.error.data()).rfind("homing stopped, then the arm reports reflex", 0), 0u) << s.error.data();
  EXPECT_NE(std::string(s.error.data()).find("restart control_deploy run"), std::string::npos);
  EXPECT_EQ(s.homing_aborted_generation, 0u);
}

TEST(ControlLoopRun, ARobotStillInMoveModeAfterTheWaitFaults) {
  RunRig r;
  r.stopTheMove(true);
  r.arm.after_stop = ArmStatusFlags{RobotMode::kMove, "[]"};
  r.loop->run();
  EXPECT_GE(r.arm.status_calls, 10);   // polled until control.settle_timeout_s (1 s on the fake clock)
  EXPECT_TRUE(r.shared->latestSnapshot().fault);
  EXPECT_NE(std::string(r.shared->latestSnapshot().error.data()).find("robot_mode Move"), std::string::npos);
}

TEST(ControlLoopRun, StopRequestedWithoutTheAbortFlagIsAFaultAsBefore) {
  RunRig r;
  r.stopTheMove(false);
  r.loop->run();
  EXPECT_EQ(r.arm.sessions, 1);
  EXPECT_EQ(r.arm.status_calls, 0);
  EXPECT_TRUE(r.shared->latestSnapshot().fault);
  EXPECT_STREQ(r.shared->latestSnapshot().error.data(), "libfranka: Move command preempted");
  EXPECT_EQ(r.loop->failureKind(), ArmError::Kind::kStopRequested);
}

TEST(ControlLoopRun, TheHomingMoveIsMarkedOnlyWhileItRuns) {   // a homing stop cannot reach the next session
  RunRig r;
  int stops_during_move = 0;
  EXPECT_FALSE(r.shared->stopHomingMove(0, [&] { ++stops_during_move; }));   // before: nothing runs
  r.arm.on_move = [&](const Arm::Observer&) {
    EXPECT_TRUE(r.shared->inHomingMove());
    EXPECT_FALSE(r.shared->stopHomingMove(1, [&] { ++stops_during_move; }));   // another homing's session
    EXPECT_TRUE(r.shared->stopHomingMove(0, [&] { ++stops_during_move; }));    // this move (session 0)
  };
  r.loop->run();
  EXPECT_EQ(stops_during_move, 1);
  EXPECT_EQ(r.arm.sessions, 2);
  EXPECT_FALSE(r.shared->inHomingMove());
  EXPECT_FALSE(r.shared->stopHomingMove(SharedState::kAnySession, [&] { ++stops_during_move; }));
  EXPECT_EQ(stops_during_move, 1);
}

TEST(ControlLoopRun, AStoppedMoveIsUnmarkedToo) {
  RunRig r;
  r.stopTheMove(true);
  r.loop->run();
  EXPECT_FALSE(r.shared->inHomingMove());
}

TEST(ControlLoopRun, ANormalFinishWithTheFlagSetIsHomedAndClearsIt) {
  RunRig r;
  r.arm.on_move = [&](const Arm::Observer&) { r.shared->requestHomingAbort(); };   // too late: the move finished
  r.loop->run();
  EXPECT_EQ(r.arm.sessions, 2);
  EXPECT_FALSE(r.shared->homingAbortRequested());
  EXPECT_EQ(r.shared->latestSnapshot().homed_generation, r.home_gen);
  EXPECT_EQ(r.shared->latestSnapshot().homing_aborted_generation, 0u);
}

// ---- the stop epoch (control_deploy/STRUCTURE.md 3.3), with FakeArm in real time ----

struct EpochRig {
  EpochRig() : cfg(test::testConfig()), arm(cfg.fake_arm, cfg.F_T_TCP, cfg.homing.q_home_rad, monotonicNowNs) {}
  // Runs a torque session of `cycles` cycles (0: until stopped) on this thread; the ArmError kind, if one ended it.
  std::optional<ArmError::Kind> session(int cycles) {
    int n = 0;
    try {
      arm.runTorqueControl([&](const ArmReading&) {
        ++n;
        running.store(true);
        return TorqueCommand{Vec7::Zero(), cycles > 0 && n >= cycles};
      });
    } catch (const ArmError& e) {
      return e.kind;
    }
    return std::nullopt;
  }
  Config cfg;
  FakeArm arm;
  std::atomic<bool> running{false};
};

TEST(ControlLoopThreads, AStopDuringACallEndsThatCallAsStopRequested) {
  EpochRig r;
  std::optional<ArmError::Kind> kind;
  std::thread t([&] { kind = r.session(0); });
  ASSERT_TRUE(test::waitUntil([&] { return r.running.load(); }, 5.0));
  r.arm.requestStop();
  t.join();
  ASSERT_TRUE(kind);
  EXPECT_EQ(*kind, ArmError::Kind::kStopRequested);
}

TEST(ControlLoopThreads, AnEarlierStopDoesNotMarkALaterCallAndALaterFaultIsAFault) {
  EpochRig r;
  r.arm.requestStop();                       // nothing runs: affects no call
  EXPECT_FALSE(r.session(5));                // a whole session runs and finishes normally
  HomingMotion short_move(r.cfg.homing.q_home_rad, r.cfg.homing.q_home_rad, 0.3, 0.02);
  EXPECT_NO_THROW(r.arm.moveToJoints(short_move, [](const ArmReading&) {}));
  r.arm.injectFault("libfranka: motion aborted by reflex! [cartesian_reflex] (injected)");
  const auto kind = r.session(0);
  ASSERT_TRUE(kind);
  EXPECT_EQ(*kind, ArmError::Kind::kReflex);   // never kStopRequested
}

TEST(ControlLoopThreads, AStopDuringAMoveEndsThatMoveAsStopRequested) {
  EpochRig r;
  HomingMotion long_move(r.cfg.homing.q_home_rad, r.cfg.homing.q_home_rad, 0.3, 20.0);
  std::optional<ArmError::Kind> kind;
  std::thread t([&] {
    try {
      r.arm.moveToJoints(long_move, [&](const ArmReading&) { r.running.store(true); });
    } catch (const ArmError& e) {
      kind = e.kind;
    }
  });
  ASSERT_TRUE(test::waitUntil([&] { return r.running.load(); }, 5.0));
  r.arm.requestStop();
  t.join();
  ASSERT_TRUE(kind);
  EXPECT_EQ(*kind, ArmError::Kind::kStopRequested);
  // the fake's status after a stop is what the test hook says (default: Idle, no errors)
  EXPECT_EQ(describeArmProblem(r.arm.status()), "");
  r.arm.statusAfterStop(ArmStatusFlags{RobotMode::kUserStopped, "[]"});
  EXPECT_NE(describeArmProblem(r.arm.status()).find("user stop"), std::string::npos);
}

}  // namespace
}  // namespace franka_control
