// from rnoro/deployment:deployment/franka_control/tests/test_supervisor.cpp @ 91b178b
// changes: Rig takes an optional test gripper; the watchdog case uses link.yaml's 1.0 s; + the keys, the hand's homing
// and stop during homing (control_deploy/STRUCTURE.md 9.1). Old cases otherwise unchanged. Fix wave: the test also
// plays the control thread's begin/endHomingMove and the gripper thread's begin/endHandHoming; stops re-sent every
// 100 ms while the move runs; no stop once a move or a hand homing has ended, none before a hand homing began.
// + key p (the start pose) and C3 start's start-pose check; Rig takes an optional config tweak.
// Supervisor with a FakeClock, a SharedState and a test Arm; no threads. The test plays the control thread: it
// publishes snapshots and takes the posted commands.
#include <gtest/gtest.h>

#include <cstdio>

#include "franka_control/pose_math.h"
#include "franka_control/supervisor.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

using test::point;

class CountingArm : public Arm {
 public:
  ArmReading readOnce() override { return {}; }
  ArmStatusFlags status() override { return {}; }
  void runTorqueControl(const TorqueStep&) override {}
  void moveToJoints(const HomingMotion&, const Observer&) override {}
  void requestStop() override { ++stops; }
  std::string describe() const override { return "CountingArm"; }
  int stops = 0;
};

class CountingGripper : public Gripper {
 public:
  void prepare() override {}
  double widthM() override { return 0.04; }
  void moveTo(double) override {}
  void stop() override { ++stops; }
  bool home() override { return true; }
  void clearStop() override {}
  int stops = 0;
};

constexpr int64_t kMs = 1000000;

struct Rig {
  explicit Rig(const Eigen::Vector3d& target = {0.45, 0.0, 0.45}, Gripper* gripper = nullptr,
               const std::function<void(Config&)>& tweak = {})
      : cfg(test::testConfig()) {
    if (tweak) tweak(cfg);
    shared = std::make_unique<SharedState>();
    shared->setGripperWidth(0.04);
    publish([&](LoopSnapshot& s) { setPose(s, target); });
    sup = std::make_unique<Supervisor>(*shared, arm, cfg, cfg.limits, clock.clock(), gripper);
    take();   // the initial kHold
  }
  static void setPose(LoopSnapshot& s, const Eigen::Vector3d& p) {
    s.O_T_TCP = test::makePose(p, test::rotvec({M_PI, 0, 0}));
    s.O_T_TCP_target = s.O_T_TCP;
  }
  void publish(const std::function<void(LoopSnapshot&)>& f) {
    current.phase = LoopSnapshot::Phase::kTorque;
    current.t_ns = clock.now();
    current.gripper_target_m = 0.04;
    if (current.q.isZero()) current.q = cfg.homing.q_home_rad;
    f(current);
    shared->snapshotSlot() = current;
    shared->publishSnapshot();
  }
  const LoopCommand* take() { return shared->takeCommand(); }
  void alive() { shared->markAlive(clock.now()); }
  void start() {
    alive();
    ASSERT_TRUE(sup->onStart().ok);
    take();
  }
  msg::SetWaypointsReply send(const std::vector<TrajectoryPoint>& pts, uint64_t chunk = 1) {
    alive();
    auto r = sup->onSetWaypoints(test::makeSetWaypoints(chunk, pts));
    EXPECT_TRUE(std::holds_alternative<msg::SetWaypointsReply>(r));
    return std::get<msg::SetWaypointsReply>(r);
  }
  TrajectoryPoint wp(double dt_s, const Eigen::Vector3d& p, double w = 0.04) const {
    return point(clock.now() + secondsToNs(dt_s), p, test::rotvec({M_PI, 0, 0}), w);
  }

  Config cfg;
  test::FakeClock clock;
  CountingArm arm;
  std::unique_ptr<SharedState> shared;
  LoopSnapshot current;
  std::unique_ptr<Supervisor> sup;
};

TEST(Supervisor, StartHoldsWithAOnePointPlan) {
  Rig r;
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  ASSERT_TRUE(r.sup->onStart().ok);
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);
  const LoopCommand* c = r.take();
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->kind, LoopCommand::Kind::kFollow);
  ASSERT_EQ(c->plan.size(), 1u);
  EXPECT_TRUE(c->plan.point(0).p.isApprox(Eigen::Vector3d(0.45, 0, 0.45)));
  EXPECT_EQ(c->plan.point(0).t_ns, r.clock.now());
  EXPECT_FALSE(r.sup->onStart().ok);   // already started
}

TEST(Supervisor, WaypointsRunAndANewRequestReplacesTheRest) {
  Rig r;
  r.start();
  auto rep = r.send({r.wp(0.05, {0.46, 0, 0.45}), r.wp(0.10, {0.47, 0, 0.45}), r.wp(0.15, {0.48, 0, 0.45})});
  EXPECT_EQ(rep.accepted, 3u);
  EXPECT_EQ(r.sup->mode(), Mode::kRunning);
  const LoopCommand* c = r.take();
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(c->plan.size(), 4u);
  EXPECT_TRUE(c->plan.point(0).p.isApprox(Eigen::Vector3d(0.45, 0, 0.45)));   // the start point
  const WaypointTrajectory first = c->plan;
  r.clock.advance(75 * kMs);
  rep = r.send({r.wp(0.05, {0.40, 0, 0.45})}, 2);
  c = r.take();
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(c->plan.size(), 2u);   // the old waypoints not yet reached are gone
  EXPECT_TRUE(c->plan.point(0).p.isApprox(first.at(r.clock.now()).p, 1e-12));
  EXPECT_EQ(c->plan.point(0).t_ns, r.clock.now());
  EXPECT_NEAR(c->plan.point(0).p.x(), 0.465, 1e-9);
}

TEST(Supervisor, AllSkippedHoldsWithoutAFault) {
  Rig r;
  r.start();
  r.send({r.wp(0.05, {0.46, 0, 0.45})});
  ASSERT_EQ(r.sup->mode(), Mode::kRunning);
  const auto rep = r.send({r.wp(-0.02, {0.46, 0, 0.45}), r.wp(0.0, {0.46, 0, 0.45})});
  EXPECT_EQ(rep.accepted, 0u);
  EXPECT_EQ(rep.skipped, 2u);
  EXPECT_TRUE(rep.rejected.empty());
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);
}

TEST(Supervisor, PulledBackAndShortenedAreCounted) {
  Rig r({0.795, 0, 0.45});
  r.start();
  const auto rep = r.send({r.wp(0.005, {0.805, 0, 0.45})});
  EXPECT_EQ(rep.accepted, 1u);
  EXPECT_EQ(rep.shortened, 1u);
  EXPECT_EQ(r.sup->gateCounters().accepted, 1u);
  EXPECT_EQ(r.sup->gateCounters().shortened, 1u);
  EXPECT_EQ(r.sup->gateCounters().pulled_back, 1u);
  const msg::ArmState s = r.sup->stateMessage(5);
  EXPECT_EQ(s.gate.pulled_back, 1u);
}

TEST(Supervisor, ThePlansEndGivesHoldingAndHeartbeatsKeepItThere) {
  Rig r;
  r.start();
  r.send({r.wp(0.05, {0.46, 0, 0.45})});
  r.clock.advance(40 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kRunning);
  r.clock.advance(10 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);
  for (int i = 0; i < 20; ++i) {
    r.clock.advance(100 * kMs);
    r.alive();
    r.sup->tick();
  }
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);
}

TEST(Supervisor, NoRequestForTheWatchdogLimitStops) {   // link.yaml timing.watchdog_limit_s: 1.0 s
  Rig r;
  r.start();
  r.clock.advance(999 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);
  r.clock.advance(1 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);   // exactly at the limit: not yet
  r.clock.advance(2 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_NE(r.sup->modeDetail().find("watchdog: no request for 1.002 s"), std::string::npos);
  const LoopCommand* c = r.take();
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->kind, LoopCommand::Kind::kHold);
  r.alive();
  EXPECT_TRUE(r.sup->onStart().ok);   // the next episode's start is accepted
}

TEST(Supervisor, SetWaypointsRefusedByTheStateIsAnErrorReplyNotAFault) {
  Rig r;
  auto refused = [&](Mode expected) {
    const auto res = r.sup->onSetWaypoints(test::makeSetWaypoints(1, {r.wp(0.05, {0.46, 0, 0.45})}));
    ASSERT_TRUE(std::holds_alternative<msg::ErrorReply>(res));
    const auto& e = std::get<msg::ErrorReply>(res);
    EXPECT_EQ(e.request, "c3.set_waypoints/v1");
    EXPECT_EQ(e.error.rfind("set_waypoints refused: mode is " + modeName(expected), 0), 0u) << e.error;
    EXPECT_EQ(r.sup->mode(), expected);
  };
  refused(Mode::kIdle);
  r.sup->onStop();
  refused(Mode::kStopped);
  r.shared->setGripperFault("gripper: lost");
  r.sup->tick();
  refused(Mode::kFault);
}

TEST(Supervisor, AGateRejectionFaultsHoldsAndCounts) {
  Rig r;
  r.start();
  const auto rep = r.send({r.wp(0.05, {0.46, 0, 0.45}), r.wp(0.10, {0.46, 0, 0.90})});
  EXPECT_EQ(rep.accepted, 0u);
  EXPECT_EQ(rep.shortened, 0u);
  ASSERT_EQ(rep.rejected.size(), 2u);
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  EXPECT_EQ(r.sup->gateCounters().rejected, 1u);
  EXPECT_EQ(r.sup->gateCounters().accepted, 0u);
  const LoopCommand* c = r.take();
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->kind, LoopCommand::Kind::kHold);
  EXPECT_NE(r.sup->stateMessage(1).error.find("safety gate rejected waypoint 1 of chunk 1"), std::string::npos);
  EXPECT_FALSE(r.sup->onStart().ok);   // never cleared
}

TEST(Supervisor, ALoopFaultInTheSnapshotFaultsWithItsText) {
  Rig r;
  r.start();
  r.publish([](LoopSnapshot& s) {
    s.fault = true;
    s.phase = LoopSnapshot::Phase::kEnded;
    std::snprintf(s.error.data(), s.error.size(), "libfranka: motion aborted by reflex! [cartesian_reflex]");
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  EXPECT_EQ(r.sup->stateMessage(1).error, "libfranka: motion aborted by reflex! [cartesian_reflex]");
}

TEST(Supervisor, ANonFiniteStateFaultsAndC2KeepsTheLastValidValues) {
  Rig r;
  r.sup->tick();
  r.publish([](LoopSnapshot& s) { s.q(0) = std::nan(""); });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  const msg::ArmState a = r.sup->stateMessage(1);
  EXPECT_EQ(a.error, "arm state not finite");
  EXPECT_TRUE(std::isfinite(a.q[0]));
}

TEST(Supervisor, TheLoopsBackstopTripStops) {
  Rig r;
  r.start();
  r.send({r.wp(0.05, {0.46, 0, 0.45})});
  r.publish([&](LoopSnapshot& s) { s.watchdog_trip_generation = 3; });   // 1 kHold, 2 start, 3 the waypoints
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_NE(r.sup->modeDetail().find("backstop"), std::string::npos);
}

TEST(Supervisor, HomingPausesTheWatchdogAndRepliesWhenTheNewSessionHolds) {
  Rig r;
  EXPECT_FALSE(r.sup->onHome().has_value());   // started: the reply comes later
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);
  const LoopCommand* c = r.take();
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(c->kind, LoopCommand::Kind::kHome);
  const uint64_t gen = c->generation;
  for (int i = 0; i < 100; ++i) {   // 10 s without a request: nothing stops
    r.clock.advance(100 * kMs);
    r.sup->tick();
    ASSERT_EQ(r.sup->mode(), Mode::kHoming);
    ASSERT_FALSE(r.sup->pollHome());
  }
  r.publish([&](LoopSnapshot& s) { s.homed_generation = gen; });   // move done, but still the old session
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);
  r.publish([&](LoopSnapshot& s) { s.session = 1; });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  const auto reply = r.sup->pollHome();
  ASSERT_TRUE(reply);
  EXPECT_TRUE(reply->ok);
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now());   // the watchdog resumes
  EXPECT_EQ(r.take()->kind, LoopCommand::Kind::kHold);
}

TEST(Supervisor, HomingTimeoutStopsTheArmAndFaults) {
  Rig r;
  r.sup->onHome();
  r.shared->beginHomingMove(0);                                                    // the control thread: the move runs
  r.publish([](LoopSnapshot& s) { s.phase = LoopSnapshot::Phase::kHoming; });
  r.clock.advanceS(29.9);
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 0);
  r.clock.advanceS(0.2);
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 1);
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  const auto reply = r.sup->pollHome();
  ASSERT_TRUE(reply);
  EXPECT_FALSE(reply->ok);
  EXPECT_EQ(reply->reason, "homing: not finished after 30.0 s");
}

TEST(Supervisor, AHomingTimeoutStopIsReSentEvery100MsUntilTheMoveIsOver) {
  Rig r;
  r.sup->onHome();
  r.clock.advanceS(30.1);   // timed out while the kHome session still settled: no stop yet (it would hit the session)
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  EXPECT_EQ(r.arm.stops, 0);
  r.shared->beginHomingMove(0);   // the move's libfranka call starts: the stop goes out once its snapshot shows it
  r.clock.advance(10 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 0);
  r.publish([](LoopSnapshot& s) { s.phase = LoopSnapshot::Phase::kHoming; });
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 1);
  for (int i = 0; i < 9; ++i) {   // 90 ms: not again yet
    r.clock.advance(10 * kMs);
    r.sup->tick();
  }
  EXPECT_EQ(r.arm.stops, 1);
  r.clock.advance(10 * kMs);      // 100 ms after the first: the stop may have been lost, send it again
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 2);
  r.shared->endHomingMove();      // the move is over (the snapshot still says kHoming): nothing more reaches the arm
  for (int i = 0; i < 30; ++i) {
    r.clock.advance(10 * kMs);
    r.sup->tick();
  }
  EXPECT_EQ(r.arm.stops, 2);
  r.publish([](LoopSnapshot& s) {   // a new session: resolved
    s.phase = LoopSnapshot::Phase::kTorque;
    s.session = 1;
  });
  r.sup->tick();
  r.shared->beginHomingMove(0);     // (were the stop still pending, this would draw one)
  r.publish([](LoopSnapshot& s) {
    s.phase = LoopSnapshot::Phase::kHoming;
    s.session = 0;
  });
  r.clock.advance(200 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 2);
}

TEST(Supervisor, ATooLongHomeMoveIsRefusedAtOnce) {
  Rig r;
  r.publish([&](LoopSnapshot& s) { s.q(0) = r.cfg.homing.q_home_rad(0) + 10.0; });
  r.sup->tick();
  const auto reply = r.sup->onHome();
  ASSERT_TRUE(reply);
  EXPECT_FALSE(reply->ok);
  EXPECT_EQ(reply->reason,
            "refused: home move would take 62.5 s > homing.timeout_s 30.0 s: move the arm closer by hand");
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  r.sup->onStart();
  EXPECT_FALSE(r.sup->onHome()->ok);   // and from holding: send stop first
}

TEST(Supervisor, AHomeTcpOutsideTheBoxFaults) {
  Rig r;
  r.sup->onHome();
  const uint64_t gen = r.take()->generation;
  r.publish([&](LoopSnapshot& s) {
    s.homed_generation = gen;
    s.session = 1;
    Rig::setPose(s, {0.31, 0.0, -0.03});   // below the floor (config.yaml min z -0.005)
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  const auto reply = r.sup->pollHome();
  ASSERT_TRUE(reply);
  EXPECT_FALSE(reply->ok);
  EXPECT_EQ(reply->reason,
            "home pose TCP (0.31, 0.00, -0.03) m is outside the workspace box: check homing.q_home_rad and the box");
}

TEST(Supervisor, AGripperFaultFaults) {
  Rig r;
  r.start();
  r.shared->setGripperFault("gripper: Franka Hand: connection lost");
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  EXPECT_EQ(r.sup->stateMessage(1).error, "gripper: Franka Hand: connection lost");
  EXPECT_EQ(r.take()->kind, LoopCommand::Kind::kHold);
}

TEST(Supervisor, StopIsNeverRefused) {
  Rig r;
  EXPECT_TRUE(r.sup->onStop().ok);
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_TRUE(r.sup->onStop().ok);
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
}

// ---- the PC keyboard, the hand's homing, stop during homing (control_deploy/STRUCTURE.md 4.3, 4.4, 5.2) ----

std::string stderrOf(const std::function<void()>& f) {
  testing::internal::CaptureStderr();
  f();
  return testing::internal::GetCapturedStderr();
}

void toFault(Rig& r) {
  r.shared->setGripperFault("gripper: lost");
  r.sup->tick();
  ASSERT_EQ(r.sup->mode(), Mode::kFault);
  r.take();
}

TEST(Supervisor, KeysHAndGAreRefusedWhileTheLaptopDrivesHomingOrFault) {
  CountingGripper g;
  for (Mode m : {Mode::kHolding, Mode::kRunning, Mode::kHoming, Mode::kFault}) {
    SCOPED_TRACE(modeName(m));
    Rig r({0.45, 0.0, 0.45}, &g);
    if (m == Mode::kHolding || m == Mode::kRunning) r.start();
    if (m == Mode::kRunning) r.send({r.wp(0.5, {0.46, 0, 0.45})});
    if (m == Mode::kHoming) r.sup->onHome();
    if (m == Mode::kFault) toFault(r);
    r.take();
    ASSERT_EQ(r.sup->mode(), m);
    const std::string log = stderrOf([&] {
      r.sup->onKey(Key::kHomeArm);
      r.sup->onKey(Key::kHomeHand);
    });
    EXPECT_EQ(r.sup->mode(), m);
    EXPECT_EQ(r.take(), nullptr);                      // no loop command
    EXPECT_EQ(r.shared->handHomingRequested(), 0u);    // no hand request
    EXPECT_NE(log.find("key h refused: mode is " + modeName(m)), std::string::npos) << log;
    EXPECT_NE(log.find("key g refused: mode is " + modeName(m)), std::string::npos) << log;
    if (m == Mode::kRunning) {
      EXPECT_NE(log.find("key h refused: mode is running: the laptop is driving an episode: press s or stop it on the "
                         "laptop first"),
                std::string::npos)
          << log;
    }
  }
}

TEST(Supervisor, KeyHHomesTheArmFromIdleAndStoppedWithoutAC3Reply) {
  for (bool from_stopped : {false, true}) {
    Rig r;
    if (from_stopped) {
      r.sup->onKey(Key::kStop);
      ASSERT_EQ(r.sup->mode(), Mode::kStopped);
      r.take();
    }
    const std::string log = stderrOf([&] { r.sup->onKey(Key::kHomeArm); });
    EXPECT_NE(log.find("key h: homing the arm: "), std::string::npos) << log;
    EXPECT_EQ(r.sup->mode(), Mode::kHoming);
    const LoopCommand* c = r.take();
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(c->kind, LoopCommand::Kind::kHome);
    const uint64_t gen = c->generation;
    r.publish([&](LoopSnapshot& s) {
      s.homed_generation = gen;
      s.session = 1;
    });
    r.sup->tick();
    EXPECT_EQ(r.sup->mode(), Mode::kIdle);
    EXPECT_FALSE(r.sup->pollHome());   // a key homing produces no C3 reply
  }
}

TEST(Supervisor, KeyGHomesTheHandWithoutALoopCommand) {
  CountingGripper g;
  Rig r({0.45, 0.0, 0.45}, &g);
  r.sup->onKey(Key::kHomeHand);
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);
  EXPECT_EQ(r.take(), nullptr);   // the 1 kHz loop keeps holding the arm
  const uint64_t gen = r.shared->handHomingRequested();
  ASSERT_NE(gen, 0u);
  for (int i = 0; i < 10; ++i) {   // C3 start and home are refused meanwhile, exactly as during an arm homing
    r.clock.advance(100 * kMs);
    r.sup->tick();
  }
  EXPECT_FALSE(r.sup->onStart().ok);
  EXPECT_FALSE(r.sup->onHome()->ok);
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);
  r.shared->reportHandHoming(gen, SharedState::kHandHomingDone);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now());   // markAlive
  EXPECT_EQ(r.take(), nullptr);
  EXPECT_FALSE(r.sup->pollHome());
  EXPECT_EQ(g.stops, 0);
}

TEST(Supervisor, AHandHomingTimeoutStopsTheHandAndFaults) {
  CountingGripper g;
  Rig r({0.45, 0.0, 0.45}, &g);
  ASSERT_FALSE(r.sup->onHandHome());
  ASSERT_TRUE(r.shared->beginHandHoming(r.shared->handHomingRequested()));   // the gripper thread: home() runs
  r.clock.advanceS(29.9);
  r.sup->tick();
  EXPECT_EQ(g.stops, 0);
  r.clock.advanceS(0.2);
  r.sup->tick();
  EXPECT_EQ(g.stops, 1);
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  EXPECT_EQ(r.sup->stateMessage(1).error, "hand homing: not finished after 30.0 s");
  EXPECT_EQ(r.arm.stops, 0);
}

TEST(Supervisor, KeyGWithoutAGripperIsRefused) {
  Rig r;
  const auto reply = r.sup->onHandHome();
  ASSERT_TRUE(reply);
  EXPECT_FALSE(reply->ok);
  EXPECT_EQ(reply->reason, "refused: no gripper");
  const std::string log = stderrOf([&] { r.sup->onKey(Key::kHomeHand); });
  EXPECT_NE(log.find("key g refused: no gripper"), std::string::npos) << log;
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
}

TEST(Supervisor, StopDuringAnArmHomingAbortsItAndTheC3HomeRepliesStopped) {
  Rig r;
  ASSERT_FALSE(r.sup->onHome().has_value());   // a C3 home
  const uint64_t gen = r.take()->generation;
  r.sup->onKey(Key::kStop);                    // the PC's s (the REP socket is busy with the home reply)
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);     // until the outcome is known
  EXPECT_TRUE(r.shared->homingAbortRequested());
  for (int i = 0; i < 3; ++i) {                // the session is still settling: no robot.stop()
    r.clock.advance(5 * kMs);
    r.sup->tick();
  }
  EXPECT_EQ(r.arm.stops, 0);
  r.shared->beginHomingMove(0);                // the control thread: the move runs
  r.publish([](LoopSnapshot& s) { s.phase = LoopSnapshot::Phase::kHoming; });
  for (int i = 0; i < 5; ++i) {
    r.clock.advance(5 * kMs);
    r.sup->tick();
  }
  EXPECT_EQ(r.arm.stops, 1);                   // once (again only after 100 ms)
  EXPECT_FALSE(r.sup->pollHome());
  r.clock.advance(5 * kMs);
  r.shared->endHomingMove();
  r.shared->clearHomingAbort();                // the control thread: stopped, a new session holds where it stopped
  r.publish([&](LoopSnapshot& s) {
    s.phase = LoopSnapshot::Phase::kTorque;
    s.session = 1;
    s.homing_aborted_generation = gen;
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_EQ(r.sup->modeDetail(), "stop during homing (PC keyboard)");
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now());
  const LoopCommand* c = r.take();
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->kind, LoopCommand::Kind::kHold);
  const auto reply = r.sup->pollHome();
  ASSERT_TRUE(reply);
  EXPECT_FALSE(reply->ok);
  EXPECT_EQ(reply->reason.rfind("stopped: ", 0), 0u) << reply->reason;
  EXPECT_EQ(reply->reason,
            "stopped: homing stopped by the PC keyboard before the home pose was reached: the arm holds where it "
            "stopped");
  EXPECT_TRUE(r.sup->onStart().ok);            // the laptop can start from there
}

TEST(Supervisor, AnAbortStopIsReSentWhileTheMoveRunsAndNeverAfterItEnded) {
  Rig r;
  ASSERT_FALSE(r.sup->onHome().has_value());
  const uint64_t gen = r.take()->generation;
  r.sup->onStop();
  r.shared->beginHomingMove(0);
  r.publish([](LoopSnapshot& s) { s.phase = LoopSnapshot::Phase::kHoming; });
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 1);
  r.clock.advance(100 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 2);   // the move still runs: sent again
  // The move finished normally just now (the control thread ended it, the abort flag is still set and the newest
  // snapshot still says kHoming): a stop now would preempt the next torque session and fault it.
  r.shared->endHomingMove();
  for (int i = 0; i < 5; ++i) {
    r.clock.advance(100 * kMs);
    r.sup->tick();
  }
  EXPECT_EQ(r.arm.stops, 2);
  r.shared->clearHomingAbort();
  r.publish([&](LoopSnapshot& s) {
    s.phase = LoopSnapshot::Phase::kTorque;
    s.session = 1;
    s.homed_generation = gen;
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);   // the normal completion wins
  EXPECT_EQ(r.arm.stops, 2);
}

TEST(Supervisor, AC3StopDuringAKeyboardHomingIsAcknowledgedAndAborts) {
  Rig r;
  r.sup->onKey(Key::kHomeArm);
  const uint64_t gen = r.take()->generation;
  const msg::Ack ack = r.sup->onStop();        // C3 stop: never refused
  EXPECT_TRUE(ack.ok);
  EXPECT_TRUE(r.shared->homingAbortRequested());
  r.publish([&](LoopSnapshot& s) {             // the control thread skipped the move (abort before it started)
    s.session = 1;
    s.homing_aborted_generation = gen;
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_EQ(r.sup->modeDetail(), "stop during homing (C3)");
  EXPECT_EQ(r.arm.stops, 0);
  EXPECT_FALSE(r.sup->pollHome());             // a keyboard homing: no C3 reply
}

TEST(Supervisor, AMoveThatFinishedFirstWins) {
  Rig r;
  r.sup->onHome();
  const uint64_t gen = r.take()->generation;
  r.sup->onKey(Key::kStop);
  r.publish([&](LoopSnapshot& s) {
    s.homed_generation = gen;
    s.session = 1;
  });
  const std::string log = stderrOf([&] { r.sup->tick(); });
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  EXPECT_NE(log.find("stop arrived after the home pose was reached"), std::string::npos) << log;
  const auto reply = r.sup->pollHome();
  ASSERT_TRUE(reply);
  EXPECT_TRUE(reply->ok);
  EXPECT_FALSE(r.shared->homingAbortRequested());   // nothing stale for the next homing
}

TEST(Supervisor, NoOtherHomeReplyStartsWithStopped) {
  std::vector<std::string> reasons;
  {   // refused (holding)
    Rig r;
    r.start();
    reasons.push_back(r.sup->onHome()->reason);
  }
  {   // refused (too long a move)
    Rig r;
    r.publish([&](LoopSnapshot& s) { s.q(0) = r.cfg.homing.q_home_rad(0) + 10.0; });
    r.sup->tick();
    reasons.push_back(r.sup->onHome()->reason);
  }
  {   // the box check
    Rig r;
    r.sup->onHome();
    const uint64_t gen = r.take()->generation;
    r.publish([&](LoopSnapshot& s) {
      s.homed_generation = gen;
      s.session = 1;
      Rig::setPose(s, {0.31, 0.0, -0.03});   // below the floor (config.yaml min z -0.005)
    });
    r.sup->tick();
    reasons.push_back(r.sup->pollHome()->reason);
  }
  {   // a loop fault
    Rig r;
    r.sup->onHome();
    r.publish([](LoopSnapshot& s) {
      s.fault = true;
      s.phase = LoopSnapshot::Phase::kEnded;
      std::snprintf(s.error.data(), s.error.size(), "libfranka: motion aborted by reflex! [joint_reflex]");
    });
    r.sup->tick();
    reasons.push_back(r.sup->pollHome()->reason);
  }
  {   // a gripper fault
    Rig r;
    r.sup->onHome();
    r.shared->setGripperFault("gripper: Franka Hand: connection lost");
    r.sup->tick();
    reasons.push_back(r.sup->pollHome()->reason);
  }
  {   // a non-finite state
    Rig r;
    r.sup->onHome();
    r.publish([](LoopSnapshot& s) { s.q(0) = std::nan(""); });
    r.sup->tick();
    reasons.push_back(r.sup->pollHome()->reason);
  }
  {   // the timeout
    Rig r;
    r.sup->onHome();
    r.clock.advanceS(30.1);
    r.sup->tick();
    reasons.push_back(r.sup->pollHome()->reason);
  }
  ASSERT_EQ(reasons.size(), 7u);
  for (const auto& reason : reasons) {
    EXPECT_NE(reason.rfind(kHomeStoppedPrefix, 0), 0u) << reason;
    EXPECT_FALSE(reason.empty());
  }
}

TEST(Supervisor, StopDuringTheHandsHomingStopsTheHand) {
  CountingGripper g;
  Rig r({0.45, 0.0, 0.45}, &g);
  r.sup->onKey(Key::kHomeHand);
  const uint64_t gen = r.shared->handHomingRequested();
  ASSERT_TRUE(r.shared->beginHandHoming(gen));      // the gripper thread: home() runs
  r.sup->onKey(Key::kStop);
  EXPECT_EQ(g.stops, 1);
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);
  EXPECT_FALSE(r.shared->homingAbortRequested());   // the arm's flag is untouched
  r.sup->onKey(Key::kStop);                          // a second s changes nothing
  EXPECT_EQ(g.stops, 1);
  r.shared->reportHandHoming(gen, SharedState::kHandHomingAborted);
  r.clock.advance(10 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now());
  EXPECT_EQ(r.arm.stops, 0);
}

TEST(Supervisor, AStopBeforeTheHandsHomingBeganSkipsItWithoutStoppingTheHand) {
  CountingGripper g;   // the gripper thread may be in a finger move: a hand stop would abort it (a gripper fault)
  Rig r({0.45, 0.0, 0.45}, &g);
  r.sup->onKey(Key::kHomeHand);
  const uint64_t gen = r.shared->handHomingRequested();
  const std::string log = stderrOf([&] { r.sup->onKey(Key::kStop); });
  EXPECT_NE(log.find("stop during hand homing (PC keyboard): the hand's homing will not start"), std::string::npos)
      << log;
  EXPECT_EQ(g.stops, 0);
  EXPECT_FALSE(r.shared->beginHandHoming(gen));     // the gripper thread skips home()
  r.shared->reportHandHoming(gen, SharedState::kHandHomingAborted);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_EQ(g.stops, 0);
}

TEST(Supervisor, AStopAfterTheHandsHomingEndedStopsNothing) {
  CountingGripper g;
  Rig r({0.45, 0.0, 0.45}, &g);
  r.sup->onKey(Key::kHomeHand);
  const uint64_t gen = r.shared->handHomingRequested();
  ASSERT_TRUE(r.shared->beginHandHoming(gen));
  r.shared->endHandHoming(gen);                     // home() returned; the re-sent finger move may start now
  r.sup->onKey(Key::kStop);
  EXPECT_EQ(g.stops, 0);
  r.shared->reportHandHoming(gen, SharedState::kHandHomingDone);
  const std::string log = stderrOf([&] { r.sup->tick(); });
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  EXPECT_NE(log.find("stop arrived after the hand's homing had finished"), std::string::npos) << log;
}

// ---- key p, the start pose, and C3 start's start-pose check ----

Vec7 startPose() {   // a start pose near the test config's home: 0.48 rad on joint 6 (a 3.0 s move)
  Vec7 q = test::testConfig().homing.q_home_rad;
  q(5) += 0.48;
  return q;
}

void withStart(Config& c) { c.homing.q_start_rad = startPose(); }

// A tilt of `deg` from the cone axis (straight down), measured on the flange's approach axis.
void setTilted(LoopSnapshot& s, const Config& cfg, const Eigen::Vector3d& p, double deg) {
  const Eigen::Matrix3d R_F =
      Eigen::AngleAxisd(deg * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix() *
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
  s.O_T_TCP = Pose::Identity();
  s.O_T_TCP.linear() = R_F * cfg.F_T_TCP.linear();
  s.O_T_TCP.translation() = p;
  s.O_T_TCP_target = s.O_T_TCP;
}

TEST(Supervisor, KeyPIsRefusedWhileQStartIsUnset) {
  Rig r;
  ASSERT_FALSE(r.cfg.homing.q_start_rad);   // config.yaml ships null
  const auto reply = r.sup->onStartPose();
  ASSERT_TRUE(reply);
  EXPECT_FALSE(reply->ok);
  EXPECT_EQ(reply->reason, "refused: homing.q_start_rad is not set: measure it and set it in control_deploy/station.yaml (franka_control: homing:)");
  const std::string log = stderrOf([&] { r.sup->onKey(Key::kStartPose); });
  EXPECT_NE(log.find("key p refused: homing.q_start_rad is not set: measure it and set it in control_deploy/station.yaml (franka_control: homing:)"),
            std::string::npos)
      << log;
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  EXPECT_EQ(r.take(), nullptr);
}

TEST(Supervisor, KeyPIsRefusedAwayFromHome) {
  Rig r({0.45, 0.0, 0.45}, nullptr, withStart);
  r.publish([&](LoopSnapshot& s) { s.q(3) = r.cfg.homing.q_home_rad(3) + 0.06; });   // > at_pose_tol_rad 0.05
  r.sup->tick();
  const auto reply = r.sup->onStartPose();
  ASSERT_TRUE(reply);
  EXPECT_FALSE(reply->ok);
  EXPECT_EQ(reply->reason, "refused: not at home: press h first");
  const std::string log = stderrOf([&] { r.sup->onKey(Key::kStartPose); });
  EXPECT_NE(log.find("key p refused: not at home: press h first"), std::string::npos) << log;
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  EXPECT_EQ(r.take(), nullptr);
  r.publish([&](LoopSnapshot& s) { s.q(3) = r.cfg.homing.q_home_rad(3) + 0.04; });   // within the tolerance
  r.sup->tick();
  EXPECT_FALSE(r.sup->onStartPose());
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);
}

TEST(Supervisor, KeyPIsRefusedOutsideIdleAndStopped) {
  CountingGripper g;
  for (Mode m : {Mode::kHolding, Mode::kRunning, Mode::kHoming, Mode::kFault}) {
    SCOPED_TRACE(modeName(m));
    Rig r({0.45, 0.0, 0.45}, &g, withStart);
    if (m == Mode::kHolding || m == Mode::kRunning) {
      r.publish([](LoopSnapshot& s) { s.q = startPose(); });   // C3 start needs the start pose
      r.sup->tick();
      r.start();
    }
    if (m == Mode::kRunning) r.send({r.wp(0.5, {0.46, 0, 0.45})});
    if (m == Mode::kHoming) r.sup->onHome();
    if (m == Mode::kFault) toFault(r);
    r.take();
    ASSERT_EQ(r.sup->mode(), m);
    const std::string log = stderrOf([&] { r.sup->onKey(Key::kStartPose); });
    EXPECT_EQ(r.sup->mode(), m);
    EXPECT_EQ(r.take(), nullptr);
    EXPECT_NE(log.find("key p refused: mode is " + modeName(m)), std::string::npos) << log;
  }
}

TEST(Supervisor, KeyPMovesFromHomeToTheStartPoseAndOpensTheHand) {
  Rig r({0.45, 0.0, 0.45}, nullptr, withStart);
  const std::string log = stderrOf([&] { r.sup->onKey(Key::kStartPose); });
  EXPECT_NE(log.find("key p: start pose: 3.0 s joint move to homing.q_start_rad"), std::string::npos) << log;
  EXPECT_NE(log.find("key p: the hand opens to 0.080 m (gripper_width_max_m) during the move"), std::string::npos)
      << log;
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);
  EXPECT_EQ(r.sup->stateMessage(1).mode, "homing");
  const LoopCommand* c = r.take();
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(c->kind, LoopCommand::Kind::kHome);
  ASSERT_TRUE(c->q_target);
  EXPECT_TRUE(c->q_target->isApprox(startPose()));
  ASSERT_TRUE(c->open_width_m);
  EXPECT_DOUBLE_EQ(*c->open_width_m, r.cfg.limits.gripper_width_max_m);
  EXPECT_FALSE(r.sup->onStart().ok);   // refused while homing
  const uint64_t gen = c->generation;
  r.publish([&](LoopSnapshot& s) {
    s.homed_generation = gen;
    s.session = 1;
    s.q = startPose();
  });
  const std::string done = stderrOf([&] { r.sup->tick(); });
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  EXPECT_NE(done.find("start pose reached"), std::string::npos) << done;
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now());
  EXPECT_EQ(r.take()->kind, LoopCommand::Kind::kHold);
  EXPECT_FALSE(r.sup->pollHome());     // a key: no C3 reply
  EXPECT_TRUE(r.sup->onStart().ok);    // the laptop starts from the start pose
}

TEST(Supervisor, KeyHStillGoesToQHomeAndLeavesTheFingers) {
  Rig r({0.45, 0.0, 0.45}, nullptr, withStart);
  r.sup->onKey(Key::kHomeArm);
  const LoopCommand* c = r.take();
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(c->kind, LoopCommand::Kind::kHome);
  ASSERT_TRUE(c->q_target);
  EXPECT_TRUE(c->q_target->isApprox(r.cfg.homing.q_home_rad));
  EXPECT_FALSE(c->open_width_m);
}

TEST(Supervisor, StopDuringKeyPAbortsItLikeAHoming) {
  Rig r({0.45, 0.0, 0.45}, nullptr, withStart);
  r.sup->onKey(Key::kStartPose);
  const uint64_t gen = r.take()->generation;
  r.sup->onKey(Key::kStop);
  EXPECT_EQ(r.sup->mode(), Mode::kHoming);
  EXPECT_TRUE(r.shared->homingAbortRequested());
  r.shared->beginHomingMove(0);
  r.publish([](LoopSnapshot& s) { s.phase = LoopSnapshot::Phase::kHoming; });
  r.sup->tick();
  EXPECT_EQ(r.arm.stops, 1);
  r.shared->endHomingMove();
  r.shared->clearHomingAbort();
  r.publish([&](LoopSnapshot& s) {
    s.phase = LoopSnapshot::Phase::kTorque;
    s.session = 1;
    s.homing_aborted_generation = gen;
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_EQ(r.sup->modeDetail(), "stop during homing (PC keyboard)");
  EXPECT_EQ(r.take()->kind, LoopCommand::Kind::kHold);
  EXPECT_FALSE(r.sup->onStart().ok);   // stopped on the way: not at the start pose
}

TEST(Supervisor, AStartPoseOutsideTheBoxFaults) {
  Rig r({0.45, 0.0, 0.45}, nullptr, withStart);
  r.sup->onStartPose();
  const uint64_t gen = r.take()->generation;
  r.publish([&](LoopSnapshot& s) {
    s.homed_generation = gen;
    s.session = 1;
    Rig::setPose(s, {0.31, 0.0, -0.03});   // below the floor (config.yaml min z -0.005)
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  EXPECT_EQ(r.sup->stateMessage(1).error,
            "start pose TCP (0.31, 0.00, -0.03) m is outside the workspace box: check homing.q_start_rad and the box");
  EXPECT_EQ(r.take()->kind, LoopCommand::Kind::kHold);
}

TEST(Supervisor, AStartPoseOutsideTheConeFaultsWithoutTheMargin) {
  for (double deg : {59.0, 61.0, 70.0}) {   // cone_max_deg 60, cone_margin_deg 2: no margin for the start pose
    SCOPED_TRACE(deg);
    Rig r({0.45, 0.0, 0.45}, nullptr, withStart);
    r.sup->onStartPose();
    const uint64_t gen = r.take()->generation;
    r.publish([&](LoopSnapshot& s) {
      s.homed_generation = gen;
      s.session = 1;
      setTilted(s, r.cfg, {0.45, 0.0, 0.30}, deg);
    });
    r.sup->tick();
    if (deg < 60.0) {
      EXPECT_EQ(r.sup->mode(), Mode::kIdle);
    } else {
      EXPECT_EQ(r.sup->mode(), Mode::kFault);
      char want[160];
      std::snprintf(want, sizeof(want),
                    "start pose tilt %.1f deg from the approach axis > safety.cone_max_deg 60.0: check "
                    "homing.q_start_rad and the cone",
                    deg);
      EXPECT_EQ(r.sup->stateMessage(1).error, want);
    }
  }
}

TEST(Supervisor, AHomeIsNotCheckedAgainstTheCone) {   // h unchanged: only the box
  Rig r;
  r.sup->onHome();
  const uint64_t gen = r.take()->generation;
  r.publish([&](LoopSnapshot& s) {
    s.homed_generation = gen;
    s.session = 1;
    setTilted(s, r.cfg, {0.45, 0.0, 0.30}, 70.0);
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
}

TEST(Supervisor, C3StartNeedsTheStartPoseOnceItIsSet) {
  Rig r({0.45, 0.0, 0.45}, nullptr, withStart);   // at home
  r.alive();
  msg::Ack a = r.sup->onStart();
  EXPECT_FALSE(a.ok);
  EXPECT_EQ(a.reason, "refused: not at the start pose: press h, then p on the PC");
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  EXPECT_EQ(r.take(), nullptr);
  r.publish([](LoopSnapshot& s) {
    s.q = startPose();
    s.q(0) += 0.049;   // within at_pose_tol_rad
  });
  r.sup->tick();
  EXPECT_TRUE(r.sup->onStart().ok);
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);
}

TEST(Supervisor, C3StartIsUnchangedWhileQStartIsUnset) {
  Rig r;
  r.publish([](LoopSnapshot& s) { s.q(0) += 1.0; });   // anywhere
  r.sup->tick();
  EXPECT_TRUE(r.sup->onStart().ok);
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);
}

TEST(Supervisor, KeySStopsInIdleAndChangesNothingInFault) {
  Rig r;
  r.sup->onKey(Key::kStop);
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_EQ(r.take()->kind, LoopCommand::Kind::kHold);
  const std::string again = stderrOf([&] { r.sup->onKey(Key::kStop); });
  EXPECT_NE(again.find("key s: already stopped"), std::string::npos) << again;
  Rig f;
  toFault(f);
  const std::string log = stderrOf([&] { f.sup->onKey(Key::kStop); });
  EXPECT_EQ(f.sup->mode(), Mode::kFault);
  EXPECT_EQ(f.take(), nullptr);
  EXPECT_NE(log.find("Ctrl-C and restart after a fault"), std::string::npos) << log;
}

TEST(Supervisor, KeysNeverFeedTheWatchdog) {
  Rig r;
  r.start();
  const int64_t alive = r.shared->lastAliveNs();
  r.clock.advance(500 * kMs);
  r.sup->onKey(Key::kHomeArm);   // refused (holding)
  r.sup->onKey(Key::kHomeHand);  // refused (no gripper)
  EXPECT_EQ(r.shared->lastAliveNs(), alive);
  r.clock.advance(501 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);   // the watchdog tripped
}

}  // namespace
}  // namespace franka_control
