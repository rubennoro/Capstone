// Supervisor with a FakeClock, a SharedState and a test Arm; no threads. The test plays the control thread: it
// publishes snapshots and takes the posted commands.
#include <gtest/gtest.h>

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

constexpr int64_t kMs = 1000000;

struct Rig {
  explicit Rig(const Eigen::Vector3d& target = {0.45, 0.0, 0.45}) : cfg(test::testConfig()) {
    shared = std::make_unique<SharedState>();
    shared->setGripperWidth(0.04);
    publish([&](LoopSnapshot& s) { setPose(s, target); });
    sup = std::make_unique<Supervisor>(*shared, arm, cfg, cfg.limits, clock.clock());
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

TEST(Supervisor, NoRequestForHalfASecondStops) {
  Rig r;
  r.start();
  r.clock.advance(499 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);
  r.clock.advance(1 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kHolding);   // exactly at the limit: not yet
  r.clock.advance(2 * kMs);
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
  EXPECT_NE(r.sup->modeDetail().find("watchdog: no request for 0.502 s"), std::string::npos);
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
    Rig::setPose(s, {0.31, 0.0, 0.03});
  });
  r.sup->tick();
  EXPECT_EQ(r.sup->mode(), Mode::kFault);
  const auto reply = r.sup->pollHome();
  ASSERT_TRUE(reply);
  EXPECT_FALSE(reply->ok);
  EXPECT_EQ(reply->reason,
            "home pose TCP (0.31, 0.00, 0.03) m is outside the workspace box: check homing.q_home_rad and the box");
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

}  // namespace
}  // namespace franka_control
