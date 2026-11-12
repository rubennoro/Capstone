#include <gtest/gtest.h>

#include "franka_control/state_publisher.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

class StillArm : public Arm {
 public:
  ArmReading readOnce() override { return {}; }
  ArmStatusFlags status() override { return {}; }
  void runTorqueControl(const TorqueStep&) override {}
  void moveToJoints(const HomingMotion&, const Observer&) override {}
  void requestStop() override {}
  std::string describe() const override { return "StillArm"; }
};

constexpr int64_t kMs = 1000000;

struct Rig {
  Rig() : cfg(test::testConfig()), codec(cfg.limits) {
    shared = std::make_unique<SharedState>();
    publishSnapshot([](LoopSnapshot&) {});
    sup = std::make_unique<Supervisor>(*shared, arm, cfg, cfg.limits, clock.clock());
    pub = std::make_unique<StatePublisher>(ctx, "tcp://127.0.0.1:*", 1000, 100.0, *sup, codec, clock.clock());
    sub.set(zmq::sockopt::subscribe, "");
    sub.connect(pub->endpoint());
  }
  void publishSnapshot(const std::function<void(LoopSnapshot&)>& f) {
    snap.phase = LoopSnapshot::Phase::kTorque;
    snap.t_ns = clock.now();
    snap.O_T_TCP = test::makePose({0.45, 0, 0.45}, test::rotvec({M_PI, 0, 0}));
    snap.O_T_TCP_target = snap.O_T_TCP;
    snap.q = cfg.homing.q_home_rad;
    f(snap);
    shared->snapshotSlot() = snap;
    shared->publishSnapshot();
  }
  std::optional<msg::ArmState> receive(int timeout_ms) {
    zmq::pollitem_t items[] = {{static_cast<void*>(sub), 0, ZMQ_POLLIN, 0}};
    zmq::poll(items, 1, std::chrono::milliseconds(timeout_ms));
    if (!(items[0].revents & ZMQ_POLLIN)) return std::nullopt;
    zmq::message_t m;
    (void)sub.recv(m);
    return std::get<msg::ArmState>(codec.decode(std::string(static_cast<const char*>(m.data()), m.size())));
  }
  // Publishes one message per period until the subscriber gets one (the TCP connection needs a moment).
  msg::ArmState next() {
    for (int i = 0; i < 500; ++i) {
      clock.advance(10 * kMs);
      sup->tick();
      shared->markAlive(clock.now());
      pub->maybePublish();
      if (auto m = receive(2)) return *m;
    }
    ADD_FAILURE() << "no C2 message";
    return {};
  }

  Config cfg;
  Codec codec;
  test::FakeClock clock;
  StillArm arm;
  zmq::context_t ctx{1};
  std::unique_ptr<SharedState> shared;
  LoopSnapshot snap;
  std::unique_ptr<Supervisor> sup;
  std::unique_ptr<StatePublisher> pub;
  zmq::socket_t sub{ctx, zmq::socket_type::sub};
};

TEST(StatePublisher, OneMessagePerPeriodAndSeqUpByOne) {
  Rig r;
  EXPECT_TRUE(r.pub->maybePublish());    // the first is due at once
  EXPECT_FALSE(r.pub->maybePublish());
  EXPECT_EQ(r.pub->nsUntilNext(), 10 * kMs);
  r.clock.advance(9 * kMs);
  EXPECT_FALSE(r.pub->maybePublish());
  r.clock.advance(1 * kMs);
  EXPECT_TRUE(r.pub->maybePublish());
  EXPECT_EQ(r.pub->lastSeq(), 2u);
  const msg::ArmState a = r.next();
  const msg::ArmState b = r.next();
  EXPECT_EQ(b.seq, a.seq + 1);
  EXPECT_EQ(r.sup->stateMessage(r.pub->lastSeq()).seq, r.pub->lastSeq());   // get_state: the last seq, not +1
}

TEST(StatePublisher, EveryModeErrorGateAndLoopLateAreMapped) {
  Rig r;
  EXPECT_EQ(r.next().mode, "idle");
  r.sup->onHome();
  EXPECT_EQ(r.next().mode, "homing");
  const uint64_t gen = r.shared->takeCommand()->generation;
  r.publishSnapshot([&](LoopSnapshot& s) {
    s.homed_generation = gen;
    s.session = 1;
    s.late_cycles_total = 7;
  });
  msg::ArmState m = r.next();
  EXPECT_EQ(m.mode, "idle");
  EXPECT_EQ(m.loop_late, 7u);
  r.sup->onStop();
  EXPECT_EQ(r.next().mode, "stopped");
  r.sup->onStart();
  EXPECT_EQ(r.next().mode, "holding");
  const auto down = test::rotvec({M_PI, 0, 0});
  r.sup->onSetWaypoints(test::makeSetWaypoints(1, {test::point(r.clock.now() + 1000 * kMs, {0.801, 0, 0.45}, down,
                                                               0.04)}));
  m = r.next();
  EXPECT_EQ(m.mode, "running");
  EXPECT_EQ(m.gate.accepted, 1u);
  EXPECT_EQ(m.gate.pulled_back, 1u);
  EXPECT_EQ(m.error, "");
  r.sup->onSetWaypoints(test::makeSetWaypoints(2, {test::point(r.clock.now() + 50 * kMs, {0.5, 0, 0.95}, down,
                                                               0.04)}));
  m = r.next();
  EXPECT_EQ(m.mode, "fault");
  EXPECT_EQ(m.gate.rejected, 1u);
  EXPECT_NE(m.error.find("safety gate rejected"), std::string::npos);
}

TEST(StatePublisher, MessagesKeepComingWithTheSameTnsWhenTheLoopHasEnded) {
  Rig r;
  r.publishSnapshot([](LoopSnapshot& s) {
    s.fault = true;
    s.phase = LoopSnapshot::Phase::kEnded;
    std::snprintf(s.error.data(), s.error.size(), "libfranka: connection lost");
  });
  const int64_t t = r.snap.t_ns;
  msg::ArmState prev = r.next();
  for (int i = 0; i < 5; ++i) {
    const msg::ArmState m = r.next();
    EXPECT_EQ(m.seq, prev.seq + 1);
    EXPECT_EQ(m.t_ns, t);
    EXPECT_EQ(m.mode, "fault");
    prev = m;
  }
}

TEST(StatePublisher, NeverBlocksWithoutASubscriber) {
  test::FakeClock clock;
  const Config cfg = test::testConfig();
  const Codec codec(cfg.limits);
  StillArm arm;
  auto shared = std::make_unique<SharedState>();
  Supervisor sup(*shared, arm, cfg, cfg.limits, clock.clock());
  zmq::context_t ctx(1);
  StatePublisher pub(ctx, "tcp://127.0.0.1:*", 2, 100.0, sup, codec, clock.clock());
  zmq::socket_t slow(ctx, zmq::socket_type::sub);   // connected, never reads
  slow.set(zmq::sockopt::rcvhwm, 1);
  slow.set(zmq::sockopt::subscribe, "");
  slow.connect(pub.endpoint());
  const int64_t t0 = monotonicNowNs();
  for (int i = 0; i < 2000; ++i) {
    clock.advance(10 * kMs);
    EXPECT_TRUE(pub.maybePublish());
  }
  EXPECT_EQ(pub.lastSeq(), 2000u);
  EXPECT_LT(nsToSeconds(monotonicNowNs() - t0), 10.0);
}

}  // namespace
}  // namespace franka_control
