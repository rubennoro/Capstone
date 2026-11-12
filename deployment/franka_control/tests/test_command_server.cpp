// CommandServer over tcp://127.0.0.1:* with a test REQ client; single thread: send, serviceOnce, receive.
#include <gtest/gtest.h>

#include "franka_control/command_server.h"
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

struct Rig {
  Rig() : cfg(test::testConfig()), codec(cfg.limits) {
    shared = std::make_unique<SharedState>();
    publishSnapshot([](LoopSnapshot&) {});
    sup = std::make_unique<Supervisor>(*shared, arm, cfg, cfg.limits, clock.clock());
    pub = std::make_unique<StatePublisher>(ctx, "tcp://127.0.0.1:*", 10, 100.0, *sup, codec, clock.clock());
    server = std::make_unique<CommandServer>(ctx, "tcp://127.0.0.1:*", *sup, *pub, codec, *shared, clock.clock());
    client = std::make_unique<test::ReqClient>(ctx, server->endpoint());
  }
  void publishSnapshot(const std::function<void(LoopSnapshot&)>& f) {
    snap.phase = LoopSnapshot::Phase::kTorque;
    snap.O_T_TCP = test::makePose({0.45, 0, 0.45}, test::rotvec({M_PI, 0, 0}));
    snap.O_T_TCP_target = snap.O_T_TCP;
    snap.q = cfg.homing.q_home_rad;
    f(snap);
    shared->snapshotSlot() = snap;
    shared->publishSnapshot();
  }
  // Sends, lets the server answer (up to a few tries for the TCP connection to come up), returns the decoded reply.
  msg::Any request(const std::string& bytes) {
    client->send(bytes);
    for (int i = 0; i < 200; ++i) {
      server->serviceOnce(5000000);
      if (auto reply = client->receive(1)) return codec.decode(*reply);
    }
    ADD_FAILURE() << "no reply";
    return msg::ErrorReply{};
  }
  msg::Any request(const msg::Any& m) { return request(codec.encode(m)); }

  Config cfg;
  Codec codec;
  test::FakeClock clock;
  StillArm arm;
  zmq::context_t ctx{1};
  std::unique_ptr<SharedState> shared;
  LoopSnapshot snap;
  std::unique_ptr<Supervisor> sup;
  std::unique_ptr<StatePublisher> pub;
  std::unique_ptr<CommandServer> server;
  std::unique_ptr<test::ReqClient> client;
};

std::string example(const std::string& name) { return test::readFile(test::examplesDir() / name); }

TEST(CommandServer, EachRequestGetsItsReplyType) {
  Rig r;
  EXPECT_TRUE(std::get<msg::Ack>(r.request(msg::Heartbeat{})).ok);
  EXPECT_EQ(std::get<msg::ArmState>(r.request(msg::GetState{})).mode, "idle");
  EXPECT_TRUE(std::get<msg::Ack>(r.request(msg::Start{})).ok);
  const auto wp = test::makeSetWaypoints(
      4, {test::point(r.clock.now() + 50000000, {0.46, 0, 0.45}, test::rotvec({M_PI, 0, 0}), 0.04)});
  const auto rep = std::get<msg::SetWaypointsReply>(r.request(wp));
  EXPECT_EQ(rep.chunk_id, 4u);
  EXPECT_EQ(rep.accepted, 1u);
  EXPECT_TRUE(std::get<msg::Ack>(r.request(msg::Stop{})).ok);
  // set_waypoints refused by the state (stopped): error/v1, not a fault
  const auto refused = std::get<msg::ErrorReply>(r.request(wp));
  EXPECT_EQ(refused.request, "c3.set_waypoints/v1");
  EXPECT_EQ(r.sup->mode(), Mode::kStopped);
}

TEST(CommandServer, BrokenRequestsGetAnErrorReplyNamingTheField) {
  Rig r;
  auto e = std::get<msg::ErrorReply>(r.request(example("c3/set_waypoints_wrong_shape.msgpack")));
  EXPECT_EQ(e.field, "O_T_TCP");
  EXPECT_EQ(e.request, "c3.set_waypoints/v1");
  e = std::get<msg::ErrorReply>(r.request(example("c3/not_msgpack.msgpack")));
  EXPECT_EQ(e.request, "");
  e = std::get<msg::ErrorReply>(r.request(example("c3/unknown_schema.msgpack")));
  EXPECT_NE(e.error.find("unknown message name"), std::string::npos);
  e = std::get<msg::ErrorReply>(r.request(example("c3/set_waypoints_schema_version.msgpack")));
  EXPECT_NE(e.error.find("version v2 not supported"), std::string::npos);
  e = std::get<msg::ErrorReply>(r.request(example("c2/state_running.msgpack")));   // C2 sent as a request
  EXPECT_NE(e.error.find("unexpected message"), std::string::npos);
  EXPECT_EQ(e.request, "c2.state/v1");
  EXPECT_TRUE(std::get<msg::Ack>(r.request(msg::Heartbeat{})).ok);   // and it keeps serving
}

TEST(CommandServer, EveryDecodedRequestFeedsTheWatchdogABrokenOneDoesNot) {
  Rig r;
  r.clock.advance(1000);
  r.request(msg::GetState{});
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now());
  r.clock.advance(1000);
  r.request(example("c3/heartbeat_unknown_field.msgpack"));
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now() - 1000);
  r.clock.advance(1000);
  r.request(msg::Stop{});
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now());
}

TEST(CommandServer, TheHomeReplyIsDeferredWhileC2KeepsPublishing) {
  Rig r;
  r.client->send(r.codec.encode(msg::Home{}));
  for (int i = 0; i < 200 && !r.server->homeReplyPending(); ++i) r.server->serviceOnce(5000000);
  ASSERT_TRUE(r.server->homeReplyPending());
  ASSERT_EQ(r.sup->mode(), Mode::kHoming);
  const LoopCommand* c = r.shared->takeCommand();
  ASSERT_NE(c, nullptr);
  const uint64_t gen = c->generation;
  EXPECT_EQ(c->kind, LoopCommand::Kind::kHome);
  const uint64_t seq0 = r.pub->lastSeq();
  for (int i = 0; i < 5; ++i) {
    r.clock.advance(10000000);
    r.sup->tick();
    r.server->serviceOnce(0);
    EXPECT_TRUE(r.pub->maybePublish());
    EXPECT_FALSE(r.client->receive(0));   // no reply yet
  }
  EXPECT_EQ(r.pub->lastSeq(), seq0 + 5);
  r.publishSnapshot([&](LoopSnapshot& s) {
    s.homed_generation = gen;
    s.session = 1;
  });
  r.sup->tick();
  r.server->serviceOnce(0);
  std::optional<std::string> reply;
  ASSERT_TRUE(test::waitUntil([&] { return (reply = r.client->receive(1)).has_value(); }, 2.0));
  EXPECT_TRUE(std::get<msg::Ack>(r.codec.decode(*reply)).ok);
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
}

}  // namespace
}  // namespace franka_control
