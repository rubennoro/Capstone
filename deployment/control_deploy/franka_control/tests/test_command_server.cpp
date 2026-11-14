// from rnoro/deployment:deployment/franka_control/tests/test_command_server.cpp @ 91b178b
// changes: the hello template; hello in EachRequestGetsItsReplyType and the watchdog test; +
// HelloReplyCarriesTheTemplateAndTheClock; a hello reply sent as a request is unexpected_message
// (control_deploy/STRUCTURE.md 9.1).
// CommandServer over tcp://127.0.0.1:* with a test REQ client; single thread: send, serviceOnce, receive.
#include <gtest/gtest.h>

#include "franka_control/command_server.h"
#include "franka_control/pose_math.h"
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

msg::HelloReply helloTemplate(const Config& cfg) {
  msg::HelloReply h;
  h.commit = "0123456789abcdef0123456789abcdef01234567";
  h.mock = true;
  h.schemas = Codec::knownSchemas();
  h.F_T_TCP = rowMajorFromPose(cfg.F_T_TCP);
  h.rotation_tol = cfg.limits.rotation_tol;
  h.gripper_width_max_m = cfg.limits.gripper_width_max_m;
  h.max_waypoints = cfg.limits.max_waypoints;
  h.watchdog_limit_s = cfg.watchdog.limit_s;
  h.homing_timeout_s = cfg.homing.timeout_s;
  h.max_lead_s = cfg.safety.max_lead_s;
  return h;
}

struct Rig {
  Rig() : cfg(test::testConfig()), codec(cfg.limits) {
    shared = std::make_unique<SharedState>();
    publishSnapshot([](LoopSnapshot&) {});
    sup = std::make_unique<Supervisor>(*shared, arm, cfg, cfg.limits, clock.clock());
    pub = std::make_unique<StatePublisher>(ctx, "tcp://127.0.0.1:*", 10, 100.0, *sup, codec, clock.clock());
    server = std::make_unique<CommandServer>(ctx, "tcp://127.0.0.1:*", *sup, *pub, codec, *shared, clock.clock(),
                                             helloTemplate(cfg));
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
  EXPECT_EQ(std::get<msg::HelloReply>(r.request(msg::Hello{})).schemas.size(), 12u);
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
  e = std::get<msg::ErrorReply>(r.request(example("c3/hello_reply.msgpack")));   // a hello reply sent as a request
  EXPECT_NE(e.error.find("unexpected message"), std::string::npos);
  EXPECT_EQ(e.request, "c3.hello.reply/v1");
  e = std::get<msg::ErrorReply>(r.request(example("c3/hello_unknown_field.msgpack")));
  EXPECT_EQ(e.field, "commit");
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
  r.clock.advance(1000);
  r.request(msg::Hello{});
  EXPECT_EQ(r.shared->lastAliveNs(), r.clock.now());
}

TEST(CommandServer, HelloReplyCarriesTheTemplateAndTheClock) {
  Rig r;
  r.clock.set(123456789012345);
  const msg::Any any = r.request(msg::Hello{});
  ASSERT_TRUE(std::holds_alternative<msg::HelloReply>(any));
  const auto& h = std::get<msg::HelloReply>(any);
  const msg::HelloReply want = helloTemplate(r.cfg);
  EXPECT_EQ(h.t_pc_ns, 123456789012345);   // the clock at handling
  EXPECT_EQ(h.commit, want.commit);
  EXPECT_EQ(h.mock, want.mock);
  EXPECT_EQ(h.schemas, want.schemas);
  EXPECT_EQ(h.F_T_TCP, want.F_T_TCP);
  EXPECT_EQ(h.rotation_tol, want.rotation_tol);
  EXPECT_EQ(h.gripper_width_max_m, want.gripper_width_max_m);
  EXPECT_EQ(h.max_waypoints, want.max_waypoints);
  EXPECT_DOUBLE_EQ(h.watchdog_limit_s, 1.0);
  EXPECT_DOUBLE_EQ(h.homing_timeout_s, 30.0);
  EXPECT_DOUBLE_EQ(h.max_lead_s, 1.5);
  // hello is read-only: no mode change, answered in every mode (homing included)
  EXPECT_EQ(r.sup->mode(), Mode::kIdle);
  r.sup->onHome(Supervisor::Source::kKeyboard);
  ASSERT_EQ(r.sup->mode(), Mode::kHoming);
  r.clock.advance(7);
  EXPECT_EQ(std::get<msg::HelloReply>(r.request(msg::Hello{})).t_pc_ns, 123456789012352);
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
