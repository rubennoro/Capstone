// Label `threads`: a short real-time run of the whole App with --mock (FakeArm + FakeGripper) on free local ports,
// driven by a test REQ client and watched through a C2 subscriber.
#include <gtest/gtest.h>

#include "franka_control/app.h"
#include "franka_control/codec.h"
#include "franka_control/fake_arm.h"
#include "franka_control/pose_math.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

class Watcher {   // the newest C2 message, read from a SUB socket
 public:
  Watcher(zmq::context_t& ctx, const std::string& endpoint, const Codec& codec)
      : sub_(ctx, zmq::socket_type::sub), codec_(codec) {
    sub_.set(zmq::sockopt::subscribe, "");
    sub_.connect(endpoint);
  }
  // The newest message (drains the queue); waits up to the timeout for one if none is queued.
  std::optional<msg::ArmState> latest(double timeout_s) {
    std::optional<msg::ArmState> out;
    zmq::message_t m;
    while (sub_.recv(m, zmq::recv_flags::dontwait)) out = decode(m);
    if (!out) out = waitFor([](const msg::ArmState&) { return true; }, timeout_s);
    return out;
  }
  // Waits for a message matching pred; returns it (or nullopt after the timeout).
  std::optional<msg::ArmState> waitFor(const std::function<bool(const msg::ArmState&)>& pred, double timeout_s) {
    const int64_t deadline = monotonicNowNs() + secondsToNs(timeout_s);
    while (monotonicNowNs() < deadline) {
      zmq::pollitem_t items[] = {{static_cast<void*>(sub_), 0, ZMQ_POLLIN, 0}};
      zmq::poll(items, 1, std::chrono::milliseconds(5));
      if (!(items[0].revents & ZMQ_POLLIN)) continue;
      zmq::message_t m;
      (void)sub_.recv(m);
      const auto s = decode(m);
      if (pred(s)) return s;
    }
    return std::nullopt;
  }

 private:
  msg::ArmState decode(const zmq::message_t& m) const {
    return std::get<msg::ArmState>(codec_.decode(std::string(static_cast<const char*>(m.data()), m.size())));
  }
  zmq::socket_t sub_;
  const Codec& codec_;
};

TEST(FakeArmRunThreads, AShortMockRunOfTheWholeApp) {
  const auto dir = test::tmpDir("run");
  std::filesystem::create_directories(dir / "ready");
  CommandLine cl;
  cl.station = test::writeStation(dir);
  cl.session = dir / "session";
  cl.ready_file = dir / "ready" / "franka_control.ready";
  cl.mock = true;
  App app(cl);
  std::atomic<int> code{-1};
  std::string fatal;
  std::thread runner([&] {
    try {
      code.store(app.run());
    } catch (const std::exception& e) {
      fatal = e.what();
      code.store(1);
    }
  });
  struct Joiner {
    App& app;
    std::thread& t;
    ~Joiner() {
      app.requestStop();
      if (t.joinable()) t.join();
    }
  } joiner{app, runner};

  ASSERT_TRUE(test::waitUntil([&] { return std::filesystem::exists(*cl.ready_file) || code.load() >= 0; }, 30.0));
  ASSERT_TRUE(std::filesystem::exists(*cl.ready_file)) << fatal;
  EXPECT_TRUE(std::filesystem::exists(dir / "session" / "franka_control" / "config_effective.yaml"));

  const Codec codec(ContractLimits::load(test::limitsFile()));
  zmq::context_t ctx(1);
  test::ReqClient req(ctx, app.commandsEndpoint());
  Watcher c2(ctx, app.stateEndpoint(), codec);
  auto call = [&](const msg::Any& m, int timeout_ms = 3000) -> msg::Any {
    auto reply = req.request(codec.encode(m), timeout_ms);
    if (!reply) {
      ADD_FAILURE() << "no reply to " << msg::schemaOf(m);
      return msg::ErrorReply{};
    }
    return codec.decode(*reply);
  };

  // idle, holding the start pose
  const auto s0 = std::get<msg::ArmState>(call(msg::GetState{}));
  EXPECT_EQ(s0.mode, "idle");
  const Pose start = poseFromRowMajor(s0.O_T_TCP);
  EXPECT_TRUE(std::get<msg::Ack>(call(msg::Start{})).ok);

  // 5 cm along x over 1 s, with heartbeats
  std::vector<TrajectoryPoint> pts;
  const int64_t t0 = monotonicNowNs();
  const Eigen::Quaterniond q(start.linear());
  for (int k = 1; k <= 10; ++k) {
    pts.push_back(test::point(t0 + k * 100000000LL, start.translation() + Eigen::Vector3d(0.005 * k, 0, 0), q,
                              s0.gripper_width_m));
  }
  const auto rep = std::get<msg::SetWaypointsReply>(call(test::makeSetWaypoints(1, pts)));
  EXPECT_EQ(rep.accepted, 10u);
  EXPECT_EQ(rep.shortened, 0u);
  EXPECT_TRUE(c2.waitFor([](const msg::ArmState& s) { return s.mode == "running"; }, 1.0));
  const Eigen::Vector3d last = pts.back().p;
  bool close = false;
  double dist = -1.0;
  std::string mode;
  for (int i = 0; i < 60 && !close; ++i) {   // up to 3 s of heartbeats every 50 ms
    EXPECT_TRUE(std::get<msg::Ack>(call(msg::Heartbeat{})).ok);
    const auto s = c2.latest(0.05);
    if (s) {
      dist = (poseFromRowMajor(s->O_T_TCP).translation() - last).norm();
      mode = s->mode;
      close = s->mode == "holding" && dist < 0.002;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  EXPECT_TRUE(close) << "the fake TCP did not settle within 2 mm of the last waypoint while holding: " << dist
                     << " m, mode " << mode;

  // heartbeats stop -> stopped within watchdog.limit_s (0.5 s) + one 5 ms messaging iteration, seen on C2 within one
  // more 10 ms period (and 10 ms for scheduling under the sanitizers)
  EXPECT_TRUE(std::get<msg::Ack>(call(msg::Heartbeat{})).ok);
  const int64_t last_request = monotonicNowNs();
  c2.latest(0.05);   // skip the queued backlog
  const auto stopped = c2.waitFor([](const msg::ArmState& s) { return s.mode == "stopped"; }, 2.0);
  const double elapsed = nsToSeconds(monotonicNowNs() - last_request);
  ASSERT_TRUE(stopped);
  EXPECT_GE(elapsed, 0.5);
  EXPECT_LE(elapsed, 0.5 + 0.005 + 0.010 + 0.010);

  // a new episode: start, then stop
  EXPECT_TRUE(std::get<msg::Ack>(call(msg::Start{})).ok);
  EXPECT_TRUE(std::get<msg::Ack>(call(msg::Stop{})).ok);
  EXPECT_EQ(std::get<msg::ArmState>(call(msg::GetState{})).mode, "stopped");

  // home: C2 shows homing, then the Ack arrives and the mode is idle
  req.send(codec.encode(msg::Home{}));
  EXPECT_TRUE(c2.waitFor([](const msg::ArmState& s) { return s.mode == "homing"; }, 2.0));
  const auto home_reply = req.receive(10000);
  ASSERT_TRUE(home_reply);
  const auto ack = std::get<msg::Ack>(codec.decode(*home_reply));
  EXPECT_TRUE(ack.ok) << ack.reason;
  const auto idle = std::get<msg::ArmState>(call(msg::GetState{}));
  EXPECT_EQ(idle.mode, "idle");
  EXPECT_LT((poseFromRowMajor(idle.O_T_TCP).translation() - Eigen::Vector3d(0.45, 0, 0.45)).norm(), 0.002);

  // a reflex in the fake arm -> fault, with its text in C2 (and C2 keeps coming)
  app.fakeArm()->injectFault("libfranka: motion aborted by reflex! [cartesian_reflex] (injected)");
  const auto fault = c2.waitFor([](const msg::ArmState& s) { return s.mode == "fault"; }, 2.0);
  ASSERT_TRUE(fault);
  EXPECT_EQ(fault->error, "libfranka: motion aborted by reflex! [cartesian_reflex] (injected)");
  const auto later = c2.waitFor([&](const msg::ArmState& s) { return s.seq > fault->seq + 2; }, 1.0);
  ASSERT_TRUE(later);
  EXPECT_EQ(later->t_ns, fault->t_ns);   // the loop has ended: the last reading is repeated
  EXPECT_FALSE(std::get<msg::Ack>(call(msg::Start{})).ok);

  // shutdown: every thread ends, exit code 0
  app.requestStop();
  runner.join();
  EXPECT_EQ(code.load(), 0) << fatal;
}

// 7.4: a torque session that never ends by itself (no settle, no settle timeout) is stopped with arm.requestStop()
// after shutdown.timeout_s, and the whole shutdown stays below the launcher's stop_timeout_s (5 s).
TEST(FakeArmRunThreads, ShutdownStaysWithinTheLaunchersStopBudget) {
  const auto dir = test::tmpDir("run_shutdown");
  std::filesystem::create_directories(dir / "ready");
  CommandLine cl;
  cl.station = test::writeStation(dir);
  cl.session = dir / "session";
  cl.ready_file = dir / "ready" / "franka_control.ready";
  cl.mock = true;
  const double timeout_s = test::testConfig().shutdown.timeout_s;
  App app(cl);
  std::atomic<int> code{-1};
  std::string fatal;
  std::thread runner([&] {
    try {
      code.store(app.run());
    } catch (const std::exception& e) {
      fatal = e.what();
      code.store(1);
    }
  });
  struct Joiner {
    App& app;
    std::thread& t;
    ~Joiner() {
      app.requestStop();
      if (t.joinable()) t.join();
    }
  } joiner{app, runner};

  ASSERT_TRUE(test::waitUntil([&] { return std::filesystem::exists(*cl.ready_file) || code.load() >= 0; }, 30.0));
  ASSERT_TRUE(std::filesystem::exists(*cl.ready_file)) << fatal;
  app.fakeArm()->ignoreFinish(true);
  const int64_t t0 = monotonicNowNs();
  app.requestStop();
  runner.join();
  const double elapsed = nsToSeconds(monotonicNowNs() - t0);
  EXPECT_EQ(code.load(), 0) << fatal;            // a stop requested at shutdown is not a fault
  EXPECT_GE(elapsed, timeout_s);                 // the session did not end by itself: the requestStop path ran
  EXPECT_LT(elapsed, timeout_s + kShutdownStopWaitS);
  EXPECT_LT(elapsed, 5.0);                       // deploy's launcher.stop_timeout_s
}

}  // namespace
}  // namespace franka_control
