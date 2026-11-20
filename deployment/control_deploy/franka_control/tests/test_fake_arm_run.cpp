// from rnoro/deployment:deployment/franka_control/tests/test_fake_arm_run.cpp @ 91b178b
// changes: AppOverrides (free ports, a key pipe) instead of station endpoints; the watchdog numbers from link.yaml
// (1.0 s); + hello, the keys h g s q through a pipe, stop during homing, the bind address; ShutdownStaysWithinThe-
// LaunchersStopBudget -> ...WithinTheWrappersStopBudget (same numbers) (control_deploy/STRUCTURE.md 9.1).
// + key p (the start pose) end to end; MockApp takes station overrides.
// Label `threads`: short real-time runs of the whole App with --mock (FakeArm + FakeGripper) on free local ports,
// driven by a test REQ client and a key pipe, watched through a C2 subscriber.
#include <gtest/gtest.h>

#include <unistd.h>

#include <cmath>

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

// The whole App with --mock on free local ports, a key pipe as its keyboard, a REQ client and a C2 watcher.
struct MockApp {
  explicit MockApp(AppOverrides extra = {}, bool with_keys = true, const std::string& station_fc = "")
      : dir(test::tmpDir("run")) {
    cl.station = test::writeStation(dir, station_fc);
    cl.session = dir / "session";
    cl.ready_file = dir / "ready";
    cl.mock = true;
    AppOverrides o = std::move(extra);
    if (!o.endpoint_state) o.endpoint_state = "tcp://127.0.0.1:*";
    if (!o.endpoint_commands) o.endpoint_commands = "tcp://127.0.0.1:*";
    if (with_keys) {
      EXPECT_EQ(pipe(key_fds), 0);
      o.keyboard_fd = key_fds[0];
    }
    app = std::make_unique<App>(cl, o);
    runner = std::thread([this] {
      try {
        code.store(app->run());
      } catch (const std::exception& e) {
        fatal = e.what();
        code.store(1);
      }
    });
  }
  ~MockApp() {
    req.reset();
    c2.reset();
    app->requestStop();
    if (runner.joinable()) runner.join();
    for (int fd : key_fds) {
      if (fd >= 0) close(fd);
    }
  }
  // Waits for the ready file, then connects the client and the watcher.
  bool ready() {
    test::waitUntil([&] { return std::filesystem::exists(*cl.ready_file) || code.load() >= 0; }, 30.0);
    if (!std::filesystem::exists(*cl.ready_file)) return false;
    req = std::make_unique<test::ReqClient>(ctx, app->commandsEndpoint());
    c2 = std::make_unique<Watcher>(ctx, app->stateEndpoint(), codec);
    return true;
  }
  msg::Any call(const msg::Any& m, int timeout_ms = 3000) {
    auto reply = req->request(codec.encode(m), timeout_ms);
    if (!reply) {
      ADD_FAILURE() << "no reply to " << msg::schemaOf(m);
      return msg::ErrorReply{};
    }
    return codec.decode(*reply);
  }
  std::string mode() { return std::get<msg::ArmState>(call(msg::GetState{})).mode; }
  void key(char c) { ASSERT_EQ(write(key_fds[1], &c, 1), 1); }
  // Waits for C2 to show this mode (skipping the queued backlog first); the message, or nullopt.
  std::optional<msg::ArmState> waitMode(const std::string& m, double timeout_s) {
    return c2->waitFor([&](const msg::ArmState& s) { return s.mode == m; }, timeout_s);
  }
  // Moves the TCP by d with waypoints over `seconds`, keeping C3 alive, and ends in holding.
  void moveBy(const Eigen::Vector3d& d, double seconds) {
    const auto s0 = std::get<msg::ArmState>(call(msg::GetState{}));
    const Pose start = poseFromRowMajor(s0.O_T_TCP_target);
    std::vector<TrajectoryPoint> pts;
    const int64_t t0 = monotonicNowNs();
    for (int k = 1; k <= 10; ++k) {
      pts.push_back(test::point(t0 + secondsToNs(seconds * k / 10.0), start.translation() + d * (k / 10.0),
                                Eigen::Quaterniond(start.linear()), s0.gripper_width_m));
    }
    EXPECT_EQ(std::get<msg::SetWaypointsReply>(call(test::makeSetWaypoints(1, pts))).accepted, 10u);
    const int64_t end = t0 + secondsToNs(seconds + 0.5);
    while (monotonicNowNs() < end) {
      call(msg::Heartbeat{});
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  std::filesystem::path dir;
  CommandLine cl;
  int key_fds[2] = {-1, -1};
  std::unique_ptr<App> app;
  std::atomic<int> code{-1};
  std::string fatal;
  std::thread runner;
  const Codec codec{ContractLimits::load(test::limitsFile())};
  zmq::context_t ctx{1};
  std::unique_ptr<test::ReqClient> req;
  std::unique_ptr<Watcher> c2;
};

Eigen::Vector3d tcpOf(const msg::ArmState& s) { return poseFromRowMajor(s.O_T_TCP).translation(); }

TEST(FakeArmRunThreads, AShortMockRunOfTheWholeApp) {
  MockApp m(AppOverrides{}, false);
  ASSERT_TRUE(m.ready()) << m.fatal;
  App& app = *m.app;
  const auto& dir = m.dir;
  EXPECT_TRUE(std::filesystem::exists(dir / "session" / "franka_control" / "config_effective.yaml"));
  const Codec& codec = m.codec;
  test::ReqClient& req = *m.req;
  Watcher& c2 = *m.c2;
  auto call = [&](const msg::Any& msg, int timeout_ms = 3000) { return m.call(msg, timeout_ms); };

  // hello: the reply carries what this process runs with, t_pc_ns on CLOCK_MONOTONIC between send and receive
  const int64_t before = monotonicNowNs();
  const auto hello = std::get<msg::HelloReply>(call(msg::Hello{}));
  const int64_t after = monotonicNowNs();
  EXPECT_TRUE(hello.mock);
  EXPECT_EQ(hello.schemas, Codec::knownSchemas());
  EXPECT_EQ(hello.schemas.size(), 12u);
  EXPECT_DOUBLE_EQ(hello.watchdog_limit_s, 1.0);
  EXPECT_DOUBLE_EQ(hello.max_lead_s, 1.5);
  EXPECT_DOUBLE_EQ(hello.homing_timeout_s, 30.0);
  EXPECT_EQ(hello.max_waypoints, 64u);
  EXPECT_GE(hello.t_pc_ns, before);
  EXPECT_LE(hello.t_pc_ns, after);

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

  // heartbeats stop -> stopped within link.yaml timing.watchdog_limit_s (1.0 s) + one 5 ms messaging iteration, seen on
  // C2 within one more 10 ms period (and 10 ms for scheduling under the sanitizers)
  EXPECT_TRUE(std::get<msg::Ack>(call(msg::Heartbeat{})).ok);
  const int64_t last_request = monotonicNowNs();
  c2.latest(0.05);   // skip the queued backlog
  const auto stopped = c2.waitFor([](const msg::ArmState& s) { return s.mode == "stopped"; }, 2.0);
  const double elapsed = nsToSeconds(monotonicNowNs() - last_request);
  ASSERT_TRUE(stopped);
  EXPECT_GE(elapsed, 1.0);
  EXPECT_LE(elapsed, 1.0 + 0.005 + 0.010 + 0.010);

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
  m.runner.join();
  EXPECT_EQ(m.code.load(), 0) << m.fatal;
}

// ---- the PC keyboard through a pipe (control_deploy/STRUCTURE.md 4.3, 4.4, 9.1) ----

TEST(FakeArmRunThreads, KeyHHomesTheArmAndKeySMidMoveStopsWhereItIs) {
  MockApp m;
  ASSERT_TRUE(m.ready()) << m.fatal;
  // h alone: homing, then idle
  m.key('h');
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  ASSERT_TRUE(m.waitMode("idle", 10.0));
  // away from home, then h, then s mid-move
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Start{})).ok);
  m.moveBy({0.0, 0.0, 0.03}, 0.5);
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Stop{})).ok);
  m.key('h');
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  std::this_thread::sleep_for(std::chrono::milliseconds(800));   // the 2 s move is running
  m.c2->latest(0.05);
  const int64_t t_key = monotonicNowNs();
  m.key('s');
  const auto stopped = m.waitMode("stopped", 2.0);
  const double reaction = nsToSeconds(monotonicNowNs() - t_key);
  ASSERT_TRUE(stopped);
  EXPECT_LT(reaction, 0.050) << "C2 stopped " << reaction << " s after the key";
  const Eigen::Vector3d where = tcpOf(*stopped);
  EXPECT_GT((where - Eigen::Vector3d(0.45, 0, 0.45)).norm(), 0.003);   // not at home: stopped on the way
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  const auto later = m.c2->latest(0.1);
  ASSERT_TRUE(later);
  EXPECT_EQ(later->mode, "stopped");
  EXPECT_LT((tcpOf(*later) - where).norm(), 0.001);   // the fake TCP holds where it stopped
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Start{})).ok);   // the laptop can start from there
  EXPECT_EQ(m.mode(), "holding");
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Stop{})).ok);
}

TEST(FakeArmRunThreads, AC3HomeStoppedByThePcKeyRepliesStopped) {
  MockApp m;
  ASSERT_TRUE(m.ready()) << m.fatal;
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Start{})).ok);
  m.moveBy({0.0, 0.0, 0.03}, 0.5);
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Stop{})).ok);
  m.req->send(m.codec.encode(msg::Home{}));
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  m.key('s');
  const auto reply = m.req->receive(5000);
  ASSERT_TRUE(reply);
  const auto ack = std::get<msg::Ack>(m.codec.decode(*reply));
  EXPECT_FALSE(ack.ok);
  EXPECT_EQ(ack.reason.rfind("stopped: ", 0), 0u) << ack.reason;
  EXPECT_TRUE(m.waitMode("stopped", 1.0));
  EXPECT_EQ(m.mode(), "stopped");
}

TEST(FakeArmRunThreads, AC3StopDuringAKeyHomingIsAcknowledgedAndStops) {
  MockApp m;
  ASSERT_TRUE(m.ready()) << m.fatal;
  m.key('h');
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Stop{})).ok);
  EXPECT_TRUE(m.waitMode("stopped", 1.0));
  EXPECT_EQ(m.mode(), "stopped");
  // the verified homing runs normally afterwards (the stop epoch: nothing sticky)
  m.key('h');
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  ASSERT_TRUE(m.waitMode("idle", 10.0));
}

TEST(FakeArmRunThreads, KeyGHomesTheHandAndTheWidthComesBack) {
  MockApp m;
  ASSERT_TRUE(m.ready()) << m.fatal;
  const double w0 = std::get<msg::ArmState>(m.call(msg::GetState{})).gripper_width_m;
  m.key('g');
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  EXPECT_TRUE(
      m.c2->waitFor([](const msg::ArmState& s) { return s.mode == "homing" && s.gripper_width_m < 0.06; }, 3.0));
  EXPECT_EQ(std::get<msg::Ack>(m.call(msg::Start{})).ok, false);   // refused while homing
  const auto idle = m.waitMode("idle", 10.0);
  ASSERT_TRUE(idle);
  EXPECT_TRUE(m.c2->waitFor([&](const msg::ArmState& s) { return std::abs(s.gripper_width_m - w0) < 0.002; }, 2.0));
}

TEST(FakeArmRunThreads, KeyPGoesFromHomeToTheStartPoseOpensTheHandAndC3StartNeedsIt) {
  // q_home_rad with joint 6 + 0.45 rad (a 2.8 s move)
  MockApp m(AppOverrides{}, true,
            "  homing:\n    q_start_rad: [0.0, -0.7853982, 0.0, -2.3561945, 0.0, 2.0207963, 0.7853982]\n");
  ASSERT_TRUE(m.ready()) << m.fatal;
  const auto refused = std::get<msg::Ack>(m.call(msg::Start{}));   // at home, not at the start pose
  EXPECT_FALSE(refused.ok);
  EXPECT_EQ(refused.reason, "refused: not at the start pose: press h, then p on the PC");
  m.key('p');
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  ASSERT_TRUE(m.waitMode("idle", 10.0));
  EXPECT_NEAR(std::get<msg::ArmState>(m.call(msg::GetState{})).q[5], 2.0207963, 0.01);
  // an episode closes the fingers; then h, and p opens them again
  ASSERT_TRUE(std::get<msg::Ack>(m.call(msg::Start{})).ok);
  const auto s0 = std::get<msg::ArmState>(m.call(msg::GetState{}));
  const Pose here = poseFromRowMajor(s0.O_T_TCP_target);
  const int64_t t0 = monotonicNowNs();
  std::vector<TrajectoryPoint> pts;
  for (int k = 1; k <= 5; ++k) {
    pts.push_back(test::point(t0 + secondsToNs(0.1 * k), here.translation(), Eigen::Quaterniond(here.linear()),
                              0.08 - 0.012 * k));
  }
  EXPECT_EQ(std::get<msg::SetWaypointsReply>(m.call(test::makeSetWaypoints(1, pts))).accepted, 5u);
  EXPECT_TRUE(m.c2->waitFor([](const msg::ArmState& s) { return s.gripper_width_m < 0.025; }, 3.0));
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Stop{})).ok);
  m.key('p');   // not at home
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(m.mode(), "stopped");
  m.key('h');
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  ASSERT_TRUE(m.waitMode("idle", 10.0));
  EXPECT_LT(std::get<msg::ArmState>(m.call(msg::GetState{})).gripper_width_m, 0.025);   // h leaves the fingers
  m.key('p');
  ASSERT_TRUE(m.waitMode("homing", 2.0));
  const auto idle = m.waitMode("idle", 10.0);
  ASSERT_TRUE(idle);
  EXPECT_TRUE(m.c2->waitFor([](const msg::ArmState& s) { return s.gripper_width_m > 0.075; }, 1.0));
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Start{})).ok);
  EXPECT_EQ(m.mode(), "holding");
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Stop{})).ok);
}

TEST(FakeArmRunThreads, KeyHIsRefusedWhileHoldingAndQQuits) {
  MockApp m;
  ASSERT_TRUE(m.ready()) << m.fatal;
  EXPECT_TRUE(std::get<msg::Ack>(m.call(msg::Start{})).ok);
  m.key('h');
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(m.mode(), "holding");   // refused, nothing changed
  EXPECT_FALSE(m.waitMode("homing", 0.2));
  const int64_t t0 = monotonicNowNs();
  m.key('q');
  ASSERT_TRUE(test::waitUntil([&] { return m.code.load() >= 0; }, 5.0));
  EXPECT_EQ(m.code.load(), 0) << m.fatal;
  EXPECT_LT(nsToSeconds(monotonicNowNs() - t0), 5.0);
}

// The binds: only the link.yaml address (never 0.0.0.0 or *); an address not on this machine is a FATAL.
TEST(FakeArmRunThreads, BindsOnlyTheLinkAddress) {
  const auto dir = test::tmpDir("bind");
  const int state = test::freePort();
  int commands = test::freePort();
  while (commands == state) commands = test::freePort();
  CommandLine cl;
  cl.station = test::writeStation(dir);
  cl.session = dir / "session";
  cl.ready_file = dir / "ready";
  cl.mock = true;
  AppOverrides o;
  o.link_file = test::writeLink(dir, "127.0.0.1", state, commands);
  App app(cl, o);
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
  ASSERT_TRUE(test::waitUntil([&] { return std::filesystem::exists(*cl.ready_file) || code.load() >= 0; }, 30.0));
  EXPECT_TRUE(std::filesystem::exists(*cl.ready_file)) << fatal;
  EXPECT_EQ(app.stateEndpoint(), "tcp://127.0.0.1:" + std::to_string(state));
  EXPECT_EQ(app.commandsEndpoint(), "tcp://127.0.0.1:" + std::to_string(commands));
  app.requestStop();
  runner.join();
  EXPECT_EQ(code.load(), 0) << fatal;

  // 192.0.2.1 (TEST-NET-1) is on no interface: FATAL naming link.yaml pc.address (main turns it into exit 1)
  const auto dir2 = test::tmpDir("bind_bad");
  CommandLine bad = cl;
  bad.station = test::writeStation(dir2);
  bad.session = dir2 / "session";
  bad.ready_file = dir2 / "ready";
  AppOverrides o2;
  o2.link_file = test::writeLink(dir2, "192.0.2.1", state, commands);
  App bad_app(bad, o2);
  try {
    bad_app.run();
    ADD_FAILURE() << "bound an address that is not on this machine";
  } catch (const FatalError& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("cannot bind the C2 endpoint tcp://192.0.2.1:"), std::string::npos) << what;
    EXPECT_NE(what.find("is link.yaml pc.address this PC's ASUS_5g address"), std::string::npos) << what;
  }
  EXPECT_FALSE(std::filesystem::exists(*bad.ready_file));
}

// 4.5: a torque session that never ends by itself (no settle, no settle timeout) is stopped with arm.requestStop()
// after shutdown.timeout_s, and the whole shutdown stays below the 5 s that control_deploy run waits before SIGKILL.
TEST(FakeArmRunThreads, ShutdownStaysWithinTheWrappersStopBudget) {
  const double timeout_s = test::testConfig().shutdown.timeout_s;
  MockApp m(AppOverrides{}, false);
  ASSERT_TRUE(m.ready()) << m.fatal;
  m.app->fakeArm()->ignoreFinish(true);
  const int64_t t0 = monotonicNowNs();
  m.app->requestStop();
  m.runner.join();
  const double elapsed = nsToSeconds(monotonicNowNs() - t0);
  EXPECT_EQ(m.code.load(), 0) << m.fatal;        // a stop requested at shutdown is not a fault
  EXPECT_GE(elapsed, timeout_s);                 // the session did not end by itself: the requestStop path ran
  EXPECT_LT(elapsed, timeout_s + kShutdownStopWaitS);
  EXPECT_LT(elapsed, 5.0);                       // control_deploy run's SIGKILL budget
}

}  // namespace
}  // namespace franka_control
