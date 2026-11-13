#include "franka_control/app.h"

#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <thread>

#include <zmq.hpp>

#include "franka_control/codec.h"
#include "franka_control/command_server.h"
#include "franka_control/config_loader.h"
#include "franka_control/control_loop.h"
#include "franka_control/fake_arm.h"
#include "franka_control/fake_gripper.h"
#include "franka_control/gripper_worker.h"
#include "franka_control/log.h"
#include "franka_control/shared_state.h"
#include "franka_control/state_publisher.h"
#include "franka_control/supervisor.h"
#ifdef FRANKA_CONTROL_HAS_LIBFRANKA
#include "franka_control/franka_hand_gripper.h"
#include "franka_control/libfranka_arm.h"
#endif

namespace franka_control {
namespace {

void sleepNs(int64_t ns) {
  if (ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
}

std::string timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm);
  return buf;
}

// Waits until done() or the budget passes, calling each() every poll; true if done.
bool waitFor(const std::function<bool()>& done, int64_t budget_ns, int64_t poll_ns,
             const std::function<void()>& each = {}) {
  const int64_t deadline = monotonicNowNs() + budget_ns;
  while (!done()) {
    if (each) each();
    if (monotonicNowNs() >= deadline) return done();
    sleepNs(poll_ns);
  }
  return true;
}

}  // namespace

App::App(CommandLine cl) : cl_(std::move(cl)) {}
App::~App() = default;

void App::requestStop() { stop_requested_.store(true); }

std::string App::stateEndpoint() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_endpoint_;
}

std::string App::commandsEndpoint() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return commands_endpoint_;
}

FakeArm* App::fakeArm() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return fake_arm_;
}

int App::run() {
  // SIGINT/SIGTERM are blocked before any thread exists (ZeroMQ's included), so only sigtimedwait below sees them.
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);

  // 1. config
  Config cfg;
  try {
    cfg = ConfigLoader::load(cl_.station, cl_.mock, FRANKA_CONTROL_CONFIG_FILE, CONTRACTS_LIMITS_FILE);
  } catch (const ConfigError& e) {
    for (const auto& p : e.problems) Log::error(p);
    throw FatalError(std::string(e.what()) + ": " + e.hint());
  }
  const int64_t poll_ns = secondsToNs(cfg.messaging.poll_max_s);

  // 2. session folder and config_effective.yaml (its only output; everything else goes to the log)
  namespace fs = std::filesystem;
  const fs::path dir = cl_.session ? *cl_.session / "franka_control"
                                   : cfg.session_root / (timestamp() + "_franka_control_alone") / "franka_control";
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) throw FatalError("cannot create the session folder " + dir.string() + " (" + ec.message() +
                           "): check session_root in the station file");
  {
    std::ofstream out(dir / "config_effective.yaml");
    out << ConfigLoader::toYaml(cfg);
    if (!out) throw FatalError("cannot write " + (dir / "config_effective.yaml").string() + ": check the disk");
  }
  Log::info("session folder " + dir.string());

  // 3. memory locked for the loop (real arm with real-time only)
  if (!cl_.mock && cfg.realtime == RealtimeMode::kEnforce && mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    Log::warning(std::string("mlockall failed (") + std::strerror(errno) + "): the loop's memory may be paged out");
  }

  auto shared = std::make_unique<SharedState>();
  const Codec codec(cfg.limits);

  // 4. arm
  std::unique_ptr<Arm> arm;
  if (cl_.mock) {
    auto fake = std::make_unique<FakeArm>(cfg.fake_arm, cfg.F_T_TCP, cfg.homing.q_home_rad, monotonicNowNs);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fake_arm_ = fake.get();
    }
    arm = std::move(fake);
  } else {
#ifdef FRANKA_CONTROL_HAS_LIBFRANKA
    try {
      arm = std::make_unique<LibfrankaArm>(cfg.robot_ip, cfg.realtime, cfg.control, cfg.load);
    } catch (const ArmError& e) {
      throw FatalError(e.what());
    }
#else
    throw FatalError("built without libfranka: run franka_control/build.sh (or use --mock)");
#endif
  }
  try {
    const std::string problem = describeArmProblem(arm->status());
    if (!problem.empty()) {
      throw FatalError("arm reports " + problem +
                       ": press and release the user stop, check Desk, then rerun deploy run");
    }
  } catch (const ArmError& e) {
    throw FatalError(std::string(e.what()) + ": check the arm and Desk");
  }
  Log::info("arm: " + arm->describe() + (cl_.mock ? " (--mock)" : ""));

  // 5. gripper: read the width and make it the target, so nothing moves at start
  std::unique_ptr<Gripper> gripper;
  if (cl_.mock) {
    gripper = std::make_unique<FakeGripper>(cfg.fake_gripper, sleepNs);
  } else {
#ifdef FRANKA_CONTROL_HAS_LIBFRANKA
    gripper = std::make_unique<FrankaHandGripper>(cfg.robot_ip, cfg.gripper, cfg.limits.gripper_width_max_m);
#endif
  }
  try {
    gripper->prepare();
    const double w = gripper->widthM();
    shared->setGripperWidth(w);
    shared->setGripperTarget(w);
  } catch (const GripperError& e) {
    throw FatalError(std::string("gripper: ") + e.what() + ": check the Franka Hand in Desk and its cable");
  }

  // 6. control thread; wait until the first torque session has held for hold_before_ready_s
  ControlLoop loop(*arm, *shared, cfg, monotonicNowNs);
  std::thread control([&loop] { loop.run(); });
  if (cfg.control_cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(*cfg.control_cpu, &set);
    if (pthread_setaffinity_np(control.native_handle(), sizeof(set), &set) != 0) {
      Log::warning("could not pin the control thread to CPU " + std::to_string(*cfg.control_cpu));
    }
  }
  bool arm_stop_sent = false;
  auto stopControl = [&] {
    shared->requestShutdown();
    gripper->stop();
    const int64_t budget = secondsToNs(cfg.shutdown.timeout_s);
    const bool ended = waitFor([&] { return loop.done(); }, budget, poll_ns, [&] {
      if (loop.inHomingMove() && !arm_stop_sent) {   // a homing move doesn't settle: stop it
        arm_stop_sent = true;
        arm->requestStop();
      }
    });
    if (!ended) {
      Log::warning("control thread still running after shutdown.timeout_s: stopping the arm");
      arm->requestStop();
      if (!waitFor([&] { return loop.done(); }, secondsToNs(kShutdownStopWaitS), poll_ns)) {
        char text[192];
        std::snprintf(text, sizeof(text),
                      "control thread still running %.1f s after arm.requestStop(): waiting for it (the launcher "
                      "kills the process after its stop_timeout_s)",
                      kShutdownStopWaitS);
        Log::error(text);
      }
    }
    control.join();
  };

  try {
    const int64_t deadline = monotonicNowNs() + secondsToNs(cfg.startup.loop_start_timeout_s);
    const int64_t hold_ns = secondsToNs(cfg.startup.hold_before_ready_s);
    int64_t first_torque_ns = -1;
    for (;;) {
      const LoopSnapshot& s = shared->latestSnapshot();
      if (s.phase == LoopSnapshot::Phase::kEnded || loop.done()) {
        const std::string err(s.error.data());
        if (loop.failureKind() == ArmError::Kind::kRealtime) {
          throw FatalError("real-time scheduling unavailable (" + err +
                           "): use a PREEMPT_RT kernel with rtprio 99, or set franka_control.realtime: ignore in "
                           "station.yaml");
        }
        throw FatalError("the 1 kHz loop ended at start-up (" + err + "): check the arm, Desk and the log");
      }
      if (s.phase == LoopSnapshot::Phase::kTorque) {
        if (first_torque_ns < 0) first_torque_ns = s.t_ns;
        if (s.t_ns - first_torque_ns >= hold_ns) break;
      }
      if (monotonicNowNs() > deadline) {
        throw FatalError("the 1 kHz loop did not start within startup.loop_start_timeout_s: check the arm, FCI in "
                         "Desk and the log");
      }
      sleepNs(poll_ns);
    }
  } catch (...) {
    stopControl();
    throw;
  }

  // 7. sockets, the messaging and gripper threads
  std::atomic<bool> messaging_stop{false};
  std::atomic<bool> thread_failed{false};
  std::mutex failure_mutex;
  std::string failure;
  auto recordFailure = [&](const std::string& text) {
    {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (failure.empty()) failure = text;
    }
    Log::error(text);
    thread_failed.store(true);
  };
  {
    zmq::context_t ctx(1);
    Supervisor supervisor(*shared, *arm, cfg, cfg.limits, monotonicNowNs);
    std::unique_ptr<StatePublisher> publisher;
    std::unique_ptr<CommandServer> server;
    try {
      publisher = std::make_unique<StatePublisher>(ctx, cfg.endpoint_state, cfg.messaging.state_send_hwm,
                                                   cfg.messaging.publish_rate_hz, supervisor, codec, monotonicNowNs);
    } catch (const zmq::error_t& e) {
      stopControl();
      throw FatalError("cannot bind the C2 endpoint " + cfg.endpoint_state + " (" + e.what() +
                       "): is another franka_control running? check endpoints.state in the station file");
    }
    try {
      server = std::make_unique<CommandServer>(ctx, cfg.endpoint_commands, supervisor, *publisher, codec, *shared,
                                               monotonicNowNs);
    } catch (const zmq::error_t& e) {
      publisher.reset();
      stopControl();
      throw FatalError("cannot bind the C3 endpoint " + cfg.endpoint_commands + " (" + e.what() +
                       "): is another franka_control running? check endpoints.commands in the station file");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      state_endpoint_ = publisher->endpoint();
      commands_endpoint_ = server->endpoint();
    }

    std::thread messaging([&] {
      try {
        // 8. the first C2 message, then the ready file; from here on C3 is answered
        publisher->maybePublish();
        if (cl_.ready_file) {
          std::ofstream ready(*cl_.ready_file);
          if (!ready) throw std::runtime_error("cannot create the ready file " + cl_.ready_file->string());
        }
        Log::info("ready: C2 on " + publisher->endpoint() + ", C3 on " + server->endpoint());
        while (!messaging_stop.load()) {
          supervisor.tick();
          server->serviceOnce(std::min(publisher->nsUntilNext(), poll_ns));
          publisher->maybePublish();
        }
      } catch (const std::exception& e) {
        recordFailure(std::string("messaging thread: ") + e.what());
      }
    });
    GripperWorker worker(*gripper, *shared, cfg.gripper, monotonicNowNs);
    std::thread gripper_thread([&] {
      try {
        worker.run();
      } catch (const std::exception& e) {
        recordFailure(std::string("gripper thread: ") + e.what());
      }
    });
    std::thread gripper_reader([&] {
      try {
        worker.runReader();
      } catch (const std::exception& e) {
        recordFailure(std::string("gripper-reader thread: ") + e.what());
      }
    });

    // wait for SIGINT/SIGTERM (or a thread failure, or requestStop() in tests)
    timespec wait{};
    wait.tv_sec = static_cast<time_t>(poll_ns / kNsPerS);
    wait.tv_nsec = static_cast<long>(poll_ns % kNsPerS);
    while (!stop_requested_.load() && !thread_failed.load()) {
      const int sig = sigtimedwait(&signals, nullptr, &wait);
      if (sig == SIGINT || sig == SIGTERM) {
        Log::info(std::string("received ") + (sig == SIGINT ? "SIGINT" : "SIGTERM") + ": shutting down");
        break;
      }
    }

    // 7.4 shutdown: the loop settles and ends while C2 keeps reporting; then messaging, then the gripper threads
    stopControl();
    messaging_stop.store(true);
    messaging.join();
    gripper_thread.join();
    gripper_reader.join();
    server.reset();
    publisher.reset();
  }
  if (thread_failed.load()) {
    std::lock_guard<std::mutex> lock(failure_mutex);
    throw FatalError(failure + ": a franka_control bug or a ZeroMQ error; see the log");
  }
  Log::info("shut down cleanly");
  return 0;
}

}  // namespace franka_control
