// from rnoro/deployment:deployment/franka_control/src/app.cpp @ 91b178b
// changes: endpoints and the watchdog limit from contracts/link.yaml, F_T_TCP from frames.yaml (via the loader); bind
// failure and hint texts name link.yaml and control_deploy; SIGHUP joins SIGINT/SIGTERM; builds the hello template
// (encoded once as a self-check) for CommandServer and passes the gripper to Supervisor; with --keyboard the keyboard
// thread (q = requestStop()) and the messaging loop drains KeyQueue; AppOverrides for tests
// (control_deploy/STRUCTURE.md 3.4, 4.1, 4.2, 5.5). Fix wave: the shutdown stop of a homing move is sent under
// SharedState::stopHomingMove and re-sent every 100 ms while the move runs (4.4).
// copied from rnoro/final-deployment:deployment/control_deploy/franka_control/src/app.cpp @ f632576
// changes: --sim: the arm and gripper are sim_pc's SimArm/SimGripper (MuJoCo, built with FRANKA_CONTROL_SIM) and
// hello.mock is true (the laptop's --mock expects it); everything else (gate, supervisor, loop, codec) unchanged.
// changes: + the sim scene's {"cmd": "key"} is attached to the messaging thread's KeyQueue (sim build only, also
// without --keyboard: the queue is drained either way) and detached before it goes away (sim_deploy P1).
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

#include <unistd.h>

#include <zmq.hpp>

#include "franka_control/build_info.h"

#include "franka_control/codec.h"
#include "franka_control/command_server.h"
#include "franka_control/config_loader.h"
#include "franka_control/control_loop.h"
#include "franka_control/fake_arm.h"
#include "franka_control/fake_gripper.h"
#include "franka_control/gripper_worker.h"
#include "franka_control/key_queue.h"
#include "franka_control/keyboard_reader.h"
#include "franka_control/log.h"
#include "franka_control/pose_math.h"
#include "franka_control/shared_state.h"
#include "franka_control/state_publisher.h"
#include "franka_control/supervisor.h"
#ifdef FRANKA_CONTROL_HAS_LIBFRANKA
#include "franka_control/franka_hand_gripper.h"
#include "franka_control/libfranka_arm.h"
#endif
#ifdef FRANKA_CONTROL_HAS_SIM
#include "sim_pc/sim_parts.h"
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

App::App(CommandLine cl, AppOverrides overrides) : cl_(std::move(cl)), overrides_(std::move(overrides)) {}
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
  // SIGINT/SIGTERM/SIGHUP are blocked before any thread exists (ZeroMQ's included), so only sigtimedwait below sees
  // them. SIGHUP: closing the terminal of control_deploy run shuts down cleanly instead of killing the process.
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  sigaddset(&signals, SIGHUP);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);

  // 1. config (config.yaml + the station's franka_control section + the three contract files)
  Config cfg;
  try {
    ContractFiles files = bakedContractFiles();
    if (overrides_.link_file) files.link = *overrides_.link_file;
    if (overrides_.frames_file) files.frames = *overrides_.frames_file;
    cfg = ConfigLoader::load(cl_.station, cl_.mock, cl_.local, FRANKA_CONTROL_CONFIG_FILE, files);
  } catch (const ConfigError& e) {
    for (const auto& p : e.problems) Log::error(p);
    throw FatalError(std::string(e.what()) + ": " + e.hint());
  }
  if (overrides_.endpoint_state) cfg.endpoint_state = *overrides_.endpoint_state;
  if (overrides_.endpoint_commands) cfg.endpoint_commands = *overrides_.endpoint_commands;
  const int64_t poll_ns = secondsToNs(cfg.messaging.poll_max_s);
  const bool keyboard = cl_.keyboard || overrides_.keyboard_fd.has_value();
  const int keyboard_fd = overrides_.keyboard_fd.value_or(STDIN_FILENO);

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

  // the hello reply (deployment/STRUCTURE.md 3.8): built once; only t_pc_ns is set per request. Encoded here once, so
  // a value that would break the contract is a FATAL now, never a broken reply later.
  msg::HelloReply hello;
  hello.commit = kGitCommit;
  hello.mock = cl_.mock || cl_.sim;   // the sim is a mock PC to the laptop (no real arm)
  hello.schemas = Codec::knownSchemas();
  hello.F_T_TCP = rowMajorFromPose(cfg.F_T_TCP);
  hello.rotation_tol = cfg.limits.rotation_tol;
  hello.gripper_width_max_m = cfg.limits.gripper_width_max_m;
  hello.max_waypoints = cfg.limits.max_waypoints;
  hello.watchdog_limit_s = cfg.watchdog.limit_s;
  hello.homing_timeout_s = cfg.homing.timeout_s;
  hello.max_lead_s = cfg.safety.max_lead_s;
  try {
    msg::HelloReply probe = hello;
    probe.t_pc_ns = monotonicNowNs();
    (void)codec.encode(probe);
  } catch (const std::exception& e) {
    throw FatalError(std::string("the hello reply would break its contract (") + e.what() +
                     "): check config.yaml and the contract files");
  }

  // 4. arm
  std::unique_ptr<Arm> arm;
  std::unique_ptr<Gripper> sim_gripper;
#ifdef FRANKA_CONTROL_HAS_SIM
  std::function<void(KeyQueue*)> sim_attach_keys;   // the sim scene's {"cmd": "key"} -> the KeyQueue below
#endif
  if (cl_.sim) {
#ifdef FRANKA_CONTROL_HAS_SIM
    try {
      sim_pc::SimParts parts = sim_pc::makeSim(cl_.station, cfg);
      arm = std::move(parts.arm);
      sim_gripper = std::move(parts.gripper);
      sim_attach_keys = std::move(parts.attach_keys);
    } catch (const std::exception& e) {
      throw FatalError(std::string("sim: ") + e.what() + ": check the station's sim: block and the model");
    }
#else
    throw FatalError("built without the sim: build build-sim/ with -DFRANKA_CONTROL_SIM=ON (sim_deploy/sim_pc/README.md)");
#endif
  } else if (cl_.mock) {
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
    throw FatalError("built without libfranka: run control_deploy/franka_control/build.sh (or use --mock)");
#endif
  }
  try {
    const std::string problem = describeArmProblem(arm->status());
    if (!problem.empty()) {
      throw FatalError("arm reports " + problem +
                       ": press and release the user stop, check Desk, then restart control_deploy run");
    }
  } catch (const ArmError& e) {
    throw FatalError(std::string(e.what()) + ": check the arm and Desk");
  }
  Log::info("arm: " + arm->describe() + (cl_.mock ? " (--mock)" : cl_.sim ? " (--sim)" : ""));

  // 5. gripper: read the width and make it the target, so nothing moves at start
  std::unique_ptr<Gripper> gripper;
  if (cl_.sim) {
    gripper = std::move(sim_gripper);
  } else if (cl_.mock) {
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
  auto stopControl = [&] {
    shared->requestShutdown();
    gripper->stop();
    const int64_t budget = secondsToNs(cfg.shutdown.timeout_s);
    int64_t arm_stop_sent_ns = -1;
    const bool ended = waitFor([&] { return loop.done(); }, budget, poll_ns, [&] {
      // a homing move doesn't settle: stop it, again every 100 ms while it runs (a stop sent just before the move's
      // libfranka call records its epoch is lost)
      const int64_t now = monotonicNowNs();
      if (arm_stop_sent_ns >= 0 && now - arm_stop_sent_ns < kHomingStopResendNs) return;
      if (shared->stopHomingMove(SharedState::kAnySession, [&] { arm->requestStop(); })) arm_stop_sent_ns = now;
    });
    if (!ended) {
      Log::warning("control thread still running after shutdown.timeout_s: stopping the arm");
      arm->requestStop();
      if (!waitFor([&] { return loop.done(); }, secondsToNs(kShutdownStopWaitS), poll_ns)) {
        char text[192];
        std::snprintf(text, sizeof(text),
                      "control thread still running %.1f s after arm.requestStop(): waiting for it "
                      "(control_deploy run kills the process 5 s after its SIGTERM)",
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
                           "control_deploy/station.yaml");
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
    Supervisor supervisor(*shared, *arm, cfg, cfg.limits, monotonicNowNs, gripper.get());
    KeyQueue keys;
#ifdef FRANKA_CONTROL_HAS_SIM
    // the sim scene pushes h, p, g, s into `keys` exactly as the keyboard thread does; detached before `keys` ends
    struct SimKeysAttachment {
      std::function<void(KeyQueue*)>& attach;
      ~SimKeysAttachment() {
        if (attach) attach(nullptr);
      }
    } sim_keys_attachment{sim_attach_keys};
    if (sim_attach_keys) sim_attach_keys(&keys);
#endif
    std::unique_ptr<StatePublisher> publisher;
    std::unique_ptr<CommandServer> server;
    try {
      publisher = std::make_unique<StatePublisher>(ctx, cfg.endpoint_state, cfg.messaging.state_send_hwm,
                                                   cfg.messaging.publish_rate_hz, supervisor, codec, monotonicNowNs);
    } catch (const zmq::error_t& e) {
      stopControl();
      throw FatalError("cannot bind the C2 endpoint " + cfg.endpoint_state + " (" + e.what() +
                       "): is link.yaml pc.address this PC's ASUS_5g address (ip -4 addr)? is another "
                       "franka_control running?");
    }
    try {
      server = std::make_unique<CommandServer>(ctx, cfg.endpoint_commands, supervisor, *publisher, codec, *shared,
                                               monotonicNowNs, hello);
    } catch (const zmq::error_t& e) {
      publisher.reset();
      stopControl();
      throw FatalError("cannot bind the C3 endpoint " + cfg.endpoint_commands + " (" + e.what() +
                       "): is link.yaml pc.address this PC's ASUS_5g address (ip -4 addr)? is another "
                       "franka_control running?");
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
        Log::info("ready: C2 on " + publisher->endpoint() + ", C3 on " + server->endpoint() +
                  (keyboard ? std::string("; ") + KeyboardReader::help() : std::string()));
        while (!messaging_stop.load()) {
          supervisor.tick();
          while (auto key = keys.pop()) supervisor.onKey(*key);   // h, g, s (q never reaches it)
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
    // the keyboard (--keyboard): after the sockets are bound; it discards keys typed before this point
    std::thread keyboard_thread;
    if (keyboard) {
      keyboard_thread = std::thread([&] {
        try {
          KeyboardReader reader(keyboard_fd, keys, [this] { requestStop(); }, *shared);
          reader.run();
        } catch (const std::exception& e) {
          recordFailure(std::string("keyboard thread: ") + e.what());
        }
      });
    }

    // wait for SIGINT/SIGTERM/SIGHUP (or a thread failure, or requestStop(): the key q, tests)
    timespec wait{};
    wait.tv_sec = static_cast<time_t>(poll_ns / kNsPerS);
    wait.tv_nsec = static_cast<long>(poll_ns % kNsPerS);
    while (!stop_requested_.load() && !thread_failed.load()) {
      const int sig = sigtimedwait(&signals, nullptr, &wait);
      if (sig == SIGINT || sig == SIGTERM || sig == SIGHUP) {
        const char* name = sig == SIGINT ? "SIGINT" : sig == SIGTERM ? "SIGTERM" : "SIGHUP";
        Log::info(std::string("received ") + name + ": shutting down");
        break;
      }
    }

    // 7.4 shutdown: the loop settles and ends while C2 keeps reporting; then messaging, then the gripper threads
    stopControl();
    messaging_stop.store(true);
    messaging.join();
    gripper_thread.join();
    gripper_reader.join();
    if (keyboard_thread.joinable()) keyboard_thread.join();   // sees shutdown within 50 ms, restores the terminal
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
