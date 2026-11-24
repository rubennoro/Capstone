// from rnoro/deployment:deployment/franka_control/include/franka_control/app.h @ 91b178b
// changes: App(CommandLine, AppOverrides) (tests only); SIGHUP; the keyboard thread with --keyboard; the hello
// template; "the launcher" -> "control_deploy run" (control_deploy/STRUCTURE.md 3.4, 4.2, 4.5).
// Start-up and shutdown: config, session folder, arm, gripper, the control / messaging / gripper / keyboard threads,
// the ready file, SIGINT/SIGTERM/SIGHUP.
#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include "franka_control/command_line.h"

namespace franka_control {

class FakeArm;

// At shutdown, if the control thread is still running after shutdown.timeout_s, the main thread calls
// arm.requestStop() and waits at most this long more. shutdown.timeout_s (3 s) + this stays below the 5 s that
// control_deploy run waits before SIGKILL.
inline constexpr double kShutdownStopWaitS = 1.5;

// A start-up or run failure: what() is "<what failed>: <what to check>", printed as the FATAL line.
struct FatalError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Tests only; never reachable from the command line.
struct AppOverrides {
  std::optional<std::string> endpoint_state;       // e.g. tcp://127.0.0.1:* (a free port)
  std::optional<std::string> endpoint_commands;
  std::optional<std::filesystem::path> link_file;  // instead of the baked contracts/link.yaml
  std::optional<std::filesystem::path> frames_file;
  std::optional<int> keyboard_fd;                  // keys from this fd (a pipe); implies --keyboard
};

class App {
 public:
  explicit App(CommandLine cl, AppOverrides overrides = {});
  ~App();
  // Runs until SIGINT/SIGTERM/SIGHUP (or requestStop()); 0 on a clean shutdown. Throws FatalError (exit code 1).
  int run();
  void requestStop();                              // the key q, and tests: the same as SIGTERM

  // Tests (valid once the ready file exists): the bound C2/C3 addresses and, with --mock, the FakeArm.
  std::string stateEndpoint() const;
  std::string commandsEndpoint() const;
  FakeArm* fakeArm() const;

 private:
  CommandLine cl_;
  AppOverrides overrides_;
  std::atomic<bool> stop_requested_{false};
  mutable std::mutex mutex_;
  std::string state_endpoint_, commands_endpoint_;
  FakeArm* fake_arm_ = nullptr;
};

}  // namespace franka_control
