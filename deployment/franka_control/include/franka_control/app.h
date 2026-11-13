// Start-up and shutdown (STRUCTURE.md 7.3-7.4): config, session folder, arm, gripper, the control / messaging /
// gripper threads, the ready file, SIGINT/SIGTERM.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "franka_control/command_line.h"

namespace franka_control {

class FakeArm;

// At shutdown, if the control thread is still running after shutdown.timeout_s, the main thread calls
// arm.requestStop() and waits at most this long more. shutdown.timeout_s (3 s) + this stays below the launcher's
// stop_timeout_s (5 s), after which it would SIGKILL (STRUCTURE.md 7.4).
inline constexpr double kShutdownStopWaitS = 1.5;

// A start-up or run failure: what() is "<what failed>: <what to check>", printed as the FATAL line.
struct FatalError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class App {
 public:
  explicit App(CommandLine cl);
  ~App();
  // Runs until SIGINT/SIGTERM (or requestStop()); 0 on a clean shutdown. Throws FatalError (exit code 1).
  int run();
  void requestStop();                              // tests: the same as SIGTERM

  // Tests (valid once the ready file exists): the bound C2/C3 addresses and, with --mock, the FakeArm.
  std::string stateEndpoint() const;
  std::string commandsEndpoint() const;
  FakeArm* fakeArm() const;

 private:
  CommandLine cl_;
  std::atomic<bool> stop_requested_{false};
  mutable std::mutex mutex_;
  std::string state_endpoint_, commands_endpoint_;
  FakeArm* fake_arm_ = nullptr;
};

}  // namespace franka_control
