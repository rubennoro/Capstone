// New (control_deploy/STRUCTURE.md 3.3, 4.3): the keyboard thread (only with --keyboard) and the terminal's cbreak
// mode. + p, the start pose.
// Keys: h home the arm, p the start pose (from home only; opens the hand), g home the hand, s stop and hold, q quit;
// upper case acts like lower case (caps lock). Keys never feed the watchdog. Ctrl-C stays SIGINT (handled by the main
// thread); Ctrl-Z and Ctrl-\ are disabled (a stopped process would miss FCI packets and trip a reflex); Ctrl-S/Ctrl-Q
// flow control is off (a paused terminal would block every log line). EOF or a read error ends the thread ("keys
// disabled"); it is never `q`.
#pragma once

#include <termios.h>

#include <cstdint>
#include <functional>
#include <optional>

#include "franka_control/key_queue.h"
#include "franka_control/shared_state.h"

namespace franka_control {

// RAII; does nothing when fd is not a TTY. cbreak: ICANON and ECHO off, ISIG kept; VSUSP and VQUIT disabled; IXON off.
class TerminalMode {
 public:
  explicit TerminalMode(int fd);
  ~TerminalMode();                                  // restores the saved termios
  TerminalMode(const TerminalMode&) = delete;
  TerminalMode& operator=(const TerminalMode&) = delete;
  bool active() const { return active_; }

 private:
  int fd_;
  bool active_ = false;
  termios saved_{};
};

class KeyboardReader {                              // keyboard thread
 public:
  static constexpr int kPollMs = 50;                // sees shutdown within this
  KeyboardReader(int fd, KeyQueue& keys, std::function<void()> on_quit, const SharedState& shared);
  void run();                                       // until shutdown, EOF or a read error
  static std::optional<Key> map(char c);            // the one key table: 'h', 'p', 'g', 's', 'q' and upper case
  static const char* help();                        // "keys: h home arm, p start pose, g home hand, s stop, q quit"

 private:
  void handle(char c);

  int fd_;
  KeyQueue& keys_;
  std::function<void()> on_quit_;
  const SharedState& shared_;
  int64_t last_help_ns_ = 0;
  bool help_logged_ = false;
};

}  // namespace franka_control
