// New (control_deploy/STRUCTURE.md 3.3, 4.3).
#include "franka_control/keyboard_reader.h"

#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "franka_control/clock.h"
#include "franka_control/log.h"

namespace franka_control {

TerminalMode::TerminalMode(int fd) : fd_(fd) {
  if (!isatty(fd_) || tcgetattr(fd_, &saved_) != 0) return;
  termios t = saved_;
  t.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
  t.c_lflag |= ISIG;                                // Ctrl-C stays SIGINT
  t.c_cc[VMIN] = 1;
  t.c_cc[VTIME] = 0;
  t.c_cc[VSUSP] = _POSIX_VDISABLE;                  // Ctrl-Z
  t.c_cc[VQUIT] = _POSIX_VDISABLE;                  // Ctrl-backslash
  active_ = tcsetattr(fd_, TCSANOW, &t) == 0;
}

TerminalMode::~TerminalMode() {
  if (active_) tcsetattr(fd_, TCSANOW, &saved_);
}

KeyboardReader::KeyboardReader(int fd, KeyQueue& keys, std::function<void()> on_quit, const SharedState& shared)
    : fd_(fd), keys_(keys), on_quit_(std::move(on_quit)), shared_(shared) {}

std::optional<Key> KeyboardReader::map(char c) {
  switch (c) {
    case 'h': return Key::kHomeArm;
    case 'g': return Key::kHomeHand;
    case 's': return Key::kStop;
    case 'q': return Key::kQuit;
    default: return std::nullopt;
  }
}

const char* KeyboardReader::help() { return "keys: h home arm, g home hand, s stop, q quit"; }

void KeyboardReader::handle(char c) {
  const std::optional<Key> k = map(c);
  if (!k) {
    if (c == '\n' || c == '\r') return;             // Enter after a key (a pipe or a cooked terminal): not a key
    const int64_t now = monotonicNowNs();
    if (!help_logged_ || now - last_help_ns_ >= kNsPerS) {   // at most once a second
      help_logged_ = true;
      last_help_ns_ = now;
      Log::info(help());
    }
    return;
  }
  if (*k == Key::kQuit) {
    Log::info("key q: shutting down");
    on_quit_();                                     // App::requestStop(), the path SIGTERM takes; never queued
    return;
  }
  keys_.push(*k);
}

void KeyboardReader::run() {
  const TerminalMode mode(fd_);                     // restored when run() returns, also after a FATAL elsewhere
  if (mode.active()) tcflush(fd_, TCIFLUSH);        // keys typed before the process was ready are discarded
  while (!shared_.shutdownRequested()) {
    pollfd p{fd_, POLLIN, 0};
    const int r = poll(&p, 1, kPollMs);
    if (r < 0) {
      if (errno == EINTR) continue;
      Log::warning(std::string("keyboard: poll failed (") + std::strerror(errno) + "): keys disabled");
      return;
    }
    if (r == 0) continue;
    if (p.revents & POLLNVAL) {
      Log::warning("keyboard: stdin is not open: keys disabled");
      return;
    }
    char buf[64];
    const ssize_t n = read(fd_, buf, sizeof(buf));
    if (n == 0) {
      Log::info("keyboard: stdin closed: keys disabled");
      return;
    }
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      if (errno == EIO) {                           // the terminal hung up
        Log::info("keyboard: stdin closed: keys disabled");
      } else {
        Log::warning(std::string("keyboard: read error (") + std::strerror(errno) + "): keys disabled");
      }
      return;
    }
    for (ssize_t i = 0; i < n; ++i) handle(buf[i]);
  }
}

}  // namespace franka_control
