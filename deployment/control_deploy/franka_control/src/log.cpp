// from rnoro/deployment:deployment/franka_control/src/log.cpp @ 91b178b
// changes: none.
// (`STRUCTURE.md n` below: rnoro/deployment's franka_control/STRUCTURE.md, old-S.)
// copied from rnoro/final-deployment:deployment/control_deploy/franka_control/src/log.cpp @ f632576
// changes: Log::setTap, sim build only (FRANKA_CONTROL_HAS_SIM): write() also passes each printed line to the tap
// (sim_deploy P1).
#include "franka_control/log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace franka_control {
namespace {

std::mutex& logMutex() {
  static std::mutex m;
  return m;
}

std::atomic<int>& minLevel() {
  static std::atomic<int> level{static_cast<int>(Log::Level::kInfo)};
  return level;
}

const char* levelName(Log::Level level) {
  switch (level) {
    case Log::Level::kDebug: return "DEBUG";
    case Log::Level::kInfo: return "INFO";
    case Log::Level::kWarning: return "WARNING";
    case Log::Level::kError: return "ERROR";
  }
  return "INFO";
}

#ifdef FRANKA_CONTROL_HAS_SIM
Log::Tap& tap() {   // guarded by logMutex()
  static Log::Tap t;
  return t;
}
#endif

}  // namespace

void Log::write(Level level, const std::string& message) {
  if (static_cast<int>(level) < minLevel().load()) return;
  const auto now = std::chrono::system_clock::now();
  const std::time_t secs = std::chrono::system_clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
  std::tm tm{};
  localtime_r(&secs, &tm);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm);
  std::lock_guard<std::mutex> lock(logMutex());
  std::fprintf(stderr, "%s,%03d franka_control %s %s\n", stamp, static_cast<int>(ms), levelName(level),
               message.c_str());
  std::fflush(stderr);
#ifdef FRANKA_CONTROL_HAS_SIM
  if (tap()) tap()(level, message);
#endif
}

void Log::debug(const std::string& message) { write(Level::kDebug, message); }
void Log::info(const std::string& message) { write(Level::kInfo, message); }
void Log::warning(const std::string& message) { write(Level::kWarning, message); }
void Log::error(const std::string& message) { write(Level::kError, message); }

void Log::fatal(const std::string& what_and_check) {
  std::lock_guard<std::mutex> lock(logMutex());
  std::fprintf(stderr, "FATAL franka_control: %s\n", what_and_check.c_str());
  std::fflush(stderr);
}

void Log::setMinLevel(Level level) { minLevel().store(static_cast<int>(level)); }

#ifdef FRANKA_CONTROL_HAS_SIM
void Log::setTap(Tap t) {
  std::lock_guard<std::mutex> lock(logMutex());
  tap() = std::move(t);
}
#endif

}  // namespace franka_control
