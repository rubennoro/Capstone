// from rnoro/deployment:deployment/franka_control/include/franka_control/log.h @ 91b178b
// changes: none.
// (`STRUCTURE.md n` below: rnoro/deployment's franka_control/STRUCTURE.md, old-S.)
// copied from rnoro/final-deployment:deployment/control_deploy/franka_control/include/franka_control/log.h @ f632576
// changes: Log::setTap, sim build only (FRANKA_CONTROL_HAS_SIM): the sim scene's {"cmd": "key"} reads the supervisor's
// one line about that key (sim_deploy P1).
// One line per message to stderr, in the Python processes' format
// "%(asctime)s %(name)s %(levelname)s %(message)s". Never used inside the 1 kHz callback (it locks and allocates).
#pragma once

#include <string>
#ifdef FRANKA_CONTROL_HAS_SIM
#include <functional>
#endif

namespace franka_control {

class Log {
 public:
  enum class Level { kDebug, kInfo, kWarning, kError };
  static void debug(const std::string& message);
  static void info(const std::string& message);
  static void warning(const std::string& message);
  static void error(const std::string& message);
  static void write(Level level, const std::string& message);
  // The process's last line on a failure: "FATAL franka_control: <what failed>: <what to check>".
  static void fatal(const std::string& what_and_check);
  static void setMinLevel(Level level);
#ifdef FRANKA_CONTROL_HAS_SIM
  // sim_deploy only: every line write() prints is also passed to the tap (empty: none), under the log lock, so the
  // tap must not log. FATAL lines are not passed.
  using Tap = std::function<void(Level, const std::string&)>;
  static void setTap(Tap tap);
#endif
};

}  // namespace franka_control
