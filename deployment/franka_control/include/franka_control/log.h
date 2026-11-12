// One line per message to stderr, in the Python processes' format
// "%(asctime)s %(name)s %(levelname)s %(message)s". Never used inside the 1 kHz callback (it locks and allocates).
#pragma once

#include <string>

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
};

}  // namespace franka_control
