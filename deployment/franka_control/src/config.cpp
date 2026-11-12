#include "franka_control/config.h"

namespace franka_control {
namespace {

std::string firstWithCount(const std::vector<std::string>& problems) {
  if (problems.empty()) return "config error";
  if (problems.size() == 1) return problems.front();
  return problems.front() + " (" + std::to_string(problems.size()) + " problems)";
}

}  // namespace

ConfigError::ConfigError(std::vector<std::string> problems_in)
    : std::runtime_error(firstWithCount(problems_in)), problems(std::move(problems_in)) {}

std::string ConfigError::hint() const {
  if (problems.empty()) return "fix the config";
  const std::string& first = problems.front();
  const auto colon = first.find(':');
  const std::string file = colon == std::string::npos ? std::string("the config") : first.substr(0, colon);
  return "fix " + file;
}

std::string realtimeName(RealtimeMode mode) { return mode == RealtimeMode::kEnforce ? "enforce" : "ignore"; }

}  // namespace franka_control
