// from rnoro/deployment:deployment/franka_control/include/franka_control/command_line.h @ 91b178b
// changes: + --local (bind 127.0.0.1) and --keyboard (read keys from stdin) (control_deploy/STRUCTURE.md 3.1).
// The common arguments, --local, --keyboard, --version and --check-config.
#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace franka_control {

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class CommandLine {
 public:
  static CommandLine parse(int argc, char** argv);   // throws UsageError
  static std::string usage();

  std::filesystem::path station;                     // --station (required, except with --version)
  std::optional<std::filesystem::path> session, ready_file;
  bool mock = false;
  bool local = false;                                // --local: bind tcp://127.0.0.1 on the link.yaml ports
  bool keyboard = false;                             // --keyboard: read the keys h g s q from stdin
  bool version = false;
  bool check_config = false;
};

}  // namespace franka_control
