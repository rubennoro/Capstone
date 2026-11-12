// The four common arguments (deploy STRUCTURE.md 4.2), plus --version and --check-config.
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
  bool version = false;
  bool check_config = false;
};

}  // namespace franka_control
