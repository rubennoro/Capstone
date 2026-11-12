#include "franka_control/command_line.h"

#include <string>

namespace franka_control {

std::string CommandLine::usage() {
  return "usage: franka_control --station FILE [--session DIR] [--ready-file FILE] [--mock] "
         "| --version | --check-config --station FILE [--mock]";
}

CommandLine CommandLine::parse(int argc, char** argv) {
  CommandLine cl;
  bool have_station = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw UsageError(name + " needs a value; " + usage());
      return argv[++i];
    };
    if (arg == "--station") {
      cl.station = value(arg);
      have_station = true;
    } else if (arg == "--session") {
      cl.session = std::filesystem::path(value(arg));
    } else if (arg == "--ready-file") {
      cl.ready_file = std::filesystem::path(value(arg));
    } else if (arg == "--mock") {
      cl.mock = true;
    } else if (arg == "--version") {
      cl.version = true;
    } else if (arg == "--check-config") {
      cl.check_config = true;
    } else {
      throw UsageError("unknown argument '" + arg + "'; " + usage());
    }
  }
  if (!cl.version && !have_station) throw UsageError("--station is required; " + usage());
  return cl;
}

}  // namespace franka_control
