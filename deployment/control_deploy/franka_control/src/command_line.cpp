// from rnoro/deployment:deployment/franka_control/src/command_line.cpp @ 91b178b
// changes: + --local and --keyboard; the usage line (control_deploy/STRUCTURE.md 3.1).
// copied from rnoro/final-deployment:deployment/control_deploy/franka_control/src/command_line.cpp @ f632576
// changes: + --sim (needs --local, refused with --mock); the usage line.
#include "franka_control/command_line.h"

#include <string>

namespace franka_control {

std::string CommandLine::usage() {
  return "usage: franka_control --station FILE [--session DIR] [--ready-file FILE] [--mock | --sim] [--local] [--keyboard] "
         "| --version | --check-config --station FILE [--mock] [--local]";
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
    } else if (arg == "--local") {
      cl.local = true;
    } else if (arg == "--keyboard") {
      cl.keyboard = true;
    } else if (arg == "--sim") {
      cl.sim = true;
    } else if (arg == "--version") {
      cl.version = true;
    } else if (arg == "--check-config") {
      cl.check_config = true;
    } else {
      throw UsageError("unknown argument '" + arg + "'; " + usage());
    }
  }
  if (!cl.version && !have_station) throw UsageError("--station is required; " + usage());
  if (cl.sim && cl.mock) throw UsageError("--sim and --mock exclude each other; " + usage());
  if (cl.sim && !cl.local && !cl.version && !cl.check_config) {
    throw UsageError("--sim needs --local (the sim never binds link.yaml pc.address); " + usage());
  }
  return cl;
}

}  // namespace franka_control
