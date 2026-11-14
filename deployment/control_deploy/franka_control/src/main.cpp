// from rnoro/deployment:deployment/franka_control/src/main.cpp @ 91b178b
// changes: --version prints "franka_control commit=<40 hex|unknown> dirty=<yes|no> libfranka=<X.Y.Z|none>";
// --check-config passes --local and the contract files (control_deploy/STRUCTURE.md 3.1).
// franka_control: parses the command line, runs App, turns failures into the FATAL line.
#include <cstdio>
#include <iostream>

#include "franka_control/app.h"
#include "franka_control/build_info.h"
#include "franka_control/command_line.h"
#include "franka_control/config_loader.h"
#include "franka_control/log.h"

using franka_control::App;
using franka_control::CommandLine;
using franka_control::ConfigError;
using franka_control::ConfigLoader;
using franka_control::FatalError;
using franka_control::Log;

int main(int argc, char** argv) {
  CommandLine cl;
  try {
    cl = CommandLine::parse(argc, argv);
  } catch (const franka_control::UsageError& e) {
    Log::fatal(std::string("bad command line: ") + e.what());
    return 2;
  }
  if (cl.version) {
    std::cout << "franka_control commit=" << franka_control::kGitCommit
              << " dirty=" << (franka_control::kGitDirty ? "yes" : "no")
              << " libfranka=" << FRANKA_CONTROL_LIBFRANKA_VERSION << std::endl;
    return 0;
  }
  if (cl.check_config) {
    try {
      const auto cfg = ConfigLoader::load(cl.station, cl.mock, cl.local, FRANKA_CONTROL_CONFIG_FILE,
                                          franka_control::bakedContractFiles());
      std::cout << ConfigLoader::toYaml(cfg);
      return 0;
    } catch (const ConfigError& e) {
      for (const auto& p : e.problems) std::cout << p << "\n";
      std::cout.flush();
      return 2;
    }
  }
  try {
    App app(cl);
    return app.run();
  } catch (const FatalError& e) {
    Log::fatal(e.what());
    return 1;
  } catch (const std::exception& e) {
    Log::fatal(std::string("unexpected error: ") + e.what() + ": a franka_control bug; see the log");
    return 1;
  }
}
