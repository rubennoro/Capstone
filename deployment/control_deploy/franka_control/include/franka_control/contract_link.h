// New (control_deploy/STRUCTURE.md 3.1): the four keys of contracts/link.yaml franka_control uses
// (deployment/STRUCTURE.md 4.3, 4.4): pc.address, ports.state, ports.commands, timing.watchdog_limit_s.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace franka_control {

struct ContractLink {
  std::string pc_address;          // IPv4 dotted literal, never 0.0.0.0
  int state_port = 0;              // C2, 1024-65535
  int commands_port = 0;           // C3, 1024-65535, != state_port
  double watchdog_limit_s = 0.0;   // > 0 (franka_control's own 0.5 s floor is checked by ConfigLoader)

  // tcp://<pc_address>:<port>, or tcp://127.0.0.1:<port> with local (--local).
  std::string endpoint(int port, bool local) const;

  // Reads the four keys with the same validation as the Python loader; ignores every other key. Problems are
  // appended ("link.yaml: pc.address: ...").
  static ContractLink load(const std::filesystem::path& path, std::vector<std::string>& problems);
};

}  // namespace franka_control
