// New (control_deploy/STRUCTURE.md 3.1). The same rules as deploy_contracts.LinkSettings for the four keys it reads.
#include "franka_control/contract_link.h"

#include <arpa/inet.h>

#include <cmath>

#include <yaml-cpp/yaml.h>

namespace franka_control {
namespace {

YAML::Node child(const YAML::Node& root, const char* section, const char* key) {
  if (!root.IsMap()) return YAML::Node(YAML::NodeType::Undefined);
  const YAML::Node s = root[section];
  if (!s || !s.IsMap()) return YAML::Node(YAML::NodeType::Undefined);
  const YAML::Node k = s[key];
  if (!k) return YAML::Node(YAML::NodeType::Undefined);
  return k;
}

bool plainScalar(const YAML::Node& n) {
  return n.IsScalar() && n.Tag() != "!" && n.Scalar() != "true" && n.Scalar() != "false";
}

}  // namespace

std::string ContractLink::endpoint(int port, bool local) const {
  return "tcp://" + (local ? std::string("127.0.0.1") : pc_address) + ":" + std::to_string(port);
}

ContractLink ContractLink::load(const std::filesystem::path& path, std::vector<std::string>& problems) {
  ContractLink link;
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception& e) {
    problems.push_back("link.yaml (" + path.string() + "): cannot read: " + e.what());
    return link;
  }
  if (!root.IsMap()) {
    problems.push_back("link.yaml (" + path.string() + "): not a map");
    return link;
  }

  const YAML::Node address = child(root, "pc", "address");
  if (!address) {
    problems.push_back("link.yaml: pc.address: missing");
  } else {
    const std::string text = address.IsScalar() ? address.Scalar() : std::string("(not a string)");
    in_addr parsed{};
    if (!address.IsScalar() || inet_pton(AF_INET, text.c_str(), &parsed) != 1) {
      problems.push_back("link.yaml: pc.address: expected an IPv4 address, got \"" + text + "\"");
    } else if (text == "0.0.0.0") {
      problems.push_back("link.yaml: pc.address: must be the PC's own address, not 0.0.0.0 (never bind every "
                         "interface)");
    } else {
      link.pc_address = text;
    }
  }

  auto port = [&](const char* key, int& out) {
    const std::string name = std::string("ports.") + key;
    const YAML::Node n = child(root, "ports", key);
    if (!n) {
      problems.push_back("link.yaml: " + name + ": missing");
      return false;
    }
    long v = 0;
    bool ok = plainScalar(n);
    if (ok) {
      try {
        v = n.as<long>();
      } catch (const YAML::Exception&) {
        ok = false;
      }
    }
    if (!ok || v < 1024 || v > 65535) {
      problems.push_back("link.yaml: " + name + ": must be an integer port 1024-65535, got " +
                         (n.IsScalar() ? n.Scalar() : std::string("(not a number)")));
      return false;
    }
    out = static_cast<int>(v);
    return true;
  };
  const bool state_ok = port("state", link.state_port);
  const bool commands_ok = port("commands", link.commands_port);
  if (state_ok && commands_ok && link.state_port == link.commands_port) {
    problems.push_back("link.yaml: ports.commands: must differ from ports.state (" + std::to_string(link.state_port) +
                       ")");
  }

  const YAML::Node w = child(root, "timing", "watchdog_limit_s");
  if (!w) {
    problems.push_back("link.yaml: timing.watchdog_limit_s: missing");
  } else {
    double v = 0.0;
    bool ok = plainScalar(w);
    if (ok) {
      try {
        v = w.as<double>();
      } catch (const YAML::Exception&) {
        ok = false;
      }
    }
    if (!ok || !std::isfinite(v) || !(v > 0.0)) {
      problems.push_back("link.yaml: timing.watchdog_limit_s: must be a finite number of seconds > 0, got " +
                         (w.IsScalar() ? w.Scalar() : std::string("(not a number)")));
    } else {
      link.watchdog_limit_s = v;
    }
  }
  return link;
}

}  // namespace franka_control
