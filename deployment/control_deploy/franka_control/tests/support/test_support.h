// from rnoro/deployment:deployment/franka_control/tests/support/test_support.h @ 91b178b
// changes: writeStation drops endpoints; testConfig passes the contract files; + framesFile, linkFile, writeLink,
// freePort (control_deploy/STRUCTURE.md 9.1).
// Test-only helpers: FakeClock, a REQ client, the contract files, the real config with a test station.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <zmq.hpp>

#include "franka_control/clock.h"
#include "franka_control/config.h"
#include "franka_control/config_loader.h"
#include "franka_control/messages.h"
#include "franka_control/types.h"

namespace franka_control {
namespace test {

class FakeClock {
 public:
  explicit FakeClock(int64_t start_ns = 1000 * kNsPerS) : t_(std::make_shared<std::atomic<int64_t>>(start_ns)) {}
  Clock clock() const {
    auto t = t_;
    return [t] { return t->load(); };
  }
  int64_t now() const { return t_->load(); }
  void advance(int64_t ns) { t_->fetch_add(ns); }
  void advanceS(double s) { advance(secondsToNs(s)); }
  void set(int64_t ns) { t_->store(ns); }

 private:
  std::shared_ptr<std::atomic<int64_t>> t_;
};

// A cppzmq REQ socket: request() sends once and waits up to the timeout (nullopt on timeout).
class ReqClient {
 public:
  ReqClient(zmq::context_t& ctx, const std::string& endpoint);
  void send(const std::string& bytes);
  std::optional<std::string> receive(int timeout_ms);
  std::optional<std::string> request(const std::string& bytes, int timeout_ms);

 private:
  zmq::socket_t socket_;
};

std::filesystem::path sourceDir();
std::filesystem::path examplesDir();
std::filesystem::path limitsFile();
std::filesystem::path framesFile();
std::filesystem::path linkFile();
std::filesystem::path configFile();
std::filesystem::path tmpDir(const std::string& name);   // a fresh, empty folder under the build tree
std::string readFile(const std::filesystem::path& p);
void writeFile(const std::filesystem::path& p, const std::string& text);

// A station.yaml with session_root under tmp (no endpoints: they come from link.yaml); `franka_control` is the
// overrides section (YAML text, may be empty).
std::filesystem::path writeStation(const std::filesystem::path& dir, const std::string& franka_control_yaml = "");
// The real config.yaml and contract files (limits, frames, link) with that station, --mock.
Config testConfig(const std::string& franka_control_yaml = "");
// The real contract files.
ContractFiles contractFiles();
// A temporary link.yaml (every key of the real one; these four values replaced).
std::filesystem::path writeLink(const std::filesystem::path& dir, const std::string& address, int state_port,
                                int commands_port, double watchdog_limit_s = 1.0);
// A TCP port on 127.0.0.1 that was free a moment ago.
int freePort();

// Waits until pred() is true or the timeout passes (polls; never a fixed sleep).
bool waitUntil(const std::function<bool()>& pred, double timeout_s);

// A rotation from a rotation vector (axis * angle).
Eigen::Quaterniond rotvec(const Eigen::Vector3d& v);
Pose makePose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q);
TrajectoryPoint point(int64_t t_ns, const Eigen::Vector3d& p, const Eigen::Quaterniond& q, double width_m);

// A set_waypoints message from points (each pose row-major).
msg::SetWaypoints makeSetWaypoints(uint64_t chunk_id, const std::vector<TrajectoryPoint>& pts);

}  // namespace test
}  // namespace franka_control
