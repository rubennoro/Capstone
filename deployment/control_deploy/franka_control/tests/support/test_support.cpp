// from rnoro/deployment:deployment/franka_control/tests/support/test_support.cpp @ 91b178b
// changes: writeStation drops endpoints; testConfig passes the contract files; + framesFile, linkFile, writeLink,
// freePort (control_deploy/STRUCTURE.md 9.1).
#include "support/test_support.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <fstream>
#include <sstream>

#include "franka_control/config_loader.h"
#include "franka_control/pose_math.h"

namespace franka_control {
namespace test {

ReqClient::ReqClient(zmq::context_t& ctx, const std::string& endpoint) : socket_(ctx, zmq::socket_type::req) {
  socket_.set(zmq::sockopt::linger, 0);
  socket_.connect(endpoint);
}

void ReqClient::send(const std::string& bytes) {
  (void)socket_.send(zmq::const_buffer(bytes.data(), bytes.size()), zmq::send_flags::none);
}

std::optional<std::string> ReqClient::receive(int timeout_ms) {
  zmq::pollitem_t items[] = {{static_cast<void*>(socket_), 0, ZMQ_POLLIN, 0}};
  zmq::poll(items, 1, std::chrono::milliseconds(timeout_ms));
  if (!(items[0].revents & ZMQ_POLLIN)) return std::nullopt;
  zmq::message_t m;
  if (!socket_.recv(m, zmq::recv_flags::dontwait)) return std::nullopt;
  return std::string(static_cast<const char*>(m.data()), m.size());
}

std::optional<std::string> ReqClient::request(const std::string& bytes, int timeout_ms) {
  send(bytes);
  return receive(timeout_ms);
}

std::filesystem::path sourceDir() { return FRANKA_CONTROL_SOURCE_DIR; }
std::filesystem::path examplesDir() { return CONTRACTS_EXAMPLES_DIR; }
std::filesystem::path limitsFile() { return CONTRACTS_LIMITS_FILE; }
std::filesystem::path framesFile() { return CONTRACTS_FRAMES_FILE; }
std::filesystem::path linkFile() { return CONTRACTS_LINK_FILE; }
ContractFiles contractFiles() { return ContractFiles{limitsFile(), framesFile(), linkFile()}; }
std::filesystem::path configFile() { return FRANKA_CONTROL_CONFIG_FILE; }

std::filesystem::path tmpDir(const std::string& name) {
  static std::atomic<int> counter{0};
  const auto dir = std::filesystem::path(FRANKA_CONTROL_TEST_TMP) /
                   (name + "_" + std::to_string(getpid()) + "_" + std::to_string(counter++));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

std::string readFile(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void writeFile(const std::filesystem::path& p, const std::string& text) {
  std::ofstream out(p, std::ios::binary);
  out << text;
}

std::filesystem::path writeStation(const std::filesystem::path& dir, const std::string& franka_control_yaml) {
  std::string text =
      "station_version: 1\n"
      "session_root: " + (dir / "sessions").string() + "\n";
  if (!franka_control_yaml.empty()) text += "franka_control:\n" + franka_control_yaml;
  const auto path = dir / "station.yaml";
  writeFile(path, text);
  return path;
}

Config testConfig(const std::string& franka_control_yaml) {
  const auto dir = tmpDir("config");
  return ConfigLoader::load(writeStation(dir, franka_control_yaml), true, false, configFile(), contractFiles());
}

std::filesystem::path writeLink(const std::filesystem::path& dir, const std::string& address, int state_port,
                                int commands_port, double watchdog_limit_s) {
  YAML::Node link = YAML::LoadFile(linkFile().string());
  link["pc"]["address"] = address;
  link["ports"]["state"] = state_port;
  link["ports"]["commands"] = commands_port;
  link["timing"]["watchdog_limit_s"] = watchdog_limit_s;
  const auto path = dir / "link.yaml";
  YAML::Emitter out;
  out << link;
  writeFile(path, std::string(out.c_str()) + "\n");
  return path;
}

int freePort() {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.sin_port = 0;
  socklen_t len = sizeof(a);
  int port = 0;
  if (bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0 &&
      getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) == 0) {
    port = ntohs(a.sin_port);
  }
  close(fd);
  return port;
}

bool waitUntil(const std::function<bool()>& pred, double timeout_s) {
  const int64_t deadline = monotonicNowNs() + secondsToNs(timeout_s);
  while (!pred()) {
    if (monotonicNowNs() > deadline) return pred();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

Eigen::Quaterniond rotvec(const Eigen::Vector3d& v) {
  const double a = v.norm();
  if (a == 0.0) return Eigen::Quaterniond::Identity();
  return Eigen::Quaterniond(Eigen::AngleAxisd(a, v / a));
}

Pose makePose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q) {
  Pose pose = Pose::Identity();
  pose.linear() = q.normalized().toRotationMatrix();
  pose.translation() = p;
  return pose;
}

TrajectoryPoint point(int64_t t_ns, const Eigen::Vector3d& p, const Eigen::Quaterniond& q, double width_m) {
  TrajectoryPoint tp;
  tp.t_ns = t_ns;
  tp.p = p;
  tp.q = q.normalized();
  tp.width_m = width_m;
  return tp;
}

msg::SetWaypoints makeSetWaypoints(uint64_t chunk_id, const std::vector<TrajectoryPoint>& pts) {
  msg::SetWaypoints m;
  m.chunk_id = chunk_id;
  for (const auto& p : pts) {
    m.t_ns.push_back(p.t_ns);
    m.O_T_TCP.push_back(rowMajorFromPose(makePose(p.p, p.q)));
    m.gripper_width_m.push_back(p.width_m);
  }
  return m;
}

}  // namespace test
}  // namespace franka_control
