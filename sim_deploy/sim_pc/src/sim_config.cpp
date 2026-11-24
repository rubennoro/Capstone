// sim_deploy/sim_pc (E4): reads the station's `sim:` block (strict).
#include "sim_pc/sim_config.h"

#include <cmath>
#include <cstdlib>
#include <set>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace sim_pc {
namespace {

std::filesystem::path expandUser(const std::string& s) {
  if (!s.empty() && s[0] == '~') {
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home ? home : "") / s.substr(s.size() > 1 && s[1] == '/' ? 2 : 1);
  }
  return std::filesystem::path(s);
}

class Block {
 public:
  Block(const YAML::Node& node, std::string where) : node_(node), where_(std::move(where)) {
    if (node_ && !node_.IsMap()) fail("", "expected a map");
  }
  [[noreturn]] void fail(const std::string& key, const std::string& text) const {
    throw std::runtime_error(where_ + (key.empty() ? "" : "." + key) + ": " + text);
  }
  bool has(const std::string& key) {
    seen_.insert(key);
    return node_ && node_[key] && !node_[key].IsNull();
  }
  Block sub(const std::string& key) {
    seen_.insert(key);
    return Block(node_ ? node_[key] : YAML::Node(), where_ + "." + key);
  }
  void number(const std::string& key, double& out, double min_exclusive = -INFINITY) {
    if (!has(key)) return;
    try {
      out = node_[key].as<double>();
    } catch (const YAML::Exception&) {
      fail(key, "expected a number");
    }
    if (!std::isfinite(out) || !(out > min_exclusive)) fail(key, "out of range");
  }
  void integer(const std::string& key, int& out, int lo, int hi) {
    if (!has(key)) return;
    try {
      out = node_[key].as<int>();
    } catch (const YAML::Exception&) {
      fail(key, "expected an integer");
    }
    if (out < lo || out > hi) fail(key, "must be in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
  }
  void boolean(const std::string& key, bool& out) {
    if (!has(key)) return;
    const std::string s = node_[key].Scalar();
    if (s != "true" && s != "false") fail(key, "expected true or false");
    out = s == "true";
  }
  void text(const std::string& key, std::string& out) {
    if (!has(key)) return;
    if (!node_[key].IsScalar()) fail(key, "expected a string");
    out = node_[key].Scalar();
  }
  void vec3(const std::string& key, std::array<double, 3>& out) {
    if (!has(key)) return;
    const YAML::Node n = node_[key];
    if (!n.IsSequence() || n.size() != 3) fail(key, "expected a list of 3 numbers");
    for (size_t i = 0; i < 3; ++i) out[i] = n[i].as<double>();
  }
  void finish() const {
    if (!node_) return;
    for (auto it = node_.begin(); it != node_.end(); ++it) {
      const std::string k = it->first.as<std::string>();
      if (!seen_.count(k)) fail(k, "unknown key");
    }
  }

 private:
  YAML::Node node_;
  std::string where_;
  std::set<std::string> seen_;
};

}  // namespace

SimConfig SimConfig::load(const std::filesystem::path& station_file) {
  SimConfig c;
  c.station_file = std::filesystem::absolute(station_file).lexically_normal();
  const std::string name = station_file.filename().string();
  YAML::Node root;
  try {
    root = YAML::LoadFile(station_file.string());
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(name + ": cannot read: " + e.what());
  }
  if (!root.IsMap() || !root["sim"] || !root["sim"].IsMap()) {
    throw std::runtime_error(name + ": no sim: block (franka_control --sim needs one; see sim_deploy/sim_pc/README.md)");
  }
  Block b(root["sim"], name + ": sim");
  auto path = [&](const std::string& key, std::filesystem::path& out) {
    std::string s;
    b.text(key, s);
    if (s.empty()) return;
    std::filesystem::path p = expandUser(s);
    if (p.is_relative()) p = c.station_file.parent_path() / p;
    out = p.lexically_normal();
  };
  path("model", c.model);
  if (c.model.empty()) b.fail("model", "missing (the scene.xml to load)");
  path("model_json", c.model_json);
  b.boolean("check_sha256", c.check_sha256);
  b.number("wait_for_model_s", c.wait_for_model_s, -1e-12);
  b.text("bind_address", c.bind_address);
  {
    Block p = b.sub("ports");
    p.integer("state", c.state_port, 1024, 65535);
    p.integer("scene", c.scene_port, 1024, 65535);
    p.finish();
    if (c.state_port == c.scene_port) b.fail("ports", "state and scene must differ");
  }
  b.integer("state_period_steps", c.state_period_steps, 1, 1000);
  {
    Block r = b.sub("realtime");
    r.number("slow_lag_ms", c.slow_lag_ms, 0.0);
    r.number("realtime_error_lag_ms", c.realtime_error_lag_ms, 0.0);
    r.number("stats_period_s", c.stats_period_s, 0.0);
    r.finish();
  }
  {
    Block d = b.sub("desk_end_effector");
    d.number("mass_kg", c.desk_ee_mass_kg, -1e-12);
    d.vec3("com_m", c.desk_ee_com_m);
    d.finish();
  }
  b.number("joint_damping_ratio", c.joint_damping_ratio, 0.0);
  {
    Block g = b.sub("gripper");
    g.number("max_gap_m", c.gripper_max_gap_m, 0.0);
    g.number("move_force_n", c.gripper_move_force_n, 0.0);
    g.number("homing_speed_m_per_s", c.gripper_homing_speed_m_per_s, 0.0);
    g.number("kp_n_per_m", c.gripper_kp_n_per_m, 0.0);
    g.number("stall_window_s", c.gripper_stall_window_s, 0.0);
    g.number("stall_motion_m", c.gripper_stall_motion_m, 0.0);
    g.number("reached_tol_m", c.gripper_reached_tol_m, 0.0);
    g.boolean("grasp_closes_fully", c.gripper_grasp_closes_fully);
    g.text("start", c.gripper_start);
    if (c.gripper_start != "home_key" && c.gripper_start != "gap") g.fail("start", "expected home_key or gap");
    g.number("start_gap_m", c.gripper_start_gap_m, -1e-12);
    g.finish();
  }
  {
    Block r = b.sub("reflex");
    r.boolean("include_finger_contacts", c.reflex_include_finger_contacts);
    r.boolean("include_model_error", c.reflex_include_model_error);
    r.number("filter_hz", c.reflex_filter_hz, 0.0);
    r.finish();
  }
  b.finish();
  return c;
}

}  // namespace sim_pc
