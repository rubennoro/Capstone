#include "franka_control/config_loader.h"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <set>
#include <sstream>

#include <yaml-cpp/yaml.h>

#include "franka_control/pose_math.h"
#include "franka_control/waypoint_trajectory.h"

namespace franka_control {
namespace {

constexpr double kWatchdogLimitMinS = 0.5;   // STRUCTURE.md 7.1: keeps the margin over message_comms' 0.42 s gap
constexpr double kJointImpedanceMax = 14250.0;   // libfranka setJointImpedance range
constexpr double kCutoffMaxHz = 1000.0;          // libfranka kMaxCutoffFrequency

std::vector<std::string> splitPath(const std::string& path) {
  std::vector<std::string> parts;
  std::stringstream ss(path);
  std::string part;
  while (std::getline(ss, part, '.')) parts.push_back(part);
  return parts;
}

// Merges `over` into `base` (a clone): maps key by key, everything else replaced. Every override key must exist.
void mergeInto(YAML::Node base, const YAML::Node& over, const std::string& path, const std::string& station_name,
               std::set<std::string>& overridden, std::vector<std::string>& problems) {
  for (auto it = over.begin(); it != over.end(); ++it) {
    const std::string key = it->first.as<std::string>();
    const std::string sub = path.empty() ? key : path + "." + key;
    const YAML::Node& cbase = base;
    if (!cbase.IsMap() || !cbase[key]) {
      problems.push_back(station_name + ": franka_control." + sub + ": not a key of franka_control/config.yaml");
      continue;
    }
    YAML::Node b = base[key];
    if (b.IsMap()) {
      if (it->second.IsMap()) {
        mergeInto(b, it->second, sub, station_name, overridden, problems);
      } else {
        problems.push_back(station_name + ": franka_control." + sub +
                           ": expected a map (as in franka_control/config.yaml)");
      }
    } else {
      base[key] = YAML::Clone(it->second);
      overridden.insert(sub);
    }
  }
}

void listLeaves(const YAML::Node& node, const std::string& path, std::vector<std::string>& out) {
  if (node.IsMap()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      const std::string key = it->first.as<std::string>();
      listLeaves(it->second, path.empty() ? key : path + "." + key, out);
    }
  } else {
    out.push_back(path);
  }
}

// Reads typed values from the merged tree; every problem names the file the value came from and its key path.
class Reader {
 public:
  Reader(YAML::Node root, std::set<std::string> overridden, std::string station_name,
         std::vector<std::string>& problems)
      : root_(std::move(root)), overridden_(std::move(overridden)), station_name_(std::move(station_name)),
        problems_(problems) {}

  std::string where(const std::string& path) const {
    for (const auto& o : overridden_) {
      if (path == o || path.rfind(o + ".", 0) == 0) return station_name_ + ": franka_control." + path;
    }
    return "franka_control/config.yaml: " + path;
  }
  void problem(const std::string& path, const std::string& text) { problems_.push_back(where(path) + ": " + text); }
  bool isOverridden(const std::string& path) const { return overridden_.count(path) > 0; }
  const std::set<std::string>& used() const { return used_; }

  YAML::Node node(const std::string& path) {
    used_.insert(path);
    const YAML::Node n = descend(root_, splitPath(path), 0);
    if (!n) {
      problem(path, "missing");
      return YAML::Node();
    }
    return n;
  }

  bool number(const std::string& path, double& out) {
    const YAML::Node n = node(path);
    if (!n) return false;
    if (!toNumber(n, out)) {
      problem(path, "expected a number");
      return false;
    }
    return true;
  }
  void positive(const std::string& path, double& out) {
    if (number(path, out) && !(out > 0.0)) problem(path, "must be > 0");
  }
  void nonNegative(const std::string& path, double& out) {
    if (number(path, out) && !(out >= 0.0)) problem(path, "must be >= 0");
  }
  bool integer(const std::string& path, int& out) {
    const YAML::Node n = node(path);
    if (!n) return false;
    try {
      out = n.as<int>();
      return true;
    } catch (const YAML::Exception&) {
      problem(path, "expected an integer");
      return false;
    }
  }
  bool boolean(const std::string& path, bool& out) {
    const YAML::Node n = node(path);
    if (!n) return false;
    if (n.IsScalar()) {
      const std::string s = n.Scalar();
      if (s == "true" || s == "false") {
        out = s == "true";
        return true;
      }
    }
    problem(path, "expected true or false");
    return false;
  }
  bool text(const std::string& path, std::string& out) {
    const YAML::Node n = node(path);
    if (!n) return false;
    if (!n.IsScalar() || n.IsNull() || n.Scalar().empty()) {
      problem(path, "expected a non-empty string");
      return false;
    }
    out = n.Scalar();
    return true;
  }
  bool numbers(const std::string& path, size_t count, std::vector<double>& out) {
    const YAML::Node n = node(path);
    if (!n) return false;
    if (!n.IsSequence() || n.size() != count) {
      problem(path, "expected a list of " + std::to_string(count) + " numbers");
      return false;
    }
    out.clear();
    for (size_t i = 0; i < count; ++i) {
      double v = 0.0;
      if (!toNumber(n[i], v)) {
        problem(path, "expected a list of " + std::to_string(count) + " numbers");
        return false;
      }
      out.push_back(v);
    }
    return true;
  }
  bool matrix(const std::string& path, size_t rows, size_t cols, std::vector<double>& out) {
    const YAML::Node n = node(path);
    if (!n) return false;
    const std::string expected =
        "expected " + std::to_string(rows) + " rows of " + std::to_string(cols) + " numbers (row-major)";
    if (!n.IsSequence() || n.size() != rows) {
      problem(path, expected);
      return false;
    }
    out.clear();
    for (size_t r = 0; r < rows; ++r) {
      if (!n[r].IsSequence() || n[r].size() != cols) {
        problem(path, expected);
        return false;
      }
      for (size_t c = 0; c < cols; ++c) {
        double v = 0.0;
        if (!toNumber(n[r][c], v)) {
          problem(path, expected);
          return false;
        }
        out.push_back(v);
      }
    }
    return true;
  }
  bool isNull(const std::string& path) {
    const YAML::Node n = node(path);
    return n && n.IsNull();
  }

 private:
  // Constructs (never assigns) nodes: assigning a yaml-cpp Node would overwrite the tree it refers to.
  static YAML::Node descend(const YAML::Node& n, const std::vector<std::string>& parts, size_t i) {
    if (i == parts.size()) return n;
    if (!n.IsMap()) return YAML::Node(YAML::NodeType::Undefined);
    const YAML::Node child = n[parts[i]];
    if (!child) return YAML::Node(YAML::NodeType::Undefined);
    return descend(child, parts, i + 1);
  }

  static bool toNumber(const YAML::Node& n, double& out) {
    if (!n.IsScalar()) return false;
    const std::string s = n.Scalar();
    if (s == "true" || s == "false" || n.Tag() == "!") return false;   // a quoted string or a bool is not a number
    try {
      out = n.as<double>();
    } catch (const YAML::Exception&) {
      return false;
    }
    return std::isfinite(out);
  }

  YAML::Node root_;
  std::set<std::string> overridden_;
  std::string station_name_;
  std::vector<std::string>& problems_;
  std::set<std::string> used_;
};

template <int N>
void allPositive(Reader& rd, const std::string& path, const std::vector<double>& v) {
  for (double x : v) {
    if (!(x > 0.0)) {
      rd.problem(path, "every value must be > 0");
      return;
    }
  }
}

std::filesystem::path expandUser(const std::string& s) {
  if (!s.empty() && s[0] == '~') {
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home ? home : "") / s.substr(s.size() > 1 && s[1] == '/' ? 2 : 1);
  }
  return std::filesystem::path(s);
}

void readStation(const YAML::Node& station, const std::filesystem::path& station_file, const std::string& name,
                 Config& c, std::vector<std::string>& problems) {
  auto required = [&](const YAML::Node& parent, const std::string& key, const std::string& path, std::string& out) {
    if (!parent || !parent.IsMap() || !parent[key]) {
      problems.push_back(name + ": " + path + ": missing (required in the station file)");
      return false;
    }
    const YAML::Node n = parent[key];
    if (!n.IsScalar() || n.Scalar().empty()) {
      problems.push_back(name + ": " + path + ": expected a string");
      return false;
    }
    out = n.Scalar();
    return true;
  };
  const YAML::Node endpoints = station["endpoints"];
  for (auto [key, target] : {std::pair<const char*, std::string*>{"state", &c.endpoint_state},
                             std::pair<const char*, std::string*>{"commands", &c.endpoint_commands}}) {
    const std::string path = std::string("endpoints.") + key;
    if (required(endpoints, key, path, *target) && target->rfind("tcp://", 0) != 0) {
      problems.push_back(name + ": " + path + ": expected \"tcp://host:port\", got \"" + *target + "\"");
    }
  }
  std::string root;
  if (required(station, "session_root", "session_root", root)) {
    std::filesystem::path p = expandUser(root);
    if (p.is_relative()) p = station_file.parent_path() / p;
    c.session_root = p.lexically_normal();
  }
}

}  // namespace

Config ConfigLoader::load(const std::filesystem::path& station_file, bool mock,
                          const std::filesystem::path& config_file, const std::filesystem::path& limits_file) {
  std::vector<std::string> problems;
  Config c;
  c.mock = mock;
  c.station_file = std::filesystem::absolute(station_file).lexically_normal();
  const std::string station_name = station_file.filename().string();

  YAML::Node base;
  try {
    base = YAML::LoadFile(config_file.string());
  } catch (const YAML::Exception& e) {
    throw ConfigError({"franka_control/config.yaml (" + config_file.string() + "): cannot read: " + e.what()});
  }
  YAML::Node station;
  try {
    station = YAML::LoadFile(station_file.string());
  } catch (const YAML::Exception& e) {
    throw ConfigError({station_name + " (" + station_file.string() + "): cannot read: " + e.what()});
  }
  if (!station.IsMap()) throw ConfigError({station_name + ": not a map"});

  YAML::Node merged = YAML::Clone(base);
  std::set<std::string> overridden;
  const YAML::Node section = station["franka_control"];
  if (section && !section.IsNull()) {
    if (!section.IsMap()) {
      problems.push_back(station_name + ": franka_control: expected a map of overrides");
    } else {
      mergeInto(merged, section, "", station_name, overridden, problems);
    }
  }
  if (overridden.count("libfranka_version")) {
    problems.push_back(station_name +
                       ": franka_control.libfranka_version: set it in franka_control/config.yaml only "
                       "(the build reads it there), then rerun franka_control/build.sh");
  }

  readStation(station, c.station_file, station_name, c, problems);
  c.limits = ContractLimits::load(limits_file, problems);
  const bool have_limits = c.limits.rotation_tol > 0.0;
  if (c.limits.max_waypoints > WaypointTrajectory::kMaxWaypoints) {
    problems.push_back("limits.yaml: max_waypoints " + std::to_string(c.limits.max_waypoints) +
                       " > the compiled plan capacity " + std::to_string(WaypointTrajectory::kMaxWaypoints) +
                       ": raise kMaxWaypoints and rebuild");
  }

  Reader rd(merged, overridden, station_name, problems);
  std::vector<double> v;

  // top level
  rd.text("binary", c.binary);
  rd.text("robot_ip", c.robot_ip);
  std::string rt;
  if (rd.text("realtime", rt)) {
    if (rt == "enforce") {
      c.realtime = RealtimeMode::kEnforce;
    } else if (rt == "ignore") {
      c.realtime = RealtimeMode::kIgnore;
    } else {
      rd.problem("realtime", "expected enforce or ignore, got \"" + rt + "\"");
    }
  }
  rd.text("libfranka_version", c.libfranka_version);
  if (!rd.isNull("control_cpu")) {
    int cpu = 0;
    if (rd.integer("control_cpu", cpu)) {
      if (cpu < 0) {
        rd.problem("control_cpu", "must be null or a CPU index >= 0");
      } else {
        c.control_cpu = cpu;
      }
    }
  }

  // control
  ControlConfig& k = c.control;
  if (rd.numbers("control.stiffness_translational_n_per_m", 3, v)) {
    k.stiffness_translational_n_per_m = Eigen::Vector3d(v[0], v[1], v[2]);
    allPositive<3>(rd, "control.stiffness_translational_n_per_m", v);
  }
  if (rd.numbers("control.stiffness_rotational_nm_per_rad", 3, v)) {
    k.stiffness_rotational_nm_per_rad = Eigen::Vector3d(v[0], v[1], v[2]);
    allPositive<3>(rd, "control.stiffness_rotational_nm_per_rad", v);
  }
  rd.positive("control.damping_ratio", k.damping_ratio);
  rd.nonNegative("control.nullspace_stiffness", k.nullspace_stiffness);
  rd.positive("control.jacobian_pinv_damping", k.jacobian_pinv_damping);
  rd.positive("control.max_position_error_m", k.max_position_error_m);
  rd.positive("control.max_rotation_error_rad", k.max_rotation_error_rad);
  if (rd.numbers("control.torque_max_nm", 7, v)) {
    for (int i = 0; i < 7; ++i) k.torque_max_nm(i) = v[static_cast<size_t>(i)];
    allPositive<7>(rd, "control.torque_max_nm", v);
  }
  rd.positive("control.torque_rate_max_nm_per_s", k.torque_rate_max_nm_per_s);
  rd.boolean("control.libfranka_limit_rate", k.libfranka_limit_rate);
  if (rd.number("control.libfranka_cutoff_frequency_hz", k.libfranka_cutoff_frequency_hz) &&
      !(k.libfranka_cutoff_frequency_hz > 0.0 && k.libfranka_cutoff_frequency_hz <= kCutoffMaxHz)) {
    rd.problem("control.libfranka_cutoff_frequency_hz", "must be in (0, 1000] Hz");
  }
  if (rd.numbers("control.collision_torque_thresholds_nm", 7, v)) {
    for (size_t i = 0; i < 7; ++i) k.collision_torque_thresholds_nm[i] = v[i];
    allPositive<7>(rd, "control.collision_torque_thresholds_nm", v);
  }
  if (rd.numbers("control.collision_force_thresholds_n", 6, v)) {
    for (size_t i = 0; i < 6; ++i) k.collision_force_thresholds_n[i] = v[i];
    allPositive<6>(rd, "control.collision_force_thresholds_n", v);
  }
  if (rd.numbers("control.joint_impedance", 7, v)) {
    for (size_t i = 0; i < 7; ++i) {
      k.joint_impedance[i] = v[i];
      if (!(v[i] > 0.0 && v[i] <= kJointImpedanceMax)) {
        rd.problem("control.joint_impedance", "every value must be in (0, 14250] Nm/rad");
        break;
      }
    }
  }
  rd.positive("control.settle_speed_rad_per_s", k.settle_speed_rad_per_s);
  rd.positive("control.settle_timeout_s", k.settle_timeout_s);

  // load
  rd.nonNegative("load.mass_kg", c.load.mass_kg);
  if (rd.numbers("load.com_m", 3, v)) {
    for (size_t i = 0; i < 3; ++i) c.load.com_m[i] = v[i];
  }
  if (rd.matrix("load.inertia_kg_m2", 3, 3, v)) {
    for (size_t i = 0; i < 9; ++i) c.load.inertia_kg_m2[i] = v[i];
  }

  // tcp
  if (rd.matrix("tcp.F_T_TCP", 4, 4, v)) {
    std::array<double, 16> m{};
    for (size_t i = 0; i < 16; ++i) m[i] = v[i];
    if (have_limits && !isRigidRowMajor(m, c.limits.rotation_tol)) {
      rd.problem("tcp.F_T_TCP", "not a rigid transform (bottom row [0, 0, 0, 1], orthonormal rotation with det > 0)");
    } else {
      c.F_T_TCP = normalized(poseFromRowMajor(m));
    }
  }

  // safety
  GateConfig& s = c.safety;
  bool box_ok = true;
  if (rd.numbers("safety.workspace_min_m", 3, v)) {
    s.workspace_min_m = Eigen::Vector3d(v[0], v[1], v[2]);
  } else {
    box_ok = false;
  }
  if (rd.numbers("safety.workspace_max_m", 3, v)) {
    s.workspace_max_m = Eigen::Vector3d(v[0], v[1], v[2]);
  } else {
    box_ok = false;
  }
  if (box_ok) {
    static const char* kAxes[3] = {"x", "y", "z"};
    for (int i = 0; i < 3; ++i) {
      if (!(s.workspace_min_m(i) < s.workspace_max_m(i))) {
        const std::string path = rd.isOverridden("safety.workspace_min_m") ? "safety.workspace_min_m"
                                                                            : "safety.workspace_max_m";
        rd.problem(path, std::string("workspace_min_m must be < workspace_max_m on ") + kAxes[i]);
      }
    }
  }
  if (rd.numbers("safety.cone_axis", 3, v)) {
    const Eigen::Vector3d axis(v[0], v[1], v[2]);
    if (!(axis.norm() > 0.0)) {
      rd.problem("safety.cone_axis", "must not be zero");
    } else {
      s.cone_axis = axis.normalized();
    }
  }
  if (rd.number("safety.cone_max_deg", s.cone_max_deg) && !(s.cone_max_deg > 0.0 && s.cone_max_deg < 180.0)) {
    rd.problem("safety.cone_max_deg", "must be in (0, 180)");
  }
  rd.nonNegative("safety.box_margin_m", s.box_margin_m);
  rd.nonNegative("safety.cone_margin_deg", s.cone_margin_deg);
  rd.positive("safety.v_max_m_per_s", s.v_max_m_per_s);
  rd.positive("safety.w_max_rad_per_s", s.w_max_rad_per_s);
  rd.positive("safety.slack_s", s.slack_s);
  rd.positive("safety.max_lead_s", s.max_lead_s);

  // watchdog
  if (rd.number("watchdog.limit_s", c.watchdog.limit_s) && !(c.watchdog.limit_s >= kWatchdogLimitMinS)) {
    rd.problem("watchdog.limit_s", "must be >= 0.5 s (message_comms' longest gap between requests is 0.42 s)");
  }
  rd.positive("watchdog.backstop_extra_s", c.watchdog.backstop_extra_s);

  // homing
  if (rd.numbers("homing.q_home_rad", 7, v)) {
    for (int i = 0; i < 7; ++i) c.homing.q_home_rad(i) = v[static_cast<size_t>(i)];
  }
  rd.positive("homing.max_joint_speed_rad_per_s", c.homing.max_joint_speed_rad_per_s);
  rd.positive("homing.min_duration_s", c.homing.min_duration_s);
  rd.positive("homing.timeout_s", c.homing.timeout_s);

  // gripper
  rd.positive("gripper.speed_m_per_s", c.gripper.speed_m_per_s);
  rd.positive("gripper.grasp_force_n", c.gripper.grasp_force_n);
  rd.positive("gripper.deadband_m", c.gripper.deadband_m);
  rd.positive("gripper.poll_period_s", c.gripper.poll_period_s);
  rd.positive("gripper.read_rate_hz", c.gripper.read_rate_hz);
  rd.number("gripper.width_offset_m", c.gripper.width_offset_m);
  rd.boolean("gripper.homing_on_start", c.gripper.homing_on_start);

  // messaging
  rd.positive("messaging.publish_rate_hz", c.messaging.publish_rate_hz);
  if (rd.integer("messaging.state_send_hwm", c.messaging.state_send_hwm) && c.messaging.state_send_hwm <= 0) {
    rd.problem("messaging.state_send_hwm", "must be > 0");
  }
  rd.positive("messaging.poll_max_s", c.messaging.poll_max_s);
  rd.positive("messaging.stats_period_s", c.messaging.stats_period_s);

  // startup, shutdown, loop_warning
  rd.positive("startup.hold_before_ready_s", c.startup.hold_before_ready_s);
  rd.positive("startup.loop_start_timeout_s", c.startup.loop_start_timeout_s);
  if (rd.number("shutdown.timeout_s", c.shutdown.timeout_s) &&
      !(c.shutdown.timeout_s > c.control.settle_timeout_s)) {
    rd.problem("shutdown.timeout_s", "must be > control.settle_timeout_s");
  }
  if (rd.number("loop_warning.success_rate_min", c.loop_warning.success_rate_min) &&
      !(c.loop_warning.success_rate_min >= 0.0 && c.loop_warning.success_rate_min <= 1.0)) {
    rd.problem("loop_warning.success_rate_min", "must be in [0, 1]");
  }
  rd.positive("loop_warning.late_period_ms", c.loop_warning.late_period_ms);

  // fake_arm, fake_gripper
  if (rd.matrix("fake_arm.home_O_T_TCP", 4, 4, v)) {
    std::array<double, 16> m{};
    for (size_t i = 0; i < 16; ++i) m[i] = v[i];
    if (have_limits && !isRigidRowMajor(m, c.limits.rotation_tol)) {
      rd.problem("fake_arm.home_O_T_TCP", "not a rigid transform");
    } else {
      c.fake_arm.home_O_T_TCP = normalized(poseFromRowMajor(m));
    }
  }
  rd.positive("fake_arm.mass_kg", c.fake_arm.mass_kg);
  rd.positive("fake_arm.inertia_kg_m2", c.fake_arm.inertia_kg_m2);
  rd.nonNegative("fake_gripper.start_width_m", c.fake_gripper.start_width_m);
  rd.positive("fake_gripper.speed_m_per_s", c.fake_gripper.speed_m_per_s);
  if (!rd.isNull("fake_gripper.object_width_m")) {
    double w = 0.0;
    rd.nonNegative("fake_gripper.object_width_m", w);
    c.fake_gripper.object_width_m = w;
  }

  // Every key of config.yaml must be one this loader reads (so config.yaml stays the full list of settings).
  std::vector<std::string> leaves;
  listLeaves(base, "", leaves);
  for (const auto& leaf : leaves) {
    if (!rd.used().count(leaf)) problems.push_back("franka_control/config.yaml: " + leaf + ": unknown key");
  }

  if (!problems.empty()) throw ConfigError(problems);
  c.effective_yaml = YAML::Dump(merged);
  return c;
}

std::string ConfigLoader::toYaml(const Config& c) {
  YAML::Node root;
  root["station_file"] = c.station_file.string();
  root["mock"] = c.mock;
  root["session_root"] = c.session_root.string();
  root["endpoints"]["state"] = c.endpoint_state;
  root["endpoints"]["commands"] = c.endpoint_commands;
  root["contract_limits"]["max_waypoints"] = c.limits.max_waypoints;
  root["contract_limits"]["rotation_tol"] = c.limits.rotation_tol;
  root["contract_limits"]["gripper_width_max_m"] = c.limits.gripper_width_max_m;
  root["franka_control"] = YAML::Load(c.effective_yaml);
  YAML::Emitter out;
  out << root;
  return std::string(out.c_str()) + "\n";
}

}  // namespace franka_control
