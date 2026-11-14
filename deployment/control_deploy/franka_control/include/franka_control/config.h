// from rnoro/deployment:deployment/franka_control/include/franka_control/config.h @ 91b178b
// changes: F_T_TCP and watchdog.limit_s are now filled from contracts/frames.yaml and contracts/link.yaml; new fields
// pc_address, local and the contract file paths; GateConfig::max_lead_s default 1.5 (control_deploy/STRUCTURE.md 3.1).
// Every setting of config.yaml (merged with the station's franka_control section), plus the contract files'
// values, as plain data (control_deploy/STRUCTURE.md section 7).
#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "franka_control/contract_limits.h"
#include "franka_control/types.h"

namespace franka_control {

// A config problem list. what() is the first problem plus "(N problems)" when there are more.
struct ConfigError : std::runtime_error {
  explicit ConfigError(std::vector<std::string> problems_in);
  std::vector<std::string> problems;
  // "fix station.yaml" when the first problem comes from station.yaml, else "fix <file>".
  std::string hint() const;
};

enum class RealtimeMode { kEnforce, kIgnore };

struct ControlConfig {
  Eigen::Vector3d stiffness_translational_n_per_m = Eigen::Vector3d::Constant(800.0);
  Eigen::Vector3d stiffness_rotational_nm_per_rad = Eigen::Vector3d::Constant(40.0);
  double damping_ratio = 1.0;
  double nullspace_stiffness = 0.5;
  double jacobian_pinv_damping = 0.2;
  double max_position_error_m = 0.03;
  double max_rotation_error_rad = 0.2;
  Vec7 torque_max_nm = (Vec7() << 87, 87, 87, 87, 12, 12, 12).finished();
  double torque_rate_max_nm_per_s = 1000.0;
  bool libfranka_limit_rate = true;
  double libfranka_cutoff_frequency_hz = 100.0;
  std::array<double, 7> collision_torque_thresholds_nm{};
  std::array<double, 6> collision_force_thresholds_n{};
  std::array<double, 7> joint_impedance{};
  double settle_speed_rad_per_s = 0.02;
  double settle_timeout_s = 1.0;
};

struct LoadConfig {
  double mass_kg = 0.0;
  std::array<double, 3> com_m{};
  std::array<double, 9> inertia_kg_m2{};   // row-major as written in config.yaml (symmetric)
};

struct GateConfig {
  Eigen::Vector3d workspace_min_m = Eigen::Vector3d(0.25, -0.40, 0.05);
  Eigen::Vector3d workspace_max_m = Eigen::Vector3d(0.80, 0.40, 0.70);
  Eigen::Vector3d cone_axis = Eigen::Vector3d(0.0, 0.0, -1.0);
  double box_margin_m = 0.01;
  double cone_max_deg = 60.0;
  double cone_margin_deg = 2.0;
  double v_max_m_per_s = 0.25;
  double w_max_rad_per_s = 1.0;
  double slack_s = 0.010;
  double max_lead_s = 1.5;
};

struct WatchdogConfig {
  double limit_s = 0.5;                   // from contracts/link.yaml timing.watchdog_limit_s
  double backstop_extra_s = 0.1;
};

struct HomingConfig {
  Vec7 q_home_rad = Vec7::Zero();
  double max_joint_speed_rad_per_s = 0.3;
  double min_duration_s = 2.0;
  double timeout_s = 30.0;
};

struct GripperConfig {
  double speed_m_per_s = 0.05;
  double grasp_force_n = 20.0;
  double deadband_m = 0.005;
  double poll_period_s = 0.02;
  double read_rate_hz = 50.0;
  double width_offset_m = 0.0;
  bool homing_on_start = true;
};

struct MessagingConfig {
  double publish_rate_hz = 100.0;
  int state_send_hwm = 100;
  double poll_max_s = 0.005;
  double stats_period_s = 1.0;
};

struct StartupConfig {
  double hold_before_ready_s = 0.2;
  double loop_start_timeout_s = 10.0;
};

struct ShutdownConfig {
  double timeout_s = 3.0;
};

struct LoopWarningConfig {
  double success_rate_min = 0.95;
  double late_period_ms = 1.5;
};

struct FakeArmConfig {
  Pose home_O_T_TCP = Pose::Identity();
  double mass_kg = 1.0;
  double inertia_kg_m2 = 1.0;
};

struct FakeGripperConfig {
  double start_width_m = 0.08;
  double speed_m_per_s = 0.1;
  std::optional<double> object_width_m;
};

struct Config {
  std::string binary;
  std::string robot_ip;
  RealtimeMode realtime = RealtimeMode::kEnforce;
  std::string libfranka_version;
  std::optional<int> control_cpu;

  ControlConfig control;
  LoadConfig load;
  Pose F_T_TCP = Pose::Identity();        // from contracts/frames.yaml (re-normalised)
  GateConfig safety;
  WatchdogConfig watchdog;
  HomingConfig homing;
  GripperConfig gripper;
  MessagingConfig messaging;
  StartupConfig startup;
  ShutdownConfig shutdown;
  LoopWarningConfig loop_warning;
  FakeArmConfig fake_arm;
  FakeGripperConfig fake_gripper;

  // From the station (session_root, required there) and the contract files.
  std::string endpoint_state;             // tcp://<link.yaml pc.address or 127.0.0.1>:<ports.state>
  std::string endpoint_commands;          // tcp://<link.yaml pc.address or 127.0.0.1>:<ports.commands>
  std::string pc_address;                 // link.yaml pc.address
  std::filesystem::path session_root;
  std::filesystem::path station_file;
  ContractLimits limits;
  bool mock = false;
  bool local = false;                     // --local: bind 127.0.0.1 instead of pc.address
  std::filesystem::path contracts_dir;    // the folder of limits.yaml (deployment/contracts)
  std::filesystem::path limits_file;
  std::filesystem::path frames_file;
  std::filesystem::path link_file;

  std::string effective_yaml;   // the merged franka_control section as YAML (ConfigLoader::toYaml adds the rest)
};

std::string realtimeName(RealtimeMode mode);

}  // namespace franka_control
