// sim_deploy/sim_pc (E4): the station's `sim:` block (DESIGN.md section 8, "E4: as built").
// franka_control's own loader ignores this top-level key (it reads only session_root and franka_control); this
// loader reads it from the same station file, strictly (an unknown key is an error).
#pragma once

#include <array>
#include <filesystem>
#include <string>

namespace sim_pc {

struct SimConfig {
  std::filesystem::path model;              // scene.xml (relative paths: to the station file's folder)
  std::filesystem::path model_json;         // model.json beside it (sha256 check); empty: no check
  bool check_sha256 = true;
  double wait_for_model_s = 0.0;            // > 0: wait this long for model.json to appear (the builder writes it last)

  std::string bind_address = "127.0.0.1";
  int state_port = 5650;                    // sim.state/v1 PUB (DESIGN 4)
  int scene_port = 5651;                    // scene control REP (DESIGN 5)
  int state_period_steps = 5;               // one state message every 5 physics steps (5 ms)

  // pacing (DESIGN 3)
  double slow_lag_ms = 5.0;                 // behind wall time by more: one `slow` count
  double realtime_error_lag_ms = 50.0;      // behind by more during a control call: ArmError kRealtime
  double stats_period_s = 10.0;             // the log's sim stats line

  // libfranka's gravity model: the end effector configured in Desk (the Franka Hand); the station's load is added
  double desk_ee_mass_kg = 0.73;
  std::array<double, 3> desk_ee_com_m{-0.01, 0.0, 0.03};   // flange frame (Desk / franka_ros F_x_Cee)

  // joint hold outside torque sessions and the homing move's joint PD (stiffness: control.joint_impedance)
  double joint_damping_ratio = 1.0;

  // gripper
  double gripper_max_gap_m = 0.04;          // the hand's stroke with the printed fingers (finger_joint1 + finger_joint2;
                                            // capped at the model's joint ranges; W0: 0.040, limits.yaml)
  double gripper_move_force_n = 40.0;       // per finger, a move (not a grasp)
  double gripper_homing_speed_m_per_s = 0.1;
  double gripper_kp_n_per_m = 2000.0;
  double gripper_stall_window_s = 0.1;      // stalled: the gap moved less than stall_motion_m in this window
  double gripper_stall_motion_m = 0.0002;
  double gripper_reached_tol_m = 0.0005;
  bool gripper_grasp_closes_fully = false;  // W0 opt-in: a grasp closes until the fingers stall (the Franka Hand's
                                            // force-controlled grasp: with nothing between them they close to 0),
                                            // not to the commanded width
  std::string gripper_start = "home_key";   // home_key (the model's key "home") | gap
  double gripper_start_gap_m = 0.04;        // with gripper_start: gap

  // reflexes (libfranka's collision behaviour: control.collision_torque_thresholds_nm)
  bool reflex_include_finger_contacts = false;
  bool reflex_include_model_error = true;   // the gravity-model mismatch (wrong load) counts as external torque
  double reflex_filter_hz = 20.0;           // tau_ext low-pass (libfranka's tau_ext_hat_filtered)

  std::filesystem::path station_file;

  // Throws std::runtime_error naming the station and the key.
  static SimConfig load(const std::filesystem::path& station_file);
};

}  // namespace sim_pc
