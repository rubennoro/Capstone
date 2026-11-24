// sim_deploy/sim_pc (E4): one MuJoCo model + data owned by one physics thread (DESIGN.md sections 3, 8).
//
// The physics thread steps 1 ms per step paced to CLOCK_MONOTONIC (never ahead of wall time) and runs all the time.
// Each step it applies, on the arm joints, one of:
//   - a torque call (SimArm::runTorqueControl): the TorqueStep is called with this step's ArmReading, its tau goes
//     through libfranka's low-pass + rate limiter, is saturated at the Panda's torque limits, and gravity
//     compensation is added as libfranka does: arm links + the Desk end effector (Franka Hand) + the configured load,
//     NOT the model's true extra mass;
//   - a joint call (SimArm::moveToJoints): a stiff joint PD (control.joint_impedance) on the HomingMotion profile;
//   - otherwise a stiff joint hold where the arm is (the robot's own controller / brakes).
// The fingers get a force-limited position servo (SimGripper). Contacts on the robot (not the fingers) are turned
// into an external joint torque; above control.collision_torque_thresholds_nm the running call ends with a reflex.
// Every state_period_steps steps a sim.state/v1 message is queued for the state PUB; scene requests run between
// two steps (sim_scene.cpp). The scene's {"cmd": "key"} (P1) pushes a PC key into franka_control's own KeyQueue
// (attachKeys) and answers with the supervisor's log line about it.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>

#include "franka_control/arm.h"
#include "franka_control/config.h"
#include "franka_control/homing_motion.h"
#include "franka_control/key_queue.h"
#include "franka_control/types.h"
#include "sim_pc/sim_config.h"

namespace sim_pc {

using franka_control::ArmError;
using franka_control::ArmReading;
using franka_control::ArmStatusFlags;
using franka_control::Vec7;

// One blocking control call of the arm (runTorqueControl or moveToJoints), served by the physics thread.
struct ArmCall {
  enum class Kind { kTorque, kJoints };
  Kind kind = Kind::kTorque;
  const franka_control::Arm::TorqueStep* step = nullptr;
  const franka_control::HomingMotion* motion = nullptr;
  const franka_control::Arm::Observer* observe = nullptr;
  uint64_t epoch = 0;
  // physics thread
  std::optional<franka_control::HomingMotion> profile;   // restarted from the measured q at the first step
  double t_s = 0.0;
  // result (under the world's call mutex)
  bool done = false;
  bool failed = false;
  ArmError::Kind error_kind = ArmError::Kind::kOther;
  std::string error_text;
};

struct StateMessage {
  int64_t t_ns = 0;
  uint64_t step = 0;
  std::vector<double> qpos;
};

struct SimStats {
  uint64_t steps = 0;
  uint64_t slow_steps = 0;          // behind wall time by more than slow_lag_ms
  uint64_t resyncs = 0;             // behind by more than realtime_error_lag_ms: physics time skipped ahead
  double max_lag_ms = 0.0;
  double wall_s = 0.0;              // since the physics thread started
  double physics_s = 0.0;
  uint64_t torque_saturations = 0;  // cycles where a commanded torque hit the Panda's limit
  uint64_t reflexes = 0;
  double step_cost_max_us = 0.0;    // physics + controller work of one step (window)
  double step_cost_mean_us = 0.0;
};

class SimWorld {
 public:
  SimWorld(const SimConfig& sim, const franka_control::Config& cfg);
  ~SimWorld();
  SimWorld(const SimWorld&) = delete;
  SimWorld& operator=(const SimWorld&) = delete;

  void start();                      // the physics thread and the io thread (state PUB, scene REP)
  void stop();

  // ---- arm (SimArm) ----
  ArmReading latestReading() const;
  ArmStatusFlags status() const;
  void runCall(ArmCall& call);       // blocks until the physics thread ends the call; throws ArmError
  void requestStop() { stop_epoch_.fetch_add(1); }
  uint64_t stopEpoch() const { return stop_epoch_.load(); }

  // ---- gripper (SimGripper) ----
  enum class GripResult { kPending, kReached, kStalled, kAborted, kTimeout };
  double gapM() const { return gap_m_.load(); }          // fingertip gap = finger_joint1 + finger_joint2
  double maxGapM() const { return sim_.gripper_max_gap_m; }
  bool graspClosesFully() const { return sim_.gripper_grasp_closes_fully; }
  uint64_t postGrip(double gap_m, double speed_m_per_s, double force_n, bool grasp);
  GripResult gripResult(uint64_t id) const;
  void abortGrip(uint64_t id);

  // ---- io (sim_scene.cpp) ----
  // JSON in, JSON out (io thread); "" = deferred: a key request, answered by ioLoop through keyReply()
  std::string handleSceneRequest(const std::string& request);
  // franka_control's KeyQueue (app.cpp, the messaging thread drains it); nullptr detaches (before the queue ends)
  void attachKeys(franka_control::KeyQueue* keys);
  SimStats stats() const;
  const std::string& modelSha() const { return model_sha_; }

 private:
  void physicsLoop();
  void ioLoop();
  void stepOnce(int64_t t_ns, double lag_ms);
  void computeGravity();
  ArmReading makeReading();
  void endCall(ArmCall* call, bool failed, ArmError::Kind kind, const std::string& text);
  void reflexCheck();
  std::string groundTruthJson(int64_t t_ns);
  std::string geomLabel(int geom) const;
  std::string pressKey(const std::string& key);           // "" = pushed, waiting for its line; else the reply
  std::optional<std::string> keyReply(bool timed_out);    // the deferred reply once the line came (or timed out)
  void onLogLine(const std::string& message);             // the Log tap (under the log lock: must not log)
  // runs fn on the physics thread between two steps; returns its result (or throws on timeout)
  std::string runOnPhysics(std::function<std::string(int64_t)> fn, double timeout_s = 1.0);

  SimConfig sim_;
  franka_control::Config cfg_;
  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;
  std::string model_sha_;
  double timestep_s_ = 0.001;

  // model indices
  std::array<int, 7> arm_qadr_{}, arm_dadr_{}, arm_jnt_{};
  std::array<int, 2> fin_qadr_{}, fin_dadr_{};
  int flange_site_ = -1, tcp_site_ = -1, flange_body_ = -1;
  int eraser_body_ = -1, eraser_qadr_ = -1, eraser_dadr_ = -1;
  std::vector<int> arm_chain_bodies_;       // flange body and its ancestors (libfranka's arm model)
  std::vector<char> robot_body_, finger_body_;
  std::vector<double> jacp_, jacr_;         // 3 x nv scratch

  // physics-thread state
  Vec7 q_hold_ = Vec7::Zero();
  Vec7 tau_last_ = Vec7::Zero();            // tau_J_d: the last commanded (filtered) torque
  Vec7 tau_ext_f_ = Vec7::Zero();
  Vec7 g_true_ = Vec7::Zero(), g_comp_ = Vec7::Zero();
  bool reflex_ = false;
  std::string reflex_text_;
  ArmCall* last_call_ = nullptr;
  uint64_t step_ = 0;
  int64_t t_phys_ns_ = 0;
  double hold_gap_m_ = 0.0;
  std::vector<double> gap_hist_;
  size_t gap_hist_i_ = 0;

  // gripper command (gripper thread <-> physics)
  struct Grip {
    uint64_t id = 0;
    double target = 0.0, speed = 0.05, force = 20.0;
    bool grasp = false;
    bool active = false;
    double sp = 0.0;               // the moving setpoint (gap)
    uint64_t start_step = 0;
    uint64_t timeout_steps = 0;
    GripResult result = GripResult::kPending;
    bool abort = false;
    bool pressing = false;         // a grasp that stalled on an object keeps pressing with its force
  };
  mutable std::mutex grip_mutex_;
  Grip grip_;
  uint64_t grip_next_id_ = 1;

  // arm call
  mutable std::mutex call_mutex_;
  std::condition_variable call_cv_;
  ArmCall* active_ = nullptr;
  std::atomic<uint64_t> stop_epoch_{0};
  std::atomic<bool> reflex_flag_{false};
  mutable std::mutex reflex_mutex_;

  // shared snapshots
  mutable std::mutex reading_mutex_;
  ArmReading reading_;
  std::atomic<double> gap_m_{0.0};

  // scene requests and the state queue
  struct Job {
    std::function<std::string(int64_t)> fn;
    std::promise<std::string> done;
  };
  std::mutex jobs_mutex_;
  std::deque<std::shared_ptr<Job>> jobs_;
  std::mutex state_mutex_;
  std::deque<StateMessage> state_queue_;
  // scene "key": lock order keys_mutex_ -> the log lock -> key_line_mutex_
  std::mutex keys_mutex_;
  franka_control::KeyQueue* keys_ = nullptr;
  std::mutex key_line_mutex_;
  std::string key_wait_;                      // "key p" while a request waits for the supervisor's line
  std::optional<std::string> key_line_;       // that line

  mutable std::mutex stats_mutex_;
  SimStats stats_;
  int64_t wall_start_ns_ = 0;
  double cost_sum_us_ = 0.0;
  uint64_t cost_n_ = 0;
  double cost_max_us_ = 0.0;

  std::atomic<bool> running_{false};
  std::thread physics_, io_;
  std::atomic<bool> io_ready_{false};
  std::string io_error_;
};

}  // namespace sim_pc
