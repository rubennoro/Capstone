// sim_deploy/sim_pc (E4): the physics thread (DESIGN.md sections 3, 8; sim_world.h).
#include "sim_pc/sim_world.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <sstream>

#include <yaml-cpp/yaml.h>

#include "franka_control/clock.h"
#include "franka_control/log.h"
#include "sim_pc/sha256.h"

namespace sim_pc {

using franka_control::Log;
using franka_control::monotonicNowNs;

namespace {

// libfranka 0.9.2 / the Panda (rate_limiting.h, robot datasheet)
constexpr double kMaxTorqueRate = 1000.0;                                   // Nm/s per joint (kMaxTorqueRate)
const Vec7 kTauMax = (Vec7() << 87, 87, 87, 87, 12, 12, 12).finished();     // Nm
constexpr int64_t kStepNs = 1000000;

std::string fmt(double v, int prec = 3) {
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(prec);
  s << v;
  return s.str();
}

int findId(const mjModel* m, int type, const std::string& name) {
  int id = mj_name2id(m, type, name.c_str());
  if (id >= 0) return id;
  // franka_sim's in-memory models prefix attached parts ("hand/finger_joint1"); E6's scene.xml strips it
  const int n = type == mjOBJ_JOINT ? m->njnt : type == mjOBJ_BODY ? m->nbody : type == mjOBJ_SITE ? m->nsite
                : type == mjOBJ_GEOM ? m->ngeom : 0;
  for (int i = 0; i < n; ++i) {
    const char* s = mj_id2name(m, type, i);
    if (!s) continue;
    const std::string full(s);
    if (full.size() > name.size() && full.compare(full.size() - name.size(), name.size(), name) == 0 &&
        full[full.size() - name.size() - 1] == '/') {
      return i;
    }
  }
  return -1;
}

void sleepUntil(int64_t t_ns) {
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(t_ns / 1000000000LL);
  ts.tv_nsec = static_cast<long>(t_ns % 1000000000LL);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) != 0) {
  }
}

// model_sha256 of sim_deploy/model/build.py: sha256 of "<sha256>  <relpath>\n" per file (scene.xml, meshes/*),
// sorted by relpath (byte order).
std::string computeModelSha(const std::filesystem::path& scene) {
  namespace fs = std::filesystem;
  const fs::path dir = scene.parent_path();
  std::vector<std::pair<std::string, std::string>> files;
  files.emplace_back(scene.filename().string(), sha256FileHex(scene));
  if (fs::is_directory(dir / "meshes")) {
    for (const auto& e : fs::directory_iterator(dir / "meshes")) {
      if (e.is_regular_file()) files.emplace_back("meshes/" + e.path().filename().string(), sha256FileHex(e.path()));
    }
  }
  std::sort(files.begin(), files.end());
  std::string text;
  for (const auto& [rel, sha] : files) text += sha + "  " + rel + "\n";
  return sha256Hex(text);
}

}  // namespace

SimWorld::SimWorld(const SimConfig& sim, const franka_control::Config& cfg) : sim_(sim), cfg_(cfg) {
  namespace fs = std::filesystem;
  // the model builder writes model.json last: wait for it if asked
  if (!sim_.model_json.empty() && sim_.wait_for_model_s > 0.0 && !fs::exists(sim_.model_json)) {
    Log::info("sim: waiting up to " + fmt(sim_.wait_for_model_s, 0) + " s for " + sim_.model_json.string());
    const int64_t deadline = monotonicNowNs() + franka_control::secondsToNs(sim_.wait_for_model_s);
    while (!fs::exists(sim_.model_json) && monotonicNowNs() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }
  if (!fs::exists(sim_.model)) throw std::runtime_error("model " + sim_.model.string() + " does not exist");
  model_sha_ = computeModelSha(sim_.model);
  if (!sim_.model_json.empty()) {
    if (!fs::exists(sim_.model_json)) throw std::runtime_error(sim_.model_json.string() + " does not exist");
    YAML::Node doc;
    try {
      doc = YAML::LoadFile(sim_.model_json.string());
    } catch (const YAML::Exception& e) {
      throw std::runtime_error("cannot read " + sim_.model_json.string() + ": " + e.what());
    }
    const std::string expected = doc["model_sha256"] ? doc["model_sha256"].as<std::string>() : "";
    if (expected != model_sha_) {
      const std::string text = "model_sha256 mismatch: model.json says " + expected + ", the files hash to " +
                               model_sha_ + " (rebuild the model: python -m sim_deploy.model build)";
      if (sim_.check_sha256) throw std::runtime_error(text);
      Log::warning("sim: " + text);
    }
  }

  char err[1000] = "";
  m_ = mj_loadXML(sim_.model.string().c_str(), nullptr, err, sizeof(err));
  if (!m_) throw std::runtime_error("mj_loadXML " + sim_.model.string() + ": " + err);
  d_ = mj_makeData(m_);
  if (std::abs(m_->opt.timestep - 0.001) > 1e-12) {
    Log::warning("sim: model timestep " + fmt(m_->opt.timestep * 1e3) + " ms: set to 1 ms (one step per cycle)");
    m_->opt.timestep = 0.001;
  }
  timestep_s_ = m_->opt.timestep;

  // names (DESIGN 2)
  for (int i = 0; i < 7; ++i) {
    const std::string n = "joint" + std::to_string(i + 1);
    const int j = findId(m_, mjOBJ_JOINT, n);
    if (j < 0 || m_->jnt_type[j] != mjJNT_HINGE) throw std::runtime_error("model: no hinge joint " + n);
    arm_jnt_[i] = j;
    arm_qadr_[i] = m_->jnt_qposadr[j];
    arm_dadr_[i] = m_->jnt_dofadr[j];
  }
  double fin_stroke_m = 0.0;
  for (int i = 0; i < 2; ++i) {
    const std::string n = "finger_joint" + std::to_string(i + 1);
    const int j = findId(m_, mjOBJ_JOINT, n);
    if (j < 0 || m_->jnt_type[j] != mjJNT_SLIDE) throw std::runtime_error("model: no slide joint " + n);
    fin_qadr_[i] = m_->jnt_qposadr[j];
    fin_dadr_[i] = m_->jnt_dofadr[j];
    fin_stroke_m += m_->jnt_limited[j] ? m_->jnt_range[2 * j + 1] : 0.5 * sim_.gripper_max_gap_m;
  }
  // W0 (2026-11-20): the hand's stroke is the model's (the printed fingers open 0..0.040: sim_rig.yaml
  // finger_max_gap_m); a larger sim.gripper.max_gap_m could never be reached (home() would time out)
  if (sim_.gripper_max_gap_m > fin_stroke_m + 1e-9) {
    Log::warning("sim: sim.gripper.max_gap_m " + fmt(sim_.gripper_max_gap_m, 4) + " m is above the model's finger " +
                 "stroke " + fmt(fin_stroke_m, 4) + " m (finger_joint1 + finger_joint2 ranges): using the model's");
    sim_.gripper_max_gap_m = fin_stroke_m;
  }
  flange_site_ = findId(m_, mjOBJ_SITE, "flange");
  tcp_site_ = findId(m_, mjOBJ_SITE, "tcp");
  if (flange_site_ < 0) throw std::runtime_error("model: no site flange");
  flange_body_ = m_->site_bodyid[flange_site_];
  eraser_body_ = findId(m_, mjOBJ_BODY, "eraser");
  if (eraser_body_ >= 0 && m_->body_jntnum[eraser_body_] == 1 &&
      m_->jnt_type[m_->body_jntadr[eraser_body_]] == mjJNT_FREE) {
    eraser_qadr_ = m_->jnt_qposadr[m_->body_jntadr[eraser_body_]];
    eraser_dadr_ = m_->jnt_dofadr[m_->body_jntadr[eraser_body_]];
  }

  // libfranka's arm model: link 7 (the body of joint7) and its ancestors; everything welded below it (hand, fingers,
  // rig) is the end effector, compensated only as Desk's Franka Hand + the configured load
  for (int b = m_->jnt_bodyid[arm_jnt_[6]]; b > 0; b = m_->body_parentid[b]) arm_chain_bodies_.push_back(b);
  const int root = m_->body_rootid[m_->jnt_bodyid[arm_jnt_[0]]];
  robot_body_.assign(m_->nbody, 0);
  finger_body_.assign(m_->nbody, 0);
  const int fb0 = m_->jnt_bodyid[m_->dof_jntid[fin_dadr_[0]]], fb1 = m_->jnt_bodyid[m_->dof_jntid[fin_dadr_[1]]];
  for (int b = 1; b < m_->nbody; ++b) {
    robot_body_[b] = m_->body_rootid[b] == root;
    for (int a = b; a > 0; a = m_->body_parentid[a]) {
      if (a == fb0 || a == fb1) finger_body_[b] = 1;
    }
  }

  // franka_control drives every robot dof: actuators on the arm or finger joints (or a tendon) are switched off
  int disabled = 0;
  for (int a = 0; a < m_->nu; ++a) {
    bool robot = m_->actuator_trntype[a] == mjTRN_TENDON;
    if (m_->actuator_trntype[a] == mjTRN_JOINT) {
      const int b = m_->jnt_bodyid[m_->actuator_trnid[2 * a]];
      robot = robot_body_[b] != 0;
    }
    if (!robot) continue;
    for (int k = 0; k < mjNGAIN; ++k) m_->actuator_gainprm[a * mjNGAIN + k] = 0.0;
    for (int k = 0; k < mjNBIAS; ++k) m_->actuator_biasprm[a * mjNBIAS + k] = 0.0;
    ++disabled;
  }

  // initial state: key "home" (E6: q_home, hand open, eraser at its default pose), then q_home of config.yaml
  const int key = mj_name2id(m_, mjOBJ_KEY, "home");
  if (key >= 0) {
    mj_resetDataKeyframe(m_, d_, key);
  } else {
    mj_resetData(m_, d_);
  }
  for (int i = 0; i < 7; ++i) d_->qpos[arm_qadr_[i]] = cfg_.homing.q_home_rad(i);
  if (sim_.gripper_start == "gap" || key < 0) {
    const double g = std::clamp(sim_.gripper_start_gap_m, 0.0, sim_.gripper_max_gap_m);
    d_->qpos[fin_qadr_[0]] = d_->qpos[fin_qadr_[1]] = 0.5 * g;
  }
  mju_zero(d_->qvel, m_->nv);
  mju_zero(d_->ctrl, m_->nu);
  mj_forward(m_, d_);
  for (int i = 0; i < 7; ++i) q_hold_(i) = d_->qpos[arm_qadr_[i]];
  hold_gap_m_ = d_->qpos[fin_qadr_[0]] + d_->qpos[fin_qadr_[1]];
  gap_m_.store(hold_gap_m_);
  jacp_.assign(3 * m_->nv, 0.0);
  jacr_.assign(3 * m_->nv, 0.0);
  gap_hist_.assign(std::max<size_t>(1, static_cast<size_t>(sim_.gripper_stall_window_s / timestep_s_)), hold_gap_m_);
  computeGravity();
  reading_ = makeReading();

  std::string chain;
  for (int b : arm_chain_bodies_) chain += std::string(chain.empty() ? "" : ",") + mj_id2name(m_, mjOBJ_BODY, b);
  double ee_mass = 0.0;
  for (int b = 1; b < m_->nbody; ++b) {
    if (robot_body_[b] && std::find(arm_chain_bodies_.begin(), arm_chain_bodies_.end(), b) == arm_chain_bodies_.end()) {
      ee_mass += m_->body_mass[b];
    }
  }
  Log::info("sim: model " + sim_.model.string() + " (MuJoCo " + mj_versionString() + ", nq " +
            std::to_string(m_->nq) + ", sha256 " + model_sha_.substr(0, 16) + "...); arm model bodies " + chain +
            "; end effector in the model " + fmt(ee_mass) + " kg, compensated " +
            fmt(sim_.desk_ee_mass_kg + cfg_.load.mass_kg) + " kg (Desk hand " + fmt(sim_.desk_ee_mass_kg) +
            " + load " + fmt(cfg_.load.mass_kg) + "); " + std::to_string(disabled) + " model actuators off");
}

SimWorld::~SimWorld() {
  stop();
  if (d_) mj_deleteData(d_);
  if (m_) mj_deleteModel(m_);
}

void SimWorld::start() {
  running_.store(true);
  wall_start_ns_ = monotonicNowNs();
  physics_ = std::thread([this] { physicsLoop(); });
  io_ = std::thread([this] { ioLoop(); });
  const int64_t deadline = monotonicNowNs() + franka_control::secondsToNs(5.0);
  while (!io_ready_.load() && monotonicNowNs() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  if (!io_error_.empty() || !io_ready_.load()) {
    const std::string e = io_error_.empty() ? std::string("io thread did not start") : io_error_;
    stop();
    throw std::runtime_error(e);
  }
}

void SimWorld::stop() {
  attachKeys(nullptr);   // the Log tap points at this world
  if (!running_.exchange(false)) {
    if (physics_.joinable()) physics_.join();
    if (io_.joinable()) io_.join();
    return;
  }
  if (physics_.joinable()) physics_.join();
  if (io_.joinable()) io_.join();
  // a call still waiting (should not happen: the control thread ends before the world)
  std::lock_guard<std::mutex> lock(call_mutex_);
  if (active_) {
    active_->done = active_->failed = true;
    active_->error_kind = ArmError::Kind::kNetwork;
    active_->error_text = "sim: physics stopped";
    active_ = nullptr;
    call_cv_.notify_all();
  }
}

// ---------------------------------------------------------------------------------------------------- the arm

ArmReading SimWorld::latestReading() const {
  std::lock_guard<std::mutex> lock(reading_mutex_);
  return reading_;
}

ArmStatusFlags SimWorld::status() const {
  if (reflex_flag_.load()) {
    std::lock_guard<std::mutex> lock(reflex_mutex_);
    return ArmStatusFlags{franka_control::RobotMode::kReflex, "[\"joint_reflex\"]"};
  }
  std::lock_guard<std::mutex> lock(call_mutex_);
  return ArmStatusFlags{active_ ? franka_control::RobotMode::kMove : franka_control::RobotMode::kIdle, "[]"};
}

void SimWorld::runCall(ArmCall& call) {
  {
    std::unique_lock<std::mutex> lock(call_mutex_);
    if (!running_.load()) throw ArmError(ArmError::Kind::kNetwork, "sim: physics not running");
    active_ = &call;
    call_cv_.wait(lock, [&] { return call.done; });
  }
  if (call.failed) throw ArmError(call.error_kind, call.error_text);
}

void SimWorld::endCall(ArmCall* call, bool failed, ArmError::Kind kind, const std::string& text) {
  std::lock_guard<std::mutex> lock(call_mutex_);
  call->failed = failed;
  call->error_kind = kind;
  call->error_text = text;
  call->done = true;
  if (active_ == call) active_ = nullptr;
  call_cv_.notify_all();
}

void SimWorld::computeGravity() {
  // the joint torques that hold the arm against gravity: g_true for every body of the model, g_comp for libfranka's
  // model (arm links + Desk's end effector + the configured load, at their centres of mass in the flange frame)
  const int nv = m_->nv;
  const double* gv = m_->opt.gravity;
  g_true_.setZero();
  g_comp_.setZero();
  auto add = [&](Vec7& g, double mass) {
    for (int i = 0; i < 7; ++i) {
      double t = 0.0;
      for (int k = 0; k < 3; ++k) t -= jacp_[k * nv + arm_dadr_[i]] * mass * gv[k];
      g(i) += t;
    }
  };
  for (int b = 1; b < m_->nbody; ++b) {
    if (!robot_body_[b] || m_->body_mass[b] <= 0.0) continue;
    mj_jacBodyCom(m_, d_, jacp_.data(), nullptr, b);
    add(g_true_, m_->body_mass[b]);
    if (std::find(arm_chain_bodies_.begin(), arm_chain_bodies_.end(), b) != arm_chain_bodies_.end()) {
      add(g_comp_, m_->body_mass[b]);
    }
  }
  const double* p = d_->site_xpos + 3 * flange_site_;
  const double* R = d_->site_xmat + 9 * flange_site_;   // row-major: world <- flange
  auto point = [&](const std::array<double, 3>& com, double mass) {
    if (mass <= 0.0) return;
    double w[3];
    for (int r = 0; r < 3; ++r) w[r] = p[r] + R[3 * r] * com[0] + R[3 * r + 1] * com[1] + R[3 * r + 2] * com[2];
    mj_jac(m_, d_, jacp_.data(), nullptr, w, flange_body_);
    add(g_comp_, mass);
  };
  point(sim_.desk_ee_com_m, sim_.desk_ee_mass_kg);
  point(cfg_.load.com_m, cfg_.load.mass_kg);
}

ArmReading SimWorld::makeReading() {
  const int nv = m_->nv;
  ArmReading r;
  for (int i = 0; i < 7; ++i) {
    r.q(i) = d_->qpos[arm_qadr_[i]];
    r.dq(i) = d_->qvel[arm_dadr_[i]];
    r.coriolis(i) = d_->qfrc_bias[arm_dadr_[i]] - g_true_(i);
  }
  r.tau_J_d = tau_last_;
  const double* p = d_->site_xpos + 3 * flange_site_;
  const double* R = d_->site_xmat + 9 * flange_site_;
  r.O_T_F = franka_control::Pose::Identity();
  Eigen::Matrix3d Rm;
  Rm << R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8];
  r.O_T_F.linear() = Rm;
  r.O_T_F.translation() = Eigen::Vector3d(p[0], p[1], p[2]);
  mj_jacSite(m_, d_, jacp_.data(), jacr_.data(), flange_site_);
  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 3; ++k) {
      r.J_F(k, i) = jacp_[k * nv + arm_dadr_[i]];
      r.J_F(3 + k, i) = jacr_[k * nv + arm_dadr_[i]];
    }
  }
  r.success_rate = 1.0;
  r.missed_packets = 0;
  return r;
}

// ---------------------------------------------------------------------------------------------------- gripper

uint64_t SimWorld::postGrip(double gap_m, double speed_m_per_s, double force_n, bool grasp) {
  std::lock_guard<std::mutex> lock(grip_mutex_);
  grip_ = Grip{};
  grip_.id = grip_next_id_++;
  grip_.target = std::clamp(gap_m, 0.0, sim_.gripper_max_gap_m);
  grip_.speed = speed_m_per_s;
  grip_.force = force_n;
  grip_.grasp = grasp;
  grip_.active = true;
  grip_.sp = gap_m_.load();
  grip_.start_step = step_;
  const double dist = std::abs(grip_.target - grip_.sp);
  grip_.timeout_steps = static_cast<uint64_t>((dist / std::max(speed_m_per_s, 1e-6) + 3.0) / timestep_s_);
  return grip_.id;
}

SimWorld::GripResult SimWorld::gripResult(uint64_t id) const {
  std::lock_guard<std::mutex> lock(grip_mutex_);
  if (grip_.id != id) return GripResult::kAborted;   // replaced by a newer command
  return grip_.result;
}

void SimWorld::abortGrip(uint64_t id) {
  std::lock_guard<std::mutex> lock(grip_mutex_);
  if (grip_.id == id && grip_.active) grip_.abort = true;
}

// ---------------------------------------------------------------------------------------------------- physics

void SimWorld::physicsLoop() {
  int64_t t0 = monotonicNowNs();
  uint64_t k = 0;
  int64_t last_stats = t0;
  while (running_.load()) {
    int64_t target = t0 + static_cast<int64_t>(k) * kStepNs;
    int64_t now = monotonicNowNs();
    if (now < target) {
      sleepUntil(target);   // never ahead of wall time
      now = monotonicNowNs();
    }
    const double lag_ms = static_cast<double>(now - target) * 1e-6;
    stepOnce(target, lag_ms);
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      ++stats_.steps;
      if (lag_ms > sim_.slow_lag_ms) ++stats_.slow_steps;
      stats_.max_lag_ms = std::max(stats_.max_lag_ms, lag_ms);
      if (lag_ms > sim_.realtime_error_lag_ms) ++stats_.resyncs;
      stats_.physics_s += timestep_s_;
      stats_.wall_s = static_cast<double>(monotonicNowNs() - wall_start_ns_) * 1e-9;
    }
    if (lag_ms > sim_.realtime_error_lag_ms) {
      t0 += now - target;   // skip physics time ahead: wall time and physics time stay one clock
    }
    ++k;
    (void)last_stats;
  }
}

void SimWorld::stepOnce(int64_t t_ns, double lag_ms) {
  const int64_t work_start = monotonicNowNs();
  const int nv = m_->nv;
  t_phys_ns_ = t_ns;
  mj_step1(m_, d_);
  computeGravity();
  const ArmReading r = makeReading();

  ArmCall* call = nullptr;
  {
    std::lock_guard<std::mutex> lock(call_mutex_);
    call = active_;
  }
  Vec7 K, D;
  for (int i = 0; i < 7; ++i) {
    K(i) = cfg_.control.joint_impedance[static_cast<size_t>(i)];
    const double Mii = d_->qM[m_->dof_Madr[arm_dadr_[i]]];
    D(i) = 2.0 * sim_.joint_damping_ratio * std::sqrt(K(i) * Mii);
  }
  auto holdTorque = [&] { return Vec7(K.cwiseProduct(q_hold_ - r.q) - D.cwiseProduct(r.dq) + g_comp_); };
  Vec7 tau_applied = holdTorque();

  if (call) {
    bool ended = false;
    if (call->epoch != stop_epoch_.load()) {
      q_hold_ = r.q;
      endCall(call, true, ArmError::Kind::kStopRequested, "libfranka: control preempted by Robot::stop() (sim)");
      ended = true;
    } else if (reflex_) {
      q_hold_ = r.q;
      endCall(call, true, ArmError::Kind::kReflex, reflex_text_);
      ended = true;
    } else if (lag_ms > sim_.realtime_error_lag_ms) {
      q_hold_ = r.q;
      endCall(call, true, ArmError::Kind::kRealtime,
              "libfranka: communication_constraints_violation (sim: physics " + fmt(lag_ms, 1) +
                  " ms behind wall time)");
      ended = true;
    }
    if (!ended && call->kind == ArmCall::Kind::kTorque) {
      franka_control::TorqueCommand c;
      try {
        c = (*call->step)(r);
      } catch (const std::exception& e) {
        q_hold_ = r.q;
        endCall(call, true, ArmError::Kind::kOther, std::string("torque step threw: ") + e.what());
        ended = true;
      }
      if (!ended) {
        Vec7 tau = c.tau;
        if (!tau.allFinite()) {
          q_hold_ = r.q;
          endCall(call, true, ArmError::Kind::kOther, "libfranka: commanded torque is not finite (sim)");
          ended = true;
        } else {
          // libfranka Robot::control: low-pass on the command, then the rate limiter, both against tau_J_d
          const double f = cfg_.control.libfranka_cutoff_frequency_hz;
          if (f < 1000.0) {
            const double gain = timestep_s_ / (timestep_s_ + 1.0 / (2.0 * M_PI * f));
            tau = gain * tau + (1.0 - gain) * tau_last_;
          }
          const double max_step = kMaxTorqueRate * timestep_s_;
          bool discontinuity = false;
          for (int i = 0; i < 7; ++i) {
            if (cfg_.control.libfranka_limit_rate) {
              tau(i) = std::clamp(tau(i), tau_last_(i) - max_step, tau_last_(i) + max_step);
            } else if (std::abs(tau(i) - tau_last_(i)) > max_step * 1.0001) {
              discontinuity = true;
            }
          }
          bool saturated = false;
          for (int i = 0; i < 7; ++i) {
            if (std::abs(tau(i)) > kTauMax(i)) saturated = true;
            tau(i) = std::clamp(tau(i), -kTauMax(i), kTauMax(i));
          }
          if (saturated) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++stats_.torque_saturations;
          }
          if (discontinuity) {
            q_hold_ = r.q;
            reflex_ = true;
            reflex_text_ = "libfranka: Motion aborted by reflex! [\"controller_torque_discontinuity\"] (sim)";
            reflex_flag_.store(true);
            endCall(call, true, ArmError::Kind::kReflex, reflex_text_);
          } else {
            tau_last_ = tau;
            tau_applied = tau + g_comp_;   // the robot adds gravity (libfranka's model) to the user torque
            if (c.finished) {
              q_hold_ = r.q;
              endCall(call, false, ArmError::Kind::kOther, "");
            }
          }
        }
      }
    } else if (!ended) {   // joint move (homing)
      if (!call->profile) {
        call->profile.emplace(r.q, call->motion->goal(), call->motion->maxJointSpeed(), call->motion->minDurationS());
        call->t_s = 0.0;
      }
      call->t_s += timestep_s_;
      const Vec7 qd = call->profile->positionAt(call->t_s);
      const Vec7 vd = call->profile->velocityAt(call->t_s);
      tau_applied = K.cwiseProduct(qd - r.q) + D.cwiseProduct(vd - r.dq) + g_comp_;
      for (int i = 0; i < 7; ++i) tau_applied(i) = std::clamp(tau_applied(i), -kTauMax(i) + g_comp_(i),
                                                               kTauMax(i) + g_comp_(i));
      try {
        (*call->observe)(r);
      } catch (const std::exception& e) {
        q_hold_ = r.q;
        endCall(call, true, ArmError::Kind::kOther, std::string("observer threw: ") + e.what());
        ended = true;
      }
      if (!ended && call->t_s >= call->profile->durationS()) {
        q_hold_ = call->profile->goal();
        tau_last_.setZero();
        endCall(call, false, ArmError::Kind::kOther, "");
      }
    }
  }
  if (!call) tau_last_ = Vec7::Zero();   // between sessions the robot's own controller holds (tau_J_d 0)

  mju_zero(d_->qfrc_applied, nv);
  for (int i = 0; i < 7; ++i) d_->qfrc_applied[arm_dadr_[i]] = tau_applied(i);

  // fingers: force-limited position servo on each finger joint (the equality keeps them together)
  {
    std::lock_guard<std::mutex> lock(grip_mutex_);
    const double gap = d_->qpos[fin_qadr_[0]] + d_->qpos[fin_qadr_[1]];
    gap_hist_[gap_hist_i_] = gap;
    const size_t next_i = (gap_hist_i_ + 1) % gap_hist_.size();
    const double gap_window_ago = gap_hist_[next_i];
    gap_hist_i_ = next_i;
    double sp = hold_gap_m_;
    double fmax = sim_.gripper_move_force_n;
    if (grip_.active) {
      if (grip_.abort) {
        grip_.active = false;
        grip_.result = GripResult::kAborted;
        hold_gap_m_ = gap;
        sp = gap;
      } else {
        const double dmax = grip_.speed * timestep_s_;
        grip_.sp += std::clamp(grip_.target - grip_.sp, -dmax, dmax);
        sp = grip_.sp;
        fmax = grip_.force;
        const uint64_t n = step_ - grip_.start_step;
        const bool at_target = std::abs(grip_.sp - grip_.target) < 1e-12;
        if (at_target && std::abs(gap - grip_.target) < sim_.gripper_reached_tol_m) {
          grip_.active = false;
          grip_.result = GripResult::kReached;
          hold_gap_m_ = grip_.target;
        } else if (n > gap_hist_.size() && std::abs(gap - gap_window_ago) < sim_.gripper_stall_motion_m &&
                   std::abs(gap - grip_.target) > sim_.gripper_reached_tol_m &&
                   (at_target || std::abs(grip_.sp - gap) > 2.0 * sim_.gripper_reached_tol_m)) {
          grip_.active = false;
          grip_.result = GripResult::kStalled;
          if (grip_.grasp) {
            grip_.pressing = true;   // libfranka grasp: keeps the grasp force on the object
          } else {
            hold_gap_m_ = gap;
          }
        } else if (n > grip_.timeout_steps) {
          grip_.active = false;
          grip_.result = GripResult::kTimeout;
          hold_gap_m_ = gap;
        }
      }
    }
    if (!grip_.active && grip_.pressing) {
      sp = grip_.target;
      fmax = grip_.force;
    }
    for (int f = 0; f < 2; ++f) {
      const int dof = fin_dadr_[f];
      const double mii = d_->qM[m_->dof_Madr[dof]];
      const double kp = sim_.gripper_kp_n_per_m;
      const double kd = 2.0 * std::sqrt(kp * mii);
      const double force = kp * (0.5 * sp - d_->qpos[fin_qadr_[f]]) - kd * d_->qvel[dof];
      d_->qfrc_applied[dof] = std::clamp(force, -fmax, fmax) + d_->qfrc_bias[dof];
    }
  }

  mj_step2(m_, d_);
  reflexCheck();
  ++step_;

  // snapshots
  {
    std::lock_guard<std::mutex> lock(reading_mutex_);
    reading_ = r;
  }
  gap_m_.store(d_->qpos[fin_qadr_[0]] + d_->qpos[fin_qadr_[1]]);

  if (step_ % static_cast<uint64_t>(sim_.state_period_steps) == 0) {
    StateMessage s;
    s.t_ns = t_ns;
    s.step = step_;
    s.qpos.assign(d_->qpos, d_->qpos + m_->nq);
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_queue_.push_back(std::move(s));
    while (state_queue_.size() > 1000) state_queue_.pop_front();
  }

  // scene requests, between two steps
  std::deque<std::shared_ptr<Job>> jobs;
  {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    jobs.swap(jobs_);
  }
  for (auto& j : jobs) {
    try {
      j->done.set_value(j->fn(t_ns));
    } catch (...) {
      j->done.set_exception(std::current_exception());
    }
  }

  const double cost_us = static_cast<double>(monotonicNowNs() - work_start) * 1e-3;
  std::lock_guard<std::mutex> lock(stats_mutex_);
  cost_sum_us_ += cost_us;
  ++cost_n_;
  cost_max_us_ = std::max(cost_max_us_, cost_us);
}

void SimWorld::reflexCheck() {
  // libfranka's tau_ext_hat_filtered: contact forces on the robot (not the fingers) as joint torques, plus the gravity
  // model's error (a wrong load), low-pass filtered; above control.collision_torque_thresholds_nm -> reflex
  const int nv = m_->nv;
  Vec7 raw = Vec7::Zero();
  std::string touching;
  for (int c = 0; c < d_->ncon; ++c) {
    const mjContact& con = d_->contact[c];
    if (con.efc_address < 0 || con.geom[0] < 0 || con.geom[1] < 0) continue;
    const int b0 = m_->geom_bodyid[con.geom[0]], b1 = m_->geom_bodyid[con.geom[1]];
    const bool r0 = robot_body_[b0], r1 = robot_body_[b1];
    if (!r0 && !r1) continue;
    if (!sim_.reflex_include_finger_contacts && (finger_body_[b0] || finger_body_[b1])) continue;
    double f6[6];
    mj_contactForce(m_, d_, c, f6);
    double F[3] = {0, 0, 0};   // force of geom[0] on geom[1], world frame
    for (int k = 0; k < 3; ++k) {
      for (int r = 0; r < 3; ++r) F[r] += f6[k] * con.frame[3 * k + r];
    }
    auto add = [&](int body, double sign) {
      mj_jac(m_, d_, jacp_.data(), nullptr, con.pos, body);
      for (int i = 0; i < 7; ++i) {
        for (int r = 0; r < 3; ++r) raw(i) += sign * jacp_[r * nv + arm_dadr_[i]] * F[r];
      }
    };
    if (r1) add(b1, 1.0);
    if (r0) add(b0, -1.0);
    if (touching.size() < 200) touching += (touching.empty() ? "" : ", ") + geomLabel(con.geom[0]) + "/" +
                                           geomLabel(con.geom[1]);
  }
  if (sim_.reflex_include_model_error) raw += g_comp_ - g_true_;
  const double a = timestep_s_ / (timestep_s_ + 1.0 / (2.0 * M_PI * sim_.reflex_filter_hz));
  tau_ext_f_ = a * raw + (1.0 - a) * tau_ext_f_;
  if (reflex_) return;
  for (int i = 0; i < 7; ++i) {
    const double lim = cfg_.control.collision_torque_thresholds_nm[static_cast<size_t>(i)];
    if (std::abs(tau_ext_f_(i)) > lim) {
      reflex_ = true;
      reflex_text_ = "libfranka: Motion aborted by reflex! [\"joint_reflex\"] (sim: joint " + std::to_string(i + 1) +
                     " external torque " + fmt(tau_ext_f_(i), 1) + " Nm > " + fmt(lim, 1) + " Nm; contacts: " +
                     (touching.empty() ? std::string("none (model error)") : touching) + ")";
      {
        std::lock_guard<std::mutex> lock(reflex_mutex_);
        reflex_flag_.store(true);
      }
      {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++stats_.reflexes;
      }
      for (int j = 0; j < 7; ++j) q_hold_(j) = d_->qpos[arm_qadr_[j]];   // the brakes
      return;
    }
  }
}

std::string SimWorld::geomLabel(int geom) const {
  const char* n = mj_id2name(m_, mjOBJ_GEOM, geom);
  if (n && *n) return n;
  const char* b = mj_id2name(m_, mjOBJ_BODY, m_->geom_bodyid[geom]);
  return b ? std::string(b) : "geom" + std::to_string(geom);
}

SimStats SimWorld::stats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  SimStats s = stats_;
  s.step_cost_max_us = cost_max_us_;
  s.step_cost_mean_us = cost_n_ ? cost_sum_us_ / static_cast<double>(cost_n_) : 0.0;
  return s;
}

std::string SimWorld::runOnPhysics(std::function<std::string(int64_t)> fn, double timeout_s) {
  auto job = std::make_shared<Job>();
  job->fn = std::move(fn);
  std::future<std::string> f = job->done.get_future();
  {
    std::lock_guard<std::mutex> lock(jobs_mutex_);
    jobs_.push_back(job);
  }
  if (f.wait_for(std::chrono::duration<double>(timeout_s)) != std::future_status::ready) {
    throw std::runtime_error("the physics thread did not serve the request within " + fmt(timeout_s, 1) + " s");
  }
  return f.get();
}

}  // namespace sim_pc
