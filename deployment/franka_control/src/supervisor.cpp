#include "franka_control/supervisor.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "franka_control/homing_motion.h"
#include "franka_control/log.h"
#include "franka_control/pose_math.h"

namespace franka_control {
namespace {

std::string fmt(double v, int precision) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
  return buf;
}

bool snapshotFinite(const LoopSnapshot& s) {
  return s.q.allFinite() && s.dq.allFinite() && s.O_T_TCP.matrix().allFinite() &&
         s.O_T_TCP_target.matrix().allFinite() && std::isfinite(s.gripper_target_m);
}

}  // namespace

Supervisor::Supervisor(SharedState& shared, Arm& arm, const Config& cfg, const ContractLimits& limits, Clock clock)
    : shared_(shared),
      arm_(arm),
      cfg_(cfg),
      clock_(std::move(clock)),
      gate_(cfg.safety, cfg.F_T_TCP, limits.gripper_width_max_m),
      watchdog_(secondsToNs(cfg.watchdog.limit_s)),
      scratch_(std::make_unique<LoopCommand>()) {
  snap_ = shared_.latestSnapshot();
  next_stats_ns_ = clock_() + secondsToNs(cfg_.messaging.stats_period_s);
  post(LoopCommand::Kind::kHold);
}

uint64_t Supervisor::post(LoopCommand::Kind kind, const WaypointTrajectory* plan) {
  scratch_->generation = next_generation_++;
  scratch_->kind = kind;
  if (plan) {
    scratch_->plan = *plan;
  } else {
    scratch_->plan.clear();
  }
  shared_.postCommand(*scratch_);
  return scratch_->generation;
}

void Supervisor::toFault(Event e, const std::string& text) {
  if (sm_.mode() == Mode::kFault) return;
  sm_.apply(e, text);
  Log::error("fault: " + text);
}

TrajectoryPoint Supervisor::snapshotTarget(int64_t now_ns) const {
  TrajectoryPoint p;
  p.t_ns = now_ns;
  p.p = snap_.O_T_TCP_target.translation();
  p.q = Eigen::Quaterniond(snap_.O_T_TCP_target.linear()).normalized();
  p.width_m = snap_.gripper_target_m;
  return p;
}

void Supervisor::tick() {
  const LoopSnapshot& s = shared_.latestSnapshot();
  const int64_t now = clock_();
  const bool homing = sm_.mode() == Mode::kHoming;

  if (snapshotFinite(s)) {
    snap_ = s;
  } else if (sm_.mode() != Mode::kFault) {
    toFault(Event::kLoopFault, "arm state not finite");
    post(LoopCommand::Kind::kHold);
    if (homing) home_reply_ = msg::Ack{false, "arm state not finite"};
  }

  if (s.fault && sm_.mode() != Mode::kFault) {   // libfranka exception, non-finite torque: the loop has ended
    const std::string text(s.error.data());
    toFault(Event::kLoopFault, text.empty() ? std::string("control loop ended with a fault") : text);
    if (homing) home_reply_ = msg::Ack{false, sm_.detail()};
  }

  if (auto g = shared_.gripperFault(); g && sm_.mode() != Mode::kFault) {
    toFault(Event::kGripperFault, *g);
    post(LoopCommand::Kind::kHold);
    if (homing) home_reply_ = msg::Ack{false, *g};
  }

  if (started() && s.watchdog_trip_generation != 0 && s.watchdog_trip_generation == follow_generation_) {
    const double age = nsToSeconds(watchdog_.ageNs(now, shared_.lastAliveNs()));
    const std::string text = "watchdog (1 kHz loop backstop): no request for " + fmt(age, 3) + " s";
    sm_.apply(Event::kWatchdog, text);
    post(LoopCommand::Kind::kHold);
    Log::warning("stopped: " + text);
  }

  watchdogExpiredNow();

  if (sm_.mode() == Mode::kRunning && !plan_.empty() && now >= plan_.lastTimeNs()) sm_.apply(Event::kPlanEnded);

  if (sm_.mode() == Mode::kHoming) {
    if (s.homed_generation == home_generation_ && s.session > home_session_ &&
        s.phase == LoopSnapshot::Phase::kTorque) {
      const Eigen::Vector3d p = snap_.O_T_TCP.translation();
      if (gate_.insideBox(p)) {
        sm_.apply(Event::kHomingDone);
        post(LoopCommand::Kind::kHold);
        shared_.markAlive(now);   // the watchdog resumes as soon as homing ends
        home_reply_ = msg::Ack{true, ""};
        Log::info("homing finished");
      } else {
        const std::string text = "home pose TCP (" + fmt(p.x(), 2) + ", " + fmt(p.y(), 2) + ", " + fmt(p.z(), 2) +
                                 ") m is outside the workspace box: check homing.q_home_rad and the box";
        toFault(Event::kHomingFailed, text);
        post(LoopCommand::Kind::kHold);
        shared_.markAlive(now);
        home_reply_ = msg::Ack{false, text};
      }
    } else if (now - home_start_ns_ >= secondsToNs(cfg_.homing.timeout_s)) {
      const std::string text = "homing: not finished after " + fmt(cfg_.homing.timeout_s, 1) + " s";
      arm_.requestStop();
      toFault(Event::kHomingFailed, text);
      shared_.markAlive(now);
      home_reply_ = msg::Ack{false, text};
    }
  }

  if (now >= next_stats_ns_) {
    next_stats_ns_ = now + secondsToNs(cfg_.messaging.stats_period_s);
    logStats(s);
  }
}

void Supervisor::logStats(const LoopSnapshot& s) {
  const LoopStats& st = s.stats;
  if (s.phase != LoopSnapshot::Phase::kTorque || st.cycles == 0) return;
  const std::string line = "1 kHz loop: " + std::to_string(st.cycles) + " cycles, period max " +
                           fmt(st.period_max_ns * 1e-6, 3) + " ms, compute max " + fmt(st.compute_max_ns * 1e-6, 3) +
                           " ms, missed packets " + std::to_string(st.missed_packets) + ", late " +
                           std::to_string(st.late_cycles) + " (total " + std::to_string(s.late_cycles_total) +
                           "), success rate min " + fmt(st.success_rate_min, 3) + ", mode " + modeName(sm_.mode());
  if (st.late_cycles > 0 || st.success_rate_min < cfg_.loop_warning.success_rate_min) {
    Log::warning(line);
  } else {
    Log::info(line);
  }
}

bool Supervisor::watchdogExpiredNow() {
  if (!started()) return false;
  const int64_t now = clock_();
  const int64_t last = shared_.lastAliveNs();
  if (!watchdog_.expired(now, last)) return false;
  const std::string text = "watchdog: no request for " + fmt(nsToSeconds(watchdog_.ageNs(now, last)), 3) + " s";
  sm_.apply(Event::kWatchdog, text);
  post(LoopCommand::Kind::kHold);
  Log::warning("stopped: " + text);
  return true;
}

msg::Ack Supervisor::onHeartbeat() { return msg::Ack{true, ""}; }

msg::Ack Supervisor::onStart() {
  const Transition t = sm_.check(Event::kStart);
  if (!t.allowed) return msg::Ack{false, t.reason};
  plan_.clear();
  plan_.append(snapshotTarget(clock_()));
  follow_generation_ = post(LoopCommand::Kind::kFollow, &plan_);
  sm_.apply(Event::kStart);
  Log::info("started: holding the current target");
  return msg::Ack{true, ""};
}

msg::Ack Supervisor::onStop() {
  const Mode before = sm_.mode();
  sm_.apply(Event::kStop, "stop");
  if (before != sm_.mode()) {
    post(LoopCommand::Kind::kHold);
    Log::info("stopped by request (was " + modeName(before) + ")");
  }
  return msg::Ack{true, ""};
}

std::optional<msg::Ack> Supervisor::onHome() {
  const Transition t = sm_.check(Event::kHome);
  if (!t.allowed) return msg::Ack{false, t.reason};
  const HomingMotion motion(snap_.q, cfg_.homing.q_home_rad, cfg_.homing.max_joint_speed_rad_per_s,
                            cfg_.homing.min_duration_s);
  if (motion.durationS() >= cfg_.homing.timeout_s) {
    return msg::Ack{false, "refused: home move would take " + fmt(motion.durationS(), 1) + " s > homing.timeout_s " +
                               fmt(cfg_.homing.timeout_s, 1) + " s: move the arm closer by hand"};
  }
  home_generation_ = post(LoopCommand::Kind::kHome);
  home_session_ = snap_.session;
  home_start_ns_ = clock_();
  home_reply_.reset();
  sm_.apply(Event::kHome);
  Log::info("homing: " + fmt(motion.durationS(), 1) + " s joint move to homing.q_home_rad");
  return std::nullopt;
}

std::optional<msg::Ack> Supervisor::pollHome() {
  std::optional<msg::Ack> r;
  r.swap(home_reply_);
  return r;
}

std::variant<msg::SetWaypointsReply, msg::ErrorReply> Supervisor::onSetWaypoints(const msg::SetWaypoints& m) {
  const Transition t = sm_.check(Event::kSetWaypoints);
  if (!t.allowed) return msg::ErrorReply{msg::SetWaypoints::kSchema, "set_waypoints " + t.reason, ""};

  const int64_t now = clock_();
  const size_t n = m.t_ns.size();
  std::vector<TrajectoryPoint> requested(n);
  for (size_t i = 0; i < n; ++i) {
    const Pose pose = poseFromRowMajor(m.O_T_TCP[i]);
    requested[i].t_ns = m.t_ns[i];
    requested[i].p = pose.translation();
    requested[i].q = Eigen::Quaterniond(pose.linear()).normalized();
    requested[i].width_m = m.gripper_width_m[i];
  }
  const TrajectoryPoint start = plan_.empty() ? snapshotTarget(now) : plan_.at(now);
  const GateResult g = gate_.check(start, requested, now);

  msg::SetWaypointsReply reply;
  reply.chunk_id = m.chunk_id;
  reply.skipped = g.skipped;
  if (!g.rejected.empty()) {
    gate_counts_.rejected += g.failedCount();
    const GateIssue* first = nullptr;
    for (const auto& issue : g.rejected) {
      reply.rejected.push_back(msg::WaypointIssue{issue.index, issue.reason});
      if (issue.failed && !first) first = &issue;
    }
    const std::string text = "safety gate rejected waypoint " + std::to_string(first ? first->index : 0) +
                             " of chunk " + std::to_string(m.chunk_id) + ": " + (first ? first->reason : "");
    toFault(Event::kWaypointsRejected, text);
    post(LoopCommand::Kind::kHold);
  } else {
    plan_.clear();
    plan_.append(start);
    for (const auto& p : g.accepted) {
      if (!plan_.append(p)) throw std::logic_error("plan does not fit WaypointTrajectory::kCapacity");
    }
    follow_generation_ = post(LoopCommand::Kind::kFollow, &plan_);
    sm_.apply(g.accepted.empty() ? Event::kWaypointsAllSkipped : Event::kWaypointsAccepted);
    reply.accepted = g.accepted.size();
    reply.shortened = g.shortened.size();
    gate_counts_.accepted += g.accepted.size();
    gate_counts_.shortened += g.shortened.size();
    gate_counts_.pulled_back += g.pulled_back.size();
    if (!g.shortened.empty() || !g.pulled_back.empty()) {
      std::string line = "chunk " + std::to_string(m.chunk_id) + ": " + std::to_string(reply.accepted) +
                         " accepted (" + std::to_string(g.shortened.size()) + " shortened, " +
                         std::to_string(g.pulled_back.size()) + " pulled back)";
      for (const auto& i : g.pulled_back) line += "; waypoint " + std::to_string(i.index) + ": " + i.reason;
      for (const auto& i : g.shortened) line += "; waypoint " + std::to_string(i.index) + ": " + i.reason;
      Log::info(line);
    }
  }
  if (reply.accepted + reply.skipped + reply.rejected.size() != n) {
    throw std::logic_error("set_waypoints reply does not account for every waypoint exactly once");
  }
  return reply;
}

msg::ArmState Supervisor::stateMessage(uint64_t seq) const {
  msg::ArmState a;
  a.t_ns = snap_.t_ns;
  a.seq = seq;
  a.O_T_TCP = rowMajorFromPose(snap_.O_T_TCP);
  a.O_T_TCP_target = rowMajorFromPose(snap_.O_T_TCP_target);
  for (size_t i = 0; i < 7; ++i) {
    a.q[i] = snap_.q(static_cast<int>(i));
    a.dq[i] = snap_.dq(static_cast<int>(i));
  }
  const double w = shared_.gripperWidth();
  a.gripper_width_m = std::isfinite(w) && w > 0.0 ? w : 0.0;
  a.mode = modeName(sm_.mode());
  a.error = sm_.mode() == Mode::kFault ? sm_.detail() : std::string();
  a.gate = gate_counts_;
  a.loop_late = snap_.late_cycles_total;
  return a;
}

}  // namespace franka_control
