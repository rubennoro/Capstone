// from rnoro/deployment:deployment/franka_control/src/supervisor.cpp @ 91b178b
// changes: onHome(Source), onHandHome(), onStop(Source) aborting a homing, onKey(Key); tick() resolves the homing
// outcomes per kind (arm: done, box check, aborted, timeout; hand: done, aborted, timeout) and sends the arm stop once
// the aborted move runs; a C3 home reply only for a C3-started homing; the gripper as the last constructor parameter
// (control_deploy/STRUCTURE.md 3.3, 4.4, 5.2). Fix wave: the arm stop (abort, homing timeout) is sent under
// SharedState::stopHomingMove, re-sent every 100 ms while that move runs and the outcome is unresolved (also after a
// timeout fault); the hand's stop under SharedState::stopHandHoming, so it never aborts a finger move.
// + key p (onStartPose): from home, onHome's arm path (startArmMove, shared with h) to homing.q_start_rad with the
// hand opened to gripper_width_max_m; its completion checks the box and the cone (cone_max_deg, no margin); onStart
// is refused away from the start pose once homing.q_start_rad is set.
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

Supervisor::Supervisor(SharedState& shared, Arm& arm, const Config& cfg, const ContractLimits& limits, Clock clock,
                       Gripper* gripper)
    : shared_(shared),
      arm_(arm),
      cfg_(cfg),
      clock_(std::move(clock)),
      gate_(cfg.safety, cfg.F_T_TCP, limits.gripper_width_max_m),
      watchdog_(secondsToNs(cfg.watchdog.limit_s)),
      scratch_(std::make_unique<LoopCommand>()),
      gripper_(gripper),
      gripper_open_m_(limits.gripper_width_max_m) {
  snap_ = shared_.latestSnapshot();
  next_stats_ns_ = clock_() + secondsToNs(cfg_.messaging.stats_period_s);
  post(LoopCommand::Kind::kHold);
}

uint64_t Supervisor::post(LoopCommand::Kind kind, const WaypointTrajectory* plan, const std::optional<Vec7>& q_target,
                          std::optional<double> open_width_m) {
  scratch_->generation = next_generation_++;
  scratch_->kind = kind;
  scratch_->q_target = q_target;
  scratch_->open_width_m = open_width_m;
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
    if (homing) setHomeReply(false, "arm state not finite");
  }

  if (s.fault && sm_.mode() != Mode::kFault) {   // libfranka exception, non-finite torque: the loop has ended
    const std::string text(s.error.data());
    toFault(Event::kLoopFault, text.empty() ? std::string("control loop ended with a fault") : text);
    if (homing) setHomeReply(false, sm_.detail());
  }

  if (auto g = shared_.gripperFault(); g && sm_.mode() != Mode::kFault) {
    toFault(Event::kGripperFault, *g);
    post(LoopCommand::Kind::kHold);
    if (homing) setHomeReply(false, *g);
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

  tickHoming(s, now);

  if (timeout_stop_pending_) {   // after a homing timeout (fault): stop the move until it is over
    if (s.phase == LoopSnapshot::Phase::kEnded || s.session > home_session_) {
      timeout_stop_pending_ = false;
    } else {
      stopArmMove(s, now);
    }
  }

  if (now >= next_stats_ns_) {
    next_stats_ns_ = now + secondsToNs(cfg_.messaging.stats_period_s);
    logStats(s);
  }
}

void Supervisor::setHomeReply(bool ok, const std::string& reason) {
  if (homing_source_ == Source::kC3) home_reply_ = msg::Ack{ok, reason};
}

void Supervisor::tickHoming(const LoopSnapshot& s, int64_t now) {
  if (sm_.mode() != Mode::kHoming) return;
  const bool by_key = abort_source_ == Source::kKeyboard;

  if (homing_kind_ == HomingKind::kHand) {   // the loop holds the arm; the gripper thread homes the hand
    const uint64_t report = shared_.handHomingReport();
    if (report != 0 && report / 4 == hand_generation_) {
      if (report % 4 == SharedState::kHandHomingAborted) {
        const std::string detail = std::string("stop during hand homing (") + (by_key ? "PC keyboard" : "C3") + ")";
        sm_.apply(Event::kHomingAborted, detail);
        shared_.markAlive(now);
        Log::info("stopped: " + detail + ": the fingers return to the commanded width");
      } else {
        sm_.apply(Event::kHomingDone);
        shared_.markAlive(now);
        if (abort_pending_) Log::info("stop arrived after the hand's homing had finished");
        Log::info("hand homing finished: the fingers return to the commanded width");
      }
      return;
    }
    if (now - home_start_ns_ >= secondsToNs(cfg_.homing.timeout_s)) {
      const std::string text = "hand homing: not finished after " + fmt(cfg_.homing.timeout_s, 1) + " s";
      if (gripper_) shared_.stopHandHoming(hand_generation_, [this] { gripper_->stop(); });
      toFault(Event::kHomingFailed, text);
      shared_.markAlive(now);
    }
    return;
  }

  // the arm: the stop is sent once the move runs (before it, the control thread skips the move itself)
  if (abort_pending_ && shared_.homingAbortRequested()) stopArmMove(s, now);
  const bool new_session = s.session > home_session_ && s.phase == LoopSnapshot::Phase::kTorque;
  if (s.homed_generation == home_generation_ && new_session) {   // the normal completion wins
    shared_.clearHomingAbort();
    if (abort_pending_) Log::info("stop arrived after the home pose was reached");
    const Eigen::Vector3d p = snap_.O_T_TCP.translation();
    const bool start_pose = arm_goal_ == ArmGoal::kStartPose;
    std::string text;   // empty: the reached pose is fine
    if (!gate_.insideBox(p)) {
      text = std::string(start_pose ? "start" : "home") + " pose TCP (" + fmt(p.x(), 2) + ", " + fmt(p.y(), 2) +
             ", " + fmt(p.z(), 2) + ") m is outside the workspace box: check " +
             (start_pose ? "homing.q_start_rad" : "homing.q_home_rad") + " and the box";
    } else if (start_pose) {
      // the cone without its margin: a start inside the margin band would have every waypoint near it pulled back
      const double tilt = gate_.tiltDeg(Eigen::Quaterniond(snap_.O_T_TCP.linear()));
      if (!(tilt <= cfg_.safety.cone_max_deg)) {
        text = "start pose tilt " + fmt(tilt, 1) + " deg from the approach axis > safety.cone_max_deg " +
               fmt(cfg_.safety.cone_max_deg, 1) + ": check homing.q_start_rad and the cone";
      }
    }
    if (text.empty()) {
      sm_.apply(Event::kHomingDone);
      post(LoopCommand::Kind::kHold);
      shared_.markAlive(now);   // the watchdog resumes as soon as homing ends
      setHomeReply(true, "");
      if (start_pose) {
        Log::info("start pose reached (fingers at " + fmt(shared_.gripperWidth(), 3) + " m): start the episode");
      } else {
        Log::info("homing finished");
      }
    } else {
      toFault(Event::kHomingFailed, text);
      post(LoopCommand::Kind::kHold);
      shared_.markAlive(now);
      setHomeReply(false, text);
    }
  } else if (s.homing_aborted_generation == home_generation_ && new_session) {
    shared_.clearHomingAbort();
    const std::string detail = std::string("stop during homing (") + (by_key ? "PC keyboard" : "C3") + ")";
    sm_.apply(Event::kHomingAborted, detail);
    post(LoopCommand::Kind::kHold);
    shared_.markAlive(now);
    setHomeReply(false, std::string(kHomeStoppedPrefix) + " homing stopped by " +
                            (by_key ? "the PC keyboard" : "a C3 stop") +
                            " before the home pose was reached: the arm holds where it stopped");
    Log::info("stopped: " + detail + ": the arm holds where it stopped");
  } else if (now - home_start_ns_ >= secondsToNs(cfg_.homing.timeout_s)) {
    const std::string text = "homing: not finished after " + fmt(cfg_.homing.timeout_s, 1) + " s";
    timeout_stop_pending_ = true;   // tick() keeps stopping the move until it is over
    stopArmMove(s, now);
    toFault(Event::kHomingFailed, text);
    shared_.markAlive(now);
    setHomeReply(false, text);
  }
}

void Supervisor::stopArmMove(const LoopSnapshot& s, int64_t now) {
  // phase kHoming of this homing's session: published only from inside the move, so its stop epoch is recorded
  if (s.phase != LoopSnapshot::Phase::kHoming || s.session != home_session_) return;
  if (arm_stop_sent_ns_ && now - *arm_stop_sent_ns_ < kHomingStopResendNs) return;
  // under the move's lock: if the control thread has already ended the move, nothing is sent
  if (shared_.stopHomingMove(home_session_, [this] { arm_.requestStop(); })) arm_stop_sent_ns_ = now;
}

void Supervisor::abortHoming(Source source) {
  if (abort_pending_) return;
  abort_pending_ = true;
  abort_source_ = source;
  const char* who = source == Source::kKeyboard ? "PC keyboard" : "C3";
  if (homing_kind_ == HomingKind::kArm) {
    shared_.requestHomingAbort();   // the control thread skips the move, or tick() stops it once it runs
    Log::info(std::string("stop during homing (") + who + "): stopping the arm's move");
  } else if (gripper_) {
    // never a finger move: the hand is stopped only while its homing runs; before that, the homing is skipped
    const SharedState::HandStop r = shared_.stopHandHoming(hand_generation_, [this] { gripper_->stop(); });
    const std::string what = r == SharedState::HandStop::kStopped       ? "stopping the Franka Hand"
                             : r == SharedState::HandStop::kBeforeStart ? "the hand's homing will not start"
                                                                        : "the hand's homing has already finished";
    Log::info(std::string("stop during hand homing (") + who + "): " + what);
  }
}

std::string Supervisor::keyRefusal(const std::string& reason) const {
  if (started()) {
    return "mode is " + modeName(sm_.mode()) +
           ": the laptop is driving an episode: press s or stop it on the laptop first";
  }
  const std::string prefix = "refused: ";
  return reason.rfind(prefix, 0) == 0 ? reason.substr(prefix.size()) : reason;
}

void Supervisor::onKey(Key key) {
  switch (key) {
    case Key::kHomeArm:
      if (auto r = onHome(Source::kKeyboard)) Log::warning("key h refused: " + keyRefusal(r->reason));
      break;
    case Key::kStartPose:
      if (auto r = onStartPose(Source::kKeyboard)) Log::warning("key p refused: " + keyRefusal(r->reason));
      break;
    case Key::kHomeHand:
      if (auto r = onHandHome()) {
        Log::warning("key g refused: " + keyRefusal(r->reason));
      } else {
        Log::info("key g: homing the hand: the Franka Hand closes and reopens, then returns to the commanded width");
      }
      break;
    case Key::kStop: {
      const Mode before = sm_.mode();
      if (before == Mode::kFault) {
        Log::warning("key s: no change: mode is fault (" + sm_.detail() + "): Ctrl-C and restart after a fault");
        break;
      }
      onStop(Source::kKeyboard);
      if (before == Mode::kStopped) Log::info("key s: already stopped");
      break;
    }
    case Key::kQuit:   // never queued: the keyboard thread calls App::requestStop() itself
      break;
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
  if (cfg_.homing.q_start_rad &&   // unset: as before (hold_still tests, the bring-up console)
      !(maxJointError(snap_.q, *cfg_.homing.q_start_rad) <= cfg_.homing.at_pose_tol_rad)) {
    return msg::Ack{false, "refused: not at the start pose: press h, then p on the PC"};
  }
  plan_.clear();
  plan_.append(snapshotTarget(clock_()));
  follow_generation_ = post(LoopCommand::Kind::kFollow, &plan_);
  sm_.apply(Event::kStart);
  Log::info("started: holding the current target");
  return msg::Ack{true, ""};
}

msg::Ack Supervisor::onStop(Source source) {
  const Mode before = sm_.mode();
  if (before == Mode::kHoming) {   // ack at once; the mode stays homing until the outcome is known (tick)
    abortHoming(source);
    return msg::Ack{true, ""};
  }
  sm_.apply(Event::kStop, "stop");
  if (before != sm_.mode()) {
    post(LoopCommand::Kind::kHold);
    Log::info(std::string(source == Source::kKeyboard ? "key s: stopped" : "stopped by request") + " (was " +
              modeName(before) + ")");
  }
  return msg::Ack{true, ""};
}

std::optional<msg::Ack> Supervisor::onHandHome() {
  if (!gripper_) return msg::Ack{false, "refused: no gripper"};
  const Transition t = sm_.check(Event::kHome);
  if (!t.allowed) return msg::Ack{false, t.reason};
  homing_kind_ = HomingKind::kHand;
  homing_source_ = Source::kKeyboard;
  abort_pending_ = false;
  arm_stop_sent_ns_.reset();
  home_reply_.reset();
  home_start_ns_ = clock_();
  ++hand_generation_;
  shared_.requestHandHoming(hand_generation_);   // no loop command: the 1 kHz loop keeps holding the arm
  sm_.apply(Event::kHome);
  return std::nullopt;
}

std::optional<msg::Ack> Supervisor::onHome(Source source) {
  return startArmMove(source, ArmGoal::kHome, cfg_.homing.q_home_rad, std::nullopt);
}

std::optional<msg::Ack> Supervisor::onStartPose(Source source) {
  if (!cfg_.homing.q_start_rad) {
    return msg::Ack{false, "refused: homing.q_start_rad is not set: measure it and set it in "
                           "control_deploy/station.yaml (franka_control: homing:)"};
  }
  const Transition t = sm_.check(Event::kHome);
  if (!t.allowed) return msg::Ack{false, t.reason};
  if (!(maxJointError(snap_.q, cfg_.homing.q_home_rad) <= cfg_.homing.at_pose_tol_rad)) {
    return msg::Ack{false, "refused: not at home: press h first"};
  }
  return startArmMove(source, ArmGoal::kStartPose, *cfg_.homing.q_start_rad, gripper_open_m_);
}

std::optional<msg::Ack> Supervisor::startArmMove(Source source, ArmGoal goal, const Vec7& target,
                                                 std::optional<double> open_width_m) {
  const Transition t = sm_.check(Event::kHome);
  if (!t.allowed) return msg::Ack{false, t.reason};
  const bool start_pose = goal == ArmGoal::kStartPose;
  const HomingMotion motion(snap_.q, target, cfg_.homing.max_joint_speed_rad_per_s, cfg_.homing.min_duration_s);
  if (motion.durationS() >= cfg_.homing.timeout_s) {
    return msg::Ack{false, std::string("refused: ") + (start_pose ? "start pose" : "home") + " move would take " +
                               fmt(motion.durationS(), 1) + " s > homing.timeout_s " + fmt(cfg_.homing.timeout_s, 1) +
                               " s: move the arm closer by hand"};
  }
  homing_kind_ = HomingKind::kArm;
  arm_goal_ = goal;
  homing_source_ = source;
  abort_pending_ = false;
  arm_stop_sent_ns_.reset();
  timeout_stop_pending_ = false;
  shared_.clearHomingAbort();   // no homing runs now: nothing stale may skip this one
  home_generation_ = post(LoopCommand::Kind::kHome, nullptr, target, open_width_m);
  home_session_ = snap_.session;
  home_start_ns_ = clock_();
  home_reply_.reset();
  sm_.apply(Event::kHome);
  const bool by_key = source == Source::kKeyboard;
  if (start_pose) {
    Log::info(std::string(by_key ? "key p: " : "") + "start pose: " + fmt(motion.durationS(), 1) +
              " s joint move to homing.q_start_rad");
  } else {
    Log::info(std::string(by_key ? "key h: homing the arm: " : "homing: ") + fmt(motion.durationS(), 1) +
              " s joint move to homing.q_home_rad");
  }
  if (open_width_m) {
    Log::info(std::string(by_key ? "key p: " : "") + "the hand opens to " + fmt(*open_width_m, 3) +
              " m (gripper_width_max_m) during the move");
  }
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
