// from rnoro/deployment:deployment/franka_control/src/control_loop.cpp @ 91b178b
// changes: run() only (stop during homing, control_deploy/STRUCTURE.md 4.4): an abort seen before the move skips it;
// a kStopRequested end of the move with the abort flag set waits for the robot to leave its move mode (at most
// control.settle_timeout_s) and starts a new torque session holding the measured pose, or faults if the arm then
// reports a problem; a normal finish clears the flag; the move is bracketed with SharedState::begin/endHomingMove, so a
// homing stop cannot land in the next torque session (fix wave). step() and observeHoming() are unchanged.
// + the move's goal is the accepted kHome's q_target (h: homing.q_home_rad, p: homing.q_start_rad), and its
// open_width_m becomes the finger target just before the move (p opens the hand); step() only stores them.
#include "franka_control/control_loop.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "franka_control/pose_math.h"

namespace franka_control {

ControlLoop::ControlLoop(Arm& arm, SharedState& shared, const Config& cfg, Clock clock)
    : arm_(arm),
      shared_(shared),
      clock_(std::move(clock)),
      F_T_TCP_(cfg.F_T_TCP),
      controller_(cfg.control, cfg.F_T_TCP),
      backstop_(secondsToNs(cfg.watchdog.limit_s + cfg.watchdog.backstop_extra_s)),
      homing_(cfg.homing),
      late_period_ns_(static_cast<int64_t>(cfg.loop_warning.late_period_ms * 1e6)),
      window_ns_(secondsToNs(cfg.messaging.stats_period_s)),
      settle_speed_(cfg.control.settle_speed_rad_per_s),
      settle_timeout_ns_(secondsToNs(cfg.control.settle_timeout_s)) {}

std::optional<ArmError::Kind> ControlLoop::failureKind() const {
  const int k = failure_kind_.load();
  if (k < 0) return std::nullopt;
  return static_cast<ArmError::Kind>(k);
}

void ControlLoop::beginSession() {
  first_cycle_ = true;
  ending_ = Ending::kNone;
  have_prev_ = false;   // the gap between sessions (a homing move) is not a late cycle
}

void ControlLoop::holdCurrent() {
  held_ = last_target_;
  follow_ = Follow::kHold;
  plan_ = nullptr;
}

void ControlLoop::setError(const char* text) {
  const size_t n = std::min(std::strlen(text), cur_.error.size() - 1);
  std::memcpy(cur_.error.data(), text, n);
  cur_.error[n] = '\0';
}

void ControlLoop::publish() {
  shared_.snapshotSlot() = cur_;
  shared_.publishSnapshot();
}

TorqueCommand ControlLoop::step(const ArmReading& r) {
  // 1. time
  const int64_t t = clock_();
  bool late = false;
  if (have_prev_) {
    const int64_t period = t - prev_t_ns_;
    window_.period_max_ns = std::max(window_.period_max_ns, period);
    if (period > late_period_ns_) late = true;
  }
  if (r.missed_packets > 0) {
    late = true;
    window_.missed_packets += r.missed_packets;
  }
  if (late) {
    ++late_total_;
    ++window_.late_cycles;
  }
  have_prev_ = true;
  prev_t_ns_ = t;
  if (!window_started_) {
    window_started_ = true;
    window_start_ns_ = t;
  }
  ++window_.cycles;
  window_.success_rate_min = std::min(window_.success_rate_min, r.success_rate);
  ++cycle_;

  // session start: hold the measured pose and the measured width
  if (first_cycle_) {
    first_cycle_ = false;
    last_target_.O_T_TCP = tcpFromFlange(r.O_T_F, F_T_TCP_);
    // after a finished move with an open width (key p), keep that finger target: the hand may still be opening, and
    // holding the width measured now would close it again (review 2026-11-20)
    last_target_.width_m = (hold_open_width_next_session_ && home_open_width_m_) ? *home_open_width_m_
                                                                                 : shared_.gripperWidth();
    hold_open_width_next_session_ = false;
    holdCurrent();
    controller_.setNullspaceTarget(r.q);
  }

  // 3. new command?
  if (const LoopCommand* c = shared_.takeCommand()) {
    if (c->generation != applied_generation_) {
      applied_generation_ = c->generation;
      switch (c->kind) {
        case LoopCommand::Kind::kHold:
          holdCurrent();
          break;
        case LoopCommand::Kind::kFollow:
          if (ending_ == Ending::kNone) {
            follow_ = Follow::kPlan;
            plan_ = &c->plan;
          } else {
            holdCurrent();
          }
          break;
        case LoopCommand::Kind::kHome:
          holdCurrent();
          if (ending_ == Ending::kNone) {
            ending_ = Ending::kHome;
            ending_start_ns_ = t;
            home_generation_ = c->generation;
            home_target_ = c->q_target ? *c->q_target : homing_.q_home_rad;   // copies: no allocation
            home_open_width_m_ = c->open_width_m;
          }
          break;
      }
    } else if (follow_ == Follow::kPlan) {
      plan_ = &c->plan;   // the same generation in a new slot: keep pointing at memory this thread owns
    }
  }
  if (ending_ == Ending::kNone && shared_.shutdownRequested()) {
    ending_ = Ending::kShutdown;
    ending_start_ns_ = t;
    holdCurrent();
  }

  // 4. backstop watchdog (only matters if the messaging thread is stuck)
  if (follow_ == Follow::kPlan && backstop_.expired(t, shared_.lastAliveNs())) {
    const TrajectoryPoint now_point = plan_->at(t);
    last_target_.O_T_TCP = Pose::Identity();
    last_target_.O_T_TCP.linear() = now_point.q.toRotationMatrix();
    last_target_.O_T_TCP.translation() = now_point.p;
    last_target_.width_m = now_point.width_m;
    holdCurrent();
    watchdog_trip_generation_ = applied_generation_;
  }

  // 5. target
  Target target = held_;
  if (follow_ == Follow::kPlan) {
    const TrajectoryPoint tp = plan_->at(t);
    target.O_T_TCP = Pose::Identity();
    target.O_T_TCP.linear() = tp.q.toRotationMatrix();
    target.O_T_TCP.translation() = tp.p;
    target.width_m = tp.width_m;
  }
  last_target_ = target;
  last_q_ = r.q;

  // 6-7. measured TCP and torque
  const TorqueResult res = controller_.compute(r, target.O_T_TCP);

  // 9. report (filled before the sanity check so a fault still carries this reading)
  cur_.t_ns = t;
  cur_.cycle = cycle_;
  cur_.session = session_;
  cur_.phase = LoopSnapshot::Phase::kTorque;
  cur_.q = r.q;
  cur_.dq = r.dq;
  cur_.O_T_TCP = res.O_T_TCP;
  cur_.O_T_TCP_target = target.O_T_TCP;
  cur_.gripper_target_m = target.width_m;
  cur_.applied_generation = applied_generation_;
  cur_.watchdog_trip_generation = watchdog_trip_generation_;
  cur_.homed_generation = homed_generation_;
  cur_.late_cycles_total = late_total_;

  // 8. sanity
  if (!res.finite) {
    fault_ = true;
    cur_.fault = true;
    setError("controller produced a non-finite torque");
    cur_.stats = last_window_;
    publish();
    return TorqueCommand{Vec7::Zero(), true};
  }

  // 10. finish?
  bool finished = false;
  if (ending_ != Ending::kNone) {
    const bool settled = (r.dq.cwiseAbs().array() < settle_speed_).all();
    finished = settled || t - ending_start_ns_ >= settle_timeout_ns_;
  }

  window_.compute_max_ns = std::max(window_.compute_max_ns, clock_() - t);
  if (t - window_start_ns_ >= window_ns_) {
    last_window_ = window_;
    window_ = LoopStats{};
    window_start_ns_ = t;
  }
  cur_.stats = last_window_;
  publish();
  shared_.setGripperTarget(target.width_m);
  return TorqueCommand{res.tau, finished};
}

void ControlLoop::observeHoming(const ArmReading& r) {
  const Pose measured = tcpFromFlange(r.O_T_F, F_T_TCP_);
  cur_.t_ns = clock_();
  ++cycle_;
  cur_.cycle = cycle_;
  cur_.session = session_;
  cur_.phase = LoopSnapshot::Phase::kHoming;
  cur_.q = r.q;
  cur_.dq = r.dq;
  cur_.O_T_TCP = measured;
  cur_.O_T_TCP_target = measured;
  cur_.applied_generation = applied_generation_;
  cur_.homed_generation = homed_generation_;
  cur_.late_cycles_total = late_total_;
  publish();
}

void ControlLoop::endWith(ArmError::Kind kind, const char* what) {
  const bool expected = kind == ArmError::Kind::kStopRequested && shared_.shutdownRequested();
  if (!expected) {
    fault_ = true;
    cur_.fault = true;
    setError(what);
    failure_kind_.store(static_cast<int>(kind));
  }
  cur_.phase = LoopSnapshot::Phase::kEnded;
  publish();
}

std::string ControlLoop::waitForArmAfterStop() {
  // The C2 t_ns repeats while this waits (deployment/STRUCTURE.md 3.6 obligation 1).
  const int64_t deadline = clock_() + settle_timeout_ns_;
  ArmStatusFlags flags;
  for (;;) {
    try {
      flags = arm_.status();
    } catch (const ArmError& e) {
      return std::string("no state (") + e.what() + ")";
    }
    if (flags.mode != RobotMode::kMove || clock_() >= deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return describeArmProblem(flags);
}

void ControlLoop::abortedHoming() {
  homing_aborted_generation_ = home_generation_;
  cur_.homing_aborted_generation = homing_aborted_generation_;   // step() leaves this field as it is
  ++session_;
}

void ControlLoop::run() {
  struct Done {
    std::atomic<bool>& flag;
    ~Done() { flag.store(true); }
  } done{done_};
  const Arm::TorqueStep step_fn = [this](const ArmReading& r) { return step(r); };
  const Arm::Observer observe_fn = [this](const ArmReading& r) { observeHoming(r); };
  for (;;) {
    if (shared_.shutdownRequested()) break;
    beginSession();
    try {
      arm_.runTorqueControl(step_fn);
    } catch (const ArmError& e) {
      endWith(e.kind, e.what());
      return;
    } catch (const std::exception& e) {
      endWith(ArmError::Kind::kOther, e.what());
      return;
    }
    if (fault_ || ending_ != Ending::kHome || shared_.shutdownRequested()) break;

    // the session ended for kHome. A stop that came before the move: no move, a new session holds the measured pose
    if (shared_.homingAbortRequested()) {
      shared_.clearHomingAbort();
      abortedHoming();
      continue;
    }

    // the joint move, then a new session holding the reached pose. The fingers (key p: open) move meanwhile: the
    // gripper thread follows this target, and after a normal finish the new session keeps it (an aborted move
    // holds the measured width).
    const HomingMotion motion(last_q_, home_target_, homing_.max_joint_speed_rad_per_s, homing_.min_duration_s);
    if (home_open_width_m_) shared_.setGripperTarget(*home_open_width_m_);
    shared_.beginHomingMove(session_);
    try {
      arm_.moveToJoints(motion, observe_fn);
    } catch (const ArmError& e) {
      shared_.endHomingMove();
      if (e.kind == ArmError::Kind::kStopRequested && shared_.homingAbortRequested() &&
          !shared_.shutdownRequested()) {
        // stopped during homing (a C3 stop or the PC's s): hold where the arm stopped, not a fault
        shared_.clearHomingAbort();
        const std::string problem = waitForArmAfterStop();
        if (!problem.empty()) {
          const std::string text = "homing stopped, then the arm reports " + problem +
                                   ": press and release the user stop, then restart control_deploy run";
          endWith(ArmError::Kind::kReflex, text.c_str());
          return;
        }
        abortedHoming();
        continue;
      }
      endWith(e.kind, e.what());
      return;
    } catch (const std::exception& e) {
      shared_.endHomingMove();
      endWith(ArmError::Kind::kOther, e.what());
      return;
    }
    shared_.endHomingMove();   // waits for a stop being sent under its lock: it cannot reach the next session
    shared_.clearHomingAbort();   // a stop that arrived as the move finished: the normal completion wins
    homed_generation_ = home_generation_;
    hold_open_width_next_session_ = home_open_width_m_.has_value();
    ++session_;
  }
  cur_.phase = LoopSnapshot::Phase::kEnded;
  publish();
}

}  // namespace franka_control
