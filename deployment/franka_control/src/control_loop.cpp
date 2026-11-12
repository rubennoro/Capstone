#include "franka_control/control_loop.h"

#include <algorithm>
#include <cstring>

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
    last_target_.width_m = shared_.gripperWidth();
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

    // the session ended for kHome: the joint move, then a new session holding the home pose
    const HomingMotion motion(last_q_, homing_.q_home_rad, homing_.max_joint_speed_rad_per_s,
                              homing_.min_duration_s);
    in_homing_.store(true);
    try {
      arm_.moveToJoints(motion, observe_fn);
    } catch (const ArmError& e) {
      in_homing_.store(false);
      endWith(e.kind, e.what());
      return;
    } catch (const std::exception& e) {
      in_homing_.store(false);
      endWith(ArmError::Kind::kOther, e.what());
      return;
    }
    in_homing_.store(false);
    homed_generation_ = home_generation_;
    ++session_;
  }
  cur_.phase = LoopSnapshot::Phase::kEnded;
  publish();
}

}  // namespace franka_control
