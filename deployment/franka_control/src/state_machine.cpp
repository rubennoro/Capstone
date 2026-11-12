#include "franka_control/state_machine.h"

namespace franka_control {

std::string modeName(Mode mode) {
  switch (mode) {
    case Mode::kIdle: return "idle";
    case Mode::kRunning: return "running";
    case Mode::kHolding: return "holding";
    case Mode::kStopped: return "stopped";
    case Mode::kHoming: return "homing";
    case Mode::kFault: return "fault";
  }
  return "fault";
}

Transition StateMachine::check(Event e) const {
  Transition t;
  t.from = mode_;
  t.to = mode_;
  const Mode m = mode_;
  const bool started = m == Mode::kHolding || m == Mode::kRunning;
  auto ok = [&](Mode to) {
    t.allowed = true;
    t.to = to;
    return t;
  };
  auto refuse = [&](const std::string& why) {
    t.allowed = false;
    std::string r = "refused: mode is " + modeName(m);
    if (m == Mode::kFault) {
      r += ": " + (detail_.empty() ? std::string("fault") : detail_);
    } else if (m == Mode::kStopped && !detail_.empty() && detail_ != "stop") {
      r += " (" + detail_ + ")";
    }
    if (!why.empty() && m != Mode::kFault) r += ": " + why;
    t.reason = r;
    return t;
  };

  switch (e) {
    case Event::kHeartbeat:
    case Event::kGetState: return ok(m);
    case Event::kStart:
      if (m == Mode::kIdle || m == Mode::kStopped) return ok(Mode::kHolding);
      if (started) return refuse("already started");
      if (m == Mode::kHoming) return refuse("homing in progress");
      return refuse("");
    case Event::kStop:   // never refused (message_comms sends it on every conclude)
      if (m == Mode::kIdle || started) return ok(Mode::kStopped);
      return ok(m);
    case Event::kHome:
      if (m == Mode::kIdle || m == Mode::kStopped) return ok(Mode::kHoming);
      if (started) return refuse("send stop first");
      if (m == Mode::kHoming) return refuse("homing in progress");
      return refuse("");
    case Event::kSetWaypoints:
      if (started) return ok(m);
      if (m == Mode::kIdle) return refuse("send start first");
      if (m == Mode::kStopped) return refuse("send start first");
      if (m == Mode::kHoming) return refuse("homing in progress");
      return refuse("");
    case Event::kWaypointsAccepted:
      if (started) return ok(Mode::kRunning);
      return refuse("not following waypoints");
    case Event::kWaypointsAllSkipped:
      if (started) return ok(Mode::kHolding);
      return refuse("not following waypoints");
    case Event::kWaypointsRejected:
      if (started) return ok(Mode::kFault);
      return refuse("not following waypoints");
    case Event::kPlanEnded:
      if (m == Mode::kRunning) return ok(Mode::kHolding);
      return refuse("not running");
    case Event::kWatchdog:
      if (started) return ok(Mode::kStopped);
      return refuse("watchdog not armed");
    case Event::kHomingDone:
      if (m == Mode::kHoming) return ok(Mode::kIdle);
      return refuse("not homing");
    case Event::kHomingFailed:
      if (m == Mode::kHoming) return ok(Mode::kFault);
      return refuse("not homing");
    case Event::kLoopFault:
    case Event::kGripperFault:
      if (m != Mode::kFault) return ok(Mode::kFault);
      return refuse("");
  }
  return refuse("");
}

Transition StateMachine::apply(Event e, const std::string& detail) {
  const Transition t = check(e);
  if (!t.allowed) return t;
  if (t.to != mode_) {
    mode_ = t.to;
    detail_ = (mode_ == Mode::kFault || mode_ == Mode::kStopped) ? detail : std::string();
  }
  return t;
}

}  // namespace franka_control
