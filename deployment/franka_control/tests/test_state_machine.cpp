#include <gtest/gtest.h>

#include <vector>

#include "franka_control/state_machine.h"

namespace franka_control {
namespace {

// Drives a fresh machine (starts idle) into `m`.
StateMachine inMode(Mode m) {
  StateMachine sm;
  switch (m) {
    case Mode::kIdle: break;
    case Mode::kHolding: sm.apply(Event::kStart); break;
    case Mode::kRunning:
      sm.apply(Event::kStart);
      sm.apply(Event::kWaypointsAccepted);
      break;
    case Mode::kStopped: sm.apply(Event::kStop, "stop"); break;
    case Mode::kHoming: sm.apply(Event::kHome); break;
    case Mode::kFault: sm.apply(Event::kLoopFault, "libfranka: motion aborted by reflex! [cartesian_reflex]"); break;
  }
  EXPECT_EQ(sm.mode(), m);
  return sm;
}

const std::vector<Mode> kAll = {Mode::kIdle, Mode::kHolding, Mode::kRunning, Mode::kStopped, Mode::kHoming,
                                Mode::kFault};

struct Row {
  Mode from;
  Event event;
  Mode to;
};

TEST(StateMachine, EveryAllowedTransitionOfTheTable) {
  const std::vector<Row> rows = {
      {Mode::kIdle, Event::kStart, Mode::kHolding},
      {Mode::kStopped, Event::kStart, Mode::kHolding},
      {Mode::kHolding, Event::kWaypointsAccepted, Mode::kRunning},
      {Mode::kRunning, Event::kWaypointsAccepted, Mode::kRunning},
      {Mode::kHolding, Event::kWaypointsAllSkipped, Mode::kHolding},
      {Mode::kRunning, Event::kWaypointsAllSkipped, Mode::kHolding},
      {Mode::kRunning, Event::kPlanEnded, Mode::kHolding},
      {Mode::kIdle, Event::kStop, Mode::kStopped},
      {Mode::kHolding, Event::kStop, Mode::kStopped},
      {Mode::kRunning, Event::kStop, Mode::kStopped},
      {Mode::kStopped, Event::kStop, Mode::kStopped},
      {Mode::kFault, Event::kStop, Mode::kFault},
      {Mode::kHolding, Event::kWatchdog, Mode::kStopped},
      {Mode::kRunning, Event::kWatchdog, Mode::kStopped},
      {Mode::kHolding, Event::kWaypointsRejected, Mode::kFault},
      {Mode::kRunning, Event::kWaypointsRejected, Mode::kFault},
      {Mode::kIdle, Event::kHome, Mode::kHoming},
      {Mode::kStopped, Event::kHome, Mode::kHoming},
      {Mode::kHoming, Event::kHomingDone, Mode::kIdle},
      {Mode::kHoming, Event::kHomingFailed, Mode::kFault},
  };
  for (const auto& r : rows) {
    StateMachine sm = inMode(r.from);
    const Transition t = sm.apply(r.event, "detail");
    EXPECT_TRUE(t.allowed) << modeName(r.from) << " event " << static_cast<int>(r.event);
    EXPECT_EQ(sm.mode(), r.to) << modeName(r.from) << " event " << static_cast<int>(r.event);
  }
  for (Mode m : {Mode::kIdle, Mode::kHolding, Mode::kRunning, Mode::kStopped, Mode::kHoming}) {
    for (Event e : {Event::kLoopFault, Event::kGripperFault}) {
      StateMachine sm = inMode(m);
      EXPECT_TRUE(sm.apply(e, "boom").allowed);
      EXPECT_EQ(sm.mode(), Mode::kFault);
      EXPECT_EQ(sm.detail(), "boom");
    }
  }
}

TEST(StateMachine, HeartbeatAndGetStateOkInEveryMode) {
  for (Mode m : kAll) {
    StateMachine sm = inMode(m);
    EXPECT_TRUE(sm.apply(Event::kHeartbeat).allowed);
    EXPECT_TRUE(sm.apply(Event::kGetState).allowed);
    EXPECT_EQ(sm.mode(), m);
  }
}

TEST(StateMachine, EveryRefusedRequestWithItsReason) {
  struct Refusal {
    Mode mode;
    Event event;
    const char* reason;
  };
  const std::vector<Refusal> refusals = {
      {Mode::kHolding, Event::kStart, "refused: mode is holding: already started"},
      {Mode::kRunning, Event::kStart, "refused: mode is running: already started"},
      {Mode::kFault, Event::kStart, "refused: mode is fault: libfranka: motion aborted by reflex! [cartesian_reflex]"},
      {Mode::kHolding, Event::kHome, "refused: mode is holding: send stop first"},
      {Mode::kRunning, Event::kHome, "refused: mode is running: send stop first"},
      {Mode::kFault, Event::kHome, "refused: mode is fault: libfranka: motion aborted by reflex! [cartesian_reflex]"},
      {Mode::kIdle, Event::kSetWaypoints, "refused: mode is idle: send start first"},
      {Mode::kStopped, Event::kSetWaypoints, "refused: mode is stopped: send start first"},
      {Mode::kFault, Event::kSetWaypoints,
       "refused: mode is fault: libfranka: motion aborted by reflex! [cartesian_reflex]"},
      {Mode::kHoming, Event::kSetWaypoints, "refused: mode is homing: homing in progress"},
  };
  for (const auto& r : refusals) {
    StateMachine sm = inMode(r.mode);
    const Transition t = sm.apply(r.event);
    EXPECT_FALSE(t.allowed);
    EXPECT_EQ(t.reason, r.reason);
    EXPECT_EQ(sm.mode(), r.mode);   // nothing changes
  }
  // stopped by the watchdog: the reason says so
  StateMachine sm = inMode(Mode::kHolding);
  sm.apply(Event::kWatchdog, "watchdog: no request for 0.512 s");
  EXPECT_EQ(sm.apply(Event::kSetWaypoints).reason,
            "refused: mode is stopped (watchdog: no request for 0.512 s): send start first");
  // the stop request is never refused
  for (Mode m : {Mode::kIdle, Mode::kHolding, Mode::kRunning, Mode::kStopped, Mode::kFault}) {
    StateMachine s = inMode(m);
    EXPECT_TRUE(s.apply(Event::kStop, "stop").allowed);
  }
}

TEST(StateMachine, AFaultIsNeverCleared) {
  StateMachine sm = inMode(Mode::kFault);
  for (Event e : {Event::kStart, Event::kStop, Event::kHome, Event::kHomingDone, Event::kWaypointsAccepted,
                  Event::kWatchdog, Event::kPlanEnded, Event::kLoopFault, Event::kGripperFault}) {
    sm.apply(e, "other");
    EXPECT_EQ(sm.mode(), Mode::kFault);
  }
  EXPECT_EQ(sm.detail(), "libfranka: motion aborted by reflex! [cartesian_reflex]");   // the first fault's text
}

}  // namespace
}  // namespace franka_control
