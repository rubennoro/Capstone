# franka_control/

The design of the C++ process that commands the Franka Panda. Written from PLAN.md sections 2, 4 (C2, C3), 5, 6.3, 7
and 9, and following contracts/STRUCTURE.md (the message definitions and wire format) and deploy/STRUCTURE.md (the
`launch.py` interface, common arguments, `station.yaml`, conventions §7). PLAN.md is the source of truth: where this
file and PLAN.md disagree, PLAN.md wins and this file is fixed.

## 1. Role

A C++17 process on libfranka. It is **the only thing connected to the Franka Panda**. It commands the arm and reports
the arm's state.

### Owns

- **The safety gate.** It sits in front of the control loop. Every incoming waypoint is checked (workspace box, tilt
  cone, gripper range, speed limits) before it is accepted, so the loop only ever sees waypoints that passed. A
  waypoint is accepted, shortened (slowed to the speed limit), pulled back (outside the box or cone by at most a small
  margin: moved onto the boundary), or rejected (a `fault`). Shortening and pulling back are normal operation
  (PLAN §5), counted in C2 `gate`.
- **The 1 kHz control loop.** It moves smoothly between accepted waypoints (linear position, slerp rotation) and
  computes Cartesian impedance torques. After the last waypoint it holds that pose. It never waits on messages.
- **The watchdog.** If `message_comms` stops sending requests for `watchdog.limit_s` (0.5 s), it stops the arm
  (`stopped`).
- **The state machine** (`idle` / `running` / `holding` / `stopped` / `homing` / `fault`), homing, the gripper, and
  reflex handling. A fault is never cleared automatically.
- **Reporting state** (C2): the hand pose at the TCP, the target, joint state, gripper width, mode, errors, gate
  counters and late loop cycles, stamped with PC time, ~100 times a second.
- **`FakeArm` and `FakeGripper`**, so everything else can run without the robot (`--mock`, unit tests).

### Doesn't own

Anything about images, the policy, relative poses, episodes or the keyboard (`message_comms`). Starting and stopping
processes (`deploy`).

### Talks to

`message_comms/` only:
- **Out:** C2 state broadcast (PUB, binds `endpoints.state`, ~100 Hz).
- **In:** C3 requests (REP, binds `endpoints.commands`): `set_waypoints`, `heartbeat`, `start`, `stop`, `home`,
  `get_state`; one reply each.

And the robot over FCI (Franka Control Interface, Franka's 1 kHz network interface to the arm, on the PC's direct
Ethernet link): the arm through `franka::Robot`, the hand through `franka::Gripper`.

### Terms used below

- **Torque control / impedance control:** the loop sends a torque for each of the 7 joints every millisecond. The
  torques make the hand behave like it is attached to the target pose by a spring (stiffness K) and a damper (D):
  pushed away, it springs back; there is no exact position command, so contact is soft.
- **Jacobian (J):** a 6×7 matrix that turns joint speeds into the hand's linear and angular speed. Its transpose
  turns a force/torque at the hand into joint torques (`τ = Jᵀ F`).
- **Nullspace:** the Panda has 7 joints but a pose has 6 numbers, so one "elbow" motion moves no part of the pose.
  The nullspace term gently holds that elbow motion near a chosen posture so it doesn't drift.
- **Coriolis torques:** torques caused by the arm's own motion; libfranka's `Model` gives them and we add them.
  Gravity is compensated by the robot itself.
- **Slerp:** "spherical linear interpolation", the constant-speed rotation between two orientations.
- **Reflex:** the robot's own safety stop (collision, too fast, too much torque). libfranka then throws an exception
  out of the control loop.
- **PREEMPT_RT, real-time:** a Linux kernel that lets a thread with real-time priority (`SCHED_FIFO`) run on time,
  every time. libfranka's `RealtimeConfig::kEnforce` requires it; `kIgnore` runs without it.
- **Wait-free:** an operation that always finishes in a fixed number of steps, whatever other threads do. The 1 kHz
  loop only uses wait-free operations to talk to other threads.
- **Sanitizers:** compiler options that make the test binary check itself while it runs: ASan (address sanitizer:
  out-of-bounds, use-after-free, leaks), UBSan (undefined behaviour: overflow, bad casts), TSan (thread sanitizer:
  data races).
- **TCP:** the gripper point the training poses describe. `F_T_TCP` is its pose in the flange frame (the flange is the
  arm's last link, where the hand is bolted on). **All poses in C2 and C3 are `O_T_TCP`: the TCP in the robot base
  frame** (PLAN §4).

## 2. File tree

One class per header/source pair (deploy §7.5). Interfaces are abstract classes (pure virtual). Small value types
share the header of the class that owns them.

```
franka_control/
  STRUCTURE.md
  launch.py                    the ManagedProcess subclass (section 11); Python, runs in the launcher
  config.yaml                  every setting, with its default (section 10)
  CMakeLists.txt               build (section 12)
  build.sh                     builds libfranka (if missing) and the binary into build/ (section 12.3)
  .gitignore                   build/, build-test/, build-tsan/
  include/franka_control/
    types.h                    value types: Vec7, Mat67, Pose, TrajectoryPoint, Target, ArmReading, TorqueCommand
    pose_math.h                free functions: flange <-> TCP, TCP Jacobian, rotation angle, rigid-transform check
    clock.h                    Clock (a function returning CLOCK_MONOTONIC ns), monotonicNowNs()
    log.h                      Log: one line per message to stderr, the Python format; never used in the 1 kHz callback
    command_line.h             CommandLine
    config.h                   Config and its section structs (plain data)
    config_loader.h            ConfigLoader
    contract_limits.h          ContractLimits
    messages.h                 msg::ArmState, msg::GateCounters, msg::SetWaypoints, msg::WaypointIssue,
                               msg::SetWaypointsReply, msg::Heartbeat, msg::Start, msg::Stop, msg::Home,
                               msg::GetState, msg::Ack, msg::ErrorReply, msg::Any (a std::variant of all of them)
    contract_error.h           ContractError
    codec.h                    Codec
    arm.h                      Arm (interface), ArmError
    arm_status.h               ArmStatusFlags, describeArmProblem() (port of derive_panda_error)
    libfranka_arm.h            LibfrankaArm            (only built with libfranka)
    fake_arm.h                 FakeArm
    homing_motion.h            HomingMotion
    gripper.h                  Gripper (interface), GripperError
    franka_hand_gripper.h      FrankaHandGripper       (only built with libfranka)
    fake_gripper.h             FakeGripper
    gripper_worker.h           GripperWorker
    safety_gate.h              SafetyGate, GateResult, GateIssue
    waypoint_trajectory.h      WaypointTrajectory
    impedance_controller.h     ImpedanceController, TorqueResult
    watchdog.h                 Watchdog
    state_machine.h            StateMachine, Mode, Event, Transition
    triple_buffer.h            TripleBuffer<T> (header-only template)
    shared_state.h             SharedState, LoopCommand, LoopSnapshot, LoopStats
    control_loop.h             ControlLoop
    supervisor.h               Supervisor
    command_server.h           CommandServer
    state_publisher.h          StatePublisher
    app.h                      App
  src/
    main.cpp                   parses the command line, runs App, turns exceptions into the FATAL line
    <one .cpp per header above, except types.h, clock.h, triple_buffer.h>
  tests/
    test_*.cpp                 GoogleTest (section 13)
    test_launch.py             pytest, for launch.py (deploy §9)
    support/                   test-only helpers: FakeClock, ReqClient (a cppzmq REQ socket), example-file reader
```

Two classes beyond PLAN §6.3's list are needed to run the threads: `ControlLoop` (the body of the control thread) and
`Supervisor` (the decisions the messaging thread makes: state machine, gate, watchdog). `HomingMotion`,
`GripperWorker`, `Codec`, `ConfigLoader` and `App` are small helpers with one job each.

## 3. Threads and `SharedState`

### 3.1 The threads

| Thread | Priority | Runs | Talks to the others only through |
|---|---|---|---|
| **control** | real-time (libfranka sets `SCHED_FIFO` inside `robot.control` with `realtime: enforce`) | `ControlLoop`: the libfranka 1 kHz torque callback, and homing moves | `SharedState` (wait-free parts only) |
| **messaging** | normal | `Supervisor`, `CommandServer` (C3), `StatePublisher` (C2) | `SharedState`; plus `Arm::requestStop()` (homing timeout only) |
| **gripper** | normal | `GripperWorker`: `franka::Gripper` calls (they block for the length of a finger move) | `SharedState` |
| **main** | normal | `App`: start-up, waits for SIGINT/SIGTERM, shutdown | everything, at start and end only |

The messaging thread repeats, at most every `messaging.poll_max_s` (5 ms):
`supervisor.tick()` → `commandServer.serviceOnce(timeout)` → `statePublisher.maybePublish()`. The timeout is the time
left until the next C2 message is due, so C2 keeps its rate while requests are served in between.

SIGINT and SIGTERM are blocked in every thread before the threads start. Only the main thread receives them (with
`sigwait`), so no signal handler runs inside libfranka or ZeroMQ.

### 3.2 `SharedState`: how the threads meet

The 1 kHz loop must never wait. A mutex shared with the messaging thread could make it wait (if the messaging thread
holds it and is descheduled, the loop misses its 1 ms deadline). So the loop only ever does single atomic operations.

**Design: triple buffers plus a few atomics.** A *triple buffer* is three copies of a struct and one atomic index.
One thread writes, one thread reads:

- The writer fills its own copy, then swaps it with the "middle" copy in one atomic exchange, marking it new.
- The reader, when it wants the newest data, swaps its own copy with the middle one in one atomic exchange (only if
  it is marked new), then reads its copy at leisure.

Both sides are wait-free (one atomic exchange each), the reader always gets the newest *complete* copy (never half old,
half new), and neither side ever retries.

Why not the alternatives:
- **Mutex:** the loop could wait (above).
- **Seqlock** (a version counter the reader checks before and after copying): the reader may have to retry, and the
  plain copy during a write is formally a data race in C++ (TSan reports it).
- **Atomic pointer swap to a newly allocated object:** needs memory allocation and freeing, which are not real-time
  safe and need a rule for when the old object may be freed. A triple buffer is the same idea with three fixed slots.

| Channel | Writer → reader | Type | What |
|---|---|---|---|
| commands | messaging → control | `TripleBuffer<LoopCommand>` | what the loop should do: hold, follow this plan, or home |
| snapshot | control → messaging | `TripleBuffer<LoopSnapshot>` | the newest arm reading, target, phase, fault, loop stats |
| `last_alive_ns` | messaging → control | `std::atomic<int64_t>` | time of the last C3 request (watchdog) |
| `gripper_target_m` | control → gripper | `std::atomic<double>` | the width the plan asks for now |
| `gripper_width_m` | gripper → messaging | `std::atomic<double>` | measured fingertip width |
| `gripper_fault` | gripper → messaging | `std::string` + `std::mutex` | gripper error text; neither thread is the loop, so a mutex is fine |
| `shutdown` | main → all | `std::atomic<bool>` | stop cleanly |

`std::atomic<double>` and `std::atomic<int64_t>` are checked with `static_assert(...::is_always_lock_free)` (true on
x86-64 and aarch64), so they are real atomic instructions, not hidden locks.

Sizes: a `LoopCommand` holds a whole plan (65 points, about 5 KB). The messaging thread writes it (a few times a
second); the loop never copies it, it reads its own slot in place. A `LoopSnapshot` is under 1 KB and is filled by
the loop in its own slot every cycle.

```
struct LoopCommand {                      // messaging -> control
  uint64_t generation;                    // +1 per posted command; the loop acts on each generation once
  enum class Kind { kHold, kFollow, kHome } kind;
  WaypointTrajectory plan;                // used with kFollow
};

struct LoopStats {                        // the last completed 1 s window, computed by the loop itself
  uint64_t cycles; int64_t period_max_ns; int64_t compute_max_ns;
  uint64_t missed_packets; uint64_t late_cycles; double success_rate_min;
};

struct LoopSnapshot {                     // control -> messaging
  int64_t t_ns;                           // CLOCK_MONOTONIC when this reading arrived (C2 t_ns)
  uint64_t cycle, session;                // cycle counter; torque session counter (+1 after each homing)
  enum class Phase { kStarting, kTorque, kHoming, kEnded } phase;
  Vec7 q, dq; Pose O_T_TCP, O_T_TCP_target; double gripper_target_m;
  uint64_t applied_generation;            // the command the loop is executing
  uint64_t watchdog_trip_generation;      // 0, or the generation the loop's backstop watchdog stopped (3.3)
  uint64_t homed_generation;              // the kHome generation whose move has finished
  bool fault; std::array<char, 256> error;   // fixed size: no allocation in the loop
  uint64_t late_cycles_total;             // late cycles since start, all sessions (C2 loop_late, 5.2)
  LoopStats stats;
};
```

`SharedState` methods (each is used by exactly one thread, named in the comment):

```
class SharedState {
 public:
  void postCommand(const LoopCommand& c);            // messaging
  const LoopCommand* takeCommand();                  // control: newest command, or nullptr if nothing new
  LoopSnapshot& snapshotSlot();                      // control: fill in place, then
  void publishSnapshot();                            // control
  const LoopSnapshot& latestSnapshot();              // messaging: newest complete snapshot
  void markAlive(int64_t now_ns);  int64_t lastAliveNs() const;
  void setGripperTarget(double w); double gripperTarget() const;
  void setGripperWidth(double w);  double gripperWidth() const;
  void setGripperFault(std::string text); std::optional<std::string> gripperFault() const;
  void requestShutdown();          bool shutdownRequested() const;
};
```

```
template <class T> class TripleBuffer {             // one writer thread, one reader thread
 public:
  T& writeSlot();                                    // writer
  void publish();                                    // writer: one atomic exchange
  bool refresh();                                    // reader: one atomic exchange if something new; true if new
  const T& readSlot() const;                         // reader
};
```

### 3.3 Rules that keep the loop real-time safe

- No memory allocation, no locks, no logging, no exceptions thrown on purpose inside the 1 kHz callback. Eigen
  matrices are fixed-size; strings are fixed `char` arrays.
- The loop never waits for the messaging thread: if no new command arrived, it continues with the one it has.
- The messaging thread owns every decision (the state machine). The loop only reports events (a fault, a finished
  homing, its backstop watchdog), and the messaging thread turns them into state changes on its next iteration
  (within ~5 ms).

## 4. The state machine

### 4.1 States

| C2 `mode` | Meaning | The loop is | Watchdog |
|---|---|---|---|
| `idle` | connected, holding the pose it had at start-up or after homing; not following waypoints | holding the target | off |
| `holding` | started; no waypoint ahead (none yet, or past the last one): holds the last target. The "slow" case. | following a plan whose end has passed | on |
| `running` | started; following waypoints that are still ahead | following a plan | on |
| `stopped` | stopped by `stop` or by the watchdog; holds the target it had at that moment | holding the target | off |
| `homing` | the blocking joint move to the home pose is running (7.2) | not running torque control (a joint motion runs) | paused |
| `fault` | a libfranka exception, a gate rejection, a gripper error, or a failed homing. Never cleared; rerun `deploy run`. | holding (gate/gripper fault) or gone (libfranka exception: the robot has stopped itself) | off |

All six are C2 `mode` values (PLAN §4 C2).

### 4.2 Transitions

| From | Trigger | To | Command posted to the loop |
|---|---|---|---|
| *(start-up)* | the loop has held the pose for `startup.hold_before_ready_s` | `idle` | `kHold` |
| `idle`, `stopped` | `start` | `holding` | `kFollow`, plan = [the current target, now] |
| `holding`, `running` | `set_waypoints`, ≥ 1 accepted | `running` | `kFollow`, the new plan |
| `holding`, `running` | `set_waypoints`, all skipped (times already passed): `accepted 0, skipped N`, not a fault | `holding` | `kFollow`, plan = [the current target, now] |
| `running` | now ≥ the plan's last waypoint time | `holding` | none (the loop already holds the last point) |
| `idle`, `holding`, `running` | `stop` | `stopped` | `kHold` |
| `holding`, `running` | watchdog: no C3 request for `watchdog.limit_s` (or the loop's backstop tripped) | `stopped` | `kHold` |
| `holding`, `running` | `set_waypoints` with any waypoint rejected by the gate | `fault` | `kHold` |
| `idle`, `stopped` | `home` | `homing` | `kHome` |
| `homing` | the move finished and the new torque session holds | `idle` | `kHold` |
| `homing` | the move failed, or `homing.timeout_s` passed | `fault` | none (timeout: `Arm::requestStop()`) |
| any but `fault` | the loop reports a fault (libfranka exception, non-finite torque) | `fault` | none (the loop has ended) |
| any but `fault` | the gripper reports an error | `fault` | `kHold` |
| `fault` | anything | `fault` | none |

Everything not in this table is refused (4.3) and changes nothing. After `stop`, the next episode's `start` is
accepted (`stopped` → `holding`). `message_comms` treats a `stopped` it did not ask for (the watchdog) as a conclude.

### 4.3 C3 requests per state

| Request | `idle` | `holding` | `running` | `stopped` | `fault` | Reply |
|---|---|---|---|---|---|---|
| `heartbeat` | ok | ok | ok | ok | ok | `Ack(ok=true)` |
| `get_state` | ok | ok | ok | ok | ok | `ArmState` (one C2 message) |
| `start` | → `holding` | **refused** ("already started") | **refused** ("already started") | → `holding` | **refused** | `Ack`; refused is `ok=false` with the reason |
| `stop` | → `stopped` | → `stopped` | → `stopped` | ok, no change | ok, stays `fault` | `Ack(ok=true)` (never refused: `message_comms` sends it on every conclude) |
| `home` | → `homing` | **refused** ("send stop first") | **refused** | → `homing` | **refused** | `Ack`, sent when the move ends (8.3) |
| `set_waypoints` | **refused** | gate | gate | **refused** | **refused** | `SetWaypointsReply`; when refused by the state, `ErrorReply` (`error/v1`), which is not a fault and changes nothing |

**`heartbeat` is accepted in every mode**, from the moment C3 answers (start-up step 8, 7.3) to shutdown. In `homing`
no request can arrive at all: the REP socket has not replied to `home` yet, and `message_comms`' REQ socket cannot
send again before it gets that reply (contracts §4.5).

A state refusal of `set_waypoints` uses `error/v1` because `SetWaypointsReply` has no "refused" field. It is not a
gate rejection: e.g. after a watchdog stop the state stays `stopped`. `message_comms` concludes on it (`RemoteError`).

Refusal reasons are one line, e.g. `refused: mode is fault: libfranka: motion aborted by reflex! [cartesian_reflex]`,
`refused: mode is running: send stop first`, `refused: mode is stopped (watchdog: no request for 0.512 s)`.

## 5. The 1 kHz loop

### 5.1 Sessions

libfranka's `robot.control(callback)` blocks its thread and calls `callback` every millisecond until the callback
returns `franka::MotionFinished`. The control thread (`ControlLoop::run`) repeats:

1. **Torque session:** `arm.runTorqueControl(step)`. At its first cycle the target is the measured TCP pose, so the
   arm holds still. It ends when the current command is `kHome` (a new generation) or `shutdown` is set, and the arm
   has settled (5.5).
2. If it ended for `kHome`: **homing move** (`arm.moveToJoints`, section 7.2), then back to 1 (a new session, holding
   the home pose; `session` +1).
3. If it ended for shutdown: return.

Any `ArmError` (libfranka exception) ends the thread: the snapshot gets `fault = true`, the error text, and
`phase = kEnded`. Nothing is retried and `automaticErrorRecovery` is never called.

### 5.2 One cycle (`ControlLoop::step`), in order

1. **Time.** `t_ns = clock_gettime(CLOCK_MONOTONIC)`, the first thing in the callback. This is C2's `t_ns` ("PC time
   the state was read") and the time the target is evaluated at. The robot's own clock (`RobotState::time`) is not
   used for anything but the missed-packet count.
2. **Read the state.** `LibfrankaArm` fills an `ArmReading` from libfranka's `RobotState` and `Model`: `q`, `dq`, the
   flange pose `O_T_F` (`model.pose(Frame::kFlange, state)`), the flange Jacobian (`model.zeroJacobian(Frame::kFlange,
   state)`), `model.coriolis(state)`, the last commanded torque `tau_J_d`, `control_command_success_rate`, and the
   number of missed packets (the callback's `franka::Duration` minus 1 ms). libfranka gives matrices column-major;
   `ArmReading` holds Eigen types, so the conversion happens once, here.
3. **New command?** `shared.takeCommand()`. A newer generation replaces the current one: `kHold` → hold the current
   target; `kFollow` → follow its plan; `kHome` → start ending the session.
4. **Backstop watchdog** (7.1): if following and `Watchdog::expired(t_ns, shared.lastAliveNs())` with the backstop
   limit, switch to holding the current target and set `watchdog_trip_generation` to the current generation. The loop
   does not follow that generation again; a newer command replaces it as usual.
5. **Target.**
   - Holding: the held target pose and width (at session start: the measured pose and `shared.gripperWidth()`).
   - Following: `plan.at(t_ns)` (5.3).
6. **Measured TCP.** `O_T_TCP = O_T_F · F_T_TCP`.
7. **Torque.** `ImpedanceController::compute(reading, target)` (5.4).
8. **Sanity.** If any torque is not finite: report a fault (`controller produced a non-finite torque`), return
   `MotionFinished` with zero torques (the robot keeps compensating gravity and holds).
9. **Report.** Fill the snapshot slot (time, `q`, `dq`, `O_T_TCP`, target, phase, generations, stats) and publish it.
   `shared.setGripperTarget(target width)`.
10. **Finish?** If ending (`kHome` or shutdown) and the arm has settled (5.5): return `MotionFinished(τ)`. Otherwise
    return `τ`.

Computation per cycle is a few tens of microseconds (two `Model` calls and 6×7 matrix products). The loop measures
its own period (between consecutive `t_ns`) and compute time; `LoopStats` keeps the worst of each 1 s window.

**Late cycles.** A cycle is *late* when the time since the previous cycle exceeds `loop_warning.late_period_ms`
(1.5 ms), or libfranka reports a missed packet (its `franka::Duration` argument is more than 1 ms). The loop counts
them in `late_cycles_total` (since start, over all sessions), which C2 sends as `loop_late` so `message_comms` can show
it on the status line. This matters mainly with `realtime: ignore` (PLAN §9). The messaging thread also writes a log
line every `messaging.stats_period_s` (1 s) with the window's stats, at WARNING level when the window had late cycles
or a success rate below `loop_warning.success_rate_min`.

### 5.3 The target from `WaypointTrajectory`

A plan is a list of points `(t_ns, position, orientation, gripper width)` with strictly increasing times. **Its first
point is always the "start point": the target at the moment the plan was made**, so a new plan begins exactly where
the old target was (no jump). Then come the accepted waypoints.

`at(t)`:
- **Before the first point** (can only happen for a fraction of a millisecond, if the loop's cycle began just before
  the plan was posted): the first point.
- **Between points i and i+1:** `a = (t − tᵢ) / (tᵢ₊₁ − tᵢ)`; position `pᵢ + a (pᵢ₊₁ − pᵢ)`; orientation
  `slerp(qᵢ, qᵢ₊₁, a)` along the shorter way (Eigen's `Quaterniond::slerp`); width linear like the position.
- **After the last point:** the last point (this is "holding").

Past and replaced waypoints are handled when the plan is built (8.2), not in the loop: waypoints whose time had passed
when they arrived are skipped (never part of a plan), and a new `set_waypoints` replaces every point after "now".
The new plan's start point is the old plan evaluated at the messaging thread's "now"; the loop picks the new plan up
within 1 ms, so the target can move by at most `2 · v_max · 1 ms` (0.5 mm) at the switch, which the impedance
spring absorbs.

### 5.4 The torque law (`ImpedanceController`)

From libfranka's `examples/cartesian_impedance_control.cpp`, with three additions (nullspace term, error clipping,
torque limits), all common in Franka controllers (franka_ros' `cartesian_impedance_example_controller` has the same
nullspace term).

With `p, R` the measured TCP position and rotation, `p_d, R_d` the target, `J` the TCP Jacobian, `q, dq` the joints:

1. **TCP Jacobian.** libfranka gives the flange Jacobian `J_F`. With `r = p_TCP − p_flange` (base frame), the TCP's
   linear rows are `J_lin = J_F,lin − [r]× J_F,ang`; the angular rows are unchanged. (`[r]×` is the matrix that
   computes `r × v`.) Doing this ourselves, rather than through libfranka's end-effector frame, keeps Desk's
   end-effector setting out of our kinematics, and the same function is unit-tested without libfranka.
2. **Position error** `e_p = p − p_d`, clipped to length `control.max_position_error_m`.
3. **Orientation error**, as in the libfranka example: quaternion `q_e = q⁻¹ · q_d` (after flipping `q` if
   `q · q_d < 0`), `e_o = −R · vec(q_e)` (vec = the x, y, z parts, ≈ half the rotation angle times the axis). Clipped
   so its angle is at most `control.max_rotation_error_rad`.
4. **Spring and damper at the TCP:** `F = −K e − D (J dq)` with `e = [e_p; e_o]`, `K = diag(stiffness_translational,
   stiffness_rotational)`, `D = 2 · damping_ratio · √K` per axis (the libfranka example's unit-mass convention).
5. **Task torque** `τ_task = Jᵀ F`.
6. **Nullspace** `τ_null = N (k_ns (q_ns − q) − 2√k_ns · dq)`, with `N = I − Jᵀ (J Jᵀ + λ² I)⁻¹ J`, `k_ns =
   control.nullspace_stiffness`, `λ = control.jacobian_pinv_damping`, and `q_ns` the joint angles at the start of the
   session (the start-up posture, or the home posture after homing).
7. **Sum:** `τ = τ_task + τ_null + coriolis`. Gravity is not added (the robot does it).
8. **Limits:** clamp each `|τᵢ| ≤ control.torque_max_nm[i]`, then limit the change from the last commanded torque
   (`tau_J_d`) to `control.torque_rate_max_nm_per_s · 1 ms`. libfranka's own rate limiter and torque low-pass filter
   also stay on (`control.libfranka_limit_rate`, `control.libfranka_cutoff_frequency_hz`).

```
struct TorqueResult { Vec7 tau; Pose O_T_TCP; double position_error_m; double rotation_error_rad; bool finite; };

class ImpedanceController {                       // control thread; pure, no allocation
 public:
  ImpedanceController(const ControlConfig& cfg, const Pose& F_T_TCP);
  void setNullspaceTarget(const Vec7& q_ns);      // at each session start
  TorqueResult compute(const ArmReading& r, const Pose& O_T_TCP_target) const;
};
```

### 5.5 Settling before a session ends

A torque session is ended (for homing or shutdown) only once the arm is still: the loop holds the current target and
waits until every `|dqᵢ| < control.settle_speed_rad_per_s`, or `control.settle_timeout_s` passes. Then it returns
`MotionFinished`. Homing is only allowed from `idle` and `stopped`, where the arm is already holding, so this is
normally immediate.

### 5.6 When libfranka throws

libfranka reports every problem as an exception out of `robot.control`: `ControlException` (a reflex; the message
lists the reflex reasons, e.g. `cartesian_reflex`), `NetworkException` (connection lost), `RealtimeException` (no
real-time priority), `CommandException`, `InvalidOperationException`. `LibfrankaArm` catches `franka::Exception`
(their common base) and throws `ArmError` with the text `libfranka: <what()>` and, for a `ControlException`, the
robot's error flags. The robot has already stopped itself; we do not touch it again. C2 keeps publishing at its rate
with `mode: fault`, `error: libfranka: ...`, and `seq` still increasing; each message repeats the last good reading, so
its `t_ns` stays unchanged (8.1). `message_comms` concludes (PLAN §5).

## 6. The safety gate

`SafetyGate::check` runs in the messaging thread on every `set_waypoints`, before anything reaches the loop.

### 6.1 Rules

Each waypoint `i` whose time has not passed (`t_ns > now`) is checked in this order. Waypoints with `t_ns ≤ now` are
**skipped**: not checked, not executed, counted in `skipped`.

| # | Rule | Result | Default |
|---|---|---|---|
| 1 | **Finite:** position, orientation and width are finite numbers (also checked by the C3 decoder; repeated here as a guard) | reject | |
| 2 | **Not too far ahead:** `t_ns ≤ now + safety.max_lead_s`. Catches a sender on the wrong clock (e.g. wall-clock ns). | reject | 5 s |
| 3 | **Gripper range:** `0 ≤ width ≤ gripper_width_max_m` (from `contracts/contracts/limits.yaml`). The policy already clips its widths into range (PLAN §5); this checks again. | reject | 0.08 m |
| 4 | **Workspace box** on the TCP position (base frame, boundary included), per axis: inside → ok; outside by at most `safety.box_margin_m` → **pulled back** onto the boundary (that coordinate set to the limit); further outside on any axis → reject | ok / pull back / reject | min `[0.25, -0.40, 0.05]`, max `[0.80, 0.40, 0.70]` m (main-sim `bridge.yaml`); margin 0.01 m |
| 5 | **Tilt cone:** the angle between the hand's approach axis (the flange z-axis, out through the fingers; flange pose = `O_T_TCP · F_T_TCP⁻¹`) and `cone_axis`. At most `cone_max_deg` → ok; at most `cone_max_deg + safety.cone_margin_deg` → **pulled back**: the orientation is turned back onto the cone boundary about the shortest axis (`z × cone_axis`, as main-sim's `safety.py`), the TCP position unchanged; beyond → reject | ok / pull back / reject | axis `[0, 0, -1]` (straight down, as at home), 60° (main-sim); margin 2° |
| 6 | **Speed (leaky bucket):** position and rotation each (6.2), on the pulled-back pose | ok / **shorten** | `v_max` 0.25 m/s, `w_max` 1.0 rad/s, slack 10 ms (main-sim) |

The cone is measured on the flange's approach axis, the direction the fingers point. With the Franka Hand `F_T_TCP`
(a rotation about z only) it is also the TCP's z axis; measuring it on the flange keeps the rule's meaning if the TCP
axes ever change.

Pulling back and shortening are normal operation (PLAN §5 "normal operation"): the waypoint is accepted, the reason is
logged, the reply's `shortened` count includes the shortened ones, and C2 `gate.pulled_back` / `gate.shortened` count
them. A waypoint can be both pulled back and shortened.

**If any waypoint is rejected, the whole request is refused:** nothing from it is followed, the state goes to `fault`
(PLAN §5: a waypoint the gate rejects concludes), and the loop holds its current target. Rules 1–5 are checked on
every non-skipped waypoint. The reply has `accepted: 0` and `shortened: 0`, and lists **every non-skipped waypoint in
`rejected`**: those that failed with their reason, the others with `not used: waypoint <i> of this request was
rejected`. So in every reply **`accepted + skipped + len(rejected) == N`** (a requirement of `message_comms`): each
waypoint is accounted for exactly once. A request whose waypoints are all past gives `accepted 0, skipped N`, no
rejections, and is not a fault.

Reasons are one line with numbers, e.g. `outside workspace box by 0.031 m on z (z = 0.019 m < min 0.050 m, margin
0.010 m)`, `tilt 65.4 deg from approach axis > 60.0 + 2.0 deg`, `pulled back onto box: z 0.046 -> 0.050 m`,
`gripper width 0.095 m outside [0, 0.080] m`, `time 7.20 s ahead > max_lead_s 5.00 s (is t_ns on CLOCK_MONOTONIC?)`,
`not finite`.

Box and cone are checked at the waypoints. Between two waypoints the loop moves on a straight line (so a path between
two points inside the box stays inside it) and a slerp (the tilt between two waypoints inside the cone can exceed the
larger of the two by a few hundredths of a degree at most, at the speed limit).

### 6.2 The speed limit (leaky bucket)

Waypoints have times, so the speed between two consecutive points is distance / time. The gate walks the accepted
waypoints in order, starting from the plan's start point (the current target at `now`), and keeps a **bucket**: the
distance it is still allowed to move.

For each waypoint, with `Δt` the time since the previous point (in s) and `d` the distance from the previous
(possibly shortened) point:

```
allowed  = min(leftover, v_max · slack_s) + v_max · Δt
if d > allowed:  move the waypoint along the line from the previous point so that d = allowed   (shortened)
leftover = allowed − min(d, allowed)
```

The same for rotation, with the rotation angle between the two orientations, `w_max`, and slerp for shortening. The
bucket starts full (`leftover = v_max · slack_s`) at each request.

What this gives:
- Over any time window `T`, the planned path moves at most `v_max · (T + slack_s)`, however the waypoints are spaced
  (5 ms, 50 ms or bursts). Within a segment the loop moves at constant speed, so the speed is at most `v_max` plus the
  small slack.
- The slack absorbs a few milliseconds of timing jitter (e.g. a first waypoint that arrives a little late), so normal
  waypoints are not shortened.
- **Shorten, never reject:** a too-fast chunk is executed slower. A shortened waypoint keeps its time; the next
  segment starts from the shortened point, so the arm catches up when the commanded motion slows down.

**Change from main-sim's `safety.py`:** there, the whole step was capped at `v_max · slack_s` (2.5 mm), which was
right for targets sent at 200 Hz and applied at once. Here a waypoint 50 ms later may move `v_max · (50 ms + slack)`,
because the loop spreads it over the 50 ms. With main-sim's formula the arm could never move faster than
2.5 mm / 50 ms = 0.05 m/s. The ported tests use the new numbers (section 13).

### 6.3 Interface

```
struct GateConfig { Eigen::Vector3d workspace_min_m, workspace_max_m, cone_axis; double box_margin_m,
                    cone_max_deg, cone_margin_deg, v_max_m_per_s, w_max_rad_per_s, slack_s, max_lead_s; };
struct GateIssue  { uint64_t index; std::string reason; };       // index: position in the request
struct GateResult {
  std::vector<TrajectoryPoint> accepted;   // in order, already pulled back / shortened; empty if any rejected
  uint64_t skipped;
  std::vector<GateIssue> shortened, pulled_back, rejected;   // pulled_back: for the C2 counter and the log
};

class SafetyGate {                         // messaging thread; pure, keeps no state between requests
 public:
  SafetyGate(const GateConfig& cfg, const Pose& F_T_TCP, double gripper_width_max_m);
  GateResult check(const TrajectoryPoint& start, const std::vector<TrajectoryPoint>& requested,
                   int64_t now_ns) const;
  bool insideBox(const Eigen::Vector3d& p_tcp) const;      // no margin; the after-homing check (7.2)
};
```

How each rule is tested: section 13, `test_safety_gate.cpp` (each main-sim case ported, plus the new rules).

## 7. Watchdog, homing, start-up, shutdown

### 7.1 Watchdog

- **The limit is the config key `watchdog.limit_s`** (in `franka_control/config.yaml`, default 0.5 s, overridable in
  `station.yaml`). `message_comms`' `launch.py` check reads the real value and requires it above its own longest gap
  between two requests outside homing, 0.42 s (`loop_ms` 0.02 s + its `commands_s` 0.2 s + its `health_s` 0.2 s).
  `ConfigLoader` refuses a value below 0.5 s, which keeps that margin.
- **Fed by every C3 request that decodes** (`heartbeat`, `set_waypoints`, `get_state`, ...), not only `heartbeat`:
  each one proves the manager is alive. `CommandServer` calls `shared.markAlive(now)`.
- **Armed only in `running` and `holding`.** In `idle` and `stopped` nothing moves, and in `fault` the arm is
  already stopped.
- **Primary check** (messaging thread, every iteration, ≤ 5 ms apart, and before handling each request):
  `now − last_alive > watchdog.limit_s` (0.5 s) → `stopped`, `kHold`. So the arm stops within 0.5 s + 5 ms.
- **Backstop check** (in the 1 kHz loop): the same rule with `limit_s + watchdog.backstop_extra_s` (0.6 s). It only
  matters if the messaging thread itself is stuck: the loop then holds by itself and reports
  `watchdog_trip_generation`; the messaging thread, when it runs again, goes to `stopped`. Normally the primary check
  acts 0.1 s earlier and the backstop never fires.
- **Paused during homing:** homing starts only from `idle` or `stopped`, where the watchdog is off, and during
  homing the torque loop (and so the backstop) is not running. When homing ends, `markAlive(now)` restarts the clock
  ("resumes as soon as homing ends", contracts §4.5).

```
class Watchdog {                                   // pure; used by both threads with their own limit
 public:
  explicit Watchdog(int64_t limit_ns);
  bool expired(int64_t now_ns, int64_t last_alive_ns) const;   // now − last_alive > limit
  int64_t ageNs(int64_t now_ns, int64_t last_alive_ns) const;
};
```

### 7.2 Homing

`home` asks for a blocking joint move to `homing.q_home_rad`.

1. Refused unless `idle` or `stopped` (4.3).
2. `HomingMotion` is built from the current `q` to `q_home`. If its duration is ≥ `homing.timeout_s`, the request is
   refused at once (`Ack(ok=false)`, "home move would take 34 s > homing.timeout_s 30 s: move the arm closer by
   hand"); nothing moves and the state does not change.
3. The Supervisor posts `kHome`. The loop settles, ends the torque session, and the control thread runs
   `arm.moveToJoints(motion, observe)`: libfranka's joint-position control (`robot.control` with a
   `franka::JointPositions` callback, the robot's internal joint impedance `control.joint_impedance`), returning
   `MotionFinished` at the goal. `observe` keeps the snapshot fresh (target = measured pose).
4. Then a new torque session starts holding the home pose, with `q_ns` = the home posture. The Supervisor sees
   `homed_generation` and the new `session`, and checks the measured TCP pose of the snapshot against the workspace
   box (`SafetyGate::insideBox`, no margin). Inside → `idle`, posts `kHold`, calls `markAlive(now)`, and sends
   `Ack(ok=true)`. Outside → `fault` ("home pose TCP (0.31, 0.00, 0.03) m is outside the workspace box: check
   homing.q_home_rad and the box"), posts `kHold`, and sends `Ack(ok=false)`. So a wrong home posture or box is caught
   the first time it is used.
5. **Timeout:** if step 4 has not happened `homing.timeout_s` after the request, the Supervisor calls
   `arm.requestStop()` (libfranka's `Robot::stop()`, which may be called from another thread and preempts the running
   motion), goes to `fault` ("homing: not finished after 30.0 s"), and replies `Ack(ok=false)`.
6. **Failure:** a libfranka exception during the move → `fault`, `Ack(ok=false, reason=<the error>)`.

`homing.timeout_s` (30 s) must stay below `message_comms`' `timeouts.home_s` (90 s), as contracts §4.5 requires, so
this process always answers `home` before `message_comms` gives up. `message_comms`' `launch.py` check compares the two
real values; this process does not hard-code the other side's number.

**`HomingMotion`** is a minimum-jerk profile: `q(s) = q₀ + Δq (10s³ − 15s⁴ + 6s⁵)`, `s = t / T`. Speed and acceleration
are zero at both ends (libfranka rejects joint motions that start or end abruptly). Its peak speed is
`1.875 |Δq| / T`, so `T = max(homing.min_duration_s, maxᵢ 1.875 |Δqᵢ| / homing.max_joint_speed_rad_per_s)`. All joints
start and finish together. The gripper is not moved by homing.

```
class HomingMotion {                               // pure
 public:
  HomingMotion(const Vec7& q_start, const Vec7& q_goal, double max_joint_speed_rad_per_s, double min_duration_s);
  double durationS() const;
  Vec7 positionAt(double t_s) const;               // t clamped to [0, T]
  Vec7 velocityAt(double t_s) const;
};
```

### 7.3 Start-up (`App::run`)

Each step that fails ends the process with a FATAL line (deploy §7.6) and exit code 1.

1. Parse the command line; load `config.yaml` merged with `station.yaml`'s `franka_control` section, plus
   `endpoints.state`, `endpoints.commands`, `session_root` (all required in the station) and `contracts/contracts/limits.yaml`
   (section 10). Every problem is listed at once.
   `FATAL franka_control: station.yaml: franka_control.safety.v_max_m_per_s: must be > 0: fix station.yaml`.
2. Session folder: `--session DIR` → `DIR/franka_control/`; alone → `<session_root>/<YYYY-MM-DD_HH-MM-SS>_franka_control_alone/franka_control/`, the same layout as in a run
   (deploy §4.2).
   Writes `config_effective.yaml` (the merged settings) there. That is its only output; everything else goes to the
   log (stderr).
3. `realtime: enforce`: `mlockall(MCL_CURRENT | MCL_FUTURE)` so the loop's memory is never paged out (a failure is a
   warning). Block SIGINT/SIGTERM.
4. **Arm.** `--mock` → `FakeArm`. Otherwise `LibfrankaArm`:
   - connect: `franka::Robot(robot_ip, kEnforce | kIgnore)`. A version mismatch with the robot
     (`IncompatibleVersionException`) → `FATAL ...: libfranka 0.9.2 does not match the robot's system version: check
     Franka's compatibility table, set libfranka_version in config.yaml, rerun build.sh`.
   - check the robot is ready: `readOnce()`, then `describeArmProblem()` on its mode and error flags. A reflex,
     error flags or a pressed user stop → `FATAL ...: arm reports reflex (current_errors: [...]): press and release
     the user stop, check Desk, then rerun deploy run`. franka_control never calls `automaticErrorRecovery()`, not
     even at start-up: clearing a reflex is always the operator's act (the user stop), and rerunning is how to try again.
   - `setCollisionBehavior(torque thresholds ×2, force thresholds ×2)` (the 4-array form, lower = upper),
     `setJointImpedance(control.joint_impedance)`, `setLoad(...)` when `load.mass_kg > 0`, `loadModel()`.
5. **Gripper.** `--mock` → `FakeGripper`. Otherwise `FrankaHandGripper(robot_ip)`, `prepare()` (homes the fingers when
   `gripper.homing_on_start`). Read the width; set `gripper_target_m` to it, so nothing moves at start.
6. **Control thread** starts (optionally pinned to `control_cpu`). The first torque session holds the measured pose.
   Wait until it has held for `startup.hold_before_ready_s` (`phase == kTorque`), at most `startup.loop_start_timeout_s`.
   A `RealtimeException` here → `FATAL ...: real-time scheduling unavailable: use a PREEMPT_RT kernel with rtprio 99,
   or set franka_control.realtime: ignore in station.yaml`.
7. **Sockets:** bind C2 PUB and C3 REP (a busy port → FATAL naming it). Start the messaging thread and the gripper
   thread. The Supervisor starts in `idle` and posts `kHold`.
8. After the first C2 message is published, create the ready file (`--ready-file`). From here on C3 is answered.

### 7.4 Shutdown (SIGINT or SIGTERM)

1. The main thread sets `shutdown`.
2. **Control thread:** a torque session holds the current target, settles (5.5) and returns `MotionFinished`; then the
   thread ends. If a homing move is running, the main thread calls `arm.requestStop()` (the move stops; the resulting
   `ArmError` is expected and not a fault).
3. The main thread joins the control thread (at most `shutdown.timeout_s`, 3 s; if it is still running:
   `arm.requestStop()`, join again, and a warning).
4. Then the messaging thread stops (so C2 kept reporting during the settle) and closes its sockets with linger 0; a
   pending `home` reply is dropped. The gripper thread calls `gripper.stop()` and ends.
5. Exit code 0, no FATAL line (deploy §7.6). A fault that happened earlier was logged at ERROR when it happened.

`shutdown.timeout_s` (3 s) stays below the launcher's `stop_timeout_s` (5 s), after which the launcher would SIGKILL.
Even then the arm is safe: when the FCI connection drops, the robot stops itself.

An unexpected exception in the messaging or gripper thread (a bug, a ZeroMQ error) triggers the same shutdown, then a
FATAL line and exit 1, because without C3/C2 the process can't do its job.

## 8. Messaging (C2, C3)

Both follow contracts/STRUCTURE.md §4.4–4.5 and the canonical msgpack form of §5 exactly. The C++ codec is proven
against the committed example files (section 13).

### 8.1 C2: `StatePublisher`

- PUB socket, binds `endpoints.state` (from `station.yaml`), send high-water mark
  `messaging.state_send_hwm`; sends with `ZMQ_DONTWAIT` (never blocks; a full queue drops the message).
- Every `1 / messaging.publish_rate_hz` (10 ms), from the newest `LoopSnapshot`:

| Field | From |
|---|---|
| `t_ns` | `snapshot.t_ns` (when that reading arrived in the loop) |
| `seq` | +1 per broadcast. A `get_state` reply carries the seq of the latest broadcast (not +1), so subscribers never see a false gap. |
| `O_T_TCP`, `O_T_TCP_target` | `snapshot` poses, row-major; rotations re-normalised (through a unit quaternion) before encoding so they pass the 1e-6 rigid check |
| `q`, `dq` | `snapshot` |
| `gripper_width_m` | `shared.gripperWidth()`, at least 0 |
| `mode` | `StateMachine`: `idle` / `running` / `holding` / `stopped` / `homing` / `fault` |
| `error` | the fault text in `fault`, else `""` |
| `gate` | Supervisor's counters since start: waypoints accepted (incl. shortened and pulled back) / shortened / pulled back to the box or cone / rejected |
| `loop_late` | `snapshot.late_cycles_total`: late 1 kHz cycles since start (5.2) |

- **C2 never pauses.** A message goes out every period with `seq` +1, even when the loop has no new reading: after a
  libfranka fault (the loop has ended) and in the few milliseconds between a torque session and a homing move. Then
  the message repeats the last reading, `t_ns` included, unchanged (`message_comms` accepts an equal `t_ns`). During
  the homing move itself the readings keep coming from `moveToJoints`' observer, so `t_ns` advances as usual.
- A snapshot that would break the C2 contract (e.g. a non-finite joint value from the arm) is not sent as it is: the
  Supervisor goes to `fault` ("arm state not finite") and C2 carries the last valid values with that error.

```
class StatePublisher {                             // messaging thread
 public:
  StatePublisher(zmq::context_t& ctx, const std::string& endpoint, int send_hwm, double rate_hz,
                 const Supervisor& supervisor, const Codec& codec, Clock clock);
  std::string endpoint() const;                    // the bound address (tests bind tcp://127.0.0.1:*)
  bool maybePublish();                             // publishes if due; true if it sent
  int64_t nsUntilNext() const;
  uint64_t lastSeq() const;
};
```

### 8.2 C3: `CommandServer` and `Supervisor`

`CommandServer::serviceOnce(timeout)`:
1. If a `home` reply is pending: ask `supervisor.pollHome()`; send it if it is ready. Do not receive (a REP socket
   must reply first).
2. Otherwise poll for a request up to `timeout`. Decode it with `Codec` (contracts §5 check order). A request that
   breaks its contract, has an unknown schema or version, or is not a C3 request → `ErrorReply` naming the field and
   problem (e.g. `c3.set_waypoints/v1: field 'O_T_TCP': value: not a rigid transform at [3]`). The server keeps
   running; concluding is `message_comms`' job (contracts §4.1, §7).
3. `shared.markAlive(now)` (watchdog, 7.1), after the Supervisor's own watchdog check for this request.
4. Dispatch to the Supervisor; encode its reply; send.

`set_waypoints` in `running` or `holding` (`Supervisor::onSetWaypoints`):
1. `now = clock()`. Convert the request: each 4×4 row-major matrix → position + unit quaternion.
2. `start` = the current plan at `now` (or the held target, if holding with a one-point plan).
3. `gate.check(start, waypoints, now)`.
4. Any rejected → `fault`, post `kHold`, add the failing waypoints to `gate.rejected`, reply with every non-skipped
   waypoint in `rejected` (6.1).
5. Else build the new plan: `[start] + accepted` (this **replaces every not-yet-reached waypoint** of the old plan),
   post `kFollow`, state → `running` (or `holding` if all were skipped: `accepted 0, skipped N`, not a fault), add to
   `gate.accepted`, `gate.shortened`, `gate.pulled_back`.
6. Reply `SetWaypointsReply` with its fields in the contract's order: `chunk_id` (echoed and logged), `accepted`,
   `shortened` (a uint64 count; the reasons go to the log), `skipped`, `rejected` (the list of `WaypointIssue`).
   Before sending, the Supervisor checks `accepted + skipped + len(rejected) == N` (a broken count is a bug: it
   throws, and the request gets `ErrorReply`). Pulled-back waypoints are counted in `accepted` and in C2
   `gate.pulled_back`; their reasons go to the log.

```
class Supervisor {                                 // messaging thread: owns StateMachine, SafetyGate, the plan
 public:
  Supervisor(SharedState& shared, Arm& arm, const Config& cfg, const ContractLimits& limits, Clock clock);
  void tick();                                     // loop/gripper events, watchdog, plan end, homing progress
  bool watchdogExpiredNow();                       // primary check; changes state if expired
  std::variant<msg::SetWaypointsReply, msg::ErrorReply> onSetWaypoints(const msg::SetWaypoints& m);
  msg::Ack onHeartbeat();
  msg::Ack onStart();
  msg::Ack onStop();
  std::optional<msg::Ack> onHome();                // nullopt: homing started, reply later
  std::optional<msg::Ack> pollHome();              // the deferred reply, once homing ended (ok, failed, timeout)
  msg::ArmState stateMessage(uint64_t seq) const;  // C2 content (also the get_state reply)
  Mode mode() const;
};

class CommandServer {                              // messaging thread
 public:
  CommandServer(zmq::context_t& ctx, const std::string& endpoint, Supervisor& supervisor,
                const StatePublisher& publisher, const Codec& codec, SharedState& shared, Clock clock);
  std::string endpoint() const;
  void serviceOnce(int64_t timeout_ns);
};
```

### 8.3 Replies

| Request | Reply |
|---|---|
| `c3.heartbeat/v1` | `ack/v1` `{ok: true, reason: ""}` |
| `c3.start/v1` | `ack/v1`, ok or refused with the reason |
| `c3.stop/v1` | `ack/v1` `{ok: true}` |
| `c3.home/v1` | `ack/v1`, sent when the home pose is reached (ok), or on refusal, failure or timeout (not ok) |
| `c3.get_state/v1` | `c2.state/v1` |
| `c3.set_waypoints/v1` | `c3.set_waypoints.reply/v1`; `error/v1` when refused by the state (4.3; not a fault) |
| anything broken or unknown | `error/v1` |

### 8.4 The codec

```
class Codec {                                      // pure; contracts/STRUCTURE.md §5, byte for byte
 public:
  explicit Codec(const ContractLimits& limits);    // max_waypoints, rotation_tol
  msg::Any decode(std::string_view bytes) const;   // throws ContractError (schema, field, problem, detail)
  std::string encode(const msg::Any& m) const;     // canonical form; validates first (a bug throws, never sends)
};
```

- msgpack-cxx (`msgpack::unpack`; trailing bytes → `not_msgpack`). Checks in the contract's fixed order:
  `not_msgpack`, `not_a_map`, `no_schema`, `unknown_schema`, `schema_version`, `missing`, `unknown_field`, then each
  field in table order (`type`, `dtype`, `shape`, `value`), then the message's own checks (`value`). A valid message
  that is not a C3 request, sent to the server, is `unexpected_message`.
- Arrays: the map `{"__nd__": true, "dtype", "shape", "data"}` with `data` little-endian, C order. 4×4 matrices are
  **row-major** on the wire; libfranka's are column-major, and Eigen's default is column-major, so conversions are
  explicit and tested (a transposed pose fails the rigid check).
- Encoding: map keys in table order, `schema` first; integers in the smallest form (msgpack-cxx `pack_int64` /
  `pack_uint64`, not the `pack_fix_*` forms); floats as float64; strings as str; bytes as bin.
- `msg::` structs are wire-shaped: matrices as `std::array<double, 16>` row-major, vectors as `std::array` or
  `std::vector`. The Supervisor converts to Eigen.

## 9. The classes

| Class | Thread | Job |
|---|---|---|
| `Arm` (interface) | control (`requestStop`: any) | read state, run the 1 kHz torque loop, homing move, stop |
| `LibfrankaArm` | control | the real Panda through `franka::Robot` + `franka::Model` |
| `FakeArm` | control | a simulated arm for `--mock` and tests |
| `Gripper` (interface) | gripper (`stop`: any) | move to a width, read the width |
| `FrankaHandGripper` | gripper | the Franka Hand with our printed fingers, through `franka::Gripper` |
| `FakeGripper` | gripper | simulated fingers |
| `GripperWorker` | gripper | the gripper thread: follows `gripper_target_m` with a deadband |
| `SafetyGate` | messaging | section 6 |
| `WaypointTrajectory` | messaging (builds), control (reads its own copy) | section 5.3 |
| `ImpedanceController` | control | section 5.4 |
| `HomingMotion` | control (and messaging, for the duration check) | section 7.2 |
| `Watchdog` | messaging (primary), control (backstop) | section 7.1 |
| `StateMachine` | messaging | section 4 |
| `SharedState`, `TripleBuffer<T>` | all | section 3.2 |
| `ControlLoop` | control | sections 5.1–5.2 |
| `Supervisor` | messaging | section 8.2 |
| `CommandServer` | messaging | C3 |
| `StatePublisher` | messaging | C2 |
| `Codec`, `ContractError` | messaging | section 8.4 |
| `ContractLimits` | main | reads `contracts/contracts/limits.yaml` |
| `Config`, `ConfigLoader` | main | section 10 |
| `CommandLine` | main | the four common arguments, plus `--version` and `--check-config` |
| `App` | main | sections 7.3–7.4 |
| `Log` | all but the 1 kHz callback | stderr lines |

### 9.1 Value types (`types.h`)

```
using Vec7 = Eigen::Matrix<double, 7, 1>;
using Mat67 = Eigen::Matrix<double, 6, 7>;
using Pose = Eigen::Isometry3d;

struct TrajectoryPoint { int64_t t_ns; Eigen::Vector3d p; Eigen::Quaterniond q; double width_m; };
struct Target          { Pose O_T_TCP; double width_m; };
struct ArmReading {                                   // one cycle's input, independent of libfranka
  Vec7 q, dq, coriolis, tau_J_d;                      // tau_J_d: the last commanded torque
  Pose O_T_F;                                         // flange pose, base frame
  Mat67 J_F;                                          // flange zero Jacobian, base frame ([linear; angular])
  double success_rate;                                // control_command_success_rate (FakeArm: 1.0)
  uint32_t missed_packets;
};
struct TorqueCommand { Vec7 tau; bool finished; };
```

### 9.2 Arm

```
struct ArmError : std::runtime_error {
  enum class Kind { kReflex, kNetwork, kRealtime, kCommand, kStopRequested, kOther } kind;
};

class Arm {
 public:
  using TorqueStep = std::function<TorqueCommand(const ArmReading&)>;
  using Observer = std::function<void(const ArmReading&)>;
  virtual ~Arm() = default;
  virtual ArmReading readOnce() = 0;                                   // start-up only, outside control
  virtual ArmStatusFlags status() = 0;                                 // robot mode + error flags (start-up check)
  virtual void runTorqueControl(const TorqueStep& step) = 0;           // blocks; step every 1 ms until finished
  virtual void moveToJoints(const HomingMotion& m, const Observer& observe) = 0;   // blocks
  virtual void requestStop() = 0;                                      // any thread; the running call throws kStopRequested
  virtual std::string describe() const = 0;                            // for the log and deploy check
};
// All methods but requestStop/describe throw ArmError.

class LibfrankaArm : public Arm {
 public:
  LibfrankaArm(const std::string& robot_ip, RealtimeMode mode, const ControlConfig& control, const LoadConfig& load);
};

class FakeArm : public Arm {
 public:
  FakeArm(const FakeArmConfig& cfg, const Pose& F_T_TCP, const Vec7& q_home, Clock clock);
  void injectFault(const std::string& text);       // tests: the next cycle throws ArmError(kReflex, text)
  void stallHoming(bool on);                       // tests: moveToJoints never finishes (homing timeout)
};
```

**`FakeArm`** is a floating rigid body that the real controller pushes. It reports a constant Jacobian `J_F = [I₆ | 0]`
(joints 1–6 move the flange directly along x, y, z and about x, y, z; joint 7 does nothing), zero Coriolis, and
`tau_J_d` = the last torque. Each 1 ms (timed with `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`) it takes the
force `τ₁..₃` and moment `τ₄..₆`, and integrates a mass `fake_arm.mass_kg` and inertia `fake_arm.inertia_kg_m2`
(semi-implicit Euler). With the default gains and unit mass/inertia this is critically damped, so the fake TCP follows
the plan with a small lag, like the real arm. `q` is the home posture plus the integrated joint speeds (meaningless
but finite). `moveToJoints` runs the `HomingMotion` in real time and moves the pose along the same profile to
`fake_arm.home_O_T_TCP`. It starts at that pose.

`describeArmProblem(const ArmStatusFlags&)` (`arm_status.h`) is the part of main-sim's `derive_panda_error` that still
applies: the robot mode (`kReflex`, `kUserStopped`, ...) and `current_errors` → one line, or `""` when the arm can be
used. ("`[]`" means no errors.) The `last_motion_errors` workaround is not needed: errors now arrive as exceptions.

### 9.3 Gripper

```
class Gripper {
 public:
  virtual ~Gripper() = default;
  virtual void prepare() = 0;                      // start-up: connect, optional finger homing
  virtual double widthM() = 0;                     // fingertip opening now
  virtual void moveTo(double width_m) = 0;         // blocks until reached, or stopped by an object
  virtual void stop() = 0;                         // any thread: ends a running moveTo
};
// All but stop throw GripperError.

class FrankaHandGripper : public Gripper {
 public:
  FrankaHandGripper(const std::string& robot_ip, const GripperConfig& cfg, double width_max_m);
};
class FakeGripper : public Gripper {
 public:
  FakeGripper(const FakeGripperConfig& cfg, std::function<void(int64_t)> sleep_ns);   // tests pass a no-op sleep
};

class GripperWorker {                              // gripper thread
 public:
  GripperWorker(Gripper& g, SharedState& shared, const GripperConfig& cfg, Clock clock);
  void run();                                      // until shutdown or an error
  void stepOnce();                                 // one iteration (tests)
};
```

- **`GripperWorker`**, every `gripper.poll_period_s` (20 ms): read `gripper_target_m`. If it differs from the last
  sent width by more than `gripper.deadband_m` (5 mm), call `moveTo(target)` (blocks while the fingers move). Then
  `widthM()` → `shared.setGripperWidth`. A `GripperError` → `shared.setGripperFault(text)` and the thread ends; the
  Supervisor goes to `fault`.
- **The width in C2 is stale while a move runs.** `moveTo` blocks for the whole finger move (up to ~1.6 s for 8 cm at
  0.05 m/s), and the width is read only after it returns, so C2 `gripper_width_m` shows the width from before the move
  until the move ends. The next target is also sent only then.
- **`FrankaHandGripper::moveTo(w)`**: hand width = `w − gripper.width_offset_m` (fingertips vs the hand's own
  reading; 0 until measured), clamped to `[0, min(width_max_m, GripperState.max_width)]`. **Closing** (narrower than now
  by more than the deadband): `grasp(width, speed, force, ε_inner = width, ε_outer = max)`, so it holds with
  `grasp_force_n` wherever an object stops the fingers. **Opening:** `move(width, speed)`. Force, speed, deadband and
  finger homing at start-up are config values (`gripper:` in section 10).
- **`FakeGripper`**: the width moves toward the target at `fake_gripper.speed_m_per_s`, and stops at
  `fake_gripper.object_width_m` when closing past it.

## 10. `config.yaml`

Every setting with its default. `station.yaml`'s `franka_control:` section may override any key (deploy §6.3: maps
merge key by key, other values are replaced, an unknown key is an error with its path). The binary reads both files
itself with yaml-cpp (deploy §7.5) and applies the same rule. Units are in the key names.

Top level:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `binary` | path | `build/franka_control` | the built program (used by `launch.py`) |
| `robot_ip` | str | `172.16.0.2` | the arm's FCI address (also the hand's) on the direct Ethernet link |
| `realtime` | `enforce` \| `ignore` | `enforce` | libfranka `RealtimeConfig` (PLAN §9 backup) |
| `libfranka_version` | str | `"0.9.2"` | **the one build setting** for libfranka: `build.sh` and CMake read it; the binary reports what it was built with; a mismatch fails `deploy check` and start-up. Not meant to be overridden in `station.yaml`. |
| `control_cpu` | int \| null | `null` | pin the control thread to this CPU core (optional) |

`control:` (the 1 kHz loop, section 5)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `stiffness_translational_n_per_m` | 3 floats | `[800, 800, 800]` | K for x, y, z (main-sim `bridge.yaml`; tune 600–1000 on the arm) |
| `stiffness_rotational_nm_per_rad` | 3 floats | `[40, 40, 40]` | K for rotation (tune 30–50) |
| `damping_ratio` | float | `1.0` | D = 2·ratio·√K |
| `nullspace_stiffness` | float | `0.5` | k_ns, pulls the elbow toward the session's start posture |
| `jacobian_pinv_damping` | float | `0.2` | λ in the nullspace projector (avoids blow-up near singular poses) |
| `max_position_error_m` | float | `0.03` | position error used by the spring is clipped to this (≤ 24 N at 800 N/m) |
| `max_rotation_error_rad` | float | `0.2` | rotation error clipped to this angle |
| `torque_max_nm` | 7 floats | `[87, 87, 87, 87, 12, 12, 12]` | per-joint torque clamp (the Panda's limits) |
| `torque_rate_max_nm_per_s` | float | `1000` | per-joint torque change limit (libfranka's `kMaxTorqueRate`) |
| `libfranka_limit_rate` | bool | `true` | libfranka's own rate limiter (`robot.control(..., limit_rate)`) |
| `libfranka_cutoff_frequency_hz` | float | `100.0` | libfranka's torque low-pass filter (`kDefaultCutoffFrequency`) |
| `collision_torque_thresholds_nm` | 7 floats | `[20, 20, 18, 18, 16, 14, 12]` | `setCollisionBehavior` joint torque thresholds (libfranka examples' defaults) |
| `collision_force_thresholds_n` | 6 floats | `[20, 20, 20, 25, 25, 25]` | `setCollisionBehavior` Cartesian force/torque thresholds |
| `joint_impedance` | 7 floats | `[3000, 3000, 3000, 2500, 2500, 2000, 2000]` | the robot's internal joint controller during homing (`setJointImpedance`) |
| `settle_speed_rad_per_s` | float | `0.02` | "still" for ending a session (5.5) |
| `settle_timeout_s` | float | `1.0` | end the session anyway after this |

`load:` (payload beyond the hand configured in Desk: printed fingers, Pi 5, cameras; PLAN §8 phase 4)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mass_kg` | float | `0.0` | 0 = no `setLoad` call. `# TODO: weigh the rig`. `deploy check` warns while 0. |
| `com_m` | 3 floats | `[0, 0, 0]` | centre of mass in the flange frame |
| `inertia_kg_m2` | 3×3 floats | zeros | inertia about the centre of mass |

`tcp:` (PLAN §6.6)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `F_T_TCP` | 4×4 floats, row-major | rotation −45° about the flange z axis; translation `[0, 0, 0.150]` | flange → TCP, a fixed constant set by the hardware: the Franka Hand's TCP axes, with our printed fingertips 150 mm out from the flange plate along its z axis (stock fingertips are at 0.1034 m). Typed in here only. Must be a rigid transform within `rotation_tol`. |

franka_control only uses `F_T_TCP`. Its agreement with `deployment/gripper.yaml` and the rig file's `T_cam_tcp` is
checked by `deploy check` and the geometry tool in `deploy/` (PLAN §6.6), not here.

`safety:` (section 6; defaults from main-sim `bridge.yaml`; `# TODO: measure around the table`)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `workspace_min_m` | 3 floats | `[0.25, -0.40, 0.05]` | box, TCP position, base frame |
| `workspace_max_m` | 3 floats | `[0.80, 0.40, 0.70]` | |
| `cone_axis` | 3 floats | `[0, 0, -1]` | the hand's approach axis at home (normalised by the loader) |
| `cone_max_deg` | float | `60.0` | max tilt from `cone_axis` |
| `box_margin_m` | float | `0.01` | outside the box by at most this → pulled back onto it; further → rejected |
| `cone_margin_deg` | float | `2.0` | tilt beyond `cone_max_deg` by at most this → pulled back; further → rejected |
| `v_max_m_per_s` | float | `0.25` | TCP speed limit (raise to 0.5 only after tuning on the arm) |
| `w_max_rad_per_s` | float | `1.0` | rotation speed limit |
| `slack_s` | float | `0.010` | leaky-bucket slack |
| `max_lead_s` | float | `5.0` | a waypoint further ahead than this is rejected |

`watchdog:`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `limit_s` | float, ≥ 0.5 | `0.5` | no C3 request for this long → `stopped` (PLAN §4 C3). The key `message_comms` cross-checks (7.1). |
| `backstop_extra_s` | float | `0.1` | the loop's own check fires at `limit_s` + this |

`homing:`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `q_home_rad` | 7 floats | `[0, -0.7853982, 0, -2.3561945, 0, 1.5707963, 0.7853982]` | Franka's "ready" posture. After every homing the measured TCP is checked against the workspace box; outside → `fault` (7.2) |
| `max_joint_speed_rad_per_s` | float | `0.3` | peak joint speed of the move |
| `min_duration_s` | float | `2.0` | shortest move |
| `timeout_s` | float, > 0 | `30.0` | homing's own timeout (contracts §4.5); `message_comms`' launch check requires it below its `timeouts.home_s` (90 s) |

`gripper:` (max opening `gripper_width_max_m` = 0.08 m comes from `contracts/contracts/limits.yaml`, with its TODO)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `speed_m_per_s` | float | `0.05` | finger speed for `move` and `grasp` |
| `grasp_force_n` | float | `20.0` | grasp force |
| `deadband_m` | float | `0.005` | smaller target changes are not sent |
| `poll_period_s` | float | `0.02` | gripper thread period |
| `width_offset_m` | float | `0.0` | fingertip opening − the hand's reported width. `# TODO: measure with the printed fingers` |
| `homing_on_start` | bool | `true` | `franka::Gripper::homing()` at start-up (the fingers open and close once) |

`messaging:`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `publish_rate_hz` | float | `100.0` | C2 rate |
| `state_send_hwm` | int | `100` | C2 send queue (messages); older ones are dropped when full |
| `poll_max_s` | float | `0.005` | longest messaging-thread iteration (watchdog and event resolution) |
| `stats_period_s` | float | `1.0` | loop-timing log line period |

`startup:`, `shutdown:`, `loop_warning:`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `startup.hold_before_ready_s` | float | `0.2` | the loop must hold this long before ready |
| `startup.loop_start_timeout_s` | float | `10.0` | waiting for the first torque session |
| `shutdown.timeout_s` | float | `3.0` | settle + join budget; below the launcher's `stop_timeout_s` |
| `loop_warning.success_rate_min` | float | `0.95` | warn when `control_command_success_rate` drops below this in a window |
| `loop_warning.late_period_ms` | float | `1.5` | a cycle more than this after the previous one is late (counted in C2 `loop_late`, 5.2) |

`fake_arm:`, `fake_gripper:` (only with `--mock` and in tests)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `fake_arm.home_O_T_TCP` | 4×4 floats | translation `[0.45, 0, 0.45]`, rotation `diag(1, -1, -1)` (TCP z down) | start and home pose of the fake |
| `fake_arm.mass_kg` | float | `1.0` | fake body mass |
| `fake_arm.inertia_kg_m2` | float | `1.0` | fake body inertia (same on all axes) |
| `fake_gripper.start_width_m` | float | `0.08` | |
| `fake_gripper.speed_m_per_s` | float | `0.1` | |
| `fake_gripper.object_width_m` | float \| null | `null` | fingers stop here when closing (null: no object) |

Also read, not from `config.yaml`:
- `station.yaml`: `endpoints.state`, `endpoints.commands` and `session_root`. All three are required in the station
  file (deploy §6.1); franka_control has no defaults for them, so a missing one is a config error naming the key.
- `contracts/contracts/limits.yaml`: `max_waypoints` (must be ≤ the compiled plan capacity, 64, or start-up fails
  "raise kMaxWaypoints and rebuild"), `rotation_tol`, `gripper_width_max_m`. Path baked in at build time
  (`CONTRACTS_LIMITS_FILE`, contracts §8).

`ConfigLoader` checks: types and lengths; positive rates, gains, limits and timeouts; `workspace_min < workspace_max`
per axis; `F_T_TCP` and `fake_arm.home_O_T_TCP` rigid; `cone_max_deg` in (0, 180); `realtime` one of two;
`shutdown.timeout_s` > `control.settle_timeout_s`; `watchdog.limit_s` ≥ 0.5 (7.1);
`safety.box_margin_m` and `safety.cone_margin_deg` ≥ 0. All problems are collected and reported together.

```
class ConfigLoader {
 public:
  static Config load(const std::filesystem::path& station_file, bool mock,
                     const std::filesystem::path& config_file /* default: next to the binary's source folder */,
                     const std::filesystem::path& limits_file);   // throws ConfigError listing every problem
  static std::string toYaml(const Config& c);                      // config_effective.yaml, --check-config output
};

class CommandLine {
 public:
  static CommandLine parse(int argc, char** argv);   // throws UsageError
  std::filesystem::path station;                     // --station (required)
  std::optional<std::filesystem::path> session, ready_file;
  bool mock, version, check_config;
};
```

## 11. `launch.py`

Python, run by the launcher (deploy §4). Imports only the standard library, `yaml`, `deploy`, `contracts`.

```
class FrankaControlProcess(ManagedProcess):
    NAME = "franka_control"
    READY_HINT = ("connected to the arm (or FakeArm) and holding its pose; check franka_control.robot_ip, "
                  "FCI enabled in Desk, brakes unlocked, user stop released, and the log")
    def program(self) -> list[str]        # [str(binary)]
    def checks(self) -> list[CheckResult]

PROCESS_CLASS = FrankaControlProcess
```

**`program()`:** `[path of the binary]`. `binary` from `station.yaml` is resolved against the station file's folder
(`station.resolve`); the `config.yaml` default against `franka_control/` (deploy §6.2). The base class appends
`--station --session --ready-file [--mock]`; the binary takes exactly these, plus two options used only by the checks:
`--version` (prints `franka_control <git hash> libfranka 0.9.2`, or `libfranka none` when built without it) and
`--check-config` (loads and validates the config as at start-up, prints the merged config as YAML, exit 0, or prints
every problem, exit 2).

**`checks()`** (quick, never moves the arm; the base class already checks the binary exists and is executable):

| Check | Status |
|---|---|
| `version`: `<binary> --version` reports the `libfranka_version` of `config.yaml` | fail on a mismatch or "none" ("rebuild: franka_control/build.sh"); with `--mock`, info only |
| `config`: `<binary> --check-config --station FILE [--mock]` | fail with its first problem line and "(N problems)" |
| ports: `port_free` for `endpoints.state` and `endpoints.commands` | fail if busy |
| `arm reachable` (not `--mock`): `host_reachable(robot_ip, 1337, 2 s)` (libfranka's command port) | fail: "arm not reachable at 172.16.0.2:1337: check the cable, the PC's static IP, FCI enabled in Desk" |
| `rt kernel` (not `--mock`, `realtime: enforce`): `/sys/kernel/realtime` reads `1` | fail: "no PREEMPT_RT kernel: install one, or set franka_control.realtime: ignore (backup, PLAN §9)" |
| `rtprio` (same): `resource.getrlimit(RLIMIT_RTPRIO)` soft limit ≥ 99 | fail: "rtprio limit N < 99: add the user to the realtime group with rtprio 99 in /etc/security/limits.conf, log in again" |
| `realtime: ignore` | **warn**: "real-time ignore mode: expect occasional reflexes under load; late cycles are shown as C2 `loop_late` on the status line and logged every second" |
| `load.mass_kg == 0` (not `--mock`) | warn: "no payload set: the rig's weight is not compensated" |
| backups | info: "arm: LibfrankaArm at 172.16.0.2, realtime enforce" or "arm: FakeArm, gripper: FakeGripper (--mock)" |

**Ready:** the default (`ready_file` exists). The binary creates it only when connected (or `FakeArm`), holding,
publishing C2 and answering C3 (7.3 step 8, deploy §4.4). A start-up failure exits with a FATAL line, which the
launcher reports at once.

## 12. Build

### 12.1 Dependencies

| Library | Version | How | Used for |
|---|---|---|---|
| libfranka | `libfranka_version` in `config.yaml`, default **0.9.2** (Panda, robot system 4.x, per Franka's compatibility table) | built from source by `build.sh` into `~/franka/libfranka-<version>`; `find_package(Franka <version> EXACT)` | `LibfrankaArm`, `FrankaHandGripper` |
| Eigen | 3.4.0 | system (`libeigen3-dev`, Ubuntu 22.04) | all math |
| libzmq | 4.3.x | system (`libzmq3-dev`, 4.3.4 on this machine) | sockets |
| cppzmq | 4.10.0 | CMake `FetchContent` (header-only) | C++ wrapper of libzmq |
| msgpack-cxx | 6.1.1 | `FetchContent`, `MSGPACK_NO_BOOST` | the wire format |
| yaml-cpp | 0.8.0 | `FetchContent`, static | `config.yaml`, `station.yaml`, `limits.yaml`, `index.yaml` |
| GoogleTest | 1.14.0 | `FetchContent`, tests only | unit tests (deploy §7.4 already names it) |
| Poco | system (`libpoco-dev`) | needed by libfranka only | |

### 12.2 `CMakeLists.txt`

- C++17, `-Wall -Wextra -Wpedantic`; Release `-O2` for the binary.
- Reads `libfranka_version` from `config.yaml` (a one-line regex on `^libfranka_version:`), so the version is written
  in one place.
- Options:
  - `FRANKA_CONTROL_WITH_LIBFRANKA` = `AUTO` (default) / `ON` / `OFF`. `AUTO` uses libfranka if `find_package` finds
    that exact version. Without it, `LibfrankaArm` and `FrankaHandGripper` are left out; the binary then runs only with
    `--mock` (otherwise FATAL "built without libfranka: run franka_control/build.sh") and `--version` says
    `libfranka none`, which `deploy check` fails on. So unit tests build on a laptop without libfranka.
  - `FRANKA_CONTROL_SANITIZE` = `OFF` / `ON`: ASan + UBSan (`-fsanitize=address,undefined -fno-omit-frame-pointer`,
    `-fno-sanitize-recover=undefined` so UB fails the test).
  - `FRANKA_CONTROL_TSAN` = `OFF` / `ON`: TSan, for the threading tests (can't be combined with ASan).
- Targets: `franka_control_core` (static library: everything but `main.cpp` and the two libfranka classes),
  `franka_control_libfranka` (the two libfranka classes, when available), `franka_control` (the binary),
  `franka_control_tests` (GoogleTest, linked to the core only; `gtest_discover_tests` for CTest).
- Compile definitions: `CONTRACTS_EXAMPLES_DIR` (default `${CMAKE_CURRENT_SOURCE_DIR}/../contracts/examples`),
  `CONTRACTS_LIMITS_FILE` (default `.../../contracts/contracts/limits.yaml`), `FRANKA_CONTROL_CONFIG_FILE`
  (`.../config.yaml`), `FRANKA_CONTROL_LIBFRANKA_VERSION` (`"0.9.2"` or `"none"`), `FRANKA_CONTROL_HAS_LIBFRANKA`.
- **Three build folders**, none in git: `build/` (the real binary: Release, no sanitizers, because they slow the
  loop), `build-test/` (unit tests with ASan + UBSan) and `build-tsan/` (the `threads` tests with TSan). Commands:

```
franka_control/build.sh                                     # libfranka (if missing) + build/franka_control
cmake -S franka_control -B franka_control/build-test -DFRANKA_CONTROL_SANITIZE=ON \
  && cmake --build franka_control/build-test && ctest --test-dir franka_control/build-test --output-on-failure
cmake -S franka_control -B franka_control/build-tsan -DFRANKA_CONTROL_TSAN=ON \
  && cmake --build franka_control/build-tsan && ctest --test-dir franka_control/build-tsan -L threads
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest franka_control/tests          # test_launch.py
```

The binary that drives the arm is therefore never a sanitizer build.

### 12.3 `build.sh`

```
franka_control/build.sh [--dry-run] [--no-apt] [--jobs N] [--prefix DIR] [--any-version] [--test]
```

1. Reads `libfranka_version` from `config.yaml`. Refuses a version outside 0.9.x (Panda) unless `--any-version`
   ("check Desk → Settings → System and Franka's compatibility table").
2. apt build dependencies (`build-essential cmake git libpoco-dev libeigen3-dev libzmq3-dev`), unless `--no-apt`.
3. libfranka: clone tag `<version>` (recursive), configure Release with `BUILD_TESTS=OFF BUILD_EXAMPLES=OFF`, build,
   install into `--prefix` (default `~/franka/libfranka-<version>`). Skipped if that version is already installed
   there. These steps are copied from main-sim's `scripts/build_libfranka_pandapy.sh` (libfranka part only), with the
   provenance header (deploy §7.6).
4. Configure and build `build/` (Release, `-DCMAKE_PREFIX_PATH=<prefix> -DFRANKA_CONTROL_WITH_LIBFRANKA=ON`).
5. Prints `build/franka_control --version`. With `--test`, also builds `build-test/` and runs CTest.

`--dry-run` prints every command and changes nothing. Idempotent.

## 13. Unit tests

GoogleTest, in `tests/`, run under ASan + UBSan (and the `threads`-labelled ones under TSan). No robot, no libfranka,
no network beyond `127.0.0.1`; sockets bind `tcp://127.0.0.1:*`. Classes that use time take a `Clock`, so most tests
drive time with a `FakeClock`; the few that need real threads wait on a condition with a timeout, never a fixed sleep
(deploy §7.4).

| File | Checks |
|---|---|
| `test_pose_math.cpp` | TCP from flange against an independent 4×4 formula (port of `test_tcp_composition_against_independent_4x4`); flange ↔ TCP applied exactly once each way; the TCP Jacobian gives `v_TCP = v_F + ω × r` for random twists; rotation angle between quaternions; the rigid check refuses a transposed matrix and a scaled rotation |
| `test_safety_gate.cpp` | every main-sim case ported (table below), plus: width < 0 and > max rejected; `t ≤ now` skipped, unchecked and counted; all past → `accepted 0, skipped N`, nothing rejected; `t > now + max_lead_s` rejected; any rejection → nothing accepted, every non-skipped waypoint in `rejected` (the failing ones with their reason); `accepted + skipped + len(rejected) == N` for every case in this file; a waypoint both pulled back and shortened; reasons name the axis and numbers; the first segment starts at the start point; a 50 ms-spaced chunk at 0.99 `v_max` is never shortened; the cone is measured on the flange axis with a tilted `F_T_TCP` |
| `test_waypoint_trajectory.cpp` | midpoint lerp of position and width; slerp at a = 0.25 against a known rotation; the shorter way when the quaternions have opposite signs; exact values at the points; before the first and after the last point (hold); capacity and non-increasing times refused by `append` |
| `test_impedance_controller.cpp` | with `J = [I₆ \| 0]`, `F_T_TCP = I`, `dq = 0`: a 1 cm position error gives `τ = K·0.01` toward the target on that axis only; a 0.1 rad rotation error about z gives the libfranka formula's torque with the right sign; damping `−D·v` for a known `dq`; clipping at `max_position_error_m` and `max_rotation_error_rad`; the nullspace term moves only joint 7 for this J; Coriolis added, no gravity; torque clamp; rate limit against `tau_J_d`; non-finite input flagged |
| `test_homing_motion.cpp` | starts and ends at rest; reaches the goal exactly at T; peak speed ≤ the limit; T ≥ `min_duration_s`; all joints finish together |
| `test_watchdog.cpp` | not expired at the limit, expired 1 ns after; age |
| `test_state_machine.cpp` | the full table of section 4.2 (every allowed transition, incl. `stop` from `idle` → `stopped`, `start` from `stopped`, `homing` → `idle`) and every refused (incl. `start` in `holding` and `running` → `Ack(ok=false)`) (state, request) pair of section 4.3, with its reason text; `heartbeat` ok in every mode |
| `test_supervisor.cpp` | (FakeClock, a `SharedState`, a test `Arm`; no threads) start → `holding` with a one-point plan; `set_waypoints` → `running` and a `kFollow` plan starting at the old plan's value at now; a second request replaces the not-yet-reached waypoints; all-skipped → `holding`, `accepted 0, skipped N`, no fault; pulled-back and shortened waypoints counted in `gate`; the plan's end passes → `holding`; holding continues while heartbeats arrive; no request for 0.5 s → `stopped` + `kHold`, not at 0.499 s; refused `set_waypoints` in `idle`/`stopped`/`fault` → `ErrorReply`, state unchanged; gate rejection → `fault` + `kHold` + counters; a loop fault in the snapshot → `fault` with its text; backstop trip → `stopped`; **homing**: no heartbeat for 10 s while homing does not stop anything, `markAlive` at the end (watchdog resumes), reply only after `homed_generation` and the new session, timeout → `requestStop()` called + `fault` + `Ack(false)`, too-long move refused at once, a home TCP outside the box → `fault` + `Ack(false)`; the reply's fields in the contract order with `shortened` a count; gripper fault → `fault` |
| `test_control_loop.cpp` | `step()` with hand-made readings: holds the measured pose at session start; follows a plan; a 2 ms gap between cycles and a missed packet each count one late cycle, a 1 ms gap none; acts on each command generation once; the backstop holds after `limit + extra` and reports the generation; `kHome` ends the session only after settling; a non-finite torque → finished + fault; snapshot fields filled |
| `test_triple_buffer.cpp` | newest wins; `refresh` false when nothing new; reader's copy is stable while the writer writes; a two-thread test (label `threads`) where every written struct has all fields equal and the reader never sees a mixed one |
| `test_shared_state.cpp` | the atomics are lock-free (`static_assert` plus a runtime check); gripper fault set/read |
| `test_codec.cpp` | reads `CONTRACTS_EXAMPLES_DIR/index.yaml` (contracts §8): every good C3 request decodes and its `values` match; every broken C3 request is refused with the same `field` and `problem`; every good C2, C3-reply, `ack` and `error` example decodes and re-encodes **byte for byte**; every good C2 example decodes; limits come from `CONTRACTS_LIMITS_FILE`; row-major ↔ Eigen conversion |
| `test_command_server.cpp` | over `tcp://127.0.0.1:*` with a test REQ client: each request gets its reply type; broken bytes → `ErrorReply` naming the field; unknown schema and wrong version → `ErrorReply`; a C2 message sent as a request → `unexpected_message`; the `home` reply is deferred while C2 keeps publishing; every decoded request feeds the watchdog |
| `test_state_publisher.cpp` | with FakeClock: one message per period; `seq` +1 per broadcast; `get_state` carries the last seq; `mode` (all six values), `error`, `gate` (incl. `pulled_back`) and `loop_late` mapping; with no new snapshot (after a fault) messages keep coming with `seq` +1 and the same `t_ns`; every message passes the C++ decoder's checks; never blocks with no subscriber |
| `test_config_loader.cpp` | the real `config.yaml` loads; station overrides merge (maps key by key, lists replaced); an unknown override key refused with its path; each bad value refused (negative `v_max`, non-rigid `F_T_TCP`, box min ≥ max, 6-element `q_home`, bad `realtime`, `watchdog.limit_s` 0.4, negative margins); all problems listed at once; endpoints and `session_root` read from `station.yaml`, a missing one refused naming the key; `--mock` works without a reachable `robot_ip`; `max_waypoints` > capacity refused |
| `test_arm_status.cpp` | `describeArmProblem` (port of `derive_panda_error`): reflex mode wins over error flags; `current_errors` text; `"[]"` and empty mean none; user stop pressed; a ready arm gives `""` |
| `test_gripper_worker.cpp` | with `FakeGripper` and a no-op sleep: changes below the deadband are not sent; closing and opening reach the target; the fingers stop at `object_width_m` (port of `test_fake_gripper_stalls_at_object`); a `GripperError` sets the gripper fault; the measured width reaches `SharedState` |
| `test_fake_arm_run.cpp` | (label `threads`) a short real-time run of the whole `App` with `--mock` on free ports and a test REQ client: ready file appears; `get_state` → `idle`; `start`; waypoints moving 5 cm over 1 s with heartbeats → `running`, then `holding`, the fake TCP within 2 mm of the last waypoint; heartbeats stop → `stopped` within 0.5 s + 10 ms; `stop` → `stopped`; `home` → C2 shows `homing`, then `Ack(ok)` and `idle`; `FakeArm::injectFault` → `fault` with the text in C2; shutdown → all threads end, exit code 0 |
| `test_launch.py` | (pytest, deploy §9) `program()` with the default and a station-overridden `binary`; `checks()` with a broken station section; `--mock` skips the arm, RT-kernel and rtprio checks; `realtime: ignore` gives a warn |

**Safety cases ported from main-sim's `tests/test_bridge_safety.py`, one for one:**

| main-sim test | Here | Change and why |
|---|---|---|
| `test_nonfinite_rejected_and_last_kept` | NaN and inf waypoints rejected (`not finite`); the decoder also refuses them; the loop keeps holding | rejection → `fault` (PLAN §5) |
| `test_workspace_box_clamp` | `(0.805, −0.405, 0.045)` (each 5 mm out) is pulled back to `(0.80, −0.40, 0.05)` and counted; `(0.811, 0, 0.3)` (11 mm out) and `(2.0, −1.0, −0.5)` are rejected, the reason naming each axis out; a point exactly on the boundary is accepted unchanged; each face separately | clamp only within the 1 cm margin; beyond it, reject |
| `test_orientation_cone_clamp` | with a 30° cone: 29.9° accepted unchanged; 31.5° pulled back to exactly 30° (rotation still orthonormal, TCP position unchanged); 32.1° and z horizontal (90°) rejected | clamp only within the 2° margin; beyond it, reject |
| `test_step_clamp_position_is_clamp_not_reject` | 10 cm in 5 ms is shortened to `0.25 · (0.005 + 0.010)` = 3.75 mm, not rejected; waypoints every 5 ms toward the same far point converge to it | new bucket formula (6.2) |
| `test_step_clamp_rotation` | 0.5 rad in 1 ms is shortened to `1.0 · (0.001 + 0.010)` = 0.011 rad | same |
| `test_long_gap_gives_only_the_slack` | savings never exceed the slack: after a 10 s still segment, a 10 cm move 5 ms later gets only `v_max · (5 ms + slack)`; and a 20 cm move spread over 10 s is accepted (that is 0.02 m/s) | a long gap between timed waypoints is a slow move, not a jump |
| `test_order_box_before_step` | from x = 0.795, a target at x = 0.805 with Δt = 10 ms is pulled back to 0.800 first, and the remaining 5 mm fits the bucket (`0.25 · (0.010 + 0.010)`), so it is not shortened; a waypoint beyond the margin is rejected even when its speed is fine; one inside but too fast is only shortened | the pull-back runs before the speed check, as in main-sim |
| `test_speed_bound_independent_of_request_rate` [200 Hz, 1 kHz, bursts] | commanded 2 m/s for 1 s with waypoints every 5 ms, 1 ms, or in bursts of 10 within 0.2 ms every 20 ms: the plan, evaluated at 1 kHz, moves ≤ `v_max · 0.1 · 1.1` in any 100 ms window, and ≥ `0.9 ×` that for the first two | evaluated on the interpolated path, not only at waypoints |
| `test_reply_jitter_does_not_clamp` | waypoints at 0.9 `v_max`, times 5 ms apart ± 2 ms jitter: never shortened | |
| gripper width-map tests | not ported: the Franka Hand reports its width itself (no servo map); replaced by `width_offset_m` and max-width tests | |

## 14. Hardware-only items (PLAN §8, phase 4)

Not testable on the laptop; each has a TODO in `config.yaml`: the workspace box and cone around the
table; the home posture (the after-homing box check catches a wrong one); the payload (`load`); the gripper's
`width_offset_m` and max opening with the printed fingers; collision thresholds and impedance gains (tracking error
≤ 2 mm, no reflexes while holding); the real-time check (libfranka's `communication_test`, success rate ≥ 0.99; the
loop-timing log line under load).

## 15. Before coding

**The first build step is to check the libfranka 0.9.2 calls this design names against that version's headers**
(`include/franka/robot.h`, `model.h`, `gripper.h`, `exception.h`, `control_types.h`, `rate_limiting.h`). They are
from knowledge of libfranka's API and were not checked here, because libfranka is not installed on the planning
machine:

- `franka::Robot(ip, RealtimeConfig::kEnforce | kIgnore)`; `Robot::control` with a torque callback
  `(const RobotState&, franka::Duration) -> franka::Torques` and its `limit_rate` / `cutoff_frequency` arguments; the
  joint-position form for homing (`franka::JointPositions`, `franka::MotionFinished`).
- `Robot::stop()` called from another thread while `control` runs, preempting it with a `ControlException`.
- `Robot::readOnce`, `setCollisionBehavior` (the 4-array form), `setJointImpedance`, `setLoad`, `loadModel`.
- `Model::pose(Frame::kFlange, state)`, `Model::zeroJacobian(Frame::kFlange, state)` (column-major, rows
  [linear; angular]), `Model::coriolis(state)`.
- `RobotState` fields `q`, `dq`, `tau_J_d`, `robot_mode`, `current_errors`, `control_command_success_rate`.
- `franka::Gripper`: `homing`, `move(width, speed)`, `grasp(width, speed, force, epsilon_inner, epsilon_outer)`,
  `readOnce` (`width`, `max_width`), `stop`.
- The exception classes of 5.6 and `IncompatibleVersionException`.

Any difference changes only `LibfrankaArm` or `FrankaHandGripper` (and this file), never the rest.
