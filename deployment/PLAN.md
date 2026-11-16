# Deployment plan

Branch `rnoro/final-deployment`. This file holds the goal, the rules, the split between the two machines, every
contract between them, the failure rule and the plan for each folder. The detailed layout of the code is in
`STRUCTURE.md` (this folder: communication and the shared contracts) and in each folder's own `STRUCTURE.md`, written
from this plan before any code.

Most of the working code comes from `rnoro/deployment` (all its tests pass there). Its interactions do not: it ran
everything on one PC with one clock and one launcher. Here the work is split over two machines, each with its own code.

## 1. Goal

**Physical deployment with continuous policy control of the Franka Panda:** the arm keeps moving while the policy
computes its next chunk.

The setup:

- The **Raspberry Pi 5** on the gripper streams both cameras (IMX519 main, IMX219 160° ultrawide) at 30 FPS over
  Wi-Fi, on the lab router **ASUS_5g**. Its sender and its contract are finished and verified (section 5.1). It sends
  to the **laptop**.
- The **laptop** (the user's machine; RTX 3050 Ti 4 GB) receives the frames and the arm state, runs the policy, and
  sends the resulting waypoints to the PC. Code: `policy_deploy/`.
- The **PC** is cabled directly to the Franka. It runs only the arm side: the safety gate, the 1 kHz control loop, the
  gripper, homing, and the state broadcast back to the laptop. Code: `control_deploy/`. The PC is too weak to run the
  policy, which is why the work is split.
- The policy is a **plain diffusion policy** first. A behavior-prompting checkpoint is served too, by
  `PromptableDiffusionPolicy` (still kind `diffusion`; it needs `policy.prompt_dataset` and `manager.task_name`).
  Until a policy trained on rig data exists, the policy on the real arm is **hold still**.
- The gripper is the **Franka Hand with our printed fingers**, at most **80 mm** open.

Already verified on the hardware: the Pi stream reaching a receiver (30 packets/s, no gaps), and `franka_control` on
the PC homing all 7 joints with `h`. Not yet verified: waypoint following on the real arm, the policy on the laptop.

Milestones:

1. This plan, then `STRUCTURE.md` and the two folder `STRUCTURE.md` files, agreed.
2. All code written, with unit tests, including contract tests run by both machines' code.
3. Each machine runs alone with stand-ins: the laptop with a fake Pi and a fake PC (`policy_deploy run --mock`), the
   PC with the fake arm (`control_deploy run --mock`).
4. Each folder's `DIRECTIONS.md` (section 11.1) followed once from scratch on its machine.
5. Hardware bring-up (section 12): the PC alone, the laptop alone, then both with hold still over Wi-Fi, then over
   wired, then the trained policy.

## 2. Rules

1. **One command per machine.** `policy_deploy run` on the laptop, `control_deploy run` on the PC. Each also has
   `check` (is everything ready?) and `--mock` (its stand-ins). Settings come from one station file per machine.
   When something fails, one line says which process, what failed and what to check.
2. **Each machine owns its partition of work.** There is no shared launcher, no ssh between the machines, and neither
   machine starts, stops or reads files of the other. They meet only through the messages in section 5.
3. **Code lives in its folder.** `policy_deploy/` and `control_deploy/` are self-contained: code, config, tests,
   directions. No folder imports the other. The only shared code is `deployment/contracts/`, and it holds **only what
   crosses between the machines** (section 5.3–5.5). Laptop-only messages live in `policy_deploy/`.
4. **Modular and swappable.** Every part that might be swapped (camera source, policy, arm, link) sits behind a small
   interface with a real and a fake implementation. Numbers that might change (rates, limits, addresses, timeouts)
   live in config, not code.
5. **Object-oriented and simple.** One job per class.
6. **Code first, with unit tests**, for every class and every contract. No simulation tests.
7. **No code before its STRUCTURE.md.**
8. **The ideal pipeline first, then failure modes**, by the rule in section 7.
9. **Real-world only.** The MuJoCo sim stays separate.
10. **Derive, don't replicate.** Copy working code from `rnoro/deployment` and say so in the file's header. Do not copy
    its process interactions where the split calls for something else.

## 3. Machines, folders and processes

```
                                LAPTOP: policy_deploy/                          PC: control_deploy/
 Pi 5 ──C0 pi_stream/v1──►  frames ──C1──► manager ◄──C4──► policy
       (Wi-Fi, ASUS_5g)                   (episodes,                            franka_control (C++)
                                           keyboard,       ◄──C2 state 100 Hz── safety gate, 1 kHz loop,
                                           pose conversion, ──C3 commands─────► gripper, homing, watchdog,
                                           PC clock map)                        keyboard (h, hand homing)
                                                                                     │ FCI (Ethernet, 172.16.0.2)
                                                                                     ▼
                                                                                Franka Panda + Franka Hand
```

```
deployment/
  PLAN.md             this file
  STRUCTURE.md        communication, the shared contracts, the codebase map
  contracts/          what crosses between the machines: C2/C3 definitions, limits, the TCP frame, the link settings,
                      example messages read by the Python and the C++ tests
  policy_deploy/      the laptop: STRUCTURE.md, DIRECTIONS.md, its processes, its contracts (C0, C1, C4), tests
  control_deploy/     the PC: STRUCTURE.md, DIRECTIONS.md, franka_control (C++), its run wrapper, tests
```

| Machine | Process | What it does |
|---|---|---|
| laptop | **frames** | Receives the Pi stream (C0), maps Pi time to laptop time, keeps the last second of frames, reports each camera's health, and on request returns 2 main + 2 ultrawide images prepared exactly as in training (C1). |
| laptop | **policy** | Loads the model on the GPU. Given images and relative state, returns 16 relative steps (C4). Knows nothing about the arm, absolute poses or the PC. |
| laptop | **manager** | The episode loop and the keyboard. Keeps the PC's state history and the PC clock map, builds the policy inputs, converts poses (absolute → relative in, relative → absolute out), sends waypoints and heartbeats to the PC, decides "slow" vs "conclude". |
| PC | **franka_control** | The only process connected to the arm. Every waypoint passes the safety gate; the 1 kHz Cartesian impedance loop moves between accepted waypoints; it handles the gripper and homing, broadcasts the state, and stops the arm when the laptop goes quiet. Its keyboard homes the arm (`h`) and the hand. |

The laptop's three processes stay separate (as in `rnoro/deployment`): JPEG decoding at 30 packets/s and the torch
diffusion loop then never delay the manager's heartbeats or its timing of arriving states, `hold_still` runs without
torch, and each process can be run and watched alone.

## 4. One policy step

1. The manager asks frames for the newest images (C1). It gets 2 main images (T − 50 ms, T) and 2 ultrawide images
   (T − 100 ms, T), each with its time on the **laptop clock**. T is the newest main frame's mid-exposure time.
2. It converts the two main image times to the **PC clock** with its PC clock map (section 5.6) and looks up the hand
   pose and gripper width at those times in its state history (C2, kept in PC time), by interpolation.
3. It makes the poses relative to the **anchor**, the pose at T, and sends images and relative state to the policy
   (C4). It gets 16 relative steps back.
4. It turns the first `exec_h` steps into absolute waypoints with the same anchor, timed `T_pc + k × 50 ms` on the PC
   clock, and sends them to the PC (C3).
5. The PC's gate checks them; they replace the waypoints not yet reached; the 1 kHz loop tracks them.
6. While the arm executes them, the manager starts the next step at once.

By the time a chunk arrives its first waypoints are already in the past and are skipped. The arm never holds if
`L + 2C + d_up ≤ (exec_h − 1) × 50 ms`, where L is the image latency (capture → laptop, about 75–110 ms measured on
the live Pi), C is one step (frames + policy + the C3 round trip) and d_up the one-way delay for the waypoints to reach
the PC (under 1 ms wired, a few ms on Wi-Fi). With `exec_h` = 8 that needs C ≤ about 120 ms. A slower
step is not a failure: the arm reaches the last waypoint and holds until new ones arrive ("slow"). The status line
shows the margin. `exec_h` is a setting (section 13).

### 4.1 The policy inputs, field by field

| C4 field | Shape | Comes from | Clock it is timed on |
|---|---|---|---|
| `main` | uint8 (2, 224, 224, 3) | frames: Pi JPEG → letterbox → 224 | Pi → laptop (one-way map from `t_send_ns`) |
| `ultrawide` | uint8 (2, 224, 224, 3) | frames: Pi JPEG → fisheye remap to the virtual pinhole → letterbox → 224 | same |
| `eef_pos` | float32 (2, 3) | C2 `O_T_TCP` (target) at the two main image times, relative to the anchor; last row 0 | PC (C2 `t_ns`) |
| `eef_rot6d` | float32 (2, 6) | same, rotation as the first two rows of R; last row identity | PC |
| `eef_rot6d_wrt_start` | float32 (2, 6) | the same rotations relative to the episode's start pose | PC |
| `gripper_width_m` | float32 (2, 1) | C2 `gripper_width_m` at the two main image times | PC |

So every policy input depends on two clock maps: Pi → laptop (in frames) and laptop → PC (in the manager). Getting an
image time wrong by 4 ms puts the pose 1 mm off at 0.25 m/s. Both maps and their error bounds are in section 5.6.

## 5. Contracts

A **contract** is the exact shape of a message: fields, types, units, reference frame, clock and timing. Both sides
are written and tested against it.

Conventions for every message:

- **Every time names its clock.** There are three: the Pi's (only inside frames), the laptop's `CLOCK_MONOTONIC`
  (C1, inside the laptop) and the PC's `CLOCK_MONOTONIC` (every time on C2 and C3). Integer nanoseconds. A time never
  crosses into another clock's message without going through a clock map.
- **Units:** metres, radians, seconds or ns. Images uint8 RGB (H, W, 3).
- **Absolute poses:** 4×4 float64, row-major, `O_T_TCP`: the TCP (section 9) in the robot base frame.
- **Every message carries `schema`** (name and version); a receiver rejects a version it doesn't know.
- **Transport:** ZeroMQ, msgpack, arrays as raw bytes plus shape and dtype. Broadcast (PUB/SUB) for state,
  request/reply for everything else. Servers bind, clients connect.

| Contract | Between | Where it is defined |
|---|---|---|
| C0 | Pi → laptop frames | `policy_deploy/` (the Pi contract, `PI_PAYLOAD.md`) |
| C1 | frames ↔ manager (laptop) | `policy_deploy/` |
| C4 | manager ↔ policy (laptop) | `policy_deploy/` |
| C2 | PC → laptop manager | `deployment/contracts/` (shared) |
| C3 | laptop manager → PC | `deployment/contracts/` (shared) |

### 5.1 C0: Pi → frames (unchanged)

`pi_stream/v1`, exactly as in `rnoro/deployment`'s `contracts/PI_PAYLOAD.md` §0, which moves to `policy_deploy/`:
the Pi binds a PUB on port 5555; one 3-part ZeroMQ message per main frame (JSON header, main JPEG, ultrawide JPEG),
both cameras 640×480 at 30 FPS; nothing is sent back. The laptop connects a SUB to the Pi's address on ASUS_5g. The
original msgpack C0 design (H.264, time-sync socket) is not carried over.

### 5.2 C1 and C4 (inside the laptop)

The same messages as in `rnoro/deployment` (C1 `get_frames` / `get_health`; C4 `config`, `reset`, `infer`, optional
`prompt`), with the fields of section 4.1. One change: the C4 `config` reply also carries the policy's `kind`
(`hold_still` / `diffusion`), so the manager can apply the `settled` rule of section 9 (`c4.config.reply/v2`). C1 times are on the laptop clock. Full field tables go in
`policy_deploy/STRUCTURE.md`. A broken chunk (contract validator) ends the run; a chunk whose step 0 is more than
2 cm / 5° from the anchor is "odd": skipped and counted.

### 5.3 C2: PC → laptop, arm state (broadcast)

Unchanged from `rnoro/deployment`. The PC binds the PUB; the laptop subscribes.

| Field | Type | Meaning |
|---|---|---|
| `t_ns` | int64 | **PC clock**: when this reading came from the arm (start of that 1 kHz cycle) |
| `seq` | uint64 | +1 per message; a gap is a missed message; going backwards means franka_control restarted |
| `O_T_TCP` | float64 (4, 4) | measured TCP pose, base frame |
| `O_T_TCP_target` | float64 (4, 4) | the pose the 1 kHz loop is tracking now |
| `q`, `dq` | float64 (7,) | joint angles (rad), speeds (rad/s): reported only, never commanded |
| `gripper_width_m` | float64 | opening between the fingertips, absolute |
| `mode` | enum | `idle` / `running` / `holding` / `stopped` / `homing` / `fault` |
| `error` | string | empty, or the fault text |
| `gate` | counters | waypoints accepted / shortened / pulled back / rejected since start |
| `loop_late` | uint64 | late 1 kHz cycles since start |

Rate: 100 Hz. The laptop needs at least 50 Hz. `t_ns` repeats unchanged after a fault that ends the 1 kHz loop (a
reflex, a libfranka error), briefly around homing, and for up to 1 s after a stopped homing
(`control.settle_timeout_s`); the laptop ignores repeated times for interpolation and for its clock map. Freshness is judged from the mapped `t_ns`, never from arrival time: a link
stall can deliver a burst of old states. Liveness ("no C2 for 2 s") is judged from arrival time, since `t_ns` can
repeat.

### 5.4 C3: laptop → PC, commands (request/reply)

The PC binds the REP; the laptop's manager is the **only** client (one C3 client at a time; the PC's own keyboard acts
inside franka_control, not as a client).

| Request | Contents | Reply |
|---|---|---|
| `hello` (new) | none | `t_pc_ns` (PC clock, read while answering), the PC's commit, `mock`, `F_T_TCP`, `gripper_width_max_m`, `max_waypoints`, `watchdog_limit_s`, `homing_timeout_s`, `max_lead_s`. Read-only. |
| `get_state` | none | one C2 message |
| `heartbeat` | none | ok. Every manager loop cycle (20 ms), with or without new waypoints. |
| `start` | none | ok (`idle`/`stopped` → `holding` at the current target), or refused with the reason |
| `stop` | none | ok: stop following, hold. Never refused. |
| `home` | none | ok when the home pose is reached (blocking joint move of all 7 joints); refused unless `idle`/`stopped`; `ok: false` with a reason starting `stopped:` when the PC's `s` stopped the move (the PC is then `stopped`) |
| `set_waypoints` | `chunk_id`; `t_ns` int64 (N,) increasing, **PC clock**; `O_T_TCP` float64 (N, 4, 4); `gripper_width_m` float64 (N,) | `accepted` (shortened ones included), `shortened`, `skipped` (time already passed), `rejected` with reasons. Replaces every waypoint not yet reached. |

`hello` replaces `rnoro/deployment`'s cross-folder read of `franka_control/config.yaml`: the laptop checks at start-up
that the PC's watchdog fits its heartbeat budget, that `F_T_TCP` matches `contracts/frames.yaml`, that the limits
match, and that the PC's `mock` flag matches its own. It is the only new C3 message, additive, and the 1 kHz path
doesn't see it. (The C++ codec also drops the laptop-only schema names: STRUCTURE.md §3.4.)

**What "policy output → arm motion" means on the PC.** The laptop sends timed Cartesian TCP poses, never joint
targets. For each waypoint the gate checks: time not passed (else skipped) and at most `max_lead_s` ahead (now
**1.5 s**, was 5 s, so a waypoint sent on the laptop clock by mistake is rejected); finite; width in [0, 0.08];
workspace box (pulled back within 1 cm, rejected beyond); tilt cone of the flange z axis (pulled back within 2°,
rejected beyond); speed (shortened to 0.25 m/s, 1 rad/s). Accepted waypoints replace the rest of the plan, starting
from the current target, so the motion has no jump. The 1 kHz loop interpolates the plan (linear position, slerp
rotation), computes the Cartesian impedance torque at the TCP (plus nullspace and Coriolis; gravity is the robot's)
and sends it to the joints. After the last waypoint it holds. A rejected waypoint is a `fault`, never cleared
remotely.

Joint targets are not used: the policy speaks relative TCP poses, a joint command would need inverse kinematics of the
7-joint arm on the laptop (with branch flips between chunks), and the gate would need forward kinematics just to check
it. The cost: the gate has no joint-limit check; a target near a limit ends in a robot reflex, which is a safe stop and
a fault.

**Watchdog.** With no C3 request for `watchdog_limit_s` while `holding`/`running`, franka_control stops following and
holds (`stopped`). The arm at most finishes the plan it already accepted (at most (`exec_h` − 1) × 50 ms ahead: 0.35 s
at `exec_h` 8, 0.55 s at 12; never more than `max_lead_s`). `idle` needs no watchdog: the loop holds the pose anyway.

### 5.5 Shared files in `deployment/contracts/`

| File | Read by | Holds |
|---|---|---|
| the C2/C3 Python message classes | laptop manager, its fake PC, the PC's run wrapper | the definitions above, with validators |
| `limits.yaml` | both (C++ too) | rotation tolerance, max gripper width 0.08, max waypoints 64 |
| `frames.yaml` | both (C++ too) | `F_T_TCP`, the one definition of the TCP (section 9) |
| `link.yaml` | both (C++ too) | the link: the PC's and the laptop's addresses, ports, watchdog, timeouts (section 6) |
| `examples/` | Python and C++ tests | example and deliberately broken C2/C3 messages |

Both machines use the same commit. `hello` carries the PC's commit and `policy_deploy check` shows both.

### 5.6 Clocks

**Pi → laptop** (in frames, unchanged): each Pi packet's `t_send_ns` against its laptop arrival; the least-delayed
packets of each second fitted for offset and drift. Laptop times come out late by the minimum Wi-Fi latency (a few ms).

**PC → laptop** (in the manager, new). The manager keeps every arm-related time on the PC clock and converts only
laptop times into PC time (the two main image times, and "now" for its freshness checks). Two estimators behind one
interface, picked by one setting:

| Estimator | Samples | Error |
|---|---|---|
| `round_trip` (default) | a `hello` twice a second: the PC's time against the midpoint of the round trip; the shortest round trips of a 10 s window, fitted for offset and drift | half the round-trip asymmetry: well under 1 ms wired, a few ms at worst on Wi-Fi |
| `one_way` (backup) | every C2: PC `t_ns` against laptop arrival, only when `t_ns` advanced; the least-delayed of a 5 s window | the minimum PC→laptop latency: about 1.5 ms wired, a few ms on Wi-Fi |

Guards for both: at least 2 s of samples before the first step (and again after a long `home`, which sends no hellos);
at start-up the map must agree with a `hello` sample within 5 ms (else refuse to start; a real check for `one_way`,
while `round_trip` builds its map from those same hellos); fit jumps are logged; and three waypoint sets in a row that
the PC skipped entirely while the laptop thought they were in the future, or sent while the newest C2 `t_ns` was more
than 20 ms ahead of the mapped "now" (a map running behind), end the run (`clock mismatch`). The PC never changes its
clock and needs no clock code. Chrony/PTP on the wall clock was rejected: it would put the PC's gate and watchdog on a
clock that can jump.

## 6. The link: Wi-Fi first, then wired

All three machines are on the lab router **ASUS_5g**: the Pi, the laptop and the PC. The laptop ↔ PC link starts on
Wi-Fi; later either machine can be cabled to a router LAN port (the laptop with a USB-C Ethernet adapter, since it has
no port; the PC with a second port, since its first goes to the Franka). The Pi always stays on Wi-Fi.

**Nothing changes in the code or the contracts between the two.** It is the same router, subnet and messages; only two
things differ, and neither needs a different code path:

| | Wi-Fi | Wired to the router |
|---|---|---|
| Addresses | the Wi-Fi interface's address | the wired interface's address: a different network card, so a different address, unless the router reserves the same one |
| Delay | a few ms, spikes of 100+ ms, rare stalls over 0.5 s | under 1 ms, steady |

So `contracts/link.yaml` holds just the PC's address (what franka_control binds and the laptop connects to), the
laptop's address (what the PC's firewall lets in) and the timing values. Going wired means changing the address there;
with reserved addresses on the router, nothing at all.

**One set of timing values, chosen to be safe on Wi-Fi**, used wired too:

| Setting | Value | Why it costs nothing when wired |
|---|---|---|
| Heartbeat | every 20 ms | |
| PC watchdog | 1.0 s | the arm finishes the plan it already accepted (at most (`exec_h` − 1) × 50 ms ahead: 0.35 s at `exec_h` 8, 0.55 s at 12) and holds anyway; the watchdog only decides when that hold becomes `stopped` |
| C3 timeout (laptop) | 0.3 s | only reached when something is wrong |
| Newest state older than (skip the step) | 100 ms | wired, the newest state is a few ms old |
| No state for (conclude) | 2.0 s | the PC protects itself; this only decides how fast the laptop reports a dead PC |
| Clock estimator | `round_trip` | accurate on both |

These are numbers in `link.yaml`, so they can be tightened once wired if ever useful (for example a 0.5 s watchdog).
The rule the laptop checks at start-up against `hello`: its worst gap between heartbeats (loop period + C3 timeout +
local health timeout + a jitter allowance: 0.02 + 0.3 + 0.05 + 0.2 = 0.57 s) must be below the PC's watchdog.

On Wi-Fi a stall longer than the watchdog stops the arm and the laptop concludes: annoying, never unsafe. The Pi
stream (about 30 Mbit/s) shares the laptop's radio with C2/C3 (well under 1 Mbit/s); a wired laptop removes that
contention, which is the main practical gain of cabling it.

**Who can command the arm.** C3 has no authentication and the PC sits on a shared lab network either way.
franka_control binds C2/C3 only to its ASUS_5g address (never `0.0.0.0`), and a firewall rule on the PC lets only the
laptop's addresses (Wi-Fi and wired) and the PC itself (its status line and console) reach those ports. Encryption (ZeroMQ CURVE) can be added later.

## 7. Failure rule

Every problem is one of two kinds:

- **Slow → keep running.** No new waypoints until things catch up; the PC finishes its plan and holds while heartbeats
  arrive; steps resume by themselves.
- **Unrecoverable → conclude.** The laptop sends `stop`, records the episode as aborted with the reason, stops its own
  processes in reverse order, and exits with one line: which process (or the PC, or the link), what failed, what to
  check. **The PC keeps running**, holding the arm, ready for the next laptop run.

**franka_control is never upset by anything else**: its 1 kHz loop never waits on messages, so trouble elsewhere can
only make the arm hold.

| Problem | Kind | Reaction |
|---|---|---|
| A camera stale (no new frame for 0.35 s), a missed frame pick | slow | skip the step |
| The ultrawide repeating an old frame (the Pi's `uw_reused`) for more than one frame | slow, then conclude | a pair is usable only if both: its capture lies within [main − 60 ms, main + 25 ms] (`pair_max_age_ms` 60, `pair_max_lead_ms` 25; the live Pi pairs the nearest frame, ultrawide 16 ms newer in the recorded packet; the lead is confirmed from `pi_collect`'s pair-gap report), and the same ultrawide frame has been the pair of at most 2 consecutive packets (`pair_max_reuse` 2: a third consecutive packet with it gets no ultrawide image, whatever the timing). Ultrawide health counts only new frames at their own capture time, and only once served as an accepted pair, so a stalled IMX219, or one whose pairs are always refused, goes stale, then down |
| A camera down (2 s), the Pi stream restarted or lost | conclude | "main camera down: check the Pi and ASUS_5g" |
| Policy slower than the waypoints it has | slow | the arm holds at the last waypoint |
| An odd chunk | slow | skip, count |
| Policy hung (5 s), errors, or a broken chunk | conclude | |
| A C2 `seq` gap, or the newest state older than 100 ms | slow | skip the step |
| No C2 for 2 s, or a C3 timeout | conclude | "PC not answering: check franka_control on the PC and the link" |
| C2 `seq` goes backwards | conclude | "franka_control restarted on the PC" |
| Clock mismatch (section 5.6), or `hello` disagreeing at start-up | conclude | names the field |
| The PC reports `fault` (reflex, lost FCI connection, rejected waypoint, homing failed) | conclude | the PC holds; a fault needs Ctrl-C and a restart of `control_deploy run` on the PC (after releasing the user stop if it was a reflex) |
| The PC `stopped` during an episode without the laptop asking (its watchdog, or `s` on the PC's keyboard) | conclude | "the PC stopped the arm (watchdog or the PC's `s`): check the link and the PC" |
| Between episodes, the PC's keys change its mode (`stopped` after `s`, `homing` after `h`/`g`) | normal | the laptop follows C2: `c` works from `idle`/`stopped`; while C2 shows `homing` the laptop refuses `c` and `h` itself (a hint on the status line; it stays idle, never concludes). A `start` the PC refuses is a hint too |
| A C3 `home` answered `ok: false` with a reason starting `stopped:` (`s` on the PC stopped the move) | normal | the arm holds where it stopped (`stopped`); the laptop shows it and stays idle. A `home` refused with `refused: mode is homing` (the PC's own `h`/`g` started first) is a hint too. Any other `ok: false` or no reply to `home` is a conclude. |
| The laptop dies, sleeps, or the link drops | (PC side) | the watchdog stops following and holds; the PC stays up; the next laptop run reconnects with `start` |
| A message failing its contract | conclude | names the message and field. A broken C3 request gets an error reply from the PC (not a fault); the laptop concludes on it. |
| Operator `s` | episode ends | `stop`, recorded `stopped`; `h` then `c` for the next one |
| Operator `q` / Ctrl-C on the laptop | conclude (normal) | `stop`, laptop shuts down; the PC keeps holding |
| At start-up: anything `check` finds wrong, or the PC in `running`/`holding` (another client) or `fault` | conclude | refuses to start and says why |

Normal operation, counted and shown on the status line: waypoints shortened to the speed limit, pulled back onto the
box or cone, widths clipped into range.

## 8. Folder plans

Each plan is the brief for that folder's `STRUCTURE.md`.

### 8.1 `control_deploy/` (PC)

**Process:** `franka_control`, copied from `rnoro/deployment` (`91b178b`, the commit that built and homed on the real
PC), disturbed as little as possible. Changes: build paths (`../contracts` → `../../contracts`; config, limits,
`frames.yaml` and example paths are baked in at build time, so rebuild after the move), hint texts, the `hello`
request, `F_T_TCP` read from `contracts/frames.yaml`, `max_lead_s` 1.5 s, the bind address from `contracts/link.yaml`, and
the keyboard below. No change to the 1 kHz cycle (`ControlLoop::step`), the gate, the impedance controller, the trajectory, the homing
motion or finger moves; the keyboard and stop-during-homing need small additive changes to the supervisor, shared
state, the control thread's session loop and the gripper worker, plus: one new state-machine event (homing aborted →
`stopped`), a new `Gripper::home()` for the hand's homing, SIGHUP handled like SIGTERM, a `--keyboard` flag, and in
`LibfrankaArm`/`FakeArm` the sticky stop flag becomes a per-call stop epoch (a sticky flag would mislabel every later
reflex as a requested stop). The last one is on the verified path and is tested at bring-up.
libfranka 0.9.2 in `~/franka/` and `build.sh` are reused.

**Entry:** `control_deploy check | run | console | firewall [--mock] [--local] [--station FILE]`, a small Python wrapper on the PC (its checks: config valid, arm
reachable, realtime mode, payload set, link address present on this PC, firewall rule; then it runs the binary in the
terminal and shows a status line). `--mock` uses the fake arm and fake gripper; `--local` binds 127.0.0.1 (tests);
`firewall` prints the nftables rule built from `link.yaml`.

**Homing and stop on the PC:** `g` runs the Franka Hand's homing (C2 shows `homing`), then returns the fingers to the
commanded width. A C3 `stop` or the PC's `s` during any homing move stops the move and holds where it stopped
(`stopped`); if the move finishes first, it ends normally (`idle`). Whether libfranka's stop mid-move needs a recovery,
and whether the hand's homing can be interrupted, is checked at bring-up; either outcome is safe.

**Start model:** the operator runs `control_deploy run` on the PC **before** the laptop. It connects, the Franka Hand
homes its fingers once, and the arm holds where it is. It stays up across laptop runs and is restarted only after a
fault or at the end of the day.

**PC keyboard** (inside franka_control, not a C3 client; `h` and `g` are refused while `running`/`holding`, i.e. while
the laptop is driving an episode; `s` and `q` are never refused):

| Key | Does |
|---|---|
| `h` | home the arm: all 7 joints to `homing.q_home_rad`, slow joint move (as verified) |
| `g` | home the hand: the Franka Hand's own homing (fingers close and reopen to calibrate) |
| `s` | stop: hold where it is |
| `q` / Ctrl-C | stop the arm and exit |

**Bring-up console** (`control_deploy console`, PC only, run instead of the laptop, never alongside it): sends C3
itself to test waypoint following before any laptop run: `start` and hold, a 1–2 cm move, a 5° rotation, a gripper
width, the watchdog (stop sending), and one waypoint deliberately outside the box (expect a fault with no motion).

**Tests:** the existing C++ tests (gate, interpolation, holding, watchdog, every state change, impedance torque, a
short fake-arm run, C2/C3 against the shared examples, sanitizers), plus `hello`, the keyboard commands, the bind
address from `link.yaml`, and the wrapper's checks.

### 8.2 `policy_deploy/` (laptop)

**Processes:** frames, policy, manager (section 3). One Python package `policy_deploy` with subpackages per process,
its own contracts (C0, C1, C4), a launcher, and a fake PC. An import test enforces that process subpackages import
only `policy_deploy`'s contracts and `deployment/contracts/`. New package names and a new environment, so nothing
clashes with `rnoro/deployment`'s editable installs.

**Entry:** `policy_deploy check | run [--mock]`. `run` starts frames, policy, manager in that order, each after the
previous is ready, and stops them in reverse. It never starts or stops anything on the PC. Each process can be run
alone (`python -m policy_deploy.frames`, `.policy`, `.manager`). A read-only link probe (`python -m
policy_deploy.manager.pc_probe`) shows the C2 rate, `seq` gaps, the clock fit and C3 round trips.

**Start-up order of the manager:** C2 arriving → while C2 shows `running`/`holding` (an earlier laptop session's
episode), wait without sending anything on C3, up to 2 × the watchdog, so the PC's watchdog can stop it; likewise
while C2 shows `homing` (a home left running by an earlier run), up to 60 s → `hello`
exchanges start and give 2 s of clock samples → `hello` checked (section 5.4) → `get_state` mode `idle`/`stopped`
(wait while `homing`; refuse in `fault`) → heartbeats start → policy config checked → both cameras ok → ready.
A commit different from the PC's is a warning, not a refusal; every difference that matters is caught by the `hello`
checks. A trained policy is refused on the real arm while `frames.yaml` says `settled: false` (section 9).

**Laptop keyboard:** `c` start an episode, `y` / `n` end it as success / failure, `s` stop, `h` home (C3 `home`),
`q` quit. Same meaning as in `rnoro/deployment`; `c` and `h` are refused locally while C2 shows `homing` (section 7).

**From `rnoro/deployment`:**
- frames: the pi_v1 path (`pi_stream_source`, `receiver`, buffer, selector, image prep copied from dataproc unchanged,
  health, `OneWayClockMap`), with the frozen-ultrawide fix (section 7). The c0/H.264/time-sync code is dropped (the
  Pi contract is final).
- policy: unchanged (hold still, diffusion with the light workspace, key map, probe), plus a `precision` setting
  (fp32 default, bf16 backup).
- manager: `message_comms` with `franka_client` → `pc_client`, `state_listener` → `pc_listener`, the PC clock map,
  the `hello` checks, the new failure codes, timeouts from `contracts/link.yaml`.
- launcher: `deploy/`'s managed-process code, for the laptop's three processes only. The geometry tool and
  `gripper.yaml` are not carried over (section 9).
- `--mock`: the fake Pi plays a recorded episode (`episode_000045`); a new Python fake PC stands in for
  franka_control with an offset, drifting clock, so every test exercises the clock map.

**Tests:** as in `rnoro/deployment` for the copied code, plus the clock maps (offset, drift, uneven delays, repeated
`t_ns`), the `hello` checks, `link.yaml` loading, the frozen-ultrawide cases, and an end-to-end `run --mock`.

**First thing to measure on the laptop:** the policy's speed and memory on the RTX 3050 Ti (35 W cap). Estimate for
the v1 architecture (two CLIP ViT-B/16 encoders + a 1D UNet, 16 diffusion steps): about 3.1–3.3 GB of the 4 GB, and
150–220 ms per inference, likely over the budget at `exec_h` 8. `run_001` is a prompting checkpoint (a different,
larger architecture, served by `PromptableDiffusionPolicy`: kind `diffusion`, with `policy.prompt_dataset` and
`manager.task_name`), so speed is measured with a small benchmark tool that
builds the plain diffusion architecture with random weights; `run_001` gives load time, host RAM and the GPU
memory of real weights (and its own speed with `bench --prompt-dataset`). Backups in section 13 (`precision: bf16` = autocast).

## 9. The TCP frame

The policy's inputs and outputs are poses **relative to the TCP frame**: a point between the fingertips with three
axes fixed to the gripper. A relative step "+1 cm along x" means "along the gripper's x axis". Training and the robot
must use the same point **and the same axes**, or every motion the policy outputs goes in a rotated direction.

- **Training** defines it from the ultrawide camera: `T_cam_tcp` in `rig_pi_v3.yaml` (on `rnoro/data-work`, from
  CAD). Its axes are the camera's, tilted 15°: x = image right (the direction the fingers close), y = image down,
  z = toward the fingertips.
- **The robot** defines it from the flange: `O_T_TCP = O_T_F × F_T_TCP`. In `rnoro/deployment`, `F_T_TCP` uses the
  Franka Hand's own axes (−45° about the flange z axis), where **y** is the direction the fingers close. That is a
  90° turn about z from training's axes.

The rule: **the TCP on the wire (C2 and C3) is the training TCP.** It is defined once, in
`contracts/frames.yaml`, read by franka_control and by the laptop. Its `settled: false` key (laptop only) makes
`policy_deploy` refuse a trained policy on the real arm until this section is done. The 90° difference is fixed there, by changing
`F_T_TCP`'s rotation (its sign follows from which side of the hand the camera is mounted), never by a correction on
the laptop. The PC's gate doesn't depend on the TCP axes (the box uses position, the cone the flange z axis).

This does not block hold still (zero motion, any axes) or the PC console tests. It must be settled, and checked on the
arm, before the first trained-policy run. Related, also before that run: the robot's camera mount should reproduce the
handheld's camera-to-fingertip geometry, so the images look like training's.

## 10. Tests

All unit tests run on a laptop without the Pi, the arm or a GPU.

- **Contract tests:** every example in `deployment/contracts/examples/` passes its validator and every broken one
  fails, in Python and in C++. Laptop-only contracts the same, inside `policy_deploy/`.
- **Message flow:** the manager against fake frames, a fake policy and the fake PC, covering the normal step and every
  row of section 7.
- **End to end per machine:** `policy_deploy run --mock` with scripted keys; `control_deploy run --mock` with a
  scripted console.
- **Across the two:** on the laptop, the laptop's processes against the real `franka_control --mock --local`, built
  there without libfranka (fake arm only).

## 11. How the work is done

Following the agent-team method: one subagent writes each `STRUCTURE.md` from its section here (top-level, then
`control_deploy/`, then `policy_deploy/`), one reviewer checks them together, every open question is settled with the
user before code. Then one executor per folder (`contracts/` first), one reviewer over all of it, all tests green,
a local commit per wave, nothing pushed without asking.

**Final step: the two `DIRECTIONS.md` files**, written after the code works, then followed once from scratch.

### 11.1 The `DIRECTIONS.md` files

One per machine: `policy_deploy/DIRECTIONS.md` (the laptop: run the policy) and `control_deploy/DIRECTIONS.md` (the
PC: run the arm). Each takes a fresh machine to a running system for someone who has **never read the code**: only
commands to copy, in order.

- **Every step has the same four parts:** one plain sentence saying what the step does and why; the exact command(s)
  to copy; what you should see when it worked; what to do if you don't.
- **Every command the folder has gets its own short section:** `check`, `run`, `run --mock`, and on the PC `console`
  and `firewall`; on the laptop the link probe and the policy timing. Each says when to use it, the command, the keys
  it takes, and what a good result looks like.
- **From scratch, in order:** clone and checkout, environments, build (PC), the one-time network setup (ASUS_5g
  addresses, `link.yaml`, the firewall), a first `--mock` run, the real run, stopping and restarting.
- **The daily routine** at the top, once set up: on the PC, Desk (unlock, FCI, release the stop) then
  `control_deploy run`; on the laptop `policy_deploy check` then `policy_deploy run`; the keys; how to stop.
- **A "when something fails" table:** the message you see → what to do.
- **The PC runs Ubuntu 20.04.** `control_deploy/DIRECTIONS.md` says what that adds, in plain steps: Python 3.10
  installed with `uv` (20.04 ships 3.8; no `sudo` needed), the same build packages as the old franka_control build
  that worked on this PC, internet during the first build (msgpack is downloaded), and starting with
  `build.sh --test`, since the PC build is the first full compile of the libfranka parts.
- Plain words, short steps, no design explanation (that stays in PLAN.md and the STRUCTURE.md files). Kept up to date
  with the code; the final review checks every command in them by running it.

## 12. Hardware bring-up order

1. **PC alone:** `control_deploy run`, then `console`: hold, small moves, rotation, gripper, watchdog, out-of-box
   fault. Set the workspace box, payload and `width_offset_m` first (they are TODO in the config).
2. **Laptop alone:** frames against the live Pi on ASUS_5g (probe), then the policy's speed and memory.
3. **Both, Wi-Fi, hold still:** the clock map, the watchdog over the link, waypoints on the real arm.
4. **Both, wired, hold still:** cable the laptop and/or the PC to the router, update the addresses in `link.yaml`;
   the same run.
5. **Trained policy:** after section 9 is settled and a rig-trained checkpoint exists.

## 13. Backups (plug and play)

Each is another implementation or value behind the same interface, picked by one setting, unit tested, and shown by
`check`.

| Part | Default | Backup |
|---|---|---|
| Link | Wi-Fi on ASUS_5g (first) | wired to the router: change the addresses in `link.yaml` |
| PC clock estimator | `round_trip` | `one_way` |
| Policy speed | the checkpoint's diffusion steps, fp32, `exec_h` 8 | raise `exec_h` to 10–12 first (training's runner used 12), then bf16, then fewer steps |
| Policy device | CUDA | CPU (slow: the arm holds between chunks) |
| Real-time scheduling (PC) | PREEMPT_RT kernel, `realtime: enforce` | libfranka's ignore mode, with the loop-timing warning (install the RT kernel early: the PC now holds for long stretches) |
| Arm | the real Panda | the fake arm (`control_deploy run --mock`) |
| PC, seen from the laptop | the real PC | the fake PC (`policy_deploy run --mock`) |
| Camera source | the real Pi | the fake Pi (a recorded episode) |

## 14. Later

Not part of this work until the whole setup runs end to end:

- **Keyboard jog commands for the hand**, on both machines: keys that move the TCP in small steps (and open/close the
  fingers), on the PC's keyboard and on the laptop's, for setting up scenes and testing.
- A hand-eye calibration check of the camera mount.
- Encryption on C3 (ZeroMQ CURVE).

## 15. Open items

- **Section 9:** the training TCP axes into `contracts/frames.yaml`, before the first trained-policy run.
- **Addresses on ASUS_5g:** reserved (fixed) addresses on the router for the PC, the laptop and the Pi, so
  `link.yaml` and the Pi address don't change between days. Later, for wired: a USB-C Ethernet adapter for the laptop
  and a second network port on the PC.
- **`exec_h`:** set after the policy is timed on the laptop.
