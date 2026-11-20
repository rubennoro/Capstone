# control_deploy/: the PC

The design of everything that runs on the PC: the C++ process `franka_control` (the only process connected to the
Franka Panda) and a small Python wrapper (`control_deploy check | run | console | firewall`). Written from PLAN.md
(§3, §5.3–5.6, §6, §7, §8.1, §9, §12, §13) and from `deployment/STRUCTURE.md` (the shared contracts and the
communication; called "top-level" below). PLAN.md wins over the top-level file, and both win over this one: where
this file disagrees with either, this file is fixed.

Source references:
- `old:<path>:<line>` points into `rnoro/deployment` at commit `91b178b`, worktree
  `/home/ruben/Capstone-deployment/deployment/` (the build that connected and homed all 7 joints on the real PC).
- `old-S §n` is a section of `old:franka_control/STRUCTURE.md`, the detailed design of the unchanged parts. It stays
  the reference for the terms (impedance, Jacobian, nullspace, reflex, PREEMPT_RT, triple buffer), the torque law,
  the gate rules and the libfranka calls. It is not copied into this branch.
- `PLAN §n` and `top §n` are sections of `deployment/PLAN.md` and `deployment/STRUCTURE.md`.

## 1. Role

### Owns

- `franka_control` (C++17, libfranka 0.9.2): the safety gate, the 1 kHz Cartesian impedance loop, the watchdog, the
  state machine, arm homing, the Franka Hand (fingers and their own homing), reflex handling, the C2 broadcast and the
  C3 server. **Unchanged in behaviour** except for what section 3 lists.
- The PC keyboard, read inside `franka_control`: `h` home the arm, `p` start pose, `g` home the hand, `s` stop and
  hold, `q` quit.
- The wrapper `control_deploy`: the PC's checks, running the binary in the terminal with a status line and a session
  folder, the bring-up console (PLAN §8.1), and the firewall rule text.
- `control_deploy/station.yaml` (this PC's settings) and `franka_control/config.yaml` (the arm's settings).

### Doesn't own

Frames, the policy, episodes, relative poses, the laptop keyboard, any clock map (top §1.2: the PC has no clock code),
and the three contract files in `deployment/contracts/` (it reads them, never writes them).

### Talks to

- The laptop's manager, only through C2 (PUB, binds `tcp://<link.yaml pc.address>:5620`) and C3 (REP, binds
  `tcp://<pc.address>:5630`) (top §2.2). Also `console` on the PC itself, never at the same time as the laptop.
- The robot over FCI (`172.16.0.2`, its own network port): the arm through `franka::Robot`, the hand through
  `franka::Gripper`.
- The operator: the keyboard and the terminal of `control_deploy run`.

## 2. Tree

```
control_deploy/
  STRUCTURE.md                 this file
  DIRECTIONS.md                from-scratch steps on the PC (outline in section 10)
  station.yaml                 this PC's settings (section 6.2)
  pyproject.toml               distribution "control-deploy", import package control_deploy
  control_deploy/              the Python wrapper (section 6)
    __init__.py
    __main__.py                python -m control_deploy == the console script
    cli.py                     argparse: check | run | console | firewall; --mock, --local, --station
    paths.py                   where things are: this folder, franka_control/, the binary, config.yaml, the repo root
    station.py                 Station: loads station.yaml (strict)
    check_result.py            CheckResult, print_table, exit_code (copied from old:deploy/deploy/check_result.py)
    check_helpers.py           file_exists, executable, port_free, host_reachable, info, warn
                               (copied from old:deploy/deploy/check_helpers.py, minus python_imports)
    binary.py                  FrankaControlBinary: runs --version and --check-config, parses their output
    netinfo.py                 local IPv4 addresses per interface (`ip -j -4 addr`); C3 peers (`ss`)
    firewall.py                FirewallRule: the nftables text from link.yaml; reads and compares the live table
    checks.py                  PcChecks: every row of `check` (section 6.3)
    session.py                 Session: the run's folder, run.yaml, copies of the yaml files
    status_line.py             StatusLine: one line from a C2 ArmState and the C3 peers
    runner.py                  Runner: `run` (spawns the binary on the terminal, tees its log, status line, signals)
    console/
      __init__.py
      console.py               Console: preflight, the step sequence, results.yaml
      steps.py                 HoldStep, MoveStep, RotateStep, GripperStep, WatchdogStep, OutOfBoxStep
      waypoints.py             pure helpers: time grid, min-jerk profile, translate / rotate a pose, chunking
  franka_control/              the C++ process (copied from old:franka_control/ at 91b178b, then section 3)
    CMakeLists.txt
    build.sh
    config.yaml                every arm setting (section 7)
    .gitignore                 build/, build-test/, build-tsan/, build-libfranka-check/, + build-nolibfranka/ (8.3)
    cmake/
      write_build_info.cmake   new: writes build_info.h with the git commit on every build (section 3.1)
    include/franka_control/    one header per class (old list, plus the new ones of section 3)
    src/                       one .cpp per header, plus main.cpp
    tests/                     GoogleTest, support/, tsan.supp (section 9.1)
  tests/                       pytest for the wrapper (section 9.2)
```

**Package name: `control_deploy`** (distribution `control-deploy`), as top §1.1 already names it. It is different
from every package of both branches: `contracts`, `deploy`, `process_frames`, `policy`, `message_comms` (old),
`deploy_contracts`, `policy_deploy` (new). It is installed editable into its own env, `~/envs/control_deploy`, never
into the old `~/envs/deploy` (top §1.4). The editable install is required: `paths.py` finds `franka_control/` as
`Path(control_deploy.__file__).resolve().parents[1] / "franka_control"`.

Dropped from old `franka_control/`: `launch.py`, `pytest.ini`, `tests/test_launch.py` (the old launcher's interface;
their checks move to `checks.py`), and `STRUCTURE.md` (this file replaces it; old-S stays the reference).

## 3. Every change vs `rnoro/deployment@91b178b`

Everything not in this table is copied byte for byte from `old:franka_control/` (apart from comments that name old
paths). In particular **unchanged**: `ControlLoop::step` (the 1 kHz cycle; apart from copying a `kHome`'s joint target
and open width, section 3.3), `SafetyGate` (apart from one reason text,
section 3.2), `WaypointTrajectory`,
`ImpedanceController`, `HomingMotion`, `Watchdog`, `TripleBuffer`, `StatePublisher`, `pose_math`, `arm_status`,
`log`, `contract_limits`, the moving parts of `GripperWorker` (apart from a finger move aborted by the shutdown, section
3.3) and `FrankaHandGripper`, and the shutdown order.

### 3.1 Build, paths, config

| File | Change | Why |
|---|---|---|
| `CMakeLists.txt` | `CONTRACTS_DIR` cache path `${CMAKE_CURRENT_SOURCE_DIR}/../../contracts` replaces the two cache paths of old:franka_control/CMakeLists.txt:39-41. Derived compile definitions on `franka_control_core`: `CONTRACTS_LIMITS_FILE`, `CONTRACTS_FRAMES_FILE`, `CONTRACTS_LINK_FILE`, `CONTRACTS_EXAMPLES_DIR` (absolute). Configure fails (`FATAL_ERROR`, naming the path) if any is missing. | The folder moved one level down; top §1.4 fixes this CMake text. |
| `CMakeLists.txt` | The configure-time `git rev-parse --short HEAD` (old:franka_control/CMakeLists.txt:43-47) and the `FRANKA_CONTROL_GIT_HASH` definition are removed. A custom target `build_info` (ALL) runs `cmake/write_build_info.cmake` on every build; the core library depends on it. | top §3.8 obligation 3: `hello.commit` is the full 40-character commit of the source the binary was built from, refreshed on every `cmake --build`. |
| `cmake/write_build_info.cmake` | New. Writes `<build>/generated/franka_control/build_info.h` with `kGitCommit` (`git rev-parse HEAD`, or `unknown`) and `kGitDirty` (`git status --porcelain --untracked-files=no -- ../../contracts ..` non-empty). Rewrites the file only when its text changes, so an unchanged commit recompiles nothing. | as above; the dirty flag is shown by `--version` and warned on by `check` (the contract allows only a hash or `unknown`). |
| `CMakeLists.txt` | Adds the new sources (`contract_frame`, `contract_link`, `keyboard_reader`, `key_queue`) and tests (section 9.1). Comment paths `control_deploy/franka_control/...`. The 91b178b fix (libfranka classes inside `franka_control_core`) is kept as is. | |
| `.gitignore` | One line added: `build-nolibfranka/` (the laptop's build for the cross test, section 8.3). | |
| `build.sh` | Only the header comment and usage text: `control_deploy/franka_control/build.sh`. The `usage()` line range (old:franka_control/build.sh:27, `sed -n '5,16p'`) follows the header. Behaviour, apt list, libfranka prefix unchanged. | It is location-independent already (`$HERE`). |
| `config.yaml` | `tcp:` removed (old:franka_control/config.yaml:37-42). `watchdog.limit_s` removed (line 57). `safety.max_lead_s` 5.0 → **1.5**. Comments: `launch.py` → `control_deploy`, `deploy check` → `control_deploy check`, `message_comms` → "the laptop's hello check", `contracts/contracts/limits.yaml` → `contracts/limits.yaml`. **2026-11-20:** `homing.q_start_rad` (`null`: unset) and `homing.at_pose_tol_rad` (0.05) added. Full file in section 7. | `F_T_TCP` now comes from `contracts/frames.yaml`, the watchdog limit from `contracts/link.yaml` (top §4.2, §4.3); `max_lead_s` per PLAN §5.4; the start pose (key `p`). |
| `include/config.h` | `Config` keeps its fields `F_T_TCP` and `watchdog.limit_s` (now filled from frames.yaml and link.yaml), so `ControlLoop`, `Supervisor`, `SafetyGate`, `FakeArm` read them exactly as before. New fields: `pc_address`, `local`, `contracts_dir`, `limits_file`, `frames_file`, `link_file`. `GateConfig::max_lead_s` default 1.5. The `endpoint_state`/`endpoint_commands` fields stay, now built from link.yaml. **2026-11-20:** `HomingConfig::q_start_rad` (`std::optional<Vec7>`, nullopt: `p` refused) and `at_pose_tol_rad` (0.05). | No change in any user of `Config`. |
| `include/config_loader.h`, `src/config_loader.cpp` | `load(station_file, mock, local, config_file, ContractFiles{limits, frames, link})`. Reads `F_T_TCP` through `ContractFrame`, `pc.address`, the two ports and `timing.watchdog_limit_s` through `ContractLink` (both new). Endpoints = `tcp://<pc.address>:<port>`, or `tcp://127.0.0.1:<port>` with `--local`. The station's `endpoints:` section is refused (old readStation, old:franka_control/src/config_loader.cpp:239-268, read it): `station.yaml: endpoints: no longer used: franka_control binds contracts/link.yaml pc.address and ports`. `session_root` is still required in the station. A station override of `tcp.*` or `watchdog.limit_s` is refused with the hint `moved to contracts/frames.yaml` / `moved to contracts/link.yaml timing.watchdog_limit_s` (the generic "not a key of config.yaml" error, old lines 36-38, gets that suffix). The 0.5 s floor (old line 17) now checks the link.yaml value, text `link.yaml: timing.watchdog_limit_s: must be >= 0.5 s (franka_control's floor)`. `toYaml` adds a `contracts:` block (the four paths, `pc_address`, `local`, the endpoints, `F_T_TCP`, `watchdog_limit_s`). **2026-11-20:** `homing.q_start_rad`: null or absent = unset, else 7 numbers inside the Panda's joint limits (the problem names the first joint outside); `homing.at_pose_tol_rad` > 0. | top §4.3, §4.4, §2.3; `check` compares the `contracts:` paths with the checkout (moved without rebuild). |
| `include/contract_frame.h`, `src/contract_frame.cpp` | New, `ContractFrame::load(path, rotation_tol, problems)`: reads `F_T_TCP` (4 rows of 4 finite numbers, rigid within `rotation_tol`), returns it re-normalised through a unit quaternion as old line 408 did. Ignores `settled` and any other key. | top §4.2, §4.4. |
| `include/contract_link.h`, `src/contract_link.cpp` | New, `ContractLink::load(path, problems)`: `pc.address` (IPv4 dotted literal by `inet_pton`, not `0.0.0.0`), `ports.state`, `ports.commands` (integers 1024–65535, different), `timing.watchdog_limit_s` (> 0). Ignores the other keys. `endpoint(port, local)`. | top §4.3, §4.4. |
| `include/command_line.h`, `src/command_line.cpp` | Two flags: `--local` (bind `127.0.0.1`) and `--keyboard` (read keys from stdin). Usage: `franka_control --station FILE [--session DIR] [--ready-file FILE] [--mock] [--local] [--keyboard] \| --version \| --check-config --station FILE [--mock] [--local]`. `--station` stays required. | top §2.3 (`--local`); the keyboard is opt-in so the unit tests never touch a developer's terminal. |
| `src/main.cpp` | `--version` prints `franka_control commit=<40 hex\|unknown> dirty=<yes\|no> libfranka=<0.9.2\|none>` (old:franka_control/src/main.cpp:25-29 printed the short hash). `--check-config` passes `--local` and the contract files. SIGPIPE is ignored (first line of `main`, before any thread). | the wrapper parses key=value; the full commit. SIGPIPE: if the `control_deploy run` wrapper dies, the next log line would otherwise kill the process mid-episode with no hold or settle; now the write fails (EPIPE) and the process keeps running, Ctrl-C and SIGTERM still work. |

### 3.2 Contracts and messages

| File | Change | Why |
|---|---|---|
| `include/messages.h`, `src/messages.cpp` | New `msg::Hello` (no fields) and `msg::HelloReply` (fields in the order of top §3.8), added to `msg::Any`, `schemaOf`, and `isC3Request` (Hello yes, HelloReply no). | top §3.7, §3.8. |
| `src/codec.cpp`, `include/codec.h` | The known-schema table (old:franka_control/src/codec.cpp:25-35, which listed C0, C1 and C4 too) becomes exactly the 12 `SHARED_SCHEMAS` of top §3.4. New public `static std::vector<std::string> Codec::knownSchemas()`: the full `name/vN` strings **sorted as whole strings** by byte order (sorting the names alone would put `c3.hello` before `c3.hello.reply`, the wrong order). Decode and encode `c3.hello/v1` and `c3.hello.reply/v1`, including a list-of-str field (`schemas`: each a well-formed `name/vN`, strictly increasing), the `commit` rule `^([0-9a-f]{40}\|unknown)$`, `F_T_TCP` rigid with this side's `rotation_tol`, the five floats finite and > 0, `max_waypoints` ≥ 1, in the check order of top §3.8. | top §3.4: a laptop-only message sent to the PC is then `unknown_schema`. |
| `include/safety_gate.h`, `src/safety_gate.cpp` | The other waypoints of a rejected request get the contract's reason `not checked: another waypoint of this request was rejected` (`kNotCheckedReason`, top §3.7.1); old: `not used: waypoint <i> of this request was rejected`. The rules are unchanged. | The laptop names the first reason that does not start with `not checked`; with the old text it named waypoint 0's "not used" line instead of the real reason. |
| `include/command_server.h`, `src/command_server.cpp` | The constructor takes a `msg::HelloReply` template (built once by `App`, section 5.5). `handle()` answers `msg::Hello`: copy the template, set `t_pc_ns = clock_()`, encode, return. The stamp is the last thing before `encode`, after decode, `watchdogExpiredNow()` and `markAlive()` (old:franka_control/src/command_server.cpp:77-79), which stay in front of every request. | top §3.8 obligations 1–2; hello feeds the watchdog like every decoded request (top §3.7.3). |

### 3.3 Keyboard, hand homing, stop during homing

| File | Change | Why |
|---|---|---|
| `include/keyboard_reader.h`, `src/keyboard_reader.cpp` | New: `KeyboardReader` (the keyboard thread), `TerminalMode` (RAII cbreak, IXON off), the key table (upper case acts like lower case). Section 4.3. | PLAN §8.1 PC keyboard. |
| `include/key_queue.h`, `src/key_queue.cpp` | New: `KeyQueue`, a queue with a mutex, keyboard thread → messaging thread; `h`, `p` and `g` dropped when 16 keys wait, `s` never dropped. | The Supervisor belongs to the messaging thread (old-S §3.3); keys reach it as events, like C3 requests. |
| `include/state_machine.h`, `src/state_machine.cpp` | One new event `kHomingAborted` (`homing` → `stopped`). `kHome` is reused for both kinds of homing (same rules). The comment on `kStop` ("message_comms sends it") names the laptop's manager. | Stop during homing (section 4.4). |
| `include/supervisor.h`, `src/supervisor.cpp` | `onHome(Source)`, `onHandHome()`, `onStop(Source)` (aborts a homing), `onKey(Key)`; the homing kind (arm or hand) and source (C3 or keyboard); `tick()` sends the arm stop once the move runs (under `SharedState::stopHomingMove`, again every 100 ms while the move runs and the outcome is unresolved, also after a homing timeout), and resolves the outcomes (aborted, hand homing done, hand homing timeout); the hand's stop goes through `SharedState::stopHandHoming`. New last constructor parameter `Gripper* gripper = nullptr` (nullptr: `g` is refused, "no gripper"; old tests compile unchanged). Section 4.4. **2026-11-20:** `onStartPose(Source)` (key `p`; refused while `homing.q_start_rad` is unset, outside `idle`/`stopped`, or away from home); `onHome` and `onStartPose` share `startArmMove(source, ArmGoal, target, open_width_m)` (`ArmGoal` `kHome`/`kStartPose`: timeout check, bookkeeping, the `kHome` command, the log line); at the end of a `p` move the TCP must be inside the box and tilt ≤ `safety.cone_max_deg` (no margin), else `fault`; `onStart` is refused away from `q_start_rad` once it is set (section 4.3). | PLAN §8.1 keys; the brief's stop-during-homing behaviour. |
| `include/shared_state.h`, `src/shared_state.cpp` | `LoopSnapshot::homing_aborted_generation` (uint64). New atomics: the homing-abort flag (messaging sets, control clears), the hand-homing request generation (messaging → gripper) and report (gripper → messaging, one `atomic<uint64_t>` holding generation × 4 + result, so the pair is read consistently). Two small locks, never taken by the 1 kHz callbacks: the hand's homing (`beginHandHoming`/`endHandHoming` on the gripper thread around `home()`, `stopHandHoming(gen, stop)` on the messaging thread) and the arm's homing move (`beginHomingMove(session)`/`endHomingMove()` on the control thread around `moveToJoints`, `stopHomingMove(session, stop)` on the messaging and main threads). The stop runs under the lock, so it reaches only the move it is meant for (section 4.4). **2026-11-20:** `LoopCommand::q_target` (`optional<Vec7>`, nullopt: `q_home_rad`) and `open_width_m` (`optional<double>`, nullopt: fingers unchanged), used with `kHome` (key `p`). | The loop and the gripper thread report outcomes the same way as `homed_generation` today. A stop must never abort a finger move (a gripper fault) or land in the torque session after a finished move (a fault). |
| `include/control_loop.h`, `src/control_loop.cpp` | `run()` only (old:franka_control/src/control_loop.cpp:222-264): before the homing move, an abort skips it; a stop-requested exception from the move with the abort flag set waits for the robot to leave its move mode and starts a new torque session instead of ending the thread; the move is bracketed with `SharedState::beginHomingMove`/`endHomingMove` (replacing the `in_homing_` atomic). Section 4.4. **2026-11-20 (key `p`):** `step()` copies the accepted `kHome`'s `q_target` (nullopt: `q_home_rad`) and `open_width_m` (no allocation); `run()` moves to that goal (`home_target_`) and sets the finger target to the open width just before the move; after a normal finish with an open width, the next session's first cycle keeps that width as the finger target instead of the measured one (`hold_open_width_next_session_`; an aborted move holds the measured width). `observeHoming()` is unchanged. | Stop during homing ends in a hold, not a fault. `p` opens the hand during the move; holding the width measured at the end would close it again. |
| `include/arm.h` | Comment only: `requestStop()` affects the control call running at that moment. | |
| `include/libfranka_arm.h`, `src/libfranka_arm.cpp` | The sticky `stop_requested_` flag (old:franka_control/src/libfranka_arm.cpp:133, 187-194) becomes a stop epoch: `requestStop()` increments it, then calls `robot_->stop()`; each `runTorqueControl`/`moveToJoints` call records the epoch at its start and classifies a `ControlException` as `kStopRequested` only if the epoch changed during that call. Hint text `rerun deploy run` → `restart control_deploy run` (line 82). **A change on the verified path** (decided, PLAN §8.1, top §8.11): tested at bring-up (section 11.1). | After an aborted homing the process keeps running; a sticky flag would mislabel every later reflex as a requested stop. |
| `include/fake_arm.h`, `src/fake_arm.cpp` | The same epoch rule (old:franka_control/src/fake_arm.cpp:46, 70). Test hook `statusAfterStop(ArmStatusFlags)`: what `status()` reports after a stop. | The fake must allow a new session after an abort. |
| `include/gripper.h` | New `virtual bool home() = 0;` blocks for the hand's own homing; true when done, false when ended by `stop()` (also by a `stop()` that came after the last `clearStop()` but before `home()` began: then the fingers do not move); throws `GripperError` on failure. New `virtual void clearStop() = 0;` (the gripper thread, before a hand homing). | `g`. |
| `include/franka_hand_gripper.h`, `src/franka_hand_gripper.cpp` | `home()`: `franka::Gripper::homing()`; a `stop_requested_` flag set by `stop()` turns a failure or `false` during a requested stop into `false` (aborted); otherwise `GripperError("Franka Hand homing failed ...")`. `home()` never resets the flag (a stop that came first skips `homing()`); `clearStop()` does. `prepare()` unchanged (it still homes at start-up when `gripper.homing_on_start`). | |
| `include/fake_gripper.h`, `src/fake_gripper.cpp` | `home()`: closes to 0 and opens to `fake_gripper.start_width_m` at its speed in 5 ms steps; `stop()` ends it (returns false), and a stop before it began skips it. Like libfranka (`CommandException` on an aborted command), a `moveTo` ended by `stop()` throws `GripperError`. `clearStop()`. Test hook `failNextHome(bool)`. | The fake must show a stop that would abort a finger move. |
| `include/gripper_worker.h`, `src/gripper_worker.cpp` | `run()` (old:franka_control/src/gripper_worker.cpp:56-64): each iteration first serves a new hand-homing request (`clearStop()`, then `beginHandHoming`: if a stop came first, report aborted without moving; else `gripper.home()`, `endHandHoming`, report done or aborted, set a `resend_` flag), then `stepOnce()` as before. `resend_` makes the next `stepOnce()` send the current target even inside the deadband (a NaN "last sent" would not work: NaN comparisons are false). A `GripperError` from `home()` goes through the existing `fail()`; one from a finger move after shutdown was requested (App's shutdown `stop()` aborted it) is logged, not a gripper fault. | The hand's homing must run on the thread that owns finger moves. |

### 3.4 App, hints, signals

| File | Change | Why |
|---|---|---|
| `src/app.cpp`, `include/app.h` | Endpoints from link.yaml (via the loader). Bind failure text (old lines 265-266, 274-275): `cannot bind the C3 endpoint tcp://192.168.50.10:5630 (<zmq error>): is link.yaml pc.address this PC's ASUS_5g address (ip -4 addr)? is another franka_control running?`. Hints `rerun deploy run` → `restart control_deploy run` (line 150), `franka_control/build.sh` → `control_deploy/franka_control/build.sh` (line 143), `station.yaml` → `control_deploy/station.yaml` (line 223), "the launcher kills" → "control_deploy run kills" (line 203, app.h:18). | PLAN §8.1 hint texts. |
| same | SIGHUP joins SIGINT and SIGTERM in the blocked set and the `sigtimedwait` set (old lines 86-91, 317-327): closing the terminal shuts down cleanly. | The process now lives in a terminal for hours; the default SIGHUP action would kill it without settling. |
| same | Builds the `HelloReply` template (section 5.5) and passes it to `CommandServer`; passes the gripper to `Supervisor`. | |
| same | The shutdown stop of a homing move goes through `SharedState::stopHomingMove` and is re-sent every 100 ms while the move runs (old: once). | A stop sent just before the move's libfranka call records its epoch is lost; the move would run to its end. |
| same | With `--keyboard`: after the sockets are bound, starts the keyboard thread (section 4.3); the messaging loop drains `KeyQueue` each iteration (section 4.1); `q` calls `App::requestStop()`. The ready log line adds `keys: h home arm, p start pose, g home hand, s stop, q quit`. | PLAN §8.1. |
| same | `App(CommandLine, AppOverrides = {})`. `AppOverrides` is for tests only and never reachable from the command line: endpoints (e.g. `tcp://127.0.0.1:*`), `link_file`, `frames_file`, `keyboard_fd`. | Unit tests bind free ports (top §2.3) and feed keys through a pipe; old tests set endpoints through the station, which is no longer allowed. |
| `include/contract_limits.h`, `src/codec.cpp`, `include/command_server.h`, `include/config_loader.h`, `include/config.h` | Header comments that name `contracts/contracts/...`, `message_comms`, `deploy` STRUCTURE sections. | |

### 3.5 Not changed, on purpose

- No new C2 field, no new mode, no change to `c2.state/v1` or the existing C3 messages (top §3.4).
- The watchdog logic (old-S §7.1): only its limit's source changed. Keys never feed it.
- Homing motion, settle rule, homing timeout, the after-homing box check (old-S §7.2; key `p` adds the cone, 4.3).
- Shutdown order and budget (old-S §7.4): 3 s + 1.5 s, under the 5 s that `control_deploy run` waits before SIGKILL.
- `ConfigLoader`'s merge of the station's `franka_control:` section (old lines 29-53, 294-308).

## 4. The process

### 4.1 Threads

| Thread | Priority | Runs | Talks to the others only through | Status |
|---|---|---|---|---|
| control | real-time (`realtime: enforce`) | `ControlLoop`: torque sessions (1 kHz), homing moves | `SharedState` (wait-free parts) | unchanged except `run()` (4.4) |
| messaging | normal | `Supervisor`, `CommandServer` (C3), `StatePublisher` (C2), key dispatch | `SharedState`, `KeyQueue`; `Arm::requestStop()` (homing timeout, homing abort); `Gripper::stop()` (hand homing abort or timeout) | + keys, + hello, + aborts |
| gripper | normal | `GripperWorker::run`: finger moves and the hand's homing | `SharedState` | + hand homing |
| gripper-reader | normal | `GripperWorker::runReader`: width at 50 Hz | `SharedState` | unchanged |
| keyboard | normal | `KeyboardReader::run` (only with `--keyboard`) | `KeyQueue`; `App::requestStop()` for `q` | new |
| main | normal | `App`: start-up, waits for SIGINT/SIGTERM/SIGHUP or a stop request, shutdown | everything, at start and end only | + SIGHUP |

The messaging loop (old:franka_control/src/app.cpp:292-296) becomes:

```
while (!messaging_stop) {
  supervisor.tick();
  while (auto key = keys.pop()) supervisor.onKey(*key);      // new: h, p, g, s (q never reaches it)
  server->serviceOnce(std::min(publisher->nsUntilNext(), poll_ns));
  publisher->maybePublish();
}
```

A key is acted on within one iteration (≤ `messaging.poll_max_s`, 5 ms). While a C3 `home` reply is pending,
`serviceOnce` only polls `pollHome()` and sleeps until the next C2 is due (old:franka_control/src/command_server.cpp:43-51),
so keys are still served every iteration.

**SharedState** (old-S §3.2, old:franka_control/include/franka_control/shared_state.h): copied, plus these members.
None is touched by `ControlLoop::step` except the two `LoopCommand` fields, which it only copies (no allocation).

| New member | Writer → reader | Type | What |
|---|---|---|---|
| `LoopSnapshot::homing_aborted_generation` | control → messaging | uint64 in the snapshot | 0, or the `kHome` generation whose homing was aborted (no move, or a move stopped) |
| `requestHomingAbort()` / `homingAbortRequested()` / `clearHomingAbort()` | messaging sets; control reads and clears | `atomic<bool>` | "stop the homing now" |
| `requestHandHoming(gen)` / `handHomingRequested()` | messaging → gripper | `atomic<uint64_t>` | the newest hand-homing request; 0 = none |
| `reportHandHoming(gen, result)` / `handHomingReport()` | gripper → messaging | `atomic<uint64_t>` = gen × 4 + result | result `kDone` 1, `kAborted` 2 (a failure is a gripper fault, as now) |
| `LoopCommand::q_target`, `LoopCommand::open_width_m` (2026-11-20) | messaging → control | `optional<Vec7>`, `optional<double>` in the command | the `kHome` goal (nullopt: `q_home_rad`) and the finger target during the move (`p`: `gripper_width_max_m`) |

All atomics stay lock-free (`static_assert`, as old shared_state.h:73-75).

### 4.2 Start-up (`App::run`) and readiness

Each failing step ends the process with one `FATAL franka_control: <what failed>: <what to check>` line and exit 1
(old-S §7.3). Steps 4–6 and 8 are unchanged; the others change as marked.

1. **Config** (changed). Parse the command line. Load `config.yaml` merged with the station's `franka_control:`
   section, the station's `session_root`, and the three contract files from their baked paths: `limits.yaml`
   (`ContractLimits`), `frames.yaml` (`ContractFrame`), `link.yaml` (`ContractLink`). Endpoints from `pc.address`
   and the two ports, or `127.0.0.1` with `--local`. Every problem is listed at once; the FATAL hint names the file:
   `FATAL franka_control: link.yaml: pc.address: expected an IPv4 address, got "pc.local": fix link.yaml`.
2. **Session folder**: `--session DIR` → `DIR/franka_control/config_effective.yaml`; alone →
   `<session_root>/<YYYY-MM-DD_HH-MM-SS>_franka_control_alone/franka_control/`. Unchanged (old lines 103-116).
3. **Memory and signals**: `mlockall` with `realtime: enforce`; block SIGINT, SIGTERM and **SIGHUP** (new).
4. **Arm**: `--mock` → `FakeArm`; else `LibfrankaArm(robot_ip)`, the ready check (`describeArmProblem`: reflex, user
   stop, error flags → FATAL "press and release the user stop, check Desk, then restart control_deploy run"),
   collision behaviour, joint impedance, `setLoad` when `load.mass_kg > 0`, model. Never `automaticErrorRecovery`.
5. **Gripper**: `FakeGripper` or `FrankaHandGripper`; `prepare()` homes the fingers once when
   `gripper.homing_on_start` (PLAN §8.1 start model); the width becomes the target, so nothing moves.
6. **Control thread**: the first torque session holds the measured pose for `startup.hold_before_ready_s`.
7. **Sockets and threads** (changed): bind C2 PUB and C3 REP on the link.yaml endpoints (FATAL text in 3.4). Start the
   messaging, gripper and gripper-reader threads, and with `--keyboard` the keyboard thread (it first discards keys
   typed before this point: `tcflush(TCIFLUSH)` on a TTY).
8. **Ready**: the first C2 is published, the ready file is created (`--ready-file`), and the log says
   `ready: C2 on tcp://192.168.50.10:5620, C3 on tcp://192.168.50.10:5630; keys: h home arm, p start pose, g home hand, s stop, q quit`.
   From here C3 is answered and keys are acted on. The mode is `idle`; the arm holds where it was.

The laptop never sees the ready file; it decides readiness from C2 and `hello` (PLAN §8.2, top §8.2).

### 4.3 The keyboard (new)

```
enum class Key { kHomeArm, kStartPose, kHomeHand, kStop, kQuit };

class KeyboardReader {                       // keyboard thread
 public:
  KeyboardReader(int fd, KeyQueue& keys, std::function<void()> on_quit, const SharedState& shared);
  void run();                                // until shutdown, EOF or a read error
  static std::optional<Key> map(char c);     // the one key table: 'h', 'p', 'g', 's', 'q' and their upper case
};

class TerminalMode {                         // RAII; does nothing when fd is not a TTY
 public:
  explicit TerminalMode(int fd);             // cbreak: ICANON and ECHO off, ISIG kept; VSUSP and VQUIT disabled; IXON off
  ~TerminalMode();                           // restores the saved termios
};

class KeyQueue {                             // keyboard -> messaging; mutex (neither is the 1 kHz loop)
 public:
  void push(Key k);                          // h, p, g: dropped with a warning when 16 are waiting; s: never dropped
  std::optional<Key> pop();
};
```

- `run()` polls the fd every 50 ms (so it sees `shutdown` quickly), reads bytes, maps each one. Unknown bytes log
  `keys: h home arm, p start pose, g home hand, s stop, q quit` (at most once a second). Upper case acts like lower case, so with
  caps lock on `S` still stops (it would otherwise print the help line).
- `q`: calls `on_quit` (= `App::requestStop()`, the same path as SIGTERM). It is not queued.
- `p` (2026-11-20, start pose): from home only (every joint within `homing.at_pose_tol_rad` of `q_home_rad`), the
  same arm-homing path as `h` (`Supervisor::startArmMove`: one shared helper for bookkeeping, stop/abort, timeout,
  logging) to `homing.q_start_rad`; the kHome `LoopCommand` carries the joint target and an open width, and
  `ControlLoop::run()` sets the finger target to `gripper_width_max_m` just before the move (outside the 1 kHz path).
  At the end: TCP inside the box **and** tilt <= `safety.cone_max_deg` (no margin, like the box), else `fault` naming
  `homing.q_start_rad`. Refused while `q_start_rad` is unset. Once it is set, C3 `start` is refused unless every joint
  is within the tolerance of it (`not at the start pose: press h, then p on the PC`); unset, `start` is unchanged.
- `h`, `g`, `p`, `s`: pushed to `KeyQueue` (`s` always gets in, even when 16 keys wait); the messaging thread calls
  `Supervisor::onKey`, which logs one line with the
  result, e.g. `key h: homing the arm: 4.2 s joint move to homing.q_home_rad` or
  `key h refused: mode is running: the laptop is driving an episode: press s or stop it on the laptop first`.
- **Ctrl-C** stays SIGINT (ISIG kept), handled by the main thread as now. **Ctrl-Z** and **Ctrl-\\** are disabled
  in `TerminalMode` (a stopped process would miss FCI packets and trip a robot reflex). **Ctrl-S/Ctrl-Q** flow control
  (IXON) is off: a paused terminal would block the wrapper's terminal write, then, once the pipe fills, every log line
  of franka_control (while holding the log mutex), the messaging thread included.
- EOF (stdin closed, `/dev/null`, terminal hung up) or a read error: log `keyboard: stdin closed: keys disabled` and
  end the thread. Never treated as `q`.
- `TerminalMode` is restored when the thread ends, including after a FATAL. If the process is killed, `control_deploy
  run` restores the terminal (section 6.4).
- Keys do not feed the watchdog (they are not C3 requests).

**Extension point (PLAN §14, later):** jog keys are new entries in `KeyboardReader::map` and new `Key` values handled
in `Supervisor::onKey`. A jog would build a short plan from the current target and pass it through
`SafetyGate::check`, exactly as `set_waypoints` does. Which mode it may run in is decided then; nothing in this design
blocks it.

### 4.4 Homing: arm, hand, and stop during homing (changed)

**Arm homing** (`h` key or C3 `home`) is the verified path (old-S §7.2, old:franka_control/src/supervisor.cpp:187-203,
104-129): `Supervisor::onHome` checks the mode and the move's duration, posts `kHome`, the loop settles, ends the
torque session, runs `moveToJoints`, starts a new session holding the home pose; the Supervisor checks the home TCP
against the box and goes to `idle`. The only difference between the two sources: a C3 `home` gets its deferred `ack`
when the homing ends; a key gets a log line. **Key `p`** (2026-11-20) takes the same path (`Supervisor::startArmMove`,
`ArmGoal::kStartPose`) to `homing.q_start_rad`, from home only, the hand opening meanwhile (section 4.3); its
completion checks the box and also the cone (`safety.cone_max_deg`, no margin), else `fault`.

**Hand homing** (`g` key only; no C3 request exists for it):

1. Allowed from `idle` and `stopped` (the `kHome` rule); refused in `holding`/`running` (the laptop is driving),
   `homing` and `fault`, and when there is no gripper.
2. The Supervisor calls `shared.requestHandHoming(gen)` and goes to `homing` (kind: hand). No loop command: the 1 kHz
   loop keeps holding the arm, and C2 `t_ns` keeps advancing. C2 shows `homing`, so the laptop waits (PLAN §8.2) and
   a C3 `start` or `home` is refused, exactly as during arm homing.
3. The gripper thread calls `gripper.home()` (the Franka Hand closes and reopens to calibrate). C2 `gripper_width_m`
   shows the motion (the reader thread keeps reading).
4. Done → `reportHandHoming(gen, kDone)`; the Supervisor goes to `idle` and calls `markAlive(now)`. The gripper
   thread then re-sends the current target width (`resend_`), so **the fingers return to the commanded width**
   (decision 2, section 12).
5. A `GripperError` → gripper fault → `fault` (the existing rule). No answer within `homing.timeout_s` → the
   Supervisor stops the hand through `shared.stopHandHoming` (only if `home()` runs, see below) and goes to `fault`
   ("hand homing: not finished after 30.0 s").

**Stop during homing.** A C3 `stop` or the PC `s` key in `homing`:

- The reply is `ack ok` at once (never refused, top §3.7.2). The mode stays `homing` until the outcome is known
  (a few ms), so nobody sees `stopped` while the arm still moves. A C3 `stop` can only arrive during a
  keyboard-started homing: during a C3 `home` the REP socket receives nothing until it replies (top §3.7.2).
- **Arm homing:** `shared.requestHomingAbort()`. Then, in `tick()`, as soon as the snapshot shows this homing's move
  running (`phase == kHoming`, its session) and the abort is still pending, `arm.requestStop()`, sent under
  `shared.stopHomingMove`: the control thread marks the move (`beginHomingMove` before `moveToJoints`,
  `endHomingMove` after it returns or throws), and the stop runs under the same lock, so it is sent only while the
  move runs and can never land in the torque session that follows it. It is re-sent every 100 ms while the move runs
  and the outcome is unresolved. The same applies to the homing-timeout stop, which keeps being re-sent after the
  `fault` until the move is over (the loop ended, or a new session). The control thread:
  - if the abort is seen **before the move starts** (the session has just ended for `kHome`): no move; clear the
    flag; `homing_aborted_generation = home_generation`; start a new torque session (session + 1) holding the
    measured pose;
  - if the move is running: `robot.stop()` preempts it with a `ControlException` (libfranka 0.9.2 `robot.h:681-689`;
    `Move command preempted`), which arrives as `ArmError(kStopRequested)` because the epoch changed during the call.
    With the abort flag set and no shutdown: clear the flag; wait until `arm.status()` shows the robot out of its
    move mode, at most `control.settle_timeout_s` (1 s); if it shows a reflex, a user stop or error flags, end with
    a fault ("homing stopped, then the arm reports <problem>: press and release the user stop, then restart
    control_deploy run"); else `homing_aborted_generation = home_generation` and start a new torque session holding
    the measured pose;
  - without the abort flag (the homing timeout, or shutdown) the old handling stays: fault, or the expected end at
    shutdown (old:franka_control/src/control_loop.cpp:210-220).
- The Supervisor sees `homing_aborted_generation == home_generation`, a newer session and `phase == kTorque`:
  `kHomingAborted` → **`stopped`** (detail `stop during homing (PC keyboard)` or `(C3)`), posts `kHold`, calls
  `markAlive(now)`, and for a C3-started homing sets the deferred reply `ack {ok: false, reason: "stopped: homing
  stopped by the PC keyboard before the home pose was reached: the arm holds where it stopped"}`. The reason starts
  with exactly the contract prefix `stopped:` (top §3.7.2); no other `home` reply of this process starts with it
  (refusals start with `refused:`; failures and timeouts carry the fault text, e.g. `arm state not finite`, and a
  unit test checks that none of those texts starts with `stopped:`). Only the PC's `s` can stop a C3-started
  homing, since the REP socket receives nothing while that reply is pending.
- **The move finished first** (the stop came in its last milliseconds): the normal completion wins: `idle`, a C3
  `home` gets `ok`, the log says `stop arrived after the home pose was reached`. The control thread clears the abort
  flag after a normal finish. Both `idle` and `stopped` hold the pose and accept `start` and `home` (decision 3,
  section 12; `start` only at the start pose once `homing.q_start_rad` is set).
- **Hand homing:** a stop must never abort a finger move (libfranka throws on an aborted command: a gripper fault,
  restart). The gripper thread may still be in the previous finger move when `g` and `s` come (up to 1.6 s after an
  episode). So the stop goes through `shared.stopHandHoming(gen, stop)`, tied to the hand-homing request number:
  - `home()` runs for that request (the gripper thread called `beginHandHoming(gen)`, not yet `endHandHoming`):
    `gripper.stop()` under the lock; `home()` returns false → `kAborted` → `stopped`, as above;
  - `home()` has not begun: nothing is sent to the hand; the request is marked aborted, the gripper thread skips
    `home()` (the fingers never move) and reports `kAborted` → `stopped`; the log says `the hand's homing will not
    start`;
  - `home()` has ended: nothing is sent; the done report → `idle` (`stop arrived after the hand's homing had
    finished`).
  The gripper thread calls `clearStop()` before `beginHandHoming`, and `home()` never resets the stop flag, so a
  `stop()` in the instant between `beginHandHoming` and the hand's `homing()` call still skips the homing.
  Whether `franka::Gripper::stop()` interrupts `homing()` is checked at bring-up (section 11); if it does not, the
  hand finishes its homing (a few seconds) and the result is `idle`.
- **Resulting state: `stopped`, holding the pose where the arm (or the fingers) stopped**; the laptop can `start`
  from there (once `homing.q_start_rad` is set, only at the start pose), `h` can home again. A stopped `p` holds the
  measured finger width.
- C2 during an aborted arm homing: `t_ns` advances during the move, then repeats unchanged while the control thread
  waits for the robot (up to 1 s), then advances again with the new session (top §3.6 obligation 1, PLAN §5.3).
- **No stop in the next session:** a `robot.stop()` meant for a move that has just finished cannot preempt the new
  torque session: the control thread ends the move (`endHomingMove`) before it starts that session, and it waits for
  a stop being sent under the lock.
- **`q`, Ctrl-C, SIGTERM, SIGHUP during homing:** the old shutdown path (old:franka_control/src/app.cpp:187-210):
  `requestStop()` on the arm move (through `shared.stopHomingMove`, re-sent every 100 ms while the move runs),
  `gripper.stop()`, settle, exit 0.

### 4.5 Shutdown

Unchanged (old-S §7.4, old:franka_control/src/app.cpp:329-336), plus: the keyboard thread sees `shutdown`, restores
the terminal and ends within 50 ms; it is joined after the gripper threads. The budget stays 3 s + 1.5 s < 5 s.

## 5. C2 and C3 on the PC

### 5.1 States

The six C2 modes of old-S §4.1, unchanged in meaning. `homing` now has two kinds, invisible on C2: arm (the loop runs
the joint move) and hand (the loop holds the arm, the Franka Hand homes).

### 5.2 Transitions

Old-S §4.2 rows, then the new ones (**bold**). "Key" means the PC keyboard; "C3" the client.

| From | Trigger | To | Command to the loop / action |
|---|---|---|---|
| *(start-up)* | the loop has held for `startup.hold_before_ready_s` | `idle` | `kHold` |
| `idle`, `stopped` | C3 `start` (**once `homing.q_start_rad` is set: every joint within `homing.at_pose_tol_rad` of it, else refused**) | `holding` | `kFollow`, plan = [current target, now] |
| `holding`, `running` | C3 `set_waypoints`, ≥ 1 accepted | `running` | `kFollow`, new plan |
| `holding`, `running` | C3 `set_waypoints`, all skipped | `holding` | `kFollow`, [current target, now] |
| `running` | now ≥ the plan's last time | `holding` | none |
| `idle`, `holding`, `running` | C3 `stop` **or key `s`** | `stopped` | `kHold` |
| `holding`, `running` | watchdog (no C3 request for `watchdog_limit_s`, or the loop's backstop at + `backstop_extra_s`) | `stopped` | `kHold` |
| `holding`, `running` | `set_waypoints` with a rejected waypoint | `fault` | `kHold` |
| `idle`, `stopped` | C3 `home` **or key `h`** | `homing` (arm) | `kHome` |
| **`idle`, `stopped`** | **key `p`, at home (`homing.at_pose_tol_rad`), `homing.q_start_rad` set** | **`homing` (arm)** | **`kHome` with `q_target` = `q_start_rad`, `open_width_m` = `gripper_width_max_m`** |
| **`idle`, `stopped`** | **key `g`** | **`homing` (hand)** | **none; hand-homing request to the gripper thread** |
| `homing` (arm) | the move finished, the new session holds, home TCP inside the box (**`p`: and tilt ≤ `safety.cone_max_deg`**) | `idle` | `kHold`; `markAlive`; C3 reply ok |
| `homing` (arm) | home TCP outside the box (**`p`: or tilt > `safety.cone_max_deg`**), the move failed, or `homing.timeout_s` passed | `fault` | `kHold` / none; timeout: `requestStop()`; C3 reply not ok |
| **`homing` (hand)** | **the gripper reports done** | **`idle`** | **`markAlive`** |
| **`homing` (hand)** | **`homing.timeout_s` passed** | **`fault`** | **`gripper.stop()`** |
| **`homing` (either)** | **C3 `stop` or key `s`** | **`homing` (until the outcome)** | **abort: arm → flag + `requestStop()` once the move runs; hand → `gripper.stop()`** |
| **`homing` (arm)** | **the loop reports the abort and holds in a new session** | **`stopped`** | **`kHold`; `markAlive`; C3 reply not ok, reason `stopped: ...`** |
| **`homing` (hand)** | **the gripper reports aborted** | **`stopped`** | **`markAlive`** |
| any but `fault` | the loop reports a fault | `fault` | none |
| any but `fault` | the gripper reports an error | `fault` | `kHold` |
| `fault` | anything (C3 or key) | `fault` | none |
| **any** | **key `q`, Ctrl-C, SIGTERM, SIGHUP** | **(process exits)** | **shutdown (4.5)** |

Everything else is refused and changes nothing. Keys refused by the mode log the reason; refusals never change state.

### 5.3 Replies per mode (C3) and keys per mode

C3, identical to top §3.7.2 (a C3 `stop` aborts a keyboard homing; a C3 `home` stopped by the PC's `s` replies
`ok: false` with the `stopped:` prefix):

| Request | `idle` | `holding` | `running` | `stopped` | `homing` (arm or hand) | `fault` |
|---|---|---|---|---|---|---|
| `hello` | reply | reply | reply | reply | reply | reply |
| `get_state` | reply | reply | reply | reply | reply | reply |
| `heartbeat` | ok | ok | ok | ok | ok | ok |
| `start` | ok → `holding` | refused | refused | ok → `holding` | refused | refused |
| `stop` | ok → `stopped` | ok → `stopped` | ok → `stopped` | ok | ok; aborts the move → `stopped`; if the move finished first → `idle` (5.2) | ok, stays `fault` |
| `home` | → `homing`, reply at the end | refused | refused | → `homing`, reply at the end | refused | refused |
| `set_waypoints` | `error/v1` | gate | gate | `error/v1` | `error/v1` | `error/v1` |
| broken, unknown, laptop-only, or a reply | `error/v1` | `error/v1` | `error/v1` | `error/v1` | `error/v1` | `error/v1` |

Refusal reasons are the state machine's one line (old:franka_control/src/state_machine.cpp:28-39), e.g.
`refused: mode is homing: homing in progress`. A laptop-only schema is `unknown_schema` (top §3.4). Once
`homing.q_start_rad` is set, `start` in `idle`/`stopped` is also refused unless every joint is within
`homing.at_pose_tol_rad` of it: `refused: not at the start pose: press h, then p on the PC` (2026-11-20).

PC keyboard:

| Key | `idle` | `holding` | `running` | `stopped` | `homing` | `fault` |
|---|---|---|---|---|---|---|
| `h` | → `homing` (arm) | refused | refused | → `homing` (arm) | refused | refused |
| `p` | → `homing` (arm) at home; refused while `homing.q_start_rad` is unset, away from home (`not at home: press h first`) or if the move would exceed `homing.timeout_s` | refused | refused | as `idle` | refused | refused |
| `g` | → `homing` (hand) | refused | refused | → `homing` (hand) | refused | refused |
| `s` | → `stopped` | → `stopped` | → `stopped` | no change | aborts → `stopped` | no change ("Ctrl-C and restart after a fault") |
| `q` | exit | exit | exit | exit | exit (the move is stopped) | exit |

A key `s` during an episode looks to the laptop exactly like a watchdog stop (C2 `stopped`, no `error`); the laptop
concludes with "the PC stopped the arm (watchdog or the PC's `s`): check the link and the PC" (PLAN §7). Between
episodes the keys may move the PC to `stopped` or `homing`; the laptop follows C2 (PLAN §7, top §8.6).

### 5.4 Watchdog

Unchanged logic (old-S §7.1, old:franka_control/src/supervisor.cpp:152-162, control_loop.cpp:123-132):

- Limit `link.yaml timing.watchdog_limit_s` (1.0 s), floor 0.5 s in the loader; the loop's backstop at
  limit + `watchdog.backstop_extra_s` (1.1 s).
- **Armed** in `holding` and `running` only. Off in `idle`, `stopped`, `fault`. In `homing` it is off too (homing
  starts only from `idle`/`stopped`), and every end of a homing (done, aborted) calls `markAlive(now)`, so it
  restarts from there.
- **Fed** by every C3 request that decodes and is a C3 request: `hello`, `heartbeat`, `get_state`, `start`, `stop`,
  `home`, `set_waypoints` (old:franka_control/src/command_server.cpp:77-79). A broken request does not feed it. Keys
  never feed it.
- **Trip**: `stopped` and `kHold`; the arm finishes at most the plan already accepted (≤ `max_lead_s`), then holds.
  Log: `stopped: watchdog: no request for 1.004 s`.
- `hello` feeding the watchdog is why top §3.7.3 rule H1 makes the laptop send it only from its main loop.

### 5.5 `hello`: reply contents

Built once in `App` after the config is loaded (the template); only `t_pc_ns` is set per request.

| Field | Value on the PC | Source |
|---|---|---|
| `t_pc_ns` | `clock_()` (`CLOCK_MONOTONIC`, the C2 `t_ns` clock), the last step before `encode` | `CommandServer::handle` |
| `commit` | 40 lowercase hex, or `unknown` (no git at build time) | `build_info.h kGitCommit` (section 3.1) |
| `mock` | true with `--mock` | `CommandLine::mock` |
| `schemas` | the 12 `SHARED_SCHEMAS`, sorted as whole strings | `Codec::knownSchemas()` |
| `F_T_TCP` | the re-normalised pose, row-major | `Config::F_T_TCP` ← `frames.yaml` |
| `rotation_tol`, `gripper_width_max_m`, `max_waypoints` | as loaded | `Config::limits` ← `limits.yaml` |
| `watchdog_limit_s` | as loaded | `Config::watchdog.limit_s` ← `link.yaml timing.watchdog_limit_s` |
| `homing_timeout_s` | effective (a station override shows here) | `Config::homing.timeout_s` ← `config.yaml homing.timeout_s` |
| `max_lead_s` | effective | `Config::safety.max_lead_s` ← `config.yaml safety.max_lead_s` (1.5) |

The template is encoded once at start-up as a self-check: a value that would break the contract is a FATAL at
start-up, never a broken reply later.

### 5.6 C2

`StatePublisher` unchanged (old-S §8.1). The PC obligations of top §3.6 hold; the one wording change is in
section 4.4 (repeated `t_ns` after an aborted homing).

## 6. The Python wrapper `control_deploy`

### 6.1 Commands

```
control_deploy check    [--mock] [--local] [--station FILE]   # the table of section 6.3; exit 2 on any fail
control_deploy run      [--mock] [--local] [--station FILE]   # check, then franka_control on this terminal (6.4)
control_deploy console  [--mock] [--local] [--station FILE] [--only STEP ...] [--yes]   # bring-up C3 client (6.5)
control_deploy firewall                                        # prints the nftables rule for link.yaml (6.6)
```

- `--station` defaults to `control_deploy/station.yaml` (`console` reads it for `session_root`, its step sizes and the
  workspace-box overrides).
- `--mock`: `run` starts the binary with `--mock` (FakeArm, FakeGripper); `check` skips the arm, real-time and
  payload rows; `console` expects `hello.mock == true`.
- `--local`: bind (`run`) or connect (`console`) on `127.0.0.1` with the link.yaml ports; `check` skips the address
  and firewall rows. For PC-only tests and for `run --mock --local` on any machine.
- The wrapper imports only the standard library, `yaml`, `numpy`, `zmq`, `msgpack` and `deploy_contracts` (a test
  enforces it). It never imports or reads `policy_deploy/`.

### 6.2 The PC's station file: `control_deploy/station.yaml`

```yaml
# This PC's settings (PLAN rule 1). Ports and addresses are in deployment/contracts/link.yaml, never here.
station_version: 1
session_root: ~/control_deploy_sessions     # one folder per run and per console session
franka_control:                             # overrides of franka_control/config.yaml keys (not libfranka_version)
  realtime: ignore                          # TODO: enforce once the PC runs a PREEMPT_RT kernel (PLAN §13)
console:                                    # the bring-up console's step sizes (6.5)
  hold_s: 3.0
  move_m: 0.015
  rotate_deg: 2.0
  gripper_delta_m: 0.02
  segment_s: 2.0
  out_of_box_m: 0.03
```

| Key | Read by | Rule |
|---|---|---|
| `station_version` | wrapper | must be 1 |
| `session_root` | binary (required, as now) and wrapper | `~` expanded; relative to the station's folder |
| `franka_control` | binary (merge, old rule) | any `config.yaml` key except `libfranka_version`; `tcp.*` and `watchdog.limit_s` refused with the "moved to" hint |
| `console.*` | wrapper | floats > 0; `move_m` ≤ 0.03, `rotate_deg` ≤ 10, `out_of_box_m` in [0.02, 0.05] |

The binary reads only `session_root` and `franka_control`, and refuses `endpoints`; the wrapper's loader is strict
(unknown keys refused, naming the key).

### 6.3 `check`: the rows

Same table format and exit code as old:deploy/deploy/check_result.py (`process  check  STATUS  detail`; exit 2 on any
fail). Never moves the arm, never connects to FCI. Values the station may override (`robot_ip`, `realtime`,
`load.mass_kg`) are read from the binary's `--check-config` output, so the wrapper never re-implements the merge.

| Row | When | Pass | Fail / warn (detail) |
|---|---|---|---|
| `env` | always | `deploy_contracts.paths.contracts_dir()` is this checkout's `deployment/contracts` | fail: "deploy_contracts comes from <path>: uv pip install -e deployment/contracts into ~/envs/control_deploy" |
| `contracts` | always | `ContractLimits`, `TcpFrame`, `LinkSettings` load | fail: the loader's message (file and key) |
| `station` | always | station.yaml valid | fail: the key and problem |
| `binary` | always | `franka_control/build/franka_control` exists, executable | fail: "build it: control_deploy/franka_control/build.sh" |
| `version` | always | `--version`: `libfranka` = config.yaml `libfranka_version`; `commit` = `git rev-parse HEAD` | fail: "built from <a>, checkout is <b>: rebuild after every pull: build.sh"; libfranka mismatch or `none`: fail ("rebuild: build.sh"), info with `--mock`; `commit=unknown`: warn; `dirty=yes` or the checkout now dirty in `contracts/` or `control_deploy/`: warn "the laptop's hello check sees only the commit" |
| `baked paths` | always | the `contracts:` paths in `--check-config` equal this checkout's | fail: "the checkout moved since the build: delete franka_control/build* and rerun build.sh" |
| `config` | always | `--check-config --station S [--mock] [--local]` exits 0 | fail: first problem line, "(N problems)" |
| `tcp frame` | always | — | info: "frames.yaml settled: false: TCP pending PLAN §9 (fine for hold still and console)"; pass when true |
| `todo` | not `--mock` | no line of config.yaml still marked `TODO` (unless the station overrides that key) | warn: "still shipped defaults: safety.workspace_min_m, ..., load.mass_kg: measure them (PLAN §12)" |
| `payload` | not `--mock` | `load.mass_kg > 0` | warn: "no payload set: the rig's weight is not compensated (the arm sags a few mm)" |
| `link address` | not `--local` | `pc.address` is on an interface (`ip -j -4 addr`) | fail: "link.yaml pc.address 192.168.50.10 is not on this PC (wlp2s0 has 192.168.50.37): reserve the address on ASUS_5g or update link.yaml" |
| `ports` | always | both ports free on the bind address | fail: "tcp://...:5630 in use: is control_deploy run already running? (ss -ltnp \| grep 56)" |
| `firewall` | not `--local` | the `inet control_deploy` table exists and matches link.yaml (6.6) | fail: absent or different ("print it with control_deploy firewall"); warn: "cannot read the firewall without root: compare sudo nft list table inet control_deploy with control_deploy firewall"; warn: "ufw is enabled: also allow the laptop there" (`/etc/ufw/ufw.conf` ENABLED=yes) |
| `fci network` | not `--mock` | an interface has an address in `172.16.0.0/24` | fail: "no 172.16.0.x address: sudo nmcli con up franka" |
| `arm reachable` | not `--mock` | TCP connect to `robot_ip:1337` within 2 s | fail: "arm not reachable at 172.16.0.2:1337: check the cable, the PC's static IP, FCI enabled in Desk" |
| `rt kernel`, `rtprio` | not `--mock`, `realtime: enforce` | `/sys/kernel/realtime` = 1; RLIMIT_RTPRIO ≥ 99 | fail, texts of old:franka_control/launch.py:102-118 with `control_deploy/station.yaml` |
| `realtime` | not `--mock`, `realtime: ignore` | — | warn: "real-time ignore mode: expect occasional reflexes; late cycles show on the status line" |
| `session root` | always | writable | fail |
| `backups` | always | — | info: "arm: LibfrankaArm at 172.16.0.2, realtime ignore; binds 192.168.50.10" or "arm: FakeArm, gripper: FakeGripper (--mock); binds 127.0.0.1" |

The rows `version`, `config`, `ports`, `arm reachable`, `rt kernel`, `rtprio`, `realtime`, `payload`, `backups` are
ported from old:franka_control/launch.py:42-118 (tests from old:franka_control/tests/test_launch.py).

### 6.4 `run`

1. Runs every check with the same flags. Any fail: prints the table, exit 2, nothing started. Warnings are printed and
   do not stop it.
2. Creates the session folder `<session_root>/<YYYY-MM-DD_HH-MM-SS>_control_deploy[_mock]/`:

   | Path | Written by | What |
   |---|---|---|
   | `run.yaml` | wrapper | start time, flags, commit, dirty, the check rows, the bind endpoints |
   | `station.yaml`, `contracts/limits.yaml`, `contracts/frames.yaml`, `contracts/link.yaml` | wrapper | copies, as they were for this run |
   | `logs/franka_control.log` | wrapper | every line the binary writes (stderr and stdout) |
   | `logs/control_deploy.log` | wrapper | its own lines (ready, exit, signals) |
   | `franka_control/config_effective.yaml` | binary (`--session`) | the merged config, unchanged mechanism |
   | `ready` | binary | the ready file |

3. Starts `franka_control --station S --session DIR --ready-file DIR/ready --keyboard [--mock] [--local]` with
   **stdin inherited** (the terminal: the binary reads the keys itself) and stdout + stderr on one pipe.
4. Copies each line of the pipe to `logs/franka_control.log` and to the terminal. On a TTY it keeps one status line
   at the bottom: before printing a log line it clears the status line (`\r\033[K`), prints the line, redraws.
5. Waits for the ready file, at most 90 s (FCI connect + finger homing + the hold). Then prints
   `control_deploy: ready: C2 tcp://192.168.50.10:5620, C3 tcp://192.168.50.10:5630; start policy_deploy on the
   laptop. Keys: h home arm, p start pose, g home hand, s stop, q quit`. Not ready after 90 s: SIGTERM, and the old READY_HINT
   ("check robot_ip, FCI enabled in Desk, brakes unlocked, user stop released, and the log").
6. **Status line**, redrawn 4 times a second from a C2 `Subscriber` (deploy_contracts) on the bound C2 endpoint, plus
   the C3 peers from `ss -Htn state established '( sport = :5630 )'`:
   `idle | tcp 0.307 0.001 0.354 m | grip 0.079 m | gate a120 s3 p0 r0 | late 0 (+0 in 10 s) | C2 100 Hz | C3 peer 192.168.50.20 | h p g s q`
   (136 columns). Over 120 columns it is compacted only as far as needed: `late 0 (+0/10s)`, then `|` without
   spaces, then no tcp unit and `C2 100Hz`. In `fault` it shows the error text. No C2 for 1 s while the binary runs: `no C2: see the log`.
7. **Signals.** The wrapper ignores SIGINT while the child runs: Ctrl-C reaches the binary through the terminal's
   process group, and the binary shuts down itself. SIGTERM or SIGHUP to the wrapper: forwarded as SIGTERM, then
   SIGKILL after 5 s (the old launcher's `stop_timeout_s`, which the binary's 4.5 s budget is built for).
8. **Exit.** Restores the terminal settings it saved before step 3 (in case the binary was killed). Prints one line:
   `control_deploy: franka_control stopped cleanly (session ...)` (exit 0), or after the binary's FATAL line
   `control_deploy: franka_control failed: see <session>/logs/franka_control.log` (exit 1).

Without a TTY on stdin (tests, scripts): no status line, keys come from whatever stdin is (EOF disables them).

### 6.5 `console` (PLAN §8.1)

A C3 client on the PC, run in a second terminal while `control_deploy run` shows `idle` or `stopped`, **never while
the laptop runs**. It is the first tool that exercises waypoint following on the real arm. It runs on the PC, so it
times waypoints with `time.monotonic_ns()`, the PC's `CLOCK_MONOTONIC`, with no clock map. Once `homing.q_start_rad` is
set, press `h`, then `p` in `run` first: the console's `start` is refused away from the start pose, as the laptop's is.

**Preflight** (any refusal: one line, exit 2, nothing sent on C3):

1. C2: subscribe; no message within 2 s → "franka_control not running (no C2 on tcp://...:5620): start control_deploy
   run first".
2. C3 peers: `ss` lists an established connection to port 5630 → "a C3 client from 192.168.50.20 is connected
   (policy_deploy on the laptop?): stop it first". Stricter than top §3.7.3 H2, possible only on the PC.
3. Rule H2 (top §3.7.3): while C2 shows `running`/`holding`, send nothing for up to 2 × `watchdog_limit_s`; still
   so → "the PC is driving an episode for another C3 client". `homing` → wait up to `homing_timeout_s`. `fault` →
   "franka_control is in fault: <error>: Ctrl-C it and run control_deploy run again".
4. `hello` (timeout `c3_timeout_s`): `compare_hello(reply, mock=--mock)` (top §3.9) must be empty, else each
   mismatch is printed. `|t_pc_ns − midpoint of the round trip| ≤ 5 ms`, else "the console must run on the PC" (its
   waypoint times use the local clock).
5. Reads `franka_control/config.yaml` and the station (same folder) for the workspace box, so it can place the
   out-of-box waypoint and warn if the current TCP is within 5 cm of a box face.

**The loop.** One C3 request every `heartbeat_period_s` (20 ms): `set_waypoints` when one is due, else `heartbeat`
(one request per cycle, as the manager does). C2 is drained every cycle. Waypoints are streamed like the manager's:
every 0.2 s a `set_waypoints` with the next 8 waypoints, 50 ms apart, the first at least 100 ms ahead, so no waypoint
is more than about 0.55 s ahead (`max_lead_s` 1.5). Profiles are minimum-jerk (`s = 10τ³ − 15τ⁴ + 6τ⁵`), out over
`segment_s`, hold, back over `segment_s`. Every reply is checked (`accepted + skipped + len(rejected) == N`, nothing
rejected, `shortened == 0`; peak speeds are far below the gate's limits: 0.028 m/s, 0.16 rad/s).

**Steps** (in this order; before each moving step it prints what will move and waits for Enter, so a hand is on the
user stop; `--yes` skips the prompts and is refused without `--mock`; `--only STEP` runs a subset, `start` first):

| Step | Sends | Success |
|---|---|---|
| 1 `hold` | `start`; heartbeats for `hold_s` (3 s) | `ack ok`; C2 `holding` within 0.1 s; TCP drift ≤ 2 mm and ≤ 1° over the hold; no new `loop_late` (warn otherwise) |
| 2 `move` | waypoints: TCP +`move_m` (1.5 cm) along base z (up, away from the table), out and back | **plan check**: C2 `O_T_TCP_target` matches the sent path at the same `t_ns` within 0.5 mm (software: gate, plan, interpolation); **tracking**: \|`O_T_TCP` − `O_T_TCP_target`\| ≤ 3 mm pass, ≤ 10 mm warn (gains, payload), > 10 mm fail; C2 `running`, then `holding` at the end |
| 3 `rotate` | waypoints: `rotate_deg` (2°) about the TCP z axis (no tilt change, so the cone is untouched), out and back | target matches within 0.05°; tracking ≤ 1° pass, ≤ 3° warn |
| 4 `gripper` | waypoints at the current pose, width − `gripper_delta_m` (2 cm; + if the width is under 3 cm), hold 1.5 s, back | C2 `gripper_width_m` within 7 mm of the command after the hold (deadband 5 mm; `width_offset_m` unmeasured → warn, not fail) |
| 5 `watchdog` | nothing at all (C2 only) | C2 `stopped` between `watchdog_limit_s` and + 60 ms after the last request; TCP drift ≤ 2 mm; then `start` again → `holding` |
| 6 `out_of_box` | one `set_waypoints`: the current pose with z = `workspace_max_m.z` + `out_of_box_m` (3 cm above the box: beyond the 1 cm margin, moving away from the table if the gate failed), 1 s ahead | reply `accepted 0`, one `rejected` with "outside workspace box"; C2 `fault` with "safety gate rejected"; TCP drift ≤ 2 mm over 1 s. If anything was accepted: `stop` at once and FAIL in capitals |

After step 6 franka_control is in `fault` **by design**: the console says "Ctrl-C control_deploy run and start it
again". Without step 6 the console ends with `stop` (mode `stopped`). Ctrl-C in the console sends `stop` and exits.
The console writes `<session_root>/<time>_control_deploy_console/` with `console.log` and `results.yaml` (each step:
pass/warn/fail and its numbers), the record of the bring-up.

### 6.6 `firewall`

Prints, from link.yaml, the nftables table the operator installs once (DIRECTIONS):

```
table inet control_deploy {
  chain input {
    type filter hook input priority filter; policy accept;
    tcp dport { 5620, 5630 } iif "lo" accept                       # the PC itself: status line, console
    tcp dport { 5620, 5630 } ip saddr { 192.168.50.20 } accept     # link.yaml laptop.addresses
    tcp dport { 5620, 5630 } drop
  }
}
```

The loopback line is needed: the status line and the console connect to `pc.address` from the PC itself, which
Linux delivers over `lo` (top §2.4, PLAN §6). `check` reads `nft -j list table inet control_deploy` (without sudo,
then `sudo -n`), and compares the accepted source addresses, the ports and the final drop with link.yaml. It never
changes the firewall.

## 7. Config: `franka_control/config.yaml` after the changes

Every key, with its default. `TODO` marks a value that needs the hardware (`check` warns while the marker is there).
Meanings are as in old-S §10 unless noted.

```yaml
# franka_control: every setting, with its default (control_deploy/STRUCTURE.md section 7).
# control_deploy/station.yaml's `franka_control:` section may override any key except libfranka_version: maps merge
# key by key, other values are replaced, an unknown key is an error. F_T_TCP is in contracts/frames.yaml, the
# watchdog limit, addresses and ports in contracts/link.yaml. Lines marked TODO need values measured on the hardware.

binary: build/franka_control          # the built program (used by control_deploy); relative to this folder
robot_ip: 172.16.0.2                  # the arm's FCI address (also the hand's) on the direct Ethernet link
realtime: enforce                     # enforce | ignore (station.yaml sets ignore until PREEMPT_RT, PLAN §13)
libfranka_version: "0.9.2"            # THE build setting for libfranka (build.sh and CMake read it); not for station.yaml
control_cpu: null                     # pin the control thread to this CPU core (optional)

control:                              # the 1 kHz loop (old-S §5)
  stiffness_translational_n_per_m: [800.0, 800.0, 800.0]    # TODO: tune 600-1000 on the arm
  stiffness_rotational_nm_per_rad: [40.0, 40.0, 40.0]       # TODO: tune 30-50 on the arm
  damping_ratio: 1.0
  nullspace_stiffness: 0.5
  jacobian_pinv_damping: 0.2
  max_position_error_m: 0.03
  max_rotation_error_rad: 0.2
  torque_max_nm: [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
  torque_rate_max_nm_per_s: 1000.0
  libfranka_limit_rate: true
  libfranka_cutoff_frequency_hz: 100.0
  collision_torque_thresholds_nm: [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]   # TODO: check on the arm (no reflexes)
  collision_force_thresholds_n: [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
  joint_impedance: [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]
  settle_speed_rad_per_s: 0.02
  settle_timeout_s: 1.0               # also the longest wait for the robot after a stopped homing (4.4)

load:                                 # payload beyond the hand configured in Desk (printed fingers, Pi 5, cameras)
  mass_kg: 0.0                        # TODO: weigh the rig (0 = no setLoad call; control_deploy check warns)
  com_m: [0.0, 0.0, 0.0]              # TODO: centre of mass in the flange frame (from CAD)
  inertia_kg_m2:                      # about the centre of mass
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]

safety:                               # the gate (old-S §6); defaults from main-sim bridge.yaml
  workspace_min_m: [0.20, -0.40, -0.005]    # TODO: measure around the table; z: 5 mm below the TCP z with the
                                            # fingertips on the table (base on the table top: about 0), grasps end there;
                                            # x: demos pull back toward the base after the grasp (sim: to 0.198 from home)
  workspace_max_m: [0.80, 0.40, 0.70]       # TODO: measure around the table
  cone_axis: [0.0, 0.0, -1.0]               # the hand's approach axis at home (normalised by the loader)
  cone_max_deg: 60.0                        # TODO: check around the table
  box_margin_m: 0.01
  cone_margin_deg: 2.0
  v_max_m_per_s: 0.25                       # raise to 0.5 only after tuning on the arm
  w_max_rad_per_s: 1.0
  slack_s: 0.010
  max_lead_s: 1.5                           # a waypoint further ahead is rejected (PLAN §5.4; reported in hello)

watchdog:                             # the limit itself is contracts/link.yaml timing.watchdog_limit_s (1.0 s)
  backstop_extra_s: 0.1               # the 1 kHz loop's own check fires at the limit + this

homing:
  q_home_rad: [0.0, -0.7853982, 0.0, -2.3561945, 0.0, 1.5707963, 0.7853982]   # TODO: check the home TCP is in the box with the rig
  max_joint_speed_rad_per_s: 0.3
  min_duration_s: 2.0
  timeout_s: 30.0                     # arm and hand homing; reported in hello (the laptop's home timeout must exceed it)
  q_start_rad: null   # TODO: measure on the arm (hand-guide to the demos' start, read with pc_probe --start-pose); null: p refused
  at_pose_tol_rad: 0.05               # every joint: "at home" for p, "at the start pose" for C3 start (once q_start_rad is set)

gripper:                              # max opening: gripper_width_max_m in contracts/limits.yaml
  speed_m_per_s: 0.05
  grasp_force_n: 20.0
  deadband_m: 0.005
  poll_period_s: 0.02
  read_rate_hz: 50.0                  # the width read into C2, also during a finger move or the hand's homing
  width_offset_m: 0.0                 # TODO: measure with the printed fingers (fingertip opening - hand reading)
  homing_on_start: true               # the Franka Hand homes once at start-up (PLAN §8.1)

messaging:
  publish_rate_hz: 100.0
  state_send_hwm: 100
  poll_max_s: 0.005                   # also the longest delay before a key is acted on
  stats_period_s: 1.0

startup:
  hold_before_ready_s: 0.2
  loop_start_timeout_s: 10.0

shutdown:
  timeout_s: 3.0                      # + 1.5 s after requestStop, below control_deploy run's 5 s before SIGKILL

loop_warning:
  success_rate_min: 0.95
  late_period_ms: 1.5

fake_arm:                             # only with --mock and in tests
  home_O_T_TCP:                       # TCP z pointing down
    - [1.0, 0.0, 0.0, 0.45]
    - [0.0, -1.0, 0.0, 0.0]
    - [0.0, 0.0, -1.0, 0.45]
    - [0.0, 0.0, 0.0, 1.0]
  mass_kg: 1.0
  inertia_kg_m2: 1.0

fake_gripper:
  start_width_m: 0.04                 # also where the fake's home() ends
  speed_m_per_s: 0.1
  object_width_m: null                # fingers stop here when closing (null: no object)
```

Read from elsewhere: `session_root` (station), `rotation_tol`, `gripper_width_max_m` (TODO there: max fingertip
opening), `max_waypoints` (limits.yaml), `F_T_TCP` (frames.yaml, pending PLAN §9), `pc.address`, `ports.*`,
`timing.watchdog_limit_s` (link.yaml). `ConfigLoader` checks are those of old-S §10 minus `tcp`/`watchdog.limit_s`,
plus the contract files' rules (3.1).

## 8. Build

### 8.1 What the PC needs

| What | How |
|---|---|
| Ubuntu 20.04 (the PC), sudo, internet for the first build | the first configure downloads msgpack-cxx 6.1.1 (FetchContent, old:franka_control/CMakeLists.txt:68-78) |
| apt build packages | installed by `build.sh` (old line 60): `build-essential cmake git pkg-config libpoco-dev libeigen3-dev libzmq3-dev libyaml-cpp-dev googletest` (system yaml-cpp, cppzmq header and GoogleTest sources, as the 91b178b build used) |
| libfranka 0.9.2 | built by `build.sh` into `~/franka/libfranka-0.9.2` (sources in `~/franka/src`); **already there on the PC**, so the step is skipped |
| Python 3.10 env | `uv venv --python 3.10 ~/envs/control_deploy`; `uv pip install --python ~/envs/control_deploy -e deployment/contracts[test] -e deployment/control_deploy[test]` |
| `nftables`, `iproute2` (`ip`, `ss`) | apt (`nftables` if missing) |
| A PREEMPT_RT kernel | later (PLAN §13); until then `realtime: ignore` in the station |

`pyproject.toml`: `requires-python >=3.10,<3.11`; dependencies `numpy>=1.26,<2.3`, `msgpack==1.1.0`,
`pyzmq==27.1.0`, `pyyaml==6.0.3` (the same pins as deploy_contracts); `deploy-contracts` is **not** listed: it is
installed separately, editable, in the same command (the rule of top §1.4 for both folders; the `env` row of `check`
fails without it); extra `test`: `pytest==8.3.5`; console script `control_deploy = control_deploy.cli:main`.

### 8.2 `build.sh` and CMake

```
control_deploy/franka_control/build.sh [--dry-run] [--no-apt] [--jobs N] [--prefix DIR] [--any-version] [--test]
```

Behaviour unchanged (old:franka_control/build.sh:1-102): reads `libfranka_version` from config.yaml (refuses outside
0.9.x without `--any-version`), apt packages, libfranka if missing, configures `build/` (Release,
`-DFRANKA_CONTROL_WITH_LIBFRANKA=ON -DFRANKA_CONTROL_BUILD_TESTS=OFF`), builds, prints `--version`; `--test` also
builds `build-test/` (ASan + UBSan) and runs CTest.

CMake options (unchanged, plus `CONTRACTS_DIR`): `FRANKA_CONTROL_WITH_LIBFRANKA` AUTO/ON/OFF,
`FRANKA_CONTROL_SANITIZE`, `FRANKA_CONTROL_TSAN`, `FRANKA_CONTROL_BUILD_TESTS`, `CONTRACTS_DIR`. Without libfranka
(`-DFRANKA_CONTROL_WITH_LIBFRANKA=OFF`) the libfranka classes are compiled out (old:franka_control/CMakeLists.txt:80-124,
`FRANKA_CONTROL_HAS_LIBFRANKA`), the binary runs only with `--mock` (without it: FATAL "built without libfranka") and
`--version` reports `libfranka=none`. That is the laptop's build for the cross test (8.3).

**Baked-in absolute paths** (compile definitions): `FRANKA_CONTROL_CONFIG_FILE`, `CONTRACTS_LIMITS_FILE`,
`CONTRACTS_FRAMES_FILE`, `CONTRACTS_LINK_FILE`, `CONTRACTS_EXAMPLES_DIR`, plus `build_info.h`. The yaml files are
read at run time (edit, then restart franka_control; no rebuild). **Moving or re-cloning the checkout needs a
rebuild**: delete `build/`, `build-test/`, `build-tsan/` (their CMake caches hold the old paths) and rerun `build.sh`.
**Rebuild after every pull** (top §7.2): `check` fails on a commit mismatch.

Test builds (the binary that drives the arm is never a sanitizer build):

```
cmake -S control_deploy/franka_control -B control_deploy/franka_control/build-test -DFRANKA_CONTROL_SANITIZE=ON \
      -DCMAKE_PREFIX_PATH=~/franka/libfranka-0.9.2 \
  && cmake --build control_deploy/franka_control/build-test && ctest --test-dir control_deploy/franka_control/build-test --output-on-failure
cmake -S control_deploy/franka_control -B control_deploy/franka_control/build-tsan -DFRANKA_CONTROL_TSAN=ON \
  && cmake --build control_deploy/franka_control/build-tsan && ctest --test-dir control_deploy/franka_control/build-tsan -L threads
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/control_deploy/bin/python -m pytest control_deploy/tests
```

### 8.3 The laptop's build for the cross test (PLAN §10)

The cross test runs on the **laptop** against the real franka_control code with the fake arm (top §8.10). It needs a
franka_control built there without libfranka; `build.sh` is not used (it would install libfranka), and the
`control_deploy` wrapper and its env are not needed. The test itself belongs to policy_deploy (its STRUCTURE.md §12.3;
its DIRECTIONS point here).

| What | How |
|---|---|
| apt packages | `build-essential cmake git pkg-config libeigen3-dev libzmq3-dev libyaml-cpp-dev googletest` (the PC's list minus `libpoco-dev`, which only libfranka needs). Eigen ≥ 3.4 (as the old build that worked on the PC; Ubuntu 20.04's own package is 3.3.7, see DIRECTIONS.md) |
| internet once | the first configure downloads msgpack-cxx 6.1.1 (FetchContent, header-only) |
| configure, build | `cmake -S deployment/control_deploy/franka_control -B deployment/control_deploy/franka_control/build-nolibfranka -DCMAKE_BUILD_TYPE=Release -DFRANKA_CONTROL_WITH_LIBFRANKA=OFF -DFRANKA_CONTROL_BUILD_TESTS=OFF` then `cmake --build deployment/control_deploy/franka_control/build-nolibfranka -j` |
| check | `build-nolibfranka/franka_control --version` ends in `libfranka=none` |
| rebuild | after every pull, as on the PC (baked paths, `commit` in `hello`) |

How the cross test runs it (the binary path comes from `FRANKA_CONTROL_BIN`):

```
$FRANKA_CONTROL_BIN --station <tmp>/station.yaml --mock --local --ready-file <tmp>/ready
```

- **Station:** generated by the test in a temporary folder, with only `session_root: <tmp>` (the binary needs nothing
  else; an `endpoints:` section would be refused). It never reads `control_deploy/station.yaml`.
- **Flags:** `--mock` (FakeArm, FakeGripper; `hello.mock` true, as the laptop's `--mock` expects), `--local` (binds
  `tcp://127.0.0.1` on the link.yaml ports 5620 and 5630, which must be free), no `--keyboard` (stdin is not a
  terminal). `realtime: enforce` in config.yaml is harmless: `--mock` skips `mlockall` (old:franka_control/src/app.cpp:119).
- It reads this checkout's `config.yaml` and the three contract files from the baked paths, so `hello` carries the
  same `F_T_TCP`, limits and watchdog as the laptop reads, and the laptop's commit.
- **Ready** when the ready file exists; **stop** with SIGTERM (exit 0 within the 4.5 s budget).

## 9. Tests

All run without the arm, the Pi or a GPU; sockets bind `tcp://127.0.0.1:*` (top §2.3). Classes that use time take a
`Clock`; threaded tests wait on conditions with a timeout, never a fixed sleep.

### 9.1 C++ (GoogleTest; ASan + UBSan; `Threads` tests also under TSan)

**Carried over** (old:franka_control/tests/, old-S §13), unchanged unless noted: `test_pose_math`,
`test_safety_gate` (incl. the ported main-sim cases; the `max_lead_s` case uses the config value; the other
waypoints of a rejected request carry the contract's `not checked: …` text),
`test_waypoint_trajectory`, `test_impedance_controller`, `test_homing_motion`, `test_watchdog`, `test_triple_buffer`,
`test_shared_state` (+ the new atomics lock-free), `test_arm_status`, `test_state_publisher`, `test_control_loop`,
`test_state_machine`, `test_supervisor`, `test_command_server`, `test_codec`, `test_config_loader`,
`test_gripper_worker`, `test_fake_arm_run`. `tests/support/`: `writeStation` drops `endpoints`; `testConfig` passes
the contract files; new `limitsFile()`, `framesFile()`, `linkFile()`, `examplesDir()` from the `CONTRACTS_*`
definitions; `writeLink(dir, address, state_port, commands_port)` for a temporary link.yaml.

**Changed:**

| File | Change |
|---|---|
| `test_codec.cpp` | `MessagesOfOtherContractsAreUnexpected` → `LaptopOnlySchemasAreUnknown` (`c1.frames/v1`, `c4.chunk/v1` → `unknown_schema`, top §3.4). New `KnownSchemasAreExactlyTheSharedOnes`: `Codec::knownSchemas()` equals the decoded `c3/hello_reply` `schemas`, in order (top §5.3 item 4). Hello requests and replies are covered by the generic example tests (decode, `values`, broken field/problem, byte-for-byte re-encode of `hello_reply`). |
| `test_config_loader.cpp` | `EndpointsAndSessionRootComeFromTheStation` → `EndpointsComeFromLinkYaml` (`tcp://<pc.address>:5620/5630`; with `local`, `tcp://127.0.0.1:...`); a station `endpoints:` section refused with its hint; station overrides of `tcp.F_T_TCP` and `watchdog.limit_s` refused with the "moved to" hints; `max_lead_s` default 1.5; `F_T_TCP` equals frames.yaml (re-normalised); `watchdog.limit_s` equals link.yaml; a link.yaml watchdog of 0.4 refused (floor). **2026-11-20:** `TheStartPoseIsUnsetUntilTheStationSetsIt` (null by default, `at_pose_tol_rad` 0.05, both set from the station, shown by `toYaml`), `AnAbsentStartPoseIsUnset`, `BadStartPoseValuesAreRefused` (not 7 finite numbers, a joint outside the Panda's limits, `at_pose_tol_rad` ≤ 0). |
| `test_command_server.cpp` | `EachRequestGetsItsReplyType` + hello; new `HelloReplyCarriesTheTemplateAndTheClock` (FakeClock: `t_pc_ns` is the clock value at handling, every other field the template's, the reply passes the decoder); `EveryDecodedRequestFeedsTheWatchdog` + hello; a `c3.hello.reply/v1` sent as a request → `unexpected_message`. |
| `test_state_machine.cpp` | `kHomingAborted`: `homing` → `stopped`; refused in every other mode; `kStop` in `homing` still allowed with no change of mode. |
| `test_supervisor.cpp` | Old cases unchanged. New: keys `h`/`g` refused in `holding`, `running`, `homing`, `fault` with the reason, state unchanged; `h` from `idle` and `stopped` → `homing`, `kHome` posted, no C3 reply produced at the end; `g` → `homing`, the hand request in SharedState, no loop command; hand done → `idle` + `markAlive`; hand timeout → `gripper.stop()` (a test gripper, while `home()` runs) + `fault`; `g` with no gripper refused; **stop during arm homing**: the abort flag set at once, `requestStop()` not called while the snapshot shows `kTorque`, called when it shows `kHoming` and the control thread marks the move, again only after 100 ms while the move runs, never once the move has ended (even with a stale `kHoming` snapshot); a homing-timeout stop waits for the move, is re-sent every 100 ms until the move is over, then never again; the loop's aborted report → `stopped`, `kHold`, `markAlive`, and the C3 home reply `ok: false` with a reason starting exactly `stopped: `; every other not-ok home reply (refused, box check, loop fault, gripper fault, timeout) does not start with `stopped:`; a `homed_generation` arriving first → `idle`, C3 reply ok; stop during hand homing → `gripper.stop()` only while `home()` runs, aborted report → `stopped`; a stop before `home()` began → no `gripper.stop()`, the homing skipped, `stopped`; a stop after it ended → nothing sent, `idle`; `s` key in `idle` → `stopped`, in `fault` → no change. **2026-11-20, key `p`:** `KeyPIsRefusedWhileQStartIsUnset`, `...AwayFromHome`, `...OutsideIdleAndStopped`; `KeyPMovesFromHomeToTheStartPoseAndOpensTheHand`; `KeyHStillGoesToQHomeAndLeavesTheFingers`; `StopDuringKeyPAbortsItLikeAHoming`; `AStartPoseOutsideTheBoxFaults`; `AStartPoseOutsideTheConeFaultsWithoutTheMargin`; `AHomeIsNotCheckedAgainstTheCone`; `C3StartNeedsTheStartPoseOnceItIsSet`; `C3StartIsUnchangedWhileQStartIsUnset`. |
| `test_control_loop.cpp` | Old cases unchanged. New (test `Arm`): the abort flag set when the session ends for `kHome` → `moveToJoints` never called, a new session, `homing_aborted_generation` set, flag cleared; `kStopRequested` from `moveToJoints` with the flag → `status()` polled, a new session holding the measured pose, nullspace target = that `q`; the same with `status()` reporting a reflex → `fault` with the problem text; `kStopRequested` without the flag → `fault` (as before); a normal finish with the flag set → homed, flag cleared; the move is marked in SharedState only while it runs (`stopHomingMove` runs the stop during it, for its session only, and never after). **2026-11-20:** `AKHomeWithoutATargetGoesToQHomeAndLeavesTheFingers`; `TheKHomeTargetIsTheMovesGoalAndItsOpenWidthTheFingerTarget`; `AStoppedPMoveHoldsTheMeasuredWidth`; `AnAbortBeforeTheMoveLeavesTheFingers`. |
| `test_gripper_worker.cpp` | New: a hand-homing request → `home()` called once on the gripper thread, `kDone` reported, the target re-sent afterwards even inside the deadband; `stop()` during `home()` → `kAborted`; `failNextHome` → gripper fault; a stop before the homing began → skipped, `kAborted`, the fingers never move; a `stop()` from before the request does not skip it; a stopped finger move throws (gripper fault), but not after shutdown was requested. `Threads`: the width in SharedState changes during the fake homing; `g` then `s` while a finger move runs → the move finishes, no gripper fault, the homing skipped; a stop while `home()` runs stops it, the fingers return. |
| `test_fake_arm_run.cpp` | Uses `AppOverrides` (free ports, a key pipe). The old run unchanged, plus: `hello` reply (`mock` true, the 12 schemas, `watchdog_limit_s` 1.0, `max_lead_s` 1.5, `t_pc_ns` between send and receive on `CLOCK_MONOTONIC`); keys through the pipe: `h` → C2 `homing` → `idle`; `h` then `s` mid-move → C2 `stopped` within 50 ms of the key, the fake TCP still afterwards (≤ 1 mm over 0.5 s), then `start` accepted; a C3 `home`, then `s` on the pipe mid-move → the deferred reply `ok: false` with a reason starting `stopped: `, C2 `stopped`; a C3 `stop` during a key `h` → `ack ok`, C2 `stopped`; `g` → `homing` → `idle`, width back to its value; `h` refused while `holding` (state unchanged); `q` → exit 0; **2026-11-20:** `KeyPGoesFromHomeToTheStartPoseOpensTheHandAndC3StartNeedsIt`. `ShutdownStaysWithinTheLaunchersStopBudget` → `...WithinTheWrappersStopBudget` (same numbers). |

**New:**

| File | Checks |
|---|---|
| `test_contract_files.cpp` | `ContractFrame`: the real frames.yaml loads and equals the literal of top §4.2; not 4×4, non-finite, transposed, reflection refused; `settled` and unknown keys ignored; a missing `F_T_TCP` named. `ContractLink`: the real link.yaml loads; `0.0.0.0`, a hostname, `256.1.1.1` refused; port 80, 70000 and equal ports refused; watchdog 0 refused; other keys ignored; `endpoint(5620, false)` = `tcp://<pc.address>:5620`, `local` → `127.0.0.1`. |
| `test_command_line.cpp` | `--local`, `--keyboard` parsed; unknown flags refused; `--version` alone; `--station` still required otherwise. |
| `test_keyboard_reader.cpp` | Over a pipe: `h`, `p`, `g`, `s` (and `H`, `P`, `G`, `S`) queued in order (`PIsQueuedAndNamed`); `q` and `Q` call `on_quit` and are not queued; unknown bytes ignored; EOF ends `run()` without `on_quit`; shutdown ends it within 0.1 s; the queue drops a 17th `h` or `g`, never an `s`. On a pty (`openpty`): `TerminalMode` sets ICANON, ECHO and IXON off, keeps ISIG, disables VSUSP and VQUIT, and restores the saved settings. On a pipe `TerminalMode` changes nothing. |
| `test_build_info.cpp` | `kGitCommit` is 40 lowercase hex or `unknown`; `--version` line format (run the built binary). |
| stop epoch (in `test_control_loop.cpp`, with `FakeArm`) | `requestStop()` during a call → that call ends `kStopRequested`; a stop requested before a call started (an earlier epoch) does not mark that call; a later fault in a new call is reported as the fault, never `kStopRequested`. |
| bind address (in `test_fake_arm_run.cpp`, `Threads`) | `BindsOnlyTheLinkAddress`: a temporary link.yaml with `pc.address: 127.0.0.1` and two free ports → `App::stateEndpoint()`/`commandsEndpoint()` are `tcp://127.0.0.1:<port>`, never `0.0.0.0`/`*`; a link.yaml address not on the machine (`192.0.2.1`, TEST-NET) → FATAL naming link.yaml pc.address, exit 1. |

The examples come from `CONTRACTS_EXAMPLES_DIR/index.yaml` (top §5): every good C2/C3 example (now including
`c2/state_stopped`, `c3/hello`, `c3/hello_reply`, `c3/hello_reply_mock`) and every broken one, with the same field,
problem and schema as Python.

### 9.2 Python (`control_deploy/tests/`, pytest)

| File | Checks |
|---|---|
| `test_station.py` | the real station.yaml loads; unknown keys, a bad `station_version`, out-of-range console values refused with the key |
| `test_paths.py` | the folders resolve from the editable install; the binary path from config.yaml `binary` |
| `test_binary.py` | with a fake binary script: `--version` parsed (commit, dirty, libfranka); `--check-config` YAML parsed (the `contracts:` paths, effective values); a hang caught by the timeout |
| `test_checks.py` | every row of 6.3 with injected inputs: commit mismatch fail, dirty warn, baked paths moved fail, port busy fail, address absent fail, firewall matched / different / absent / unreadable / ufw enabled, TODO markers warn (and not when the station overrides the key), payload warn, rt kernel and rtprio (as old test_launch.py), `--mock` skips arm/real-time/payload/fci/todo, `--local` skips address/firewall |
| `test_firewall.py` | the printed table for one and two laptop addresses; parsing `nft -j` output (sample JSON) into addresses, ports and the drop; a missing drop rule is a mismatch |
| `test_netinfo.py` | parsing `ip -j -4 addr` and `ss -Htn` samples |
| `test_status_line.py` | the line from the C2 example states (`deploy_contracts.load_index()`): each mode, the fault text, gate counters, peers; width fits 120 columns |
| `test_runner.py` | with a fake binary: the exact arguments (`--keyboard`, `--local`, `--mock`); log lines reach the log file and the terminal; the ready banner; not ready in time → SIGTERM; exit codes passed through; SIGTERM forwarded, SIGKILL after 5 s; termios restored on a pty after the child is killed; checks failing → nothing started, exit 2 |
| `test_waypoints.py` | time grid strictly increasing; ≤ 8 per request; every lead ≤ 0.6 s; min-jerk endpoints and peak speed; translation along base z; rotation about TCP z leaves the flange z axis unchanged |
| `test_console.py` | against a fake PC built from `deploy_contracts` `RpcServer` + `Publisher` in the test: no C2 refused; a C3 peer present refused; H2 wait then refuse; `homing` waited; `fault` refused; hello mismatch listed; clock check; step evaluation from synthetic C2 sequences (pass / warn / fail per step); an accepted out-of-box waypoint → `stop` sent and FAIL; `--yes` without `--mock` refused |
| `test_console_mock.py` | marked `needs_binary` (skipped if `franka_control/build/franka_control` is missing): `run --mock --local` with stdin from a pipe, then `console --mock --local --yes`: every step passes, the last leaves C2 in `fault`; a key `q` on the pipe ends `run` with exit 0. This is PLAN §10's "`control_deploy run --mock` with a scripted console". Also: the binary alone with its stdout and stderr on a pipe whose read end is closed after ready (the wrapper gone) keeps running (SIGPIPE ignored) and exits 0 on SIGTERM |
| `test_import_rules.py` | `control_deploy` imports only stdlib, yaml, numpy, zmq, msgpack, `deploy_contracts`; never `policy_deploy` or the old packages |

## 10. DIRECTIONS.md outline

1. **What you need:** the PC (Ubuntu 20.04, two network connections: a cable to the Franka, Wi-Fi on ASUS_5g),
   sudo, internet once, Desk access in a browser.
2. **Get the code:** clone, `git checkout rnoro/final-deployment`, the same commit as the laptop. Where to clone
   (any folder; moving it later means a rebuild).
3. **Python env:** install `uv`; create `~/envs/control_deploy`; install `deployment/contracts` and
   `deployment/control_deploy` editable. Check: `control_deploy --help`.
4. **Build:** `control_deploy/franka_control/build.sh` (libfranka 0.9.2 already in `~/franka` is reused). Check: the
   last line ends in `libfranka=0.9.2`. Rebuild after every pull.
5. **Robot network:** the Franka port at `172.16.0.1/24` (`nmcli con add ... con-name franka`), `ping 172.16.0.2`.
6. **ASUS_5g address:** connect the Wi-Fi; read the address (`ip -4 addr`); ask for a reserved address on the router
   for the PC (and the laptop); put it in `contracts/link.yaml pc.address` (and the laptop's in `laptop.addresses`),
   commit, pull on both machines. Going wired later: change the address there (PLAN §6).
7. **Firewall:** `control_deploy firewall | sudo tee /etc/nftables.d/control_deploy.nft`, include it from
   `/etc/nftables.conf`, `sudo systemctl enable --now nftables`; if ufw is enabled, also allow the laptop there.
   Redo after changing `laptop.addresses`.
8. **Station:** `control_deploy/station.yaml`: `session_root`, `realtime: ignore` until the RT kernel.
9. **Desk, every session:** open `https://172.16.0.2`; robot system 4.x (for libfranka 0.9.2); end effector set to the
   Franka Hand; unlock the brakes; activate FCI; release the user stop.
10. **Without hardware:** `control_deploy check --mock --local`, `control_deploy run --mock --local` (keys h p g s q),
    then in a second terminal `control_deploy console --mock --local`.
11. **The real run:** `control_deploy check` (no fail), `control_deploy run`. What start-up looks like (fingers home
    once, the arm holds), the status line, the keys. Keep a hand on the user stop. Start `policy_deploy` on the
    laptop only after `ready`. The PC stays up across laptop runs.
12. **Bring-up** (first time, section 11 checklist): measure and set the TODO values, then `console` step by step.
13. **After a fault:** read the line; release the user stop if it was a reflex; Ctrl-C (if still running); `control_deploy
    run` again. The laptop just reruns.
14. **When something fails:** table of messages → fix (version/commit → rebuild; baked paths → rebuild; link address →
    link.yaml or the router; bind failed → address or another instance; arm not reachable; not ready → Desk; ports in
    use; firewall; real-time).
15. **Tests:** ctest commands, pytest.
16. **Files you edit:** `station.yaml`, `franka_control/config.yaml`, `contracts/link.yaml` (shared: commit it).

## 11. Hardware bring-up on the PC (PLAN §12 step 1)

### 11.1 Checklist

Before running anything:

- [ ] Weigh the rig beyond the Franka Hand (printed fingers, Pi 5, cameras, mount); centre of mass from CAD; set
      `load.mass_kg`, `com_m` (not also in Desk: Desk's end effector is the Hand alone, `setLoad` adds the rest).
- [ ] Measure the workspace box in the robot base frame around the table, a few cm inside the real limits; set
      `safety.workspace_min_m`/`max_m`. The floor (`min z`): 5 mm below the `tcp z` with the fingertips resting on the
      table (grasps of a flat object end at the table; a target within `box_margin_m` below is pulled up onto it).
- [ ] Check `homing.q_home_rad` with the rig on: no cable or camera collision; its TCP must be inside the box (the
      after-homing check faults otherwise).
- [ ] Desk: system version 4.x, end effector Franka Hand, FCI on, brakes unlocked, user stop released.
- [ ] `control_deploy check`: no fail. Expected warnings: `realtime` (ignore), `todo` for what is still unmeasured.

First run:

- [ ] `control_deploy run`: the fingers home once, `ready`, mode `idle`. Watch 5 minutes: no drift, no reflex,
      `loop_late` on the status line.
- [ ] `g`: the fingers close, reopen, return to the commanded width; mode `homing` → `idle`.
- [ ] Measure `gripper.width_offset_m` (calipers at two openings vs C2 `gripper_width_m`) and the max fingertip
      opening (`limits.yaml gripper_width_max_m`, shared: commit). Restart `run` after editing.
- [ ] `h`: the verified homing; `idle` at the home pose.
- [ ] `h`, then `s` mid-move (**new on hardware**): expect `stopped`, holding where it stopped. If it ends in
      `fault` instead, note the libfranka text (section 12, decision 5) and restart.
- [ ] **The stop epoch (a change on the verified path, section 3.3):** right after that stopped homing, `h` again
      (the verified homing must run normally: `idle` at home), then `s` in `idle` and `console` step 1 (`start`, hold,
      `stop`): no `fault`. A requested stop must never appear as a reflex, and a later reflex (if one happens during
      bring-up) must be reported as a reflex fault, not as `stop requested`: note the log line.
- [ ] `g`, then `s` during the hand's homing: `stopped` (or `idle` if the hand cannot be interrupted).
- [ ] `console`, one step at a time, Enter before each: `hold`, `move`, `rotate`, `gripper`, `watchdog`,
      `out_of_box` (ends in `fault` by design; restart `run`). Keep `results.yaml`.
- [ ] **The start pose** (2026-11-20): Ctrl-C `run`; in Desk guiding mode (FCI off) move the arm by hand to the
      training demos' start (fingertips ~48° from straight down, ~16 cm above the table); FCI on, `run` again (it
      holds there); on the laptop `pc_probe --start-pose`; paste its `q_start_rad` line into `control_deploy/station.yaml`
      (`franka_control: homing:`); restart `run`.
- [ ] `h`, then `p`: the hand opens fully, the start pose is reached (`idle`, no box or cone fault); then `start` is
      accepted there (`console --only hold`); `p` again: refused (`not at home`).
- [ ] Only then the laptop (PLAN §12 step 3).

### 11.2 Risks

| Risk | What happens | Mitigation |
|---|---|---|
| Payload not set (`load.mass_kg` 0) | The rig's weight is a constant force the spring must hold: about m·g/800 N/m of sag (0.4 kg → ~5 mm down, ~0.5° of tilt); the extra torque counts toward the collision thresholds | weigh it first; `check` warns; the console's tracking numbers show it |
| Workspace box and cone unmeasured (main-sim defaults) | The gate may allow targets into the table or fixtures | measure first; start with a smaller box |
| Untuned gains (800 N/m, 40 Nm/rad) | Sluggish tracking or oscillation; tracking error over 3 mm | console steps 2–3 report it; tune in the station |
| Low collision thresholds (20 Nm) | Reflexes on faster moves or with an uncompensated payload → `fault` | console steps are slow; raise only with a reason |
| `realtime: ignore` (no PREEMPT_RT) | Late cycles; a `communication_constraints_violation` reflex while holding for hours leaves a `fault` for the next laptop run | install the RT kernel early (PLAN §13); watch `loop_late` |
| Home pose not checked with the rig | Collision during `h` | check by hand in Desk guiding mode first |
| Stop during homing untested | libfranka's preempt might leave the robot needing recovery → `fault` (safe, restart) | first try with the arm far from anything |
| `F_T_TCP` pending (PLAN §9) | none for these tests: the box uses the TCP position, the cone the flange z axis, the console rotates about TCP z, which is the flange z for a rotation about z | settle before the first trained policy |
| The PC's Wi-Fi address changes (DHCP) | bind fails at start (FATAL), or a running process stays bound to the old address and the laptop cannot connect | reserved address on ASUS_5g (PLAN §15) |
| No joint-limit check in the gate | a target near a joint limit ends in a reflex → `fault` | small console moves from the home pose |

## 12. Decisions (were open questions and gaps)

Settled by the lead on 2026-11-13, with PLAN.md and the top-level STRUCTURE.md updated to match. Nothing is open.

1. **`g` puts C2 `mode` to `homing`** (PLAN §8.1, top §3.7.2's `homing` column): the laptop then waits instead of
   starting an episode while the fingers calibrate, and `start`/`home` are refused by the existing rule.
2. **After `g`, the fingers return to the commanded width** (PLAN §8.1): the target stays the truth; no loop change.
3. **A stop arriving as a homing move finishes: the normal completion wins** (`idle`, C3 `home` ok; PLAN §8.1, top
   §3.7.2). Both modes hold the pose and accept the same requests.
4. **Firewall tooling:** a dedicated nftables table `inet control_deploy`, printed by `control_deploy firewall` and
   read by `check` (without sudo, then `sudo -n`; warn when it cannot read it). It accepts `laptop.addresses` and
   loopback (top §2.4, PLAN §6). An optional sudoers line for exactly `nft -j list table inet control_deploy` lets the
   row pass instead of warn.
5. **Hardware unknowns are settled at bring-up, not in the design:** whether `Robot::stop()` during a joint move
   leaves the robot ready for a new torque session without recovery, and whether `franka::Gripper::stop()` interrupts
   `homing()`. The design handles both outcomes (a fault, or the hand finishing its homing); record what happens in
   `results.yaml` and the log (section 11.1).
6. **A C3 `home` stopped by the PC's `s` replies `ok: false` with the fixed prefix `stopped:`** (top §3.7.2, section
   4.4); the laptop shows it and stays idle (PLAN §7). Any other not-ok reply concludes on the laptop.
7. **`LibfrankaArm`'s sticky stop flag becomes a per-call stop epoch** (section 3.3; PLAN §8.1, top §8.11): needed so
   the process survives a stopped homing without a later reflex being mislabelled as a requested stop. It is a
   change on the verified path and is tested at bring-up (section 11.1).
8. **The change list of PLAN §8.1** now names every change of section 3: the keyboard, stop during homing (supervisor,
   shared state, the control thread's session loop, the gripper worker), the new state-machine event,
   `Gripper::home()`, SIGHUP, `--keyboard`, and the stop epoch. `ControlLoop::step`, the gate, the homing motion and
   finger moves stay unchanged.
9. **The entry is `check | run | console | firewall [--mock] [--local] [--station FILE]`** (PLAN §8.1).
10. **C2 `t_ns` repeats for up to 1 s after a stopped arm homing** (top §3.6 obligation 1, PLAN §5.3); the laptop
    ignores repeated `t_ns` and judges liveness by arrival.
11. **The status line of `control_deploy run` is a C2 subscriber on the PC** (top §2.2, §3.11), read-only.
12. **A dirty tree on the PC is invisible to the laptop:** `hello.commit` is a hash or `unknown` (top §3.8); `check`
    warns on `dirty=yes`. No contract change.
13. **System packages for yaml-cpp, cppzmq and GoogleTest**, as the 91b178b build that ran on the PC (old-S §12.1
    named FetchContent; the code is followed).
14. **The cross test runs on the laptop** against the libfranka-less build of section 8.3 (PLAN §10, top §8.10);
    policy_deploy owns the test.
15. **`deploy-contracts` is not a dependency in `pyproject.toml`**; it is installed editable alongside (top §1.4).
