# teleop: design

The `teleop` process (`python -m teleop_collect.teleop`). It is the laptop's only C3 client. It turns the page's key
state into waypoints, runs the episode lifecycle and records C2 and the commands while an episode records. This file
covers everything in `teleop/` except `motion/` and `episode/`. Those are pure logic with their own DESIGN.md files.

**Does:** C2 in, C3 out (rule H1: every C3 request on the main thread), the PC clock map, start-up (PLAN 8), the 20 ms
loop (PLAN 5.1), the R1 calls to frames, the U1 server the ui talks to, the failure rule (PLAN 7), the session log,
and `pc_probe`.
**Does not:** touch the PC beyond C2/C3, read the cameras (frames does), or serve the page (the ui does).

## Threads and data flow

| Thread | Class | Does |
|---|---|---|
| main | `TeleopLoop` (+ `EpisodeFlow`, `StartupSequence`) | every decision, every C3 request (`PcClient`), R1 (`FramesClient`), U1 (`U1Server`, served from the loop) |
| pc-listener | `PcListener` | each C2 into `StateHistory`, `PcClock`, `LinkStats` and, while recording, E4's `StateRecorder` (`set_recorder`) |

One cycle, every `link.yaml heartbeat_period_s` (`teleop_loop.py`, STRUCTURE 6.3):
1. the signal flags;
2. the state rules (`FailureMonitor.check_state`), the listener's error, and the recorder's error;
3. `KeyState.current` (deadman: stale or unfocused means no keys);
4. **exactly one C3 request**: a home that was asked for, else a hello when due, else the `MotionController.step`
   set, else a heartbeat;
5. R1 health every `health_poll_ms`;
6. up to `u1_max_per_cycle` U1 requests;
7. the home reference and `at_limit`, keys.csv and the clock_pc.csv fit, then the session-log rows.

**U1 buttons in step 6.** The start, stop and stop_arm buttons send their C3 request (start or stop) inside step 6 of
the same cycle. So does the `get_state` that names the cause when the PC refuses a request. Home is the exception: it
is queued for the next cycle and becomes that cycle's request. A cycle with a button press therefore sends at most one
keep-alive plus that one command. Rule H1 still holds: all of these are on the main thread, on the one REQ socket. The
worst gap between two C3 requests is unchanged, because the command itself feeds the watchdog.

`now_pc` and the base state come from one `PcClock.snapshot()` and the newest C2. A C2 seq gap, a clock map that is
not ready, or a state older than `state_max_age_ms` pauses motion (SLOW). E4's controller enforces the age guard
itself.

U1 requests reach `TeleopLoop.dispatch` from inside `RpcServer.serve_once`. If a handler raises (a C3 timeout during
Start, a conclude), the exception is kept and re-raised on the loop after the reply has gone out, so `_guard`
concludes as usual. While homing, only `get_status` and `keys` are answered (keys are ignored). Everything else gets
the STRUCTURE 6.7 text.

The episode lifecycle (`episode_flow.py`, STRUCTURE 6.4):
- **Start:** a new folder, the `EpisodeRecording`, R1 `start_recording`, C3 `start`, motion begun at the C2 width,
  then the state recorder is switched on.
- **Stop:** C3 `stop`, the state recorder off, R1 `stop_recording`, the recorders closed, then discard or
  `build_meta` + `finish`.
- **Abort** (Kind.EPISODE): the same steps, then the folder is discarded.
- **`abort_open`:** the path for conclude, Ctrl-C and SIGTERM. It never raises.

Every C3 request made while recording goes to commands.csv: start, sets, hellos (with the HelloReply), heartbeats and
stop.

## Decisions

- **Kind.EPISODE** (PLAN 7): camera down, PC stopped, Stop arm and recording failures abort the episode and the app
  stays up. With no episode open they count as NORMAL; between episodes, a camera that is down only disables Start.
- **PC_REFUSED is kept** (STRUCTURE 6.5 does not list it). A set refused because the PC stopped is turned into
  PC_STOPPED by `_refusal_cause`, and `_guard` then aborts the episode instead of concluding.
- **Home** is a U1 request carried out in the next cycle, as that cycle's C3 request. The home reference is set from
  the first C2 showing `idle` after a successful home, since the reply can arrive before that C2. A C2 `homing` ->
  `idle` transition also sets it (E4's `observe`).
- **Start allowed:** the rules and texts of STRUCTURE 6.4 in order. A queued home counts as `homing`.
- **The task and episode lists are caches.** Reading every meta.yaml on the main loop stalls the 20 ms cycle (about
  10 ms per episode) and trips the PC's watchdog. So the lists are read from disk only at start-up (after
  `clean_incomplete`) and after a task change (select, create). A save, a discard or a delete updates the list from
  what was just written. Nothing is re-read on a timer, by a status request, or while recording or stopping.
- **A write error never concludes.** An OSError writing commands.csv, keys.csv, clock_pc.csv or events.jsonl through
  `EpisodeFlow.record_*` is kept, and the next cycle's step 2 aborts the episode with RECORDING_FAILED (PLAN 7). So do
  E4's state-recorder error, a failed close, and a failed meta.yaml / COMPLETE.
- **R1 health timeouts:** up to 3 `get_health` timeouts in a row (`timeouts.health_s`, 50 ms) are each a warning on
  the page. The fourth in a row concludes NO_ANSWER. One answer resets the count. The worst-gap budget is unchanged.
- **SIGTERM text:** the launcher sends SIGTERM both for an operator Ctrl-C and when another process exits, so the text
  is "stopped by the launcher (Ctrl-C, or another process exited)".
- **`laptop_commit()`** counts untracked files as dirty, as the launcher does.
- **The R1 stop never ends the run.** On Stop the C3 `stop` goes out first, so the PC is `stopped` and its watchdog
  is not armed while R1 `stop_recording` waits. That wait is `timeouts.frames_s` = 4 s, because frames' drain can take
  up to 1 s + 2 s. The worst-gap budget is therefore unaffected. A timeout, a broken reply or "not recording" aborts
  the episode with RECORDING_FAILED (folder deleted, app up); the abort paths only log such a failure.
- **Delete is refused while an episode records or stops** ("stop the episode first"), so no rmtree runs on the loop
  while the watchdog is armed.
- **The start width is clipped** into [0, limits.yaml gripper_width_max_m] before `MotionController.begin`. An open
  hand can read 0.081 m, which the PC's gate would refuse (a fault). A clip is shown and logged. Every other width
  comes from the controller (0 or gripper_width_max_m).
- **A camera down for longer than frames' `health.down_s`** gives the Start refusal "<camera> camera down: check the Pi
  (after a Pi restart: Ctrl-C and run teleop_collect again)". frames latches a camera down after a Pi stream restart.
  The time is measured from the first health poll that showed the camera down.
- The ready file is written through `ProcessArgs.write_ready()`. The FATAL line is printed last on stderr (`app.py`).

## Copied from fd: (`policy_deploy/policy_deploy/manager/` @ 2ffe641)

- **Copied, imports only:** `pc_client`, `pc_clock`, `link_stats`, `heartbeat_sender`, `state_history`. In
  `state_history`, `pose_math` now comes from `teleop_collect.common`.
- **Adapted:**
  - `errors`, `problems`, `failure_monitor`, `settings`, `startup_checks`, `startup_sequence`, `session_log`;
  - `pc_listener` (gains the recorder hook);
  - `frames_client` (now R1);
  - `pc_probe` (behaviour and JSON rows unchanged);
  - `teleop_loop` (from `episode_loop`);
  - `app` / `__main__` (from `__main__`).
- **New:** `episode_flow`, `u1_server`. Each file's header lists its changes.

## Tests (`tests/teleop/`, helpers in `tests/teleop_helpers.py`)

- **Copied from pd: and adapted:** the clock, state history, listener (+ the recorder hook), heartbeat, client (+ R1)
  and startup-check tests.
- **`test_teleop_loop.py`:** runs on `TeleopHarness`, which drives the real loop, flow and E4 classes against a fake
  PC client, a fake listener and a fake R1 on a fake clock. It covers:
  - start-up order;
  - one C3 request per cycle and hello priority;
  - move, release once, stale keys, lost focus, the old-state guard, the fingers, `at_limit`;
  - start, stop, success, discard and delete, and that numbers are never reused;
  - every start refusal;
  - home and the U1 traffic during it;
  - aborts (camera down, recording error, PC stopped, a refused set, Stop arm);
  - conclude (fault, gate rejected, clock mismatch, state lost), Ctrl-C, SIGTERM during a home;
  - the status contract.
- **`test_sockets.py`:** the real clients and U1 server against `SocketPc` (a small fake PC on real sockets) and a
  small R1 server. It covers a whole episode over U1, H2 (another client busy, an orphaned holding), `pc_probe
  --check --json`, and the process's FATAL line.

## Extending

- **A new U1 request:** its contract class (E1), one handler method, and one `U1_HANDLERS` entry.
- **A new failure:** one `ProblemCode` and one `RULES` entry; act on it with `TeleopLoop._problem`.
- **A new recorded file:** E4's `EpisodeRecording`, fed from `EpisodeFlow.record_*`.
