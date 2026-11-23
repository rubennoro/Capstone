# teleop_collect: design

The laptop side of teleop collection: an operator drives the Franka Panda's end effector from a page in the browser
with the keyboard, and every demonstration is recorded on disk (both Pi camera streams as received, every arm state,
every command). The goal, the rules and the decisions are in [PLAN.md](PLAN.md); the interfaces between the parts in
[STRUCTURE.md](STRUCTURE.md). This file is the whole design in short, with links to each part's own DESIGN.md.

**Does:** receive the Pi's C0 stream; talk C2/C3 to the PC's unchanged `franka_control`; turn held keys into
waypoints; run the episode lifecycle; write episodes under `data_root`; serve one page on `127.0.0.1`.
**Does not:** change, start, stop or read anything on the PC or the Pi (PLAN §2 rules 2 and 4); compute joints (the
PC's 1 kHz loop tracks a Cartesian TCP pose); convert episodes into training data; replace the physical user stop.

## The processes and how they talk

```
                LAPTOP (teleop_collect run)                                        PC (control_deploy run, unchanged)
 Pi 5 ──C0──► frames ◄──R1──► teleop ◄────────── C2 state, 100 Hz (SUB) ──────── franka_control
      pi_stream  │  ▲  REQ/REP   │  ▲  ─────────── C3 requests (REQ) ─────────►  gate, 1 kHz loop, gripper,
      /v1 (SUB)  │  │            │  │             start stop set_waypoints         homing, watchdog
                 │  U2           │  U1                hello heartbeat get_state
                 ▼  │ REQ/REP    ▼  │ REQ/REP
          episode files       ui (aiohttp, 127.0.0.1:8765) ◄── WebSocket ── browser page (keys, buttons)
          (*.mkv, *.csv)   episode files (robot_states, commands, keys, clock_pc, events, meta, COMPLETE)
```

| Link | Between | What | Where defined |
|---|---|---|---|
| C0 | Pi → frames | `pi_stream/v1`: one packet per main frame, both JPEGs, Pi times | `teleop_collect/contracts/pi_stream.py` (fd's) |
| C2 | PC → teleop | `c2.state/v1` at 100 Hz: mode, TCP and target poses, joints, width, gate counters | `deployment/contracts/` (byte copy of fd's) |
| C3 | teleop → PC | `hello`, `heartbeat`, `get_state`, `start`, `stop`, `set_waypoints` (never `home`) | `deployment/contracts/` |
| R1 | teleop → frames | `start_recording(dir)`, `stop_recording` → summary, `get_health` | `teleop_collect/contracts/r1_recorder.py` |
| U1 | ui → teleop | `get_status`, `keys`, `gripper`, the buttons (task, start, stop, label, delete episode, delete task, storage folder, stop arm) | `teleop_collect/contracts/u1_teleop.py` |
| U2 | ui → frames | `get_preview` → the newest JPEG of each camera, health | `teleop_collect/contracts/u2_preview.py` |

R1, U1 and U2 are local REQ/REP on `127.0.0.1` with the station's ports. **Only teleop's main thread sends C3**, on
its one REQ socket (rule H1); the ui never talks to the PC and frames never talks to teleop's PC link. Three
processes, because writing about 4 MB/s of JPEGs and serving previews must never delay the 20 ms loop that feeds the
PC's watchdog (PLAN §3).

The launcher starts `fake_pc` (only under `--mock`), then `frames`, `teleop`, `ui`, each after the previous one wrote
its ready file, and stops them in reverse, so teleop sends its C3 `stop` while frames and the PC are still up. Any
process failing ends the run with one `FAILED <process> ...: FATAL ...` line; teleop exiting 0 is a normal stop.

## One cycle of the teleop loop

Every `link.yaml heartbeat_period_s` (20 ms), on teleop's main thread (`teleop/teleop_loop.py`):

1. Signal flags (Ctrl-C, SIGTERM).
2. State rules: a fault, `seq` going back, no C2 for `state_lost_s`; the listener's and recorders' errors.
3. The key state: older than `key_stale_ms` (150 ms) or not focused = all keys released (the deadman).
4. **Exactly one C3 request:** a `hello` when due (every 0.5 s, feeding the PC clock map), else the `set_waypoints`
   from `MotionController.step`, else a `heartbeat`.
5. R1 health every 500 ms.
6. Up to 4 U1 requests; Start, Stop and Stop arm add their own C3 `start`/`stop` here, on the same socket.
7. Status bookkeeping (`at workspace limit`, the start pose reference), keys.csv and the clock rows.

**From keys to waypoints** (`teleop/motion/`): the held keys give a twist (±`v_max` per translation axis, the twist
about the TCP's own axis, the tilt about the base axes through the TCP point, both tilts normalised to `ω_max`,
Shift ×0.25). The base of every set is **the PC's own latest target** (C2 `O_T_TCP_target` at its PC time `t_s`), so
the laptop never integrates a pose of its own. `K` = 2 waypoints at `now_pc + k·Δ` (Δ = 50 ms) are the base advanced
by the twist over `t_k − t_s`. On release one last waypoint holds the arm; after it, only heartbeats. Measured: x+
held 1 s moves the target 50.0 mm at 50.0 mm/s, on the real `franka_control --mock --local` and on the fake PC.

## The episode flow

Phases: `starting → idle ⇄ recording`, `recording → stopping → idle`, any → `done`.

- **Start-up** (PLAN §8): incomplete episode folders deleted; C2 arriving; the H2 wait (no C3 while another client
  drives the arm); hellos until the clock map is ready; `hello` compared (schemas, `mock`, `F_T_TCP`, limits,
  watchdog) and the gap budget checked; both cameras `ok`; `data_root` writable; **the arm's pose now becomes the
  session's start pose** (the operator pressed `h`, then `p` on the PC first). Nothing moves the arm.
- **The start pose** (`teleop/motion/start_pose.py`): there is no laptop home; `h` and `p` are the PC's keys (as
  policy_deploy since fd:`927a5ba`). Until the PC accepts a start, the pose after each moving `homing` → `idle` on C2
  (the PC's `h`, then `p`) replaces the learned one; the PC's first accepted start confirms it for the session, and a
  start it refuses with `refused: not at the start pose` drops it. The PC, with `homing.q_start_rad` set, has the
  final say.
- **Start** (only at the start pose, idle or stopped, task selected, cameras `ok`, clock map ready): a new numbered
  folder, teleop's recorders, R1 `start_recording`, C3 `start`, motion on.
- **Stop** (Save: success / fail / Discard): motion off, C3 `stop`, R1 `stop_recording`, the recorders closed, then
  `meta.yaml` and `COMPLETE` written last, or the folder deleted.
- **Warn** (a camera down or stale, a recording error, while recording): the arm and the episode go on. Shown once on
  the page, noted in the episode's `events.jsonl` and `meta.yaml` `warnings`. After a recording error the episode can
  only be discarded (Save is refused with the reason).
- **Abort** (Stop arm, the PC stopped: its `s` or the watchdog): as Stop, and the folder is deleted. The app stays up;
  the reason goes to the page and the session log.
- **Conclude** (PC lost, restarted or faulted, a contract break, a clock mismatch, Ctrl-C): C3 `stop`, the open
  episode deleted, the processes stopped, one line. The PC keeps holding the arm.

## The safety rules

- **No software stops while collecting** (2026-11-20): only Franka's own limits and reflexes, the operator (keys
  released, Stop arm / Esc, the PC's `s`, the user stop) and a dead link (the PC's 1 s watchdog) stop the arm. Data
  problems are warnings; a C2 seq gap never pauses motion.
- **The PC's gate is the one safety authority** (PLAN §5.3). The laptop never copies its box or cone to go faster. For
  teleop the PC runs its teleop block (the box open except the floor, cone 179°, margins 0.05 m / 10°: the PC's
  directions, step 8, "Teleop settings").
- **The bound:** a waypoint sticks out past the gate's pulled-back boundary by at most `speed × (lead + state age)` =
  0.05 m/s × (0.3 + 0.25) s = 27.5 mm and 5.5°, below 0.8 × the assumed PC margins (40 mm, 8°), so teleop never
  faults the PC at the floor. `MotionSettings` refuses settings beyond that at load. The PC does not report its
  margins: `teleop_collect check` shows them as the INFO row `PC margins`, and start-up as a page message. With the
  PC's default margins (10 mm, 2°) a press into the floor can fault the PC.
- **No motion on a stale state:** no set while the newest C2 is older than 250 ms (mapped); the arm ends its last plan
  within the 300 ms lead and holds.
- **The deadman:** stale keys (150 ms) or a page without focus = released. The ui sends "no keys" when the page is
  silent or its socket closes; teleop has its own timeout as the second line.
- **Stop arm, Esc, Ctrl-C and every conclude send C3 `stop`.** One C3 client at a time (H2 at start-up).
- **Start only at the start pose**, so every episode begins from the pose the deployment starts from. The laptop
  never moves the arm to a pose; the PC's `h` and `p` do.
- **The page is local only:** bound on `127.0.0.1`, with Host and Origin checks.
- **The physical user stop is the emergency stop** (PLAN §2 rule 10).

## The data on disk

`<data_root>/<task>/task.yaml` and `episode_NNN/` (PLAN §6); `--mock` runs use `mock_data_root` instead, so stand-in
episodes never mix with the real training data (the two folders may not be equal or inside one another). The storage
folder can be changed in the page (or with `teleop_collect run --data-root PATH`); it is remembered in
`storage.local.yaml` next to `station.yaml` (gitignored), which replaces station.yaml's `data_root` /
`mock_data_root`. The page can also delete a whole task folder. frames writes `main.mkv` and `ultrawide.mkv` (the JPEG
bytes as received, muxed into Matroska, never re-encoded; frame `i` of a video is row `i` of its CSV), `*_frames.csv`,
`packets.csv`, `clock_pi.csv`. teleop writes `robot_states.csv`, `commands.csv`, `keys.csv`, `clock_pc.csv`,
`events.jsonl`, then `meta.yaml` (with the full `hello`, so `F_T_TCP`) and `COMPLETE` last. Every time is integer ns
with its clock in the name (`_pi_ns`, `_pc_ns`, `_ns`), raw next to mapped. Episode numbers only count up. A folder
without `COMPLETE` is deleted at the next start-up. Session logs go to `session_root`, never to `data_root`.

## The parts

| Part | What it does | Design |
|---|---|---|
| `contracts/` | C0 parsing, clock fits, station and config loading, process arguments, the R1/U1/U2 messages | [contracts/DESIGN.md](teleop_collect/contracts/DESIGN.md) |
| `frames/` | the Pi stream, camera health and the pair rule, the recorder (`.mkv` + CSVs), previews, the R1/U2 servers, `pi_collect` | [frames/DESIGN.md](teleop_collect/frames/DESIGN.md) |
| `teleop/` | the PC link, clock map, start-up, the 20 ms loop, the episode flow, the failure rule, the U1 server, `pc_probe` | [teleop/DESIGN.md](teleop_collect/teleop/DESIGN.md) |
| `teleop/motion/` | keys → twist → waypoints, the §5.3 bound, the start pose reference | [teleop/motion/DESIGN.md](teleop_collect/teleop/motion/DESIGN.md) |
| `teleop/episode/` | the task and episode store, teleop's recorders, `meta.yaml` | [teleop/episode/DESIGN.md](teleop_collect/teleop/episode/DESIGN.md) |
| `ui/` | the web server, the page, the deadman, `ScriptedPage` for tests | [ui/DESIGN.md](teleop_collect/ui/DESIGN.md) |
| `launcher/` | `teleop_collect check` and `run`, the check rows, start in order, stop in reverse | [launcher/DESIGN.md](teleop_collect/launcher/DESIGN.md) |
| `fake_pc/` | the Python stand-in for `franka_control` under `--mock` | [fake_pc/DESIGN.md](teleop_collect/fake_pc/DESIGN.md) |
| `deployment/contracts/` | the shared C2/C3 code, a byte-for-byte copy of `rnoro/final-deployment`'s | fd:STRUCTURE.md §3 |

Copied code says so in its header (`copied from` / `adapted from rnoro/final-deployment:... @ 2ffe641`, or `@ ea9f7fb`
for what the 2026-11-19 re-sync took and `@ 927a5ba` for the 2026-11-20 start pose re-sync, with every change listed).
The rig file is the deployment's `contracts/rigs/rig_pi_v3_640.yaml`, and `F_T_TCP` is `contracts/frames.yaml`'s (+45
deg about flange z since `ea9f7fb`, fingertips 235.65 mm out since `927a5ba`). Nothing imports `policy_deploy` or `control_deploy` (`tests/test_import_rules.py`).

## Tests

- **Unit tests** per part, without the Pi, the arm or a GPU, on free ports (each part's DESIGN.md).
- **Integration** (`tests/integration/`, helpers in `tests/integration_helpers.py`), on the real processes:
  - `test_run_mock.py`: `teleop_collect run --mock` driven by `ScriptedPage` (the page's WebSocket protocol): a task,
    Start at the fake PC's start pose (confirmed), x+ for 1 s, close, open, save, the fake PC's `h`, `p` after it; every episode file, both `.mkv` demuxed against their CSVs, row
    counts, `F_T_TCP`; discard, Stop arm, delete, numbers never reused; Ctrl-C mid-episode (exit 0, folder gone, no
    process left).
  - `test_cross_franka_control.py`: the same against the real libfranka-less `franka_control --mock --local` (only
    run from `~/Capstone-final-deployment`, never built there), with `homing.q_start_rad` set and `h`, `p` typed on
    its stdin: the PC's own start refusal away from the start pose, a session started at home; the speed measurement.
  - `test_failure_rows.py`: the PLAN §7 rows that need several processes, with an in-process fake PC on the
    `link.yaml` ports (and an in-process fake Pi): a camera down mid-episode (a warning, the episode goes on), keys held
    without focus and focus lost mid-hold, Stop arm, the PC's `s`, no C2 for 2 s, a C3 timeout, the PC restarting, a PC
    fault.
  - `test_fake_pc_speed.py`: the fake PC follows a plan replaced every 20 ms at the commanded speed.

## How to extend

- **A new key:** STRUCTURE §4.1's names (contracts), the page's map (ui), its axis in `twist_from_keys` and, if it can
  raise a speed, `worst_step` (motion).
- **A new recorded file:** frames' recorder or teleop's `EpisodeRecording`, plus PLAN §6 and the integration check
  `EPISODE_FILES` in `tests/integration_helpers.py`.
- **A new button:** a U1 request (contracts), its handler in teleop, one entry in the ui's `COMMANDS` and the page.
- **A new failure row:** a `ProblemCode` and its `RULES` entry (teleop); a multi-process test in
  `tests/integration/test_failure_rows.py`.
