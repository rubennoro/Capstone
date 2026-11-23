# Teleop collection plan

Branch `rnoro/teleop-collect-deploy`. This file holds the goal, the rules, the lab procedure, the processes, the teleop
loop, the episode layout, the failure rule and the open questions. `STRUCTURE.md` (the detailed layout of the code) is
written from this plan once it is agreed, and the code only after that.

Every source reference `fd:<path>` points into `rnoro/final-deployment` @ `2ffe641`
(`~/Capstone-final-deployment/deployment/`). That branch is only read and copied from, never changed.
Re-synced to `ea9f7fb` (2026-11-19): `deployment/contracts/` (the new `F_T_TCP`, `rigs/rig_pi_v3_640.yaml`), the rig
file setting and `frames/_dataproc/config.py`; those files' headers name `ea9f7fb`.
Re-synced to `927a5ba` (2026-11-20): `deployment/contracts/` (`F_T_TCP` 235.65 mm out), the start pose (the PC's key
`p`; no laptop home any more, §8) and `pc_probe --start-pose`; the files it touched name `927a5ba` in their header.
No software stops (2026-11-20): only Franka's limits, the operator and a dead link stop the arm; data problems are
warnings (§7); lead 300 ms, state age 250 ms, assumed PC margins 0.05 m / 10° (the PC's teleop block, §5.3). The PC
runs fd:`447910b` (`927a5ba` + the default box x 0.20).

## 1. Goal

**Collect pick/place demonstrations on the real Franka Panda by driving it from the laptop's keyboard**, and record
everything a policy would later need: both Pi camera streams and the arm's state. Episodes are saved per task under
`~/CapstoneTeleop/<task>/`.

Why: a policy trained on handheld (UMI-style) rig data may not transfer well enough to the robot. Teleop data is
recorded on the robot itself, with the same cameras, the same 640×480 stream, the same TCP definition and the same
clocks that `policy_deploy` uses when it runs a policy. A policy trained on it sees in training exactly what it sees in
deployment.

The setup is the one `rnoro/final-deployment` already built:

- The **Raspberry Pi 5** on the gripper streams both cameras (IMX519 main, IMX219 160° ultrawide, 640×480, 30 FPS) over
  Wi-Fi (ASUS_5g) in the C0 contract `pi_stream/v1` (fd:policy_deploy/PI_PAYLOAD.md). Its sender is fixed.
- The **PC** runs `control_deploy run` **unchanged**, from `rnoro/final-deployment`. It is the only machine that drives the
  arm: safety gate, 1 kHz Cartesian impedance loop, gripper, homing, watchdog, and the C2 state broadcast at 100 Hz.
- The **laptop** runs the new `teleop_collect run`. It receives the Pi stream and the C2 state, turns held keys into C3
  `set_waypoints`, records episodes, and serves the operator's page in the browser.

The operator commands **only the end effector**: 3D translation, 3D rotation, and fingers open/close. The joints follow,
because the PC's 1 kHz loop tracks a Cartesian TCP pose. The laptop never computes joint angles.

Out of scope: converting episodes into training data (done elsewhere, later); a policy; any change on the PC or the Pi.

## 2. Rules

1. **`rnoro/final-deployment` is never modified.** Code is copied from it as needed, and each copied file says so in its
   header: `# copied from rnoro/final-deployment:deployment/<path> @ 2ffe641` (or `adapted from`, with a `changes:`
   line), as that branch did with `rnoro/deployment`.
2. **The PC and the Pi are unchanged.** The laptop speaks only the existing C2/C3 contracts to the PC (fd:STRUCTURE.md
   §3) and only C0 from the Pi. No new C3 message, no PC rebuild, no PC configuration beyond what
   fd:control_deploy/DIRECTIONS.md already asks for.
3. **One command on the laptop:** `teleop_collect check | run [--mock] [--station FILE]`. `check` says whether
   everything is ready, `run` starts the processes and opens the page, and `--mock` replaces the Pi and the PC with
   stand-ins. Settings come from one station file. When something fails, one line says which process, what failed and
   what to check.
4. **The laptop never starts, stops or reads files on the PC or the Pi**, as in fd:PLAN.md rule 2.
5. **Self-contained folder:** `deployment/teleop_collect/` holds its code, config, tests and directions. It imports
   nothing from `policy_deploy` or `control_deploy`. The shared C2/C3 code is `deployment/contracts/`, a **byte-for-byte
   copy** of fd:contracts/ (the `deploy_contracts` package, `limits.yaml`, `frames.yaml`, `link.yaml`, `examples/`,
   its tests). It is never edited in this branch except `link.yaml`'s addresses, which must always equal the copy the PC
   runs with (§4.1).
6. **Modular and object-oriented**, one job per class. Every part that may be swapped (the camera source, the PC, the
   key source) has a real and a fake implementation. Numbers live in config.
7. **Code first, with unit tests**, for every class and every new contract. The tests run on the laptop without the Pi,
   the arm or a GPU.
8. **No code before its STRUCTURE.md.** The agent-team method as before: one executor per part, one reviewer over all of
   it, every open question settled with the user before code, a local commit per wave, nothing pushed without asking.
9. **Real world only.** The MuJoCo sim stays separate.
10. **The physical user stop is the emergency stop.** Nothing in software replaces it.

## 3. Machines, folders and processes

```
                         LAPTOP: teleop_collect/                                 PC: control_deploy/ (unchanged)
 Pi 5 ──C0 pi_stream/v1──► frames ──R1──► teleop ◄──C2 state 100 Hz────────── franka_control
       (Wi-Fi ASUS_5g)     (receive,      (episodes,  ──C3 start/stop/───────►  (gate, 1 kHz loop,
                            record,        teleop loop,    set_waypoints,       gripper, homing,
                            preview)       PC clock map,   hello, heartbeat)    watchdog, keyboard)
                              ▲            record state)        │                    │ FCI
                              │ U2               ▲ U1           │                    ▼
                              └───────── ui (web server, 127.0.0.1) ◄── browser   Franka Panda + Franka Hand
```

| Process | What it does | Derived from |
|---|---|---|
| **frames** | Receives the Pi stream (C0), checks it against the contract and the rig file, and maps Pi time to laptop time (one-way fit from `t_send_ns`). It tracks each camera's health and pair rule. While an episode records, it writes **every frame's JPEG exactly as received into one video file per camera** (MJPEG in Matroska, `main.mkv` / `ultrawide.mkv`: the JPEG bytes are muxed as they are, never re-encoded), plus their times, into the episode folder. It keeps the newest JPEG of each camera for the page's preview. It does not prepare 224×224 images, because nothing consumes them. | fd:policy_deploy/policy_deploy/frames/ (receiver, Pi stream source, clock map, health, pair rule, fake Pi) |
| **teleop** | The 20 ms main loop: keep-alives and every C3 request (rule H1), the PC clock map, the C2 history. It turns the key state into waypoints (§5), runs the episode lifecycle, owns the task/episode store, and records C2 and the commands while an episode records. | fd:policy_deploy/policy_deploy/manager/ (pc_client, pc_listener, pc_clock, heartbeat_sender, startup_sequence, startup_checks, failure monitor, state history, session log) |
| **ui** | A small web server on `127.0.0.1` only. It serves one page: task picker, episode list and buttons, two live previews, the status panel, and the key capture. It forwards the key state and button presses to teleop (U1) and pulls previews and health from frames (U2). It owns nothing; it shows and forwards. | new |
| **fake_pc** (`--mock`) | The Python stand-in for franka_control: C2/C3 on 127.0.0.1, a drifting offset clock, the fake gate. | fd:policy_deploy/policy_deploy/fake_pc/ |
| fake Pi (`--mock`, inside frames) | Plays the recorded Pi episode `~/Downloads/data/incoming/episode_000152` as a C0 stream. | fd:policy_deploy/policy_deploy/frames/fake_pi_sender.py |

The launcher starts `fake_pc` (only under `--mock`), then `frames`, `teleop` and `ui`, each after the previous one is
ready, and stops them in reverse. Any process exiting stops the rest (fd:policy_deploy/policy_deploy/launcher/).

Why three processes: the same reason as in policy_deploy. Writing about 4 MB/s of JPEGs and serving previews to a
browser must never delay the 20 ms loop that keeps the PC's watchdog fed and times the waypoints. Each process can be
started and watched on its own (`python -m teleop_collect.<frames|teleop|ui>`).

**New laptop-only contracts**, defined and tested inside `teleop_collect/` (like policy_deploy's C1/C4). All are local
REQ/REP on `127.0.0.1` with ports from the station:

| Contract | Between | Requests (detailed in STRUCTURE.md) |
|---|---|---|
| R1 | teleop → frames | `start_recording(episode_dir)`, `stop_recording()` → per-camera frame counts, gaps, first/last times; `get_health` |
| U1 | ui → teleop | `get_status`, `keys(held, slow, focus)` (sent on every change and at least every 50 ms), `gripper(close|open)`, `select_task(name)`, `create_task(name, description)`, `start_episode`, `stop_episode(outcome)`, `delete_episode(id)`, `label_episode(id, outcome)` (save an unsaved one), `delete_task(name)` (the folder and its episodes), `set_data_root(path)` (the storage folder, remembered), `stop_arm` |
| U2 | ui → frames | `get_preview` → the newest JPEG of each camera and its age, plus health |

A C3 request comes only from teleop's main thread: the ui never talks to the PC.

## 4. The lab procedure

### 4.1 Once, before the first collection day

1. **The PC's bring-up**, exactly as fd:control_deploy/DIRECTIONS.md steps 1–10 (not done yet on the real arm). In
   particular:
   - **Measure the workspace box** and put it in the PC's `control_deploy/station.yaml`: it is the box of the
     bring-up and of a policy run (default `workspace_min_m` x 0.20 since fd:`447910b`). **For teleop only its floor
     is kept** (the PC's teleop block, §5.3): the sides and the cone are opened, the collision thresholds raised, and
     the margins set to 0.05 m / 10°. The floor (`workspace_min_m` z) is 5 mm below the fingertips resting on the
     table (about `-0.005`, fd:'s default since `927a5ba`): grasps end at the table. Teleop's moves stay within the
     gate's 5 cm pull-back margin below it (§5.3), so they never fault there.
   - **Measure the start pose** (`homing.q_start_rad`, fd:control_deploy/DIRECTIONS.md step 10): hand-guide the arm to
     the training demos' start, read it with `pc_probe --start-pose`, paste it into the PC's station, then commit and
     push it on `rnoro/final-deployment` (the one value tying teleop data to the deployment). Every teleop episode
     starts there (§8).
   - Weigh the rig into the payload, set `gripper.width_offset_m`, and run the `console` steps (`hold`, `move`,
     `rotate`, `gripper`, `watchdog`, `out_of_box`). This is the first time the arm follows waypoints.
2. **The addresses** in `contracts/link.yaml` (the PC's and the laptop's) and the Pi's address in the laptop's station:
   the same values as fd:policy_deploy/DIRECTIONS.md step 4. **This branch's `contracts/link.yaml` and the PC's
   (`rnoro/final-deployment`) are edited the same way by hand.** A wrong address fails the connection with a clear
   line; a differing watchdog is caught by `hello` at start-up.
3. **The laptop:** its own environment `~/envs/teleop_collect` (Python 3.10), with this branch's `deployment/contracts`
   and `deployment/teleop_collect` installed editable. **Never install into `~/envs/policy_deploy*`:** those envs
   install `deploy_contracts` from `~/Capstone-final-deployment`, and a second editable install of the same package name
   would repoint them. Then `teleop_collect run --mock` once (fake Pi, fake PC), then `teleop_collect check` against the
   live Pi and PC.

### 4.2 Every collection session

1. **Physical:** router on, arm controller on, user stop plugged in and within reach, table clear apart from the scene.
2. **PC, Desk:** `https://172.16.0.2`: unlock the brakes, activate FCI, release the user stop.
3. **PC:** `control_deploy run`, then wait for `control_deploy: ready`. The fingers home once and the arm holds. It stays up
   all session. Then `h` (the whole arm to the home pose), then `p` (to the start pose, the hand opens).
4. **Pi:** press the C button; the LED turns blue.
5. **Laptop:** `teleop_collect run`. It runs the checks (Pi reachable, C2 arriving, `hello` matching, clock map ready),
   waits for both cameras, takes **the arm's pose now as this session's start pose** (§8; the arm does not move), and
   prints `teleop_collect: ready: open http://127.0.0.1:<port>`. The page opens in the browser.

**Always the PC first, then the Pi, then the laptop.** The page can open earlier, but **Start episode** stays disabled
until the arm, the link and both cameras are healthy. So the moment Start is clicked, data flows.

### 4.3 In the page

```
┌ task: pick_block ▾  [+ new task] ──────────────────────────────────────────────────────────────────┐
│ ┌ main (IMX519) ────────┐ ┌ ultrawide (IMX219) ───┐   arm: idle   tcp 0.452 0.003 0.301 m       │
│ │                       │ │                       │   grip 0.080 m (open)   gate a0 s0 p0 r0     │
│ │      live preview     │ │      live preview     │   cams ok 30/ok 30 fps   c2 100 Hz           │
│ │                       │ │                       │   rtt 4 ms   clock ok   speed normal         │
│ └───────────────────────┘ └───────────────────────┘                                              │
│ [Start episode]  [Stop arm]                    keys: W/S A/D R/F move · I/K J/L U/O rotate ·      │
│ at the start pose ✓ ready to start                    N close · M open fingers · Shift slow        │
│ episodes: 000 success 24.1 s   001 fail 31.0 s   002 success 19.8 s  [delete]                      │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Pick a task** from the list, or **New task** (a name, lowercase letters, digits and `_`, plus an optional
   one-line description). The episode list shows that task's episodes.
2. **The start pose before every episode:** `h`, then `p` on the PC's `control_deploy run` (its own keys). **Start
   episode is enabled only while the arm is at this session's start pose** (§8), so every episode starts from the same
   pose, the one the deployment starts from. Then set up the scene.
3. **Start episode:** the laptop sends C3 `start` (the PC holds the start pose), frames starts writing, teleop starts
   recording C2, and the keys go live. The page shows `REC episode 003 0:00`. **Wait about a second before the first
   key**, so the recording opens on the still scene.
4. **Drive:** hold keys to move, release to stop (§5). `N` closes the fingers (a grasp with force), `M` opens them.
5. **Stop episode**, then choose **Save: success**, **Save: fail** or **Discard**. Motion stops, the laptop sends C3
   `stop` (the arm holds where it is), and the recording is closed. A saved episode is complete on disk at once;
   a discarded one is deleted.
6. Repeat from 2: `h`, `p`, scene, Start. **Delete** removes any listed episode, after a confirm.
7. **Quit:** close with Ctrl-C in the terminal (the page then shows "disconnected"). The PC keeps holding the arm.
   Then the Pi's C button (LED off); and at the end of the day, on the PC, `q`, then lock the brakes in Desk.

The PC's own keys (`h` home, `p` start pose, `g` home the hand, `s` stop, `q` quit) are the only way to move the arm
to a pose: the laptop has no home of its own (as policy_deploy since fd:`927a5ba`). While the PC shows `homing`, the
page refuses Start with a hint.

### 4.4 When something goes wrong during an episode

- **Hands off the keys:** the arm holds at once (§5.2). The user stop is always the hard stop.
- **Stop arm** (button, or `Esc`): C3 `stop`, motion off. The open episode is aborted, which means **deleted** (§7).
  Between episodes it also stops a move of the PC's `h` or `p`.
- **Nothing in the software stops the arm** except Franka's own limits and reflexes, the operator (keys released,
  Stop arm, the PC's `s`, the user stop) and a dead link (the PC's 1 s watchdog). Data problems (a camera down or
  stale, a failed recording) are warnings: the episode goes on (§7).
- The page states anything that ended an episode by itself, in one line (§7).

## 5. The teleop loop

### 5.1 One cycle (every `link.yaml timing.heartbeat_period_s`, 20 ms)

The teleop main loop is policy_deploy's `EpisodeLoop` rhythm with the policy step replaced by a teleop step. Each
cycle it:

1. Checks the signal flags and the C2 state rules (fault, `seq` going back, no C2 for `state_lost_s`), as
   fd:policy_deploy/policy_deploy/manager/failure_monitor.py.
2. Reads the newest key state from the ui (U1). A key state older than `key_stale_ms` (150 ms) counts as **all keys
   released**, which is the deadman rule.
3. Sends **exactly one C3 request** (rule H1, on the one REQ socket):
   - a `hello`, when one is due (every 0.5 s; it feeds the PC clock map, as before);
   - otherwise, while an episode is live and a key is held or was just released, or the finger command changed: a
     `set_waypoints` (§5.2), which also feeds the PC's watchdog;
   - otherwise a `heartbeat`.
4. Polls frames' health (R1) every 500 ms.
5. Handles one U1 request from the ui (buttons, status).
6. Writes the cycle's records (§6) when an episode records.

A button handled in step 5 (Start, Stop, Stop arm) adds its own C3 request (`start`, `stop`) in the same cycle, still
on the main thread and the one REQ socket. Rule H1 holds and the worst gap is unchanged.

The C3 timeout stays `link.yaml timing.c3_timeout_s` (0.3 s), so the worst gap between C3 requests is far below the
PC's 1.0 s watchdog, as checked at start-up against `hello` (fd:STRUCTURE.md §3.9).

### 5.2 From held keys to waypoints

- The held keys give a **twist**: a translation velocity `v` along the robot base axes and a rotation rate `ω` about the
  fingertip point (§13.2 for which axes the rotation uses). Each key adds ±`v_max` or ±`ω_max` on one axis; opposite keys
  cancel. **Shift** scales both by `slow_factor` (0.25).
- The base of every command is **the PC's own latest target**: C2 `O_T_TCP_target` from the newest state, at its PC time
  `t_s`. The laptop never integrates a pose of its own, so it cannot drift away from where the arm really is.
- A `set_waypoints` holds `K` waypoints (6), at PC times `t_k = now_pc + k × Δ` (Δ = 50 ms, so the lead is 300 ms), with
  `now_pc` from the PC clock map. Each pose is the base pose moved by the twist over `t_k − t_s`: translation added in the
  base frame, rotation applied about the TCP point. The width of every waypoint is the commanded finger width.
- **Release:** when every key is released, one last set holds the arm. Its single waypoint at `now_pc + Δ` is the base
  pose advanced by the old twist over the state's age only (its best guess of where the target is now). After that, only
  heartbeats go out, and the arm holds after the last waypoint. The release never advances past the end of the last
  plan sent, so after a stale gap the arm doesn't creep.
- **Only while following:** no set is sent unless the newest C2 mode is `running` or `holding` (the PC refuses
  `set_waypoints` otherwise). The width commanded at Start is the measured width **clipped into [0,
  `gripper_width_max_m`]**, since an open hand can read slightly above the PC's limit.
- **Fingers:** `N` sets the commanded width to 0 (closed) and `M` to `gripper_width_max_m` (0.08, open). Closing
  makes the PC's Franka Hand **grasp with force** and hold wherever an object stops the fingers (fd:control_deploy/
  franka_control/src/franka_hand_gripper.cpp). `N` or `M` with no motion key held sends one set with the current pose and the new
  width.
- The PC does the rest, as for a policy: waypoints already past are skipped, the rest are gated, and they replace the
  plan. The 1 kHz loop interpolates from its current target, so there is no jump (fd:PLAN.md §5.4). In C2 the mode is
  `running` while it follows and `holding` once the plan ends.

### 5.3 Why teleop never faults the PC at the box and the cone

The PC's gate **pulls back** a waypoint that lies at most `box_margin_m` outside the workspace box or
`cone_margin_deg` outside the tilt cone, and **rejects** it beyond that, which is a `fault` and needs a PC restart.
For teleop the PC runs its **teleop block** (the PC's directions, step 8, "Teleop settings"): the box open except the
floor, the cone 179°, and the margins **0.05 m / 10°**, which teleop assumes (`motion.assumed_pc_box_margin_m`,
`assumed_pc_cone_margin_deg`). The PC does not report its margins, so `teleop_collect check` shows them as an INFO row
(`PC margins`) and start-up as a page message: `teleop assumes the PC's safety.box_margin_m 0.05 and cone_margin_deg
10 (the teleop block of the PC's station.yaml); with smaller PC margins a press into the floor can fault the PC`.
Every command starts from the PC's target, which the gate has already pulled back onto the boundary. So a command can
stick out past the boundary by at most `speed × (lead + state age)`. The settings keep that inside `margin_use` (0.8)
of the margins:

| | Default | Worst step beyond the boundary, lead 300 ms + state age ≤ 250 ms | 0.8 × PC margin |
|---|---|---|---|
| translation `v_max` | 5 cm/s | 27.5 mm | 40 mm |
| rotation `ω_max` | 10 °/s | 5.5° | 8° |

- **The guard:** teleop sends no set at all while the newest C2 is older than `state_max_age_ms` (250 ms, mapped PC
  time), only heartbeats. The arm finishes its last plan (at most the 300 ms lead) and holds, so the bound holds even on
  a bad Wi-Fi moment. A holding set built on a stale state could stick out further than the margin, so none is sent.
  A C2 `seq` gap alone never pauses motion: the planner extrapolates from the newest target, so a lost state only
  matters at a key change (counted and logged).
- **Both tilt keys together** (I/K with J/L): the tilt rate is normalised to at most `ω_max`, so the cone bound holds for
  diagonal tilts too. Translation keys are per axis, like the box itself.
- **What you see:** at the floor (or a box wall or the cone edge, with the deployment block) the arm simply stops
  moving in that direction, and the page shows `at workspace limit` from C2's `gate.pulled_back` counter.
- **Speed is never pushed** (§13.3): these defaults stay, and nothing on the laptop copies the PC's box or cone
  to go faster. The PC's gate stays the one safety authority.
- **The other gate rules** never trigger with these settings: the speed limit is 0.25 m/s and 1 rad/s, the lead limit
  1.5 s, the width range [0, 0.08].

### 5.4 Default key map

Recognised only while the page has focus and an episode is live. Losing focus counts as releasing every key.

| Keys | Motion |
|---|---|
| W / S | +x / −x (away from / toward the robot base) |
| A / D | +y / −y (left / right, seen from behind the robot) |
| R / F | +z / −z (up / down) |
| U / O | twist the hand about its own approach axis |
| I / K | tilt forward / back |
| J / L | tilt left / right |
| N | close the fingers (grasp with force) |
| M | open the fingers fully |
| Shift (held) | slow mode |
| Esc | **Stop arm** (C3 `stop`; the open episode is aborted and deleted) |

## 6. The episode on disk

### 6.1 Layout

```
~/CapstoneTeleop/                                  station: data_root
  <task>/
    task.yaml                                      name, description (the task in one line), created, the settings
    episode_000/
      meta.yaml                                    written at stop (6.3)
      main.mkv                                     every main frame's JPEG exactly as received, one video frame each
      ultrawide.mkv                                every new ultrawide frame's JPEG exactly as received (a reused one is
                                                   not written again)
      packets.csv                                  one row per C0 packet (6.2)
      main_frames.csv, ultrawide_frames.csv        one row per written frame (6.2)
      robot_states.csv                             every C2 message while recording (6.2)
      commands.csv                                 every C3 request teleop sent while recording, with its reply (6.2)
      keys.csv                                     every key state change from the ui
      clock_pi.csv                                 the Pi -> laptop fit, once per second
      clock_pc.csv                                 every hello sample (send, receive, t_pc_ns) and the PC fit, once per second
      events.jsonl                                 start, stop, hints, problems, with laptop times
      COMPLETE                                     written last: an episode without it is incomplete
    episode_001/
```

- **Everything raw, with every clock.** Times are integer ns: Pi times end `_pi_ns`, PC times `_pc_ns`, laptop
  times `_ns`, as fd:STRUCTURE.md §6.1. Mapped values are stored next to the raw ones, never instead of them. The raw
  samples of both clock maps are kept, so the alignment can be refitted offline.
- **Images are the JPEGs the Pi sent**, byte for byte (640×480), stored as the frames of one MJPEG video per camera
  (Matroska container). Nothing is decoded or re-encoded, so it costs almost no CPU, needs no per-frame files, and
  plays in any video player. Frame `i` of the video is row `i` of `<camera>_frames.csv`, which holds its true times
  (the video's own timestamps are only for playback). The page's preview uses the same bytes. A smaller H.264 file
  would mean decoding and re-encoding every frame (CPU while recording, and pixels that differ from what the deployed
  policy receives), so it is not used (§13.6).
- **Episode numbers** just count up within a task (`episode_000`, `episode_001`, ...). A deleted or discarded number
  is never reused, so gaps are normal and mean nothing.
- **Crash safety:** files are appended while the episode records. `meta.yaml` and `COMPLETE` are written at stop. A
  folder without `COMPLETE` (the laptop crashed or lost power mid-episode) is deleted at the next start-up, like any
  aborted episode, and the session log says so.

### 6.2 The files, column by column

| File | Columns |
|---|---|
| `packets.csv` | `packet_no`, `t_send_pi_ns`, `arrival_ns`, `main_frame_no`, `ultrawide_frame_no`, `uw_reused`, `pair_used` (the pair rule's verdict), `pair_gap_ms` |
| `main_frames.csv`, `ultrawide_frames.csv` | `video_index` (the frame's position in the `.mkv`), `frame_no`, `t_sensor_pi_ns`, `exposure_us`, `t_mid_pi_ns`, `t_mid_ns` (mapped to the laptop), `packet_no`, `arrival_ns`, `bytes` |
| `robot_states.csv` | `t_pc_ns`, `t_ns` (mapped to the laptop), `arrival_ns`, `seq`, `mode`, `O_T_TCP` (16 values, row-major), `O_T_TCP_target` (16), `q` (7), `dq` (7), `gripper_width_m`, `gate_accepted`, `gate_shortened`, `gate_pulled_back`, `gate_rejected`, `loop_late`, `error` |
| `commands.csv` | `t_send_ns`, `t_reply_ns`, `request` (schema), `chunk_id`, `n`, per waypoint `t_pc_ns`, pose (16) and width, the twist (6), `slow`, the reply's `accepted`, `shortened`, `skipped`, `rejected` (a count), `rejected_reasons`, or the ack (`ack_ok`, `ack_reason`) |
| `clock_pc.csv` | one file with a `kind` column: `hello` rows (send, receive, `t_pc_ns`) and `fit` rows (offset, drift, samples) |
| `keys.csv` | `t_ns` (arrival at teleop), `held` (the key names), `slow`, `focus` |

### 6.3 `meta.yaml`

Task and episode number; outcome (`success` or `fail`; an aborted episode is never saved); start and stop times on the laptop and
the PC; duration. Counts: main and ultrawide frames written, `packet_no` gaps, `frame_no` gaps, `uw_reused`, C2
messages, C2 `seq` gaps, waypoint sets sent, skipped, shortened, `pulled_back` during the episode, and loop-late cycles
during the episode, and camera-down / recording-failed counts. `warnings`: each data problem seen while recording
(`code`, `what`, `t_s` since Start), once per code and camera. The full `hello` reply (the PC's commit, `mock`,
**`F_T_TCP`**, the limits, `max_lead_s`). The laptop's commit and whether it was dirty. The rig file's path and its
content. The teleop settings (speeds, K, Δ). The final Pi and PC clock fits.

`F_T_TCP` is stored because every pose on C2 is `O_T_F × F_T_TCP`. With it, the flange pose and any other TCP can be
recovered offline if the TCP definition changes later (fd:PLAN.md §9; §13.9).

### 6.4 Rates and size

About 30 main and 30 ultrawide JPEGs per second (about 48 kB and 75 kB each on the live Pi), so about 3.7 MB/s, about
110 MB for a 30 s episode (the container adds well under 1 %). C2 adds 100 rows per second (well under 1 MB/s). frames writes on its own writer thread with a
bounded queue. If the queue fills, that is a warning: the episode can then only be discarded (§7).

## 7. Failure rule

Every problem is one of four kinds:

- **Slow → keep going.** Motion pauses (the arm holds) until things catch up, and the recording continues. Counted and
  shown on the page.
- **Warn → keep going, note it.** A data problem: the arm and the episode go on. Shown once per episode on the page,
  written to the episode's `events.jsonl` and to `meta.yaml` `warnings`. Outside an episode it is normal.
- **Episode → abort the episode.** The laptop stops the arm (C3 `stop`). The open episode is **aborted: its folder is
  deleted**, and the reason is shown on the page and written to the session log (`~/teleop_collect_sessions/`), never
  to the data root. The app stays up, ready for the next episode.
- **Conclude → end the run** (a failure). As in policy_deploy: C3 `stop`, the open episode aborted (deleted), the laptop's processes
  stopped in reverse order, and one line saying which process, what failed and what to check. **The PC keeps running
  and holding** (§13.8).

| Problem | Kind | Reaction |
|---|---|---|
| Newest C2 older than 250 ms (mapped) | slow | no motion this cycle (no set, only heartbeats) |
| A C2 `seq` gap | normal | counted and logged ("a C2 seq gap (motion goes on)"); motion goes on |
| Wi-Fi spike: a waypoint set skipped entirely by the PC | slow | counted; the next cycle sends a new one |
| The page lost focus, the ui stopped sending keys (150 ms) | slow | keys released: the arm holds; recording continues |
| Waypoints pulled back at the box or cone | normal | the arm stops at the limit; `at workspace limit` |
| A camera stale (no new frame for 0.35 s) during an episode | warn | the arm and recording go on; noted in `meta.yaml` `warnings` |
| A camera down (2 s), the Pi stream restarted, during an episode | warn | the arm and recording go on: "episode_NNN: main camera down: check the Pi and ASUS_5g; the episode goes on (noted in its meta.yaml)"; it can be saved (teleop-train's frame-gap check will likely exclude it) |
| The PC `stopped` without the laptop asking (its watchdog, or `s` on the PC) | episode | aborted: "the PC stopped the arm" |
| The PC `homing` started from its keyboard during an episode | (cannot happen) | the PC refuses `h`/`p`/`g` while `running`/`holding` |
| Operator **Stop arm** / Esc | episode | aborted: "operator stop" |
| frames' write queue full, a disk write error, during an episode | warn | the arm goes on; Save is refused ("this episode cannot be saved (...): Discard it"), only Discard works |
| The same found only at Stop (after the C3 stop) | episode | the episode deleted: "recording could not keep up / disk error" |
| A camera down, or the clock map not ready, **between** episodes | normal | Start is disabled, with the reason |
| No C2 for 2 s, a C3 timeout | conclude | "PC not answering: check franka_control on the PC and the link" |
| C2 `seq` goes backwards | conclude | "franka_control restarted on the PC" |
| The PC reports `fault` (reflex, rejected waypoint, lost FCI) | conclude | the PC holds; on the PC Ctrl-C and `control_deploy run` again (release the user stop after a reflex) |
| Clock mismatch (fd:PLAN.md §5.6), `hello` disagreeing at start-up | conclude | names the field |
| A message failing its contract (C0, C2, C3, R1, U1, U2) | conclude | names the message and field |
| Start refused by the PC with `refused: not at the start pose` | normal | nothing recorded; the learned start pose is dropped (§8); `the PC refused start: ...` |
| frames, teleop or ui exits | conclude | the launcher stops the rest |
| At start-up: anything `check` finds wrong, the PC in `running`/`holding` (another C3 client) or `fault` | conclude | refuses to start and says why |
| Operator Ctrl-C | conclude (normal) | `stop`; an open episode aborted (deleted); exit 0. The launcher prints `stopped by Ctrl-C`; teleop logs "stopped by the launcher (Ctrl-C, or another process exited)", since it only sees the launcher's SIGTERM |

**One C3 client at a time** (fd:STRUCTURE.md §3.7.3): `teleop_collect`, `policy_deploy`, `pc_probe` and the PC's
`console` never run together. Rule H2 makes `teleop_collect` refuse to start while another client drives the arm.

## 8. Start-up of `teleop` (in order)

As fd:policy_deploy/STRUCTURE.md §6.3, without the policy: C2 arriving → H2 wait (no C3 while C2 shows
`running`/`holding`, up to 2× the watchdog, or `homing`, up to 60 s; `fault` refuses) → hellos every cycle until the PC
clock map is ready (≥ 2 s and ≥ 10 samples) → `compare_hello` (schemas, `mock`, `F_T_TCP`, the limits, the watchdog)
and the laptop's budget checks → `get_state` shows `idle`/`stopped` → heartbeats → frames reports both cameras ok →
the data root is writable → **the start pose: the arm's pose now** (from the newest C2, when it shows `idle` or
`stopped`; while the PC is moving the arm, the pose after that move) → ready. **No step moves the arm.**

**The start pose, before every episode.** The PC owns it: its key `p` moves the arm from home to
`homing.q_start_rad` (the training demos' start, measured once by hand-guiding, §4.1), and once that is set its C3
`start` is refused anywhere else (`refused: not at the start pose: press h, then p on the PC`). The laptop never reads
the PC's station; it learns the pose from the arm, as follows:

- **At start-up:** the operator pressed `h`, then `p` on the PC before `teleop_collect run`, so the arm's pose at the
  end of start-up is the **candidate** (joints `q` and the TCP target position).
- **Until confirmed:** a C2 `homing` → `idle` in which the arm moved (a joint by more than `start_pose.tolerance_rad`
  or the TCP target by more than `start_pose.tolerance_m`) replaces the candidate with the pose after it: the PC's
  `h`, then `p` end at the start pose. The PC's `g` (the hand only, the same `homing` mode) moves neither and never
  counts.
- **The PC has the final say:** the first start it accepts **confirms** the candidate, fixed for the session. A start
  it refuses with `refused: not at the start pose` drops the candidate; the pose after the PC's next move is the new
  one.

Start episode is enabled only while every joint of the newest C2 is within `start_pose.tolerance_rad` (0.01 rad) of the
start pose, its TCP target within `start_pose.tolerance_m` (5 mm), and the mode is `idle` or `stopped`. Otherwise the
page says `put the arm at the start pose: h, then p on the PC` (none known), `not at the start pose: h, then p on the
PC`, or `the PC refused the start here (not the start pose): press p on the PC`.

**With `homing.q_start_rad` unset** the PC refuses `p` and checks nothing at `start`: the operator presses only `h`,
the start pose is home, and the first start confirms it. **The start pose must be the same for collecting and for
deploying:** a policy trained on demos that started at home must be deployed with `homing.q_start_rad` unset too, or
the demos recorded again once it is set.

**Why the laptop never moves the arm to a pose.** `control_deploy` stays unchanged (rule 2), and since fd:`927a5ba`
homing and the start pose are its keyboard keys only (policy_deploy has no laptop home either). At its own start it
homes only the fingers and holds the arm where it is, so the operator presses `h`, then `p` there. A commit different
from the PC's is expected (this branch versus `rnoro/final-deployment`). It is shown and recorded, never refused. The
`settled` rule of fd:PLAN.md §9 is about a trained policy and does not apply to teleop.

## 9. What is copied, and from where

Candidates. The final list, file by file, is STRUCTURE.md's job.

| teleop_collect part | Copied or adapted from (fd:) |
|---|---|
| shared C2/C3 code and the three contract files | `contracts/`, copied whole and unchanged to `deployment/contracts/` (rule 5) |
| C0 parsing, internal packet, clock fit, station, config merge, process args | `policy_deploy/policy_deploy/contracts/` (`pi_stream.py`, `c0_packet.py`, `clock_fit.py`, `station.py`, `config_merge.py`, `process_args.py`, `paths.py`) |
| frames: receiver, Pi stream source, health, pair rule, rig file, fake Pi | `policy_deploy/policy_deploy/frames/` (`receiver.py`, `pi_stream_source.py`, `camera_health.py`, `pair_rule.py`, `packet_pairer.py`, `rig_file.py`, `fake_pi_sender.py`, `fake_pi.py`, `pi_recording.py`, `errors.py`); the rig file is the deployment's `contracts/rigs/rig_pi_v3_640.yaml`, read from `../contracts`. Changes: no image prep or selection; a recording tap on the raw JPEG bytes; the latest JPEG for the preview |
| teleop: the PC link, clock map, start-up, failure monitor | `policy_deploy/policy_deploy/manager/` (`pc_client.py`, `pc_listener.py`, `pc_clock.py`, `heartbeat_sender.py`, `link_stats.py`, `state_history.py`, `startup_sequence.py`, `startup_checks.py`, `failure_monitor.py`, `problems.py`, `pose_math.py`, `session_log.py`), and `episode_loop.py`'s rhythm. Changes: no policy and no frames C1; the teleop step instead of the policy step; set_waypoints may be the cycle's keep-alive (§5.1) |
| fake PC | `policy_deploy/policy_deploy/fake_pc/` (whole) |
| launcher | `policy_deploy/policy_deploy/launcher/` (`launcher.py`, `managed_process.py`, `processes.py`, `check_helpers.py`, `check_result.py`, `session.py`, `cli.py`) |
| tools | `frames/pi_collect.py` and `manager/pc_probe.py`, unchanged in behaviour |
| **new** | the teleop step (key state → twist → waypoints), the episode controller and store, the frame and state recorders, R1/U1/U2, the ui server and its one static page |

## 10. Tests

All unit tests run on the laptop without the Pi, the arm or a GPU (rule 7).

- **Contracts:** R1/U1/U2 examples pass their validators and broken ones fail. The copied C2/C3 code keeps its tests.
- **Teleop step:** twist from keys (cancelling, slow mode, stale keys = released); waypoint times and poses for a given
  base state; the release set; the `N`/`M` finger commands; Start refused away from the start pose; the margin bound of §5.3 for every key combination and every state
  age up to the guard; no motion while C2 is too old.
- **Recorder:** every frame of each `.mkv` demuxes to exactly the JPEG bytes that arrived, and `video_index` matches; the CSV rows; a reused ultrawide not written
  twice; `COMPLETE` last; an aborted or incomplete episode deleted; delete; numbering with gaps.
- **Episode flow:** against the fake PC and a recorded Pi stream, through U1: create task, start, keys, stop as
  success/fail/discard, delete; every row of §7.
- **End to end:** `teleop_collect run --mock` with a scripted page client (keys and buttons over U1).
- **Across:** the laptop's processes against a real libfranka-less `franka_control --mock --local` (it is already built
  at `~/Capstone-final-deployment/deployment/control_deploy/franka_control/build-nolibfranka/`; run, never rebuilt or
  changed there).

## 11. Hardware bring-up order

1. **The PC alone:** fd:control_deploy/DIRECTIONS.md steps 1–10 (§4.1). Nothing teleop-related can be tested before the
   arm follows waypoints in `console`.
2. **The laptop and the Pi:** `teleop_collect run` with the PC not running fails its check, as it should. With
   `frames` alone, the previews and `pi_collect`.
3. **First teleop:** arm in the middle of the box, slow mode only, one hand on the user stop. Each key, then the box
   walls and floor (expect `at workspace limit`, no fault), then the fingers on an object.
4. **First recorded episode:** record about 10 s, discard nothing, and inspect the files (frame counts, times, poses
   against the video).
5. **Collection.**

## 12. How the work is done

The agent-team method, arranged for speed. **All the design that crosses between agents is fixed in STRUCTURE.md
before any executor starts**, so the executors of a wave work in parallel without waiting on each other. Every open
question of §13 is settled with the user first. **One reviewer checks the entire set of changes at the end.**

### 12.1 Waves

| Wave | Who | Work | Done when |
|---|---|---|---|
| **0** | the lead (no agent) | (1) Copy fd:contracts/ to `deployment/contracts/` (sources, yaml, examples, tests; no build or cache files) and prove it identical with `diff -r`. (2) Create `~/envs/teleop_collect`. (3) Write `STRUCTURE.md`: the package tree with **an owner for every file**, every class and function that crosses between two executors (signatures, types, errors), the R1/U1/U2 field tables and examples, config keys and defaults, station keys, ports, the source of every copied file, and the test list per executor. | the user agrees to STRUCTURE.md; local commit |
| **1** | 1 executor: **E1 foundation** | The package skeleton (`pyproject.toml`, `__main__`, `config/*.yaml`, `station.yaml`, `rigs/`). `teleop_collect/contracts/`: the copied C0 parsing and internal packet, the clock fit, the station and config loaders, the process arguments and paths; the new R1/U1/U2 messages and their examples. Unit tests. `teleop_collect/contracts/DESIGN.md` (§12.5). Everything else imports this, so it goes alone, and it is small. | its tests green; walk-through; local commit |
| **2** | 5 executors **in parallel**, each in its own files | **E2 frames:** `teleop_collect/frames/`: the copied receiver, Pi source, health, pair rule, rig file and fake Pi; the frame recorder (writer thread, bounded queue, `packets.csv`, `*_frames.csv`, and the two MJPEG `.mkv` files muxed from the received JPEG bytes); the preview; the R1 and U2 servers; the process entry; `pi_collect`. **E3 teleop link:** `teleop_collect/teleop/` without `motion/` and `episode/`: the copied PC client, listener, PC clock map, heartbeat sender, link stats, state history, start-up sequence and checks, failure monitor and problem codes, session log, `pc_probe`; the main loop (§5.1), the U1 server, the process entry. **E4 teleop core** (pure logic, no sockets): `teleop_collect/teleop/motion/` (key state, twist, the waypoint planner of §5.2–5.3) and `teleop_collect/teleop/episode/` (the task and episode store, the state, command and key recorders, `meta.yaml`, `COMPLETE`). **E5 ui:** `teleop_collect/ui/`: the web server, the one static page (key capture, focus rule, previews, buttons, status), the U1/U2 clients, a scripted page client for tests. **E6 launcher and fake PC:** `teleop_collect/launcher/` (`check` and `run`, the four processes, the check rows) and `teleop_collect/fake_pc/` (copied). **Each executor also writes the DESIGN.md of every folder it owns** (§12.5): E2 `frames/`, E3 `teleop/`, E4 `teleop/motion/` and `teleop/episode/`, E5 `ui/`, E6 `launcher/` and `fake_pc/`. | each executor's own tests green; walk-through; local commit for the wave |
| **3** | 1 executor: **E7 integration** | `teleop_collect run --mock` end to end with the scripted page client; the cross test against the libfranka-less `franka_control --mock --local`; every row of §7 that needs more than one process; seam fixes in any folder (each one listed in its report). The documents of §12.5: `teleop_collect/DIRECTIONS.md` (the laptop), `deployment/control_deploy/DIRECTIONS.md` (the PC), and `teleop_collect/DESIGN.md` (the whole design: how the parts fit, linking each folder's DESIGN.md). | all tests green, `run --mock` works; walk-through; local commit |
| **4** | 1 reviewer: **R** over everything | The whole diff of this branch against `main`, waves 0–3 together, code and documents (below). It reports findings ranked by severity; it does not fix them. | the report |
| **5** | the lead, or the owning executor per finding | The fixes. Then R checks only the fixed findings again. | R clean; walk-through; local commit |
| **6** | the lead: **the final step** | **Run every test**: `deployment/contracts/tests`, all of `teleop_collect/tests`, `teleop_collect check --mock`, `teleop_collect run --mock` with the scripted page client, and the cross test against `franka_control --mock --local`. Report each command with its result line. Anything failing goes back to wave 5. | everything green; walk-through; local commit |

E3 and E4 share `teleop/`, but in disjoint subfolders. E3 codes against E4's interfaces as written in STRUCTURE.md,
with small fakes in its own tests, so neither waits. The same holds for E5 against U1/U2 (E1 defines them, E2 and E3 serve
them).

### 12.2 Rules every agent is given

- **The brief:** PLAN.md, the agent's sections of STRUCTURE.md, its list of files, and the fd: sources it copies. It
  writes **only its own files**. An interface problem with another agent's part goes to the lead in its report; it never
  edits that part.
- **`~/Capstone-final-deployment` is read-only:** read and copy only, no edits, no builds, no `git` commands that change it.
  `deployment/contracts/` in this branch is never edited.
- **No `git commit`, `push`, `merge` or `stash`.** The lead commits locally after each wave. Nothing is pushed without asking.
- **One environment:** `~/envs/teleop_collect` only. Never install into `~/envs/policy_deploy*` or `~/envs/control_deploy`.
- **Tests:** each executor writes and runs only its own unit tests: `unset PYTHONPATH`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
  They bind free ports only (`tcp://127.0.0.1:*`), because five agents test at once. Nothing runs `teleop_collect run`
  or binds the fixed station ports before wave 3.
- **No hardware:** nothing connects to the real PC or Pi addresses.
- **Copied files** carry the header of rule 1, with every change listed in `changes:`, and otherwise stay as close to the
  source as possible.
- **The report** (the only thing the lead sees): the files written; every deviation from STRUCTURE.md and why; the test
  command and its result line; open issues.

### 12.3 What the reviewer checks

1. **PLAN.md and STRUCTURE.md against the code:** every class, contract, config key, failure row (§7) and the episode
   layout (§6).
2. **Copies:** each copied file diffed against its fd: source at the commit its header names (`2ffe641`, or `ea9f7fb`
   / `927a5ba` after the re-syncs), and every difference listed in its header.
   `deployment/contracts/` byte-identical to fd:contracts/ (only `link.yaml` addresses may differ, and only when the user
   set them).
3. **`rnoro/final-deployment` untouched:** `~/Capstone-final-deployment` still at `927a5ba` (was `2ffe641`, then
   `ea9f7fb`), never changed from this branch's sessions.
4. **The dependency rules:** no import of `policy_deploy` or `control_deploy`; process subpackages import only
   `teleop_collect.contracts` and `deploy_contracts`.
5. **Safety:** the §5.3 bound in code and tests (speeds, lead, state-age guard, stale keys = released, focus loss, the
   release set); no C3 request outside the main loop; rule H2 at start-up; Stop arm and Ctrl-C always send `stop`.
6. **Data:** bytes on disk equal bytes received; every time stored with its clock; `COMPLETE` last; discard and delete
   remove the right folder and only it.
7. **Running it:** the full test suite, `teleop_collect check --mock` and `run --mock` with the scripted client, and
   every command in both DIRECTIONS.md files that can run without hardware.
8. **The documents** (§12.5): both DIRECTIONS.md files can be followed command by command from a fresh machine, with no
   step missing or out of order. Every DESIGN.md matches its code.

### 12.4 Commits

A local commit by the lead after waves 0, 1, 2, 3, 5 and 6, named `teleop_collect: wave N: ...`. **Before every
commit the lead walks the user through the changes** (what was added or changed, file by file, and anything that
deviates from this plan) and commits only after the user agrees. No push without asking.

### 12.5 Documents

**Two DIRECTIONS.md files, one per machine:**

| File | Machine | Covers |
|---|---|---|
| `deployment/control_deploy/DIRECTIONS.md` | the PC | getting `rnoro/final-deployment` on the PC, the env, the `franka_control` build, the arm cable, `link.yaml`, the measured arm values (the workspace box first), the bring-up `console`, the start pose (`homing.q_start_rad`), the daily `control_deploy run`, and the keys (`h`, then `p` before every episode). Documents only: the PC's code stays `control_deploy` from `rnoro/final-deployment`, unchanged (rule 2). It shortens fd:control_deploy/DIRECTIONS.md to what teleop collection needs. The real addresses go into the PC checkout's `contracts/link.yaml` as a local edit, never committed to `rnoro/final-deployment`. |
| `deployment/teleop_collect/DIRECTIONS.md` | the laptop | the code, `~/envs/teleop_collect`, `link.yaml` and the Pi's address, `run --mock`, `check`, the session routine (PC, Pi, laptop), the page and its keys, episodes on disk, the tools (`pi_collect`, `pc_probe`), and stopping. |

Rules for both:

- **Command by command.** A person who never read the code follows them top to bottom on a fresh machine and ends
  with a working setup, without problems. The steps go in the order they must be done, with no step missing.
- **Few words.** Each step has the command to copy, plus one or two lines: what it does and why it matters. Each
  argument is explained only if the command has one. Then **Good:** what you see when it worked, and **If not:**
  what the likely outcomes mean and what to do.
- No design explanations (those go in DESIGN.md), no history.
- A short "daily routine" box at the top, and a "when something fails" table at the end (the message → what to do).
- Every command in them is run during the review (§12.3) where that is possible without hardware.

**One DESIGN.md per part**, so it is clear how each part was designed: `teleop_collect/DESIGN.md` (the whole: the
processes, how they talk, the episode flow, and links to the rest), then `contracts/`, `frames/`, `teleop/`,
`teleop/motion/`, `teleop/episode/`, `ui/`, `launcher/` and `fake_pc/`. Each says:
- what the part does and does not do;
- its classes and threads, and how data moves through them;
- the decisions behind it and why, pointing to the PLAN.md sections;
- what was copied from fd: and what changed;
- how it is tested;
- how to extend it (e.g. a new key, a new recorded file).

Each is written by the part's executor, from its code, and kept short. `deployment/contracts/` gets no DESIGN.md: it is a
byte-for-byte copy, and its design is fd:STRUCTURE.md.

## 13. Decisions (were open questions)

All settled with the user on 2026-11-18; nothing is open.

1. **Shared C2/C3 code:** copied byte for byte into `deployment/contracts/` (rule 5, §4.1). Done in wave 0.
2. **Rotation axes and key map:** the twist turns the hand about its own approach axis, whatever its tilt. The tilts
   lean it forward/back and left/right about the robot base's axes, pivoting at the fingertip point (§5.2, §5.4).
3. **Speeds:** never pushed. 5 cm/s, 10 °/s, slow ×0.25, with the PC's gate as the only safety authority (§5.3).
4. **Aborted episodes are deleted**, whatever ended them (the PC stopping the arm, Stop arm, a conclude, Ctrl-C, a
   crash). The reason goes to the page and the session log only (§7). Data problems never abort since 2026-11-20:
   they are warnings in the episode's `meta.yaml`; a failed recording can only be discarded.
5. **Numbering:** episode numbers just count up. Deleted or discarded numbers are never reused, so gaps are normal (§6.1).
6. **Images:** one MJPEG video per camera holding the received JPEGs unchanged (§6.1).
7. **No pre-roll:** the recording starts at the Start click, and the operator waits about a second before the first key
   (§4.3).
8. **A lost or faulted PC is a failure:** conclude, with the one-line reason; rerun `teleop_collect run` (§7).
9. **The TCP:** the `F_T_TCP` in `contracts/frames.yaml` as the PC runs it now (since `ea9f7fb`: training's TCP axes,
   +45 deg about flange z; since `927a5ba`: fingertips 235.65 mm out, the printed finger adapter's CAD). Training and the deployed policy both use
   the same robot TCP, so the teleop data is consistent; `F_T_TCP` is saved with every episode. Not part of this branch:
   to deploy a teleop-trained policy, `policy_deploy` refuses any trained policy while `frames.yaml` says
   `settled: false`. That flag is a `rnoro/final-deployment` matter for later.
10. **Fingers:** `N` closes (a grasp with force), `M` opens fully. There is no toggle (§5.2, §5.4).
11. **The start pose** (changed 2026-11-20 with fd:`927a5ba`; was a laptop home): no laptop home at all. The operator
    presses `h`, then `p` on the PC before `teleop_collect run` and **before every episode**; teleop takes the arm's
    pose at start-up as the session's start pose, and the PC's first accepted start confirms it. Start is enabled only
    there (§8), so every episode starts from the pose the deployment starts from.
12. **Metadata:** only what helps policy training: the task name and its one-line description. No operator names or
    free-text notes.
