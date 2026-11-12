# Deployment plan

Branch `rnoro/deployment`. This file holds the goals, rules, contracts and a plan for each process. Each folder's
detailed design goes in its `STRUCTURE.md`, written from its plan here (section 6) before that folder's code.

## 1. Goal

**Final goal: physical deployment with continuous policy control of the Franka Panda.** The arm keeps moving while
the policy computes its next chunk: no stopping and starting between chunks.

The setup:

- The **Raspberry Pi 5** on the gripper sends live 30 FPS streams from the IMX519 (main) and IMX219 160° (ultrawide)
  cameras **over Wi-Fi** (the Pi and the PC are on the same network). Its pipeline is unchanged.
- A **separate PC**, cabled directly to the Franka, runs the four processes below. Its specs are not known yet, so
  every part that depends on them has a plug-and-play backup (section 9).
- The policy is a **plain diffusion policy** first. Prompting (behavior prompting) can be added later without
  changing the rest (section 6.2).
- The gripper is the **Franka Hand with our printed fingers**.

Milestones:

1. Each process's `STRUCTURE.md` written and agreed (section 8, phase 1).
2. All code written, with unit tests, including contract tests (section 7).
3. **`deploy run` working end to end** with stand-ins: a fake Pi sender, the fake arm, and the "hold still" policy.
4. **`INSTRUCTIONS.md`** written and followed once from scratch on the PC, start to finish.
5. **Physical deployment:** the same `deploy run` on the real arm. First the "hold still" policy (checks frames,
   timing and safety on the real arm), then the trained policy in continuous control.

## 2. Rules

1. **Simple to run.** The final product is two commands:
   - `deploy check`: is everything ready (Pi, arm, config, model)?
   - `deploy run`: start everything and run episodes from the keyboard.

   An **episode** is one attempt at the task: `c` starts it from the home pose, and `y` (success), `n` (fail) or `s`
   (stop) ends it. A **run** is one `deploy run`, from start-up until `q` or Ctrl-C (or an unrecoverable failure)
   shuts everything down. One run holds many episodes: `h` homes the arm between them.

   Both read their settings from one file, `station.yaml`, so they need few or no arguments. When something fails,
   the message says which process failed and what to check.
2. **Easy to troubleshoot.** `deploy run` starts the four processes in the background and waits until each is ready.
   Each process can also be started alone with its own command, so a misbehaving one can be run and watched in its own
   terminal.
3. **Modular.**
   - Processes talk only through the messages in section 4. Any process can be swapped for a stand-in that speaks the
     same messages.
   - The only shared Python code is `contracts/`. No process folder imports another process folder.
4. **Code lives in its process folder.** Each folder holds its own code, tests, config defaults and its own launch
   description. The `deployment/` level only has the launcher, `contracts/`, `station.yaml` and `gripper.yaml`.
5. **Object-oriented and simple.**
   - Each class has one job.
   - Every part that might be swapped (camera source, policy, arm) sits behind a small interface with a real and a
     fake implementation.
   - Numbers that might change (rates, limits, ports, thresholds) live in config, not in code.
6. **Code first, with unit tests.** Write all the code first, with unit tests for each class and for every contract.
   Simulation tests are not part of this work for now.
7. **No code before its STRUCTURE.md.** Each folder's design is written and agreed first.
8. **The ideal pipeline first, then failure modes**, following the rule in section 5.
9. **Real-world only.** This work targets the real Pi, PC and arm in real time. The MuJoCo sim (`rnoro/main-sim`)
   stays separate and untouched for now; nothing here depends on it.
10. **Set up from scratch with `INSTRUCTIONS.md`.** `deployment/INSTRUCTIONS.md` takes a fresh PC (and the Pi side
    once it exists) to a working `deploy run`: create both Python environments (the light one, and the torch one for
    `policy/`), build `franka_control`, fill in `station.yaml`, run the tests, `deploy check`, `deploy run`. Minimal
    words, copy-paste commands, one short section per step, and what to do when a step fails. It is written and kept
    up to date with the code.

## 3. Layout and the four processes

```
deployment/
  PLAN.md
  INSTRUCTIONS.md     set everything up from scratch: short, copy-paste commands (rule 10)
  gripper.yaml        our hand-entered camera mount constants: ultrawide camera position and tilt (6.6)
  station.yaml        every setting deploy check / deploy run need: hosts, ports, paths, which policy, which rig file
  deploy/             the launcher: deploy check, deploy run (thin; knows nothing about frames, poses or the arm)
  contracts/          the message definitions, validators and example messages (section 4)
  process_frames/     STRUCTURE.md, code, tests, launch.py
  policy/             STRUCTURE.md, code, tests, launch.py
  franka_control/     STRUCTURE.md, C++ code, C++ tests, launch.py
  message_comms/      STRUCTURE.md, code, tests, launch.py
```

```
                                  message_comms (manager)
 Pi 5 ────C0────► process_frames ──C1──►  routing + conversions  ──C4──►  policy (model)
                                          (abs → rel in,          ◄─C4──
                                           rel → abs out)
                                              ▲            │
                                C2 state broadcast         │ C3 waypoints / commands / heartbeat
                                              │            ▼
                                franka_control (C++): safety gate → 1 kHz loop ──FCI──► Franka Panda
```

| Folder | Process | What it does |
|---|---|---|
| `process_frames/` | frames | Receives both cameras' frames from the Pi and converts Pi timestamps to PC time. Keeps the recent frames. When asked, returns 2 main and 2 ultrawide frames shrunk to 224×224 exactly as in training. Reports each camera's health. Owns the rig config (stand-in until v2). |
| `policy/` | model | Loads the trained model on the GPU. Given images and relative robot state, returns 16 relative steps. Knows nothing about the arm or absolute poses. |
| `franka_control/` | C++ on libfranka | The only process connected to the arm. Every incoming waypoint passes the **safety gate** first. The **1 kHz loop** moves smoothly between accepted waypoints using Cartesian impedance control. Broadcasts the arm's state, and handles the gripper and homing. |
| `message_comms/` | manager | The episode loop and its rhythm, plus the keyboard. Routes messages between the other three, and converts poses: absolute → relative on the way into the policy, relative → absolute on the way out. Decides "slow" vs "conclude" (section 5). |
| `deploy/` | launcher (not a process) | Reads `station.yaml`, checks each process, starts them in order, waits until each is ready, and stops them all in reverse. |

**One policy step (continuous control):**

1. `message_comms` asks `process_frames` for the newest frames (C1).
2. It looks up the hand pose at each image time in its state history (from C2), then converts the poses to relative,
   anchored on the pose at the newest main image.
3. It sends images and relative state to `policy`, and gets 16 relative steps back (C4).
4. It converts the steps to absolute, timed waypoints using the same anchor, and sends the first `exec_h` of them to
   `franka_control` (C3).
5. While the arm executes those, it starts the next step right away. The new waypoints replace the rest of the old
   ones.

Waypoint k is timed `t_anchor + k × 50 ms`, and by the time a chunk arrives its first waypoints are already in the
past (they are skipped). The arm never holds if `L + 2C ≤ (exec_h − 1) × 50 ms`, where L is the image latency
(capture → PC) and C is the time one step takes (frames + policy + conversion). With `exec_h` = 8 and L = 80 ms, C must
be ≤ 135 ms. If a step is slower, the arm reaches the last waypoint and holds there until new ones arrive. That's
"slow", not a failure. The status line shows the margin.

## 4. Contracts

A **contract** is the exact shape of a message: its fields, types, units, reference frame and timing. Both sides of a
connection are written and tested against it.

The contracts live as code in `contracts/`. For each message there is a validator, plus example messages that the
Python tests and the C++ tests both read. That way a change on one side breaks a test instead of the robot.

Conventions for all messages:

- **Time:** `t_ns` is integer nanoseconds on the PC's monotonic clock (`CLOCK_MONOTONIC`). Every process runs on the
  same PC, so these times can be compared directly. Only `process_frames` sees Pi time, and it converts it.
- **Units:** metres, radians, seconds (or ns). Images are uint8 RGB, (H, W, 3).
- **Absolute poses:** 4×4 float64 matrices `O_T_TCP`. This is the pose of the TCP (the gripper point the training
  poses describe) in the robot base frame.
- **Every message carries `schema`** (name and version). A process rejects a version it doesn't know.
- **Transport:** ZeroMQ, with addresses and ports set in `station.yaml`. C1–C4 run on local ports on the PC; C0
  runs over Wi-Fi from the Pi. **Broadcast** (publish/subscribe) is
  used for state; **request/reply** for everything else. The encoding is msgpack, with arrays as raw bytes plus shape and dtype.

### C0: Pi 5 → `process_frames` (frame stream over Wi-Fi, plus clock sync)

**Stand-in.** The Pi's sending side isn't written yet. This is what `process_frames` expects; whoever writes the Pi
sender implements it (TODO: confirm with the Pi side). Our fake Pi sender speaks exactly this, so `process_frames` can
be built and tested now.

The Pi sends every frame of both cameras as it's captured: 30 FPS each, one message per frame. `process_frames`
connects to the Pi's address from `station.yaml` and subscribes. The Pi doesn't wait for requests.

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `c0.frame/v1` |
| `camera` | string | `main` or `ultrawide` |
| `sensor` | string | `imx519` or `imx219`; checked against the rig file, so a swapped cable is caught |
| `seq` | uint64 | per camera; goes up by 1 per frame, so a gap means a dropped frame |
| `t_pi_ns` | int64 | libcamera `SensorTimestamp`: end of the first row's exposure, on the Pi's `CLOCK_BOOTTIME`. The same meaning as the `t_ns` in the Pi's recordings. |
| `exposure_ns` | int64 | exposure time in ns (libcamera `ExposureTime` × 1000); 0 if unknown, in which case `process_frames`' `default_exposure_ns` (config.yaml, 5 ms capped at half a frame period) is used |
| `width`, `height` | int | frame size; must match the rig file |
| `encoding` | string | `h264` (default) or `jpeg` (backup) |
| `keyframe` | bool | H.264 only: true for a frame that can be decoded on its own |
| `data` | bytes | one encoded frame |

H.264 requirements on the Pi side:
- The same encoder settings as recording (about 10 Mbit/s), so live images match training.
- Low-latency settings: no B-frames, so each frame can be decoded as soon as it arrives.
- A **keyframe at least once a second**, so the decoder can recover after a dropped frame.
- Annex-B framing (one access unit per message), with SPS/PPS sent before every keyframe.
- `seq` counts captured frames, so a frame the Pi drops before sending also shows as a gap.

**Clock sync** (request/reply, `process_frames` → Pi, about once a second): `time_sync` → the reply is the Pi's
current `CLOCK_BOOTTIME` in ns. `process_frames` measures the round trip and estimates the Pi → PC clock offset and
drift from many of these exchanges.

### C1: `process_frames` ↔ `message_comms` (request/reply)

`get_frames` (no arguments) → reply:

| Field | Type / shape | Meaning |
|---|---|---|
| `main` | uint8 (2, 224, 224, 3) | 2 main frames, oldest first, 50 ms apart (20 Hz), prepared exactly as in training |
| `main_t_ns` | int64 (2,) | PC mid-exposure time of each main frame |
| `ultrawide` | uint8 (2, 224, 224, 3) | 2 ultrawide frames, 100 ms apart (10 Hz), remapped and prepared as in training |
| `ultrawide_t_ns` | int64 (2,) | PC mid-exposure time of each |
| `health` | per camera: `status` (`ok` / `stale` / `down`), `last_frame_age_ms`, `fps`, `dropped` | used for failure handling (section 5) |

If a camera can't supply valid frames, the reply has `status != ok` for that camera and **no images**. It never
returns old images as if they were new.

`get_health` returns only the `health` part. It's used by `deploy check` and the status line.

The spacings (50 ms / 100 ms), the nearest-frame tolerance and the stale/down limits come from config. Mid-exposure
time follows training: `t = t_pi_ns − exposure_ns/2 + time_offset_ns`, then mapped to PC time.

### C2: `franka_control` → `message_comms` (broadcast, ~100 Hz)

| Field | Type / shape | Meaning |
|---|---|---|
| `t_ns` | int64 | PC time the state was read from the arm |
| `seq` | uint64 | goes up by 1 per message; a gap means a missed message |
| `O_T_TCP` | float64 (4, 4) | measured hand pose at the TCP, base frame |
| `O_T_TCP_target` | float64 (4, 4) | the target the 1 kHz loop is tracking right now |
| `q`, `dq` | float64 (7,) | joint angles (rad), joint speeds (rad/s) |
| `gripper_width_m` | float64 | opening between the fingertips (m), absolute |
| `mode` | enum | `idle` / `running` / `holding` / `stopped` / `homing` / `fault` |
| `error` | string | empty, or the libfranka or safety error text |
| `gate` | counters | waypoints accepted / shortened / pulled back to the box or cone / rejected since start |
| `loop_late` | uint64 | 1 kHz cycles that ran late since start (shown on the status line; mainly for the ignore-real-time backup) |

### C3: `message_comms` → `franka_control` (request/reply)

| Request | Contents | Reply |
|---|---|---|
| `set_waypoints` | `chunk_id`, plus three parallel arrays of length N: `t_ns` int64 (N,) increasing, `O_T_TCP` float64 (N, 4, 4), `gripper_width_m` float64 (N,) | `accepted` (count, shortened ones included), `shortened` (count), `skipped` (count of waypoints whose time had already passed), `rejected` with reasons. Replaces any waypoints not yet reached. |
| `heartbeat` | none | ok. Sent every loop cycle even when there are no new waypoints, so `franka_control` knows the manager is alive. |
| `start` | none | ok, or refused (e.g. in `fault`) |
| `stop` | none | ok: stop following waypoints, hold position |
| `home` | none | ok when the home pose is reached (blocking joint move). The watchdog pauses while homing; homing has its own timeout. |
| `get_state` | none | one C2 message, for start-up checks |

- After the last waypoint, `franka_control` **holds** that pose for as long as heartbeats keep arriving. That's the
  "slow" case.
- With **no heartbeat** for the watchdog limit (proposed 0.5 s), it treats the manager as gone: it stops and goes to
  `stopped`.
- The final limits go in `franka_control/STRUCTURE.md`.

### C4: `message_comms` ↔ `policy` (request/reply)

| Request | Contents | Reply |
|---|---|---|
| `config` | none | `obs_steps` (2), `action_steps` (16), `step_ms` (50), image size (224), which inputs it uses, `uses_prompt` (false for a plain diffusion policy) |
| `reset` | none | ok; clears the policy's history (must answer in < 0.5 s) |
| `infer` | the observation below | the action chunk below, or an error |
| `prompt` (optional) | `task_name`, `prompt_index` | ok. Only sent when `config` says `uses_prompt: true`, once after each `reset`. A policy without prompting never receives it. |

Observation. Everything is relative to the anchor (the hand pose at the newest main image time), oldest step first:

| Field | Type / shape | Meaning |
|---|---|---|
| `eef_pos` | float32 (2, 3) | position relative to the anchor (m); the last row is always 0 |
| `eef_rot6d` | float32 (2, 6) | orientation relative to the anchor as rot6d (the first two rows of the rotation matrix); the last row is always [1,0,0,0,1,0] |
| `eef_rot6d_wrt_start` | float32 (2, 6) | orientation relative to the episode start pose |
| `gripper_width_m` | float32 (2, 1) | absolute opening |
| `main`, `ultrawide` | uint8 (2, 224, 224, 3) | straight from C1 |

Action chunk:

| Field | Type / shape | Meaning |
|---|---|---|
| `eef_pos` | float32 (16, 3) | step k's position relative to the anchor (m), 50 ms apart |
| `eef_rot6d` | float32 (16, 6) | step k's orientation relative to the anchor |
| `gripper_width_m` | float32 (16, 1) | absolute opening, in [0, max opening] (the policy clips it into range) |
| `width_clipped` | uint64 | how many of this chunk's widths the policy clipped into range; shown on the status line |

A valid chunk (checked by the contract validator) has: the right shapes and types, finite values, and rot6d rows of
unit length and perpendicular to each other. A broken chunk ends the run.

Separately, `message_comms` checks that step 0 is within 2 cm / 5° of the anchor. A chunk that fails only this
check is **odd, not broken**: that step is skipped (the arm holds), the chunk is counted, and the count is shown on
the status line (a "slow" case, section 5).

These are the same quantities as main-sim's 5-topic contract (`controlSW/docs/adding_a_policy.md`), with cleaner names
and 224×224 images. `policy/` maps them to whatever key names its model was trained with.

## 5. Failure rule

Every problem is one of two kinds:

- **Slow → keep running.** No new waypoints are sent until things catch up. `franka_control` finishes its waypoints
  and holds the last one, its 1 kHz loop running normally, while heartbeats keep arriving. When things catch up, the
  policy steps resume by themselves.
- **Unrecoverable → conclude immediately.** No retries, no partial recovery:
  1. `message_comms` sends `stop` to `franka_control`.
  2. The episode is recorded as aborted, with the reason.
  3. `deploy run` shuts every process down in reverse order and exits with one clear message: which process, what
     failed, what to check.
  4. To try again, run `deploy run` again.

**`franka_control` is never upset by anything else.** Its 1 kHz loop runs inside its own process and never waits on
messages, so slowness elsewhere can only make it hold.

| Problem | Kind | Reaction |
|---|---|---|
| A camera `stale` (no new frame for > stale limit, proposed 0.35 s) | slow | skip policy steps; resume once both cameras have fresh frames |
| A single dropped frame, or no frame within the picking tolerance | slow | skip that step |
| Policy reply slower than the waypoints it has | slow | arm holds at the last waypoint, then continues |
| A missed state message (C2 `seq` gap), or the newest state older than its limit | slow | skip that step |
| A camera `down` (no frame for > down limit, proposed 2 s) or the Pi connection lost | conclude | stop, abort, shut down: "main camera down: check the Pi stream and cable" |
| A Pi stream restart (that camera's `seq` or `t_pi_ns` goes backwards) | conclude | that camera is latched `down`; same as above |
| Clock sync with the Pi lost for > 10 s | conclude | frames stop being used, so the cameras go stale, then `down`; same as above |
| Policy hung (no reply for > hang limit, proposed 5 s), returns an error, or a broken chunk (fails the contract validator) | conclude | stop, abort, shut down |
| An odd chunk: step 0 more than 2 cm / 5° from the anchor | slow | skip that step, count it, show it on the status line |
| `franka_control` reports `fault`: a libfranka reflex, a lost connection to the arm, or a waypoint the gate rejects (outside the box or cone by more than the margin, width outside [0, max opening], non-finite) | conclude | the arm is stopped by `franka_control` itself; shut down. A fault is never cleared automatically. |
| Any process exits or stops answering | conclude | shut down. If `message_comms` itself dies, the heartbeat stops and `franka_control` stops the arm. |
| A message that fails its contract (wrong schema version, shape or type) | conclude | shut down, naming the message and field |
| Operator presses `s` (stop) | episode ends (normal) | `stop` to `franka_control` (the arm holds), the episode is recorded as `stopped`, and everything keeps running. `h` homes the arm, `c` starts the next episode. |
| Operator presses `q` (quit) or Ctrl-C | conclude (normal) | `stop`, then shut down in reverse order; the run ends |
| At start-up: anything `deploy check` finds wrong | conclude | `deploy run` refuses to start, and says what is wrong |

These are normal operation, not failures. Each is counted and shown on the status line:
- **Speed:** a waypoint that moves too fast is shortened to the speed limit (`franka_control`).
- **Box and cone:** a waypoint outside the workspace box or tilt cone by at most a small margin (proposed 1 cm /
  2°) is pulled back to the boundary (`franka_control`). Beyond the margin it's rejected (see the table).
- **Width:** a policy width outside [0, max opening] is clipped into range and counted (`policy`). `franka_control`
  checks the width again.

## 6. Process plans

Each plan below is the brief for that folder's planning subagent (section 8). Its `STRUCTURE.md` turns the plan into
files and classes. Every folder has the same shape:

```
<folder>/
  STRUCTURE.md
  launch.py       how deploy/ starts this process, checks it, and knows it is ready (section 6.5)
  config.yaml     this process's defaults (station.yaml can override them)
  <package>/      the code: one class per file where possible
  tests/          unit tests, including this process's contract tests
```

### 6.1 `process_frames/`

**Inputs:**
- Two 30 FPS streams from the Pi over Wi-Fi, received directly from the Pi as C0. They do not pass through
  `message_comms`.
- Compressed, because raw frames don't fit: two raw streams are roughly 130 MB/s plus 110 MB/s, far more than Wi-Fi
  carries.
- **H.264 is the default**, decoded with PyAV like the training videos, so live images match training as closely as
  possible. JPEG per frame is the backup (section 9).
- Wi-Fi adds delay and occasional drops. Those are "slow" cases (section 5).
- **Rig file:** a stand-in copy of `rig_pi_v1.yaml` with both cameras set to 30 FPS. Its header says
  `TODO: replace with the v2 rig file`, and `deploy check` warns while it's in use. The resolutions, lens models and
  TCP in it are nominal until v2 arrives.

**Two sides, running at the same time:**

```
 Pi 5 ──C0 (every frame, 30 FPS each)──► receiving side: decode → Pi time to PC time → ring buffers → health
                                                │
 message_comms ──C1 get_frames─────────► serving side: pick 2+2 frames from the buffers → image prep → reply
```

**Receiving side** (always on, paced by the Pi):
1. Takes in every C0 message as it arrives, whether or not anyone has asked for frames.
2. Checks it against the rig file (sensor, size) and checks `seq` for drops.
3. Decodes it to RGB. After a dropped H.264 frame, the following frames are marked unusable until the next keyframe,
   so a corrupted image is never served.
4. Converts its time to PC mid-exposure time, using the clock sync.
5. Puts it in that camera's ring buffer, and updates that camera's health.

**Serving side** (only when `message_comms` sends `get_frames`):
1. Picks 2 main and 2 ultrawide frames from the buffers.
2. Image prep turns them into 224×224.
3. Replies with images, times and health (C1).

**Classes:**

| Class | Job | Swappable implementations |
|---|---|---|
| `FrameSource` (interface) | yields C0 frame messages | `PiStreamSource` (subscribes to the Pi, or to the fake Pi sender, over the network), `ReplaySource` (reads a recorded Pi episode straight from files, for unit tests) |
| `FrameDecoder` (interface) | encoded frame → RGB array; reports frames that can't be used (after a drop, before the next keyframe) | `H264Decoder` (PyAV, default), `JpegDecoder` (backup), `RawDecoder` (tests) |
| `ClockSync` | runs the `time_sync` exchanges with the Pi | |
| `ClockMap` | Pi time → PC time, from the clock sync's offset and drift | |
| `FrameBuffer` | the last ~1 s of frames for one camera | |
| `FrameSelector` | picks the frames nearest the wanted times, within tolerance | |
| `ImagePrep` | letterbox / remap → 224×224 | |
| `CameraHealth` | `ok` / `stale` / `down` per camera, from frame ages and rates | |
| `RigConfig` | loads and validates the rig file | |
| `Receiver` | the receiving side: source → decoder → clock map → buffers → health, in its own thread | |
| `FramesServer` | the serving side: answers C1 | |
| `FakePiSender` | a stand-in Pi: replays a recorded Pi episode as C0 messages at 30 FPS, and answers `time_sync`. Can drop frames, pause or stop on command, for tests. | |

**Code to start from:**
- `rnoro/data-work`, `data-processing/dataproc/`:
  - `imageprep.py`, `geometry/camera_model.py`, `config.py` and the `NAME_RE` part of `paths.py`: image prep and the
    rig loader, copied unchanged so the images match training;
  - `_vendor/cv_util.py` as the reference in tests;
  - the tests `test_imageprep.py` and `test_vendored_camera_model.py`;
  - `raw_episode.py` (`read_video_frames`): PyAV decoding, if the Pi sends H.264;
  - `pi_format.py`: the Pi's file layout, for `ReplaySource` and `FakePiSender`;
  - `configs/rigs/rig_pi_v1.yaml`: the stand-in rig file.
- `rnoro/main-sim`, `controlSW/franka_deploy/`:
  - `tasks/frame_select.py`: the frame-picking rule;
  - `peripherals/clock_map.py`: Pi clock → PC clock.

**Unit tests:**
- 224 images identical to `cv_util` on several input sizes.
- The remap puts a known point at the expected pixel.
- Mid-exposure times match the formula.
- Frame picking at 30 and 60 FPS, with jitter and dropped frames.
- Clock sync and clock mapping with offset, drift and uneven round trips (like Wi-Fi).
- C0 messages from `FakePiSender` pass the contract validator. Wrong sensor, size, schema or encoding is refused.
- A dropped H.264 frame: nothing is served until the next keyframe, then serving resumes.
- Health transitions ok → stale → ok and ok → down, driven by a `FakePiSender` that pauses or stops.
- C1 replies pass the contract validator.
- The rig loader rejects broken rig files.

**Made easy to change:** the camera settings come only from the rig file, so the v2 file replaces the stand-in with
no code change. A new Pi format is a new `FrameDecoder`. A change to C0 is confined to `FrameSource` and the contract.
Rates and limits live in `config.yaml`.

### 6.2 `policy/`

**Normal case:** at start-up, load the checkpoint onto the GPU and run one warm-up inference. Answer `config`,
`reset` and `infer` (C4). **The first target is a plain diffusion policy** (no prompting).

**Classes:**

| Class | Job | Swappable implementations |
|---|---|---|
| `Policy` (interface) | `config()`, `reset()`, `infer(obs) → chunk` | `DiffusionPolicy` (a trained plain diffusion checkpoint, the first target), `HoldStillPolicy` (always returns "stay at the anchor") |
| `PromptablePolicy` (interface, extends `Policy`) | adds `prompt(task_name, prompt_index)`; `config()` reports `uses_prompt: true` | `BehaviorPromptingPolicy`, added later if needed |
| `KeyMap` | contract field names ↔ the model's own input/output names | |
| `PolicyServer` | answers C4; validates requests and replies against the contract. Answers `prompt` only if its policy is a `PromptablePolicy`. | |

Adding prompting later is one new class plus one line in `station.yaml`. `message_comms` already sends `prompt` when
`config` says so, and nothing else changes.

No policy has been trained on rig data yet: `run_001` was trained on iPhone images, and isn't expected to work on the
rig. Until a rig-trained diffusion policy exists, `HoldStillPolicy` is the policy used on the real arm.

**Code to start from:**
- `behavior_prompting/behavior_prompting/deployment/detached_inference.py`: loads a checkpoint, runs inference.
- `rnoro/main-sim` `controlSW/franka_deploy/franka_deploy/`:
  - `inference/server.py` and `inference/light_workspace.py`: a loading wrapper around it;
  - `mocks/fake_policy_server.py` (with `--no-prompt-rules`): the stand-in;
  - `tools/policy_contract.py`: the protocol checker.
- `main-sim`'s image resize inside `detached_inference._process_obs` is **not** used: images arrive already 224×224.

**Unit tests:**
- `HoldStillPolicy` replies pass the contract validator.
- `KeyMap` round trip.
- `reset` answers in < 0.5 s.
- An invalid request gives a clear error, not a crash.
- `DiffusionPolicy` with a tiny dummy network: shapes and dtypes. The real checkpoint isn't needed for unit tests.
- `PolicyServer` refuses `prompt` for a non-promptable policy, and a fake `PromptablePolicy` gets exactly one
  prompt after each reset.

**Made easy to change:** a new model is a new `Policy` subclass plus a `KeyMap`. Which policy runs is one line in
`station.yaml`. This process has its own Python environment (torch + behavior_prompting), separate from the others.

### 6.3 `franka_control/` (C++)

**Normal case:**
1. Connect to the arm and hold the current pose.
2. On `start`, follow waypoints. Each incoming waypoint passes the safety gate.
3. The 1 kHz loop interpolates the accepted waypoints for "now" (linear position, slerp rotation) and computes the
   Cartesian impedance torque.
4. Broadcast the state at ~100 Hz.
5. After the last waypoint, hold it while heartbeats arrive.

**Classes:**

| Class | Job | Swappable implementations |
|---|---|---|
| `Arm` (interface) | read state, send torques at 1 kHz, homing move | `LibfrankaArm` (the real Panda), `FakeArm` (a simple simulated arm, so everything can be tested without the robot) |
| `Gripper` (interface) | move to a width, read the width | `FrankaHandGripper` (the Franka Hand with our printed fingers, through the libfranka gripper API, in its own thread), `FakeGripper` |
| `SafetyGate` | checks each waypoint: workspace box, tilt cone (pulled back within a small margin), speed limits (shortened), width; accept / shorten / pull back / reject | |
| `WaypointTrajectory` | holds the accepted waypoints; gives the target for any time | |
| `ImpedanceController` | target + state → joint torques | |
| `Watchdog` | heartbeat timing → `stopped` | |
| `StateMachine` | `idle` / `running` / `holding` / `stopped` / `homing` / `fault`, and the allowed moves between them | |
| `SharedState` | the hand-off between the 1 kHz loop and the messaging thread, without the loop ever waiting | |
| `CommandServer` | answers C3 | |
| `StatePublisher` | broadcasts C2 | |

**Code to start from:**
- libfranka's own `examples/cartesian_impedance_control.cpp`: the torque law.
- `rnoro/main-sim` `controlSW/franka_deploy/franka_deploy/bridge/` (Python, to be rewritten in C++ with the same
  behaviour):
  - `safety.py`: box, cone, leaky-bucket speed limit;
  - `franka_bridge.py`: watchdog and state machine;
  - `backends.py`: `FakeBackend` and `derive_panda_error`;
  - `tests/test_bridge_safety.py`: the safety cases to port one for one.
- `scripts/build_libfranka_pandapy.sh`: the libfranka build steps only.

**libfranka version:** we assume the PC uses a libfranka version that supports our Panda's system version. The
version is a single build setting (`config.yaml` / the build script), so changing it doesn't touch the code.

**Unit tests (C++):**
- Every safety gate case.
- Interpolation, including time already passed and replaced waypoints.
- Holding after the last waypoint.
- Watchdog timing.
- Every allowed and refused state change.
- The impedance torque for known errors.
- C2/C3 messages read from and written to the shared example files.
- A short `FakeArm` run.
- Built with address and undefined-behaviour sanitizers in tests.

**Made easy to change:** gains, limits, home pose and `F_T_TCP` live in `config.yaml`. A different arm or controller is
a new `Arm` or controller class. The messaging thread and the 1 kHz loop only meet in `SharedState`.

### 6.4 `message_comms/`

**Normal case:**
1. Connect to the three processes and subscribe to the state. Ask `policy` for its `config`.
2. The operator presses start. `message_comms` resets the policy, and sends `prompt` only if `config` said
   `uses_prompt`.
3. The episode loop runs: frames, state lookup, input conversion, policy, output conversion, waypoints. The heartbeat
   is sent every cycle.
4. Stop or success/fail ends the episode.

**Classes:**

| Class | Job | Swappable implementations |
|---|---|---|
| `FramesClient`, `PolicyClient`, `FrankaClient` | one client per process (C1, C4, C3), each validating replies against the contract | fakes for tests |
| `StateListener` + `StateHistory` | receive C2; keep ~1 s; give the pose at any time by interpolation | |
| `PoseConverter` | absolute → relative (inputs) and relative → absolute + times (waypoints), using one anchor per step | |
| `EpisodeLoop` | the rhythm: when to ask, how many waypoints to send, heartbeats | |
| `FailureMonitor` | classifies each problem as slow or conclude (section 5) | |
| `Operator` | keyboard → `c` start an episode, `y` success / `n` fail / `s` stop (each ends the episode; the run continues), `h` home, `q` quit (ends the run); the status line | `ScriptedOperator` for tests |
| `SessionLog` | per-episode records and the reason it ended | |

**Code to start from:**
- `behavior_prompting/deps/real-env/real_env/agents/policy_agent.py` (`predict_actions`, the obs building) and
  `robot_utils.pose_utils` (`get_relative_poses`, `get_absolute_pose`): the relative/absolute math, rewritten into
  `PoseConverter`.
- `rnoro/main-sim` `controlSW/franka_deploy/franka_deploy/`:
  - `tasks/chunk_timing.py`: chunk timing;
  - `tasks/franka_task.py` (`run_policy_control`): the loop and scheduling, as a reference only;
  - `tasks/stop_watcher.py`;
  - `eval/`: episode records.

**Unit tests:**
- `PoseConverter`:
  - absolute → relative → absolute gives back the input;
  - known poses give known relative values;
  - rotation, not just translation;
  - the same anchor is used both ways;
  - `wrt_start`.
- `StateHistory`: interpolation, old states, `seq` gaps.
- `EpisodeLoop` with fake clients:
  - the next chunk is asked while the current one runs;
  - waypoints replace correctly;
  - heartbeats are sent during slow steps.
- `FailureMonitor`: every row of the section 5 table gives the stated reaction.
- Every client rejects a reply that breaks its contract.

**Made easy to change:** the loop only sees the clients' interfaces. A new failure rule is one entry in
`FailureMonitor`. Rates, `exec_h` and limits live in `config.yaml`.

### 6.5 `deploy/` (the launcher) and each `launch.py`

The launcher stays small and generic. Each process folder describes itself; the launcher only runs the descriptions.

**Classes:**

| Class | Where | Job |
|---|---|---|
| `Station` | `deploy/` | loads and checks `station.yaml` |
| `ManagedProcess` (base class) | `deploy/` | start a command, wait until ready, check health, stop; capture its log to the session folder |
| `ProcessFramesProcess`, `PolicyProcess`, `FrankaControlProcess`, `MessageCommsProcess` | each folder's `launch.py` | subclasses: which command, which settings, what "ready" means, what `deploy check` tests for it |
| `Launcher` | `deploy/` | `check`: runs every process's checks. `run`: start in order (`franka_control`, `process_frames`, `policy`, `message_comms`), wait for each to be ready, then wait. If any process exits, stop all in reverse. |

**Commands:** `deploy check` and `deploy run`. Both are optional: `--station <file>`, which defaults to
`deployment/station.yaml`, and `--mock`, which uses every process's stand-in.

**Each process alone:** `python -m process_frames`, `python -m policy`, `python -m message_comms` and the
`franka_control` binary each start that process with the same `station.yaml`, for troubleshooting.

**Code to start from:** `rnoro/main-sim` `controlSW/franka_deploy/franka_deploy/launch/launcher.py` (start order,
ready files, shutdown) as a reference only. It's much larger than this needs to be.

**Unit tests:**
- Start order.
- Waiting for ready, and timing out with a clear message.
- One process exiting stops all, in reverse order.
- `deploy check` reports every failing check by process.
- `--mock` picks every stand-in.

### 6.6 Gripper geometry (the TCP)

**The problem.** The policy learned to move one specific point on the gripper: the **TCP** (tool centre point), the
spot between the fingertips. Training and the robot must describe exactly that point, with the same position **and**
the same axes. Otherwise the policy's "tilt 20°" turns into a swing around the wrong point, and its "forward" points
somewhere else.

- **During training,** SLAM tracks the ultrawide camera. `dataproc` turns each camera pose into a TCP pose with the
  rig file's `T_cam_tcp`.
- **On the robot,** the Franka knows its flange (the metal plate at the end of the arm). `franka_control` turns it
  into the TCP with `F_T_TCP`.

**The three constants.** Each is a fixed 4×4 offset: the gripper is rigid, so they only change if the hardware
changes.

| Constant | From → to | Where it lives | How it's set |
|---|---|---|---|
| `F_T_TCP` | flange → TCP | `franka_control/config.yaml` | Fixed by the hardware: the Franka Hand's TCP axes (−45° about the flange z axis), with our printed fingertips **150 mm** out from the plate (stock is 103.4 mm, so `finger_extra_m` = 0.0466, typed in by us, here only) |
| `F_T_cam` | flange → ultrawide camera (the camera's constant pose on the gripper) | `deployment/gripper.yaml` | **Typed in by us** from our own measurements, in the hand frame: the lens is 55 mm above the centre line, 10 mm out from the plate (the back end of the 100 mm mount), on the hand's +x side, tilted 15° toward the fingers. The 55 mm, 10 mm and +x side are **assumed true** for now (not yet measured on the hardware) |
| `T_cam_tcp` | ultrawide camera → TCP | the rig file (`process_frames/rigs/…`), read by `dataproc` and `process_frames` | **Calculated, never typed in:** `T_cam_tcp = inverse(F_T_cam) × F_T_TCP` |

**Frames used:**
- **Flange (`F`):** Franka's convention. The origin is at the centre of the plate, and z points out of the plate
  toward the fingers.
- **Hand (`H`):** the Franka Hand's own axes, which are the flange rotated −45° about z. y is the direction the
  fingers close, and x is "up/down" on the gripper. `gripper.yaml` is written in this frame, because that's how the
  mount is measured. `F_T_cam = F_T_H × H_T_cam`.
- **Camera (`cam`):** OpenCV optical, as in `dataproc`. z points out of the lens, x right in the image, y down in the
  image. It is the **ultrawide (IMX219)** camera, because it's the one SLAM tracks. The IMX519's mount doesn't enter
  this calculation.
- **TCP:** the Franka Hand TCP frame. Calculating `T_cam_tcp` from `F_T_TCP` gives the training TCP these same axes
  automatically.
- **Rig file rotations:** scipy `from_euler("xyz", …)`, as `dataproc/config.py` reads them. The tool writes the
  rotation as a matrix, to avoid any ambiguity.

**`deployment/gripper.yaml`:** a short hand-written file holding only our own camera mount constants (the finger
length lives only in `F_T_TCP`, so no number is written twice):
- `ultrawide.position_mm` in the hand frame;
- `ultrawide.up_side`: the hand-frame side the camera sits on, "up" in the image (one of ±x, ±y);
- `ultrawide.tilt_deg`: how far it looks toward the fingers.

With our numbers, `T_cam_tcp`'s rotation is the 15° tilt combined with a 90° swap of axes (the TCP uses the Franka
Hand's axes, the camera OpenCV's). Its position puts the fingertips 149.5 mm in front of the lens and 16.9 mm below
it (150.4 mm away, 6.4° below the image centre). The stand-in rig file carries this value.

There is no CAD import.

**The geometry tool** (`python -m deploy.geometry`, part of `deploy/`, run by hand during setup and after any
hardware change):
1. Reads `gripper.yaml` and `franka_control`'s `F_T_TCP`.
2. Computes `F_T_cam` and `T_cam_tcp`.
3. Writes `T_cam_tcp` into the rig file, with a header saying it was computed and from which `gripper.yaml`.
4. Prints the distance from the camera to the fingertips, as a sanity check.

Its unit tests are the worked check made during planning (`deploy/tests/tcp_check_reference.py`, verified):
- the triangle closes: `F_T_cam × T_cam_tcp = F_T_TCP`;
- the rig-file form round-trips;
- the fingertips are in front of and below the camera;
- for 1,000 random arm poses, the training side and the robot side give the identical point and axes;
- a policy motion moves the fingertips identically on both sides.

**`deploy check`** recomputes the triangle from `gripper.yaml`, `franka_control`'s `F_T_TCP` and the rig file's
`T_cam_tcp`. It **fails** if they disagree by more than 2 mm / 0.5°, for example after someone edits one file but
not the others.

**The stand-in rig file's `T_cam_tcp` must be replaced.** The value in data-work's `rig_pi_v1.yaml` (15°,
(0, 6 cm, 20 cm)) describes the **iPhone gripper**: fingertips 20 cm in front of the phone's camera and 6 cm below it.
iPhUMI's deployment worked because the robot carried the same phone mount and the same long printed fingers, so the
triangle held there. It doesn't describe our rig: with stock-length Franka fingertips (`finger_extra_m` = 0) it would
put the camera about 7.4 cm behind the plate. It would only fit if our fingers stuck out about 7.4 cm or more beyond
the stock ones. Our camera, mount, tilt and fingers are our own, so `T_cam_tcp` comes from our own `gripper.yaml`
and `F_T_TCP`. The stand-in rig file (`process_frames/rigs/rig_pi_v1_standin.yaml`) already carries a computed
value from the placeholder `gripper.yaml`. The geometry tool overwrites it once `gripper.yaml` has our numbers.
**Training data must be processed with the same `T_cam_tcp`,**
so the same rig file (or at least the same `T_cam_tcp`) goes to `dataproc`. The handheld rig and the robot rig must
have identical camera-to-finger geometry.

**Hand-eye calibration (optional, phase 4).** It's a check that our typed-in `F_T_cam` matches the real, printed
mount:
1. The arm holds the camera in about 15 poses, looking at a fixed checkerboard.
2. OpenCV's `calibrateHandEye` solves for the real `F_T_cam`.
3. If it differs from `gripper.yaml` by more than about 5 mm / 2°, update `gripper.yaml`, rerun the geometry tool,
   and reprocess the training data.

It's a small separate tool, not part of the four processes, and is only built if the first real-arm runs suggest the
numbers are off.

## 7. Tests

All unit tests run on a laptop, without the Pi, the arm or a GPU. No simulation tests for now. Each folder's tests
are listed in its plan (section 6). Across folders:

- **Contract tests** (`contracts/`):
  - every example message passes its validator;
  - every deliberately broken example fails it (wrong shape, dtype, unit, missing field, wrong schema version);
  - each process's tests check its real output with the validator;
  - the C++ tests read the same example files for C2/C3, so Python and C++ agree.
- **Message flow:** `message_comms` against fake versions of the other three. This covers the full normal step and
  every row of section 5. The shutdown order is tested in `deploy/`'s launcher tests.
- **End to end (milestone 3):** `deploy run --mock` with scripted keys runs episodes and shuts down cleanly. Owned
  by `deploy/`: `deploy/tests/test_end_to_end_mock.py`, skipped until all four `launch.py` files exist.

## 8. How the work is done (subagents)

**Phase 1: plans.**
- One planning subagent per process folder, plus one for `deploy/` + `contracts/`. Each turns its section 6 plan into
  its `STRUCTURE.md`: files, classes and their methods, config keys, tests, and how it connects to the others through
  section 4.
- They are launched one after another, not all at once.
- **One reviewer** then checks all of them together: do the plans agree with each other and with sections 4 and 5?
- Every open question is settled with you before any code is written.

**Phase 2: build.**
- `contracts/` first, since everything depends on it.
- Then one executor subagent per process folder, each building its folder with its unit tests.
- **One reviewer** looks over all the executors' work and how it fits together.
- Gate: all tests pass. Then a local commit; nothing is pushed without asking.

**Phase 3: `deploy run`.**
- Wire everything through `deploy/` and `message_comms/`.
- Reach milestone 3, with the same reviewer, gate and commit.
- Write `INSTRUCTIONS.md`, then follow it in a fresh environment to prove every command works (milestone 4).

**Phase 4: physical deployment.** On the real PC and arm:
1. `franka_control` alone (the 1 kHz loop's timing).
2. Homing.
3. The safety box.
4. The TCP: `gripper.yaml` filled in, the geometry tool run, `deploy check` passing (6.6). Optionally, the
   hand-eye calibration check.
5. The "hold still" policy through the whole pipeline.
6. The trained policy in continuous control.

## 9. Backups (plug and play)

The PC's specs are not known, so we assume everything works, and write a backup for each part that depends on the PC
or the network.

- Each backup is another implementation behind the same interface.
- It's picked by one setting in `station.yaml`. No code changes.
- Each one is unit tested like the default.
- `deploy check` reports which ones are active.

| Part | Default | Backup | Use the backup when |
|---|---|---|---|
| Frame encoding (`process_frames`) | H.264 stream, PyAV decode | JPEG per frame (`JpegDecoder`) | H.264 decoding is too slow or unreliable on the PC, or the Pi side can't stream H.264 |
| Policy device (`policy`) | GPU (CUDA) | CPU | no usable GPU. The policy will be slow, so the arm holds between chunks (a "slow" case); fewer diffusion steps (below) helps |
| Diffusion steps (`policy`) | the checkpoint's own number | fewer steps (a config value) | inference too slow for continuous control |
| Waypoints per step `exec_h` (`message_comms`) | 8 (0.4 s) | more waypoints per step | the policy is slower than 0.4 s per step |
| Real-time scheduling (`franka_control`) | enforce: the PC has a PREEMPT_RT kernel, and the loop runs at real-time priority | libfranka's "ignore real-time" mode, plus a loop-timing warning on the status line | no RT kernel installed yet. Fine for first tests; expect occasional reflexes under load |
| Arm (`franka_control`) | `LibfrankaArm` | `FakeArm` | testing without the robot (`deploy run --mock`) |
| Camera source (`process_frames`) | the real Pi | `FakePiSender` (a recorded Pi episode, sent as C0) | testing without the Pi (`deploy run --mock`) |

## 10. Open items

None of them block the plans or the code.

- **The Pi's sending side:** who writes it, and confirming it matches C0. Until then, `FakePiSender` stands in.
- **`exec_h`:** the final number of waypoints per step. It starts at 8 and is tuned once the policy is timed on the
  real PC.
