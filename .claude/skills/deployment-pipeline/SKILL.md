---
name: deployment-pipeline
description: Map of the real-robot deployment pipeline on branch rnoro/final-deployment (deployment/ folder) - the Raspberry Pi 5 camera stream, the laptop's policy_deploy (frames, policy, manager, launcher, fake PC) and the PC's control_deploy (franka_control C++ 1 kHz loop, safety gate, console). Use whenever working on, debugging, extending, running or explaining deployment, policy_deploy, control_deploy, franka_control, deploy_contracts, the C0-C4 contracts, the clock maps, the safety gate, link.yaml / station.yaml, or teleop / data collection that has to plug into this pipeline.
---

# The deployment pipeline (Pi → laptop → PC → Franka Panda)

Source of truth: branch `rnoro/final-deployment`, worktree `~/Capstone-final-deployment`, folder `deployment/`
(state read on 2026-11-18 at commit `2ffe641`). Paths below are relative to `deployment/`. If this pipeline has been
merged into the current branch, the same relative paths apply there.

Docs, in order of authority: `PLAN.md` (goal, rules, contracts, failure rule) > `STRUCTURE.md` (communication, shared
contracts, codebase map) > `policy_deploy/STRUCTURE.md`, `control_deploy/STRUCTURE.md` (per-folder design). Operator
docs: `DIRECTIONS.md` (start here), `policy_deploy/DIRECTIONS.md`, `control_deploy/DIRECTIONS.md`,
`policy_deploy/TEST_FRAMES.md`, `policy_deploy/PI_PAYLOAD.md` (C0). Read the relevant STRUCTURE section before
changing code. The repo rule is no code before its STRUCTURE.md.

**On branch `rnoro/teleop-collect-deploy`, the laptop side is replaced by `teleop_collect`:** keyboard teleop and
episode recording against this unchanged PC side, built and tested. Its docs are in `deployment/teleop_collect/`:
`DESIGN.md` (the whole, linking each part's DESIGN.md), `DIRECTIONS.md` (the laptop), `PLAN.md`, `STRUCTURE.md`; the
PC's are in `deployment/control_deploy/DIRECTIONS.md`.
- On that branch `deployment/` holds only `teleop_collect/`, the PC directions and a byte copy of `contracts/`. The
  policy_deploy and control_deploy code paths below are in `~/Capstone-final-deployment/deployment/`.
- That branch is never modified or merged; code is copied from it with derive headers.
- teleop's failure rule differs from policy_deploy's in one way: a PC `stopped` or a camera down during an episode
  aborts (deletes) only that episode, and the app stays up.

Contract field tables, the C3 replies per mode, the gate rules, the failure table and the config keys are in
[reference.md](reference.md).

## 1. Topology

```
 Pi 5 (on the hand)          LAPTOP (Ubuntu 22.04, RTX 3050 Ti 4 GB)           PC (Ubuntu 20.04, cabled to arm)
 IMX519 main + IMX219 160°   policy_deploy/                                    control_deploy/
 PUB :5555 ──C0 pi_stream/v1──► frames ──C1──► manager ◄──C4──► policy
   (Wi-Fi ASUS_5g, 30 pkt/s)  REP 127.0.0.1:5610    │ ▲         REP :5640       franka_control (C++)
                                                    │ └──C2 c2.state/v1 100 Hz── PUB <pc.address>:5620
                                                    └────C3 c3.* REQ/REP───────► REP <pc.address>:5630
                                                                                   │ FCI, Ethernet 172.16.0.2
                                                                                 Franka Panda + Franka Hand
```

- All three machines are on the lab router **ASUS_5g**. The PC also has a direct cable to the Franka controller
  (172.16.0.x). Going wired to the router changes only addresses in `contracts/link.yaml`.
- **The Pi sender is not in this repo.** It lives on the Pi (`admin@RaspberryPi5`, Wi-Fi MAC `88:a2:9e:a1:70:d4`) and
  is fixed and verified. Pressing the rig's **C button** starts it (LED blue); pressing again stops it. The laptop adapts to
  the Pi: C0 is defined in `policy_deploy/PI_PAYLOAD.md`. Receive-only tools: `policy_deploy.frames.pi_collect`
  and `TEST_FRAMES.md`.
- Servers and publishers bind, clients connect. The PC binds both C2 and C3, and the laptop only connects. **C3 has
  exactly one client at a time**: the manager in a run, otherwise `pc_probe` (laptop) or `console` (PC), never together.
  C2 can have any number of subscribers.

## 2. Hard rules of the codebase (PLAN §2)

1. One command per machine: `policy_deploy run` (laptop) and `control_deploy run` (PC), each with `check` and `--mock`.
2. Neither machine starts, stops, ssh's into or reads files of the other. They meet only through C2/C3 messages.
3. `policy_deploy/` and `control_deploy/` never import, include or read each other. The only shared code is
   `contracts/` (Python package `deploy_contracts`, plus `limits.yaml`, `frames.yaml`, `link.yaml` and `examples/`,
   which the C++ also reads). It holds **only what crosses between the machines**: C0/C1/C4 are laptop-only and
   live in `policy_deploy/policy_deploy/contracts/`. Import tests enforce this.
4. A station or config file can never override a key of the three contract yaml files.
5. Swappable parts sit behind a small interface with a real and a fake implementation. Tunable numbers live in config.
6. Code is copied from `rnoro/deployment@91b178b` (`~/Capstone-deployment`) where possible, and each file header says
   what changed ("Derive, don't replicate").
7. Agent-team method: STRUCTURE first, settle open questions with the user, then one executor per folder and one
   reviewer, a local commit per wave, and no push without asking.

## 3. The laptop: `policy_deploy/`

`policy_deploy check | run [--mock] [--station FILE]` is the launcher (`policy_deploy/launcher/launcher.py`). It starts
processes in the order `fake_pc` (only `--mock`) → `frames` → `policy` → `manager`, each after the previous one's
ready file appears, and stops them in reverse. Any process exiting stops the rest. Each process also runs alone with
`python -m policy_deploy.<frames|policy|manager|fake_pc>`.

| Process | Code | What it does |
|---|---|---|
| **frames** | `policy_deploy/frames/` | Receiver thread: SUB to the Pi → `pi_stream` checks → JPEG decode → `UltrawideStore` → `OneWayClockMap` (Pi→laptop from `t_send_ns`, the least-delayed packets of each second fitted for offset and drift) → 1 s `PacketBuffer` + `CameraHealth`. Main thread: C1 REP `FramesServer`. `get_frames` picks 2 main images (T−50 ms, T) and 2 ultrawide images (T−100 ms, T) with `PacketSelector`, judges the pair with `PairRule` (ultrawide within [main−60 ms, main+25 ms], same ultrawide frame used for at most 2 packets), and prepares them as in training (`_dataproc/imageprep.py`: main letterboxed to 224, ultrawide fisheye remapped to a virtual pinhole then letterboxed to 224). Rig file `rigs/rig_pi_v2_640.yaml` (both cameras 640×480 at 30 fps). |
| **policy** | `policy_deploy/policy/` | C4 REP `PolicyServer` that enforces reset → (prompt) → infer. `policy_factory.KINDS`: `hold_still` (no torch, returns the anchor, so zero motion) and `diffusion` (`DiffusionPolicy`, or `PromptableDiffusionPolicy` when the checkpoint has `shape_meta.use_prompting`, which needs `policy.prompt_dataset` plus `manager.task_name`/`prompt_index`). Settings: `precision` fp32/bf16 (autocast), `diffusion_steps`, `device`. Warm-up inferences run before ready. `kind: diffusion` runs in the torch env (`python.torch`). `bench.py` times the policy. |
| **manager** | `policy_deploy/manager/` | `EpisodeLoop` on the main thread, one cycle every `heartbeat_period_s` (20 ms): flags → C2 state rules → **one C3 keep-alive (hello or heartbeat, rule H1)** → collect the worker result → key → health poll → submit the next step → status line. Every C3 request goes out on the main thread. `StepWorker` thread runs `PolicyStep` (C1 get_frames → map the times → look up poses in `StateHistory` → `PoseConverter` absolute→relative → C4 infer → odd-chunk check → relative→absolute timed waypoints). `PcListener` thread puts every C2 into `StateHistory` and `PcClock`. `KeyboardOperator` reads the keys. |
| **fake_pc** | `policy_deploy/fake_pc/` | Python stand-in for franka_control under `--mock`: C2/C3 on 127.0.0.1, a clock at laptop×(1+30 ppm)+1000 s (so every test exercises the clock map), a fake gate, `schemas=SHARED_SCHEMAS`. |

Laptop keys: `h` home (C3 `home`, blocking), `c` start an episode (policy reset (+prompt), then C3 `start`),
`y`/`n` end it as success/failure, `s` stop, `q` quit. `c` and `h` are refused locally while C2 shows `homing`.

Manager start-up (`manager/startup_sequence.py`): wait for C2 → H2 wait (no C3 while C2 shows `running`/`holding`,
up to 2× watchdog, or `homing`, up to 60 s; `fault` refuses) → hellos every cycle until the PC clock map is ready
(≥2 s, ≥10 samples) → `compare_hello` and the laptop's budget checks → `get_state` shows `idle`/`stopped` → heartbeats →
C4 config (sizes, kind, the `settled` rule) → both cameras ok → ready file, phase IDLE.

Session output: `~/policy_deploy_sessions/<date>_<time>/` with `logs/`, `manager/episodes.jsonl`, `steps.jsonl`,
`clock.jsonl`, `link.jsonl`, and `tap/` when `save_tap: true`.

Other tools: `python -m policy_deploy.manager.pc_probe --seconds 30` (read-only link check, a C3 client),
`python -m policy_deploy.frames.pi_collect --station station.yaml --seconds 30` (Pi stream and pair-gap report),
`python -m policy_deploy.frames.probe --save DIR` (the 4 policy images as PNGs),
`~/envs/policy_deploy_torch/bin/python -m policy_deploy.policy.bench --steps 16,8 --precision fp32,bf16`.

## 4. The PC: `control_deploy/`

`control_deploy check | run | console | firewall [--mock] [--local] [--station FILE]` is a Python wrapper
(`control_deploy/control_deploy/`). `run` runs the checks, creates a session folder (`~/control_deploy_sessions/`),
runs the binary `franka_control/build/franka_control --station ... --keyboard` on this terminal (stdin inherited,
output teed), draws a status line from C2, and forwards SIGTERM/SIGHUP. Readiness comes from a ready file.

**franka_control** (C++, `franka_control/src/`, libfranka 0.9.2 in `~/franka/`, built by `franka_control/build.sh [--test]`):

| Thread | Code | Job |
|---|---|---|
| control (1 kHz) | `control_loop.cpp` `ControlLoop::step` | Each cycle: take the newest `LoopCommand` (hold / follow plan / home) → backstop watchdog → target = `plan.at(t)` (linear position, slerp rotation) → `ImpedanceController` (Cartesian impedance at the TCP + nullspace + Coriolis; gravity is the robot's) → torque. It never waits on messages. Homing is a separate libfranka joint move (`HomingMotion`) between torque sessions. |
| messaging | `app.cpp` loop | `Supervisor::tick()` (mode logic, watchdog, homing outcomes) → keys → `CommandServer::serviceOnce` (C3 REP) → `StatePublisher::maybePublish` (C2 at 100 Hz). |
| gripper + gripper-reader | `gripper_worker.cpp` | Finger moves toward the commanded width (deadband 5 mm). The width is read at 50 Hz into C2. Hand homing (`g`) runs here. |
| keyboard | `keyboard_reader.cpp` | `h` home arm, `g` home hand, `s` stop, `q` quit. `h`/`g` are refused while `holding`/`running`. Keys never feed the watchdog. |

`Supervisor` (`supervisor.cpp`) owns the `StateMachine` (`state_machine.cpp`), the `SafetyGate` (`safety_gate.cpp`),
the watchdog and the plan. `onSetWaypoints`: decode → gate → if anything is rejected, **fault** (the whole request is
refused, and a fault needs a restart of `control_deploy run`); otherwise the plan = current target + accepted waypoints
(replacing everything not yet reached), mode `running`, then `holding` when the plan ends. `O_T_TCP = O_T_F × F_T_TCP`,
with `F_T_TCP` from `contracts/frames.yaml`.

`console` is the bring-up C3 client run in place of the laptop: steps `hold move rotate gripper watchdog out_of_box`
(the last one faults on purpose). `firewall` prints an nftables rule that lets only `link.yaml laptop.addresses` and
loopback reach 5620/5630.

## 5. One policy step, end to end (PLAN §4)

1. The worker asks frames for the newest images (C1): main at T−50 ms and T, ultrawide at T−100 ms and T, with times on
   the **laptop clock**. T is the newest main frame's mid-exposure.
2. One `PcClock` snapshot maps both main image times and "now" to **PC time**. The step is skipped if the map is not
   ready, the newest image is older than 250 ms, or the newest C2 is older than 100 ms (mapped).
3. The poses (C2 `O_T_TCP_target` by default, `measured` as a fallback when they are more than 2 cm apart) and widths at
   those PC times are interpolated from `StateHistory`. They are made relative to the **anchor** (the pose at T):
   `eef_pos`, `eef_rot6d`, `eef_rot6d_wrt_start`, `gripper_width_m`, plus the 4 images, go to C4.
4. The policy returns 16 relative steps. An odd chunk (step 0 more than 2 cm or 5° from the anchor) is skipped and counted.
5. The first `exec_h` steps (default 8) become absolute poses on the same anchor, timed `T_pc + k×50 ms` (+
   `lookahead_ms`), and are sent as `set_waypoints`. The PC skips the ones already past, gates the rest, and they
   replace the remaining plan.
6. The worker starts the next step at once (continuous control). The arm holds at the last waypoint if the policy is
   slow. No-hold condition: `L + 2C' + d_up ≤ (exec_h−1)×50 ms` (exec_h 8 → 350 ms). The status line `margin` shows it.

Measured with `run_001` (a prompting checkpoint) on the 3050 Ti: fp32/16 steps ≈ 300 ms, bf16/8 steps ≈ 186 ms per
inference, so it needs `precision: bf16`, `diffusion_steps: 8`, `exec_h: 12`.

## 6. Clocks (never mix them)

| Clock | Lives | Carried by | Mapped by |
|---|---|---|---|
| Pi (libcamera `SensorTimestamp`) | inside frames only | C0 `t_sensor_ns`, `t_mid_ns`, `t_send_ns` | frames `OneWayClockMap` → laptop |
| laptop `CLOCK_MONOTONIC` | laptop | C1 times, arrival stamps | none |
| PC `CLOCK_MONOTONIC` | PC | **every time on C2/C3** (`t_ns`, waypoint `t_ns`, hello `t_pc_ns`) | manager `PcClock`: `round_trip` (hello midpoints, default) or `one_way` (C2 arrivals); `link.yaml clock.estimator` |

Naming in laptop code: `_pc_ns` for PC times, `_pi_ns` for Pi times, `_ns` for laptop times. The PC has no clock code.
Freshness is judged from the mapped `t_ns`, liveness ("no C2 for 2 s") from arrival time, because `t_ns` repeats around
faults and homing. `get_state` replies are mode checks only, never samples. Guard: three waypoint sets in a row that
the PC skipped entirely although the laptop timed them ≥100 ms ahead, or sent while C2 was more than 20 ms in the
future, conclude with `clock mismatch`.

## 7. Failure rule (PLAN §7)

Every problem is either **slow** (skip the step; the PC finishes its plan and holds; steps resume by themselves) or
**conclude** (the laptop sends `stop`, records the episode as aborted, stops its processes in reverse order, and exits
with one line `policy_deploy: FAILED <process> ...: <what>: <what to check>`). **The PC keeps running and holding.**
The PC protects itself: watchdog 1.0 s → `stopped`, gate rejection → `fault`, reflex → `fault`. The full table is in
reference.md §5.

## 8. Running (this laptop)

Envs on this laptop, all installed editable from `~/Capstone-final-deployment`: `~/envs/policy_deploy` (light),
`~/envs/policy_deploy_torch` (torch + behavior_prompting), `~/envs/control_deploy`. A libfranka-less franka_control
is built at `control_deploy/franka_control/build-nolibfranka/franka_control` for the cross test. Always run with
`unset PYTHONPATH`.

```bash
cd ~/Capstone-final-deployment/deployment/policy_deploy && unset PYTHONPATH
~/envs/policy_deploy/bin/policy_deploy check [--mock]
~/envs/policy_deploy/bin/policy_deploy run --mock          # fake Pi + fake PC, hold_still
```

- `policy_deploy run --mock` needs the recorded episode `frames.mock_episode` (default `~/Downloads/episode_000045`).
- Test suites exist (`contracts/tests`, `policy_deploy/tests`, `control_deploy/tests`, C++ via `build.sh --test`, run with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), but **only run them when the user asks**. They take minutes and use the 56xx ports.
- **Never** run `control_deploy run --mock --local` and `policy_deploy run --mock` together on one machine (same ports),
  **except** for the cross test: `launcher.fake_pc: false` with a real `franka_control --mock --local` listening.
- Operator order on hardware: **the PC first** (Desk: unlock brakes, FCI, release the user stop → `control_deploy run`
  → `ready`), then the Pi C button, then the laptop (`policy_deploy check`, then `run`). To end: laptop `q`, Pi
  button, PC `q`, lock the brakes.

## 9. Changing things

| Change | What to do (STRUCTURE.md §4.5, §7) |
|---|---|
| A C2/C3 field (add, remove, rename, type, unit, frame, clock) | Contract change: bump the schema, update the Python class, `python -m deploy_contracts.make_examples`, update the C++ codec (`franka_control/src/codec.cpp`) and `messages.h`, all in one commit. |
| A key in limits/frames/link.yaml | Update the Python loader **and** the C++ reader in the same commit. |
| A value in limits.yaml / frames.yaml | Shared data: same commit on both machines, rerun `make_examples`. `hello` compares the values at every start. |
| A value in link.yaml | Tunable: edit it on both machines (commit + pull) and restart both. |
| A laptop-only message (C0/C1/C4) | `policy_deploy/policy_deploy/contracts/`, with its examples built in memory, never in `contracts/examples/`. |
| A new policy kind | A `Policy` subclass + a `policy_factory.KINDS` entry. `config.reply/v2` carries `kind`. |
| A new laptop key | One `Key` member + one `KEY_HANDLERS` entry in `episode_loop.py` (jog keys are planned, PLAN §14). |
| Anything on the PC | Rebuild after every pull (absolute contract paths and the commit are baked in). `hello.commit` is shown by `policy_deploy check`. |

Both machines must be on the same commit. A commit difference is a warning; `compare_hello` mismatches refuse.

## 10. Open items (as of 2026-11-18)

- **TCP axes (PLAN §9):** `frames.yaml F_T_TCP` is still rnoro/deployment's Franka Hand TCP (−45° about flange z, 150 mm
  out), whose axes are turned 90° about z from training's `T_cam_tcp` (`rig_pi_v3.yaml` on rnoro/data-work).
  `settled: false` makes the laptop refuse a trained policy on the real arm, so only `hold_still` runs until this is fixed in
  `F_T_TCP` (never as a correction on the laptop) and checked on the arm.
- `link.yaml` and `station.yaml pi.address` hold PLACEHOLDER addresses (192.168.50.10/.20/.30). Reserve them on the
  router.
- `franka_control/config.yaml` TODOs: workspace box, payload (`load.mass_kg` 0), `gripper.width_offset_m`, gains,
  collision thresholds, cone, home pose versus box. Put measured values in `control_deploy/station.yaml`.
  `realtime: ignore` until there is a PREEMPT_RT kernel.
- `exec_h` is chosen per checkpoint from `policy.bench`. `pair_max_lead_ms` is set from the `pi_collect` report.
- Verified on hardware: the Pi stream (30 pkt/s, no gaps) and `h` homing on the PC. **Not yet verified:** waypoint
  following on the real arm, the laptop↔PC link, a trained policy on the arm.
- Later (PLAN §14): keyboard jog keys on both machines, a hand-eye check, ZeroMQ CURVE on C3.
