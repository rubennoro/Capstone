# Hardware day: ordered procedure

Do these steps in order. Each one has a pass number and names the log field that proves it. Don't start a step until
the one before it has passed. All of this was built and tested on the laptop with fakes. **None of it has run on
the Jetson, the arm, or the gripper.**

Conventions:

- `FD=controlSW/franka_deploy`.
- Commands run on the Jetson from the repo root, with `PYTHONPATH=$FD` and the deploy venv
  (`scripts/jetson_deploy_setup.sh`).
- `<S>` is a session directory under `work_root`. `$S` in the commands below is any scratch directory with free space.
- Every number written as "pass:" is the acceptance threshold.
- Keep a paper log. Note each step's result and the path of its JSON next to it.

People and safety, throughout:

- Two people. One holds the external activation device (user stop) at all times. The other types.
- The zone is clear. Speeds start at the values below and are raised only where this document says so.
- **Abort rule** (the whole day, and the evaluation): after **2 safety events**, stop. A safety event is a reflex,
  a user-stop press, an unexpected contact, or a watchdog stop. Debug before continuing.

---

## Which path this day follows

Two hardware paths exist. The sections below say which one a step belongs to.

| Path | Gripper | Camera | Profile | Steps |
|------|---------|--------|---------|-------|
| **Servo** (original, PURCHASING §2) | motorized iPhUMI gripper (Feetech/Dynamixel) | Jetson rig (`RigCamera`) | (base, no `--profile`) | 0-10 as written. Steps marked "servo gripper only" are skipped on the Franka Hand path |
| **Franka Hand + iPhone** (the sim evaluator's path) | Franka Hand with printed fingers (`hand_server` on 5566) | iPhone app over Wi-Fi (`WirelessCamera`) | `real-iphone` (`real-rig` for the rig's v2 stream) | 0 (skip 0.6, 0.7), 1, 2, 3, **11**, **12**, 4, 5, **13**, 7, 8, **14**, 9, 10, **15** |

## Decisions to record first (nothing below is decided for you)

Record each answer on paper and in [TODO.md](../../../TODO.md) before the step that needs it. **OPEN** = still waiting on the user.

| ID | Decision | State | What it needs |
|----|----------|-------|---------------|
| D1 | **Panda or FR3** | **OPEN** | Pins libfranka / panda-py (`scripts/build_libfranka_pandapy.sh --robot`); `check-config` fails until it is set (`robot.fci.libfranka_version` is a critical hardware placeholder). The Stage 1 sim found both arms about equally feasible with the same tooling (43/45 demos each). Confirm which arm the lab has |
| D2 | **Gripper actuator**: Franka Hand + printed fingers, or a servo-driven iPhUMI gripper | **OPEN** | The sim, `real-iphone` and §11-13 evaluate the **Franka Hand**; PURCHASING §2 still recommends the servo (Feetech, same fingers and tags as training). Franka Hand: finger adapters, the width map (§13), `hand_server` (§12). Servo: the `FeetechGripper` driver (0.6), servo width table (§6) |
| D5 | **v_max and the success rule** | **OPEN** | v_max 0.25 vs 0.5 m/s, and the success rule (lift, hold time, timeout, slip). Placeholders today: `criteria/block_raise.yaml` `limits` 0.25 / 1.0 and `success` lift 0.05 m, hold 2 s, 60 s; TODO D5 says lift 3 cm, §10 says 5 cm. All three disagree until you pick |
| D9 | **Tooling geometry: printed fingers about 10 cm longer than stock, phone on the -y side** | **OPEN, needs your confirmation** | Stage 1 (iPhone housing only): +10 cm and -y gives **43/45 demos feasible on Panda and FR3**. Stock-length fingers put the phone into links 5/6: 31/45 (+y) or 8/45 (-y) on Panda, 37/45 or 27/45 on FR3, worst tooling clearance -22 to -62 mm (`controlSW/docs/sim_plan/results/stage1/summary.txt`). **Caveats (A3 review):** 37/45 demos leave the 60 degree bridge cone, so the SafetyFilter would clamp them on hardware (the cone is not part of feasibility); joint speeds stay within limits for only 19/43 (FR3) and 36/43 (Panda) of the feasible demos. **Not yet re-run with the 30 mm-thick rig housing.** Sets `tooling.finger_adapter.extension_m` and `tooling.tcp_x_along_hand` (still 0.0 and "+y") |
| D10 | **Gates still open**: Stage 2b live-vs-offline p50/p95/n_pairs, Stage 3 `obs_motion` 3 mm | **OPEN** | `criteria/block_raise.yaml` (all marked placeholder). obs_motion = arm speed x image age (about 5-9 mm at demo speeds) can never meet 3 mm, so pick the gate or turn on `--match-camera-latency` (off today, your decision) |
| D11 | **Is the real TCP the pad centre?** (A6) | **OPEN** | The station `grasp_tcp_height_m` 0.01 puts the 30 mm pads 5 mm into the table if the TCP is the pad centre, so the sim uses 0.018. If the TCP is the pad centre, set it to 0.018. Measure in 5 / 4.3 (pivot) and 12.8 |

What the sim could not settle, and which step measures it:

| Open assumption | Where it lives | Measured in |
|-----------------|----------------|-------------|
| iPhone capture latency 0.08 s (`sensor.clock.latency_s`, `provenance: placeholder`) | `iphone16pro_wifi.yaml` | 11.5 (QR clock) |
| Rig `residual_latency_s` 0 | `rig1_wifi.yaml` | 11.9 |
| Deployment stream K vs station K vs the two training K variants | `streams.*.K` | 11.6 |
| `width_map.a` = 1.0 is a **critical placeholder**; `b` = 0.0431 is **derived** (= `train_closed_m`, the training width at gap 0; the old b 0.008 made the demo hold widths 0.046-0.062 imply a 38-54 mm gripped side instead of the 20 mm eraser) | `franka_hand_iphumi_*.yaml` | 13 |
| `hand.grasp_epsilon_inner_m` 0.015 (needs >= about 0.009 so the 20 mm side is not missed; a no-object close still stops >= 3 mm above 0) | tooling yaml | 12.8 |
| libfranka Hand semantics (GIL, `stop()`, `grasp` without an object, force per finger) | `hand_server.py` LibfrankaHand | 12 (checklist §7) |
| panda-py `CartesianImpedance` law. The sim's implementation is **under-damped (15-34 % overshoot on ideal 2 cm steps)** and is a model of the documented law, not of the source | `franka_sim/simcore.py` | 14 (checklist §6) |

Limits use the **minimum rule**: every speed limit is the minimum of the station `criteria.limits`, deploy.yaml `task.v_max/w_max` and `bridge.override.safety.v_max/w_max`. `--match-camera-latency` is **off** (user decision): the task samples the robot state at "now", so the observation pairs an image that is about 126 ms old with the current proprio; at demo speeds that is the 5-9 mm obs_motion above.

## 0. Before power-on (desk work)

| # | Item | Pass |
|---|------|------|
| 0.1 | Robot model and system version (Desk → Settings → System). FCI enabled. | recorded on paper |
| 0.2 | `scripts/jetson_deploy_setup.sh --franka-if <iface>`. Run `--dry-run` first, then for real, then reboot. | script ends with "done" |
| 0.3 | `python -m franka_deploy.tools.jetson_verify --json verify.json` | **12/12 PASS**: `preempt_rt`, `cmdline`, `nic_irq`, `nvpmodel`, `limits`, `static_ip`, `robot_ping`, `camera_0`, `camera_1`, `cuda_torch`, `imagecodecs` (2023.9.18), `l4t` |
| 0.4 | `scripts/build_libfranka_pandapy.sh --robot panda\|fr3`. Panda 4.x → 0.9.x; FR3 → ≥ 0.10 (check the compatibility table for the exact system version). Copy the two printed lines into `configs/bridge.yaml`. | `<prefix>/franka_versions.json` written; `import panda_py` works |
| 0.5 | Re-check the panda-py names used by `PandaPyBackend` against the pinned version (list in `docs/panda_py_checklist.md` §0). | every name exists |
| 0.6 | *(servo gripper only)* Implement `FeetechGripper` (or `DynamixelGripper`) `_write_position/_read_position/_write_current_limit` in `gripper/driver.py`. It is a `NotImplementedError` stub today. | `make_gripper({"type": "feetech", ...})` opens and reads a position |
| 0.7 | *(rig camera only; the iPhone path has no rig)* Rig calibrated (`sensor_rig/docs/calibration.md`). Copy `rig_calibrated.yaml` to `camera.rig_config_path`. | `check-config` row `rig_not_default` OK |
| 0.8 | Deploy envs built clean on the Jetson, without relying on `~/.local` or a ROS `PYTHONPATH` (D0 G3). Copy the checkpoint and prompt zarr to the paths in `deploy.yaml`; set `prompt_zarr_sha256`. | `python -m franka_deploy check-config --config deploy.yaml` (no `--profile`: the deploy.yaml base is the Jetson): **all OK** (including `panda_py_versions_pinned`) |
| 0.9 | Disk: video is about **4.3 GB/h** (robologger fixes 5 Mbit/s per camera). Free space ≥ the session plan + `disk_budget_gb` (5 GB). | `df -h` noted |
| 0.10 | Real spacemouse connected (`spacemouse.mock: false`). | the spacemouse server starts |

## 1. Real-time go/no-go (Jetson + robot, arm holding only)

| # | Command | Pass / log field |
|---|---------|------------------|
| 1.0 | **Pre-test, on its own (not alongside libfranka):** `sudo cyclictest -q -m -a 5 -t1 -p 80 -i 1000 -h 2000 -D 10m > cyclictest.txt` while `stress-ng --cpu 5 --taskset 0-4` runs. p99.9: `python -c "from franka_deploy.tools.gonogo import parse_cyclictest as p; print(p(open('cyclictest.txt').read()))"`; max: `# Max Latencies`. | wake-up latency p99.9 < 0.5 ms and max < 1 ms on core 5 |
| 1.1 | libfranka `communication_test 172.16.0.2` (from the libfranka build dir) | success rate ≥ 0.99 (printed) |
| 1.2 | `python -m franka_deploy gonogo run --out gonogo_30min --config <bridge.yaml with backend.type panda_py> --bridge-core 5 --bridge-rtprio 80 --hold-s 60 --duration-s 1800 --stress --tegrastats --load-cmd "<realenv-py> -m real_env.peripherals.rig_camera wrist_right" --load-cmd "<bp-py> -m franka_deploy.tools.bench_inference --ckpt <ckpt> --work-dir <scratch> --dataset-path <zarr> --n 100000"` | `gonogo_30min/gonogo.json`: **`go: true`**, meaning `reflexes == 0`, `watchdog_stops == 0`, **`min_success_rate ≥ 0.99`** (Franka's `control_command_success_rate`, worst second). On hardware these are the gates. The loop-timing gate comes from the 1.0 pre-test: panda-py reports no per-cycle times, so copy `cyclictest.txt` into the run dir before `gonogo evaluate`. Otherwise evaluate reports "no loop timing". Also record `tegrastats.temp_max_c` (tj below the throttle point, ~95 °C), `tegrastats.ram_used_max_mb`, `client.rtt_p999_ms_max` |
| 1.2b | **Micro-clamp rule:** run the stack (task idle, spacemouse teleop, or `run --replay-episode` at 1×) under the same load. Count `clamp_micro` (and `clamp_pos`) against `ticks` in the controller shutdown line. | if `clamp_micro / ticks` > **1 %**, raise `safety.slack_s` 0.010 → **0.020** in `bridge.yaml` (overshoot bounded by v_max·slack = 5 mm at 0.25 m/s) and repeat |
| 1.3 | Fail → move only the bridge to an x86 PREEMPT_RT PC (`bridge.endpoint: tcp://<rt-pc>:5565`). Repeat 1.1–1.2 there. | same |

The arm moves ±2 cm at 1 Hz in 1.2. The amplitude (`--amp-m`) and axis (`--axis`) are set for that.

**Timing source.** On the laptop, the gonogo loop gate uses the FakeBackend's per-cycle *compute time* (a
`cycle_hist_us` histogram). That is a laptop-only check of the tooling, not a real-time claim. On hardware the gates
are Franka's `control_command_success_rate` (≥ 0.99 every second) and 0 reflexes. Wake-up latency comes from the
separate cyclictest pre-test (1.0). Don't run cyclictest at RT priority on core 5 at the same time as libfranka's
thread: it would compete with the loop it is meant to qualify.

## 2. Safety pre-checks (before anything moves on its own)

| # | Item | Pass / log field |
|---|------|------------------|
| 2.1 | User stop: press while the bridge holds. | arm stops; bridge log `reflex` or the robot mode changes; recovery only through `recover` |
| 2.2 | FCI active, Desk in "Execution", brakes open. | `get_state.mode` |
| 2.3 | Payload: gripper + rig + Jetson (~1–1.5 kg, plus CoM) in Desk or `setLoad`. | values recorded; idle `O_F_ext` within ±3 N |
| 2.4 | Collision thresholds lowered (`backend.panda_py.collision`, ~20 N / 10 Nm to start). | values in `bridge.yaml` |
| 2.5 | Measure the workspace box and cone around the table. Enter `safety.workspace_min/max` and `cone_*` in `bridge.yaml`. Test each rule with deliberately bad targets through `BridgeClient` (`docs/panda_py_checklist.md` §2). | NaN → `target_rejected_nonfinite` + hold; outside the box → `safety.box` count; outside the cone → `safety.cone` count; a 10 cm jump → executed at ≤ 0.25 m/s (`safety.leaky`) |
| 2.6 | Watchdog: stop sending targets. | `watchdog_hold` within **0.2 s**, `watchdog_stop` within **2 s** (bridge.jsonl `age_s`) |
| 2.7 | Reflex: push past the thresholds. | bridge `reflex`; controller status `fault`; no automatic recovery |

## 3. Bridge alone (no controller, no task)

| # | Command | Pass / log field |
|---|---------|------------------|
| 3.1 | Home (joint move, `move_joints`), then `start`. Compare `get_state` with Desk. | O_T_EE matches Desk to < 1 mm |
| 3.2 | Steps: `gonogo run --out steps --motion steps --amp-m 0.02 --hold-s 10 --duration-s 60` (2 cm per axis at 0.05 m/s, 2 s holds). Tune impedance (600–1000 N/m, 30–50 Nm/rad) and filter. | `steps/gonogo.json` `tracking.lag_ms` = **τ** and `tracking.steady_err_mm_median` = **e** (record both; e ≤ 2 mm); `reflexes 0` |
| 3.3 | Sine: `gonogo run --out sine --motion sine --amp-m 0.03 --hold-s 10 --duration-s 60` (±3 cm, 1 Hz, 0.19 m/s peak) | `tracking.lag_ms` (τ under motion), `tracking.track_err_mm_max`; `go: true` |
| 3.4 | Set `controller.lookahead_time_s = 0.1 + τ` in `franka_cartesian.yaml`. | edited |

## 4. Frames: TCP via CAD, then hand-eye

| # | Command | Pass / log field |
|---|---------|------------------|
| 4.1 | CAD F_T_C of the rig's virtual main camera on the flange → `cad.json` (4x4). F_T_TCP(CAD) = F_T_C @ C_T_TCP (`franka_cartesian.yaml` comment). | written |
| 4.2 | Fixed ChArUco board (9x7, 30 mm squares, 22 mm markers, DICT_5X5_100, printed flat; measure the square size). Camera running (`rig_camera wrist_right`). Arm hand-guided, bridge **not** running a controller. `python -m franka_deploy fdcal capture handeye --out he` gives ≥ 20 poses with large rotation diversity (tilt ±30–40°, roll ±60°, board 0.3–0.5 m away). | ≥ 20 rows in `he/poses.json` (frames without a detected board are rejected) |
| 4.3 | `python -m franka_deploy fdcal handeye --data he --cad-F_T_C cad.json --out he/result.json` | **`cad_agreement.accept`: ≤ 3 mm / 1°** (`cad_agreement.mm`, `.deg`); `tsai_vs_park_mm_deg` ≤ [1 mm, 0.3°]. Copy `F_T_TCP` into the tooling YAML (`tooling.F_T_TCP`, status `measured`); servo path: `franka_cartesian.yaml` |
| 4.4 | Either `ee_is_tcp: false` (the default: the bridge commands the flange), or set the EE in Desk / `NE_T_EE` with `ee_is_tcp: true`. Never both. | one of the two |

## 5. Pivot test

| # | Command | Pass / log field |
|---|---------|------------------|
| 5.1 | Fingertip centre on a marked point, bridge `start`. `python -m franka_deploy fdcal capture pivot --out piv --F_T_TCP <yaml holding the measured F_T_TCP> --angle 20` (drives ±20° about TCP x, y, z at ≤ 10°/s and ≤ 2 cm/s, records each extreme). | **`piv/pivot.json`: `tcp_excursion_mm ≤ 2` and `F_T_TCP_translation_error_mm ≤ 2`** (`accept: true`). Also look: the fingertip stays on the mark |

## 6. Gripper width map (servo gripper only: on the Franka Hand skip to §13)

| # | Command | Pass / log field |
|---|---------|------------------|
| 6.1 | Finger AR tags visible in the ultrawide. `python -m franka_deploy fdcal capture width --out wd --controller-config configs/franka_cartesian.yaml --positions <closed>:<open>:<step>` (closed first, ≥ 10 positions). | `wd/sweep.json` |
| 6.2 | `python -m franka_deploy fdcal width --data wd --out wd/calib.yaml` (iPhUMI's tag detector and width definition). Copy `calibration` into `franka_cartesian.yaml`. | **`checks.zero_at_closed`, `checks.monotonic`, `checks.covers_required` all true** |
| 6.3 | *(servo)* Current limit and squeeze δ on the eraser. Log commanded vs reached width (D2 risk: the history stays "open" if a current-limited squeeze never reaches the command). | eraser held without slipping; the reached width is logged |
| 6.4 | *(servo)* Hold during a grasp: send `hold`. | commanded width unchanged; the object doesn't drop |
| 6.5 | **Commanded vs reached width.** During 5 grasps of the eraser, log commanded width (`target_width`) and reached width (`get_width()` through the width map). The policy's gripper proprio is `task.width_proprio: commanded` (the upstream action-as-proprio convention). | reached ≈ the eraser width (± 2 mm) while commanded is lower by the squeeze δ. If reached differs from commanded by more than δ + 3 mm when the fingers are *not* blocked, fix the map (6.2). If grasps stall far from the command, A/B `width_proprio: measured` vs `commanded` in 9.1 (5 trials each) before scoring |

## 7. Controller + teleop

| # | Command | Pass / log field |
|---|---------|------------------|
| 7.1 | Launcher with the task idle, spacemouse teleop. | controller status **`hop_ms_p99` < 5**, `period_err_ms_p99` < 1 under load; no `transient` errors |
| 7.2 | Measure the home pose → `deploy.yaml home.pose` and `home.xyz`. | the `r` key homes; `c` refuses when FK is > 5 cm from home |
| 7.3 | **Camera timing.** If the main sensor runs at 30 fps (lens-dependent), the 20 Hz obs pick main frames 33/67 ms apart instead of an even 50 ms (training: 60 fps). Check the camera stats line (packet rate) and the tap's main/ultrawide timestamps. Prefer a 60 fps main mode if the FOV allows. | preflight `camera_rate_and_selection` OK; main frame spacing recorded (50 ms, or 33/67 ms at 30 fps) |
| 7.4 | **Video budget:** robologger fixes 5 Mbit/s per camera, ~70–85 MB per minute, **~4.3 GB/h**. Size the session plan (N trials × duration) against free space. | free space ≥ planned hours × 4.3 GB + `disk_budget_gb` |

## 8. Open-loop replay (no policy)

| # | Command | Pass / log field |
|---|---------|------------------|
| 8.1 | `python -m franka_deploy run --config deploy.yaml --replay-episode <train demo> --replay-speed 0.5`, then `c` from home. The arm replays the demo's relative motion through the full controller/bridge path. | episode data: TCP path vs the demo's start-relative path **≤ 1 cm / 3° over 5 s** (the laptop mock gives 0.55–0.76 mm); clamp counters = 0 |
| 8.2 | Same at `--replay-speed 1.0`. | same pass numbers; clamp counters reported (a few `leaky` clamps are expected at v_max 0.25 for fast demo segments) |

## 9. Closed loop (policy)

| # | Command | Pass / log field |
|---|---------|------------------|
| 9.1 | `python -m franka_deploy run --config deploy.yaml`: pause mode, **exec_h 8, v_max 0.25 (station `criteria.limits`, task.v_max, bridge: min of all three), prompt_index 0**, inference timeout 2 s. 10 trials from home. | per call `info_anchor_dev_m` in the tap: > 2 cm → fallback + reseed (`anchor_fallbacks`), > 5 cm → hold; 0 safety events |
| 9.1b | **Settle gate** (pause mode): before each capture the task waits until the bridge backlog is ≤ 2 mm, `track_dev` ≤ 3 mm and the arm is still (0.5 mm over 50 ms), with a 1 s timeout. | `task_counters.settle_waits` and settle times in the tap; no `SettleTimeoutError`; settle p95 < 0.3 s (otherwise lower impedance lag / check τ from 3.2) |
| 9.1c | **Anchor sweep, measured vs target:** 5 trials each with `task.anchor: target` (default) and `task.anchor: measured` in `deploy.yaml`, same start buckets. | per call `info_anchor_dev_m` (median, p95), `anchor_fallbacks`, clamp counters, success. Keep `target` unless `measured` is clearly better; record the choice before section 10 |
| 9.2 | Stop paths: stop file / command key → hold ≤ 0.1 s (`hold_request_t` → `hold_applied_t` in the controller shutdown line); user stop works. | ≤ 0.1 s |
| 9.3 | Latency on the Orin: `bench_inference` warm p50 / p95 (fp32 vs bf16), first call cold. Then the end-to-end **capture → reply** latency from the tap during 9.1. | recorded; < 2 s for pause mode. **Continuous mode (`pause: false`) only if the measured capture → reply p95 < 0.42 s** (otherwise the next chunk arrives after the current one runs out); stay in pause mode if not |
| 9.4 | Observation check on real images: compare `agent_capture` taps against the training distribution (the eval OOD fraction). D1 covered the remap chain only, so a real rig's viewpoint and photometric differences are still open. | `ood.frac_outside` recorded |
| 9.5 | Only after B3-hw tuning: raise the speed limit to 0.5 m/s. **Rule: every limit is the MINIMUM of its sources** (station `criteria.limits.v_max_m_s` / `w_max_rad_s`, the deploy.yaml `task.v_max/w_max`, and the deploy.yaml `bridge.override.safety.v_max/w_max`), so lowering any one of them lowers the limit and raising needs all of them: add `criteria.limits.v_max_m_s=0.5` to the profile's `station_set` (or `--set`), and set `task.v_max: 0.5` (or `null` = take the station value) in deploy.yaml; remove any lower `bridge.override.safety`. The launcher prints a loud `WARNING: limit mismatch` and logs it when the sources differ, and `config/bridge.yaml` / the task command show the value used. | the 9.1 checks hold again; no limit-mismatch warning |

## 10. Evaluation protocol (fixed before the first scored trial)

- **Success rule**: the eraser is lifted ≥ 5 cm above the table and held for 2 s, within 60 s of `c`, with no human
  contact. Written down before trial 1. Annotate with
  `python -m franka_deploy annotate <episode> --success yes|no --category ... --annotator <name of the person judging>`.
  `--annotator` is required.
- **Conditions** (fixed in advance; one config each): e.g. prompt_index 0 vs another, v_max 0.25 vs 0.5. Change one
  thing at a time. Each condition's `deploy.yaml` is snapshotted in the session (`config/`).
- **N ≥ 20 trials per condition.** Start poses are randomised over pre-marked buckets: eraser position on a 3x3 grid
  and 2 orientations. Draw the bucket order with a seeded random sequence written down beforehand.
- **Abort** after 2 safety events (see the top of this document).
- Report: `python -m franka_deploy eval <sessions...> --out eval_report` (env_bp).
  - Fields: `success_rate.{k,n,p,ci95}` (Wilson); `by_condition`; `clamps_total`; `holds_per_h`;
    `faults_per_h`.
  - Per episode: `similarity.ratio_to_demo_baseline` (demo LOO baseline 30.9 mm) and `ood.frac_outside`.

## 11. iPhone wireless path (profile `real-iphone`)

The app streams both cameras as 960x720 JPEGs over Socket.IO (main on port 5555, ultrawide on 5556). The receiver
(`WirelessCamera`) dates each frame as arrival minus `clock.latency_s`, then republishes both at 60 Hz exactly like
`RigCamera`. Nothing here has run with the real app. The rig's v2 stream is 11.9.

| # | Command / action | Pass / log field |
|---|------------------|------------------|
| 11.1 | **App settings** (the station records them in `sensor.app_settings_required`): Deployment mode, connection **Ethernet**, host IP = this machine, Stream resolution **High** (960x720 after the app's x0.5; the app default **Low is 320x240** and fails the receiver's `on_size_mismatch` with a message naming the setting), **30 fps** (the app default is 60; the station requires 30). | the receiver prints no `size_mismatch` warning; `streams.*.size_mismatches` 0 |
| 11.2 | **Network:** a dedicated 5 GHz access point, the policy host wired to it, the phone on that AP only, auto-lock off and the phone on power. Open inbound TCP **5555 and 5556** on the host (`sudo ufw allow 5555:5556/tcp`). `ping -i 0.2 -c 100 <phone>`. | 0 % loss; RTT p95 recorded (suggested < 10 ms) |
| 11.3 | `python -m franka_deploy check-config --config configs/deploy.yaml --profile real-iphone` | every CRITICAL line is a value this document measures (`robot.desk_end_effector`, `robot.fci.libfranka_version`, `robot.impedance.collision_thresholds`, `tooling.width_map.a`, `sensor.clock.latency_s`, ...). A real run refuses to start until each is measured and applied, or listed in `allow_placeholders` |
| 11.4 | **Connect, receiver alone.** Write the receiver config once (command below), then `python -m franka_deploy.peripherals.wireless_camera --config camera.yaml --stats-path stats.json`; in the app tap Connect. | both streams print `connect #1 transport=websocket` (not polling); `fps_in` 30 +- 1 per stream; `decode_ms_p95` < 10 ms; `size_mismatches` 0; `hello_problems` n/a (v1) |
| 11.5 | **Latency (QR clock).** Stop the receiver from 11.4 (the tool binds the same ports), then `python -m franka_deploy wireless-latency measure --config configs/deploy.yaml --profile real-iphone --duration-s 60 --display-hz <screen refresh Hz> --redraw-hz 60 --fullscreen --out lat_iphone`. Point the phone at the QR window so the code fills about a third of the main view and is also in the ultrawide view. Repeat once after 10 minutes of streaming (the phone warms up). | per stream: **decode rate >= 0.8**; **p95 - p50 <= 30 ms**; first-vs-last window drift **<= 5 ms**. The tool prints `--set sensor.clock.latency_s.main=...` / `.ultrawide=...` and a `provenance` line: apply both by hand (it never edits YAML). It also suggests `sensor.task.camera_latency_s` (the pause wait: p95 arrival delay; **not a stamp correction**) |
| 11.6 | **Stream intrinsics.** ChArUco board (9x7, 30 mm, 22 mm markers, DICT_5X5_100) on a table. `python -m franka_deploy fdcal capture stream --out strm --n 30` (needs the receiver running as in 11.4 and the camera server publishing; Enter saves a view when the board is seen), 20-30 views with the board near every part of the frame, tilted +-30 degrees. Then `python -m franka_deploy fdcal stream-intrinsics --data strm --config configs/deploy.yaml --profile real-iphone --out strm/result`. | per stream **`match`**: \|df\| <= 0.5 % and \|dc\| <= 3 px against the station K; reprojection rms < 0.5 px; coverage >= 8/9 cells. The tool also prints the nearest of the two training variants (39 demos / 6 demos). A mismatch is a **finding**, not a fix: the policy was trained at the training K, so the snippet for `sensor.streams.<s>.K` records what the deployment stream really is. Try `--distortion k1k2` if rms is high |
| 11.7 | **Reconnect check** (task idle, stack running): in the app toggle Wi-Fi off for 5 s, then on. | `health.reconnects` increments for both streams; while it is down the receiver stops publishing (`stale_ticks` rises after `max_age_s` 0.3 s) and the task refuses images older than 0.35 s (`FrameSelectionError`, no actions from stale frames); after reconnect it resumes by itself, no restart; `reconnects` grows by 1, not unboundedly |
| 11.8 | **Receiver health under the full stack:** start the stack as in §8 (replay) or §9 and read the camera stats line and `stats.json`. | `age_s p95` <= 0.15 s (Stage 3 gate `receiver_age_s_p95_max`); preflight rows `camera_rate_and_selection` and `wireless_health_ok` OK; `stale_ticks` 0 in steady state |
| 11.9 | **Rig (v2) stream** (`real-rig`): same as 11.1-11.5 with `sensor-rig stream-send` on the Jetson; `wireless-latency measure ... --profile real-rig --out lat_rig`. | per stream **p95 - p50 <= 5 ms** (= `cam_stamp_err_ms_p95_max` v2); the tool prints `sensor.clock.residual_latency_s.*` = the old value + the p50 error; `hello_problems` empty; clock offset/drift in the stats line |

One-time receiver config for 11.4 / 11.6 (the launcher writes the same file into every session's `config/camera.yaml`):

```
python -c "import yaml; from franka_deploy.launch.launcher import load_config; from franka_deploy.station import station_from_deploy_cfg as f; c = load_config('configs/deploy.yaml', 'real-iphone'); yaml.safe_dump(f(c).camera_config(c['camera']['endpoint'], c['camera']['logger_endpoint']), open('camera.yaml', 'w'))"
```

**Display bias.** The QR time is the moment the code was drawn, not when its light left the panel, so `wireless-latency`
over-estimates the latency by the display lag: up to one refresh plus panel processing, roughly **10-25 ms at 60 Hz**.
Use a 120 Hz or faster screen if you can, and write the screen's refresh rate in `--display-hz` so the report states the bias; `--redraw-hz` (default 60) is how often the QR is redrawn and must not exceed the refresh rate. The
v1 value to apply is the p50 as printed; subtract your own estimate of the display lag only if you measured it.

## 12. Franka Hand (hand_server; checklist [§7](panda_py_checklist.md))

`LibfrankaHand` has never touched a hand. Run `hand_server` alone first, drive it with a `BridgeClient`, then the
controller. Names are from `panda_py_checklist.md` §7; each step below is one of its items.

```
python -m franka_deploy.gripper.hand_server --endpoint tcp://localhost:5566 --backend libfranka --hostname <fci-ip>
python - <<'EOF'
from franka_deploy.bridge.link import BridgeClient
c = BridgeClient("tcp://localhost:5566", timeout_s=0.5)
print(c.request("hand_state"))
print(c.request("hand_cmd", {"kind": "move", "width_m": 0.04, "speed_m_s": 0.05, "force_n": 0.0, "seq": 1}))
EOF
```

| # | Item (checklist §7) | Pass / log field |
|---|---------------------|------------------|
| 12.1 | Names exist in the pinned panda-py: `panda_py.libfranka.Gripper(hostname)`, `homing()`, `move`, `grasp(width, speed, force, eps_in, eps_out)`, `stop()`, `read_once()` (`.width`, `.is_grasped`, `.max_width`). | every name imports; `hand_state` answers |
| 12.2 | **GIL:** `hand_state` replies while a `move` runs (loop the request at 100 Hz during a 0.08 m move at 0.02 m/s). | reply time **< 5 ms** throughout. If not: the driver counts `rpc_timeouts`/`stale_reads`, the controller is unaffected, but hand_server needs the device in a subprocess |
| 12.3 | **`stop()` mid-motion:** send `stop` during a `grasp` and during a `move`. | the blocked call returns at once: `last_result: stopped` within 0.2 s (suggested); a new `hand_cmd` while busy replaces the old one (latest wins) |
| 12.4 | **`read_once()` rate** idle and during a command (`gap_vel_m_s` is a finite difference of it). | >= 50 Hz both ways (the state thread polls at 50 Hz) |
| 12.5 | **Homing** after power-on: time from `hand_server` start to the first `hand_state`. | **<= 20 s** (the launcher waits 20 s); record whether `move`/`grasp` need homing first |
| 12.6 | **`grasp` without an object**, and with an object narrower than `width - eps_inner`. `grasp` to 0.04 with `eps_inner 0.005`, hand empty; then onto the 20 mm eraser side from a start gap 0.05. | record: closes to 0, or stops at `width - eps_inner` (the sim and FakeHand stop there)? Does it still close on the eraser? A difference changes what `grasp_ok` means in the sim |
| 12.7 | **Force per finger vs total:** a force gauge between the pads, `grasp` with `force_n 30`. | gauge reads **30 N** (sim: 30 N per pad, tendon force 2F). If it reads about 15 or 60 N, rescale `hand.force_n` and record it |
| 12.8 | **Eraser epsilons:** grasp the 20 mm side with the station's `hand.grasp_epsilon_inner_m` **0.015** (`outer` 0.08) from the policy's trigger widths (train width 0.061-0.064). The pads should stop on the eraser at gap about 0.012 (pad offset 0.008). | `grasp_ok` at the eraser; held 2 s without slipping; `eps_inner` >= about 0.009 holds (below it the grasp is reported as failed); an empty close stops >= 3 mm above 0 |
| 12.9 | **Re-grasp:** while GRASPING, make the hand idle and not holding (stop it mid-hold, or restart `hand_server`). | up to `policy.max_regrasps` (2) new grasps 0.25 s apart; the fingers do not cycle |
| 12.10 | **Shutdown while holding:** SIGTERM `hand_server` with the eraser held. | the grasp force stays (a finished grasp is not `stop()`ped); release with Desk / `homing` |
| 12.11 | **Hysteresis on the hand:** `gripper.policy.close_below_train_m` **0.064** / `open_above_train_m` **0.070** (from the firstreal-single widths; the closest demo trigger is 0.02 mm under 0.064, so these are fitted to this dataset). Teleop a close and an open. | no chatter while holding; exactly one release; re-check after §13 changes `a`/`b` |
| 12.12 | **With the controller:** `gripper.type: franka_hand` (profile `real-iphone`; endpoint `tcp://localhost:5566` from `tooling.hand.endpoint`). `check-config` rows `controller_gripper`, `franka_hand_driver`, `hand_server`. Teleop close/open. | driver status: `rpc_timeouts` 0, `rpc_errors` 0, `stale_reads` ~0, `cmds_sent` rises |

## 13. Franka Hand width map

The policy sees a training width, `train = a * gap + b` (`tooling.width_map`). `a` = 1.0 is a **critical placeholder**;
`b` = 0.0431 is **derived** (= `train_closed_m`, the tag separation of the training gripper at gap 0). Both must come from the hand.

| # | Command | Pass / log field |
|---|---------|------------------|
| 13.1 | Finger AR tags visible in the ultrawide at every gap (the printed fingers carry them), camera server running (11.4). | tags 0 and 1 detected at gap 0 and 0.08 |
| 13.2 | `python -m franka_deploy fdcal capture width --out wd --hand-endpoint tcp://localhost:5566 --gaps 0.0:0.08:0.0025` (for each gap: `hand_cmd move`, wait for `move_ok` (timeout = travel / speed + 1 s), save an ultrawide frame, log `hand_state.gap_m`). A 2.5 mm step keeps the slope's 1-sigma near 0.2 % with 0.3 mm tag noise. | 33 rows in `wd/sweep.json`; every move `move_ok` (the tool warns otherwise) |
| 13.3 | `python -m franka_deploy fdcal width --linear --data wd --deploy-config configs/deploy.yaml --profile real-iphone --out wd/fit.json` | **monotonic**; **residual p95 <= 1 mm**; the gaps that map to the training range 0.0447-0.0844 lie inside the sweep and inside 0-0.080. Prints `--set tooling.width_map.a=..` / `.b=..` and a provenance line: apply by hand |
| 13.4 | **Which `b`.** The fit gives raw tag separation = `a * gap + c`. `--anchor closed` (default) sets `b = train_closed_m` (the training value, not the sweep minimum), so the policy sees the training convention whatever `c` is. `--anchor raw` sets `b = c`. The tool prints `c - train_closed_m`. | if \|c - train_closed_m\| > 3 mm the printed fingers' tags are not the training gripper's: a finding. Decide anchor closed vs raw (**open**); record it |
| 13.5 | **Which eraser side the demos grip (A5).** The tool prints the gap the demo hold widths 0.046-0.062 map to (3-19 mm at a = 1, b = 0.0431, i.e. a 11-27 mm gripped side with the 8 mm pad offset). Watch a demo video (`results/block_raise/run_001_2026-10-05/videos/`, or the raw episode) and say whether the demos hold the **20 mm** side (eraser 60 x 20 x 12 mm). The old b 0.008 implied 38-54 mm, which matches no side. | the mapped gripped side matches the side the demos hold (20 mm: pad contact at gap 0.012) |
| 13.6 | Re-check with the new `a`, `b`: `hand.grasp_epsilon_inner_m` (0.015) and the hysteresis `close_below` 0.064 / `open_above` 0.070 (the gap at which they trigger moves with `a`, `b`): 12.8 and 12.11 again. | grasp holds the eraser; no chatter |
| 13.7 | **Commanded vs reached width** during 5 eraser grasps (as 6.5, through the width map). `task.width_proprio: commanded` is the upstream convention. | reached gap +- 2 mm of the eraser side while commanded is lower by the squeeze; if not, A/B `width_proprio: measured` vs `commanded` in 9.1 before scoring |

## 14. Arm sim-fidelity (checklist [§6](panda_py_checklist.md))

The sim implements panda-py's `CartesianImpedance` from its documented law; the source was not available offline.
**The sim's law is under-damped: 15-34 % overshoot on ideal 2 cm steps** (effective damping ratio 1/sqrt(Lambda), Lambda 4.6-8.3 kg;
panda-py's real law may differ, and real joint friction damps more). Everything the sim concludes about tracking and
overshoot rests on this.

| # | Item | Pass / log field |
|---|------|------------------|
| 14.1 | **Read the pinned panda-py `CartesianImpedance` source** (`panda_py/controllers/`). Record, against `franka_sim/simcore.py`: the task law (`J^T(-K e - D J dq)`), the damping definition (`D = 2 zeta sqrt(K)` elementwise vs mass-normalised), the orientation error, the null-space term, **pinv damping** (franka_example_controllers use lambda 0.2: then set `backend.mujoco.pinv_damping`), the target filter (`filter_coeff`), the torque rate limit (1 Nm/ms). | each item written down as same / different. Every difference is a sim parameter to set or a finding |
| 14.2 | **Step response at 600 / 800 / 1000 N/m:** `python -m franka_deploy gonogo run --out steps_<K> --motion steps --amp-m 0.02 --hold-s 10 --duration-s 60 --config <bridge.yaml with that stiffness>`, against the sim: `python -m franka_sim characterize --robots panda --stiffness 600,800,1000 --out sim_char.json` (baseline: `controlSW/franka_sim/franka_sim/configs/characterization_baseline.json`). gonogo prints lag and steady error; overshoot is `max(fk - cmd)` during the holds in `steps_<K>/trace.npz` (`t`, `cmd`, `fk`), as a percentage of the 2 cm. | record real vs sim for lag (`tracking.lag_ms`) and overshoot per stiffness. Suggested: real overshoot <= the sim's and lag within 30 ms; a real arm that is much better damped means the sim over-penalises tracking (the Stage 3 gate is 5 mm p95): adjust the sim before trusting it |
| 14.3 | **Payload:** the Desk / `setLoad` mass and CoM against the weighed tooling. A wrong payload shows as a z offset of about m g / K_z (3.7 mm per 0.3 kg at 800 N/m). | `O_F_ext` idle within +-3 N; z offset recorded |
| 14.4 | **Collision thresholds:** `robot.impedance.collision_thresholds` / `backend.panda_py.collision` equal what `set_collision_behavior` received (libfranka default tau [20,20,18,18,16,14,12] Nm, F [20,20,20,25,25,25] N). | values recorded in the station |
| 14.5 | **Desk end-effector setting** = `robot.desk_end_effector` (`none`: the bridge commands the flange; `franka_hand`: F_T_NE / NE_T_EE apply and `ee_is_tcp` / EE_T_TCP must follow). | one consistent setting (see 4.4) |
| 14.6 | **Phone / cradle vs links 5 and 6:** large q7 motions with the real housing (the sim reflexes on a q7 self-collision; MuJoCo itself does not collide the tooling with link 6). Check the real clearance with the **+10 cm fingers and the -y phone side** once printed (D9). | no contact through the demos' wrist range |
| 14.7 | **Approach speed and the contact reflex (A6):** a ~0.5 m/s approach strikes the table at about 40 N per finger and trips the 20 N reflex. This is real, not a threshold artefact (the sim exaggerates the overshoot). Keep v_max <= 0.25 m/s and do not raise the reflex threshold to hide it. Also: the sim's `slip` metric currently measures arm overshoot, so do not gate on slip in sim until true in-hand slip lands (A7 adds TCP z). | a table touch at 0.25 m/s does not reflex; record the peak contact force |

## 15. After hardware day: which key gets which value, and the re-runs

| Measured in | Station key (apply by hand with the printed `--set` line and provenance) |
|-------------|-----------------------------------------------------------------------|
| 11.5 | `sensor.clock.latency_s.main` / `.ultrawide` (iPhone), `sensor.task.camera_latency_s` |
| 11.9 | `sensor.clock.residual_latency_s.*` (rig) |
| 11.6 | `sensor.streams.<s>.K` (only if you decide to record the deployment K) |
| 12.7, 12.8 | `tooling.hand.force_n`, `tooling.hand.grasp_epsilon_inner_m` |
| 13.3 / 13.4 | `tooling.width_map.a`, `tooling.width_map.b` |
| 14.1 / 14.2 | `backend.mujoco.pinv_damping`, damping of the sim, `robot.impedance.*` |
| 14.3-14.5 | `robot.desk_end_effector`, `robot.impedance.collision_thresholds`, `robot.fci.libfranka_version` |
| D9 | `tooling.finger_adapter.extension_m`, `tooling.tcp_x_along_hand` |
| 4.3 | `tooling.F_T_C_main`, `tooling.F_T_TCP` |
| 5.1 | `tooling.F_T_TCP` |

Then re-run the sim stages with the measured values, so the pass/fail it reports matches the hardware (outputs go outside the repo):

```
python -m franka_sim eval --profile sim-iphone-fake --stage 1 --out $S/after_hw/stage1
python -m franka_sim eval --profile sim-iphone --stage 2b --demos 0,1 --out $S/after_hw/stage2b
python -m franka_sim eval --profile sim-iphone --stage 3 --out $S/after_hw/stage3
```

For the sim re-run, enter the measured camera mount as `tooling.F_T_C_main` (status `measured`; the sim then uses F_T_C_main @ C_T_TCP). Set `tooling.F_T_TCP` (status `measured`) only if it agrees with it (and with `extension_m` / `tcp_x_along_hand`) within 1 mm / 0.1 deg, or the sim run refuses to start (preflight `sim_tcp`). Leave `tooling.F_T_TCP` null in sim-only work: the launcher derives it.

- Stage 1: kinematic feasibility with the real tooling geometry (D9), including the rig housing (30 mm thick) if the rig path is used.
- Stage 2b: latency and stamps with the measured `clock.latency_s` (the gate `cam_stamp_err_ms_p95_max`, and the open live-vs-offline gates).
- Stage 3: the closed loop with the measured width map, epsilons and arm dynamics.
- Station values come from `controlSW/configs/station/`; use `--set host.work_root=<dir>` on a disk with space (`pc_laptop` points at `/home`, which is about 99 % full).

---

## Every hardware-only item from D0–D6 (not executed anywhere yet)

| From | Item | Where above |
|------|------|-------------|
| D0 | Every Franka, gripper and Jetson item. Clean Jetson envs (no `~/.local`, no ROS `PYTHONPATH`). | 0.2–0.10 |
| D1 | Model latency on the Orin (cold/warm, fp32/bf16). imagecodecs 2023.9.18 installs on aarch64. Real-rig viewpoint/photometric match. | 9.3, 0.3, 9.4 |
| D2 | Commanded vs reached gripper width (proprio risk; width_proprio commanded vs measured). 30 vs 60 fps main sensor mode, 33/67 ms spacing (lens decision). Continuous mode only if capture → reply < 0.42 s. Real spacemouse. | 6.3, 6.5, 7.3, 9.3, 0.10 |
| D3 | `PandaPyBackend` names vs the pinned panda-py. Versions pinned (B0). v_max 0.25 → 0.5 after B3-hw. Bridge hop latency on the Jetson. | 0.4–0.5, 9.5, 7.1 |
| D4 | Jetson profile: pinned versions, calibrated rig.yaml, Feetech driver, real spacemouse. Video disk budget (~4.3 GB/h). Home pose. Settle gate on real dynamics. | 0.4–0.10, 7.2, 7.4, 9.1b |
| D5 | The laptop showed one `leaky` clamp from bridge request stalls > 10 ms slack. Micro-clamp rule (> 1 % of ticks → slack 20 ms). Anchor measured vs target sweep. Evaluation protocol. | 1.2b, 8.2, 9.1c, 10 |
| Sim evaluator | iPhone latency 0.08 s, deployment K vs training K, Franka Hand semantics (checklist §7), width map a/b, panda-py impedance law (under-damped in the sim), finger extension +10 cm / phone -y, v_max and success rule, 2b and obs_motion gates | 11, 12, 13, 14, decisions block |
| D6 | RT setup (PREEMPT_RT, isolcpus, IRQ pinning, nvpmodel/clocks, limits, static IP, torch wheel). libfranka/panda-py source build. 30-min go/no-go. Hand-eye, pivot and width capture on real data. Open-loop replay on the arm. | 0.2–0.4, 1, 4–6, 8 |
