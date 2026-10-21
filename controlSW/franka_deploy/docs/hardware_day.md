# Hardware day: ordered procedure

Do these steps in order. Each one has a pass number and names the log field that proves it. Don't start a step until
the one before it has passed. All of this was built and tested on the laptop with fakes. **None of it has run on
the Jetson, the arm, or the gripper.**

Conventions:

- `FD=controlSW/franka_deploy`.
- Commands run on the Jetson from the repo root, with `PYTHONPATH=$FD` and the deploy venv
  (`scripts/jetson_deploy_setup.sh`).
- `<S>` is a session directory under `work_root`.
- Every number written as "pass:" is the acceptance threshold.
- Keep a paper log. Note each step's result and the path of its JSON next to it.

People and safety, throughout:

- Two people. One holds the external activation device (user stop) at all times. The other types.
- The zone is clear. Speeds start at the values below and are raised only where this document says so.
- **Abort rule** (the whole day, and the evaluation): after **2 safety events**, stop. A safety event is a reflex,
  a user-stop press, an unexpected contact, or a watchdog stop. Debug before continuing.

---

## 0. Before power-on (desk work)

| # | Item | Pass |
|---|------|------|
| 0.1 | Robot model and system version (Desk → Settings → System). FCI enabled. | recorded on paper |
| 0.2 | `scripts/jetson_deploy_setup.sh --franka-if <iface>`. Run `--dry-run` first, then for real, then reboot. | script ends with "done" |
| 0.3 | `python -m franka_deploy.tools.jetson_verify --json verify.json` | **12/12 PASS**: `preempt_rt`, `cmdline`, `nic_irq`, `nvpmodel`, `limits`, `static_ip`, `robot_ping`, `camera_0`, `camera_1`, `cuda_torch`, `imagecodecs` (2023.9.18), `l4t` |
| 0.4 | `scripts/build_libfranka_pandapy.sh --robot panda\|fr3`. Panda 4.x → 0.9.x; FR3 → ≥ 0.10 (check the compatibility table for the exact system version). Copy the two printed lines into `configs/bridge.yaml`. | `<prefix>/franka_versions.json` written; `import panda_py` works |
| 0.5 | Re-check the panda-py names used by `PandaPyBackend` against the pinned version (list in `docs/panda_py_checklist.md` §0). | every name exists |
| 0.6 | Implement `FeetechGripper` (or `DynamixelGripper`) `_write_position/_read_position/_write_current_limit` in `gripper/driver.py`. It is a `NotImplementedError` stub today. | `make_gripper({"type": "feetech", ...})` opens and reads a position |
| 0.7 | Rig calibrated (`sensor_rig/docs/calibration.md`). Copy `rig_calibrated.yaml` to `camera.rig_config_path`. | `check-config` row `rig_not_default` OK |
| 0.8 | Deploy envs built clean on the Jetson, without relying on `~/.local` or a ROS `PYTHONPATH` (D0 G3). Copy the checkpoint and prompt zarr to the paths in `deploy.yaml`; set `prompt_zarr_sha256`. | `python -m franka_deploy check-config --config deploy.yaml --profile jetson`: **all OK** (including `panda_py_versions_pinned`) |
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
| 4.3 | `python -m franka_deploy fdcal handeye --data he --cad-F_T_C cad.json --out he/result.json` | **`cad_agreement.accept`: ≤ 3 mm / 1°** (`cad_agreement.mm`, `.deg`); `tsai_vs_park_mm_deg` ≤ [1 mm, 0.3°]. Copy `F_T_TCP` into `franka_cartesian.yaml` |
| 4.4 | Either `ee_is_tcp: false` (the default: the bridge commands the flange), or set the EE in Desk / `NE_T_EE` with `ee_is_tcp: true`. Never both. | one of the two |

## 5. Pivot test

| # | Command | Pass / log field |
|---|---------|------------------|
| 5.1 | Fingertip centre on a marked point, bridge `start`. `python -m franka_deploy fdcal capture pivot --out piv --F_T_TCP configs/franka_cartesian.yaml --angle 20` (drives ±20° about TCP x, y, z at ≤ 10°/s and ≤ 2 cm/s, records each extreme). | **`piv/pivot.json`: `tcp_excursion_mm ≤ 2` and `F_T_TCP_translation_error_mm ≤ 2`** (`accept: true`). Also look: the fingertip stays on the mark |

## 6. Gripper width map

| # | Command | Pass / log field |
|---|---------|------------------|
| 6.1 | Finger AR tags visible in the ultrawide. `python -m franka_deploy fdcal capture width --out wd --controller-config configs/franka_cartesian.yaml --positions <closed>:<open>:<step>` (closed first, ≥ 10 positions). | `wd/sweep.json` |
| 6.2 | `python -m franka_deploy fdcal width --data wd --out wd/calib.yaml` (iPhUMI's tag detector and width definition). Copy `calibration` into `franka_cartesian.yaml`. | **`checks.zero_at_closed`, `checks.monotonic`, `checks.covers_required` all true** |
| 6.3 | Current limit and squeeze δ on the eraser. Log commanded vs reached width (D2 risk: the history stays "open" if a current-limited squeeze never reaches the command). | eraser held without slipping; the reached width is logged |
| 6.4 | Hold during a grasp: send `hold`. | commanded width unchanged; the object doesn't drop |
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
| D6 | RT setup (PREEMPT_RT, isolcpus, IRQ pinning, nvpmodel/clocks, limits, static IP, torch wheel). libfranka/panda-py source build. 30-min go/no-go. Hand-eye, pivot and width capture on real data. Open-loop replay on the arm. | 0.2–0.4, 1, 4–6, 8 |
