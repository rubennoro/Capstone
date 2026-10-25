# Roadmap: from the Jetson sensor rig to a policy running on the Franka

Replacing the iPhone on the iPhUMI gripper with separate parts: **2 CSI cameras + 1 IMU, all on one NVIDIA Jetson
Orin Nano**. Two pieces:

- **A.** Sensors → full episode recording (the iPhone's way) → training data → trained policy.
- **B.** Trained policy → live observations in the exact training format → actions → Franka arm + gripper.

Detailed sources (in `sensor_rig/docs/planning/`): [SPEC_sensors.md](sensor_rig/docs/planning/SPEC_sensors.md)
(37 requirements, iPhone reference, compliance matrix), [PLAN_A_data_to_policy.md](sensor_rig/docs/planning/PLAN_A_data_to_policy.md),
[PLAN_B_deployment.md](sensor_rig/docs/planning/PLAN_B_deployment.md), research in `research/`. Code:
[sensor_rig/README.md](sensor_rig/README.md).

**Principle:** the software is spec-agnostic (all sensor specifics live in `rig.yaml`). The parts below are what
we bring up first, and each stage must meet the iPhone-derived requirements in the spec.

---

## Status (2026-10-17): both pieces are done in software; every next step needs hardware

Built step by step, each step independently re-verified by a reviewer, with the regression guard green after every
step.

**Piece A, data collection:** `sensor_rig/`, 400 tests.

| Where | Command | What it does |
|---|---|---|
| Jetson | `sensor_rig/scripts/jetson_setup.sh` | install |
| Jetson | `sensor-rig doctor` | preflight |
| Jetson | `sensor-rig collect --rig R --out RAW --session rig<name>` | the one command: checked gripper calibration, then demos with Enter start/stop |
| Jetson | `sensor-rig stamp-offset` | LED camera-timing test |
| laptop | `sensor-rig process-session` | raw session → iPhone-format export, prints the `run_pipeline.sh` line |

- Procedure: [sensor_rig/docs/first_plugin.md](sensor_rig/docs/first_plugin.md). Every command in it is tested against mocks.
- Dry run: collect → pose → export → real iPhUMI processing matches the iPhone zarr (TCP 6e-4, width 0.2 mm), in
  30 fps fill mode.

**Piece B, deployment:** `controlSW/franka_deploy/`, ~190 tests.

| Command | What it does |
|---|---|
| `python -m franka_deploy run --config deploy.yaml [--mock]` | the one command: preflight, starts bridge → controller → RigCamera → inference → task, supervised, hold-first teardown |
| `annotate`, `eval`, `offline-eval`, `reinfer` | evaluation |

- Verified with the real run_001 model on mock hardware:
  - it reproduces training's val MSE (1.65e-4 vs 1.48e-4);
  - deployment observations equal the dataset path (Δaction < 1e-4);
  - the safety filter, watchdog, reflex latch, stop path ≤ 12 ms and every kill / Ctrl+C case end in a hold.
- Procedure: [controlSW/franka_deploy/docs/hardware_day.md](controlSW/franka_deploy/docs/hardware_day.md).

**Sim evaluator (branch `sim-eval`, local commits, not pushed):** a MuJoCo Panda/FR3 + Franka Hand model, the iPhone and
rig wireless sensor path (Wi-Fi emulator, receiver, clock sync) and a configurable station (`controlSW/configs/`) so a
policy can be evaluated before hardware. Resume point and phase list: [controlSW/docs/sim_plan/STATUS.md](controlSW/docs/sim_plan/STATUS.md).

| Stage | What it checks | State |
|---|---|---|
| 1 | offline kinematic feasibility of the 45 demos | built, run: **+10 cm fingers + phone on -y gives 43/45 on Panda and FR3**; stock length hits links 5/6 (iPhone housing only; not yet re-run with the 30 mm-thick rig housing) |
| 1b | full-stack demo replay with physics | not built (A8) |
| 2a / 2b | chunk reachability; live-vs-offline stamps and prediction error | built; the real-policy 2b gap (p95 10.8 mm iPhone, 29.8 mm rig) is expected from JPEG and latency, gates are placeholders |
| 3 | closed loop with a judge, sweeps with resume | built on the stub simulator; MuJoCo closed loop depends on the scene/render phases |
| hardware-day tools | `wireless-latency` (QR clock), `fdcal stream-intrinsics`, `fdcal width --linear` | built and self-tested; never run on a device |

What is still **placeholder** (each has a hardware-day step, [hardware_day.md](controlSW/franka_deploy/docs/hardware_day.md) §11-15):

- iPhone capture latency 0.08 s (`sensor.clock.latency_s`), rig `residual_latency_s` 0;
- `width_map.a` = 1.0 (critical); `b` = 0.0431 is derived;
- the arm's impedance law: the sim implements the documented `CartesianImpedance` and is **under-damped (15-34 % overshoot)**, unverified against panda-py;
- Franka Hand semantics (GIL, `stop()`, grasp without an object, force per finger), `grasp_epsilon_inner_m` 0.015;
- `--match-camera-latency` is off (user decision): obs_motion 5-9 mm at demo speeds;
- limits use the minimum rule (station `criteria.limits`, deploy `task.v_max`, bridge safety).

**New decisions D9 and D10 (TODO.md):** confirm printed fingers about 10 cm longer than stock with the phone on the -y side (before printing).

**To-do list:** [TODO.md](TODO.md). **What to buy:** [PURCHASING.md](PURCHASING.md).

**Decisions needed from the user**

1. Panda or FR3 (pins libfranka/panda-py).
2. Gripper actuator: motorized iPhUMI gripper (Feetech or Dynamixel) **or** the Franka Hand with printed fingers (what the sim evaluates; PURCHASING §2b).
3. Lens purchase: 2.1 mm main lens for the B0180. Ultrawide decided 2026-10-19: Arducam B0179 (fixed 175° fisheye, IR-cut version).
4. Where the 1 kHz loop runs: Jetson first, gated by the go/no-go test; RT PC as fallback.
5. v_max for the reported evaluation, plus the success rule and prompt indices; the open sim gates (Stage 2b, obs_motion).
6. **D9:** printed fingers +10 cm and the phone on the -y side (Stage 1 result: 43/45; but 37/45 demos leave the 60 degree cone and joint speeds are within limits for only 19/43 FR3 and 36/43 Panda), if the Franka Hand path is chosen.
7. **D10:** the open sim gates (Stage 2b live-vs-offline p50/p95/n_pairs, Stage 3 obs_motion) and the laptop work_root on the full /home.
8. Commit `sensor_rig/` changes, `controlSW/franka_deploy/`, `scripts/regression/`, ROADMAP.md.
9. A machine with ≥ 20 GB free to build the ORB-SLAM3 image.

**Top hardware-day risks**

- **Data collection:**
  1. ORB-SLAM3 on real ultrawide footage (never run).
  2. Camera timestamp quality (run `stamp-offset`; argus_sof helper is unbuilt).
  3. Two-camera, IMU and JPEG bring-up on the Orin Nano.
- **Deployment:**
  1. The rig-vs-iPhone perception gap (fallback: rig data + fine-tune).
  2. RT + libfranka + GPU on one Orin Nano.
  3. Gripper commanded vs reached width.
  4. TCP / hand-eye error.

---

## 0. Ground rule: the iPhone pipeline keeps working exactly as before

The iPhone path (app → `data/raw` → iPhUMI processing → `process_dataset.sbatch` → training; iPhone peripheral →
real-env → inference) is the reference and must not change behaviour.

1. **Additive only.** Rig work lives in new files: `sensor_rig/`, new real-env files (`rig_camera.py`, `rig_wrist_*.yaml`),
   and `controlSW/franka_deploy/` (Franka controller and task as subclasses; no edits to existing tasks/controllers).
   So far the only touches to shared files are an appended `"JetsonRig1,1"` entry in `gripper_calibration.yaml`
   (existing entries byte-identical) and one new entry-point line in real-env `pyproject.toml`.
2. **Shared-code fixes are opt-in.** Any change to shared real-env / behavior_prompting / iPhUMI code must keep the
   current behaviour as the default (new parameter with the old value as default, new subclass, or new config), unless
   you explicitly approve a behaviour change. Pending B2 items:

   | Change | Effect on iPhone path |
   |---|---|
   | `inference_timeout_s` parameter (default 5.0) | none (default = today) |
   | `task_name` attribute in `BaseTask` | none (additive) |
   | action replay: single-arm support, `--sides auto`, `--action-offset` default 1 | none (default = today) |
   | controller error topic `"errors"` → `"error"` (N1) | **behaviour change**: iPhone/ARX5/UR5 deployments would start seeing controller errors they currently miss. Needs your approval. |
   | camera-client read order (E10) | deferred: touches iPhone deployment timing |

3. **Separate names on the cluster.** Rig sessions/datasets never reuse iPhone names (e.g. session `rig1016`,
   dataset `rig-v1-single`, sim `simrigthirty`), so iPhUMI's shared `tmp_demonstrations/` cache and
   `~/CapstoneProject/datasets/` can't mix them. Rig runs are labelled in their results README.
4. **Regression guard** ([scripts/regression/](scripts/regression/README.md)): run before every commit that touches
   shared code and before pulling on the cluster:
   - `check_protected.sh`: iPhone-path files unchanged vs the pre-rig baseline `0ddedef` (calibration yaml: append-only).
   - `iphone_pipeline_check.py`: the real iPhUMI processing on 5 fixed `firstreal` episodes matches a golden made at
     the baseline commit.
   - `test_iphone_deploy_unchanged.py`: iPhone peripheral configs compose and import as at baseline.

   **Status (2026-10-16): all three pass.**
   - Protected files: 0 violations vs `0ddedef`, 5 allowed additions.
   - Pipeline: all 18 zarr arrays are identical to the golden, including every image frame (2965 main, 495
     ultrawide); about 45 s.
   - Deploy check: 11/11.

   So, as of today, the iPhone path is provably unchanged by the rig work.

---

## 1. Hardware and specs (fixed setup)

| Component | Part (recommended) | Key specs | vs iPhone 16 Pro |
|---|---|---|---|
| Compute | Jetson Orin Nano Super dev kit, JetPack 6, NVMe | 6-core A78AE, 8 GB shared, CUDA; no HW video encoder | records only (pose offline); runs inference at deployment |
| Main camera | Arducam B0180 (IMX219, 4:3, M12) + **2.1 mm low-distortion lens** | 1640×1232 full-FOV @ 30 fps, rolling shutter (~23 ms readout), 82°×67° | iPhone: 1920×1440 @ 60 fps, 68.6°×54.2°. Coverage 1.000 incl. mount tolerances; 938 px/rad ≥ 703 needed |
| Ultrawide | **Arducam B0179** (IMX219, fixed 175° diagonal fisheye, f ≈ 1.50 mm) | 1640×1232 @ 30 fps → exported at 10 Hz, ~140°×105° | iPhone: 640×480 @ 10 Hz, 102°×85.6°. Coverage 1.000; tags ≥ 20 px |
| IMU | ICM-42688-P breakout on SPI | 1 kHz, gyro 0.0028 °/s/√Hz, accel ~70 µg/√Hz, 1 µs FIFO stamps | ~3× better than the requirement; bias stability needs an Allan run |
| Pose | ORB-SLAM3 mono-inertial (UMI fork), offline on laptop; OpenVINS fallback | trajectory at camera rate | iPhone: ARKit 60 Hz, jitter ≤ 0.22 mm / 0.055° (measured) |
| Cables | 15→22-pin CSI, ≤ 30 cm (main); 22→22-pin included with the B0179 | | |

Avoid: a main lens > 2.3 mm (coverage 0.955 at 2.5 mm), an ultrawide > 1.75 mm, the 1280×720@60 mode (crops the
view), Camera Module 3 / IMX708 on Jetson (fragile drivers).

**Key requirements** (IDs and derivations in the spec): main ≥ 30 fps (60 target); ultrawide ≥ 10 Hz; image↔pose bias
≤ 3 ms, stamp jitter ≤ 1 ms rms, camera↔IMU residual ≤ 1 ms; coverage ≥ 0.99; exposure ≤ 4 ms locked; IMU ≥ 200 Hz;
pose jitter ≤ 0.5 mm / 0.1°, error over the 0.8 s action horizon ≤ 2 % + 1 mm and 0.5°; deployment latency
≤ 100 ms (camera), stamps ±5 ms.

**Gaps (ranked):**
- **G1 (decision):** full-FOV is 30 fps vs the 60 Hz grid. Recommended: accept (nearest-frame fill, ≤ ±16.7 ms ≈ 7.7 mm
  at p99 hand speed; fine for rig-only training).
- **G2 (software, planned):** Argus appsink stamps are arrival times, which can't meet the timing requirements. Write a
  C++ libargus helper with sensor start-of-frame stamps, then verify with the LED test + Kalibr.
- **G3:** no hardware sync between cameras (accepted; per-frame stamps).
- **G4:** rolling shutter (~0.9° skew at p99 rotation; global-shutter camera is the fallback).
- **G5 (biggest unknown):** ORB-SLAM3 accuracy on this rig. Acceptance test: rig on the Franka flange vs kinematics.

---

## 2. Piece A: sensors → training data → trained policy

### How the frequencies line up (verified in code)

```
sensors      main 30 fps · ultrawide 30 fps · IMU 1 kHz        (one clock: CLOCK_MONOTONIC ns)
raw episode  JPEG frames + per-frame stamps · imu0/data.csv
pose (VIO)   ORB-SLAM3 on ultrawide + IMU → trajectory          (offline)
export       60 Hz grid (iPhUMI hardcodes 1/60); main = nearest frame (fill mode);
             ultrawide decimated to 10 Hz, slotted; pose interpolated at grid times
iPhUMI       align on 60 Hz; gripper width from ultrawide tags (~10 Hz → 60 Hz); image = latest past frame
zarr         main + low-dim at 60 Hz, ultrawide at 10 Hz (+ index maps)
training     policy runs at 20 Hz: 2 main frames 50 ms apart, 2 ultrawide frames 100 ms apart, 2 proprio steps;
             predicts 16 actions × 50 ms (0.75 s) relative to the latest pose; prompt = one demo
```

Action labels come only from pose (VIO) and gripper width (ultrawide tags); main-camera timing affects only
observations. At 30 fps, the two observation frames are 33 or 67 ms apart instead of 50 ms, which is acceptable for
rig-only training. Deployment samples frames the same way (RigCamera publishes the latest frame at 60 Hz).

### Done
- `sensor_rig`: Argus + ICM-42688-P drivers, mock drivers, recorder with live abort rules, raw format, pose-prep for
  ORB-SLAM3/OpenVINS, pose-import with gravity alignment and checks, iPhone-format export + validator, calibration
  tools (Kalibr import, coverage, alignment, TCP), simulator.
- Verified in simulation: real iPhone demos → simulated rig recordings → export → real iPhUMI pipeline → zarr
  matches the iPhone zarr (width median 0.1 mm, relative motion 1e-5, image MAE ~1).

### Remaining (🖥 software now · 🔧 needs hardware · ☁ cluster)

| WP | What | Type | Done when |
|---|---|---|---|
| A0 | Housekeeping: DESIGN.md entry for `sensor_rig/`, pull on the cluster | 🖥 | cluster can run `python -m sensor_rig` |
| A1 | Fill-mode hardening: e2e in 30 fps fill mode, tighter drop/freeze aborts, VIO jump detection, lock AE/AWB in configs | 🖥 | e2e passes in fill mode; image↔pose lag zero-mean within ±17 ms |
| A2 | Sim-rig 30 fps dataset from all 45 `firstreal` demos → `process_dataset.sbatch` (first real run of it) → train vs a 60-epoch iPhone baseline | ☁🖥 | same valid-demo count; val MSE within ~20 % |
| A3 | C++ libargus timestamp helper (G2) behind the same camera interface | 🖥 then 🔧 | stamps meet ≤ 3 ms bias / ≤ 1 ms jitter in the LED test |
| A4 | ORB-SLAM3 (UMI fork) in Docker on the laptop: `pose-prep --target umi`, finger mask, map → localize, import | 🖥 then 🔧 | plumbing passes on sim; pilot ≥ 90 % demos tracked |
| A5 | Parts + mount (lenses at the iPhone lens positions) | 🔧 | `calib-check --live` coverage ≥ 0.99, both tags visible |
| A6 | Jetson bring-up (checklist): fps, drops, IMU, LED stamp test, thermal | 🔧 | 0 drops, fps ± 1 %, offsets ≤ 2–3 ms |
| A7 | Calibration: Kalibr cameras + camera-IMU, Allan, calib-align, TCP → `rig_v1.yaml`, `gripper_calibration.yaml` entry | 🔧 | calib-check acceptance |
| A8 | First real data: gripper calibration + demos → VIO → export → processing | 🔧☁ | ≥ 90 % demos pass align_validate; 10-D actions; no black borders |
| A9 | Validation: rig on Franka flange vs kinematics; width vs iPhone calibration; dataset stats | 🔧 | spec pose/timing targets met or documented |
| A10 | Training on rig data: smoke (10–20 demos), then ~60 epochs at ≥ 45 demos | ☁ | val MSE same order as run_001 → checkpoint to piece B |

**Critical path:** A5 → A6 → A7 (Kalibr) → A4 pilot → A8 → A10. Earliest real-rig training ≈ 3–5 weeks after parts
arrive. A1–A4 (software) can be done before then.

---

## 3. Piece B: trained policy → Franka

### Observation → policy → action contract (run_001)

- **In:** `camera_right_main_rgb` and `camera_right_ultrawide_rgb`, each 2×3×224×224 (RigCamera 960×720 → pad to square
  → resize 224; main at T and T−50 ms, ultrawide at T and T−100 ms). Proprio = the commanded-action history (pos3 +
  rot6d relative to the newest, 50 ms apart; gripper width in metres, training range 0.045–0.084). Prompt = demo
  `prompt_index` of task `block_raise`, re-sent at every episode start.
- **Out:** 16 steps × 10-D (pos3 + rot6d relative to the TCP pose at observation time, + absolute width), 50 ms apart.
  `PolicyAgent` anchors on measured FK at the training TCP and stamps step k at T + k·50 ms. 8 steps executed per chunk.
- **To motors:** task → `FrankaCartesian` controller (200 Hz target interpolation) → bridge process (panda-py Cartesian
  impedance, 1 kHz libfranka loop) → Franka; gripper via its own driver (motorized iPhUMI gripper recommended so images
  match training).
- **Latency (estimated, Jetson):** 0.20–0.40 s bf16, 0.35–0.70 s fp32. Continuous motion needs < ~0.42 s, otherwise
  stop-and-go.
- **1 kHz loop placement:** try the Jetson first (PREEMPT_RT, isolated core, direct Ethernet) with a 1–2 day go/no-go
  under full load; if it fails, move only the bridge to a small x86 RT PC (~$150–250). Same code either way.

### Done
- `RigCamera` peripheral (drop-in for the iPhone wrist camera, same remap as export, latency applied once, libx264
  instead of NVENC), with mock mode and equivalence tests (deploy vs export images MAE < 0.1).
- Existing: inference server, `PolicyAgent`, real-env tasks, ARX5/UR5 controllers as templates.

### Remaining

| WP | What | Type | Done when |
|---|---|---|---|
| B0 | Decisions + procurement: gripper actuator, Panda vs FR3 (libfranka version), RT kernel plan | — | decided |
| B1 | Laptop harness: mock controller + mock spacemouse + RigCamera replay → real inference server + PolicyAgent | 🖥 | contract + timing checks pass; actions sane vs the demo |
| B2 | Upstream fixes (opt-in, see §0) | 🖥 | applied after approval; regression guard green |
| B3 | `controlSW/franka_deploy/`: bridge (panda-py, fake backend, safety limits), `FrankaCartesian`, `iPhumiFrankaTask`, configs; Jetson RT setup | 🖥 then 🔧 | B1 passes with it; go/no-go: 0 reflexes in 30 min under load |
| B4 | TCP calibration (CAD + hand-eye with ChArUco), pivot test, gripper width map, open-loop demo replay | 🔧 | replay ≤ 1 cm / 3° over 5 s |
| B5 | Closed loop: pause mode first, then continuous | 🔧 | ≥ 7/10 successes at the canonical spot |
| B6 | Performance on the Jetson: fp32 vs bf16, DDIM 16 vs 8, memory (8 GB), CPU placement | 🖥 then 🔧 | latency/memory table, pause vs continuous decision |

**Critical path:** B0 → B3-hw (RT go/no-go → bridge on the arm) → B4 → B5. B1, B2, B3-sw and the B6 method are
software-only (~2 weeks for one person).

---

## 4. Decisions for you

1. Approve the B2 upstream fixes, in particular the behaviour-changing error-topic fix (N1).
2. G1: accept 30 fps fill mode (recommended) vs a custom ~40 fps mode / different sensor.
3. Rig-only training (recommended first) vs mixing with iPhone data (only after A9 confirms width scale and views).
4. ORB-SLAM3 UMI fork (recommended) vs stock with OpenVINS backup.
5. Gripper actuator (motorized iPhUMI gripper recommended) and Panda vs FR3.
6. Approve the A2 cluster jobs (sim-rig dataset + 60-epoch iPhone baseline).

## 5. What can start now (no hardware)

A1 → A2 (cluster), A3 (code), A4 (Docker + plumbing), B2 (after approval) → B1 → B3-sw, B6 method on the laptop.
