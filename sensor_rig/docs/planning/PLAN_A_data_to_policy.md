# PLAN A: 3 sensors on a Jetson → iPhone-equivalent episodes → zarr → training → trained policy

**Hardware (fixed by the lead):**
- One NVIDIA **Jetson Orin Nano dev kit** (JetPack 6) is the only computer on the rig.
- Two off-the-shelf **IMX219** CSI modules (4:3) with M12 lenses, one on each 22-pin CSI port:
  - "main": a ≈ 90° diagonal-class lens, pinhole ≥ ~70×55°, which covers the virtual main (68.6×54.2°);
  - "ultrawide": a ≈ 160° fisheye (coverage 1.0).
- One **ICM-42688-P** IMU on SPI.

**Constraints:**
- The two cameras are **not hardware-synced**.
- The IMX219 full-FOV 4:3 mode is **1640×1232 @ 30 fps**. 60 fps is only available in a cropped 16:9 1280×720 mode.
- There is **no hardware video encoder**: JPEG is encoded on the CPU.
- Pose is computed **offline** (ORB-SLAM3 mono-inertial on UW + IMU), not on the Jetson.
- Backend: `SR/drivers/argus_camera.py` + `SR/configs/rig_default.yaml`. The Raspberry Pi / Picamera2 path is
  ignored except for footnote ¹.

Scope is piece A only: recording, pose, export, processing, training. The deployment peripheral and robot control
are piece B. Everything was checked against the working tree on 2026-10-16, which includes the uncommitted
`sensor_rig/`. Where R2's `align.py` line numbers differ, the numbers here are current.

Path abbreviations:
- `SR/` = `sensor_rig/sensor_rig/`
- `DP/` = `iPhUMI/python_package/iphumi/demonstration_processing/`
- `BP/` = `behavior_prompting/behavior_prompting/train_network/`
- `RC` = `results/block_raise/run_001_2026-10-05/run_config/config.yaml`

**[measured]** marks numbers produced for this plan:
- iPhone side: real `firstreal` demos processed by the real pipeline, in `scratchpad/e2e/lead_run2/pipe_A`.
- Rig side: a simulation that calls the real export functions (`build_grid`, `iso_ms_array`, `select_uw`,
  `slot_uw`) and then applies iPhUMI's `align` indexing rule. Script: `scratchpad/lagcheck.py` plus inline runs.

---

## 0. TL;DR

- **The policy learns at 20 Hz:**
  - Observations: 2 main images and 2 proprio samples 50 ms apart, plus 2 UW frames 100 ms apart.
  - Actions: 16 targets 50 ms apart (0.75 s), relative to the latest observed pose.
  - Prompts: one full demo of the same task, 1 token per second.
  - Everything is cut from a **60 Hz** grid fixed by iPhUMI (`align.fps`, `detect.py:69`).
- **Action labels come only from pose (VIO on the UW) and gripper width (ArUco on the UW).** Main-camera timing
  affects observations only. So the hard requirements are pose accuracy and timestamp accuracy (camera ↔ IMU, main ↔
  UW). Frame rate matters less.
- **On this hardware the main camera is 30 fps, so export always runs in fill mode.** Each main frame appears in two
  60 Hz slots, and that is sufficient for training:
  - The two observation frames are always distinct ([measured]: 0 % identical).
  - Their spacing is 33/67 ms (50/50 mix) instead of the iPhone's constant 50 ms.
  - Image-vs-pose error is −17…+33 ms (mean +10) with today's code, −17…+17 after fix F2. The iPhone has 0…+17 ms.
  - Fine for **rig-only** training if deployment samples frames the same way. A real domain difference if mixed with
    iPhone data.
  - **Do not use the 1280×720@60 mode.** It crops the field of view (UW coverage 0.3–0.6, `docs/calibration.md:33-34`)
    and is 16:9.
- **No hardware camera sync is needed.** Each stream is placed by its own timestamp. What matters is that each stamp
  is close to mid-exposure.
  - On the Jetson the stamp is the GStreamer appsink buffer PTS, i.e. *when nvarguscamerasrc received the buffer*,
    not the sensor start of frame (`SR/drivers/argus_logic.py:11-12`).
  - The −30 ms prior is a placeholder (`rig_default.yaml:31,50`).
  - **Measuring and stabilizing that offset is the main timing risk.**
- **Real gaps found (§1.5):**
  1. Fill mode was **never run through the real iPhUMI pipeline** (D10c ran in native mode).
  2. A fill-mode alignment bias (F2).
  3. **No finger mask** for ORB-SLAM3.
  4. **No mapping-video / localization flow.**
  5. **No VIO jump or reset check.**
  6. Fill-mode drop tolerance is too loose (a ~0.45 s freeze passes).
  7. **Exposure and white balance are not locked** in `rig_default.yaml`.
- **Critical path:**
  lenses + mount → Jetson bring-up (Argus both cameras + SPI IMU, stamp-offset LED test) → Kalibr camera-IMU →
  **ORB-SLAM3 (UMI fork) working on real UW footage, with mask and map** → export → `process_dataset.sbatch` → smoke
  train.
- **Two milestones need no hardware and can start now:**
  - Train on *simulated* 30 fps rig data made from all 45 `firstreal` demos. This tests the rig path,
    `process_dataset.sbatch` and training on the cluster, and measures the 30 fps effect on identical demos.
  - ORB-SLAM3 in Docker on the laptop, plus the new pose-prep target.
- **Earliest real-rig training:** a smoke run as soon as 10–20 demos, one gripper calibration and working VIO exist.
  TCP, `calib-align` and Allan variance are not required, because raw episodes re-export later with
  `--rig rig_v1.yaml`.

---

## 1. Frequency and timing chain, end to end

### 1.1 The chain (verified in code)

| # | Stage | iPhone | Jetson rig (`rig_default.yaml`, Argus) | Code |
|---|---|---|---|---|
| 1 | Main camera native rate | ARKit ≈ 60 Hz VFR ([measured] ms-string dt 16/17 ms). App aborts below 31 fps | **IMX219 mode 3, 1640×1232 @ 30 fps** (2×2 binned, full FOV). 1280×720@60 is cropped and not usable | `rig_default.yaml:15-21`; `docs/calibration.md:31-34`; app rule R2 §2 (VC:306,1818-1843) |
| 2 | UW native rate | ≈ 10 Hz (every 6th ARFrame; 166 UW for 996 main) | **IMX219 1640×1232 @ 30 fps**, ≈ 160° fisheye | `rig_default.yaml:34-39` |
| 3 | Camera sync | one sensor pipeline, one timestamp per ARFrame | **None.** Two free-running sensors, each stamped independently | R4 §1.6 |
| 4 | Camera timestamp | ARFrame time | `base_time + buffer PTS` on a pipeline clock forced to CLOCK_MONOTONIC = time nvarguscamerasrc got the buffer (late by readout + ISP + queueing). `t_capture = t_stamp + offset_prior_ns + kalibr_timeshift`, applied only at pose-prep/export. Prior **−30 ms (placeholder)**. Arrival latency tracked by `LatencyTracker` (drift warn 5 ms) | `SR/drivers/argus_camera.py:6-7,69,117-127`; `SR/drivers/argus_logic.py:11-12,134-136,146-175`; `rig_default.yaml:29-32,48-51` |
| 5 | Rolling shutter | yes | IMX219 binned readout ≈ 23.3 ms per frame (mid-row is +11.6 ms after SOF) | `docs/raspberry_pi.md:207` (sensor-level number, valid for any host) |
| 6 | Exposure / WB | ARKit auto | exposure range 0.1–4 ms, gain 1–8, **`aelock: false`, `awblock: false`** (auto) | `rig_default.yaml:22,40` |
| 7 | Encoding on the rig | HEVC/H.264 in the app (HW) | **CPU JPEG** (OpenCV, Q90, 2 workers per camera, 0.75 s queue). ≈ 490 KB per UW frame and similar for main → **≈ 25–30 MB/s** for both | `rig_default.yaml:96-99`; `docs/raspberry_pi.md:72`; `HARDWARE_CHECKLIST.md` §3 (≈ 20 MB/s) |
| 8 | IMU | internal to ARKit | ICM-42688-P **ODR 1000 Hz**, written at 1000 Hz (FIFO, 20-byte high-res timestamps, chip clock → MONOTONIC). VIO input block-averaged to ~200 Hz (`--imu-rate`) | `rig_default.yaml:70-77`; `SR/drivers/chip_clock.py`; `SR/pose/cli_pose.py:17` |
| 9 | Pose source and rate | ARKit `camera.transform` per ARFrame, at the image time | **Offline** ORB-SLAM3 mono-inertial on UW + IMU (laptop). `f_<name>.txt` = Twb (IMU pose) per tracked UW frame, **≈ 30 Hz**, on the corrected clock (pose-prep writes corrected stamps) | `rig_default.yaml:86-88`; `SR/pose/orbslam3.py:3,94-100` |
| 10 | Export grid (60 Hz) | n/a (one row per ARFrame) | Median fps < 50 → **fill mode always on this rig**: uniform 60 Hz grid, each slot shows the **nearest** main frame (ties alternate) | `SR/export/grid.py:32-46,49-59`; `SR/export/exporter.py:36-37,199` |
| 11 | Export fps abort | app: < 31 fps aborts | Fill: ≥ 0.52 × 30 = **15.6 frames in every 1-s window** | `SR/export/exporter.py:43,201-204`; `SR/export/grid.py:62-77` |
| 12 | Export poses | identical to image time | Linear position + SLERP at slot times. Max key gap 0.10 s. Head/tail trimmed; a mid-episode gap aborts. `poseTimes` = `rgbTimes` (keeps `poseTimes[0] == rgbTimes[0]`, which the gripper-width time base needs) | `SR/export/exporter.py:39,122-130,208-225,293` |
| 13 | Export UW | slotted into ARFrame rows, `""` elsewhere | 30 fps → greedy decimation to **10 Hz** (every 3rd frame) → nearest slot within half the spacing + 1 ms. The string written is the slot's main time. PTS relative to the first occupied slot, timescale 600. [measured] slot vs capture ≤ 4.3 ms; spacing 99–101 ms | `SR/export/slotting.py:12-25,28-57,60-66`; `SR/export/exporter.py:38,233-244` |
| 14 | Export images | — | physical → virtual iPhone cameras (main 1920×1440, UW 640×480 pinhole), libx264 mp4. In fill mode every main frame is encoded twice | `SR/export/exporter.py:247-276` |
| 15 | iPhUMI detect (UW time base) | PTS + (first non-empty slot)/60 | same; **exact** on the uniform fill grid | `DP/process_stages/detect.py:65-69` |
| 16 | iPhUMI align grid | `arange(poseTimes[0], poseTimes[-1], 1/60)` | same | `DP/process_stages/align.py:81-82,128`; `DP/config/process_demos_iphone.yaml:62-64` |
| 17 | Align image index | **most recent past** frame | same, applied on top of export's nearest-frame choice (see F2) | `DP/process_stages/align.py:28-33,141-144` |
| 18 | Align pose | linear per component of [pos, rotvec] | same | `DP/process_stages/align.py:19-25` |
| 19 | align_validate | 3 m/s per 60 Hz step | same | `process_demos_iphone.yaml:67,80`; `align_validate.py:52-53` |
| 20 | Gripper width | ArUco on UW (~10 Hz), linear interpolation onto aligned times relative to `aligned_times[0]` | same. Tag time = slot time (≤ 4–8 ms from the corrected UW stamp) | `DP/utils/gripper_util.py:97-107` |
| 21 | zarr | low-dim + main RGB at **60 Hz** (N rows); UW at **~10 Hz** + `upsample_index` (N,) / `downsample_index` (N_uw,); 224×224 pad-to-square | same | `DP/build_umi_dataset_stages/gen_replay_buffer_zarr.py:183,359-379,384-385`; `common/replay_buffer.py:613-618` |
| 22 | Training obs | main + eef pos/rot + width: horizon 2, step 3 → **t and t−50 ms**. UW: horizon 2, step 1 in UW index → the 2 most recent distinct 10 Hz frames (100 ms apart) | same | `RC:76-79,105-170`; `BP/config/task/umi.yaml:8-9`; `BP/common/sampler.py:674-700` |
| 23 | Training action | 10-D (rel pos 3 + rot6d 6 + width 1), horizon 16, step 3 → t, t+50, …, t+750 ms. **Effective control rate 20 Hz**, relative to the latest obs pose | same | `RC:80,91-104,173-175`; `BP/common/sampler.py:712-726` |
| 24 | Offline rollout / exec | `exec_action_horizon` 12 (0.6 s); steps the dataset by 3 | same | `RC:219`; `BP/env_runner/umi_prompting_runner.py:156-159` |
| 25 | Prompt | `pair` mode: one full demo of the same task. Actions at 20 Hz chunked by 20 (**1 token/s**), obs at 1 Hz. `max_sequence_length` 1200 → ≤ 60 tokens (demo ≲ 60 s) | same | `RC:84-90`; `BP/common/sampler.py:502-545,926-951`; `BP/model/prompt/prompt_obs_encoder.py:484-489` |
| 26 | Split | `val_ratio` 0.05 by task → ~2 val demos at 45, so the metric is noisy | same | `RC:208-211` |

### 1.2 Measured timing at the policy's sampling points

All values are image time relative to pose time at each aligned 60 Hz sample (positive = image older than the pose).

| Case | Main image − pose | Main obs-pair spacing (k vs k−3) | UW image − pose |
|---|---|---|---|
| iPhone (real `firstreal`, 4 demos) [measured] | 0 … +16.7 ms, **mean +13.2…14.2** | **50 ms in 99.8 %** | 0 … +116 ms, **mean ≈ 55** |
| **Jetson rig, 30 fps fill, current code** (sim with real export code, 0.2–2 ms stamp jitter) | **−17 … +33 ms, mean +9.5…10.4** | **33/67 ms = 50/50 %**, never identical | ≈ iPhone (10 Hz sample-and-hold) |
| Jetson rig after fix F2 | −17 … +17, mean ≈ 0 | 33/67 = 50/50 (inherent to 30 fps) | unchanged |

Why the current fill mode is asymmetric:
- Export chooses the nearest frame per slot (zero-mean).
- Align then takes the most recent slot whose **ms-rounded** string is ≤ the float grid time `t0 + k/60`. When
  rounding puts the string just after that time, align takes slot k−1, which adds 16.7 ms on about half the samples.
- The iPhone also runs "one slot behind": [measured] `rgb_frame_idx != k` for 99–100 % of samples, hence its +14 ms
  mean. But its frames are 60 Hz, so the spacing stays a constant 50 ms.

### 1.3 What the training data requires of the recording (Jetson numbers)

| Requirement | Value | Why (and the baseline the policy already tolerates) |
|---|---|---|
| Main rate | **30 fps** in the full-FOV 1640×1232 mode (≥ 20 is the absolute floor for distinct 50 ms obs). Never the cropped 1280×720@60 | Obs are 50 ms apart. Resolution is irrelevant above ~224 px of content (training is 224×224; deployment renders 960×720 → 224) |
| UW rate | 30 fps (VIO needs ≥ 20–30); obs and tags use only 10 Hz | ORB-SLAM3 mono-inertial |
| IMU | 1 kHz recorded, rate within 1 % of ODR under recording load, no FIFO overflow; ≥ 200 Hz into VIO | `HARDWARE_CHECKLIST.md` §2 |
| Drops, main | ≤ 1 %, **no gap > 100 ms** (a proposed new check, F6) | A gap shows as a frozen image while the pose moves. The current fill rule (≥ 15.6 frames per 1-s window) lets a ~0.45 s freeze through |
| Drops, UW | 0 `queue_full` / `sensor_gap` (checklist); no gap > ~250 ms (proposed) | Width interpolation, VIO tracking |
| Camera stamp → mid-exposure, per camera | Measure `offset_prior_ns` with an LED/GPIO test (`HARDWARE_CHECKLIST.md` §1). Residual systematic error ≤ 3 ms. Appsink latency jitter ≤ 2 ms RMS and no drift (`LatencyTracker` warns at 5 ms drift) | Below the iPhone's 0–17 ms image staleness. A systematic error > 17 ms shifts images past anything in iPhone data. Appsink stamps include ISP and queueing delay, so **jitter and load-dependence are the unknowns** |
| UW camera ↔ IMU (for VIO) | **≤ 1 ms** after Kalibr `timeshift_cam_imu` (accept a raw shift < 10 ms, repeatable to 1 ms). pose-import warns above 5 ms | ORB-SLAM3 does not estimate td online. At ~2 rad/s wrist rotation, 5 ms = 0.6° per frame |
| Main ↔ UW | **No hardware sync needed.** Relative stamp error ≤ 5 ms | Main gives the obs image; UW gives width and (through VIO) pose. Only consistency matters |
| Pose accuracy (VIO + TCP) | Relative error over 1 s ≲ 3 mm and ≲ 0.5°. Scale error ≲ 2 %. **No resets or jumps inside a demo.** Initialized before the motion (head trim allowed; mid-gap aborts) | Actions are relative motions over 0.75 s, ~10–25 mm per 50 ms step |
| Pose latency | none in the *file*: trajectory stamps = corrected UW stamps (pose-prep) | Offline VIO, so latency is irrelevant if the clocks are right |
| Hard limit | Systematic ≥ 50 ms (one action step) breaks learning. ≥ 17 ms leaves the iPhone envelope. Target ≤ 5 ms | — |
| Exposure | **Locked**, ≤ 4 ms (motion blur, VIO, Kalibr). Gain and WB fixed per session | Auto-exposure and auto-WB change the image statistics between episodes and can break the photometric consistency VIO relies on |
| Gripper width | both finger tags in the virtual UW, closed and open. One gripper-calibration episode per session and side | `calibrate.py` needs ≥ 3 two-tag frames and a 0.2 detection fraction |
| Demo length | 1 s … ~60 s | `prompt_obs_encoder.py:489`; `exporter.py:44` |
| CPU and disk budget | 2 × 1640×1232 × 30 fps JPEG on the Orin Nano CPU, plus Argus→BGR conversion, plus the Python IMU reader. Must sustain 0 `queue_full`; NVMe ≥ 30 MB/s; no thermal throttling at 15 W | No hardware encoder; check with `bench-encode` and the 10-min run (`HARDWARE_CHECKLIST.md` §3) |

### 1.4 Where the rig differs from the iPhone, and whether it matters

| Difference | Effect | Matters given the sampling actually used? |
|---|---|---|
| **30 vs 60 fps main** (fill mode, each frame in 2 slots) | Obs spacing 33/67 ms instead of 50. Image-pose error ±17 ms (after F2) instead of 0…17 | **Minor for rig-only training:** observation noise roughly the size of the iPhone's own staleness; action labels untouched. **Matters for mixing** with iPhone data, and piece B must sample deployment frames the same way. No alternative exists on this hardware (the 60 fps mode is cropped). |
| No camera sync + appsink stamps | Main and UW stamps carry independent, load-dependent pipeline delays | **Yes, if not measured.** Fine once each camera's offset is known to ≤ 3 ms and stable. The C++ libargus helper (`getSensorTimestamp`, SOF) is the upgrade path if appsink jitter is too high (`HARDWARE_CHECKLIST.md` §4). |
| Image–pose offset | iPhone: one timestamp for image and pose. Rig: pose from UW+IMU, image from main | Covered by the row above. |
| UW timing | 30 fps decimated to an exact 10 Hz; string = slot time (≤ 4–8 ms) | No: same 10 Hz pair as on the iPhone; width time error ≤ 8 ms. |
| Rolling shutter (IMX219 ≈ 23 ms readout) | Skew within a frame | Training images: no (training and deployment use the same camera). **VIO: yes.** ORB-SLAM3 doesn't model it. Mitigations: ≤ 4 ms exposure; the mid-row offset is part of `offset_prior_ns`. UMI's GoPro is rolling shutter too and works. |
| Pose source | VIO (scale from IMU, init failures, resets) vs ARKit | **Yes, the main risk** (§4). |
| View / FOV / colour / width scale | Virtual cameras reproduce the iPhone K; the lens must cover them. Width depends on tag geometry | Only for mixing with iPhone data or reusing iPhone policies. |

### 1.5 Mismatches and gaps found (flags)

| ID | Finding | Evidence | Fix (inside `sensor_rig/` unless noted) |
|---|---|---|---|
| **F1** | **Fill mode, which is the only mode on this rig, was never run through the real iPhUMI pipeline.** D10c simulated a native ≈ 59 fps main | `e2e/lead_run2/rig_export/.sensor_rig/export_log.jsonl`: all `grid_mode: native`. Fill is covered only by unit tests (`tests/test_export.py:54`, `tests/test_export_poses.py:55`) | Add `--sim-rig sim_rig_main30.yaml` (or `--main-fps 30`, `SR/sim/cli.py:17`) to `tests/e2e/run_pipeline_compare.py` and rerun D10c |
| **F2** | Fill mode + align's most-recent-past rule on ms strings → −17…+33 ms image lag (mean +10) instead of zero-mean ±17 | §1.2 [measured] | Emit fill-mode slot strings so that `rgbTimes[k] ≤ t0 + k/60 − margin` (grid start on a ms boundary + 0.5 ms, floor to ms) and interpolate poses at the string times. Test: align returns `rgb_frame_idx == k` |
| **F3** | **No finger/gripper mask for VIO.** The UW sees fingers and tags moving rigidly with the camera; ORB features there break the static-scene assumption | `SR/pose/` has no mask; UMI masks the gripper | UMI ORB-SLAM3 fork (mask support) or keypoint masking. Painting black is not enough (static edges) |
| **F4** | **No mapping-video → localize flow.** Per-demo mono-inertial init on short tabletop clips is fragile | R4 §3, §7; `SR/pose/orbslam3.py:100` runs stock `mono_inertial_euroc` per episode | `pose-prep --target umi`: map once per session (save atlas), localize each demo (load atlas) |
| **F5** | **No jump or reset detection.** pose-import checks overlap, gravity and delay only. Export aborts at > 7 m/s, align_validate at > 3 m/s. ORB-SLAM3 multi-map output can contain frame jumps | `SR/pose/trajectory.py:229-265`; `SR/export/exporter.py:133-145` | Gyro-vs-VIO angular-rate residual (RMS and max) plus a per-step translation-jump check, recorded in `trajectory.json`; refuse or flag above thresholds |
| **F6** | Fill-mode drop rule too loose (≥ 15.6 frames per 1-s window → a ~0.45 s freeze passes) | `SR/export/exporter.py:43,201` | `max_main_gap_s` 0.10 and `max_uw_gap_s` 0.25 |
| **F7** | `rig_default.yaml` leaves **AE and AWB unlocked** (`aelock: false`, `awblock: false`) | `rig_default.yaml:22,40` | Lock exposure (≤ 4 ms), gain and WB for recording and calibration. Record the values in rig.yaml |
| **F8** | Appsink stamps are not SOF. The −30 ms offset is a placeholder; jitter under JPEG load is unknown | `SR/drivers/argus_logic.py:11-12`; `rig_default.yaml:31,50` | LED/GPIO test per camera, under full recording load. If jitter > 2 ms RMS or the offset is load-dependent: C++ libargus helper with `getSensorTimestamp` |
| F9 | Component-wise rotvec interpolation in align (`align.py:19-25`) is wrong near \|θ\| ≈ π. iPhone data reach 149°; rig exports 27–81° [measured]. The VIO world yaw is arbitrary | [measured] | Warn above 170°, or pick a per-episode world yaw (free under the relative representation) |
| F10 | `sensor_rig/` and the `"JetsonRig1,1"` entry in `gripper_calibration.yaml` are **uncommitted**, so the cluster can't process rig exports | git status; `gripper_calibration.yaml:16-21` | Commit (user decision) |
| F11 | `isErrorCorrection` is always false in exports | `SR/export/iphone_json.py:76` | Pass a flag through `meta.json` if EC demos are used |
| F12 | `process_dataset.sbatch` has never run | AGENTS.md | WP2 |

### 1.6 Recommended settings (Jetson)

- **Main:** IMX219 mode 3, 1640×1232 @ 30 fps, ≈ 90° diagonal-class M12 lens (verify coverage ≥ 0.99 with
  `rig-validate` before buying). Exposure locked ≤ 4 ms, fixed gain and WB.
  - Accept fill mode.
  - **Fix F1, F2 and F6 before collecting real data.**
  - Never use 1280×720@60.
- **UW:** IMX219 mode 3, 1640×1232 @ 30 fps, ≈ 160° fisheye (coverage 1.0), same locks. VIO uses all 30 fps; export
  uses 10 Hz.
- **IMU:** ODR 1 kHz with on-chip low-pass. VIO input at 200 Hz (`--imu-rate 200`). If the Python reader misses the
  rate under load, move it to C++ (`HARDWARE_CHECKLIST.md` §4).
- **Timestamps:** measure both cameras' `offset_prior_ns` under full recording load (both cameras + JPEG + IMU). Then
  apply Kalibr `timeshift_cam_imu` on top.
- **JPEG/CPU:**
  - Keep Q90 with 2 workers per camera.
  - Confirm with `bench-encode` on the Orin Nano at 1640×1232 (2 streams × 30 fps).
  - If the budget fails, lower the quality before lowering the resolution. The UW must stay full resolution for VIO.
  - Record to NVMe.
- **Export:** defaults (`--uw-rate 10`, 60 Hz fill grid).
- **Training:** unchanged (`obs_down_sample_steps: 3`, `ultrawide_down_sample_steps: 1`, `align.fps: 60`), since
  export always produces 60 Hz + 10 Hz. Use ~60 epochs (run_001 overfit after ~50).

---

## 2. Done vs remaining (piece A, Jetson)

### (a) Hardware and mount

Done:
- Coverage analysis (77°/120° lenses = 0.83; ≈ 90° main / 160° fisheye = 1.0).
- IMU SPI wiring on the 40-pin header (`rig_default.yaml:70`, jetson-io).

Remaining:
- Buy the lenses and check them with `rig-validate`.
- **CAD mount.** Main and UW lens centres where the iPhone 16 Pro lenses were (a remap fixes rotation, not position),
  IMU rigid on the same plate, fingers and tags in UW view, CSI cable strain relief.
- Power, NVMe, fan; the Jetson's own mounting or tether.
- Glue the focus.

### (b) Drivers and recording bring-up

Done (no hardware):
- `ArgusCamera` (GStreamer → appsink, MONOTONIC pipeline clock, latency tracking, sensor-gap detection).
- `Icm42688Spi` (FIFO decode, chip-clock mapping).
- Recorder with live aborts, JPEG workers, drops accounting, mock mode; `bench-encode`.
- 199 unit tests pass.

Remaining (hardware), `HARDWARE_CHECKLIST.md` §0–3:
- Mode list; both sensor-ids running at the same time; CAM1 second-instance issue on JP 6.2.1 (5 restarts).
- 300-frame checks: no seq gaps, 33.3 ms steps, positive and stable latency.
- **LED/GPIO stamp-offset test.**
- Exposure/WB locks.
- IMU: WHO_AM_I, gravity, 1 kHz ± 1 % under load.
- 60-s recordings with 0 drops; 10-min thermal run at 15 W; disk-full abort.

### (c) Calibration

Done (tooling):
- `calib-import`, `calib-check`, `calib-align`, `calib-tcp`, `rig-validate`, `bench-remap`.
- Procedure in `docs/calibration.md`.
- Placeholder `"JetsonRig1,1"` entry (a copy of iPhone17,1).

Remaining (hardware):
- Allan variance (≥ 3 h static).
- Kalibr cameras (≤ 0.3 / 0.5 px).
- Kalibr camera-IMU (twice, agree to 1 ms / 0.2° / 3 mm).
- `calib-import --strict` → `rig_v1.yaml`.
- `calib-align` against the existing iPhone `firstreal` gripper-calibration export.
- `calib-tcp` from CAD.
- `artag_z_distance_from_ultrawide` from `calib-check`.
- One rig gripper-calibration episode per session.
- `bench-remap` on the Orin Nano (deployment budget, piece B).

### (d) VIO

Done:
- `pose-prep` (EuRoC folder, ORB-SLAM3 settings with KannalaBrandt8, OpenVINS/Kalibr yamls).
- `pose-import` (parse, gravity alignment, overlap / gravity / delay checks).

Remaining:
- **ORB-SLAM3 build. Recommendation: follow UMI's pipeline with the UMI fork in its Docker image on the laptop**
  (x86; the cluster only if Apptainer is available). Not on the Jetson.
- Mask (F3); mapping flow (F4); jump checks (F5).
- A real-footage pilot; accuracy vs ground truth (WP8).
- OpenVINS as fallback (online td, no map).

### (e) Export and processing on real data

Done:
- `export` / `validate`, atomic output.
- D10c native-mode equivalence with the real pipeline.
- Validator passes on all 47 real iPhone exports.

Remaining:
- F1, F2, F6, F9, F10.
- Flow: laptop `export --out data/rig_export` (layout `<date>/<NAME>/`), then
  `scripts/run_pipeline.sh process|all --raw-dir data/rig_export --sessions <rigsession> --task block_raise`.
  `group.py` globs `**/*.json` and skips the hidden `.sensor_rig/`.
- Labels: let `process_dataset.sbatch` label "None" demos; don't also pass `--label-task`.
- Session names without underscores, distinct from iPhone sessions.
- First `process_dataset.sbatch` run (F12).

### (f) Validation

Done:
- Simulation equivalence (width median diff ~0.1 mm, relative motion ~1e-5, image MAE ~1).

Remaining:
- VIO accuracy against the Franka; width range vs iPhone; timestamp offsets.
- Sanity viewers (`calib-check --episode`, exported mp4s, zarr frames and trajectories).
- Dataset statistics vs `firstreal`.

### (g) Training

Done:
- `firstreal-single` trains (run_001, best val action MSE 1.48e-4 at epoch 50).
- Smoke path proven (run_002).

Remaining:
- Sim-rig 30 fps run (WP2).
- Real rig smoke and full runs; comparisons; decision on mixing.

---

## 3. Work packages (dependency order)

Sizes: **S** ≤ 1 day, **M** 2–5 days, **L** 1–2 weeks.
- 🖥 = software-only, can start now
- 🔧 = needs the rig
- ☁ = cluster submission (ask the user before submitting)

### WP0 Commit and housekeeping 🖥 S
1. User decision to commit `sensor_rig/` and the `gripper_calibration.yaml` entry.
2. Add `sensor_rig/` to DESIGN.md (layout, and the vendored-changes list: yaml entry, +1 line in real-env
   `pyproject.toml`).

**Done when:** pushed, and the cluster clone is pulled.

### WP1 Fill-mode hardening and e2e 🖥 M
1. F2: align-friendly slot strings plus a unit test that reproduces align's indexing.
2. F6: gap aborts.
3. F9: angle guard.
4. F5: jump / gyro-residual metrics in pose-import.
5. F7: lock AE/AWB in `rig_default.yaml` (document the values to set on hardware).
6. F1: `run_pipeline_compare.py --sim-rig sim_rig_main30.yaml`, plus a variant with `--drop-rate 0.01
   --ts-jitter-ns 2000000`. Add the §1.2 lag statistic to `report.json`.

**Done when:** D10c passes in fill mode; fill-mode lag is zero-mean within ±17 ms; unit tests pass.

### WP2 Sim-rig 30 fps dataset → `process_dataset.sbatch` → training ☁🖥 M (after WP0, ideally WP1)
1. `python -m sensor_rig sim --iphone data/raw/2026-10-01 --main-fps 30 --session simrigthirty --out
   data/rig_raw_sim`. The session must differ from `firstreal` to avoid `tmp_demonstrations` collisions.
2. `export --out data/rig_export_sim`.
3. `scripts/run_pipeline.sh all --task block_raise --dataset simrig30-single --sessions simrigthirty --raw-dir
   data/rig_export_sim --smoke-first -- training.num_epochs=60`.
4. A 60-epoch `firstreal-single` baseline. Check that the val split selects the same demos.

**Compare:** val action MSE components and attention videos.

**Done when:** both datasets keep the same valid-demo count and the runs agree within noise (e.g. ≤ 20 % relative val
MSE, given ~2 val demos).

This tests the whole rig path on the cluster except real VIO, lenses and timing noise.

### WP3 Lenses and mount 🔧 M
1. Buy the ≈ 90° main and ≈ 160° fisheye M12 lenses for IMX219 4:3 modules; `rig-validate`.
2. CAD from the iPhone 16 Pro lens positions on the existing gripper mount.
3. Print, assemble, glue focus, route the CSI cables, mount the NVMe.

**Done when:** `calib-check --live` shows both tags (closed and open) and coverage ≥ 0.99 for both cameras.

### WP4 Jetson bring-up 🔧 M (after WP3)
1. `HARDWARE_CHECKLIST.md` §0–3.
2. `bench-encode` (2 × 1640×1232 @ 30).
3. **LED/GPIO stamp test** for both cameras under full load. Set `offset_prior_ns` and check that the stamp jitter
   is ≤ 2 ms.
4. Locked exposure, gain and WB.
5. 10-min thermal run.
6. A 60-s demo + gripper calibration through pose-prep / export with the placeholder rig.

**Done when:** 0 drops, fps ± 1 %, IMU ± 1 %, offsets known to ≤ 2–3 ms. Otherwise trigger the C++ libargus stamp
helper (+M).

### WP5 Calibration 🔧 M (after WP4; Allan can start once the IMU works)
1. Allan.
2. Kalibr cameras.
3. Kalibr camera-IMU (twice).
4. `calib-import --strict`.
5. `calib-align`.
6. `calib-tcp`.
7. `calib-check` acceptance.
8. Version `rig_v1.yaml` and the reports.

Only the Kalibr intrinsics and camera-IMU extrinsics/timeshift are on the VIO critical path. The rest can follow,
because raw episodes re-export with `--rig`.

### WP6 VIO (UMI-style ORB-SLAM3, offline on the laptop) 🖥 now + 🔧 pilot — L — **critical-path risk**

Software, now:
1. Build or pull UMI's ORB-SLAM3 Docker image; reproduce on a sample.
2. Add `pose-prep --target umi`. It writes that fork's inputs (UW video or images + IMU + settings + a **static finger
   mask** covering the full finger sweep) and a map → localize run script.
3. pose-import for its output.
4. Plumbing test on simulated episodes. The iPhone UW is only 10 Hz, so this tests plumbing, not accuracy.

Hardware pilot:
1. Record a 60–90 s mapping video, 5–10 demos and a wiggle clip.
2. Run map → localize.
3. pose-import checks (gravity < 3°, delay < 5 ms, F5 metrics); count the fraction of demos tracked end to end.

**Done when:** ≥ 90 % of demos are tracked without mid-gaps, pose-import passes, and the WP8 accuracy targets hold.

**Fallback:** OpenVINS (online td), or longer wiggle starts without a map.

### WP7 Real data: export → processing 🖥 + ☁ S–M (after the WP6 pilot, WP1, WP0)
1. Record session `rig<date>`: gripper calibration, then demos. Copy raw data off the Jetson NVMe with
   `rsync --exclude '.*'`.
2. Laptop: pose-prep → ORB-SLAM3 → pose-import → `export --out data/rig_export` → `validate --decode`.
3. Sanity: `calib-check --episode`, mp4 scrub, trajectory and width plots.
4. `run_pipeline.sh process`.

**Done when:** ≥ 90 % of demos pass align_validate, the zarr has `gripper_right_gripper_width` (10-D actions), and
images have no black borders.

### WP8 Validation against ground truth 🔧 M (in parallel with WP7)
- **VIO and TCP:** rig on the Franka flange, a scripted 6-DoF trajectory at hand speeds, VIO (after hand-eye
  alignment) vs forward kinematics: RPE over 1 s, scale, drift.
- **Width:** rig gripper-calibration `min_width` / `max_width` vs iPhone `firstreal` (0.043 / 0.128 raw).
- **Timing:** LED residuals, Kalibr repeatability, per-demo pose-import delay.
- **Dataset statistics** vs `firstreal`: lengths, speeds, widths, valid fraction.

**Done when:** the §1.3 targets are met, or the deviations are documented with a decision.

### WP9 Training on rig data ☁ M (after WP7)
1. Smoke run on the first 10–20 demos.
2. A ~60-epoch full run at ≥ 45 demos (≈ 38k frames, ~2.1 min/epoch on an H200).

Compare with run_001 and the WP2 sim run:
- val action MSE components;
- attention videos;
- offline cross-evaluation of the iPhone checkpoint on rig val data and vice versa (a small eval script or
  `task.eval_dataset_path`);
- optional mixed dataset, only after WP8 confirms width scale and view alignment.

**Done when:** the rig-only policy reaches val MSE of the same order as run_001 with sensible rollout videos. Hand
that checkpoint to piece B.

### Dependency graph and critical path

```
WP0 ─┬─ WP1 ─┬─────────────────────────────────────────────┐
     │       └─ WP2 (sim-rig 30 fps train, ☁)                │
     └─ WP6-software (ORB-SLAM3 Docker, umi target, mask)    │
WP3 (lenses + mount) ─ WP4 (Jetson bring-up, stamp test) ─ WP5-Kalibr ─ WP6-pilot ─ WP7 ─ WP9-smoke ─ WP9-full
                                                       └─ WP5-rest (Allan, align, TCP)    └─ WP8 (parallel)
```

- **Critical path:** WP3 → WP4 (including the appsink stamp test) → Kalibr → WP6 pilot (WP6 software ready by then)
  → WP7 → WP9 smoke.
- **The longest unknowns:** mount fabrication, appsink timestamp quality (may add the C++ libargus helper) and VIO
  robustness.
- **Earliest real-rig training:** the day after the WP6 pilot works on ~10–20 demos. Roughly 3–5 weeks after the
  parts arrive.
- Demos can be recorded before calibration is final as long as the hardware is frozen. Raw episodes keep unmodified
  stamps and re-export later.

---

## 4. Risks and decisions

### Risks (highest first)
1. **VIO robustness on short tabletop demos:** init failures, resets, finger features, IMX219 rolling shutter, 30 fps.
   Mitigations:
   - UMI recipe (map + localize + mask);
   - wiggle at the start of each demo;
   - locked exposure ≤ 4 ms;
   - F5 checks;
   - OpenVINS fallback.
2. **Timestamp quality on Argus.**
   - Appsink PTS is not SOF, and the two free-running cameras carry independent, possibly load-dependent delays.
   - The Python IMU reader may miss 1 kHz under JPEG load.
   - Mitigations: LED test under load, Kalibr timeshift, per-demo pose-import delay check, C++ libargus stamps and a
     C++ IMU reader as upgrades.
3. **CPU budget without a hardware encoder** (2 × 1640×1232 @ 30 JPEG + BGR conversion + IMU, thermal at 15 W).
   `bench-encode` and the 10-min run will tell; lower the JPEG quality first.
4. **JetPack 6 CAM1 second-instance issue** (checklist): both cameras must come up reliably.
5. **Fill mode untested end to end (F1/F2).** Cheap to close now.
6. **Lens coverage and mount position.** Below 0.98 coverage adds black borders; a positional offset shifts the
   viewpoint and invalidates the copied iPhone TCP.
7. **Gripper width:** tags near the fisheye edge; a width scale different from the iPhone.
8. **`process_dataset.sbatch` untested.** The silent "Skipping gripper width" failure removes the width key, so check
   the zarr keys after each run.
9. **Small validation split** (~2 demos).

### Decisions for the user
1. **Commit `sensor_rig/`** and the `gripper_calibration.yaml` entry now? Needed for WP2 and every cluster step.
2. **Approve the WP1 changes** (F2, F5, F6, F7, F9), all inside `sensor_rig/`.
3. **Accept 30 fps fill mode** as the only option on IMX219, which makes rig-only training the default. Agree that
   piece B must reproduce the same frame sampling at deployment.
4. **VIO stack:** UMI's ORB-SLAM3 fork with map/localize, in laptop Docker (recommended), or stock ORB-SLAM3 with
   OpenVINS as backup.
5. **Rig-only or mixed with iPhone data?** Recommendation: rig-only first; mix only after WP8.
6. **Ground truth:** approve using the Franka flange for VIO and TCP validation.
7. **Approve ☁ submissions** for WP2 (processing + smoke + 60-epoch sim-rig run, plus a 60-epoch `firstreal-single`
   baseline).
8. **If appsink stamps prove noisy:** approve the C++ libargus timestamp helper (+M).

---

¹ Footnote: `sensor_rig` also contains a Raspberry Pi 5 / Picamera2 backend (`SR/drivers/picamera2_camera.py`,
`configs/rig_pi5.yaml`, `docs/raspberry_pi.md`). It is out of scope for this hardware setup. Everything from pose-prep
onward is identical for both.
