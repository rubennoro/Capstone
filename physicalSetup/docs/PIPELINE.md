# physicalSetup pipeline: iPhone-equivalent training episodes from the Jetson rig

Status: design, 2026-10-27, branch `rnoro/jetson-sensors`. Implementation plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

The rig replaces the iPhone on the iPhUMI handheld gripper:
- Jetson Orin Nano;
- Arducam IMX519 autofocus main camera;
- Arducam IMX219 160° ultrawide;
- ST LSM6DSOX IMU on an Arduino Modulino Movement over I2C.

Code lives in `physicalSetup/`, in C++:
- `common/`: shared by collection and deployment;
- `episode_collection/`: the three programs below;
- `deployment/`: policy deployment on the Jetson, later.

The Python `sensor_rig/` package (built for older hardware choices) stays as the reference implementation and the
source of golden test data.

Tags: **[repo]** = read in this repo (file:line). **[V]** = read in a vendor or paper source. **[I]** = inferred; check on hardware.

---

## 1. The idea in one page

An episode is a **package**: a directory that a chain of independent processes advances, one stage at a time, until
it is a training-ready piece of the behavior-prompting dataset.

```
 ON THE JETSON, LIVE                                     LATER, OFFLINE (Jetson or laptop, any time)
 ───────────────────                                     ────────────────────────────────────────────
 ec-calibrate ──► calibration package ──┐
   (gripper open/close; fit on exit)    │
                                        ▼
 ec-record  ──►  episode package  ──►  ec-process: validate_raw → slam → traj_fit → pose → width → validate
   capture all sensors                  → images → export ──► <session>.zarr.zip ──► behavior_prompting training
   + live checks (iPhone bounds)
   + live ORB-SLAM3 30 Hz (checks + display)
```

**Why the training pose is offline.**
- The iPhone runs ARKit's SLAM live, which is the only way it can produce pose. Our team found the phone **overheats
  and cannot record successive episodes** under that load (on top of 60 Hz video encoding).
- In practice, throttling shows up as frame-rate drops, which trip the app's own fps < 31 abort
  (ViewController.swift:1819 [repo]).
- Doing the accurate pose computation offline removes it from the recording-time budget. Collection only has to
  capture and check, so we can keep recording episode after episode.
- Offline also means no deadline: every frame is processed, there is a full bundle adjustment, and a continuous-time
  trajectory fit with all the IMU data. That is the most accurate pose we can produce, and **it is the only pose that
  goes into training.**

**Why there is still a live SLAM.**
- The iPhone enforces collection bounds while recording, and one of them (pose jump > 7 m/s) needs a live pose.
- A lightweight **30 Hz live ORB-SLAM3** provides that pose, and drives the on-screen SLAM view.
- It is used for live checks and display only. Nothing it computes is written into the training data.

**Why it is modular.**
- Once the sensors are integrated and collecting, collection does not care what happens next. A recorded package can
  be processed straight away, stored and processed later, re-processed when a calibration improves, or fed to a
  different algorithm (another SLAM, another trajectory fit, another width method) by swapping one stage.
- The capture layer, time base, shared-memory bus and image preparation are the same code at deployment, where the
  Jetson feeds live camera frames to the trained policy and its actions go to the Franka controller.

---

## 2. Target: what "1:1 with the iPhone" means

The policy never sees raw iPhone sensors. It sees the zarr built by `iPhUMI/python_package/iphumi/demonstration_processing/`,
and at deploy time an observation assembled to the same spec. The contract is that zarr and that observation, not the
iPhone's raw files.

| Item | iPhone path | Source [repo] |
|---|---|---|
| Main RGB | 60 Hz 1920x1440 (4:3). Stored as 224x224: letterbox to square → `INTER_AREA` resize → JPEG-XL. No undistort. | ViewController.swift:1819; cv_util.py:5-34; gen_replay_buffer_zarr.py:430 |
| Ultrawide RGB | ~10 Hz 640x480 (fx 246.6). Stored at native rate, with up/down index maps to the 60 Hz grid. | ViewController.swift:87-114; gen_replay_buffer_zarr.py:313-379 |
| Pose | ARKit `W_T_I` at **60 Hz**, one per main frame (gravity-aligned, y-up), then `W_T_I @ I_T_TCP`. The policy reads it at 20 Hz. | ViewController.swift:1810; gripper_util.py:185-220 |
| Gripper width | ArUco 4x4_50 16 mm finger tags on the ultrawide, PnP (zero distortion), width = Δx, mapped by a calibration episode, interpolated to 60 Hz | detect_aruco_iphone.py:45-118; calibrate.py:85-168 |
| Raw IMU | not recorded (ARKit-internal) | no CoreMotion in the app |
| Depth | recorded, excluded from the dataset | build_umi_dataset_iphone.yaml:10-14 |

Zarr keys:
- `data/camera_right_main_rgb (N,224,224,3)`;
- `data/camera_right_ultrawide_rgb (N_uw,224,224,3)`;
- `data/gripper_right_eef_pos (N,3)`, `data/gripper_right_eef_rot_axis_angle (N,3)`;
- `data/gripper_right_gripper_width (N,1)`;
- `meta/episode_ends`, task labels, and the ultrawide index maps.

Golden reference: `scripts/regression/golden/iphone_firstreal_5ep.json`.

The policy (run_001) reads:
- main: 2 frames at 20 Hz;
- ultrawide: 2 frames at 10 Hz;
- proprio: pos + rot6d + width, 2 steps at 20 Hz;
- actions: 16 steps at 20 Hz.

**Image views (decided 2026-10-27).** Rig policies are trained on rig data and deployed on the same rig, so the image
views only have to match *between rig training and rig deployment*, not match the iPhone.
- **Main:** the IMX519's own view, no remap. The raw frame is letterboxed to a square and resized to 224 with
  `INTER_AREA`, exactly the iPhUMI `cv_util` operation applied to iPhone frames. This is needed because the IMX519 stock
  lens (about 66°x38° at 16:9) sees less than the iPhone main (about 74°x59°). Remapping to the iPhone view would black
  out about 40 % of the height.
  - Consequence: the rig's main images cannot be mixed with iPhone main-camera data, and iPhone-trained checkpoints
    (run_001) are not expected to transfer to the rig.
- **Ultrawide:** remapped to the virtual iPhone ultrawide pinhole (640x480, fx 246.6, about 105°x88°). The 160° fisheye
  fully covers it. This undistorts the fisheye, so iPhUMI's zero-distortion tag PnP is valid, and keeps the ultrawide
  geometry identical to the iPhone's.

All poses are **relative to the current pose** (pose_repr_util.py:62-64). Consequences:
- **Pose:** what matters is local accuracy over about 1 s windows, and timestamp accuracy (1 ms ≈ 1 mm at 1 m/s).
- **Deployment:** pose comes from the Franka (proprio = commanded action history, policy_agent.py:141-166), and width
  from the gripper. Deployment needs **cameras only**. IMU, SLAM and tags are collection-only.

---

## 3. Hardware facts that shape the design

| Part | Role | Facts |
|---|---|---|
| **IMX519** 16 MP AF | main RGB | Rolling shutter, 2-lane. 1080p60 / 720p120 (16:9); full 4:3 only at 4656x3496 ~10 fps [V]. Driver for JP 6.2 / L4T 36.4.3; installs are kernel-exact, so **pin JetPack** [V]. Focus: VCM at I2C 0x0C, userspace; can show `UU` on JP 6.2 [V]. No hardware sync [V]. |
| **IMX219 160°** | ultrawide RGB, **SLAM camera**, tag camera | Only 1640x1232@30 and 3280x2464@21 are full-FOV; 720p60 and 1080p30 are crops [repo picamera2_logic.py:99-104]. Fixed focus. Fisheye model: KannalaBrandt8 for ORB-SLAM3; Kalibr `ds`/`eucm` above ~150° [V]. |
| **LSM6DSOX** (Modulino Movement, 0x6A, 3.3 V, Qwiic) | IMU for SLAM | Bare breakout, no MCU [V]. Wire **directly to the Jetson** 40-pin I2C, **not** through the Ventuno Q (a second clock domain for no gain). FIFO 512 x 7-byte tagged words, chip timestamp at 25 µs/LSB with a per-chip trim [V]. |
| **Orin Nano** | everything | 6 A78AE cores, Ampere GPU, **no NVENC**, active fan. Thermal behaviour under sustained collection must be measured (section 6.3). |

**Rates:**

| Stream | Rate and mode | Why |
|---|---|---|
| Main | 60 Hz, captured at 1920x1080 (1280x720 is a 2/3 crop on the IMX519), stored at 1280x720 | matches the 60 Hz grid |
| Ultrawide | 30 Hz at 1640x1232 | full FOV; better for SLAM than a 60 Hz crop |
| IMU | 833 Hz over 400 kHz I2C, about 30 % bus load [I] | |

---

## 4. Programs (what an operator runs)

All three are command-line programs on the Jetson. `ec-process` also builds and runs on the laptop.

### 4.1 `ec-calibrate`: the per-session gripper calibration run

```
ec-calibrate --session block-raise-2026-10-30        # start; follow on-screen instructions; press Enter to end
```

1. Starts the cameras and IMU, then shows the camera view with live tag overlays, so you can check that both finger
   tags are visible.
2. Records a `grippercalibration` package while the operator opens and closes the gripper fully, about 3 times.
3. On Enter:
   - stops recording;
   - runs tag detection on the recorded ultrawide frames (remapped to the virtual iPhone ultrawide);
   - fits min/max width and the robust left↔right single-tag maps, with the same math as `calibrate.py:85-168`.
4. Prints **PASS/FAIL** with the numbers (range, tag detection rate, fit residuals). On PASS it writes
   `calibration/gripper_range.json` (plus an iPhUMI-compatible yaml) into the session folder, then exits.
5. Run it once per session, and again whenever the fingers or tags are remounted.
6. It does not need SLAM: it records nothing pose-related.

One-off rig calibration (camera intrinsics, camera↔IMU extrinsics and time offset via Kalibr, IMU Allan variance,
LED camera↔camera offset) is a separate mode, `ec-calibrate --rig {kalibr|allan|led}`. It records a package in the same
format, and the matching offline tools consume it.

### 4.2 `ec-record`: one demonstration per run

```
ec-record --session block-raise-2026-10-30 --task block_raise [--viz cameras|slam|both|none]
```

1. **Arming.**
   - Refuses to start if the session has no PASSing gripper calibration (override: `--no-calibration`).
   - Starts the cameras, IMU and live SLAM.
   - Screen: "SLAM initializing: move the gripper". Live ORB-SLAM3 mono-inertial needs a few seconds of motion.
   - The raw streams **are** already written to disk during arming, as a **pre-roll**: `events.csv` marks `arm` and
     `rec_start`. The offline SLAM pass also starts from scratch and needs the same initialization motion. It
     initializes on the pre-roll, so the episode has an accurate pose from its first frame.
   - The training window is `[rec_start, rec_stop]` only. The pre-roll is never exported. If the run is ended before
     recording starts, the pre-roll is deleted.
2. **Recording.**
   - Starts **automatically once live SLAM reports tracking OK** and every stream is at steady rate. Decided
     2026-10-27.
   - Screen: `REC <episode name> 00:12`. All raw streams go to disk.
3. **Live checks** (section 6.2) run on every sample. Any abort rule stops recording, **discards the episode**, and
   shows the reason, exactly like the iPhone. Decided 2026-10-27.
4. **End:** press Enter, or Ctrl-C. The program finishes writing, fsyncs, writes `live_checks.json`, marks the package
   `recorded`, and exits.
   - Then run it again for the next episode; every episode of a session is a separate run.
   - SLAM re-initializes each run, which is a few seconds of motion before recording starts again.
5. **Visualization** (`--viz`), at a lower rate than capture:
   - `cameras`: main + ultrawide mosaic at about 10 Hz, with a status overlay (REC/armed, per-stream fps, IMU rate,
     drops, finger-tag count, disk, temperature);
   - `slam`: ORB-SLAM3's Pangolin viewer (map points, keyframes, trajectory, tracked features);
   - `both` (default).

   The windows open on a monitor plugged into the Jetson (HDMI/DP + keyboard; decided 2026-10-27). Enter and Ctrl-C
   are read from the terminal that runs `ec-record`.

### 4.3 `ec-process`: offline, turns packages into training data

```
ec-process <session_dir> [--stages all|<list>] [--episodes <glob>] [--force] [--impl slam=orbslam3,traj_fit=bspline]
ec-process <session_dir> --status                     # table of every package and the stage it has reached
```

- Separate from collection. Run it right after a session, overnight, on the laptop, or months later.
- It advances every package through the stages in section 7, skipping the ones already done (idempotent, resumable).
- On the last stage it builds `<session>.zarr.zip`, which drops into the existing training pipeline unchanged.

---

## 5. Live architecture (`ec-record` / `ec-calibrate`)

```
 ┌──────────────────────────── ec-record (one process) ─────────────────────────────┐
 │  CaptureService (common/)                                                         │
 │   ├─ cam_main  thread ─┐                                                          │
 │   ├─ cam_uw    thread ─┼─► StreamHub: per-subscriber bounded queues, never blocks │
 │   └─ imu       thread ─┘        ├─► RawWriter (JPEG pool + CSV)     → package/raw │
 │                                 ├─► LiveChecks  ◄── SLAM status/pose (shm)        │
 │                                 └─► ShmPublisher ──────────────┐                  │
 └────────────────────────────────────────────────────────────────┼──────────────────┘
                                                                  │ POSIX shm rings
        ┌─────────────────────────────────────────────────────────┼─────────────┐
        ▼                                                         ▼             ▼
  ec-slam-live (process)                                 ec-viz (process)   [deployment: obs reader]
  ORB-SLAM3 mono-inertial, uw 30 Hz + IMU                camera mosaic 10 Hz
  publishes pose + tracking state + heartbeat ──► shm    reads SLAM status
  Pangolin viewer (lower refresh)
```

**Design rules:**
- **Producers never block.** Each subscriber has its own bounded queue.
  - For the RawWriter, a full queue means lost data, which is an abort rule.
  - For display subscribers, a full queue drops frames silently and counts them.
- **Process isolation.** Live SLAM and the viewers run as separate processes, started and supervised by `ec-record`.
  A viewer crash never touches recording. A SLAM stall or crash is detected by heartbeat and is an abort rule, because
  the pose check can no longer run.
- **CPU placement.** Capture and writer threads get pinned cores and elevated priority. Each backend's `start()` runs
  on its own pinned sensor thread, so driver-internal threads (IMU poll, Argus acquire) inherit the placement. The IMU
  has a core of its own: the config loader rejects an IMU core shared with a camera (or with live SLAM). Live SLAM
  gets its own pinned cores (it has a deadline now). Viewers run at normal priority and drop frames. SCHED_FIFO needs
  root or an rtprio limit; without it, placement failures are logged and counted, and capture continues.
- **One clock.** Every timestamp is int64 ns `CLOCK_MONOTONIC_RAW`, the domain of Argus sensor timestamps and
  IIO/chip-mapped IMU stamps. This gets verified on hardware.
  - Image time: Argus `getSensorTimestamp()` (r36.4.3) is "the time that the first data from this capture arrives
    from the sensor", i.e. the end of row 0's exposure. So `t_sof = ts − exposure` and
    `t_mid = ts − exposure/2 + readout/2` (mid-exposure, mid-frame).
  - The Argus clock domain is not settled by NVIDIA's documentation. A resolver classifies the first frames
    (MONOTONIC_RAW, TSC − offset, or converted MONOTONIC/REALTIME). It accepts stamps up to 2 ms in the future and
    counts them as anomalies. Verified on hardware day.
  - IMU time = chip timestamp mapped to host by a lower-envelope linear fit over bracketed register reads.
  - Calibrated offsets (Kalibr camera↔IMU, LED camera↔camera) are applied **only** when consuming (offline stages and
    deployment), never baked into raw files.
- **The shm bus is a deployment interface too.** Fixed-slot seqlock rings, one writer and many readers, each with
  sequence numbers so a reader detects overruns. The deployment process reads the same rings for the newest frames.

**Capture details:**
- **Cameras:** libargus (EGLStream consumer) with sensor timestamps and per-frame exposure/gain metadata.
  - Exposure, gain and white balance are locked by setting each range to min = max. Exposure is short on the
    ultrawide (≤ 4–8 ms, to limit motion blur for SLAM and tags).
  - IMX519: a fixed VCM focus position written once at start and recorded in the package; autofocus never runs.
  - A GStreamer `nvarguscamerasrc` backend exists as the bring-up fallback. Mock and replay backends exist for tests.
  - **Colour path (one path for training and deployment):**
    - Argus delivers **full-range NV12** (`NVBUF_COLOR_FORMAT_NV12_ER`, i.e. JFIF YCbCr), and every frame carries
      `FrameMeta::color_range` (also over shm).
    - The recorder JPEG-encodes **straight from the NV12 planes** (libjpeg raw data, 4:2:0, JFIF), never through BGR.
    - `imageprep::frame_to_bgr` converts full-range frames exactly as libjpeg decodes them. Live frames and decoded
      training JPEGs therefore differ only by compression loss (q90: mean 1.2, max 19 LSB).
    - The main camera is captured at 1920x1080 and stored at 1280x720 through one shared resize function, used by
      both the recorder and deployment.
- **IMU:** a userspace `i2c-dev` driver for day one: FIFO with tagged accel/gyro/timestamp words, polled every 10 ms,
  watermark and overflow detection.
  - Registers per datasheet DS12814, each one read back after writing. Timestamp batching adds one FIFO word per
    sample, so the poll budget counts 3 words per ODR.
  - Needs the I2C bus at ≥ 400 kHz (the FIFO stream is about 39 % of a 400 kHz bus). **INT1 must stay unconnected**:
    INT1 high at power-up selects I3C-only mode. The kernel IIO `st_lsm6dsx` driver is a later alternative behind the same
  `ImuBackend` interface.

---

## 6. Episode package format, live checks and thermal

### 6.1 Session and package layout

```
<data_root>/<session>/                                   e.g. /data/rig/block-raise-2026-10-30/  (session: [A-Za-z0-9-]+, no "_": iPhUMI group.py:36-37 splits names on "_")
  session.json          session id, task, operator, created, rig.yaml hash, software version
  rig.yaml              frozen copy of the rig config used for every package in this session
  calibration/
    <ISO>_<5d>_<session>_grippercalibration_right/       a package (same layout as below)
    gripper_range.json  fit result: min/max width, L↔R maps, residuals, PASS/FAIL
  episodes/
    <ISO>_<5d>_<session>_demonstration_right/            naming identical to iPhUMI (ViewController.swift:1092-1100)
      manifest.json     schema version, kind, stage states (section 7.1)
      raw/
        cam_main/  data.csv (#t_ns,filename) · data/<t_ns>.jpg · frame_meta.csv (t_sof, exposure, gain, seq, color_range) · drops.csv
        cam_uw/    same (EuRoC-compatible, so ORB-SLAM3 and Kalibr tooling read it directly)
        imu0/      data.csv (EuRoC: t_ns, w_xyz [rad/s], a_xyz [m/s²]) · chip.csv · clock_fit.json
        events.csv      arm, rec_start, rec_stop, warnings (t_ns, type, detail)
        live_checks.json  every rule: value, threshold, pass/fail; max pose speed (as the iPhone UI shows)
        live_slam.csv     the live 30 Hz pose and tracking state (diagnostic only; never used for training)
      stages/<stage>/   each offline stage's outputs + result.json
      export/episode.zarr
```

Main camera capture: 1920x1080 at 60 fps (IMX519 1280x720 is a 2/3 crop of the 16:9 field: R1 / upstream imx519.c), stored at 1280x720 via one shared resize function used by the recorder and deployment.

Stored resolutions:
- ultrawide at full 1640x1232: it is the offline SLAM and tag input;
- main at a configurable resolution, default 1280x720, JPEG q90. Anything ≥ 640 px wide is plenty for the 224 output,
  and the cap keeps CPU encoding within budget (there is no NVENC).

Aborted episodes are deleted (decided 2026-10-27). Their `events.csv` + `live_checks.json` are kept in
`<session>/aborted/` so failure rates can be tracked.

### 6.2 Live checks: the iPhone's collection bounds, plus ours

| Rule | iPhone source [repo] | Ours | Action |
|---|---|---|---|
| Main fps < 31 after the first 1 s (per frame, 1/dt) | ViewController.swift:1817-1825 | Same rule on main sensor timestamps, same 1 s grace. At 60 fps this means any gap > 32 ms. | **abort** |
| Frame-to-frame pose speed > 7.0 m/s ("typically a SLAM tracking failure") | ViewController.swift:306, 1829-1843 | Same rule on live ORB-SLAM3 at 30 Hz (‖Δp‖/dt) | **abort** |
| Live SLAM tracking lost or reset, a new map created, or a stalled heartbeat (> 0.5 s) | (ARKit equivalent: tracking failure) | from the live SLAM status in shm | **abort** |
| No ultrawide frames | DemonstrationData.swift:574 | Same; also ultrawide fps < 27 over a 1 s window | **abort** |
| (ours) IMU gap > 20 ms, FIFO overflow, or rate < 0.95 x 833 Hz over 1 s | n/a | IMU driver stats | **abort** |
| (ours) Raw writer queue overflow (a frame would be lost) | n/a | StreamHub | **abort** |
| (ours) Free disk < 5 GB | n/a | statvfs | refuse to arm / abort |
| (ours) Thermal: SoC near its throttle point, or clocks throttled | n/a | `/sys/class/thermal`, clock readback | warn; abort if throttling is detected |
| (ours) Finger tags not seen for > 0.5 s | n/a (iPhUMI checks this offline) | low-rate live tag detection for the overlay | warn only |

The thresholds come from the iPhone app as given above. Ours are in `rig.yaml` with these defaults.

The offline `validate` stage re-applies 7 m/s and iPhUMI's 3 m/s `align_validate` rule to the 60 Hz training pose. Live
checks catch problems while the operator can redo the demo; offline checks guarantee that nothing bad reaches training.

### 6.3 Thermal and the live SLAM budget

Live SLAM is the one piece of live work that resembles what overheats the iPhone, so it is budgeted and tested
explicitly:
- **Placement:** ORB-SLAM3 gets 2–3 pinned cores. The input is the 30 Hz ultrawide downscaled to about 800x600, with
  about 1000 features. The viewer refreshes at about 10 Hz.
- **Hardware-day acceptance test:**
  - back-to-back `ec-record` runs for ≥ 15 minutes, live SLAM at 30 Hz, both viewers on, in the chosen `nvpmodel`
    power mode;
  - pass = no throttling, no fps aborts, and live SLAM holding ≥ 28 Hz.
- **Fallbacks, in order:**
  1. the GPU ORB-SLAM3 fork (32 fps on an Orin Nano reported [V]);
  2. live SLAM at 15 Hz. The 7 m/s rule divides by dt, so it stays valid with coarser sampling.

Viewer CPU is optional and is dropped first.

---

## 7. Offline pipeline (`ec-process`)

### 7.1 Package state machine

`manifest.json` holds one record per stage:
- `status`: `done`, `failed` or `skipped`;
- `impl`: which algorithm ran;
- `inputs_hash`: hashes of input files plus the relevant config/calibration keys;
- `outputs`, `tool_version`, `t_start`/`t_end`, `log` path.

A stage re-runs only when its `inputs_hash` changes or `--force` is given. For example, a new gripper calibration
re-runs `width` and everything after it, but not `slam`.

The derived package state is:
`recorded → raw_ok → slam_done → traj_done → posed → width_done → valid|invalid → exported`.

Stages talk only through files in the package, never through memory. That is what makes them swappable.
- In-tree stages are C++ classes registered in a `StageRegistry`, with `impl` selected in config.
- Any external program can also be a stage through a JSON contract:
  `stage_exe --package DIR --config stage.json` writes its outputs plus `result.json`.
  This is how a Python, ROS or other SLAM tool plugs in without touching the pipeline.

### 7.2 Stages

| # | Stage | Input → output | Default impl (alternatives) |
|---|---|---|---|
| 0 | `validate_raw` | raw/ → report | Pre-roll present (`arm` < `rec_start`); monotonic timestamps; drops versus `drops.csv`; durations agree across streams; clock fit residuals; `live_checks.json` is PASS |
| 1 | `slam` | uw frames + IMU (from `arm`, including the pre-roll) + rig calib → refined `W_T_C_uw` per uw frame, plus velocity, IMU biases and tracking state | `orbslam3_mi`: offline ORB-SLAM3 mono-inertial, every frame, initialized on the pre-roll, loop closure + full BA at the end, KannalaBrandt8. Needs a small saver patch to export velocity and biases. (`orbslam3_mi_map`: UMI-style localize in a session map; `openvins`; `external_tum`) |
| 2 | `traj_fit` | stage 1 + IMU → continuous-time trajectory | `imu_preint`: IMU preintegration between refined frames from velocity/biases, endpoint-corrected (`bspline`: cumulative cubic SE(3) B-spline with pose + IMU residuals, Ceres; `slerp`: last resort, < 1 mm at hand speeds) |
| 3 | `pose` | trajectory + main-frame times → 60 Hz TCP pose | Evaluate at each main frame's capture time (mid-exposure + calibrated offset), then `W_T_TCP = W_T_C_uw @ C_uw_T_TCP` in a gravity-aligned y-up frame (ARKit convention), as pos + rotvec. This step is needed in any case: the cameras are not synced, so main frames never coincide with SLAM frames. |
| 4 | `width` | uw frames + `gripper_range.json` → 60 Hz width | Remap uw → virtual iPhone ultrawide (640x480, fx 246.6), then ArUco DICT_4X4_50 16 mm, subpixel, `solvePnP`, `|z−0.098|<0.08`, width = right.x − left.x. Single-tag fallback via the L↔R maps, clip, `−min_width`, linear interpolation to the main frame times. The same math as iPhUMI detect/calibrate/gripper_util. |
| 5 | `validate` | stages 0–4 → valid/invalid + reasons | Pose speed > 7 m/s or > 3 m/s (iPhUMI align_validate); mid-episode pose gap; SLAM tracking loss or map change; width coverage (single-tag/none fraction); trim pose-less ends |
| 6 | `images` | raw frames (`[rec_start, rec_stop]` only) → 224x224 main (60 Hz) + uw (10 Hz sub-stream, nearest frame to a 10 Hz schedule) | `common/imageprep`. Main: native view, letterbox to square, `INTER_AREA` → 224. Uw: remap to the virtual iPhone ultrawide (640x480, fx 246.6), then letterbox → 224. **The same library the deployment uses.** |
| 7 | `export` | stages 3, 4, 6 → `export/episode.zarr`; session level: `<session>.zarr.zip` | iPhUMI keys, dtypes, JPEG-XL level 99 chunks, ultrawide index maps, `episode_ends`, task labels (whole demo = task, as `label_whole_demo_as_task.py`) |

**Note on stage 7:** the zarr is written by a thin Python step that calls the C++ image prep through pybind11 bindings.
The training side reads images through Python `imagecodecs`' JPEG-XL numcodecs codec. Writing those chunks from C++
would mean re-implementing that codec's container and metadata exactly, which risks a subtle incompatibility for no
benefit. Everything upstream of it is C++. (Open decision 1.)

### 7.3 Accuracy budget (training pose)

| Source | Size |
|---|---|
| SLAM itself (offline, full BA) | mm to low cm; UMI offline ORB-SLAM3 ≈ 6 mm / 3.5° [V] |
| Timestamp error | 1 ms ≈ 1 mm at 1 m/s |
| Calibration (`T_cam_imu`, `C_uw_T_TCP`) | goes straight into the pose |
| Fit / evaluation at 60 Hz | < 1 mm; lower with the IMU-constrained fit |

Measured on hardware day with the **arm ground-truth test**:
- Mount the rig on the Franka and run repeatable trajectories.
- Compare against forward kinematics: relative pose error over about 1 s windows, aligned absolute error, tracking
  failures.
- Cases: 30 Hz full FOV (default), 720p60 cropped, 15 Hz decimated, fast motion.
- Also compare live 30 Hz versus offline 60 Hz, to see how often live checks would abort an episode that offline
  would have tracked.

---

## 8. Module map and reuse for deployment

```
physicalSetup/
  common/                      ← shared library `rigcommon` (+ Python bindings `rigpy`)
    time/        clocks, ChipClockMapper
    config/      rig.yaml schema + validation
    drivers/     CameraBackend {argus, gst, mock, replay}, ImuBackend {lsm6dsox_i2c, (lsm6dsox_iio), mock, replay}, Imx519Focus
    capture/     CaptureService, StreamHub
    shm/         seqlock ring bus (publisher/reader)
    camera_model/ pinhole-radtan, fisheye KB/equidistant, double-sphere, VirtualPinhole, RemapTable
    imageprep/   [uw: remap →] letterbox → 224 (bit-exact with the iPhUMI cv_util path)
    package/     package/manifest/session IO, naming, atomic writes
  episode_collection/
    apps/        ec-record, ec-calibrate, ec-slam-live, ec-viz, ec-doctor, ec-imu-probe, ec-cam-probe
    record/      RawWriter, LiveChecks, operator UI
    offline/     ec-process, StageRegistry, stages/{validate_raw, slam, traj_fit, pose, width, validate, images}, export_py/
    configs/     rig_jetson_imx519_imx219w160_lsm6dsox.yaml
  deployment/                  ← later: CaptureService → shm → ObsAssembler → policy → Franka (controlSW/franka_deploy)
  third_party/   ORB-SLAM3 (pinned + patches), Pangolin, CLI11
  docs/          PIPELINE.md, IMPLEMENTATION_PLAN.md, BRINGUP.md
```

| Module | Collection | Deployment |
|---|---|---|
| drivers, time, config, capture, shm | yes | **yes**: identical capture and timestamps |
| camera_model, imageprep (+ bindings) | offline `images` stage | **yes**: identical 224 images, which keeps train and deploy equivalent |
| package, RawWriter, LiveChecks, ec-* apps | yes | no (optional episode logging during evaluation) |
| live SLAM, offline stages | yes | no: pose comes from the Franka, width from the gripper |

---

## 9. Decisions (all resolved 2026-10-27; no open questions)

| # | Decision |
|---|---|
| D1 | Training pose is offline (section 7). Live ORB-SLAM3 at 30 Hz is for live checks + display only. |
| D2 | A live pose failure (jump > 7 m/s, tracking lost) **aborts** the episode, like the iPhone. Aborted episodes are deleted; their logs are kept. |
| D3 | Recording starts automatically once live SLAM is tracking. The arming pre-roll is kept in raw/ for offline SLAM initialization and is never exported. |
| D4 | Hardware: **IMX519** main + **IMX219 160°** ultrawide (the PDF is current; ROADMAP's B0180/B0179 entry is outdated). LSM6DSOX Modulino wired **directly to the Jetson** 40-pin I2C; the Ventuno Q is not used. |
| D5 | Main camera 16:9 at **60 fps**, rig-native view (section 2). Ultrawide remapped to the virtual iPhone ultrawide. |
| D6 | Camera→TCP transform: from the CAD of the mount and fingers (sim-eval finger kit geometry), checked and refined by the arm ground-truth test. |
| D7 | Rig calibration (Kalibr: intrinsics, camera↔IMU extrinsics and time offset) is **deferred to its own discussion** (TODO.md). Until then, live work uses nominal intrinsics + CAD extrinsics marked `provenance: nominal`. Training-grade offline poses wait for it. |
| D8 | Stage 7 export in a thin Python step (zarr + `imagecodecs` JPEG-XL, the libraries training reads with; the encoder is libjxl C++ either way). Everything else is C++. |
| D9 | Jetson: **JetPack 6.2 / L4T 36.4.3** (installed; no apt upgrades of kernel/L4T packages). Data on its **NVMe SSD**. Internet on hardware day. No SSH before then. Visualization on a monitor plugged into the Jetson. |
| D10 | Laptop builds: core C++ only, in the session scratch directory, deleted afterwards. ORB-SLAM3, Pangolin, Ceres and the TUM-VI dataset are **not** built or downloaded on the laptop for now. SLAM code compiles behind CMake options and gets its first real build on the Jetson. |
| D11 | Execution: one executor agent per task; one reviewer per wave over all tasks and their integration; local commit per wave; no push. |
