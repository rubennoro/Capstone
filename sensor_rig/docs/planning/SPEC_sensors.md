# SPEC_sensors: sensor specification for the iPhUMI rig (iPhone 16 Pro reference → Jetson Orin Nano rig)

Status: planning document, 2026-10-16. Nothing here changes repo files.
Inputs: R1 (iPhone export format), R2 (processing pipeline), R3 (deployment path), R4 (Jetson/VIO stack),
`results/block_raise/run_001_2026-10-05/run_config/config.yaml`, `sensor_rig/` (README, docs/calibration.md,
configs/rig_default.yaml, record/HARDWARE_CHECKLIST.md), and new measurements on the 47 real exports in
`data/raw/2026-10-01/` (scripts: `scratchpad/spec/{measure_iphone,noise,tags,wrate,cov}.py`).

Provenance tags used everywhere: **[M]** measured from our data, **[D]** documented (vendor doc, datasheet or our
code), **[L]** literature, **[E]** engineering estimate or derivation (shown), **[?]** unknown, must be measured.

---------------------------------------------------------------------------------------------------

## 0. Summary

- Training and processing really use: a 60 Hz pose/time grid, main images at **20 Hz** (`obs_down_sample_steps: 3`),
  ultrawide (UW) images at **10 Hz**, all at **224x224**; gripper width from 16 mm ArUco tags seen by the virtual UW at
  640x480 (~40 px per tag side); relative actions over a **0.8 s** horizon (16 x 50 ms).
- On the iPhone, image, pose and UW share one ARKit timestamp, so image↔pose sync error is zero by construction.
  ARKit pose jitter is ≤ 0.22 mm / 0.055° rms. Hand motion in the dataset is slow: camera speed p99 0.46 m/s, angular rate
  p99 0.67 rad/s.
- The chosen parts (2x IMX219 with M12 lenses, ICM-42688-P on SPI, offline ORB-SLAM3 mono-inertial) can meet every
  geometric and resolution requirement, with margin, if the lenses are a **2.1 mm rectilinear main** (fx ≈ 938 px at
  1640x1232) and a **~1.55 mm f-theta fisheye UW** (fx ≈ 690 px). Both give coverage 1.000 even with 0.15 mm lens
  decentration plus 3° tilt.
- The real gaps are **temporal**:
  1. The full-FOV mode runs at 30 fps, not 60. The 60 Hz grid is filled with the nearest frame, which gives an image↔pose
     error of up to ±16.7 ms (≤ 7.7 mm at p99 speed). On the iPhone this error is 0.
  2. The current Argus backend stamps frames at appsink delivery, not at start of frame (SOF).
  3. The cameras have no hardware sync.
  4. The rolling-shutter readout is ~23 ms.
  5. Pose accuracy of mono-inertial VIO on this hardware is unproven.
  All of these are measurable on the bench, and items 1 and 2 have concrete mitigations (§5).

---------------------------------------------------------------------------------------------------

## 1. iPhone reference spec (iPhone 16 Pro, `deviceIdentifier` "iPhone17,1", as used for `firstreal`)

Dataset measured: 45 demonstrations + 2 gripper calibrations, 37,971 main frames, mean episode 14.05 s.

### 1.1 Main (wide) camera, ARKit `capturedImage`

| property | value | src |
|---|---|---|
| Hardware | 48 MP "Fusion" camera, 24 mm equiv., f/1.78, 2nd-gen sensor-shift OIS | [D] [Apple specs](https://support.apple.com/121031) |
| Recorded resolution | 1920x1440 (4:3), landscape, YCbCr 420f → H.264 yuvj420p | [M] R1 §6 |
| Intrinsics (pinhole, no distortion coefficients exported) | two variants in one day: fx=fy=**1406.34**, c=(953.25, 723.22) in 39 files; fx=fy=**1278.45**, c=(954.60, 723.53) in 8 files (the first 6 demos + calibrations). ARKit reports K once per view session, and the 10% focal change between sessions is real (ARKit chooses the crop/scale per session). The rig's virtual main uses the 1406 variant. | [M] |
| FOV (1406 variant) | HFOV 68.6°, VFOV 54.2°, DFOV 80.9° (1278 variant: 73.8° x 58.8°) | [M]/[E] |
| Frame rate | 59.97 fps effective. dt median 17 ms, mean 16.674 ms, std 0.47 ms (all of it ms quantization of the JSON strings), max 17.0 ms. | [M] |
| Dropped frames | **0 in 37,971** (no dt > 17 ms in any demo) | [M] |
| Shutter | rolling (all phone CMOS); readout time not published | [D]/[?] |
| Exposure / WB / focus | ARKit defaults: the app sets no camera options (`ARWorldTrackingConfiguration()` + sceneDepth + collaboration), so auto exposure, auto WB, and autofocus (`isAutoFocusEnabled` defaults to true) are on | [D] ViewController.swift:458-560 |
| Stabilization / distortion | ARKit images are not electronically stabilized (EIS would break VIO). OIS is present in hardware; whether ARKit locks it is undocumented. ARKit's pinhole K with zero distortion is treated as exact by the pipeline (pointmaps disabled, so main K is unused downstream). | [D]/[?] |
| Timestamp | `ARFrame.timestamp` (mach uptime, capture time) → UNIX via one offset sampled at record start, stored as ISO ms strings. MP4 PTS (1/600) agree within 1.3 ms. | [D]/[M] R1 §3 |
| `recordingStartTime − poseTimes[0]` | +14 to +51 ms (pipeline latency at start) | [M] |

### 1.2 Ultrawide camera, ARKit private `_capturedUltraWideImage`

| property | value | src |
|---|---|---|
| Hardware | 48 MP, 13 mm equiv., f/2.2, 120° (diagonal, native) | [D] [Apple specs](https://support.apple.com/121031) |
| Recorded resolution | 640x480 rectified pinhole (the pipeline assumes zero distortion) | [M]/[D] |
| Intrinsics | fx=fy=**259.02**, c=(320.29, 239.67) (39 files); 246.65 (8 files) | [M] |
| FOV (259 variant) | HFOV 102.0°, VFOV 85.6°, DFOV 114° (rectified: less than the native 120°) | [E] |
| Rate | **10.00 Hz** effective. Slot spacing in main-frame indices: 6 (6176x), 5 (309x), 7 (284x), 4/8 (3x), 12-13 (10x = dropped UW frames, ~0.15%). Time spacing mean 99.99 ms, std 5.05 ms, p1 83 ms, p99 117 ms, max 134 ms. | [M] |
| Sync to main | The UW image is attached to a main ARFrame and gets the **main frame's** timestamp. The true UW timestamp is discarded, so the error is ≤ ~8.3 ms (half a 60 Hz period), actual value unknown. | [D] R1 §3.2 |
| Focus | locked by ARKit (app comment, DeploymentViewController) | [D] |
| ArUco view (measured on 4 processed demos) | 16 mm tags at depth z = 0.102–0.106 m (median 0.105); tag side **38.8 / 40.5 / 43.4 px** (p5/median/p95), min 14 px; tags sit ~43° off axis (|x| median 55 mm, y ≈ 58 mm); detection rate 100% / 99.8% (id 0 / id 1) | [M] |
| Width calibration | `max_width` 0.1275, `min_width` 0.0431 (tag-x units, m); single-tag linear-map MAE **0.55 mm** | [M] |
| Width rate | |dw/dt| median 0.6 mm/s, p99 0.14 m/s, max 0.20 m/s | [M] |

### 1.3 IMU (internal, used only by ARKit)

| property | value | src |
|---|---|---|
| Part / noise | not published by Apple. Core Motion exposes processed motion to apps at ≤ 100 Hz. ARKit's internal IMU rate and noise are unknown. | [?] |
| Export | **none**. No IMU data in the iPhUMI export. | [D] |
| Sync | internal to ARKit (hardware-timestamped on the same mach clock) | [L]/[?] |

### 1.4 LiDAR (unused)

| property | value | src |
|---|---|---|
| Output | ARKit `sceneDepth` 256x192 float16 (m) at 60 Hz, never repeated in this dataset; preview mp4 | [M] R1 §7 |
| Use | **none**: `include_depth: false`, pointmaps disabled. The rig writes `hasDepth:false`. | [D] |

### 1.5 ARKit pose

| property | value | src |
|---|---|---|
| Quantity | `camera.transform` = world_T_maincam, ARKit axes (+x right, +y up, +z back), gravity-aligned y-up world, metres, float32 | [D] |
| Rate | 60 Hz, 1:1 with main frames (`poseTimes == rgbTimes`) | [M] |
| Image↔pose sync | **0 by construction** (same ARFrame) | [D] |
| Short-term jitter | HF residual vs a 15-frame (250 ms) cubic Savitzky-Golay fit: position **0.22 mm rms** (p95 0.42 mm), rotation **0.055° rms** (p95 0.10°). This is an upper bound on jitter, because it includes real high-frequency motion. | [M] |
| Motion in the data (60 Hz grid) | speed median 0.090, p95 0.280, **p99 0.463**, max 0.855 m/s; angular rate median 0.17, p95 0.45, **p99 0.67**, max 1.19 rad/s; acceleration p99 3.15 m/s² | [M] |
| Displacement over the 0.8 s action horizon | median 76 mm, p95 217 mm, max 508 mm | [M] |
| Drift (literature) | Room/building scale, walking: RPE ≈ 0.02 m/s, final drift 0.02–0.26 m over 50–69 m loops [Kim 2022]. iPhone Pro vs Vicon: relative ATE < 1% of path in 9/10 sequences, rotational RPE < 4° [MobileEgo 2026 preprint]. No published tabletop (< 1 m) evaluation found. | [L] |
| Failure handling | app aborts if fps < 31 after 1 s or speed > 7 m/s; no covariance exported | [D] |

### 1.6 Deployment (iPhone as wrist camera)

| property | value | src |
|---|---|---|
| Stream | AVCaptureMultiCam main + UW, 960x720 JPEG at 30 fps over USB (`iphone_wrist.yaml`) | [D] R3 |
| Latency | measured ≈ 100 ms (iPhUMI docs). Config `camera_latency_s: 0.08`, subtracted twice (`iphone.py:90` + `base_camera.py:221`), so frames are effectively stamped 160 ms early. | [D] R3 §2.5 |
| Pose | none (robot FK) | [D] |

---------------------------------------------------------------------------------------------------

## 2. Spec-agnostic requirements (any replacement rig)

Derivation constants (from §1 unless noted): v99 = 0.46 m/s, vmax = 0.86 m/s, ω99 = 0.67 rad/s, ωmax = 1.19 rad/s,
a99 = 3.15 m/s², width rate p99 0.14 m/s; obs step 50 ms; action horizon 0.8 s; virtual main 1920x1440 (f 1406.34),
virtual UW 640x480 (f 259.02); deploy outputs 960x720 (main f 703.17, UW f 388.52); training 224x224 after
pad-to-square (main f_eff = 1406·224/1920 = **164 px**, UW f_eff = 259·224/640 = **91 px**).
Rule used for the error budgets: a timing error Δt costs v·Δt of position and ω·Δt of rotation. 1 ms ≈ 0.46 mm /
0.04° at p99 motion.

### 2.1 Cameras: rate and timing

| ID | requirement | threshold | derivation | verify |
|---|---|---|---|---|
| REQ-CAM-1 | Main camera frame rate | **≥ 30 fps hard minimum, 60 fps target** | Policy obs are 20 Hz on a 60 Hz grid. With nearest-frame fill at rate F, the image↔pose time error is uniform ±1/(2F). At 30 fps that is ±16.7 ms (rms 9.6 ms): ≤ 7.7 mm / 0.64° at p99, rms 0.9 mm at median speed. That is ≤ 1/3 of a 50 ms obs step, which is acceptable. At 20 fps (±25 ms) it would be half a step, which is not. 60 fps gives 0 error, like the iPhone. | `frames_meta.csv` dt histogram; `fps_measured` within 1% of nominal |
| REQ-CAM-2 | Main frame drops | ≤ 0.1% of frames, no gap > 2 frame periods; live abort < 31 frames/s (fill mode 0.52·F) | iPhone 0 / 37,971. align_validate has no drop check, so gaps hide silently (R2 §5.2). | `drops.csv`, export report |
| REQ-CAM-3 | UW effective rate | **≥ 10 Hz delivered after decimation** (physical ≥ 10 fps; ≥ 20 fps if the UW is also the VIO camera, see REQ-VIO-2) | `ultrawide_down_sample_steps: 1` at 10 Hz; training takes the 2 most recent UW frames, 100 ms apart. Gripper width is interpolated from ~10 Hz detections. | `ultrawideRGBTimes` slot gaps: mostly 6, max ≤ 12 |
| REQ-CAM-4 | UW-to-slot time error | **≤ 8.3 ms** (half a 60 Hz slot), zero mean | Matches the iPhone (UW tagged with the main frame time). Width error ≤ 0.14 m/s · 8.3 ms = 1.2 mm at the p99 width rate, comparable to the 0.55 mm fit MAE. | export slotting report (|t_uw − t_slot| max) |
| REQ-SYNC-1 | Image↔pose time alignment (constant bias, after calibration) | **≤ 3 ms** | 3 ms ≈ 1.4 mm at p99, 6% of a 50 ms step (p99 step motion 23 mm). The iPhone bias is 0. The pipeline does no latency compensation (R2 §5.2), so any bias goes straight into the data. | Kalibr `timeshift_cam_imu` repeatability (two runs agree ≤ 1 ms); pose-import gyro↔|ω| cross-correlation < 3 ms; LED/GPIO test |
| REQ-SYNC-2 | Per-frame timestamp jitter (camera stamp vs true mid-exposure) | **≤ 1 ms rms, ≤ 3 ms max** | VIO can estimate a constant offset but not jitter (OpenVINS/Kalibr model one constant `t_d`). 1 ms at ωmax gives 0.07°, below ARKit jitter (0.055° rms) in practice. | dt std of SOF stamps (expect µs-level); appsink-vs-SOF comparison |
| REQ-SYNC-3 | Camera↔IMU residual offset after Kalibr | **≤ 1 ms** | VIO residual budget: 1 ms · ω99 = 0.04°, and 0.46 mm when converted to position | Kalibr; OpenVINS online `calib_cam_timeoffset` converges to ≤ 1 ms |
| REQ-SYNC-4 | Main↔UW relative timing | each stream stamped on the same clock to REQ-SYNC-2. Hardware trigger **not** required. | The pipeline links them only through slotting (REQ-CAM-4). | skew histogram (`RigCamera` status line: p95 ≤ ½ UW period) |
| REQ-CLK-1 | Clock | one monotonic host clock for all sensors (CLOCK_MONOTONIC ns); UNIX = monotonic + one offset sampled at record start; ms-precision ISO strings | R1 §3.1 | `meta.clock`; validator |
| REQ-CLK-2 | Episode timestamps strictly increasing; `poseTimes[0] == rgbTimes[0]` | exact | R2 latent time-base bug | `sensor_rig validate` |

### 2.2 Cameras: geometry, resolution, image quality

| ID | requirement | threshold | derivation | verify |
|---|---|---|---|---|
| REQ-FOV-1 | Coverage of the virtual main (68.6° x 54.2°) | **≥ 0.99 (hard ≥ 0.98)**, under 0.15 mm lens decentration + 3° mounting tilt | Black borders are a domain shift the policy never saw. In equivalent terms, a rectilinear lens needs f ≤ 1201 px at 1640x1232 (HFOV ≥ 68.6°, VFOV ≥ 54.2°) plus ≥ 3–4° margin per side for M12 centring tolerance (±0.1–0.15 mm ≈ ±2.5–3.7° at f 2.3 mm). | `rig-validate` / `calib-check` coverage after Kalibr |
| REQ-FOV-2 | Coverage of the virtual UW (102° x 85.6°, 114° diag) | **≥ 0.99 (hard ≥ 0.98)**, same tolerances | For an equidistant lens on the IMX219 (half-height 1.38 mm), VFOV is the binding axis: θ_half = 42.8° + 4° margin = 0.817 rad, so f ≤ 1.38 mm / 0.817 = **1.69 mm** (r = f·θ) | same |
| REQ-RES-1 | Main sampling density at the deploy output (960x720, f 703) | physical px/rad ≥ **703** at centre; ≥ 0.9x the virtual density everywhere in the virtual main FOV | Deployment renders 960x720 directly. Training only needs f_eff ≥ 164 px, so the 1920x1440 export may be upsampled (cosmetic). | compute from Kalibr K; `bench-remap` |
| REQ-RES-2 | UW sampling density in the tag region (~35–50° off axis) at the virtual 640x480 | physical px/rad ≥ **1.0x the virtual density** (virtual at 43°: radial 484, tangential 354 px/rad) | ArUco PnP runs on the 640x480 export with dist=0. Tags must keep ≥ the iPhone's ~40 px side and its corner precision. | detect on a rig calibration episode: tag side ≥ 36 px median, detection rate ≥ 95%, `tag_det_threshold` ≥ 0.2 |
| REQ-RES-3 | Tag pixel size anywhere in the gate (z 0.018–0.178 m) | ≥ **20 px side** (DICT_4X4: 6 cells incl. border, ≥ 3 px/cell) | at z = 0.178 m: 259·0.016/0.178 = 23 px, OK | same |
| REQ-IMG-1 | Aspect ratio of delivered virtual images | exactly 4:3 (1920x1440, 640x480; deploy 960x720) | pad-to-square geometry must match training | validator |
| REQ-IMG-2 | Exposure time | **≤ 4 ms** (locked) | Rotational blur at ω99: 0.67 rad/s · 4 ms = 2.7 mrad, i.e. 0.44 px at 224 training res and ~1.9 px on a 690 px/rad fisheye for VIO features. Translation blur at v99, 0.3 m depth: 0.46·0.004/0.3 = 6 mrad, i.e. ~1 px at 224. Tags are rigid to the camera, so only finger motion blurs them (≤ 0.2 m/s·4 ms = 0.8 mm ≈ 2 px at 640x480). | `frames_meta.csv` exposure_ns constant; brightness check |
| REQ-IMG-3 | Rolling-shutter readout (VIO camera) | **≤ 25 ms** full frame; skew at ω99 ≤ 1° top-to-bottom | ORB-SLAM3/OpenVINS do not model RS. 0.67 rad/s · 25 ms = 0.96°. UMI's GoPro (RS) worked at similar motion. | Kalibr RS calibrator (line delay) or LED-bar test |
| REQ-IMG-4 | AE / AWB / focus | **locked** during an episode (fixed exposure, gain, WB gains, focus glued) | ARKit uses auto modes, but VIO and photometric consistency want locks. Deployment frames must look like training frames, so use the same locks at deployment. | cover-half-the-lens test: brightness and colour must not change |
| REQ-IMG-5 | Colour pipeline | sRGB-like, full-range JPEG/H.264 yuvj420p, gamma and WB tuned once against a colour target; deploy path identical to export path | domain gap vs iPhone matters only if datasets are mixed | `test_deploy_equivalence`; ColorChecker ΔE report |
| REQ-MECH-1 | Lens positions | physical main and UW optical centres within **±3 mm** of the iPhone lens positions (rotation is fixed in software) | A remap fixes rotation, not translation. 3 mm at a 0.3 m object depth is 0.57°, i.e. 1.6 px at 224 (f_eff 164). The width scale depends on the camera↔finger geometry. | CAD + `calib-align` residual |

### 2.3 IMU and pose

| ID | requirement | threshold | derivation | verify |
|---|---|---|---|---|
| REQ-IMU-1 | Rate | ODR ≥ **200 Hz** delivered to the VIO (sample at ≥ 1 kHz, anti-alias filter, decimate) | VIO practice; UMI GoPro ≈ 200 Hz [R4] | `rate_measured` within 1% of ODR; no FIFO overflow |
| REQ-IMU-2 | Noise | gyro ≤ **0.01 °/s/√Hz**, accel ≤ **200 µg/√Hz** (raw, before VIO inflation) | consumer-VIO class (BMI088 ~0.014 °/s/√Hz works in PX4/VINS) | Allan variance ≥ 3 h static |
| REQ-IMU-3 | Range | gyro ≥ ±500 °/s, accel ≥ ±8 g | ωmax 68 °/s, a99 0.3 g; margin for knocks | config |
| REQ-IMU-4 | Timestamp | per-sample host time from chip counter + linear clock fit; jitter ≤ **50 µs**, no gaps > 2/ODR, monotonic | Kalibr's most common failure mode is bad IMU stamps | dt histogram; `imu0/aux.csv` |
| REQ-IMU-5 | Mounting | rigid on the same plate as the VIO camera; no foam; known `T_imu_cam` (Kalibr: 2 runs agree ≤ 0.2°, 3 mm) | Kalibr rigid-body assumption | Kalibr repeatability |
| REQ-POSE-1 | Pose rate on the 60 Hz grid | VIO poses ≥ **30 Hz** (interpolated to the grid) | linear-interpolation error ≤ a·dt²/8 = 3.15·(1/30)²/8 = **0.44 mm** at p99 acceleration | compare 30 Hz vs IMU-propagated |
| REQ-POSE-2 | Short-term jitter | position ≤ **0.5 mm rms**, rotation ≤ **0.1° rms** (same HF-residual metric as §1.5) | ≤ ~2x ARKit (0.22 mm / 0.055°); ≤ 5% of median per-step motion (4.5 mm) | the same script on the exported poses |
| REQ-POSE-3 | Relative pose error over 0.8 s | translation ≤ **2% of displacement + 1 mm** (≈ 2.5 mm median, 5 mm p95), rotation ≤ **0.5°** | relative actions span 0.8 s; 2% ≈ the mono-inertial scale error level (ORB-SLAM3 EuRoC mono-inertial: 0.9% [Campos 2021]) | Franka ground truth: mount the rig on the flange, replay demo-like paths at ≤ 0.5 m/s, hand-eye align, compute RPE(0.8 s) |
| REQ-POSE-4 | Per-episode absolute error (SE3-aligned) | ATE ≤ **10 mm**, ≤ **2°** | UMI ORB-SLAM3 benchmark: 6.1 mm / 3.5° mean [Chi 2024]; ORB-SLAM3 TUM-VI room mono-inertial 1.1 cm avg [Campos 2021] | same Franka test; ≥ 10 trajectories |
| REQ-POSE-5 | Tracking availability | ≥ 98% of grid slots posed; no mid-episode gap (export aborts otherwise); trimmed head/tail ≤ 1 s | iPhone: 100% | export report; `pose_gap_mid_episode` count |
| REQ-POSE-6 | Gravity and frame | gravity-aligned within 3° (pose-import check); never exactly identity rotation; ARKit camera axes on export | R2 hazard (NaN in `pose_6d_to_4x4`) | pose-import checks; validator |
| REQ-TCP-1 | Camera↔TCP extrinsic | measured `tcp_pose_in_arkit_frame` for `JetsonRig1,1`, ≤ 1 mm / 0.5° | the current entry is a copy of the iPhone's | `calib-tcp`; Franka hand-eye |

### 2.4 Deployment

| ID | requirement | threshold | derivation | verify |
|---|---|---|---|---|
| REQ-DEP-1 | Capture→publish latency | ≤ **100 ms p99** (iPhone ≈ 100 ms) | Actions are timestamped from the newest obs. 8 of 16 actions (0.4 s) are executed. Latency eats into the executable part. | LED/QR timecode (E9) |
| REQ-DEP-2 | Published timestamp accuracy | true capture time ± **5 ms** in the deployment host's monotonic clock; `camera_latency_s: 0` (latency applied once) | double subtraction in the iPhone path | LED test |
| REQ-DEP-3 | Publish rate | 60 Hz packets (`fps % 20 == 0`, selection tolerance 2/fps = 33 ms); physical main ≥ 30 fps | real-env assertion; trained 20 Hz | BaseCamera FPS line under inference load |
| REQ-DEP-4 | Image equivalence deploy vs export | MAE ≤ 1 level at 224 (same remap code) | already met in D10d (MAE 0.085) | `test_deploy_equivalence` |
| REQ-DEP-5 | Compute headroom | remap p99 ≤ 4 ms per camera; JPEG encode (recording) sustains 2 x 30 fps at 1640x1232 with no `queue_full` drops at the 15 W power mode | Orin Nano has no NVENC [R4] | `bench-remap`, `bench-encode`, `tegrastats` |

---------------------------------------------------------------------------------------------------

## 3. Component specs for the chosen parts

### 3.1 Image sensor: Sony IMX219 (both cameras)

| property | value | src |
|---|---|---|
| Format | 1/4", 3280x2464 active (8.1 MP), 1.12 µm pixels, active area 3.67 x 2.76 mm, diag 4.59 mm, colour RGGB, 10-bit RAW, rolling shutter, MIPI CSI-2 2-lane | [D] [Commonlands IMX219](https://commonlands.com/pages/image-sensors/imx219), [Arducam docs](https://docs.arducam.com/Nvidia-Jetson-Camera/Native-Camera/imx219/) |
| Jetson JP6 modes (in-tree driver, jetson-io "IMX219 Dual") | 3280x2464@21, 3280x1848@28, 1920x1080@30 (crop), **1640x1232@30 (2x2 binned, full FOV, 2.24 µm effective pixels)**, 1280x720@60 (binned + crop: kills VFOV, UW coverage 0.3–0.6) | [D]/[F] R4 §1.2 |
| Sensor capability beyond the Jetson modes | Raspberry Pi libcamera runs 1640x1232 at **41.85 fps** (10-bit) and 83.7 fps (8-bit), 1920x1080 at 47.6 fps, from the same sensor. A custom Jetson DT mode could raise the full-FOV rate to ~40 fps (still not 60). | [D] [libcamera-rs PR](https://github.com/lit-robotics/libcamera-rs/pull/75), [Arducam RPi docs](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/8MP-IMX219/) |
| Line time / readout | mainline driver: pixel rate 182.4 MHz, line length 3448 → **18.9 µs/line**. The binned 1640x1232 mode (VTS 1763 at 30 fps) reads 1232 lines in **≈ 23.3 ms**. The Jetson driver's PLL may differ: measure. | [D]/[E] Linux `imx219.c` |
| Dynamic range / SNR | not in public sources (small 1.12 µm pixel, so expect noticeably more noise than an iPhone at equal exposure; indoor lighting matters) | [?] |
| Exposure/gain control | Argus `exposuretimerange`, `gainrange`, `aelock`, `awblock`, `wbmode` (rig.yaml `argus:` block) | [D] |
| Timestamp | Argus `ICaptureMetadata::getSensorTimestamp()` = SOF in CLOCK_MONOTONIC_RAW (forum evidence) or TSC via `ISensorTimestampTsc`. GStreamer `nvarguscamerasrc` PTS is pipeline running-time, which is what the current backend effectively stamps (appsink delivery, `offset_prior_ns` ≈ −30 ms). | [D]/[F] R4 §1.5, rig_default.yaml |

### 3.2 Camera modules and lenses

Both modules: an M12 (S-mount) IMX219 board with the Raspberry-Pi-v2-compatible pinout (24 MHz clock, works with the
in-tree Jetson driver), connected to J20 (CAM0, 2-lane) and J21 (CAM1) through a **15-to-22-pin 0.5 mm FFC**.

| role | module | lens target | candidate lenses | resulting geometry at 1640x1232 | src |
|---|---|---|---|---|---|
| **main** | Arducam IMX219 M12 board (e.g. B0184/B0183 family, or B0180 board with its lens swapped). Retailers disagree on whether the stock lens is swappable on every SKU, so confirm with Arducam. | **rectilinear (low-distortion) f = 2.1 mm** (acceptable range 2.0–2.3 mm), 1/4" image circle ≥ 4.6 mm, IR-cut | Arducam 1/4" M12 2.1 mm low-distortion (< 1%) lens [Arducam blog table](https://www.arducam.com/blog/?p=11548); Commonlands CIL821 (2.1 mm, 92° H on IMX219) [Commonlands](https://commonlands.com/pages/image-sensors/imx219) | fx ≈ **938 px**; HFOV 82°, VFOV 67°, DFOV 95°. Coverage of the virtual main **1.000** nominal, 1.000 with 0.15 mm decentration + 3° tilt. Density 938 px/rad ≥ 703 (deploy), 1.33x margin. | [E] `cov.py` |
| main (not recommended) | Arducam low-distortion 2.8 mm (70–75° H) or "77°" 2.96 mm modules | — | — | coverage 0.92 / 0.83 (ideal pinhole) | [E] |
| **ultrawide** | Arducam **B0180** (IMX219, M12, ships with a 175° lens, "drop-in for RPi V2 / Jetson") | **f-theta fisheye f ≈ 1.55 mm** (range 1.45–1.69 mm; DFOV 155–180° on 1/4"), image circle ≥ sensor height (2.76 mm) and ideally ≥ diagonal, IR-cut | Commonlands **CIL216** 1.55 mm f/2.2: 145° H x 105° V on IMX219, covers the sensor; The Pi Hut 1.56 mm M12 fisheye (141° on a 1/4" Pi camera); or B0180's stock 175° lens if its VFOV ≥ 92° | fx ≈ **690 px**; coverage **1.000** nominal and with tolerances. Density 690 px/rad vs virtual 484 (radial) / 354 (tangential) at the 43° tag angle: **1.4–1.9x**. Corner of the virtual UW (57°): 0.8x radial (upsampled, cosmetic). Deploy virtual UW (f 388.5) at 43°: 0.95x. | [D] [Arducam B0180](https://www.arducam.com/b0180-arducam-imx219-wide-angle-camera-module-drop-replacement-raspberry-pi-v2-nvidia-jetson-camera.html), [Commonlands](https://commonlands.com/pages/image-sensors/imx219), [Pi Hut](https://thepihut.com/products/m12-lens-185-degree-fisheye-1-2-5-optical-format-1-56mm-focal-length) |
| ultrawide (not recommended) | Waveshare IMX219-160 (fixed, non-M12) | — | — | spec sheet is self-inconsistent (3.15 mm, 160° diag). IR-CUT variant 124° H x **91° V**, which is only 2.7° per side over the 85.6° need, below the 4° margin | [D] [Waveshare wiki](https://www.waveshare.com/wiki/IMX219-160_Camera) |
| ultrawide (not recommended) | ~120° modules | — | — | coverage 0.83 | [E] |

Lens notes:
- Fisheye calibration model: Kalibr `pinhole-equi` = ORB-SLAM3 KannalaBrandt8 = OpenVINS `equidistant`. Wider than
  ~170° fits worse. Keep the useful FOV ≤ ~160° (calibration.md §3 acceptance: UW reprojection < 0.5 px).
- Glue the focus ring after focusing at ~0.3 m (main) and ~0.15 m (UW, so the tags at 0.1 m and the scene are both
  acceptable; depth of field at f/2.2, 1.55 mm is very large).
- Coverage table from `sensor_rig.camera_model.coverage` (nominal | 0.15 mm decentre | 3° tilt | both):
  main 2.1 mm 1.000/1.000/1.000/1.000; 2.3 mm 1.000/1.000/1.000/0.996; 2.5 mm 1.000/0.999/0.983/0.955;
  UW f-theta 1.55 mm 1.000 x4; 1.75 mm 1.000/0.998/1.000/1.000; 1.88 mm (140°) 0.995/0.978/0.986/0.987.

Cables: FFC 22-pin (Jetson side) to 15-pin (module side), 0.5 mm pitch, **≤ 30 cm** (reliable for MIPI at the IMX219's
~0.9 Gbps/lane). Mount the Jetson on the gripper body or forearm. For longer runs use a CSI-over-HDMI extension kit
(adds EMI risk). Route the FFC away from the gripper hinge and strain-relieve it.

### 3.3 IMU: TDK ICM-42688-P (datasheet DS-000347)

| property | value | src |
|---|---|---|
| Gyro FSR | ±15.6 … ±2000 °/s | [D] [TDK product page](https://invensense.tdk.com/products/icm-42688-p/) |
| Gyro noise | **0.0028 °/s/√Hz** @10 Hz (0.028 °/s rms at 100 Hz BW) | [D] DS-000347 rev 1.6, Table 1 |
| Gyro ZRO | ±0.5 °/s initial, ±0.005 °/s/°C | [D] |
| Gyro sensitivity | ±0.5% initial, ±0.005 %/°C, nonlinearity ±0.1%, cross-axis ±1.25% | [D] |
| Accel FSR | ±2 … ±16 g | [D] |
| Accel noise | **65 µg/√Hz** (X/Y), **70 µg/√Hz** (Z) | [D] Table 2 |
| Accel zero-g | ±20 mg initial, ±0.15 mg/°C | [D] |
| ODR | gyro 12.5 Hz–32 kHz, accel 1.5625 Hz–32 kHz; UI LPF 42–3979 Hz at ODR ≥ 1 kHz | [D] |
| FIFO | 2 kB; 20-byte hi-res packets (19/18-bit gyro/accel + 16-bit timestamp), timestamp resolution 1 µs (TMST_RES=0) or 16 µs; RTC/CLKIN input, FSYNC input | [D] |
| Bias instability | **not specified** by TDK. Measure with Allan variance (≥ 3 h). | [D]/[?] |
| Interface | SPI ≤ 24 MHz (rig.yaml uses 10 MHz on `/dev/spidev0.0`, header pins 19/21/23/24); I²C ≤ 1 MHz; I3C | [D] |
| Breakouts | any SPI-exposed ICM-42688-P board, e.g. the open-hardware [SUSTAINA-OP module](https://github.com/SUSTAINA-OP/ICM-42688-P-Module), ~$10–20 | [D] R4 §8 |
| rig.yaml settings | ODR 1 kHz, ±16 g / ±2000 °/s, LPF ODR/4, FIFO watermark 16 packets, 5 ms poll, chip-clock → CLOCK_MONOTONIC fit | [D] |

Margin vs REQ-IMU-2: gyro is 3.6x better than the threshold and accel 3x better. Consider ±8 g / ±500 °/s for finer
resolution. The FIFO hi-res path makes this mostly moot.

### 3.4 Compute: Jetson Orin Nano Super dev kit (JetPack 6.x / L4T 36.x)

| property | value | src |
|---|---|---|
| CPU / GPU | 6x Cortex-A78AE (1.7 GHz Super), 1024-core Ampere, 67 TOPS (Super), 8 GB LPDDR5 (68 GB/s base, 102 GB/s MAXN SUPER) | [D] vendor listings |
| Camera I/O | 8 CSI-2 lanes on the module; dev kit has 2x 22-pin (J20 = CAM0 2-lane, J21 = CAM1 2/4-lane); **no frame-sync lines** on the RPi-style pinout | [D] R4 §1.1, §1.6 |
| Video | **no NVENC** (H.264 encode in software only, ~1080p30 on 1–2 cores). Record JPEG (CPU OpenCV/turbojpeg, 2 workers per camera) or raw to NVMe. NVJPG availability unverified, so run `bench-encode`. | [D]/[?] |
| Power | 7–25 W modes; use 15 W for capture (over-current throttling reports at 25 W) | [F] R4 §7 |
| Timestamps | Argus SOF (MONOTONIC_RAW), TSC; spidev userspace for the IMU | [D] |

### 3.5 Pose software (offline)

| stack | expected accuracy (literature) | notes |
|---|---|---|
| **ORB-SLAM3 mono-inertial** (KB8 fisheye, UW + IMU) | EuRoC avg RMS ATE **4.3 cm**, scale error 0.9%. TUM-VI room1-6 (hand-held fisheye, small room) **1.1 cm** avg (0.6–1.8 cm) [Campos 2021, Tables II, IV]. UMI (GoPro fisheye RS + IMU, map + relocalize): **6.1 mm / 3.5°** mean per gripper vs mocap [Chi 2024] | Needs inertial initialization (wiggle or mapping video). Saves all frames (≥ 30 Hz), which meets REQ-POSE-1. GPLv3. |
| OpenVINS mono+IMU | MSCKF. Online cam-IMU extrinsic and **time-offset** estimation, which is the main value here as a cross-check of REQ-SYNC-1/3 [Geneva 2020] | odometry only, no relocalization; no RS model |

Neither stack models rolling shutter. Accuracy on IMX219 RS at 30 fps with tabletop demos is unproven ([?]), and
the Franka test in REQ-POSE-3/4 is the acceptance gate.

---------------------------------------------------------------------------------------------------

## 4. Compliance matrix

Legend: **P** pass, **G** gap (with mitigation, §5), **M** needs measurement on hardware, **n/a**.

| REQ | threshold | iPhone 16 Pro | chosen rig (2x IMX219 M12: 2.1 mm main + 1.55 mm fisheye UW, ICM-42688-P, ORB-SLAM3) | status |
|---|---|---|---|---|
| CAM-1 main rate | ≥ 30 (60) fps | 59.97 fps [M] | 30 fps full FOV (60 only with crop) | **G1**: meets the minimum, fill error ±16.7 ms |
| CAM-2 drops | ≤ 0.1% | 0 / 37,971 [M] | unknown; JPEG budget dependent | M |
| CAM-3 UW rate | ≥ 10 Hz | 10.00 Hz [M] | 30 fps → decimated 10 Hz | P |
| CAM-4 UW slot error | ≤ 8.3 ms | ≤ 8.3 ms by design [D] | nearest-slot ≤ 8.3 ms given correct stamps | P (with SYNC-2) |
| SYNC-1 image↔pose bias | ≤ 3 ms | 0 by construction | appsink stamps: ~30 ms prior + Kalibr; SOF: few ms | **G2**, M |
| SYNC-2 stamp jitter | ≤ 1 ms rms | sub-ms (ARKit HW clock) [L] | appsink: ms-level jitter [F]; SOF: µs | **G2**, M |
| SYNC-3 cam↔IMU | ≤ 1 ms | internal to ARKit | Kalibr + OpenVINS estimate | M |
| SYNC-4 main↔UW | same clock | shared ARFrame | free-running, no HW sync, per-frame stamps | P (**G3** noted) |
| CLK-1/2 | — | P | P (implemented + validator) | P |
| FOV-1 main coverage | ≥ 0.99 | 1.0 (is the reference) | 1.000 with tolerances [E] | P (verify after Kalibr) |
| FOV-2 UW coverage | ≥ 0.99 | 1.0 | 1.000 with tolerances [E] | P (verify) |
| RES-1 main density | ≥ 703 px/rad | 1406 (export) | 938 px/rad | P |
| RES-2 UW tag density | ≥ 1.0x virtual | 1.0 (reference) | 1.4–1.9x at 43° | P |
| RES-3 tag ≥ 20 px | ≥ 20 px | 38.8 px p5 [M] | same virtual K → same px | P |
| IMG-1 4:3 | exact | P | P (virtual cameras) | P |
| IMG-2 exposure ≤ 4 ms | locked ≤ 4 ms | auto (unknown) [?] | locked 0.1–4 ms in rig.yaml; low light raises gain/noise | M (lighting) |
| IMG-3 RS readout ≤ 25 ms | ≤ 25 ms | unknown [?] | ≈ 23.3 ms [E] | **G4**, M |
| IMG-4 locks | locked | AE/AWB/AF auto | locks supported (Argus) | P (better than iPhone for VIO) |
| IMG-5 colour | tuned once | Apple ISP | Argus ISP default tuning, different look | **G6** (only if mixing data) |
| MECH-1 lens position ±3 mm | ±3 mm | reference | depends on the bracket | M (CAD) |
| IMU-1 rate | ≥ 200 Hz | n/a (not exported) | 1 kHz ODR | P |
| IMU-2 noise | ≤ 0.01 °/s/√Hz, ≤ 200 µg/√Hz | unknown | 0.0028 / 65–70 [D] | P (Allan to confirm) |
| IMU-3 range | ±500 °/s, ±8 g | — | ±2000 °/s, ±16 g | P |
| IMU-4 stamp jitter | ≤ 50 µs | — | chip 1 µs counter + fit (not yet run on HW) | M |
| IMU-5 mount | rigid | — | bracket design | M |
| POSE-1 rate | ≥ 30 Hz | 60 Hz | 30 Hz (+ IMU) | P |
| POSE-2 jitter | ≤ 0.5 mm / 0.1° rms | 0.22 mm / 0.055° [M] | unknown | **G5**, M |
| POSE-3 RPE(0.8 s) | 2% + 1 mm, 0.5° | not measured at tabletop scale (likely P) | literature suggests ~1% scale | **G5**, M |
| POSE-4 ATE | ≤ 10 mm / 2° | [L] < 1% of path | UMI 6.1 mm / 3.5° (rot is marginal) | **G5**, M |
| POSE-5 availability | ≥ 98% | 100% | init failures possible | M |
| POSE-6 gravity/frame | 3° | P | pose-import check | P |
| TCP-1 | 1 mm / 0.5° | iPhone entry (copied from 15 Pro) | `JetsonRig1,1` is a placeholder copy | **G7** |
| DEP-1 latency | ≤ 100 ms | ≈ 100 ms | est. 25–45 ms SOF→delivery + ~2 ms remap + IPC | M (likely P) |
| DEP-2 stamp accuracy | ±5 ms | 160 ms effective error (double subtraction) | capture time, latency 0 | P after SOF (G2) |
| DEP-3 publish 60 Hz | 60 Hz | 30 fps config (trips `fps % 20`) | `fixed_rate` 60 Hz (duplicates 30 fps) | P |
| DEP-4 equivalence | MAE ≤ 1 | — | 0.085 (D10d) | P |
| DEP-5 compute | remap ≤ 4 ms, no encode drops | — | laptop remap 1.6 ms; Orin unknown | M |

---------------------------------------------------------------------------------------------------

## 5. Gaps and mitigations (ranked by impact on data quality)

**G1. Main camera 30 fps (full FOV) vs the 60 Hz grid.** *Decision needed.*
Impact: image↔pose time error uniform within ±16.7 ms (rms 9.6 ms, zero mean): ≤ 7.7 mm / 0.64° at p99 motion,
0.9 mm rms at median motion. The iPhone has 0. Options:
(a) Accept (current `export/grid.py` nearest fill). It's within REQ-CAM-1, and ColorJitter/RandomCrop dominate anyway.
(b) Custom Jetson sensor mode 1640x1232 at ~40 fps (the sensor does 41.85 fps on the Pi). That cuts the error to ±12.5 ms.
    It needs a device-tree mode table edit, and 40 is not a divisor of 60, so fill remains.
(c) Interpolate the pose at each **image** time instead of the slot time (the image-consistent label). This removes
    image↔pose error but makes action spacing jitter 33–67 ms. Not recommended.
(d) Different sensor for main (e.g. IMX477/IMX296 on a Pi 5, or a 60 fps 4:3 sensor). Changes the parts.
Recommendation: (a) now, and measure in a pilot whether policy quality differs. Revisit (b) only if needed.

**G2. Camera timestamps: appsink delivery vs SOF.** *No part change; software work.*
The current Argus/GStreamer backend stamps delivery time with a −30 ms prior. That prior is a guess, and the jitter is
ms-level, so it fails REQ-SYNC-1/2. Kalibr/OpenVINS remove only the constant part.
Mitigation: the ~200-line C++ libargus helper emitting `getSensorTimestamp()` (SOF, MONOTONIC_RAW → MONOTONIC via
`clock.RawToMono`). Add exposure/2 + readout/2 for mid-exposure (≈ +2 + 11.7 ms). Then verify with the LED/GPIO test
(HARDWARE_CHECKLIST) and Kalibr. Expected result: µs jitter, residual ≤ 1–2 ms. Optionally tie exposure to IMU time
through the ICM-42688-P FSYNC pin, if a module exposes XVS/strobe (stock RPi-style IMX219 boards don't).

**G3. No hardware sync between the two CSI cameras.** *Accept.*
They free-run at 30 fps with a slowly drifting phase. The pipeline only needs UW frames slotted within 8.3 ms
(REQ-CAM-4), the same as the iPhone, and VIO uses the UW alone. Not a blocker with per-frame SOF stamps.
Don't build stereo VIO from these two cameras.

**G4. Rolling shutter ~23 ms on the VIO camera.** *Measure; part change only if VIO fails.*
Skew is 0.9° at ω99 and 1.6° at ωmax. Mitigations: exposure ≤ 4 ms, ORB-SLAM3 on the fisheye (small px/rad), demos
without wrist flicks (the current data already has ω99 0.67 rad/s), Kalibr RS line-delay measurement for the record.
If POSE-3/4 fail: global-shutter VIO camera (IMX296 community driver, or a hardware-synced stereo+IMU module such as
RealSense D435i / OAK-D with cuVSLAM stereo-inertial).

**G5. Pose accuracy and initialization of mono-inertial VIO.** *Biggest unknown; gate with a Franka test.*
Literature: 1.1 cm (TUM-VI room), 6.1 mm / 3.5° (UMI). ARKit jitter is 0.22 mm. Plan:
1. Record a mapping video, then demos; use the UMI-style map + relocalize flow.
2. Start each demo with a short wiggle for inertial init.
3. Mount the rig on the Franka flange, replay ≥ 10 demo-like trajectories (≤ 0.5 m/s), hand-eye align, and report
   jitter, RPE(0.8 s) and ATE against REQ-POSE-2..4.
4. Cross-check the time offset with OpenVINS.
If the rotation ATE fails at > 2°, try the IMU-heavier settings, a higher camera rate, or the G4 part change.

**G6. ISP / colour / noise differences vs the iPhone.** *Matters only when mixing datasets.*
IMX219 1.12 µm pixels with a 4 ms exposure cap need ≥ ~500 lux on the workspace. Lock AE/AWB/gain and tune WB once
with a grey card or ColorChecker. Keep export and deploy on the same remap and colour path (already done). Train
rig-only, or rig + iPhone with ColorJitter, and measure.

**G7. Mechanical: TCP extrinsic and lens positions.** *Must be measured.*
`gripper_calibration.yaml` `JetsonRig1,1` is a copy of the iPhone entry. Measure it with `calib-tcp` / Franka hand-eye
(REQ-TCP-1). Place the lens centres within ±3 mm of the iPhone positions (REQ-MECH-1). Re-record a rig
gripper-calibration episode, because the width scale changes with geometry.

**G8. Lens tolerances / availability.** *Procurement.*
Retailers disagree on M12 lens swap-ability per Arducam SKU. Buy B0180 (M12, swappable) for both, plus the two
lenses. Avoid lenses > 2.3 mm for the main (2.5 mm drops to 0.955 under tolerances) and > 1.75 mm for the UW. Verify
coverage with `calib-check` right after Kalibr.

**G9. Compute / encode budget and thermal.**
No NVENC, so JPEG on the CPU, 2 workers per camera at 1640x1232@30. Run `bench-encode` and `bench-remap` on the Orin
in the 15 W mode with the fan. Fallbacks: turbojpeg, Q85, or UW at 15 fps (still ≥ 10 Hz).

**G10. IMU driver path untested.**
The userspace spidev backend and chip-clock fit have unit tests only. Run the checklist: no 65.5 ms timestamp wraps,
rate within 1%, 3 h Allan variance.

Decisions for the user: G1 (accept 30 fps fill vs custom mode vs different main sensor), the G2 work item (write the
libargus SOF helper before the first real dataset), and whether to buy the G4 fallback before or after the Franka
pose test.

---------------------------------------------------------------------------------------------------

## 6. Recommended part list (sensors only)

| qty | part | why | approx. |
|---|---|---|---|
| 2 | Arducam **B0180** IMX219 M12 module (RPi-v2-compatible pinout) | in-tree JP6 driver; M12 lens freedom | $15–25 each |
| 1 | M12 **2.1 mm low-distortion** lens, 1/4", IR-cut (Arducam 1/4" 2.1 mm < 1% series or Commonlands CIL821) | main: coverage 1.000 with tolerances, 938 px/rad | $10–30 |
| 1 | M12 **~1.55 mm fisheye** (Commonlands CIL216 1.55 mm f/2.2, or the Pi Hut 1.56 mm), IR-cut | UW + VIO: 145° x 105° on IMX219, coverage 1.000 | $15–40 |
| 2 | 15→22-pin 0.5 mm FFC, 15–30 cm | CSI to J20/J21 | $3–6 each |
| 1 | ICM-42688-P SPI breakout (SPI pads exposed; INT1 and FSYNC broken out preferred) | VIO IMU, 0.0028 °/s/√Hz | $10–20 |
| 1 | NVMe SSD 256–512 GB | JPEG logging | $25–40 |
| 1 | Aprilgrid 6x6, 30 mm tags, rigid board; grey card / ColorChecker | Kalibr; colour lock | $20–40 |
| 1 | 3D-printed or aluminium rigid plate holding both cameras + IMU, lens centres at the iPhone positions | REQ-MECH-1, REQ-IMU-5 | — |
| (fallback) | IMX296 global-shutter module, or RealSense D435i / OAK-D | only if G4/G5 fail | $50–330 |

---------------------------------------------------------------------------------------------------

## 7. References

Scholarly:
- Campos C., Elvira R., Gómez Rodríguez J. J., Montiel J. M. M., Tardós J. D. "ORB-SLAM3: An Accurate Open-Source
  Library for Visual, Visual–Inertial, and Multimap SLAM." *IEEE Transactions on Robotics* 37(6):1874–1890, 2021.
  https://doi.org/10.1109/TRO.2021.3075644 (Tables II–IV used: EuRoC mono-inertial 0.043 m, 0.9% scale; TUM-VI room
  mono-inertial 0.011 m avg.)
- Kim P., Kim J., Song M., Lee Y., Jung M., Kim H.-G. "A Benchmark Comparison of Four Off-the-Shelf Proprietary
  Visual–Inertial Odometry Systems." *Sensors* 22(24):9873, 2022. https://doi.org/10.3390/s22249873 (ARKit RPE
  0.02–0.04 m/s, final drift 0.02–0.26 m on 50–69 m loops.)
- Geneva P., Eckenhoff K., Lee W., Yang Y., Huang G. "OpenVINS: A Research Platform for Visual-Inertial Estimation."
  *2020 IEEE ICRA*. https://doi.org/10.1109/ICRA40945.2020.9196524
- Chi C., Xu Z., Pan C., et al. "Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild
  Robots." RSS 2024; arXiv:2402.10329, https://arxiv.org/abs/2402.10329 (SLAM vs mocap: 6.1 mm / 3.5° mean.)
- "MobileEgo Anywhere: Open Infrastructure for long horizon egocentric data on commodity hardware." arXiv preprint
  2605.05945 (2026), https://arxiv.org/abs/2605.05945 (ARKit vs Vicon; not peer reviewed.)

Vendor / documentation:
- Apple iPhone 16 Pro tech specs: https://support.apple.com/121031
- TDK ICM-42688-P product page https://invensense.tdk.com/products/icm-42688-p/ and datasheet DS-000347 (rev 1.6
  used; current 1.9).
- Sony IMX219 summary: https://commonlands.com/pages/image-sensors/imx219 ; Arducam IMX219 Jetson docs:
  https://docs.arducam.com/Nvidia-Jetson-Camera/Native-Camera/imx219/
- Linux `drivers/media/i2c/imx219.c` (timing constants): https://gbmc.googlesource.com/linux/+/refs/heads/dev-6.6/drivers/media/i2c/imx219.c
- Arducam B0180: https://www.arducam.com/b0180-arducam-imx219-wide-angle-camera-module-drop-replacement-raspberry-pi-v2-nvidia-jetson-camera.html ;
  Arducam low-distortion lens table: https://www.arducam.com/blog/?p=11548
- Waveshare IMX219-160: https://www.waveshare.com/wiki/IMX219-160_Camera
- RPi IMX219 mode list: https://github.com/lit-robotics/libcamera-rs/pull/75
- Jetson/Argus/IMU/VIO forum and vendor links: see R4 (`scratchpad/research/R4_jetson_vio_stack.md`).
