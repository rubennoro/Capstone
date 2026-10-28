# physicalSetup implementation plan

Status: plan, 2026-10-27. Design: [PIPELINE.md](PIPELINE.md); all decisions are in its section 9, with no open
questions. Sensors arrive **2026-10-29** (hardware day, "H").

## 0. Ground rules

- **Execution:** one **executor agent per task**. One **reviewer agent per wave** reviews every task in the wave
  *and how they connect* (interfaces, end-to-end flows), then the gate (section 7). After the gate, **one local commit
  per wave** on `rnoro/jetson-sensors`. The user does all pushing.
- **Interfaces first.** Wave 0 (T0) freezes the shared headers. Wave 1 and 2 executors code against them in
  parallel, and may change a contract only through a note to the orchestrator, never silently.
- **C++17**, CMake ≥ 3.22, warnings as errors (`-Wall -Wextra -Wpedantic -Werror`), clang-format file in T0.
  Namespace `rig::`. GoogleTest. Builds must work with both gcc 9 / OpenCV 4.5.4 (laptop) and gcc 11 / OpenCV 4.8
  (JetPack 6.2). The ArUco API differs between those versions; `#if CV_VERSION_*` guards are tested on the laptop side
  and compiled on the Jetson.
- **Laptop build policy (D10):**
  - Build trees live only in the session scratch directory `$SCRATCH/build/<task>`, where `$SCRATCH` =
    `/tmp/claude-1000/-home-ruben-Capstone/024e666a-499e-4cbd-801d-deb863b2bcb1/scratchpad`. They are deleted after
    each wave.
  - Only libraries already on the laptop are used: OpenCV, Eigen, yaml-cpp, spdlog, nlohmann_json, gtest, GStreamer,
    libjpeg. CLI11 is vendored as one header.
  - **No** ORB-SLAM3, Pangolin, Ceres or TUM-VI on the laptop. SLAM code sits behind `RIG_WITH_ORBSLAM3=OFF` and is
    first compiled on the Jetson.
  - Small pip venvs for Python golden tests (opencv-python-headless, numpy, zarr, imagecodecs, pybind11) go in
    `$SCRATCH/venv`, about 300 MB.
  - Optional: libargus *headers* may be extracted from the L4T r36.4 multimedia-API package into `$SCRATCH`, for a
    compile-only check of the Argus backend.
- **No changes outside `physicalSetup/`**, except `TODO.md`, `AGENTS.md` and `ROADMAP.md` status lines. Shared iPhUMI,
  real-env and sensor_rig code is read-only; the regression guard in `scripts/regression/` must therefore stay green
  trivially. The gate checks this with `git diff --name-only`.
- **Sources of truth** that executors must cite in code comments (file:line or document section):
  - PIPELINE.md;
  - iPhUMI Swift/Python for every iPhone-parity rule;
  - for the LSM6DSOX, ST's BSD-3 `lsm6dsox-pid` register header + datasheet DS12814 + AN5272;
  - for libargus, the Jetson Multimedia API reference r36.4.
- **Rigor for sensor integration** (T1, T2, T3, T5, T8): every hardware-facing value (register address, bit field,
  scale factor, Argus call, timestamp domain) is cited, unit-tested against a simulator, and listed in BRINGUP.md
  with the on-hardware command that verifies it on day H.

## 1. Timeline

| When | Work |
|---|---|
| Day 0 (2026-10-27/28) | Wave 0: T0 contracts + build skeleton (one executor), reviewed, committed |
| Day 1 (2026-10-28) | Wave 1: T1 IMU, T2 cameras, T3 capture/shm/package, T4 camera model + image prep (4 executors in parallel) → reviewer R1 → gate → commit |
| Day 2 (2026-10-28/29) | Wave 2: T5 ec-record, T6 ec-viz + ec-slam-live, T7 ec-calibrate, T8 bring-up kit (4 executors) → reviewer R2 → gate → commit |
| **H (2026-10-29+)** | Hardware bring-up per BRINGUP.md (section 5), with the user at the Jetson. Fix tasks are spawned as needed. |
| H+1 … | Wave 3: offline pipeline (T9–T12). Wave 4: calibration tooling + arm test (after the Kalibr discussion). Wave 5: deployment scaffold. |

**Must-have for H:** T0–T5, T8, the cameras mode of T6, and T7. Live ORB-SLAM3 (T6) is complete in code with a mock
backend; its first real build is on H.

## 2. Wave 0: contracts

### T0. Build skeleton and frozen interfaces (executor)

**Deliverables:**
- `physicalSetup/CMakeLists.txt` with options `RIG_WITH_ARGUS`, `RIG_WITH_GST`, `RIG_WITH_ORBSLAM3`, `RIG_WITH_CERES`,
  `RIG_WITH_PYTHON` and `RIG_SANITIZE={address,undefined,thread}`.
- `common/`, `episode_collection/` and `deployment/` subdirectories with their own CMakeLists.
- `.clang-format`, `third_party/CLI11.hpp`, `README.md`.

**Frozen headers** in `common/include/rig/`:
- `types.hpp`:
  - `using TimeNs = int64_t;` (CLOCK_MONOTONIC_RAW).
  - `struct ImageFrame { TimeNs t_sof_ns, t_mid_ns, t_arrival_ns; uint64_t seq; int width, height; PixelFormat fmt; std::shared_ptr<const uint8_t> data; size_t stride; FrameMeta meta; }`, where `FrameMeta` = exposure_ns, analog_gain, digital_gain, readout_ns, sensor_mode, focus_position.
  - `struct ImuSample { TimeNs t_ns; int64_t chip_ticks; Eigen::Vector3d gyro_rps, acc_mps2; float temp_c; }`.
  - `Status` / `Result<T>`.
- `drivers/camera_backend.hpp`: `class CameraBackend { start(const CameraConfig&); stop(); read(timeout) -> std::optional<ImageFrame>; stats() -> CameraStats; modes() -> std::vector<SensorMode>; }`.
- `drivers/imu_backend.hpp`: `class ImuBackend { start(const ImuConfig&); stop(); read(std::vector<ImuSample>&, timeout) -> size_t; stats() -> ImuStats; }`.
- `drivers/i2c_bus.hpp`: `class I2cBus { write_read(addr, const uint8_t* w, size_t wn, uint8_t* r, size_t rn); write(addr, ...); }`. The interface lets T1 test against a simulator.
- `capture/stream_hub.hpp`: subscriber registration, `Lossless` versus `DropOk` policy, stats.
- `shm/ring.hpp`: `ShmRingWriter` / `ShmRingReader`, fixed slots, seqlock, header {magic, version, slot_size, n_slots, stream_kind, write_seq}.
- `shm/status_channel.hpp`: small fixed struct channels for recorder state and SLAM status
  `{t_ns, tracking_state, map_id, pose[7], heartbeat_seq}`.
- `config/rig_config.hpp`: structs mirroring `rig.yaml`, with `load(path) -> Result<RigConfig>` (implemented in T3).
- `package/package.hpp`: `Session`, `Package`, `Manifest`, `EventLog` (implemented in T3).
- `slam/slam_backend.hpp`: `class SlamBackend { init(settings); track(const ImageFrame&, std::span<ImuSample>) -> SlamStatus; shutdown(); }`.

Also:
- `episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml`: a complete example with every key
  documented. It uses nominal intrinsics (lens-spec KB model for the ultrawide, pinhole for the main) and CAD
  extrinsics, each marked `provenance: nominal` until calibration.
- A CI-style script `physicalSetup/scripts/check.sh <build_dir> [--sanitize X]`: configure, build, ctest. The gate uses it.

**Acceptance:** an empty-implementation build passes on the laptop; headers compile standalone; the orchestrator
reviews and freezes them.

## 3. Wave 1: foundations (4 executors in parallel)

### T1. LSM6DSOX IMU driver + register-level simulator

**Owns:** `common/drivers/imu/lsm6dsox_*`, `common/drivers/i2c/linux_i2c_bus.*`, `tests/imu/*`,
`episode_collection/apps/ec-imu-probe`.

**Register map and protocol:**
- `lsm6dsox_regs.hpp`, from ST's BSD-3 `lsm6dsox-pid` header (vendored with its license), plus the datasheet. Covers
  WHO_AM_I (expect 0x6C), CTRL1_XL, CTRL2_G, CTRL3_C (BDU, IF_INC, SW_RESET), CTRL10_C (TIMESTAMP_EN), FIFO_CTRL1–4,
  FIFO_STATUS1/2 (level, watermark, overrun), FIFO_DATA_OUT_TAG + 6 data bytes, TIMESTAMP0–3, INTERNAL_FREQ_FINE.
- Configuration, defaults in rig.yaml:
  - ODR 833 Hz for accel and gyro; accel ±8 g; gyro ±2000 dps;
  - FIFO in continuous mode, timestamp batching every sample, watermark about 50 % of the FIFO.
- Startup sequence: probe WHO_AM_I → soft reset → wait → configure. Read back every written register and verify it.
- Read path: poll FIFO_STATUS every `poll_ms` (10 ms) on an absolute-time loop; burst-read N×7 bytes, verifying
  address wrap-around per the datasheet and ST driver; decode tags (gyro, accel, timestamp; others ignored).
- Each IMU sample is paired with the most recent timestamp word. Scale to SI: g = 9.80665, rad/s. Detect saturation.
- Detect overrun, unknown tags, I2C errors and timestamp regressions, and count each in `ImuStats`. An overrun is
  reported upward and is an abort rule in T5.

**Clock:**
- Read the 32-bit TIMESTAMP registers bracketed by host clock reads (`t0`, `ts`, `t1`) every 100 ms.
- `ChipClockMapper` (shared with T3's time module): lower-envelope or robust linear fit of host versus chip ticks over
  a sliding window, nominal 25 µs/LSB with the trim applied, configurable slope clamp.
- Output: `t_ns` in MONOTONIC_RAW, fit residual stats, and `clock_fit.json`.

**`LinuxI2cBus`:** `/dev/i2c-N` with `I2C_RDWR` combined transactions (repeated start); retries with a counted error
budget.

**`Lsm6dsoxSim : I2cBus`:** a register-file simulator for tests:
- a real FIFO with tags, ODR timing driven by a virtual clock, timestamp counter with configurable drift (±2 %) and
  offset;
- overrun behaviour, injected I2C NACKs and errors, and a WHO_AM_I mismatch case;
- synthetic motion (sinusoids plus noise) so that scale and axes are checked.

**`ec-imu-probe`:** for H. Prints WHO_AM_I, the config readback, live sample rate, FIFO level max, overruns, clock
residuals, and mean/std of accel and gyro at rest (gravity magnitude check 9.81 ± 0.2). Option `--record N s` writes
EuRoC CSV.

**Tests:**
- decode of every tag type;
- scale factors for every full-scale setting;
- 60 s simulated run at 833 Hz: zero loss, rate within 0.1 %, timestamps monotonic;
- clock mapping error < 50 µs with 2 % drift and jitter;
- overrun detected and counted;
- I2C error recovery;
- reset and readback verification.

TSan run of the driver thread.

**Acceptance:** all tests pass under ASan, UBSan and TSan. Every register constant cites its source. A BRINGUP.md
snippet (handed to T8) lists the exact `i2cdetect` / `i2cget` / `ec-imu-probe` commands and expected outputs.

### T2. Camera backends (Argus primary, GStreamer fallback, mock, replay) + IMX519 focus lock

**Owns:** `common/drivers/camera/*`, `tests/camera/*`, `episode_collection/apps/ec-cam-probe`.

**`ArgusCamera`** (behind `RIG_WITH_ARGUS`):
- Setup: `CameraProvider` → device by `sensor_id` → choose `ISensorMode` by W×H and fps (fail loudly if absent; list
  the available modes).
- Controls:
  - `setFrameDurationRange` fixed at 1/fps;
  - AE lock and AWB lock;
  - exposure range min = max, gain range min = max, ISP digital gain 1–1;
  - denoise and edge enhancement off, so SLAM and tags see raw texture. Configurable.
- Capture: EGLStream `IFrameConsumer`; copy to an NV12 CPU buffer via `NvBufSurface` / `NvBufSurfTransform`. The
  JPEG writer encodes from YUV directly, so there is no BGR conversion on the hot path.
- Metadata: `ICaptureMetadata::getSensorTimestamp()` → `t_sof_ns`, plus `getSensorExposureTime`,
  `getSensorAnalogGain` and `getFrameReadoutTime`. `t_mid = t_sof + exposure/2 + readout/2`, with the formula's
  assumption cited and an H verification step.
- Drops: detect from sequence and timestamp gaps; reason logged.
- Optional: a compile-only check against extracted r36.4 headers.

**`GstCamera`** (fallback): `nvarguscamerasrc sensor-id=N sensor-mode=M aelock=true awblock=true exposuretimerange="e e" gainrange="g g" ispdigitalgainrange="1 1" ! nvvidconv ! video/x-raw,format=NV12 ! appsink`.
- Timestamps: buffer PTS + pipeline base time, converted to MONOTONIC_RAW. Documented as about 30 ms late and jittery
  (fallback only).
- **Tested on the laptop** with `videotestsrc` substituted through the config (`gst_source_override`).

**`MockCamera`:** deterministic patterns, configurable fps, jitter, drops and stalls.
**`ReplayCamera`:** from a package's `raw/cam_*` or any EuRoC-style folder (real-time paced or as fast as possible).
These two drive all of T5's tests.

**`Imx519Focus`:**
- set a fixed lens position at start: VCM at I2C 0x0C on the camera's muxed bus (bus number from config), with the
  register write sequence cited from Arducam's `Focuser.py`;
- if the address is kernel-bound (`UU`), fall back to a V4L2 `focus_absolute` control on the sensor subdev if present;
- otherwise log `focus: fixed-at-power-up` and record it;
- the position is written into frame meta and the package.

**`ec-cam-probe`:** for H. Lists sensors and modes, captures N seconds per camera, and reports fps, inter-frame jitter
(p50/p99/max), drops, `t_arrival − t_sof` latency distribution, exposure and gain readback (confirms the locks),
the focus status, and saves sample frames.

**Tests:** mock/replay timing and drop detection; GStreamer path with `videotestsrc`; mode selection logic (table
driven, using Arducam's published IMX519/IMX219 mode tables); timestamp math. ASan, UBSan, TSan.

**Acceptance:** all backends behind one interface; the GStreamer path runs on the laptop; Argus compiles where headers
exist; BRINGUP snippet with `nvargus-daemon` / `v4l2-ctl --list-formats-ext` / `ec-cam-probe` commands and expected
outputs.

### T3. Time, config, capture service, StreamHub, shm bus, package format

**Owns:** `common/time/*`, `common/config/*`, `common/capture/*`, `common/shm/*`, `common/package/*` and their tests.

- **time:** `now_raw_ns()`, `now_mono_ns()`, `now_unix_ns()` and session anchor sampling. Generic `ChipClockMapper`,
  used by T1.
- **config:**
  - `rig.yaml` loader with full validation: every key typed and range-checked; errors carry the YAML path, for example
    `cameras.uw.mode: 1280x720@60 is a cropped mode on imx219; full-FOV modes are ...`;
  - unknown keys are errors;
  - a stable content hash for the session freeze.
- **capture:**
  - `CaptureService` owns the backends: one thread per sensor, pinned cores and priorities from config;
  - steady-state detection (N frames at nominal rate per stream);
  - fans out to `StreamHub`, which has per-subscriber bounded queues with the `Lossless` policy (overflow → an event,
    escalated by T5's checks) or the `DropOk` policy (counted);
  - stats are published to a status channel.
- **shm:**
  - POSIX shm (`shm_open` + `mmap`), fixed-slot seqlock ring, one writer and many readers;
  - readers detect overrun by sequence; versioned header; clean unlink on exit plus stale-segment recovery;
  - frame rings carry a downscaled or native frame (configurable per consumer);
  - IMU ring; status channels.
- **package:**
  - `Session::create/open` (`session.json`, frozen `rig.yaml`, calibration dir);
  - `Package::create(kind, side)` with exact iPhUMI naming (`<ISO>_<5 digits>_<session>_<kind>_<side>`, per
    ViewController.swift:1092-1100);
  - directory layout per PIPELINE 6.1;
  - `Manifest` (JSON, schema version, stage records, atomic write = temp + fsync + rename + dir fsync);
  - `EventLog` (`events.csv`); CSV writers;
  - `discard()`, which moves `events.csv` + `live_checks.json` into `aborted/` and deletes the rest;
  - a `validate_layout()` used by tests and by offline stage 0.

**Tests:** config validation (every error path); multi-process shm test (writer and readers in forked processes,
overrun, crash of the writer); seqlock under TSan; manifest atomicity (simulated crash between steps); naming parity
with iPhUMI's parser (the Python regex from `group.py` / `generic_util.py` is run in the golden test venv).

**Acceptance:** sanitizers clean; `CaptureService` + `MockCamera` + `Lsm6dsoxSim` integration test (60 s, zero
lossless drops) once T1 and T2 land. The reviewer runs it.

### T4. Camera models + image prep (+ Python bindings)

**Owns:** `common/camera_model/*`, `common/imageprep/*`, `common/python/*` (pybind11), tests.

- **Models:**
  - PinholeRadtan; KannalaBrandt (equidistant KB4, the model ORB-SLAM3 uses for fisheye);
  - DoubleSphere and EUCM (for Kalibr wide-angle);
  - VirtualPinhole.
  - Each has project, unproject and their Jacobians where needed, ported from `sensor_rig/sensor_rig/camera_model.py`
    with cited lines.
- **`RemapTable`:** physical → virtual pinhole maps (`cv::remap`, `INTER_LINEAR`), plus a coverage report (the
  fraction of virtual pixels that see real sensor pixels).
- **`ImagePrep`:**
  - `letterbox_square` + `resize(224, INTER_AREA)`, matching iPhUMI `common/cv_util.py:5-34` exactly (pad colour,
    placement, rounding);
  - main = letterbox only (D5); uw = remap to the virtual iPhone ultrawide (640x480, fx 246.6, cx/cy from an iPhone
    JSON), then letterbox.
- **Bindings:** `rigpy.imageprep` and `rigpy.camera_model`, used by stage 7 and later by deployment.

**Golden tests:**
- Python cv2 (venv) on real iPhone frames from `data/raw/2026-10-01/…firstreal…` (decoded mp4 frames) versus C++
  output: **bit-exact**, or a documented ≤ 1 LSB with the reason (OpenCV version);
- model round-trip and projection versus the Python `camera_model.py` on random points, to 1e-9;
- coverage of the 160° nominal model onto the iPhone ultrawide = 100 %.

**Acceptance:** golden tests pass; the bindings import in the venv and reproduce the same bytes.

**Reviewer R1** (after T1–T4): reads all four diffs against T0 contracts and PIPELINE.md, and does the following.
- Clean build in `$SCRATCH/build/r1`; all tests plus the ASan, UBSan and TSan configurations.
- Integration: `CaptureService` with Sim IMU + Mock cameras → `StreamHub` → shm → a reader, 5 min, checking rates,
  timestamps and zero lossless drops.
- **Independent check of the IMU register configuration**: derive register values from the datasheet and AN5272
  without looking at T1's code, then diff.
- Hardware-facing values all cited; BRINGUP snippets complete.
- Findings go back to the owning executors (resumed, not respawned). The reviewer re-verifies, then the gate.

## 4. Wave 2: programs (4 executors in parallel)

### T5. `ec-record`: RawWriter, LiveChecks, supervisor, operator UI

**Owns:** `episode_collection/record/*`, `apps/ec-record`, tests.

- **RawWriter:**
  - JPEG encoder pool (libjpeg, encoding from YUV planes, q90 by default; uw at full 1640x1232, main at the configured
    storage size);
  - `data.csv` / `frame_meta.csv` / `drops.csv` per camera; IMU EuRoC CSV + `chip.csv` + `clock_fit.json`;
  - writes start at **arm** (pre-roll), with bounded memory;
  - fsync and manifest `recorded` on stop.
  - `ec-bench-encode` measures the sustainable encode rate on the Jetson (used on H to choose the main storage size).
- **LiveChecks:** every rule in PIPELINE 6.2, with thresholds from rig.yaml.
  - **iPhone parity is exact:** main per-frame 1/dt < 31 after 1 s of recording (ViewController.swift:1817-1825);
    pose ‖Δp‖/dt > 7.0 m/s from the live SLAM channel (ViewController.swift:306, 1829-1843); no uw frames
    (DemonstrationData.swift:574).
  - Ours: SLAM lost, reset or map change; heartbeat > 0.5 s; uw fps < 27 over 1 s; IMU gap > 20 ms, overrun, or rate
    < 0.95x; lossless queue overflow; disk < 5 GB; thermal.
  - Each rule is a small class with a unit test of its edge cases, including the 1 s grace and a SLAM first-pose
    (no Δ before two poses).
  - `live_checks.json` records the value, threshold and result for every rule, and the max pose speed.
- **State machine:** `INIT → ARMING (pre-roll recording; wait for SLAM tracking_ok and steady streams) → RECORDING →
  STOPPING → DONE | ABORTED(reason)`. The `rec_start`/`rec_stop`/`arm` events are written. Refuses to arm without a
  PASSing `gripper_range.json` unless `--no-calibration`. Arming timeout (default 30 s) → clean exit, nothing kept.
- **Supervisor:**
  - `posix_spawn` of `ec-slam-live` and `ec-viz` with the shm names and config;
  - watches the SLAM heartbeat; kills children on exit;
  - a viewer crash is logged only; a SLAM crash is an abort.
- **UI:** terminal status line (state, timer, fps per stream, IMU rate, drops, max pose speed, temperature); Enter to
  stop; Ctrl-C to stop and save; the abort reason is printed in red.

**Tests:**
- end-to-end runs with Mock/Replay cameras + Sim IMU + `MockSlam`:
  - a normal 60 s episode;
  - each abort reason injected, checking the package is discarded and the logs kept;
  - Ctrl-C during each state; arming timeout; no calibration;
- the produced package passes `validate_layout()` and has the pre-roll before `rec_start`;
- 10-minute soak on the laptop with mocks: memory flat, zero lossless drops.

### T6. `ec-viz` + `ec-slam-live`

**Owns:** `episode_collection/apps/ec-viz`, `apps/ec-slam-live`, `common/slam/*`, `episode_collection/viz/*`, tests.

- **`ec-viz`:**
  - reads the shm frame rings + status channels;
  - mosaic of main and uw at about 10 Hz (OpenCV highgui) with an overlay: state, REC timer, fps, IMU rate, drops,
    SLAM state, max pose speed, temperature, disk;
  - optional low-rate finger-tag overlay using T7's detector;
  - `--headless-snapshot` renders to PNG for tests.
- **`ec-slam-live`:**
  - `SlamBackend` implementations: **`MockSlam`** (replays a TUM trajectory with injectable jumps, losses, map
    changes, stalls; compiled and tested now) and **`OrbSlam3Backend`** (`RIG_WITH_ORBSLAM3`, first built on H):
    - `ORB_SLAM3::System(voc, settings, IMU_MONOCULAR, use_viewer)`;
    - `TrackMonocular(img, t, imu_batch)` with input downscaled to about 800x600 grey;
    - tracking state mapped to `tracking_ok / initializing / lost`; map change detected via the Atlas map id;
  - publishes pose + state + heartbeat at the frame rate to the status channel; pinned cores from config; Pangolin
    viewer at a lower refresh.
  - ORB-SLAM3 settings writer for the KannalaBrandt8 camera + IMU noise + `T_b_c` from rig.yaml, ported from
    `sensor_rig/sensor_rig/pose/orbslam3.py`, with a golden test against the Python output.
- `third_party/orbslam3/`: the pinned commit hash, the patch list (OpenCV 4.8 / C++17 fixes, the saver patch for
  velocity and biases used later by T10), and the build script for aarch64. **Not built on the laptop.**

**Tests:** MockSlam-driven end-to-end with T5's recorder (abort on an injected 8 m/s jump, on a loss, on a stall);
viz snapshot test; settings-writer golden test.

### T7. `ec-calibrate` (gripper range) + tag detector

**Owns:** `episode_collection/calib/*`, `common/tags/*`, `apps/ec-calibrate`, tests.

- **Tag detector:** ArUco DICT_4X4_50, 16 mm, ids and corner-refinement parameters exactly as iPhUMI
  `scripts/detect_aruco_iphone.py:45-118` + `calibration/aruco_config.yaml`; `solvePnP` with the same flag; input =
  the remapped uw virtual image (T4); OpenCV 4.5 and 4.8 API paths.
- **Fit:** an exact port of iPhUMI `calibrate.py:85-168` (tag-pair choice, the |z − 0.098| < 0.08 filter,
  width = right.x − left.x, min/max, robust linear L↔R maps). Output: `gripper_range.json` plus an iPhUMI-compatible
  yaml.
- **PASS/FAIL rules:** both tags detected in ≥ 80 % of frames; observed range ≥ 0.06 m; map residual below a
  threshold. Printed with the numbers.
- **Program:** arms immediately (no SLAM; D-flow in PIPELINE 4.1); on-screen instructions; Enter to stop; fit; PASS
  writes the calibration; exits.

**Golden equivalence test (1:1 proof):** on the real iPhone calibration episode
`data/raw/2026-10-01/2026-10-01T21-10-44.057Z_15647_firstreal_grippercalibration_left`:
- run iPhUMI's own Python detect + calibrate in the venv → expected numbers;
- run the C++ detector + fit on the same decoded frames with the same intrinsics and zero distortion;
- per-frame widths within 1e-4 m, and fit outputs within 1e-4.

### T8. Bring-up kit: BRINGUP.md, `jetson_setup.sh`, `ec-doctor`, device-tree overlay

**Owns:** `physicalSetup/docs/BRINGUP.md`, `physicalSetup/scripts/jetson_setup.sh`, `apps/ec-doctor`,
`physicalSetup/bringup/*`.

**BRINGUP.md** is a numbered, copy-paste procedure for H. Every step has its expected output and what to do if it
differs:
1. Confirm JetPack 6.2 / L4T 36.4.3 (`cat /etc/nv_tegra_release`, `uname -r`); hold L4T and kernel packages
   (`apt-mark hold`).
2. Network and SSH (the user does this).
3. NVMe: mount, `data_root`, a write-speed test.
4. Power mode (`nvpmodel`) and fan; thermal readout.
5. Dependency install and the physicalSetup build (`jetson_setup.sh`; idempotent; prints every command; never touches
   L4T/kernel packages).
6. ORB-SLAM3 + Pangolin build on the NVMe (T6's script) and the vocabulary file.
7. IMU wiring:

   | Modulino Qwiic | 40-pin pin |
   |---|---|
   | 3V3 | 1 |
   | SDA | 3 |
   | SCL | 5 |
   | GND | 6 |

   Then `i2cdetect -y -r 7` → `6a`, followed by T1's checks.
8. Arducam IMX519 driver install for 36.4.3 (exact Arducam commands, researched and cited).
9. **Mixed IMX519 + IMX219 device-tree overlay** (the highest-risk item):
   - the method from NVIDIA's mixed-sensor forum guidance;
   - a prepared `.dts` in `bringup/`;
   - the `extlinux.conf` entry with a **fallback boot entry** so a bad overlay can't brick boot;
   - rollback steps.
10. Camera checks (T2's commands); focus lock check; mode table dump.
11. `ec-doctor`, which must be all PASS.
12. Live tests in order:
    - `ec-record --viz cameras` with `MockSlam`;
    - then real live SLAM;
    - the **15-minute back-to-back thermal acceptance test** (PIPELINE 6.3);
    - the first `ec-calibrate`, then the first real episodes.
13. Timestamp-domain verification: Argus sensor timestamp versus MONOTONIC_RAW arrival (a stable positive latency);
    IMU clock-fit residuals.
14. AprilGrid print spec (for the later Kalibr work; see TODO): Kalibr 6x6 AprilGrid, 36H11, tag size and spacing for
    A3, mounted flat on rigid board.

**`ec-doctor`:**
- checks with PASS/FAIL + fix hint: L4T version, package holds, I2C bus present, IMU WHO_AM_I, cameras enumerated with
  the configured modes, focus control available, NVMe free space and write speed, clocks and power mode, thermal
  zones readable, display available (`$DISPLAY`), ORB-SLAM3 vocabulary present, shm usable;
- `--sysroot` override so it is unit-tested against fake sysfs/procfs trees on the laptop.

**Reviewer R2** (after T5–T8): reviews all four plus their integration with Wave 1.
- Full end-to-end on the laptop: `ec-calibrate` (replay of the iPhone calibration episode) → `ec-record` × 3 (replay +
  Sim IMU + MockSlam, including one injected abort) → session folder inspected against PIPELINE 6.1.
- **Dry-run of BRINGUP.md** against the code: every command and flag it names exists and behaves as described.
- Sanitizers; T1–T8 traceability table: each PIPELINE requirement → code → test.

## 5. Hardware day (H): bring-up with the user

Run BRINGUP.md in order. The orchestrator explains each command before the user runs it, and asks before anything
that changes state on the Jetson (driver install, device tree, extlinux). Acceptance on H:

| # | Test | Pass |
|---|---|---|
| H1 | `ec-imu-probe` 10 min | 833 Hz ± 0.5 %, 0 overruns, \|g\| 9.81 ± 0.2, clock residual p99 < 100 µs |
| H2 | `ec-cam-probe` both cameras, 5 min | main 60 fps and uw 30 fps; p99 jitter < 2 ms; 0 unexplained drops; exposure and gain locked; focus status known |
| H3 | Both cameras + IMU together (`ec-record --viz cameras`, MockSlam), 10 min | no drops; encode keeps up (`ec-bench-encode` chooses the main storage size) |
| H4 | Live ORB-SLAM3 builds and tracks (nominal intrinsics) | initializes within 10 s of motion; holds ≥ 28 Hz |
| H5 | **Thermal:** back-to-back `ec-record`, live SLAM + both viewers, ≥ 15 min | no throttling, no fps aborts, SLAM ≥ 28 Hz. Fallbacks per PIPELINE 6.3 otherwise. |
| H6 | `ec-calibrate` on the real gripper | PASS, plausible range |
| H7 | 5 real `ec-record` episodes | packages valid; pre-roll present; abort reasons sensible |

Fix work found on H is done by new executor tasks (T5-fix-…), reviewed by a reviewer pass, and committed.

## 6. Waves 3–5 (after H)

**Wave 3: offline pipeline (`ec-process`)**
- **T9:** orchestrator, `StageRegistry`, manifest stage records and `inputs_hash`, external-stage JSON contract,
  `--status`, the `validate_raw` and `validate` stages.
- **T10:** `slam` stage, offline ORB-SLAM3 mono-inertial on the full uw + IMU from `arm` (pre-roll init), saver patch
  for velocity and biases, `external_tum` impl. Built on the Jetson, or on the laptop once disk allows.
- **T11:** `traj_fit` (`imu_preint` in Eigen; `slerp`; `bspline` behind `RIG_WITH_CERES`) + `pose` stage (evaluate at
  main mid-exposure times, `C_uw_T_TCP`, gravity y-up, pos + rotvec). Tested on synthetic trajectories with simulated
  IMU and noisy 30 Hz poses (error budget from PIPELINE 7.3).
- **T12:** `width` stage (T7 detector on all uw frames, the session calibration, interpolation to main times),
  `images` stage (T4), and Python `export` (zarr + imagecodecs JPEG-XL level 99, iPhUMI keys, ultrawide index maps,
  session `.zarr.zip`).
  - **1:1 acceptance:** build a sim rig package from 5 firstreal iPhone episodes (`sensor_rig/sim`, with frames and
    timestamps from the iPhone, `external_tum` = the ARKit pose), run stages 0–7, and compare to the golden
    `scripts/regression/golden/iphone_firstreal_5ep.json`: keys, shapes, dtypes, episode_ends, image bytes, width
    within 1e-4 m, pose within the interpolation tolerance.
- **Reviewer R3.**

**Wave 4: calibration + accuracy** (after the Kalibr discussion, TODO):
- Kalibr record mode and import into rig.yaml; Allan variance tool; LED camera↔camera offset test;
- arm ground-truth test harness (rig on the Franka, FK versus offline pose, live versus offline comparison);
- CAD-derived `C_uw_T_TCP` from the sim-eval finger kit.
- **Reviewer R4.**

**Wave 5: `deployment/` scaffold:** `CaptureService` → shm → `ObsAssembler` (20 Hz main / 10 Hz uw selection rule of
umi_utils.py:37-76) → `rigpy` → policy → `controlSW/franka_deploy`. **Reviewer R5.**

## 7. Gate before every commit (the orchestrator does this)

1. Clean configure and build in a fresh `$SCRATCH/build/gate`: `scripts/check.sh` in the default, ASan+UBSan and TSan
   configurations. All tests pass.
2. `git diff --name-only` shows only allowed paths (section 0). The iPhone regression guard is then unaffected by
   construction; run it as well if the env is available.
3. The reviewer's findings are all closed or explicitly deferred, with a TODO entry.
4. Commit: `physicalSetup: wave N (<tasks>)` with the attribution trailer. Delete the build trees.

## 8. Pause and resume ("wrap up")

The state lives in files, never only in a conversation:

- **[STATUS.md](STATUS.md):** the single source of truth. Wave and task table (`todo / running / review / fixing /
  done / paused`), the commit of the last gate, the open reviewer findings, and "Next actions". The orchestrator
  updates it whenever a task changes state.
- **`physicalSetup/docs/worklogs/<task>.WORKLOG`:** each executor (and the reviewer) keeps one. It holds:
  - what is done (files and tests), what is in progress (the exact file/function), and what is next, as a checklist;
  - decisions or contract questions raised;
  - the exact build/test commands and their last result.
  The agent updates it at least after every meaningful step. A fresh agent can continue from the worklog alone.

**When the user says "wrap up":**
1. The orchestrator sends every running agent `WRAP UP` (SendMessage).
2. Each agent:
   - finishes or reverts the edit it is in the middle of, so the files are in a consistent state;
   - updates its WORKLOG, setting "Next" to the exact resume point;
   - replies with its state.
   No new work starts.
3. The orchestrator then:
   - runs a quick build of what exists (best effort) and records the result;
   - sets each task in STATUS.md to `paused` with a pointer to its WORKLOG;
   - commits **everything** as `physicalSetup: WIP pause (<tasks>)`. This is a WIP commit, not a gate. The next gate
     commit on top of it is the reviewed one.
   - updates AGENTS.md ("Current state" / "Next time": "resume physicalSetup from docs/STATUS.md");
   - deletes the scratch build trees.

**Resume in a new session:**
1. Read AGENTS.md → STATUS.md.
2. For each `paused` task, launch a fresh executor with the original task brief (section 2–4 of this plan) plus "resume
   from `worklogs/<task>.WORKLOG`; verify the build and tests first".
3. Continue the wave as normal.

Agent IDs are not needed to resume, because everything an agent knew is in its worklog.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| Mixed IMX519+IMX219 overlay fails on 36.4.3 | Prepared dts + fallback boot entry (T8). Worst case: bring up each camera alone first, so the IMU + one-camera path still gets validated on H. |
| IMX519 VCM inaccessible (`UU`) | V4L2 fallback, else fixed at power-up and recorded (T2). |
| Argus sensor timestamp domain is not MONOTONIC_RAW | H13 measurement; a conversion layer in T2 (TSC offset) behind the same `t_sof_ns`. |
| Live ORB-SLAM3 below 28 Hz or overheating (H4/H5) | Pinned cores; GPU fork; 15 Hz live SLAM (PIPELINE 6.3). |
| CPU JPEG encode can't keep up (no NVENC) | `ec-bench-encode` on H; smaller main storage size; NVJPG hardware encoder if present (checked on H). |
| ORB-SLAM3 never built before H | Pinned commit + patches + script prepared in T6; MockSlam keeps everything else testable. |
| Laptop disk | Builds in scratch only, deleted per wave (D10). |
| Kalibr not yet done | Nominal intrinsics + CAD extrinsics (`provenance: nominal`) for live work. Training-grade offline poses wait for Wave 4. |
