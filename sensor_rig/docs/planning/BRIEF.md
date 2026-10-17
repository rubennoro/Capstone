# Sensor rig redesign: lead's design brief (binding decisions)

Goal: replace the iPhone on the iPhUMI gripper with separate sensors on a Jetson Orin Nano (JetPack 6):
two CSI cameras ("main" ~77°, "ultrawide" ~120° fisheye) + a standalone IMU (ICM-42688-P on SPI).
Two deliverables:
1. **Data collection**: record an episode on the rig and turn it into a folder that is indistinguishable (to the
   existing Python pipeline) from an iPhone export, so `process_demos_iphone.py` / `build_umi_dataset_iphone.py` /
   training run unchanged.
2. **Deployment**: a real-env camera peripheral fed by the rig that produces exactly the observation images the
   iPhone produced at deployment (and therefore what the policy saw in training).

Research (read these; they are the source of truth, with file:line refs):
- R1 iPhone export format: `../research/R1_iphone_export_format.md`
- R2 processing pipeline: `../research/R2_processing_pipeline.md`
- R3 deployment path: `../research/R3_deployment_path.md`
- R4 Jetson/VIO hardware+software: `../research/R4_jetson_vio_stack.md`
(paths relative to this file; absolute dir: <scratchpad>/)

Repo: /home/ruben/Capstone (read AGENTS.md, DESIGN.md). A test conda env with the iphumi deps (no pypeertalk) is at
`<scratchpad>/env_iphumi` (python: `<scratchpad>/env_iphumi/bin/python`). Real iPhone exports: `data/raw/2026-10-01/`.

## Decisions

D1. **New package** `/home/ruben/Capstone/sensor_rig/` (Python package `sensor_rig`, installable with its own
    `pyproject.toml`). Core deps: numpy, opencv, pyyaml, scipy, av (PyAV, for MP4 writing). Hardware backends
    (GStreamer/Argus, spidev) are imported lazily so everything else imports and tests on a laptop.
    Tests in `sensor_rig/tests/` (pytest), runnable with the scratch env.

D2. **Three-stage data path** (pose is computed offline, like original UMI; the Orin Nano only records):
    1. `record` (Jetson): writes a **raw rig episode** (EuRoC-like layout: per-camera `data.csv` with
       `#timestamp [ns],filename` + JPEG frames, `imu0/data.csv` in EuRoC columns, `meta.json`, snapshot of `rig.yaml`).
    2. `pose` (laptop/cluster): convert raw episode(s) to the input format of an offline VIO (ORB-SLAM3
       mono-inertial on the ultrawide, OpenVINS alternative), run it externally, and import the resulting trajectory
       (TUM format: `t x y z qx qy qz qw`, on the rig clock, pose of a named sensor frame). We do NOT build ORB-SLAM3
       here; we provide adapters, configs and docs. An optional live-pose hook can exist but is not required.
    3. `export` (laptop/cluster): raw episode + trajectory + rig.yaml → iPhone-format episode folder that satisfies
       every invariant in R1 §0 / §11. Then the existing pipeline runs unchanged.

D3. **Virtual iPhone cameras.** rig.yaml describes each *physical* camera (model: `pinhole-radtan` or
    `fisheye-equidistant` à la Kalibr/OpenCV, intrinsics, resolution) and each *virtual* camera to reproduce:
    - virtual main: pinhole 1920x1440, K = iPhone 16 Pro values from R1 §5 (fx=fy=1406.3428, cx=953.2503, cy=723.22003)
    - virtual ultrawide: pinhole 640x480, K (fx=fy=259.01538, cx=320.2935, cy=239.67455), zero distortion
    - optional rotation between physical and virtual camera (default identity).
    One shared module `sensor_rig/camera_model.py` builds remap tables (physical → virtual, any output scale) and
    reports coverage (fraction of virtual pixels the physical lens actually sees). **Both export and deployment use
    this same code**, so training data and deployment observations are produced identically.

D4. **Pose convention.** Export the pose of the *virtual main camera* as ARKit `camera.transform`: camera-to-world,
    camera axes +x right, +y up, +z backward (T_arkit = T_cv @ diag(1,-1,-1,1)), world gravity-aligned **+y up**,
    metres, 4x4 row-major. Chain: VIO gives T_W_S (sensor S, world z-up gravity-aligned for inertial VIO) →
    T_W_main = T_W_S @ T_S_main (calibration) → rotate world z-up → y-up → apply a fixed configurable world offset
    (default non-zero translation + yaw) so no exported pose is ever exactly identity (R2: identity rotation → NaN in
    `common/transform_util.py`).

D5. **One clock.** Every timestamp on the Jetson is CLOCK_MONOTONIC in integer nanoseconds. Per-sensor fixed offsets
    (camera latency/exposure-mid, Kalibr cam-IMU `timeshift_cam_imu`) live in rig.yaml and are applied at export.
    Export maps monotonic → UNIX with one offset captured at record start (wall-clock stored in meta.json), and writes
    ISO-ms strings like the app.

D6. **Export timing.** Main stream drives the 60 Hz record grid (iPhone pipeline hardcodes 1/60, R2). If the main
    camera runs at ~60 fps, grid = its actual frame times (like the iPhone). If slower (e.g. 30 fps), build a 60 Hz
    grid and repeat the latest frame; poses are always interpolated (SLERP + linear) at grid times. Ultrawide frames
    are attached to the grid slot nearest their (offset-corrected) capture time, at most one per slot, with the slotted
    `""` representation, and the UW MP4 PTS follow R1 invariant 4. Grid slots without a valid pose (VIO not yet
    initialised / lost) are trimmed from the episode ends; gaps in the middle abort the episode (mirror the app).

D7. **Device identity.** Exported `deviceIdentifier` = `"JetsonRig1,1"`. Add an entry for it to
    `iPhUMI/python_package/iphumi/demonstration_processing/calibration/gripper_calibration.yaml` (initially copied
    from `iPhone17,1` with a TODO to measure the mount; if the virtual main camera is placed where the phone's main
    camera was, the iPhone TCP transform is correct by construction).

D8. **Deployment peripheral** `RigCamera(BaseCamera)` in
    `behavior_prompting/deps/real-env/real_env/peripherals/rig_camera.py` (+ `configs/peripherals/rig_wrist_right.yaml`),
    importing `sensor_rig` for capture + remap. Output contract identical to the iPhone peripheral (R3 §6):
    `{"main","ultrawide"}` 720x960x3 uint8 BGR (virtual cameras at the deployment scale), one monotonic timestamp,
    name `right_wrist_camera_0`, fps 60. Keep edits to vendored upstream code minimal; list (don't silently fix)
    upstream deployment bugs from R3 §8 and propose fixes separately.

D9. **Backends.** Camera: `MockCamera` (replay/synthetic) and `ArgusCamera` (GStreamer `nvarguscamerasrc` → appsink,
    best-effort timestamp mapping to CLOCK_MONOTONIC, documented as an upgrade path to a C++ libargus capture with SOF
    timestamps). IMU: `MockImu` (synthesised from a trajectory) and `Icm42688Spi` (userspace spidev, FIFO burst reads,
    on-chip timestamps mapped to monotonic). Hardware backends cannot be run here: keep their pure logic (register
    decoding, FIFO parsing, timestamp unwrapping/mapping) in testable functions with unit tests on fake bytes.

D10. **Verification without hardware (the acceptance test).**
    a. `sensor_rig/sim/` builds a *simulated raw rig episode* from a real iPhone demo: warp the iPhone main/ultrawide
       frames into the physical rig camera models (inverse of D3), synthesise IMU from the ARKit trajectory, and emit a
       TUM trajectory as a VIO would (sensor frame, z-up world, different origin), with realistic rig timestamps.
    b. Export it, then check: all R1 invariants; exported poses match the original ARKit poses up to a rigid world
       transform (relative poses equal to ~1e-4); images match the originals where covered.
    c. Run the real iPhUMI processing pipeline (scratch env) on a session of exported episodes (incl. a gripper
       calibration episode) and on the original episodes; compare zarr outputs (eef pos/rot relative motion, gripper
       width, image keys/shapes).
    d. Deployment: `RigCamera` with `MockCamera` replaying the simulated rig frames produces images that, after the
       inference-time transform (pad-to-square + resize 224), match the training zarr images.

## Out of scope for this pass
Building ORB-SLAM3/OpenVINS, the Franka controller, flashing the Jetson, buying parts. Document them as next steps.
