# sensor_rig

A Jetson Orin Nano sensor rig that replaces the iPhone on the iPhUMI gripper. It has two CSI cameras ("main" and a
fisheye "ultrawide") and an ICM-42688-P IMU on SPI. The package does two jobs:

1. **Data collection.** It records episodes on the rig and turns them into folders that the existing iPhUMI pipeline
   (`process_demos_iphone.py`, `create_session_iphone.py`, `build_umi_dataset_iphone.py`) cannot tell apart from
   iPhone exports.
2. **Deployment.** It runs a camera peripheral that produces the same observation images the iPhone produced (see
   the deployment section).

Hardware backends (GStreamer/Argus, spidev) are imported lazily. Everything else runs and is tested on a laptop.

## Data path

```
Jetson : sensor-rig doctor      -> preflight table (devices, IMU, clock, disk, rig.yaml)
Jetson : sensor-rig collect     -> raw rig episodes (doctor, calibration, demos; or `record`)  <raw>/<YYYY-MM-DD>/<NAME>/   (JPEG + CSV + meta.json + rig.yaml)
laptop : sensor-rig pose-prep   -> EuRoC folder + ORB-SLAM3 settings (or OpenVINS/Kalibr yamls)
extern : ORB-SLAM3 mono-inertial (or OpenVINS) -> trajectory file
laptop : sensor-rig pose-import -> <NAME>/trajectory/trajectory.{tum,json}  (with sanity checks)
laptop : sensor-rig export      -> <out>/<YYYY-MM-DD>/<NAME>/<NAME>{.json,_rgb.mp4,_ultrawidergb.mp4}
laptop : sensor-rig validate    -> checks every iPhone-format invariant (export also runs it)
then   : the unchanged iPhUMI pipeline, with group.iphone_dir=<out>
```

One command for the laptop part: `sensor-rig process-session` (below) runs pose-prep → VIO → pose-import → export →
validate for a whole session and prints the `scripts/run_pipeline.sh` line.

The raw episode folder name is already the final iPhone-style NAME
(`<UTC time with ':'→'-'>_<5 digits>_<session>_<demonstration|grippercalibration>_<side>`). It is chosen at record
start, like the app does.

## Install and test

The package needs numpy, opencv, pyyaml, scipy and PyAV. On a laptop, install it with `pip install -e sensor_rig`
(extras: `[jetson]`, `[rosbags]`, `[test]`). On the Jetson, use `scripts/jetson_setup.sh` instead (below). Run
commands with the `sensor-rig` entry point. `python -m sensor_rig` fails when run from the repo root: Python imports
the project folder `sensor_rig/` as an empty namespace package. The tests don't need an install, because
`tests/conftest.py` puts `sensor_rig/` on `sys.path`.

```bash
PY=<python with numpy, opencv, scipy, pyyaml, av, pytest>
$PY -m pytest sensor_rig/tests -m "not e2e"          # from the repo root
```

Tests marked `data` use the real iPhone exports in `data/raw/2026-10-01` (override the location with
`SENSOR_RIG_IPHONE_DATA`). They are skipped if the folder is missing. `pyproject.toml` disables the ROS Humble
pytest plugins that `/opt/ros` puts on `PYTHONPATH`, because they fail to import.

## Conventions (normative)

- **Clock:** every rig timestamp is CLOCK_MONOTONIC in integer ns. Raw files store the stamps exactly as recorded.
  Per-sensor offsets from rig.yaml (latency/exposure-mid prior + Kalibr `timeshift_cam_imu`) are applied later, by
  `rig.capture_time_ns(cam, t)` in pose-prep and export. UNIX time = monotonic + one offset sampled at record start
  (`meta.clock.start`).
- **Transforms:** `T_a_b` maps frame-b points to frame a. Camera frames use OpenCV axes. Frame names are `imu, main,
  ultrawide, virtual_main, virtual_ultrawide`. Trajectory files may call the IMU `imu0`.
- **Exported pose** (iPhone `camera.transform`):
  `T = rig.export.T_world_offset @ T_YUP_ZUP @ T_Wz_S(t) @ rig.T_S_virtual_main_arkit(S)`. VIO world is z-up and
  gravity-aligned; ARKit world is y-up; the camera has ARKit axes (+y up, +z back). The non-identity world offset
  makes sure no pose is ever exactly the identity, which would give NaN in iphumi `pose_6d_to_4x4`.

## Raw rig episode (`sensor_rig.raw.v1`)

```
<raw>/<YYYY-MM-DD>/<NAME>/
  meta.json            kind/session/side/gripper_id/calibration run, clock samples, per-sensor stats, status
  rig.yaml             snapshot of the rig config (RigConfig.dump)
  cam_main/ cam_ultrawide/
    data.csv           "#timestamp [ns],filename" (strictly increasing), data/<t_ns>.jpg
    frames_meta.csv    seq,t_ns,t_arrival_ns,sensor_seq,exposure_ns,gain,encode_ms,bytes
    drops.csv          seq,t_ns,reason,count   (sensor_gap | queue_full | encode_error | write_error | nonmonotonic_ts)
  imu0/data.csv        EuRoC columns: t_ns, gyro xyz [rad/s], accel xyz [m/s^2], IMU frame
  imu0/aux.csv         t_ns,chip_us,temp_c,batch,host_read_ns
  trajectory/          added by pose-import
```

While recording, the folder is called `.<NAME>.recording`. An aborted episode is deleted, or kept as
`.<NAME>.aborted`. Every reader skips hidden folders. `sensor_rig.raw_episode` is the only module that reads or
writes this format (`RawEpisodeWriter` is used by the recorder and the simulator; `RawEpisode` is the reader).

## Pose stage

```bash
sensor-rig pose-prep --raw <episode|date dir|raw root> --out <dir> --target orbslam3 --scale 0.5 --imu-rate 200
<dir>/<NAME>/run_orbslam3.sh                      # needs $ORB_SLAM3; writes f_<NAME>.txt
sensor-rig pose-import --raw <episode> --traj f_<NAME>.txt --format orbslam3-euroc --frame imu
```

- `pose-prep` writes `mav0/cam0` (grayscale PNG, optionally downscaled with scaled intrinsics; `--scale` snaps to
  the nearest exact uniform scale, e.g. 0.4 on 1640x1232 becomes 615x462), `mav0/imu0`
  (optionally block-averaged), and `times.txt`. All of them use **corrected** timestamps, so the VIO trajectory comes
  out on the clock that export uses.
- With `--target orbslam3` it also writes `orbslam3_settings.yaml` (KannalaBrandt8 or PinHole, `IMU.T_b_c1 =
  T_imu_cam`, noise from `rig.imu.noise.for_vio()`) and a run script.
- With `--target openvins` it writes `kalibr_imucam_chain.yaml`, `kalibr_imu_chain.yaml` and an
  `estimator_config.yaml` template. Building the ROS bag is left to `rosbags-convert` or Kalibr's `bagcreater`.
- `pose-import` parses TUM (`t x y z qx qy qz qw`; s or ns detected automatically, parsed exactly to ns) or ORB-SLAM3
  EuRoC output. By default it first **gravity-aligns** the trajectory: it rotates the world so that the mean IMU
  specific force (rotated into the world) points along +z, leaving yaw untouched. It only does this when the estimated
  tilt is > 2° (`--gravity-align-min-deg`), because the estimate is only good to ~0.5° while an inertial VIO that is
  already gravity-aligned knows better; `--no-gravity-align` turns it off. This is required for ORB-SLAM3, whose `SaveTrajectoryEuRoC` world is the first keyframe's IMU frame and so
  is tilted. The applied angle is recorded in `trajectory.json` (`gravity_alignment`). It then runs three checks:
  - overlap with the main camera span must be ≥ 50%;
  - **gravity**: the mean accelerometer reading rotated into the VIO world must point along +z within 3°;
  - **time offset**: |ω| cross-correlation between the trajectory and the gyro; it warns if the delay is > 5 ms.

  - **motion** (T3): gyro vs VIO angular rate over each pose interval, with a constant bias removed. It refuses
    above an RMS of 0.15 rad/s or a max of 1.5 rad/s, or on a per-step speed > 5 m/s (VIO jumps, resets,
    relocalisation).

  It refuses on a failed overlap, gravity or motion check unless you pass `--force`. `--time-offset-ns` (default
  `auto` = `meta.sim.replay_offset_ns` of an episode recorded by `collect --mock-replay`) shifts the trajectory onto
  the episode's clock. Otherwise it writes
  `trajectory/trajectory.tum` and `trajectory.json` (frame, checks, source sha256).

## Process a session (laptop)

```bash
sensor-rig process-session --raw <raw root> --session rig1020 --out data/rig_export/rig1020 \
    --vio {import-dir --traj-dir D | cmd --vio-cmd "TEMPLATE" | orbslam3-docker [--docker-image IMG]} \
    [--rig calibrated.yaml] [--task-label block_raise] [--workers 4] [--overwrite] [--force-episode NAME] [--allow-partial]
```

- **Checks first:** the session name must be rig-only (`rig*`) and unused in `data/raw`. Every demo must name a kept
  calibration of the session; hidden aborted / discarded / rejected episodes are never read.
- **Demos:** pose-prep (EuRoC folder, ORB-SLAM3 settings, finger mask `mask_ultrawide.png` under
  `<out>/.sensor_rig/vio/<session>/`) → VIO hook → pose-import (gravity, overlap and motion checks, replay offset).
  - `import-dir` looks for `<NAME>.tum`, `<NAME>/trajectory.tum`, `f_<NAME>.txt`, or for a `collect
    --mock-replay` episode, the source's `sim/vio_trajectory.tum`.
  - `cmd` runs a shell template with `{prep} {settings} {times} {name} {mask} {out}`.
  - `orbslam3-docker` runs ORB-SLAM3 in the image from `docker/orbslam3/`. That image is not built yet; see its
    README.
- **Calibration episodes skip the pose stage:** export reads a trajectory only for demonstrations.
- **Export and validate:** export runs calibrations first, then validate runs on the export root.
- **Failures and resume:** a failed episode never stops the others, and re-runs resume finished steps
  (`--overwrite` redoes them).
- **Report:** `<out>/.sensor_rig/process_<session>.json` gives per-episode status and reason, VIO residuals, lag
  stats, coverage and trim.
- **Exit code:** 0 only if the calibration and ≥ 1 demo exported and no demo failed (or `--allow-partial`).
- **Next step:** it prints the `scripts/run_pipeline.sh all --task ... --dataset rig-<session>-single --sessions
  <session> --raw-dir <repo-relative export root> --smoke-first` command; run it yourself.

## Export

```bash
sensor-rig export --raw <raw root|date dir|episode ...> --out <export_root> [--uw-rate 10|all]
       [--label-task block_raise] [--calibration-run NAME] [--rig better_rig.yaml] [--trajectory f.tum
       --trajectory-frame imu] [--workers N] [--overwrite] [--force] [--fix-world-yaw]
       [--max-main-gap 0.10] [--max-uw-gap 0.25]
sensor-rig validate <export_root|episode> [--gripper-calibration .../gripper_calibration.yaml] [--decode]
```

What export does (code in `sensor_rig/export/`):

- **Grid** (`grid.py`): the main camera drives the 60 Hz record, because the pipeline hardcodes 1/60.
  - Main camera at ≥ 50 fps: the grid is the actual main frame times (dropped frames stay gaps, like the iPhone).
  - Slower main camera (e.g. 30 fps, the Jetson's only full-FOV mode): **align-exact fill grid** (T3).
    - The ms time strings are written so that iPhUMI's align stage (`np.arange` over the pose strings at 1/60 s,
      then "most recent string ≤ sample") maps sample k to slot k exactly. The sample times are reproduced bit for
      bit, including numpy's float drift, with ≥ 0.5 ms of slack.
    - Poses are sampled at the string times. Each slot shows one of the two main frames around align's sample time,
      chosen by error diffusion, so the image-vs-pose lag is zero-mean (|mean| < 2 ms). The worst case is half a
      frame interval plus the grid phase (≈ 17 ms; ≈ 19 ms when the camera clock drifts, as in the iPhone-derived
      sims).
    - `export_log.jsonl` records the lag stats per episode. The k / k−3 observation pair never shows the same frame.
  - Episodes with fewer than 31 frames in any 1-s window after the first second abort. In fill mode the threshold
    is 0.52 × nominal fps.
- **Poses** (`exporter.py`, `pose/`): linear position + SLERP rotation at grid times.
  - Slots without a pose (VIO not yet initialised, or lost) at the head or tail are trimmed. A gap in the middle
    aborts the episode (`pose_gap_mid_episode`).
  - It also aborts on: speed > 7 m/s, an identity rotation, NaN, duration < 1 s, or a frame gap > 0.10 s (main) /
    > 0.25 s (ultrawide). The gap reasons name the gap and its time; change the limits with `--max-main-gap` /
    `--max-uw-gap`.
  - Rotation angles > 170° get a warning: align interpolates rotation vectors component by component, which is
    wrong near 180°. `--fix-world-yaw` rotates the (arbitrary) world yaw per episode to keep them small. Relative
    motion is unchanged.
  - A demonstration without `gripperCalibrationRunName` aborts early with `no_calibration_run`; pass
    `--calibration-run <NAME>`.
- **Ultrawide** (`slotting.py`): decimated to ~10 Hz (like the iPhone), then each frame is attached to the nearest
  grid slot, at most one per slot.
  - `ultrawideRGBTimes[i]` is the slot's main time string or `""`.
  - UW MP4 PTS = slot time − first occupied slot time (R1 invariant 4).
- **Images:** physical → virtual iPhone cameras through `rig.remap(cam, "export")`. Main is 1920×1440 pinhole with
  the iPhone 16 Pro K; ultrawide is 640×480 zero-distortion pinhole, which ArUco PnP needs. Coverage goes in the
  report, with a warning if it is below 0.98.
- **Videos** (`video.py`): libx264 in a QuickTime container (`.mp4`), timescale 600, VFR PTS from real times, first
  PTS 0.
  - The header frame count must equal the JSON counts and the packet PTS must equal the planned PTS. Export checks
    this before publishing. `gen_replay_buffer_zarr.py` leaves frames past `in_stream.frames` black, so the count
    matters.
  - Pixel format is `yuvj420p` (full range, BT.601), like the iPhone. Frames are converted with OpenCV's full-range
    YCrCb rather than PyAV's `reformat`, which darkened every frame by ~2 levels. The decoded colours now stay within
    ~1 level of the input on average (`test_colour_round_trip_unbiased`), which keeps training images close to
    deployment images (deployment skips H.264).
- **JSON** (`iphone_json.py`): all 32 keys, compact Swift-like output (float32 shortest repr), `hasDepth:false`
  with empty depth lists, `deviceIdentifier: "JetsonRig1,1"`. Output for a given input is byte-identical.
- **Atomic output:** an episode is built in `<out>/.partial-<NAME>/`, validated, then moved into place. One report
  line per episode is appended to `<out>/.sensor_rig/export_log.jsonl`. Both are hidden, so group.py's `**/*.json`
  glob never picks them up.
- Calibrations are exported before demos. `validate` on an export root also checks that every demo's
  `gripperCalibrationRunName` resolves to a calibration of the same side and gripperID. The validator passes on all
  47 real iPhone exports.

After export, run the pipeline on `<export_root>` exactly as for iPhone data. The pipeline's
`gripper_calibration.yaml` needs a `default: {"JetsonRig1,1": ...}` entry.

Before the first real recording, work through the hardware checklist in
[sensor_rig/record/HARDWARE_CHECKLIST.md](sensor_rig/record/HARDWARE_CHECKLIST.md).

## First plug-in

The full hardware-day procedure, from a bare Jetson to the printed cluster command, with pass/fail numbers for each
step: [docs/first_plugin.md](docs/first_plugin.md). Its commands are run by `tests/test_first_plugin.py`.

## Jetson setup and preflight

```bash
sensor_rig/scripts/jetson_setup.sh [--dry-run]       # once per Jetson; idempotent, prints every command it runs
source ~/sensor_rig_venv/bin/activate
sensor-rig doctor --rig <rig.yaml> --out /mnt/nvme/raw [--json doctor.json]   # before every session
sensor-rig doctor --mock --out /tmp/raw                  # laptop: hardware checks skipped, mock sensors
```

- The setup script installs JetPack's apt packages (`python3-gi`, GStreamer, `python3-numpy/-opencv/-scipy/-yaml/-spidev`).
  It creates a `--system-site-packages` venv (default `~/sensor_rig_venv`), upgrades pip in it to ≥ 23, and runs
  `pip install --no-deps -e sensor_rig`. It never pip-installs OpenCV, scipy or PyAV.
  - The recorder needs only numpy, cv2, yaml, scipy (rig transforms), spidev and gi. PyAV is needed only for export,
    which runs on the laptop.
  - It then checks every import, adds the `video`/`spi` groups and an spidev udev rule, and loads spidev at boot.
    Finally it prints the manual jetson-io, power and clock steps.
  - Tested on Ubuntu 22.04's own apt stack (python 3.10, numpy 1.21, scipy 1.8, OpenCV 4.5.4, the same base as
    JetPack 6), not on a Jetson.
- `doctor` prints a PASS/WARN/FAIL table, and each FAIL/WARN line names its fix. It exits 1 on any FAIL.
  - Checks: Python deps, entry point, rig.yaml (placeholders, AE/AWB lock, exposure ≤ 4 ms, coverage), wall clock
    (year < 2026 = FAIL: no RTC battery) and NTP, free disk (minutes at 30 MB/s) and write speed.
  - Hardware checks: GStreamer + `nvarguscamerasrc`, `nvargus-daemon`, `/dev/videoN`, the Argus sensor mode vs
    rig.yaml, `/dev/spidev*` access, IMU WHO_AM_I, a 2 s capture on both cameras at once (fps, stamp latency sign,
    drops), and 1 s of IMU (rate, |a| ≈ 9.81 at rest).
  - Every hardware call has a deadline, so a hung camera gives a FAIL instead of a hang.
  - `run_doctor()` in `sensor_rig/doctor.py` is reusable (`quick=True` skips the mode listing and the write test).

## Stamp offsets and start-of-frame stamps

- `sensor-rig stamp-offset --rig rig.yaml --led gpio:7` blinks an LED at random phases while both cameras record
  under full load. It prints each camera's stamp → mid-exposure offset (error ≤ ~0.5 ms with 0.5 ms jitter in
  simulation), jitter and drift, plus the `cameras.<c>.time.offset_prior_ns` patch. `--mock` runs it on a laptop.
- Backend `argus_sof` (`capture.backends.<cam>: argus_sof`): a libargus helper (`native/argus_sof/`, C++, not built
  yet) that delivers sensor start-of-frame timestamps instead of appsink arrival times. Doctor checks the helper
  binary.

## Recording (Jetson)

One command per session (run it inside `tmux` so an SSH drop doesn't end it):

```bash
sensor-rig collect --rig <rig.yaml> --out /mnt/nvme/raw --session rig1020 --task-label block_raise   # --side left (default)
sensor-rig collect --mock --out /tmp/raw --session mocktest              # laptop: mock cameras + IMU
sensor-rig collect --mock-replay <raw episodes...> --out /tmp/raw --session replaytest   # one source per episode
```

- **Flow:** a quick `doctor` runs first (it refuses on any FAIL unless you pass `--force`). The sensors are then opened
  **once** for the whole session.
- **Calibration first:** the first episode is always the gripper calibration (open and close the fingers fully 2-3
  times, both tags visible, ≥ 3 s), as in the iPhone app. It is checked: duration ≥ 3 s, both cameras ≥ 0.9× nominal
  fps, IMU ≥ 0.95× ODR, no dropped frames. A failed calibration is moved to `.<NAME>.rejected`, and demos stay
  blocked until one passes. `--reuse-calibration` takes the last accepted one of the same session/side/gripper instead.
- **Keys** (letter + Enter):

  | Key | Action |
  |---|---|
  | Enter | start / stop an episode |
  | `a` | abort the running episode (kept as `.<NAME>.aborted`) |
  | `d` | discard the last kept episode (renamed to `.<NAME>.discarded`, never deleted) |
  | `c` | record a new calibration next |
  | `s` | status |
  | `q` | quit |

  While recording, a status line prints every second. After each stop, a verdict line shows the duration, frames /
  fps / drops per camera, IMU rate, size, and KEPT / ABORTED / REJECTED with the reason, plus the running counts.
- **Tag check:** after the calibration, ~10 ultrawide frames are remapped to the virtual iPhone ultrawide and run
  through ArUco (DICT_4X4_50; both the OpenCV < 4.7 and the ≥ 4.7 APIs). If no frame shows both finger tags, the
  calibration is rejected; below 20 % it gets a warning. The check is off for `--mock` (synthetic images have no
  tags).
- **`--mock-replay`:** each episode replays the next source of its kind (calibration sources for calibrations), with
  every stamp shifted by one constant. The episode stops by itself when the source ends. The shift is stored in
  `meta.json` (`sim.replay_offset_ns`), and `pose-import` applies it to the source trajectory.
  `tests/e2e/run_collect_dryrun.py` runs the whole chain: collect → pose-import → export → real iPhUMI pipeline.
- **Push button** (optional, off by default): `capture.button: {enabled: true, pin: 7}` in rig.yaml, or
  `--button gpio:7` (BOARD numbering, Jetson.GPIO). A short press acts as Enter, a long press as abort.
- **Signals:** Ctrl-C, SIGTERM and SSH hangup abort the running episode (kept hidden) and close the sensors. The
  session summary `<out>/.sensor_rig/sessions/<session>_<UTC>.json` is written anyway: every episode, its status,
  verdict and calibration, the doctor result and the counts.
- **Rig-only names:** use session names that never collide with iPhone sessions (e.g. `rig1020`). Calibration and
  demos always share the session, side and gripper ID, and every demo names its calibration explicitly.
- **Extra live abort rules for collect:** the main camera's fps rule also applies to the ultrawide (the VIO camera),
  and any camera aborts on a frame gap > 0.25 s. Set them with `capture.live_abort.{all_cameras_fps,
  max_frame_gap_s}`. Both are off for `record` with older rig.yaml files.

Lower-level tools:

```bash
sensor-rig record --rig <rig.yaml> --out <raw root> --session rig1020 --side left [--kind gripper_calibration]  # one kind per run
sensor-rig record --mock --duration 5 ...      # same recorder with mock cameras + IMU, runs on a laptop
sensor-rig bench-encode                        # JPEG encode budget check (run it on the Orin Nano)
```

Code in `sensor_rig/drivers/` (backends) and `sensor_rig/record/` (recorder). Everything is stamped with
CLOCK_MONOTONIC ns. Argus stamps (MONOTONIC_RAW) are converted in the backend, and ICM-42688-P FIFO timestamps are
mapped to the host clock by `drivers/chip_clock.py`. Rig offsets (`rig.yaml` `cameras.*.time`) are applied only at
export and deployment, never on disk. Live abort rules mirror the app (fps, disk space, sensor errors). The Argus and
SPI backends have not run on hardware yet; see the checklist above.

## Raspberry Pi 5

Data collection also works on a Raspberry Pi 5 (Bookworm, libcamera / Picamera2): set
`capture.backends: {main: picamera2, ultrawide: picamera2}` and give each camera a `picamera2:` block (camera number,
full-FOV sensor mode, locked exposure / gain / white balance, manual focus). The ICM-42688-P stays on SPI0, on the same
header pins as on the Jetson. Everything after recording is unchanged. Example: `configs/rig_pi5.yaml`, with an HQ
Camera IMX477 + 4.5 mm main at 1014x760 and an IMX219 + 160° fisheye ultrawide at 1640x1232; both have coverage 1.0.
Setup, wiring, lens choice, the CPU JPEG budget and the timestamp research (Pi 5 `SensorTimestamp` = CLOCK_MONOTONIC
at start of frame) are in [docs/raspberry_pi.md](docs/raspberry_pi.md). The bring-up steps are in the Raspberry Pi
section of the [hardware checklist](sensor_rig/record/HARDWARE_CHECKLIST.md).

## Calibration

The procedure (Kalibr in Docker, Allan variance, virtual-camera alignment, TCP) is in
[docs/calibration.md](docs/calibration.md). Tools: `calib-import` (Kalibr yamls → rig.yaml), `calib-check` (renders
the virtual views + coverage), `calib-align`, `calib-tcp`, `rig-validate`, `bench-remap`.

**Lens choice:** a remap can rotate and undistort but cannot invent field of view. Coverage of the virtual iPhone
cameras (`rig-validate` prints it):

| lens | coverage |
|---|---|
| IMX219 ~77° main, ideal pinhole | 0.83 (0.91 with k1 = −0.1) |
| ~120° fisheye ultrawide | 0.84 |
| ~140° fisheye ultrawide | 0.996 |
| ≥ 90° main / ~160° fisheye (`rig_default.yaml` placeholders) | 1.0 / 1.0 |
| 2.1 mm main / 1.55 mm fisheye (`configs/rig_jetson_imx219_2p1_1p55.yaml`, the charter parts, nominal) | 1.0 / 1.0 |

Below ~0.98 the exported images get black borders the policy never saw in iPhone data.

## Deployment

`RigCamera` (`behavior_prompting/deps/real-env/real_env/peripherals/rig_camera.py`, configs
`rig_wrist_right.yaml` / `rig_wrist_mock.yaml`) is a drop-in for the iPhone wrist camera: same name, endpoints and
output (`{"main","ultrawide"}`, 960×720 BGR, rendered by the same remap code as export). See
[docs/deployment.md](docs/deployment.md), which also lists upstream real-env issues that are **not applied** and
need a decision.

## Acceptance tests (D10, no hardware)

```bash
PY=<env python>
$PY -m pytest sensor_rig/tests -m "not e2e"                    # unit + D10a/b/d
SENSOR_RIG_E2E=1 $PY -m pytest sensor_rig/tests/test_e2e.py    # D10c (~4 min)
$PY sensor_rig/tests/e2e/run_pipeline_compare.py --workdir /tmp/d10c --workers 6   # same, keeps the work dir
$PY sensor_rig/tests/e2e/run_pipeline_compare.py --workdir /tmp/d10c30 --main-fps 30 [--ts-jitter-ns 1000000 --drop-rate 0.01]   # fill mode
$PY sensor_rig/tests/e2e/run_collect_dryrun.py --workdir /tmp/dry      # collect --mock-replay -> ... -> zarr
```

Fill-mode runs use the separate, justified tolerances `TOL_FILL` in `run_pipeline_compare.py`. Native tolerances
are unchanged.

D10c simulates rig recordings from real iPhone demos (`sensor-rig sim`), then runs pose-import, export and
validate. It runs the real iPhUMI pipeline on the originals and on the exports in isolated work dirs and compares the
two zarrs (keys, lengths, relative TCP motion, gripper width, images, tag detection). It writes `report.json`, exits
0 only if every check passes, and never writes to iPhUMI's `tmp_*` folders or `data/raw`.
