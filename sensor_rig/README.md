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
Jetson : python -m sensor_rig record      -> raw rig episode  <raw>/<YYYY-MM-DD>/<NAME>/   (JPEG + CSV + meta.json + rig.yaml)
laptop : python -m sensor_rig pose-prep   -> EuRoC folder + ORB-SLAM3 settings (or OpenVINS/Kalibr yamls)
extern : ORB-SLAM3 mono-inertial (or OpenVINS) -> trajectory file
laptop : python -m sensor_rig pose-import -> <NAME>/trajectory/trajectory.{tum,json}  (with sanity checks)
laptop : python -m sensor_rig export      -> <out>/<YYYY-MM-DD>/<NAME>/<NAME>{.json,_rgb.mp4,_ultrawidergb.mp4}
laptop : python -m sensor_rig validate    -> checks every iPhone-format invariant (export also runs it)
then   : the unchanged iPhUMI pipeline, with group.iphone_dir=<out>
```

The raw episode folder name is already the final iPhone-style NAME
(`<UTC time with ':'→'-'>_<5 digits>_<session>_<demonstration|grippercalibration>_<side>`). It is chosen at record
start, like the app does.

## Install and test

The package needs numpy, opencv, pyyaml, scipy and PyAV. Install it with `pip install -e sensor_rig` (extras:
`[jetson]`, `[rosbags]`, `[test]`). The tests don't need an install: `tests/conftest.py` puts `sensor_rig/` on
`sys.path`. `python -m sensor_rig` does need the install when run from the repo root. Without it, Python imports the
project folder `sensor_rig/` as an empty namespace package and fails with "cannot be directly executed".

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
python -m sensor_rig pose-prep --raw <episode|date dir|raw root> --out <dir> --target orbslam3 --scale 0.5 --imu-rate 200
<dir>/<NAME>/run_orbslam3.sh                      # needs $ORB_SLAM3; writes f_<NAME>.txt
python -m sensor_rig pose-import --raw <episode> --traj f_<NAME>.txt --format orbslam3-euroc --frame imu
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

  It refuses on a failed overlap or gravity check unless you pass `--force`. Otherwise it writes
  `trajectory/trajectory.tum` and `trajectory.json` (frame, checks, source sha256).

## Export

```bash
python -m sensor_rig export --raw <raw root|date dir|episode ...> --out <export_root> [--uw-rate 10|all]
       [--label-task block_raise] [--calibration-run NAME] [--rig better_rig.yaml] [--trajectory f.tum
       --trajectory-frame imu] [--workers N] [--overwrite] [--force]
python -m sensor_rig validate <export_root|episode> [--gripper-calibration .../gripper_calibration.yaml] [--decode]
```

What export does (code in `sensor_rig/export/`):

- **Grid** (`grid.py`): the main camera drives the 60 Hz record, because the pipeline hardcodes 1/60.
  - Main camera at ≥ 50 fps: the grid is the actual main frame times (dropped frames stay gaps, like the iPhone).
  - Slower main camera (e.g. 30 fps): a uniform 60 Hz grid where each slot shows the **nearest** main frame (zero-mean
    image-vs-pose time error; exact ties alternate).
  - Episodes with fewer than 31 frames in any 1-s window after the first second abort. In fill mode the threshold
    is 0.52 × nominal fps.
- **Poses** (`exporter.py`, `pose/`): linear position + SLERP rotation at grid times.
  - Slots without a pose (VIO not yet initialised, or lost) at the head or tail are trimmed. A gap in the middle
    aborts the episode (`pose_gap_mid_episode`).
  - It also aborts on: speed > 7 m/s, an identity rotation, NaN, or duration < 1 s.
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

## Recording (Jetson)

```bash
python -m sensor_rig record --rig <calibrated rig.yaml> --out <raw root> --session firstrig --side left   # see --help
python -m sensor_rig record --mock --duration 5 ...      # same recorder with mock cameras + IMU, runs on a laptop
python -m sensor_rig bench-encode                        # JPEG encode budget check (run it on the Orin Nano)
```

Code in `sensor_rig/drivers/` (backends) and `sensor_rig/record/` (recorder). Everything is stamped with
CLOCK_MONOTONIC ns. Argus stamps (MONOTONIC_RAW) are converted in the backend, and ICM-42688-P FIFO timestamps are
mapped to the host clock by `drivers/chip_clock.py`. Rig offsets (`rig.yaml` `cameras.*.time`) are applied only at
export and deployment, never on disk. Live abort rules mirror the app (fps, disk space, sensor errors). The Argus and
SPI backends have not run on hardware yet; see the checklist above.

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
```

D10c simulates rig recordings from real iPhone demos (`python -m sensor_rig sim`), then runs pose-import, export and
validate. It runs the real iPhUMI pipeline on the originals and on the exports in isolated work dirs and compares the
two zarrs (keys, lengths, relative TCP motion, gripper width, images, tag detection). It writes `report.json`, exits
0 only if every check passes, and never writes to iPhUMI's `tmp_*` folders or `data/raw`.
