# dataproc design

`dataproc` runs on the Jetson. It takes the recordings the Pi 5 delivered to `/data/incoming/` (its format:
[PI_ADAPTER.md](PI_ADAPTER.md); dataproc's own raw format: [CONTRACT.md](CONTRACT.md)) and turns them into training
`.zarr.zip` files that train with both `umi_policy_dunetp` (behavior prompting) and `umi_policy_dunet` (plain
diffusion), unchanged. Everything runs offline, after recording.

## Commands

```
dataproc calibrate TASK EPISODE_DIR [--source /data/incoming] [--rig rig_pi_v1]
dataproc run TASK_FOLDER START_DIR END_DIR [--source /data/incoming] [--slam auto|orbslam3|oracle|none]
                                          [--retry-failed] [--force] [--dry-run]
dataproc package DATASET FOLDER [FOLDER ...] [--overwrite]
dataproc status TASK_FOLDER
dataproc sim --out DIR [--episodes N] [--calibration] [--fast] [--imu-gap SEC] [--contract] ...
```

- **`calibrate`** (`dataproc/calibrate/`): fit and check the gripper calibration on one recording **before**
  anything is created. On success, create a new task folder (`TASK`, else `TASK_2`, `TASK_3`, …) holding it; on
  failure print why and create nothing. **One folder = one calibration = one session** on one hardware setup.
  Details: [PI_ADAPTER.md](PI_ADAPTER.md).
- **`run`** (`dataproc/run/`): process the recordings START_DIR..END_DIR (inclusive) of a calibrated folder, one at
  a time, with the folder's rig and calibration. Only recordings that pass every stage become `episode_N`
  (numbered from **`episode_1`**); the others go to `failed/`. It builds no zip. Details:
  [PI_ADAPTER.md](PI_ADAPTER.md); the stages: [PROCESS.md](PROCESS.md).
- **`package`** (`dataproc/package/`): combine the exported episodes of any task folders (sessions of one task
  and/or several tasks) into one `.zarr.zip` in the cluster's dataset layout ([EXPORT.md](EXPORT.md)).
- **`status`** (`dataproc/status.py`, read only) and **`sim`** (`dataproc/sim/`, synthetic Pi-format recordings
  with ground truth, [SIM.md](SIM.md)).

`dataproc/cli.py` only parses the arguments and calls each folder's `cli_*` function. `intake`
([INTAKE.md](INTAKE.md)) and `process` ([PROCESS.md](PROCESS.md)) are modules that `calibrate` and `run` use; they
are no longer commands.

## Folder layout

Root: `$DATAPROC_ROOT`, default `~/data` (`dataproc/paths.py`).

```
pi_sources.csv                         every Pi recording taken (calibration / demonstration; ingested / failed /
                                       quarantined), keyed by the sha256 of its MANIFEST.sha256
incoming/                              brief staging of converted copies (.adapt-tmp/ while converting); empty
                                       between commands
tasks/<folder>/                        one calibration = one session: block_raise, block_raise_2, ...
  task.yaml                            label (= TASK, also for block_raise_2), rig, created, notes
  episodes.csv                         ledger: one row per slot a recording entered (append-only)
  gripper_range.json                   the calibration result (copy)
  calibration/<pi_name>/{raw/, processed/}
  episode_N/{raw/, processed/}         passed every stage (processed/episode.zarr exists); N from 1, gap-free
  failed/<pi_name>/{raw/, processed/, reason.json}   failed or rejected at some stage; never packaged
  .staging/<pi_name>/{raw/, processed/}              in progress, or waiting for SLAM; never packaged
  report.json                          the last run: outcome per recording
datasets/<label>/<DATASET>/<DATASET>.zarr.zip + package.json      written by package (several labels: multi/)
rigs/<rig_id>.yaml                     rig description, copied from configs/rigs/ if missing
quarantine/<pi_name>[__k]/             recordings intake rejected (a duplicate) + reason.json
.run.lock                              flock held by calibrate / run
```

`raw/` is the converted recording, moved in unchanged and never written to afterwards; `processed/` holds the stage
outputs and `status.json`. Directories sort numerically (`episode_2` before `episode_10`). Task, folder, dataset and
rig names are `[A-Za-z0-9_-]+`; Pi recording dir names are `[A-Za-z0-9][A-Za-z0-9._-]*`
(`paths.validate_episode_dir_name`). Stage status and inputs hashes do not depend on the episode dir's path, so the
`.staging/<pi_name>` → `episode_N` rename keeps every processed output valid.

## Intake (used by `calibrate` and `run`)

`dataproc/intake.py` files converted copies from `incoming/` into a task folder: contract checks, dedupe, ledger,
quarantine, crash-safe moves. `calibrate` uses it with a new folder (`TASK`, `TASK_2`, …) and the calibration slot;
`run` with the **staging target** (an existing folder; the recording goes to `.staging/<pi_name>/raw/`, and `run`
later moves it to `episode_N` or `failed/`). Rules and decisions: [INTAKE.md](INTAKE.md).

- **Checks**, each failure → `quarantine/<pi_name>/` with `reason.json`: every `RawEpisode.validate` error;
  `duplicate` (the Pi dir name or the (`session_id`, `episode_number`) pair is already in the `episodes.csv` of any
  task folder); `rig_mismatch` (`meta.json` `rig_id` ≠ `task.yaml` `rig`).
- **Moves** are `os.rename` on one filesystem; `EXDEV` is an error, never a copy.
- **`episodes.csv`**: `slot,pi_name,session_id,episode_number,kind,rig_id,start_unix_ns,intake_unix_ns,
  manifest_sha256`. `slot` = `episode_N`, `calibration/<name>`, `.staging/<name>` or `failed/<name>`; the newest
  row of a recording is where it is now.

## Conventions

**Frames.** The training world frame is gravity-aligned **z-up** (sensor_rig `gravity_align` convention).
`world_offset` (rig yaml, optional, default translation [0.1, 0.2, 0.3] m and yaw 30°) is a yaw about +z then a
translation, applied on the left; its yaw must be at least 1° away from a multiple of 360 so poses near the SLAM origin never have rotation
vector 0 (iPhUMI's `pose_6d_to_4x4` is NaN there). `T_cam_tcp` is in the **physical** ultrawide optical frame; the
virtual ultrawide is rotated from it by `virtual_ultrawide.rotation` (`R_phys_virt`, `d_phys = R_phys_virt @ d_virt`,
default identity). ArUco tag `tvec` and `aruco.nominal_z_m` are in the virtual frame.

**Time.** Raw `t_ns` is the libcamera SensorTimestamp verbatim (CONTRACT §2). The Jetson owns its meaning via
`cameras.<role>.timestamp_ref` in the rig yaml (`start_of_exposure` | `end_of_exposure`; rig_v1: `end_of_exposure`,
nominal):

```
mid-exposure camera time  t_cam = t_ns + s * (exposure_ns // 2) + time_offset_ns
                          s = +1 for start_of_exposure, -1 for end_of_exposure
IMU time                  t_imu = t_ns + imu.time_offset_ns
```

Kalibr: `time_offset_ns = round(timeshift_cam_imu_s * 1e9)`, with Kalibr run on these mid-exposure stamps; a
constant rolling-shutter half-readout is absorbed in `time_offset_ns`. Code: `RigConfig.camera_time_ns(role, t_ns,
exposure_ns)` and `RigConfig.imu_time_ns(t_ns)`. VIO noise: `imu.noise` × the optional `imu.vio_inflation`
factors (`ImuNoise.for_vio()`).

**SLAM is pluggable.** A backend takes the raw episode + rig and returns a camera trajectory with validity:
- `orbslam3`: ORB-SLAM3 mono-inertial (native build on the Jetson, [ORBSLAM3.md](ORBSLAM3.md)) on the fisheye
  ultrawide + IMU;
- `oracle`: synthetic truth + noise, for tests.

## Output zarr contract

Same keys, dtypes, chunks and codec as `scripts/regression/golden/iphone_firstreal_5ep.json` (iPhUMI's output),
written as a `.zarr.zip` with zarr 2.18.3's `ZipStore`. Groups `data/`, `meta/`, `labels/` (empty); no root
attributes. N = total main-rate steps (60 Hz), M = total ultrawide steps (~10 Hz), E = episodes.

| Array | Shape | dtype | Chunks | Compressor |
|---|---|---|---|---|
| `data/camera_right_main_rgb` | (N, 224, 224, 3) | uint8 | (1, 224, 224, 3) | `imagecodecs_jpegxl`, level 99 |
| `data/camera_right_ultrawide_rgb` | (M, 224, 224, 3) | uint8 | (1, 224, 224, 3) | `imagecodecs_jpegxl`, level 99 |
| `data/gripper_right_eef_pos` | (N, 3) | float32 | replay-buffer optimal | none |
| `data/gripper_right_eef_rot_axis_angle` | (N, 3) | float32 | replay-buffer optimal | none |
| `data/gripper_right_gripper_width` | (N, 1) | float32 | replay-buffer optimal | none |
| `meta/episode_ends` | (E,) | int64 | (E,) | none |
| `meta/episode_names` | (E,) | object (`vlen-utf8`) | (E,) | none |
| `meta/is_episode_error_correction` | (E,) | bool | (E,) | none |
| `meta/task_names` | (E,) | object (`vlen-utf8`) | (E,) | none |
| `meta/task_lengths`, `meta/task_data_ends`, `meta/task_labels_ends` | (E,) | int64 | (E,) | none |
| `meta/is_task_error_correction` | (E,) | bool | (E,) | none |
| `meta/episode_ends_camera_right_ultrawide_rgb` | (E,) | int64 | (E,) | none |
| `meta/downsample_index_camera_right_ultrawide_rgb` | (M,) | int64 | (M,) | none |
| `meta/upsample_index_camera_right_ultrawide_rgb` | (N,) | int64 | replay-buffer optimal | none |

Semantics (from the golden file): `episode_names` = the Pi episode dir name (`pi_name` in `episodes.csv`); one
task per episode; `task_names` = `task.yaml` `label`; `task_lengths` = episode length − 1;
`task_labels_ends` = `cumsum(task_lengths)`; `task_data_ends` = `episode_ends − 1`; the ultrawide maps relate the ~10 Hz ultrawide stream
to the 60 Hz main stream exactly as iPhUMI writes them. Positions in metres, rotations as axis-angle (rotvec) of
the TCP in the world frame, width in metres.

## Code organisation

| Module | Role |
|---|---|
| `dataproc/cli.py` | argparse front end: parses, then calls `<command>/cli.py` |
| `dataproc/calibrate/`, `dataproc/run/`, `dataproc/package/` | one folder per command (`cli.py` prints and returns the exit code) |
| `dataproc/pi_adapter.py`, `dataproc/pi_intake.py` | Pi → contract conversion; the adapt + intake part calibrate and run share |
| `dataproc/pi_format.py` | the reverse (contract → Pi format), for `dataproc sim` and the tests |
| `dataproc/intake.py`, `dataproc/process.py`, `dataproc/stages/`, `dataproc/export/` | intake, the stages, the zarr writer |
| `dataproc/paths.py` | root, task-folder and episode-dir resolution (pure path logic) |
| `dataproc/raw_episode.py` | the single reader/validator of a raw episode |
| `dataproc/config.py` | `task.yaml` and rig yaml schemas |

Code taken from `sensor_rig/` or `iPhUMI/` is vendored (copied) with its source path and commit in a header
comment; `dataproc` imports neither.

## Development environment

The versions below (they match iPhUMI's `environment.yml`) are pinned in `pyproject.toml`. On the Jetson (Python
3.12, aarch64) the same pins install; `numcodecs` 0.13.1 has no aarch64/3.12 wheel there and is compiled (needs
`build-essential`). Laptop recipe (Python 3.10 venv):

```bash
ENV=/tmp/claude-1000/-home-ruben-Capstone/5f0864d3-a420-45bc-b87f-4236508c9d74/scratchpad/env_dataproc
/usr/bin/python3.10 -m venv $ENV
$ENV/bin/pip install --upgrade pip
$ENV/bin/pip install "numpy==2.2.*" "opencv-python-headless==4.12.*" "zarr==2.18.3" "imagecodecs==2025.3.30" \
    numcodecs av scipy pyyaml transforms3d numba pytest
$ENV/bin/pip install -e "data-processing[test]"          # from the repo root
$ENV/bin/python -m pytest data-processing/tests -q       # from the repo root
```

Installed: numpy 2.2.6, opencv-python-headless 4.12.0.88, zarr 2.18.3, imagecodecs 2025.3.30, numcodecs 0.13.1,
av 17.1.0, scipy 1.15.3, PyYAML 6.0.3, transforms3d 0.4.2, numba 0.68.1, pytest 9.2.0.

On the laptop, ROS Humble's `PYTHONPATH` leaks into the venv: pip prints harmless dependency warnings about ROS
packages, and the ROS pytest plugins are disabled in `pyproject.toml` (`addopts`).
