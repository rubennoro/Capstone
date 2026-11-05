# dataproc design

`dataproc` runs on the Jetson. It takes raw episodes that the Pi 5 delivered to `incoming/` (format:
[CONTRACT.md](CONTRACT.md)) and turns a task's episodes into one `output/<folder>.zarr.zip` that trains with both
`umi_policy_dunetp` (behavior prompting) and `umi_policy_dunet` (plain diffusion), unchanged. Everything runs
offline, after recording.

## Folder layout

Root: `$DATAPROC_ROOT`, default `/data/rig` (`dataproc/paths.py`).

```
incoming/<pi_episode>/                 as delivered by the Pi (+ .verified from the Jetson)
tasks/<folder>/
  task.yaml                            label, rig, created, notes (dataproc/config.py)
  episodes.csv                         one row per intaken episode
  calibration/<pi_episode>[__k]/raw/   gripper-calibration episodes (--calibration)
  episode_N/raw/                       the raw episode, moved unchanged; never written to afterwards
  episode_N/processed/                 stage outputs + status.json
  output/<folder>.zarr.zip, output/report.json
rigs/<rig_id>.yaml                     rig description (configs/rigs/rig_v1.yaml is the nominal one)
quarantine/<pi_episode>[__k]/          rejected episodes + reason.json
.intake.lock                           flock held by a running intake
```

`episode_N` starts at `episode_0` (no leading zeros); directories sort numerically (`episode_2` before
`episode_10`). Task, folder and rig names are `[A-Za-z0-9_-]+`; Pi episode dir names are
`[A-Za-z0-9][A-Za-z0-9._-]*` (`paths.validate_episode_dir_name`). `--task` accepts a folder name, `name/episode_N`, or a path inside
`<root>/tasks/`.

## `dataproc intake --task NAME [--new] [--calibration]`

- `--task` is required.
- **Lock:** `fcntl.flock(LOCK_EX | LOCK_NB)` on `<root>/.intake.lock`; if it is held, exit 1 with "another intake
  is running".
- **Target folder** (`paths.resolve_intake_target`):
  - no folder → create `<task>/`;
  - the folder exists → add to it (it must be a directory);
  - `--new` while it exists → create the first free `<task>_1`, `<task>_2`, … (any existing entry counts as taken).
- **Candidates:** only `incoming/` dirs that have `.verified`. A dir that contains any other dot-prefixed entry (an
  rsync temp file: transfer or re-send in progress) is **left in `incoming/`** and reported as "skipped: transfer in
  progress". Nothing waiting → no folder is created.
- **Order:** an episode whose `meta.json` is invalid is quarantined first, before sorting. The rest are taken
  oldest first by `EpisodeMeta.start_unix_ns`, ties broken by (`session_id`, `episode_number`).
- **Checks**, each failure → quarantine with that reason:
  - every `RawEpisode.validate` error (manifest + counts; reason = the problem codes);
  - `duplicate`: the Pi dir name or the (`session_id`, `episode_number`) pair already appears in the
    `episodes.csv` of **any** task folder (never ingest twice, never delete);
  - `rig_mismatch`: `meta.json` `rig_id` differs from the target `task.yaml` `rig`.
- **Quarantine:** `quarantine/<pi_name>/` + `reason.json`; on a name collision `<pi_name>__2`, `__3`, …
- **Move:** with `os.rename` (same filesystem) to `episode_N/raw/`; if it fails with `EXDEV`, error out, never copy.
  Numbering starts at `episode_0`, otherwise max existing + 1 (gaps are not refilled; any `episode_N` entry counts).
- `--calibration` sends the next waiting episode to `calibration/<pi_name>/raw/` instead (`__2`, `__3`, … on a
  name collision).
- **`task.yaml`** is written on folder creation, atomically (temp file + rename), with `label` = the task name (so
  `block_raise_1` is still labelled `block_raise`) and `rig` taken from `meta.json`.
- **`episodes.csv`**: header + one row per intaken episode, append-only (temp + rename, or an fsync'd append):

  ```
  slot,pi_name,session_id,episode_number,kind,rig_id,start_unix_ns,intake_unix_ns,manifest_sha256
  ```

  `slot` = `episode_N` or `calibration/<name>`; `kind` = `demonstration` | `calibration`; `manifest_sha256` = the
  sha256 of the `MANIFEST.sha256` file.

## `dataproc process --task NAME [--episode N] [--force]`

- `--task` is required. The folder must exist ("run intake first"); `process` never creates task folders.
- Never writes into `raw/`.
- Skips stages whose `inputs_hash` (in `processed/status.json`) is unchanged, unless `--force`.
- A failed or rejected episode is recorded, and the run continues.
- Always rebuilds `output/<folder>.zarr.zip` + `output/report.json` from every valid episode.

`dataproc status --task NAME` prints the per-episode status. `dataproc sim --out DIR --episodes N [--calibration]`
writes synthetic episodes in the raw format plus `truth.npz` (pose, width), for tests.

## Stages

Per episode, in order (each writes into `processed/` and records its `inputs_hash`; the orchestrator, wave 2,
fixes the exact stage names and files):

1. **check**: `RawEpisode.validate` (the contract; errors reject the episode). Also rejects when a `meta.json`
   camera `resolution` or `fps` differs from `rigs/<rig>.yaml`, or `meta.json` `rig_id` differs from `task.yaml`
   `rig`.
2. **images**: decode; remap the fisheye ultrawide to the virtual pinhole (`rigs/<rig>.yaml` `virtual_ultrawide`,
   640×480, fx 246.6); letterbox both cameras to 224×224; pick the ~10 Hz ultrawide sub-stream.
3. **pose**: SLAM backend → gravity-aligned z-up `T_W_cam(t)` → `T_W_tcp = T_world_offset @ T_W_cam @ T_cam_tcp`,
   sampled on a fixed 60 Hz grid over `[button_start, button_stop]` (trimmed to where pose is valid). Inputs
   include the rig's `T_cam_tcp`, `world_offset` and time model (below).
4. **width**: ArUco `DICT_4X4_50` 16 mm finger tags on the virtual ultrawide, zero-distortion PnP,
   `right.x − left.x`, mapped with the task's gripper range.
5. **validate**: thresholds: speed > 3 m/s and > 7 m/s, ultrawide gap / IMU gap > 20 ms, tag coverage, minimum
   length; trimming.

Per task: **gripper_range** (from `calibration/`: min/max width and the left↔right linear maps, iPhUMI math) and
**zarr** (merge every valid episode into `output/<folder>.zarr.zip`, write `report.json`).

**Frames.** The training world frame is gravity-aligned **z-up** (sensor_rig `gravity_align` convention).
`world_offset` (rig yaml, optional, default translation [0.1, 0.2, 0.3] m and yaw 30°) is a yaw about +z then a
translation, applied on the left; its yaw must be non-zero so poses near the SLAM origin never have rotation
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
- `orbslam3`: ORB-SLAM3 mono-inertial in Docker on the fisheye ultrawide + IMU;
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
| `dataproc/paths.py` | root, task-folder and episode-dir resolution (pure path logic) |
| `dataproc/raw_episode.py` | the single reader/validator of a raw episode |
| `dataproc/config.py` | `task.yaml` and rig yaml schemas |
| `dataproc/cli.py` | argparse front end |

Code taken from `sensor_rig/` or `iPhUMI/` is vendored (copied) with its source path and commit in a header
comment; `dataproc` imports neither.

## Development environment

Python 3.10 venv (versions match iPhUMI's `environment.yml`):

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
