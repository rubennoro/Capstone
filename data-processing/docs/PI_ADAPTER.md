# Pi episodes → dataproc (`dataproc run`, `dataproc/pi_adapter.py`)

The Pi 5 rig delivers episodes into `/data/incoming/<dir>/` in its own format. That folder is the ground truth:
dataproc only reads it. `dataproc run` converts each episode into a contract copy (`docs/CONTRACT.md`,
`dataproc.raw.v1`), then runs the normal intake and process on the copy.

```bash
dataproc run --task block_raise --calibration-episode episode_000001   # first time: names the calibration recording
dataproc run --task block_raise                                        # later: takes new episodes, or resumes
dataproc run --task block_raise --dry-run                              # only lists what it would take
```

- `--task`: the task folder (`$DATAPROC_ROOT/tasks/<task>/`) and the label written into the zarr.
- `--calibration-episode`: a directory name in `--source` holding the gripper-calibration recording (the gripper
  opening and closing with the finger ArUco tags in view). It is required while the task has no calibration.
- Every other complete, valid source directory that was never taken before is a demonstration of `--task`.
- `--episodes DIR [DIR ...]`: take only these source directories as demonstrations. The rest stay for later runs.
- `--no-calibration`: process a task that has no calibration episode. Everything up to the width stage runs
  (convert, check, SLAM) and is kept. Gripper width needs the calibration, so no `episode.zarr` and no zip are
  built until you run again with `--calibration-episode`; then only width → export run.
- `--source` (default `/data/incoming`), `--rig` (default `rig_pi_v1`), `--slam auto|orbslam3|none|oracle`
  (default `auto` = ORB-SLAM3 if `~/ORB_SLAM3` (or `$ORB_SLAM3_DIR`) has the built binary, else `none`).
- Exit codes: 0 = `output/<task>.zarr.zip` built, 3 = stopped at SLAM (`--slam none`), 1 = anything else.

## Steps

1. **Plan.** Scan `--source`. A source is skipped (and listed) when it has no `COMPLETE` (still syncing), a file
   doesn't match its `MANIFEST.sha256`, the Pi marked it `valid: false` or with `problems`, or it was already
   taken (`$DATAPROC_ROOT/pi_sources.csv`, keyed by the sha256 of its manifest).
2. **Adapt** each source into `$DATAPROC_ROOT/incoming/pi-<date>_episode_NNNNNN/`. It is built under
   `incoming/.adapt-tmp/`, contract-checked, given `.verified`, and then renamed into place.
3. **Intake** (unchanged): the calibration goes to `calibration/<name>/`, the demonstrations to `episode_N/`.
   Contract failures go to `quarantine/` with a `reason.json`; the source stays untouched either way.
4. **Record** every taken source (ingested or quarantined) in `pi_sources.csv`. To retry a *quarantined* source after fixing it, delete its row (an ingested one would be rejected as a
   duplicate; reprocess it with `dataproc process --episode N --force` instead).
5. **Process** (unchanged): calibration → check → slam → grid → pose → width → validate → images → export → zip.
   Already-done stages are skipped, so re-running `dataproc run --task T` after installing SLAM resumes from slam.

## Field mapping

| Contract (`dataproc.raw.v1`) | From the Pi episode |
|---|---|
| `cam0` = main | the camera whose `model` is `imx519` (the Pi names it `cam1`) |
| `cam1` = ultrawide | the camera whose `model` is `imx219` (the Pi names it `cam0`) |
| `camX.mp4` | copied byte for byte |
| `frame_idx,t_ns` | `frame_index,t_ns` (verbatim) |
| `sensor_seq` | Pi column if present, else synthesized: cumulative `round(Δt / frame period)` (a drop = a gap) |
| `exposure_ns` | Pi column if present, else **5 ms** (capped at half a frame period) |
| `imu.csv` gyro (rad/s) | counts × `gyro_dps_per_lsb` × π/180 |
| `imu.csv` accel (m/s²) | counts × `accel_g_per_lsb` × 9.80665 |
| `sample_idx` | `sample_index` |
| `clock` | `CLOCK_BOOTTIME` |
| `session_id` | `pi-<local date of COMPLETE>` |
| `episode_number` | Pi `episode` |
| `rig_id` | `--rig` |
| `button_start_ns` / `button_stop_ns` | the overlap of the three streams: **the whole recording** is training data |
| `wall_anchor` | (`t_end_boottime_ns`, mtime of `COMPLETE`): approximate, only used to order episodes |
| `pi_meta`, `adapter` | the Pi's original meta.json, and what was synthesized |
| other files in the Pi manifest | copied and listed |

## Assumptions and limits

- **Exposure.** Mid-exposure times use the 5 ms guess until the Pi records libcamera `ExposureTime` as an
  `exposure_ns` column; the adapter uses the column automatically once it exists.
- **Whole clip.** With no button presses, idle time before and after the demonstration is trained on. Trim at
  recording time, or add button times to the Pi meta later.
- **Session ids** are dates. If the Pi's episode counter restarts on the same day, the repeated
  (`session_id`, `episode_number`) pair is quarantined as `duplicate` (visible, never silent).
- **SLAM needs motion.** Mono-inertial ORB-SLAM3 makes no map from a static recording. The slam stage then fails
  with `never initialised (0 keyframes)`.
- **Calibration needs both finger tags visible** while the gripper opens and closes, or it fails with
  `finger tags 0/1 never both detected` and every episode stops at width.
- **Rig values are nominal** (`configs/rigs/rig_pi_v1.yaml`) until the cameras, IMU and TCP are calibrated.
