# Pi recordings → dataproc (`dataproc calibrate` / `run`, `dataproc/pi_adapter.py`)

The Pi 5 rig delivers recordings into `/data/incoming/<dir>/` in its own format. That folder is the ground truth:
dataproc only reads it. `calibrate` and `run` convert each recording into a contract copy (`docs/CONTRACT.md`,
`dataproc.raw.v1`), then use the normal intake and processing on the copy. The shared part (rig copy, staging,
adapt, intake, `pi_sources.csv`) is `dataproc/pi_intake.py`.

```bash
dataproc calibrate block_raise episode_000001                        # -> tasks/block_raise (or block_raise_2, ...)
dataproc run block_raise episode_000002 episode_000030               # the range, inclusive, natural sort
dataproc run block_raise episode_000002 episode_000030 --dry-run     # only lists what it would do
dataproc package block_raise-1110 block_raise                        # -> datasets/block_raise/block_raise-1110/
```

## `dataproc calibrate TASK EPISODE_DIR [--source DIR] [--rig RIG_ID]` (`dataproc/calibrate/`)

1. **Inspect** `--source/EPISODE_DIR` (read only). It fails, with nothing created, when the directory is missing,
   has no `COMPLETE` (still syncing), doesn't match its `MANIFEST.sha256`, the Pi marked it `valid: false` or with
   `problems`, or it is already in `$DATAPROC_ROOT/pi_sources.csv`.
2. Under `FileLock($DATAPROC_ROOT/.run.lock)`: copy `configs/rigs/<rig>.yaml` to `rigs/` if missing (an existing
   copy without the `gripper` section while the repo's has one is an error: it predates the opening check), require
   an empty `incoming/`, and **adapt** the recording into `incoming/pi-<date>_episode_NNNNNN/` (built under
   `incoming/.adapt-tmp/`, contract-checked, then renamed into place).
3. **Fit and check** (`stages/calibration.fit_calibration`): contract + video decode; rig id and camera
   resolution/fps vs `--rig` (`rig_mismatch`, `rig_resolution_mismatch`, `rig_fps_mismatch`); ArUco tags on every
   ultrawide frame; `fit_gripper_range`; then the **opening check**: `max_width - min_width` must be within
   `gripper.opening_tolerance_m` of `gripper.max_opening_m` (`rig_pi_v1`: 0.110 m ± 0.010 m), else
   `opening_out_of_range`. Fewer than 5 open/close cycles is only a warning.
4. **Failure:** prints the tag statistics and likely causes (`dataproc/calibrate/diagnostics.py`), deletes the staged
   copy, creates no folder and records nothing, so the same recording can be retried. Exit 1.
5. **Success:** intake the staged copy as `calibration/<pi_name>/` of the first free folder `TASK`, `TASK_2`,
   `TASK_3`, … (`task.yaml`: `label: TASK`, `rig: RIG_ID`), write the fit to `calibration/<pi_name>/processed/` and
   `gripper_range.json` exactly as `CalibrationStep` reuses it (it is never fitted again), and record the recording
   in `pi_sources.csv` (kind `calibration`). Exit 0.

## `dataproc run TASK_FOLDER START_DIR END_DIR [...]` (`dataproc/run/`)

1. **Plan** (`run/plan.py`): `tasks/TASK_FOLDER` must exist with a successful calibration, else exit 1 ("run
   `dataproc calibrate <task> <episode_dir>` first"). START_DIR and END_DIR must be directories in `--source` with
   START ≤ END (natural sort); the range is every non-dot directory between them, inclusive. Each one is **skipped
   and listed** (nothing changes for it) when it is still syncing, fails its manifest, is marked invalid by the Pi,
   is a calibration, is already an episode of this folder, was taken by another folder, or failed earlier
   (`failed earlier at <stage>: <reason>`, unless `--retry-failed`). `--dry-run` prints this and stops.
2. Under `FileLock(.run.lock)`: the rig copy, an empty `incoming/`, and repair of what a crashed run left
   (`run/staging.finish_interrupted`). The folder context (task.yaml, its rig, the SLAM backend, the calibration)
   is loaded once.
3. **Each recording, one at a time** (`run/episode.py`): adapt it, intake it into
   `tasks/<folder>/.staging/<pi_name>/raw/` (recorded in `pi_sources.csv`), run the per-episode stages there, then:
   - passed every stage (export done, `processed/episode.zarr` written): renamed to the next free **`episode_N`,
     starting at `episode_1`**;
   - failed or rejected at any stage: renamed to `failed/<pi_name>/` with `reason.json` (`stage`, `reasons`,
     `message`, time); `pi_sources.csv` status `failed`;
   - no SLAM backend (`--slam none`, or `auto` without ORB-SLAM3): stays in `.staging/`; any later `run` resumes it
     from slam. Exit 3.
   Every move is one same-filesystem rename. A recording that cannot be converted is skipped and not recorded (it
   can be retried).
4. **Summary:** one line per recording, then `dataproc status`, then the `dataproc package` command.
   `tasks/<folder>/report.json` holds the per-recording outcome. Exit 0 = done (also with failures), 3 = waiting for
   SLAM, 1 = the run could not run.

`--retry-failed` moves the range's `failed/<pi_name>/` back to `.staging/` and processes them again (stages whose
inputs changed re-run; add `--force` to re-run all). `--force` re-runs every stage of the selected range (also of
existing episodes, in place). To retry a *quarantined* recording (intake: a duplicate) after fixing it, delete its
row from `pi_sources.csv`.

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
- **Calibration needs both finger tags visible** while the gripper opens and closes fully, or `dataproc calibrate`
  fails (and creates no folder).
- **Rig values are nominal** (`configs/rigs/rig_pi_v1.yaml`) until the cameras, IMU and TCP are calibrated.
