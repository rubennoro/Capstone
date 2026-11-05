# `dataproc process` and `dataproc status`

`process` turns a task folder's raw episodes into `output/<folder>.zarr.zip` (schema: [DESIGN.md](DESIGN.md),
"Output zarr contract") plus `output/report.json`. `status` shows where every episode is. The step names below are
the method and class names in the code, in the same order.

```bash
dataproc process --task NAME [--episode N] [--force] [--slam orbslam3|oracle]   # default --slam orbslam3
dataproc status  --task NAME
```

`--slam oracle` uses the synthetic ground truth (`raw/truth.npz`, written by `dataproc sim`). On an episode without
that file, the slam stage fails with `oracle_no_truth` and a message pointing to `--slam orbslam3`.

## Code layout

| Module | Contents |
|---|---|
| `dataproc/process.py` | `TaskProcessor` (the command), `EpisodeProcessor` (one episode), `run_process`, `cli_process` |
| `dataproc/stages/base.py` | the `Stage` interface, `StageFailed`, `ProcessContext`, `Episode`, `Calibration`, `rig_sections` |
| `dataproc/stages/calibration.py` | `CalibrationStep` (task level), `newest_calibration` |
| `dataproc/stages/check.py` | `CheckStage` (+ the pure `check_episode`, `rig_consistency`) |
| `dataproc/stages/geometry.py` | `SlamStage`, `GridStage`, `PoseStage`, `WidthStage` (they call `slam/`, `stages/pose.py`, `stages/width.py`) |
| `dataproc/stages/validate.py` | `ValidateStage` (+ the pure `validate_episode` and every threshold constant) |
| `dataproc/stages/output.py` | `ImagesStage`, `ExportStage` |
| `dataproc/files.py` | process-specific helpers: JSON (atomic), canonical hashing of stage inputs, `status.json` read/write |
| `dataproc/fsutil.py` | generic filesystem plumbing shared with intake: `write_atomic` (dot temp file + fsync + rename), renames, locks |
| `dataproc/report.py` | `episode_summary`, `format_table` (shared by process and status) |
| `dataproc/status.py` | `TaskStatus` (the status command) |

A stage is a class with `name`, `version`, `deps`, `inputs(ctx, ep) -> dict` (what goes into its hash) and
`run(ctx, ep) -> dict` (the `outputs` recorded in `status.json`). It raises
`StageFailed("failed" | "rejected", reasons, message)`.

## `TaskProcessor.run()`

1. **`load_task`**. Reads `tasks/<folder>/task.yaml`, `episodes.csv` and `rigs/<task.yaml rig>.yaml`. It lists
   the episodes: `episode_N` directories that have a `raw/`, in numeric order. Empty `episode_N/` directories and
   dot-entries (`.intake_pending.json`, temp files) are ignored. Writes nothing. **Exit 1, with nothing created**,
   when any of these holds:
   - the task folder is missing ("run intake first");
   - `task.yaml` is missing ("run intake first") or invalid;
   - the rig yaml is missing or invalid;
   - `episodes.csv` is malformed;
   - `--episode N` names no `episode_N` with a `raw/`.

   `process` never creates folders other than `processed/` (and its subdirectories) and `output/`. It never writes
   into `raw/`.
2. **`process_calibration`** (`CalibrationStep`). Picks the newest `calibration/<name>/raw`, by ledger
   `intake_unix_ns`; one without a ledger row counts as oldest. It runs these checks in order:
   - the contract check;
   - the calibration's `meta.rig_id` must equal the task's rig;
   - decode the ultrawide, remap each frame with the run's shared `ImagePrep.to_virtual` (`ctx.prep`), detect the
     tags;
   - `fit_gripper_range`.

   It writes `calibration/<name>/processed/{gripper_range.json, status.json}` and copies `gripper_range.json` to
   `tasks/<folder>/gripper_range.json`. The task-level copy always mirrors the current result: it is removed when
   there is no calibration or the newest one failed. The step is skipped (the saved range is reused) when its hash
   (manifest, `aruco`, `virtual_ultrawide`, `cameras.ultrawide`, task rig, version) is unchanged and its status is
   `done` or `failed`, unless `--force`. Results:
   - `done`;
   - `no_calibration`;
   - `calibration_failed`, with a reason: `invalid_episode`, `rig_mismatch`, `gripper_range_fit` or `exception`.

   An unexpected exception writes `calibration/<name>/processed/error.txt`, which is removed after a later
   successful fit. A saved `gripper_range.json` that cannot be loaded (for any reason) is fitted again.

   When the result is not `done`, every episode's width stage fails with `no_calibration` / `calibration_failed`.
3. **`episodes_to_process`**. All episodes, or only `--episode N`.
4. **`process_episode`**. Runs `EpisodeProcessor(ctx, episode).run()` on each episode, sequentially. One
   `ImagePrep` (remap table) is shared by the whole run.
5. **`build_output`**. Always rebuilds `output/<folder>.zarr.zip` from every episode whose export is `done`, also
   those not processed in this call (for example with `--episode N`), in `episode_N` order. The build writes a
   temp file and renames it, so a failed build leaves the previous zip as it was. **An existing zip is never
   deleted.**
   - The build fails (for example ENOSPC): the old zip is kept untouched and the traceback goes to
     `output/error.txt`. The report has `zip: null` and a `zip_note` naming the failure.
   - Nothing is exportable: no zip is built and the old zip is kept untouched. The report has `zip: null` and
     `zip_note: "no exportable episodes; previous zip kept and is stale"`.
   - In both cases the exit code is 1, and `dataproc status` shows the kept zip as `stale`. `output/error.txt` is
     removed after a later successful build.
6. **`write_report`**. Writes `output/report.json` and prints the summary table. The report has:
   - `task`, `label`, `rig_id`, `rig_nominal_sections` (`rig.nominal_sections()`), `slam_backend`,
     `dataproc_version`, `created`, `duration_s`;
   - `calibration`: `{state, name, reason, ran, gripper_range}`;
   - `episodes`: one row each, `{slot, pi_name, status, last_stage, stages, reasons, n_steps, ran_this_call}`;
   - `totals`: counts by status, plus `n_steps`;
   - `zip` (`path`, `n_episodes`, `n_steps`, `n_ultrawide`, `episode_names`) when this call built it, else
     `null` with a `zip_note`.

   **Exit code:** 0 when the zip was built. 1 when nothing is exportable or a fatal error occurred.

## `EpisodeProcessor.run()`

`STAGES = (CheckStage, SlamStage, GridStage, PoseStage, WidthStage, ValidateStage, ImagesStage, ExportStage)`.
Each stage goes through the same steps:
1. compute its `inputs_hash`;
2. skip it if it is up to date;
3. run it;
4. record it in `processed/status.json`.

The episode **stops at the first `failed` or `rejected` stage**, whether it failed now or in an earlier run. The
stages after it are recorded as `skipped`, with reason `"<stage> <status>"`.

- **`inputs_hash`** is the sha256 over canonical JSON of:
  - the stage name and `version`;
  - the dataproc `__version__`;
  - the episode's `MANIFEST.sha256` bytes;
  - the stage's `inputs()` (the parsed rig-yaml sections it uses, without `provenance`/`notes`, plus its constants);
  - the hashes of its `deps`.

  So the `--slam` backend name, `task.yaml` `label` and the `gripper_range.json` bytes reach every later stage
  through the chain. Only the backend *name* is hashed: the options of a backend passed in from Python (tests) are
  not.
- **Up to date** means all of the following:
  - no `--force`;
  - none of its `deps` ran in this call;
  - same `inputs_hash`;
  - last status `done`, `failed` or `rejected`.

  So failed and rejected episodes are retried only when their inputs change, or with `--force`. A stage that
  `feeds` another (images → export: the images live in memory only) also needs that stage to be up to date.
- **An unexpected exception** fails the stage with reason `exception`. The traceback goes to
  `processed/error.txt`, and the next episode is processed. The file is removed when that stage runs again.
- **Status record:** `{status: done|failed|rejected|skipped, reason, reasons, inputs_hash, outputs, details?,
  error_file?, t_start, t_end, version, dataproc_version}` (times in unix seconds).

| Stage | Reads | Writes (in `processed/`) | Fails / rejects | `inputs()` |
|---|---|---|---|---|
| **check** | the whole `raw/`: manifest hashes, CSVs, meta, both MP4s decoded | `check.json` | rejected: every `RawEpisode.validate` error code; `rig_resolution_mismatch` / `rig_fps_mismatch` (meta camera vs rig yaml, per role); `rig_mismatch` (meta `rig_id` ≠ task.yaml `rig`) | rig id, camera resolution/fps, task rig |
| **slam** | ultrawide + IMU (+ `truth.npz`) | `slam/{trajectory.tum, slam.npz, result.json}`, backend work in `slam/work/` | failed: `slam_error`, `oracle_no_truth` | backend name, `cameras.ultrawide`, `imu` |
| **grid** | meta button times | `grid.npy`: `arange(T0, T1, 16_666_667)`, `T0/T1 = rig.imu_time_ns(button_start/stop_ns)` | rejected: `too_short` (< 2 steps) | `imu.time_offset_ns` |
| **pose** | slam trajectory, grid | `pose.npz` (`pos`, `rotvec`, `valid`, `reasons`) | (none; invalid rows are trimmed later) | `tcp`, `world_offset`, ultrawide `T_imu_cam`, max gap 100 ms |
| **width** | every ultrawide frame via `to_virtual`, gripper range, grid | `width.npz` (grid `width`/`valid`, per-frame `frame_has_width`, detections), `width_detections.json` | failed: `no_calibration`, `calibration_failed` | `aruco`, `virtual_ultrawide`, `cameras.ultrawide`, gripper_range.json hash, max gap 200 ms |
| **validate** | grid, pose, width, slam, raw frame/IMU times | `validate.json` (status, reasons, kept span, metrics, thresholds) | rejected: see below | camera/IMU time sections, all thresholds |
| **images** | kept grid rows, both MP4s | `images.npz` (frame indices; images stay in memory) | failed: `main_frame_missing` | camera sections, `virtual_ultrawide`, 10 Hz, 25 ms |
| **export** | images, kept pose/width rows | `episode.zarr` | — | `task.yaml` `label`, `pi_name` |

**Names.** `episode_names` uses the ledger `pi_name` of the slot. If that is missing or empty (a reconciled
row), it falls back to `<meta session_id>_episode_<NNNNNN>`.

### validate (`dataproc/stages/validate.py`)

1. A grid row is usable when its pose, width and main image are all valid. The main image is valid when the
   nearest main frame is within `MAIN_MAX_DT_NS` = 25 ms. The **kept span** is the longest contiguous run of usable
   rows; the earliest wins a tie.
2. These reasons come from the kept span itself:
   - `no_valid_span`: fewer than 2 usable rows;
   - `too_short`: the span lasts < `MIN_KEPT_S` = 2 s;
   - `kept_fraction`: it covers < `MIN_KEPT_FRACTION` = 0.8 of the grid (the button span).
3. These are checked inside the kept span `[t_lo, t_hi]`, and every one that fails is recorded:

| Reason | Rule (constant) |
|---|---|
| `speed` | TCP speed (per-step position delta / dt, as iPhUMI `align_validate`) > `MAX_SPEED_MPS` = 3 m/s |
| `speed_extreme` | the same > `EXTREME_SPEED_MPS` = 7 m/s (also reported separately) |
| `ultrawide_gap` | a gap between ultrawide frames (or a span edge and the nearest frame) > `MAX_ULTRAWIDE_GAP_NS` = 100 ms |
| `imu_gap` | the same for IMU samples > `MAX_IMU_GAP_NS` = 20 ms |
| `slam_tracking` | SLAM-tracked fraction of the ultrawide frames in the span < `MIN_TRACKED_FRACTION` = 0.95 |
| `scale_mismatch`, `scale_unobservable` | the SLAM result's scale-check problems (whole episode) |
| `tag_coverage` | fraction of ultrawide frames in the span with ≥ 1 usable finger tag (a frame the width stage measured; z within tolerance) < `MIN_TAG_FRACTION` = 0.8 |

These constants are part of the validate hash, so changing one re-runs validate and everything after it.

### What re-runs when something changes

| Change | Re-runs |
|---|---|
| nothing | nothing (only the zip and report are rebuilt) |
| `--force` | calibration and every stage of every selected episode |
| raw episode re-delivered (MANIFEST bytes) | everything for that episode |
| `world_offset` or `tcp` in the rig yaml | pose, validate, images, export |
| `task.yaml` `label` | images, export |
| a new or changed calibration | width, validate, images, export |
| `--slam` backend | slam and everything after it |

A data file changed **after** intake without its manifest changing is not detected by the hash, because the hash
covers `MANIFEST.sha256`, not the data files. `--force` re-verifies it (check: `hash_mismatch`).

## `dataproc status` (`TaskStatus.run()`)

Read-only.
1. **`load_task`**: reads `task.yaml` and `episodes.csv`; exit 1 if the folder or `task.yaml` is missing.
2. **`episode_rows`**: one row per `episode_N` with a `raw/`: slot, pi_name, last stage, status
   (`exported` / `rejected` / `failed` / `incomplete` / `pending`), reasons, n_steps.
3. **`calibration_state`**: the newest calibration, `done` / `calibration_failed` / `not_processed` /
   `no_calibration`.
4. **`output_state`**: the zip is `up to date` when all of these hold:
   - it exists;
   - the last `report.json` says it was built (`zip` is not `null`);
   - its own `meta/episode_names` equal the names of the episodes exported now, in order;
   - it is newer than every exported `episode.zarr` (mtime).

   Otherwise it is `stale (run dataproc process)` (for example a zip kept after a failed build, or with nothing
   exportable) or `missing`.

## Timing (laptop, oracle backend, `--fast` sim episode: 5 s, 150 grid steps)

About 6.5 s per episode on a first run: check 0.3 s, slam 0.15 s, width 3.3 s (tag detection on 150 remapped
frames), images 1.9 s, export 1.1 s. A second run with nothing changed takes ~0.3 s for 3 episodes (zip rebuild
only).
