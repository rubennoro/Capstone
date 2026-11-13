# Processing: the calibration and the per-episode stages

`dataproc run` (via `dataproc/run/`) processes each recording with the stages below; `dataproc calibrate` fits the
calibration they use. Each exported episode is one `processed/episode.zarr` in the final schema
([DESIGN.md](DESIGN.md), "Output zarr contract"); **no zip is built here**: `dataproc package` combines them
([EXPORT.md](EXPORT.md)). `dataproc status TASK_FOLDER` shows where every episode is. The step names below are the
function, method and class names in the code.

`--slam oracle` uses the synthetic ground truth (`raw/truth.npz`, written by `dataproc sim`). On a recording
without that file, the slam stage fails with `oracle_no_truth` and a message pointing to `--slam orbslam3`.

## Code layout

| Module | Contents |
|---|---|
| `dataproc/process.py` | `load_folder_context`, `process_calibration`, `EpisodeProcessor` (one episode dir); `TaskProcessor` / `run_process` / `cli_process` (every `episode_N` of a folder, `output/report.json`; Python API only, not a command) |
| `dataproc/stages/base.py` | the `Stage` interface, `StageFailed`, `ProcessContext`, `Episode`, `Calibration`, `rig_sections` |
| `dataproc/stages/calibration.py` | `fit_calibration` (pure), `write_calibration_result`, `CalibrationStep` (reuse), `newest_calibration` |
| `dataproc/stages/check.py` | `CheckStage` (+ the pure `check_episode`, `rig_consistency`) |
| `dataproc/stages/geometry.py` | `SlamStage`, `GridStage`, `PoseStage`, `WidthStage` (they call `slam/`, `stages/pose.py`, `stages/width.py`) |
| `dataproc/stages/validate.py` | `ValidateStage` (+ the pure `validate_episode` and every threshold constant) |
| `dataproc/stages/output.py` | `ImagesStage`, `ExportStage` |
| `dataproc/files.py` | process-specific helpers: JSON (atomic), canonical hashing of stage inputs, `status.json` read/write |
| `dataproc/fsutil.py` | generic filesystem plumbing shared with intake: `write_atomic` (dot temp file + fsync + rename), renames, locks |
| `dataproc/report.py` | `episode_summary`, `format_table` (shared by run, process and status) |
| `dataproc/status.py` | `TaskStatus` (the status command) |

A stage is a class with `name`, `version`, `deps`, `inputs(ctx, ep) -> dict` (what goes into its hash) and
`run(ctx, ep) -> dict` (the `outputs` recorded in `status.json`). It raises
`StageFailed("failed" | "rejected", reasons, message)`.

## The folder context (`load_folder_context`)

Reads `tasks/<folder>/task.yaml`, `episodes.csv` and `rigs/<task.yaml rig>.yaml`, and resolves the SLAM backend
(`auto` → `orbslam3` if built, else `none`). Writes nothing. A missing or invalid task folder, `task.yaml`, rig yaml
or ledger stops the command before anything is changed. One `ImagePrep` (remap table) is shared by the whole run.

## The calibration (`fit_calibration`, `CalibrationStep`; version 2)

**`fit_calibration(raw_dir, rig, prep, task_rig)`** reads one calibration recording and writes nothing. Its checks,
in order (the first failure is the result, with its reason code). As in iPhUMI, only the fit can fail a calibration
that is otherwise readable; the gripper values are warnings:
1. the contract check, videos decoded (`invalid_episode`);
2. `rig_consistency`: meta `rig_id` vs the rig (`rig_mismatch`), each camera's resolution and fps vs the rig yaml
   (`rig_resolution_mismatch`, `rig_fps_mismatch`);
3. every ultrawide frame remapped with `ImagePrep.to_virtual`, the finger tags detected;
4. `fit_gripper_range` (`gripper_range_fit`: too few usable both-tag frames);
5. **the gripper warnings** (iPhUMI has none of these; they never fail): a warning when `max_width - min_width`
   (metres, exactly the full opening the width stage exports) is more than `gripper.opening_tolerance_m` from
   `gripper.max_opening_m`, and when `min_width` (fully closed) is that far from the optional
   `gripper.closed_width_m`, which usually means a tag was hidden when the gripper closed. Without a `gripper`
   section in the rig yaml there is one warning saying so.

An unexpected exception is `exception` (traceback kept). Fewer than `MIN_CYCLES` = 5 open/close cycles is a warning
only. The result carries `calibration_diagnostics` (per-tag and both-tag detection counts, z statistics, cycle
count, closed/open tag distance), which `dataproc calibrate` prints with hints.

**The rig yaml `gripper` section** (`config.GripperConfig`, part of the calibration hash):

```yaml
gripper:
  max_opening_m: 0.080          # rig_pi_v3: the Franka Hand's stroke (tag-to-tag x, open - closed)
  opening_tolerance_m: 0.010    # for both values
  closed_width_m: 0.044         # optional: tag-to-tag x fully closed (episode_000056)
```

`rig_v1` (and the `--fast` sim rig) use 0.060 m, the opening the simulator sweeps.

**`write_calibration_result`** saves a fit as `calibration/<name>/processed/{gripper_range.json, status.json}`
(`error.txt` for an exception) and the task copy `tasks/<folder>/gripper_range.json`. `dataproc calibrate` calls it
before the folder is used, with the same inputs hash `CalibrationStep` computes.

**`CalibrationStep`** (`process_calibration`, at the start of every run) picks the newest
`calibration/<name>/raw` (by ledger `intake_unix_ns`; a folder made by `calibrate` has exactly one) and **reuses**
the saved result when its hash (manifest, `aruco`, `virtual_ultrawide`, `cameras.ultrawide`, `gripper`, task rig,
`CALIBRATION_VERSION`, dataproc version) is unchanged and its status is `done` or `failed`; otherwise (a rig edit,
`--force`, an unreadable saved range) it fits again. The task copy always mirrors the current result: it is removed
when the newest calibration failed. Results: `done`, `no_calibration`, `calibration_failed` (with a reason). When
not `done`, every episode's width stage fails with `no_calibration` / `calibration_failed`; `run` refuses a folder
whose calibration is not `done`.

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

**Names.** `episode_names` uses the ledger `pi_name` of the slot (for `run`: the converted name,
`pi-<date>_episode_NNNNNN`). If that is missing or empty (a reconciled row), it falls back to
`<meta session_id>_episode_<NNNNNN>`.

**After the stages** (`dataproc run`, `dataproc/run/episode.py`): export `done` → `.staging/<pi_name>` is renamed to
the next free `episode_N` (from `episode_1`); `failed`/`rejected` at any stage → `failed/<pi_name>/` with
`reason.json` (`stage`, `reasons`, `message`, time); slam failed with `slam_unavailable` (no backend) → it stays in
`.staging/` and waits (exit 3).

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
| nothing | nothing |
| `--force` | calibration and every stage of every selected recording |
| raw episode re-delivered (MANIFEST bytes) | everything for that episode |
| `world_offset` or `tcp` in the rig yaml | pose, validate, images, export |
| `task.yaml` `label` | images, export |
| a changed calibration result (e.g. after a rig edit) | width, validate, images, export |
| `--slam` backend | slam and everything after it |

A data file changed **after** intake without its manifest changing is not detected by the hash, because the hash
covers `MANIFEST.sha256`, not the data files. `--force` re-verifies it (check: `hash_mismatch`).

## `dataproc status TASK_FOLDER` (`TaskStatus.run()`)

Read-only.
1. **`load_task`**: reads `task.yaml` and `episodes.csv`; exit 1 if the folder or `task.yaml` is missing.
2. **`episode_rows`**: one row per `episode_N` with a `raw/`: slot, pi_name, last stage, status
   (`exported` / `rejected` / `failed` / `incomplete` / `pending`), reasons, n_steps.
3. **`unfinished_rows`**: "not episodes (never packaged)": every `.staging/<pi_name>` (`in progress`, or
   `waiting for SLAM`) and every `failed/<pi_name>` (its `reason.json` stage and reasons).
4. **`calibration_state`**: `done` / `calibration_failed` / `not_processed` / `no_calibration`.

## Timing (laptop, oracle backend, `--fast` sim episode: 5 s, 150 grid steps)

About 6.5 s per episode on a first run: check 0.3 s, slam 0.15 s, width 3.3 s (tag detection on 150 remapped
frames), images 1.9 s, export 1.1 s. Re-running a range with nothing new does no processing.
