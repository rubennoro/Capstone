# umiproc design

`umiproc` runs UMI's SLAM recipe (one map per session, then every demo localized against it) on the Pi 5 rig's
recordings and turns them into per-task folders of checked episodes, ready to package. It runs on the Jetson.

Decided with the user on 2026-11-16. Nothing here changes without asking.

## Scope

- In: lens calibration (optional), `map`, `calibrate` (gripper), `run`, `status`, the checks, DIRECTIONS.md.
- Out for now: packaging (`umiproc/dp/package` exists as part of the copy but is not wired to a command), camera–IMU
  calibration (OpenICC), IMU Allan variance.

## Self-contained: the dataproc copy

- `umiproc/dp/` is a **copy** of `data-processing/dataproc` at commit `e69b4b1` (2026-11-16), with its tests in
  `tests/dp/` and its rig files in `configs/rigs/`. Nothing in `umi-processing/` imports or reads `data-processing/`.
- The copy was mechanical: `dataproc` imports became `umiproc.dp`, and the copy's repo paths point at
  `umi-processing/configs/rigs`. Format identifiers (`dataproc.raw.v1`, `dataproc.rig.v1`, `dataproc.episode_zarr.v1`,
  …) are unchanged on purpose: they name file formats, and rig files and raw copies contain them.
- The copy is ours: change it where umiproc needs to, and list every change under "Changes to the copy" below. It is
  never synced with `data-processing/` automatically.
- Tests: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests` in an environment that has only
  `pip install -e umi-processing[test]` (no dataproc), so a leftover `dataproc` import fails loudly.

## Where it runs and the data root

- Jetson: recordings in **`/data/incoming/episode_NNNNNN/`** (the Pi's format; only read, never written).
- Output root **`~/data/umi/`** (override with `$UMIPROC_ROOT`; `/data` is refused, as dataproc refuses it):

```
~/data/umi/
  pi_sources.csv            every recording taken, with its kind: map | calibration | demonstration
  rigs/                     the data-root copies the processing reads (rig yaml, SLAM config, optional lens file)
  lens/<pi_name>/           lens-calibrate results (lens.yaml, report.json, coverage.png)
  incoming/                 staging for the Pi-format conversion (as dataproc)
  tasks/TASK/               one folder = one session = one map + one gripper calibration
    task.yaml  map/  calibration/  gripper_range.json  .staging/  episode_N/  failed/<pi_name>/  episodes.csv  report.json
```

## The session (UMI's order)

| Step | Record | Command | Result |
|---|---|---|---|
| 1 | Mapping clip | `umiproc map TASK MAP_DIR` | creates `tasks/TASK` (`TASK_2`, … if taken), builds and checks the map |
| 2 | Gripper calibration clip | `umiproc calibrate TASK_FOLDER CAL_DIR` | `gripper_range.json` in the folder |
| 3 | Demos | `umiproc run TASK_FOLDER START END` | `episode_N`, or `failed/<pi_name>/reason.json` |

- `run` needs the map. Without the gripper calibration it still localizes every demo and runs every pose check, then
  leaves it in `.staging/` as **waiting for gripper calibration** (exit 3); a later `calibrate` + `run` finishes it,
  re-running only the width and export steps.
- The map, calibration and demo recordings are recorded in `pi_sources.csv` by kind, so `run`'s range skips the map and
  calibration recordings by itself.

## SLAM (UMI's ORB-SLAM3 fork)

- Fork `github.com/cheng-chi/ORB_SLAM3` at `b741dca`, Pangolin at the pinned `d484494`, built natively (no Docker) with
  `native/build.sh` on the Jetson and the laptop; our patches in `native/patches/`.
- Front-end `native/umi_slam/dataproc_umi_slam.cc`: reads the EuRoC folder the copied `write_euroc` writes (real
  per-frame times, starting at 0 for the fork's map loading), `--save_map` / `--load_map`, `--mask_img`, real-time
  pacing (ORB-SLAM3 skips keyframes otherwise), input checks, `--max_lost_frames`.
- **Gripper mask, UMI's exact mechanism:** a mask PNG drawn once from our polygons (`configs/slam/rig_pi_v3.yaml`),
  nonzero = masked; the front-end sets those pixels to 0 in every frame it feeds ORB-SLAM3 (`im.setTo(0, mask)`, as
  `gopro_slam.cc`), in the mapping run **and** every localization run. The frames on disk, tag widths and training
  images are never masked. (The copied rig's `cameras.ultrawide.slam_mask` stays unset, so `write_euroc` does not mask
  a second time.)
- **Table tag** `DICT_4X4_50` id 13, 0.16 m (UMI's): the fork's tag-based initialisation fallback is on for mapping
  (`--init_tag_id 13 --init_tag_size 0.16`), and the tag pose in the map is a map check.
- Sensor values: lens = the optional calibrated lens file, else the rig's nominal model (reported as "nominal,
  uncalibrated"); camera→IMU = the rig's CAD `T_imu_cam`; time offset = 0 (the copied `time_offset_check` is reported
  as a diagnostic on every map and episode); IMU noise = datasheet densities (LSM6DSOX Rn 3.8 mdps/√Hz; An 75 µg/√Hz at
  ±4 g, the Pi's range) and the two random walks as documented assumptions.
- Every map and episode records the exact sensor values and SLAM settings it was made with; changing them re-runs
  the map and then every episode.

## Checks

### `map` (UMI has none; a failed map attaches nothing and prints which check failed and what to change)

| # | Check |
|---|---|
| M1 | the mapping run ends with exactly one map that has keyframes |
| M2 | the IMU is fully initialised (both of ORB-SLAM3's inertial refinement stages finished) |
| M3 | the mapping clip, re-localized against the saved map (as UMI's `03` does): relocalized within 0.5 s and ≥ 95 % of frames visually tracked (state OK) |
| M4 | the copied gravity and metric-scale checks pass on that re-localized trajectory; speed ≤ 3 m/s |
| M5 | tag 13: ≥ 20 sightings passing UMI's filters (keyframes only, 0.3–2 m, centre within 0.6·H/2 of the centre of OUR 1280×960 image), position spread ≤ 1 cm, tag normal within 5° of gravity |

### `run`: UMI's rules, converted to the same time at our 30 fps (UMI counts frames at 60 fps)

| UMI at 60 fps | time | ours at 30 fps |
|---|---|---|
| localization stops after 60 lost frames (no trajectory → demo dropped) | 1 s | 30 frames |
| > 10 lost frames in the demo → dropped | 0.17 s | > 5 frames |
| < 60 tracked frames → dropped | 1 s | < 30 frames |
| the demo is **split at its lost frames**; pieces < 24 frames dropped; every other piece is its own episode | 0.4 s | < 12 frames |

"Tracked" means visually tracked (state OK), never an IMU-only prediction. Added to UMI's rules: relocalized within the
first 0.5 s (a later relocalization is where the fork's inertial link was wrong); the copied validate's speed, scale,
gravity and finger-tag-coverage checks on every piece; the gripper calibration is the copied method plus UMI's rule that
both finger tags are seen in ≥ 80 % of the calibration frames.

## Episodes

- Same layout as dataproc: `.staging/<pi_name>/` while processing, then `episode_N/` (from `episode_1`) or
  `failed/<pi_name>/` with `reason.json` (stage, reasons, message, time).
- A recording that splits into k pieces becomes k consecutive `episode_N` folders, each with its own
  `processed/episode.zarr`; the raw copy is stored once and the other pieces point to it.
- `processed/episode.zarr` is the copied export's per-episode file, so the copied packaging can read the folders later.

## Build order and DIRECTIONS.md

| Step | Built | DIRECTIONS.md |
|---|---|---|
| 0 | `lens-board`, `lens-calibrate` | §1 lens calibration (optional, 1 min) |
| 1 | native build + patches, data root and organization, rig + SLAM config, `status` | §2 Jetson setup |
| 2 | `map` and M1–M5 | §3 prove the environment: record a mapping clip, run `map`, read the checks |
| 3 | `run` (UMI's rules, splitting, waiting for gripper calibration) and `calibrate` | §4 prove demos, §5 gripper calibration and finishing |
| 4 | smoke test on existing recordings, final pass | §3–5 adjusted |

## Changes to the copy

Apart from the import rewrite and the repo paths:

- `process.py`, `stages/calibration.py`: the stage-hash key `"dataproc"` is `"umiproc"` (our own hashes; no data made by
  dataproc is ever re-used).
- `tests/dp/`: paths one folder deeper; `test_cli.py` runs `python -m umiproc.dp`; `test_intake.py`'s subprocess code
  imports `umiproc.dp`.
- `configs/rigs/rig_pi_v3.yaml` `imu.noise`: accelerometer noise density 7.36e-4 (datasheet at ±4 g; was 7.0e-4, the
  ±2 g figure), each value labelled datasheet or assumption.
- User-facing messages still name `dataproc` commands (e.g. "dataproc calibrate first"); the steps that wire a module
  into a `umiproc` command change its messages.

Known failures inherited from `e69b4b1` (they fail in `data-processing/` too, after its "remove hard cap on calibration"
change): `tests/dp/calibrate/test_calibrate_task.py::{test_opening_out_of_tolerance_only_warns,
test_closed_width_far_off_warns, test_failed_fit_record_is_not_written_anywhere}`. Step 3 (`umiproc calibrate`)
resolves them.
