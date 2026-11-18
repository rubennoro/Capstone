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

### The root (`umiproc/paths.py`)

- `umiproc.paths.data_root()` is the only source of the root: `--root`, else `$UMIPROC_ROOT`, else `~/data/umi`. The
  copied dataproc code asks it too (`umiproc.dp.paths.data_root`, and every `DataRoot` checks its path), so nothing
  umiproc runs reads `$DATAPROC_ROOT` or defaults to `~/data`.
- Refused as the root (also through a symlink or `..`): `/data`, and dataproc's own root, `~/data` and
  `$DATAPROC_ROOT` when it is set (umiproc's `tasks/`, `incoming/`, `rigs/` and `pi_sources.csv` would mix with
  dataproc's). A folder inside them, such as the default `~/data/umi`, is fine, except:
- Refused as the root, and every folder inside them: the Pi's `/data/incoming` (umiproc would write `.run.lock`,
  `incoming/`, `tasks/` and `pi_sources.csv` into the Pi's delivery folder), and dataproc's managed subfolders
  `incoming/`, `tasks/`, `rigs/`, `quarantine/`, `datasets/` of `~/data` and of `$DATAPROC_ROOT` when it is set.

### Full layout

```
~/data/umi/
  pi_sources.csv              one row per recording taken: source_name, source_manifest_sha256, converted_name, task
                              (the folder), kind (map | calibration | demonstration), status, adapted_unix_ns
  .run.lock                   held by every command that takes recordings (one at a time)
  rigs/<rig>.yaml             copy of configs/rigs/<rig>.yaml       (schema dataproc.rig.v1)
  rigs/<rig>.slam.yaml        copy of configs/slam/<rig>.yaml       (schema umiproc.slam.v1)
  rigs/<rig>.lens.yaml        optional: `umiproc lens-calibrate --apply` (schema umiproc.lens.v1); an older one is
                              kept as <rig>.lens.yaml.bak-<UTC time>
  lens/<pi_name>/             lens-calibrate results: report.json, coverage.png, lens.yaml (only when every check passed)
  incoming/<pi_name>/         a recording converted to the copied raw format, while a command checks it; empty between
                              commands (a leftover stops the next command until it is moved away)
  quarantine/<pi_name>/       a converted recording the copied intake refused (validation, duplicate), with reason.json
  maps_failed/<pi_name>/<UTC>/  a failed map's report.json, logs, CSVs, settings, mask, fingerprint, tag detections
                              (no frames; no atlas, except for a map that passed every check but that intake did not
                              take); nothing else records a failed map
  .work/map/<pi_name>/        umiproc map's work folder (EuRoC frames, SLAM runs); removed at the end, cleared by the
                              next map after a crash
  .work/run/                  umiproc run's work folder (the SLAM inputs, one demo's EuRoC frames at a time); removed
                              at the end
  tasks/TASK/                 TASK, TASK_2, ... (task.yaml label stays TASK)
    task.yaml                 label, rig, created
    episodes.csv              the folder's ledger: one row per slot (slot, pi_name, ..., kind), append-only
    map/<pi_name>/raw/        the map recording (exactly one per folder)
    map/<pi_name>/            next to raw/ (umiproc map, only when every check passed): map_atlas.osa,
                              tx_slam_tag.json, mapping/ and relocalize/ (trajectory.csv, .t0_ns, umiproc_slam.log),
                              orbslam3_settings.yaml, slam_mask.png, slam_fingerprint.json, tag_detections.json,
                              report.json (moved in last: a map folder without it is an interrupted map)
    calibration/<pi_name>/raw/  the gripper calibration recording
    gripper_range.json        the gripper calibration's result
    .staging/<pi_name>/       a demonstration being processed, or waiting for the gripper calibration
    episode_N/                passed every check (from episode_1)
    failed/<pi_name>/         failed a check: reason.json (stage, reasons, message, time)
    report.json               the last run's outcome per recording
```

`<pi_name>` is the converted name `pi-YYYY-MM-DD_episode_NNNNNN` (the local date the Pi wrote `COMPLETE`).

### Taking recordings (`umiproc/recordings.py`)

- A command takes a recording in two steps under `.run.lock`: `inspect_recording` (read only: `COMPLETE`, every file
  against `MANIFEST.sha256`, `meta.json` valid, not in `pi_sources.csv` yet), `stage` (convert it into
  `incoming/<pi_name>/` with the copied adapter), then `take(kind)` (the copied intake moves it into its slot and writes
  both ledgers) or `discard` (nothing recorded: it can be retried). So a failed map or calibration attaches nothing.
- By kind: `map` creates the session's folder (`TASK`, else `TASK_2`, ... as the copied calibrate does) with the
  recording in `map/<pi_name>/raw`; `calibration` and `demonstration` go into an existing folder (made by `map`), to
  `calibration/<pi_name>/raw` and `.staging/<pi_name>/raw`.
- `run`'s range reads the kind from `pi_sources.csv` (`kind_of`, `not_a_demonstration`) and skips the map and
  calibration recordings with a reason.
- A recording the copied intake refuses (contract errors, or the same converted name as one already taken) goes to
  `quarantine/` and is recorded as `quarantined`, as in dataproc.

### The rig file, SLAM config and lens file (`umiproc/sensors.py`)

- The processing reads only the data-root copies in `rigs/`. A missing rig file or SLAM config is copied from the repo
  (`ensure_rig_files`; never over an existing copy). An existing copy is compared with the repo by value (`720` equals
  `720.0`; free-text notes ignored): differences are printed and shown by `umiproc status`, and the copy is used (it may
  be edited on purpose). Refused: a rig copy that predates a section the repo has (the copied `check_rig_copy` rule) and
  a rig file with `cameras.ultrawide.slam_mask` (the mask is the SLAM config's only, see "SLAM").
- `lens-calibrate` reads the data-root copies when they exist, else the repo's files; it copies nothing (the first
  `map` does).
- `load_sensors` gives the **effective sensor values**: the rig with the lens file's model in place of the nominal
  ultrawide model when the lens file exists (provenance `calibrated`, its date and source), else the nominal model
  (`nominal, uncalibrated`); the CAD `T_imu_cam`; time offsets 0 (a rig copy with another offset is refused); the IMU
  noise as in the rig file (`imu.vio_inflation` applied for SLAM); the IMU rate 104 Hz (the LSM6DSOX rate the Pi
  sets, `meta.json imu.odr_hz`). Refused: a lens file for another rig, camera, sensor or resolution, and a SLAM config
  whose mask resolution is not the ultrawide's. The effective rig is the one every stage uses (SLAM, tags, images).

### SLAM inputs (`umiproc/slam/inputs.py`)

Every SLAM run (map and localization) gets the same two files, written from the effective sensor values:

- `orbslam3_settings.yaml`, format `File.version "1.0"`: the fork picks its `Settings` parser for it
  (`src/System.cc`), which aborts on a missing key or a wrong type (reals need a `.` or an `e`, `IMU.T_b_c1` must be
  `dt: f`). Every key is checked against `src/Settings.cc` and by running the real binary
  (`tests/test_umiproc_slam_inputs.py`).

  | Key | Value | Source / reason |
  |---|---|---|
  | `Camera.type` | `"KannalaBrandt8"` | the rig's fisheye-equidistant = Kannala-Brandt k1..k4 |
  | `Camera1.fx fy cx cy k1..k4` | effective ultrawide model, native pixels | lens file, else the rig's nominal model |
  | `Camera.width/height` | 1280 / 960 | native size, no scaling (UMI scaled its GoPro to 960x720) |
  | `Camera.fps` | 30 | the ultrawide's rate (patches 0001/0005 use it) |
  | `Camera.RGB` | 0 | frames are grey PNGs read as grey: not used |
  | `IMU.T_b_c1` | the ultrawide's CAD `T_imu_cam` | camera -> IMU body |
  | `IMU.NoiseGyro/NoiseAcc` | 6.63e-5 / 7.36e-4 | datasheet densities x `vio_inflation.noise_density` (1.0) |
  | `IMU.GyroWalk/AccWalk` | 2e-6 / 1e-4 | assumed random walks x `vio_inflation.random_walk` (1.0) |
  | `IMU.Frequency` | 104.0 | the LSM6DSOX rate the Pi sets (only scales the noise per sample) |
  | `IMU.InsertKFsWhenLost` | 1 | the parser's default, which UMI's settings use (they omit the key) |
  | `ORBextractor.*` | SLAM config `orb` (1500, 1.2, 8, 20, 7) | |
  | `System.thFarPoints` | 20.0 | UMI's |
  | `Viewer.*` | UMI's values | required by the parser; no viewer runs |

  No file path goes into it, so its hash does not depend on where the root is.
- `slam_mask.png`: the SLAM config's polygons drawn by `mask_keep` at 1280x960, 8-bit grey, **255 = masked** (the
  front-end sets nonzero pixels to 0 in every frame). Pixel-identical to the mask of the 2026-11-16 experiments.
- **Fingerprint** (`slam_fingerprint`, `fingerprint_hash`): the effective sensor values, the SLAM config (mask
  polygons, table tag, ORB), the mask's pixel hash, the settings file's sha256, the umiproc_slam binary's path and
  sha256, and its build: the sha256 of the build's `BUILD_INFO` (next to the binary; without its `built <time>` line, which every `build.sh` run rewrites), the patch list parsed from that
  `BUILD_INFO` (names and content hashes: what the build applied), and the path and sha256 of the `libORB_SLAM3.so`
  the binary loads (all the patches live in that library). The library is resolved with `ldd` on the binary, as
  `build.sh`'s own check does, and must be the `library` that `BUILD_INFO` records. A missing binary, `BUILD_INFO` or
  library is an error naming `native/build.sh`. Its hash is the sha256 of the canonical JSON. Maps and episodes
  store it; a different hash means they are out of date (re-run the map, then every episode).

## The session (UMI's order)

| Step | Record | Command | Result |
|---|---|---|---|
| 1 | Mapping clip | `umiproc map TASK MAP_DIR` | creates `tasks/TASK` (`TASK_2`, … if taken), builds and checks the map |
| 2 | Gripper calibration clip | `umiproc calibrate TASK_FOLDER CAL_DIR` | `gripper_range.json` in the folder |
| 3 | Demos | `umiproc run TASK_FOLDER START END` | `episode_N`, or `failed/<pi_name>/reason.json` |

- `run` needs the map. Without the gripper calibration it still localizes every demo and runs every pose check, then
  leaves it in `.staging/` as **waiting for gripper calibration** (exit 3); a later `calibrate` + `run` finishes it,
  re-running only the width and export steps.
- `run` refuses (exit 2, nothing changed) a map that is not complete and passed, and a map whose fingerprint differs
  from the one the current rig files, lens file, mask, settings and SLAM build give (out of date: every demo must be
  localized with exactly the map's settings; the message names the differing keys).
- `calibrate` goes into a folder whose map is complete and passed. One calibration per session: a second one replaces
  the first (the newest is used) only while the folder has no `episode_N`; once episodes exist it is refused.
- The map, calibration and demo recordings are recorded in `pi_sources.csv` by kind, so `run`'s range skips the map and
  calibration recordings by itself.

## SLAM (UMI's ORB-SLAM3 fork)

- Fork `github.com/cheng-chi/ORB_SLAM3` at `b741dca`, Pangolin at the pinned `d484494`, built natively (no Docker) with
  `native/build.sh` on the Jetson and the laptop (it builds its own OpenCV 4.5.4 + contrib: JetPack's 4.8 has the
  newer aruco API the fork cannot build against); our patches in `native/patches/`, listed in
  `~/ORB_SLAM3_umi/umiproc_build/BUILD_INFO`:

  | Patch | Fixes |
  |---|---|
  | 0001 | frame period from `Camera.fps` (was hard-coded 1/59.97 s, UMI's GoPro) |
  | 0002 | `SaveTrajectoryCSV` crashed when no map had a keyframe |
  | 0003 | a relocalization against the loaded map gave the new keyframe an IMU link spanning everything since frame 0 |
  | 0004 | the table-tag pose used uninitialised distortion values |
  | 0005 | `mnFramesToResetIMU` is never set for the settings format UMI uses (0 in our runs); set to `Camera.fps`, as ORB-SLAM3's older settings reader does. **A deliberate difference from UMI's behaviour** (user, 2026-11-16): demo 119 localized 20/362 frames without it, 340–362/362 with it |
  | 0006 | `System::MapSummary()`, read only (the map checks) |
  | 0007 | the first keyframe after a relocalization started with zero IMU bias; it now starts from the map's |
  | 0008 | after a relocalization `mnLastKeyFrameId` held a keyframe id where every other place holds a frame id |
  | 0009 | every relocalization shifted the map's keyframe times again (−6.4 → −13.1 → −23.8 s); now only the first does |
  | 0010 | upstream bug: when Tracking started a new map while LocalMapping still had keyframes of the old one queued, their new points went into the new map; its first IMU initialisation (FullInertialBA) then dereferenced a NULL keyframe vertex (the SIGSEGV of map run 128). Points now go to their keyframe's map |
  | 0011 | upstream bug: an IMU initialisation (VIBA 1/2) that ends after Tracking started a new map forced Tracking to OK on the empty new map (SearchByBoW(NULL), SIGSEGV). Now only if the map being initialised is still current |
  | 0012 | `System::GetAtlas()`, read only (the map export `--save_points` for `umiproc slam-3d`; a build without it refuses that option) |

  0007–0009 were checked mechanically on demo recordings only (each removes its symptom; no crash or regression in
  the earlier runs); their effect on pose quality is judged on the first real mapping session (user, 2026-11-16),
  against a 0001–0006 build: `PATCHES_UPTO=0006 native/build.sh` → `~/ORB_SLAM3_umi/umiproc_build_0006/` (its own
  library), selected with `UMIPROC_SLAM_BINARY`. Not patched: a first relocalization late in a clip still adds the
  whole wait to LocalMapping's `mTinit` (as the fork intends).

- Front-end `native/umi_slam/umiproc_slam.cc` (binary `~/ORB_SLAM3_umi/umiproc_build/umiproc_slam`): reads the EuRoC
  folder the copied `write_euroc` writes (real per-frame times, starting at 0 for the fork's map loading),
  `--save_map` / `--load_map`, `--mask_img`, real-time pacing (ORB-SLAM3 skips keyframes otherwise), input checks,
  `--max_lost_frames`; its last line `UMIPROC_SUMMARY {json}` gives every map's keyframes, IMU init and VIBA 1/2,
  and the pacing: per-frame tracking time (median, p95, max in ms), the frames whose tracking took longer than their
  frame period, and the achieved feed rate against the recording's (whether the Jetson keeps up).
  Python side: `umiproc/slam/{runner,log,trajectory_csv}.py`.
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
- **IMU noise A/B on the first real mapping session** (user, 2026-11-16): the datasheet values stay the documented
  default (`imu.vio_inflation` 1.0 / 1.0 in `configs/rigs/rig_pi_v3.yaml`). The first mapping clip is run twice: as
  is, and with `imu.vio_inflation: {noise_density: 10, random_walk: 10}` set in the data-root rig copy
  (`~/data/umi/rigs/rig_pi_v3.yaml`; the settings writer multiplies both densities by `noise_density` and both
  random walks by `random_walk`, and the fingerprint changes with it). The two are compared on checks M2–M4,
  alongside the 0006-vs-0009 patch comparison. No other change to the noise values.
- Every map and episode records the exact sensor values and SLAM settings it was made with; changing them re-runs
  the map and then every episode.

## Checks

### `map` (UMI has none; a failed map attaches nothing and prints which check failed and what to change)

Since 2026-11-17 (branch `rnoro/data-work-auto`, the user's decision): only **M1** and **M5's sightings** block
(`umiproc.map.checks.BLOCKING`); M2, M3, M4 and M5's spread and tilt are warnings (printed, in the report's
`warnings`, the folder is created anyway); the test demos (DIRECTIONS.md section 5) are the gate, as in UMI. Likewise
`lens-calibrate` has no pass/fail: checks (a)–(h) all warn, any fit is written and `--apply` installs it
(`metrics.checks_outside_limits` in the lens file); only no fit at all fails.

Since 2026-11-18 (the user's decision): **`map --first-map`**. When the whole clip's mapping run ends with more than one
map with keyframes, the clip is cut after the first map piece's last tracked frame (pieces from the run's per-frame
states, `checks.map_pieces`; the first piece tracking ≥ 16 frames, so maps dropped right after initialising are
skipped) and the mapping run, re-localization and M1–M5 run again on the cut clip (times.txt shortened, same frames).
The whole clip's run stays in `mapping_full/`; the cut is in `report.json` (`first_map`), not in the SLAM fingerprint
(the fingerprint is the SLAM inputs a demo shares with its map). The cut is found automatically, never a given time.
Without the flag, a multi-map M1 names the first piece and suggests the flag. `map` draws nothing (2026-11-18): its
mapping run(s) store the map export (`--save_points`, when the build has it) and `umiproc slam-video FOLDER` /
`umiproc slam-3d FOLDER` write the MP4 + plot-run figure / the 3-D page into `tasks/<folder>/analysis/` on demand (a
run folder in `maps_failed/` writes into its `analysis/` next to it).

| # | Check |
|---|---|
| M1 | the mapping run ends with exactly one map that has keyframes |
| M2 | the IMU is fully initialised (both of ORB-SLAM3's inertial refinement stages finished) |
| M3 | the mapping clip, re-localized against the saved map (as UMI's `03` does): relocalized within 0.5 s and ≥ 95 % of frames visually tracked (state OK) |
| M4 | the copied gravity and metric-scale checks pass on that re-localized trajectory; the SLAM orientation agrees with gravity throughout (every 2.5 s window's mean specific force in the SLAM world within 3° of the clip's mean, ≥ 4 windows); speed ≤ 3 m/s |
| M5 | tag 13: ≥ 20 sightings passing UMI's filters (keyframes only, 0.3–2 m, centre within 0.6·H/2 of the centre of OUR 1280×960 image), position spread ≤ 1 cm per axis, the sightings' median tag normal within 5° of gravity |

How `map` computes them (`umiproc/map/`; DIRECTIONS.md §4 for the user's side):

- **Runs** (UMI `02_create_map.py`, `03_batch_slam.py` for `map*`): the copied `write_euroc` with the effective rig
  (native 1280x960, no mask in the frames, IMU not decimated); the mapping run `--save_map --mask_img --init_tag_id 13
  --init_tag_size 0.16 --aruco_dict_id` (from the SLAM config's table tag), real time; then the same EuRoC folder
  `--load_map --mask_img --max_lost_frames 30`, real time, without tag options (umiproc_slam then passes tag id -1 to
  the fork). UMI's `03` run keeps `gopro_slam`'s tag defaults (id 13, 0.16 m), and a localization run that loses the
  loaded map can start a new map through `MonocularInitialization`, whose tag fallback would then use it
  (`Tracking.cc:2213-2237`); with -1 that fallback never finds a tag, and `--max_lost_frames 30` stops the run long
  before a new map would matter. The re-localization runs whenever the mapping run saved a map with keyframes, also
  when M1/M2 failed, so a failed map reports every check it can.
- **M1, M2** from the mapping run's `UMIPROC_SUMMARY` (exit 0, `maps_with_keyframes`, the map saved; that map's
  `imu_initialized`, `inertial_ba1`, `inertial_ba2`). A run killed by a signal names it in M1's hint (e.g. `SIGSEGV`,
  exit -11: not a recording problem, read the log). Their hints change when the mapping run reset its map ≥ 5 times
  and at least half of the dropped maps tracked ≤ 15 frames (0.5 s; counted from the mapping CSV's per-frame states,
  else from the log: "Fail to track local map!" right after "New Map created"): most likely the camera model (lens
  calibration) or the camera–IMU geometry, not the recording, and no "record longer" (seen on real clip 128 with the
  nominal lens: 34 resets, 20 of the 34 dropped maps lost within 8 frames). **M3** on the re-localized CSV: the first visually tracked frame
  (state OK, not lost) ≤ 0.5 s after the first frame, and tracked frames / **all** frames of times.txt ≥ 95 % (frames
  never fed after a stop at 30 lost count as lost). **M4**: the visually tracked rows as `T_W_imu = T_W_cam ·
  T_imu_cam⁻¹` through the copied `finalize_trajectory` (gravity alignment, then `gravity_check` and `scale_check`;
  after the alignment the gravity angle is ~0 by construction, so it informs through |g|, as in the copy). The
  orientation sub-check (`orientation_consistency`) tests what that angle cannot: the IMU samples between tracked
  frames ≤ 0.2 s apart, rotated into the SLAM world (`R_W_imu · f`), are averaged per 2.5 s window of the tracked span
  (a window counts when its samples cover ≥ half of it) and compared with their mean over the whole span; the largest
  angle must be ≤ 3° with ≥ 4 windows (else "not enough tracked span"); the max, median and window count are
  reported. A camera–IMU rotation or lens error makes the estimated gravity turn with the rig, a bending map makes it
  drift; a consistent trajectory stays within the hand's mean acceleration over 2.5 s (≤ ~1.4°) plus an
  accelerometer bias seen from different orientations (20 mg ≈ 1.2°). 3° is the copied gravity check's own angle
  limit. The speed is the camera's between consecutive visually tracked frames. **M5** (`umiproc/map/tag.py`, UMI's `cv_util.py`
  detection and `calibrate_slam_tag.py` filters and aggregation): tag 13 detected in every ultrawide frame (default
  detector parameters, subpixel corners, the effective fisheye model, `solvePnP` as UMI's OpenCV 4.7
  `estimatePoseSingleMarkers`), sightings = posed keyframe rows of the re-localized CSV (UMI's `~is_lost &
  is_keyframe`), 0.3–2 m, centre within 288 px of (640, 480); geometric median, strict 0.9 quantile, the kept
  sighting nearest the mean is `tx_slam_tag`. The spread limit applies to each axis' std (ddof 0) of the kept
  positions, as UMI prints it; the 3-D `|std_xyz|` is reported too. The normal judged is the median normal of the kept
  sightings (geometric median of their z axes, renormalised; one sighting's rotation is noisier than its position),
  against the world's up = the re-localized trajectory's mean specific force in the SLAM world (the copied
  `gravity_align` estimate); the chosen `tx_slam_tag` sighting's normal is reported too.
- **Diagnostics** in `report.json` (never judged): the copied `time_offset_check` and `motion_check`, both runs'
  pacing (a warning below 90 % of the recording's frame rate), keyframes, frames with the tag, lens provenance,
  `vio_inflation`, the build (binary, patches, library).
- **Pass / fail**: a pass writes `report.json` (with the folder and slot) into the work folder, takes the recording
  (`take(MAP, task=TASK)`, both ledgers), moves the map next to `raw/` and moves `report.json` in last (`os.replace`).
  An intake error or quarantine at that point keeps the report, logs and atlas in `maps_failed/` (exit 2). A pass
  interrupted after `take()` leaves an INCOMPLETE slot (`status`: record a new mapping clip; the clip is taken), with
  the map still in `.work/map/<pi_name>/` after a hard kill (manual recovery: DIRECTIONS.md §4.8). A failed map records **nothing** (no `pi_sources.csv` row, no folder): the staged copy is removed and the report goes
  to `maps_failed/`, so the same clip can be run again (A/B variants, or after fixing a setting). A recording the copied
  intake would refuse (validation, duplicate) is found before SLAM and taken as `quarantined`, as the copy does.
  Exit codes: 0 pass, 1 a check failed, 2 cannot run on this input.
- **A/B variants**: one data root per variant (`UMIPROC_ROOT=~/data/umi_ab/<variant>`, DIRECTIONS.md Appendix A), so a root
  never holds two variants; every report prints and stores the build (patches) and `vio_inflation`, and `status`
  flags a map whose fingerprint differs from the current files and build as out of date.

### `run`: UMI's rules, converted to the same time at our 30 fps (UMI counts frames at 60 fps)

| UMI at 60 fps | time | ours at 30 fps |
|---|---|---|
| localization stops after 60 lost frames (no trajectory → demo dropped) | 1 s | 30 frames |
| > 10 lost frames in the demo → dropped | 0.17 s | > 5 frames |
| < 60 tracked frames → dropped | 1 s | < 30 frames |
| the demo is **split at its lost frames**; pieces < 24 frames dropped; every other piece is its own episode | 0.4 s | < 12 frames |

"Tracked" means visually tracked (state OK), never an IMU-only prediction. Added to UMI's rules: relocalized within the
first 0.5 s (a later relocalization is where the fork's inertial link was wrong); the copied gravity and scale checks
and M4's orientation check **once per demo** on all its tracked frames; the copied validate's speed and
finger-tag-coverage checks on every piece; the gripper calibration is the copied method plus UMI's rule that both
finger tags are seen in ≥ 80 % of the calibration frames.

How `run` computes them (`umiproc/run/`; DIRECTIONS.md §5 and §7 for the user's side):

- **Localization** (UMI `03_batch_slam.py:113-123`): the copied `write_euroc` with the effective rig (native
  1280x960, unmasked frames, IMU not decimated) into `<root>/.work/run/<pi_name>/euroc` (removed after the run), then
  `umiproc_slam --load_map <map_atlas.osa> --mask_img --max_lost_frames 30`, real time, without tag options. The
  front-end counts the frames `LocalizeMonocular` did not track over the whole run (not in a row), as the fork's
  `gopro_slam.cc:248-258`; unlike `gopro_slam` it still saves the CSV after the stop, but a stopped run is dropped
  whatever it tracked (UMI has no trajectory then: `06_generate_dataset_plan.py:592-596`). A run that crashed, timed
  out or left no readable CSV is `failed` (`slam_error`); a run that ends with more than one map with keyframes is
  rejected (`new_map`: the CSV's world is the biggest map's first keyframe, then not necessarily the loaded map's).
- **UMI's rules** (`umiproc/run/rules.py`, mirroring `06_generate_dataset_plan.py` for one gripper camera): the
  counts are over every frame of times.txt (frames never fed after a stop count as lost); `06:604-608` > 10 lost ->
  ours > 5; `06:610-614` < 60 tracked -> ours < 30; then `06:719-726` with `get_bool_segments` (`06:41-54`): the
  tracked frames are cut at every lost frame and runs shorter than `min_episode_length` (`06:88`, 24) -> ours 12 are
  dropped; `06:729-763`: every remaining run is one episode. UMI's window shift for multi-camera alignment
  (`06:555-568`, one frame) does not apply to one camera. The 0.5 s relocalization rule: the first tracked frame at
  most 0.5 s after the first frame (as M3). Because every frame before it is lost, the > 5 lost rule already asks for
  the demo to find itself within its first 6 frames (0.17 s); the 0.5 s rule only names the cause.
- **Table frame** (UMI `06:110-119, 617-626`): `T_tag_cam = inv(tx_slam_tag) @ T_W_cam` with the map's
  `tx_slam_tag.json` (the localized CSV's world is the map's world, the same one M5 measured the tag in),
  `T_tag_imu = T_tag_cam @ inv(T_imu_cam)`, and the copied pose stage's TCP `T_tag_imu @ T_imu_cam @ T_cam_tcp` on
  the copied 60 Hz grid (the button span), as UMI's `tx_tag_tcp = tx_tag_cam @ tx_cam_tcp`. The copy's
  `T_world_offset` (the rig file's training-world offset, yaw 30° and a shift from rig_v1) is set to the identity in
  code for these calls (`umiproc.run.pieces.table_rig`; the copy's yaml loader refuses a yaw near 0), so every
  episode's `eef_pos` / `eef_rot_axis_angle` really are in the tag frame (`piece.json`: `"world": "table_tag",
  "world_offset": "identity"`). The copied export's refusal of a zero rotation vector cannot hit a real pose: the
  TCP looking down at the table is ~180° from the tag frame (the zero vector would be the gripper looking up with
  its axes on the tag's), and the copied `tcp_poses` already marks rows with |rotvec| ≤ 1e-6 invalid.
- **Pose checks once per demo** (UMI has none; the map's scale is judged by M4): the copied `finalize_trajectory` on
  **all** the demo's visually tracked IMU poses in the SLAM world (gravity-aligned first, as M4, so the tag's small
  tilt does not enter): `gravity_check` (|g| within 0.5 m/s²) and `scale_check` (|s - 1| ≤ 0.1; it interpolates over
  the demo's short losses, ≤ 5 frames = 167 ms, and splits only at gaps > 200 ms). The demo fails on `gravity` or an
  observable `scale_mismatch`; `scale_unobservable` (too short or too little motion: it needs about 1 s of real
  motion) is reported in `localize.json` (`demo_checks`) and the log, never a failure. M4's
  `orientation_consistency` (every 2.5 s window within 3° of the demo's mean) on the same frames, judged when the
  demo has ≥ 4 windows (10 s tracked), else reported as not judged. A failure of any of these fails the demo (every
  piece; `reason.json` names the demo's reason). Judging gravity and scale per piece would drop pieces UMI keeps: the
  scale check cannot measure under ~1.2 s, and |g| over a short piece includes the hand's mean acceleration
  (|Δv|/T). Even on the whole demo a demo of about 1 s (the 30-frame minimum) with brisk motion can reach the 0.5
  m/s² gravity limit (seen on the synthetic clip: up to 0.48 m/s² over 1 s).
- **Pose checks per piece**: the copied validate's TCP speed per 60 Hz step ≤ 3 m/s (`speed_extreme` > 7 m/s: a SLAM
  jump), and grid rows inside the button span; UMI's 12-frame piece length is the rules'.
- **Widths and export per piece** (needs the gripper range): the copied `detect_episode_tags` / `episode_widths` /
  `width_on_grid` on the effective rig, then the copied `validate_episode` on the piece's grid rows (kept span = its
  longest run with pose, width and a main frame within 25 ms; `kept_fraction` >= 0.8 of the piece; speed; ultrawide and
  IMU gaps; `tag_coverage` >= 0.8), **except** its `too_short` (kept span >= 2 s), which UMI's piece length replaces:
  the kept span must still cover >= 12 of the piece's frames (`piece_too_short`). What the kept span leaves out is
  reported, never silent: `trimmed` in `piece.json`, in each piece of `reason.json` / `localize.json`, in the log and
  the outcome line (seconds, grid rows, the piece's frames not covered, and per cause: no finger width, no
  main-camera frame, no pose, usable but outside the longest stretch). UMI instead splits at width gaps
  (`06:669-684` interpolates the widths and warns below 0.9): an open decision (TODO.md). Then the copied images and
  `write_episode_zarr`. The copied `slam_tracking` check always passes inside a piece (every frame of it is tracked).
- **Gripper calibration** (`umiproc/calibrate/gripper.py`): the copied `fit_calibration` with the effective rig
  (`ImagePrep` of the lens model), then UMI's rule (`scripts/calibrate_gripper_range.py:22, 28-35, 41-50, 59-61`):
  each tag's rate = frames it was detected in / all frames, the gripper's rate = the smaller one, >= 0.8. The copied
  fit keeps its own 0.2 floor; the 80 % rule runs on the same detections afterwards. `run` applies the rule again to
  the calibration it uses (the copied calibration step can refit after a rig edit). A failure records nothing.

## Episodes

- Same layout as dataproc: `.staging/<pi_name>/` while processing, then `episode_N/` (from `episode_1`) or
  `failed/<pi_name>/` with `reason.json` (stage, reasons, message, time; umiproc adds the rules' counts and every
  piece's outcome).
- A recording that splits into k pieces becomes k consecutive `episode_N` folders, each with its own
  `processed/episode.zarr`; the raw copy is stored once and the other pieces point to it.
- `processed/episode.zarr` is the copied export's per-episode file, so the copied packaging can read the folders later.

In detail (`umiproc/run/recording.py`):

```
.staging/<pi_name>/processed/
  status.json        the copied stage records (check, slam, grid, pose, width, validate, images, export): the copied
                     status, reason.json and packaging read them as before. slam = localization + UMI's rules,
                     pose = the table frame + the pose checks
  check.json grid.npy    the copied check and grid stages
  localize/          trajectory.csv (+ .t0_ns), umiproc_slam.log, times.txt
  localize.json      map, fingerprint, the run, the rules, demo_checks (gravity, scale, orientation on the whole
                     demo) and the pieces (frames, pose checks, finish with trimmed, slot), orientation;
                     phase: localized (pose checks done: waiting, or ready for the widths) | finished (exported,
                     being moved into episode_N) | failed
  pieces/<k>/        while finishing: episode.zarr, status.json, piece.json
episode_N/           the first exported piece: raw/ (the recording), processed/ with its episode.zarr, status.json,
                     piece.json, localize.json, localize/
episode_N+1/ ...     the other pieces: raw -> ../episode_N/raw (relative symlink), processed/{episode.zarr,
                     status.json, piece.json (raw_of: episode_N)}; its own episodes.csv row (a copy of episode_N's)
```

- A waiting recording is resumed by any later `run` of the folder: when its `localize.json` is `localized` for the
  same map and fingerprint, only the widths, validate, images and export run.
- `episode.zarr`'s name is the recording's `pi_name`, or `<pi_name>_piece<k>` when it was split (the copied packaging
  refuses two episodes with one name). Each episode's status.json has every stage `done`.
- A piece that fails a check is dropped and the others are kept; a recording with no exported piece goes to
  `failed/<pi_name>/` with the stage it got furthest to (`slam`, `pose`, `validate`, `images`).
- An interrupted promotion is finished by the next `run`, without a new localization or export: a recording whose
  `localize.json` says `finished` while it is still in `.staging/` is moved on from where it stopped (the first
  piece's files still in `pieces/<k>/` are moved, then the folder); pieces left in `episode_N/processed/pieces/`,
  a piece slot without its `raw` link or ledger row, are finished at the start of every run. Each step is a rename
  that can be repeated; the other pieces' `piece.json` names their slot before it is made, so a slot left empty by
  a crash is reused (no gap, no second ledger row).
- A piece's `raw` is a link to the first piece's: deleting the first piece's folder alone leaves the others
  dangling. `status` lists them ("pieces without their recording") and packaging skips them with a warning that
  names them (DIRECTIONS.md §7.4).
- One gripper calibration per session: `run` refuses (exit 2) when the calibration would be fitted again (the
  copied `calibration_inputs_hash` changed: the rig file's `aruco`, `gripper` or ultrawide section, the lens file,
  the calibration step's version) while the folder has episodes; without episodes it is fitted again as before.
  `status` shows the calibration `run` uses (newest by the ledger's intake time, the copied `newest_calibration`).
- Everything that can refuse a `run` (folder, map, fingerprint, range, the calibration rule above, a non-empty
  `incoming/`) is checked before the repairs of an interrupted run, so "nothing was changed" holds; the plan is
  made again after the repairs when they changed what it reads.
- `--retry-failed` moves `failed/<pi_name>` back and runs it from scratch (a new localization).

## Build order and DIRECTIONS.md

| Step | Built | DIRECTIONS.md |
|---|---|---|
| 0 | `lens-board`, `lens-calibrate` | §2 lens calibration (optional, 1 min) |
| 1 | native build + patches, data root and organization, rig + SLAM config, settings, mask, fingerprint, `status` | §1 Jetson setup |
| 2 | `map` and M1–M5 | §4 the map: record a mapping clip, run `map`, read the checks |
| 3 | `run` (UMI's rules, splitting, waiting for gripper calibration) and `calibrate` | §5 test demos, §6 gripper calibration, §7 the rest of the demos |
| 4 | smoke test on existing recordings, final pass | §4–7 adjusted |

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
- `paths.py`: the root is `umiproc.paths.data_root()` (`$UMIPROC_ROOT`, default `~/data/umi`); `ENV_ROOT` and
  `DEFAULT_ROOT` are umiproc's; `DataRoot` refuses `/data` and dataproc's root (`umiproc.paths.check_root`).
  `$DATAPROC_ROOT` is no longer read anywhere.
- `intake.py`: a fifth slot kind, `map` (`map/<pi_name>/raw`, one per folder, one episode per intake, like
  `calibration`): `run_intake(..., map_slot=True)`, `TaskTarget.next_slot(..., map_slot)`; reconcile, rollback, the
  quarantine `mode` and the writable checks know it.
- `pi_intake.py`: `intake_adapted(..., map_slot=False)` passes it through (the `kind` written to `pi_sources.csv` was
  already free text).
- `run/plan.py`: a recording recorded as `map` is skipped ("is the map recording (of task folder ...)"), like a
  calibration.
- Messages and help naming `$DATAPROC_ROOT` now name `$UMIPROC_ROOT` (`cli.py`, `calibrate/task.py`, `run/plan.py`,
  docstrings of `pi_adapter.py`, `pi_intake.py`, `package/build.py`).
- `tests/dp/`: `$DATAPROC_ROOT` became `$UMIPROC_ROOT`; `conftest.py` points `$UMIPROC_ROOT` at a temporary folder
  for the whole session and removes `$DATAPROC_ROOT`, so no copied test can reach a real root.

Step 2 (`umiproc map`) changed nothing in the copy (it calls `write_euroc`, `finalize_trajectory`, the intake's
validation and `DuplicateIndex` as they are).

Step 3 (`umiproc calibrate`, `umiproc run`):

- `package/collect.py` (`collect_folder`): an `episode_N` whose `raw` is a dangling link (a piece of a split
  recording whose first piece was deleted) is excluded (`raw link dangling`) and named in a warning, instead of
  being skipped silently like an empty `episode_N/`. (`run` passes `T_world_offset` = identity to the copied
  `tcp_poses` without changing the copy.)

- **Effective rig** in the copied stages that loaded the rig file themselves, so tags, widths and training images use
  the lens file (`umiproc.sensors.effective_rig`: the data-root rig file with the lens file's ultrawide model in
  place, the rig `load_sensors` gives as `SensorValues.rig`, without needing the SLAM config):
  `process.py` `load_folder_context` (default: the effective rig; new `rig=` argument: one already loaded, as `run`
  passes `SensorValues.rig`; a lens file that cannot be used is a `ProcessError`), `calibrate/task.py` `_load_rig`,
  `calibrate/cli.py` `_rig` (the hints). Through the context this reaches `CalibrationStep`, the width, images and
  export functions and the stage hashes (`rig_sections` of the effective rig).
- `run/plan.py`: `check_folder(..., require_calibration=True)` and `plan_run(..., require_calibration=True)`; `umiproc
  run` passes False (its demos wait for the calibration instead of the run refusing).
- `tests/dp/calibrate/test_calibrate_task.py`: the three tests that failed after dataproc's "remove hard cap on
  calibration" (`8eefea2`) were out of date, not the code (the code is that commit's: the opening and closed width
  only warn). `test_opening_out_of_tolerance_only_warns` and `test_closed_width_far_off_warns` re-used `cal_0` under
  another name, but a copy has the same `MANIFEST.sha256` and is refused as already taken (they now use `cal_1`,
  `cal_2`); `test_failed_fit_record_is_not_written_anywhere` kept the old opening failure's `n_in_window > 0`, but the
  failure it now provokes (tags outside the depth window) has `n_in_window == 0` by construction (it now asserts the
  diagnostics are there: frames, both-tag frames, none in the window). The 31 tests of `tests/dp/calibrate/` pass.
