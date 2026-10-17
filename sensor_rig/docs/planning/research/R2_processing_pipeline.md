# R2: iPhUMI processing pipeline, end to end (raw episode folder to zarr to training obs)

Scope: what the pipeline in `iPhUMI/python_package/iphumi/demonstration_processing` reads, computes and writes, so a
Jetson rig (CSI "main" + "ultrawide" cameras, standalone IMU, VIO pose) can write episodes it accepts unchanged.

Path abbreviations:
- `DP/` = `/home/ruben/Capstone/iPhUMI/python_package/iphumi/demonstration_processing/`
- `CM/` = `/home/ruben/Capstone/iPhUMI/python_package/iphumi/common/`
- `APP/` = `/home/ruben/Capstone/iPhUMI/ios_app/iPhUMI/` (the writer side, used as ground truth for field semantics)
- `BP/` = `/home/ruben/Capstone/behavior_prompting/behavior_prompting/train_network/`

Everything here comes from reading the code plus inspecting real episodes in `data/raw/2026-10-01/`. I parsed the MP4
headers by hand because ffprobe/PyAV aren't installed locally. Nothing was modified.

---

## 0. TL;DR for the new rig

To be accepted unchanged, an episode must be one folder holding:

```
<ISO-time-with-dashes>_<5digits>_<session>_<type>_<side>.json            (required)
<same-prefix>_rgb.mp4            main camera, ~60 Hz, every frame logged    (required for demonstrations)
<same-prefix>_ultrawidergb.mp4   ultrawide, ~10 Hz                           (required for both types)
<same-prefix>_depth.raw          256x192 float16 frames                      (optional; only if include_depth)
```
- No underscores in the session name. `type` is `demonstration` or `grippercalibration`. `side` is `left`, `right` or `head`.
- The JSON uses ARKit-convention poses: a row-major 4x4 `world_T_maincam`, camera axes x right, y up, z toward the user.
- All timestamps are UNIX-UTC ISO-8601 strings with ms precision, on one clock.
- `ultrawideRGBTimes` is *slotted* into the main-camera frame grid (`""` where no ultrawide frame was captured).
- `deviceIdentifier` and `gripperID` select a `tcp_pose_in_arkit_frame` entry in `calibration/gripper_calibration.yaml`.
- The ultrawide video has to show the two 16 mm ArUco tags (DICT_4X4_50, ids 0/1) at ~0.098 m depth. Detection treats it
  as an undistorted pinhole image with the K in `ultrawideCameraIntrinsics`.

---

## 1. Entry points, configs, and how this repo invokes them

### 1.1 Repo invocation (`scripts/cluster/process_dataset.sbatch`)
Run from `DP/` in the `iphumi` conda env (`process_dataset.sbatch:24,32-33`):
1. `python process_demos_iphone.py filters.session_name="[S1,S2]" group.iphone_dir=$REPO/$RAW_DIR stages="[group]"` (`:36`)
2. `python scripts/label_whole_demo_as_task.py --demonstrations-dir tmp_demonstrations --session-name S --task-name TASK`
   for each session (`:39-42`). It edits the *grouped copies* `tmp_demonstrations/*/*/{left,right,head}.json`. When
   `type=="demonstration"`, `sessionName` matches and `labelType=="None"`, it sets `labelType="Predefined"`,
   `taskNames=[TASK]`, `taskStartTimestamps=[first non-empty rgbTimes]` and `taskEndTimestamps=[last]`
   (`scripts/label_whole_demo_as_task.py:23-39`).
3. `python process_demos_iphone.py filters.session_name=... num_workers=$NW skip_stages="[group,visualize]"` (`:45`).
   This runs group_validate, detect, calibrate, align, align_validate and label.
4. `python create_session_iphone.py input_session_filters="[...]" output_session_name=$DATASET sides=$SIDES`
   (SIDES defaults to `single`, `:18,48`).
5. `python build_umi_dataset_iphone.py session_dir=tmp_sessions/$DATASET replay_buffer.num_workers=$NW` (`:49`).
6. Move `tmp_sessions/$DATASET/replay_buffer_$DATASET.zarr.zip` to `~/CapstoneProject/datasets/<task>/<dataset>/` (`:53`).

Per AGENTS.md this sbatch has never run end to end. `firstreal-single` was built by hand with the same commands.

### 1.2 `process_demos_iphone.py` (Hydra, `config/process_demos_iphone.yaml`)
- Stage list `['group','group_validate','detect','calibrate','label','align','align_validate','visualize']`
  (`process_demos_iphone.py:17`). The **execution order is fixed by the code**: group, group_validate, detect, calibrate,
  align, align_validate, label, visualize (`:29-60`). Label must run after align because it needs `aligned_frame_times`.
- Optional overrides: `process_custom=<name>` loads `config/process_custom/<name>.yaml` (`process_demos_iphone.yaml:5`).
  Examples are in `behavior_prompting/iphumi_configs/process_custom/` (bimanual align_validate thresholds only).
- Key config values (`process_demos_iphone.yaml`):
  - `demonstrations_dir: tmp_demonstrations` (`:9`)
  - `group.iphone_dir` (`:30`), `group.gripper_calibration: calibration/gripper_calibration.yaml` (`:34`)
  - `detect.aruco_config: calibration/aruco_config.yaml` (`:41`)
  - `calibrate.tag_det_threshold: 0.2` (`:46`)
  - `align.fps: 60` (`:64`), `align_validate.*` (`:66-89`, listed in §5)
  - `filters.max_position_delta: null` (`:23`)

### 1.3 `create_session_iphone.py` (`config/create_session_iphone.yaml`)
`sessions_dir: tmp_sessions`, `include_invalid: false`, `sides: ''` (auto), `strict_side_matching: true`.

### 1.4 `build_umi_dataset_iphone.py` (`config/build_umi_dataset_iphone.yaml`)
- `stages: [dataset_plan, replay_buffer]`, `output_format: zarr`.
- `replay_buffer.include_depth: false`, `pad_depth_to_square: false`, `upsample_ultrawide_to_full_frequency: false`,
  `include_demo_start_end_pose: false`, `num_workers: -1`, `max_episodes: null`.
- Hidden keys read via `cfg.get` with defaults:
  - `out_res_rgb` (224), `out_res_pointmap` (512), `out_res_depth`
  - `save_pointmap` (false), `save_pointmap_side` ("head"), `save_look_at_point`
  - `iphone_calibration` (only needed for pointmaps)
  - `episode_keep_ratio`, `episode_keep_ratio_name_filter`
  - (`gen_replay_buffer_zarr.py:179-184,36-58,384-398,425`)

### 1.5 Calibration files
- `DP/calibration/aruco_config.yaml`: `DICT_4X4_50`. Marker size is 0.016 m by default and for ids 0, 1, 6, 7. Every id
  in the dictionary gets the default size (`DP/utils/cv_util.py:22-31`).
- `DP/calibration/gripper_calibration.yaml`: keyed **`gripperID` → `deviceIdentifier`**, e.g. `default` → `"iPhone16,1"`
  (15 Pro) and `"iPhone17,1"` (16 Pro, copied from the 15 Pro).
  - Each entry holds `gripper_transform.tcp_pose_in_arkit_frame` (4x4) and `gripper_transform.artag_z_distance_from_ultrawide`
    (0.098079 m).
  - There are no camera intrinsics in it; intrinsics come from the episode JSON (`gripper_calibration.yaml:1-5`).
- `DP/calibration/compute_iphone15pro_calibration.py` shows where the 4x4 comes from: `R = Rx(15°) @ diag(1,-1,-1)`,
  `t = [0, -0.06441, -0.211397]` in ARKit main-camera coordinates (`:3,13-27`).

---

## 2. Raw episode format (what the iPhone writes)

Example `data/raw/2026-10-01/2026-10-01T21-13-57.489Z_75494_firstreal_demonstration_left/`:

| File | Content (measured) |
|---|---|
| `<prefix>.json` | metadata + per-frame arrays (keys below) |
| `<prefix>_rgb.mp4` | H.264, 1920x1440, 996 samples, timescale 600, ~60 fps (VFR timestamps from ARKit) |
| `<prefix>_ultrawidergb.mp4` | H.264, 640x480, 166 samples for 996 main frames, i.e. ~10 Hz |
| `<prefix>_depth.raw` | 97,910,784 B = 996 × 256×192 × 2 B (float16, metres) |
| `<prefix>_depthpreview.mp4` | 256x192 preview. **Never read by the pipeline** |

Gripper calibration episode (`..._firstreal_grippercalibration_left/`): only `.json` + `_ultrawidergb.mp4` (265 frames out
of 1589 slots). Its `hasRGB`/`hasDepth` are false, and `rgbTimes`/`poseTimes`/`poseTransforms`/`depthTimes` are empty.

JSON keys in a demonstration (`APP/DataCollection/DemonstrationData.swift:118-122,240-262,271-350`):

| Key | Example / semantics | Used by pipeline? |
|---|---|---|
| `rgbTimes` [N] | `"2026-10-01T21:13:57.449Z"`, one per main frame = ARFrame time converted to UNIX (`ViewController.swift:1810-1812,2083-2090`) | **required** (align, gripper width, label, label_whole_demo) |
| `poseTimes` [N] | identical strings to `rgbTimes` (same `poseTime`, `DemonstrationData.swift:320,346`) | **required** (align) |
| `poseTransforms` [N][4][4] | **row-major** `frame.camera.transform` = world_T_maincam, ARKit convention; Swift encodes rows explicitly (`DemonstrationData.swift:16-46`). Example last row `[0,0,0,0.99999994]` | **required** (align, filter) |
| `ultrawideRGBTimes` [N] | same length as rgbTimes; holds the ARFrame time if that ARFrame carried an ultrawide image, else `""` (`DemonstrationData.swift:323-329`). Observed pattern: non-empty at idx 1,7,13,... (every 6th) | **required** (detect, align) |
| `depthTimes` [N] | same strings as rgbTimes | optional (align falls back to `rgbTimes`, `align.py:191`) |
| `depthFrameIsRepeat` [N] | bool | not used |
| `mainCameraIntrinsics` 3x3 | `[[1278.45,0,954.60],[0,1278.45,723.53],[0,0,1]]` (row-major) | copied to `{side}_calibration.yaml`; not used downstream (pointmaps disabled) |
| `ultrawideCameraIntrinsics` 3x3 | `[[246.65,0,320.16],[0,246.65,239.54],[0,0,1]]` | **required in practice** (detect solvePnP) |
| `deviceIdentifier` | `"iPhone17,1"` (default `"iPhone16,1"` if absent, `generic_util.py:94`) | **required** (calibration lookup) |
| `gripperID` | `"default"` | **required** for left/right (calibration lookup) |
| `gripperCalibrationRunName` | full name of the cal episode incl. side suffix, e.g. `2026-10-01T21-10-44.057Z_15647_firstreal_grippercalibration_left`; `""` for cal episodes | **required** for gripper width (left/right) |
| `sidesPresent` | `["left"]` | optional (group_validate completeness) |
| `sessionName` | `"firstreal"` | display + `label_whole_demo_as_task` + legacy filter |
| `type` | `"demonstration"` / `"gripper_calibration"` | only `label_whole_demo_as_task.py:30`; the pipeline uses the folder-name suffix instead |
| `labelType`, `isVoiceHost`, `taskNames`, `taskStartTimestamps`, `taskEndTimestamps` | label inputs | label stage (`label.py:54-130`); missing `labelType` means no tasks |
| `isErrorCorrection` | bool | create_session stats, zarr meta |
| `note` | optional | display only |
| `deviceName`, `frameCount`, `hasRGB`, `hasDepth`, `hasUltrawideRGB`, `hasAudio`, `recordingStartTime`, `formatVersion`, `side`, narration* | — | **not used** by Python |

App-side checks before anything reaches Python: a recording is aborted if fps < 31 after the first second, or if the pose
moves faster than 7 m/s (`ViewController.swift:306,1818-1843`). Python never sees dropped episodes.

---

## 3. Stage by stage

### 3.1 `group` (`DP/process_stages/group.py`)
- **Reads:** `glob(iphone_dir/**/*.json)` recursively (`:24`).
- **Filename parsing:** `name.split('_')` must give exactly **5** parts:
  `time, randomizer, session, recording_type, side`. Anything else is skipped with "unknown or old name format"
  (`:36-42`). So **session names cannot contain `_`**.
- **Filters** via `keep_demonstration` (`generic_util.py:334-388`): `demonstration_names`, `demonstration_regex`,
  `task_names` (reads JSON `taskNames`), `session_name` (from the title split), `max_position_delta` (reads
  `poseTimes`/`poseTransforms`).
- **Output dir:** `tmp_demonstrations/<YYYY-MM-DD>/<time>_<rand>_<session>_<type>/`. The date is parsed from the text before
  `T`, and `_left`/`_right`/`_head` are stripped (`generic_util.py:248-259`). Left, right and head files of one recording
  land in the same folder.
- **Copies** every file matching `<dir>/<title>*` and strips `<title>_` from the name. The result is `left.json`,
  `left_rgb.mp4`, `left_ultrawidergb.mp4`, `left_depth.raw`, `left_depthpreview.mp4` (`:68-76`), with symlinks if
  `group.symlink`.
- **Writes `{side}_calibration.yaml`** (`:79-83`, `generic_util.py:91-120`):
  - Copies `calibration_dict[gripperID][deviceIdentifier]` (asserts both keys exist when side != head and gripperID is
    non-empty).
  - Adds `camera_calibration.main.intrinsics` and `camera_calibration.ultrawide.intrinsics` from the JSON.
  - **Fails** (RuntimeError) if the gripperID or deviceIdentifier is missing from the yaml.

### 3.2 `group_validate` (`DP/process_stages/validate.py`)
- Sides present = which of `left.json`, `right.json`, `head.json` exist (`generic_util.py:212-218`).
- For demonstrations it reads `sidesPresent` and **deletes the demo folder** if a listed side is missing (`:158-181`).
  Without `sidesPresent` the check is skipped.

### 3.3 `detect` (`DP/process_stages/detect.py`, `DP/scripts/detect_aruco_iphone.py`)
- Runs on both demonstrations and grippercalibrations, for every side (head included).
- **Reads** `{side}_ultrawidergb.mp4`, plus `ultrawideCameraIntrinsics` from the JSON (falling back to
  `{side}_calibration.yaml`) and `ultrawideRGBTimes` (`detect.py:47,60-69`).
- **time_offset** = `(index of first non-empty ultrawideRGBTimes) * 1/60` (`detect.py:65-69`). This hardcodes **60 Hz main
  cadence** and the slotted layout. An IndexError is raised if every entry is empty.
- **Detection:** PyAV decodes every frame as `rgb24` (`detect_aruco_iphone.py:38-46`), then runs
  `cv2.aruco.ArucoDetector(DICT_4X4_50, CORNER_REFINE_SUBPIX)` (`:78-82`).
  - Each known id is posed with `cv2.solvePnP` on a square of side `marker_size` (0.016 m) using K = ultrawide intrinsics
    and **dist_coeffs = zeros** (`:95-109`).
  - So the ultrawide image is assumed to be an **undistorted pinhole** image at the resolution K refers to.
- **Per-frame time** = `frame.pts * time_base + time_offset` (`:47,57`), in seconds relative to the first main-camera
  frame. This relies on the UW mp4 PTS starting at about 0 at the first UW frame (the app starts the writer at the first
  UW frame's time, `DemonstrationData.swift:285-291`).
- **Writes** `{side}_tag_detection.json`: a list of `{frame_idx, time, tag_dict:{id:{rvec,tvec,corners}}}`.

### 3.4 `calibrate` (`DP/process_stages/calibrate.py`), grippercalibration episodes only
- Skips head, and any side without `gripper_transform` (`:26`).
- **Gripper id:** tag pairs are (2k, 2k+1); left finger = even, right finger = odd. Score =
  min(detection fraction of left tag, of right tag), and the best gripper id wins. It **asserts** score ≥
  `tag_det_threshold` = 0.2 (`:45-79`).
- **Outlier gate:** use a tag only if `tvec.z` is within `artag_z_distance_from_ultrawide ± 0.08` m, i.e. 0.018 to 0.178 m
  (`:85-107`).
- **Width sample** = `right_tvec.x − left_tvec.x` (metres along the ultrawide x axis), over frames where both tags are seen.
  Needs ≥3 such frames (`:109-129`). Records `min_width` and `max_width` (raw extrema, `:132-133`).
- **Linear maps** left_x→right_x and right_x→left_x, fitted with a 3·MAD outlier refit. They predict the missing finger
  when only one tag is visible (`:135-151`).
- **Writes** `{side}_gripper_range.json`: `{gripper_id, left_finger_tag_id, right_finger_tag_id, max_width, min_width,
  x_basis:'tvec0', left_to_right:{a,b}, right_to_left:{a,b}, fit_stats}`.

### 3.5 `align` (`DP/process_stages/align.py`), demonstrations only
- **Reads** `{side}.json`: `poseTimes`, `poseTransforms` (counts must match, else ValueError `:157-160`), `rgbTimes`,
  `depthTimes` (fallback rgbTimes) and `ultrawideRGBTimes`.
- **Common time grid:** `start = max(first poseTime over sides)`, `end = min(last poseTime over sides)`,
  `sample_times = arange(start, end, 1/fps)` with fps = 60 (`:166-179`). Empty overlap means the demo is skipped (printed,
  not fatal).
- **Pose resampling:** 4x4 → [pos, rotvec] → `scipy interp1d` **per component, linear** (rotvec components are interpolated
  independently, not slerp), then back to 4x4 (`:70-76`). Outside the range it holds the end values.
- **Frame indices:** for each grid time, the **most recent past frame** = `searchsorted(ts, t, 'right') − 1`, clipped
  (`:79-84`). This applies to rgb, depth, and ultrawide (with `""` entries filtered out, so the index counts UW video
  frames). Images are never interpolated, so they can be up to one frame period stale.
- **Writes:**
  - `metadata.yaml: aligned_frame_times` (ISO strings, `:184-186`).
  - `{side}_aligned.csv` with columns `relative_aligned_frame_idx, relative_aligned_timestamp, x,y,z, q_x,q_y,q_z,q_w
    (ARKit world_T_maincam), rgb_frame_idx, depth_frame_idx, ultrawide_frame_idx` (`:87-126`).
  - Checks CSV length == metadata length and that timestamps are monotonic (`generic_util.py:175-205`).

### 3.6 `align_validate` (`DP/process_stages/align_validate.py`)
Writes `valid.txt` or `invalid.txt` (listing the reasons) in each demo. Thresholds are in §5.3.

### 3.7 `label` (`DP/process_stages/label.py`)
- **Picks the side:** the voice host if exactly one side is the host, else head > left > right among sides with
  labelType != 'None' (`:68-83`).
- For `Narration`/`Predefined`, task start/end ISO times map to aligned indices via `searchsorted(...,'right')−1`. It
  **asserts** `end > start` (`:105-132`).
- **Writes** `labels_aligned.json` = `{label_source_side, tasks:[{name, relative_aligned_start_frame_idx,
  relative_aligned_end_frame_idx, labels:{}}]}` (`:194-214`).
- `task_extraction.visualize: true` re-encodes `{side}_rgb.mp4` into `{side}_subtasks.mp4` with overlay text (`:98-100,135-191`).
  Not skipped by the sbatch, and costs CPU.
- Note: `label_whole_demo_as_task` sets the task end to the last rgbTime. That sits after the aligned end, so it clamps
  to `aligned_frame_count − 1`.

### 3.8 `visualize` (skipped by this repo)
Hardcodes depth as 256x192 float16 (`visualize.py:169`) and builds 4:3 tiles. Not needed for the dataset.

### 3.9 `create_session_iphone.py`
- For each demo it needs `valid.txt` or `invalid.txt`. If neither exists it deletes the session and raises (`:118-123`).
  `manual_valid.txt`/`manual_invalid.txt` override.
- **Side handling:** with `sides=single`, demos with exactly one of left/right qualify (`:33-34`). Without `sides`, all demos
  must share the same side set (`:279-298`).
- Symlinks demos plus their referenced grippercalibration folders (via `gripperCalibrationRunName`) into
  `tmp_sessions/<name>/demos/` (`:150-157,312-313`).
- Writes `session_info.json` with `sides` (the active mode) and counts.

### 3.10 `build_umi_dataset_iphone.py` → `gen_dataset_plan.py` → `gen_replay_buffer_zarr.py`
- **Plan** (`gen_dataset_plan.py`), per demo and side (after the session-mode filter):
  - `arkit_poses = read_aligned_csv(...)`
  - `tcp_pose = mat_to_pose(iphone_to_tcp_poses(...))`: 6D [pos, rotvec] (`:185-194`)
  - `gripper_width = get_demo_gripper_width(..., aligned_video_times=aligned_times)` (`:197-206`)
  - Video paths (`{side}_rgb_masked.mp4` is preferred if present) and the three index arrays (`:219-233`)
- **Canonical side:** in `single` mode, left and right are both renamed **`right`** (`generic_util.py:21-34`,
  `gen_dataset_plan.py:177`). That is why `firstreal` (recorded `left`) trains as `camera_right_*` / `gripper_right_*`.
- **Silent failure:** if gripper-width extraction raises AssertionError, IndexError, ValueError or FileNotFoundError, it
  prints "Skipping gripper width" and stores `None` (`gen_dataset_plan.py:204-206`). The zarr then has **no**
  `gripper_<side>_gripper_width` key, so training builds a 9-D action instead of 10-D and fails later.
- **Replay buffer:** see §6.

---

## 4. Gripper width, intrinsics and calibration in detail

- **Camera:** gripper width always comes from the **ultrawide** stream (640x480 at about 10 Hz), never from main
  (`detect.py:47`).
- **Tags:** ArUco DICT_4X4_50, 16 mm.
  - Finger pair for gripper k: left = 2k, right = 2k+1. Default gripper is 0, so ids 0 and 1. The yaml also sizes ids 6
    and 7 (bimanual/other grippers).
  - Nominal tag depth 0.098 m from the ultrawide center, accepted within ±0.08 m (hardcoded in `calibrate.py:87` and
    `gripper_util.py:308`).
- **Intrinsics:** taken from **the episode JSON** (`ultrawideCameraIntrinsics`, captured from ARKit's private
  `_ultraWideCamera.intrinsics`, `ViewController.swift:1771-1773,116-128`).
  - **Distortion is assumed zero** (`detect_aruco_iphone.py:105`). Nothing in the code is hardcoded per iPhone; the K
    simply has to match the encoded UW frames.
  - It's unknown whether ARKit's UW buffer is already distortion-corrected (see open questions).
- **Calibration episode role:** it provides `min_width` (closed) and `max_width` (open) in raw tvec-x units, plus
  single-tag prediction maps. Demos find it through `gripperCalibrationRunName` → `tmp_demonstrations/<date>/<title>/`,
  falling back to the sibling folder (`generic_util.py:224-240`).
  - The file used is `{side}_gripper_range.json` of **the same side name**, so a left demo needs a left calibration.
- **Per-demo width** (`gripper_util.py:194-301`):
  - For each UW detection: both tags give `right_x − left_x`; one tag gives a prediction via the linear map. The value
    is clipped to [min, max].
  - It is then mapped by `interp1d([min,max] → [0, max−min])`, so **0 = fully closed** and units are approximately metres
    of tag-center x separation (not true finger width).
  - Linear interpolation over detection times, holding the first/last value outside them.
  - Sampled at `aligned_times − aligned_times[0]` (seconds).
- **Per-device calibration keying:** the yaml key is `gripperID` then `deviceIdentifier` (Apple model string such as
  `"iPhone17,1"`), not `deviceName`. A new rig only needs a new key (e.g. `deviceIdentifier: "jetson-rig-v1"`) with its
  own `tcp_pose_in_arkit_frame` and `artag_z_distance_from_ultrawide`.

**Latent time-base bug (matters for multi-device, harmless for single):** tag times are relative to *that side's first
rgb frame* (via `time_offset`), while widths are sampled at times relative to `aligned_times[0]` = max over sides of the
first poseTime. For a single side on the iPhone, `poseTimes[0] == rgbTimes[0]`, so they agree. **A new rig must keep
`poseTimes[0] == rgbTimes[0]`**, or gripper width will be shifted by the difference.

---

## 5. Poses, frames, time alignment and validation

### 5.1 Pose chain
- **Raw:** `W_T_I` = ARKit `frame.camera.transform`, the pose of the *main (wide) camera* in the ARKit world.
  - World is gravity-aligned (default `ARWorldTrackingConfiguration`, y up), origin at session start
    (`ViewController.swift:458`).
  - Camera axes in landscape (charging port right): **x right, y up, z toward the user** (OpenGL-style; comment in
    `compute_iphone15pro_calibration.py:5`).
- **TCP:** `W_T_TCP = W_T_I @ I_T_TCP` (`gripper_util.py:360-395`), with `I_T_TCP` from
  `{side}_calibration.yaml: gripper_transform.tcp_pose_in_arkit_frame`:
  ```
  [[1, 0,          0,          0        ],
   [0, -0.9659258, 0.25881904, -0.06441 ],
   [0, -0.25881904,-0.9659258, -0.211397],
   [0, 0, 0, 1]]
  ```
  - That is a 180° flip about x (ARKit to "z forward, x right, y down") composed with the 15° mount tilt; the TCP sits
    6.4 cm below and 21.1 cm in front of the main camera.
  - **TCP convention:** z points away from the user along the fingers, x right, y down (`compute_iphone15pro_calibration.py:6`).
  - Head side: rotation `diag(1,-1,-1)`, no translation (`gripper_util.py:385-392`).
- **Stored:** `gripper_<side>_eef_pos` = TCP translation in the ARKit world, and `gripper_<side>_eef_rot_axis_angle` = TCP
  rotvec (`mat_to_pose`, `CM/pose_util.py:32`). Both are **absolute** in that world.
- **Training makes them relative:** `pose_repr: relative` for both obs and action, expressed relative to the latest obs
  pose (`BP/dataset/umi_task_dataset.py:418-449`), with rot converted to 6D. For single-arm, the absolute world frame and
  its gravity alignment therefore drop out. What matters is the **TCP frame definition** and pose accuracy.
  `*_wrt_start` is computed on the fly from the first frame of the task or episode (`:369-410`).
- **Multi-device world:** ARKit collaboration (`isCollaborationEnabled = true`, `ViewController.swift:539`) gives all
  phones one shared world. Python *assumes* all sides' poses are already in one frame on one wall clock. Clocks come from
  per-device `Date() − uptime` (`ViewController.swift:684-689`), with no explicit sync. Not relevant for a single Jetson
  rig.

### 5.2 Synchronization summary
- **Master grid:** 60 Hz over the pose-time overlap (`align.fps`).
- **Pose:** linearly interpolated (component-wise rotvec).
- **Main RGB / depth / ultrawide:** the latest frame at or before the grid time (sample-and-hold). Ultrawide stays at its
  native ~10 Hz in the zarr, with index maps (§6).
- **Gripper width:** linear interpolation of the ~10 Hz detections onto the grid.
- **No latency compensation anywhere.** `CM/latency_util.py` exists but is unused in `demonstration_processing`. All
  modalities on the iPhone share the ARFrame timestamp, so camera-to-pose latency is implicitly zero. A VIO stack with
  its own latency must timestamp poses at **exposure time** on the camera clock.
- **Dropped frames:** not explicitly handled. Python assumes video frame *i* corresponds to `rgbTimes[i]` (decode order),
  and UW frame *j* to the *j*-th non-empty `ultrawideRGBTimes`. If the encoder drops a frame, every later index shifts
  silently. The app's fps < 31 abort is the only guard.
- **Timestamp precision:** ms (ISO strings), so about 6% of a 60 Hz period.

### 5.3 Validation thresholds (`process_demos_iphone.yaml:66-89`, `align_validate.py`)

| Check | Default | Applies to single-arm? |
|---|---|---|
| `max_pose_delta_m_per_s` (per aligned step, i.e. 0.05 m/frame at 60 fps) | 3.0 m/s | **yes** |
| `max_initial_gripper_to_gripper_distance` | 1.50 m | bimanual only |
| `max_initial_gripper_to_gripper_{x,y,z}_distance` (in left ARKit camera frame) | 1.00 / 0.50 / 0.50 m | bimanual |
| `max_initial_gripper_angle_apart` | 150° | bimanual |
| `detect_flipped_grippers` (right must have +x in the left *ARKit camera* frame) | true | bimanual |
| `max_initial_gripper_to_head_distance` | 2.00 m | with head |
| trajectory `left_right_*`, `left_head_*`, `right_head_*` min/max conditions | null | optional |
| `filters.max_position_delta` (raw consecutive-frame jump, applied at group/iterate time) | null | optional |
| calibrate `tag_det_threshold` | 0.2 | cal episodes |
| tag z window | nominal ±0.08 m | width |
| app-side abort: fps < 31, pose speed > 7 m/s | — | iPhone app only |

Note: the bimanual checks run on **ARKit camera poses from the CSV, not TCP poses**, so "x" means camera x.

---

## 6. Final zarr layout (`gen_replay_buffer_zarr.py`, `CM/replay_buffer.py`)

For `firstreal-single` (sides=single, so canonical side `right`, no depth). N = 37,971 aligned 60 Hz frames over 45 episodes:

**data/**

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `gripper_right_eef_pos` | (N,3) | float32 | TCP position, ARKit world (`:259`) |
| `gripper_right_eef_rot_axis_angle` | (N,3) | float32 | TCP rotvec (`:260`) |
| `gripper_right_gripper_width` | (N,1) | float32 | absent if width extraction failed (`:269-273`) |
| `camera_right_main_rgb` | (N,224,224,3) | uint8 | JPEG-XL level 99, chunk (1,224,224,3) (`:430,446-452`) |
| `camera_right_ultrawide_rgb` | (N_uw,224,224,3) | uint8 | native ~10 Hz, N_uw ≈ N/6 (`:453-460`) |
| `camera_right_depth` (opt) | (N,192,256,1) | float16 | only with `include_depth` (Blosc zstd) |
| `gripper_right_demo_{start,end}_pose` (opt) | (N,6) | float32 | `include_demo_start_end_pose` |
| `camera_<side>_pointmap` / `_lookatpoint` (opt) | (N,512,512,3) / (N,3) | float16 | needs `iphone_calibration` yaml with `depth.intrinsicMatrix`; off |

**meta/**
- `episode_ends` (E,) int64, `episode_names` (E,) object, `is_episode_error_correction`
- `task_names`, `task_lengths`, `task_data_ends`, `task_labels_ends`, `is_task_error_correction` (`replay_buffer.py:828-833`)
- `upsample_index_camera_right_ultrawide_rgb` (N,): 60 Hz index to UW buffer index
- `episode_ends_camera_right_ultrawide_rgb` (E,)
- `downsample_index_camera_right_ultrawide_rgb` (N_uw,): UW index to the first 60 Hz index using it (`:359-379`)
- Per episode, UW frames 0..max-used are stored, so any UW frames before the aligned start are included.

`labels/` is empty (labels {}).

**Image transform** (`CM/cv_util.py:5-64`):
- Decode with PyAV `to_ndarray('rgb24')`, so storage is **RGB**. `bgr_to_rgb=False`.
- Pad with black to a square of side max(w,h), centered. 1920x1440 becomes 1920x1920 (240 px bars top and bottom);
  640x480 becomes 640x640.
- `cv2.resize` to 224x224 with INTER_AREA.
- Input resolution is read **once from the first episode's** main and UW videos (`:402-410`). Every episode must share
  those resolutions, or the `assert img.shape == (ih, iw, 3)` fails.
- Frames whose index is ≥ `in_stream.frames` are **skipped silently**, leaving zeros (black) (`:588-595`). The container
  must report an accurate frame count, as MP4 `stsz` does.

**Training mapping** (`results/block_raise/run_001_2026-10-05/run_config/config.yaml`, task `umi_prompt` → `BP/config/task/umi.yaml`):
- **obs:**
  - `camera_right_main_rgb` [3,224,224]: horizon 2, down_sample_steps 3 (60 to 20 Hz)
  - `camera_right_ultrawide_rgb` [3,224,224]: horizon 2, down_sample_steps 1 *in UW index space* (two distinct 10 Hz
    frames, via `map_upsample_index`, `BP/common/sampler.py:674-688`)
  - `gripper_right_eef_pos` (3), `gripper_right_eef_rot_axis_angle` (raw 3 → rot6d 6), `gripper_right_gripper_width` (1):
    horizon 2, step 3
  - `gripper_right_eef_rot_axis_angle_wrt_start` (computed, not stored)
- **action** (10-D, horizon 16, step 3): built by concatenating `eef_pos`, `eef_rot_axis_angle` and `gripper_width` per
  prefix (`BP/common/sampler.py:274-286`), then converted to relative pos + rot6d + width.
- `image_resolution: 224`. Augmentation: RandomCrop 0.95 + ColorJitter (train), CenterCrop 0.95 (eval).
  `sample_type: task`, so tasks in meta are required (hence `label_whole_demo_as_task`).
- Key names are hardwired to `camera_right_*` / `gripper_right_*` in `umi.yaml:38-47` and the run config.

---

## 7. iPhone-specific assumptions, ranked by how hard a new rig is to match (hardest first)

| # | Assumption | Where | What the Jetson rig must do (or what to parameterize) |
|---|---|---|---|
| 1 | **Gripper width from ArUco tags seen by the ultrawide**, posed by solvePnP with an **undistorted pinhole** model (dist = 0), K = `ultrawideCameraIntrinsics` for the encoded resolution; 16 mm DICT_4X4_50 ids (2k, 2k+1) at ~0.098 m ±0.08 m depth. Width = tvec-x difference (geometry-dependent units). | `detect_aruco_iphone.py:95-109`, `calibrate.py:55-133`, `gripper_util.py:303-356`, `aruco_config.yaml` | The new UW (probably a fisheye or wide CSI lens) must see both finger tags near the edge of view. Either **rectify frames to pinhole before encoding** and write the rectified K, or add distortion support (a code change). Measure the new `artag_z_distance_from_ultrawide`. Width values won't numerically match iPhone data unless the camera-to-finger geometry is identical, which matters for mixing datasets and for mapping width to the robot gripper at deployment. A grippercalibration episode recorded on the rig is required. |
| 2 | **Pose semantics:** `poseTransforms` = row-major `world_T_maincam` in **ARKit camera axes (x right, y up, z back)**, metric, low drift, and **on the same clock as the frames at exposure time** (zero implicit latency). The TCP is reached through one fixed 4x4 per (gripperID, deviceIdentifier). | `align.py:195-196`, `gripper_util.py:377-394`, `gripper_calibration.yaml` | Either convert VIO output to world_T_maincam in ARKit convention and reuse a TCP offset measured for the new mount, or report any body frame and put a hand-eye-calibrated `body_T_TCP` into a new yaml entry (no code change). The TCP must stay *z forward, x right, y down*, because relative actions are learned in that frame and deployment assumes it. VIO latency must be compensated before writing. **Hazard:** many VIO stacks start at an exact identity rotation. `pose_6d_to_4x4` divides the rotvec by its norm (`CM/transform_util.py:81-93`), so a zero rotvec probably gives a NaN pose (not executed locally). Avoid exact identity or patch the code. Gravity alignment isn't needed for single-arm (relative repr) but is harmless. |
| 3 | **Timing/format of streams:** one UNIX-time clock; ms-precision ISO-8601 `...Z` strings; main ~60 Hz; `len(rgbTimes) == decoded frames of _rgb.mp4` (no encoder drops); `poseTimes` 1:1 with frames and **`poseTimes[0] == rgbTimes[0]`**; `ultrawideRGBTimes` same length as `rgbTimes`, **slotted** into main ticks with `""` placeholders; first non-empty slot index × 1/60 = UW start offset; UW mp4 PTS starting at 0 at the first UW frame and spaced by real time. | `detect.py:65-69`, `align.py:79-84,155-198`, `gripper_util.py:272-282` | Hardware-trigger or timestamp both CSI cameras on a shared clock. Assign each UW frame to the main frame at or just before it (write that main frame's time, or the true UW time; align uses the string values, detect uses PTS plus slot offset). Interpolate VIO poses to main-frame times when writing. Make sure the encoder never drops frames (or drop the matching time entries). Sustaining 1920x1440 at 60 fps from CSI with HW encode on Jetson needs checking. Alternative: parameterize the 1/60 in `detect.py:69` and `align.fps`. |
| 4 | **Image domain:** main 1920x1440 (4:3, HFOV ≈ 74°, fx 1278), UW 640x480 (4:3, HFOV ≈ 105°, fx 247), RGB, phone-mount viewpoint; pad-to-square depends on aspect ratio. | `gen_replay_buffer_zarr.py:402-410,579-604`, `cv_util.py` | The pipeline accepts any resolution if it is constant across a dataset. But a different FOV, aspect (e.g. 16:9 gives different black bars), color response or viewpoint is a domain shift from iPhone data and from any deployment camera. To mix with or reuse iPhone-trained policies, match 4:3, similar FOV, and the mount pose. Training from scratch on rig data is fine. |
| 5 | **Main-camera frame count fidelity:** PyAV with `in_stream.frames` > 0 (MP4 with sample tables); otherwise frames are skipped and stay black. | `gen_replay_buffer_zarr.py:588-604`, `detect_aruco_iphone.py:45` | Write standard MP4 (H.264/H.265) via GStreamer `mp4mux` with proper finalization. Don't use MKV or raw streams. |
| 6 | **File and folder naming:** `<YYYY-MM-DDTHH-MM-SS.mmmZ>_<rand>_<session>_<demonstration|grippercalibration>_<left|right|head>` plus suffixes `.json`, `_rgb.mp4`, `_ultrawidergb.mp4`, `_depth.raw`; exactly 5 underscore fields (no `_` in session); date before `T` parsed for the folder. | `group.py:33-42,68-70`, `generic_util.py:132-133,248-259,306-308` | Easy: copy the convention. The side is normally `left` or `right` (single mode normalizes to `right`). |
| 7 | **JSON schema keys:** listed in §2 (required: `rgbTimes`, `poseTimes`, `poseTransforms`, `ultrawideRGBTimes`, `ultrawideCameraIntrinsics`, `deviceIdentifier`, `gripperID`, `gripperCalibrationRunName`; for labels `labelType`, `taskNames`, `taskStart/EndTimestamps`; `type` + `sessionName` for `label_whole_demo_as_task`; recommended `sidesPresent`, `isErrorCorrection`, `mainCameraIntrinsics`, `depthTimes`). | §2 | Easy. Calibration episodes need `ultrawideRGBTimes` + `ultrawideCameraIntrinsics` + device/gripper ids, and can have empty pose/rgb arrays. |
| 8 | **Calibration keying by Apple model string** (`deviceIdentifier`, default `"iPhone16,1"`) under `gripperID`. | `generic_util.py:91-120` | Easy: add a yaml entry for the rig's identifier. |
| 9 | **Depth:** 256x192 float16 raw, frame-aligned with `depthTimes`; hardcoded shape. | `depth_util.py:5`, `gen_replay_buffer_zarr.py:200-201`, `visualize.py:169` | Optional. Omit `_depth.raw`, keep `include_depth=false`, and omit `depthTimes` (or set it equal to `rgbTimes`). If depth is ever provided, it must be resampled to 256x192 float16 metres or the shape parameterized. |
| 10 | **60 Hz / 10 Hz constants in training config:** `obs_down_sample_steps: 3` ("relative to 60Hz"), `ultrawide_down_sample_steps: 1` ("relative to 10Hz"). | `BP/config/task/umi.yaml:8-9`, run config | If the rig runs at other rates, adjust these to keep 20 Hz obs and actions with ~10 Hz UW history, or keep 60/10. |
| 11 | **ARKit gravity-aligned shared world for multi-device; bimanual checks in ARKit-camera axes.** | `align_validate.py:108-152` | Not needed for one rig. For two rigs, a shared VIO map plus common clock would be needed. |

---

## 8. Other gotchas noticed

- `process_dataset.sbatch` doesn't skip the label stage's `visualize: true`, so `{side}_subtasks.mp4` is re-encoded for
  every demo.
- `iterate_demonstrations` asserts every `tmp_demonstrations/*/*` dir has a side JSON (`generic_util.py:403`), so stray
  dirs crash the run.
- `get_demonstration_path` uses `.replace('_left','')` etc. on the whole name. That is fine only because sessions can't
  contain `_`.
- Rotvec component-wise linear interpolation (`align.py:70-76`) is fine at 60 Hz VIO-to-frame resampling. It would
  misbehave near |θ| ≈ π if poses were sparse.
- `include_depth` with `depthTimes=[]` would give index −1 entries that are silently skipped. Harmless when depth is off.
- `demonstration_has_data_for_side` checks `{side}.mp4`, which the iPhone never writes. Dead code.

## 9. Open questions

1. Is the ARKit ultrawide buffer (`_capturedUltraWideImage`) already distortion-corrected? If not, iPhone widths carry
   distortion bias near the periphery and the "pinhole" assumption was always approximate. This affects how faithfully
   a rectified CSI ultrawide should mimic it.
2. Will the rig record both cameras on a hardware-synced clock, and can it sustain 1920x1440 at 60 fps main? If not,
   parameterize `1/60` (`detect.py:69`) and `align.fps`.
3. Which frame does the VIO stack report (IMU, left cam, body), and does its first pose equal identity (NaN hazard)?
4. Should rig data be mixed with iPhone data (`firstreal`) in one dataset? If so, FOV/aspect/mount and the width scale
   must match closely. If not, only internal consistency matters.
5. Does the deployment side (controlSW on the Pi 5/Franka) assume the same TCP convention and width scale (0 = closed,
   tag-x metres)? A rig change to TCP or width units must be mirrored there.
6. Who generates `gripperCalibrationRunName` and the 5-digit randomizer on the rig? Any 5-field-safe token works, but the
   cal episode must have been grouped into the same `tmp_demonstrations` tree.
