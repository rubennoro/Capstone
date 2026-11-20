# teleop-train: plan

Turns the episodes recorded by `deployment/teleop_collect` into the training dataset that the `train/` scripts on
`rnoro/training` upload and train on, so a teleop dataset trains exactly like a handheld one: plain diffusion
(`umi_policy_dunet`) or behavior prompting (`umi_policy_dunetp`).

`fd:<path>` = `rnoro/final-deployment` @ `ea9f7fb` (`~/Capstone-final-deployment/deployment/<path>`), read-only.
(Waves 0-1 copied from `2ffe641`; `ea9f7fb` changed the TCP and the deployment rig, §8.)
`up:<path>` = `rnoro/data-work-auto` @ `839d23b` (`~/Capstone-data-work/umi-processing/umiproc/dp/<path>`), read-only.
`tr:<path>` = `rnoro/training` @ `d58817b` (`~/Capstone/<path>`).

## 1. What and why

- **In:** `~/CapstoneTeleop/<task>/episode_NNN/` (teleop_collect PLAN §6). Each episode holds two MJPEG videos with
  per-frame times, the arm state at 100 Hz, and `meta.yaml`.
- **Out:** `~/Capstone/data/<task>/package/<dataset>/{<dataset>.zarr.zip, package.json}`. This is the layout
  `tr:train/all.sh` uses as-is when it is run without `--pull` / `--package`.
- **Why a separate tool:** handheld data needs SLAM, maps, tags and gripper calibration (`umi-processing/`). Teleop data
  already has the TCP pose from the arm and the times on one clock. What is left: resample onto a grid, prepare the
  images, write the zarr. Those three pieces are copied; nothing else of `umi-processing/` is needed.

**Out of scope:**
- changes to `deployment/teleop_collect/`, to `rnoro/training` or to `policy_deploy`;
- mixing teleop and handheld episodes in one dataset;
- the dataset report's teleop-specific sections (it fails soft, §7).

## 2. Assumption: the training side

This plan assumes `rnoro/training` gets a **`--config NAME` option** on `train/submit.sh` and `train/all.sh` (built
separately, not part of this plan):

- `NAME` is `umi_policy_dunet` (plain diffusion) or `umi_policy_dunetp` (behavior prompting, the current default).
- `train.sbatch` runs `train.py --config-name=NAME`.
- `job.json` / `job_info.json` record it.
- `eval.sbatch` skips its attention-video step for a plain run.

Both configs read the same zip (§6), so nothing here depends on which one is chosen.

## 3. Rules

1. **The raw episodes are read-only.** Nothing under `~/CapstoneTeleop/` is ever written, moved or deleted.
2. **Training images = deployment images.** The image prep is `fd:policy_deploy/policy_deploy/frames/_dataproc/`, the
   copy `policy_deploy` runs live (it differs from `umi-processing`'s). It is copied unchanged.
3. **Copied code says so.** Each copied file starts with `# copied from <branch>:<path> @ <commit>` (plus
   `# changes: imports only` or the list of changes), as in `teleop_collect`. No import from another branch, and no
   import of `teleop_collect` (its files are read as plain YAML / CSV).
4. **Refuse, don't guess.** An episode that fails a check (§5.3) is left out with a one-line reason. It is never
   patched up.
5. **Deterministic.** The same episodes and settings give the same arrays. Only the zip's timestamps may differ.
6. **One environment:** `~/envs/teleop_train` (§10). Never install into `~/envs/teleop_collect` or `~/envs/policy_deploy*`.
7. **Python 3.10**, `from __future__ import annotations`, dataclasses, type hints, integer ns times. Errors are one
   line naming what failed and what to check.

## 4. Data flow, locations, commands

```
~/CapstoneTeleop/<task>/                       teleop_collect's data_root (read only)
  task.yaml
  episode_NNN/  meta.yaml  COMPLETE  robot_states.csv  main.mkv  main_frames.csv  ultrawide.mkv  ultrawide_frames.csv
        │
        │  teleop_train convert <task>
        ▼
~/Capstone/data/<task>/                        rnoro/training's LOCAL_DATA_ROOT/<folder>
  teleop_source.yaml                           marker: made by teleop_train, from which raw task folder
  task.yaml                                    copied from the raw task folder
  episode_NNN/processed/episode.zarr           one per converted episode (same name as the raw episode)
  episode_NNN/processed/status.json            converted / excluded, why, window, settings
        │
        │  teleop_train package <task> <dataset>
        ▼
~/Capstone/data/<task>/package/<dataset>/<dataset>.zarr.zip + package.json
        │
        │  cd ~/Capstone && train/all.sh <task> <dataset> --config umi_policy_dunet
        ▼
cluster training → ~/Capstone/data/<task>/policy/<run>/
```

### 4.1 Commands

Run as `~/envs/teleop_train/bin/python -m teleop_train <command>`; the console script `teleop_train` does the same.

| Command | What it does |
|---|---|
| `convert TASK [--episodes 3,7-9] [--force]` | Converts every saved episode of the task that is not yet converted with the current settings. Prints one row per episode: number, outcome, raw and kept duration, steps, state (`converted` / `excluded` / `up to date`) and the reason. `--force` redoes the converted ones too. |
| `package TASK DATASET [--episodes N] [--include-fail] [--overwrite]` | Runs `convert` for whatever is missing, then builds the zip and `package.json` from the converted episodes, in episode order. `--episodes N` keeps the first N, so 10/20/30 are nested subsets. Finally it prints the `train/all.sh` command to run next. |

Common options:
- `--raw-root DIR` (default `~/CapstoneTeleop`);
- `--out-root DIR` (default `~/Capstone/data`);
- `--pose-source target|measured` (default `target`, §8).

**Exit codes:** 0 done (some episodes may be excluded); 1 nothing usable or a refusal (the message says why).

**Refusals:**
- `~/Capstone/data/<task>/` exists without `teleop_source.yaml` (a handheld folder of the same name);
- `teleop_source.yaml` names a different raw folder;
- `DATASET` already exists, unless `--overwrite` is given (the cluster refuses a reused name with a different zip anyway).

## 5. Converting one episode

### 5.1 Inputs read

| File | Used |
|---|---|
| `COMPLETE` | must exist (teleop_collect deletes folders without it, but never trust that) |
| `meta.yaml` | `meta_version` (must be 1), `task`, `episode`, `outcome`, `rig` (the rig file's content), `pc_hello.F_T_TCP` |
| `robot_states.csv` | `t_ns` (laptop clock), `seq`, `O_T_TCP_00..33` or `O_T_TCP_target_00..33` (§8), `gripper_width_m` |
| `main_frames.csv`, `ultrawide_frames.csv` | `video_index`, `t_mid_ns` (mid-exposure, laptop clock) |
| `main.mkv`, `ultrawide.mkv` | the JPEG frames; frame `i` = row `i` of the CSV |

The arm times and the camera times are both already mapped to the laptop clock by teleop_collect, so no clock work
happens here.

### 5.2 Steps

1. **Window:** from the latest first sample to the earliest last sample of the arm states, the main frames and the
   ultrawide frames.
2. **Grid:** 60 Hz over the window, `t_k = t_0 + round(k × 1e9 / 60)`. Why 60 Hz: `tr:behavior_prompting/.../task/umi.yaml`
   downsamples ×3 to 20 Hz and assumes a ~10 Hz ultrawide; the handheld data uses the same grid.
3. **Pose on the grid:** position by linear interpolation; rotation by `scipy` `Slerp` between the two neighbouring
   states. Then convert to `eef_pos` (m) and `eef_rot_axis_angle` (rotation vector). The frame is the robot base.
   Training makes every pose relative to the current one, so the world frame cancels (unlike the handheld table-tag
   frame, nothing has to agree across sessions).
4. **Width on the grid:** linear interpolation of `gripper_width_m`.
5. **Trim the idle ends:**
   - A step is **moving** when any of these holds, each speed averaged over 0.25 s:
     - TCP speed > 5 mm/s;
     - rotation speed > 1 °/s;
     - |width rate| > 5 mm/s.
   - The thresholds are below the slowest real motion: slow mode is 12.5 mm/s per axis and 2.5 °/s.
   - Keep from 0.5 s before the first moving step to 0.5 s after the last one, clamped to the window. Pauses in the
     middle are kept.
   - Why 0.5 s: an action chunk spans 0.8 s (16 steps at 20 Hz), so the still samples at the start carry the coming
     motion and the policy learns to start from home rather than to wait. The 0.5 s at the end teaches it to stop.
6. **Main images:** for each grid step, the nearest main frame (`frames.nearest_frame_index`, at most 50 ms away).
   At 30 fps every step normally has one within 16.7 ms. 50 ms is half the 100 ms main-frame gap limit of §5.3, so
   a single dropped frame (a step up to 33 ms away) passes and the gap rule decides. At teleop speeds (at most
   5 cm/s), 50 ms is at most 2.5 mm between the pose and the image.
7. **Ultrawide images:** the ~10 Hz sub-stream and its per-step index (`frames.ultrawide_substream`, rate 10 Hz).
8. **Decode and prepare:**
   - Demux the `.mkv` with `av` (no decoding) to get each used frame's JPEG bytes, each once. Decode them exactly
     as `fd:policy_deploy/policy_deploy/frames/jpeg_decoder.py` does live:
     `cv2.imdecode(..., cv2.IMREAD_COLOR)` then `cv2.COLOR_BGR2RGB`. Then the pixels match the deployment bit for
     bit; `av`'s own decoder upsamples chroma differently.
   - Check each frame against the rig's resolution (640×480 for `rig_pi_v3_640`).
   - Prepare with `ImagePrep(parse_rig(meta.rig))`: `prep_main` letterboxes to 224; `prep_ultrawide` maps the fisheye
     to the virtual pinhole, then letterboxes to 224.
9. **Write:** `EpisodeArrays(name="<task>_episode_NNN", task_label=<task>, ...)`, then `write_episode_zarr` to
   `episode_NNN/processed/episode.zarr` (atomic replace), then `status.json`.

### 5.3 Checks (any failure → the episode is excluded, with the reason in `status.json`)

| Check | Limit |
|---|---|
| `COMPLETE` and a readable `meta.yaml` of version 1 | required |
| outcome `success` (unless `package --include-fail`) | `fail` episodes are converted but marked `outcome: fail` |
| `t_ns` / `t_mid_ns` strictly increasing; `video_index` = row number; `.mkv` frame (packet) count = CSV rows | exact |
| gap between consecutive arm states inside the kept span | ≤ 50 ms (2 s of `seq` gaps never pass) |
| gap between consecutive main frames inside the kept span | ≤ 100 ms |
| gap between consecutive ultrawide frames inside the kept span | ≤ 200 ms |
| every grid step has a main frame within 50 ms | all steps |
| some motion found | at least one moving step |
| kept span | ≥ 2 s |
| frame resolution = the rig's | exact |
| `O_T_TCP*` rows are rigid transforms | rotation orthonormal within 1e-6 |

### 5.4 `status.json`

- `schema: teleop_train.episode.v1`;
- `state` (`converted` / `excluded`) and `reason`;
- `outcome`;
- the raw path and the sha256 of its `meta.yaml`;
- window and kept span (ns), `n_steps`, `n_uw`;
- `pose_source`, `F_T_TCP`, `rig_id` and the sha256 of the rig content;
- the settings block (all §5.2 / §5.3 numbers), `CONVERT_VERSION` and `code_sha256` (a hash of the converting code:
  `episode_reader`, `trajectory`, `images`, `convert` and `_copied/`);
- the teleop_train commit and whether it was dirty.

`convert` treats an episode as **up to date** when its `status.json` matches the current `CONVERT_VERSION`, settings,
`code_sha256`, `pose_source` and raw `meta.yaml` sha256. So a code change converts again even without a version bump.

## 6. Packaging

1. **Selection:** the converted episodes, in episode number order. `fail` episodes are left out unless
   `--include-fail` is given. Then the first N with `--episodes N`.
2. **Consistency (refuse if any differs across the selection):** `rig_id` and rig sha256, `F_T_TCP`, `pose_source`,
   `CONVERT_VERSION`. A policy can only be deployed with one TCP and one rig.
   **The deployment's TCP:** also refuse when the shared `F_T_TCP` differs from this branch's
   `deployment/contracts/frames.yaml` (within 1e-9 per element, as the hello check), the TCP the deployment runs (§8).
   **The deployment's rig:** also refuse when the shared rig sha256 differs from that of
   `deployment/contracts/rigs/rig_pi_v3_640.yaml`, the rig the deployment preps images with (§8).
   **Fresh:** refuse a status whose settings or `code_sha256` differ from the current ones (the CLI converts first, so
   this only matters through the Python API).
3. **Zip:** `build_task_zarr_zip([...episode.zarr], <dataset>.zarr.zip)`, copied unchanged. The images are copied byte
   for byte; there is one task per episode, named `<task>`.

   The zip has exactly the keys of `task/umi.yaml`:
   - `camera_right_main_rgb`, `camera_right_ultrawide_rgb` (224×224×3, JPEG-XL);
   - `gripper_right_eef_pos`, `gripper_right_eef_rot_axis_angle`, `gripper_right_gripper_width`;
   - the `meta/` episode ends, task names and ultrawide index maps.

   So both `umi_policy_dunet` and `umi_policy_dunetp` read it.
4. **`package.json`:** what the `train/` scripts read, plus what makes the dataset traceable.

   | Key | Content |
   |---|---|
   | `dataset`, `label_dir`, `labels` | the name; `label_dir` = `labels[0]` = `<task>` (**read by upload.sh / submit.sh**) |
   | `totals` | `episodes`, `n_steps` (**read by submit.sh** to size the epochs), `per_label`, `excluded` |
   | `zip` | `name`, `size_bytes`, `sha256` (**checked by upload.sh**) |
   | `episodes` | per episode: `folder`, `slot` (`episode_NNN`), `pi_name` (`<task>_episode_NNN`), `label`, `n_steps`, `kept_s`, `outcome` |
   | `excluded` | per episode left out: `entry`, `state`, `reason` |
   | `source` | `teleop_collect`, the raw task folder |
   | `rig`, `pose_source`, `F_T_TCP` | shared by every episode (§6 step 2) |
   | `warnings` | e.g. an episode longer than 60 s after trimming (§7) |
   | `teleop_train_version`, `created` | |

## 7. Training on it

```bash
cd ~/Capstone
train/all.sh <task> <dataset> --config umi_policy_dunet     # plain diffusion
train/all.sh <task> <dataset> --config umi_policy_dunetp    # behavior prompting
```

With neither `--pull` nor `--package`, `all.sh` takes `data/<task>/package/<dataset>/` as it is and runs the usual
steps: upload, smoke run, full run, fetch, reports, offline eval.

Notes:
- **Long episodes:** `task/umi.yaml` sets `max_sequence_length: 1200`, which is 60 s at 20 Hz. Teleop is slow (5 cm/s
  per axis), so episodes can run longer. `package` warns per episode above 60 s. For prompting, raise the limit with a
  Hydra override (`train/submit.sh ... -- task.shape_meta.max_sequence_length=<steps>`). Plain diffusion only samples
  windows, so the limit matters there only if the dataset code enforces it (to be checked in wave 3, §11).
- **Dataset report:** `dataset_report.py` expects handheld files (`episodes.csv`, `validate.json`). `all.sh` treats
  its failure as non-fatal and goes on, so a teleop dataset trains even if that step fails.
- **Epochs:** `submit.sh` sizes them from `totals.n_steps`; the 60-epoch default stays.

## 8. Matching the deployment

- **Images:** the same `ImagePrep` code (rule 2) and the same rig content as recorded in `meta.yaml`.
  teleop_collect records with the deployment's rig, `deployment/contracts/rigs/rig_pi_v3_640.yaml` (training's rig with the
  measured ultrawide lens, as `fd:policy_deploy` uses it since `ea9f7fb`), so the recorded rig is the deployment's.
  The main image is only letterboxed, so the rig does not change it. The ultrawide is remapped through the lens model,
  so a different rig gives different ultrawide images: with the old nominal `rig_pi_v2_640`, 75 % of the pixels
  differ.
- **Pose source:** the default is `target` (`O_T_TCP_target`), because `fd:policy_deploy/config/manager.yaml` uses
  `pose_source: target` for both the proprioception and the anchor. With that, the policy observes, predicts and
  commands the same quantity.

  `--pose-source measured` exists for comparison. The deployed `pose_source` must then be set to match.
  `package.json` records which one a dataset uses.
- **TCP:** the one TCP is `deployment/contracts/frames.yaml` (a copy of `fd:contracts/frames.yaml`). Since `ea9f7fb` it
  is training's TCP axes: x = finger closing (image right), y = image down, z = toward the fingertips; +45° about
  flange z, fingertips 150 mm out. Still `settled: false`: `policy_deploy` refuses any trained policy on the arm until
  it is checked (`pc_probe --axes`). The PC sends every pose as `O_T_F × F_T_TCP`, and the laptop checks that the PC's
  `F_T_TCP` equals its own at start-up. So the recorded poses are in the TCP the deployment runs, and `package` refuses a
  dataset whose `F_T_TCP` differs from `frames.yaml` (§6 step 2). `F_T_TCP` is recorded in `package.json`.
- **Rotation vectors:** with the TCP pointing down, `eef_rot_axis_angle` has norm ~π and can flip sign between steps.
  Training (`sampler.py` only slices the arrays; `UmiTaskDataset` applies `pose_to_mat` before making poses relative)
  and deployment (`pose_converter.py`, 4×4 matrices) both work on matrices, so a flip changes nothing: checked
  numerically across 3 flips, obs and actions equal to 1e-15.
- **Rates:** obs every 3rd grid step (20 Hz), ultrawide ~10 Hz. These are the same as `policy_deploy` assumes for the
  handheld policies.
- **Gripper:** the policy predicts widths (metres) as before. How `policy_deploy` and the PC turn widths into Panda
  gripper commands is unchanged.

## 9. Tree

```
teleop-train/
  PLAN.md                        this file
  DESIGN.md  DIRECTIONS.md       written in wave 3
  pyproject.toml                 name teleop-train, script teleop_train = teleop_train.cli:main
  teleop_train/
    __init__.py  __main__.py  cli.py
    paths.py                     roots, folder names, the marker, the refusals of §4.1
    episode_reader.py            reads one raw episode (§5.1) into arrays + meta; the file-level checks
    trajectory.py                grid, interpolation, motion detection, trimming (§5.2 steps 1-5)
    images.py                    frame selection + decode + ImagePrep (§5.2 steps 6-8)
    convert.py                   one episode end to end, the §5.3 checks, status.json, up-to-date rule
    package.py                   selection, consistency, zip, package.json (§6)
    _copied/
      frames.py                  up:export/frames.py                 (unchanged)
      zarr_writer.py             up:export/zarr_writer.py            (imports only)
      _vendor/replay_buffer.py   up:_vendor/replay_buffer.py         (unchanged)
      _vendor/imagecodecs_numcodecs.py  up:_vendor/...               (unchanged)
      _dataproc/                 fd:policy_deploy/policy_deploy/frames/_dataproc/
        config.py  paths.py  imageprep.py  geometry/{camera_model,transforms}.py   (imports only)
  tests/
    synthetic_episode.py         builds a raw episode folder with a known trajectory and generated JPEG frames
    test_copies.py               each _copied file equals its source apart from the header and import lines
    test_episode_reader.py  test_trajectory.py  test_images.py  test_convert.py  test_package.py
```

## 10. Environment

`~/envs/teleop_train`, Python 3.10, with `pip install -e "teleop-train[test]"`:

| Package | Pin | Why |
|---|---|---|
| numpy | 2.2.6 | as `umi-processing` |
| scipy | 1.15.3 | Slerp, rotation vectors |
| opencv-python-headless | 4.12.0.88 | ImagePrep |
| av | 17.1.0 | MJPEG decode |
| pyyaml | 6.0.3 | meta.yaml |
| zarr / numcodecs / imagecodecs | 2.18.3 / 0.13.1 / 2025.3.30 | the replay buffer and JPEG-XL, as `umi-processing` |
| pytest (test extra) | 8.3.5 | |

## 11. Tests

- **Unit tests, on synthetic episodes** (`tests/synthetic_episode.py`): a known trajectory (still, move, rotate-only,
  gripper-only, still) at 100 Hz, and MJPEG videos at 30 fps whose frames encode their own index.
  - **reader:** columns, both pose sources, every malformed-file refusal.
  - **trajectory:** interpolation exact on linear/constant-rate inputs; slerp across a rotation near π; trimming
    bounds, with rotate-only and gripper-only motion counted as moving; mid-episode pauses kept.
  - **images:** the right frame index per step (read back from the encoded index); the ultrawide maps; prep output
    equal to calling the copied `ImagePrep` directly.
  - **convert:** every §5.3 row, the up-to-date rule, `--force`, the raw folder unchanged (hash before and after).
  - **package:** selection and `--episodes N`; consistency refusals; zip keys, shapes and dtypes as `task/umi.yaml`;
    the zip read back with the vendored `ReplayBuffer`; the `package.json` fields `train/` reads; sha256 correct.
  - **cli** (in test_convert.py and test_package.py): exit codes, the handheld-folder refusal, the printed next
    command.
- **Copy check:** `test_copies.py` compares each copied file with its source at the recorded commit, when that worktree
  exists (skipped otherwise).
- **Mock episode (lead, wave 3):** `teleop_collect run --mock` records a few episodes to `~/CapstoneTeleop_mock`
  (real Pi frames, fake-PC poses); then `teleop_train package` with `--raw-root ~/CapstoneTeleop_mock` and a scratch
  `--out-root`.
- **Training-side check (lead, wave 3):** load that zip with `behavior_prompting`'s `UmiTaskDataset` under both
  `umi_policy_dunet` and `umi_policy_dunetp` in the inference env (`env_bp`), draw one batch each, and check
  `max_sequence_length` (§7). No GPU and no cluster.

Test command, from `teleop-train/`: `unset PYTHONPATH; ~/envs/teleop_train/bin/python -m pytest tests -q`.

## 12. Build order

One executor per task, one reviewer over all of it. Each wave ends with a walk-through and a local commit (no push).

**Subagents:** the lead, and any executor, may dispatch subagents whenever that helps. Typical uses:
- read-only lookups across the source branches (`fd:`, `up:`, `tr:`);
- an independent check of a finished piece;
- splitting a large task into parallel parts.

A subagent gets a self-contained brief: the goal, the files and sections of this plan it needs, and what is already
ruled out. Its result is verified before it is used. Subagents follow the same rules (§3) and write only inside
their task's files. The executor that dispatched them owns the result and reports it.

| Wave | Who | Work | Done when |
|---|---|---|---|
| 0 | lead | the env; `pyproject.toml`; `_copied/` with headers; `test_copies.py`; `synthetic_episode.py` | copies import, copy check green |
| 1 | 3 executors in parallel | **E1** `episode_reader.py` + `paths.py`; **E2** `trajectory.py`; **E3** `images.py` | each one's tests green |
| 2 | 2 executors in parallel | **E4** `convert.py` (+ the `convert` command); **E5** `package.py` (+ the `package` command) | tests green; reviewer pass over waves 0-2 |
| 3 | lead + E6 | mock-episode run and training-side check (§11); **E6** `DESIGN.md`, `DIRECTIONS.md` (short, command by command) | a mock dataset loads in `UmiTaskDataset` under both configs |

After wave 3, the first real use: record teleop episodes, `teleop_train package`, then `train/all.sh` with the
`--config` option of §2 (once it exists on `rnoro/training`).

## 13. Decisions

| Decision | Why |
|---|---|
| Separate `teleop-train/` at the repo root, copying 3 pieces instead of depending on `umi-processing/` | no SLAM / tags / calibration needed; no cross-branch import |
| Output into `~/Capstone/data/<task>/` in the handheld layout | `train/all.sh` uses it without `--pull` / `--package` and with no change to `train/` |
| 60 Hz grid, 224 images, ~10 Hz ultrawide, the handheld zarr keys | `task/umi.yaml` (plain and prompting) reads it unchanged |
| Trim idle ends by translation, rotation and width speed, 0.5 s margin; keep mid-episode pauses | every episode starts still at home; untrimmed, the policy learns to wait |
| `success` episodes only by default (`--include-fail` to add the others) | behavior cloning copies whatever it is shown |
| `pose_source: target` by default | matches `policy_deploy`'s default proprioception and anchor |
| `ImagePrep` from `final-deployment`, not `umi-processing` | it is the code that runs live |
| Refuse mixed rig / TCP / pose source in one dataset | one policy deploys with one TCP and one rig |
| Refuse a dataset whose `F_T_TCP` differs from `deployment/contracts/frames.yaml` | a policy deployed with another TCP moves in the wrong frame |
| teleop_collect records with the deployment rig `contracts/rigs/rig_pi_v3_640.yaml` | the ultrawide prep depends on the rig's lens model |
