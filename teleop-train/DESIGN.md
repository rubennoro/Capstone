# teleop-train: design

Turns teleop episodes into the training dataset of `rnoro/training`'s `train/` scripts. The episodes are recorded by
`deployment/teleop_collect`, and the dataset is a `<dataset>.zarr.zip` + `package.json`. The goal, the rules and every
decision are in [PLAN.md](PLAN.md); the commands are in [DIRECTIONS.md](DIRECTIONS.md). This file is the design in
short.

**Does:**
- read raw episodes;
- put the arm pose, gripper width and both cameras on one 60 Hz grid;
- trim the still ends;
- prepare 224×224 images exactly as the deployment does live;
- write one `episode.zarr` per episode;
- join the usable episodes into the training zip, in the handheld (iPhUMI) format.

**Does not:**
- write anything under the raw root;
- train;
- deploy;
- run SLAM or calibration (teleop poses come from the arm);
- mix teleop and handheld data in one dataset.

## Data flow

```
~/CapstoneTeleop/<task>/episode_NNN/          teleop_collect's data_root, read only
  meta.yaml  robot_states.csv  main.mkv + main_frames.csv  ultrawide.mkv + ultrawide_frames.csv
      │ episode_reader: arrays + file checks
      ▼
  window (all three streams overlap) → 60 Hz grid → interpolate pose (lerp + slerp) and width → trim still ends
      │ trajectory: gap checks on the kept span
      ▼
  nearest main frame per step, ~10 Hz ultrawide sub-stream → JPEG bytes → cv2 decode → ImagePrep (224×224)
      │ images
      ▼
~/Capstone/data/<task>/episode_NNN/processed/episode.zarr + status.json       convert
      │ select (success, --episodes N), check one rig / TCP, the deployment's rig and TCP
      ▼
~/Capstone/data/<task>/package/<dataset>/<dataset>.zarr.zip + package.json    package
      │ cd ~/Capstone && train/all.sh <task> <dataset>   (TRAIN_CONFIG: umi_policy_dunet | umi_policy_dunetp)
```

Both clocks are already mapped to the laptop's by teleop_collect, so no clock work happens here:
- arm states: `t_ns`;
- camera frames: mid-exposure `t_mid_ns`.

The output arrays are the handheld zarr keys that `task/umi.yaml` reads:
- `camera_right_main_rgb`, `camera_right_ultrawide_rgb`;
- `gripper_right_eef_pos`, `gripper_right_eef_rot_axis_angle`, `gripper_right_gripper_width`;
- the `meta/` index maps.

Plain diffusion and behavior prompting read the same zip. Actions are not stored: training builds them from the
future poses, relative to the current one.

## Matching the deployment

A policy trained on this data sees in training what `policy_deploy` gives it live:

| What | How |
|---|---|
| Images | the same decoder (`cv2.imdecode` + BGR→RGB, as `policy_deploy`'s `JpegDecoder`) and the same `ImagePrep` code, copied from `final-deployment` @ `ea9f7fb`, built from the recorded rig |
| Rig | teleop_collect records with the deployment's `contracts/rigs/rig_pi_v3_640.yaml`; `package` refuses another one, because the ultrawide prep depends on the lens model |
| TCP | the PC sends poses as `O_T_F × F_T_TCP`; `package` refuses a dataset whose `F_T_TCP` differs from `deployment/contracts/frames.yaml` |
| Pose | `O_T_TCP_target` by default, as `policy_deploy`'s `pose_source: target` |
| Rotation | stored as rotation vectors; training and deployment both turn them into matrices first, so the sign flips near π change nothing (checked, 1e-15) |

## The parts

| File | What it does |
|---|---|
| `paths.py` | names, folder layout, the `teleop_source.yaml` marker (refuses a handheld folder of the same name or an output that overlaps the raw root) |
| `episode_reader.py` | reads one episode into arrays; refuses missing files and columns, non-increasing times, `video_index` ≠ row, frame count ≠ CSV rows, non-rigid poses |
| `trajectory.py` | the exact 60 Hz grid, lerp/slerp, motion detection (translation, rotation or width speed), the 0.5 s trim, the gap limits |
| `images.py` | frame selection (copied `frames.py`), JPEG bytes demuxed with `av`, decoded and prepared once each |
| `convert.py` | one episode end to end; `status.json` (written last) for converted and excluded episodes; the up-to-date rule; the `convert` command |
| `package.py` | selection, consistency and deployment checks, the atomic dataset folder, `package.json`; the `package` command |
| `cli.py` | `teleop_train convert` / `package` and their shared options |
| `_copied/` | code copied unchanged from `rnoro/data-work-auto` @ `839d23b` (frame selection, zarr writer, replay buffer, JPEG-XL codec) and `rnoro/final-deployment` @ `ea9f7fb` (image prep, rig parser); `tests/test_copies.py` checks each against its source |

## The data on disk

- **`status.json` per episode** (`teleop_train.episode.v1`):
  - the state (`converted` / `excluded`) and the one-line reason;
  - the outcome;
  - the window and the kept span;
  - the step counts;
  - `pose_source`, `F_T_TCP`, the rig and its sha256;
  - the settings, `CONVERT_VERSION`, and `code_sha256` (a hash of the converting code);
  - the raw `meta.yaml` sha256.
- **Up to date:** an episode is up to date when all of these match. So a changed setting, a code change, another pose
  source or an edited raw episode converts again.
- **`package.json`:** the keys the `train/` scripts read (`label_dir`, `zip.sha256`, `totals.n_steps`,
  `totals.episodes`), plus:
  - the episodes and the excluded ones, with reasons;
  - the rig, `pose_source` and `F_T_TCP`;
  - warnings (an episode over 60 s, the prompting limit).
- **Atomic writes:** a crash never leaves a half file that looks complete:
  - zarrs are written under a temp name and renamed;
  - the dataset folder is built as a temp sibling and renamed;
  - `status.json` is removed before a conversion and written last.

## Tests

`tests/`, 340 tests, about 2.5 min (`unset PYTHONPATH; ~/envs/teleop_train/bin/python -m pytest tests -q`).

- **`synthetic_episode.py`:** builds teleop_collect-format episodes with a known trajectory (still, move, rotate,
  grip, still) and fault options. Their JPEG frames encode their own index, so tests can read back which frame
  landed on which step.
- **Per part:** reader, paths, trajectory (against the exact truth), images (bit-identical to decode + `ImagePrep`),
  convert (every check, the up-to-date rule, image/grid alignment), package (selection, every refusal, atomicity,
  the zip read back), and the copies.
- **Checked once by hand (wave 3, PLAN §12):** real `teleop_collect run --mock` episodes through `package`, and the
  zip loaded with `behavior_prompting`'s `UmiTaskDataset` under both configs.

## How to extend

- **A new conversion setting:**
  1. Add it as a module constant.
  2. Add it to `convert.SETTINGS`, which makes old conversions stale.
  3. Add its §5.3 row in PLAN.md and a test.
- **Another deployment rig or TCP:**
  1. Re-sync `deployment/contracts/` with final-deployment.
  2. Point `package.DEPLOY_RIG_YAML` at the new rig file.
  3. Re-copy `_copied/_dataproc/` if the parser changed, and update `tests/test_copies.py`.
- **Re-copying a file into `_copied/`:** keep its `# copied from <branch>:<path> @ <commit>` line. The code hash makes
  existing conversions stale by itself.
