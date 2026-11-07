# DIRECTIONS: episodes in `/data/incoming` → training data

Step by step, for the Jetson. What each step does, and why, is in [README.md](README.md) (sections 3 and 4).
Every command here is safe to repeat. `/data/incoming` is only ever read, never changed.

## 0. Open a terminal and get ready

```bash
source ~/envs/env_dataproc/bin/activate
echo "${DATAPROC_ROOT:-not set}"
```

The `echo` must print `not set`. If it prints a folder (e.g. `~/dataproc_practice` from the practice run), run
`unset DATAPROC_ROOT`. Otherwise your output goes to that folder instead of `~/data`.

## 1. Record on the Pi

Use one task name per batch of recordings, e.g. `block_raise`.

**Calibration recording (once per task, and again if the gripper, tags or camera mount change).** One clip where:
- the gripper opens fully, closes fully and opens again, slowly;
- **both finger ArUco tags stay visible to the ultrawide (IMX219) camera** the whole time;
- the rig is held roughly still.

Write down its folder name when it arrives (e.g. `episode_000010`).

**Demonstrations.** Each clip is 15 s, and the **whole clip** becomes training data.
- **Start moving straight away:** about 2 s of gentle side-to-side / up-down movement in front of a textured scene.
  ORB-SLAM3 can't start mapping without motion. Static recordings always fail at SLAM.
- Then do the task, smoothly (faster than 3 m/s is rejected).
- Keep the finger tags in view (needed in at least 80% of frames).
- Good light, a scene with texture (not a blank wall).
- Behavior prompting needs **at least 2** good demos per task. Aim for dozens.

> **Finish one task before recording the next.** A run takes *every* unused episode in `/data/incoming` as a demo of
> the task you name (unless you pick them with `--episodes`).

## 2. Check what arrived

```bash
ls /data/incoming
ls /data/incoming/episode_000010        # each folder needs a COMPLETE file
```

Folders still being copied (no `COMPLETE`) are skipped automatically and picked up by a later run.

## 3. Preview (changes nothing)

```bash
dataproc run --task block_raise --calibration-episode episode_000010 --dry-run
```

You'll see:

```
data root: /home/capstone-gripper/data
source /data/incoming: calibration episode_000010, 5 new demonstration(s): episode_000011, ..., episode_000015
  skip episode_000016: no COMPLETE (still being recorded or transferred)
slam: orbslam3 (auto)
```

Check that the calibration and demo list are right. Anything skipped is listed with the reason.

## 4. Run it

Pick the line that fits:

```bash
# First time for a task: calibration + all new demos
dataproc run --task block_raise --calibration-episode episode_000010

# Later sessions for the same task: just the new demos (the task keeps its calibration)
dataproc run --task block_raise

# No calibration yet (e.g. testing whether SLAM tracks): runs everything up to the width step
dataproc run --task block_raise --no-calibration

# Only some episodes (the rest stay unused for later)
dataproc run --task block_raise --no-calibration --episodes episode_000004 episode_000005
```

It takes about **1.5 minutes per episode**, mostly ORB-SLAM3. While it runs you can follow progress from a second
terminal with `dataproc status --task block_raise`.

## 5. What the output means

The output appears in this order:

```
data root: /home/capstone-gripper/data                                    ← where everything goes
source /data/incoming: calibration (none: --no-calibration), 2 new demonstration(s): episode_000004, episode_000005
slam: orbslam3 (auto)                                                      ← ORB-SLAM3 found and used
rig: copied rig_pi_v1.yaml to /home/capstone-gripper/data/rigs/...        ← first run only
  episode_000004 -> pi-2026-11-06_episode_000004 -> episode_0             ← converted copy name -> task slot
  episode_000005 -> pi-2026-11-06_episode_000005 -> episode_1
process: task block_raise, slam orbslam3
calibration: no_calibration: no calibration/<name>/raw in the task folder ← or "done", or "calibration_failed: why"
episode_0: failed  [ran: check,slam; 76.6 s]                             ← one line per episode as it finishes
episode_1: failed  [ran: check,slam; 75.1 s]
slot       pi_name                       status  n_steps  reasons         ← summary table
episode_0  pi-2026-11-06_episode_000004  failed  -        slam: slam_error; slam: orbslam3: never initialised (0 keyframes; ...)
...
note: rig rig_pi_v1 still has nominal sections: cameras.main, ...         ← expected: rig not calibrated yet
no zip built: no exportable episodes                                      ← or: wrote .../block_raise.zarr.zip
```

(That example is the test recordings from 2026-11-06, which are static, so SLAM can't start on them. With moving
recordings and a calibration, the episodes read `exported` and the last line names the zip.)

**Per-episode status:**

| Status | Meaning |
|---|---|
| `exported` | Good: in the zip |
| `failed` at `slam` | ORB-SLAM3 didn't work on it (see the reason) |
| `failed` at `width` | No calibration, or it failed |
| `rejected` at `validate` | Usable data, but outside a quality limit (speed, tracking, tag visibility, sensor gaps) |
| `rejected` at `check` | A file or format problem |

The full list of reasons and fixes is in the [README troubleshooting table](README.md#8-troubleshooting).

**Exit code** (`echo $?` right after): `0` = zip built, `3` = no SLAM installed, `1` = otherwise.

## 6. Look at the results

```bash
dataproc status --task block_raise          # the table again, any time
```

In the Files app (Ctrl+L, paste the path) or VS Code:

| Path (under `~/data/tasks/block_raise/`) | What to look at |
|---|---|
| `episode_N/raw/cam0.mp4`, `cam1.mp4` | The episode's videos (main, ultrawide). Double-click to play. |
| `episode_N/processed/status.json` | Every stage: done / failed / skipped, and why |
| `episode_N/processed/slam/work/orbslam3.log` | What ORB-SLAM3 said |
| `episode_N/processed/validate.json` | Quality numbers: speed, tracking, tag coverage |
| `calibration/<name>/processed/` | The calibration result (`gripper_range.json`) |
| `output/report.json` | Summary of the whole task |
| `output/block_raise.zarr.zip` | **The training file** |
| `~/data/pi_sources.csv` | Which `/data/incoming` episode went where |

Each processed episode takes ~300 MB, most of it SLAM's frame images.

## 7. Train on the cluster

When the zip exists, the run prints the commands. Pick a dataset name (e.g. `rig-2026-11-10`), then from the Jetson:

```bash
ssh <you>@xfer.discovery.neu.edu mkdir -p CapstoneProject/datasets/block_raise/<dataset>
rsync -av ~/data/tasks/block_raise/output/block_raise.zarr.zip \
    <you>@xfer.discovery.neu.edu:CapstoneProject/datasets/block_raise/<dataset>/<dataset>.zarr.zip
```

and from the laptop:

```bash
scripts/run_pipeline.sh train --task block_raise --dataset <dataset> --note "first rig data"
```

That trains behavior prompting (`umi_policy_dunetp`). The same file works for regular diffusion
(`umi_policy_dunet`); see [../DIRECTIONS.md](../DIRECTIONS.md) for training options.

## 8. Next time

| Situation | Do |
|---|---|
| More demos of the same task | Record, then `dataproc run --task block_raise`. Only new episodes are processed; the zip is rebuilt with all of them. |
| Added a calibration after `--no-calibration` | `dataproc run --task block_raise --calibration-episode <dir>`. Only width → export rerun. |
| New calibration (gripper/tags changed) | Same command with the new folder. The newest calibration is used. |
| A different task | Record it after finishing the previous batch, then `dataproc run --task <other_task> --calibration-episode <dir>` |
| Reprocess an episode already taken in | `dataproc process --task block_raise --episode N --force`. Taking the same recording in a second time isn't supported (it would be rejected as a duplicate). |
| Rig values changed (`~/data/rigs/rig_pi_v1.yaml`) | `dataproc process --task block_raise --force` |
