# DIRECTIONS: recordings in `/data/incoming` → training data

Step by step, for the Jetson. What each step does, and why, is in [README.md](README.md) (sections 3–5).
Every command here is safe to repeat. `/data/incoming` is only ever read, never changed.

One recording session = **calibrate → run → package**:

```bash
dataproc calibrate block_raise episode_000010                      # creates tasks/block_raise (or block_raise_2, ...)
dataproc run block_raise episode_000011 episode_000040             # the session's demos -> episode_1, episode_2, ...
dataproc package block_raise-1110 block_raise                      # -> ~/data/datasets/block_raise/block_raise-1110/
```

## 0. Open a terminal and get ready

```bash
source ~/envs/env_dataproc/bin/activate
echo "${DATAPROC_ROOT:-not set}"
```

The `echo` must print `not set`. If it prints a folder (e.g. `~/dataproc_practice` from the practice run), run
`unset DATAPROC_ROOT`. Otherwise your output goes to that folder instead of `~/data`.

## 1. Record on the Pi

Use one task name per session, e.g. `block_raise`.

**Calibration recording (first, at the start of every session, and again if the gripper, tags or camera mount
change).** One clip where:
- the gripper opens **fully** and closes **fully**, several times (at least 5 cycles), slowly;
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

Write down the first and last demo folder names (e.g. `episode_000011` … `episode_000040`). `run` takes exactly
that range, so other recordings in `/data/incoming` never mix in.

## 2. Check what arrived

```bash
ls /data/incoming
ls /data/incoming/episode_000010        # each folder needs a COMPLETE file
```

Folders still being copied (no `COMPLETE`) are skipped and listed; run the same command again later.

## 3. Calibrate (creates the task folder)

```bash
dataproc calibrate block_raise episode_000010
```

- **OK:** `calibrate: OK -> tasks/block_raise (label block_raise, rig rig_pi_v1)`, the measured opening (it must be
  0.110 m ± 0.010 m on `rig_pi_v1`), tag statistics, and the `dataproc run` command to use next. If `block_raise`
  already exists (an earlier session), the new folder is `block_raise_2`, then `block_raise_3`, … — **use the folder
  name it prints** in the next steps. All of them are labelled `block_raise` in the training file.
- **FAILED:** it prints why (tag statistics and likely causes), creates nothing, and the recording can be tried
  again after fixing the cause, or record a new calibration. The common causes are in the
  [README troubleshooting table](README.md#8-troubleshooting).
- A warning `only N open/close cycle(s) detected` is not fatal, but record more cycles next time.

## 4. Run the session's demonstrations

Preview first (changes nothing):

```bash
dataproc run block_raise episode_000011 episode_000040 --dry-run
```

Each recording in the range is listed with what will happen (`take`) or why it is skipped (still syncing, invalid,
a calibration, already an episode, failed earlier). Then run it:

```bash
dataproc run block_raise episode_000011 episode_000040
```

It goes one recording at a time, about **1.5 minutes each**, mostly ORB-SLAM3. Each line reads `-> episode_N`
(passed every stage) or `-> failed (<stage>: <reason>)`. While it runs you can follow progress from a second
terminal with `dataproc status block_raise`.

- Episodes are numbered from **`episode_1`**, gap-free, in the order they passed. Failed recordings go to
  `tasks/block_raise/failed/<name>/` with `reason.json` and never get a number.
- Re-running the same range is safe: it only picks up recordings that are new (e.g. finished syncing since).
- **Exit code** (`echo $?` right after): `0` = done (also if some failed; they're listed), `3` = waiting for SLAM
  (ORB-SLAM3 not found; build it and run the same command again), `1` = it could not run (message on screen).

## 5. What the output means

```
calibration: done (pi-2026-11-10_episode_000010)
run: tasks/block_raise, 30 recording(s) to process, 0 skipped, slam orbslam3
[1/30] episode_000011: take
[1/30] episode_000011 -> episode_1  [ran: check,slam,grid,pose,width,validate,images,export; 84.2 s]
...
run summary: tasks/block_raise, episode_000011..episode_000040
  episode_000011  ->  episode_1
  episode_000012  ->  failed (validate: slam_tracking)
  ...
  28 episode(s), 2 failed, 0 waiting for SLAM, 0 skipped
<the dataproc status table>
Next: package the episodes for training (pick a dataset name):
  dataproc package block_raise-1110 block_raise
```

| Failed at | Meaning |
|---|---|
| `check` | A file or format problem, or the recording's cameras don't match the rig file |
| `slam` | ORB-SLAM3 didn't work on it (see the reason; usually not enough motion) |
| `validate` | Usable data, but outside a quality limit (speed, tracking, tag visibility, sensor gaps) |
| `images` | A step had no main camera frame |

The full list of reasons and fixes is in the [README troubleshooting table](README.md#8-troubleshooting). After
fixing a cause (e.g. the rig file), `dataproc run block_raise <first> <last> --retry-failed` re-runs the failed
ones from the first stage whose inputs changed (add `--force` to re-run all their stages).

## 6. Look at the results

```bash
dataproc status block_raise          # the table again, any time
```

In the Files app (Ctrl+L, paste the path) or VS Code:

| Path (under `~/data/tasks/block_raise/`) | What to look at |
|---|---|
| `episode_N/raw/cam0.mp4`, `cam1.mp4` | The episode's videos (main, ultrawide). Double-click to play. |
| `episode_N/processed/status.json` | Every stage: done / failed / skipped, and why |
| `episode_N/processed/slam/work/orbslam3.log` | What ORB-SLAM3 said |
| `episode_N/processed/validate.json` | Quality numbers: speed, tracking, tag coverage |
| `failed/<name>/reason.json` | Why a recording failed (stage, reason codes) |
| `.staging/<name>/` | A recording waiting for SLAM (or interrupted); the next `run` resumes it |
| `calibration/<name>/processed/`, `gripper_range.json` | The calibration result |
| `report.json` | The last run's outcome per recording |
| `~/data/pi_sources.csv` | Which `/data/incoming` recording went where |

Each processed episode takes ~300 MB, most of it SLAM's frame images.

## 7. Package and train on the cluster

Pick a dataset name (e.g. `block_raise-1110`) and list the folders to include (one session, or several):

```bash
dataproc package block_raise-1110 block_raise                     # one session
dataproc package block_raise-all block_raise block_raise_2        # two sessions of the same task
```

It writes `~/data/datasets/block_raise/<dataset>/<dataset>.zarr.zip` plus `package.json`, lists what it included
and excluded (failed and unfinished recordings never go in), and prints the commands to copy it. From the Jetson:

```bash
ssh <you>@xfer.discovery.neu.edu mkdir -p CapstoneProject/datasets/block_raise/<dataset>
rsync -av ~/data/datasets/block_raise/<dataset>/<dataset>.zarr.zip \
    <you>@xfer.discovery.neu.edu:CapstoneProject/datasets/block_raise/<dataset>/<dataset>.zarr.zip
```

and from the laptop:

```bash
scripts/run_pipeline.sh train --task block_raise --dataset <dataset> --note "first rig data"
```

That trains behavior prompting (`umi_policy_dunetp`). The same file works for regular diffusion
(`umi_policy_dunet`); see [../DIRECTIONS.md](../DIRECTIONS.md) for training options. Several tasks in one dataset
go to `~/data/datasets/multi/<dataset>/`.

## 8. Next time

| Situation | Do |
|---|---|
| A new session of the same task | Record a calibration first, `dataproc calibrate block_raise <dir>` (→ `block_raise_2`), then `dataproc run block_raise_2 <first> <last>` |
| More demos for the same session, same setup | Record, then `dataproc run <folder> <first new> <last new>` |
| Gripper, tags or camera mount changed | That's a new session: calibrate again |
| A different task | `dataproc calibrate <other_task> <dir>`, then run its range |
| ORB-SLAM3 was missing (exit 3) | Build it, then the same `run` command; it resumes from SLAM |
| Fixed something after failures | `dataproc run <folder> <first> <last> --retry-failed` |
| Rig values changed (`~/data/rigs/rig_pi_v1.yaml`) | `dataproc run <folder> <first> <last> --force`, then package again with `--overwrite` |
| A new training file with more sessions | `dataproc package <new name> <folder> <folder> ...` |
