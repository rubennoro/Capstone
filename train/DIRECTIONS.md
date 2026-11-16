# train/ directions: Ruben's configuration

How to go from recordings processed on the Jetson to a trained policy on the laptop. This file applies to
**Ruben's configuration**, which is already set up: the laptop `ruben-XPS-15-9520`, the Jetson
`capstone-gripper`, and the Explorer account `noroian.r`. To set up a new machine or account, use
[SETUP.md](SETUP.md). Script details are in [README.md](README.md), and the analysis tools in
[analysis/README.md](analysis/README.md).

```
Jetson (records + processes)  ──pull──▶  laptop ~/Capstone/data/<folder>/  ──upload──▶  Explorer cluster (GPU)
                                          ▲                                               │ submit → Slurm job
                                          └────────────────── fetch (policy) ◀────────────┘
```

All commands run **on the laptop, from the repo root** (`~/Capstone`), unless they start with `ssh`.

## Ruben's configuration (verified 2026-11-15)

| What | Value |
|---|---|
| Jetson | `capstone-gripper@129.10.156.38`, key login. Set in `train/config.local.env` (ignored by git). dataproc data in `~/data`, venv `~/envs/env_dataproc`. |
| Cluster | `noroian.r@login.explorer.northeastern.edu` (jobs) and `noroian.r@xfer.discovery.neu.edu` (transfers), with the Research Computing key set in `~/.ssh/config` |
| GPU jobs | H200 on partition `gpu` (8 h) for full runs and `gpu-short` (2 h) for smoke runs; conda env `bp` |
| Everything else | the defaults in `train/config.env` |

**If something stops working:**
- **The Jetson can't be reached:** its IP may have changed. Run `hostname -I` on the Jetson and put the new address
  in `train/config.local.env` (`JETSON_HOST="capstone-gripper@<new-ip>"`).
- **`Permission denied (publickey)` from the cluster:** the Research Computing key isn't being used. See
  [SETUP.md](SETUP.md) step 3.
- **You want to change a setting** (time limit, epochs, ...): add the line to `train/config.local.env`. See
  [SETUP.md](SETUP.md) step 2.

**Still open:**
- The Jetson has **no trainable episodes yet**: every episode in `eraser_pick_1` and `eraser_pick_2` failed at SLAM.
  Start below once `dataproc status <folder>` shows exported episodes (step 0).
- *(Optional, only for offline eval)* The cluster clone is at `06511cd` and needs a pull:
  `ssh noroian.r@login.explorer.northeastern.edu 'git -C CapstoneProject/Capstone pull --ff-only'`.

---

## Run order: every new dataset

The examples use the Jetson task folder **`eraser_pick_2`** and a new dataset name **`ep2-v1`**. A dataset name is
used once and never reused, so pick a new one for every package.

**0. See what's on the Jetson.** Get the task folder name and check that its episodes processed:

```bash
ssh capstone-gripper@129.10.156.38 'ls data/tasks'
ssh capstone-gripper@129.10.156.38 'source ~/envs/env_dataproc/bin/activate && dataproc status eraser_pick_2'
```

The first line of the status says `N episodes, M exported`. Go on only if **M is greater than 0**:

- **`0 exported` and every entry under `failed/`** (for example `slam failed slam_error`): nothing can be trained
  yet, and step 1 would stop with an error. The `reason.json` inside each `failed/<episode>/` says why. Fix the
  processing on the Jetson first (`rnoro/data-work`, `data-processing/README.md`), then come back.
  - "orbslam3: never initialised (0 keyframes...)" means SLAM never locked on. Each clip needs about 2 s of clear
    movement at the start, in a textured scene.
- **How many episodes:** a smoke run works with a few. A real run needs about 40 or more (the first iPhone run used
  45). 5% of the episodes per task are held out for validation.
- **One task = one label.** The label is the name given to `dataproc calibrate`. `eraser_pick_1` and
  `eraser_pick_2` have different labels, so they are two different tasks to the trainer. To add more recordings to
  a task, run `dataproc run` on the **same folder** again; new episodes are numbered after the existing ones.
  `train/` packages one folder per dataset.

**1. Pull the data to the laptop.** This packages the dataset on the Jetson and copies everything into
`data/eraser_pick_2/`. Use `--dataset ep2-v1` instead of `--package` if it is already packaged. Drop `--no-raw` to
copy the raw recordings too (about 300 MB per episode; check your free space).

```bash
train/pull.sh eraser_pick_2 --package ep2-v1 --no-raw
```

**2. Check the dataset before using GPU time:** episode counts, failures, lengths, and the estimated training time.

```bash
python3 train/analysis/dataset_report.py eraser_pick_2 --dataset ep2-v1
```

**3. Upload it to the cluster.**

```bash
train/upload.sh eraser_pick_2 ep2-v1
```

**4. Submit training.** This runs a 3-epoch smoke run first. The full run starts only if the smoke run succeeds.
Note the run names it prints.

```bash
train/submit.sh eraser_pick_2 ep2-v1 --smoke-first --note "first rig data"
```

Steps 1, 3 and 4 in one command: `train/go.sh eraser_pick_2 ep2-v1 --package --no-raw --note "first rig data"`.

**5. Watch it.** `--wait` checks every 5 minutes until the run is ok, failed or timeout. A full 60-epoch run on about
40 episodes takes about 2 h.

```bash
train/status.sh eraser_pick_2 --wait
```

**6. Fetch the policy** into `data/eraser_pick_2/policy/<run>/`. The checkpoint is about 2.5 GB.

```bash
train/fetch.sh eraser_pick_2 run_001_2026-11-14
```

The policy file is `data/eraser_pick_2/policy/run_001_2026-11-14/checkpoints/top1_*.ckpt`.

**7. Analyse it.**

```bash
python3 train/analysis/run_report.py eraser_pick_2 run_001
python3 train/analysis/compare_runs.py eraser_pick_2 --results-dir results/block_raise --no-smoke

# offline eval (error in mm/deg per demo): runs as a GPU job, then fetch it and report
train/analysis/eval.sh eraser_pick_2 run_001_2026-11-14 --ckpt top1 --per-demo 8
train/status.sh eraser_pick_2 run_001_2026-11-14        # repeat until the "top1:" line says ok
train/fetch.sh eraser_pick_2 run_001_2026-11-14 --no-checkpoints
python3 train/analysis/eval_report.py eraser_pick_2 run_001
```

`status.sh` prints one line per eval (for example `top1: ok`). Don't use `--wait` here: it only waits for the
training run, which has already finished, so it returns straight away. The fetch then brings back `eval/top1/`.

**8. Clean up on the cluster** once a run is fetched. Runs count against the home quota at
`~/CapstoneProject/policies/<label>/<run>/`.

---

## Where things end up

| What | Laptop | Cluster |
|---|---|---|
| Processed episodes + calibration | `data/<folder>/` (copy of the Jetson's `~/data/tasks/<folder>/`) | not uploaded |
| Training input (one file) | `data/<folder>/package/<dataset>/<dataset>.zarr.zip` + `package.json` | `~/CapstoneProject/datasets/<label>/<dataset>/` |
| Trained policy + logs, plots, wandb | `data/<folder>/policy/<run>/` | `~/CapstoneProject/policies/<label>/<run>/` |
| Analysis output | `data/<folder>/analysis/` and `data/<folder>/policy/<run>/analysis/` | none |
| Job logs | none (`status.sh` shows the tail) | `~/CapstoneProject/logs/` |

`data/` is ignored by git. If something goes wrong, see the Troubleshooting section of [README.md](README.md).
