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
  Start below once `dataproc status <folder>` shows exported episodes (step 1).

The cluster clone is already up to date for the offline eval (checked 2026-11-15: `main` at `1e4dc64`, which
includes `controlSW/franka_deploy`; `robotmq` is in the `bp` env).

---

## One command: every step

When the data is good and time is short, `train/all.sh` runs steps 2 to 8 below in one go and waits for the cluster
jobs (about 2.5 h with the eval). Do steps 0 and 1 first, then start it **inside tmux** so it survives closing the
terminal or a dropped ssh session:

```bash
tmux new -s train                                     # later: Ctrl-b d to detach, `tmux attach -t train` to come back
cd ~/Capstone
export TASK_NAME=eraser_pick_2 DATASET_NAME=ep2-v1
train/all.sh $TASK_NAME $DATASET_NAME --package --no-raw --results-dir results/block_raise --note "first rig data"
```

- Add `--dry-run` first to see every command without running anything. `--no-eval` skips the offline eval;
  `--no-smoke` skips the smoke run.
- **When it ends**, the last lines say `DONE: every step ok`, `DONE with N problem(s)` (the policy is on the laptop,
  but a report or the eval failed), or `STOPPED at step '...'` with the reason and what to do next.
- **Everything is logged** in `data/$TASK_NAME/logs/`: `all_<time>_<dataset>.log` has the full output, and
  `all_<time>_<dataset>.steps` has one line per step with its result and reason.
- **If it stops after submitting** (Ctrl-C, laptop asleep, a failed fetch), the cluster jobs keep running. Continue
  with the command it printed, `train/all.sh $TASK_NAME --run <full run name>`. This starts again at the wait.
- **The laptop must stay awake and online** while it waits: a suspended laptop pauses the checks. The jobs carry on
  either way.

---

## Run order: every new dataset

**0. Name the task (do this first, in every new terminal).** The commands below don't contain a task name. They
use variables you set once, so the same commands work for any task:

```bash
cd ~/Capstone
export TASK_NAME=eraser_pick_2      # the Jetson task folder (ssh capstone-gripper@129.10.156.38 'ls data/tasks')
export DATASET_NAME=ep2-v1          # a NEW name for this package: letters, digits, - and _ only; never reuse one
```

- `export` only lasts for the terminal window you typed it in. If you open a new terminal, run both lines again.
- **To train a different task**, set new values and repeat the steps, for example
  `export TASK_NAME=eraser_pick_1` and `export DATASET_NAME=ep1-v1`. To work on two tasks at once, use two terminal
  windows, one per task.
- To see what's set: `echo $TASK_NAME $DATASET_NAME`. If a command complains about a missing argument, the
  variables are probably empty: you're in a new terminal, so run the two `export` lines again.

**1. Check the task on the Jetson** to make sure its episodes processed:

```bash
ssh capstone-gripper@129.10.156.38 "source ~/envs/env_dataproc/bin/activate && dataproc status $TASK_NAME"
```

The first line of the status says `N episodes, M exported`. Go on only if **M is greater than 0**:

- **`0 exported` and every entry under `failed/`** (for example `slam failed slam_error`): nothing can be trained
  yet, and step 2 would stop with an error. The `reason.json` inside each `failed/<episode>/` says why. Fix the
  processing on the Jetson first (`rnoro/data-work`, `data-processing/README.md`), then come back.
  - "orbslam3: never initialised (0 keyframes...)" means SLAM never locked on. Each clip needs about 2 s of clear
    movement at the start, in a textured scene.
- **How many episodes:** a smoke run works with a few. A real run needs about 40 or more (the first iPhone run used
  45). 5% of the episodes per task are held out for validation.
- **One task = one label.** The label is the name given to `dataproc calibrate`. `eraser_pick_1` and
  `eraser_pick_2` have different labels, so they are two different tasks to the trainer. To add more recordings to
  a task, run `dataproc run` on the **same folder** again; new episodes are numbered after the existing ones.
  `train/` packages one folder per dataset.

**2. Pull the data to the laptop.** This packages the dataset on the Jetson and copies everything into
`data/$TASK_NAME/`. If the dataset is already packaged, use `--dataset $DATASET_NAME` instead of `--package`. Drop
`--no-raw` to copy the raw recordings too (about 300 MB per episode; check your free space).

```bash
train/pull.sh $TASK_NAME --package $DATASET_NAME --no-raw
```

**3. Check the dataset before using GPU time:** episode counts, failures, lengths, and the estimated training time.

```bash
python3 train/analysis/dataset_report.py $TASK_NAME --dataset $DATASET_NAME
```

**4. Upload it to the cluster.**

```bash
train/upload.sh $TASK_NAME $DATASET_NAME
```

**5. Submit training.** This runs a 3-epoch smoke run first. The full run starts only if the smoke run succeeds.

```bash
train/submit.sh $TASK_NAME $DATASET_NAME --smoke-first --note "first rig data"
```

It prints two run names: the smoke run (ending in `_smoke`) and the full run. Save the full run's name for the
next steps:

```bash
export RUN_NAME=run_002_2026-11-16        # copy the full run's name from the submit output
```

Steps 2, 4 and 5 in one command: `train/go.sh $TASK_NAME $DATASET_NAME --package --no-raw --note "first rig data"`.
Steps 2 to 8 in one command: `train/all.sh` (see "One command: every step" above).

**6. Watch it.** `--wait` checks every 5 minutes until the run is ok, failed or timeout. A full 60-epoch run on about
40 episodes takes about 2 h.

```bash
train/status.sh $TASK_NAME $RUN_NAME --wait
```

**7. Fetch the policy** into `data/$TASK_NAME/policy/$RUN_NAME/`. The checkpoint is about 2.5 GB.

```bash
train/fetch.sh $TASK_NAME $RUN_NAME
ls data/$TASK_NAME/policy/$RUN_NAME/checkpoints/      # the policy file: top1_epoch=...ckpt
```

**8. Analyse it.**

```bash
python3 train/analysis/run_report.py $TASK_NAME $RUN_NAME
python3 train/analysis/compare_runs.py $TASK_NAME --results-dir results/block_raise --no-smoke

# offline eval (error in mm/deg per demo): runs as a GPU job, then fetch it and report
train/analysis/eval.sh $TASK_NAME $RUN_NAME --ckpt top1 --per-demo 8
train/status.sh $TASK_NAME $RUN_NAME --eval top1 --wait     # until the eval job is ok, failed or timeout
train/fetch.sh $TASK_NAME $RUN_NAME --no-checkpoints
python3 train/analysis/eval_report.py $TASK_NAME $RUN_NAME
```

`--eval top1` follows the eval job instead of the training run. If it ends with `STATUS=failed`, the `message:` line
and the log tail say why. The fetch then brings back `eval/top1/`, including `eval_job.json`.

**9. Clean up on the cluster** once a run is fetched. Runs count against the home quota at
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
