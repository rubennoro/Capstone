# train/ directions: day 0

This guide takes you from recordings processed on the Jetson to a trained policy on your laptop. Part 1 is one-time
setup: what you need, which blanks to fill, and how to check each connection. Part 2 is the order you run things in
for every new dataset. Details for each script are in [README.md](README.md), and the analysis tools are described
in [analysis/README.md](analysis/README.md).

```
Jetson (records + processes)  ──pull──▶  laptop ~/Capstone/data/<folder>/  ──upload──▶  Explorer cluster (GPU)
                                          ▲                                               │ submit → Slurm job
                                          └────────────────── fetch (policy) ◀────────────┘
```

All commands run **on the laptop, from the repo root** (`~/Capstone`), unless they start with `ssh`.

---

## Part 1: one-time setup

### 1.1 What you need first

| Need | How to check |
|---|---|
| The repo on branch `rnoro/training` (`train/` exists only there) | `git -C ~/Capstone branch --show-current` |
| A Jetson with `dataproc` installed that has already processed a task: `dataproc calibrate`, then `dataproc run`. That procedure is in `data-processing/README.md` on branch `rnoro/data-work`. | `ssh <jetson> 'ls data/tasks'` lists your task folder |
| A Northeastern Explorer account with GPU access | you can log in to `login.explorer.northeastern.edu` |
| On the cluster: the repo clone `~/CapstoneProject/Capstone` and the conda env `bp` (both already set up for this project) | see 1.3 |
| On the laptop: bash, ssh, rsync, and python3 with numpy, matplotlib, pandas and pyyaml (Anaconda has them) | `python3 -c "import numpy, matplotlib, pandas, yaml"` |

### 1.2 Fill in the blanks

All settings are in **`train/config.env`**. Every line has a comment saying what it is for, plus an example. If a
script needs a value that is still a `<...>` placeholder, it stops and names the setting.

**You must fill:**

| Setting | What to put | Example |
|---|---|---|
| `JETSON_HOST` | How you ssh into the Jetson: `user@ip` or a `Host` alias from `~/.ssh/config` | `ruben@192.168.1.50` |

**Check these; the defaults match the current setup:**

| Setting | Default | Change it if |
|---|---|---|
| `JETSON_SSH_OPTS` | empty | the Jetson needs a port or key: `-p 2222 -i ~/.ssh/id_jetson` |
| `JETSON_DATAPROC_ROOT` | `data` (= `~/data` on the Jetson) | `DATAPROC_ROOT` is set to something else on the Jetson |
| `JETSON_ACTIVATE` | `source ~/envs/env_dataproc/bin/activate` | dataproc lives in another venv |
| `CLUSTER_USER` | `noroian.r` | someone else runs it (the login and transfer hosts follow from it) |
| `SLURM_ACCOUNT` | empty (no `--account`) | your cluster allocation requires an account |
| `SLURM_GRES`, `SLURM_TIME` | `gpu:h200:1`, `08:00:00` | you want another GPU or time limit (epochs are sized to fit the time limit) |
| `DEFAULT_EPOCHS` | `60` | you want longer or shorter full runs by default |
| `CLUSTER_RUNS` | `CapstoneProject/policies` | the home quota is too small (one run ≈ 2.5 GB) |

To change a setting just for yourself without editing the tracked file, put the line in
`train/config.local.env` (same syntax, ignored by git, read after `config.env`), for example
`JETSON_HOST="jetson"`.

### 1.3 Check every connection once

Run these by hand. Each one should succeed without asking for a password; Duo is fine. If one fails, fix it before
going on.

```bash
# ssh keys: run once per host if you get a password prompt
ssh-copy-id <JETSON_HOST>
ssh-copy-id noroian.r@login.explorer.northeastern.edu

# Jetson: task folders and packaged datasets are visible, and dataproc runs
ssh <JETSON_HOST> 'ls data/tasks data/datasets'
ssh <JETSON_HOST> 'source ~/envs/env_dataproc/bin/activate && dataproc --help | head -3'

# Cluster login node: project folder and clone are there
ssh noroian.r@login.explorer.northeastern.edu 'ls CapstoneProject && git -C CapstoneProject/Capstone log --oneline -1'

# Cluster transfer node: accepts commands (upload.sh checks sha256 there)
ssh noroian.r@xfer.discovery.neu.edu 'sha256sum --version | head -1'

# The bp conda env exists
ssh noroian.r@login.explorer.northeastern.edu 'module load anaconda3/2024.06 && conda env list | grep -w bp'
```

**Cluster clone version.** `train/analysis/eval.sh` needs the clone at commit `4be9982` or newer. Training works
without it. If the `git log` above shows an older commit, update the clone (this changes files on the cluster):

```bash
ssh noroian.r@login.explorer.northeastern.edu 'git -C CapstoneProject/Capstone pull'
```

### 1.4 Dry run

This prints every ssh, rsync and sbatch command the real run would send, and runs nothing remote:

```bash
train/go.sh block_raise br-test --package --no-raw --dry-run
```

If it ends without an error, setup is done.

---

## Part 2: every new dataset, in order

The examples use the Jetson task folder **`block_raise`** and a new dataset name **`br-nov14`**. A dataset name is
used once and never reused, so pick a new one for every package.

**0. See what's on the Jetson.** Get the task folder name and check that its episodes processed:

```bash
ssh <JETSON_HOST> 'ls data/tasks'
ssh <JETSON_HOST> 'source ~/envs/env_dataproc/bin/activate && dataproc status block_raise'
```

**1. Pull the data to the laptop.** This packages the dataset on the Jetson and copies everything into
`data/block_raise/`. Use `--dataset br-nov14` instead of `--package` if it is already packaged. Drop `--no-raw` to
copy the raw recordings too (about 300 MB per episode; check your free space).

```bash
train/pull.sh block_raise --package br-nov14 --no-raw
```

**2. Check the dataset before using GPU time:** episode counts, failures, lengths, and the estimated training time.

```bash
python3 train/analysis/dataset_report.py block_raise --dataset br-nov14
```

**3. Upload it to the cluster.**

```bash
train/upload.sh block_raise br-nov14
```

**4. Submit training.** This runs a 3-epoch smoke run first. The full run starts only if the smoke run succeeds.
Note the run names it prints.

```bash
train/submit.sh block_raise br-nov14 --smoke-first --note "first rig data"
```

Steps 1, 3 and 4 in one command: `train/go.sh block_raise br-nov14 --package --no-raw --note "first rig data"`.

**5. Watch it.** `--wait` checks every 5 minutes until the run is ok, failed or timeout. A full 60-epoch run on about
40 episodes takes about 2 h.

```bash
train/status.sh block_raise --wait
```

**6. Fetch the policy** into `data/block_raise/policy/<run>/`. The checkpoint is about 2.5 GB.

```bash
train/fetch.sh block_raise run_001_2026-11-14
```

The policy file is `data/block_raise/policy/run_001_2026-11-14/checkpoints/top1_*.ckpt`.

**7. Analyse it.**

```bash
python3 train/analysis/run_report.py block_raise run_001
python3 train/analysis/compare_runs.py block_raise --results-dir results/block_raise --no-smoke

# offline eval (error in mm/deg per demo): runs as a GPU job, then fetch it and report
train/analysis/eval.sh block_raise run_001_2026-11-14 --ckpt top1 --per-demo 8
train/status.sh block_raise run_001_2026-11-14        # repeat until the "top1:" line says ok
train/fetch.sh block_raise run_001_2026-11-14 --no-checkpoints
python3 train/analysis/eval_report.py block_raise run_001
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
