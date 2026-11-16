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

> **Status on Ruben's laptop (checked 2026-11-15).** Part 1 is done. You can skip it unless the IP or keys change.
>
> **Done:**
> - `JETSON_HOST="capstone-gripper@129.10.156.38"` is in `train/config.local.env`, and key login to the Jetson works.
> - The new Explorer key is installed and works for both the login and transfer nodes.
> - The `bp` env, the H200 partitions and the Jetson's dataproc defaults are all confirmed.
>
> **Still open:**
> - *(Optional, only for offline eval)* The cluster clone is at `06511cd`; update it as described at the end of 1.3.
> - The Jetson has no trainable episodes yet: every episode in `eraser_pick_1` and `eraser_pick_2` failed at SLAM.
>   Start Part 2 once `dataproc status <folder>` shows exported episodes (see Part 2, step 0).

---

## Part 1: one-time setup

### 1.1 What you need first

| Need | How to check |
|---|---|
| The repo on branch `rnoro/training` (`train/` exists only there) | `git -C ~/Capstone branch --show-current` |
| A Jetson with `dataproc` installed that has already processed a task: `dataproc calibrate`, then `dataproc run`. That procedure is in `data-processing/README.md` on branch `rnoro/data-work`. | `ssh <jetson> 'ls data/tasks'` lists your task folder |
| A Northeastern Explorer account with GPU access, and **your Research Computing SSH key** on the laptop (required since 2026-11-06; see 1.3) | `ssh noroian.r@login.explorer.northeastern.edu hostname` prints a node name |
| On the cluster: the repo clone `~/CapstoneProject/Capstone` and the conda env `bp` (both already set up for this project) | see 1.3 |
| On the laptop: bash, ssh, rsync, and python3 with numpy, matplotlib, pandas and pyyaml (Anaconda has them) | `python3 -c "import numpy, matplotlib, pandas, yaml"` |

### 1.2 Fill in the blanks

> **Already done on Ruben's laptop:** `train/config.local.env` contains
> `JETSON_HOST="capstone-gripper@129.10.156.38"`, and all the defaults below were checked against the real Jetson
> and cluster on 2026-11-15. Nothing to fill in. Come back here only for a new laptop, a new Jetson IP, or if you
> want different defaults.

All settings are in **`train/config.env`**. Every line has a comment saying what it is for, plus an example. If a
script needs a value that is still a `<...>` placeholder, it stops and names the setting.

**You must fill:**

| Setting | What to put | Example |
|---|---|---|
| `JETSON_HOST` | How you ssh into the Jetson: `user@ip` or a `Host` alias from `~/.ssh/config` | `capstone-gripper@129.10.156.38` |

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

#### How to change a setting (step by step)

There are two ways. **Way B is recommended**, especially for `JETSON_HOST`, because it keeps your Jetson's address
out of git.

**Way A: edit `config.env` directly.**

1. In VS Code's file list on the left, open the `train` folder and click `config.env`.
2. Find the line for the setting, for example `JETSON_HOST="<user@jetson-host or ~/.ssh/config alias>"`.
3. Replace only the text **between the two double quotes**. Keep the quotes, and don't add spaces around the `=`.
   Before: `JETSON_HOST="<user@jetson-host or ~/.ssh/config alias>"`
   After:  `JETSON_HOST="capstone-gripper@129.10.156.38"`
4. Save with **Ctrl+S**.

Downside: `config.env` is tracked by git, so this change gets committed the next time someone commits `train/`.

**Way B: put your changes in a separate file, `config.local.env`.**

The scripts read `config.env` first, then `config.local.env`. Anything you write in the second file replaces the
default, and git ignores that file, so it never gets committed.

1. In VS Code's file list on the left, **right-click the `train` folder** and choose **New File...**
2. Name it exactly `config.local.env` and press **Enter**. The empty file opens.
3. For every setting you want to change, copy its line from `config.env` into the new file and change the value
   between the quotes. Copy only the lines you are changing, one per line. For example:
   ```
   JETSON_HOST="capstone-gripper@129.10.156.38"
   SLURM_TIME="04:00:00"
   ```
   Same rules as in Way A: keep the double quotes, no spaces around `=`, and the name on the left must be spelled
   exactly as in `config.env` (capital letters and underscores).
4. Save with **Ctrl+S**.

The file sits in your laptop's copy of `train/`. When you submit a job, it is also copied to the cluster
automatically.

**Check that it worked.** In the VS Code terminal (**Terminal → New Terminal**), run:

```bash
cd ~/Capstone
train/pull.sh eraser_pick_2 --dataset test --dry-run
```

Nothing is copied or changed. Look at the printed lines that start with `[dry-run] rsync`: they should contain your
value, for example `capstone-gripper@129.10.156.38:data/tasks/eraser_pick_2/`. If you see
`error: JETSON_HOST is not set (value: '<user@jetson-host ...>')`, the scripts did not pick up your value: check the
file name (`config.local.env`, inside `train/`), the spelling of `JETSON_HOST`, and the quotes, then save and run
the command again.

**To undo a change:** delete that line from `config.local.env` and save. The default from `config.env` applies again.
To undo all of them, delete the whole `config.local.env` file.

### 1.3 Check every connection once

The scripts can't type passwords, so both machines need key login.

**Jetson key (once).** This asks for the Jetson password one time and installs your laptop key:

```bash
ssh-copy-id capstone-gripper@129.10.156.38
```

`129.10.156.38` is an address on the Northeastern network and can change when the Jetson reconnects. If the scripts
suddenly can't reach the Jetson, run `hostname -I` on the Jetson and put the new address in
`train/config.local.env`. The name in the Jetson's terminal prompt (`capstone-gripper`) does not work from the
laptop: you get "Temporary failure in name resolution".

**Cluster key (once).** Since 2026-11-06, Explorer accepts only the key Research Computing generated for you
([RC instructions](https://rc.northeastern.edu/2026/10/23/ssh-instructions-for-explorer/)). Passwords, older keys
and `ssh-copy-id` no longer work. The error is `Permission denied (publickey)`.

1. In a browser, log in to **ood.explorer.northeastern.edu**, then go to **Files → Home Directory**, turn on
   **Show dot files**, and open **`.ssh`**.
2. Download the two files named like `noroian.r-2026XXXXTXXXXXXZ` and `noroian.r-2026XXXXTXXXXXXZ.pub`. Don't delete
   anything in that folder. If the files aren't there, email rchelp@northeastern.edu.
3. On the laptop, use your real file name in place of `noroian.r-2026XXXXTXXXXXXZ`:
   ```bash
   mv ~/Downloads/noroian.r-2026* ~/.ssh/
   chmod 600 ~/.ssh/noroian.r-2026XXXXTXXXXXXZ
   chmod 644 ~/.ssh/noroian.r-2026XXXXTXXXXXXZ.pub
   ```
4. Add this block at the **very top** of `~/.ssh/config`. Ssh uses the first matching block, so it wins over any
   older lines for these hosts:
   ```
   Host login.explorer.northeastern.edu xfer.discovery.neu.edu
       User noroian.r
       IdentityFile ~/.ssh/noroian.r-2026XXXXTXXXXXXZ
       IdentitiesOnly yes
   ```

**Now check every connection.** Each command should succeed without asking for a password; Duo is fine. If one
fails, fix it before going on.

```bash
# Jetson: task folders and packaged datasets are visible, and dataproc runs
ssh <JETSON_HOST> 'ls data/tasks data/datasets'      # data/datasets only exists after the first package
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
ssh noroian.r@login.explorer.northeastern.edu 'git -C CapstoneProject/Capstone pull --ff-only'
```

### 1.4 Dry run

This prints every ssh, rsync and sbatch command the real run would send, and runs nothing remote:

```bash
train/go.sh eraser_pick_2 ep2-test --package --no-raw --dry-run
```

If it ends without an error, setup is done.

---

## Part 2: every new dataset, in order

The examples use the Jetson task folder **`eraser_pick_2`** and a new dataset name **`ep2-v1`**. A dataset name is
used once and never reused, so pick a new one for every package.

**0. See what's on the Jetson.** Get the task folder name and check that its episodes processed:

```bash
ssh <JETSON_HOST> 'ls data/tasks'
ssh <JETSON_HOST> 'source ~/envs/env_dataproc/bin/activate && dataproc status eraser_pick_2'
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
