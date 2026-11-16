# train/ setup: one time, on a new machine

Use this only to set up `train/` on a laptop, Jetson or cluster account that isn't configured yet. Ruben's laptop is
already set up: for running training, go to [DIRECTIONS.md](DIRECTIONS.md).

When you're done here, the laptop can ssh into the Jetson and the Explorer cluster without a password, and
`train/config.env` (plus your `train/config.local.env`) points at the right machines.

All commands run **on the laptop, from the repo root** (`~/Capstone`), unless they start with `ssh`. In this file
`<jetson-user>@<jetson-ip>` means your Jetson login and `<username>` means your Explorer username.

---

## 1. What you need first

| Need | How to check |
|---|---|
| The repo on branch `rnoro/training` (`train/` exists only there) | `git -C ~/Capstone branch --show-current` |
| A Jetson with `dataproc` installed that has processed a task (`dataproc calibrate`, then `dataproc run`). The procedure is in `data-processing/README.md` on branch `rnoro/data-work`. | `ssh <jetson-user>@<jetson-ip> 'ls data/tasks'` lists your task folder |
| A Northeastern Explorer account with GPU access | you can log in to ood.explorer.northeastern.edu |
| On the cluster: a repo clone at `~/CapstoneProject/Capstone` (branch `main`) and the conda env `bp` | see step 4 |
| On the laptop: bash, ssh, rsync, and python3 with numpy, matplotlib, pandas and pyyaml (Anaconda has them) | `python3 -c "import numpy, matplotlib, pandas, yaml"` |

## 2. Fill in the settings

All settings are in **`train/config.env`**. Every line has a comment saying what it is for, plus an example. If a
script needs a value that is still a `<...>` placeholder, it stops and names the setting.

**You must fill:**

| Setting | What to put | Example |
|---|---|---|
| `JETSON_HOST` | How you ssh into the Jetson: `user@ip` or a `Host` alias from `~/.ssh/config` | `capstone-gripper@129.10.156.38` |

Use the Jetson's **IP address** (`hostname -I` on the Jetson). The name in the Jetson's terminal prompt usually
doesn't work from the laptop: you get "Temporary failure in name resolution".

**Check these. The defaults match the project's machines:**

| Setting | Default | Change it if |
|---|---|---|
| `JETSON_SSH_OPTS` | empty | the Jetson needs a port or key: `-p 2222 -i ~/.ssh/id_jetson` |
| `JETSON_DATAPROC_ROOT` | `data` (= `~/data` on the Jetson) | `DATAPROC_ROOT` is set to something else on the Jetson |
| `JETSON_ACTIVATE` | `source ~/envs/env_dataproc/bin/activate` | dataproc lives in another venv |
| `CLUSTER_USER` | `noroian.r` | you are not Ruben (the login and transfer hosts follow from it) |
| `SLURM_ACCOUNT` | empty (no `--account`) | your cluster allocation requires an account |
| `SLURM_GRES`, `SLURM_TIME` | `gpu:h200:1`, `08:00:00` | you want another GPU or time limit (epochs are sized to fit the time limit) |
| `DEFAULT_EPOCHS` | `60` | you want longer or shorter full runs by default |
| `CLUSTER_RUNS` | `CapstoneProject/policies` | the home quota is too small (one run ≈ 2.5 GB) |

### How to change a setting (step by step)

There are two ways. **Way B is recommended**, especially for `JETSON_HOST`, because it keeps the Jetson's address
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

Nothing is copied or changed. The printed lines that start with `[dry-run] rsync` should contain your value, for
example `capstone-gripper@129.10.156.38:data/tasks/eraser_pick_2/`. If you see
`error: JETSON_HOST is not set (value: '<user@jetson-host ...>')`, the scripts did not pick up your value. Check the
file name (`config.local.env`, inside `train/`), the spelling of `JETSON_HOST`, and the quotes, then save and run
the command again.

**To undo a change:** delete that line from `config.local.env` and save; the default from `config.env` applies
again. To undo all of them, delete the whole `config.local.env` file.

## 3. ssh keys

The scripts can't type passwords, so both machines need key login.

**Jetson.** This asks for the Jetson password one time and installs your laptop key:

```bash
ssh-copy-id <jetson-user>@<jetson-ip>
```

The Jetson's IP on the Northeastern network can change when it reconnects. If the scripts suddenly can't reach
it, run `hostname -I` on the Jetson and update `JETSON_HOST`.

**Explorer cluster.** Since 2026-11-06, Explorer only accepts the key Research Computing generated for you
([RC instructions](https://rc.northeastern.edu/2026/10/23/ssh-instructions-for-explorer/)). Passwords, older keys
and `ssh-copy-id` no longer work. The symptom is `Permission denied (publickey)`.

1. In a browser, log in to **ood.explorer.northeastern.edu**, go to **Files → Home Directory**, turn on
   **Show dot files**, and open **`.ssh`**.
2. Download the two files named like `<username>-2026XXXXTXXXXXXZ` and `<username>-2026XXXXTXXXXXXZ.pub`. Don't
   delete anything in that folder. If the files aren't there, email rchelp@northeastern.edu.
3. On the laptop (use your real file name):
   ```bash
   mv ~/Downloads/<username>-2026* ~/.ssh/
   chmod 600 ~/.ssh/<username>-2026XXXXTXXXXXXZ
   chmod 644 ~/.ssh/<username>-2026XXXXTXXXXXXZ.pub
   ```
4. Add this block at the **very top** of `~/.ssh/config`. Ssh uses the first matching block, so this one wins over
   any older lines for these hosts:
   ```
   Host login.explorer.northeastern.edu xfer.discovery.neu.edu
       User <username>
       IdentityFile ~/.ssh/<username>-2026XXXXTXXXXXXZ
       IdentitiesOnly yes
   ```

## 4. Check every connection

Each command should succeed without asking for a password (a Duo prompt is fine). If one fails, fix it before
going on.

```bash
# Jetson: task folders are visible and dataproc runs (data/datasets appears after the first package)
ssh <jetson-user>@<jetson-ip> 'ls data/tasks data/datasets'
ssh <jetson-user>@<jetson-ip> 'source ~/envs/env_dataproc/bin/activate && dataproc --help | head -3'

# Cluster login node: project folder and clone are there
ssh <username>@login.explorer.northeastern.edu 'ls CapstoneProject && git -C CapstoneProject/Capstone log --oneline -1'

# Cluster transfer node: accepts commands (upload.sh checks sha256 there)
ssh <username>@xfer.discovery.neu.edu 'sha256sum --version | head -1'

# The bp conda env exists
ssh <username>@login.explorer.northeastern.edu 'module load anaconda3/2024.06 && conda env list | grep -w bp'
```

**Cluster clone version.** `train/analysis/eval.sh` needs the clone at commit `4be9982` or newer; training works
without it. If the `git log` above shows an older commit, update the clone. This changes files on the cluster:

```bash
ssh <username>@login.explorer.northeastern.edu 'git -C CapstoneProject/Capstone pull --ff-only'
```

## 5. Dry run

This prints every ssh, rsync and sbatch command a real run would send, and runs nothing remote:

```bash
train/go.sh <folder> test-dataset --package --no-raw --dry-run
```

If it ends without an error, setup is done. Continue with [DIRECTIONS.md](DIRECTIONS.md) to train.
