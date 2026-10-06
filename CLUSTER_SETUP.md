# Cluster Setup & Training Log

Running notes for training the behavior prompting policy (`umi_policy_dunetp`) on the
Northeastern Explorer cluster: first on the provided example data, then on our own iPhUMI data.

- **Cluster login:** `ssh noroian.r@login.explorer.northeastern.edu`
- **Transfer node (use for big copies):** `noroian.r@xfer.discovery.neu.edu`
- **Project root on cluster:** `~/CapstoneProject/Capstone` (a clone of `git@github.com:rubennoro/Capstone.git`)

> Paths below are relative to the repo root. `behavior_prompting/` is the outer vendored repo;
> the Python package (and `train_network/`, `scripts/`) lives one level deeper in
> `behavior_prompting/behavior_prompting/`.

---

## What training requires (summary of the docs)

Sources: `behavior_prompting/docs/iphumi.md`, `behavior_prompting/docs/laundry_folding.md`,
`iPhUMI/docs/usage.md`, configs in `behavior_prompting/behavior_prompting/train_network/config/`.

- **Policy:** behavior prompting = `--config-name=umi_policy_dunetp` (single arm).
  Defaults: batch 70, 140 epochs, bf16, checkpoint every 10 epochs. Needs **~46 GB GPU memory**.
  It uses task `umi_prompt` (`sample_type: task`), so **the dataset must contain task labels**.
- **Single-arm configs read `camera_right_*` / `gripper_right_*` keys** (`config/task/umi.yaml`).
- **Example data:** HF dataset `austinpatel/iphumi_example_zarr_datasets`;
  single-arm file is `replay_buffer_test-data-042926-single.zarr(.zip)`.
- **Training expects `.zarr`, not `.zarr.zip`** → unzip with `behavior_prompting/behavior_prompting/scripts/unzip_zarr.py`.
- **Environment:** `behavior_prompting/environment.yml` → conda env `bp` (Python 3.10, torch 2.7.1 cu118).
  It pip-installs local packages (`-e .`, `-e deps/icrt`, `-e deps/LIBERO`) with paths relative to
  `behavior_prompting/`, so `conda env create` must be run **from inside `behavior_prompting/`**.
- **Train command shape:**
  ```bash
  cd behavior_prompting/behavior_prompting/train_network
  accelerate launch --num_processes=<N_GPUS> train.py \
    --config-name=umi_policy_dunetp \
    task.dataset.dataset_path=datasets/umi/iphumi_example_zarr_datasets/replay_buffer_test-data-042926-single.zarr \
    exp_name="example_data_single_arm_bp" \
    group_tag="initial_experiments" \
    logging.mode=offline
  ```
  Per-experiment overrides go in a modifiers file (e.g. `+modifiers/umi/laundry_folding=task_dunetp`)
  rather than editing base configs.

---

## Steps completed (2026-10-05)

### 1. Inspected `~/CapstoneProject` on the cluster
`~/CapstoneProject` existed and was **empty**, so there was no old `Capstone` folder to delete.

### 2. Cloned the repo
```bash
cd ~/CapstoneProject && git clone git@github.com:rubennoro/Capstone.git
```
- Result: `~/CapstoneProject/Capstone` at commit `620c7b6` (matches local `main`).
- Checking out about 6,400 files takes a few minutes because the home filesystem is networked and the vendored deps contain many small files.

### 3. Checked storage
- Home usage was **8.7 GB** before copying data. `quota` doesn't work on this storage system, but my understanding is that the per-user home limit is about 75 GB, which is enough for the data, the conda env (~10 GB) and checkpoints.
- `/scratch/noroian.r` exists. Scratch is purged periodically, so only put regenerable things there, such as checkpoints if home gets tight.

### 4. Copied our custom data (`data/`, gitignored) to the cluster
- **Two attempts through the login node failed** (`rsync: Broken pipe`, exit 255), both after about 20–25 seconds and roughly 500 MB. **The login node kills long or heavy processes**, so don't do large transfers or heavy work there.
- Deleted the partial copy and copied from scratch **through the transfer node**, which worked at about 17 MB/s (≈5 min):
  ```bash
  rsync -a --partial --info=progress2,stats2 \
    -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
    /home/ruben/Capstone/data/ noroian.r@xfer.discovery.neu.edu:~/CapstoneProject/Capstone/data/
  ```
- Verified: **229 files / 5,432,472,296 bytes** on both ends.
- Location on cluster: `~/CapstoneProject/Capstone/data/raw/2026-10-01/`. It contains about 50 raw iPhone exports from session `firstreal`: demonstrations plus one gripper calibration, all from the **left** gripper, recorded with **no task labels**.

### 5. Surveyed conda and GPUs
- Conda is `anaconda3/2024.06`, already initialized by `~/.bashrc`; there's no `mamba`. `conda env create` in this version uses the libmamba solver, so it replaces `mamba env create`.
- Other modules include `miniconda3/24.11.1`, `miniconda3/25.9.1` and `cuda/12.1.1`, `12.3.0`, `12.8.0`, `13.2.0`.
- **`conda env list` was `Killed` on the login node**, so env creation has to run as a Slurm job.
- GPUs that fit the ~46 GB policy:

  | Partition         | GPUs        | Time limit |
  |-------------------|-------------|------------|
  | `gpu`             | H200, A100  | **8 h**    |
  | `gpu-short`       | H200, A100  | 2 h        |
  | `gpu-interactive` | H200, A100  | 2 h        |
  | `sharing`         | H100, H200, A100, L40S, ... | 1 h |

  Prefer **H200 (141 GB)**, because some A100s are only 40 GB. With the 8-hour limit, the full 140-epoch run will probably take several jobs that resume from checkpoints.
- At the time of this survey, neither the `bp` env nor the HF example dataset existed yet.

### 6. Submitted the `bp` env build (job `10838595`)
- Submitted with the sbatch script in step A below. It's running on node `c0613` in the `short` partition.
- At 25 minutes, conda had solved and verified the env and was on "Executing transaction". The pip step (torch, mujoco, robomimic, local `-e` installs) comes after that.
- Log: `~/CapstoneProject/logs/bp_env_10838595.out`.

- **The job failed after about 35 minutes.** Conda finished, but pip stopped with `ERROR: deps/icrt is not a valid editable requirement`.
  - Cause: upstream, `deps/icrt`, `deps/LIBERO` and `deps/real-env` are **git submodules**. Removing `behavior_prompting/.git` to vendor it into this monorepo left them as empty folders.
  - Fix (commit `d0298b9`): vendored all three at the commits upstream pins (listed in the commit message) and deleted `behavior_prompting/.gitmodules`.
  - Gotcha: the `data` rule in `behavior_prompting/.gitignore` silently ignored 3 source files (`icrt/data/{dataset,utils}.py`, `libero/configs/data/default.yaml`), so they had to be force-added with `git add -f`.
  - The failed build left a partial `bp` env, so the rebuild has to run `conda env remove -n bp -y` first.
- Pushed the fix to GitHub, then ran `git pull --ff-only` on the cluster through `xfer.discovery.neu.edu`. The cluster clone is now at `3c54bee`, with `deps/icrt` 19 MB, `deps/LIBERO` 424 MB and `deps/real-env` 458 KB.
- **Resubmitted the env build as job `10841186`** (2026-10-05). It uses the same sbatch script with `conda env remove -n bp -y || true` added before `conda env create`.

- **Job `10841186` succeeded.** The final check printed `2.7.1+cu126 12.6`. torch came from pip's default **CUDA 12.6** wheels rather than cu118, which is fine for the H200s; confirm `torch.cuda.is_available()` in the smoke test. The `bp` env now exists.

### 8. Submitted the example-data download and our data processing (2026-10-05)
- Job scripts now live in `scripts/cluster/`. They're copied with `scp` and submitted with `sbatch`, because pasting long heredocs into the terminal got line-wrapped and mangled (jobs `10847006` and `10847112` failed that way).
- **`10847436`** (`get_example_data.sbatch`, step B): the download took about 15 s (3 zips, about 350 MB in total). Unzipping the single-arm zarr (2,296 small files) runs at only about 1–2 files/s on the networked home filesystem.
- **`10848060`** (`build_iphumi_env.sbatch`): built the `iphumi` env in 45 min (`iphumi env OK 2.18.3 4.12.0`).
- **`10848071`** (`process_firstreal.sbatch block_raise`, step E): started after `10848060` succeeded (`--dependency=afterok`) and ran every step of E with task name **`block_raise`** (raising an eraser with one gripper).
  - Group copied 45 demos + 1 calibration. The labeling script tagged 45 demos. `align_validate` **passed 45, failed 0**.
  - **Dataset: `iPhUMI/python_package/iphumi/demonstration_processing/tmp_sessions/firstreal-single/replay_buffer_firstreal-single.zarr.zip`** (845 MB): 45 episodes, 37,971 frames, `block_raise` × 45, no error-correction demos.
  - With `sides=single` the pipeline **already names the single gripper `right`** (`camera_right_*`, `gripper_right_*`), even though it was recorded as `L`, so `rename_zarr_side.py` isn't needed.
  - **Cancelled during the unzip step** (`scancel`, after 54 min). Unzipping writes about 44,000 small files at about 4 files/s, which would take about 3 h. Training reads the zip through the LMDB cache instead (see 9), so the unzipped copy isn't needed. The optional `visualize` stage never ran.
- Logs: `~/CapstoneProject/logs/<job-name>_<jobid>.out`.

### 9. Smoke tests (2026-10-05)
- **The storage (VAST `/home` and `/scratch`) was very slow tonight:** about 1.3 MB/s home reads and 10 MB/s scratch writes, measured with `dd`. The first `import torch` from the new `bp` env took **about 30 min**, stuck in `D` state on `folio_wait`. Later imports were fast because the files were cached.
- **`10848996` (example data):** crashed after the imports. **LIBERO asks an interactive `(Y/N)` question on its first import** and got `EOFError` in batch mode. Fix in `train_smoke.sh`: `echo n | python -c "from libero.libero import get_libero_path"` before training, which creates `~/.libero/config.yaml`. Slurm reported `COMPLETED` because `| tee` hid the exit code, so the scripts now use `set -o pipefail`.
- **`10850472` (example data):** reached dataset loading in 3.5 min, downloaded the CLIP weights (so **GPU nodes have internet**) and confirmed the **offline wandb path `train_network/wandb/offline-run-*`**. Then it failed: the example single-arm zarr has **only 3 episodes and 5 tasks, one demo each**, so the per-task validation split leaves nothing to train on. **It can't be trained with `val_ratio > 0`, and it's useless for timing.**
- **`10850802` (our `block_raise` data, from the zip with `task.dataset.cache_dir=/tmp/noroian.r/zarr_cache`): passed.** 3 epochs in 13.7 min. Train loss 0.213 → 0.040 → 0.029, val action MSE 0.0389 → 0.0295 → 0.0192.
  - **4.66 steps/s, 518 steps/epoch, about 2.1 min/epoch of training.** Peak GPU memory **34 GB**. Full plots, videos and CSVs are in **`first_smoketest/`** (see its README).
  - **Each checkpoint is 6.4 GB** (weights + EMA + optimizer). A run keeps up to 5: top-2 by val MSE, top-2 by train loss, and `latest.ckpt` at the final epoch. The smoke run alone left 30 GB in home (still to delete).
  - The per-epoch attention videos are all copies of the final epoch: the runner overwrites `vis/`, and offline wandb only copies media when the run ends.
- `scripts/wandb_offline_to_csv.py` turns an offline run into `history.csv` / `epochs.csv` and loss, action-MSE, lr/grad-norm and GPU-memory PNGs. It needs `wandb==0.27.2`, pandas and matplotlib, all available in the `bp` env.

- **Full run submitted:** job `10851655` (`block_raise_bp_v1`, 140 epochs, 1 H200), using the old `train_full.sbatch`. Its results get packaged into `~/CapstoneProject/results/block_raise_bp_v1/`.

### 10. Reusable pipeline (2026-10-06)
The one-off scripts (`train_full.sbatch`, `train_smoke.sh`, `process_firstreal.sbatch`, `get_example_data.sbatch`) were replaced by a general per-task pipeline: `submit.sh` / `run_pipeline.sh`, `process_dataset.sbatch`, `train.sbatch` and `package_run.py`. Results now go to `results/<task>/run_NNN_<date>/`, and the job pushes them to GitHub itself. `first_smoketest/` became `results/block_raise/run_000_2026-10-05_smoke/`.

> **NEXT SESSION, START HERE:** check job `10851655`, then do the one-time setup in **[DIRECTIONS.md](DIRECTIONS.md) §0** (pull on the cluster, git identity, move the dataset, repackage v1 as `run_001`). How it all fits together: **[DESIGN.md](DESIGN.md)**.

### 7. Read the training code and made the decisions it calls for
- **No resume support** (details in step D). This has to be sorted out before a run that's longer than 8 hours.
- **Logging: wandb fully offline** (details in "Logging plan" below).

---

## Logging plan (decided: wandb fully offline, no account)
- Add **`logging.mode=offline`** to every train command, so no `wandb login` or account is needed.
- What gets saved:
  - **Checkpoints:** `train_network/runs/<date>/<time>_<exp_name>_.../checkpoints/*.ckpt`. These are saved whether or not wandb is used.
  - **wandb offline run:** `train_network/wandb/offline-run-<timestamp>-<id>/` (confirmed in the smoke test).
    - Metrics go into a binary `run-<id>.wandb` file.
    - The config and the attention-visualization images are normal files under `files/` (images in `files/media/`).
  - **Slurm log:** the printed loss per epoch, readable as plain text.
- **`scripts/wandb_offline_to_csv.py` (done):** reads the `.wandb` file with wandb's own reader (no account) and writes `history.csv`, `epochs.csv` and loss-curve PNGs. It was tested on the smoke run, and the output is in `first_smoketest/`.
- To view results on the laptop, copy the run folders back through the transfer node:
  ```bash
  rsync -a noroian.r@xfer.discovery.neu.edu:~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/train_network/runs/ ./runs/
  ```
  Do the same for the `wandb/` folder.
- Optional later: `wandb sync <offline-run-dir>` uploads a run to wandb.ai if we ever make an account.

---

## Remaining work

### A. Build the `bp` conda env (DONE: job `10841186`, after `10838595` failed)
Submit as a CPU batch job so it survives disconnects:
```bash
ssh noroian.r@login.explorer.northeastern.edu 'mkdir -p ~/CapstoneProject/logs && cat > ~/CapstoneProject/build_env.sbatch <<"EOF"
#!/bin/bash
#SBATCH --job-name=bp_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/noroian.r/CapstoneProject/logs/%x_%j.out
module load anaconda3/2024.06
cd ~/CapstoneProject/Capstone/behavior_prompting
conda env create -f environment.yml
conda run -n bp python -c "import torch, behavior_prompting; print(torch.__version__, torch.version.cuda)"
EOF
sbatch ~/CapstoneProject/build_env.sbatch'
```
- Watch progress with `squeue -u noroian.r` and `tail -f ~/CapstoneProject/logs/bp_env_<jobid>.out`.
- The parts most likely to fail to build are `free-mujoco-py`, `spnav` (a GitHub tarball) and the `robomimic` git install.
- If the env gets partly built and then fails, run `conda env remove -n bp` before retrying.

### B. Download and unzip the example dataset (DONE: job `10847436`; the dataset turned out to be untrainable, see 9)
Run on a compute node or as a short batch job:
```bash
conda activate bp
cd ~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/train_network
mkdir -p datasets/umi && cd datasets/umi
hf download austinpatel/iphumi_example_zarr_datasets --repo-type=dataset --local-dir ./iphumi_example_zarr_datasets
cd ~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/scripts
python unzip_zarr.py -i ../train_network/datasets/umi/iphumi_example_zarr_datasets/replay_buffer_test-data-042926-single.zarr.zip
```

### C. Smoke-test training on a GPU node (DONE on our data: job `10850802`, see 9 and `first_smoketest/`)
> What actually worked: `scripts/cluster/train_smoke.sh` submitted with `sbatch --wrap` on `gpu-short` (interactive `srun --pty` isn't needed).
1. Get an interactive H200 node for a short test:
   ```bash
   srun -p gpu-interactive --gres=gpu:h200:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
   ```
2. Check the GPU and torch:
   ```bash
   nvidia-smi
   conda activate bp
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Run the training command from the summary above with `--num_processes=1` and **`logging.mode=offline`**. Check that it starts, the loss goes down, and it doesn't run out of memory.
4. Things to measure or check during the smoke test:
   - **Time per epoch**, including the attention-visualization "rollout" that runs every 10 epochs (`rollout_every: 10`). This decides the strategy for the 8-hour limit in step D.
   - Peak GPU memory, from `nvidia-smi`.
   - Where the offline wandb folder is created. Then test `scripts/wandb_offline_to_csv.py` on it.
   - **`dataloader.num_workers=8`** should match the CPUs requested.

### D. Full training run as an `sbatch` job
- Partition `gpu`, `--gres=gpu:h200:1` (or more GPUs together with a matching `--num_processes`), `--time=08:00:00`.
- **Resume: checked. Training cannot resume from a checkpoint.**
  - `logging.resume` is only a wandb option. `train.py` → `TrainPolicyWorkspace.run()` always starts at epoch 0 with a fresh optimizer and LR scheduler, and it never calls `load_checkpoint`. Only `rollout_policy_workspace.py` loads checkpoints.
  - With `umi_policy_base` (`save_last_ckpt: False`), the only mid-run saves are `train_loss` top-k checkpoints every 10 epochs (`save_train_loss_ckpt_every: 10`). Those store the **EMA weights** as `model`, and the LR scheduler state isn't saved at all.
  - Each launch creates a new `runs/...` folder.
- **Decided from the smoke test: a single 8 h job on 1 H200** (140 epochs × about 2.1 min + evaluation every 10 epochs ≈ 5.2 h). Script: `scripts/cluster/train_full.sbatch` (dataset zip + node-local LMDB cache, run dir on `/scratch`).
- The fallback strategy, if a bigger dataset stops fitting:
  1. If 140 epochs fit in about 7.5 h on 1 GPU, submit a single job.
  2. If not, use more GPUs: `--gres=gpu:h200:N` with `--num_processes=N`. The code scales the LR by the number of processes automatically.
  3. If not, reduce `training.num_epochs`.
  4. As a last resort, patch in resume support: save a `latest` checkpoint with the raw model, EMA, optimizer, scheduler and epoch, and reload it on restart.
- Output and checkpoints go under the hydra `run.dir`. `train_full.sbatch` sets it to `/scratch/$USER/bp_runs/...`, because 5 checkpoints × 6.4 GB is too much for home. Scratch is purged, so copy the keeper back.
- Use `scripts/slim_checkpoint.py` to remove optimizer state before sharing or deploying a checkpoint.

### E. Process our custom data into a training dataset (DONE: jobs `10848060` + `10848071`, see 8; the scripted version is `scripts/cluster/process_firstreal.sbatch`)
The raw data in `data/raw/2026-10-01/` has to go through the iPhUMI pipeline, which uses a **separate `iphumi` conda env** from `iPhUMI/python_package/environment.yml`:
1. Build the `iphumi` env on a compute node: `cd iPhUMI/python_package && conda env create -f environment.yml`.
2. Process the demos:
   ```bash
   cd iPhUMI/python_package/iphumi/demonstration_processing
   python process_demos_iphone.py filters.session_name=firstreal \
     group.iphone_dir=~/CapstoneProject/Capstone/data/raw/2026-10-01
   ```
   - The `group` stage copies the exports into `tmp_demonstrations/`.
   - Confirm the session name from the demo JSON `sessionName` field. The folder names suggest `firstreal`.
   - The iPhone 16 Pro calibration was already added in commit `620c7b6`.
3. **Add task labels.** These demos were recorded with task label type "None", but dunetp samples by task. Run this after the `group` stage, then rerun the later stages:
   ```bash
   python ~/CapstoneProject/Capstone/scripts/label_whole_demo_as_task.py \
     --demonstrations-dir tmp_demonstrations --session-name firstreal --task-name <task_name>
   ```
4. Create a session and build the zarr:
   ```bash
   python create_session_iphone.py input_session_filters="[firstreal]" output_session_name=firstreal-single sides=single
   python build_umi_dataset_iphone.py session_dir=tmp_sessions/firstreal-single
   ```
5. Unzip the zarr with `unzip_zarr.py`, then **rename left to right**, because the single-arm configs read `right` keys:
   ```bash
   python ~/CapstoneProject/Capstone/scripts/rename_zarr_side.py <path>.zarr --from-side left --to-side right
   ```
6. Sanity-check the dataset with `print_replay_buffer.py` / `vis_replay_buffer.py`.
7. Train with the same command as in D, pointing `task.dataset.dataset_path` at the new `.zarr`. Optionally add a modifiers file under `config/modifiers/umi/<our_task>/`.

---

## Gotchas learned so far
- **Don't run heavy commands on the login node**: big rsyncs, `conda env create` and even `conda env list` get killed. Use `xfer.discovery.neu.edu` for transfers and Slurm (`sbatch` / `srun`) for everything else.
- **`data/` is gitignored**, so it never arrives with `git clone` and always has to be copied separately.
- **When vendoring a repo, check `.gitmodules`.** Submodules arrive as empty folders. Also run `git ls-files --others --ignored --exclude-standard <dir>` to catch code that our ignore rules drop.
- **Plan for the 8-hour GPU job limit**: training can't resume from checkpoints. Our current data fits in one job (about 5.2 h).
- **Never unzip big zarrs on the networked filesystem.** Train from the `.zarr.zip` with `task.dataset.cache_dir=/tmp/$USER/zarr_cache`, which makes a one-time LMDB copy on node-local disk that multiple dataloader workers can read safely.
- **Batch jobs have no stdin**: anything that calls `input()` (LIBERO's first import) crashes. Answer it with `echo n |` first.
- **`cmd | tee log` hides failures** unless `set -o pipefail` is set.
- **Don't paste long multi-line commands into the terminal.** Write the script locally, `scp` it over, then `sbatch` it.
- **`scancel` returns before the job stops.** Wait a few seconds before deleting files the job was writing.
