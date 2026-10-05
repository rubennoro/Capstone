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
- Not yet created: the `bp` env (no conda env exists yet) and the HF example dataset (not downloaded yet).

### 6. Submitted the `bp` env build (job `10838595`)
- Submitted with the sbatch script in step A below. It's running on node `c0613` in the `short` partition.
- At 25 minutes, conda had solved and verified the env and was on "Executing transaction". The pip step (torch, mujoco, robomimic, local `-e` installs) comes after that.
- Log: `~/CapstoneProject/logs/bp_env_10838595.out`.

### 7. Read the training code and made the decisions it calls for
- **No resume support** (details in step D). This has to be sorted out before a run that's longer than 8 hours.
- **Logging: wandb fully offline** (details in "Logging plan" below).

---

## Logging plan (decided: wandb fully offline, no account)
- Add **`logging.mode=offline`** to every train command, so no `wandb login` or account is needed.
- What gets saved:
  - **Checkpoints:** `train_network/runs/<date>/<time>_<exp_name>_.../checkpoints/*.ckpt`. These are saved whether or not wandb is used.
  - **wandb offline run:** `wandb/offline-run-<timestamp>-<id>/`. It's probably under `train_network/wandb/`, because the code doesn't set a wandb `dir` and Hydra 1.2 doesn't change the working directory. **Confirm the exact path during the smoke test.**
    - Metrics go into a binary `run-<id>.wandb` file.
    - The config and the attention-visualization images are normal files under `files/` (images in `files/media/`).
  - **Slurm log:** the printed loss per epoch, readable as plain text.
- **To do: write `scripts/wandb_offline_to_csv.py`.** It should read the `.wandb` file with wandb's own Python reader (no account), write the logged metrics to a **CSV**, and save **loss-curve PNGs** (train loss, val loss, action MSE). Write and test it against a real offline run from the smoke test, using `wandb==0.27.2` from the `bp` env.
- To view results on the laptop, copy the run folders back through the transfer node:
  ```bash
  rsync -a noroian.r@xfer.discovery.neu.edu:~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/train_network/runs/ ./runs/
  ```
  Do the same for the `wandb/` folder.
- Optional later: `wandb sync <offline-run-dir>` uploads a run to wandb.ai if we ever make an account.

---

## Remaining work

### A. Build the `bp` conda env (submitted as job `10838595`; running)
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

### B. Download and unzip the example dataset
Run on a compute node or as a short batch job:
```bash
conda activate bp
cd ~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/train_network
mkdir -p datasets/umi && cd datasets/umi
hf download austinpatel/iphumi_example_zarr_datasets --repo-type=dataset --local-dir ./iphumi_example_zarr_datasets
cd ~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/scripts
python unzip_zarr.py -i ../train_network/datasets/umi/iphumi_example_zarr_datasets/replay_buffer_test-data-042926-single.zarr.zip
```

### C. Smoke-test training on a GPU node (example data)
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
- **Strategy for the 8-hour limit, chosen from the smoke test's time per epoch:**
  1. If 140 epochs fit in about 7.5 h on 1 GPU, submit a single job.
  2. If not, use more GPUs: `--gres=gpu:h200:N` with `--num_processes=N`. The code scales the LR by the number of processes automatically.
  3. If not, reduce `training.num_epochs`.
  4. As a last resort, patch in resume support: save a `latest` checkpoint with the raw model, EMA, optimizer, scheduler and epoch, and reload it on restart.
- Output and checkpoints go under `train_network/runs/` (the hydra `run.dir`). Move them to `/scratch` if home space gets tight.
- Use `scripts/slim_checkpoint.py` to remove optimizer state before sharing or deploying a checkpoint.

### E. Process our custom data into a training dataset
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
- **Plan for the 8-hour GPU job limit**: training has to be able to resume from checkpoints.
