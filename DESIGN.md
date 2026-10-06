# Design: how this repo is organized

The goal: record demos with an iPhone gripper (iPhUMI), turn them into a dataset, train a **behavior prompting** policy on
the Northeastern Explorer cluster, and get a standard results folder back for every training job. To actually run it,
see [DIRECTIONS.md](DIRECTIONS.md).

## The pipeline at a glance

```
iPhone app ──export──▶ laptop data/raw/<date>/ ──rsync──▶ cluster data/raw/<date>/
                                                              │  process_dataset.sbatch (CPU, iphumi env)
                                                              ▼
                                     ~/CapstoneProject/datasets/<task>/<dataset>/<dataset>.zarr.zip
                                                              │  train.sbatch (GPU, bp env)
                                                              ▼
              /scratch run dir (raw training output) ──package_run.py──▶ results/<task>/run_NNN_<date>/
                                                                                  │  git push (by the job)
                                                                                  ▼
                                                                    GitHub ──git pull──▶ laptop
```
`scripts/run_pipeline.sh` (laptop) and `scripts/cluster/submit.sh` (cluster) wrap all of this into one command.

## Where things live

| Place | What's there | In git? |
|---|---|---|
| **Laptop** `~/Capstone` | The repo, raw exports in `data/raw/`, any fetched checkpoints | Code and results yes; data and checkpoints no |
| **GitHub** `rubennoro/Capstone` | The bridge: code goes laptop → GitHub → cluster; results go cluster → GitHub → laptop | n/a |
| **Cluster** `~/CapstoneProject/Capstone` | A clone of the repo. Jobs run its scripts and commit results into it | Yes |
| **Cluster** `~/CapstoneProject/datasets/<task>/<dataset>/` | Processed datasets (`.zarr.zip`; `firstreal-single`, 45 demos, is 845 MB) | No |
| **Cluster** `~/CapstoneProject/logs/` | Console output of every job (`<job-name>_<id>.out`) and GPU memory CSVs | No |
| **Cluster** `~/CapstoneProject/wandb_runs/<task>/<run>/` | Raw offline wandb run of each successful job (the CSVs in results are made from it) | No |
| **Cluster** `/scratch/noroian.r/bp_runs/<task>/<run>/` | Hydra run dir while training. Deleted after a successful package; kept on failure for debugging. Scratch is purged periodically | No |
| **Cluster** conda envs `bp`, `iphumi` | Training env / processing env (`~/.conda/envs/`) | Built by `scripts/cluster/build_*_env.sbatch` |

## Repo layout and the important files

```
Capstone/
├── DIRECTIONS.md  DESIGN.md       how to use it / how it's built
├── AGENTS.md                      current status and next steps (updated each working session)
├── results/<task>/                one folder per task
│   ├── README.md                  what the task is, which datasets exist (written by hand)
│   └── run_NNN_<date>[_smoke]/    one folder per training job (layout below)
├── scripts/
│   ├── run_pipeline.sh            LAPTOP entry point: rsync raw data, update cluster clone, call submit.sh
│   ├── package_run.py             THE analysis: run dir → standard results folder (same for every job)
│   ├── wandb_offline_to_csv.py    offline wandb run → history/epochs CSVs + the 5 plots (used by package_run.py)
│   ├── slim_checkpoint.py         drop optimizer state from a checkpoint, ~6.4 → ~3 GB (used by package_run.py)
│   ├── label_whole_demo_as_task.py  label "no label" demos as one task (used by process_dataset.sbatch)
│   └── cluster/
│       ├── submit.sh              CLUSTER entry point: picks run numbers, submits/chains the jobs below
│       ├── process_dataset.sbatch raw exports → .zarr.zip (any task, sessions, raw dir)
│       ├── train.sbatch           train (full or smoke) → package → commit + push results
│       ├── build_bp_env.sbatch    rebuild the training env
│       └── build_iphumi_env.sbatch  rebuild the processing env
├── iPhUMI/                        vendored upstream: iPhone app + processing pipeline
│   └── python_package/iphumi/demonstration_processing/
│       ├── process_demos_iphone.py, create_session_iphone.py, build_umi_dataset_iphone.py
│       ├── calibration/gripper_calibration.yaml   per-phone gripper calibration (we added iPhone 16 Pro)
│       └── tmp_demonstrations/  tmp_sessions/      processing cache (gitignored, shared by all datasets)
├── behavior_prompting/            vendored upstream: the policy and training code
│   ├── environment.yml            the `bp` env
│   └── behavior_prompting/train_network/
│       ├── train.py               training entry point (Hydra)
│       ├── config/umi_policy_dunetp.yaml, config/task/umi*.yaml   policy + dataset config
│       └── workspace/train_policy_workspace.py   the training loop (one small Capstone change, below)
└── data/raw/<date>/               raw iPhone exports (gitignored)
```

## The results folder (same for every job)

```
results/<task>/run_NNN_<date>[_smoke]/
├── README.md         status · data (dataset, path, split) · training settings · per-eval results table · speed/GPU · files
├── summary.json      the same key numbers, machine-readable (for comparing runs)
├── epochs.csv        one row per epoch: train/val loss, action MSE (all, position, rotation, gripper width)
├── history.csv       every gradient step: train loss, lr, grad norm, steps/s
├── plots/            loss_per_step · loss_per_epoch · action_mse_per_epoch · lr_and_grad_norm · gpu_memory
├── videos/           epoch_XXXX/ for each rollout (every 10 epochs) + final/
├── logs/             train_log.txt · slurm_job.txt (progress bars stripped) · gpu_memory.csv
├── run_config/       config.yaml + overrides.yaml (exact Hydra config) · normalizer.pkl · job_info.json
└── checkpoints/      best_<epoch, val MSE>.ckpt: the only checkpoint kept, slimmed; gitignored (lives on the cluster)
```
Run numbers are allocated per task at submit time (`run_001`, `run_002`, …), and `_smoke` marks 3-epoch tests.
`run_000_2026-10-05_smoke` is the first smoke test, migrated from the old `first_smoketest/` folder.

## Design decisions (and why)

| Decision | Why |
|---|---|
| **The job pushes its own results to GitHub** | The cluster can't reach the laptop, so this is the only automatic path back. Each job commits only its own run folder, under a file lock (`flock`) with pull-rebase-retry, so concurrent jobs can't conflict. |
| **Packaging always runs** (success, failure, or 10 min before the time limit via `--signal=USR1@600`) | Every job leaves a results folder that explains what happened. A failed run shows its last log lines in its README. |
| **Only the best checkpoint** (`checkpoint.topk.k=1`, no train-loss checkpoints, `latest.ckpt` not kept) | Checkpoints are 6.4 GB each. "Best" = lowest validation action MSE. It's slimmed (no optimizer state) because training can't resume anyway, and deployment loads without the optimizer. |
| **Train from the `.zarr.zip` with `task.dataset.cache_dir=/tmp/...`** | Unzipping writes ~44k small files, which takes hours on the networked home filesystem. The cache makes one LMDB copy on the GPU node's local disk, which multiple dataloader workers can read safely. |
| **Run dir on `/scratch`, results in home** | Training output is big and temporary. The packaged results are small and permanent. |
| **wandb fully offline** (`logging.mode=offline`) | No account needed. `wandb_offline_to_csv.py` reads the binary run file directly. |
| **Smoke and full runs share one script and one layout** | A smoke test is the same job with `training.num_epochs=3` and evaluation every epoch, on the 2 h `gpu-short` partition. |
| **Datasets are never overwritten** | Results READMEs refer to datasets by name, so a name must always mean the same data. |
| **Processing cache is shared** | `tmp_demonstrations/` keeps processed demos, so building a second dataset from an already-processed session reuses them. |

## Changes we made to the vendored code

Keep this list current: these edits are lost if the upstream code is ever re-imported.
- `behavior_prompting/.../workspace/train_policy_workspace.py`: after each rollout, copy `vis/*.mp4` into
  `vis/epoch_XXXX/` (the runner overwrites them otherwise). Marked `# Capstone:`.
- `iPhUMI/.../calibration/gripper_calibration.yaml`: added the iPhone 16 Pro (`iPhone17,1`) entry.
- `behavior_prompting/deps/{icrt,LIBERO,real-env}`: vendored copies of upstream git submodules.

## Cluster quirks the scripts already handle
- **Login node kills heavy work** (big copies, conda, even large Python imports): use `xfer.discovery.neu.edu` for transfers and Slurm for everything else.
- **LIBERO asks a Y/N question on first import**, which crashes batch jobs. The scripts answer `n` up front.
- **Headless GPU nodes**: the scripts set `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy`.
- **`cmd | tee` hides failures**: the scripts use `set -o pipefail`.
- **8-hour GPU job limit, and no resume support**: size epochs to fit (see DIRECTIONS.md §4).
