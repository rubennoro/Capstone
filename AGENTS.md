# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-10-06)

- **Task `block_raise`** (raise an eraser with one gripper). Dataset **`firstreal-single`**: session `firstreal`, 45 demos,
  37,971 frames, 845 MB.
  - At the standard location: `~/CapstoneProject/datasets/block_raise/firstreal-single/firstreal-single.zarr.zip`.
- **Runs** (in `results/block_raise/`):
  - `run_000_2026-10-05_smoke/`: 3 epochs, val action MSE 0.0192.
  - `run_001_2026-10-05/`: first full run (job `10851655`, 140 epochs, 4h55m on 1 H200). Best val action MSE **0.000148
    at epoch 50**. Val MSE plateaus from ~epoch 30 and val loss creeps up after ~50 (mild overfitting), so ~60 epochs is
    enough next time.
    - Copied by hand from the old sbatch output (since deleted from the cluster; all of it is in this folder) and
      pushed (`8ac2faf`). Only final-epoch videos exist.
    - Slim best checkpoint (2.5 GB) is in `run_001_2026-10-05/checkpoints/` (gitignored) on both the laptop and the cluster
      clone. The full-size one (with optimizer, for resuming) plus its `.hydra/` config is the only thing left in
      `/scratch/noroian.r/bp_runs/2026.10.05/23.22.47_block_raise_bp_v1/`; scratch gets purged eventually.
- **Pipeline** (`scripts/run_pipeline.sh`, `scripts/cluster/{submit.sh,process_dataset.sbatch,train.sbatch}`,
  `scripts/package_run.py`): the cluster clone is now pulled to `b87c819` with its git identity set. DIRECTIONS.md §0 is fully done.
  - **Pipeline tested end to end:** job `10878177` → `results/block_raise/run_002_2026-10-06_smoke/` (11 min). Trained, packaged
    in-job, and pushed to GitHub from the compute node by itself, with per-epoch videos. The checkpoint stayed on the cluster.
    Val action MSE 0.0191988, identical to run_000 (training is deterministic).
  - `scripts/check_job.sh <jobid> [--wait]`: read-only status check (Slurm, log tail, packaged/pushed, what's on GitHub).
  - Still never run: `process_dataset.sbatch`.
  - `package_run.py` gets killed on the login node (too heavy). Run it inside a job, or locally.
- Plots: the user is happy with them as they are. Don't change `scripts/wandb_offline_to_csv.py` unless asked.

## Next time

1. Watch the v1 videos (`results/block_raise/run_001_2026-10-05/videos/final/`): does the attention follow the prompt?
2. Decide the next real training run (e.g. ~60 epochs, since v1 overfit after ~50), or record more data and use
   `scripts/run_pipeline.sh all ...` to test `process_dataset.sbatch`.
3. Optional: set a real git email on the laptop (`git config --global user.email ...`) so commits link to the GitHub account.
