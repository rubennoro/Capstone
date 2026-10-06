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
  - Still at its old location: `iPhUMI/python_package/iphumi/demonstration_processing/tmp_sessions/firstreal-single/replay_buffer_firstreal-single.zarr.zip`
    (to be moved in DIRECTIONS.md §0).
- **Smoke test done:** `results/block_raise/run_000_2026-10-05_smoke/`. 3 epochs, val action MSE 0.0389 → 0.0192,
  ~2.1 min/epoch, 34 GB GPU memory. Its best checkpoint is on the laptop only (gitignored).
- **Full run in progress:** job **`10851655`** (`block_raise_bp_v1`, 140 epochs, 1 H200 on `gpu`), started 2026-10-05 23:22.
  - At epoch 18 after 42 min (~2.3 min/epoch); val action MSE 0.00077 at epoch 10. Expected to finish ~04:40 on 2026-10-06.
  - It uses the **old** `train_full.sbatch`. It packages into `~/CapstoneProject/results/block_raise_bp_v1/` and keeps the
    full-size best checkpoint in `/scratch/noroian.r/bp_runs/<date>/<time>_block_raise_bp_v1/`.
- **Reusable pipeline committed and pushed** (`f02cfcb`), but **not yet pulled on the cluster**, so the running job
  isn't affected. It consists of `scripts/run_pipeline.sh`, `scripts/cluster/{submit.sh,process_dataset.sbatch,train.sbatch}`
  and `scripts/package_run.py`.
  - Tested: `package_run.py` on the real smoke data, and `submit.sh` with a fake `sbatch`.
  - Not yet run on the cluster: `train.sbatch`, `process_dataset.sbatch`, and pushing to GitHub from a compute node.

## Next time

1. Check job `10851655`: `squeue -u noroian.r; sacct -j 10851655 -X; tail ~/CapstoneProject/logs/bp_train_10851655.out`.
2. When it's done, do the **one-time setup in DIRECTIONS.md §0**:
   - pull on the cluster and set the git identity
   - move `firstreal-single` to `~/CapstoneProject/datasets/block_raise/firstreal-single/`
   - delete the stale `~/CapstoneProject/*.sbatch` copies
   - repackage v1 with `scripts/package_run.py` into `results/block_raise/run_001_2026-10-05/`. Its run dir is on scratch; its wandb run is the newest in `train_network/wandb/`; pass `--job-info` by hand.
3. Review the v1 results together: loss curves, whether val MSE plateaued, whether the attention videos follow the prompt.
4. **End-to-end test of the new pipeline:**
   `scripts/run_pipeline.sh train --task block_raise --dataset firstreal-single --smoke --note "pipeline test"`.
   - Confirm `results/block_raise/run_00N_..._smoke/` gets pushed to GitHub, with `videos/epoch_000{0,1,2}/`.
   - Confirm the checkpoint stays only on the cluster.
   - If the push from the compute node fails (SSH to GitHub blocked), move the commit + push into a small follow-up job that runs where GitHub is reachable (`--dependency=afterany`), or switch the clone to an HTTPS remote with a token.
5. Optional: set a real git email on the laptop (`git config --global user.email ...`) so commits link to the GitHub account.
