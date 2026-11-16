# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-11-15)

- **Branch `rnoro/training` (this checkout) holds `train/`: the training workflow for rig data.** Local commits only,
  not pushed. Run order for Ruben's (already configured) setup: [train/DIRECTIONS.md](train/DIRECTIONS.md); one-time setup
  for a new machine: [train/SETUP.md](train/SETUP.md); script details in
  [train/README.md](train/README.md), analysis in [train/analysis/README.md](train/analysis/README.md).
  - Flow: `pull.sh` (Jetson task folder + `dataproc package` -> `data/<folder>/`), `upload.sh`, `submit.sh`
    (smoke-first Slurm chain), `status.sh`, `fetch.sh` (-> `data/<folder>/policy/<run>/`), `analysis/*`.
  - Decisions: data lives in gitignored `~/Capstone/data/<folder>/`; one slim checkpoint per run (TOPK=1, ~2.5 GB);
    cluster runs in `~/CapstoneProject/policies/` (not the clone); tools synced to `~/CapstoneProject/train_tools/` on
    every submit, so the cluster clone stays on main; the new job does not git-push; `scripts/cluster/` untouched.
  - Tests (all offline): `train/tests/run_tests.sh` (94), `train/tests/cluster/run_tests.sh` (168),
    `train/analysis/tests/run.sh` (20). Regression guard passes.
- **Connections verified 2026-11-15:**
  - Jetson `capstone-gripper@129.10.156.38` (NEU IP, may change), key login, in gitignored `train/config.local.env`.
    dataproc defaults (`~/data`, `~/envs/env_dataproc`) confirmed.
  - Explorer needs the Research Computing key since 2026-11-06 (downloaded via OOD, `~/.ssh/config` block); login and
    xfer both work, xfer accepts commands, `gpu` and `gpu-short` have H200s, `bp` has robotmq.
  - Cluster clone at `06511cd`, clean; needs `git pull --ff-only` (>= `4be9982`) before offline eval. Not done yet
    (asks first).
- **Blocker: no trainable rig data.** Jetson tasks `eraser_pick_1` (eps 61-65) and `eraser_pick_2` (117-121): calibration
  done, every episode failed at SLAM ("orbslam3: never initialised, 0 keyframes"). That is dataproc work on
  `rnoro/data-work` (`~/Capstone-data-work`), not `train/`.
- Other branches: dataproc on `rnoro/data-work`; deployment on `rnoro/deployment`. Each has its own AGENTS.md.
- Earlier iPhone-era results stay in `results/block_raise/` (best: run_001, val action MSE 0.000148 at epoch 50 of 140;
  overfit after ~50). Old pipeline: root [DIRECTIONS.md](DIRECTIONS.md).

## Next time

1. Fix SLAM initialisation on the Jetson (read `orbslam3.log` under `~/data/tasks/<folder>/.staging/<ep>/processed/slam/work/`),
   until `dataproc status <folder>` shows exported episodes.
2. First real run: `train/go.sh <folder> <new-dataset> --package --no-raw --note "..."`. The smoke run is the first time
   the trainer reads a dataproc zip.
3. Pull the cluster clone (with OK) so `train/analysis/eval.sh` works.
4. Decide whether to push `rnoro/training`.
