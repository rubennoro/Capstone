# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-11-20, 02:10)

- **Branch `rnoro/training` (this checkout) holds `train/`: the training workflow for rig data.** Local commits only,
  not pushed. Run order: [train/DIRECTIONS.md](train/DIRECTIONS.md); setup for a new machine: [train/SETUP.md](train/SETUP.md);
  script details: [train/README.md](train/README.md); analysis: [train/analysis/README.md](train/analysis/README.md).
  - Flow: `pull.sh`, `upload.sh`, `submit.sh`, `status.sh`, `fetch.sh`, `analysis/*`; `all.sh` chains them (without
    `--pull/--package` it uses `data/<folder>/package/<dataset>/` as it is, e.g. a symlink to a umiproc package).
  - The policy type is picked per submit with `--policy plain|prompting` (ee168f2; `submit.sh`, `go.sh`, `all.sh`),
    otherwise `TRAIN_CONFIG` in `train/config.local.env`: `umi_policy_dunetp` behavior prompting (default) or
    `umi_policy_dunet` plain diffusion. Frozen per run: check the last `TRAIN_CONFIG=` line in
    `<run>/run_config/submit_config.env`.
  - 0f0863b: datasets with several labels (`label_dir multi`, e.g. PICK_TEST1 + PICK_TEST2) are allowed.
- **Data comes from umi-processing on the laptop** (`umiproc package`, `~/data/umi/datasets/<label>/<dataset>/`), not
  the Jetson's dataproc. Subsets "first N episodes of each folder" are built with a temporary `UMIPROC_ROOT` whose task
  folders link only `episode_1..N` (umiproc has no range option).
- **All 14 runs are done and fetched; every checkpoint is on the laptop only** (two tasks: PICK_TEST1 = one day, 30 eps;
  PICK_DOUBLE = PICK_TEST1 + PICK_TEST2, 62 eps). A run lives on the laptop in `~/Capstone/data/<FOLDER>/policy/<run>/`
  (checkpoint, summary.json, epochs.csv, plots, logs). The cluster keeps the same run folder without its checkpoint in
  `~/CapstoneProject/policies/<label>/<run>/` (label `PICK_TEST1`, or `multi` for PICK_DOUBLE).
  Checkpoint: `<run>/checkpoints/top1_epoch=...-val_action_mse_error_all=<val MSE>.ckpt` (BP 2.6 GB, plain 1.1 GB).

  | Task, policy | Episodes | Dataset | Run (`data/<FOLDER>/policy/...`) | Best val MSE |
  |---|---|---|---|---|
  | PICK_TEST1, behavior prompting | 10 | `pick1-ep10-v1` | `PICK_TEST1/run_020_2026-11-19` | 0.000835 |
  | | 15 | `pick1-ep15-v1` | `PICK_TEST1/run_012_2026-11-18` | 0.000408 |
  | | 20 | `pick1-ep20-v1` | `PICK_TEST1/run_014_2026-11-18` | 0.000329 |
  | | 25 | `pick1-ep25-v1` | `PICK_TEST1/run_016_2026-11-18` | 0.000230 |
  | | 30 | `pick1-v1` | `PICK_TEST1/run_006_2026-11-18` | 0.000201 |
  | PICK_TEST1, plain diffusion | 10 | `pick1-ep10-v1` | `PICK_TEST1/run_030_2026-11-19`* | 0.000862 |
  | | 15 | `pick1-ep15-v1` | `PICK_TEST1/run_028_2026-11-19` | 0.000397 |
  | | 20 | `pick1-ep20-v1` | `PICK_TEST1/run_031_2026-11-19` | 0.000372 |
  | | 25 | `pick1-ep25-v1` | `PICK_TEST1/run_024_2026-11-19` | 0.000269 |
  | | 30 | `pick1-v1` | `PICK_TEST1/run_022_2026-11-19` | 0.000212 |
  | PICK_DOUBLE, plain diffusion | 30 (15+15) | `pick1-double-ep15-v1` | `PICK_DOUBLE/run_003_2026-11-19` | 0.000227 |
  | | 40 (20+20) | `pick1-double-ep20-v1` | `PICK_DOUBLE/run_004_2026-11-19` | 0.000335 |
  | | 50 (25+25) | `pick1-double-ep25-v1` | `PICK_DOUBLE/run_005_2026-11-19` | 0.000565 |
  | | 62 (30+32) | `pick1-double-v2` | `PICK_DOUBLE/run_002_2026-11-19` | 0.000498 |

  \* `run_030` trained all 60 epochs; its job died while copying wandb/.hydra (home full), so its cluster `STATUS` still
  says `running` (fetched with `--force`; the laptop copy is complete). Ignore these cluster run folders (never had a
  usable checkpoint): `PICK_TEST1/run_001`..`004`, `007`, `008`, `010`, `017`..`019`, `run_026_2026-11-19`, all
  `*_smoke`, `multi/run_001_2026-11-19`.
  Val MSE is only roughly comparable across runs: each holds out its own 1-3 val episodes. In PICK_DOUBLE it rises
  with size (more PICK_TEST2 episodes in val), so it cannot rank those four; compare them with offline eval on one
  shared episode set. Full notes on every policy (datasets, folder contents, best epochs, training times):
  `data/RUNS.md` (gitignored).
- **2026-11-19 incidents:** (1) the cluster home filled at 20:49: `run_030` packaging died and two running jobs hung at
  0% GPU; fixed by deleting checkpoints already on the laptop. After the final fetch (2026-11-20) all checkpoints were
  deleted on the cluster (`policies/` 369 MB). Keep fetched checkpoints off the cluster; check the quota on the OOD
  dashboard (`quota` is not allowed on the login node).
  (2) Uploads from the laptop to xfer drop after ~40-70 s (`client_loop: send disconnect: Broken pipe`); retrying with
  `rsync --partial` resumes, see `data/PICK_DOUBLE/upload_and_submit.sh`. Cause unconfirmed (network?). Downloads
  (fetch) had no drops.
  (3) One `sbatch` refusal ("submission failed"), fine on retry.
- Explorer queue: `gpu` QOS 4 running / 8 queued per user; account `eece5640.202530` (in config.local.env).
- Other branches: umi-processing + dataproc on `rnoro/data-work*` (`~/Capstone-data-work`); deployment on
  `rnoro/final-deployment` (`~/Capstone-final-deployment`). Each has its own AGENTS.md.

## Next time

1. Offline eval on one shared set of episodes so the runs can be ranked (val MSE cannot do it):
   `train/analysis/eval.sh <FOLDER> <run> --ckpt top1`; reports with `python3 train/analysis/run_report.py <FOLDER> <run>`.
   Compare BP vs plain at each size, and two days (15+15 ... 62) vs one day (30).
2. Deployment candidate: the 62-episode plain policy `PICK_DOUBLE/run_002_2026-11-19` (or the eval winner). Check that
   `~/Capstone-final-deployment` loads a plain-diffusion checkpoint before robot trials.
3. Laptop disk: ~18 GB free with all 14 checkpoints (~23 GB) on it; move unused ones off if space gets tight.
4. Decide whether to push `rnoro/training`.
