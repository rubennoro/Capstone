# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-11-19)

- **Branch `rnoro/training` (this checkout) holds `train/`: the training workflow for rig data.** Local commits only,
  not pushed. Run order: [train/DIRECTIONS.md](train/DIRECTIONS.md); setup for a new machine: [train/SETUP.md](train/SETUP.md);
  script details: [train/README.md](train/README.md); analysis: [train/analysis/README.md](train/analysis/README.md).
  - Flow: `pull.sh`, `upload.sh`, `submit.sh`, `status.sh`, `fetch.sh`, `analysis/*`; `all.sh` chains them (without
    `--pull/--package` it uses `data/<folder>/package/<dataset>/` as it is, e.g. a symlink to a umiproc package).
  - `TRAIN_CONFIG` picks the policy (`umi_policy_dunetp` behavior prompting = default, `umi_policy_dunet` plain
    diffusion); set it in `train/config.local.env`, it is frozen per run on submit.
  - 0f0863b: datasets with several labels (`label_dir multi`, e.g. packaged from PICK_TEST1 + PICK_TEST2) are allowed.
- **Data now comes from umi-processing on the laptop** (`umiproc package`, `~/data/umi/datasets/`), not the Jetson's dataproc.
- **PICK_TEST1 (one day, 30 eps):** scaling study 10/15/20/25/30 eps. Behavior prompting done (val MSE 0.00084 at 10
  -> 0.00020 at 30, no overfit in 60 epochs); plain diffusion `run_022`..`run_030` submitted 2026-11-19 18:00. Which run
  is which: [data/PICK_TEST1/RUNS.md](data/PICK_TEST1/RUNS.md) (gitignored).
- **PICK_DOUBLE (PICK_TEST1 + PICK_TEST2, plain diffusion), started 2026-11-19, paused on the laptop side:**
  - Packages in `~/data/umi/datasets/multi/`, linked in `data/PICK_DOUBLE/package/`: `pick1-double-v2` (62 = 30+32),
    `pick1-double-ep15-v1` / `-ep20-v1` / `-ep25-v1` (first 15/20/25 of each day; built with a temporary
    `UMIPROC_ROOT` whose task folders link only those episodes).
  - Submitted: `pick1-double-v2` -> `multi/run_001_2026-11-19`, job 12000883 (queued behind the 4-running limit).
  - Not submitted yet: the three subsets. `train/config.local.env` still has the TEMPORARY
    `TRAIN_CONFIG="umi_policy_dunet"` line they need (the driver deletes it after its last submit).
- Explorer queue: `gpu` QOS 4 running / 8 queued per user; account `eece5640.202530` (in config.local.env).
- Other branches: umi-processing + dataproc on `rnoro/data-work*` (`~/Capstone-data-work`); deployment on
  `rnoro/final-deployment` (`~/Capstone-final-deployment`). Each has its own AGENTS.md.

## Next time

1. Queue the rest (laptop must stay awake while it runs; it only submits when <= 5 of your jobs are queued):
   `setsid nohup data/PICK_DOUBLE/run_sweep.sh > data/PICK_DOUBLE/logs/sweep_driver.log 2>&1 < /dev/null &`
   and finish the 62-episode run: `train/all.sh PICK_DOUBLE --run run_001_2026-11-19` (wait, fetch, reports, eval).
2. Fetch the PICK_TEST1 plain-diffusion runs (`train/fetch.sh PICK_TEST1 <run>`) and update RUNS.md.
3. Compare: plain 15+15 (two days) vs `pick1-v1` (30, one day); the 62-episode policy is the deployment candidate.
   Check that `~/Capstone-final-deployment` loads a plain-diffusion checkpoint before robot trials.
4. Decide whether to push `rnoro/training`.
