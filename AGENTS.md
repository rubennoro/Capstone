# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-10-27)

- **Franka simulation evaluator: all 19 phases built and committed; W10 end-to-end verification run (all but the
  viewer).** Resume from [controlSW/docs/sim_plan/STATUS.md](controlSW/docs/sim_plan/STATUS.md).
  - Branch `sim-eval` (local; `origin/sim-eval` is at `3bfadbf`) now holds everything.
    - `sim-eval-w10b` was brought in commit by commit on 2026-10-27, and its worktree can be removed.
  - **A9 finger kit (2026-10-27):** finger modes `stock | box | kit`, set by `tooling.finger_adapter.mode`.
    - Docs: `controlSW/franka_sim/docs/finger_model.md`.
    - Review FIX open: kit masses, camera-check tag ids.
    - The real UMI finger kit from the user's Onshape STLs is built but still in the scratchpad.
  - Covers: station configs, MuJoCo Panda + Franka Hand + printed fingers + phone/rig mount, SimCore/MuJoCo arm
    and hand, scene + ground truth, renderer, wireless sensor path, eval runner with Stages 1/1b/2a/2b/3, judge,
    sweeps, hardware-day tools. `python -m franka_sim eval --profile sim-iphone --stage all --out DIR`.
  - The arm is the **Panda** (decided 2026-10-27). Stage 1 at +10 cm fingers / phone -y: 43/45 demos feasible.
  - Verified 2026-10-26/27: suites + regression guard green; Stages 1-2b on iPhone and rig; bp_run001 passes the
    policy contract and runs closed loop on the laptop GPU (fails tracking/obs-motion gates, 0/1 success); smoke sweep
    works. FAILs are findings (2 infeasible demos, the 60 deg bridge cone/box clamps refuse every sim-rig 1b start).
  - Waiting on the user: confirm fingers +10 cm / phone -y (tooling YAMLs still stock); cone/workspace box; how
    1b/2a gates treat infeasible demos; merge. Not yet run: the viewer (needs a display).

- **Custom sensor rig (Jetson Orin Nano + 2 IMX219 CSI cameras + ICM-42688-P IMU): software done and verified, not
  yet run on hardware.** Status, commands, decisions and risks are in [ROADMAP.md](ROADMAP.md) ("Status" section).
  - Data collection: `sensor_rig/`, 400 tests, committed in `987dcd6` with newer changes uncommitted. On the Jetson:
    `jetson_setup.sh`, `sensor-rig doctor`, `sensor-rig collect`, `stamp-offset`. On the laptop: `sensor-rig
    process-session`. Procedure in `sensor_rig/docs/first_plugin.md`.
  - Deployment: `controlSW/franka_deploy/` (uncommitted, ~190 tests): `python -m franka_deploy run --config
    deploy.yaml [--mock]`, plus annotate/eval/offline-eval/reinfer. Procedure in `controlSW/franka_deploy/docs/hardware_day.md`.
  - Built by executor + reviewer agent pairs. Every step was independently re-verified.
  - **iPhone path guard:** `scripts/regression/` (uncommitted).
    - It proves the iPhone pipeline and deployment are unchanged since `0ddedef`: protected paths, the real
      processing on 5 firstreal episodes against an exact golden, and the iPhone peripheral configs.
    - Run it before any commit touching shared code. The rule is additive-only; upstream fixes are not applied
      (subclassed in franka_deploy instead).
  - Only shared-file touches: an appended `"JetsonRig1,1"` entry in `gripper_calibration.yaml` (a copy of iPhone17,1;
    needs a measured TCP), +1 line in real-env `pyproject.toml`, and new `rig_*` files in real-env.
  - Local scratch envs (not in the repo, in the session scratchpad):
    - env_iphumi (processing + sensor_rig);
    - env_bp (inference; torch shared from `~/.local`);
    - env_realenv (real-env stack).
  - Planning docs and research are in `sensor_rig/docs/planning/`. A Raspberry Pi 5 backend also exists
    (`rig_pi5.yaml`) but the chosen platform is the Jetson.

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

- **First:** A9 fixes, the real kit into `assets/finger_kits/`, and the real-kit checks. Also confirm the cone/workspace box.
  Then continue from `controlSW/docs/sim_plan/STATUS.md` ("Next actions").
0. Sensor rig (see ROADMAP.md "Status"; checklist in [TODO.md](TODO.md); parts in [PURCHASING.md](PURCHASING.md)):
   - Decide: gripper actuator; buy cameras (B0180 + 2.1 mm main lens; B0179 175° ultrawide, IR-cut version: decided 2026-10-19); where the 1 kHz
     loop runs; v_max and the success rule for evaluation; whether to commit the new work.
   - Hardware day:
     - data collection: `sensor_rig/docs/first_plugin.md`;
     - arm: `controlSW/franka_deploy/docs/hardware_day.md`.
   - Build the ORB-SLAM3 image on a machine with ≥ 20 GB free (the laptop disk is ~90% full).
1. Watch the v1 videos (`results/block_raise/run_001_2026-10-05/videos/final/`): does the attention follow the prompt?
2. Decide the next real training run (e.g. ~60 epochs, since v1 overfit after ~50), or record more data and use
   `scripts/run_pipeline.sh all ...` to test `process_dataset.sbatch`.
3. Optional: set a real git email on the laptop (`git config --global user.email ...`) so commits link to the GitHub account.
