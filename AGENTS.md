# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-10-17; teleop added 2026-11-18; teleop-train 2026-11-19)

- **Teleop data collection** (this branch, `rnoro/teleop-collect-deploy`): `deployment/teleop_collect/` is built and
  tested; it has not run on hardware yet.
  - What it is: keyboard teleop of the Panda from a browser page on the laptop, recording both Pi streams (MJPEG .mkv)
    and every arm state into `~/CapstoneTeleop/<task>/episode_NNN/`.
  - The PC runs `control_deploy` from `rnoro/final-deployment`, unchanged and never modified. `deployment/contracts/`
    here is its byte copy @ `ea9f7fb` (re-synced 2026-11-19, `fd177e1`).
    - TCP: +45° about flange z (training's axes), still `settled: false`.
    - teleop_collect records with the deployment rig `contracts/rigs/rig_pi_v3_640.yaml`.
    - Mock episode: `~/Downloads/data/incoming/episode_000152`.
    - `pc_probe --axes` added.
  - Docs: `teleop_collect/{PLAN,STRUCTURE,DESIGN,DIRECTIONS}.md` and `deployment/control_deploy/DIRECTIONS.md`.
  - Tests: `teleop_collect` has 1301 (1295 passing at the re-sync, plus the `--axes` tests), including the mock run
    and the cross test against `franka_control --mock`; `contracts` has 582. The env is `~/envs/teleop_collect`.
    - Known flaky: `test_failure_rows.py::test_a_camera_down_mid_episode...`, a race in the test, not fixed.

- **teleop-train** (this branch, `teleop-train/`, commits `745390a`, `fd230a8` + wave 3): teleop episodes → the
  training `.zarr.zip` + `package.json` that `rnoro/training`'s `train/` uses as-is.
  - Commands: `teleop_train convert <task>`, then `teleop_train package <task> <dataset>`. Output goes into
    `~/Capstone/data/<task>/`.
  - Training: `train/all.sh <task> <dataset>` (no `--pull`/`--package`).
  - Docs: `teleop-train/{PLAN,DESIGN,DIRECTIONS}.md`.
  - Tests: 340. The env is `~/envs/teleop_train`.
  - Images match the deployment bit for bit (same decoder, same `ImagePrep` @ `ea9f7fb`). `package` refuses a TCP or
    rig other than `deployment/contracts`'.
  - Verified on real `teleop_collect run --mock` episodes, and the zip loads in `UmiTaskDataset` under
    `umi_policy_dunet` and `umi_policy_dunetp`.
  - Policy choice on `rnoro/training`: `train/all.sh` / `submit.sh` `--policy plain|prompting` (`umi_policy_dunet` /
    `umi_policy_dunetp`), passed to the cluster and recorded per run. The default is prompting.

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

- Teleop:
  1. Push `rnoro/teleop-collect-deploy` (all local commits so far).
  - Other branches are read only from this branch's sessions.
  - Train a first teleop dataset (`teleop-train/DIRECTIONS.md`): `teleop_train package <task> <dataset>`, then
    `cd ~/Capstone && train/all.sh <task> <dataset> --policy plain`.
  2. On the PC: the bring-up in `deployment/control_deploy/DIRECTIONS.md`, measuring the workspace box first, then
     the `console` steps.
  3. On the laptop: `teleop_collect/DIRECTIONS.md`. Set the real addresses in `link.yaml` on both machines and in
     `station.yaml`.
  4. First teleop: slow mode, hand on the user stop.
0. Sensor rig (see ROADMAP.md "Status"; checklist in [TODO.md](TODO.md); parts in [PURCHASING.md](PURCHASING.md)):
   - Decide: Panda vs FR3; gripper actuator; buy lenses (2.1 mm + ~1.55 mm fisheye M12 on IMX219); where the 1 kHz
     loop runs; v_max and the success rule for evaluation; whether to commit the new work.
   - Hardware day:
     - data collection: `sensor_rig/docs/first_plugin.md`;
     - arm: `controlSW/franka_deploy/docs/hardware_day.md`.
   - Build the ORB-SLAM3 image on a machine with ≥ 20 GB free (the laptop disk is ~90% full).
1. Watch the v1 videos (`results/block_raise/run_001_2026-10-05/videos/final/`): does the attention follow the prompt?
2. Decide the next real training run (e.g. ~60 epochs, since v1 overfit after ~50), or record more data and use
   `scripts/run_pipeline.sh all ...` to test `process_dataset.sbatch`.
3. Optional: set a real git email on the laptop (`git config --global user.email ...`) so commits link to the GitHub account.
