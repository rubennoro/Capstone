# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-10-17; teleop added 2026-11-18; teleop-train 2026-11-20; first real teleop data and training 2026-11-20)

- **Teleop data collection** (this branch, `rnoro/teleop-collect-deploy`): `deployment/teleop_collect/` **ran on the
  real arm on 2026-11-20** and works.
  - What it is: keyboard teleop of the Panda from a browser page on the laptop, recording both Pi streams (MJPEG .mkv)
    and every arm state into `~/CapstoneTeleop/<task>/episode_NNN/`.
  - The PC runs `control_deploy` from `rnoro/final-deployment` @ `b4030f6` (real addresses, `q_start_rad` from the
    simulation in `abba9e8`), never modified from here. `deployment/contracts/` here is its byte copy @ `b4030f6`
    (`dd20222`: gripper_width_max_m 0.04, F_T_TCP settled, PC 192.168.50.186, laptop .227, Pi .59).
    - teleop_collect records with the deployment rig `contracts/rigs/rig_pi_v3_640.yaml`.
  - Start pose: on the PC `h`, then `p` before `teleop_collect run` and before every episode. Collect and deploy with
    the same `q_start_rad`.
  - No software stops outside Franka's limits; the PC's `station.yaml` needs its **teleop block** while collecting
    (PC DIRECTIONS "Teleop settings") and the deployment block before a policy.
  - **Episodes are never deleted automatically (2026-11-20):** Stop arm / Esc, the PC's `s`, the watchdog, a fatal
    problem or Ctrl-C keep the open episode `unsaved` (meta.yaml with outcome unsaved, no COMPLETE, the reason in
    `warnings`); the page's episode list offers save success / save fail / delete (U1 `label_episode`), also at the
    next run. Recording or disk errors never block Save (they go into `warnings`). Verified end to end in `--mock`
    (5 scenarios); the old pytest suite was **not** updated and has tests that still expect deletion.
  - The page shows a labeled key legend (W forward, F down toward the table, ...).
  - Data: `~/CapstoneTeleop/teleoppick1/` (1 episode, eraser) and **`~/CapstoneTeleop/teleop_marker/`: 24 success
    episodes** (19-45 s, 10.1 min; cameras and arm data clean, all start at the same pose).
  - Docs: `teleop_collect/{PLAN,STRUCTURE,DESIGN,DIRECTIONS}.md` and `deployment/control_deploy/DIRECTIONS.md`.
  - The env is `~/envs/teleop_collect`. When another session holds the ports, run things in
    `unshare -rn sh -c 'ip link set lo up && ...'`.

- **teleop-train** (this branch, `teleop-train/`, `745390a`..`0fdec8e`, all pushed): teleop episodes → the training
  `.zarr.zip` + `package.json` that `rnoro/training`'s `train/` uses as-is. Built and checked end to end on mock
  episodes; not yet run on real teleop data.
  - Commands: `teleop_train convert <task>`, then `teleop_train package <task> <dataset>`. Output goes into
    `~/Capstone/data/<task>/`.
  - Training: `cd ~/Capstone && train/all.sh <task> <dataset> --policy plain|prompting` (no `--pull`/`--package`).
  - Docs: `teleop-train/{PLAN,DESIGN,DIRECTIONS}.md`.
  - Tests: 340. The env is `~/envs/teleop_train`.
  - Images match the deployment bit for bit (same decoder, same `ImagePrep` @ `ea9f7fb`). `package` refuses a TCP or
    rig other than `deployment/contracts`'.
  - Verified on real `teleop_collect run --mock` episodes, and the zip loads in `UmiTaskDataset` under
    `umi_policy_dunet` and `umi_policy_dunetp`.
  - Policy choice: `rnoro/training` `ee168f2` (2026-11-20, pushed; `~/Capstone` is on it) adds
    `--policy plain|prompting` to `submit.sh`, `go.sh` and `all.sh` (`umi_policy_dunet` / `umi_policy_dunetp`). It is
    passed to the cluster and frozen with each run. Without it, `TRAIN_CONFIG` decides (behavior prompting by
    default). `--attention` needs prompting.
  - Other branches (`rnoro/training`, `rnoro/final-deployment`, data-work) are read only from this branch's
    sessions: their changes are reported as dependencies, never made here.

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

- Teleop (`rnoro/teleop-collect-deploy`, pushed 2026-11-20):
  1. Training of `teleop_marker` (plain diffusion, account eece5640.202530), submitted 2026-11-20, one package per
     data size (first N episodes, `~/Capstone/data/teleop_marker/package/teleop_marker-nN/`):
     n8 `run_002_2026-11-20`, n16 `run_004`, n20 `run_006`, n24 `run_008` (each after its smoke run 001/003/005/007).
     tmux session `marker` on the laptop waits, fetches and evals each (`train/all.sh teleop_marker --run <run>`;
     rerun that if the laptop slept). Then compare the four runs (data-size curve).
  2. Deploy the best policy on the deployment block (PC station.yaml), same `q_start_rad`.
  3. Update the teleop_collect tests to the keep-unsaved behavior (only when asked).
  - Open: measure `gripper.width_offset_m` on the PC (the closed hand reads ~0 mm around the eraser);
    final-deployment moves → re-sync `contracts/` and the copy pins.
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
