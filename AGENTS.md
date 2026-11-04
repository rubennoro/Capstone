# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-11-03)

- **The phone is dropped.** Focus: the custom sensor rig + custom printed fingers on the Franka **Panda**.
  - Real rig hardware, from the C++ `physicalSetup/` on branch `rnoro/jetson-sensors` (first hardware session 2026-10-31):
    IMX519 AF main, IMX219 160° ultrawide, LSM6DSOX IMU on I2C, Jetson Orin Nano.
  - The Python `sensor_rig/`, ROADMAP and PURCHASING still describe the old B0180/B0179/ICM-42688-P plan, so they are stale.
- **Branch `rnoro/main-sim`** (local, not pushed; = `main` + 7 commits): a phone-free MuJoCo closed loop.
  - Covers the Panda + Franka Hand + your Onshape finger kit (default; box fingers with `--box`) + the rig mount.
    The policy slot holds a scripted pick-and-place stand-in.
  - **The command:** `controlSW/franka_sim/run_viewer.sh [POSES] [--box] [--set ...]`. It opens the MuJoCo viewer;
    the report is in `~/sim_runs/scripted_<time>/summary.html`. Details: `controlSW/franka_sim/README.md`.
  - **Verdict = safety only:**
    - rules: contact, reflex, workspace box, 60° cone, joint margin 5°, EE speed ≤ 1.1 × v_max and angular speed ≤ 1.1 × w_max;
    - the task outcome (none / grasped / lifted / placed) is reported separately and never counts as failure;
    - criteria are in `controlSW/configs/run/criteria/pick_place.yaml`; scene `station/scene/pick_place_table.yaml`
      (place target 0.45, 0.15).
  - **Verified 2026-11-03:**
    - box runs and the kit run: Safety 3/3, placed 3/3, live judge and offline ground truth agree;
    - EE speed max 0.15–0.21 m/s (limit 0.275; the lag snap-back still shows as a spike); joint margin min about 23°;
    - forced `--set criteria.safety.joint_margin_deg=60` → FAIL `joint_margin` at once;
    - `policy-contract` PASS; `tests/test_scripted_policy.py` 4 passed.
  - **Stand-in notes:** peak command speed is 0.6 × limits (0.9 let the impedance lag, ~21 mm, snap back to 0.29 m/s).
    Chunks are relative to the task's commanded anchor.
  - **Known:** the offline ground truth reads 25 Hz lines, so it can miss short speed spikes the 100 Hz judge sees.
  - **Placeholders still open:**
    - rig camera K and extrinsics: still the virtual-iPhone K, and `C_T_TCP` is a copy of iPhone17,1;
    - finger tag positions (kit camera-check is 23 mm off per side);
    - adapter CAD (D9), payload mass.
  - `sim-eval` is kept as is (committed `187c247`, 1 ahead of origin) but no longer developed. Its Stages
    1/1b/2a/2b, sweeps, camera-check and iPhone paths were not brought over.
- **Environment:** one env, `~/envs/env_sim`, built by `controlSW/franka_sim/setup_env.sh` (py3.10, mujoco 3.3.0, no torch).
  - `env_bp`, `env_realenv`, the `~/.local` py3.10 torch and the pip/conda caches were deleted (disk now about 32 GB free).
  - A learned-policy (torch) env must be rebuilt when needed.
- **Training data/runs** (unchanged; all iPhone-era, so not deployable on the rig):
  - dataset `firstreal-single` (45 demos) at `~/CapstoneProject/datasets/block_raise/firstreal-single/`;
  - `results/block_raise/run_001_2026-10-05/` (best val action MSE 0.000148 at epoch 50; slim checkpoint gitignored);
  - pipeline `scripts/run_pipeline.sh` + `scripts/cluster/*` tested end to end (job `10878177`); `process_dataset.sbatch`
    is still never run.
- Plots: the user is happy with them as they are. Don't change `scripts/wandb_offline_to_csv.py` unless asked.

## Next time

1. Watch `run_viewer.sh 3` (kit) yourself.
   - Decide whether to push `rnoro/main-sim`, and whether to merge it to main.
2. Replace the sim placeholders with the real rig:
   - IMX519/IMX219-160 intrinsics and mount (camera K, `C_T_TCP`, `C_main_T_C_uw`);
   - measured finger tags and adapter;
   - weighed payload.
3. Policy: a plain diffusion policy first (Behavior Prompting is a stretch goal).
   - Needs rig-recorded pick-and-place demos and an inference env with torch.
   - Swap it in with `--policy` (`controlSW/docs/adding_a_policy.md`).
4. Rig software: C++ `physicalSetup/` Wave 2 (`ec-record`, `ec-calibrate`) on `rnoro/jetson-sensors`.
   - Reflash the Jetson to JetPack 6.2 for the IMX519 (back up the NVMe first).
   - Clean up the stale hardware entries in ROADMAP/PURCHASING/TODO.
5. Add a robot-feasibility filter for demos (speed ≤ limits, reach, tilt cone, width) to the data pipeline.
