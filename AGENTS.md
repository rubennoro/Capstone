# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-11-20, branch `rnoro/final-deployment`)

- **Deployment on the real Panda:** Pi 5 (cameras) → laptop (`deployment/policy_deploy`: frames, policy, manager) →
  PC (`deployment/control_deploy`: franka_control). It runs end to end. The latest commit is `f632576`, pushed.
  - How to run trials: [deployment/policy_deploy/DIRECTIONS.md](deployment/policy_deploy/DIRECTIONS.md), "Running trials".
    - It starts with a U/F/D/RUN block to paste into both terminals and lists all 14 trained policies.
    - It covers recording each trial with `probe --live 10 --save` plus a NOTES.md.
  - Trials so far, where they are and what happened: [deployment/policy_deploy/TRIALS.md](deployment/policy_deploy/TRIALS.md).
    - 9 trials on 2026-11-20, from run_022, run_006, run_003 and run_002 (the double-62 candidate).
    - The recordings are local only, under `~/Capstone/data/<task>/policy/<run>/deploy/`.
    - Each trial folder also has a video, made by `python -m policy_deploy.frames.trial_video`.
  - Real addresses: PC 192.168.50.186, laptop 192.168.50.227, Pi 192.168.50.59.
  - `frames.yaml` is `settled: true`, with F_T_TCP +45° about flange z and z 0.23565 m.
  - The gripper opens 0 to 0.04 m.
  - Added on the arm day, in `b4030f6`:
    - a laptop workspace clamp (`workspace_clamp_m 0.002`, using the PC's box);
    - `GripperMap.to_model` clamps the width (no more negative widths);
    - `warmup_gripper_width_m 0.04`;
    - `probe --live` and `pi_collect --show` (both need the torch env's python).
  - **Open problem:** every PICK_DOUBLE policy aims 2–8 cm below the table. That causes dozens of clamps, and every
    episode ends on a user stop.
    - Suspects: the start pose, or the table height. `homing.q_start_rad` in `control_deploy/station.yaml` is the
      sim's value and was never checked on the arm. The policies act relative to the start pose.
    - Once, `pc_probe --start-pose` "at p" read z −0.007 and tilt 35.9°, but the arm was not actually at the start pose then.
  - Also seen: the ultrawide obs pair spacing was once 100 ms (main 66.7 ms); one rot6d perpendicularity failure with run_006.
- Other branches each have their own worktree (`~/Capstone-<name>`): training, umi-processing / data-work, teleop-collect,
  main-sim. Treat them as read only from here.
- Older work on `main` (block_raise runs, the Jetson sensor rig, the iPhone pipeline) is superseded: no iPhone, and the
  rig is the Pi 5 one.

## Next time

1. Fix the start pose, on the arm:
   1. With the arm low, measure fingertips to table with a ruler. Is the table at base z ≈ 0?
   2. Press `h`, then `p`, then run `pc_probe --start-pose`.
   3. If it is wrong, hand-guide a start about 16–20 cm above the table and tilted about 48°.
   4. Paste the printed `q_start_rad` into the PC's `control_deploy/station.yaml`.
   5. Commit, push, and pull on the laptop.
2. Rerun run_002 (double-62) and compare the clamp counts with [TRIALS.md](deployment/policy_deploy/TRIALS.md).
   Add a row for each new trial.
3. Optional ideas, offered but not started:
   - measure the start heights of the pick1-double-v2 demos;
   - add a `probe --video` option;
   - look into the 100 ms ultrawide pair spacing.
