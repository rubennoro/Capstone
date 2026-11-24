# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-11-22, 23:40)

- **Branch `rnoro/main-sim` (worktree `~/Capstone-main-sim`, nothing committed since 3633fcb): `sim_deploy/`, a sim that
  stands in for the Pi and the PC so the real laptop code runs against it.** Plan: [sim_deploy/PLAN.md](sim_deploy/PLAN.md)
  §11; design `sim_deploy/DESIGN.md`; how to run `sim_deploy/DIRECTIONS.md` (§3 Evaluate); write-ups
  `sim-experiments/03_policy_pipeline/` (one README per W0-W4).
  - W0 done (copy = upstream f632576 + 20 sim-only files; the 20th = today's rot6d Gram-Schmidt in
    `policy/base_policy.py`, handoff for final-deployment `sim_deploy/ports/rot6d_gram_schmidt.md`, not ported yet).
    W1 done except the timing bench. W2 done, gate not met; 30 mm rerun 14/14 robot cell = real arm.
  - Sim eraser 30 mm wide everywhere (user decision; `sim_deploy/eraser.py`, make_stations --check). Upstream repo/ref
    via `SIM_DEPLOY_UPSTREAM_REPO` / `SIM_DEPLOY_UPSTREAM_REF` (`sim_deploy/upstream.py`).
  - UI: look selector (robot default / train / current), bf16/8/12, upstream start pose check, runs recorded as `ui` and
    scored on finish, frames button.
  - Eval split (all uncommitted, user decisions 2026-11-22): an eval trial is a **rollout** everywhere (code, CLI
    `--rollouts`, folders `rollouts/NNN/`, schema; "episode" = training demo only; old data migrated). Success = lift
    only: eraser centre ≥ settled + 5 cm held ≥ 1 s AND clear of the table (lowest corner ≥ table + 1 cm); grasp
    reported, `lift_without_grasp` flagged (score v4). Starts = start set `eval/start_sets/w3.yaml`: shared training
    demos 0–9 (one held out per checkpoint, recorded) + 8 real-trial eraser poses. Sim payload 0 (= real PC). Batch
    marks rollouts overlapping a GPU fault window invalid; run.sh/batch sweep orphaned processes. Old smoke/W3 results
    deleted (user OK).
  - Pre-launch findings (details: `sim-experiments/03_policy_pipeline/W3_closed_loop/README.md` "Pre-launch findings"):
    code path verified identical to training (obs + actions, also final-deployment); the policies fail on the images:
    robot fingers open ≤ 40 mm look like a training grasp (→ lift/back-off), and the robot cell's look holds them still
    at the start. Tilt cone kept at 60° (75° experiment: 0 successes, rig into the table). User: no physical changes.
- **W3 batch `w3_2026-11-22_1834` PAUSED by the user 23:36 (clean, between pieces)**, user is considering the
  cluster's GPUs instead. Resumed 21:08 after the reboot with slower pacing added to the frozen plan.yaml
  (piece_rollouts 3, gpu_cool_c 58, gpu_pause_s 120, gpu_cool_max_s 1200): no GPU fault, peak 79 C (temp guard
  `sim_evals/_logs/w3_temp_guard.sh` would switch to 2-rollout pieces > 80 C; never fired; stopped). The latched-fault
  sim restart works (0 `no_accepted_waypoints` since). Log `sim_evals/_logs/w3_batch_resume_2026-11-22_2108.log`.
  - Done + valid: run_014, run_016, run_020 (18/18 faults each, 0 grasps), run_022 (10 faults, 6 timeouts, 2 invalid).
    run_024 at 6/18. Pending retries (invalid = PC latched before the fix): run_006 [2-7,11,14], run_012 [3-17],
    run_022 [5,13]. Untouched: run_028/030/031, PICK_DOUBLE run_002-005.
  - Findings so far: every fault = PC tilt gate (62-65.5° > 60+2), arm closest to the eraser at t ≈ 0 then moves
    away (back-off); time to fault 5-11 s (014, 020), up to 28 s (016), 19-46 s (022). **New structural failure:**
    homing after a 60 s timeout hits a joint-7 reflex, contact link5/rig_cover_col (run_022 rollouts 5, 13): the
    home path swings the rig cover into link 5.
- Envs: `~/envs/sim_deploy`, `~/envs/sim_policy_deploy{,_torch}`. Runs need AC + `powerprofilesctl set performance` + load
  < 3 (else sim Pi start > 30 s). Never wait-loop on `pgrep -f "sim_pi|franka_control"` (matches itself).

## Next time

1. Batch `w3_2026-11-22_1834` (state `sim_evals/_batches/w3_2026-11-22_1834/`) is paused; the user decides laptop vs
   cluster. Laptop: checks (nvidia-smi, no `NVRM: Xid` this boot, load < 3, AC, performance), then
   `~/envs/sim_deploy/bin/python -m sim_deploy.eval batch --resume w3_2026-11-22_1834` (`--dry-run` first: 11 runs,
   ~5-6 h). Cluster caveat (told the user): rollouts run in real time, so an A100's much faster inference (laptop
   policy p50 ~223 ms, margin < 0) changes behaviour → not comparable with the laptop runs or the real arm unless all
   14 checkpoints are rerun there, ideally with latency matched; also needs the commit/push or scp of sim_deploy/,
   cluster envs, headless EGL, load_max/powerprofilesctl gates relaxed, fixed localhost ports (one job per node),
   checkpoints + upstream final-deployment repo on the cluster. When done: report
   (`sim_evals/_reports/w3_2026-11-22_1834/`) plus per-checkpoint ranking by time to fault, max tilt, closest approach,
   freeze vs drift, grasp attempts (tilt/contact analysis: `sim-experiments/03_policy_pipeline/W3_closed_loop/tools/rollout_motion.py EVAL...`), then
   **flag every structural failure** (tilt-gate stops, freezing, z-ceiling climbs, rotation lag, anything new).
2. Walk the user through the changes file by file and commit locally (no push without asking), incl. `deployment/`
   (untracked so far) so sim edits have history. Confirm real-trial labels.
3. Hand off `sim_deploy/ports/rot6d_gram_schmidt.md` to the final-deployment owner (after W3; user's call).
4. Open: W1 timing bench (sim cycle ~200 ms vs real ~130 ms); image fixes for the finger gap (pixel edit / finger
   mask + retrain / teleop data on the arm) are user decisions.
