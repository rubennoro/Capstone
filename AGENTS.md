# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-11-23)

- **Local = individual policy tests only (user, 2026-11-23):** the UI and the whole sim setup stay usable on the
  laptop for testing one policy at a time (a few rollouts). Comprehensive evals (batches, all checkpoints x start set)
  go through branch `rnoro/sim-cluster` on the cluster, never here. Focus of `rnoro/main-sim` = a good UI.
- **Branch `rnoro/main-sim` (worktree `~/Capstone-main-sim`), all committed and pushed to origin:** `deployment/` =
  upstream final-deployment f632576 + 24 sim-only files (20 + 4 tests), `sim_deploy/`, `sim-experiments/` write-ups.
  Plan [sim_deploy/PLAN.md](sim_deploy/PLAN.md) §11; design `sim_deploy/DESIGN.md`; how to run
  `sim_deploy/DIRECTIONS.md`; UI design `sim_deploy/ui/DESIGN.md` (U6 redesign 2026-11-23).
  - UI: settings sections (Rollouts, Eraser starts random | W3 start set, Scene: arm start + look, Policy settings
    folded), Progress panel from the driver's `[eval] step n ...` lines, results table; words in `ui/static/help.js`,
    step model in `steps.js`; `--stub` runs the page with fake runner/live/catalog (checked on :8771 2026-11-23).
  - Eval rules (unchanged, 2026-11-22): trial = rollout; success = lift (centre ≥ settled + 5 cm held ≥ 1 s, clear of
    the table), score v4; start set `eval/start_sets/w3.yaml` (training demos 0-9 + 8 real-trial poses); eraser 30 mm.
  - Pre-launch findings (`sim-experiments/03_policy_pipeline/W3_closed_loop/README.md`): the policies fail on the
    images (robot fingers ≤ 40 mm look like a training grasp → back-off). W3 batch `w3_2026-11-22_1834` (paused
    2026-11-22, 4 runs done, every fault = PC tilt gate; homing after a timeout can hit a joint-7 reflex,
    link5/rig_cover_col) is not to be resumed locally (comprehensive eval → sim-cluster).
- **Tests (2026-11-23, sim envs):** sim-specific failures fixed (policy may not import manager / scipy: local
  Gram-Schmidt in `base_policy.py`, numpy Rodrigues in `replay_policy.py`; tests follow the replay kind and the rot6d
  repair). Remaining failures are upstream's: 4 contracts + 6 policy tests assert stale values (PC IP .10, gripper 0.08,
  TCP settled); ~105 frames/launcher tests need `tests/frames/data/` + `tests/launcher/data/`, gitignored (`data/`)
  in rnoro/final-deployment, so absent from the copy and from any fresh clone of upstream.
- Envs: `~/envs/sim_deploy`, `~/envs/sim_policy_deploy{,_torch}`. Local test runs need AC + `powerprofilesctl set
  performance` + load < 3 (else sim Pi start > 30 s). Never wait-loop on `pgrep -f "sim_pi|franka_control"`.

## Next time

1. UI work on this branch (the user's focus); check with `--stub` / static checks; single-policy tests through the UI
   or `run.sh` are fine (ask before starting one).
2. Comprehensive evals: through `rnoro/sim-cluster` (not this worktree). Cluster caveat stands: faster inference changes behaviour
   in real time, so compare only runs from the same machine (or latency-matched).
3. Hand off `sim_deploy/ports/rot6d_gram_schmidt.md` (now: local Gram-Schmidt copy, no pose_math import) and the
   upstream test-data gitignore finding to the final-deployment owner (user's call).
4. Open: W1 timing bench (sim cycle ~200 ms vs real ~130 ms); image fixes for the finger gap (pixel edit / finger
   mask + retrain / teleop data on the arm) are user decisions; confirm real-trial labels.
