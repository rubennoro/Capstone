# AGENTS.md: where we're at

Running status between working sessions. Update it at the end of each conversation: replace "Current state" and
"Next time" rather than appending history. How to run things: [DIRECTIONS.md](DIRECTIONS.md). How the repo is built:
[DESIGN.md](DESIGN.md).

- Cluster login: `ssh noroian.r@login.explorer.northeastern.edu` · transfers: `noroian.r@xfer.discovery.neu.edu`
- Cluster repo clone: `~/CapstoneProject/Capstone` · datasets: `~/CapstoneProject/datasets/` · job logs: `~/CapstoneProject/logs/`
- Working style: explain cluster commands before running them, and ask before anything that changes state (submitting,
  cancelling, deleting). Read-only checks are fine. Auto mode blocks writing files on the cluster over ssh: commit the
  script and `git pull` on the cluster, or have the user run `scp`.

## Current state (2026-11-24)

- **Sim tests = one rollout at a time, visually in the UI, step by step with the user** (2026-11-24). No headless or
  multi-rollout runs from the agent; comprehensive evals go through branch `rnoro/sim-cluster` on the cluster, never
  here. Protect the laptop GPU: one sim at a time (the UI server's live render is load too), load < 3, AC,
  `powerprofilesctl set performance`. Focus of `rnoro/main-sim` = a good UI.
- **Branch `rnoro/main-sim` (worktree `~/Capstone-main-sim`), all committed and pushed:** `deployment/` = upstream
  final-deployment f632576 + 25 sim-only files (20 + 4 tests + `launcher/use_policy.py`), `sim_deploy/`,
  `sim-experiments/`. Plan [sim_deploy/PLAN.md](sim_deploy/PLAN.md) §11; design `sim_deploy/DESIGN.md`; how to run
  `sim_deploy/DIRECTIONS.md`; UI design `sim_deploy/ui/DESIGN.md` (U6 redesign 2026-11-23).
  - UI: settings sections, Progress panel from the driver's `[eval] step n ...` lines, results table; words in
    `ui/static/help.js`, step model in `steps.js`; `--stub` = fake runner/live/catalog. Page default seed 523 (rollout
    0: eraser ~(0.534, -0.068), the in-range spot nearest the teleop demos; API/run.sh default stays 0).
  - **Teleop gripper fix (dc9bca3):** `policy_deploy use` gives a run whose dataset package.json says
    `source.kind: teleop_collect` an identity `gripper_map` (0-0.040 m). Before, the handheld map turned the open
    39.9 mm into 27 mm (< the 30 mm eraser) and showed the policy 90 mm. Upstream final-deployment has the same gap.
- **Teleop findings (teleop_marker run_002/006/008 = n8/n20/n24, 2026-11-23/24):**
  - Demos (raw `~/CapstoneTeleop/teleop_marker`): same start pose as the sim, object on one line (TCP at close x ~0.53,
    y -0.16..-0.03 = 0-13 cm ahead, mostly 3-8 cm), close at 5-13 s at table height, heading kept (<= 2.4 deg), 5 cm/s.
    The sim's random range (from the handheld pick1 demos: 9-22 cm ahead, 0.3-3.5 cm aside) barely overlaps.
  - Seed 3 (outside the demo line): approaches, closes in mid-air ~12 cm up or descends 9.5 cm off the eraser.
  - Eraser pinned at the median demo spot (0.530, -0.104) via `--random <file>`, 3 x 120 s, n24: all descend to the
    table in ~12 s and put the fingers around the eraser's near end, then never close (100 s waiting). Seed 523 in
    the UI: grippers close, still no pick. Likely the close decision is visual (sim look); placement mattered a lot.
  - Shake: new chunk every ~210 ms, ~half the waypoints skipped late (policy p50 ~200 ms on the laptop).
  - **Scoring quirk:** `grasped` is contact-based and fires with the fingers open (demo-spot rollout 2: gap 40 mm all
    along) → grasp rates overstate; flag for the cluster batch.
- **Tests (2026-11-23, sim envs):** sim-specific failures fixed; remaining are upstream's (4 contracts + 6 policy
  stale values; ~105 frames/launcher tests need `tests/frames/data/` + `tests/launcher/data/`, gitignored upstream).
- Envs: `~/envs/sim_deploy`, `~/envs/sim_policy_deploy{,_torch}`. Never wait-loop on `pgrep -f` with a pattern the
  loop's own command line contains (use `[s]im_deploy` or a pid).

## Next time

1. Teleop: the user judges whether the sim's policy view at the grasp moment looks like the teleop camera (the
   close-decision question); then look fixes vs more varied teleop data (object position in both directions) vs an
   earlier checkpoint (val loss lowest ~epoch 20, saved = 59).
2. Before merging into `rnoro/sim-cluster`'s batch: teleop checkpoints are not in `eval/start_sets/w3.yaml` (driver
   refuses `--starts set` for them) and the random range is the handheld one → decide teleop starts; the cluster's
   data layout must have `data/<folder>/package/<dataset>/package.json` or the gripper fix falls back silently (note
   only); fix or flag the open-gripper `grasped`. Faster cluster inference changes behaviour: compare same-machine runs.
3. Hand off to the final-deployment owner (user's call): `sim_deploy/ports/rot6d_gram_schmidt.md` (local copy, no
   pose_math import), the teleop gripper_map gap in `use_policy.py`, the gitignored test data.
4. Open: W1 timing bench (sim cycle ~200 ms vs real ~130 ms); fewer diffusion steps to cut the shake; image fixes for
   the finger gap are user decisions; confirm real-trial labels.
