# 02 C2: policies with the fixes C1 points to

**Status:** not run; folded into 03_policy_pipeline W2 (attribution with W1's harness) · **Date:** 2026-11-20 (skeleton; closed 2026-11-20) · **Who:** phase agent C (skeleton: lead agent) ·
**Plan ref:** sim_deploy/PLAN.md §10 phase C2; A2 ([A_baseline](../A_baseline/README.md)) is the baseline

> **Not run.** The camera study was paused on 2026-11-20. C1 never ran, so no fix level was chosen. No `campC2*`
> eval folder, `index.csv` row or `sim_evals/_camera_study/C2/` exists. Its baseline A2 never ran either
> ([A_baseline](../A_baseline/README.md)). PLAN §11 folds the camera work into **W2** (realistic look, camera
> colour / noise / JPEG, attribution with W1's harness; the IMX219 / IMX519 view corrections stay an opt-in
> experiment): [../../03_policy_pipeline/W2_sim_look/README.md](../../03_policy_pipeline/W2_sim_look/README.md). Closed-loop policy evals with the fixes belong to W3 ("every checkpoint x N
> episodes from the start pose, real limits"). The Question and Method below are kept as the plan.

## Question

With the fixes C1 points to, applied one level at a time (realistic look; + IMX219 virtual-camera correction; +
IMX519 crop / zoom), do the runnable checkpoints approach, grasp and lift the eraser from `leftx55_t48_h16` at real
limits, and how much does each level change success / grasp / approach / faults and the first-chunk direction
against A2 at the same seeds?

## Why it matters for the real robot

- A fix that only changes the look tells nothing new about the arm; it makes the sim usable for ranking
  checkpoints and start poses.
- A camera correction that turns retreat into approach is a candidate change for the real deployment (rig file /
  frame prep). It stays sim-only here; D decides whether to recommend it, after A-C.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `<sha at run time>` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Model | per level: `current` `166a0649571a…` only if C1 says geometry only; otherwise B's `real` (or `real+mesh`) build, sha recorded in each eval.yaml |
| Home / start pose | `--home start`, `leftx55_t48_h16` (as A2) |
| Eraser placement | `scene/randomization_start.yaml`, A2's seed |
| Stations / overlays | `stations/control_sim.yaml` + `start_pose.yaml`, no v_max overlay (real limits); the look build for the sim Pi and sim PC; the IMX219 correction (`sim_deploy/corrections/rig_pi_v3_640_corr.yaml`, per B) and the IMX519 option (`sim_deploy/corrections/main_crop.py`, per B) as sim-only switches, default off. How `run.sh` selects them is not in place yet (today it takes only `SIM_DEPLOY_SIM_PC_OVERLAY`, `SIM_DEPLOY_SIM_PC_STATION`, `SIM_DEPLOY_PY`, `SIM_DEPLOY_START_TRIES` / `_WAIT_S`, `SIM_DEPLOY_MAX_RESTARTS`): to be added and documented by the phase agent |
| Policy / checkpoint | every runnable checkpoint (14, [study overview](../README.md), "Runnable checkpoints") |
| Inference settings | as A2 (BP `--precision bf16 --diffusion-steps 8 --exec-h 12`; plain as A2 recorded) |
| Seeds / episodes | 3 episodes per checkpoint and fix level, **the same `--seed` as A2** |
| Timeout | as A2 |
| Sim-only settings | the look; the IMX219 / IMX519 corrections; everything else as A2 |

Real limits: v_max 0.25 m/s, w_max 1.0 rad/s, cone 60 deg, box [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70].

## Method

For each fix level C1 names (in order: realistic look; + IMX219 correction; + IMX519 correction; levels C1 rules out
are skipped and listed here as "not run"), every checkpoint x 3 episodes through `sim_deploy/run.sh` with `--tap`.
The same per-episode and per-checkpoint rows as A2 (`sim_evals/_camera_study/A/tools/a2_rows.py`: outcome, grasp /
lift / success, faults with rule and time, approach `d_start_cm` / `d_min_cm`, first chunk toward the eraser and along
the TCP z), and the difference to A2 per checkpoint (same seeds = same eraser poses).

## Commands to reproduce

To be confirmed by the phase agent (the switches for the look build and the corrections do not exist in `run.sh`
yet):

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# BP
<LOOK/CORRECTION SWITCHES> sim_deploy/run.sh --policy ~/Capstone/data/PICK_TEST1/policy/<RUN> --episodes 3 --seed <A2_SEED> \
  --timeout <A2_T> --tap --precision bf16 --diffusion-steps 8 --exec-h 12 --tag campC2_<level>
# plain diffusion
<LOOK/CORRECTION SWITCHES> sim_deploy/run.sh --policy ~/Capstone/data/<TASK>/policy/<RUN> --episodes 3 --seed <A2_SEED> \
  --timeout <A2_T> --tap --tag campC2_<level>
~/envs/sim_deploy/bin/python sim_evals/_camera_study/A/tools/a2_rows.py campC2_<level> sim_evals/_camera_study/C2/<EP>.csv sim_evals/_camera_study/C2/<CKPT>.csv
```

(tag names `campC2_<level>` are the lead's suggestion; the phase agent sets them.)

## Results

Not run (study paused 2026-11-20, folded into 03 W2).

## Analysis

Not run (study paused 2026-11-20, folded into 03 W2).

## Conclusion

Not run (study paused 2026-11-20, folded into 03 W2).

## Limitations / sim-vs-real caveats

Not run (study paused 2026-11-20, folded into 03 W2).

## Open questions

Not run (study paused 2026-11-20, folded into 03 W2).

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| eval folders | `sim_evals/<task>/<run_folder>/<date>_<time>_<C2 tag>/` |
| tables | `sim_evals/_camera_study/C2/` (to be created by the phase agent) |
| index rows | `sim_evals/index.csv` (C2 tags) |
| logs | `sim_evals/_logs/<date>_<time>/` |
| copied into this folder | none (not run) |
