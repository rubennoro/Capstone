# <NN>_<name>: <one-line title>

<!-- Copy this file into a new experiment folder as README.md (or as <n>_<topic>.md for one part of a bigger
experiment). Keep every heading, in this order; write "none" or "not run" rather than deleting a section.
Every number must come from a raw file under sim_evals/ (or a log) that the "Raw data" section names. Where a number
in sim_deploy/PLAN.md or DESIGN.md disagrees with the raw file, use the raw file and say so under "Analysis". -->

**Status:** planned | running | done | superseded (by ...) · **Date:** YYYY-MM-DD · **Who:** agent / user ·
**Plan ref:** sim_deploy/PLAN.md §..., DESIGN.md "..."

## Question

One or two sentences: what this experiment decides. Phrase it so the conclusion can answer it with yes / no / a
number.

## Why it matters for the real robot

What changes on the Franka (or in training, or in the deployment config) depending on the answer. If the answer is a
sim-only matter, say that here.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `<sha>` (+ uncommitted `sim_deploy/`, `deployment/`: `eval.yaml` `git.sim_deploy` then reads `<sha>-dirty`, so the model sha and the date identify the version) |
| Model | `model_sha256` `<first 8>…` (`build/sim_rig/model.json` = `eval.yaml` `model.sha256`; table top z 0 or 0.05, appearance, ...: see the README's model table) |
| Home / start pose | `--home start` with `q_start_rad [...]` (pose id, TCP O (x, y, z), height above the table, tilt, heading) / `--home deploy` (TCP O (0.307, 0, 0.355), straight down) |
| Eraser placement | `scene/randomization_start.yaml` (`--home start`) / `scene/randomization.yaml` (`--home deploy`), seeded / replay placement of episode `<i>` |
| Stations / overlays | `stations/control_sim.yaml` + `start_pose.yaml` (+ overlay `stations/<file>.yaml` via `SIM_DEPLOY_SIM_PC_OVERLAY`: what it changes; `eval.yaml` does not record it, so put it in the tag) |
| Policy / checkpoint | `hold_still` / `replay:<zarr>#<i>` / `~/Capstone/data/<FOLDER>/policy/<run>` (BP or plain, dataset, episodes, `.ckpt` name: see the README's Checkpoints) |
| Inference settings | precision, diffusion steps, `exec_h` |
| Seeds / episodes | `--seed S`, N episodes (episode n uses `default_rng([S, n])`) |
| Timeout | `--timeout` s (default 60) |
| Sim-only settings | everything that differs from the real robot's config beyond the README's "Sim-only settings" table (v_max, fast homing, look, camera correction ...) |

## Method

What was run, in which order, and what each run is meant to show. Name the tool (script, module, CLI) for every
step and what it measures. Say what counts as a pass before showing results.

## Commands to reproduce

Every command, copy-pasteable, from a fresh terminal (`cd ~/Capstone-main-sim && unset PYTHONPATH` first). Put the
expected run time and anything that must not run at the same time (sim ports, GPU).

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# ...
```

## Results

Tables first (one row per run / pose / episode; columns with units; success as k / n with the Wilson 95 % interval
from `metrics.json`, and the outcome counts), then figures (`![caption](file.png)`, files in this folder if small,
otherwise a path into `sim_evals/`). Quote the raw value, not a rounded paraphrase.

## Analysis

What the results mean, the causes found (with the evidence), and where the docs and the raw data disagree.

## Conclusion

The answer to the Question, in a few lines. What was decided or changed because of it.

## Limitations / sim-vs-real caveats

What in the sim makes this result weaker for the real robot (rendering, contact model, timing, missing clutter,
stand-in values ...).

## Open questions

Numbered; each one says what would answer it.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| eval folder(s) | `sim_evals/<task>/<run>/<YYYY-MM-DD_HHMM>_<tag>/` (also `_baseline/`, `_replay/`) |
| logs | `sim_evals/_logs/<YYYY-MM-DD_HHMMSS>/` (`sim_pc.log`, `sim_pi.log`, `eval.log`) |
| copied into this folder | `file.csv`, `figure.png` (source path) |
