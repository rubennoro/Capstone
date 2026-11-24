# 02 D: review of A-C and the recommendation

**Status:** not run; folded into 03_policy_pipeline W2 (attribution with W1's harness) · **Date:** 2026-11-20 (skeleton; closed 2026-11-20) · **Who:** reviewer agent D (skeleton: lead agent) ·
**Plan ref:** sim_deploy/PLAN.md §10 phase D ("Recommendation only after the complete sim eval (A-C)")

> **Not run.** The camera study was paused on 2026-11-20 with A1 partial, B1 done and A2, B2-B5, C1, C2 not run.
> There was nothing for D to review, so no study table exists in `sim_evals/_camera_study/`. No real-robot change was
> recommended. PLAN §11 folds the work into **W2** (each workstream has its own reviewer gate):
> [../../03_policy_pipeline/W2_sim_look/README.md](../../03_policy_pipeline/W2_sim_look/README.md). The closing record of what ran is in the [study overview](../README.md) and
> [A_baseline](../A_baseline/README.md). The Question and Method below are kept as the plan.

## Question

Do A1, A2, B, C1 and C2 hold up against their raw files, and on that evidence: is the policies' retreat from the
start pose caused by the start pose, the camera geometry, or the sim's appearance? What, if anything, should change
on the real deployment (start pose, IMX219 / IMX519 correction, mount), and what stays sim-only?

## Why it matters for the real robot

This is the only phase that may recommend a real-robot change (PLAN §10). Until then `rnoro/final-deployment` stays
untouched and every correction is sim-only.

## Setup

| Item | Value |
|---|---|
| Inputs | the phase READMEs ([A_baseline](../A_baseline/README.md), [B_geometry_appearance](../B_geometry_appearance/README.md), [C1_attribution](../C1_attribution/README.md), [C2_fixes](../C2_fixes/README.md)) and their raw data under `sim_evals/_camera_study/` and the eval folders they name |
| Runs | none of its own (re-computes tables from raw files only) |
| Checkpoints | the 14 in the [study overview](../README.md) |
| Sim-only settings | as listed by A-C |

## Method

1. For each phase: every number in its README re-read from the raw file it names (eval.yaml / episodes.jsonl /
   metrics.json / the phase CSVs); the seeds of A2 and C2 checked equal; the model sha and station of each eval
   checked against the phase's Setup; "Docs vs raw data" items collected.
2. One table in `sim_evals/_camera_study/` (per checkpoint: A2 vs each C2 level; success / grasp / approach / faults,
   first-chunk direction; plus A1's replay summary and C1's verdict per cell), generated from the raw files.
3. The recommendation: for the real arm (start pose stands or not; correction or mount change, or nothing) and for
   the sim (which look and corrections become defaults for future benchmarks), each line citing the rows it rests on.

## Commands to reproduce

To be confirmed by the reviewer (the tool that builds the table is to be written).

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# (to be written by the reviewer): build sim_evals/_camera_study/<TABLE> from A/, C1/, C2/ and the eval folders
```

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
| the study table | `sim_evals/_camera_study/<TABLE>` (name to be set by the reviewer) |
| inputs | `sim_evals/_camera_study/{A,B,C1,C2}/`, the A1 / A2 / C2 eval folders |
| copied into this folder | none (not run) |
