# 02 C1: attribution: geometry or appearance?

**Status:** not run; folded into 03_policy_pipeline W2 (attribution with W1's harness) · **Date:** 2026-11-20 (skeleton; closed 2026-11-20) · **Who:** phase agent C (skeleton: lead agent) ·
**Plan ref:** sim_deploy/PLAN.md §10 phase C1; §6 "P3" table row G7; DESIGN.md "P3/P4" (G7 offline first chunk)

> **Not run.** The camera study was paused on 2026-11-20 before C1 started. `sim_evals/_camera_study/C1/` does not
> exist and no forward pass was made over B's sets (B3 rendered none). PLAN §11 folds C1 into **W2** ("attribution
> with W1's harness: sim vs real first-chunk response per checkpoint"): [../../03_policy_pipeline/W2_sim_look/README.md](../../03_policy_pipeline/W2_sim_look/README.md). W1
> (`sim-experiments/03_policy_pipeline/W1_offline/`) supplies the harness that feeds real training episodes through
> the copied laptop pipeline. The Question, Method and decision logic below are kept as the plan that W2 inherits.
> Inputs that exist: B1's real reference inputs (`sim_evals/_camera_study/B/reference/`, 15 frames, see
> [B's Status at pause](../B_geometry_appearance/README.md)) and P3's `sim_evals/_start_pose_sweep/p3/` logs.

## Question

With model forward passes on fixed images: which of {handheld, mount, corrected mount} x {current look, realistic
look} make the policies' first chunk point toward the eraser like the real training frames do, and which make it back
away? Is it the camera **geometry** (a real-world problem), the **appearance** (a sim-only problem), both, or
neither? And per camera: the IMX219, the IMX519, or both?

## Why it matters for the real robot

The verdict decides what C2 runs and what D can recommend. Geometry: the real arm's mounted cameras see the same
offset, and the fix (IMX219 virtual-camera tilt + zoom, IMX519 crop / zoom, or a mount change) is a real deployment
change, recommended only after C2 and D. Appearance only: nothing changes on the arm; the sim needs the realistic look
before it can rank policies or start poses.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `<sha at run time>` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Model | B's builds: `current` = `sim_deploy/build/sim_rig/` `166a0649571a…`; `real`, `real+mesh` = B's look builds (shas in B's README) |
| Home / start pose | images: B's sets (`handheld`, `mount`, `mount_start` at `leftx55_t48_h16`); low-dim input: the start (eef_pos 0, rot6d identity, as at the first observation of an episode) |
| Eraser placement | as in B's sets (per reference frame, `T_O_tag` in `sim_evals/_camera_study/B/layout.json`) |
| Stations / overlays | the copy's policy settings from an A2 eval's `station.yaml` per checkpoint (as P3's tool loads them) |
| Policy / checkpoint | every runnable checkpoint (14, [study overview](../README.md), "Runnable checkpoints"); at least run_006 (P3's reference) and PICK_DOUBLE run_002 (the deployment candidate) if time is short |
| Inference settings | to be fixed by the C1 agent: P3 ran CPU fp32, 8 diffusion steps, 5 samples per case; BP prompt = PICK_TEST1 `pick1-v1` index 14 (P3 log) |
| Seeds / episodes | no episodes; samples per image set and checkpoint: to be fixed (P3: 5) |
| Timeout | none |
| Sim-only settings | looks `real` / `real+mesh`; corrections from B (`sim_deploy/corrections/rig_pi_v3_640_corr.yaml`, `sim_deploy/corrections/main_crop.py`), applied to images only |

Reference images: B1's real 224 inputs of the 15 frames (PICK_TEST1 `pick1-v1` indices 14, 2, 9, 19, 25 x rows
`first`, `approach_1of3`, `approach_2of3`); P3's `sim_evals/_start_pose_sweep/p3/ep152_ref.npz` for index 14.
Runs "on the chain" in PLAN §10's sense: it needs the GPU (or many CPU-hours), so it waits for A2 to finish and blocks
C2 until it is done.

## Method

1. For each checkpoint x image set (looks `current` / `real` (/ `real+mesh`) x poses `handheld` / `mount` /
   `mount_start` x correction none / IMX219 / IMX219 + IMX519) and for the real frames, one forward pass per sample
   with the same low-dim input, as `sim_evals/_start_pose_sweep/tools/offline_first_chunk.py` does.
2. Read the first chunk: its last step along the TCP z axis (toward the fingertips; run_006 on episode_000152's
   training images: +5.6 cm; the demos: +5.7 cm in 0.75 s) and toward the eraser (A2's projection).
3. Per camera: repeat with P3's swap (sim IMX519 + real IMX219 and the reverse, `SPLIT=1`) so the IMX219 and IMX519
   effects are read separately.
4. Apply the decision table in the [study overview](../README.md) ("Decision logic"). The thresholds for
   "approaches" / "rejects" are written here **before** the passes are run (lead's proposal: approaches = mean
   first chunk toward the fingertips / eraser >= half the real-frame value for that checkpoint and frame; rejects = <=
   0; in between = reported as such).
5. Output: the verdict and the list of fixes for C2 (realistic look; + IMX219 correction; + IMX519 correction).

## Commands to reproduce

To be confirmed by the phase agent. P3's tool (template; a C1 tool that takes B's image sets is still to be written):

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
cd deployment/policy_deploy
~/envs/sim_policy_deploy_torch/bin/python ../../sim_evals/_start_pose_sweep/tools/offline_first_chunk.py <A2_EVAL>            # sim vs training images
SPLIT=1 ~/envs/sim_policy_deploy_torch/bin/python ../../sim_evals/_start_pose_sweep/tools/offline_first_chunk.py <A2_EVAL>    # camera swap
# C1 over B's sets: (to be written by the phase agent), outputs in sim_evals/_camera_study/C1/
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
| forward-pass tables, logs | `sim_evals/_camera_study/C1/` (to be created by the phase agent) |
| input image sets | `sim_evals/_camera_study/B/sets/`, `sim_evals/_camera_study/B/reference/` |
| P3 reference | `sim_evals/_start_pose_sweep/p3/` (`offline.log`, `offline_split.log`, `ep152_ref.npz`) |
| copied into this folder | none (not run) |
