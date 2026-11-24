# 03 W1: the deploy path = training, offline, on real images (no rendering)

**Status:** steps 1, 2, 3 and 5 done; step 4 **waiting: bench rerun under the performance power profile** (settings
bf16 / 8 steps / `exec_h` 12 stay provisional) · **Date:** 2026-11-20/21 · **Who:** W1 coordinator (harness,
prompting, open-loop, real-trials subagents) + reviewer · **Plan ref:** sim_deploy/PLAN.md §11 (W1 row); §6 G7; DESIGN.md
"E1: as built"

## Question

Fed real training episodes (real images, real low-dim state), does every one of the 14 checkpoints, loaded and run
through the copied laptop pipeline, predict what training says it should (open-loop error vs the recorded actions, and
vs the training code's own offline eval), and which precision / diffusion steps / `exec_h` does each need to fit the
step budget? Output: yes / no per checkpoint plus one settings file per checkpoint.

## Why it matters for the real robot

The real laptop runs exactly this pipeline (ImagePrep, pose converter, gripper map, C4). If a checkpoint already
mispredicts here, on real images with no sim in the loop, the fault is in the deploy path (image prep, key map, rot6d,
relative poses, prompting, the gripper map) and it will mispredict on the arm too; nothing in W2 or W3 can be read
until this passes. The timing bench gives the settings the arm should use per checkpoint; on 2026-11-20 every policy
ran on the arm with bf16 / 8 steps / `exec_h` 12 (upstream TRIALS.md), chosen for run_006 and not measured per
checkpoint.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Pipeline version used | the copy in `deployment/` = **upstream `f632576`** for everything on the policy path: `policy/`, `manager/` and `frames/_dataproc/imageprep.py` byte-identical to `git show f632576:...` (checked 2026-11-21 03:1x); the only differences are W0's sim-only `replay` kind (`config/policy.yaml`, `policy/settings.py`, `policy/policy_factory.py`, `policy/self_check.py`, `launcher/use_policy.py`) and the sim launcher/frames changes, none on the diffusion path. Every result file records `common.deploy_version()` / `prompting_common.version_info()` (file hashes) |
| Pipeline under test | the copy's `deployment/policy_deploy`: `frames/_dataproc/imageprep.py` `ImagePrep` (`MainPrep`, `UltrawidePrep`), `manager/pose_converter.py` `PoseConverter` (+ the workspace clamp after W0), `policy/gripper_map.py` `GripperMap` (after W0; kind `diffusion`, both BP and plain), `policy/diffusion_policy.py` / `promptable_diffusion_policy.py`, `policy/model_runner.py`, C4 `ActionChunk` (`contracts/sizes.py`: 16 steps of 50 ms) |
| Real episodes | the packages on the laptop: `~/data/umi/datasets/PICK_TEST1/{pick1-ep10-v1, pick1-ep15-v1, pick1-ep20-v1, pick1-ep25-v1, pick1-v1}`, `~/data/umi/datasets/multi/{pick1-double-ep15-v1, -ep20-v1, -ep25-v1, pick1-double-v2}`; each checkpoint on its own dataset, train and val split |
| Second real source (optional) | the real-arm policy inputs of 2026-11-20 (robot mount, lab): `~/Capstone/data/<task>/policy/<run>/deploy/2026-11-20_*_trial*_probe_live/` (9 folders: run_022 x3, run_006 x1, PICK_DOUBLE run_003 x2, run_002 x3; 4 PNGs per set, 155-701 sets each) |
| Checkpoints | all 14 ([top README](../../README.md#checkpoints)) |
| Prompting (BP) | `policy.prompt_dataset` = what `policy_deploy use RUN_FOLDER` sets (`launcher/use_policy.py training_dataset`), `prompt_split` train / val / all, `manager.task_name`, `manager.prompt_index` (0 by default) |
| Inference settings | swept: precision fp32 / bf16, diffusion steps (the checkpoint's own `num_inference_steps` and 8), `exec_h` |
| Hardware | the laptop's 4 GB GPU, one policy process at a time; on AC power |

## Method

1. **Harness.** Feed real episodes through the copied pipeline for both kinds: raw images -> `ImagePrep` (as the
   frames process would) -> observation (low-dim through `PoseConverter.to_observation`, widths through
   `GripperMap.to_model`) -> the policy -> chunk (`to_robot`, `make_chunk` clipping) -> `PoseConverter.to_waypoints`.
   The P3 tool `sim_evals/_start_pose_sweep/tools/offline_first_chunk.py` (first chunk on episode_000152) is the
   starting point. Check that the inputs equal training's (image diff vs the dataset's stored images; low-dim keys).
2. **Prompting specifics.** BP: reset -> prompt -> predict order as in training's evaluator; the prompt demo from the
   training dataset (train split), `task_name`, `prompt_index`; check the prompt that is used against `use`'s choice.
3. **Open-loop error.** Per checkpoint, per episode step: predicted vs recorded actions (position mm, rotation deg,
   width; along the chunk at k = 0, 4, 8, 15 as `train/analysis/eval_report.py` reports them), train vs val. Compare
   with the training code's own offline eval (`train/analysis/eval.sh` -> `eval/<tag>/eval.json` ->
   `eval_report.py`); none of the 14 runs has an `eval/` folder yet, and `eval.sh` runs on the cluster (the user's OK
   first). Also the first chunk along the TCP z (P3's measure: +5.6 cm for run_006 on episode_000152, demos +5.7 cm in
   0.75 s), which W2 reuses.
4. **Timing bench -> settings.** `sim_deploy/offline/bench_checkpoints.py` runs the copy's `policy_deploy.policy.bench`
   per checkpoint (steps S and 8, fp32 and bf16, GPU lock, `env.json` with load / GPU / AC state). Pick per checkpoint
   the settings whose p95 fits the `exec_h` budget (`exec_h` 12 = 12 x 50 ms = 600 ms of waypoints per chunk) and
   write them to a settings file W4 reads. Earlier numbers to compare: run_006 bf16 / 8 / 12 in the sim 177-182 ms p50
   (p95 198, E7 G7), 188-270 ms p50 in P3 (loaded laptop); upstream DIRECTIONS (`--mock`): fp32 16 steps about 300 ms,
   bf16 16 about 226 ms, bf16 8 about 186 ms.

Pass: every checkpoint loads in the copied pipeline and its open-loop error on real episodes matches the training
code's eval within a tolerance the coordinator fixes before running; a settings file exists for each.


Fixed before step 3 (open question 1): **T1** "model reproduces training": direct-path (training-exact inputs) action
MSE (torch `mse_loss` over the unnormalised (16, 10) action, as `train_policy_workspace.compute_action_mse`) on the
checkpoint's val and on its train episodes / the training log's value at the checkpoint's epoch (`epochs.csv`) in
[1/3, 3]. **T2** "deploy = training": deploy-path MSE / direct-path MSE <= 1.5 per split, and the median per-sample
|deploy - direct| at k = 15 <= 1.5 x the median seed-to-seed spread of the direct path. Pass = T1 and T2. The
cluster-side `train/analysis/eval.sh` was not run (needs the user's OK); `epochs.csv` (the training run's own val /
train action MSE) is the training-eval reference instead.

## Commands to reproduce

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
PY=~/envs/sim_policy_deploy_torch/bin/python; LOCK="GPU_LOCK_HELD=1 flock $HOME/Capstone-main-sim/sim_evals/.gpu.lock"
# step 1 harness smoke (CPU obs + one GPU process per checkpoint)
$PY -m sim_deploy.offline.smoke
# step 2 prompting (cwd = the copy)
cd deployment/policy_deploy
$PY ../../sim_deploy/offline/prompting_choice.py                       # CPU, Q1/Q2, all 5 BP runs
$PY ../../sim_deploy/offline/prompting_sweep.py gt                     # CPU, once
env $LOCK $PY ../../sim_deploy/offline/prompting_sweep.py run run_006_2026-11-18   # GPU, one run per process (x5)
$PY ../../sim_deploy/offline/prompting_sweep.py summary
cd ~/Capstone-main-sim
# step 3 open-loop error
$PY -m sim_deploy.offline.openloop plan && $PY -m sim_deploy.offline.openloop obs        # CPU
env $LOCK $PY -m sim_deploy.offline.openloop run --id PICK_TEST1/run_006_2026-11-18       # GPU, x14
$PY -m sim_deploy.offline.openloop summary
# step 4 timing bench (AC power, performance profile, nothing else on GPU/CPU), then the settings file
env $LOCK ~/envs/sim_deploy/bin/python sim_deploy/offline/bench_checkpoints.py --only <id> --tag <T>   # per checkpoint
~/envs/sim_deploy/bin/python sim_deploy/offline/make_settings.py
# step 5 real trials (2026-11-20 arm inputs)
~/envs/sim_policy_deploy/bin/python -m sim_deploy.offline.real_trials_inventory
$PY -m sim_deploy.offline.real_trials_train_ref
~/envs/sim_policy_deploy/bin/python -m sim_deploy.offline.real_trials_images
cd deployment/policy_deploy && env $LOCK $PY ~/Capstone-main-sim/sim_deploy/offline/real_trials_replay.py \
    ~/Capstone/data/PICK_DOUBLE/policy/run_002_2026-11-19        # x4: PICK_DOUBLE run_002/003, PICK_TEST1 run_006/022
cd ~/Capstone-main-sim && ~/envs/sim_deploy/bin/python -m sim_deploy.offline.real_trials_analysis
```

GPU lock: every GPU process runs under `flock sim_evals/.gpu.lock` (shared with W0/W2); with `GPU_LOCK_HELD=1` the
scripts' own lock (`common.gpu_lock`, `prompting_common.gpu_lock`, `bench_checkpoints.py`) is a no-op, since a second
flock on the same file from the same process tree would block forever.

## Results

### Per checkpoint (step 3: open-loop on real training episodes; step 4: settings)

All 14 load in the copied pipeline (both kinds, gripper map on = convention B) and **all 14 pass T1 and T2**.
MSE x 1e-4, unnormalised; "train-log" = `epochs.csv` at the checkpoint's epoch; direct = training's inputs, deploy =
the copied laptop pipeline, both fp32 at the checkpoint's 16 steps, seed 0; bf16/8 = deploy at bf16, 8 steps. k15 =
mean position error vs the recorded action at the last chunk step (mm, deploy). Samples: all val episodes + 3 fixed
train episodes, rows 6, 21, ... (stride 15).

| checkpoint | kind | val: train-log / direct / deploy / bf16-8 | val k15 mm | train: train-log / direct / deploy / bf16-8 | train k15 mm | T1 / T2 | settings (provisional) |
|---|---|---|---|---|---|---|---|
| PICK_TEST1 run_006 | BP | 2.01 / 2.33 / 2.46 / 2.50 | 34.8 | 0.43 / 0.41 / 0.48 / 0.52 | 10.5 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_012 | BP | 4.08 / 3.91 / 3.74 / 3.85 | 43.3 | 0.97 / 0.64 / 0.73 / 0.82 | 11.8 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_014 | BP | 3.29 / 3.27 / 3.14 / 3.19 | 43.0 | 0.49 / 0.48 / 0.55 / 0.61 | 11.1 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_016 | BP | 2.30 / 2.85 / 3.02 / 3.05 | 39.9 | 0.52 / 0.45 / 0.53 / 0.57 | 9.8 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_020 | BP | 8.35 / 8.24 / 8.17 / 8.49 | 73.1 | 1.51 / 1.31 / 1.39 / 1.66 | 12.1 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_022 | plain | 2.12 / 2.35 / 2.41 / 2.40 | 33.1 | 0.35 / 0.37 / 0.43 / 0.44 | 9.5 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_024 | plain | 2.69 / 3.26 / 3.37 / 3.37 | 41.0 | 0.33 / 0.37 / 0.43 / 0.43 | 9.3 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_028 | plain | 3.97 / 3.46 / 3.39 / 3.43 | 41.8 | 0.44 / 0.41 / 0.46 / 0.47 | 9.9 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_030 | plain | 8.62 / 9.15 / 9.00 / 9.03 | 77.3 | 0.78 / 0.62 / 0.67 / 0.70 | 11.2 | pass / pass | bf16 / 8 / 12 |
| PICK_TEST1 run_031 | plain | 3.72 / 3.18 / 3.09 / 3.11 | 42.1 | 0.38 / 0.38 / 0.45 / 0.46 | 10.3 | pass / pass | bf16 / 8 / 12 |
| PICK_DOUBLE run_002 | plain | 4.98 / 4.52 / 4.47 / 4.47 | 42.0 | 0.57 / 0.40 / 0.49 / 0.50 | 8.3 | pass / pass | bf16 / 8 / 12 |
| PICK_DOUBLE run_003 | plain | 2.27 / 2.41 / 2.49 / 2.48 | 34.0 | 0.35 / 0.30 / 0.38 / 0.39 | 6.8 | pass / pass | bf16 / 8 / 12 |
| PICK_DOUBLE run_004 | plain | 3.35 / 3.57 / 3.63 / 3.63 | 42.9 | 0.25 / 0.28 / 0.33 / 0.35 | 7.0 | pass / pass | bf16 / 8 / 12 |
| PICK_DOUBLE run_005 | plain | 5.65 / 6.82 / 6.92 / 6.89 | 41.5 | 0.31 / 0.30 / 0.36 / 0.37 | 7.4 | pass / pass | bf16 / 8 / 12 |

Notes: the error is on the model's own (16, 10) output; `to_waypoints` is checked only in the step-1 smoke. T2's
k15 test (deploy vs direct at the same seed against the spread between two seeds) is a loose bar; the MSE ratio is the
tighter one. Ratios T1 0.66-1.24, T2 deploy/direct 0.96-1.25 (train split highest: the deploy path's JPEG + frame selection adds a
little where the model fits tightly). bf16/8 vs fp32/16 deploy MSE 0.996-1.20x (val <= 1.04x; the 1.20x is run_020's train split). Full table with
per-k pos/rot/width and the seed spread: `sim_evals/_pipeline/W1/openloop/summary.md`.

### Step 1: harness

`sim_deploy/offline/` (`deploy_path.py`, `direct_path.py`, `build_obs.py`, `policy_runner.py`, `compare.py`,
`openloop.py`): recorded episode (staged Pi-format videos) -> the copy's `PiRecording`, `Receiver`, `PacketSelector`,
`ImagePrep` (JPEG q90 as streamed) -> `StateHistory` + `PolicyStep.run` (`PoseConverter.to_observation`,
`GripperMap`) -> C4 encode/decode -> the policy -> `to_waypoints`; next to it training's own `UmiTaskDataset` sample
(direct). Smoke (`s1_smoke/summary.json`): direct path = B1's replica and task sampling = episode sampling, both
exactly (max diff 0); deploy images vs training's frames differ only by JPEG noise (PSNR 42-48 dB; one main frame
differs in selection at row 6). The superset packages (pick1-v1, pick1-double-v2) serve every subset package's
episodes (checked: low-dim and sampled frames equal).

### Step 2: prompting (5 BP checkpoints)

- What deploy uses (`policy_deploy use`): `prompt_dataset` = the checkpoint's own package, `prompt_split` train,
  `task_name` PICK_TEST1, `prompt_index` 0 = episode_000138 (561 steps) for run_006/012/014/016 and episode_000139
  for run_020 (its pick1-ep10 train split starts there). The split equals the checkpoint's training split.
- Deploy prompting = training: `reset -> prompt -> predict` gives bit-identical chunks to training's
  `predict_action_training` with the pair prompt (collated, and as `BatchedByTaskDataset` gives it): max abs diff 0.0
  on 4 samples x 2 seeds for every BP run; `reset()` clears the prompt (predict without one asserts).
- bf16: autocast is on for the UNet in predict, off while encoding the prompt; the prompt-token cache stays fp32;
  bf16 vs fp32 same seed <= 1.2 mm.
- Which demo is the prompt hardly matters (the BP checkpoints effectively ignore it; the seed moves the output
  7-11 mm): over every train/val demo x 15 references x 3 seeds the mean position error
  spans 0.05-0.13 mm between the best and worst prompt (run_006 8.75-8.77 mm, run_012 15.23-15.26, run_014 9.28-9.29,
  run_016 11.18-11.19, run_020 17.15-17.21). The default prompt is fine; prompting does not explain any behaviour.

### Step 5: the 2026-11-20 real-arm trials (offline replay of the real policy inputs)

9 trials (PICK_DOUBLE run_002 x3, run_003 x2; PICK_TEST1 run_006 x1, run_022 x3). The real probe images at the
episode start, with the stationary low-dim the manager sends, through the deploy-built policy (5 seeds), vs the
nearest training first frame and camera swaps. First chunk, fingertip base dz at k = 15 (cm, bf16/8 as ran):

| trial | real | nearest training frame | real IMX519 + train IMX219 | train IMX519 + real IMX219 |
|---|---|---|---|---|
| PICK_DOUBLE run_002 t1 / t2 / t3 | -3.4 / -3.4 / -5.7 | +0.9 / +0.9 / -4.0 | -3.1 / -3.2 / -7.1 | +0.2 / +0.4 / -2.8 |
| PICK_DOUBLE run_003 t1 / t2 | -3.7 / -2.2 | +1.0 / +1.0 | -2.6 / -1.1 | +0.2 / +0.1 |
| PICK_TEST1 run_006 t1 | -0.6 | +0.6 | -1.2 | +0.4 |
| PICK_TEST1 run_022 t2 / t3 / t4 | -1.8 / -1.9 / -2.9 | +1.2 / +1.2 / -3.4 | +2.5 / +2.7 / -4.5 | -2.2 / -2.4 / -1.9 |

- Not the deploy pipeline: on training images the same pipeline predicts like training (step 3), and fp32/16 vs
  bf16/8 (<= 0.06 cm) and the width convention (A vs B vs model width 0.087: <= 0.1 cm, run_006 0.9 cm) change the
  first chunk very little.
- **The first chunk on real images is not abnormal.** The real descents (-0.6 to -5.7 cm) lie within the demos' own
  first chunks (ground truth over the 62 demos: median about -3 cm, many -4 to -8 cm; e.g. run_002's 8 baselines -8.1
  to +4.7 cm). The "nearest training frame" column is one demo (`train_first/pick1-double-v2/ep13`, a demo whose first
  chunk rises, +0.9 cm) for 7 of 9 trials, so the "real vs train" gap and the camera-swap columns compare against one
  atypical reference and **do not** show that the real view asks for an abnormal descent, nor which camera carries
  it. Redoing the swap against several demos is open (question 6).
- Mount geometry (calculated, not measured): the sim's mounted camera (from the STEP) sits [-0.3, -31.1, -25.1] mm
  from the handheld's camera (`rig_pi_v3`) in the TCP frame. A policy that brings the camera to where the handheld
  camera was would put the robot fingertip about 4 cm lower (visual offset [-3.3, -2.0, -39.8] mm in the base frame
  at the start orientation). This would act near the grasp, not in the first 0.75 s, and is untested; the rotation
  lever arm adds only 2-4 mm. A candidate cause for the arm going below the table, to confirm on the real rig.
- Start pose vs the demos: each demo's own motion to its grasp, applied from the real start pose, ends below the
  table in 34 of 62 demos (median -0.5 cm, min -12.9 cm). The real start (fingertip ~0.16 m) is about 2.5 cm above
  the demos' median start height (0.135 m), yet the demos' motions leave essentially no margin from it.
- Executed: PICK_DOUBLE run_002 t1-t3 and run_003 t2 clamped 11-81 chunks (run_003 t1: 0 clamped, aborted by the PC's
  gate at waypoint 10), commanded z down to -0.079 m; within clamped chunks the policy kept asking for 2-4 cm (up to
  7.4 cm) more descent.

## Analysis

1. The copied laptop pipeline (f632576 + sim-only replay kind) is training-exact up to JPEG noise and the frame
   selector: deploy/direct MSE 0.96-1.25 for every checkpoint and split, below the seed-to-seed spread at k = 15.
   Nothing in image prep, key map, rot6d, relative poses, gripper map or prompting explains the arm's behaviour.
2. The val error is large for every checkpoint (k15 33-77 mm, vs 7-12 mm on train episodes), as in training's own
   log (val MSE 5-10x train). The 10-episode runs (run_020 BP, run_030 plain) are worst (73 / 77 mm). This is the
   policies' generalisation from 10-62 demos, not a deploy fault. Caveat: for 9 of 14 checkpoints the val split is a
   single episode (28-35 samples), so the val numbers rest on very few demos.
3. bf16 / 8 steps costs at most 1.20x the fp32/16 MSE (often <1.05x): accuracy does not argue against the arm's
   settings. Whether a checkpoint needs them for timing is step 4.
4. BP vs plain: no systematic difference on the same data (run_006 vs run_022, 2.46 vs 2.41 val MSE).
5. The real-arm failures are not explained by the deploy code (points 1, 3). The first chunk on real images descends
   no more than the demos do; the arm's later descent below the table is consistent with two candidates, neither
   proven: the mount's camera-TCP geometry (calculated ~4 cm lower fingertip if the policy servos the camera) and a
   start pose from which the demos' own motions end at or below the table. W2 (sim vs real response) and a
   measurement of the real mount's camera offset are where to look.
6. The BP checkpoints effectively ignore the prompt (prompt choice moves the output 0.05-0.13 mm, the seed 7-11 mm), so
   W3's BP-vs-plain table compares models, not prompting.

## Conclusion

Pass for steps 1-3: every one of the 14 checkpoints loads in the copied pipeline and predicts like training on real
images (T1 and T2). Step 4 is not done: the bench numbers on disk (5 checkpoints; run_012's folder is empty) were thermally throttled
(735-1035 MHz SM vs 2100 max) or loaded (run_004 load 5.8, run_005 10.6), giving bf16/8 162-190 ms vs ~97 ms measured
on the real station; the rerun waits on the user (performance power profile, AC, idle laptop). Until then every
checkpoint uses the arm's bf16 / 8 steps / `exec_h` 12 (provisional; `sim_deploy/offline/checkpoint_settings.yaml`
holds only run_002 / run_003 from the throttled bench, with `exec_h` 11: **ignore it until the rerun**; W3/W4 use
bf16 / 8 / 12).

## Limitations / sim-vs-real caveats

- Open loop on recorded episodes: the policy's own actions never change what it sees next.
- The training images are handheld views; the real-arm inputs are mount views. Step 5 shows the two differ in
  exactly the way that matters (the first chunk's descent).
- Step 5's low-dim is the stationary start observation; probe sets after the start are an approximation.
- Training-eval reference = `epochs.csv` (the training run's sampled val/train MSE), not the cluster `eval.sh`.
- Timing depends on the laptop's load, power and thermal state; `env.json` + `samples.csv` record it.

## Open questions

1. Tolerance: fixed as T1 / T2 above. Answered.
2. Gripper map effect on the width error: convention A vs B give the same model input (the map is affine and
   inverted exactly); width error vs recorded at k15 on val 1.4-2.2 mm, 7.5 mm for the two 10-episode runs (run_020, run_030), <= 0.6 mm on train. On real images the width convention moves the
   first chunk <= 0.1 cm (run_006: 0.9 cm).
3. 8 steps vs 16: bf16/8 <= 1.20x fp32/16 MSE. Answered.
4. Step 4 timing: open until the bench rerun (performance profile, AC, idle; one tag for all 14).
5. Mount camera geometry: confirm the mounted camera's offset vs the handheld on the real rig (W2 / hardware).
6. Step 5's camera swap against several demos' first frames (not the single nearest one) to say which camera, if
   any, makes the real response differ from the demos'.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| code | `sim_deploy/offline/` (harness: `common, episodes, deploy_path, direct_path, build_obs, compare, policy_runner, approach, smoke`; step 2: `prompting_*`; step 3: `openloop.py`; step 4: `bench_checkpoints, c4_overhead, make_settings`; step 5: `real_trials_*`) |
| obs cache | `sim_evals/_pipeline/W1/obs/<package>/ep<ii>_r<row>.{npz,json,deploy.npz,deploy.json}`; variants `obs_100hz/`, `obs_lead30ms/`; staging `staging/` |
| step 1 smoke | `sim_evals/_pipeline/W1/s1_smoke/` (`summary.json`; `run_006_2026-11-18_fp32_s8.*` = fp32 at **8** steps, renamed from `_fp32_ckpt`: `policy_runner` passed `diffusion_steps=None`, which kept the station template's 8; fixed, the step count is now resolved and named `_s<steps>`) |
| step 2 | `sim_evals/_pipeline/W1/prompting/` (`prompt_choice.json`, `prompt_sweep_<run>.{json,npz}`, `summary.json`, `gt_refs.*`, `logs/`) |
| step 3 | `sim_evals/_pipeline/W1/openloop/` (`plan.json`, `<TASK>_<run>.{json,npz}`, `summary.{json,md}`, `logs/`) |
| step 4 | `sim_evals/_pipeline/W1/bench/<TASK>_<run>[/<tag>]/{bench.jsonl,bench.log,env.json}`, `samples.csv`, `c4_overhead.json`, `sessions.json`, `summary.json`; settings `sim_deploy/offline/checkpoint_settings.yaml` (provisional) |
| step 5 | `sim_evals/_pipeline/W1/real_trials/` (`inventory.*`, `images.json`, montages, `train_first/`, `replay_<TASK>_<run>.{json,npz}`, `analysis.{json,md}`, `logs/`) |
