# 03 W3: closed-loop functional evals, every checkpoint, BP vs plain

**Status:** ready, not run (plan `sim_deploy/eval/batches/w3.yaml`: 14 checkpoints x look `robot` x the 18 starts of
the start set `w3`, dry run OK 2026-11-22; the first attempt with random placements and both looks stopped on a GPU
fault, its folders are deleted and not W3 data) · **Plan ref:** sim_deploy/PLAN.md §11 (W3 row) · **Protocol:**
[sim_deploy/evaltool/SCHEMA.md](../../../sim_deploy/evaltool/SCHEMA.md)

## Question

From the real start pose, with real limits, the arm's settings and the real robot cell's look, does each of the 14
checkpoints approach, grasp and lift the eraser, from where its training demos had it and from where the real-arm
trials had it? How do behavior prompting (BP) and plain diffusion compare? Every failure gets a cause.

**Why this protocol (user decisions 2026-11-22).** The goal is a sim result that predicts the real arm, so the sim
differs from it as little as possible: only the `robot` look, the arm at `q_start_rad`, the real PC's limits and load
(mass 0, no setLoad, as the real station), and the same eraser starts for every checkpoint so they are compared on
equal terms:

- **Shared demo starts (rollouts 0-9).** All 9 training datasets begin with the same 10 demos (pick1-ep10-v1), so
  every checkpoint trained on 9 of them and held exactly one out for validation. The eraser goes where each demo
  grasped it, carried to the start pose; the report gives the 9 trained demos and the 1 held-out demo apart (the
  held-out one is a small generalisation test, not mixed into the trained rate).
- **Real-trial poses (rollouts 10-17).** The 2026-11-20 real-arm trials did **not** put the eraser at demo positions:
  it lay 3-8 cm ahead of the start fingertip (W2 fit to the camera images), while the demo starts carried to the sim
  start pose lie 12-22 cm ahead. These 8 poses are the calibration set: the sim result there is what the next real
  trials, run the same way, should be compared with. The real object was the 51 mm eraser; the sim places its 30 mm
  one (the real arm will use a <= 38 mm object) at the fitted centre x, y and yaw, flat.

## Protocol

Fixed by SCHEMA.md "Scoring constants" (user decisions 2026-11-21 / 22); the values live once in
`sim_deploy/evaltool/constants.py` and are copied into every `score.json`.

| Item | Value |
|---|---|
| Success | eraser centre >= settled z + 0.05 m and the eraser's lowest corner >= table top + 0.01 m (lifted clear of the table), both held >= 1.0 s; grasp not required (score v4). Grasp (left and right pad or holder on the eraser in one gt row) is reported; a success without a grasp row in its hold window is flagged `lift_without_grasp` for a check by hand. Also reported: success within 20 s and 30 s |
| Timeout | 60 s per rollout; the rollout stops at success |
| Start | `--home start`: `h`, PC `p` to `stations/start_pose.yaml` (= upstream `abba9e8`), then the eraser at start n (below), flat on the table, settled |
| Eraser placement | `--starts set` (plan `starts: set`): rollout n = `starts[n]` of `sim_deploy/eval/start_sets/w3.yaml`, the same 18 for every checkpoint: 0-9 the shared demos 0-9 at their replay placement (assumptions in sim_deploy/DESIGN.md "Start set"), 10-17 the real-trial poses. Each checkpoint's held-out demo is recorded `held_out: true` |
| Look | `robot` only (the real robot cell) |
| Seeds | seed 0 is recorded but places nothing; rollout n is start n (`seed_key` [0, n] pairs rollouts across checkpoints) |
| Settings | bf16 / 8 diffusion steps / exec_h 12 for every checkpoint |
| Limits | real: v_max 0.25 m/s, w_max 1.0 rad/s, cone 60 deg, box [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70]; no sim overlay |
| Load | the real PC's: mass 0, no setLoad (the rig was never weighed; the sim model carries it, so the sim sags as the real arm) |
| Eraser | 30 mm wide (sim_deploy/eraser.py) |
| Invalid rollout | stop reason `user` / `error`, loop_late > 100 per s, sim-lag fault, no gt rows, no waypoint accepted in a timeout, ran during a GPU fault window (`gpu_fault`: [wall_start, wall_end] overlaps [kernel Xid, cleared at the batch's resume after the reboot)): excluded from rates, re-run once by the batch (same start) |
| Execution | each run in run.sh calls of at most 6 rollouts (`piece_rollouts`, the same eval folder continued with `--resume`), the GPU cooled below 65 C and >= 60 s idle before every call (Xid 62 twice under sustained load, 2026-11-22); only the time between calls changes, no effect on rollouts or `protocol_sha` |
| Comparable | `eval.yaml` `protocol_sha` (sha256 of the `protocol` block: start set, every rollout's start and pose, timeout, settings, look, model sha, start pose, judge, limits): one sha for all 14 checkpoints |

**Known gaps (reported, not fixed).** (1) Control cycle: the sim's is ~200 ms p50 (policy ~170 ms on the laptop GPU)
against ~110-130 ms on the real arm (2026-11-20 steps: run_006 119 / 132 ms policy / cycle, plain runs ~97 / ~110):
the policy replans less often in the sim (user decision: report it; the report's "Timing vs the real arm" table).
(2) The object: 30 mm wide in the sim, 51 mm in the trials whose poses are used. (3) For the 5 BP checkpoints the
prompt (`task_name PICK_TEST1, prompt_index 0`) is index 0 of the checkpoint's own **train** split
(deployment/policy_deploy `use_policy.py` + the training split mask): for run_006 / 012 / 014 / 016 that is shared
demo 0, the same demonstration as rollout 0; run_020 holds demo 0 out, so its prompt is demo 1, the same
demonstration as rollout 1 (its held-out rollout 0 is not its prompt). (4) Demo starts carry the demo's start-to-grasp motion to a start pose that is not the demo's
(`height_error_m` -12.8 to +0.1 cm): the eraser lies where the policy would grasp if it repeated the demo, not where
the demo's eraser was relative to the table. (5) The 8 real-trial poses are not 8 independent starts: they form
~5 distinct clusters (run_003 trial1 ~ run_006 trial1; run_002 trial1 ~ run_003 trial2 ~ run_022 trial2), so their
rate is reported as correlated samples, not 8 independent ones. (6) The W2 fit gives eraser lengths of 100-112 mm
against the real 127 mm: the fitted centre may be off by up to ~13 mm along the eraser's length.

## Matrix

| Pass | Plan | Runs | Rollouts |
|---|---|---|---|
| 1 | `batches/w3.yaml` | 14 checkpoints (explicit list) x look `robot` = 14 | N = 18 each: 10 demo starts + 8 real-trial poses (252 + retries) |
| 2 | a plan with the best 3 of pass 1 (to be written then) | 3 | more rollouts need more starts (open) |

Start set `w3` (`python -m sim_deploy.eval starts show`; every start passes on table / in box / in the IMX219 view;
x, y in the robot base frame, ahead / left relative to the start fingertip):

| Rollout | Source | Demo / trial | x m | y m | yaw deg | ahead cm | left cm |
|---|---|---|---|---|---|---|---|
| 0 | demo | 0 (ep 138) | 0.550 | -0.042 | -89.0 | 11.7 | -1.9 |
| 1 | demo | 1 (ep 139) | 0.583 | 0.011 | 89.6 | 17.1 | -5.2 |
| 2 | demo | 2 (ep 140) | 0.515 | 0.044 | 89.1 | 20.3 | 1.6 |
| 3 | demo | 3 (ep 141) | 0.578 | 0.026 | 84.6 | 18.5 | -4.7 |
| 4 | demo | 4 (ep 142) | 0.536 | 0.049 | -86.6 | 20.8 | -0.5 |
| 5 | demo | 5 (ep 143) | 0.567 | 0.063 | -89.9 | 22.2 | -3.6 |
| 6 | demo | 6 (ep 144) | 0.553 | 0.049 | 87.5 | 20.7 | -2.2 |
| 7 | demo | 7 (ep 145) | 0.550 | 0.024 | 85.0 | 18.3 | -1.9 |
| 8 | demo | 8 (ep 146) | 0.512 | 0.015 | -89.1 | 17.3 | 1.9 |
| 9 | demo | 9 (ep 147) | 0.555 | 0.023 | 86.6 | 18.2 | -2.4 |
| 10 | real_trial | PICK_DOUBLE run_002 trial1 | 0.523 | -0.083 | 88.9 | 7.6 | 0.8 |
| 11 | real_trial | PICK_DOUBLE run_002 trial2 | 0.518 | -0.095 | 84.8 | 6.4 | 1.3 |
| 12 | real_trial | PICK_DOUBLE run_002 trial3 | 0.469 | -0.116 | 74.0 | 4.3 | 6.2 |
| 13 | real_trial | PICK_DOUBLE run_003 trial1 | 0.525 | -0.109 | -77.6 | 5.0 | 0.6 |
| 14 | real_trial | PICK_DOUBLE run_003 trial2 | 0.522 | -0.088 | 89.5 | 7.1 | 0.9 |
| 15 | real_trial | PICK_TEST1 run_006 trial1 | 0.525 | -0.110 | -80.5 | 5.0 | 0.6 |
| 16 | real_trial | PICK_TEST1 run_022 trial2 | 0.511 | -0.082 | 82.6 | 7.7 | 2.0 |
| 17 | real_trial | PICK_TEST1 run_022 trial3 | 0.536 | -0.126 | 87.0 | 3.3 | -0.5 |

Held-out demo (= rollout) per checkpoint (the checkpoint's split = `get_val_mask` recomputed, all 14):

| Checkpoint | Kind | Dataset | Val demos | Held-out rollout |
|---|---|---|---|---|
| PICK_TEST1 run_006 | BP | pick1-v1 | 2, 23 | 2 |
| PICK_TEST1 run_022 | plain | pick1-v1 | 2, 23 | 2 |
| PICK_TEST1 run_016 | BP | pick1-ep25-v1 | 2 | 2 |
| PICK_TEST1 run_024 | plain | pick1-ep25-v1 | 2 | 2 |
| PICK_TEST1 run_014 | BP | pick1-ep20-v1 | 1 | 1 |
| PICK_TEST1 run_031 | plain | pick1-ep20-v1 | 1 | 1 |
| PICK_TEST1 run_012 | BP | pick1-ep15-v1 | 1 | 1 |
| PICK_TEST1 run_028 | plain | pick1-ep15-v1 | 1 | 1 |
| PICK_TEST1 run_020 | BP | pick1-ep10-v1 | 0 | 0 |
| PICK_TEST1 run_030 | plain | pick1-ep10-v1 | 0 | 0 |
| PICK_DOUBLE run_002 | plain | pick1-double-v2 | 5, 40, 47 | 5 |
| PICK_DOUBLE run_005 | plain | pick1-double-ep25-v1 | 4, 38 | 4 |
| PICK_DOUBLE run_004 | plain | pick1-double-ep20-v1 | 3, 30 | 3 |
| PICK_DOUBLE run_003 | plain | pick1-double-ep15-v1 | 2, 23 | 2 |

Checkpoints: BP PICK_TEST1 run_006 / 012 / 014 / 016 / 020; plain PICK_TEST1 run_022 / 024 / 028 / 030 / 031;
plain PICK_DOUBLE run_002 / 003 / 004 / 005 (list: [top README](../../README.md#checkpoints)).

Wall time of pass 1 (dry run): 3 h 09 min (rollouts end after ~15 s) to 6 h 18 min (all time out), plus retries and
waits.

## How to run and report

Before: AC power, `powerprofilesctl set performance`, heavy apps closed, load (1 min) < 3, no UI run
(sim_deploy/DIRECTIONS.md §3c).

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
PY=~/envs/sim_deploy/bin/python
$PY -m sim_deploy.eval batch --plan sim_deploy/eval/batches/w3.yaml --dry-run     # run list, estimate, checks
$PY -m sim_deploy.eval batch --plan sim_deploy/eval/batches/w3.yaml               # prints <BATCH_ID> = w3_<date>_<HHMM>
$PY -m sim_deploy.eval batch --resume <BATCH_ID>                                  # after a stop or crash
$PY -m sim_deploy.evaltool report --batch <BATCH_ID> --out sim_evals/_reports/<BATCH_ID>   # again, by hand
```

The batch writes the report itself at the end. It has BP vs plain pooled, kind x look (robot only now), per
checkpoint success k / n with Wilson 95 %, grasp, approach / close error, faults by gate rule, clamps; success within
20 s / 30 s; the start split (9 trained demos / 1 held-out demo / 8 real-trial poses); and sim vs real chunk timing.
A GPU fault stops the batch (exit 2) and invalidates the rollouts running at or after the kernel Xid; reboot, then
`--resume` (it closes the fault window at the reboot, so the re-run rollouts count).

Analysis per checkpoint: compare with W1's first-chunk measure (`checkpoints.csv`
`w1_first_chunk_toward_grasp_cm`) and W2's attribution; a failure is explained when its cause is named with evidence
(`frames.jpg`, `gt.jsonl`, a log line).

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| eval folders | `sim_evals/<task>/<run_folder>/<date>_<time>_w3_<look>[_retryN]/` |
| batch state | `sim_evals/_batches/<BATCH_ID>/` (`plan.yaml`, `manifest.jsonl`, `logs/`) |
| report | `sim_evals/_reports/<BATCH_ID>/` (`report.md`, `report.html`, CSVs) |
| run logs | `sim_evals/_logs/<date>_<time>/` |

## Pre-launch findings (2026-11-22, before the batch)

Smoke and check runs before launch (robot look, start set, bf16/8/12) found three things; read W3's numbers with them.

1. **rot6d PolicyFatal (fixed in sim only).** The model's raw rot6d rows were refused by C4 (`rot6d_dot_max` 0.1)
   although the manager Gram-Schmidts them anyway; it killed run_006, run_012 and run_020 rollouts after 6–50 s. The
   sim copy now orthonormalizes in `policy/base_policy.py make_chunk` (motion unchanged). Check (`*_gs_check`, 12
   rollouts on run_006/012/020): 0 rot6d faults, 17 chunks straightened (raw \|cos\| up to 0.30). The real branch
   still has the raw check: handoff `sim_deploy/ports/rot6d_gram_schmidt.md`.
2. **Tilt cone: kept at 60°** (user decision). All 12 check rollouts then stopped at the PC gate's 60+2° tilt limit
   after 5–36 s. A sim-only 75° overlay (`stations/cone75_experiment.yaml`; evals `*_cone75`, `*_cone60`, 28
   rollouts on run_006/012/020/022, PICK_DOUBLE run_002) gave 0 successes too: the prompting checkpoints kept tilting
   up to the new limit (74–80°), and PICK_DOUBLE run_002 drove the hand and the rig (Pi 5 housing, cover) into the
   table for ~20 s per rollout. Plain checkpoints mostly froze at 60° (timeout, tilt 50°).
3. **Why the policies fail: images, not code.** Four parallel checks (scripts in the session scratchpad; results
   here):
   - Action path: waypoints = anchor × chunk exactly; executed tilt follows commanded (r 0.92–0.98), displacement
     direction cosine 0.97–0.99; open-loop replay of demo 19 reproduces its tilt-down 48→38° and descent
     (`_replay/.../2026-11-22_1723_inv_replay`, `*_inv_tap`). Minor: the arm lags in rotation by ~2–7° and does not
     recover gate-shortened steps.
   - Observation path: identical to training (TCP axes at q_start match the demo starts, rot6d rows, relative math
     to 1e-15, gripper map 0.030/0.040 ↔ 0.051/0.090, horizons). No F_T_TCP error. Same files in final-deployment.
     Minor: the two pose rows are 33/67 ms apart instead of 50 ms.
   - The policy itself commands the retreat: at rest with identity proprio, run_020's first chunk goes ~3 cm along
     the closing axis and backs off; demos go ~0 sideways and +2.7 cm forward.
   - Image causes (`sim_evals/_pipeline/W2/fingergap_2026-11-22/fingergap.md`): the camera mount pose (41–53 mm
     further back than handheld) is a small effect in the helpful direction. The **finger gap** dominates: the robot's
     fingers open at most 40 mm (the training images show ~88 mm open), so open robot fingers look like a training
     grasp ("already holding" → the lift: tilt up, back off, rise); +0.6–7.1° tilt and −1–9 cm approach per
     checkpoint, strongest near the eraser. At the real-trial starts in the robot cell the policies barely respond
     even with the fingers rendered at 88 mm: the **robot cell's appearance** (W2 APPEARANCE term) holds them still,
     as on the real arm.
   - The gripper *numbers* are already mapped (robot 0.040/0.030 ↔ training 0.090/0.051); the finger *pixels* are
     not. Software-only options (not done, user: no physical changes, 40 mm max): edit the finger pixels to the mapped
     training width at deployment; mask the fingers in training + deployment and retrain; record teleop demos on the
     arm (fixes fingers and cell at once).

Expected W3 outcome: success near 0% for all 14; the report therefore also ranks by time to fault, max tilt,
closest approach, freeze vs drift and grasp attempts.

## Results

(running: see the batch id in AGENTS.md)

## Open questions

1. Pass 2: pool its rollouts with pass 1's (same starts) or report it alone?
2. Does any checkpoint need another start pose? Answered by the failures' causes.
3. Pass 2 (more rollouts per checkpoint) needs more starts: more shared demos do not exist (the datasets share only
   10), so repeats, more real-trial poses, or per-dataset demos.
4. The sim's ~200 ms cycle vs the real ~130 ms: if W3 failures look timing-driven, a faster sim inference setting
   would be a separate protocol (not done: user decision to report it).
