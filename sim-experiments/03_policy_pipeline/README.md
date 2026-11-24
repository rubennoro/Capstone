# 03_policy_pipeline: every policy runs in the sim as it would on the robot, easily

**Status:** running (W0, W2 done 2026-11-21; W1 step 4 open; W3 ready, not run; W4 in progress; W5 = these docs) · **Date:** 2026-11-20 · **Who:** docs lead
(this overview), coordinator agents W0-W4 (one per workstream, each with its own subagents and a reviewer) ·
**Plan ref:** sim_deploy/PLAN.md §11 (goal, progress, workstreams), §10 (camera study, folded into W2), §9 (start
pose), §6 (checks G1-G7); DESIGN.md "P3/P4", "E7: integration"

## Question

Can every runnable checkpoint, behavior prompting (BP) or plain diffusion, run in the sim through the same laptop
pipeline, with the same per-checkpoint settings, that it uses on the Franka, started by one command or one click?
Answered per checkpoint: yes / no, and for a no, the reason (pipeline, images, physics, timing).

## Why it matters for the real robot

- **The same laptop pipeline.** The sim runs the laptop's `policy_deploy` as on the robot (`policy_deploy run --mock`
  against the sim PC and sim Pi; only the hardware checks are skipped). Whatever the sim shows about image prep, the
  pose converter, the gripper map, timing and the C4 chunks is what the arm gets, as long as the copy in `deployment/`
  matches upstream. Since W0 (2026-11-21) it does; before, it did not: upstream `rnoro/final-deployment` moved from `ea9f7fb` to `f632576`
  (gripper max 0.04 m + `policy/gripper_map.py`, the laptop's workspace clamp, real addresses, `F_T_TCP` settled,
  `q_start_rad` in `station.yaml`).
- **Settings per checkpoint.** Precision, diffusion steps and `exec_h` decide whether a chunk arrives in time, and the
  step count changes the policy's behaviour. On the arm every policy ran on 2026-11-20 with bf16 / 8 steps / `exec_h`
  12 (upstream `deployment/policy_deploy/TRIALS.md` at `0c5903a`). W1 measures what each checkpoint needs; W4 makes the
  sim (and so the real launch) apply it without hand-typed flags.
- **The sim as the test bench before arm time.** The first day on the arm (TRIALS.md, 9 recorded trials of run_022,
  run_006, PICK_DOUBLE run_003 and run_002) ended in user stops: every PICK_DOUBLE policy aimed 2-8 cm below the table
  (z down to -0.079 m) and the laptop clamped dozens of chunks. A sim in which policies behave as offline replay
  predicts (W1, W2, W3) can separate pipeline errors from start pose / table height errors without the arm.

## Where we start (PLAN §11 "Progress so far", 2026-11-20)

| Part | State | Record |
|---|---|---|
| Sim chain (sim PC = copied `franka_control` + MuJoCo, sim Pi, copied `policy_deploy` with the listed sim-only changes) | built; G1-G5 pass (E7, P3) | [00_integration_checks](../00_integration_checks/) |
| Rig, mount, cameras from the STEP, printed finger adapter, fin-ray compliance, eraser, table at z 0 | built (E2, E3, E7, P3/P4) | [00_integration_checks](../00_integration_checks/) |
| Start pose key `p`, start-pose tool, eval flow h -> p -> c | built (P1, P2) | [01_start_pose](../01_start_pose/) |
| UI (pick a policy, Play, live views, results) | built (U1-U5); not yet used for a real run | `sim_deploy/ui/DESIGN.md` |
| Start pose | settled: `[-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]` (`leftx55_t48_h16`; = upstream `abba9e8`) | [01_start_pose](../01_start_pose/) |
| Policies in the sim | **not functional**: run_006 backs away from every start pose; the camera swap points at the sim's IMX219 image (P3/P4) | [01_start_pose](../01_start_pose/), [02_camera_study](../02_camera_study/README.md) |
| Camera study (§10) | paused, folds into W2 | [02_camera_study](../02_camera_study/README.md) |
| Copy of `deployment/` | since W0: `f632576` + 19 sim-only files (before: `ea9f7fb` + sim changes + the 09:07 snapshot) | W0 below |

Known issues (§11): the sim's IMX219 look; v_max 0.25 cuts fast demo motions (closed loop should cope); replays need
their own start height; parallel sim chains overloaded the CPU once (realtime fault); other sessions' pytest suites
briefly bind 5555/5620.

State of the camera study's raw data, checked for this overview (differs from §11 in places, see "Docs vs raw data"):
A1 has 23 eval folders tagged `campA1_v025` (3 grasp, `index.csv` / `metrics.json`) and 7 tagged `campA1_v15`
(2 grasp) under `sim_evals/_replay/PICK_TEST1/`; B1's 15 reference frames are in `sim_evals/_camera_study/B/reference/`
(eps 02, 09, 14, 19, 25 x `first`, `approach_1of3`, `approach_2of3`); `B/corr_main/` holds an IMX519 fit (`fit.json`,
10:55); `B/look/` and `B/corr_uw/` are empty; `B/sets/` holds only `_real`.

## Workstreams (PLAN §11)

Each is run by a coordinator agent that gives every step to its own subagent; a reviewer gates each workstream.

| | Workstream | Steps (one subagent each) | Done when (gate) | Depends on | Folder | Status | Finding |
|---|---|---|---|---|---|---|---|
| W0 | **Re-sync with the real deployment** | (1) port `ea9f7fb..f632576` into the copy and re-apply the sim-only changes; (2) the sim gripper with the printed fingers' 0.04 m max and `gripper_map` semantics, and what it means for the 51 mm eraser; (3) start pose from upstream `station.yaml` (`scene start-pose --from-git`); (4) rebuild + G1/G5 smoke | copy = `f632576` + listed sim changes; the chain runs `hold_still` and a replay | - (first; everything depends on it) | [W0_resync](W0_resync/README.md) | done (2026-11-21) | copy = f632576 + 19 sim-only files; G1/G5/replay/load pass; sim eraser 30 mm wide (the 51 mm one cannot fit the real 40 mm fingers) |
| W1 | **Deploy path = training, offline, on real images** (no rendering) | (1) harness feeding real training episodes through the copied laptop pipeline (ImagePrep, pose converter, gripper map, C4) for both kinds; (2) prompting specifics (prompt dataset, `task_name`, `prompt_index`); (3) per-checkpoint open-loop error vs the recorded actions and vs the training code's own offline eval; (4) timing bench per checkpoint -> settings (precision, diffusion steps, `exec_h`) | every checkpoint loads and predicts like training on real images; a settings file per checkpoint | W0 (the ported pipeline) | [W1_offline](W1_offline/README.md) | steps 1-3, 5 done; step 4 waiting (bench rerun under the performance power profile) | all 14 load and predict like training on real images (pipeline = f632576; deploy/direct MSE 0.96-1.25x, bf16/8 <= 1.20x fp32/16); deploy prompting bit-identical to training; real-arm failures not explained by the pipeline (first chunk on real images within the demos' range); candidates, unproven: mount camera-TCP geometry (calculated ~4 cm lower fingertip) and a start pose with no margin; BP checkpoints effectively ignore the prompt; settings bf16 / 8 / exec_h 12 provisional |
| W2 | **Sim images close enough to real** (continues §10 B / C1) | (1) realistic look (table, eraser colour, light, workspace mesh); (2) camera colour / noise / JPEG; (3) attribution with W1's harness: sim vs real first-chunk response per checkpoint; (4) IMX219 / IMX519 view corrections stay an opt-in experiment | the sim response is within tolerance of the real-image response for every checkpoint | W0; W1's harness for step 3 | [W2_sim_look](W2_sim_look/README.md) | done 2026-11-21 06:45, gate **not met** (`sim_evals/_pipeline/W2/attribution/attribution.md`) | Look, not camera model: realistic table / eraser / light / desk extent moved run_006 +9.6 cm (camera model, kit colours, eraser size, motion blur < 0.3 cm each). Robot cell (look `robot`) vs the real arm's images: within tolerance for 13 / 14 (run_016 not); the real arm's ~0 cm start response is reproduced. Training scene at the mount: 0 / 14 within (geometry: IMX519 for plain, IMX219 for run_006; the IMX519 crop does not fix it); handheld static starts 8 / 14. Use look `robot` / `train` for W3, not `current` |
| W3 | **Closed-loop functional evals** | every checkpoint x N rollouts from the start pose, real limits, one chain at a time; BP vs plain table | each checkpoint approaches / grasps as its offline behaviour predicts; failures explained | W0, W1, W2 | [W3_closed_loop](W3_closed_loop/README.md) | ready, not run (`eval/batches/w3.yaml`: 14 x robot/train x N=10, seed 0; then N=20 for the best 3) | - |
| W4 | **Easy to run any policy** | per-checkpoint settings applied automatically (catalog + driver + UI); `eval batch` for a list or "all"; UI end to end on a real run; DIRECTIONS | one command / one click runs any checkpoint correctly | W1's settings files; in parallel with W3 | [W4_ease_of_use](W4_ease_of_use/README.md) | in progress: look, arm settings, `eval batch`, `evaltool` built; smoke batch and UI end to end open | one setting for all (bf16 / 8 / 12) until W1's bench |
| W5 | **Docs** | this folder (W0-W4); finish 00 / 01 / 02; AGENTS.md | every result reproducible from the docs | throughout | this README | running | - |

## Order and resource rules

- **Order (§11):** W0 -> (W1 in parallel with W2) -> W3 in parallel with W4; W5 throughout. W1 (20:25) and W2 (20:28)
  started before any W0 output exists; W1 has to use the `f632576` pipeline (gripper map, clamp) whether from the
  ported copy or upstream, and say which (`bench_checkpoints.py` records whether `policy/gripper_map.py` exists).
- **One sim chain at a time.** Every chain run binds the sim ports (DESIGN.md §1: :5555 sim Pi stream; :5610 C1,
  :5640 C4 policy_deploy; :5620 C2, :5630 C3, :5650 state, :5651 scene sim PC; :8770 the sim UI). More than one only if
  a coordinator proves real time holds (network namespaces, §11); parallel chains overloaded the CPU once.
- **One GPU policy process at a time.** The laptop GPU has 4 GB; the policy used 3.3 GB of it with the renderer in
  E7's G7. W1's offline harness, W2's renders and attribution, W3's evals and W4's UI runs take turns.
- **Real limits for every policy run:** v_max 0.25 m/s, w_max 1.0 rad/s, cone 60 deg, sim box [0.20, -0.40, -0.005]
  ..[0.80, 0.40, 0.70] (as in 02). Sim-only speed overlays are for replays only, named in the tag.
- Watch for other sessions' test suites on 5555 / 5620 (`SIM_DEPLOY_START_TRIES` / `SIM_DEPLOY_START_WAIT_S` retry,
  sim_deploy/DIRECTIONS.md).

## Checkpoints

The 14 runnable checkpoints and their kinds, datasets, epochs and val MSE are listed once, in the top README's
[Checkpoints](../README.md#checkpoints) section (verified again for this overview: 14 of the 36 run folders under
`~/Capstone/data/{PICK_TEST1,PICK_DOUBLE}/policy/` hold one `checkpoints/*.ckpt`; 5 BP with no `TRAIN_CONFIG` line,
9 plain with `TRAIN_CONFIG="umi_policy_dunet"` in `run_config/submit_config.env`). `python -m sim_deploy.ui.catalog`
lists them (it imports no torch; it writes only the replay cache `sim_deploy/ui/.cache/replays.json`, which
`--no-replays` skips).

| Workstream | Which checkpoints |
|---|---|
| W0 | none for the gate (`hold_still` and one replay); one BP (run_006) and one plain (PICK_DOUBLE run_002) only to check that both kinds load with the gripper map |
| W1 | all 14, each on episodes of its own training dataset (train and val split) |
| W2 | all 14 for the attribution; run_006 first (the P3 reference: +5.6 cm on the training images) and run_002 (the deployment candidate) |
| W3 | all 14, same seeds for every checkpoint |
| W4 | any; the end-to-end check on at least one BP and one plain |

## Where the raw data goes

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| W0: G1 diff report, smoke evals | report: `sim_evals/_pipeline/W0/G1/G1_report.md` (+ `gripper/`, `load_check/`); `hold_still`: `sim_evals/_baseline/hold_still/<date>_<time>_<tag>/`; replay: `sim_evals/_replay/<task>/<dataset>_ep<NNN>/<date>_<time>_<tag>/` |
| W1: offline harness output, settings files | `sim_evals/_pipeline/W1/` (created by the W1 coordinator; `prompting/` at 20:28, empty); code in `sim_deploy/offline/` (`bench_checkpoints.py`, 20:30); settings file location TBD |
| W2: looks, renders, attribution | `sim_evals/_pipeline/W2/` (`sets/<look>/<pose>/<ref>/`, `attribution/attribution.{md,json}` + `chunks/`, `camera/`, `look/`, `logs/`; see [W2_sim_look](W2_sim_look/README.md) "Raw data"); the §10 study's data stays in `sim_evals/_camera_study/` |
| W3: closed-loop evals | `sim_evals/<task>/<run_folder>/<date>_<time>_w3_<look>/`; batch state `sim_evals/_batches/<BATCH_ID>/`; report (BP vs plain, kind x look) `sim_evals/_reports/<BATCH_ID>/` |
| W4 | end-to-end eval folders; dev reports `sim_evals/_reports/_dev_*`; imported real trials `sim_evals/_real/` |
| logs | `sim_evals/_logs/<date>_<time>/` (`sim_pc.log`, `sim_pi.log`, `eval.log`) |

## Overall conclusion

(to be integrated by the docs lead after W3 and W4)

## Docs vs raw data (found while writing this overview)

1. PLAN §11 "A1 replays partly run (3/24 grasp at v 0.25)": `sim_evals/_replay/PICK_TEST1/` and `index.csv` hold 23
   `campA1_v025` evals, 3 with a grasp (3/23); plus 7 `campA1_v15`, 2 with a grasp.
2. PLAN §11 "B1 reference frames done; B2-B5 ... not done": B5's IMX519 fit exists in part
   (`sim_evals/_camera_study/B/corr_main/fit.json` and overlays, 10:55; `sim_deploy/corrections/main_crop*.yaml`);
   B2 (`look/`) and B4 (`corr_uw/`) are empty, B3 (`sets/`) has only `_real`.
3. (Resolved by W0, 2026-11-21: robot side as upstream, sim eraser 0.030; see W0_resync.) Gripper width semantics differed between the sim and upstream (details in [W0_resync](W0_resync/README.md)): the sim
   (D8 revised by E7) treats training's width as the fingertip gap and holds the 51 mm eraser at 0.052-0.056 m;
   upstream's `gripper_map.py` treats it as the handheld's ArUco marker distance and maps training 0.051 (closed on
   the eraser) / 0.090 (open) to robot 0.030 / 0.040. The top README's "Conventions / width" row (0.080 the hand's full
   stroke) and the "Sim-only settings" reference (`447910b`) predate `b4030f6`.
4. (Fixed by W0.) The top README's TCP row says `settled: false` upstream; upstream has `settled: true` since `b4030f6`.
5. PLAN §9 still says upstream `q_start_rad` is `null`; it is set since `abba9e8` (§11 says so).
6. Upstream `rnoro/final-deployment` is at `0c5903a` (19:41: TRIALS.md, `frames/trial_video.py`), one commit past the
   W0 target `f632576`; `scene start-pose --from-git` reads the branch tip by default (`DEFAULT_GIT_REF =
   "rnoro/final-deployment"`); its `station.yaml` is the same at both commits.
7. The top README's index links `00_integration_checks/README.md`, which does not exist yet (the folder holds
   `G1_G4_copy_geometry_cameras_stream.md`, `G5_pc_behaviour.md`, `G6_replay.md`, `G7_policies.md`).
