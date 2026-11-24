# 02_camera_study: is it the start pose or the camera?

**Status:** paused 2026-11-20; folded into 03 W2 · **Date:** 2026-11-20 (A1 10:37-10:56, B 10:40-10:56; closed
2026-11-20) · **Who:** lead agent (this overview), phase agents A, B; C and D never started ·
**Plan ref:** sim_deploy/PLAN.md §10 (study), §11 ("Progress so far": "Camera study (§10) paused", workstream W2),
§9 (start pose), §6 "P3" table row G7; DESIGN.md "P3/P4", "E5: as built" ("What differs from the real lab"), "E7:
integration" ("Appearance"). DESIGN.md has no camera-study section.

> **Paused on 2026-11-20 and folded into PLAN §11 workstream W2** ("Sim images close enough to real": realistic look,
> camera colour / noise / JPEG, attribution with W1's harness, the IMX219 / IMX519 view corrections as an opt-in
> experiment): [../03_policy_pipeline/W2_sim_look/README.md](../03_policy_pipeline/W2_sim_look/README.md). The study's question, whether the start pose, the camera geometry or the sim's look
> makes the policies back away, is **not answered**. A1 ran in part: from `leftx55_t48_h16` the replayed demos grasp
> where their recorded path lands at the table, every floor fault is a replay start-height artefact, and the pose
> caused no fault. B1 (the real reference inputs) is done. A2, B2-B5, C1, C2 and D did not run. The Status table
> below gives each phase. The rest of this file is the plan as written before the runs, kept for W2.

## Question

From the recommended start pose (`sim_deploy/stations/start_pose.yaml`, `leftx55_t48_h16`), do (1) the demos' recorded
motions and (2) the trained policies run in the sim? If the motions run and the policies do not, the problem is the
camera view, and then: is it the **geometry** of the view (where the Franka mount puts the cameras relative to the
fingers and scene, vs where the handheld put them in training) or its **appearance** (how the sim renders the scene)?

## Why it matters for the real robot

- **Geometry is a real-world problem.** The rig on the Franka mount does not sit where it sat on the handheld: PLAN §10
  puts the mount's view of the fingers / scene at about 3-4 % off the handheld view (see "Docs vs raw data" below:
  this figure has no raw source yet; B measures it). E3 found the fingertips 11.8 mm further from the cameras than the
  rig study assumed (TCP 235.65 mm vs the STEP's 223.9 mm), and G3 found training's `T_cam_tcp` ~9 mm off the image
  evidence. If geometry makes the policy back away, the real arm will see the same offset: the fix (a virtual-camera
  correction for the IMX219, a crop / zoom for the IMX519, or a mount change) has to go to the real deployment, and
  the recommendation says so.
- **Appearance is a sim-only problem.** If only the look (table texture, light, shadows, colours, background) makes
  the policy back away, the real lab does not have that problem; the sim needs a realistic look before any benchmark
  can rank policies or start poses, and nothing changes on the arm.
- Until this is answered no sim result can rank checkpoints or start poses: in P4's screen every one of 23 poses
  showed run_006 retreating (DESIGN.md "P3/P4", "Screen and full eval").

## Hypothesis (from P3, untested)

P3's G7 (run_006, behavior prompting, bf16 / 8 diffusion steps / exec_h 12) faulted in all 3 episodes after rising and
backing off from the first chunk. Offline, with the same model and the same low-dim input
(`sim_evals/_start_pose_sweep/tools/offline_first_chunk.py`, logs `sim_evals/_start_pose_sweep/p3/offline.log`,
`offline_split.log`; mean of 5 samples, last step of the first chunk along the TCP z axis = toward the fingertips):

| IMX519 image | IMX219 image | First chunk along TCP z (cm) |
|---|---|---|
| training (episode_000152) | training | +5.6 (the demos: +5.7 cm in 0.75 s) |
| sim | sim | -1.3 |
| sim | training | +5.4 |
| training | sim | -0.5 |

So **the sim's IMX219 render is what the policy rejects**. The policy images at the start pose vs episode_000152
(`sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/g7_policy_images_vs_ep152.png`): geometry
matches (eraser, fingers, tags where training has them); table texture, shadows, colours and the IMX219's background
do not. P3's **guess is appearance, not geometry; it is untested**: no render with a realistic look and no render from
the handheld pose went through the policy. Caveats on these numbers: one checkpoint (run_006), one training episode,
5 samples, CPU fp32 with 8 diffusion steps (the tool overrides the station's bf16), and the P3 stand-in start pose (49.5 deg, 16.2 cm, TCP O (0.439, -0.164, 0.162)), not `leftx55_t48_h16`.

## Phases (PLAN.md §10)

| Phase | Runs on | What | Must produce before the next phase starts (gate) | Folder |
|---|---|---|---|---|
| A1 | chain | **pose and physics**: replay the PICK_TEST1 demos from the start pose, eraser placed per demo, at the real v_max 0.25 and once with a sim-only v_max 1.5 (`stations/vmax15_campA.yaml`; open-loop replays cannot re-aim) | one row per replay (grasp, lift, success, path error, closing error, faults with their rule) at both speeds, and the verdict: do the motions carry from `leftx55_t48_h16`? If they do not at 1.5, the start pose / physics is the problem and the study reports that before A2 is read | [A_baseline](A_baseline/README.md) |
| A2 | chain, after A1 | **policies as they are** (no correction): every runnable checkpoint (14) x 3 episodes from the start pose, real limits | per episode and per checkpoint: outcome, success / grasp / approach, faults with rule and time, and the first-chunk direction (toward the eraser, along the TCP z); the seeds used (C2 reuses them) | [A_baseline](A_baseline/README.md) |
| B | offline (no sim chain, no policy), in parallel with A | **camera geometry and appearance**: renders of the same scene from the handheld pose vs the Franka mount, each with the current look and with a realistic look, as separate model builds; offsets vs real training first frames; fit the IMX219 virtual-camera correction (tilt + zoom) and the IMX519 crop / zoom (sim-only, default off) | the image sets {handheld, mount, mount_start} x looks {`current`, `real`, `real+mesh`} x {no correction, corrected} for 15 real reference frames (PICK_TEST1 `pick1-v1` indices 14, 2, 9, 19, 25 x 3 rows each, `sim_evals/_camera_study/B/layout.json`); the measured offsets (224 px, %); the fitted corrections in sim-only files (`sim_deploy/corrections/rig_pi_v3_640_corr.yaml`, `sim_deploy/corrections/main_crop.py`, per B's README) | [B_geometry_appearance](B_geometry_appearance/README.md) |
| C1 | chain, after A and B | **attribution** (model forward passes): {handheld, mount, corrected mount} x {current, realistic look} vs the real frame | the first-chunk table per cell (and per camera with the P3 swap), the verdict geometry / appearance / both / neither, and the list of fixes C2 runs | [C1_attribution](C1_attribution/README.md) |
| C2 | chain, after C1 | **policies with the fixes C1 points to** (realistic look; + IMX219 correction; + IMX519 correction): every runnable checkpoint x 3 episodes, same seeds as A2 | the same table as A2, per fix level, at A2's seeds | [C2_fixes](C2_fixes/README.md) |
| D | reviewer, after C2 | check A-C, one table in `sim_evals/_camera_study/`, then the recommendation | the checked table and the recommendation (what goes to the real deployment, what stays sim-only) | [D_review](D_review/README.md) |

**One chain at a time.** A1 -> A2 -> C1 -> C2 run strictly one after another: every chain run uses the sim ports
(DESIGN.md §1: C0 :5555 sim Pi stream; C1 :5610, C4 :5640 policy_deploy; C2 :5620, C3 :5630, state :5650, scene
:5651 sim PC; :8770 the sim UI) and the laptop's 4 GB GPU (the policy alone used 3.3 GB with the renderer in E7's G7). PLAN calls B CPU-only
rendering; B's README says EGL offscreen, < 280 MB VRAM per process (the sim Pi's EGL renderer runs on the NVIDIA
GPU, DESIGN.md "E5"). B runs in parallel with A, binds none of those ports, and keeps its VRAM small enough to share
the 4 GB with A2's policy (about 3.3 GB).

## Decision logic (C1 -> C2)

C1 feeds each image set through each checkpoint with the same low-dim input as at the real start and reads the first
chunk (the P3 measure: last step along the TCP z axis; plus the component toward the eraser). "Realistic" = B's look `real` (and `real+mesh` if B finds the workspace mesh usable); "corrected mount" = B's
fitted IMX219 / IMX519 corrections applied to the mount renders. The real training first
frames are the reference (run_006 on episode_000152: +5.6 cm). Lead's proposed reading, to be fixed by the C1 agent
**before** it runs (and written in C1's Method): a cell "approaches" when the mean first chunk points toward the
fingertips / eraser with at least half the real-frame value, "rejects" when it is <= 0.

| Handheld, current | Mount, current | Handheld, realistic | Mount, realistic | Corrected mount, realistic | Verdict | C2 runs |
|---|---|---|---|---|---|---|
| rejects | rejects | approaches | approaches | (approaches) | **appearance** (sim-only) | realistic look only |
| rejects | rejects | approaches | rejects | approaches | **geometry** (real-world), on top of appearance | realistic look + IMX219 correction (+ IMX519 correction if B / C1 show the IMX519 matters) |
| approaches | rejects | approaches | rejects | approaches | **geometry only** (real-world) | IMX219 (+ IMX519) correction, current look; and with the realistic look |
| rejects | rejects | rejects | rejects | rejects | **neither**: something else (low-dim input, prompt, scene content) | no C2 until the cause is found; report |

The P3 camera swap (sim IMX519 + real IMX219 and the reverse) is repeated per cell so the verdict is per camera:
the IMX219 correction and the IMX519 crop / zoom are judged separately. A cell between the two thresholds is
reported as such, not rounded into a verdict.

## Constraints

- No hardware changes; `rnoro/final-deployment` (`~/Capstone-final-deployment`) untouched.
- Every correction (realistic look, IMX219 virtual-camera tilt + zoom, IMX519 crop / zoom, v_max 1.5) is a sim-only
  experiment, off by default, in sim-only files (`sim_deploy/stations/`, separate model builds, an option in the copy
  `deployment/` that defaults to off), until D.
- **Recommendation only after the complete sim eval (A-C).** No phase recommends a real-robot change on its own.
- Real limits for every policy run (A2, C2): v_max 0.25 m/s, w_max 1.0 rad/s, cone 60 deg, sim box [0.20, -0.40,
  -0.005]..[0.80, 0.40, 0.70] (DESIGN.md "P3/P4", "Real PC handoff"). The sim-only v_max 1.5 is for A1 replays only.

## Shared setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Model | `sim_deploy/build/sim_rig/model.json` `model_sha256` `166a0649571a…` (P3/P4: table top = O z 0; E7 appearance: white pads, blue holders, ArUco tag textures, procedural wood table, `table_texrepeat` 1.2) |
| Start pose | `--home start`, `stations/start_pose.yaml` = `leftx55_t48_h16`: `q_start_rad [-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]`, TCP O (0.531, -0.159, 0.160), 16.0 cm above the table, tilt 48.0 deg, fingertips to the robot's left (PLAN §9, DESIGN.md "The chosen pose") |
| Eraser | policies: `scene/randomization_start.yaml` (median 15.9 cm ahead, 1.9 cm right of the fingertips); replays: placed as in the replayed demo |
| Stations | `stations/control_sim.yaml` + `start_pose.yaml`; A1's second pass adds `stations/vmax15_campA.yaml` (v_max 1.5, sim-only) |
| Inference | behavior prompting (BP, `umi_policy_dunetp`): `--precision bf16 --diffusion-steps 8 --exec-h 12` on this laptop, on AC power; plain diffusion (`umi_policy_dunet`): to be confirmed by the A2 agent (DIRECTIONS.md names the flags for BP only) |
| Seeds | A2 and C2 use the same `--seed` (episode n gets `default_rng([S, n])`, the same eraser pose); A2 records it |
| Raw data | `sim_evals/_camera_study/` (A/, B/, then C1/, C2/, D's table) + the eval folders below |

### Runnable checkpoints (14; listed on 2026-11-20, `~/Capstone/data/{PICK_TEST1,PICK_DOUBLE}/policy/*/checkpoints/*.ckpt`)

| Run folder | Kind | Dataset (episodes) | Checkpoint epoch / val MSE |
|---|---|---|---|
| `PICK_TEST1/policy/run_020_2026-11-19` | BP | `pick1-ep10-v1` (10) | 59 / 0.0008353 |
| `PICK_TEST1/policy/run_012_2026-11-18` | BP | `pick1-ep15-v1` (15) | 59 / 0.0004078 |
| `PICK_TEST1/policy/run_014_2026-11-18` | BP | `pick1-ep20-v1` (20) | 59 / 0.0003294 |
| `PICK_TEST1/policy/run_016_2026-11-18` | BP | `pick1-ep25-v1` (25) | 59 / 0.0002300 |
| `PICK_TEST1/policy/run_006_2026-11-18` | BP | `pick1-v1` (30) | 59 / 0.0002007 |
| `PICK_TEST1/policy/run_030_2026-11-19` | plain | `pick1-ep10-v1` (10) | 50 / 0.0008616 |
| `PICK_TEST1/policy/run_028_2026-11-19` | plain | `pick1-ep15-v1` (15) | 59 / 0.0003967 |
| `PICK_TEST1/policy/run_031_2026-11-19` | plain | `pick1-ep20-v1` (20) | 59 / 0.0003717 |
| `PICK_TEST1/policy/run_024_2026-11-19` | plain | `pick1-ep25-v1` (25) | 59 / 0.0002694 |
| `PICK_TEST1/policy/run_022_2026-11-19` | plain | `pick1-v1` (30) | 50 / 0.0002123 |
| `PICK_DOUBLE/policy/run_003_2026-11-19` | plain | `pick1-double-ep15-v1` (30 = 15+15) | 59 / 0.0002274 |
| `PICK_DOUBLE/policy/run_004_2026-11-19` | plain | `pick1-double-ep20-v1` (40 = 20+20) | 59 / 0.0003348 |
| `PICK_DOUBLE/policy/run_005_2026-11-19` | plain | `pick1-double-ep25-v1` (50 = 25+25) | 50 / 0.0005648 |
| `PICK_DOUBLE/policy/run_002_2026-11-19` | plain | `pick1-double-v2` (62 = 30+32) | 40 / 0.0004976 (the current deployment candidate) |

Kinds and datasets from `~/Capstone/data/RUNS.md`; epoch and MSE from the checkpoint file names.

## Where the raw data goes

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| study folder (tables, tools, renders) | `sim_evals/_camera_study/` (`A/`, `B/` exist; `A/tools/a1_rows.py`, `A/tools/a2_rows.py`, `B/layout.json`; B's planned `B/{index.json, summary.md, reference/, look/, sets/, corr_uw/, corr_main/}`). At the pause: `A/` = the interim `a1_summary.md`, `a1_episodes.csv`, 2 figures, `tools/`; `B/` = `layout.json`, `reference/` (B1), `sets/_real/`, `corr_main/` (unreviewed), empty `look/` and `corr_uw/`; no `index.json`, `summary.md`, `C1/`, `C2/` |
| A1 replays | `sim_evals/_replay/PICK_TEST1/pick1-v1_ep<iii>/<date>_<time>_campA1_v025/`, `..._campA1_v15/` |
| A2, C2 policy evals | `sim_evals/<task>/<run_folder>/<date>_<time>_<tag>/` (A2 tags `campA2*`; C2 tags to be set by the C2 agent) |
| logs | `sim_evals/_logs/<date>_<time>/` |
| P3 evidence this study starts from | `sim_evals/_start_pose_sweep/p3/`, `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/` |

## Status

| Phase | Status | Folder | Finding (one line) |
|---|---|---|---|
| A1 | **partial, paused** (30 of 60 replay evals: 23 at v_max 0.25, 7 at 1.5) | [A_baseline](A_baseline/README.md) | 3 of 15 physical v025 replays and 2 of 4 physical v15 replays grasp, lift and succeed. All 9 floor (box z) faults are demos that started 21.6-27.1 cm high, whose path carried to the 16.0 cm start ends 7.1-12.7 cm below the table (a replay artefact). The pose caused no fault. 11 of 30 evals were ended by the host (9 health aborts, 2 realtime faults from CPU load) |
| A2 | not run | [A_baseline](A_baseline/README.md) | - (no `campA2` eval; becomes W3's first table) |
| B1 | **done** | [B_geometry_appearance](B_geometry_appearance/README.md) | 15 real reference frames (5 episodes x 3 rows) as training's 224 inputs and through the deploy path. Recorded deploy-vs-training difference: MAD 0.40-0.53 (IMX519) / 0.57-0.69 (IMX219) 8-bit levels, no shift. No pass threshold was set |
| B2 (looks) | not done | [B_geometry_appearance](B_geometry_appearance/README.md) | - (`B/look/` and `sim_deploy/look/assets/` empty; no look build) |
| B3 (render sets) | not done | [B_geometry_appearance](B_geometry_appearance/README.md) | - (`B/sets/` holds only the real `_real/` export) |
| B4 (IMX219 correction) | not done | [B_geometry_appearance](B_geometry_appearance/README.md) | - (`B/corr_uw/` empty) |
| B5 (IMX519 crop / zoom) | not done | [B_geometry_appearance](B_geometry_appearance/README.md) | - (an unreviewed first fit, `B/corr_main/fit.json` at 10:55:51, is on disk; not a result) |
| C1 | not run | [C1_attribution](C1_attribution/README.md) | - (folded into W2: attribution with W1's harness) |
| C2 | not run | [C2_fixes](C2_fixes/README.md) | - |
| D | not run | [D_review](D_review/README.md) | - (no real-robot change recommended) |

## What carries over to W2

- **The hypothesis, still untested.** P3's offline camera swap above: run_006's first chunk goes toward the
  fingertips with training's IMX219 image (+5.4 cm) and backs away with the sim's (-1.3 cm all-sim, -0.5 cm with only
  the sim IMX219). P3 guessed appearance, not geometry. Its caveats stand: one checkpoint, one training episode,
  5 samples, CPU fp32, the P3 stand-in pose. The decision logic and the proposed thresholds above ("approaches" >=
  half the real-frame value, "rejects" <= 0, set before the passes) are the plan for W2's attribution.
- **B1's real reference inputs** (`sim_evals/_camera_study/B/reference/`, 15 frames + 10 extra raw frame pairs,
  [B's Status at pause](B_geometry_appearance/README.md)). These are the real-image side of W2's "sim vs real
  first-chunk response". They are also a cross-check for W1's harness: the deploy path and training's sampler
  already agree there on the low-dim input (<= 1.2e-7) and on the images (no shift; MAD < 1.1 levels with JPEG q90).
- **`layout.json`'s `T_O_tag`**: it places a sim scene like a real frame (the demos' median grasp at O (0.55, 0)).
- **The tools** (sim-only, default off; none reviewed): `sim_deploy/camera_study/reference.py` (B1, ran),
  `sets_real_export.py` (ran), `render_sets.py` (B3, never produced a set); `sim_deploy/look/make_assets.py` (B2, no
  assets built); `sim_deploy/corrections/geometry.py`, `fit_uw.py` (B4, no output), `geometry_main.py`,
  `fit_main.py`, `main_crop.py`, `export_main_refs.py` (B5, one unreviewed fit and `main_crop_corr*.yaml`); P3's
  `sim_evals/_start_pose_sweep/tools/offline_first_chunk.py`.
- **Not for W2, recorded for W3 / W0:** A2's plan and `sim_evals/_camera_study/A/tools/a2_rows.py`. From A1: replays
  need each demo's own start height, and the sim chain must run one at a time unless real time is shown to hold. The
  check is the `run totals` max lag, not the rtf.

## Overall conclusion

None: the study was paused before it could answer its question. What it established: from `leftx55_t48_h16` the
recorded demo motions can grasp and lift (A1: demos 0, 1, 13), and the failures seen were replay start-height
artefacts and host load, not the pose (details and caveats in [A_baseline](A_baseline/README.md)). Whether the
policies' retreat comes from the camera geometry or from the sim's look remains open, and moves to W2.

## Docs vs raw data (items 1-5 found while writing this overview; 6-13 added when the study was closed)

1. PLAN §10's "the Franka mount sees the fingers / scene ~3-4 % off the handheld view" is not backed by any number in
   DESIGN.md or a raw file found so far (DESIGN has: fingertips 11.8 mm further from the cameras than the rig study
   assumed, E3; training `T_cam_tcp` ~9 mm off the image evidence and tags 13-17 % too small with it, G3 / E5). B's
   measured offset replaces it.
2. PLAN §10's summary of P3 ("with training's IMX219 swapped in, run_006 approaches (+5.4 cm); with the sim's it backs
   away (-1.3 cm)") compares a camera swap with the all-sim case; the matching swap is training IMX519 + sim IMX219 =
   -0.5 cm (offline_split.log). Same conclusion, but the table above is the precise one.
3. The P3 offline numbers were measured at the P3 stand-in start pose (G7 eval `2026-11-20_0957_p3_g7_start`:
   tilt 49.51 deg, 16.18 cm), not at `leftx55_t48_h16`; A2 and C1 measure at the chosen pose.
4. DESIGN.md E5 "What differs from the real lab" lists a plain grey table, grey holders and plain black tags; E7
   ("Appearance") replaced those (wood texture, blue holders, white pads, ArUco tags). The differences P3 still saw:
   the orange striped table vs the pale lab table, hard shadows from two directional lights, the saturated eraser,
   the IMX219's empty table to the horizon and grey sky vs lab clutter; and from E5: no LED glow, motion blur,
   vignetting, sensor noise or auto-exposure / white balance (the real frames are pink-tinted).
5. PLAN §10 says "B is CPU-only rendering"; B's README (`B_geometry_appearance/README.md`, Commands) says "EGL offscreen,
   < 280 MB VRAM per process", i.e. B renders on the GPU while A2's policy needs ~3.3 GB of the 4 GB. To be checked
   by B and D (VRAM headroom while A2 / C1 / C2 run).
6. PLAN §11 "Progress so far": "A1 replays partly run (3/24 grasp at v 0.25 ...)". Raw: **3 of 23** v025 eval
   folders grasp (22 ran: ep 17 is `not_run`; 3 of the 15 not ended by the host). No 24th v025 eval exists
   (`sim_evals/_replay/PICK_TEST1/*/*campA1_v025/episodes.jsonl`). PLAN gives no v1.5 count; raw **2 of 7**.
7. PLAN §11 Known issues: "parallel sim chains overloaded the CPU **once** (realtime fault)". Raw: **two**
   `libfranka: communication_constraints_violation` faults: v025 ep 27 (53.1 ms, `_logs/2026-11-20_104528/sim_pc.log`
   line 97, 3 chains running) and v025 ep 17 (50.1 ms at policy_deploy start-up,
   `.../pick1-v1_ep017/2026-11-20_1056_campA1_v025/policy_deploy.log`, one A1 chain). The interim
   `A/a1_summary.md` (10:54) and `01_start_pose/README.md` list only ep 27. Host-ended runs: interim 9 of 27, raw
   **11 of 30**.
8. PLAN §11 "floor faults are replay start-height artefacts": confirmed for all 9. Their height part is -59.5 to
   -117.4 mm of the carried grasp z, the start-tilt part -14.5 to +3.4 mm. Two other demos (1, 28) carry ~5 cm low
   because of their start tilt (62-64 deg vs 48 deg), not their height; neither faulted
   ([A_baseline](A_baseline/README.md), Analysis).
9. PLAN §11 "B1 reference frames done; B2-B5 ... not done": right, but not empty. B5 left an unreviewed first fit
   (`sim_evals/_camera_study/B/corr_main/fit.json`, `generated_utc` 15:55:51Z, and
   `sim_deploy/corrections/main_crop_corr*.yaml`). B2's tool `sim_deploy/look/make_assets.py` exists with no assets.
   B3 / B4 have tools and no output.
10. Item 5 above (B on the GPU): never tested. No B step rendered before the pause (`B/sets/` holds only the real
    export), and the B tools that ran (`reference.py`, `sets_real_export.py`, `fit_main.py`) are CPU-only by their
    docstrings.
11. "Where the raw data goes" (above) and B's README name `B/index.json`, `B/summary.md`, `A/summary.md`,
    `sim_deploy/corrections/rig_pi_v3_640_corr.yaml` and `stations/sim_rig_look_real.yaml`: none exist. A's tables
    are the interim `A/a1_summary.md` / `a1_episodes.csv`.
12. `sim_evals/index.csv` has a row for every one of the 30 `campA1` folders (rows 44-73). The closing brief's "some
    without an index row" does not hold. The interim `A/a1_summary.md` has internal prose / table mismatches (1662
    vs 1730 shortened, rho 0.27 vs 0.30, 18 vs 22 slow steps, max lift 20.0 vs 20.1 cm), listed in
    [A_baseline](A_baseline/README.md) "Docs vs raw data" 5.
13. Outside this folder, not edited here: `sim-experiments/README.md` (Index row 02: "running (A, B)", finding "-";
    the "Sim-only settings" appearance row: "under study in 02_camera_study"). PLAN §11's "Docs" row ("02 skeleton
    done") is also outdated.
