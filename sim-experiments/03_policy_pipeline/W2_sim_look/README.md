# 03 W2: sim images close enough to real

**Status:** done 2026-11-21 06:45; gate **not met**: robot cell within tolerance for 13 of 14 checkpoints, training scene (mount) for 0 of 14 (details below); reviewed (read-only reviewer, issues fixed or listed) · **Date:** 2026-11-20 (started) to 2026-11-21 · **Who:** W2 coordinator (two sessions;
the first was cut off at 21:29 during run_022's chunks) + subagents (motion blur) + reviewer · **Plan ref:**
sim_deploy/PLAN.md §11 (W2 row), §10 (B and C1, which W2 continues), §6 "P3" G7; DESIGN.md "E5: as built" ("What
differs from the real lab"), "E7: integration" ("Appearance")

> **Eraser width (2026-11-21, after this write-up):** every number below was rendered with the look stations' eraser
> 51 mm wide (127 x 51 x 12; `train` 111 x 52 x 12; robot_trials with each trial's fitted width). Since then every sim
> eraser is 30 mm wide (`sim_deploy/eraser.py` WIDTH_M; lengths and thickness unchanged; robot_trials renders the
> fitted length with that width; `look.make_stations --check` fails on any other width), so a rerun of W2's attribution
> now renders 30 mm and its numbers will differ. Not rerun.

## Question

With a realistic look (table, eraser colour, light, workspace background) and a camera model (colour, noise, JPEG),
is each checkpoint's response to sim images within tolerance of its response to real images of the same situation
(first chunk)? Yes / no per checkpoint, and for a no: which camera (IMX219 / IMX519) and which difference (look,
camera model, geometry).

## Why it matters for the real robot

- The policies did not run in the sim because they reject the sim's images (P3: run_006 +5.6 cm on training images,
  -1.3 cm on the sim's). Until the sim's images pass for a checkpoint, a closed-loop sim result for it (W3) says
  nothing about the arm.
- A look or camera-model fix is sim-only. A **geometry** difference (Franka mount vs the handheld in training) and a
  difference of the **real robot cell** (its table, object, fingers) are real-world ones and go to the deployment as
  recommendations only. The IMX219 / IMX519 view corrections stay an opt-in experiment, off by default.
- The arm's first trials (2026-11-20) recorded the policy's real mount-view inputs. W2 uses their static start sets
  as a second real reference: "the sim at the real arm's start vs the real arm's own images".

## Two real references, two gates

| Gate | Sim side | Real side | What a "yes" means |
|---|---|---|---|
| **Training scene** | look `train` (below), handheld pose (appearance only) and the sim's mount cameras, same arm configuration as the recorded TCP | the 15 B1 training references (5 PICK_TEST1 episodes x `first`, `approach_1of3`, `approach_2of3`) | the sim scene + camera model reproduce the training images well enough for the policy |
| **Robot cell** | look `robot` (the real robot cell from the trial frames), arm at q_start, gap 0.040, eraser fitted per trial | the real arm's policy inputs at each trial's static start (8 trials, 6 sets each; trial 4 of run_022 with a black object left out) | the sim reproduces what the policy saw on the real arm |

Response = the first chunk's end point (step 15, 0.75 s) along the demo's own displacement (`approach`), and along
TCP z; N paired samples (same seeds for every image set). **Tolerance (fixed before the sim sets were run, first
session):** per checkpoint `2 x RMS over references of the std over the neighbouring real frames {T-6, T-3, T, T+3,
T+6} + SE`, i.e. how much the real response moves between real frames 50-100 ms apart (60 neighbour observations per
checkpoint); a cell is within when the RMS over references of the per-reference difference sim - real is <= tol
(the robot cell: RMS over trials, same tol). The mean difference is reported next to the RMS.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` + uncommitted `sim_deploy/`; the copy `deployment/` as W0 left it during the runs (`run_info.json` records the policy files' sha256 per invocation) |
| Inference | `sim_deploy/camera_study/first_chunk.py` (the copy's `Settings.load` + `build_policy`, as `policy_deploy` builds it; gripper map on), bf16, 8 DDIM steps, exec_h 12 (what every checkpoint ran with on the arm on 2026-11-20), CUDA under `sim_evals/.gpu.lock`; N 16 for run_006 / run_022 / PICK_DOUBLE run_002, N 8 for the other 11; prompting checkpoints: prompt `(PICK_TEST1, 0)` from `policy_deploy use` |
| Deploy path check | deploy path (B1 `deploy_obs.npz`) - training path: approach -0.09 cm (RMS 0.12) for run_006, -0.08 (0.12) for run_022 |
| Looks (stations, `sim_deploy/look/make_stations.py`) | see "Looks" below; image-only stations: physics as sim_rig except the eraser box (the looks keep the real 127 x 51 mm eraser, `train` 111 x 52 mm; sim_rig.yaml's is W0's 30 mm one since 2026-11-21 03:17). `check_look physics`: real / real_kit / real_bg_kit bit-identical to each other; `train` differs only in the eraser's size / inertia |
| Camera model | `sim_deploy/look/camera_look_real.yaml` (fitted by `look/fit_camera_look.py`, first session): per camera blur, colour matrix, tone, radial / tilt colour shading, signal-dependent noise; JPEG q90 as the sim Pi; IMX219 table colour W1 distance 25.7 -> 7.4 DN, IMX519 9.1 -> 8.4 DN; edge 10-90 % widths sim 1.28 -> 2.07 (IMX519) / 2.47 (IMX219) px vs real 2.9 |

### Looks

| Look | What changes vs the previous row | Station |
|---|---|---|
| `current` | E7's look: procedural wood, yellow eraser, white pads, key light with shadows | `sim_rig.yaml` |
| `real_nocam` | real table texture from the IMX519 frames at its real place, lime eraser colour, soft diffuse light, grey sky; no camera model | `sim_rig_look_real_nocam.yaml` |
| `real` | + camera model | `sim_rig_look_real.yaml` |
| `real_kit` | + training's kit colours: light-blue pads, blue holders (sampled under the sim's labels in the real frames: pads real (121, 180, 224) vs sim (213, 213, 211)) | `sim_rig_look_real_kit.yaml` |
| `real_bg_kit` | + the PICK_TEST1 workspace mesh and backdrop walls; v2 (used): the sim table cut to the real desk's extent (x -0.05..1.10, y -0.30..0.55 m in O, from the mesh's table faces) and the mesh's low faces over the work area dropped (3 130 faces: dark blobs next to the eraser) | `sim_rig_look_real_bg_kit.yaml` |
| **`train`** | + the eraser at the size the training frames show (111 x 52 mm, not 127 x 51) | `sim_rig_look_train.yaml` |
| `train_blur` | + motion blur: 13 sub-frames over a 40 ms exposure centred on the frame (recorded TCP interpolated between rows, IK per sub-frame, averaged in linear light); run_006 only | `sim_rig_look_train.yaml` + `render_sets --motion-blur-s 0.04 --motion-blur-k 13` |
| `train_c519` (opt-in) | `train` mount sets with the IMX519 crop/zoom correction (`corrections/main_crop_corr.yaml`, scale 1.21) before the copy's ImagePrep | `corrections/main_crop_sets.py` |
| `robot_nocam` / **`robot`** | the real robot cell (white table of the real extent, backdrop painted from the trial frames, dark grey holders, yellow object), without / with the camera model | `sim_rig_look_robot{_nocam}.yaml` |
| `robot_lime`, `robot_kit`, `robot_wood`, `robot_train` | what-if: the robot cell with ONE training element put back (lime eraser; blue pads + holders; wood table), or all three | `sim_rig_look_robot_*.yaml` |

## Method

1. Looks as separate builds (table above); each judged by the attribution response, not by eye.
2. Camera model fitted on region statistics (first session), then judged the same way (`real` vs `real_nocam`).
3. Attribution per checkpoint (`attribution_table.py`): real (15 references), 60 neighbours (tolerance), the real
   arm's static sets (`rm__`), and the sim sets per look: handheld, mount, mount_start (the arm at q_start with the
   eraser where it lay), robot_trials, with the per-camera swaps (sim IMX519 + real IMX219 = "sim IMX519 only", and
   the reverse). Derived: GEOMETRY = mount - handheld (same look), APPEARANCE = look - current.
4. Opt-in: the IMX519 crop correction on the mount sets (`train_c519`). The IMX219 virtual-camera correction
   (`corrections/fit_uw.py`) was not run.

## Commands to reproduce

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
PY=~/envs/sim_deploy/bin/python
# looks: stations + builds (CPU)
$PY -m sim_deploy.look.make_stations
$PY -m sim_deploy.model build --station sim_deploy/stations/sim_rig_look_train.yaml        # one per look
$PY -m sim_deploy.look.check_look physics --a sim_deploy/build/sim_rig_look_real --b sim_deploy/build/sim_rig_look_train
# image sets (EGL on the GPU: under the lock)
flock sim_evals/.gpu.lock $PY -m sim_deploy.camera_study.render_sets --station sim_deploy/stations/sim_rig_look_train.yaml --look-id train --no-robot-trials
flock sim_evals/.gpu.lock $PY -m sim_deploy.camera_study.robot_trials --station sim_deploy/stations/sim_rig_look_robot.yaml --look-id robot
$PY -m sim_deploy.corrections.main_crop_sets --look train --poses mount mount_start        # opt-in, CPU
# attribution for one checkpoint (plans the jobs, then first_chunk.py under the GPU lock)
POSES="handheld mount mount_start robot_trials" SWAP_POSES="handheld mount" \
  sim_deploy/camera_study/attr_run.sh ~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 16 train train_c519 robot
# all 14: sim_evals/_pipeline/W2/attribution/_work/all2.sh; then the tables:
$PY -m sim_deploy.camera_study.attribution_table table
```

## Results

### 1. The look ladder (run_006, N 16, tol 2.7 cm; real +10.2 cm)

approach diff sim - real, mean / RMS over the 15 references (cm); `first` = the 5 static starts only.

| Look | handheld | handheld `first` | mount | mount, sim IMX519 only | mount, sim IMX219 only |
|---|---|---|---|---|---|
| `current` | -11.9 / 12.7 | -9.3 / 9.6 | -12.9 / 14.0 | -5.5 / 6.7 | -7.3 / 7.7 |
| `real_nocam` (table, eraser colour, light) | -4.0 / 6.1 | -1.3 / 2.8 | -7.2 / 8.8 | -0.9 / 4.5 | -6.1 / 6.6 |
| `real` (+ camera model) | -4.1 / 6.0 | -1.5 / 2.7 | -7.5 / 9.1 | -0.8 / 4.4 | -6.5 / 6.9 |
| `real_kit` (+ kit colours) | -4.2 / 6.2 | -1.3 / 2.7 | -7.2 / 8.7 | -0.6 / 4.1 | -6.4 / 6.8 |
| `real_bg_kit` (+ desk extent, lab mesh) | -2.4 / 5.2 | +1.2 / 2.7 | -5.3 / 7.5 | -0.5 / 4.1 | -4.7 / 5.4 |
| **`train`** (+ eraser 111 mm) | **-2.3 / 5.1** | **+0.9 / 2.6 yes** | **-5.0 / 7.2** | -0.4 / 4.0 | -4.4 / 5.0 |
| `train_blur` (+ motion blur 40 ms) | -2.3 / 5.1 | +0.8 / 2.5 | -4.9 / 7.0 | -0.4 / 3.9 | -4.3 / 5.0 |
| `train_c519` (opt-in IMX519 crop) | | | -5.4 / 7.2 | -0.8 / 3.6 | -4.4 / 5.0 |

The bg look before the desk extent (v1, table drawn to the horizon): handheld -3.3 / 5.7, mount -5.6 / 7.7
(`attribution_run006_bg_v1.md`).

### 2. Gate summary, all 14 checkpoints (`attribution.md`, top)

| checkpoint | N | tol | real | handheld | handheld first | handheld moving | mount | mount first | mount, sim IMX519 only | mount, sim IMX219 only | mount + IMX519 crop (opt-in) | robot cell vs real arm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PICK_TEST1__run_006_2026-11-18 | 16 | 2.7 | +10.2 | -2.3 / 5.1 **no** | +0.9 / 2.6 yes | -3.9 / 5.9 **no** | -5.0 / 7.2 **no** | -0.4 / 4.0 **no** | -0.4 / 4.0 **no** | -4.4 / 5.0 **no** | -5.4 / 7.2 **no** | -0.9 / 1.4 yes (real arm +0.2, sim -0.7) |
| PICK_TEST1__run_022_2026-11-19 | 16 | 2.7 | +10.3 | -1.5 / 3.9 **no** | +2.0 / 2.7 yes | -3.3 / 4.4 **no** | -7.8 / 9.8 **no** | -2.0 / 4.2 **no** | -5.9 / 7.9 **no** | -1.6 / 2.0 yes | -7.7 / 9.8 **no** | -1.4 / 1.5 yes (real arm +0.9, sim -0.5) |
| PICK_DOUBLE__run_002_2026-11-19 | 16 | 3.1 | +10.3 | -0.4 / 3.7 **no** | +3.4 / 3.8 **no** | -2.2 / 3.7 **no** | -7.5 / 10.0 **no** | -0.6 / 5.6 **no** | -6.7 / 9.1 **no** | -0.7 / 1.0 yes | -6.3 / 9.3 **no** | +0.4 / 1.0 yes (real arm +1.6, sim +1.9) |
| PICK_DOUBLE__run_003_2026-11-19 | 8 | 3.2 | +11.2 | -1.1 / 4.0 **no** | +2.2 / 3.5 **no** | -2.8 / 4.2 **no** | -8.4 / 10.4 **no** | -1.8 / 4.4 **no** | -6.9 / 9.1 **no** | -1.2 / 1.4 yes | -8.6 / 11.0 **no** | -0.5 / 0.8 yes (real arm +3.1, sim +2.6) |
| PICK_DOUBLE__run_004_2026-11-19 | 8 | 3.2 | +10.9 | -1.7 / 3.7 **no** | +1.6 / 2.1 yes | -3.4 / 4.3 **no** | -9.4 / 10.6 **no** | -4.3 / 5.3 **no** | -7.9 / 9.2 **no** | -1.3 / 1.5 yes | -8.2 / 10.1 **no** | +1.0 / 2.2 yes (real arm +1.1, sim +2.1) |
| PICK_DOUBLE__run_005_2026-11-19 | 8 | 3.2 | +10.8 | -0.9 / 3.3 **no** | +2.7 / 3.0 yes | -2.6 / 3.5 **no** | -9.2 / 10.7 **no** | -3.3 / 4.9 **no** | -8.3 / 9.8 **no** | -0.7 / 0.9 yes | -8.9 / 10.9 **no** | -0.8 / 1.5 yes (real arm +2.2, sim +1.4) |
| PICK_TEST1__run_012_2026-11-18 | 8 | 3.3 | +11.7 | -4.9 / 6.4 **no** | -1.6 / 3.1 yes | -6.6 / 7.5 **no** | -5.8 / 7.3 **no** | -1.9 / 3.7 **no** | -3.3 / 5.2 **no** | -2.5 / 2.7 yes | -6.5 / 7.7 **no** | -2.2 / 2.4 yes (real arm -1.8, sim -4.0) |
| PICK_TEST1__run_014_2026-11-18 | 8 | 3.1 | +10.6 | -5.5 / 7.1 **no** | -2.2 / 3.4 **no** | -7.2 / 8.4 **no** | -7.4 / 8.7 **no** | -3.4 / 5.0 **no** | -3.4 / 5.0 **no** | -3.8 / 4.2 **no** | -7.7 / 8.9 **no** | -2.1 / 2.4 yes (real arm -1.4, sim -3.5) |
| PICK_TEST1__run_016_2026-11-18 | 8 | 3.0 | +10.9 | -3.5 / 5.3 **no** | -1.1 / 2.0 yes | -4.8 / 6.3 **no** | -6.4 / 7.7 **no** | -2.9 / 4.3 **no** | -2.0 / 4.4 **no** | -4.1 / 4.5 **no** | -6.5 / 7.6 **no** | -2.9 / 3.1 **no** (real arm +0.4, sim -2.5) |
| PICK_TEST1__run_020_2026-11-19 | 8 | 3.3 | +11.1 | -7.2 / 8.3 **no** | -4.3 / 5.3 **no** | -8.6 / 9.4 **no** | -8.1 / 9.3 **no** | -4.3 / 5.7 **no** | -4.4 / 6.2 **no** | -3.7 / 3.8 **no** | -8.4 / 9.3 **no** | -1.7 / 3.0 yes (real arm -0.9, sim -2.7) |
| PICK_TEST1__run_024_2026-11-19 | 8 | 2.9 | +10.9 | -2.1 / 4.1 **no** | +0.8 / 2.4 yes | -3.5 / 4.8 **no** | -6.6 / 8.5 **no** | -0.7 / 3.8 **no** | -4.5 / 7.1 **no** | -1.8 / 2.2 yes | -6.5 / 8.8 **no** | -0.8 / 1.0 yes (real arm +1.6, sim +0.8) |
| PICK_TEST1__run_028_2026-11-19 | 8 | 3.5 | +11.5 | -5.9 / 7.2 **no** | -2.7 / 4.2 **no** | -7.6 / 8.3 **no** | -10.3 / 11.2 **no** | -5.7 / 6.6 **no** | -7.4 / 8.5 **no** | -2.7 / 2.9 yes | -9.9 / 10.9 **no** | -1.9 / 2.9 yes (real arm +2.3, sim +0.3) |
| PICK_TEST1__run_030_2026-11-19 | 8 | 3.1 | +10.9 | -5.7 / 6.6 **no** | -2.7 / 4.4 **no** | -7.2 / 7.5 **no** | -10.1 / 11.3 **no** | -5.3 / 6.8 **no** | -6.3 / 7.8 **no** | -3.7 / 4.2 **no** | -9.7 / 10.9 **no** | -0.2 / 0.7 yes (real arm +0.7, sim +0.5) |
| PICK_TEST1__run_031_2026-11-19 | 8 | 3.1 | +10.8 | -3.6 / 5.4 **no** | -0.3 / 3.0 yes | -5.2 / 6.2 **no** | -7.0 / 8.1 **no** | -3.3 / 4.3 **no** | -5.7 / 6.8 **no** | -1.3 / 1.5 yes | -6.5 / 8.0 **no** | -0.5 / 1.0 yes (real arm +1.8, sim +1.2) |

### 3. Robot cell: what moves the response (N 16)

approach (cm) at the real arm's trial starts: sim mean, and sim - real arm mean / RMS over 8 trials.

| Look | run_006 (real arm +0.2) | run_022 (+0.9) | PICK_DOUBLE run_002 (+1.6) |
|---|---|---|---|
| `current` | -4.3; -4.5 / 4.7 **no** | -0.8; -1.7 / 1.8 yes | -3.5; -5.1 / 5.2 **no** |
| `robot_nocam` | -0.9; -1.1 / 1.4 yes | - | - |
| **`robot`** | -0.7; -0.9 / 1.4 yes | -0.5; -1.4 / 1.5 yes | +1.9; +0.4 / 1.0 yes |
| `robot_lime` (training's lime eraser) | -0.6 | -0.4 | +1.9 |
| `robot_kit` (blue pads + holders) | -1.0 | -0.5 | +2.1 |
| `robot_wood` (wood table) | +3.1 | +0.3 | -0.3 |
| `robot_train` (all three) | +1.3 | +0.3 | +0.0 |
| for comparison: `train` scene at the mount, arm at q_start (`mount_start`) | +7.5 | +7.7 | +9.5 |
| real training `first` frames | +8.8 | +9.4 | +9.2 |


## Analysis

- **The look is what made the policies back away in the sim, and most of it is the scene, not the camera.** On
  run_006 the realistic table, eraser colour and soft light moved the handheld response by +7.8 cm (-11.9 -> -4.0)
  and the static starts from -9.3 to -1.3 cm. Cutting the table at the real desk's edge and putting the lab behind it
  added +1.7 (IMX219 only: -3.6 -> -1.9). The camera model (colour, shading, noise, JPEG; it brings the IMX219 table
  colour from 25.7 to 7.4 DN) moved the response by -0.1 to +0.3 cm, the kit colours by < 0.2 cm, the eraser size by
  +0.1 to +0.3 cm, and 40 ms of motion blur by < 0.2 cm on every cell. P3's "the sim's IMX219 image" was right
  about the camera, but the cause was the table running to the horizon in the fisheye, not its colour.
- **Robot cell: sim = real arm for 13 of 14 checkpoints** (look `robot`; run_016 -2.9 / 3.1 against tol 3.0). The
  `current` look was outside for run_006 (-4.5) and run_002 (-5.1); the robot look brings both within (-0.9, +0.4).
  On the real arm every checkpoint's response at the start is near zero (-1.8 to +3.1 cm) where the training starts
  give +9 to +12: the arm's "no approach" of 2026-11-20 is reproduced by the sim.
- **Why the real arm does not approach (sim what-ifs, not tested on the arm):** the training scene at the mount
  cameras from the start pose gives +7.5 / +7.7 / +9.5 cm (run_006 / 022 / 002), within 1-2 cm of the training
  starts, so the mount geometry at the start costs little. The robot cell gives ~0. Putting training's lime eraser
  or kit colours into the robot cell changes < 0.5 cm; the wood table +3.8 cm for run_006 and nothing for the plain
  ones; all three together +0 to +2. So most of the gap is in what the robot cell shows beyond those three: the
  surroundings (person, walls, the white table's extent) and where the object lies relative to the fingers at the
  start.
- **Training scene, mount: not within for any checkpoint, and it is geometry.** With the `train` look, GEOMETRY (mount -
  handheld, same look) is -0.9 to -2.9 cm for the prompting checkpoints and -3.5 to -8.4 cm for the plain ones
  (PICK_DOUBLE -7.2 to -8.4); the static starts are close (mount
  `first` -0.4 to -5.7) and the references in motion carry most of it. The per-camera swap splits it by kind: the
  plain checkpoints (run_022/024/028/030/031, PICK_DOUBLE 002-005) lose it on the **IMX519** (sim IMX519 only -4.5
  to -8.3; sim IMX219 only within for 8 of 9), the prompting run_006 on the **IMX219** (-4.4; IMX519 only -0.4).
  The fitted IMX519 crop/zoom (scale 1.21) does not fix it (-6.5 to -9.9, about unchanged): a 2-D crop cannot
  reproduce the mount's different viewpoint.
- **Handheld (appearance only) is close but not within:** the static starts pass for 8 of 14 checkpoints, the
  moving references are 2-9 cm short for all. Motion blur is not the cause (no change). Still open: the per-episode
  eraser placement (fitted, up to 14 px off in the IMX519) matters most when the eraser is near, and the sim's table
  texture has less fine detail than the real wood (Laplacian variance 8 vs 18-24).
- **Prompting vs plain:** the prompting checkpoints (012/014/016/020) are the farthest off on the handheld
  (-3.5 to -7.2) and run_016 is the only robot-cell miss; the plain ones are closer on appearance and worse on the
  IMX519 geometry.

## Conclusion

**Gate not met.** Per checkpoint, sim response within tolerance of the real-image response:

| Checkpoint | Kind | Robot cell vs real arm (look `robot`) | Training scene, mount (look `train`) | Training scene, handheld (appearance only) |
|---|---|---|---|---|
| PICK_TEST1 run_006 | prompting | yes | no (IMX219) | no (static starts yes) |
| PICK_TEST1 run_012 | prompting | yes | no | no (static yes) |
| PICK_TEST1 run_014 | prompting | yes | no | no |
| PICK_TEST1 run_016 | prompting | **no** (-2.9 / 3.1, tol 3.0) | no | no (static yes) |
| PICK_TEST1 run_020 | prompting | yes | no | no |
| PICK_TEST1 run_022 | plain | yes | no (IMX519) | no (static yes) |
| PICK_TEST1 run_024 | plain | yes | no (IMX519) | no (static yes) |
| PICK_TEST1 run_028 | plain | yes | no | no |
| PICK_TEST1 run_030 | plain | yes | no | no |
| PICK_TEST1 run_031 | plain | yes | no (IMX519) | no (static yes) |
| PICK_DOUBLE run_002 | plain | yes | no (IMX519) | no |
| PICK_DOUBLE run_003 | plain | yes | no (IMX519) | no |
| PICK_DOUBLE run_004 | plain | yes | no (IMX519) | no (static yes) |
| PICK_DOUBLE run_005 | plain | yes | no (IMX519) | no (static yes) |

For W3: run the closed loop in the `robot` look (sim_rig_look_robot.yaml) to reproduce the arm, and in the `train`
look to see what the policies do in their training scene; the `current` look is not usable for policy runs. For the
real deployment (recommendations, not changes): the robot cell, not the mount at the start pose, is what makes the
policies stay put (training scene at the mount from the start: +7.5 to +9.5 cm); the IMX519 / IMX219 mount
geometry matters once the arm moves (plain checkpoints: IMX519).

## Limitations / sim-vs-real caveats

- First-chunk response is a proxy: a checkpoint can pass it and still drift later in a rollout (W3 checks that).
- All 14 checkpoints are judged on PICK_TEST1 references (B1 has none for PICK_DOUBLE): for the four PICK_DOUBLE
  checkpoints the training-scene comparison is on a neighbouring task's frames (their real response there is still
  +10 cm, i.e. in distribution enough to approach).
- The robot cell's real reference is 8 trials on one day (of run_022, run_006, PICK_DOUBLE run_002 / 003), at the
  static start only; every checkpoint is judged on those same 8 starts, and the real arm's response there is small
  for all 14 (-1.8 to +3.1 cm), so "within" there is a test near zero motion.
- The tolerance comes from real frame-to-frame spread at 60 Hz neighbours; it is tight (2.7-3.5 cm) against
  per-reference sim differences that scatter by 3-5 cm even when their mean is near 0.
- The looks are fitted to B1's 5 episodes (table texture, eraser colour, kit colours, camera model); the references
  used to judge them are the same episodes' frames (no held-out episodes).

## Open questions

1. What in the robot cell, beyond the eraser colour, kit colours and table texture, holds the response near zero?
   Next what-ifs: the backdrop (person, walls) swapped for the training lab, the object placed where the training
   starts had it relative to the fingers.
2. Why the moving references come out 2-9 cm short on the handheld (appearance only): eraser placement accuracy near
   the fingers, table texture detail; not motion blur.
3. The IMX219 virtual-camera correction (`corrections/fit_uw.py`) was not run; the IMX519 crop did not help. A
   geometry fix would have to move the cameras (rig mount), which is a hardware recommendation (memory: IMX219
   ~20-25 mm back).
4. Tolerance: one SE in it although a difference carries two (sqrt(2) x SE, ~0.1 cm more); the `first`-only cells
   (5 references) are noisy.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| attribution tables (gate summary on top) | `sim_evals/_pipeline/W2/attribution/attribution.{md,json}` |
| first chunks per checkpoint and observation | `sim_evals/_pipeline/W2/attribution/chunks/<task>__<run>/<id>.{npz,json}` + `run_info.json` |
| the bg look v1 (before the desk extent) | `sim_evals/_pipeline/W2/attribution/attribution_run006_bg_v1.{md,json}`, `chunks_archive/real_bg_kit_v1_run006/` |
| jobs files, frozen stations, batch scripts | `sim_evals/_pipeline/W2/attribution/_work/` (`all.sh` stopped after run_022; `all2.sh` the final run) |
| motion blur calibration (exposure 8-50 ms, Laplacian ratio moving / static) | scratch only: `/tmp/claude-1000/-home-ruben-Capstone-main-sim/be23ff1a-12cc-4a44-ad2b-37ded29df196/scratchpad/blur/` (not kept); the chosen parameters are in each `train_blur` set's `meta.json` |
| batch logs | `sim_evals/_pipeline/W2/logs/` (`first_chunk_<task>__<run>_<date>_<time>.log`, `render_sets_<look>.log`, `robot_trials_<look>.log`) |
| image sets per look and pose | `sim_evals/_pipeline/W2/sets/<look>/<pose>/<ref>/`, `sets/index.json`, contact sheets `sets/contact_<look>.png` |
| real arm static sets | `sim_evals/_pipeline/W2/attribution/real_mount/` |
| real neighbours | `sim_evals/_pipeline/W2/attribution/real_neighbours/` |
| camera model fit | `sim_evals/_pipeline/W2/camera/camera_look_sim_rig_look_real.{json,yaml}` (installed as `sim_deploy/look/camera_look_real.yaml`) |
| look checks (physics, side by side, robot compare) | `sim_evals/_pipeline/W2/look/` |
| study it continues | [02_camera_study](../../02_camera_study/README.md) |

## 30 mm rerun, 2026-11-22

**Question:** W2's robot-cell attribution (look `robot`, 8 real-arm trial starts) was rendered with each trial's
fitted eraser width (46-51 mm, the real 127 x 51 mm object). Since 2026-11-21 every sim eraser is 30 mm wide. Does the
policy's first chunk in the robot look change materially at 30 mm, and does any checkpoint's robot-cell pass / fail
change? There is no real image of a 30 mm object: the comparison is sim (30 mm) vs the real arm's frames (51 mm
object) with W2's tolerance, and sim (30 mm) vs sim (51 mm) at the same pose and seeds.

**Procedure (same as W2):** same 14 checkpoints, N (16 for run_006 / run_022 / PICK_DOUBLE run_002, 8 otherwise),
bf16 / 8 DDIM steps / exec_h 12, seeds, metrics and per-checkpoint tolerance; W2's `robot_trials_fit.json` reused
(same eraser poses, fitted length / thickness), only the rendered width is 30 mm (`robot_trials.py` as it stands,
look id `robot_w30`, with the per-camera swaps). Two controls: (a) the current tools re-rendering with the fitted
width (`robot_w51ctl`) give images **bit-identical** to W2's `robot` sets; (b) W2's 51 mm images re-inferred today
(`robotrep`) give **bit-identical** first-chunk metrics (difference 0.00 for every checkpoint). So the only change
between the columns below is the eraser width. 30 mm vs 51 mm changes ~300-500 px per IMX519 image and ~100-200 px
per IMX219 image (of 224 x 224, > 10 DN).

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH; B=sim_evals/_pipeline/W2/w30_rerun_2026-11-22
flock sim_evals/.gpu.lock ~/envs/sim_deploy/bin/python $B/_work/render_w30.py w30 w51ctl   # sets (EGL)
~/envs/sim_deploy/bin/python $B/_work/jobs_w30.py                                         # 32 jobs per checkpoint
$B/_work/run_all.sh                                                                       # first chunks, GPU lock per checkpoint
~/envs/sim_deploy/bin/python $B/_work/analyse_w30.py                                      # -> $B/robot_w30_vs_w51.{md,json}
```

**Result** (approach, cm; sim - real arm mean / RMS over the 8 trials vs tol; `30-51` = sim 30 mm - sim 51 mm, mean /
RMS over trials):

| Checkpoint | tol | real arm | 51 mm (W2) | 30 mm | 30-51 |
|---|---|---|---|---|---|
| PICK_TEST1 run_006 | 2.7 | +0.2 | -0.9 / 1.4 yes | -0.7 / 1.3 yes | +0.20 / 0.30 |
| PICK_TEST1 run_012 | 3.3 | -1.8 | -2.2 / 2.4 yes | -1.4 / 1.8 yes | +0.78 / 0.98 |
| PICK_TEST1 run_014 | 3.1 | -1.4 | -2.1 / 2.4 yes | -2.2 / 2.4 yes | -0.07 / 0.20 |
| PICK_TEST1 run_016 | 3.0 (2.97) | +0.4 | -2.9 / 3.11 **no** | -2.7 / 2.96 yes (by 0.01) | +0.23 / 0.47 |
| PICK_TEST1 run_020 | 3.3 | -0.9 | -1.7 / 3.0 yes | -1.5 / 2.7 yes | +0.25 / 0.50 |
| PICK_TEST1 run_022 | 2.7 | +0.9 | -1.4 / 1.5 yes | -1.1 / 1.3 yes | +0.21 / 0.29 |
| PICK_TEST1 run_024 | 2.9 | +1.6 | -0.8 / 1.0 yes | -0.7 / 1.0 yes | +0.07 / 0.33 |
| PICK_TEST1 run_028 | 3.5 | +2.3 | -1.9 / 2.9 yes | -2.3 / 3.1 yes | -0.34 / 0.44 |
| PICK_TEST1 run_030 | 3.1 | +0.7 | -0.2 / 0.7 yes | -0.0 / 0.7 yes | +0.13 / 0.22 |
| PICK_TEST1 run_031 | 3.1 | +1.8 | -0.5 / 1.0 yes | -0.6 / 1.0 yes | -0.04 / 0.13 |
| PICK_DOUBLE run_002 | 3.1 | +1.6 | +0.4 / 1.0 yes | +0.6 / 1.2 yes | +0.24 / 0.52 |
| PICK_DOUBLE run_003 | 3.2 | +3.1 | -0.5 / 0.8 yes | -0.1 / 0.9 yes | +0.42 / 0.57 |
| PICK_DOUBLE run_004 | 3.2 | +1.1 | +1.0 / 2.2 yes | +0.3 / 1.9 yes | -0.64 / 0.74 |
| PICK_DOUBLE run_005 | 3.2 | +2.2 | -0.8 / 1.5 yes | -0.5 / 1.7 yes | +0.32 / 0.52 |

Per-camera swaps at 30 mm (sim IMX519 only / sim IMX219 only) are within for all 14; z and the chunk distance d are in
`robot_w30_vs_w51.md` (z: run_016 stays outside, -2.9 / 3.1 vs 3.0; all others within; d within for all 14 at both
widths).

**Interpretation:**
- The first chunk does **not** change materially with the eraser width: sim 30 mm - sim 51 mm is -0.6 to +0.8 cm in
  the mean (RMS over trials <= 1.0 cm, chunk distance 0.2-0.6 cm), well below every tolerance (2.7-3.5 cm) and below
  the trial-to-trial scatter of sim - real arm. The policies stay near zero approach in the robot cell with either
  object; most checkpoints move slightly toward the real arm (mean |sim - real| shrinks for 10 of 14).
- Pass / fail: 13/14 at 51 mm becomes 14/14 at 30 mm. The one change is run_016, which sat on the line at 51 mm
  (RMS 3.11 vs tol 2.97) and is now just inside (2.96): a borderline flip within noise, not a real change; its z stays
  outside at both widths. **W2's conclusion is unchanged**: the robot look reproduces what the policies saw on the
  real arm, and the 30 mm sim eraser does not alter it.
- Caveat: this says the policy is about equally (in)sensitive to 30 vs 51 mm *in the robot cell at the start pose*,
  where every checkpoint barely moves. It does not tell how the policies respond to a narrow object once they
  approach (W3), and the real arm has not been run with a <= 38 mm object.

**Raw data:** `sim_evals/_pipeline/W2/w30_rerun_2026-11-22/` (`robot_w30_vs_w51.{md,json}`; `sets/robot_w30`,
`sets/robot_w51ctl` + contact sheets; `chunks/<task>__<run>/` the new chunks; `_work/` scripts, jobs, frozen stations,
`merged/` = symlinks of W2's chunks + the new ones used by `attribution_table.analyse`; `logs/`). W2's 51 mm results
are untouched.
