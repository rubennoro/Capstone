# 02 B: camera geometry vs appearance (handheld vs mount, sim look vs lab)

## Status at pause (2026-11-20, added when the study was closed)

**B1 done; B2-B5 and the verification step not done.** The camera study was paused on 2026-11-20 (the last B file
was written at 10:55:52) and folded into workstream W2 of `sim_deploy/PLAN.md` §11 ("Sim images close enough to
real"): [../../03_policy_pipeline/W2_sim_look/README.md](../../03_policy_pipeline/W2_sim_look/README.md). Everything
below this section is phase agent B's plan, written at 10:45. Its status line ("running") and its "pending" / "TODO-verify"
entries date from then. Checked against the files: none of its numbers was proven wrong, so nothing below was edited.
This section states only what is on disk. It claims no geometry or appearance result.

### Steps at the pause

B2-B5 use the numbering in B's Method below (PLAN §10 phase B: looks, renders, IMX219 correction, IMX519 crop / zoom).
Paths are relative to `~/Capstone-main-sim/`. Times are local (EST) file mtimes.

| Step | What (B's Method) | State | On disk |
|---|---|---|---|
| B1 | real 224 inputs of the 15 reference frames, through training's sampler and through the deploy copy's `ImagePrep` | **done** | `sim_evals/_camera_study/B/reference/` (details below); tool `sim_deploy/camera_study/reference.py` (582 lines, 10:53) |
| B2 | looks `real` / `real+mesh` (assets, model builds, histogram distance) | not done | `B/look/` empty; `sim_deploy/look/assets/` empty; `sim_deploy/look/make_assets.py` (342 lines, 10:55) exists, but `camera_look.py` and `fit_look.py` (named in `sim_deploy/look/__init__.py`) do not; no look station in `sim_deploy/stations/` (no `sim_rig_look_real.yaml`) and no look build (`sim_deploy/build/` has only `sim_rig/`, `166a0649…`) |
| B3 | render sets handheld / mount / mount_start x looks x corrections | not done | `B/sets/` holds only `_real/`: 15 `.npz` (the real training images and low-dim rows of the 15 references, exported by `sim_deploy/camera_study/sets_real_export.py` at 10:48 as `render_sets.py`'s input cache). No rendered set. `sim_deploy/camera_study/render_sets.py` (1189 lines, last edited 10:56:09) |
| B4 | IMX219 virtual-camera rotation + zoom (`corr_uw`) | not done | `B/corr_uw/` empty; no `sim_deploy/corrections/rig_pi_v3_640_corr.yaml`; tools `sim_deploy/corrections/fit_uw.py` (646 lines, 10:56:18) and `geometry.py` (457 lines) |
| B5 | IMX519 crop / zoom (`corr_main`) | not done: one first fit on disk, unreviewed | `B/corr_main/` 7 files, 6.8 MB: `fit.json` (`generated_utc` 2026-11-20T15:55:51Z = 10:55:51 EST, a minute before the pause), `refs_main.npz` (10:50), `e2e_dots_640_in.png`, `e2e_dots_224_corrected.png`, `overlay_all.png`, `overlay_ep14_first_fits.png`, `overlay_ep14_approach_2of3_fits.png`. The same run wrote `sim_deploy/corrections/main_crop_corr.yaml`, `main_crop_corr_fingers_tags.yaml` and `main_crop_corr_table.yaml` (the parameters are in these YAMLs; `main_crop.py` only applies them). Tools: `fit_main.py`, `geometry_main.py`, `main_crop.py`, `export_main_refs.py`. Nobody reviewed it, step 5 did not check it, and nothing uses it. Its numbers are not results of this study |
| 5 | verification: projection vs detection | not done | none |
| - | `B/index.json`, `B/summary.md` (in Raw data below) | not written | none |

The B tools present at the pause are sim-only and off by default (`sim_deploy/corrections/__init__.py`: "Nothing here
is a default"). By their docstrings, `reference.py` stage 1, `fit_uw.py` and `fit_main.py` run on the CPU without
rendering. Only `render_sets.py` (B3) renders.

### B1: what is done

The 15 reference frames are from `sim_evals/_camera_study/B/layout.json` (10:40): package `PICK_TEST1 pick1-v1`,
5 episodes x 3 rows (60 Hz data rows from the episode start). `first` = row 6, the first row with two distinct IMX219
frames. `approach_1of3` / `approach_2of3` are about 1/3 and 2/3 of the way to the recorded grasp start, on a multiple
of 6.

| Package index | Episode (`reference/index.json`) | Grasp start row | Rows: `first` / `approach_1of3` / `approach_2of3` |
|---|---|---|---|
| 2 | `episode_000140` | 109 | 6 / 36 / 72 |
| 9 | `episode_000147` | 121 | 6 / 42 / 78 |
| 14 | `episode_000152` (the G3 / P3 reference) | 114 | 6 / 36 / 78 |
| 19 | `episode_000157` | 112 | 6 / 36 / 72 |
| 25 | `episode_000163` | 110 | 6 / 36 / 72 |

`layout.json` also fixes `T_O_tag` (package table frame -> O). The yaw puts the demos' median start heading
(-177.79 deg in the tag frame) on O +y, the median grasp TCP xy (0.0121, 0.1168) maps to O (0.55, 0.00), and tag z 0
= O z 0. The file says it is owned by coordinator B ("do not change").

`sim_evals/_camera_study/B/reference/` (51 MB, 247 files, all written 10:54:15-10:54:57):

- 15 folders `ep<ii>_<label>/`, 15 files each: `obs.npz` (the training input: exactly the `Observation` keys),
  `main_0/1.png`, `uw_0/1.png` (the 224 images training saw, oldest first); `deploy_obs.npz`, `deploy_main_0/1.png`,
  `deploy_uw_0/1.png` (the same frames through the deploy path: raw video -> INTER_AREA 640x480 -> the copy's
  `ImagePrep` with `rig_pi_v3_640.yaml`); `raw640_main_0/1.png`, `raw640_uw_0/1.png` (the 640x480 decoded raw
  frames); `meta.json` (rows, frame indices, widths, TCP poses in the tag frame at T-6 / T-3 / T and at the grasp,
  the image comparisons, the checks).
- `index.json` (the 15 references, their files and episodes), `raw_extra/`: 10 more 640x480 frame pairs at row 6
  (package indices 0, 6, 7, 8, 10, 12, 16, 21, 28, 29, picked by farthest-point sampling of the grasp TCP xy) plus
  its `index.json`.

What the 15 `meta.json` record (read here and not recomputed; MAD = mean absolute difference in 8-bit levels,
`reference.py` line 424):

| Check | IMX519 (main), 30 images | IMX219 (uw), 30 images |
|---|---|---|
| deploy path vs the training (zarr) image: MAD / PSNR | 0.402-0.533 / 48.39-49.87 dB | 0.570-0.688 / 45.77-47.36 dB |
| deploy path + JPEG q90 vs the training image: MAD / PSNR | 0.592-0.707 / 46.46-47.67 dB | 0.937-1.098 / 42.17-43.50 dB |
| best integer shift deploy vs training | (0, 0) in all 30 | (0, 0) in all 30 |
| low-dim: training conversion vs the copy's `PoseConverter` | max abs difference <= 1.2e-7, `check_values`: "ok (train and deploy)" in all 15 | |
| row -> frame map vs the episode's stored IMX219 upsample index | `uw_upsample_matches_episode_zarr: true` in all 15 | |
| index 14 vs P3's `ep152_ref.npz` | `p3_ep152_ref_check` all 0 (in the three `ep14_*` references) | |

B's Method gave B1's pass as "the two agree (pending threshold)". No threshold was ever set, so B1 is "done" as a
set of files, not as a passed check. Copied here: the `ep14_first` row-T images from training and from the deploy
path, [figures/](figures/) `ep14_first_main_1.png`, `ep14_first_uw_1.png`, `ep14_first_deploy_main_1.png`,
`ep14_first_deploy_uw_1.png` (from `sim_evals/_camera_study/B/reference/ep14_first/`).

Commands from `reference.py`'s docstring (not rerun while closing):

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.camera_study.reference package   # stage 1 (CPU)
~/envs/sim_policy_deploy/bin/python       -m sim_deploy.camera_study.reference raw       # stage 2 (PyAV decode)
~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.camera_study.reference all       # both
```

The B1 command in B's "Commands to reproduce" below (`python sim_deploy/camera_study/reference.py`, TODO-verify)
does not match: the tool runs as a module, with a stage argument.

### Corrections to the plan below (paths, not numbers)

- `sim_deploy/corrections/rig_pi_v3_640_corr.yaml` and `stations/sim_rig_look_real.yaml` do not exist.
- The IMX519 crop / zoom parameters are in `sim_deploy/corrections/main_crop_corr*.yaml`, applied by `main_crop.py`.
- `B/index.json` and `B/summary.md` were never written.

---

**Status:** running · **Date:** 2026-11-20 · **Who:** agent B (+ sub-agents B1-B6) ·
**Plan ref:** sim_deploy/PLAN.md §10 phase B; DESIGN.md "E5", "E7", "P3/P4"

## Question

In the policy's 224 x 224 inputs: (1) how far (px, %) is the Franka mount's view of the fingers and scene from the
handheld training view, per camera; (2) how far is the sim's look from the lab's (histogram distance per look);
(3) how much of the geometric offset do the candidate corrections (IMX219 virtual-camera rotation + zoom, IMX519
crop/zoom) remove?

## Why it matters for the real robot

Geometry is a real-world problem: the Franka's camera mount puts the cameras where they are, so an offset found here
is on the real robot too, and a correction would go into the real rig file / frame prep. Appearance (table texture,
eraser colour, light, background) is a sim-only problem: the real robot sees the lab.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb3b23a` (+ uncommitted `sim_deploy/`, `deployment/`, `sim-experiments/`) |
| Model | `sim_deploy/build/sim_rig/model.json` `model_sha256` `166a0649571a…` (look "current": table top O z 0, procedural wood table, directional lights, kit colours, ArUco tag textures; built 2026-11-20T14:40:51Z from `3633fcb-dirty`). Look variants: pending (shas of `build/sim_rig_look_real/`, `…real+mesh`) |
| Home / start pose | `mount_start` only: `q_start_rad [-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]` (`leftx55_t48_h16`, TCP O (0.531, -0.159, 0.160), 48 deg tilt, heading left) |
| Eraser placement | pending (per reference frame, from the package via `T_O_tag`) |
| Stations / overlays | `stations/sim_rig.yaml` (current look); look variants `stations/sim_rig_look_real.yaml` (TODO-verify), real+mesh: pending |
| Policy / checkpoint | none (no forward passes in B; those are C1) |
| Inference settings | not run |
| Seeds / episodes | reference frames: package `PICK_TEST1/pick1-v1` indices 14 (= episode_000152, the G3/P3 reference), 2, 9, 19, 25 x 3 rows (labels `first` / `approach_1of3` / `approach_2of3`, 60 Hz rows from `layout.json`) = 15 frames |
| Timeout | not used |
| Sim-only settings | look variants; camera corrections (default off) |

Variants (every set = 15 reference frames x 2 cameras):

| Axis | Values |
|---|---|
| look | `current` (sim_rig) / `real` (table texture, eraser colour sampled from training frames, diffuse light) / `real+mesh` (+ PICK_TEST1 workspace mesh from `umiproc viz` as background) |
| camera pose | `handheld` = training's `T_cam_tcp` (rig file) + E7's fix (IMX219 and IMX519 rigidly translated (-4.0, +2.5, +8.8) mm in IMX219 axes; nominal pose reported alongside) / `mount` = sim model's mounted cameras at the same TCP pose / `mount_start` = mounted cameras with the arm at the start pose |
| correction | none / IMX219 rig-file virtual camera rotation + zoom (`sim_deploy/corrections/rig_pi_v3_640_corr.yaml`) / IMX519 crop/zoom (`sim_deploy/corrections/main_crop.py`) |

Frame map: `T_O_tag` (`layout.json`) = package table (tag) frame -> O: yaw so the demos' median start heading
(-177.79 deg in the tag frame) = O +y, median grasp TCP xy (0.0121, 0.1168) tag -> O (0.55, 0.00), tag z 0 = O z 0.

## Method

1. **B1 reference inputs.** The real 224 images of the 15 frames, two ways: through training's sampler
   (`behavior_prompting/.../train_network/common/sampler.py`, `config/task/umi.yaml`) and through the deploy copy's
   `ImagePrep` (`deployment/policy_deploy/policy_deploy/frames/_dataproc/imageprep.py`, `rig_pi_v3_640.yaml`).
   Pass: the two agree (pending threshold).
2. **B2 looks.** Model builds `real` and `real+mesh` (`python -m sim_deploy.model build --station ...`); per look the
   colour histogram distance of sim renders vs the real frames.
3. **B3 renders.** Handheld vs mount (vs mount_start) for every frame and look, through the sim Pi chain
   (`sim_deploy/sim_pi/` render, JPEG q90, then `ImagePrep`) -> the policy's 224 images.
4. **B4 / B5 corrections.** Projection-based fits: B4 the IMX219 virtual-camera rotation + zoom
   (`sim_deploy/corrections/fit_uw.py`), B5 the IMX519 crop/zoom (`sim_deploy/corrections/fit_main.py`); offsets
   before/after.
5. **Verification.** Geometry offsets by two methods: projection of known points (fingertips, tags) vs detection in
   the rendered / real images (ArUco tags as in `sim_deploy/sim_pi/handheld_check.py`). Pass: the two agree (pending
   threshold).

## Commands to reproduce

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# No sim chain ports, no policy; EGL offscreen, < 280 MB VRAM per process. Arguments: TODO-verify.
~/envs/sim_policy_deploy_torch/bin/python sim_deploy/camera_study/reference.py       # B1, TODO-verify
~/envs/sim_deploy/bin/python -m sim_deploy.model build --station sim_deploy/stations/sim_rig_look_real.yaml  # B2, TODO-verify
~/envs/sim_deploy/bin/python sim_deploy/camera_study/render_sets.py                  # B3, TODO-verify
~/envs/sim_deploy/bin/python sim_deploy/corrections/fit_uw.py                        # B4, TODO-verify
~/envs/sim_deploy/bin/python sim_deploy/corrections/fit_main.py                      # B5, TODO-verify
```

Run time: pending.

## Results

pending.

Geometry offsets (224 px; mean / max over the 15 frames):

| Camera | Class | handheld vs real | mount vs real | mount vs handheld | mount + corr vs handheld |
|---|---|---|---|---|---|
| IMX519 (main) | fingers / tags | pending | pending | pending | pending |
| IMX519 (main) | scene (table, eraser) | pending | pending | pending | pending |
| IMX219 (ultrawide) | fingers / tags | pending | pending | pending | pending |
| IMX219 (ultrawide) | scene (table, eraser) | pending | pending | pending | pending |

Projection vs detection (verification):

| Camera | Pose | projection offset (px) | detection offset (px) | difference (px) |
|---|---|---|---|---|
| pending | | | | |

Fitted corrections:

| Correction | Parameters | File |
|---|---|---|
| IMX219 virtual camera | rotation (deg), zoom | `sim_deploy/corrections/rig_pi_v3_640_corr.yaml` |
| IMX519 crop/zoom | crop, zoom | `sim_deploy/corrections/main_crop.py` |

Appearance (histogram distance sim vs real, per camera; mean over the 15 frames):

| Look | model_sha256 | IMX519 | IMX219 |
|---|---|---|---|
| current | `166a0649571a…` | pending | pending |
| real | pending | pending | pending |
| real+mesh | pending | pending | pending |

Image-set inventory (`sim_evals/_camera_study/B/sets/`):

| Set | Look | Pose | Correction | Frames x cameras | Path |
|---|---|---|---|---|---|
| pending | | | | | |

## Analysis

pending.

## Conclusion

pending.

## Limitations / sim-vs-real caveats

pending.

## Open questions

pending.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| frame selection + `T_O_tag` | `sim_evals/_camera_study/B/layout.json` |
| index of all sets | `sim_evals/_camera_study/B/index.json` |
| summary | `sim_evals/_camera_study/B/summary.md` |
| B1 real reference inputs | `sim_evals/_camera_study/B/reference/` |
| B2 looks | `sim_evals/_camera_study/B/look/` |
| B3 render sets | `sim_evals/_camera_study/B/sets/` |
| B4 IMX219 correction fit | `sim_evals/_camera_study/B/corr_uw/` |
| B5 IMX519 correction fit | `sim_evals/_camera_study/B/corr_main/` |
| copied into this folder | none yet |
