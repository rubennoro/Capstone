# HANDOFF: SLAM gripper mask for `dataproc` (work in progress)

> **Temporary file.** Next agent: read this before touching `data-processing/`. When the work below is finished and
> verified, **delete this file and its pointer bullet in `AGENTS.md` ("Current state")** in the same commit as the
> final work. The user asked for this on 2026-11-16.

## Why this exists
Every Pi recording failed `dataproc run` at the slam stage with
`orbslam3: never initialised (0 keyframes ...)`.

**Cause.** The ultrawide camera (IMX219, 1280x960) is bolted to the gripper, so these always sit in the same place in
its image:
- the fingers and their tags;
- the held object;
- the housing, board, fan and LED.

ORB-SLAM3 matches them with zero parallax, so its monocular start-up keeps frame 0 as its reference for the whole clip
and never succeeds. This was measured on 2026-11-15/16: 250–450 matches against frame 0 for 7 s, nearly all on the
gripper.

**Fix.** A rig setting, `cameras.ultrawide.slam_mask`: polygons blacked out **only** in the PNG frames that
`write_euroc` writes for ORB-SLAM3. Unaffected:
- training images (`dataproc/export/images.py`) and tag widths (`dataproc/stages/width.py`), which read the MP4s;
- the raw videos;
- deployment.

**The mask is necessary but not enough on its own.** Recordings also need texture across the table and about 2 s of
brisk side-to-side *sliding* at the start. The current recordings (session 1: 61–113 on a bare gray table; session 2:
117–121 with a few objects behind the eraser) mostly still fail later, at `validate` (too little of the clip has poses).

## Update 2026-11-15 (laptop session): mask parked, not applied by `dataproc run`
- **The mask is out of the active rig.** The `slam_mask` block moved from `configs/rigs/rig_pi_v3.yaml` to the draft
  `data-processing/configs/slam_masks/rig_pi_v3.yaml`. All the mask code stays; a rig without a mask processes
  exactly as before. To try it, paste the draft into a scratch data root's `rigs/rig_pi_v3.yaml`.
- **Jetson:** `~/data/rigs/rig_pi_v3.yaml` still has the mask, and dataproc does not warn when only the copy has
  one. Delete that copy (it is re-copied from the repo without the mask) before the next real run.
- **Why parked: the mask edge.** Blacking out leaves a hard black border whose corners never move. In wall-facing
  frames about half of the remaining ORB features sit within 20 px of it (episode 117). Idea to test: fill the
  masked area with a heavily blurred image instead of black, so the border has no edge.
- **Step 3 is written and tested** (138 targeted tests). The new error message, run on the 10 real logs: session 1
  (61–65) never initialised; session 2 (117–121) initialised 2–6 times, then lost tracking before IMU init.
- **Per-episode measurement** (usable ORB features = outside the mask, > 20 px from its border; IMU over the
  first 3 s):

  | | features/frame (median) | frames < 150 | gyro rms | accel std |
  |---|---|---|---|---|
  | session 1 (61–65) | 83–121 | ~60% | 20–34 °/s | 0.29–0.39 m/s² |
  | session 2 (117–121) | 636–1413 | 3–9% | 5–14 °/s | 0.08–0.17 m/s² |

  Session 1 lacks texture (re-record). Session 2 has texture but a gentle start (IMU init).
- **Laptop:** everything up to ORB-SLAM3 runs (scratch root: `calibrate` from 115, `run` 117–121 stopped at
  "mono_inertial_euroc not found"). ORB-SLAM3 is not built here: all apt deps are present, so it needs no sudo,
  about 2 GB and about 10 min. Not decided yet.
- `--retry-failed` cannot be reproduced from a copy of `tasks/` alone: the root-level `pi_sources.csv` is needed.

## Status (pushed as a WIP commit on `rnoro/data-work`)
| Step | What | Status |
|---|---|---|
| 1 | Trace the mask | **Done.** In `data-processing/configs/rigs/rig_pi_v3.yaml`; overlays in `data-processing/docs/media/slam_mask_overlay_session{1,2}.jpg`. 29% of the image; fingers at every opening, held-eraser notch, housing/board/fan/LED, ≥ 12 px margin; the gripper sits in the same place in both sessions (≤ 2 px). |
| 2 | Rig schema, validation, re-run hashing, stale-copy guard | **Done**, 156 targeted tests pass (full suite not run). |
| 3 | Apply the mask to SLAM's frames, clearer error message | **In progress when this file was committed** (partly written, untested). Finish it if no later commit did. |
| 4 | `scripts/preview_slam_mask.py` + docs | Not started. |
| 5 | Verification | Partly done, see "Results" below. |

**Step 2 files:**
- `dataproc/config.py`: `SlamMask`, `CameraConfig.slam_mask`, `_slam_mask()` validation.
- `dataproc/stages/base.py`: `rig_sections` gets a separate `cameras.ultrawide.slam_mask` entry, only when a mask is set.
- `dataproc/stages/geometry.py`: `SlamStage.inputs` includes it, only when set.
- `dataproc/pi_intake.py`: `check_rig_copy` raises when the data-root copy lacks the repo's mask, warns when they differ;
  `ensure_rig` logs the warnings.
- `dataproc/sim/episode.py`: `fast_rig` scales the polygons.
- Tests: `test_config.py`, `test_slam_mask_hash.py`, `test_process_rerun.py`, `run/test_run_range.py`,
  `calibrate/test_calibrate_rig_copy.py`, `test_sim_episode.py`.

**Step 2 re-run rule (proven by tests):** adding or changing a mask re-runs exactly `slam, pose, validate, images,
export`. Check, grid, width and the calibration fit don't re-run, and editing only the mask's `source` note re-runs
nothing. For recordings that had failed at slam, `--retry-failed` with the mask runs slam plus the stages it had
blocked.

**Step 3 scope** (see the "Design" list):
- new `dataproc/slam/mask.py` with `keep_mask()` and `slam_keep_mask()`;
- `dataproc/slam/orbslam3.py`: `euroc_image()` used by `write_euroc`, and `"slam_mask": {n_polygons,
  masked_fraction}` in the prep diagnostics only when a mask is set;
- the clearer error: when the log shows ORB-SLAM3 initialised and then reset before IMU init, say
  `initialised N time(s) but lost tracking before the IMU finished initialising ...` instead of "never initialised";
- `tests/test_slam_mask.py` and additions to `tests/test_geometry_slam.py`.

**Step 4 scope:**
- `scripts/preview_slam_mask.py RECORDING [--rig ID|PATH] [--out FILE]`: draws the rig's mask over a few ultrawide
  frames. Pi dirs have `cam0` = ultrawide; task `raw/` dirs have `cam1` = ultrawide; pick the MP4 by the rig's
  ultrawide resolution.
- Docs: `docs/ORBSLAM3.md` (replace the "Finger mask" follow-up with a section), `docs/GEOMETRY.md` (EuRoC prep),
  `docs/PROCESS.md` (YAML snippet next to the gripper one, plus a re-run table row), `README.md` (SLAM paragraph and
  the "never initialised" troubleshooting row).

**Step 5 still to do:**
- Confirm on 1–2 episodes that dataproc's own path (`Orbslam3Backend` with the masked rig) gives the same SLAM outcome
  as the hand-masked runs.
- Full test suite, once.
- Re-run test in a scratch `DATAPROC_ROOT` copy of `tasks/eraser_pick_2`.
- Then hand the user the real commands (below).

## Design essentials
- **YAML**, under `cameras.ultrawide`: `slam_mask: {source: "<note>", polygons: [[[x, y], ...], ...]}`.
  - Coordinates are pixel **edges** at native resolution: `0 <= x <= W`, `0 <= y <= H`.
  - Each polygon has ≥ 3 points and area ≥ 1 px². Only `cameras.ultrawide` may have one.
  - Parsed to float tuples; `source` is a note and isn't hashed.
- **Drawing:** 255 keep / 0 masked.
  - `pts = round(((p * scale) - 0.5) * 256)`, then `cv2.fillPoly(..., shift=8)`.
  - **Fill one polygon per call.** A single call with several polygons leaves overlaps *unfilled*; this happened for
    real.
- **Applying:** gray → mask at the frame's own size if a resize follows → `INTER_AREA` resize → mask at the output
  size. Without a mask, the output must be byte-identical to before.
- **Jetson data root:** `~/data/rigs/rig_pi_v3.yaml` was updated to the masked rig on 2026-11-16, with the user's OK.
  - The pre-mask copy is backed up in that Jetson session's scratchpad (`/tmp/claude-1000/...`, temporary).
  - If the two copies' masks differ, dataproc warns. If the copy lacks the mask, calibrate/run stop with a
    paste-ready snippet.

## Results so far (Jetson, real ORB-SLAM3, this mask)
- **Episode 117:**
  - camera-only: starts at 2.8 s (4.8 s with the earlier rough mask) and tracks to the end;
  - camera+IMU: 3 resets, then a map of 20 keyframes survives, with poses for 10.6–13.9 s of the 13.9 s clip (24%);
  - so slam should now produce a trajectory, and validate should reject it because it needs about 80% of the clip.
- **Multi-episode runs:** 61, 62, 63, 70, 88, 96, 104, 112, 117–121, each masked, plus unmasked for 70, 88, 96, 104
  and 112. These were running when this file was committed. Add the result table here in the follow-up commit.

## Where things can run
- **Laptop:** code and unit tests (fake ORB-SLAM3 runner). `pip install -e data-processing[test]`, then:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest data-processing/tests/test_slam_mask.py
  data-processing/tests/test_geometry_slam.py data-processing/tests/test_config.py
  data-processing/tests/test_slam_mask_hash.py -q`. The full suite takes about 7 minutes.
- **Jetson only:**
  - ORB-SLAM3 (`~/ORB_SLAM3`), the recordings (`/data/incoming`), and the task folders `~/data/tasks/eraser_pick_1`
    (calibration 60; 61–63 tried) and `eraser_pick_2` (calibration 115; 117–121 tried). All failed at slam before the
    mask. Episode 120's source is gone from `/data/incoming`, but its copy is in `failed/`.
  - The two folders have different task labels (`eraser_pick_1` vs `eraser_pick_2`). Rename one before packaging
    both together for prompting policies; it doesn't matter for plain `umi_policy_dunet`.
- **User commands once Steps 3–5 are done** (the user runs these):
  `source ~/envs/env_dataproc/bin/activate`, then
  `dataproc run eraser_pick_2 episode_000117 episode_000121 --retry-failed` and
  `dataproc run eraser_pick_1 episode_000061 episode_000063 --retry-failed`.

## Working agreements with the user
- The user runs `dataproc` pipeline commands on real data. Give the exact command and what it will print; don't run
  it.
- Explain each command in plain words before running it. Ask before changing state (data root, commits, pushes).
- Work step by step with a short plain-language summary after each step. The user is learning the system.
- Don't run the ~7-minute full suite unless it's needed. The user asked to skip it on 2026-11-16.
- Delegating steps to subagents is fine; check their output before reporting it.

## When finished
Delete this file and the `AGENTS.md` pointer bullet in the commit that completes the work.
