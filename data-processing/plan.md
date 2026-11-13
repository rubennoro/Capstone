# Plan: make `dataproc calibrate` behave like iPhUMI's gripper calibration

Status: **draft, not implemented.** Opened 2026-11-12. Decisions needed are marked **D1–D4**.

## Why

`dataproc calibrate` rejects good calibration recordings. `episode_000056` (both tags 83%, 6 full open/close
cycles, closed 44 mm, open 138 mm) fails only because its measured opening, 0.093 m, is outside the rig file's
`gripper.max_opening_m: 0.110 ± 0.010`. That check exists only in dataproc. iPhUMI has no expected opening, and
0.110 m was a planned value that was never measured.

The iPhone pipeline measures **0.0865 m** on its own calibration video, using the phone's factory-calibrated lens
data. Deployment builds the robot's width map with the same tag method and checks that it covers **0.085 m**
(`controlSW/franka_deploy`, `fdcal width`, `covers_0p085`). So the gripper "opening" in this whole system means
*tag-centre spacing, fully open minus fully closed*, and is about 0.086–0.093 m, not 0.110 m.

## How iPhUMI calibrates (reference, commit `4be9982`)

Files under `iPhUMI/python_package/iphumi/demonstration_processing/`.

1. **Tags** (`scripts/detect_aruco_iphone.py`): ArUco `DICT_4X4_50` with OpenCV default settings plus sub-pixel
   corners, on each ultrawide frame. `solvePnP` with the phone's own `K`, zero distortion and 16 mm tags
   (`calibration/aruco_config.yaml`) gives `tvec` per tag.
2. **Range fit** (`process_stages/calibrate.py`, `calibrate_gripper_range_iphone`):
   - pick the tag pair (0/1, 2/3, ...) with the best `min(left rate, right rate)`; **fail if below
     `tag_det_threshold` (0.2)**;
   - keep frames with both tags within `artag_z_distance_from_ultrawide` (0.098 m) ± 0.08 m;
   - width = right x − left x; `min_width` / `max_width` = raw min / max; **fail if fewer than 3 such frames**;
   - fit left↔right straight-line maps (one 3-MAD outlier pass);
   - write `<side>_gripper_range.json`. **No other checks.**
3. **Per-recording width** (`utils/gripper_util.py`): both tags → right − left; one tag → predict the other from
   the map; clip to [min, max]; subtract `min_width` (closed = 0, open = max − min). No real-world scaling.

## dataproc today vs iPhUMI

| Step | iPhUMI | dataproc | Match? |
|---|---|---|---|
| Tag dictionary, marker size, PnP | `DICT_4X4_50`, 16 mm, `solvePnP`, zero distortion | same | ✅ |
| Image the tags are found in | phone's ultrawide, phone's `K` | Pi fisheye converted to a virtual iPhone view using **guessed** fisheye lens values (`provenance: nominal`) | ⚠️ can't match exactly (see D3) |
| Detector settings | OpenCV defaults; RGB frame passed as if BGR | 10 px/cell, 0.2 margin, RGB→grey (commit `730800b4`) | ⚠️ deliberate difference (D2) |
| Which tag pair | best-detected pair | fixed `aruco.left_id` / `right_id` from the rig file | ⚠️ same result while only tags 0/1 exist |
| Detection-rate threshold | `min(left, right) ≥ 0.2`, else fail | same | ✅ |
| Depth window | nominal ± 0.08 m (0.098 m) | same (`aruco.nominal_z_m` 0.098 m; ours measure 0.116 m) | ✅ |
| ≥ 3 both-tag frames | fail otherwise | same | ✅ |
| min / max | raw extremes | same | ✅ |
| Left↔right maps | polyfit + one 3-MAD refit | same | ✅ |
| **Expected opening** | **none** | **fail unless 0.110 ± 0.010 m** (`stages/calibration.py` `_check_opening`) | ❌ the difference that blocks us |
| Open/close cycles | not checked | warning below 5 (`MIN_CYCLES`) | ⚠️ warning only, harmless |
| Per-recording width (clip, subtract min) | as above | same (`stages/width.py` `gripper_width_offset`, `calibration_interpolator`, `episode_widths`) | ✅ |
| Rate | ultrawide ~10 Hz → main 60 Hz | ultrawide 30 Hz → 60 Hz grid | ✅ same definition |

## The risk in matching exactly

The opening check is the only thing that caught a bad calibration so far. In `episode_000045` the right tag slid
behind the Pi board whenever the gripper closed, so the fit's "closed" was 62 mm instead of about 43 mm and the
range came out at 74 mm. iPhUMI's rules (and option A below) would **accept** that recording. Every demonstration
would then get widths that never reach 0 when the gripper is closed.

## Decisions

### D1. What replaces the 0.110 m pass/fail check?

| Option | Pass/fail | What it catches | Notes |
|---|---|---|---|
| **A. Exact iPhUMI** | detection rate and ≥ 3 frames only | nothing more | `episode_000045` would pass. Simplest. |
| **B. iPhUMI + warnings** | as A | prints the opening, closed/open widths and cycle count, and warns when they look odd | `episode_000045` passes with a warning you might not read. |
| **C. iPhUMI + a measured expected opening** (recommended) | as A, plus `|opening − expected| ≤ tolerance`, with `expected` measured for this gripper with this method | a truncated or partial calibration (`episode_000045`: 74 mm) | keeps today's code; only the rig value changes. Set `gripper.max_opening_m` from repeated good calibrations (`episode_000056`: 0.093 m) and the ruler check, tolerance 0.010 m. |
| **D. iPhUMI + a closed-width check** | as A, plus fail when the closed width is far above the expected closed spacing | specifically a closing hidden by an obstruction | needs a new rig value (expected closed spacing, ~0.044 m) |

Recommendation: **C**. It is a one-value rig change, keeps the safety net, and matches iPhUMI's widths exactly.
The plan below assumes C; A/B/D change only step 2.

### D2. Keep dataproc's finer detector settings?

They change *which frames* find a tag, not what a found tag measures: on the same frames, widths differ by a median
of 0.02–0.15 mm (max 3.3 mm), and the iPhone video gives 43.3 / 129.8 mm either way. Recommendation: **keep**.
Without them `episode_000056`'s left tag drops to 24% and most demonstrations would fail `tag_coverage`.

### D3. Is our absolute scale close enough to the iPhone's?

Same-looking gripper, same method: iPhone 86.5 mm, Pi rig 93 mm (7% apart). Only matters if Pi-rig and iPhone
data are mixed, or the iPhone-based deployment width map is reused for a Pi-trained policy. **Ruler check** (tag
centre to tag centre, fully closed and fully open) decides which is right:
- ruler ≈ 44 / 138 mm → our scale is right; nothing to do;
- ruler ≈ 43 / 130 mm (like the iPhone) → our virtual view is about 7% off → calibrate the ultrawide lens
  (checkerboard, OpenCV fisheye) and replace the nominal values in `rig_pi_v2.yaml`.

### D4. Update `aruco.nominal_z_m` to our measured depth?

Ours: 0.116 m median against 0.098 m. Inside the ± 0.08 m window either way; changing it only centres the
window. Recommendation: set it to 0.116 when the lens calibration lands (it may change the depth), not now.

## Implementation (assuming D1 = C, D2 = keep)

1. **Rig file** `configs/rigs/rig_pi_v2.yaml`: `gripper.max_opening_m: 0.093`, comment saying it is
   *tag-centre spacing, open − closed, as measured by this method* (not the fingertip gap), where it came from
   (`episode_000056`, plus the ruler check), and that iPhUMI itself has no such check.
2. **Code** `dataproc/stages/calibration.py`: no logic change. Reword the `_check_opening` docstring and the
   failure message to say it is a dataproc-only sanity check against the measured opening.
   `dataproc/calibrate/diagnostics.py`: reorder the hints for `opening_out_of_range` so "a tag was hidden or the
   gripper did not fully close/open" comes first; it is the cause we actually hit.
3. **Version** `CALIBRATION_VERSION` 2 → 3 only if the logic changes (D1 = A, B or D). For C the rig value is
   already part of the inputs hash, so saved results are refitted automatically.
4. **Docs**: `README.md` (calibration row and troubleshooting), `docs/PROCESS.md`, `docs/PI_ADAPTER.md`: replace
   "0.110 m" with the new value and one sentence on what it measures and how iPhUMI differs.
5. **Tests**: the existing opening tests (`tests/calibrate/test_calibrate_task.py`
   `test_opening_out_of_tolerance`, `test_rig_without_gripper_section_skips_the_opening_check`,
   `tests/calibrate/test_calibrate_rig_copy.py`) use their own rig values and should still pass; update any
   hard-coded `0.11` text in `test_calibrate_rig_copy.py`. Add one test: a calibration whose closing is hidden
   (range ≈ 0.67 × expected) fails with `opening_out_of_range`.
6. **Copied rig files**: `~/data/rigs/rig_pi_v2.yaml` and `~/data/test/rigs/rig_pi_v2.yaml` are copies made by
   earlier runs and keep 0.110 until deleted or edited; note this in the README troubleshooting table.

## Validation

| Recording | Expected after the change |
|---|---|
| `episode_000056` | passes: opening ~0.093 m, both tags 83%, 6 cycles |
| `episode_000045` | still fails `opening_out_of_range` (0.074 m; the closing is hidden) |
| `episode_000044` | still fails `gripper_range_fit` (right tag never seen) |
| iPhone calibration video, through `fit_gripper_range` | unchanged: 43.3 / 129.8 mm |
| Test suite | full run, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest data-processing/tests -q` |

Then the first real `dataproc calibrate block_raise <dir>` and a `dataproc run` in the test folder.

## Out of scope here (tracked separately)

- IMU-to-camera rotation (gravity suggests ~90° about the lens axis against the rig file's 0°).
- Ultrawide colour cast (IMX219 lens shading; colour correction or a tuning file).
- Main-camera focus lock; IMU rate 104 → 208/416 Hz; tape over the blue LED near the left tag.
