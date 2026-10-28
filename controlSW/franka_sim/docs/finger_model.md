# Finger model: what is simulated, how accurate it is, and what is still open

Scope: every simulation outcome that depends on the gripper fingers (reach, grasp, lift, camera view of the tags).
Written for the sim evaluator on branch `sim-eval` (finger-kit phase A9). Status words used below:
**measured** (taken from hardware or training data), **derived** (computed from another number or a mesh),
**datasheet** (vendor value), **decision** (a design choice), **placeholder** (a stand-in until CAD or hardware exists),
**unverified** (the author could not confirm it from the sources listed).
Source tags: `A9.md` = `controlSW/docs/sim_plan/phases/A9.md`; `A9c.WORKLOG` step N = `.../phases/A9c.WORKLOG`;
`model.py` = `controlSW/franka_sim/franka_sim/model.py`; `YAML` = both
`controlSW/configs/station/tooling/franka_hand_iphumi_{iphone16pro,rig1}.yaml` (same finger values in both).

## 1. What is modelled and why

- **Arm and hand:** the Franka **Panda** with the Franka Hand (menagerie model, `assets/franka_emika_panda/hand.xml`). The
  FR3 build still exists but only the Panda matters for the lab (A9.md line 5).
- **Fingers as built for iPhUMI:** Franka jaw, then an **adapter**, then the UMI `finger_holder_left/right` (printed, carries
  the ArUco tag), then the TPU **soft finger** (A9.md lines 4-5, "assembly A").
- **Why it matters:** the policy sees the tags through the phone camera and was trained on the width they imply; the
  grasp depends on pad shape, offset and contact; the arm reach (Stage 1) depends on how far the phone sits from the
  jaws, which the finger length sets.
- **Why modes exist:** until the real STLs arrive, the fingers are stand-ins. The three modes (section 2) let each
  stage state which stand-in it ran with.

## 2. The three finger modes

| Mode | What is in the model | TCP (tool centre point) | Tags | Use |
|---|---|---|---|---|
| `stock` | Unmodified menagerie fingers, all stock fingertip pad collisions on. No adapter, no printed parts. `fingertip_pad_collision_1` of each finger is renamed `pad_l`/`pad_r` so scene and campaign code resolves the pads (`model.py` `_stock_fingers`, A9c.WORKLOG step 10). | midpoint of the two stock pad centres, `tcp_at = "stock_pads"` | none | grasp physics and reach comparison only. **Not vision-equivalent** (no tags in the images). |
| `box` (default) | Stock finger collisions off where they would intrude; two rigid box pads `pad_l/pad_r` (20 x 8 x 30 mm) at `extension_m`, tag patches at fitted offsets (`model.py` box branch of build_model step 3). | midpoint of the pad centres, `tcp_at = "pad_centre"` | fitted patches (section 6) | everything today; byte-identical to before A9 (section 5.1) |
| `kit` | STL finger kit (`finger_kit/v1`): TPU finger as one mesh geom `pad_{l,r}`, UMI holder as `finger_holder_{l,r}`, placeholder adapter block `finger_adapter_{l,r}`, tags from the kit's frames (`model.py` `_mount_kit`). | midpoint of the two kit `tip` points, `tcp_at = "tip"` | from CAD frames | the target configuration; **no real kit exists yet** |

Kit variants (`kit_variant`): `full` (soft finger + holder, tags) and `finger_only` (soft fingers only, defined in
the kit's `kit.yaml` under `variants`). Both now work (A9c.WORKLOG step 11 had `finger_only` raising; D1 implemented it afterwards). A tag whose carrier
part is absent is skipped and listed in `tags_skipped`: with `finger_only` both tags are skipped, so the kit tag separation is empty
and the images show no tags (like stock, not vision-equivalent); masses lose the holders (section 8). The kit masses (pad, holder, adapter block) are added to the finger bodies' inertials by the model, so the 2-finger dynamics see them; the added mass is `info["finger_kit"]["finger_body_mass_kg"]` (A9h; before A9h the kit masses were only reported, never simulated: A9.REVIEW finding 1).

### How to switch

In the tooling YAML under `finger_adapter:` (documented in `controlSW/configs/README.md:61-70`), or per run with `--set`:

```yaml
finger_adapter:
  mode: box                     # stock | box | kit   (absent: kit if `kit` is set, else box)
  kit_variant: full             # full | finger_only
  kit: franka_sim/franka_sim/assets/finger_kits/<id>   # relative to controlSW; null = box fingers
  kit_sha256: <64 hex>          # must equal kit_digest(kit dir); required when kit is set
  adapter_length_m: 0.0         # L, kit mode only
  lateral_offset_m: auto        # dy: "auto" or a number in m
  adapter_yaw_deg: 0.0
  adapter_section_m: [0.021, 0.024]
```

```
--set tooling.finger_adapter.mode=stock
--set tooling.finger_adapter.mode=kit --set tooling.finger_adapter.kit=<dir> --set tooling.finger_adapter.kit_sha256=<hex>
--set tooling.finger_adapter.kit_variant=finger_only
```

(the path under `kit:` above is illustrative: no kit directory is shipped yet.) `mode: kit` with `kit: null` is an error;
`kit` set with `mode: box|stock` logs one line `finger_adapter.kit is set but mode=<m>: kit ignored` and records
`info["finger_kit_ignored"]` (`model.py` build_model step 3, A9f.REVIEW). Both YAMLs currently carry `mode: box`, `kit: null`.

## 3. Frame chain, TCP and tags

Per finger, all frames have z toward the fingertip and y outward (away from the other finger) (A9.md lines 14-36):

```
 hand (Franka Hand body, menagerie)
   |  slide joint (menagerie, unchanged; finger body at hand z = 58.4 mm, hand.xml:66)
   v
 finger body (hand/left_finger, hand/right_finger)    right finger = left rotated 180 deg about z
   |  J = (0, y_c, z_base) in the finger body frame, axes = finger body axes
   v
 J   jaw mount = base face of the stock finger_0 mesh      (from the mesh at build time, never hard-coded)
   |  J_T_M = Trans(0, dy, L) . Rz(yaw)     L = adapter_length_m, dy = lateral_offset_m or auto
   |  [placeholder adapter block spans J -> M, section adapter_section_m, PLA 1240 kg/m^3]
   v
 M   mount frame of the kit part on that side
   |  inv(kit_T_M)   (kit_T_M is given in kit/CAD coordinates)
   v
 kit (CAD) frame:  meshes, `tip`, tag centre/normal/up are all stored here
```

Full chain: `body_T_kit = J . J_T_M . inv(kit_T_M)` (`finger_kit.py:1-24`). Only `camera_face` is expressed about the M axes.

- **J from the menagerie mesh:** `jaw_mount()` (`model.py`) takes `finger_0` vertices within 1 mm of its lowest z as the
  footprint. Result: footprint x in [-10.5, 10.5] mm, y in [2.5, 26.3] mm, so y_c = 14.4 mm; base z about 0.13 mm
  (float32 recentring); tip z 53.90 mm (`_finger_tip_z`, max over all finger meshes; 53.86 mm is finger_0's own max); identical in both finger frames (A9.md line 24; A9c.WORKLOG step 0).
- **Side selection:** each part declares `camera_face: +x | -x` in its M frame. The builder computes the camera direction
  in M (`J . Rz(yaw)` applied to the camera direction in the finger frame) and mounts the kit side whose sign matches;
  error if |x component| < 0.5 (yaw near 90 deg) (`model.py` `_mount_kit`, A9c.WORKLOG step 7). Kit sides are named by part
  (`left` = holder_left), not by Franka finger. The camera direction comes from `tcp_x_along_hand`: `+y` puts the camera on
  hand +x, `-y` on hand -x (`model.py` `_tcp_axes`, YAML comment). Stage 1 recommends `-y`.
- **`auto_dy`:** `lateral_offset_m: auto` picks dy so the two contact parts just touch at Franka gap 0 (min inner-face y in
  the finger body frame = 0). Reported in `info["finger_kit"]["dy"]`.
- **TCP per mode** (all use the same axes: TCP x = sgn . hand y, z = hand z, `model.py` `_tcp_axes`; origin on the hand centre line):

  | Mode | TCP z in the hand frame | Flange to TCP |
  |---|---|---|
  | stock | stock pad_1 centre z = 58.4 + 44.5 mm | **0.1029 m** (hand.xml:66, :21; A9c.WORKLOG step 10; Franka documents 0.1034) |
  | box | 58.4 + (tip_z + extension - pl/2), pl = 30 mm, tip_z = 53.90 mm | 0.0973 m + extension (0.1973 m at +10 cm; matches the Stage 1b override in `controlSW/docs/sim_plan/results/stage1b/campaign.txt`) |
  | kit | mean z of the two placed `tip` points at gap 0 | box value + pl/2 only when L and dy are the box-equivalent ones (`box_equivalent_placement`). With L = 0 and `auto` (section 8 run) it is **0.0885 m**: M sits at J, so tip = 58.4 + 0.13 + 30 mm. Reported as `flange_to_tip_m`; residual x/y in `tcp_residual_mm` |

  iPhUMI's own definition is "the pose at the end of the gripper in the middle vertically and horizontally of the
  fingers" (`compute_iphone15pro_calibration.py:8`). The kit mode follows it literally (fingertip, mid-gap). Box mode puts the
  TCP at the pad centre, 15 mm higher; the camera calibration `C_T_TCP` (section 4) is fixed relative to the TCP, so this
  difference moves the camera in the sim by the same amount. Unverified: which of the two matches the real iPhUMI mount to
  the millimetre (see pending item P1).
- **Tags:** in kit mode a tag is the kit's centre, normal (out of the tag face, toward the camera), up, id and size, placed as
  `tag_{l,r}` (white, 1 mm thick, centre lifted 0.5 mm off the surface) and `tag_{l,r}_ink` (black, 0.6 x size, 0.1 mm proud
  of the white) (`model.py` `_mount_kit`, `finger_kit.py:41`). Box mode places them with the fitted offsets (next section).

## 4. Numbers

Stages: **1** offline kinematic replay (feasibility, finger length, phone side); **1b** physics replay of the demos
(`results/stage1b`); **2a** offline policy predictions + IK; **2b** live open loop with emulated Wi-Fi; **3** closed loop in
MuJoCo; **CC** camera-check (sim tags vs detected real tags).

| Name | Value | Unit | Status | Source | Stages |
|---|---|---|---|---|---|
| Stock finger_0 jaw footprint (J) | x [-10.5, 10.5], y [2.5, 26.3], y_c 14.4 (14.41) | mm | derived (menagerie mesh, not Franka drawing) | `model.py` `jaw_mount`; A9c.WORKLOG step 0 | 1, 1b, 3, CC |
| Stock finger tip z (finger frame) | 53.90 (finger_0 alone 53.86) | mm | derived | `model.py` `_finger_tip_z`; A9.md line 24 | all |
| Finger body offset in hand | 58.4 | mm | datasheet-level (menagerie) | `hand.xml:66` | all |
| Finger travel | 0 to 40 each, gap 0 to 80 | mm | datasheet | `hand.xml:11`, YAML `hand.stroke_m` 0.080 | all |
| Stock pad (`fingertip_pad_collision_1`) | 17 x 8 x 17 (x, y, z); centre y 5.5, z 44.5 | mm | datasheet-level (menagerie) | `hand.xml:21` | stock mode |
| Stock pad offset at gap 0 | 3.0 (1.5 per side) | mm | derived | `model.py` `_stock_fingers`; A9c.WORKLOG step 10 | stock mode |
| Stock `flange_to_tip_m` / mesh tip | 0.1114 (pad_1 top: TCP + pad_length/2) / 0.1123 (`flange_to_mesh_tip_m`: finger z 53.90 + 58.4 mm, flange z) | m | derived; 0.9 mm gap is a label convention, not a bug | `finger-kit report` (stock); A9.REVIEW finding 5 | stock mode |
| Stock TCP, flange to TCP | 0.1029 (Franka doc 0.1034) | m | derived; 0.5 mm gap to the doc is **unexplained** | A9c.WORKLOG step 10; test `test_stock_mode_pads_named_no_tags_tcp` | stock mode, 1, 1b |
| Box pad size | 20 x 8 x 30 (x, thickness, length) | mm | placeholder | YAML `pad_size_m`; provenance "iphumi_finger_v1 print; not measured on the Franka Hand" | box mode |
| Box pad offset (inner faces at gap 0) | 8 | mm | placeholder | `model.py` `PAD_OFFSET_M` | box mode, 1b, 3 |
| Box `extension_m` | 0.0 in YAML; Stage 1 recommends 0.10 | m | placeholder / decision pending | YAML; `STATUS.md` Stage 1 table | 1, 1b, 2, 3 |
| Box tag size | 16 | mm | decision | `model.py` `TAG_SIZE_M` | CC, 2, 3 |
| Box tag y extra (per side outward) | 31.2 | mm | fitted (camera-check on 5 demos / 200 frames; constant to ~0.1 mm) | `model.py` `TAG_Y_EXTRA_M` | CC, 2, 3 |
| Box tag z from pad centre | -108 | mm | fitted (explains the (0, +23, -104) mm UW-frame offset); real tags ~98 mm from the UW camera | `model.py` `TAG_Z_FROM_PAD_M` | CC, 2, 3 |
| Box closed tag separation (gap 0) | 86.4 (= 2 x (b/2 + tag/2 + y_extra) = 2 x (4 + 8 + 31.2)) | mm | derived from the three rows above | `model.py` box branch tag_c formula | CC |
| Kit `adapter_length_m` L | 0.0 | m | placeholder; Stage 1 re-run picks it | YAML provenance | 1, 1b |
| Kit `lateral_offset_m` dy | `auto` | m | placeholder | YAML | 1b, 3, CC |
| Kit `adapter_yaw_deg` | 0.0 | deg | placeholder | YAML | all |
| Kit `adapter_section_m` | 21 x 24 (default = stock footprint) | mm | placeholder | YAML | 1 (collision), 1b |
| Adapter density | 1240 (PLA) | kg/m^3 | datasheet-level | `model.py` `ADAPTER_DENSITY_KG_M3` | 1b, 3 (inertia) |
| Kit part density / mass | kit `materials` (TPU 1210, PLA 1240) or a measured `mass_kg` | kg/m^3 | per kit, template values | A9.md lines 50-51 | 1b, 3 |
| Contact (pads and eraser) solref | [2 x timestep, 1] = [0.002, 1] (timestep 0.001 s, `configs/run/sim/physics.yaml:3`; the generated block confirms) | s, - | decision (A5: 15 g eraser, 2 x 30 N squeeze) | `model.py` `grip_contact` | 1b, 3 |
| Contact solimp | [0.95, 0.99, 0.001, 0.5, 2] | - | decision | `model.py` `grip_contact` | 1b, 3 |
| Pad friction (box) | 1.0 (sliding; torsional/rolling MuJoCo defaults 0.005/0.0001) | - | placeholder | YAML `friction`; `model.py` `_box` | 1b, 3 |
| Pad friction (kit) | kit `contact.friction`, template [1.0, 0.005, 0.0001] | - | placeholder | A9.md line 55 | 1b, 3 |
| Eraser | 60 x 20 x 12 mm, 15 g, friction 0.9 | - | placeholder | `configs/station/scene/block_raise_table.yaml:5` | 1b, 3 |
| Solver | elliptic cone, impratio 10, timestep 0.001 s (the sim-iphone/sim-rig built models; `configs/run/sim/physics.yaml`), implicitfast | - | decision | `model.py` build_model step 7 (defaults when the run config omits them) | 1b, 3 |
| Multi-point CCD (`mjENBL_MULTICCD`) | on in kit mode only | - | decision (A9c fix) | `model.py` build_model step 7; A9c.WORKLOG step 17 | 1b, 3 |
| Grasp height rule | pad/finger bottom 5 mm above the table | mm | decision | `scene.py:32,61-76`; A9c.WORKLOG step 13 | 1b, 3 |
| Hand force / max force | 30 / 70 | N | decision / datasheet | YAML `hand.force_n`, `force_max_n` | 1b, 3 |
| Width map | train width = 1.0 x gap + 0.0431 | m | derived (b = tag separation at closed in training data) | YAML `width_map`; "A5 review" | 2a, 2b, 3 |
| Training width range | 0.0447 to 0.0844 | m | measured (firstreal-single) | YAML `train_min_m`, `train_max_m` | 2, 3 |
| `C_T_TCP` | 15 deg camera tilt about x; translation (0, 0.0644, 0.2114) m | - | measured for iPhone 15 Pro, copied to 16 Pro (assumes same mount) | `gripper_calibration.yaml:14` (iPhone17,1 entry, its comment at :12); rig1 entry is a copy, **placeholder** | all camera stages |
| `artag_z_distance_from_ultrawide` | 0.098079 | m | measured (outlier filter, "roughly correct") | `gripper_calibration.yaml:15` | CC |
| `tcp_x_along_hand` | `+y` in the YAML (Stage 1 recommends `-y`) | - | placeholder / decision pending | YAML comment; `STATUS.md` | 1, 1b, all |

Note on the closed tag separation: the 86.4 mm is arithmetic from the formula in the box branch (finger origins sit on the centre
line at gap 0, inner pad faces at +-b/2). Camera-check measured the real tags at "finger gap + 87 mm" before the fit, and the
fit (31.2 mm per side) was chosen to close the -62.3 mm difference (`model.py` `TAG_Y_EXTRA_M` comment). The generated block (section 8, box) prints 0.0864 m: confirmed. The kit run prints 0.0784 m, 8 mm less, because `auto` dy closes the 8 mm placeholder pad offset (kit pad_offset 0).

Two different "widths" must not be mixed up: the **training width** (0.0431 m at closed, from the tag separation in
the data) and the **pad offset** (8 mm box placeholder, a geometric gap between pad faces). the `PAD_OFFSET_M` comment in `model.py` records
the earlier mistake of tying them together.

## 5. Validation evidence so far

### 5.1 Box mode byte identity (A9c.WORKLOG steps 1, 3, 6)

- Before editing `model.py`, the SHA-256 of `spec.to_xml()` and of 27 compiled-model arrays plus geom names was recorded at
  HEAD `f230a96` for sim-iphone / sim-rig x panda / fr3 (mujoco 3.3.0, run twice, stable). The hashes are hard-coded in
  `test_box_mode_byte_identical_to_head` and `test_box_mode_grasp_height_unchanged` in `tests/test_model_kit.py` (a `git show HEAD` comparison would compare the file to itself once committed).
- After the edit all four digests were identical, and a direct `to_xml()` diff against a HEAD copy (with and without
  scene) was identical (152,486 bytes). Re-checked after the stock/kit addendum (step 3 and 12).
- Caveat: the digests are valid for mujoco 3.3.0 only; the test skips otherwise (`test_box_mode_byte_identical_to_head`).

### 5.2 Box-equivalent kit matches box mode (A9c.WORKLOG step 5; `test_model_kit.py`, tolerance `TOL = 1e-9`)

`make_box_kit` with `box_equivalent_placement` L and dy reproduces the box pads, tags and contacts for Panda, FR3 and a rotated `kit_T_M`.

| Quantity | Achieved | Test tolerance | Why that size |
|---|---|---|---|
| Pad vertices, float64 placement | 1e-17 m | 1e-9 | ASCII STL keeps float64 |
| Pad vertices, compiled (float32) | <= 2.2e-9 m | 5e-9 | MuJoCo stores float32 vertices |
| Tag poses | 6e-17 m | 1e-9 | |
| Pad-eraser `mj_geomDistance` | 6.3e-10 m | 1e-9 | box-box and mesh colliders return different contact *points*, so equivalence is tested on distance, not point lists (step 4) |
| TCP offset to box mode | pl/2 = 15 mm, to 2e-16 m | 1e-9 | by design: tip vs pad centre |
| TCP residual x/y | < 1e-9 mm | 1e-9 | |
| float32 STL (binary) | ~1e-9 m rounding at 30 mm | | `make_box_kit` writes ASCII by default for this reason (A9a.WORKLOG step 2) |

Also: an A9a reviewer probe on independent code reproduced the pad corners to 3.2e-9 m (binary STL), tip and tags to < 1e-15
(A9a.REVIEW check 2).

### 5.3 `auto_dy` closure

- Reviewer probe: contact part min y after placement = 7e-18 / 0 m for yaw 0 and 25 deg, both sides, and 7e-18 for J rotated
  180 deg about z (A9a.REVIEW check 2).
- Model level: auto dy for the box-equivalent kit equals `-y_c - b/2` to 1e-12; the two pad inner faces touch (offset 0
  to 1e-12) and `mj_geomDistance` at gap 0 matches the commanded gap to 1e-8 (`test_lateral_offset_auto_pads_touch_at_gap_zero`).

### 5.4 Stock mode pick

Scripted pick with the stock pads (contact parameters from `grip_contact()`): 2 of 2 seeds lift and hold; TCP 0.1029 m
(`test_stock_mode_pads_named_no_tags_tcp`); grasp registers in `Scene.grasped` (A9c.WORKLOG steps 10, 13). Two seeds is a smoke test.

### 5.5 The 20 N reflex in kit mode, diagnosis and fix (A9c.WORKLOG steps 14-17)

- **Symptom:** the scripted kit pick tripped the 20 N Cartesian reflex (|O_F_ext z| 20.1 N) on 3/3 seeds at 5 mm tip height; it
  passed at 7 and 10 mm.
- **Not** descent overshoot: the descent is identical in box and kit mode (min pad bottom 3.52 mm above the table, no table contact).
- **Cause:** MuJoCo's mesh-versus-box collision returns **one** contact point per pad, placed at an edge of the overlap with the
  eraser. The two pads' points sat at opposite edges (pad_l at eraser z 12.47 mm, pad_r at 5.76 mm, about 24.8 N each, depth
  -0.39 / -0.42 mm), a couple that pivots the eraser; |O_F_ext| reached (12.7, 17.9, 21.0) N. Box pads use the box-box collider
  (4 points per pad) and peak at 5.1 N.
- **Ruled out:** raising the grasp height only shrinks the overlap (5 seeds, grasp-phase |Fz| peak: 5.5 mm 0/5 lifts, 6 mm 1/5,
  6.5-7 mm 4/5, 7.5-8 mm 5/5 at 15-19.5 N, 9-10 mm 5/5 with 14-20 N outliers, 11 mm 5/5 at ~5 N with only 1 mm overlap).
- **Fix:** `mjENBL_MULTICCD` in kit mode only (box digests unchanged). Result: tip 5 mm, 5/5 seeds lift, peak |Fz| 5.0-6.2 N
  (box 5.1 N), slip 0.02 mm, 4 contact points per pad at eraser z about 5.2 and 11.9 mm.
- The test that guards this (`test_model_kit.py`, contacts test) checks the deepest point equals `mj_geomDistance`.

### 5.6 Other evidence

- Suites at A9c close: 414 passed, 3 failed, 11 skipped; the 3 failures (`test_cam_uw_is_main_times_extrinsic` x2,
  `test_render::test_rig_b0179_mode`) reproduce on a clean HEAD copy, unrelated to the fingers (A9c.WORKLOG step 12).
- Station validator: rejects bad `mode`, `kit_variant`, non-finite numbers, kit without sha; accepts `{}` (A9f.REVIEW, PASS).
- **Not validated yet:** anything with a real kit (section 7).

## 6. Known limits and their expected effect

| Limit | Where it bites | Expected effect |
|---|---|---|
| **Rigid TPU.** The soft finger is a convex hull with stiff contact (solref 0.002 s), friction a placeholder. | 1b, 3 (grasp) | A real soft finger conforms, spreads the load and tolerates more width error; the sim grasp is stricter on pad alignment and has no compliance in squeeze. Lift success is not calibrated against real TPU. Magnitude: unknown (**unverified**). |
| **Single-point mesh contact** (MuJoCo mesh-vs-box). Fixed by MULTICCD in kit mode only. | 1b, 3 (kit) | After the fix peak force 5.0-6.2 N vs 5.1 N for box (section 5.5). Box mode is not changed so its contacts are as before. MULTICCD costs run time (unmeasured). |
| **Placeholder adapter block** (length, section, yaw, lateral offset). | 1 (phone reach), 1b, 3 | Finger length sets how far the phone sits from the arm. Stage 1 gave +10 cm for box fingers (42/45 demos on the Panda). The real adapter's length changes `flange_to_tip_m` and so the whole 1/1b result; **Stage 1 must be re-run** (P2). |
| **J from the menagerie mesh, not Franka's finger-interface drawing.** | all kit results | A footprint error of a few tenths of a mm shifts the pad position by the same amount. Confirm on hardware day. Magnitude **unverified**. |
| **Friction placeholder.** | 1b, 3 | Eraser holds in sim with friction 1.0 and 30 N; the real margin is unknown. |
| **Stock mode has no tags.** | 2a, 2b, 3 (anything that reads images) | The policy cannot run in stock mode with a faithful image. Use stock only for reach and grasp-physics comparisons. |
| **UW extrinsic `C_main_T_C_uw` is `fitted`.** | CC | Provenance status `fitted` (A9 W10B finding 5) marks a value fitted to camera-check data rather than measured; the iPhone `C_main_T_C_uw` uses it. camera-check pairs sim and detected tags by ArUco id (`tags.tag_pairing = "by_id"` when the model records `info["finger_kit"]["tag_ids"]`; box mode falls back to image u-order, `"by_image_order"`), so a swapped id mapping gives large pixel errors instead of passing. |
| **Box tags are fitted, not geometric.** | CC, 2, 3 | Offsets (31.2 mm, -108 mm) were fitted to 5 demos and are confounded with the camera model: the UW camera extrinsic was fitted at the same time, with y 19 mm larger than the SIFT/lens prior (`YAML` provenance of `C_main_T_C_uw`). Pixel error with the fit: holdout p50 6.6 / 9.0 px and 9.2 / 4.3 px (main / UW) on two holdout sets of 5 demos. |
| **TCP differs between modes** (pad centre vs tip vs stock pad). | 1, 1b | A mode change moves the TCP (kit = box + 15 mm for the equivalent kit; stock 0.1029 vs box 0.0973 m at extension 0). Results across modes are comparable only with the TCP and the grasp-height rule (pad bottom at 5 mm) in mind. |
| **Stage 1 reach, box vs box-equivalent kit** (Panda, 45 demos, `-y`, A9.REVIEW section D). | 1 | box `extension_m` +0.10: 43/45 feasible, worst tooling clearance +23.8 mm; the box-equivalent kit (L = 0.123767, auto dy), same tip 0.2123 m: 43/45, +31.9 mm. Same feasible count and placement; the 8.1 mm clearance difference is the TCP convention: the kit TCP sits at the tip, 15 mm beyond the box pad-centre TCP, and `C_T_TCP` is fixed to the TCP, so the phone sits 15 mm further out along the flange z axis, away from links 5/6. The "+10 cm" answer stands as the box result. |
| **Stage 1b today** (box, +10 cm, `-y`): 3/43 demos within the tracking criterion at 0.5 speed, 36/45 reached the lift (ground-truth success) | 1b | The campaign is an arm-dynamics and speed-clamp result more than a finger result (most failures are `safety_clamp`); do not read finger accuracy from it. `controlSW/docs/sim_plan/results/stage1b/campaign.txt`. The 36/45 figure also appears in the plan file. |
| **Training-camera link `C_T_TCP` copied** for the 16 Pro and the Jetson rig. | CC, 2, 3 | The camera position relative to the TCP is assumed equal to the iPhone 15 Pro mount (`gripper_calibration.yaml:12`); the rig value is a copy to be measured. |

## 7. Pending evidence (needs the real STLs)

Acceptance thresholds are from `A9.md` and the plan file `/home/ruben/.claude/plans/pasted-content-id-b776-1-the-sprightly-cloud.md`
section "Verification". None of these has been run.

| # | Check | Acceptance threshold | State |
|---|---|---|---|
| P0 | Placement: `finger-kit check` and `show` on the real kit | tags face the camera, ids 0 and 1 on the expected image sides; fingers touch at gap 0 (`auto_dy`); no kit geom intersects the Hand body at gap 0 or 0.08 | pending (kit not delivered; CLI by D2) |
| P1 | camera-check with the real kit, no tag fitting (`python -m franka_sim camera-check --zarr $ZARR --stage1 S1/stage1.json --out DIR`) | tag pixel error p50 <= today's fitted result (holdout main 6.6 px / UW 9.0 px); real vs sim tag separation within a few mm; TCP-to-tag z offset about 0.098 m (`artag_z_distance_from_ultrawide`) | pending. This is the acceptance test that the CAD placement matches the training geometry. |
| P2 | Stage 1 re-run on the Panda over `adapter_length_m` x camera side | new recommended adapter length; `flange_to_tip_m` compared to the old +10 cm box result (0.1973 m flange to TCP) | pending (D3 prepares the sweep) |
| P3 | Stage 1b campaign with the real kit (about 1 h) | grasp gap for the eraser about 0.020 m (training width 0.063); lift count compared to 36/45 | pending |
| P4 | Gates | franka_sim and franka_deploy suites, `scripts/regression/check_protected.sh`, `test_iphone_deploy_unchanged.py` | per-piece reviewers passed their parts; final integration review pending (`A9.REVIEW.md`) |
| P5 | Confirm J against Franka's finger-interface drawing | hardware day | pending |
| P6 | Measure TPU friction, a real TCP (`C_T_TCP` for the rig), adapter dimensions | replace placeholders in the YAML provenance | pending |

Until P1 passes, nothing in the kit mode is claimed to reproduce the training images; until the real kit replaces the box fingers
in the tooling YAML (gated on the final reviewer), `mode: box` stays the default.

## 8. Generated numbers

The tables below are produced by the tooling, not typed. Regenerate with
`python -m franka_sim finger-kit report --profile sim-iphone [--set ...] --out DIR` and paste `finger_report.md` between the markers.

<!-- BEGIN finger-kit report: python -m franka_sim finger-kit report --profile sim-iphone --out DIR -->
Run on 2026-10-27, profile sim-iphone, panda, with the working tree of that day. The kit is the throwaway box-equivalent kit from `make_box_kit` in `$S/d4/kit` (id `box_equivalent`, pinned with `finger-kit pin`), not a real finger. Long paths in the overrides are shortened. Rows `mount_mass_kg`, stock `friction` and `flange_to_mesh_tip_m` and the stock `flange_to_tip_m` label were updated by hand after A9h2 changed the report (regenerate to confirm).

#### (a) box mode (default)

Finger report: robot `panda`, tooling `franka_hand_iphumi_iphone16pro`, mode `box`, station fingerprint `0b3ef298b94b9c56`

| number | value | unit | status | source |
|---|---|---|---|---|
| mode | box |  | decision | A9 addendum: stock \| box \| kit; box until the iPhUMI kit passes camera-check |
| extension_m | 0 | m | placeholder | iphumi_finger_v1 print; not measured on the Franka Hand |
| pad_length_m | 0.03 | m | placeholder | iphumi_finger_v1 print; not measured on the Franka Hand |
| pad_offset_m | 0.008 | m | placeholder | model.PAD_OFFSET_M placeholder (CAD pending) |
| tcp_residual_x_mm | 0 | mm | derived (model) | box TCP on the centre line by construction |
| tcp_residual_y_mm | 0 | mm | derived (model) | box TCP on the centre line by construction |
| friction | [1] |  | placeholder | iphumi_finger_v1 print; not measured on the Franka Hand |
| tcp_at | pad_centre |  | derived (model) | model |
| solref | [0.002, 1] |  | decision | run/sim grip_solref (model.grip_contact) |
| solimp | [0.95, 0.99, 0.001, 0.5, 2] |  | decision | run/sim grip_solimp (model.grip_contact) |
| mount_mass_kg | 0.299 | kg | derived (model) | phone mount only (housing + cradle), sum of the tooling geom masses; the finger kit is not in it |
| tooling_mass_kg | 0.299 | kg | derived (model) | deprecated alias of mount_mass_kg (kept for one release) |
| flange_to_tcp_m | 0.0973 | m | derived (model) | F_T_TCP z (flange frame, along the approach axis) |
| flange_to_tip_m | 0.1123 | m | derived (model) | distal pad edge on the centre line |
| tag_separation_closed_m | 0.0864 | m | derived (model) | tag box centres at gap 0 |
| F_T_TCP_xyz_mm | [0, 0, 97.3] | mm | derived (model) | TCP from the finger geometry |
| F_T_TCP_rpy_deg | [0, 0, 45] | deg | derived (model) | TCP from the finger geometry |
| F_T_TCP_source | derived |  | derived (model) | model |
| C_T_TCP_xyz_mm | [0, 64.4, 211.4] | mm | measured | gripper_calibration.yaml iPhone17,1 via fdcal/handeye.py:24 |
| C_T_TCP_rpy_deg | [15.009, 0, 0] | deg | measured | gripper_calibration.yaml iPhone17,1 via fdcal/handeye.py:24 |
| F_T_cam_main_xyz_mm | [82.696, -82.696, -90.211] | mm | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_main_rpy_deg | [-15.009, 0, 45] | deg | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_uw_xyz_mm | [57.819, -81.356, -95.455] | mm | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_uw_rpy_deg | [-15.009, 0, 45] | deg | derived (model) | F_T_TCP @ inv(C_T_TCP) |

#### (b) stock mode, `--set tooling.finger_adapter.mode=stock`

Finger report: robot `panda`, tooling `franka_hand_iphumi_iphone16pro`, mode `stock`, station fingerprint `29a5923c1db4543c`, overrides ['tooling.finger_adapter.mode=stock']

| number | value | unit | status | source |
|---|---|---|---|---|
| mode | stock |  | decision | A9 addendum: stock \| box \| kit; box until the iPhUMI kit passes camera-check |
| pad_length_m | 0.017 | m | datasheet | menagerie fingertip_pad_collision_1 z extent |
| pad_offset_m | 0.003 | m | datasheet | stock fingertip_pad_collision_1 inner faces at gap 0 |
| tcp_residual_x_mm | 0 | mm | derived (model) | stock pad midpoint off the hand centre line |
| tcp_residual_y_mm | 0 | mm | derived (model) | stock pad midpoint off the hand centre line |
| friction | [1, 0.005, 0.0001] |  | datasheet | compiled model geom_friction of pad_l (menagerie stock pad, unchanged) |
| tcp_at | stock_pads |  | derived (model) | model |
| solref | [0.002, 1] |  | decision | run/sim grip_solref (model.grip_contact) |
| solimp | [0.95, 0.99, 0.001, 0.5, 2] |  | decision | run/sim grip_solimp (model.grip_contact) |
| mount_mass_kg | 0.299 | kg | derived (model) | phone mount only (housing + cradle), sum of the tooling geom masses; the finger kit is not in it |
| tooling_mass_kg | 0.299 | kg | derived (model) | deprecated alias of mount_mass_kg (kept for one release) |
| flange_to_tcp_m | 0.1029 | m | derived (model) | F_T_TCP z (flange frame, along the approach axis) |
| flange_to_tip_m | 0.1114 | m | derived (model) | pad top (distal pad edge on the centre line; stock: finger_1 pad top) |
| flange_to_mesh_tip_m | 0.1123 | m | derived (model) | stock finger mesh tip (_finger_tip_z) along the flange z axis; the pad top is flange_to_tip_m |
| F_T_TCP_xyz_mm | [0, 0, 102.9] | mm | derived (model) | TCP from the finger geometry |
| F_T_TCP_rpy_deg | [0, 0, 45] | deg | derived (model) | TCP from the finger geometry |
| F_T_TCP_source | derived |  | derived (model) | model |
| C_T_TCP_xyz_mm | [0, 64.4, 211.4] | mm | measured | gripper_calibration.yaml iPhone17,1 via fdcal/handeye.py:24 |
| C_T_TCP_rpy_deg | [15.009, 0, 0] | deg | measured | gripper_calibration.yaml iPhone17,1 via fdcal/handeye.py:24 |
| F_T_cam_main_xyz_mm | [82.696, -82.696, -84.611] | mm | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_main_rpy_deg | [-15.009, 0, 45] | deg | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_uw_xyz_mm | [57.819, -81.356, -89.855] | mm | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_uw_rpy_deg | [-15.009, 0, 45] | deg | derived (model) | F_T_TCP @ inv(C_T_TCP) |

#### (c) kit mode, box-equivalent kit (`make_box_kit(variants=True)`, L = 0, dy auto), `kit_variant=full`

Finger report: robot `panda`, tooling `franka_hand_iphumi_iphone16pro`, mode `kit`, station fingerprint `d5bcfb38ec9e4821`, overrides ['tooling.finger_adapter.mode=kit', 'tooling.finger_adapter.kit=<scratch>/d4/kit', 'tooling.finger_adapter.kit_sha256=7f87e26ff4bc91979ea4f5fd3be42f5ad04d5c5a250212af237bad569dd12257', 'tooling.finger_adapter.lateral_offset_m=auto', 'tooling.finger_adapter.kit_variant=full']

| number | value | unit | status | source |
|---|---|---|---|---|
| mode | kit |  | decision | A9 addendum: stock \| box \| kit; box until the iPhUMI kit passes camera-check |
| kit_id | box_equivalent |  | decision | A9: box fingers until the iPhUMI kit passes camera-check |
| kit_sha256 | 7f87e26ff4bc91979ea4f5fd3be42f5ad04d5c5a250212af237bad569dd12257 |  | decision | A9: box fingers until the iPhUMI kit passes camera-check |
| kit_variant | full |  | decision | A9 addendum: kit variant, full = soft finger + UMI holder (tags); finger_only = soft fingers only |
| kit_path | <scratch>/d4/kit |  | decision | tooling.finger_adapter.kit |
| adapter_length_m | 0 | m | placeholder | A9: parametric adapter block, CAD pending; Stage 1 picks the length |
| dy_l_m | -0.018414 | m | derived (model) | auto_dy: contact part inner face at finger-body y = 0 at gap 0 |
| side_l | left |  | derived (model) | kit side whose camera_face matches the camera direction |
| dy_r_m | -0.018414 | m | derived (model) | auto_dy: contact part inner face at finger-body y = 0 at gap 0 |
| side_r | right |  | derived (model) | kit side whose camera_face matches the camera direction |
| yaw_deg | 0 | deg | placeholder | A9: parametric adapter block, CAD pending; Stage 1 picks the length |
| tag_separation_closed_m | 0.0784 | m | derived (kit) | kit box_equivalent geometry: placeholder: make_box_kit: model.py box pads |
| pad_offset_m | 0 | m | derived (kit) | kit box_equivalent geometry: placeholder: make_box_kit: model.py box pads (contact parts' inner faces at gap 0) |
| tags_skipped | [] |  | derived (model) | variant: tags whose carrier part is absent |
| mass_pad_l_kg | 0.0048 | kg | derived (kit) | kit box_equivalent part masses: pad=density, holder_left=density, holder_right=density |
| mass_finger_holder_l_kg | 0.008 | kg | derived (kit) | kit box_equivalent part masses: pad=density, holder_left=density, holder_right=density |
| mass_pad_r_kg | 0.0048 | kg | derived (kit) | kit box_equivalent part masses: pad=density, holder_left=density, holder_right=density |
| mass_finger_holder_r_kg | 0.008 | kg | derived (kit) | kit box_equivalent part masses: pad=density, holder_left=density, holder_right=density |
| tcp_residual_x_mm | 0 | mm | derived (model) | tip midpoint off the hand centre line |
| tcp_residual_y_mm | 0 | mm | derived (model) | tip midpoint off the hand centre line |
| pad_length_m | 0.03 | m | derived (kit) | kit box_equivalent geometry: placeholder: make_box_kit: model.py box pads (contact part z extent) |
| friction | [1, 0.005, 0.0001] |  | derived (kit) | kit box_equivalent contact.friction |
| tcp_at | tip |  | derived (model) | model |
| multiccd | on (kit mode: multi-point contacts for the mesh pads) |  | derived (model) | model: kit mode only |
| solref | [0.002, 1] |  | decision | run/sim grip_solref (model.grip_contact) |
| solimp | [0.95, 0.99, 0.001, 0.5, 2] |  | decision | run/sim grip_solimp (model.grip_contact) |
| mount_mass_kg | 0.299 | kg | derived (model) | phone mount only (housing + cradle), sum of the tooling geom masses; the finger kit is not in it |
| tooling_mass_kg | 0.299 | kg | derived (model) | deprecated alias of mount_mass_kg (kept for one release) |
| flange_to_tcp_m | 0.088533 | m | derived (model) | F_T_TCP z (flange frame, along the approach axis) |
| flange_to_tip_m | 0.088533 | m | derived (model) | kit tip midpoint |
| F_T_TCP_xyz_mm | [0, 0, 88.533] | mm | derived (model) | TCP from the finger geometry |
| F_T_TCP_rpy_deg | [0, 0, 45] | deg | derived (model) | TCP from the finger geometry |
| F_T_TCP_source | derived |  | derived (model) | model |
| C_T_TCP_xyz_mm | [0, 64.4, 211.4] | mm | measured | gripper_calibration.yaml iPhone17,1 via fdcal/handeye.py:24 |
| C_T_TCP_rpy_deg | [15.009, 0, 0] | deg | measured | gripper_calibration.yaml iPhone17,1 via fdcal/handeye.py:24 |
| F_T_cam_main_xyz_mm | [82.696, -82.696, -98.978] | mm | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_main_rpy_deg | [-15.009, 0, 45] | deg | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_uw_xyz_mm | [57.819, -81.356, -104.222] | mm | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_uw_rpy_deg | [-15.009, 0, 45] | deg | derived (model) | F_T_TCP @ inv(C_T_TCP) |

#### (d) same kit, `kit_variant=finger_only`

Finger report: robot `panda`, tooling `franka_hand_iphumi_iphone16pro`, mode `kit`, station fingerprint `36f70ad95ecf5572`, overrides ['tooling.finger_adapter.mode=kit', 'tooling.finger_adapter.kit=<scratch>/d4/kit', 'tooling.finger_adapter.kit_sha256=7f87e26ff4bc91979ea4f5fd3be42f5ad04d5c5a250212af237bad569dd12257', 'tooling.finger_adapter.lateral_offset_m=auto', 'tooling.finger_adapter.kit_variant=finger_only']

| number | value | unit | status | source |
|---|---|---|---|---|
| mode | kit |  | decision | A9 addendum: stock \| box \| kit; box until the iPhUMI kit passes camera-check |
| kit_id | box_equivalent |  | decision | A9: box fingers until the iPhUMI kit passes camera-check |
| kit_sha256 | 7f87e26ff4bc91979ea4f5fd3be42f5ad04d5c5a250212af237bad569dd12257 |  | decision | A9: box fingers until the iPhUMI kit passes camera-check |
| kit_variant | finger_only |  | decision | A9 addendum: kit variant, full = soft finger + UMI holder (tags); finger_only = soft fingers only |
| kit_path | <scratch>/d4/kit |  | decision | tooling.finger_adapter.kit |
| adapter_length_m | 0 | m | placeholder | A9: parametric adapter block, CAD pending; Stage 1 picks the length |
| dy_l_m | -0.018414 | m | derived (model) | auto_dy: contact part inner face at finger-body y = 0 at gap 0 |
| side_l | left |  | derived (model) | kit side whose camera_face matches the camera direction |
| dy_r_m | -0.018414 | m | derived (model) | auto_dy: contact part inner face at finger-body y = 0 at gap 0 |
| side_r | right |  | derived (model) | kit side whose camera_face matches the camera direction |
| yaw_deg | 0 | deg | placeholder | A9: parametric adapter block, CAD pending; Stage 1 picks the length |
| tag_separation_closed_m |  | m | derived (kit) | kit box_equivalent geometry: placeholder: make_box_kit: model.py box pads |
| pad_offset_m | 0 | m | derived (kit) | kit box_equivalent geometry: placeholder: make_box_kit: model.py box pads (contact parts' inner faces at gap 0) |
| tags_skipped | [{'id': 0, 'on': 'holder_left', 'finger': 'l', 'side': 'left'}, {'id': 1, 'on': 'holder_right', 'finger': 'r', 'side': 'right'}] |  | derived (model) | variant: tags whose carrier part is absent |
| mass_pad_l_kg | 0.0048 | kg | derived (kit) | kit box_equivalent part masses: pad=density, holder_left=density, holder_right=density |
| mass_pad_r_kg | 0.0048 | kg | derived (kit) | kit box_equivalent part masses: pad=density, holder_left=density, holder_right=density |
| tcp_residual_x_mm | 0 | mm | derived (model) | tip midpoint off the hand centre line |
| tcp_residual_y_mm | 0 | mm | derived (model) | tip midpoint off the hand centre line |
| pad_length_m | 0.03 | m | derived (kit) | kit box_equivalent geometry: placeholder: make_box_kit: model.py box pads (contact part z extent) |
| friction | [1, 0.005, 0.0001] |  | derived (kit) | kit box_equivalent contact.friction |
| tcp_at | tip |  | derived (model) | model |
| multiccd | on (kit mode: multi-point contacts for the mesh pads) |  | derived (model) | model: kit mode only |
| solref | [0.002, 1] |  | decision | run/sim grip_solref (model.grip_contact) |
| solimp | [0.95, 0.99, 0.001, 0.5, 2] |  | decision | run/sim grip_solimp (model.grip_contact) |
| mount_mass_kg | 0.299 | kg | derived (model) | phone mount only (housing + cradle), sum of the tooling geom masses; the finger kit is not in it |
| tooling_mass_kg | 0.299 | kg | derived (model) | deprecated alias of mount_mass_kg (kept for one release) |
| flange_to_tcp_m | 0.088533 | m | derived (model) | F_T_TCP z (flange frame, along the approach axis) |
| flange_to_tip_m | 0.088533 | m | derived (model) | kit tip midpoint |
| F_T_TCP_xyz_mm | [0, 0, 88.533] | mm | derived (model) | TCP from the finger geometry |
| F_T_TCP_rpy_deg | [0, 0, 45] | deg | derived (model) | TCP from the finger geometry |
| F_T_TCP_source | derived |  | derived (model) | model |
| C_T_TCP_xyz_mm | [0, 64.4, 211.4] | mm | measured | gripper_calibration.yaml iPhone17,1 via fdcal/handeye.py:24 |
| C_T_TCP_rpy_deg | [15.009, 0, 0] | deg | measured | gripper_calibration.yaml iPhone17,1 via fdcal/handeye.py:24 |
| F_T_cam_main_xyz_mm | [82.696, -82.696, -98.978] | mm | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_main_rpy_deg | [-15.009, 0, 45] | deg | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_uw_xyz_mm | [57.819, -81.356, -104.222] | mm | derived (model) | F_T_TCP @ inv(C_T_TCP) |
| F_T_cam_uw_rpy_deg | [-15.009, 0, 45] | deg | derived (model) | F_T_TCP @ inv(C_T_TCP) |

#### Reconciliation with the hand-derived numbers

| Quantity | Doc (hand) | Generated | Result |
|---|---|---|---|
| Box flange to TCP (extension 0) | 0.0973 m | 0.0973 m | match |
| Box closed tag separation | 86.4 mm | 86.4 mm | match |
| Box pad offset / length | 8 / 30 mm | 8 / 30 mm | match |
| Stock flange to TCP | 0.1029 m | 0.1029 m | match |
| Stock pad offset / length | 3 / 17 mm | 3 / 17 mm | match |
| Contact solref | 0.004 (typed, wrong) | 0.002 | doc fixed: timestep is 0.001 s |
| Kit TCP, box-equivalent kit, L = 0 | box + 15 mm (typed) | 0.0885 m | doc fixed: true only with box-equivalent L; at L = 0 the kit is shorter |
| Kit tag separation | not stated | 78.4 mm | 8 mm below box: auto dy closes the pad offset |

Resolved open items (A9.REVIEW findings 3 and 4): `tooling_mass_kg` (0.299 kg in every mode) is the phone mount only,
housing 0.199 + cradle back plate 0.10 kg; the report now calls it `mount_mass_kg` (the old key stays as an alias for one
release). It does not include the finger kit, whose masses are in the `mass_<geom>_kg` rows. The stock `friction` is the compiled
`geom_friction` of `pad_l`, [1, 0.005, 0.0001].
<!-- END -->
