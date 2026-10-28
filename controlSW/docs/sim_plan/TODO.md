# Sim evaluator: open work (2026-10-28)

Everything still open on branch `sim-eval`, in order. Status and evidence: [STATUS.md](STATUS.md). Finger model and its
limits: [../../franka_sim/docs/finger_model.md](../../franka_sim/docs/finger_model.md). Latest review:
[phases/A9.REVIEW2.md](phases/A9.REVIEW2.md) (verdict FIX). Work stopped here on 2026-10-28 by the user's decision: the review
findings below are recorded, not fixed.

## 1. Finger kit (A9): before the real kit can be the default

The real kit `iphumi_umi_soft_v1` is in `franka_sim/franka_sim/assets/finger_kits/` and works opt-in. Box mode stays the
default in both tooling YAMLs until items 1.1 and 1.2 pass.

- [ ] **1.1 Tag positions (blocker, needs a measurement).** The kit's tags sit about 23 mm too far inward per side and about 8 mm
  too close to the camera compared with the training video. Camera check: 56.6 / 47.5 px against 6.6 / 9.0 px for the fitted box tags.
  - Closed tag separation: 40.6 mm in the kit, about 88 mm in the data.
  - On the real gripper used for `firstreal`, measure:
    - the tag-centre-to-gripping-face distance;
    - the closed tag separation;
    - the tag height above the holder.
  - Also check that this gripper is the Onshape UMI assembly at all.
  - Then fix `tags.centre` in `kit.yaml`. If it is fitted rather than measured, mark the provenance `fitted`. Re-pin `kit_sha256` and rerun P1:
    `python -m franka_sim camera-check --profile sim-iphone --robot panda --set <kit sets> --zarr $ZARR --stage1 S1/stage1.json --demos 10,20,30,39,42 --out DIR`.
  - Accept at p50 <= 6.6 / 9.0 px.
- [ ] **1.2 Stage 1b regression with the kit.** 9/45 lifts against box 36/45, and 40/45 runs end in a reflex.
  - The scripted pick works (5/5), so look at the replayed approach:
    - log the contact pairs at reflex time in the campaign;
    - try a tip height of 7-10 mm (`scene.py` `PAD_TABLE_CLEARANCE_M`, 5 mm today);
    - try a tooling-clearance rule for the 26 mm wide tips.
  - Rerun 1b on an idle machine (`--jobs 3` beside other heavy jobs gives render-fps preflight faults).
- [ ] **1.3 camera-check id fallback** (`camera_check.py:510-515`). In kit mode, return rc 2 unless the kit's tag ids are exactly
  {0, 1}. Today other ids fall back to image order silently, and a kit with one tag raises KeyError. Box mode keeps image order.
- [ ] **1.4 Payload report** (`model.py:926-944`). Add a kit-inclusive `difference_kg`, or warn when the configured and modelled
  payload differ by more than 0.1 kg. When the kit becomes the default, raise `payload.mass_kg` from 1.15 to about 1.27 kg (L 0.008),
  then replace it with the weighed parts.
- [ ] **1.5 Switch the tooling YAMLs** (both `configs/station/tooling/franka_hand_iphumi_*.yaml`), only after 1.1 and 1.2.
  - Set `mode: kit`, the kit path and `kit_sha256`, `adapter_length_m: 0.008` and `tcp_x_along_hand: -y`, each with provenance.
  - Rerun `eval --profile sim-iphone --stage 1 --out DIR` without `--set`. Expect 43/45 and clearance about +39 mm.
- [ ] **1.6 Adapter CAD.** Stage 1 recommends L = 0.008 m (the kit minimum; the holder clears the hand from about 0.0074 m).
  Design the adapter at that length, export it as `adapter_*.stl` into the kit (replacing the placeholder block), and re-pin.
- [ ] **1.7 Hardware day (P5/P6):**
  - check the J frame (stock finger base face from the menagerie mesh) against Franka's finger-interface drawing;
  - weigh the printed parts;
  - measure the TPU friction;
  - measure the real TCP and the rig `C_T_TCP`.

## 2. Decisions waiting on the user

- [ ] **D9 tooling:** box fingers +10 cm or the real kit at L 0.008 m. Phone on -y either way: −y is better at every length in both
  designs. See STATUS.md "Stage 1 result".
- [ ] **Bridge cone (60°) and workspace box (placeholder).**
  - They cause the iPhone 1b tracking FAILs and refuse every sim-rig 1b start (the demos tilt 68-72°).
  - Options: widen the cone, measure the box, or start the campaign inside the cone.
- [ ] **1b / 2a gates with infeasible demos (18, 32):** exclude the Stage-1-infeasible demos, or accept FAIL-by-finding.
- [ ] Whether to push `sim-eval` (`origin/sim-eval` is at `3bfadbf`).

## 3. Runs still to do

- [ ] After the cone/box decision: sim-rig `--stage 1b`, the iPhone 1b, and a longer bp_run001 Stage 3
  (`--poses 3 --episodes-per-pose 3`).
- [ ] PLAN §11 step 7, a viewer run: needs a display. Copy deploy.yaml with
  `profiles.sim-iphone.bridge.override.backend.mujoco.viewer: true`.
- [ ] Find what causes the Stage 1 worst-margin demo (1.1-1.5°).
- [ ] Is the Stage 3 tracking 8 mm / observation motion 5.4 mm the policy or the sim? Compare with a real-robot run.

## 4. Housekeeping

- [ ] Remove the w10b worktree: remove its `data` symlink first, then `git worktree remove .claude/worktrees/w10b`.
  Everything from it is on `sim-eval`.
- [ ] Scratch env paths (`/tmp/claude-1000/.../299e5dab.../scratchpad/env_*`) are hardcoded in README.md,
  `station/host/pc_laptop.yaml` and the sim profiles in `deploy.yaml`. If the scratchpad is purged, recreate the envs and
  update those paths.
- [ ] Copy anything wanted from `$S/a9r2/` (full real-kit run outputs, probes, renders) before the scratchpad goes. The small
  summaries are already in `results/a9_real_kit/`.
