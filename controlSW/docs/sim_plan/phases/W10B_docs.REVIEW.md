# W10B docs-vs-code review (system reviewer)

Verdict: FIX (documentation only; no code defect found). CLI names, profile names, stage names (incl. 1b), ports,
tcp_x_along_hand semantics and the derived-sim-TCP text in INTERFACES I-TCP / tooling YAMLs all match the code.
`check-config` on sim-iphone, sim-rig (OK, `sim_tcp` row 0.000 mm, `tooling.F_T_TCP derived (sim)` note) and
real-iphone, real-rig (FAIL on critical placeholders / Jetson paths, as designed) behave as documented.
STATUS.md is stale (2026-10-18); orchestrator rewrites it (not itemised).

## PLAN.md section 11
1. L337 step 3 `eval --profile sim-iphone --stage 1` lacks the required `--out DIR`; "for Panda and FR3" is not
   possible with one eval call (no --robot flag). Use: `python -m franka_sim eval --profile sim-iphone --stage 1 --out $S/s1`
   and for both arms `python -m franka_sim stage1 --zarr $ZARR --ids robot=panda,tooling=franka_hand_iphumi_iphone16pro,sensor=iphone16pro_wifi,scene=block_raise_table --robots panda,fr3 --out $S/s1_both`.
   `camera-check` also needs `--zarr Z --stage1 S1.json --out DIR`.
2. L338 step 4 same: add `--out DIR` to both eval commands.
3. L340 step 6 `policy-contract` requires `--endpoint`: `python -m franka_deploy policy-contract --endpoint tcp://localhost:18765 [--expect run/policy/<id>.yaml]`.
4. L341 step 7 "`viewer: true`" has no home: it is `backend.mujoco.viewer` (backend.py:19), set via a copy of deploy.yaml
   `profiles.sim-iphone.bridge.override.backend.mujoco.viewer: true` (needs a display; `--set` is station-only).
5. L336 step 2: `--config deploy.yaml` is `--config controlSW/franka_deploy/configs/deploy.yaml` (cwd controlSW/franka_deploy: `configs/deploy.yaml`).

## hardware_day.md
6. L121 (4.3) and L128 (5.1) say "Copy F_T_TCP into franka_cartesian.yaml" / `--F_T_TCP configs/franka_cartesian.yaml`; on the station profiles
   (real-*) the controller TCP comes from `tooling.F_T_TCP` (station.py:338) and L287 says so. Replace 4.3 tail with
   "Copy `F_T_TCP` into the tooling YAML (`tooling.F_T_TCP`, status `measured`); servo path: `franka_cartesian.yaml`", and
   5.1 with "`--F_T_TCP <yaml holding the measured F_T_TCP>`".
7. L276-295 (section 15 table + re-run): a measured `tooling.F_T_TCP` is NOT a sim input that wins: the launcher keeps it
   only if status is not placeholder/hw, and then preflight `sim_tcp` FAILS if it differs from the model TCP by > 1 mm / 0.1 deg
   (launcher.py:203-290). Add under the table: "For the sim re-run, enter the measured camera mount as `tooling.F_T_C_main` (status
   `measured`; the sim then uses F_T_C_main @ C_T_TCP). Set `tooling.F_T_TCP` (status `measured`) only if it agrees with it
   (and with extension_m / tcp_x_along_hand) within 1 mm / 0.1 deg, or the sim run refuses to start. Leave `tooling.F_T_TCP`
   null in sim-only work: the launcher derives it." Change table row `4.3, 5.1 | tooling.F_T_TCP` to `4.3 | tooling.F_T_C_main, tooling.F_T_TCP; 5.1 | tooling.F_T_TCP`.
8. L241 (12.12) `check-config` row `hand_endpoint` does not exist; the rows are `controller_gripper`, `franka_hand_driver`, `hand_server`.
9. L72 (0.8) and L32 `--profile jetson`: no such profile (deploy.yaml base = Jetson; unknown names are silently ignored). Replace with
   `check-config --config deploy.yaml` (no --profile); table L32 profile cell `jetson` -> `(base, no --profile)`. Same text in
   franka_deploy/README.md:33 and TODO.md D1.

## franka_deploy/README.md
10. L82-91 "see profile `station-mock`" omits the sim profiles. Add: "Profiles `sim-iphone-fake|sim-rig-fake|sim-iphone|sim-rig` are driven by
    `python -m franka_sim eval --profile P --stage 1|1b|2a|2b|3|all --out DIR`; on `sim-*` MuJoCo runs the launcher fills `tooling.F_T_TCP`
    from the sim model when it is null (provenance `derived (sim)`, INTERFACES I-TCP); real-* never do."

## controlSW/franka_sim/README.md
11. L3-4 "Phase A0 only vendors ... model building comes later" is stale. Replace with: "Package: MuJoCo Panda/FR3 + Franka Hand model, kinematics, SimCore/MuJoCoBackend, scene, renderer, Stage 1/1b campaign, and the eval CLI (`python -m franka_sim --help`: check-assets, model, stage1, campaign, camera-check, camera-gap, characterize, eval)."

## controlSW/configs/README.md
12. L5 "Nothing consumes it yet in B1; B2 switches the launcher ..." is stale: the launcher, run_task, fdcal and the sim all consume it. Replace with "Consumed by the launcher, `run_task.py`, `fdcal` and `franka_sim`."
13. Layout block: `sweeps/   block_raise_default` -> `block_raise_default, block_raise_smoke`.

## ROADMAP.md
14. L59 Stage 1b "not built (A8)" -> "built (`python -m franka_sim campaign` / `eval --stage 1b`); 45-demo result: tracking 3/43 within gate (a finding, not a pass)".
15. L60 Stage 3 "built on the stub simulator; MuJoCo closed loop depends on the scene/render phases" -> "built; runs on the stub (`--sim-stub`) and on MuJoCo via BridgeSim with rendered images (W9c)".

## TODO.md
16. D6 (L~66) lists `controlSW/franka_deploy/` and ROADMAP/PURCHASING/TODO as uncommitted; they are committed on `sim-eval` except the W10B working-tree changes. Reword to "uncommitted: W10B (sim TCP derivation, judge/eval fixes), newer sensor_rig/, scripts/regression/".
17. D1 `check-config --profile jetson` same as item 9.

## sim_plan/README.md
18. Status table ends at W7 and the footer says PAUSED 2026-10-18; add rows W8-W10 (A6/A7/A8, B8/B9/B10, W10A/W10B) or point to STATUS.md. Baseline "RE 145 passed" is old.

No issues found in: INTERFACES.md ports/CLI/I-CAMPAIGN/I-TCP, tooling YAML tcp_x_along_hand comments (+y = hand +x side, -y = hand -x side, matches model.py:391), F_T_TCP null/derived comments, adding_a_policy.md commands, PURCHASING.md 2b.
