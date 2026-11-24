# 00_integration_checks: checks G1-G7 the sim must pass before any benchmark

**Status:** done: G1-G5 pass; G6 pass for the conventions (open-loop replay fails on fast demo segments); G7 fail
(policies not functional in the sim) · **Date:** 2026-11-20 · **Who:** agents E2-E7 and P3 (runs), documented by
agent · **Plan ref:** sim_deploy/PLAN.md §6 (E7 results column and the "P3" table), §11 (progress, W0);
DESIGN.md "E7: integration", "P3/P4"

## Question

Does the MuJoCo sim stand in for the robot side faithfully enough that a policy's sim result means something: a
faithful copy of the deployment code (G1), the rig geometry (G2), camera images (G3), the Pi's stream (G4), the PC's
behaviour (G5), end-to-end conventions (G6), and learned policies that run (G7)?

## Why it matters for the real robot

PLAN: "No benchmark runs until every check in §6 passes." Several findings here are real-robot findings: the gate's
`v_max` 0.25 m/s cuts the demos' fast motions (G6), the copy's box floor (z 0.05) refuses every pinch on the table
(G5/G6, now -0.005 in the sim and upstream), and the handheld IMX219 sits ~9 mm closer to the fingers than
`rig_pi_v3` says (G3).

## Overview

| | Check | Pass when (PLAN §6) | E7 result (deploy / trainstart home, old table z 0.05) | P3 result (start pose, table O z 0) | Status now (PLAN §11) | W0 re-sync (upstream `f632576`): re-run? |
|---|---|---|---|---|---|---|
| G1 | Copy is faithful | diff shows only the listed changes; `policy_deploy check --mock` passes against the sim | **pass**: 18 files (17 modified + `replay_policy.py`); check 21 PASS / 0 WARN | not re-run (P1 ported the start-pose snapshot: 61 files differ now) | pass | **yes**: W0 changes the copy (W0 lists "G1 smoke") |
| G2 | Geometry | rig poses match the STEP; 11-17 numbers ±1 mm; rig parts where the STEP puts them | **pass**: TCP 235.65 mm, masses and cameras identical to the wave-2 model; tags 44.29 mm apart at gap 0 | not re-run; current model `166a0649` has the same masses, cameras and TCP | pass | **yes** (partly): the model is rebuilt and the gripper gets the printed fingers' 0.04 m max + `gripper_map`; recheck TCP, finger travel, tag distance |
| G3 | Cameras vs reality | handheld pose: finger/tag pixels within a few px of a real training frame | **pass with a residual**: 1.01 / 3.30 px (mean / max) with the camera moved (-4.0, +2.5, +8.8) mm; with training's `T_cam_tcp` tag means 4.97 / 12.03 px, tags 13-17 % small | not re-run (independent of the table) | pass | only if W0 changes the width mapping (`gripper_map`) or the rig file; otherwise no |
| G4 | Stream and clocks | 30 pkt/s, no stale camera, clock converges, pair rule, probe images like training | **pass**: 900 packets in 30 s, 0 gaps / bad, pair gap -16.3 ms, clock synced at once, drift 20.1-22.5 ppm, refused 0/0/0 | not re-run as such (every P3 eval reaches `[idle]` against the sim Pi) | pass | **yes** (smoke): upstream frames/manager changes; one `pi_collect` + frames stats run |
| G5 | PC behaviour | hold still; tracking; out-of-box -> fault; no C3 1 s -> stopped; homing | **pass**: drift <= 0.040 mm / 0.090 deg (deploy), <= 0.069 mm / 0.122 deg (trainstart) over 20 s; fault latched + run.sh resume; stopped 0.77 s after holding; tracking not run as such | **pass**: 0.066 mm / 0.121 deg, start pose to 0.000114 rad; `c` and `p` refusals | pass | **yes**: W0 lists "G5 smoke" (hold_still), start pose from upstream `station.yaml` |
| G6 | Conventions end to end | replay: TCP follows the recorded path; grasp closes on the eraser | **fail**: ep 2, 178 waypoints shortened, TCP 11.4 cm above the grasp, rejected waypoint x 0.238 < 0.250 | **pass for the conventions**: ep 19 success (8.8 mm median, 14.5 mm at the grasp), ep 0 success (16.8 / 26.6 mm); ep 28 / 16 80-107 mm off at v_max 0.25 and 0.5 (29 of 30 demos > 0.5 m/s before the grasp) | conventions pass; open loop limited by v_max | **yes**: W0's "chain runs a replay"; `gripper_map` changes what width the replay commands, so the grasp must be rechecked |
| G7 | Policies run | each checkpoint (plain and prompting) 3 episodes, no fault, inference in budget, tap saved | **fail**: run_006, 2 faults (box z top) after 7.28 / 5.98 s; 177-182 ms p50, margin min 10.7 ms; no tap | **fail**: run_006, 3 faults (box y, cone 62 / 63 deg) after 18.45 / 16.89 / 11.44 s; rises 40-44 cm; offline: the sim's IMX219 image flips the first chunk (+5.6 -> -1.3 cm) | **not functional** (PLAN §11) | **yes**, after W0-W2 (W3): upstream manager / pose_converter / diffusion_policy and gripper changes |

## Parts

| File | Checks | Raw evidence |
|---|---|---|
| [G1_G4_copy_geometry_cameras_stream.md](G1_G4_copy_geometry_cameras_stream.md) | G1, G2, G3, G4 | session scratchpad only (no eval folders); copies in `data/g1_g4/` |
| [G5_pc_behaviour.md](G5_pc_behaviour.md) | G5 (hold, fault, watchdog, homing, refusals) | `sim_evals/_baseline/hold_still/*`, `_replay/.../pick1-v1_ep003/*`; C2 recordings in the scratchpad; `data/g5/` |
| [G6_replay.md](G6_replay.md) | G6 (replays, path error, demo speeds) | `sim_evals/_replay/PICK_TEST1/pick1-v1_ep0{00,02,16,19,28}/*`; `data/g6/` |
| [G7_policies.md](G7_policies.md) | G7 (run_006 smoke and from the start pose; offline first chunk) | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_{0149_smoke,0957_p3_g7_start}`; `data/g7/` |
| [data/evals.csv](data/evals.csv) | one row per eval folder used by G5-G7: stage, home, table z, box in force, v_max, overlay, model sha, outcomes, markers, superseded | eval.yaml, episodes.jsonl |
| [../01_start_pose/6_camera_swap.md](../01_start_pose/6_camera_swap.md) | the offline camera-swap follow-up of G7 | `sim_evals/_start_pose_sweep/p3/` |

`$S` in the parts = `/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad`, the
session scratchpad (temporary, not in sim_evals).

## Setup

Common to all parts (details per run in each part and in `data/evals.csv`):

| Item | E7 (2026-11-20 01:20-01:53) | P3 (2026-11-20 09:39-10:01) |
|---|---|---|
| Repo | `rnoro/main-sim` @ `3633fcb` + uncommitted `sim_deploy/`, `deployment/` (eval.yaml `git.sim_deploy` `3633fcb3…-dirty`) | same |
| Model sha (eval.yaml) | `701815be08cd…` | `166a0649571a…` (and `7746a439…` in the superseded 0939) |
| Table | top O z 0.05 on a pedestal | top O z 0 (the base on the table) |
| Homes | `deploy` (TCP O (0.307, 0.000, 0.354)), `trainstart` (TCP O (0.438, -0.164, 0.211)) | `start` (P3 stand-in, TCP O (0.439, -0.164, 0.162), 16.2 cm, 49.5 deg, heading left); `deploy` for the `c` refusal |
| Box in force | min x 0.25, z 0.05; max z 0.70 m (the copy's `config.yaml`; from the gate's texts) | min [0.20, -0.40, -0.005], max [0.80, 0.40, 0.70] m |
| v_max / cone | 0.25 m/s / 60 + 2 deg | same; sim-only 0.5 m/s in three G6 replays (`stations/vmax05_g6.yaml`) |

Since P4 the sim default start pose is `leftx55_t48_h16` (`stations/start_pose.yaml`), not the P3 stand-in; to
reproduce a P3 run exactly, pass the P3 stand-in with `--start-pose`.

## Method

Each part follows sim-experiments/TEMPLATE.md. Every number was re-read from the raw files (eval folders, sim_evals
logs, the scratchpad's C2 recordings and reports); recomputations are scripts in `data/` (read-only, they print to
stdout). Where PLAN/DESIGN differ, the raw value is used and listed under "Docs vs raw data".

## Commands to reproduce

See each part. All closed-loop checks go through `sim_deploy/run.sh` (one chain at a time; ports 5555, 5610, 5620,
5630, 5640, 5650, 5651 must be free; G7 needs the GPU and an idle laptop on AC power):

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
sim_deploy/run.sh --policy hold_still --episodes 1 --seed 0 --tag g5_hold --timeout 20
sim_deploy/run.sh --policy "replay:/home/ruben/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip#19" --episodes 1 --tag g6 --timeout 25
sim_deploy/run.sh --policy /home/ruben/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 --episodes 3 --seed 0 --tag g7 \
  --timeout 60 --precision bf16 --diffusion-steps 8 --exec-h 12 --tap
```

## Results

The overview table above; per-run tables in the parts.

### Fix log (the fixes that affected the checks)

| # | Stage | Where | What was wrong -> fix | Affects | Evidence in the raw data |
|---|---|---|---|---|---|
| 1 | E7 | D8 (adapter, stations, `sim_pi/handheld_check`) | width offset 0.0443 -> 0; kit tags +1.855 mm outward; G3 variants E7 / E7+camera / D8_old | G2, G3, G6, G7 (the width the fingers take) | current `model.json` `finger_kit.tag_separation_closed_m` 0.044287 (wave 2: 0.040577); `$S/e7/g3/g3_report.json` variants |
| 2 | E7 | `run.sh` | the port check was inverted (refused when every port was free) | every run.sh run | run.sh ran from 01:30 (`$S/e7/run_hold.log`) |
| 3 | E7 | `eval/pd_session.py` | a message after the status line's erase code was taken as status: the session folder was never found, steps/tap never collected -> split at `\x1b[K` | G5-G7 (steps, inference numbers, tap) | `0130_g5_hold`: "policy_deploy ready (session None)", 0 steps; from 0134 on steps are collected (552 sent); `pd_session.py` line 87 |
| 4 | E7 | `eval/driver.py` | `[idle]` arrives before the ready line: wait up to 5 s for the session folder | same as 3 | as 3 |
| 5 | E7 | `deployment/.../policy/replay_policy.py` (+ STRUCTURE.md 5.5) | the cursor advanced exec_h steps per infer while the manager replans every ~33 ms: the replay ran ~12x fast -> cursor = time since the first infer | G6, G5's fault replay | the file is dated 01:40, before the first replay eval (01:41) |
| 6 | E7 | `eval/driver.py`, `pd_session.py` | a FAILED start (PC in fault) was not recognised: waited 180 s -> parse `FAILED`, exit early, exit code 3 | G5 fault recovery, G7 | `0141_g5_fault_trainstart` ep 1 `not_run`, problem "no [idle] within 180 s" (before); 0143_43 exits at once (after) |
| 7 | E7 | `run.sh`, `eval/driver.py --resume`, `report.append_index` | after a latched fault the eval stopped -> run.sh restarts the sim PC + Pi and resumes the folder (<= 5 times); one index row per eval folder | G5, G6, G7 | `_logs/2026-11-20_014332/eval.log` "resuming ...: 1 of 2 episodes done"; `index.csv` one row per folder |
| 8 | E7 | `run.sh` (`set +e` around the driver) | with `pipefail` + `set -e` the script ended at the driver's exit code before reading it | G5 recovery | `_logs/2026-11-20_014251/eval.log` (a first `0143_g5_fault_recovery`) stops after "the sim PC latched a fault" with no restart (inferred: before fix 8) |
| 9 | E7 | `eval/driver.py` (UI runner) | SIGINT dropped the episode in progress -> `interrupted` | none of the checks | - |
| 10 | E7 | `eval/driver.py`, `scene/placement.py`, `scene/__main__.py`, `run.sh` | `--home deploy|trainstart` | G4 images, G5 trainstart holds and fault | eval.yaml `home.variant` trainstart in 0136, 0141, 0143_43 |
| 11 | E7 | `model/build.py`, `sim_rig.yaml`, `control_sim.yaml` | appearance + fin-ray contact pairs; finger kp 20000 | G2, G3, G6 grasp, G7 images | `model.json` `appearance`, `finger_contact` (absent in wave 2) |
| 12 | E7 | `control_sim.yaml` | `safety.workspace_min_m [0.20, -0.40, 0.045]`; eraser region / default pose centred | G6, G7 | **not in force in the E7 evals** (their gate texts say min x 0.250, z 0.050); first visible in P3 as "min 0.200 m" |
| P1 | P3/P4 | `stations/sim_rig.yaml`, `model/build.py`, `control_sim*.yaml`, `ui/live.py` | layout: base on the table (z 0), no pedestal, box floor -0.005 | G5-G7 (P3) | eval.yaml `scene.station_scene.table.top_z_m` 0.0 from 0943 on; start TCP z 0.161 |
| P2 | P3/P4 | `run.sh` | the sim PC could die after its scene hello (C2 bind refused while another program held :5620) -> wait for `ready: C2 on`, retry the start (3 x 3 s) | P3 runs | `_logs/2026-11-20_094132`, `095002`, `095223`: sim_pc.log "FATAL ... cannot bind the C2 endpoint tcp://127.0.0.1:5620 (Address already in use)", no eval.log |
| P3 | P3/P4 | `eval/driver.py` | no TCP path in the results -> `trace.json` gets `tcp_O_xyz_wxyz` | G5 (P3 drift), G6, G7 | column present in P3 traces only |
| P4 | P3/P4 | `eval/driver.py`, `run.sh` | one start pose file only -> `--start-pose FILE` | reproducing P3 now | - |
| P5 | P3/P4 | `sim_pc/merge_station.py`, `run.sh` | no sim-only settings per run -> `SIM_DEPLOY_SIM_PC_OVERLAY` (refused if it sets `q_start_rad`); `vmax05_g6.yaml` | G6 v_max 0.5 | `_logs/2026-11-20_094921/sim_pc.log` "(v_max 0.50 m/s)"; eval.yaml does not record the overlay |
| P6 | P3/P4 | `scene/README.md` | "all 30 replays grasp within 2 cm from the start pose": only the median does; 7 are `grasp_ok` | G6 episode choice | `scene/README.md` (episodes 0, 7, 14, 16, 19, 23, 28) |
| P7, P8 | P3/P4 | `scene/start_sweep.py`, `scene/gate_check.py` | new: sweep runner; demos through a Python port of the gate | not the checks (P4; gate_check explains G6) | [../01_start_pose/](../01_start_pose/README.md) |

Rows 1-12 are DESIGN "E7: integration" -> "Fix log"; P1-P8 are DESIGN "Fix log (P3/P4)" rows 1-8. P1/P2's own work
(the key `p` port, `--home start`) is in [01_start_pose](../01_start_pose/README.md); it made G5's refusals possible.

### Superseded runs

| Eval folder (`sim_evals/...`) | Stage | Why superseded | SUPERSEDED.txt? | Replaced by |
|---|---|---|---|---|
| `_baseline/hold_still/2026-11-20_0939_p3_g5_hold` | P3 | old table z 0.05, model `7746a439` | **yes** | `2026-11-20_0943_p3_g5_hold_z0` |
| `_baseline/hold_still/2026-11-20_0130_g5_hold`, `_0134_g5_hold_deploy`, `_0136_g5_hold_trainstart`, `_0145_g5_watchdog` | E7 | old table z 0.05, model `701815be`, the copy's box, deploy / trainstart homes from before the start pose | **no** | P3 0943 (hold), 0945 (refusal); the watchdog was not redone |
| `_replay/PICK_TEST1/pick1-v1_ep003/2026-11-20_0141_g5_fault_trainstart`, `.../2026-11-20_0143_g5_fault_recovery_43` | E7 | as above; 0141 also shows the pre-fix-6 bug | **no** | not redone (the fault/recovery mechanism is unchanged; P3's G6 ep 28 and G7 faults exercise it again) |
| `_replay/PICK_TEST1/pick1-v1_ep002/2026-11-20_0147_g6` | E7 | as above (deploy home) | **no** | P3 G6 replays |
| `PICK_TEST1/run_006_2026-11-18/2026-11-20_0149_smoke` | E7 | as above (deploy home) | **no** | `2026-11-20_0957_p3_g7_start` |
| (folder gone) `_replay/PICK_TEST1/pick1-v1_ep003/2026-11-20_0143_g5_fault_recovery` | E7 | first try, stopped after the fault (before fix 8) | - | `..._0143_g5_fault_recovery_43`; only `_logs/2026-11-20_014251/` remains, no index row |

Other markers: `_baseline/hold_still/2026-11-20_0945_p3_g5_refuse_c_deploy/NOTE.txt` (intended `start_failed`).
`index.csv` keeps the rows of superseded runs. `--home trainstart` is an alias of `start` since P2, so the E7
trainstart runs cannot be reproduced with that flag.

### Model versions (`model.sha256` in eval.yaml; `model.json` `model_sha256`)

| sha (first 12) | What it is | Seen | Used by |
|---|---|---|---|
| `cc928f8306fa` | an E6 build (DESIGN "E4: as built" smoke: "E6's model sha cc928f83…"); contents not recorded | 00:43-00:47 | `_dry_run` folders 0043, 0046, 0047 only |
| `9c75edf8d681` | wave-2 / E6 final: old table z 0.05, CAD tag positions (tags 40.58 mm apart at gap 0), no appearance, no contact pairs; built 05:49:54Z (00:49 local) | 00:53 | `_dry_run/.../0053_final2`, `0053_g6`; `$S/e7/backup/model_before.json` |
| `701815be08cd` | E7: D8 tags (44.29 mm), appearance, fin-ray contact pairs; old table z 0.05 | 01:30-01:53 | **every E7 check run** (G5, G6 ep 2, G7 smoke); `$S/e7/sim_logs/` |
| `7746a4391b50` | built between 01:53 and 09:18 (no build log found); old table z 0.05 | 09:18-09:39 | P1 smoke logs (`$S/p1/fc_*.log`), P2 dry runs (`_dry_run/... 0923-0930`), the superseded `0939_p3_g5_hold` |
| `166a0649571a` | P3/P4: table top O z 0, no pedestal; built 14:40:51Z (09:40 local); current | 09:41 on | every P3 run from 0943 on |

## Analysis

The checks split in two. G1-G5 show the sim reproduces the deployment's software path, geometry, stream and PC
behaviour. G6 and G7 show what happens when motions pass through the real gate: open-loop replays of human-speed
demos lose distance they cannot recover (a gate effect that applies on the robot too), and run_006 backs away from
the sim's images (a sim appearance problem: swapping in a real IMX219 image restores the approach). PLAN §11 now
calls policies in the sim "not functional" and plans W0 (re-sync with upstream `f632576`) before W1-W3.

### Docs vs raw data

Collected from the parts (each part has its own list with the full context):

1. **Model sha of E7.** sim-experiments/README.md "`7746a439…` (E7, D8 tags)" and PLAN §6 G2 "one sha (`7746a439…`
   now)": every E7 check eval.yaml says `701815be08cd…`; `7746a439…` is P1/P2 + the superseded 0939
   (`sim_evals/*/2026-11-20_01*/eval.yaml`, `$S/p1/fc_*.log`).
2. **Box during E7.** sim-experiments/README.md (sim-only table, "every sim run") and DESIGN E7 fix 12 give the sim
   box min [0.20, -0.40, 0.045]: the E7 evals ran with min x 0.250 / z 0.050 (gate texts in
   `0141_.../episodes.jsonl`, `0147_g6/episodes.jsonl`, `$S/e7/sim_logs/sim_pc.log`).
3. **G3 px.** PLAN §6 G3 "with training's `T_cam_tcp` 5.0 / 12.0 px": the two tags' means (4.97 / 12.03); maxima 8.20 /
   17.71 px; both-tag nominal 8.40 / 17.18 (`$S/e7/g3/g3_report.json`).
4. **G4 drift.** PLAN §6 G4 "drift 20.1 ppm": stats lines 20.2, 20.1 and 22.5 ppm (`$S/e7/g4/frames.log`).
5. **G1 file count.** PLAN §6 G1 "18 listed files": true at E7 (`$S/p1/copy_before/`); 61 now (P1's port).
6. **G5 hold duration.** PLAN §6 G5 "over 20 s": the first E7 hold (`0130_g5_hold`) ran 15 s (`--timeout 15`); the
   quoted drift is from 0134 / 0136 (20.1 s), measured from C2 recordings in the scratchpad, not in sim_evals.
7. **G5 drift P3.** PLAN "0.07 mm / 0.12 deg": `0943_p3_g5_hold_z0` 0.066 mm / 0.121 deg; 0.070 / 0.123 is the
   superseded 0939 (trace.json).
8. **G5 start pose.** PLAN "within 0.0001 rad": `max_joint_err_rad` 0.000114 (`0943_p3_g5_hold_z0/episodes.jsonl`).
9. **G5 fault.** PLAN "out-of-box waypoint -> fault" reads as a scripted test: it was a replay of ep 3 pinching below
   the old box floor (z 0.034 / 0.037 / 0.038 m < 0.050).
10. **G5 watchdog.** PLAN "stopped ~1 s after the last C3": raw `holding` -> `stopped` 0.77 s; the last C3 time was
    not recorded (`$S/e7/g5_wd_c2.jsonl`).
11. **G5 tracking.** "scripted waypoints tracked within a set error": no run tests it.
12. **G6 E7 box.** PLAN "then left the old box (x 0.238)": a rejected waypoint at x 0.238; the TCP's minimum x was
    0.2771 m (`$S/e7/g6_c2.jsonl`).
13. **G6 demo speeds.** PLAN "peaks 0.6-1.5 m/s": the 29 demos' peaks are 0.514-1.529 m/s, five below 0.6
    (`data/g6/demo_speed.csv`, recomputed here).
14. **G6 offline grasp.** PLAN "Grasp itself checked offline: 5 N/pad at 0.054, no slip": no raw file found (from
    DESIGN "Fin-ray compliance (sim)").
15. **Overlay record.** sim-experiments/README.md "Each eval's eval.yaml records the merged station": eval.yaml has only
    the path `sim_deploy/build/control_sim_start.merged.yaml` (rewritten by later runs); the v_max 0.5 overlay is
    visible only in `_logs/<time>/sim_pc.log`.
16. **G7 E7 height.** PLAN "TCP up to z 0.71 > box 0.70": 0.715 / 0.712 m are rejected waypoint z values; the reached
    TCP height is not recorded (no TCP column in E7 traces). Fault times 7.28 / 5.98 s ("after 6-7 s").
17. **G7 tap.** "tap images saved" is part of the criterion; the E7 smoke ran without `--tap`.
18. **G7 offline prompt.** The offline first-chunk logs use prompt `('PICK_TEST1', 14)`, the eval `prompt_index` 0
    (`sim_evals/_start_pose_sweep/p3/offline.log` vs `0957_p3_g7_start/eval.yaml`).
19. **G7 "demos +5.7 cm in 0.75 s"** and **VRAM "3.3 GB"**: the first has no raw file (from DESIGN); the second is a
    3306 MiB maximum in `$S/e7/g7_vram.txt`.
20. **Failed sim starts.** DESIGN "P3 results": "two sim starts failed with cannot bind the C2 endpoint": three
    `_logs` folders show it (`2026-11-20_094132`, `095002`, `095223`).
21. **Upstream values.** sim-experiments/README.md "Upstream final-deployment still says 150 mm" and "x 0.25, z 0.05
    (upstream)": upstream `rnoro/final-deployment` has F_T_TCP z 0.23565 m since `927a5ba` (`settled: true` at
    `f632576`) and box min [0.20, -0.40, -0.005] at `447910b` (`git show <ref>:deployment/contracts/frames.yaml`,
    `.../franka_control/config.yaml`); 150 mm and x 0.25 / z 0.05 are only in the ea9f7fb base of the copy.

## Conclusion

G1-G5 pass: the copy, geometry, cameras (with the explained ~9 mm camera residual), stream and PC behaviour stand in
for the robot. G6 passes for what it checks (frames and conventions: replays 19 and 0 grasp and lift) and shows that
open-loop replays cannot recover what the 0.25 m/s gate cuts from the demos' fast bursts. G7 fails: run_006 backs away
from the sim's IMX219 images, so no benchmark runs yet (PLAN §11: policies "not functional"; W0 re-sync first, then
W1/W2 on the image gap, W3 closed-loop evals). After W0, G1, G4, G5 and G6 must be re-run, G2/G3 rechecked where the
gripper changes, and G7 redone in W3.

## Limitations / sim-vs-real caveats

- G1-G4's evidence lives in a temporary scratchpad; only the copies in `data/` are durable.
- Most checks have one or two episodes and one seed.
- The E7 runs used the old layout, an older model and the copy's box; they are superseded but not marked.
- Contact, lighting and timing are models (see the parts).

## Open questions

1. After W0: does the chain still pass G1/G4/G5 (W0's smoke) and does a replay still grasp with the 0.04 m gripper
   max and `gripper_map`?
2. Tracking under scripted waypoints (G5's unrun item): a scripted line / square at 0.1-0.25 m/s.
3. Which part of the IMX219 render the policy rejects, and does a plain-diffusion checkpoint behave the same (W1/W2)?

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| eval folders | listed in `data/evals.csv` (`sim_evals/_baseline/hold_still/`, `sim_evals/_replay/PICK_TEST1/`, `sim_evals/PICK_TEST1/run_006_2026-11-18/`) |
| index | `sim_evals/index.csv` (one row per eval folder) |
| logs | `sim_evals/_logs/2026-11-20_01*` (E7), `sim_evals/_logs/2026-11-20_09*` (P3) |
| scratchpad (temporary) | `$S/e7/` (G1-G7 E7 evidence), `$S/p1/`, `$S/p34/` |
| copied / computed here | `data/evals.csv`, `data/g1_g4/`, `data/g5/`, `data/g6/`, `data/g7/` (about 0.5 MB) |
