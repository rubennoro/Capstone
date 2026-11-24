# 01_start_pose / 3: early-failure screen of the feasible start poses with run_006

**Status:** done (stopped after 23 of 26 poses; the full stage not run) · **Date:** 2026-11-20 · **Who:** agent ·
**Plan ref:** sim_deploy/PLAN.md §9 (P4 row), §6 "P3" table row G7, §10; DESIGN.md "P3/P4" -> "Screen and full eval";
DIRECTIONS.md §2b, §2c

Part 3 of 6 of [01_start_pose](README.md). Before: [1_p_key_and_standins.md](1_p_key_and_standins.md),
[2_candidates_envelope.md](2_candidates_envelope.md) (where the feasible poses come from). After:
[4_gate_check.md](4_gate_check.md) (the offline ranking that replaced this screen),
[5_real_pc_handoff.md](5_real_pc_handoff.md), [6_camera_swap.md](6_camera_swap.md).

## Question

Run with the trained policy from each feasible start pose for one short episode: does any pose fault early
(`fault_early_<rule>`, a fault in the first 10 s), and does any pose reach approach, grasp or lift, so that the
screen can rank the poses for the full stage? Answer: one early fault in 23 poses; no approach, grasp or lift
anywhere; the screen cannot rank the poses.

## Why it matters for the real robot

The start pose (`homing.q_start_rad`, PC key `p`) still has to be measured by hand-guiding the real arm. A pose
where the policy faults within seconds, or never comes near the eraser, would be a bad target for that. If the screen
had separated the poses, its ranking would have gone into the recommendation for the real arm. It did not, so the
choice (chosen pose `leftx55_t48_h16`) rests on the offline checks in [2_candidates_envelope.md](2_candidates_envelope.md)
and [4_gate_check.md](4_gate_check.md). On the real arm the policy has to be judged on its own: what this screen
shows is the policy's reaction to the sim's images, not to the poses.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`); every `eval.yaml` `git.sim_deploy`: `3633fcb3b23aba2f7eef74ceb0543f5924987720-dirty` |
| Model | `sim_deploy/build/sim_rig/model.json` `model_sha256` `166a0649571a…` (full: `166a0649571ab688ee9b18704c189f7d0aea6c80814a6be52fac10ef1e17b075`, same in all 24 `eval.yaml`): P3/P4 layout, base on the table, table top = O z 0, sim box floor z -0.005 m; current appearance (procedural table, directional lights) |
| Home / start pose | `--home start`: PC `h` (to `q_home_rad`), then PC `p` to the pose under test, given per eval with `--start-pose sim_evals/_start_pose_sweep/poses/<pose id>.yaml` (not `stations/start_pose.yaml`). Pose ids `<heading>_t<tilt>_h<height>`: tilt {38, 48, 58} deg, fingertip height {10, 16, 22} cm above the table, heading left / away / right / away_left / away_right (see [2_candidates_envelope.md](2_candidates_envelope.md)); the eraser region centre at O (0.45, 0) m for left / right, (0.50, 0) m for the diagonals, (0.60, 0) m for away |
| Eraser placement | `sim_deploy/scene/randomization_start.yaml` (`pick_eraser_start_v1`, relative to the start pose: ahead 0.09-0.217 m, left -0.035 to -0.003 m, yaw -6.6 to +0.8 deg), episode seed `[0, 0]`; the same relative placement at every pose (episode 0 of `left_t38_h22`: ahead 0.171 m, left -0.026 m, yaw -6.3 deg); not clipped in any of the 23 episodes |
| Stations / overlays | `sim_deploy/stations/control_sim.yaml` + the pose file + overlay `sim_deploy/stations/homing_fast_sweep.yaml` (`--fast-homing`: `franka_control.homing.max_joint_speed_rad_per_s` 0.6 instead of 0.3 rad/s), merged per pose into `sim_deploy/build/control_sim_sweep_<pose id>.merged.yaml` (all 24 have 0.6) |
| Policy / checkpoint | `PICK_TEST1/run_006_2026-11-18` (`~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18`), kind prompting (behavior prompting, `config_name` `umi_policy_dunetp`; policy_deploy kind `diffusion` with `prompt_dataset` `pick1-v1.zarr.zip`, `prompt_index` 0), training dataset `pick1-v1` (30 PICK_TEST1 episodes), 60 epochs, checkpoint `top1_epoch=0059-val_action_mse_error_all=0.0002007.ckpt` |
| Inference settings | bf16, 8 diffusion steps, `exec_h` 12 (from each `eval.yaml` `command` and `policy.station_policy` / `station_manager`; `start_sweep.run` sets them for every checkpoint) |
| Seeds / episodes | `--seed 0`, 1 episode per pose (`episode_seeds` `[[0, 0]]`) |
| Timeout | `--timeout 25` s (judge `timeout_s` 25.0) |
| Sim-only settings | homing 0.6 rad/s (`--fast-homing`; the `p` move took 4.96-11.14 s, `t_start_pose_s` in `episodes.jsonl`; the policy never sees homing); sim box x min 0.20 m, y -0.40..0.40 m, z -0.005..0.70 m (`control_sim.yaml`); no external video (`--video-fps 0`); finger contact, width offset and appearance as listed in [../README.md](../README.md) "Sim-only settings" |

## Method

1. Tool: `python -m sim_deploy.scene.start_sweep run --stage screen --poses feasible` (`sim_deploy/scene/start_sweep.py`
   `run`). `--poses feasible` takes every `ok` row of `sim_evals/_start_pose_sweep/candidates.json` in file order.
   For each pose it starts one sim (sim PC `franka_control --sim` with the merged station, sim Pi), then runs the
   eval driver (`python -m sim_deploy.eval run ... --start-pose <pose file> --tag sweep_screen_<pose id>`), which
   presses `h`, `p`, places the eraser relative to the start pose and presses the laptop's `c`; a latched fault
   (driver exit 3) would restart the sim and `--resume` (not needed here: one episode per pose).
2. The judge (`sim_deploy/eval/driver.py`) polls the sim's ground truth at 20 Hz and writes `episodes.jsonl`
   (outcome, C2 error text, `duration_s`) and `episodes/000/trace.json` (`t_s`, `lift_m`, `grasped`, `finger_gap_m`,
   `tcp_O_xyz_wxyz`).
3. Classes (`start_sweep.classify`): `fault_early_<rule>` if the outcome is `fault` and `duration_s` < 10 s; else
   lift, grasp, approach (closest TCP to the settled eraser centre < 5 cm over the 20 Hz path), no_progress. Rule from
   the C2 error text: box ("outside workspace box"), cone ("from approach axis"), reflex, v_max. The fault time is
   `duration_s`, the time from the episode start (`c`) to the fault.
4. Pass, as planned (PLAN §9 P4): poses that do not fail early, and among them those that reach approach / grasp /
   lift, go to the full stage (DIRECTIONS §2c: `--stage full`, chosen poses x checkpoints, `--episodes 2 --timeout 60
   --fast-homing`). The code has the same defaults for both stages (`--episodes 1`, `--timeout 25`, `--seed 0`); the
   full stage differs only by the flags given.
5. Verification for this file: a read-only script recomputed every row of `sim_evals/_start_pose_sweep/summary.csv`
   from the eval folders (`eval.yaml`, `episodes.jsonl`, `metrics.json`, `episodes/000/trace.json`), plus the start
   distance, the time of the closest distance, the TCP motion along the fingertip direction in the first 3 s and the
   highest TCP z, into `data/p3screen_per_pose.csv`.

## Commands to reproduce

About 27 min wall time for the 23 poses (first sim start 10:02:15, the 24th sim start 10:28:51; 46-105 s per pose).
Needs the sim ports (5555, 5620, 5630, 5640, 5650, 5651) and the laptop GPU (4 GB) to itself: nothing else on the sim
chain may run at the same time, and test suites that bind :5620 / :5555 make the sim start retry (it did once here,
at `left_t48_h10`).

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# the candidates and pose files (already there; regenerating rewrites candidates.json and poses/)
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep candidates
# the screen (equivalent to the 2026-11-20 10:02 run; each eval.yaml records the driver flags): every feasible pose, run_006, 1 episode, 25 s, seed 0, fast homing, no video
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep run --stage screen --poses feasible \
    --policies ~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 --episodes 1 --timeout 25 --seed 0 \
    --fast-homing --video-fps 0
# only the 23 poses of this record (today's candidates.json also has the 14 feasible shifted leftx* poses):
#   --poses left_t38_h22,left_t48_h10,...,away_right_t48_h22  (the list in Results)
# the summary (run automatically at the end of `run`; skips folders with INTERRUPTED.txt or SUPERSEDED.txt)
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep summary
```

The full stage that was not run would have been (DIRECTIONS §2c):
`... start_sweep run --stage full --poses <ID>,<ID> --policies <RUN_FOLDER>,<RUN_FOLDER> --episodes 2 --timeout 60 --fast-homing`.

## Results

### Per pose

All 23 recorded evals are under `sim_evals/PICK_TEST1/run_006_2026-11-18/`. Success, grasp and lift are 0 in every
row (`max_lift_m` <= 8e-9 m). Distances are from the TCP (fingertip point) to the settled eraser centre, from the
20 Hz path in `trace.json`. Every row of `summary.csv` agrees with its eval folder (outcome, rule, fault time,
class, closest distance, pose fields); no row is flagged.

| pose | tilt (deg) | height (cm) | heading | outcome | fault rule (C2 error) | fault time (s) | class | start TCP-eraser (m) | closest TCP-eraser (m) | eval folder |
|---|---|---|---|---|---|---|---|---|---|---|
| `left_t38_h22` | 38 | 22 | left | fault | box: z 0.710 m > max 0.700 m | 14.44 | no_progress | 0.275 | 0.275 | `2026-11-20_1002_sweep_screen_left_t38_h22` |
| `left_t48_h10` | 48 | 10 | left | fault | box: y -0.413 m < min -0.400 m | 20.76 | no_progress | 0.197 | 0.192 | `2026-11-20_1003_sweep_screen_left_t48_h10` |
| `left_t48_h16` | 48 | 16 | left | fault | box: y -0.411 m < min -0.400 m | 18.30 | no_progress | 0.231 | 0.230 | `2026-11-20_1004_sweep_screen_left_t48_h16` |
| `left_t48_h22` | 48 | 22 | left | fault | cone: tilt 62.1 deg > 60 + 2 deg | 15.44 | no_progress | 0.275 | 0.275 | `2026-11-20_1005_sweep_screen_left_t48_h22` |
| `left_t58_h10` | 58 | 10 | left | fault | box: y -0.411 m < min -0.400 m | 19.25 | no_progress | 0.197 | 0.197 | `2026-11-20_1006_sweep_screen_left_t58_h10` |
| `left_t58_h16` | 58 | 16 | left | fault | box: y -0.412 m < min -0.400 m | 20.12 | no_progress | 0.231 | 0.231 | `2026-11-20_1008_sweep_screen_left_t58_h16` |
| `left_t58_h22` | 58 | 22 | left | fault | cone: tilt 62.1 deg > 60 + 2 deg | 19.60 | no_progress | 0.275 | 0.266 | `2026-11-20_1009_sweep_screen_left_t58_h22` |
| `away_t48_h16` | 48 | 16 | away | fault | cone: tilt 68.4 deg > 60 + 2 deg | 12.50 | no_progress | 0.231 | 0.206 | `2026-11-20_1011_sweep_screen_away_t48_h16` |
| `right_t38_h22` | 38 | 22 | right | fault | box: z 0.718 m > max 0.700 m | 19.84 | no_progress | 0.275 | 0.177 | `2026-11-20_1012_sweep_screen_right_t38_h22` |
| `right_t48_h10` | 48 | 10 | right | timeout | - | - | no_progress | 0.197 | 0.105 | `2026-11-20_1013_sweep_screen_right_t48_h10` |
| `right_t58_h10` | 58 | 10 | right | timeout | - | - | no_progress | 0.197 | 0.107 | `2026-11-20_1015_sweep_screen_right_t58_h10` |
| `right_t58_h16` | 58 | 16 | right | timeout | - | - | no_progress | 0.231 | 0.137 | `2026-11-20_1016_sweep_screen_right_t58_h16` |
| `away_left_t38_h22` | 38 | 22 | away_left | fault | box: z 0.716 m > max 0.700 m | 17.34 | no_progress | 0.275 | 0.275 | `2026-11-20_1017_sweep_screen_away_left_t38_h22` |
| `away_left_t48_h10` | 48 | 10 | away_left | fault | box: x 0.188 m < min 0.200 m | 22.82 | no_progress | 0.197 | 0.193 | `2026-11-20_1018_sweep_screen_away_left_t48_h10` |
| `away_left_t48_h16` | 48 | 16 | away_left | fault | box: x 0.189 m < min 0.200 m | 20.59 | no_progress | 0.231 | 0.225 | `2026-11-20_1019_sweep_screen_away_left_t48_h16` |
| `away_left_t48_h22` | 48 | 22 | away_left | fault | cone: tilt 62.1 deg > 60 + 2 deg | 19.25 | no_progress | 0.275 | 0.263 | `2026-11-20_1020_sweep_screen_away_left_t48_h22` |
| `away_left_t58_h10` | 58 | 10 | away_left | fault | cone: tilt 62.2 deg > 60 + 2 deg | 9.77 | fault_early_cone | 0.197 | 0.189 | `2026-11-20_1021_sweep_screen_away_left_t58_h10` |
| `away_left_t58_h16` | 58 | 16 | away_left | fault | box: x 0.190 m < min 0.200 m | 19.40 | no_progress | 0.231 | 0.229 | `2026-11-20_1022_sweep_screen_away_left_t58_h16` |
| `away_left_t58_h22` | 58 | 22 | away_left | fault | box: z 0.711 m > max 0.700 m | 18.65 | no_progress | 0.275 | 0.271 | `2026-11-20_1023_sweep_screen_away_left_t58_h22` |
| `away_right_t38_h22` | 38 | 22 | away_right | fault | cone: tilt 63.2 deg > 60 + 2 deg | 23.71 | no_progress | 0.275 | 0.273 | `2026-11-20_1024_sweep_screen_away_right_t38_h22` |
| `away_right_t48_h10` | 48 | 10 | away_right | fault | cone: tilt 62.0 deg > 60 + 2 deg | 20.35 | no_progress | 0.197 | 0.196 | `2026-11-20_1025_sweep_screen_away_right_t48_h10` |
| `away_right_t48_h16` | 48 | 16 | away_right | timeout | - | - | no_progress | 0.231 | 0.231 | `2026-11-20_1026_sweep_screen_away_right_t48_h16` |
| `away_right_t48_h22` | 48 | 22 | away_right | timeout | - | - | no_progress | 0.275 | 0.275 | `2026-11-20_1027_sweep_screen_away_right_t48_h22` |
| `away_right_t58_h10` | 58 | 10 | away_right | interrupted (no `episodes.jsonl`, `trace.json` or `metrics.json`) | - | - | - | - | - | `2026-11-20_1029_sweep_screen_away_right_t58_h10` |

The 24th eval (`away_right_t58_h10`) has `INTERRUPTED.txt`: "INTERRUPTED (P4, 2026-11-20): the screen was stopped
here on the coordinator's change of plan; not in the sweep summary." Its logs show the episode running: in
`policy_deploy.log` up to "ep 1 running 23.7 s, step 83"; in `sim_evals/_logs/2026-11-20_102851_sweep_screen_away_right_t58_h10/sim_pc.log`
the start pose reached at 10:29:32, mode `running` to the end, no refused waypoint, then "received SIGTERM" at
10:29:57, about 1 s before the 25 s timeout. Its result is not recorded. It has no `sim_evals/index.csv` row (the
other 23 have one each, outcomes as above).

### Counts

| Count | Value (raw) |
|---|---|
| poses screened (recorded) | 23 (+ 1 interrupted) |
| outcomes | 18 fault, 5 timeout, 0 success |
| fault rules | box 11 (z > 0.700 m: 4; y < -0.400 m: 4, all heading left at 10 / 16 cm; x < 0.200 m: 3, all away_left at 10 / 16 cm), cone 7 (tilt 62.0-68.4 deg) |
| timeouts | `right_t48_h10`, `right_t58_h10`, `right_t58_h16`, `away_right_t48_h16`, `away_right_t48_h22` |
| classes | `fault_early_cone` 1 (`away_left_t58_h10`, 9.77 s); no_progress 22; approach, grasp, lift 0 |
| fault times | early: 9.77 s; the other 17 faults: 12.50-23.71 s |
| start TCP-eraser distance | 0.197 / 0.231 / 0.275 m at 10 / 16 / 22 cm height (the same at every heading and tilt: the placement is relative to the start pose) |
| closest TCP-eraser distance | 0.105-0.275 m; never below 0.05 m (so no approach) |
| closer than at the start | by >= 0.5 mm in 17 of 23; by 8.9-9.8 cm at the 4 right-heading poses, 2.5 cm at `away_t48_h16`, 0.05-1.2 cm at 12 others; never closer at 4 (`left_t38_h22`, `left_t48_h22`, `left_t58_h10`, `away_left_t38_h22`), less than 0.5 mm closer at 2 (`away_right_t48_h16`, `away_right_t48_h22`) |

### Early motion and timing

TCP motion from the start along the fingertip direction (`fingertip_dir_O` in `eval.yaml`) and in z, 3 s into the
episode; the highest TCP z of the 20 Hz path; inference from `metrics.json` `steps`; the gate's waypoint counts
from `episodes.jsonl` `c2.gate`. Skipped steps: 0 in every eval.

| pose | closer than start (cm) | at (s) | along fingertips at 3 s (cm) | z at 3 s (cm) | max TCP z (m) | policy_ms_p50 | policy_ms_p95_max | margin_ms_min | steps sent | gate shortened / pulled back / rejected |
|---|---|---|---|---|---|---|---|---|---|---|
| `left_t38_h22` | 0.0 | 0.0 | -6.2 | +6.5 | 0.690 | 270.3 | 279.1 | -133.6 | 50 | 53 / 3 / 3 |
| `left_t48_h10` | 0.5 | 3.66 | -0.8 | +2.4 | 0.591 | 271.3 | 278.4 | -137.1 | 72 | 86 / 57 / 4 |
| `left_t48_h16` | 0.1 | 0.85 | -1.7 | +2.3 | 0.590 | 269.5 | 277.3 | -115.5 | 64 | 64 / 64 / 1 |
| `left_t48_h22` | 0.0 | 0.0 | -3.1 | +3.2 | 0.576 | 273.0 | 282.0 | -164.3 | 53 | 61 / 64 / 3 |
| `left_t58_h10` | 0.0 | 0.0 | -0.9 | +1.7 | 0.482 | 281.2 | 329.4 | -256.8 | 61 | 89 / 124 / 1 |
| `left_t58_h16` | 0.1 | 0.96 | -3.8 | +3.6 | 0.556 | 284.5 | 307.8 | -270.4 | 63 | 96 / 135 / 1 |
| `left_t58_h22` | 0.9 | 1.07 | -0.9 | +1.7 | 0.565 | 303.0 | 359.0 | -382.8 | 56 | 57 / 95 / 2 |
| `away_t48_h16` | 2.5 | 3.11 | +2.3 | -0.2 | 0.484 | 265.9 | 281.1 | -157.0 | 43 | 61 / 0 / 4 |
| `right_t38_h22` | 9.8 | 3.80 | +8.4 | -5.8 | 0.692 | 276.6 | 282.3 | -163.3 | 67 | 67 / 1 / 4 |
| `right_t48_h10` | 9.2 | 3.71 | +7.8 | -3.7 | 0.454 | 277.1 | 284.8 | -169.9 | 86 | 121 / 0 / 0 |
| `right_t58_h10` | 8.9 | 4.21 | +8.6 | -3.7 | 0.478 | 274.7 | 284.4 | -188.5 | 86 | 85 / 0 / 0 |
| `right_t58_h16` | 9.4 | 3.41 | +9.0 | -4.5 | 0.498 | 275.9 | 287.9 | -155.7 | 86 | 77 / 0 / 0 |
| `away_left_t38_h22` | 0.0 | 0.0 | -5.2 | +5.1 | 0.698 | 268.7 | 278.9 | -182.4 | 60 | 79 / 17 / 4 |
| `away_left_t48_h10` | 0.4 | 0.90 | -1.7 | +2.6 | 0.566 | 267.3 | 275.9 | -142.9 | 80 | 93 / 133 / 4 |
| `away_left_t48_h16` | 0.7 | 1.11 | -1.4 | +1.6 | 0.594 | 268.0 | 277.6 | -147.7 | 72 | 84 / 97 / 5 |
| `away_left_t48_h22` | 1.2 | 1.70 | +0.4 | +0.7 | 0.615 | 265.8 | 272.3 | -152.3 | 68 | 83 / 42 / 2 |
| `away_left_t58_h10` | 0.8 | 0.95 | -1.6 | +3.0 | 0.245 | 265.3 | 279.6 | -131.0 | 34 | 33 / 57 / 4 |
| `away_left_t58_h16` | 0.3 | 0.85 | -0.8 | +2.5 | 0.595 | 264.6 | 271.9 | -114.2 | 69 | 53 / 182 / 4 |
| `away_left_t58_h22` | 0.4 | 0.85 | +1.0 | +1.4 | 0.691 | 262.6 | 267.9 | -104.4 | 67 | 65 / 183 / 3 |
| `away_right_t38_h22` | 0.1 | 0.85 | -2.0 | +1.5 | 0.688 | 273.4 | 277.9 | -119.9 | 82 | 92 / 97 / 4 |
| `away_right_t48_h10` | 0.1 | 0.35 | -2.8 | +2.1 | 0.300 | 270.5 | 276.8 | -120.4 | 71 | 78 / 56 / 1 |
| `away_right_t48_h16` | 0.0 | 0.30 | -2.7 | +1.7 | 0.526 | 272.8 | 278.9 | -139.0 | 88 | 107 / 127 / 0 |
| `away_right_t48_h22` | 0.0 | 0.30 | -3.3 | +1.9 | 0.502 | 278.2 | 288.4 | -151.8 | 85 | 145 / 60 / 0 |

Inference over the 23 evals: `policy_ms_p50` 262.6-303.0 ms, `margin_ms_min` -382.8 to -104.4 ms, 0 skipped.
E7's smoke run had 177-182 ms p50 (PLAN §6 G7 row); the slower steps here match G7 under P3 (270 / 270 / 188 ms p50),
which DESIGN.md "P3 results" puts down to another session's test suites loading the laptop. During this screen at
least one other program held :5620 for a while (the sim start for `left_t48_h10` failed once with "cannot bind the C2
endpoint tcp://127.0.0.1:5620 (Address already in use)" and succeeded on the retry; coordinator's console log, not in
the repo).

## Analysis

**What the screen shows.** At every pose the episode ends far from the eraser and climbing: the highest TCP z is
0.454-0.698 m in 21 of 23 (the start heights are 0.10-0.22 m), the box faults are the TCP leaving the sim box at
the top (z 0.70 m), at the robot's right edge (y -0.40 m, heading left) or at the robot-side face (x 0.20 m, heading
away_left), and the cone faults are the hand tilting past 62 deg on the way. The two lower maxima are the early cone
fault (`away_left_t58_h10`, 0.245 m at 9.77 s) and `away_right_t48_h10` (0.300 m, cone fault at 20.35 s). This is
the G7 behaviour: run_006 backs away and rises from the sim's images (PLAN §6 "P3" G7 row; stand-in eval
`sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/`: 3 faults at 18.45, 16.89 and 11.44 s, box y
-0.411 m and cone 62.0 / 63.0 deg, the same rules as here).

**The right heading is the exception in the first seconds.** The 4 right-heading poses move 7.8-9.0 cm forward along
the fingertips and 3.7-5.8 cm down in the first 3 s and get 8.9-9.8 cm closer to the eraser (closest 0.105-0.177 m at
3.4-4.2 s); then they turn and climb (3 timeouts, 1 box fault at z 0.718 m). `away_t48_h16` moves 2.3 cm forward
(closest 0.206 m at 3.1 s). Everywhere else the first 3 s go up (+0.7 to +6.5 cm) and back or barely forward (-6.2 to
+1.0 cm along the fingertips). So "the retreat from the first chunk" holds for 18 of 23 poses, not for all; even the
right-heading start is still 10.5 cm or more from the eraser at its closest, twice the approach threshold. Why the
right heading starts forward is not known: at that heading the hand is rolled near joint 7's limit
([2_candidates_envelope.md](2_candidates_envelope.md)), so the policy's images see the scene differently, but no
image was checked for these poses (open question 2).

**Why the screen cannot rank the poses.** A ranking needs a signal that depends on the pose. Here the class is
no_progress at 22 of 23 poses, the one early fault is a cone fault 0.23 s before the 10 s line (9.77 s), and the
fault rules follow the direction the arm retreats in (box y at heading left, box x at away_left, box z from 22 cm,
3 of 4 at 38 deg), i.e. where the retreat hits a wall first, not how good the pose is for a pick. The cause is the same as
G7: offline, run_006's first chunk goes +5.6 cm toward the fingertips on training images and -1.3 cm on the sim's;
swapping in training's IMX219 image gives +5.4 cm, swapping in training's IMX519 image -0.5 cm, so it is the sim's
IMX219 render that the policy rejects (DESIGN.md "P3 results"; [6_camera_swap.md](6_camera_swap.md)). Until that is
fixed ([../02_camera_study/README.md](../02_camera_study/README.md)), a closed-loop policy run in this sim measures
the policy's reaction to the render, not the start pose.

**Why the full stage was not run.** Coordinator decision (DESIGN.md "Screen and full eval": "it cannot rank poses
either, and it competes for the GPU"). The full stage would have been chosen poses x checkpoints, 2 episodes, 60 s
(DIRECTIONS §2c); longer episodes of the same retreat add no ranking signal, and the laptop's 4 GB GPU and the sim
ports were needed for the camera study. The ranking was done offline instead ([4_gate_check.md](4_gate_check.md)).

**Coverage.** `--poses feasible` at the screen's start (10:02) was the 26 feasible poses of the first 45 candidates
(today's `candidates.json` rows 0-44: left 7, away 1, right 4, away_left 7, away_right 7). The shifted `leftx50_*` /
`leftx55_*` rows were added later (their pose files are dated 10:14-10:15, `candidates.json` 10:15), so they were never
in the list. Screened and recorded: 23. Not recorded: `away_right_t58_h10` (the 24th, started 10:29:01, interrupted),
`away_right_t58_h16` and `away_right_t58_h22` (never started). None of the 14 feasible shifted poses, the chosen pose
`leftx55_t48_h16` among them, nor the P3 stand-in was screened.

**Docs vs raw data.**

1. DESIGN.md "P3/P4" -> "Screen and full eval" says "the TCP never got closer to the eraser than it started (10-28
   cm)"; raw (`trace.json` of each eval folder, `data/p3screen_per_pose.csv`): 17 of 23 got closer by >= 0.5 mm, the
   4 right-heading poses by 8.9-9.8 cm (e.g. `right_t48_h10` 0.197 -> 0.105 m in
   `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_1013_sweep_screen_right_t48_h10/episodes/000/trace.json`) and
   `away_t48_h16` by 2.5 cm (0.231 -> 0.206 m); "10-28 cm" is the range of the closest distance (0.105-0.275 m), the
   start distances are 19.7-27.5 cm.
2. DESIGN.md "Screen and full eval" says "the rest at 11-24 s"; raw (`episodes.jsonl` `duration_s`): the 17 later
   faults are at 12.50-23.71 s (earliest `away_t48_h16`, 12.50 s, in
   `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_1011_sweep_screen_away_t48_h16/episodes.jsonl`); 11 s is the
   earliest G7 fault (11.44 s), not a screen value.
3. DESIGN.md "Screen and full eval" says "Every pose shows the same retreat as G7", and the G7 row (PLAN §6 "P3",
   DESIGN "P3 results") says the policy "rises and backs off from the first chunk"; raw (`trace.json`): every pose
   ends climbing, but the 4 right-heading poses first move 7.8-9.0 cm forward along the fingertips and 3.7-5.8 cm down
   in 3 s (e.g. `right_t58_h16`: TCP (0.469, 0.159, 0.160) m -> (0.445, 0.081, 0.114) m at 3 s, in
   `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_1016_sweep_screen_right_t58_h16/episodes/000/trace.json`).
4. The other screen statements in DESIGN.md "Screen and full eval" match the raw files: 23 of the 26 first feasible
   poses; 18 faults (box 11, cone 7), 5 timeouts; no grasp, no approach; one early fault, cone, 9.77 s at
   `away_left_t58_h10`; the 24th eval marked `INTERRUPTED.txt`. `summary.csv` / `summary.md` agree with every eval
   folder. `stations/homing_fast_sweep.yaml` (0.6 rad/s) and DIRECTIONS §2c's screen flags match every `eval.yaml`
   and merged station.

## Conclusion

No feasible pose of the 23 screened lets run_006 approach the eraser in the sim (closest 10.5 cm, threshold 5 cm),
and only one faults early (`away_left_t58_h10`, cone, 9.77 s). The screen cannot rank the start poses, because the
policy retreats from the sim's IMX219 render at every pose; the right heading is the only one that starts toward the
eraser (about 9 cm) before turning away. Decided: the full stage was not run, the screen was stopped after 23 poses,
and the chosen pose `leftx55_t48_h16` was picked on the offline checks
([2_candidates_envelope.md](2_candidates_envelope.md), [4_gate_check.md](4_gate_check.md)), not on this screen.

## Limitations / sim-vs-real caveats

- One checkpoint (run_006, prompting), one episode and one eraser placement per pose, 25 s. PLAN §9 P4 asked for every
  runnable checkpoint and a few episodes each; a plain-diffusion checkpoint was not screened.
- The result is dominated by the sim's IMX219 render (appearance), which the real arm does not have; a real-arm
  episode from any of these poses can behave completely differently.
- The eraser placement is relative to the start pose, so the eraser lies at a different point on the table at each
  heading (region centre 0.45-0.60 m out); the box faults depend on where the retreat meets the sim box, which is
  set by heading and layout, not by the policy's task skill.
- Homing at 0.6 rad/s instead of 0.3 rad/s: only the `h` / `p` moves differ; every episode starts at the same pose.
- Inference ran 262.6-303.0 ms p50 with negative step margins (to -382.8 ms) under load from other work; the gate
  shortened 33-145 waypoints per episode. A faster policy step could change the timing of the retreat, not its
  direction (the offline first chunk already points away).
- The fault rules are against the sim box (x min 0.20 m, z floor -0.005 m), which is the value recommended for the
  real box but not yet set there.
- No screen result exists for the chosen pose `leftx55_t48_h16`, the other shifted poses, the 3 unrecorded
  away_right 58 deg poses, or the P3 stand-in (the stand-in has G7 instead).

## Open questions

1. Does the screen separate the poses once the IMX219 render is fixed? Answer: rerun the screen (same command, the
   poses incl. `leftx55_t48_h16`) after [../02_camera_study/README.md](../02_camera_study/README.md) C2 gives a render
   the policy approaches from.
2. Why does run_006 start toward the eraser at the right heading only? Answer: save the four policy images at
   `right_t48_h10` and at `left_t48_h10` (`policy_deploy.frames.probe --save`) and run
   `sim_evals/_start_pose_sweep/tools/offline_first_chunk.py` on them, as for G7.
3. Would `away_right_t58_h10` (interrupted 1 s before its timeout, no fault logged) have been a sixth timeout? Answer:
   one screen eval of that pose; it only matters if the screen is used for ranking again.
4. Does a plain-diffusion checkpoint (e.g. from the PICK_TEST1 / PICK_DOUBLE plain runs) behave the same at these
   poses? Answer: the screen with `--policies` listing it.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| eval folders (23 recorded) | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_1002_sweep_screen_left_t38_h22/` ... `2026-11-20_1027_sweep_screen_away_right_t48_h22/` (each: `eval.yaml`, `episodes.jsonl`, `metrics.json`, `episodes/000/trace.json`, `policy_deploy.log`, `station.yaml`, `start_pose.yaml`) |
| interrupted eval | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_1029_sweep_screen_away_right_t58_h10/` (`INTERRUPTED.txt`, `eval.yaml`, `policy_deploy.log`; no episode results) |
| logs | `sim_evals/_logs/2026-11-20_100215_sweep_screen_left_t38_h22/` ... `2026-11-20_102851_sweep_screen_away_right_t58_h10/` (`eval.log`, `sim_pc.log`, `sim_pi.log`; 24 folders) |
| sweep summary and candidates | `sim_evals/_start_pose_sweep/summary.csv`, `summary.md`, `candidates.json`, `candidates.md`, `poses/<pose id>.yaml` |
| index rows | `sim_evals/index.csv`, tags `sweep_screen_*` (23 rows) |
| merged sim PC stations | `sim_deploy/build/control_sim_sweep_<pose id>.merged.yaml` |
| G7 comparison | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/` |
| code / overlay | `sim_deploy/scene/start_sweep.py` (`run`, `classify`, `summary`), `sim_deploy/stations/homing_fast_sweep.yaml` |
| copied into this folder | `data/p3screen_summary.csv`, `data/p3screen_summary.md` (from `sim_evals/_start_pose_sweep/summary.{csv,md}`, 10:32); `data/p3screen_per_pose.csv` (derived here from the eval folders: outcome, C2 error text, fault time, class, start / closest TCP-eraser distance, motion at 3 s, highest TCP z, inference timing, gate counts, agreement with `summary.csv`) |
