# 01_start_pose / 5: the real PC handoff for the chosen pose `leftx55_t48_h16`

**Status:** done (settled: adopted upstream in rnoro/final-deployment `abba9e8`, to be checked on the arm) · **Date:** 2026-11-20 · **Who:** agent ·
**Plan ref:** sim_deploy/PLAN.md §9 (start pose, key `p`), §10 (camera study); DESIGN.md "P3/P4" -> "The chosen pose
(sim default and recommendation for the real arm)", "P4 candidates and the envelope of q_start_rad"; DIRECTIONS.md
§2b, §2c

Part 5 of 6 of [01_start_pose](README.md). Parts: [1_p_key_and_standins.md](1_p_key_and_standins.md),
[2_candidates_envelope.md](2_candidates_envelope.md), [3_early_failure_screen.md](3_early_failure_screen.md),
[4_gate_check.md](4_gate_check.md), this file, [6_camera_swap.md](6_camera_swap.md).

## Question

Does the chosen pose `leftx55_t48_h16` pass every rule of the real station's PC (key `p`, the box, the cone, the joint
limits, the 30 s homing timeout, C3 `start` at the pose), with which margins, and what exactly should the user see and
check on the Franka when it is set there? Answer: yes, with the margins in the Results table.

## Why it matters for the real robot

When this part was written (final-deployment HEAD `447910b`), `homing.q_start_rad` was `null` on the real station
(final-deployment `config.yaml`, no override in its `station.yaml`): `p` was refused and episodes started wherever the
arm was. **Since then (read 2026-11-20, HEAD `0c5903a`):** `abba9e8` set this q in `deployment/control_deploy/station.yaml`
(adopted upstream, to be checked on the arm), and `b4030f6` set `gripper_width_max_m` **0.04 m** (`contracts/limits.yaml`,
the printed fingers' measured opening) and `F_T_TCP` `settled: true`. So on the real arm `p` now opens the fingers to
0.04 m and `pc_probe` prints `gripper width: 40.0 mm`, not the 80.0 mm below (which is this repo's ea9f7fb copy and the
sim); the kinematic numbers (q, TCP, tilt, margins) do not change. This file is the handoff: the 7 numbers to
paste (or the pose to hand-guide to), what `pc_probe --start-pose` should print there, the layout the policy expects
(where the eraser goes), the margins against the real PC's rules, and the checklist before the first `c`. Nothing in
the real deployment was changed by this part.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Model | `sim_deploy/build/sim_rig/model.json` `model_sha256` `166a0649571a…` (P3/P4: base on the table, table top = O z 0, x -0.40..1.00 m, y ±0.80 m; `F_T_TCP` z 0.23565 m) |
| Home / start pose | `q_home_rad [0.0, -0.7853982, 0.0, -2.3561945, 0.0, 1.5707963, 0.7853982]`; chosen pose `leftx55_t48_h16` `q_start_rad [-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]` (TCP O (0.531, -0.159, 0.160) m, tilt 48.0 deg, heading left, height 16.0 cm) |
| Eraser placement | `sim_deploy/scene/randomization_start.yaml` (`pick_eraser_start_v1`, from PICK_TEST1 `pick1-v1`, 30 episodes) |
| Stations / overlays | sim: `stations/control_sim.yaml` + `stations/start_pose.yaml`; real station: `~/Capstone-final-deployment` (`rnoro/final-deployment` HEAD `447910b`, read-only) `deployment/control_deploy/franka_control/config.yaml` + `deployment/contracts/frames.yaml` + `deployment/control_deploy/station.yaml` |
| Policy / checkpoint | none (kinematic and rule checks only) |
| Inference settings | none |
| Seeds / episodes | none |
| Timeout | none (the PC's `homing.timeout_s` 30 s is one of the checked rules) |
| Sim-only settings | none used here (no sim was started). The sim box and the real box now have the same values. |

The three boxes used below:

| Name | min (m) | max (m) | TCP | Source |
|---|---|---|---|---|
| sim box | [0.20, -0.40, -0.005] | [0.80, 0.40, 0.70] | 235.65 mm | `sim_deploy/stations/control_sim.yaml` (overrides `workspace_min_m` only) + the copy's `config.yaml` max |
| real box (the real station today) | [0.20, -0.40, -0.005] | [0.80, 0.40, 0.70] | 235.65 mm | final-deployment `config.yaml` (z -0.005 since `927a5ba`, 2026-11-20 09:58; x 0.20 since `447910b`, 10:05); `frames.yaml` `F_T_TCP` z 0.23565 since `927a5ba` |
| upstream ea9f7fb (old snapshot) | [0.25, -0.40, 0.05] | [0.80, 0.40, 0.70] | 150 mm | `git show ea9f7fb:deployment/...`; the copy in this repo (`deployment/control_deploy/franka_control/config.yaml`) still carries this box; the copy's `frames.yaml` was changed to 235.65 mm by sim D7 |

Where "the user's real station" in DESIGN.md and in `start_sweep handoff` comes from: hard-coded constants in
`sim_deploy/scene/start_sweep.py` (`REAL_BOX = ([0.20, -0.40, -0.005], [0.80, 0.40, 0.70])` "the user's real station
(2026-11-20 working tree: x 0.20, z -0.005)", `REAL_CONE_DEG 60.0`, `REAL_MAX_JOINT_SPEED 0.3`,
`REAL_HOMING_TIMEOUT_S 30.0`, `REAL_MIN_DURATION_S 2.0`, `REAL_BOX_UPSTREAM` = the ea9f7fb box, `REAL_F_T_TCP_Z 0.150`
"final-deployment @ ea9f7fb"). They are not read from a file. The working tree they describe is now committed
upstream: final-deployment HEAD has exactly those box values, cone 60 deg, `v_max_m_per_s` 0.25,
`homing.max_joint_speed_rad_per_s` 0.3, `min_duration_s` 2.0, `timeout_s` 30.0, `at_pose_tol_rad` 0.05,
`q_start_rad: null`, and `F_T_TCP` z 0.23565 m (`settled: false`). There is a real station file
(`deployment/control_deploy/station.yaml`, in final-deployment and in the copy): it only sets `realtime: ignore` and
the console step sizes; it has no `homing:` block.

## Method

1. Read the q of the chosen pose from `sim_deploy/stations/start_pose.yaml` and
   `sim_evals/_start_pose_sweep/poses/leftx55_t48_h16.yaml` (and the 22 cm alternative from `poses/leftx55_t48_h22.yaml`).
2. Read the sweep's own handoff numbers for both poses: `sim_evals/_start_pose_sweep/candidates.json` entries (written
   by `start_sweep.evaluate()`, the same function `start_sweep handoff` prints; `real_pc` block = the real-PC rules).
3. Recompute every kinematic number without MuJoCo and without sim_deploy code, with an independent script
   (`data/p5_recompute.py`: the Panda's published modified-DH parameters + flange 0.107 m + `frames.yaml`'s `F_T_TCP`;
   the box, cone and joint margins; the `p` move time with franka_control's `HomingMotion` formula
   `max(min_duration_s, 1.875 max|dq| / max_joint_speed_rad_per_s)` from `homing_motion.cpp`; the eraser's median
   centre from `randomization_start.yaml`; the exact `pc_probe --start-pose` lines with the format strings of
   `deployment/policy_deploy/policy_deploy/manager/pc_probe.py` `start_pose_lines()`).
4. Check the rules against the code that enforces them: `supervisor.cpp` `onStartPose` / `startArmMove` (from home
   within `at_pose_tol_rad`, move time < `homing.timeout_s`, box and cone without margin at arrival), `safety_gate.cpp`
   `insideBox` / `tiltDeg` (cone on the flange z axis), C3 `start` refused unless every joint is within 0.05 rad of
   `q_start_rad`.
5. Read the demo-path and gate results for the two poses (`demo_paths.json`, `gate/gate_h16.json`, `gate/gate_h22.json`)
   only for the numbers this handoff quotes; the analysis of those is [4_gate_check.md](4_gate_check.md).

Pass: every rule accepted at the pose, the joint margin >= 0.10 rad (the sweep's `JOINT_MARGIN_MIN`), the move time
< 30 s, and the recomputation equal to the raw file to the printed digits.

`start_sweep handoff` was **not run**. Its code path (`main` -> `evaluate(q, pid="handoff", out_dir=--out)`) calls
`sp.write_start_pose(..., out_dir/"poses"/"handoff.yaml")` after `mkdir`, so it writes a file
(`sim_evals/_start_pose_sweep/poses/handoff.yaml` with the default `--out`), and it sets `MUJOCO_GL=egl` and loads the
scene for the region check (`placement.Scene.load(...).check_region()`). Its output for this q is the
`candidates.json` entry (same function, same model sha, same stations), which is copied here, and every kinematic
number in it was recomputed independently.

## Commands to reproduce

No sim, no ports, no GPU; under 1 s.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# the independent recomputation (reads the yaml files, prints JSON + text)
python3 -I sim-experiments/01_start_pose/data/p5_recompute.py
# the sweep's numbers for the two poses
python3 -I -c "import json; [print(json.dumps(r, indent=1)) for r in json.load(open('sim_evals/_start_pose_sweep/candidates.json')) if r['id'] in ('leftx55_t48_h16', 'leftx55_t48_h22')]"
# the real station, read-only
GIT_OPTIONAL_LOCKS=0 git -C ~/Capstone-final-deployment log --oneline -5 -- deployment/contracts/frames.yaml deployment/control_deploy/franka_control/config.yaml
grep -n "workspace_m\|cone_max_deg\|q_start_rad\|at_pose_tol\|max_joint_speed\|timeout_s: 30" ~/Capstone-final-deployment/deployment/control_deploy/franka_control/config.yaml
grep -n "0.23565" ~/Capstone-final-deployment/deployment/contracts/frames.yaml
GIT_OPTIONAL_LOCKS=0 git -C ~/Capstone-final-deployment show ea9f7fb:deployment/control_deploy/franka_control/config.yaml | grep -n workspace_m
# the sweep's own handoff (writes <out>/poses/handoff.yaml; not run here). With an --out outside sim_evals/:
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep handoff --out /tmp/p5_handoff \
    --q="-0.5928536,0.5219462,-0.0897168,-1.5930073,0.6595725,1.5068439,-1.5301472"
```

## Results

### The pose to hand over

```
q_start_rad: [-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]
```

Identical in `sim_deploy/stations/start_pose.yaml` (written 2026-11-20T10:18:12, source "P4 sweep ... leftx55_t48_h16"),
`sim_evals/_start_pose_sweep/poses/leftx55_t48_h16.yaml`, `candidates.json` `q`, `gate/gate_h16.json` `q`, and DESIGN.md
"P3/P4".

| | Raw (`candidates.json`) | Recomputed (`data/p5_recompute.txt`) |
|---|---|---|
| TCP (fingertip) in O, 235.65 mm TCP | (0.531, -0.159, 0.16) m | (0.531, -0.159, 0.160) m |
| height above the table (table = O z 0) | 0.16 m | 16.0 cm |
| fingertip direction (TCP z in O) | (0.0, 0.74, -0.67) | (+0.0001, +0.7431, -0.6691) |
| tilt | 48.0 deg | 48.00 deg |
| heading | 90.0 deg, "pointing to the robot's left" | +y: "pointing to the robot's left" |
| TCP in O with the old 150 mm TCP | (0.531, -0.2226, 0.2173) m | (0.531, -0.2226, 0.2173) m |
| deploy home TCP (235.65 mm), for reference | - | (0.3069, 0.0, 0.3546) m: 35.5 cm above the table |

### Expected `pc_probe --start-pose` printout at the pose

From `start_pose_lines()` (q with 4 decimals, z with 3, direction `{:+.2f}`, tilt 1 decimal, width in mm with 1
decimal), with the real station's 235.65 mm TCP and the hand opened by `p` to `gripper_width_max_m` 0.08 m
(`contracts/limits.yaml`; `width_offset_m` 0.0):

```
q_start_rad: [-0.5929, 0.5219, -0.0897, -1.5930, 0.6596, 1.5068, -1.5301]   # paste under franka_control: homing: in the PC's control_deploy/station.yaml
fingertip (TCP) height: z 0.160 m in the robot base frame
fingertip direction (TCP z axis) -> base (+0.00, +0.74, -0.67): tilted 48.0 deg from straight down, pointing to the robot's left
gripper width: 80.0 mm          # 40.0 mm on the real PC since upstream b4030f6 (gripper max 0.04 m)
demos start: fingertips tilted ~48 deg from straight down (38-64), ~16 cm above the table (6-27), hand open; object ~16 cm ahead horizontally and ~17 cm below the fingertips
```

With the old 150 mm TCP (upstream ea9f7fb, or a PC that still has it) the height line reads
`fingertip (TCP) height: z 0.217 m in the robot base frame`; the direction and tilt lines are the same (the TCP only
shifts along its own z). The q line shows the arm's measured joints, which after `p` equal the set q to within the
PC's settle.

### Assumed layout (where the policy expects the eraser)

From `randomization_start.yaml` (frame: origin = the start TCP projected on the table, ahead = the fingertip direction
projected on the table = +y here, left = up x ahead = -x here):

| | Raw | In O at the chosen pose |
|---|---|---|
| eraser centre, demos' median | `median: {ahead_m: 0.159, left_m: -0.019, yaw_rel_deg: -2.2}` | (0.550, 0.000) m (recomputed): 15.9 cm ahead of the fingertips (+y), 1.9 cm to their right (+x) |
| region ahead | `ahead_m: [0.09, 0.217]` | 9.0-21.7 cm ahead (O y -0.069..0.058 m) |
| region left | `left_m: [-0.035, -0.003]` | 0.3-3.5 cm to the right of the fingertips (O x 0.534..0.566 m) |
| eraser yaw (long axis vs ahead) | `yaw_rel_deg: [-6.6, 0.8]` | -6.6..+0.8 deg; the fingers close across its 51 mm width |
| region in the box, on the table, in both camera views | `region.ok: true`, 9 corners | - |

### Margins against the real PC's rules

| Rule (real station) | Margin, raw (`candidates.json`) | Recomputed | Passes |
|---|---|---|---|
| joint limits (Panda) | `joint_margin_rad` 1.241, joint 2 | 1.241 rad, joint 2 (all: 2.304, 1.241, 2.808, 1.479, 2.238, 1.524, 1.367 rad) | yes |
| real box, 235.65 mm TCP | `real_pc.tcp_235mm.box_margin_m` 0.165, axis z | x 0.269, y 0.241, z 0.165 m | yes |
| real box, 150 mm TCP | 0.1774, axis y | x 0.269, y 0.1774, z 0.2223 m | yes |
| upstream ea9f7fb box [0.25, -0.40, 0.05] | `upstream_box_ok` true (both TCPs) | 235.65 mm: z 0.110 m; 150 mm: z 0.1673 m | yes (at the pose) |
| cone 60 deg (flange z = TCP z) | `cone_margin_deg` 12.0 | 12.0 deg | yes |
| `p` move time < `homing.timeout_s` 30 s | `real_pc.move_time_s` 14.5 (sim `move_time_s_sim` 14.47) | max\|dq\| 2.3155 rad (joint 7): 1.875 x 2.3155 / 0.3 = 14.472 s | yes, 15.5 s spare |
| joint 7 turn during `p` | - | \|dq7\| = 2.3155 rad = 132.7 deg (dq7 = -1.5301472 - 0.7853982) | - |
| `p` only from home (`at_pose_tol_rad` 0.05) | - | `supervisor.cpp` `onStartPose`: "refused: not at home: press h first" | by design |
| `p` path from home: contacts | `path.contacts` [] | - (MuJoCo only) | yes |
| `p` path: closest to the table | `table_clearance_m` 0.1472 at s 1.0, `left_finger` | - | yes (14.7 cm) |
| `p` path: closest hand / rig / fingers / links 6-7 to links 0-3 | `self_clearance_m` 0.2397 at s 0.0, `left_finger`-`link1` | - | yes (24.0 cm) |
| accepted by the real PC | `p_accepted_real_tcp` true (150 mm), `p_accepted_fingertip_tcp` true (235.65 mm) | - | yes |

During episodes the start pose is never what is refused: through the Python port of the gate, `max_lead` never
fires and the binding rule is `v_max` 0.25 m/s (all 30 PICK_TEST1 demos shortened; `gate/gate_h16.json`); details in
[4_gate_check.md](4_gate_check.md).

### The 22 cm alternative (`leftx55_t48_h22`)

| | `leftx55_t48_h16` (chosen) | the 22 cm alternative |
|---|---|---|
| q (rad) | [-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472] | [-0.5926811, 0.4637061, -0.1004182, -1.5344559, 0.6613767, 1.4082918, -1.4666293] |
| TCP O (m) / height | (0.531, -0.159, 0.160) / 16.0 cm | (0.531, -0.159, 0.220) / 22.0 cm |
| tilt / heading | 48.0 deg / left | 48.0 deg / left |
| joint margin | 1.241 rad (joint 2) | 1.299 rad (joint 2) |
| real box margin (235.65 mm) | 0.165 m (z) | 0.225 m (z) |
| cone margin | 12.0 deg | 12.0 deg |
| `p` move time / joint 7 turn | 14.5 s / 2.3155 rad (132.7 deg) | 14.1 s (14.075 recomputed) / 2.252 rad (129.0 deg) |
| `p` path table / self clearance | 14.72 / 23.97 cm | 20.72 / 23.97 cm |
| `pc_probe` height line | z 0.160 m (150 mm TCP: 0.217) | z 0.220 m (150 mm TCP: 0.277) |
| demos carried into the table (`demo_paths.json` `box_violations_by_axis` `z<`) | 14 / 30 | 6 / 30 |
| demos refused by the gate, first refusal (`gate_h16/h22.json`, PICK_TEST1) | 17 (box: approach 14, retreat 3) | 9 (box: approach 6, retreat 3) |
| IK along the demo paths, min joint margin | 0.041 rad, 0 unreachable | 0.085 rad, 0 unreachable |

Copies: `data/p5_leftx55_t48_h22.yaml`, `data/p5_candidates_leftx55_t48.json`.

## Analysis

**The pose passes every real-PC rule with wide margins.** The smallest margins are the box floor (16.5 cm: the
height itself) and the cone (12.0 deg). Joints are 1.24 rad from their limits; the `p` move takes half the homing
timeout. The independent DH recomputation reproduces every kinematic number of `candidates.json` to the printed
digits, so the sweep's MuJoCo model and franka_control's kinematics agree on this pose. The only parts not recomputed
are the MuJoCo-only ones (contacts and clearances along the `p` path, the camera-view region check).

**What the real PC checks and what it does not.** `p` is checked only at the start (from home, move time < 30 s) and
at arrival (box and cone without margin, `supervisor.cpp`); the 1 kHz joint move itself does not go through the gate
and the real PC has no table model. The 14.7 cm table clearance and the 24.0 cm self clearance are sim knowledge: they
hold on the arm only if the real table top is the base's mounting plane (O z 0) and nothing else stands in the swept
volume. `pc_probe`'s z and tilt are computed by the PC from the joints and its `F_T_TCP`; they confirm the config and
the joints, not the physical fingertip height. A ruler (16.0 cm, fingertip to table) is the physical check of
`F_T_TCP` 235.65 mm (still `settled: false`).

**The 22 cm alternative vs the chosen pose.** The 22 cm alternative has more margin everywhere and carries fewer
demos into the table (6 vs 14 of 30). That count is an artifact of applying recorded paths open-loop: demos that
started higher than the start pose end below the table. The chosen pose sits at the demos' median start (tilt 48.2 deg,
fingertip 16.3 cm above the table, `randomization_start.yaml` `demos`), so the first images are closest to what most
demos started from. The 22 cm alternative is the fallback if 16 cm turns out tight on the arm (e.g. a real table
above the base plane).

**Why the hand-guiding window is 45-50 deg, 15-20 cm, heading left, eraser about 0.55 m out** (the envelope is in
[2_candidates_envelope.md](2_candidates_envelope.md), the gate results in [4_gate_check.md](4_gate_check.md)):

- tilt 38 deg at 10 and 16 cm: the eraser region leaves the IMX519 view (`region:in_view_imx519` for every heading).
- tilt 58 deg: feasible, but the cone margin is 2.0 deg and 9 of 30 demos then cross the cone (`demo_paths.json`
  `leftx55_t58_*` `cone_violations` 9); the demos start up to 64.0 deg.
- heading away: no IK or joint 4/5 at its limit (`away_t48_h22`, `away_t58_h10` no solution; others 0.004-0.088 rad);
  only `away_t48_h16` with the eraser at 0.60 m is feasible (0.12 rad), and 7 of 30 demo grasps or ends are
  unreachable from it (`demos_grasp_and_end_reachable` 23, min margin 0.001 rad).
- heading right: joint 7 runs into its limit (0.003-0.077 rad at 48/16, 48/22, 58/22, 38/16, 38/10).
- heading left with the eraser at 0.45 m (the stand-in's layout, `left_*`): the demos' retreat crosses the box face
  x 0.20 in 21 of 30 (`box_violations_by_axis` `x<` 21), vs 3 of 30 at 0.55 m.

**Sim success is not real success.** No checkpoint has succeeded in this sim from any start pose: only run_006 has
been run with a policy (`sim_evals/index.csv`: smoke, `p3_g7_start`, 23 `sweep_screen_*` rows, 0 successes), and the
chosen pose itself has not been run with a policy yet (no `leftx*` row in `sim_evals/_start_pose_sweep/summary.csv`;
the camera study's A2 is to do that). Offline, run_006's first chunk goes +5.6 cm toward the fingertips on training
images and -1.3 cm on the sim's; swapping in training's IMX219 image alone gives +5.4 cm, so the policy rejects the
sim's IMX219 render ([6_camera_swap.md](6_camera_swap.md), [../02_camera_study/README.md](../02_camera_study/README.md)).
The start pose was chosen on kinematics and the demos' paths, not on sim success; whether the policy works from it has
to be judged on the arm (or after the camera study fixes the render).

### Docs vs raw data

- Doc says the object sits at "~(0.53, 0.00)" at `~/Capstone-final-deployment` commit `abba9e8` (commit message and the comment in `deployment/control_deploy/station.yaml` lines 7-9); raw: the demos' median eraser centre for this pose is O **(0.55, 0.00)** m (`sim_evals/_start_pose_sweep/candidates.json` `leftx55_t48_h16.grasp_centre_O_m`; also start TCP x 0.531 + 0.019 m "right" offset from `sim_deploy/scene/randomization_start.yaml`). 0.53 is the TCP's x, not the eraser's; the placement error would be 2 cm toward the robot.
- Doc says the upstream real robot box is "x 0.25, z 0.05 (upstream, "TODO: measure")" at
  `sim-experiments/README.md` "Sim-only settings vs the real robot", row "workspace box min"; raw: final-deployment
  HEAD `447910b` has `workspace_min_m: [0.20, -0.40, -0.005]` at
  `~/Capstone-final-deployment/deployment/control_deploy/franka_control/config.yaml` line 43. The old values are only in
  ea9f7fb and in this repo's copy.
- Doc says "Upstream final-deployment still says 150 mm" at `sim-experiments/README.md` "Frames and poses", row TCP;
  raw: `F_T_TCP` z 0.23565 since `927a5ba` at `~/Capstone-final-deployment/deployment/contracts/frames.yaml` line 10.
- Doc says "upstream config @ ea9f7fb ... keep the user's -0.005 (and x 0.20 for the retreats)" and "the user's real
  station: box [0.20, -0.40, -0.005]..." at `sim_deploy/DESIGN.md` "P3/P4" -> "The chosen pose", handoff table row
  "upstream config @ ea9f7fb", and `sim_deploy/scene/start_sweep.py` lines 68-70 call it a "2026-11-20 working tree"
  and `REAL_F_T_TCP_Z = 0.150` the real TCP; raw: both are committed upstream (`927a5ba`, `447910b`), so the real
  station today is the 235.65 mm TCP with the -0.005 / 0.20 box. In `candidates.json` the field
  `p_accepted_real_tcp` is the 150 mm (old) TCP, `p_accepted_fingertip_tcp` the current real one. No number changes;
  both pass.
- Doc says "in this sim no checkpoint succeeds yet" at `sim_deploy/DESIGN.md` "The chosen pose", last sentence; raw:
  one checkpoint (run_006) was tried, from the P3 stand-in and 23 sweep poses, never from `leftx55_t48_h16`
  (`sim_evals/index.csv`, `sim_evals/_start_pose_sweep/summary.csv`).
- Doc says ">= 24.0 cm to the robot's own links" at `sim_deploy/DESIGN.md` "The chosen pose", row "p path from
  q_home"; raw: `self_clearance_m` 0.2397 is only between links 6-7 / hand / rig / fingers and links 0-3
  (`start_sweep.py` `FAR_BODIES` x `NEAR_BODIES`), at s 0.0 (the home pose, `left_finger`-`link1`) in
  `sim_evals/_start_pose_sweep/candidates.json`.
- Doc says the demos' object is "~17 cm below the fingertips" at `deployment/policy_deploy/policy_deploy/manager/pc_probe.py`
  `_DEMOS_START`, `sim_deploy/PLAN.md` §9 (P2 row) and the comment of `randomization_start.yaml`; raw: `below_m`
  median 0.161 m (range 0.064-0.272) in `sim_deploy/scene/randomization_start.yaml` `demos`.
- Doc says the start-pose table row "Fingertips tilted from straight down, pointing away from the robot, camera on
  top" and the object "centred left-right" at `~/Capstone-final-deployment/deployment/control_deploy/DIRECTIONS.md`
  step 10 "The start pose" (line 230); raw: the chosen pose points to the robot's left (`heading_deg` 90.0), and
  heading away is infeasible or at 0.001-0.12 rad joint margin (`candidates.json`, `demo_paths.json`
  `away_t48_h16`). Following those upstream directions would hand-guide the arm into the heading this sweep rules out. (That was HEAD
  `447910b`. At HEAD `0c5903a` step 10 checks the adopted q first, and line 272 says "fingers pointing to the robot's
  left", with the object at "about (0.53, 0.00) m": the heading is fixed, the placement is 2 cm short of the raw
  O (0.55, 0.00), see the first item.)
- Rounding only, no disagreement: DESIGN.md "p move 14.5 s ... joint 7 turns 2.32 rad" and "the hand rolls about
  133 deg": raw 14.472 s and 2.3155 rad = 132.7 deg (2.32 rad itself would be 132.9 deg); DESIGN.md "region 9-22 cm
  ahead ... yaw -7..+1 deg": raw 0.09-0.217 m and -6.6..0.8 deg.

## Conclusion

**Adopted upstream in `abba9e8`, to be checked on the arm.** rnoro/final-deployment commit `abba9e8` (2026-11-20 15:45) put this q into `deployment/control_deploy/station.yaml` (`franka_control.homing.q_start_rad`), identical to the 7 decimals above (`git -C ~/Capstone-final-deployment show abba9e8:deployment/control_deploy/station.yaml`, read 2026-11-20). Its comment and commit message place the object at "~(0.53, 0.00)"; the raw eraser centre is O (0.55, 0.00) m (see "Docs vs raw data").

Yes: `leftx55_t48_h16` passes every rule of the real station (box margin 0.165 m on z, cone margin 12.0 deg, joint
margin 1.241 rad on joint 2, `p` 14.472 s of 30 s with joint 7 turning 2.3155 rad), with the current 235.65 mm TCP and
with the old 150 mm one. The handoff is the q above; the expected probe lines, the eraser layout and the checklist
below go with it. The 22 cm alternative is the fallback. Nothing was changed in the real deployment.

**Hand-guiding (if the user measures instead of pasting the 7 numbers):** tilt 45-50 deg, fingertips 15-20 cm above
the table (ruler), fingertips pointing to the robot's left (+y), eraser about 0.55 m out from the base axis on the
robot's centre line, i.e. the fingertip point at about O (0.53, -0.16) m. Then paste the printed `q_start_rad` line
into final-deployment's `deployment/control_deploy/station.yaml` under `franka_control: homing:`; the sim follows with
`python -m sim_deploy.scene start-pose --from-git` (DIRECTIONS §2b).

**Checklist on the arm:**

1. `h`, then `p` slowly with a hand on the user stop: 14.5 s, the hand rolls about 133 deg (joint 7) and opens to
   80 mm. Check the cables of the rig for that roll. `p` again: refused (`not at home: press h first`).
2. On the laptop: `~/envs/policy_deploy/bin/python -m policy_deploy.manager.pc_probe --seconds 3 --start-pose`.
   Expect `z 0.160 m` (+-5 mm), `tilted 48.0 deg` (+-1 deg), `pointing to the robot's left`, `gripper width: 40.0 mm` with upstream since `b4030f6` (gripper max 0.04 m; 80.0 mm in the sim and the ea9f7fb copy),
   every joint within 0.05 rad of the q above (else C3 `start` is refused: `not at the start pose: press h, then p on
   the PC`). A `z 0.217 m` means the PC still has the 150 mm TCP. Measure the fingertip height with a ruler: 16.0 cm.
3. Place the eraser about 16 cm ahead of the fingertips (+y) and 2 cm to their right (+x), i.e. its centre at about
   O (0.55, 0.00) m, long axis along +y (within -7..+1 deg).
4. Save the four policy images: `~/envs/policy_deploy/bin/python -m policy_deploy.frames.probe --save ~/probe_images`
   (`--save DIR` exists in `deployment/policy_deploy/policy_deploy/frames/probe.py`), and compare them with training's
   episode_000152 first frame: the eraser centred, about a third up from the fingertips in the IMX219.
5. Only then `c` on the laptop.

## Limitations / sim-vs-real caveats

- The real table must be the base's mounting plane (O z 0), flat, with nothing else in reach; the PC does not check
  the `p` path for collisions.
- `F_T_TCP` 235.65 mm is from CAD (`settled: false`); a different real value moves the printed z and the TCP the gate
  uses, not the joints.
- The eraser layout is PICK_TEST1 (`pick1-v1`, 30 episodes); PICK_DOUBLE's day 2 was not used for the region.
- The kinematic recomputation uses the Panda's nominal DH parameters; the real arm's calibration differs by
  millimetres.
- Sim clearances use the sim's collision geometry of the rig and fin-ray fingers.
- No policy was run from this pose; see "Sim success is not real success".

## Open questions

1. Does the real arm reach the pose with the probe lines above and a 16.0 cm ruler height? Answer: the checklist,
   steps 1-2.
2. Does any checkpoint grasp from this pose? Answer: on the arm, or in the sim after the camera study
   ([../02_camera_study/README.md](../02_camera_study/README.md) A2, C2).
3. Should final-deployment's `control_deploy/DIRECTIONS.md` step 10 table say "pointing to the robot's left" and the
   eraser 1.9 cm right of the fingertips instead of "pointing away" and "centred"? Answer: the user, after step 1.
4. Do the rig's cables tolerate the 132.7 deg roll of joint 7 on every `p` and `h`? Answer: step 1 on the arm.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| chosen pose (sim default) | `sim_deploy/stations/start_pose.yaml` |
| pose files | `sim_evals/_start_pose_sweep/poses/leftx55_t48_h16.yaml`, `poses/leftx55_t48_h22.yaml` |
| sweep checks incl. real PC | `sim_evals/_start_pose_sweep/candidates.json` (10:15), `candidates.md` |
| demo paths / gate | `sim_evals/_start_pose_sweep/demo_paths.json`, `gate/gate_h16.json`, `gate/gate_h22.json` |
| policy evals so far | `sim_evals/index.csv`, `sim_evals/_start_pose_sweep/summary.csv` |
| eraser region | `sim_deploy/scene/randomization_start.yaml` |
| rule code | `deployment/control_deploy/franka_control/src/{supervisor,safety_gate,homing_motion}.cpp`, `deployment/policy_deploy/policy_deploy/manager/pc_probe.py`, `.../frames/probe.py` |
| real station (read-only, outside the repo) | `~/Capstone-final-deployment/deployment/control_deploy/franka_control/config.yaml`, `deployment/contracts/frames.yaml`, `deployment/control_deploy/station.yaml` @ `447910b` |
| copied into this folder | `data/p5_start_pose.yaml` (`sim_deploy/stations/start_pose.yaml`), `data/p5_leftx55_t48_h16.yaml`, `data/p5_leftx55_t48_h22.yaml` (`sim_evals/_start_pose_sweep/poses/`), `data/p5_candidates_leftx55_t48.json` (the two `candidates.json` and `demo_paths.json` entries), `data/p5_recompute.py` + `data/p5_recompute.txt` (the independent recomputation and its output) |
