# 01_start_pose / 2: Candidate start poses and the feasibility envelope

**Status:** done · **Date:** 2026-11-20 · **Who:** agent ·
**Plan ref:** sim_deploy/PLAN.md §9 (step P4), DESIGN.md "P3/P4: start-pose checks and sweep" -> "### P4 candidates and
the envelope of q_start_rad"

## Question

Of a grid of candidate start poses over the demos' range (tilt, fingertip height, heading, eraser layout), how many are
**feasible** (pass every offline sim check and the real PC's `p` rules), why do the others fail, and how do the 30
PICK_TEST1 demo paths behave when carried from each feasible pose?

## Why it matters for the real robot

`homing.q_start_rad` on the real PC is still `null` (to be measured by hand-guiding). This part gives the region the
user should hand-guide into and the regions to avoid: where the IMX519 does not see the near eraser positions, where
joint 7 or joint 4/5 sit at their limit, where the p move passes close to the robot's own links, and where the demos'
retreat leaves the real box at x 0.20. The chosen pose `leftx55_t48_h16` (the sim default and the recommendation for
the real arm, see [5_real_pc_handoff.md](5_real_pc_handoff.md)) comes from this grid. The joint-limit, reach and
clearance results are pure kinematics of the Panda + rig model and carry over to the real arm as far as the model's
geometry is right; the camera-view check depends on the sim's camera models.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Model | `sim_deploy/build/sim_rig/model.json` `model_sha256` `166a0649571a…` (P3/P4: table top at O z 0, x -0.40..1.00 m, y -0.80..0.80 m, no pedestal; TCP 235.65 mm). Model joint ranges = the Panda limits (`scene.xml`, `start_pose.PANDA_Q_MIN/MAX`). |
| Home / start pose | no episode is run. Every candidate's p move starts at `q_home_rad [0.0, -0.7853982, 0.0, -2.3561945, 0.0, 1.5707963, 0.7853982]` (`deployment/control_deploy/franka_control/config.yaml`). Orientation reference: the P3 stand-in `STAND_IN_Z0` `[-0.6419329, 0.3706554, -0.1347388, -1.8275169, 0.5918946, 1.4884701, -1.6761742]` (the E7 trainstart stand-in re-solved for the table at z 0; see [1_p_key_and_standins.md](1_p_key_and_standins.md)) |
| Eraser placement | `sim_deploy/scene/randomization_start.yaml` (PICK_TEST1 pick1-v1, 30 demos): region ahead 0.09..0.217 m, left -0.035..-0.003 m, yaw_rel -6.6..+0.8 deg; median ahead 0.159 m, left -0.019 m, yaw_rel -2.2 deg |
| Stations / overlays | `sim_deploy/stations/control_sim.yaml` over the copy's `config.yaml`: sim box [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70] m, cone axis (0, 0, -1), cone_max 60 deg, homing 0.3 rad/s, min 2.0 s, timeout 30 s. No overlay. |
| Policy / checkpoint | none (offline kinematics only) |
| Inference settings | n/a |
| Seeds / episodes | IK seeds: `q_home`, `STAND_IN_Z0`, 22 random `clip(q_home + N(0, 0.6 rad))` from `np.random.default_rng(0)`; demo paths: PICK_TEST1 `pick1-v1.zarr.zip` episodes 0..29 (60 Hz rows) |
| Timeout | n/a |
| Sim-only settings | the sim box (its values are the user's real-station box of 2026-11-20, which current upstream final-deployment also has since 447910b; only the ea9f7fb copy in this repo still has [0.25, -0.40, 0.05]); TCP 235.65 mm (current upstream since 927a5ba; the ea9f7fb copy says 150 mm: both TCPs are checked for the real PC) |

## Method

All steps are offline (MuJoCo kinematics and distance queries, no physics episode, no ports, no GPU).

**1. The grid** (`sim_deploy/scene/start_sweep.py candidates`): tilt {38, 48, 58} deg x fingertip height {10, 16, 22}
cm above the table x heading {left (+y), away (+x), right (-y), away_left, away_right} = 45 poses, ids
`<heading>_t<tilt>_h<height>`. Then the left heading with the layout moved out
(`sim_evals/_start_pose_sweep/tools/left_out.py`): the same 3 x 3 tilt x height with G (0.50, 0) and (0.55, 0) m,
ids `leftx50_*` and `leftx55_*` = 18 poses. 63 in all.

- **Orientation** (`target_pose`): `R = Rz(heading) · Ry(tilt0 - tilt) · Rz(-heading0) · R0`, with R0, tilt0 and
  heading0 the TCP orientation, tilt and heading of `STAND_IN_Z0` (the demos' mean start orientation, closing axis on
  the table). Headings: left 90 deg, away 0 deg, right -90 deg, away_left 45 deg, away_right -45 deg.
- **Position**: the demos' median eraser centre (15.9 cm ahead, 1.9 cm right of the fingertips) is put on a fixed
  point G of the table (`GRASP_CENTRE`, verified in the code): left and right (0.45, 0) m (the stand-in's layout),
  away_left and away_right (0.50, 0) m, away **(0.60, 0) m**; the shifted left layouts (0.50, 0) and (0.55, 0) m.
  TCP xy = G - 0.159 m · ahead - (-0.019 m) · left; TCP z = table (0) + height. So the TCP of a left pose at G 0.45 is
  at O (0.431, -0.159), of an away pose at O (0.441, 0.019).
- **IK** (`solve` -> `trainstart.solve_ik`): damped least squares on joints 1-7 (lambda 1e-4, step clip 0.1 rad,
  null-space pull to the seed 0.05, 400 iterations), 24 seeds (above); a solution counts if the position error is
  <= 1e-4 m and the angle error <= 0.05 deg; among those the one with the lowest
  `||q - q_home||(joints 1-6) - 0.5 · min(joint margin, 0.3 rad)` wins.

**2. The checks** (`evaluate`; a pose is feasible when all pass, the failing ones are listed in `fails`):

| Check | Code | Threshold |
|---|---|---|
| Panda joint limits | `start_pose.describe` `panda_joint_limits` | every joint inside `PANDA_Q_MIN/MAX` (j1/3/5/7 ±2.8973, j2 ±1.7628, j4 -3.0718..-0.0698, j6 -0.0175..3.7525 rad) |
| model joint limits | `describe` `model_joint_limits` | model margin > 0 rad |
| sim box | `describe` `box` | TCP inside [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70] m, no margin |
| cone | `describe` `cone` | TCP z within 60 deg of (0, 0, -1), no margin |
| contact | `describe` `no_contact` | no contact (except with the eraser) at the pose |
| heading | `describe` `heading` | horizontal part of the fingertip direction >= 0.035 (else the eraser cannot be placed ahead) |
| move time | `describe` `move_time` | `max(2.0 s, 1.875 · max abs(q - q_home) / 0.3 rad/s)` < 30 s |
| joint margin | `JOINT_MARGIN_MIN` | min distance to the Panda limits >= **0.10 rad** |
| region | `Scene.check_region` (home start) | the 8 corners of the region (ahead x left x yaw_rel extremes, before clipping) and the median, 9 placements: eraser footprint on the table, centre and footprint in the sim box, all 8 eraser corners inside the IMX219 **and** the IMX519 pinhole image (margin 0 px, no occlusion test) at the pose |
| p path | `path_check` | linear joint interpolation home -> q (HomingMotion's path), **121 samples**: no contact (except the eraser, parked away); clearance >= **0.02 m** from links 2-7, hand, fingers, rig to the table, and from links 6-7, hand, fingers, rig to links 0-3 (`mj_geomDistance`, capped at 0.5 m) |
| real PC (`real_pc`) | `p_accepted_real_tcp`, `p_accepted_fingertip_tcp` | the same q with TCP 150 mm and 235.65 mm: TCP in the real box (`REAL_BOX` [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70] m, no margin), tilt <= 60 deg (no margin), joint margin > 0, move time `max(2.0 s, 1.875 · max abs(dq) / 0.3 rad/s)` < 30 s; plus the real box margin and what `pc_probe --start-pose` would print. Not part of `ok`/`fails`: only reported. |

**3. The demo paths** (`sim_evals/_start_pose_sweep/tools/demo_paths.py`, every feasible pose in candidates.json):
each of the 30 PICK_TEST1 recorded TCP paths applied relative to the start pose, `W(t) = O_T_start · inv(P_0) · P_t`
(as the replay and the policy's relative actions do). Counted per pose: the **pick** (start to the grasp row + 90
rows = 1.5 s at 60 Hz; grasp row = `grasp_rows(widths)["start"]`, the half-closing row) inside the real box enlarged by
the gate's 10 mm pull-back margin ([0.19, -0.41, -0.015]..[0.81, 0.41, 0.71] m) and tilt <= 62 deg (cone 60 + 2 deg
margin); the same for the whole path; per axis, the number of demos that leave the box on that side; IK of the grasp
row and the last row seeded from the start q (counted reachable with position error <= 1e-3 m and angle error <= 0.5
deg), and the minimum joint margin over those two poses (model joint ranges = Panda limits).

Pass criteria: a pose is feasible when `fails` is empty. The demo-path counts are not pass/fail; they rank the
feasible poses (higher = better) and fed the choice of `leftx55_t48_h16` together with
[4_gate_check.md](4_gate_check.md).

## Commands to reproduce

Not re-run for this record (other agents run the sim now). All three write into `sim_evals/_start_pose_sweep/`; none
binds a port or uses the GPU. Run them in this order: `candidates` rewrites `candidates.json` with the 45 grid rows
only (dropping the shifted rows) and the `poses/<id>.yaml` files; `left_out.py` adds the 18 shifted rows back and
rewrites `candidates.md`; `demo_paths.py` reads the feasible rows and writes `demo_paths.json`.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# 1. the 45-pose grid, IK, checks -> sim_evals/_start_pose_sweep/candidates.{json,md}, poses/<id>.yaml
#    (~35 s: pose files written 09:56:12..09:56:47 on 2026-11-20; prints one OK/FAIL line per pose,
#    then "<n> of 45 feasible")
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep candidates
# 2. the shifted left layouts leftx50_* / leftx55_* -> appended to candidates.json, candidates.md rewritten,
#    poses/leftx*.yaml (~21 s: 10:14:48..10:15:09; prints "<id> <ok> <fails> <joint margin>" per pose, not saved)
~/envs/sim_deploy/bin/python sim_evals/_start_pose_sweep/tools/left_out.py
# 3. the 30 demo paths carried from every feasible pose -> demo_paths.json
#    (needs ~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip; <= 2 min: candidates.json 10:15:09,
#    demo_paths.json 10:17:09; prints one line per pose, not saved)
~/envs/sim_deploy/bin/python sim_evals/_start_pose_sweep/tools/demo_paths.py
# read-only views of the result
column -s '|' -t < sim_evals/_start_pose_sweep/candidates.md | less -S
```

A re-run today would not reproduce the stored real-PC box margins of the first 45 rows: they were computed with the
older `REAL_BOX` (see Analysis, "Docs vs raw data" 1); every pass/fail would stay the same.

## Results

### Feasible counts (`sim_evals/_start_pose_sweep/candidates.json`)

**Feasible: 40 of 63** (26 of the first 45, 14 of the 18 shifted): the DESIGN numbers are confirmed.

| Heading (G, m) | feasible / 9 | Not feasible |
|---|---|---|
| left (0.45, 0) | 7 | t38_h10, t38_h16 |
| away (0.60, 0) | 1 | all but t48_h16 |
| right (0.45, 0) | 4 | t38_h10, t38_h16, t48_h16, t48_h22, t58_h22 |
| away_left (0.50, 0) | 7 | t38_h10, t38_h16 |
| away_right (0.50, 0) | 7 | t38_h10, t38_h16 |
| leftx50 (0.50, 0) | 7 | t38_h10, t38_h16 |
| leftx55 (0.55, 0) | 7 | t38_h10, t38_h16 |
| **all** | **40 / 63** | 23 |

Every candidate with an IK solution (61 of 63) passes the real PC's `p` rules with both TCPs; the two without one
(away_t48_h22, away_t58_h10) have no q to check. Every pose reaches its target tilt and height (IK position
error <= 1.0e-4 m, angle error <= 0.0025 deg). The cone margin is 22.0 / 12.0 / 2.0 deg at tilt 38 / 48 / 58 deg. No
p path has a contact. Move time from home on the real PC (0.3 rad/s): 7.4-23.0 s (< 30 s everywhere).

### Every pose

Real box margins in this table are recomputed from the stored TCP positions (`real_pc.tcp_150mm.tcp_O_m`,
`real_pc.tcp_235mm.tcp_O_m`) against the current real box [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70] m; the stored
values of the first 45 rows are against the older ea9f7fb box (Analysis 1). Columns: joint margin to the Panda limits
and the joint that sets it; sim box margin of the fingertip TCP; cone margin; p path minimum clearance to the table and
between the far bodies (links 6-7, hand, fingers, rig) and links 0-3; IMX519 view: how many of the 9 region
placements fall outside the IMX519 image (no other region test fails anywhere). Same data with more columns (q, IK
error, move times, where the minimum clearances are):
[data/p2_candidates_compact.csv](data/p2_candidates_compact.csv).

| id | feasible | TCP O (m) | joint margin (rad, joint) | sim box margin (m) | cone margin (deg) | p path: table / self clearance (cm) | real box margin, current box: 150 mm / fingertip TCP (m) | IMX519 corners out (of 9) | fails |
|---|---|---|---|---|---|---|---|---|---|
| left_t38_h10 | no | (0.431, -0.159, 0.100) | 0.993 (j4) | 0.105 | 22.0 | 8.9 / 24.0 | 0.173 (z) / 0.105 (z) | 4 | IMX519 view |
| left_t38_h16 | no | (0.431, -0.159, 0.160) | 1.036 (j4) | 0.165 | 22.0 | 14.9 / 24.0 | 0.188 (y) / 0.165 (z) | 3 | IMX519 view |
| left_t38_h22 | yes | (0.431, -0.159, 0.220) | 1.106 (j4) | 0.225 | 22.0 | 20.9 / 24.0 | 0.188 (y) / 0.225 (z) | 0 | - |
| left_t48_h10 | yes | (0.431, -0.159, 0.100) | 1.125 (j4) | 0.105 | 12.0 | 8.7 / 24.0 | 0.162 (z) / 0.105 (z) | 0 | - |
| left_t48_h16 | yes | (0.431, -0.159, 0.160) | 1.154 (j4) | 0.165 | 12.0 | 14.7 / 24.0 | 0.177 (y) / 0.165 (z) | 0 | - |
| left_t48_h22 | yes | (0.431, -0.159, 0.220) | 1.210 (j4) | 0.225 | 12.0 | 20.7 / 23.9 | 0.177 (y) / 0.225 (z) | 0 | - |
| left_t58_h10 | yes | (0.431, -0.159, 0.100) | 1.125 (j7) | 0.105 | 2.0 | 8.5 / 22.9 | 0.150 (z) / 0.105 (z) | 0 | - |
| left_t58_h16 | yes | (0.431, -0.159, 0.160) | 1.186 (j7) | 0.165 | 2.0 | 14.5 / 22.4 | 0.168 (y) / 0.165 (z) | 0 | - |
| left_t58_h22 | yes | (0.431, -0.159, 0.220) | 1.253 (j7) | 0.225 | 2.0 | 20.5 / 22.1 | 0.168 (y) / 0.225 (z) | 0 | - |
| away_t38_h10 | no | (0.441, 0.019, 0.100) | 0.088 (j4) | 0.105 | 22.0 | 8.9 / 4.7 | 0.173 (z) / 0.105 (z) | 4 | joint margin < 0.10 rad (j4); IMX519 view |
| away_t38_h16 | no | (0.441, 0.019, 0.160) | 0.020 (j4) | 0.165 | 22.0 | 14.9 / 6.4 | 0.188 (x) / 0.165 (z) | 3 | joint margin < 0.10 rad (j4); IMX519 view |
| away_t38_h22 | no | (0.441, 0.019, 0.220) | 0.014 (j4) | 0.225 | 22.0 | 20.9 / 7.8 | 0.188 (x) / 0.225 (z) | 0 | joint margin < 0.10 rad (j4) |
| away_t48_h10 | no | (0.441, 0.019, 0.100) | 0.082 (j5) | 0.105 | 12.0 | 8.7 / 3.1 | 0.162 (z) / 0.105 (z) | 0 | joint margin < 0.10 rad (j5) |
| away_t48_h16 | yes | (0.441, 0.019, 0.160) | 0.120 (j4) | 0.165 | 12.0 | 14.7 / 7.0 | 0.177 (x) / 0.165 (z) | 0 | - |
| away_t48_h22 | no | target (0.441, 0.019, 0.22) | - | - | - | - | - | - | no IK solution |
| away_t58_h10 | no | target (0.441, 0.019, 0.1) | - | - | - | - | - | - | no IK solution |
| away_t58_h16 | no | (0.441, 0.019, 0.160) | 0.004 (j5) | 0.165 | 2.0 | 14.5 / 0.7 | 0.168 (x) / 0.165 (z) | 0 | joint margin < 0.10 rad (j5); p path self < 2 cm |
| away_t58_h22 | no | (0.441, 0.019, 0.220) | 0.046 (j4) | 0.225 | 2.0 | 20.5 / 4.6 | 0.168 (x) / 0.225 (z) | 0 | joint margin < 0.10 rad (j4) |
| right_t38_h10 | no | (0.469, 0.159, 0.100) | 0.073 (j7) | 0.105 | 22.0 | 8.9 / 23.6 | 0.173 (z) / 0.105 (z) | 4 | joint margin < 0.10 rad (j7); IMX519 view |
| right_t38_h16 | no | (0.469, 0.159, 0.160) | 0.012 (j7) | 0.165 | 22.0 | 14.9 / 23.3 | 0.188 (y) / 0.165 (z) | 3 | joint margin < 0.10 rad (j7); IMX519 view |
| right_t38_h22 | yes | (0.469, 0.159, 0.220) | 0.372 (j7) | 0.225 | 22.0 | 20.9 / 23.6 | 0.188 (y) / 0.225 (z) | 0 | - |
| right_t48_h10 | yes | (0.469, 0.159, 0.100) | 0.148 (j7) | 0.105 | 12.0 | 8.7 / 23.4 | 0.162 (z) / 0.105 (z) | 0 | - |
| right_t48_h16 | no | (0.469, 0.159, 0.160) | 0.077 (j7) | 0.165 | 12.0 | 14.7 / 22.4 | 0.177 (y) / 0.165 (z) | 0 | joint margin < 0.10 rad (j7) |
| right_t48_h22 | no | (0.469, 0.159, 0.220) | 0.003 (j7) | 0.225 | 12.0 | 20.7 / 21.3 | 0.177 (y) / 0.225 (z) | 0 | joint margin < 0.10 rad (j7) |
| right_t58_h10 | yes | (0.469, 0.159, 0.100) | 0.181 (j7) | 0.105 | 2.0 | 8.5 / 21.5 | 0.150 (z) / 0.105 (z) | 0 | - |
| right_t58_h16 | yes | (0.469, 0.159, 0.160) | 0.123 (j7) | 0.165 | 2.0 | 14.5 / 20.6 | 0.168 (y) / 0.165 (z) | 0 | - |
| right_t58_h22 | no | (0.469, 0.159, 0.220) | 0.051 (j7) | 0.225 | 2.0 | 20.5 / 19.9 | 0.168 (y) / 0.225 (z) | 0 | joint margin < 0.10 rad (j7) |
| away_left_t38_h10 | no | (0.374, -0.099, 0.100) | 0.200 (j4) | 0.105 | 22.0 | 8.9 / 13.5 | 0.137 (x) / 0.105 (z) | 4 | IMX519 view |
| away_left_t38_h16 | no | (0.374, -0.099, 0.160) | 0.223 (j4) | 0.165 | 22.0 | 14.9 / 14.2 | 0.137 (x) / 0.165 (z) | 3 | IMX519 view |
| away_left_t38_h22 | yes | (0.374, -0.099, 0.220) | 0.294 (j4) | 0.174 | 22.0 | 20.9 / 16.6 | 0.137 (x) / 0.174 (x) | 0 | - |
| away_left_t48_h10 | yes | (0.374, -0.099, 0.100) | 0.236 (j4) | 0.105 | 12.0 | 8.7 / 14.2 | 0.129 (x) / 0.105 (z) | 0 | - |
| away_left_t48_h16 | yes | (0.374, -0.099, 0.160) | 0.230 (j4) | 0.165 | 12.0 | 14.7 / 13.5 | 0.129 (x) / 0.165 (z) | 0 | - |
| away_left_t48_h22 | yes | (0.374, -0.099, 0.220) | 0.273 (j4) | 0.174 | 12.0 | 20.7 / 15.1 | 0.129 (x) / 0.174 (x) | 0 | - |
| away_left_t58_h10 | yes | (0.374, -0.099, 0.100) | 0.295 (j4) | 0.105 | 2.0 | 8.5 / 15.9 | 0.123 (x) / 0.105 (z) | 0 | - |
| away_left_t58_h16 | yes | (0.374, -0.099, 0.160) | 0.267 (j4) | 0.165 | 2.0 | 14.5 / 14.1 | 0.123 (x) / 0.165 (z) | 0 | - |
| away_left_t58_h22 | yes | (0.374, -0.099, 0.220) | 0.276 (j4) | 0.174 | 2.0 | 20.5 / 14.1 | 0.123 (x) / 0.174 (x) | 0 | - |
| away_right_t38_h10 | no | (0.401, 0.126, 0.100) | 0.269 (j4) | 0.105 | 22.0 | 8.9 / 11.5 | 0.164 (x) / 0.105 (z) | 4 | IMX519 view |
| away_right_t38_h16 | no | (0.401, 0.126, 0.160) | 0.167 (j7) | 0.165 | 22.0 | 14.9 / 17.2 | 0.164 (x) / 0.165 (z) | 3 | IMX519 view |
| away_right_t38_h22 | yes | (0.401, 0.126, 0.220) | 0.349 (j7) | 0.201 | 22.0 | 20.9 / 18.8 | 0.164 (x) / 0.201 (x) | 0 | - |
| away_right_t48_h10 | yes | (0.401, 0.126, 0.100) | 0.265 (j7) | 0.105 | 12.0 | 8.7 / 11.2 | 0.156 (x) / 0.105 (z) | 0 | - |
| away_right_t48_h16 | yes | (0.401, 0.126, 0.160) | 0.285 (j7) | 0.165 | 12.0 | 14.7 / 12.5 | 0.156 (x) / 0.165 (z) | 0 | - |
| away_right_t48_h22 | yes | (0.401, 0.126, 0.220) | 0.142 (j7) | 0.201 | 12.0 | 20.7 / 17.3 | 0.156 (x) / 0.201 (x) | 0 | - |
| away_right_t58_h10 | yes | (0.401, 0.126, 0.100) | 0.312 (j4) | 0.105 | 2.0 | 8.5 / 12.3 | 0.150 (x) / 0.105 (z) | 0 | - |
| away_right_t58_h16 | yes | (0.401, 0.126, 0.160) | 0.235 (j7) | 0.165 | 2.0 | 14.5 / 12.3 | 0.150 (x) / 0.165 (z) | 0 | - |
| away_right_t58_h22 | yes | (0.401, 0.126, 0.220) | 0.252 (j7) | 0.201 | 2.0 | 20.5 / 13.6 | 0.150 (x) / 0.201 (x) | 0 | - |
| leftx50_t48_h16 | yes | (0.481, -0.159, 0.160) | 1.298 (j7) | 0.165 | 12.0 | 14.7 / 24.0 | 0.177 (y) / 0.165 (z) | 0 | - |
| leftx50_t48_h22 | yes | (0.481, -0.159, 0.220) | 1.358 (j7) | 0.225 | 12.0 | 20.7 / 24.0 | 0.177 (y) / 0.225 (z) | 0 | - |
| leftx50_t48_h10 | yes | (0.481, -0.159, 0.100) | 1.236 (j7) | 0.105 | 12.0 | 8.7 / 24.0 | 0.162 (z) / 0.105 (z) | 0 | - |
| leftx50_t58_h16 | yes | (0.481, -0.159, 0.160) | 1.212 (j2) | 0.165 | 2.0 | 14.5 / 23.1 | 0.168 (y) / 0.165 (z) | 0 | - |
| leftx50_t58_h22 | yes | (0.481, -0.159, 0.220) | 1.285 (j2) | 0.225 | 2.0 | 20.5 / 23.1 | 0.168 (y) / 0.225 (z) | 0 | - |
| leftx50_t58_h10 | yes | (0.481, -0.159, 0.100) | 1.121 (j2) | 0.105 | 2.0 | 8.5 / 23.5 | 0.150 (z) / 0.105 (z) | 0 | - |
| leftx50_t38_h16 | no | (0.481, -0.159, 0.160) | 1.185 (j4) | 0.165 | 22.0 | 14.9 / 24.0 | 0.188 (y) / 0.165 (z) | 3 | IMX519 view |
| leftx50_t38_h22 | yes | (0.481, -0.159, 0.220) | 1.255 (j4) | 0.225 | 22.0 | 20.9 / 24.0 | 0.188 (y) / 0.225 (z) | 0 | - |
| leftx50_t38_h10 | no | (0.481, -0.159, 0.100) | 1.141 (j4) | 0.105 | 22.0 | 8.9 / 24.0 | 0.173 (z) / 0.105 (z) | 4 | IMX519 view |
| leftx55_t48_h16 | yes | (0.531, -0.159, 0.160) | 1.241 (j2) | 0.165 | 12.0 | 14.7 / 24.0 | 0.177 (y) / 0.165 (z) | 0 | - |
| leftx55_t48_h22 | yes | (0.531, -0.159, 0.220) | 1.299 (j2) | 0.225 | 12.0 | 20.7 / 24.0 | 0.177 (y) / 0.225 (z) | 0 | - |
| leftx55_t48_h10 | yes | (0.531, -0.159, 0.100) | 1.162 (j2) | 0.105 | 12.0 | 8.7 / 24.0 | 0.162 (z) / 0.105 (z) | 0 | - |
| leftx55_t58_h16 | yes | (0.531, -0.159, 0.160) | 1.098 (j2) | 0.165 | 2.0 | 14.5 / 24.0 | 0.168 (y) / 0.165 (z) | 0 | - |
| leftx55_t58_h22 | yes | (0.531, -0.159, 0.220) | 1.165 (j2) | 0.225 | 2.0 | 20.5 / 24.0 | 0.168 (y) / 0.225 (z) | 0 | - |
| leftx55_t58_h10 | yes | (0.531, -0.159, 0.100) | 1.012 (j2) | 0.105 | 2.0 | 8.5 / 24.0 | 0.150 (z) / 0.105 (z) | 0 | - |
| leftx55_t38_h16 | no | (0.531, -0.159, 0.160) | 1.355 (j4) | 0.165 | 22.0 | 14.9 / 24.0 | 0.188 (y) / 0.165 (z) | 3 | IMX519 view |
| leftx55_t38_h22 | yes | (0.531, -0.159, 0.220) | 1.424 (j2) | 0.225 | 22.0 | 20.9 / 24.0 | 0.188 (y) / 0.225 (z) | 0 | - |
| leftx55_t38_h10 | no | (0.531, -0.159, 0.100) | 1.298 (j2) | 0.105 | 22.0 | 8.9 / 24.0 | 0.173 (z) / 0.105 (z) | 4 | IMX519 view |

### Why poses fail: reason x where (from `fails` in candidates.json)

Each failing pose can have several reasons; 23 poses fail with 28 reasons.

| Reason (code text) | Poses | Where |
|---|---|---|
| `region:in_view_imx519` | 14 | tilt 38 deg at 10 and 16 cm, every heading and layout (7 x 2); 4 of 9 placements out at 10 cm, 3 of 9 at 16 cm; none at 22 cm |
| `joint_margin<0.1 (joint 7)` | 5 | right: t38_h10 0.073, t38_h16 0.012, t48_h16 0.077, t48_h22 0.003, t58_h22 0.051 rad |
| `joint_margin<0.1 (joint 4)` | 4 | away: t38_h10 0.088, t38_h16 0.020, t38_h22 0.014, t58_h22 0.046 rad |
| `joint_margin<0.1 (joint 5)` | 2 | away: t48_h10 0.082, t58_h16 0.004 rad |
| `ik: no solution (reach / joint limits)` | 2 | away: t48_h22, t58_h10 |
| `path:self` | 1 | away_t58_h16: 0.68 cm between link6 and link2 at the end of the p move |

The same 23 poses by combination: IMX519 view only 10 (left, away_left, away_right, leftx50, leftx55 at t38_h10 /
t38_h16); joint margin + IMX519 view 4 (away and right at t38_h10 / t38_h16); joint margin only 6 (away_t38_h22,
away_t48_h10, away_t58_h22, right_t48_h16, right_t48_h22, right_t58_h22); joint margin + p path self 1
(away_t58_h16); no IK 2.

The 4 shifted poses that fail are leftx50_t38_h10, leftx50_t38_h16, leftx55_t38_h10, leftx55_t38_h16, all on the
IMX519 view only (4 / 3 of 9 placements out, as at G 0.45).

Per heading, over the poses with an IK solution:

| Heading | joint margin (rad) | p path self clearance (cm) | p path table clearance |
|---|---|---|---|
| left (G 0.45) | 0.993-1.253 | 22.07-23.97 | the start height minus 1.06-1.46 cm (a finger, at the end of the move), the same for every heading |
| leftx50 | 1.121-1.358 | 23.05-23.97 | same |
| leftx55 | 1.012-1.424 | 23.97 | same |
| away_left | 0.200-0.295 | 13.51-16.64 | same |
| away_right | 0.142-0.349 | 11.24-18.83 | same |
| right | 0.003-0.372 | 19.85-23.64 | same |
| away | 0.004-0.120 | 0.68-7.81 | same |

Real box margin (current box) over all 61 poses with an IK solution: >= 0.123 m with the 150 mm TCP (away_left at 58
deg, axis x), >= 0.105 m with the fingertip TCP (every 10 cm pose, axis z). As stored in candidates.json (ea9f7fb box
for the first 45 rows): >= 0.0728 m and >= 0.05 m.

The chosen pose `leftx55_t48_h16`: TCP O (0.531, -0.159, 0.160) m, tilt 48.0 deg, joint margin 1.241 rad (joint 2),
sim box margin 0.165 m (z), cone margin 12.0 deg, p path 14.72 cm to the table / 23.97 cm to links 0-3, sim move
time 14.5 s, real box margin 0.1774 m (y, 150 mm TCP) / 0.165 m (z, fingertip TCP).

### The demo paths carried from each feasible pose (`sim_evals/_start_pose_sweep/demo_paths.json`)

40 entries = the 40 feasible poses (demo_paths.py ran at 10:17, after left_out.py at 10:15, so the shifted layouts are
included). Box = the real box with the gate's 10 mm margin, so "x < 0.19 m" is the robot-side face x 0.20 m minus 10
mm and "z < -0.015 m" is the table top (z -0.005 floor) minus 10 mm. Each count is a number of demos of 30. Also in
[data/p2_demo_paths.csv](data/p2_demo_paths.csv).

| id | pick in box + cone (of 30) | whole path in box + cone | x < 0.19 m (retreat) | x > 0.81 m | z < -0.015 m (into the table) | tilt > 62 deg | grasp + end reachable (of 30) | joint margin min / p10 (rad) |
|---|---|---|---|---|---|---|---|---|
| left_t38_h22 | 20 | 4 | 21 | 0 | 10 | 0 | 30 | 0.321 / 0.475 |
| left_t48_h10 | 4 | 1 | 21 | 0 | 26 | 0 | 30 | 0.25 / 0.42 |
| left_t48_h16 | 16 | 3 | 21 | 0 | 14 | 0 | 30 | 0.32 / 0.48 |
| left_t48_h22 | 24 | 7 | 21 | 0 | 6 | 0 | 30 | 0.421 / 0.575 |
| left_t58_h10 | 8 | 1 | 21 | 0 | 21 | 9 | 30 | 0.386 / 0.552 |
| left_t58_h16 | 18 | 4 | 21 | 0 | 12 | 9 | 30 | 0.438 / 0.599 |
| left_t58_h22 | 27 | 7 | 21 | 0 | 2 | 9 | 30 | 0.521 / 0.643 |
| away_t48_h16 | 16 | 16 | 0 | 0 | 14 | 0 | 23 | 0.001 / 0.043 |
| right_t38_h22 | 20 | 17 | 0 | 3 | 10 | 0 | 27 | 0.427 / 0.451 |
| right_t48_h10 | 4 | 3 | 0 | 3 | 26 | 0 | 7 | 0.015 / 0.035 |
| right_t58_h10 | 8 | 4 | 0 | 3 | 21 | 9 | 8 | 0.015 / 0.055 |
| right_t58_h16 | 18 | 10 | 0 | 3 | 12 | 9 | 7 | 0.001 / 0.01 |
| away_left_t38_h22 | 20 | 13 | 7 | 0 | 10 | 0 | 21 | 0.021 / 0.035 |
| away_left_t48_h10 | 4 | 1 | 8 | 0 | 26 | 0 | 11 | 0.017 / 0.044 |
| away_left_t48_h16 | 16 | 8 | 8 | 0 | 14 | 0 | 11 | 0.085 / 0.098 |
| away_left_t48_h22 | 24 | 16 | 8 | 0 | 6 | 0 | 17 | 0.017 / 0.066 |
| away_left_t58_h10 | 8 | 2 | 8 | 0 | 21 | 9 | 11 | 0.042 / 0.06 |
| away_left_t58_h16 | 18 | 7 | 8 | 0 | 12 | 9 | 11 | 0.064 / 0.106 |
| away_left_t58_h22 | 27 | 15 | 8 | 0 | 2 | 9 | 15 | 0.053 / 0.059 |
| away_right_t38_h22 | 20 | 20 | 0 | 0 | 10 | 0 | 20 | 0.001 / 0.035 |
| away_right_t48_h10 | 4 | 4 | 0 | 0 | 26 | 0 | 29 | 0.018 / 0.157 |
| away_right_t48_h16 | 16 | 16 | 0 | 0 | 14 | 0 | 29 | 0.051 / 0.198 |
| away_right_t48_h22 | 24 | 24 | 0 | 0 | 6 | 0 | 8 | 0.001 / 0.001 |
| away_right_t58_h10 | 8 | 5 | 0 | 0 | 21 | 9 | 29 | 0.033 / 0.175 |
| away_right_t58_h16 | 18 | 12 | 0 | 0 | 12 | 9 | 29 | 0.038 / 0.178 |
| away_right_t58_h22 | 27 | 20 | 0 | 0 | 2 | 9 | 29 | 0.09 / 0.229 |
| leftx50_t48_h16 | 16 | 9 | 10 | 0 | 14 | 0 | 30 | 0.389 / 0.565 |
| leftx50_t48_h22 | 24 | 16 | 10 | 0 | 6 | 0 | 30 | 0.483 / 0.649 |
| leftx50_t48_h10 | 4 | 2 | 10 | 0 | 26 | 0 | 30 | 0.325 / 0.524 |
| leftx50_t58_h16 | 18 | 8 | 10 | 0 | 12 | 9 | 30 | 0.505 / 0.677 |
| leftx50_t58_h22 | 27 | 15 | 10 | 0 | 2 | 9 | 30 | 0.583 / 0.75 |
| leftx50_t58_h10 | 8 | 4 | 10 | 0 | 21 | 9 | 30 | 0.456 / 0.644 |
| leftx50_t38_h22 | 20 | 12 | 10 | 0 | 10 | 0 | 30 | 0.384 / 0.55 |
| leftx55_t48_h16 | 16 | 13 | 3 | 0 | 14 | 0 | 30 | 0.474 / 0.679 |
| leftx55_t48_h22 | 24 | 21 | 3 | 0 | 6 | 0 | 30 | 0.562 / 0.745 |
| leftx55_t48_h10 | 4 | 3 | 3 | 0 | 26 | 0 | 30 | 0.416 / 0.641 |
| leftx55_t58_h16 | 18 | 10 | 3 | 0 | 12 | 9 | 30 | 0.587 / 0.776 |
| leftx55_t58_h22 | 27 | 18 | 3 | 0 | 2 | 9 | 30 | 0.661 / 0.824 |
| leftx55_t58_h10 | 8 | 4 | 3 | 0 | 21 | 9 | 30 | 0.543 / 0.73 |
| leftx55_t38_h22 | 20 | 17 | 3 | 0 | 10 | 0 | 30 | 0.465 / 0.655 |

Summary over the feasible poses (the counts that depend only on tilt and height are the same for every heading and
layout; checked over all 40 entries):

| Tilt / height | pick in box + cone | z < -0.015 m (into the table) | tilt > 62 deg |
|---|---|---|---|
| 38 deg / 22 cm | 20 | 10 | 0 |
| 48 deg / 10 cm | 4 | 26 | 0 |
| 48 deg / 16 cm | 16 | 14 | 0 |
| 48 deg / 22 cm | 24 | 6 | 0 |
| 58 deg / 10 cm | 8 | 21 | 9 |
| 58 deg / 16 cm | 18 | 12 | 9 |
| 58 deg / 22 cm | 27 | 2 | 9 |

| Heading (G, m) | x < 0.19 m (retreat) | x > 0.81 m | grasp + end reachable | joint margin min (rad) |
|---|---|---|---|---|
| left (0.45) | 21 (every pose) | 0 | 30 (every pose) | 0.25-0.521 |
| leftx50 (0.50) | 10 | 0 | 30 | 0.325-0.583 |
| leftx55 (0.55) | 3 | 0 | 30 | 0.416-0.661 (0.474 at `leftx55_t48_h16`) |
| away (0.60), 1 pose | 0 | 0 | 23 | 0.001 |
| away_left (0.50) | 7 (t38_h22), 8 (others) | 0 | 11-21 | 0.017-0.085 |
| away_right (0.50) | 0 | 0 | 8-29 | 0.001-0.09 |
| right (0.45), 4 poses | 0 | 3 | 7-27 | 0.001-0.427 |

## Analysis

**The envelope.** Four limits shape where `q_start_rad` can be, each from the raw rows above:

1. *IMX519 view (tilt and height).* At 38 deg tilt and 10 or 16 cm the region's near placements leave the IMX519
   image (4 / 3 of 9) for every heading and layout, so the limit is the camera's downward reach, not the arm. At 22 cm
   (or at 48 deg and more) every placement is in both views. Which placements fail is not stored (only the count); 4
   at 10 cm matches the 4 near corners (ahead 0.09 m x 2 left x 2 yaw) but this is an inference.
2. *Wrist limits for away (heading away from the base).* With the fingertips pointing +x the flange sits between
   the TCP and the base, so joint 4 (elbow) or joint 5 runs to its limit: 2 of 9 have no IK, 7 of the other 8 have a
   joint margin of 0.004-0.088 rad, and the one feasible pose (away_t48_h16) has 0.120 rad and 7.01 cm between link6
   and link2 on the p move; away_t58_h16 comes within 0.68 cm. This is with the eraser already at G 0.60 m; the code
   docstring says closer layouts are worse (no raw rows for them).
3. *Joint 7 for right.* The hand roll the camera-up orientation needs at the robot's right is near joint 7's limit:
   5 of 9 right poses are below 0.10 rad on joint 7 (0.003-0.077 rad), and the 4 feasible ones have only 0.123-0.372
   rad. From those 4 poses the demos' grasp and end poses are reachable in only 7-8 of 30 at 10-16 cm (27 at 38 deg /
   22 cm).
4. *The cone at 58 deg.* Cone margin 2.0 deg at the start; 9 of 30 demos tilt beyond 62 deg when carried from any
   58 deg pose (0 at 38 and 48 deg), so a real start at 58 deg or more would have the gate pulling back or refusing
   the demos' tilting (the demos start at up to 64 deg).

The left heading has none of these limits: joint margin 0.993-1.424 rad, self clearance 22.07-23.97 cm, and every
demo's grasp and end pose reachable from every left pose with >= 0.25 rad. What still separates the left poses is the
demo paths: the start height decides how many carried demos go into the table (26 / 14 / 6 of 30 at 10 / 16 / 22 cm
at 48 deg), and the layout decides how many retreats leave the box at the robot-side face (21 / 10 / 3 of 30 at G 0.45
/ 0.50 / 0.55 m). The diagonal and away headings avoid most of the retreat crossings (away_left still 7-8) but reach the
demos' grasps with 0.001-0.09 rad. `leftx55_t48_h16` sits at the demos' median tilt and height (48.2 deg, 16.3 cm,
`randomization_start.yaml`), with the fewest retreat crossings of the left layouts and 0.474 rad of joint margin on the
demos; the 22 cm alternative `leftx55_t48_h22` carries 6 instead of 14 demos into the table. How the real gate treats
those paths (pull-back vs refusal, v_max) is [4_gate_check.md](4_gate_check.md).

**Why the carried counts do not depend on heading.** The carried path is the start pose times the demo's motion
relative to its own first pose; turning the start pose about the vertical turns the whole path about the vertical, so
z and tilt are unchanged. Tilt does change them, because the relative motion is expressed in the tilted TCP frame:
at 22 cm, 10 / 6 / 2 demos go into the table at 38 / 48 / 58 deg.

### Docs vs raw data

1. Code `sim_deploy/scene/start_sweep.py` (`REAL_BOX`, module docstring "Real PC": box [0.20, -0.40, -0.005]) and
   DESIGN.md "### P4 candidates and the envelope of q_start_rad" ("Every candidate passes the real PC's `p` rules
   (below)", i.e. the user's real station) say the real-PC check uses the current box; raw
   `sim_evals/_start_pose_sweep/candidates.json`: the 43 grid rows with IK (`left_*` .. `away_right_*`) have
   `real_pc` box margins that match the ea9f7fb box [0.25, -0.40, 0.05] and no `upstream_box_ok` key (e.g.
   left_t38_h10 stored 0.1175 m (z) / 0.05 m; current box 0.1725 / 0.105 m); only the 18 `leftx*` rows (computed at
   10:14-10:15, after the code change) use the current box. The same mix is in candidates.md's "real box margin
   150 mm" column. Every pose still passes: the current box contains the ea9f7fb one.
2. DESIGN.md "### P4 candidates ..." fails table, "joint 7 margin < 0.10 rad: heading right at 38/16, 48/16, 48/22,
   58/22"; raw candidates.json: also `right_t38_h10` (0.073 rad, joint 7, together with the IMX519 view), 5 right
   poses in all.
3. DESIGN.md same table, away row: "with the eraser at 0.45-0.50 m from the base axis no solution (joint 4 at
   -3.07)": not in the raw files (candidates.json has the away heading only at G 0.60 m). The code docstring
   (`start_sweep.py`, `GRASP_CENTRE`) says G 0.50: no solution, joint 4 at its limit, 39 mm residual; G 0.40: no
   solution (joints 1, 2, 4); it does not mention 0.45 or -3.07. Raw at G 0.60: 2 of 9 away poses have no IK
   (t48_h22, t58_h10).
4. DESIGN.md same table, left row: "table clearance = the start height"; raw candidates.json `path.table_clearance_m`:
   the start height minus 1.06-1.46 cm (8.54-20.94 cm for 10-22 cm), set by a finger at the end of the move.
5. DESIGN.md "**The demos carried from each pose**": "the pick depends only on the start height (... 14/30 at 16 cm,
   6/30 at 22 cm, 26/30 at 10 cm) and on the tilt (58 deg: 9/30 over the cone)"; raw demo_paths.json: 14 / 6 / 26 are
   the 48 deg values; at 58 deg 12 / 2 / 21 at 16 / 22 / 10 cm, at 38 deg / 22 cm 10: the tilt changes the
   into-the-table count too, not only the cone count. The 9/30 over the cone at 58 deg is confirmed (whole path,
   tilt > 62 deg).
6. DESIGN.md same paragraph: "The away-type headings avoid that face"; raw demo_paths.json: away_left crosses x <
   0.19 m in 7 (t38_h22) to 8 of 30; away and away_right 0; right 0 at the robot side but 3 of 30 beyond x 0.81 m.
7. DESIGN.md same paragraph, "(... the shifted layouts: `tools/left_out.py`)" for the demo numbers;
   `sim_evals/_start_pose_sweep/tools/left_out.py` only adds the 18 shifted candidates to candidates.json/md; their
   demo numbers (leftx50 10/30, leftx55 3/30 retreat crossings) come from `demo_paths.py` (`demo_paths.json`, all 40
   feasible poses).
8. DESIGN.md same paragraph, "crosses the robot-side box face x 0.20 in 21/30": the count is at x < 0.19 m
   (`demo_paths.py`: the box enlarged by the gate's 10 mm pull-back margin); 21 / 10 / 3 confirmed.
9. Not in the raw files: "the retreat after the grasp moves ~25 cm to the hand's left" (demo_paths.json has counts
   only); "demos that started higher than the start carry into the table" (an interpretation; no per-demo heights in
   the sweep files); "at 38 deg the corners 9 cm ahead fall below its image" (candidates.json stores only how many
   of the 9 placements fail per camera, not which or where).

Confirmed as written in DESIGN: 40 of 63 (26 + 14); IMX519 view at tilt 38 / 10 and 16 cm for every heading; away at
0.60 m only 48/16 feasible, joint margin 0.12 rad, self clearance 7 cm (7.01), the 0.7 cm self clearance (0.68 cm,
away_t58_h16); left joint margin 1.0-1.4 rad (0.993-1.424) and self clearance 22-24 cm (22.07-23.97); every candidate
passes the real PC's `p` rules (with the caveat in 1); cone margin 2 deg at 58; 21 / 10 / 3 retreat crossings;
away-type grasp/end joint margin 0.001-0.09 rad; away-left 11-21/30 reachable; left 30/30 reachable with >= 0.25 rad,
0.474 at `leftx55_t48_h16`.

## Conclusion

40 of 63 candidate start poses are feasible (26 of the 45-pose grid, 14 of the 18 shifted left layouts). The
envelope of `q_start_rad`: tilt 48 deg or more (38 deg only at 22 cm, for the IMX519 view), tilt below 58 deg in
practice (cone margin 2 deg there, 9 of 30 demos over the cone), fingertips pointing to the robot's left (the right
runs joint 7 into its limit, away runs joint 4/5 into theirs; the diagonals are feasible but leave the demos'
grasps at <= 0.09 rad), with the eraser 0.55 m out rather than 0.45 m (retreat crossings 3 instead of 21 of 30). This
is the basis of the chosen pose `leftx55_t48_h16` (sim default, `sim_deploy/stations/start_pose.yaml`, and the
recommendation for the real arm, [5_real_pc_handoff.md](5_real_pc_handoff.md)); the policy screen could not rank the
poses ([3_early_failure_screen.md](3_early_failure_screen.md)).

## Limitations / sim-vs-real caveats

- Offline kinematics only: no physics, no policy. The demo paths are open-loop carries of recorded motion; a
  closed-loop policy can correct (or worsen) them, and v_max / pull-back are not modelled here
  ([4_gate_check.md](4_gate_check.md) does that).
- The view check uses the sim's pinhole camera models at the rig's mount pose, without occlusion; the Franka mount
  sees the scene ~3-4 % off the handheld view (PLAN §10, [6_camera_swap.md](6_camera_swap.md)). The 38 deg limit
  should be checked with the real IMX519.
- Joint limits are the Panda's nominal ones; the real arm also has velocity / acceleration limits near them and the
  rig and finger geometry are CAD values, so a 0.1 rad margin is a sim threshold, not a measured safe value.
- One IK solution per pose (the best of 24 seeds by the cost above); another branch could have more margin, so a
  failing joint margin does not prove that no better q exists.
- The grid is coarse (3 tilts, 3 heights, 5 headings, 3 left layouts) and does not cover the demos' extremes
  (tilt up to 64 deg, heights 6-27 cm); only PICK_TEST1 (30 demos) is carried here.
- The real-PC numbers of the first 45 rows were computed against the ea9f7fb box (Analysis 1).
- The tool scripts in `sim_evals/_start_pose_sweep/tools/` were saved there at 10:32:55, after their outputs (10:15 /
  10:17); the outputs match their code (keys, counts), but the copies are not proven identical to what ran.

## Open questions

1. Is the IMX519 view limit at 38 deg / 16 cm real? Answer: on the real arm at that pose, save the IMX519 frame
   (`policy_deploy.frames.probe --save`) with the eraser 9 cm ahead of the fingertips.
2. Is there an away layout between G 0.50 and 0.60 m (or beyond 0.60 m) with joint margin >= 0.10 rad over more
   tilts and heights? Answer: a `left_out.py`-style run for the away heading at G 0.55 / 0.65 / 0.70 m (no raw rows
   exist below or above 0.60 m).
3. Would G 0.60 m for the left heading remove the last 3 retreat crossings without losing joint margin or the region
   on the table / in view? Answer: `left_out.py` with gx 0.60 m, then `demo_paths.py`.
4. Should candidates.json be regenerated so the first 45 rows' real-PC margins use the current box? Answer: re-run
   steps 1-3 (writes into `sim_evals/`; ask first; pass/fail will not change).

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| candidates (all checks, 63 rows) | `sim_evals/_start_pose_sweep/candidates.json` (150 KB, not copied), `sim_evals/_start_pose_sweep/candidates.md` |
| pose files (start_pose.yaml format, 61 candidates + `standin_z0`, the P3 stand-in checked with `start_sweep handoff --id standin_z0` at 09:58:52) | `sim_evals/_start_pose_sweep/poses/<id>.yaml` |
| demo paths carried from the 40 feasible poses | `sim_evals/_start_pose_sweep/demo_paths.json` |
| code | `sim_deploy/scene/start_sweep.py`, `sim_deploy/scene/start_pose.py` (`describe`), `sim_deploy/scene/placement.py` (`check_region`, `in_view`), `sim_deploy/scene/trainstart.py` (`solve_ik`), `sim_evals/_start_pose_sweep/tools/left_out.py`, `sim_evals/_start_pose_sweep/tools/demo_paths.py` |
| region and demo statistics | `sim_deploy/scene/randomization_start.yaml` |
| logs | none saved (the three commands print to the terminal only) |
| copied into this folder | `data/p2_candidates.md` (copy of `sim_evals/_start_pose_sweep/candidates.md`), `data/p2_candidates_compact.csv` (derived from candidates.json: one row per pose, all check values, q, the stored and the current-box real box margins), `data/p2_demo_paths.csv` (derived from demo_paths.json, one row per feasible pose) |
