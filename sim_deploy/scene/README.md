# sim_deploy/scene: where the eraser goes

The table and the eraser are part of the one model file (`python -m sim_deploy.model build`, values in
`../stations/sim_rig.yaml` `scene:`). This folder decides **where the eraser is at the start of each rollout**: a
seeded random pose for benchmarks, or the "replay placement" for check G6. The eval driver sends the pose to the sim
PC with the scene control `set_object` (DESIGN.md section 5). Runs in `~/envs/sim_deploy`.

| File | What it is |
|---|---|
| `randomization_start.yaml` | P2, `--home start` (default): the eraser region relative to the start pose (ahead / left of the fingertips, yaw to their heading), from the demos, and the demos' start numbers |
| `randomization.yaml` | `--home deploy`: the region (x, y in the robot base frame O, metres) and the yaw range (degrees) of the eraser, and the settle time after placing it |
| `start_pose.py` | P2: `../stations/start_pose.yaml` (read, write, from a probe printout or git), its checks in the sim (FK), `start-data` (the demos' numbers) |
| `placement.py` | `Scene.load()`, `random_pose(seed, rollout)`, `replay_placement("replay:<zarr>#<i>")`, `check_region()` |
| `zarr_lowdim.py` | reads a umiproc package's low-dim arrays without the zarr package (the same reader as the copy's replay policy) |

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
~/envs/sim_deploy/bin/python -m sim_deploy.scene check                     # region corners at the start pose: table, box, in view
~/envs/sim_deploy/bin/python -m sim_deploy.scene --home deploy check       # ... at the deploy home
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose --show         # the start pose in use, its checks
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-data                # the eraser relative to the demos' start (--write: the yaml)
~/envs/sim_deploy/bin/python -m sim_deploy.scene sample --seed 0 --rollouts 5
~/envs/sim_deploy/bin/python -m sim_deploy.scene replay "replay:$HOME/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip#2"
~/envs/sim_deploy/bin/python -m sim_deploy.scene replay-all ~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip
```

## The scene (values in sim_rig.yaml)

- Table top at O z = 0 (P3/P4, the user: the robot base is mounted directly on the table, the table top is its
  mounting plane), x -0.40 to 1.00 m, y ±0.80 m, all around the base; no pedestal. link0 and link1 do not collide
  with it (link0 sits on it), every other link, the hand, rig and fingers do. The sim PC's box floor is
  `workspace_min_m` z -0.005 (table - 5 mm): any fingertip height above the table is inside the box, and a TCP
  pressed up to 15 mm into the table is pulled back by the gate's margin, not refused. (Until P3: table top z 0.050,
  x 0.16-0.96, y ±0.50, the robot on a pedestal; box floor 0.045.)
- Eraser: `size_m` [length, width, thickness] = 127 × 51 × 12 mm (the user, 2026-11-20), `resting: flat`
  (`on_long_edge` and `on_end` are the other choices; both are parameters in `sim_rig.yaml`), 30 g, yellow,
  friction 1.0 / torsional 0.01 / rolling 0.0001, condim 4, the stiff finger-pad contact (solref 2 ms). At yaw 0 its
  length is along O x, so the fingers (closing along O y at home) face its 51 mm side.
- Home TCP (fingertip point) at O (0.307, 0.000, 0.355). Eraser default pose (0.30, 0.00), centred under it (the user, E7).
- `--home start` (P2, the default): rollouts start at the PC's start pose (key `p`, `../stations/start_pose.yaml`); the
  eraser goes ahead of the fingertips as in the demos (below). `--home trainstart` is an alias: E7's `trainstart.py` IK
  solution is the stand-in start pose; `trainstart.yaml`, `randomization_trainstart.yaml` and
  `../stations/control_sim_trainstart.yaml` are no longer read.

## The start pose (P2)

`../stations/start_pose.yaml` holds only `franka_control: homing: q_start_rad: [7 rad]` (the key of the real PC's
`station.yaml`) and a comment block (source, date, the pose in the sim). `run.sh` merges it into the sim PC's station
(`../sim_pc/merge_station.py`); `--home deploy` merges `q_start_rad: null` instead. Write it only through:

```bash
python -m sim_deploy.scene start-pose --q "q1,...,q7" | --from-probe FILE | --from-git [REF]  [--source TEXT] [--force] [--dry]
python -m sim_deploy.scene start-pose --show [--json]
```

- `--from-probe FILE` reads the last `q_start_rad: [...]` line of a `pc_probe --start-pose` printout (`-` = stdin).
- `--from-git [REF]` (default `rnoro/final-deployment`): `GIT_OPTIONAL_LOCKS=0 git show REF:deployment/control_deploy/station.yaml`,
  `franka_control.homing.q_start_rad`; until it holds 7 numbers it fails with "not set yet" (the committed file, not
  final-deployment's working tree).
- Printed every time, from the model's forward kinematics: the TCP, its height above the sim table, the fingertip
  direction's tilt from straight down and its heading in pc_probe's words, and the checks: the Panda's joint limits
  (franka_control's loader table) and the model's, the TCP inside the sim PC's box, the tilt within `cone_max_deg`
  (no margin: what the PC checks at the end of `p`), no contact, the `p` move from home within `homing.timeout_s`
  (`1.875 * max|dq| / 0.3 rad/s`: 15.1 s for the stand-in), a heading (fingertips not straight down). A failing pose
  is not written without `--force`. Information: tilt and height against the demos' (`randomization_start.yaml
  demos`), and (`--show`) whether the eraser region fits at that pose.

## The eraser relative to the start pose (P2)

Frame: origin = the start TCP on the table; **ahead** = the fingertip direction projected on the table; **left** = up
× ahead; **yaw_rel** = the eraser's long axis relative to ahead. `start-data` measures it on every demo (the eraser
centre = the grasp TCP, half the eraser height back along its approach axis, relative to the episode's first pose):

| PICK_TEST1 `pick1-v1`, 30 episodes | median | 10th-90th pct (the region) | min-max |
|---|---|---|---|
| ahead (cm) | 15.9 | 9.0 to 21.7 | 7.7 to 22.5 |
| left (cm) | -1.9 (right of the fingertips) | -3.5 to -0.3 | -5.3 to +2.0 |
| yaw_rel (deg) | -2.2 | -6.6 to +0.8 | -7.9 to +3.5 |
| eraser centre below the fingertips (cm) | 16.1 | 8.6 to 23.9 | 6.4 to 27.2 |
| fingertip height above the table (cm) | 16.3 | 8.7 to 24.2 | 6.2 to 27.1 |
| tilt from straight down (deg) | 48.2 | 40.5 to 61.2 | 38.1 to 64.0 |

pc_probe's text ("~16 cm ahead horizontally, ~17 cm below the fingertips, centred left-right") agrees within 1 cm;
the 17 cm is E7's start-to-grasp-TCP height (16.6 cm), the eraser's centre is 0.5 cm lower again; the data's
left-right centre is 1.9 cm to the right (PICK_DOUBLE `pick1-double-v2`: ahead 17.7, left -2.5, yaw_rel -1.9).
Rollout n of seed S draws ahead, left, yaw_rel uniformly in the 10th-90th percentile ranges (`rng([S, n])`), then the
centre is clipped so the whole eraser stays on the table and inside the box (xy); `random_placement` records
`clipped`. At the stand-in every region corner is on the table, in the box and in both camera views (`check`), and
none of 100 seeded draws is clipped. Replays from the start pose: the 30 PICK_TEST1 episodes grasp between -12.1
and +8.0 cm of the resting eraser's height (median +0.2 cm; from the deploy home +7.5 cm); 7 are within 2 cm
(episodes 0, 7, 14, 16, 19, 23, 28: the catalog's `grasp_ok`), so G6 checks the grasp on those. (P3 corrected
P2's "all 30 within 2 cm": only the median is.)

## Random placements

Rollout `n` of an eval with `--seed S` draws from `numpy.random.default_rng([S, n])`: x, y uniform in the region, yaw
uniform in the range (added to the station's default yaw). Same seed, same placements, for every policy.

The region comes from the training data. `replay-all` on PICK_TEST1 `pick1-v1` carries every recorded grasp to the
sim's home TCP (below): the eraser sat 0 to 7 cm behind the start TCP (toward the base) and up to 6 cm to its right
(x 0.238 to 0.332, y −0.057 to +0.016, median (0.284, −0.021)); its long axis was within −1 to +5 deg of O x. The
region was x 0.26 to 0.34, y −0.06 to +0.02; by the user's choice (E7) it is now centred under the home TCP: x 0.27 to
0.34, y −0.03 to +0.03, yaw −8 to +12 deg. `check` confirms every corner keeps the whole eraser on the table, inside the box, and inside the IMX219's
view at home (the model's pinhole stand-in, the policy's virtual camera; it is also inside the IMX519's view).

## Replay placement (check G6)

The replay policy repeats the recorded motion relative to wherever the TCP is when the rollout starts:
`O_T_TCP(t) = O_T_TCP_start · inv(P_s) · P_j`. So the eraser must be where the recorded grasp happens, carried over
the same way: `O_T_grasp = O_T_TCP_home · inv(P_s) · P_g`, with P_s the episode's first recorded TCP pose and P_g the
TCP pose when the fingers start closing. Row g comes from the width: the last row before the width falls below
`open − 15 %` of the drop from the open width (median of the first rows) to the hold width (the episode's minimum).

The data has no object pose, so three assumptions fill the gap:

1. The eraser centre is on the grasp TCP's approach axis, `--behind-tip` (default: half its resting height) back
   from the fingertip point toward the hand: the fingertips at the eraser's bottom, i.e. at the table. The
   recordings agree: their frame has z up with the table at z ≈ 0, and every grasp's fingertip point is at −1 to
   +1 cm.
2. The fingers close across the eraser's body y (its width when flat): body y is laid along the grasp closing axis
   (TCP x) projected on the table, its length along the projected TCP y.
3. It rests on the table: x, y from step 1, z = table top + half its resting height. The gap between the step-1
   height and the resting height is reported as `height_error_m`. The output also gives the recording's own start
   and grasp tip heights and tilts.

`height_error_m` says whether the replay can close on the eraser from the sim's home. For `pick1-v1` it ranges from
−4.5 to +17.2 cm (median +7.5): the handheld started 12 to 24 cm above the table, the sim's home TCP was 30.5 cm (35.5 cm since P3's table at z 0: add 5 cm to the deploy-home numbers here)
above it. Episode 14 (E1's G6 episode) is +11.6 cm: the recorded close happens above the eraser, so G6 on it checks the
path and the axes, not the grasp. Episodes with a small error, for a grasp check: 2 (−0.4 cm), 8 (+1.6 cm),
3 (−1.4 cm), 10 (+2.4 cm).

**The start pose differs from training.** Every recording starts with the approach axis 40 to 51° from vertical
(the camera looking ahead over the table) and grasps at 28 to 40°; the sim's home (`q_home_rad`) points the fingers
straight down. The replay keeps the relative motion, so in the sim it grasps at about 10 to 20° from vertical. A
learned policy sees a start view unlike its training's first frames. Starting from a training-like pose needs
another home pose (E4: `set_arm` is always refused, the arm is reset only by homing).

**Grasp feasibility (E7, before W0: the 0.08 stroke; DESIGN.md "E7: integration").** The eraser lies flat and is gripped across its
51 mm width (the user, the IMX519 frames of episode_000152). Training's width is the fingertip gap (D8 revised: umiproc
records the tag distance minus the closed one): the holds 0.052-0.056 are a 52-56 mm gap around the 51 mm eraser,
the opening 0.090 is the full stroke, and the policy's clip at 0.08 still opens 80 mm. The sim's pads carry a soft
contact layer (`sim_rig.yaml finger_contact`) so that a commanded 0.054 holds the eraser with ~5 N per pad. The
default `resting: flat` is right.

**W0 (2026-11-20/21): with the printed fingers' 40 mm max opening the 51 mm eraser cannot be grasped.** Upstream
f632576 measured the real fingers' max fingertip opening as 0.040 m (`limits.yaml gripper_width_max_m`), and the sim
model now has the same stroke (`sim_rig.yaml finger_max_gap_m`). The gripping faces are flat and parallel over the
whole 123 mm pad, so the gap is 40 mm at every depth: the 51 mm eraser never fits between them. In a scripted grasp
the pads land on its top, slide over it while closing, and leave it on the table. With upstream's `gripper_map`, C2
then read the commanded ~0.031 (a check made before `grasp_closes_fully`), which the model sees as 0.054, the training hold. A grasp needs an opening of about
0.06 m or more. **Decision (W0):** the robot side stays upstream's (stroke 0.040, `gripper_map`, `width_offset_m`
0, the Franka Hand's grasp closing until it stalls), and the sim's eraser is narrowed to **0.030 m** across the grasp
(`sim_rig.yaml scene.eraser.size_m [0.127, 0.030, 0.012]`) = upstream's `gripper_map.robot_closed_m` ("the eraser's
width: the robot holding it reads this"). Offline (arm held, `sim_evals/_pipeline/W0/gripper/`): a grasp from the
40 mm open stop closes on it at a 22 mm gap with 20 N per pad and holds it hanging (0.5 mm slip in 5 s); C2 then
reads 0.022, which `to_model` clamps to 0.030 = training 0.051 (closed on the eraser). This is a sim-only stand-in:
the real eraser is 51 mm wide, so on the arm the same commands do not grasp it either (open question for the user).
W2's look stations keep the 51 mm eraser for images (`look/make_stations.py`).
