# 01_start_pose / 4: the demos carried to a start pose through a port of the real gate

**Status:** done · **Date:** 2026-11-20 · **Who:** agent ·
**Plan ref:** sim_deploy/PLAN.md §6 (P3 G6), §9 (P4), §10; DESIGN.md "P3/P4" -> "Screen and full eval", "The chosen pose"

Part 4 of 6 of [01_start_pose](README.md). Before: [1_p_key_and_standins.md](1_p_key_and_standins.md),
[2_candidates_envelope.md](2_candidates_envelope.md), [3_early_failure_screen.md](3_early_failure_screen.md). After:
[5_real_pc_handoff.md](5_real_pc_handoff.md), [6_camera_swap.md](6_camera_swap.md).

## Question

If the training demos' TCP motions are replayed relative to a start pose (what the policy's relative actions do with
perfect tracking), how often does the real PC's safety gate refuse, pull back or shorten them, and which rule is
binding? Specifically: (a) is the chosen pose `leftx55_t48_h16` better than the 22 cm alternative and than the same
pose with the eraser 0.45 m or 0.50 m out, and (b) which real-PC gate values (box, cone, v_max, w_max) would the
demos need?

## Why it matters for the real robot

The gate is `franka_control`'s `safety_gate.cpp`, unchanged on the real PC. A refused request is a fault on the arm
(latched, the episode ends); a shortened one is a motion that is never made up. The answer says (1) which start pose
to hand-guide the arm to, (2) whether the box (x min, floor z) and cone in the real config can stay as they are, and
(3) whether v_max 0.25 m/s will hold the policy back. The policy screen in the sim could not rank poses
([3_early_failure_screen.md](3_early_failure_screen.md): the policy backs away from the sim's IMX219 view), so this
offline check is what the pose ranking rests on.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Model | `sim_deploy/build/sim_rig/model.json` `model_sha256` `166a0649571a…` (P3/P4: table top at O z 0). Used only for the start pose's forward kinematics and the IK along the path; the gate port itself uses no model. The JSONs do not record the sha; this is the model in place when they were written (10:32) and now. |
| Home / start pose | no homing; each demo is carried from `q_start_rad` of the pose under test (table below). All six poses: tilt 48 deg, heading left, TCP y -0.159 m |
| Eraser placement | none (no scene); the eraser region is implied by the pose id (`leftx<G>`: region centred at O x 0.<G> m) |
| Stations / overlays | the gate values = `deployment/control_deploy/franka_control/config.yaml` `safety:` (lines 44-54) with the box min from `sim_deploy/stations/control_sim.yaml` line 11; copied into `GATE` in `sim_deploy/scene/gate_check.py` lines 38-40 (values below) |
| Policy / checkpoint | none: the recorded demos. PICK_TEST1 `~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip` (30 demos) and PICK_DOUBLE `~/data/umi/datasets/multi/pick1-double-v2/pick1-double-v2.zarr.zip` (62 = day 1 (30) + day 2 (32)) |
| Inference settings | modelled as the manager sends: one request per 0.1 s (`CYCLE_S`), `exec_h` 12 waypoints at 50 ms (`EXEC_H`, `STEP_S`) |
| Seeds / episodes | deterministic; every demo of both datasets |
| Timeout | none |
| Sim-only settings | `--no-floor` variants: box floor at z -1 m (to separate "carried into the table" from the rest). Otherwise the real values (box min the user's real station, which current upstream also has, see "Docs vs raw data") |

Gate values used (`gate` block of every JSON; identical in all eight files except the floor):

| Key | Value | Source |
|---|---|---|
| `workspace_min_m` | [0.20, -0.40, -0.005] m (no-floor: [0.20, -0.40, -1.0]) | `control_sim.yaml` line 11 (the ea9f7fb copy's `config.yaml` line 45 says [0.25, -0.40, 0.05]; current upstream `~/Capstone-final-deployment/.../config.yaml` line 43 @ 447910b says [0.20, -0.40, -0.005]) |
| `workspace_max_m` | [0.80, 0.40, 0.70] m | `config.yaml` line 46 |
| `box_margin_m` | 0.01 m | line 49 |
| `cone_axis`, `cone_max_deg`, `cone_margin_deg` | (0, 0, -1), 60 deg, 2 deg | lines 47, 48, 50 |
| `v_max_m_per_s`, `w_max_rad_per_s`, `slack_s` | 0.25 m/s, 1.0 rad/s, 0.010 s | lines 51-53 |
| `max_lead_s` | 1.5 s | line 54 |

The eight result files (identified by matching each JSON's `q` to `sim_evals/_start_pose_sweep/poses/<id>.yaml` to
1e-6 rad; the JSONs record no pose id and no command line). **Each file holds both datasets** (`datasets` keys
`pick1-v1.zarr.zip` and `pick1-double-v2.zarr.zip`):

| File (`sim_evals/_start_pose_sweep/gate/`) | Pose id | Start TCP O (m) | Eraser region centre | Box floor z | IK |
|---|---|---|---|---|---|
| `gate_h16.json/.txt` | `leftx55_t48_h16` (chosen pose; = `stations/start_pose.yaml`) | (0.531, -0.159, 0.160) | O x 0.55 m | -0.005 m | yes |
| `gate_h16_nofloor.json/.txt` | `leftx55_t48_h16` | (0.531, -0.159, 0.160) | 0.55 m | -1.0 m | no |
| `gate_h22.json/.txt` | `leftx55_t48_h22` (the 22 cm alternative) | (0.531, -0.159, 0.220) | 0.55 m | -0.005 m | yes |
| `gate_h22_nofloor.json/.txt` | `leftx55_t48_h22` | (0.531, -0.159, 0.220) | 0.55 m | -1.0 m | no |
| `gate_left_t48_h16.json/.txt` | `left_t48_h16` | (0.431, -0.159, 0.160) | 0.45 m | **-1.0 m** | yes |
| `gate_left_t48_h22.json/.txt` | `left_t48_h22` | (0.431, -0.159, 0.220) | 0.45 m | **-1.0 m** | yes |
| `gate_leftx50_t48_h16.json/.txt` | `leftx50_t48_h16` | (0.481, -0.159, 0.160) | 0.50 m | **-1.0 m** | yes |
| `gate_leftx50_t48_h22.json/.txt` | `leftx50_t48_h22` | (0.481, -0.159, 0.220) | 0.50 m | **-1.0 m** | yes |

The `left_*` and `leftx50_*` files were run with `--no-floor` but with IK (their `gate.workspace_min_m` z is -1.0 and
the demos have an `ik` record). That is why there are no separate no-floor files for them: their refusal counts
already are the "without the floor" numbers. PICK_DOUBLE's episodes 0-29 give results identical to PICK_TEST1's 30
in every field (checked on `gate_h16.json`): day 1 of pick1-double-v2 is PICK_TEST1.

## Method

**The tool.** `sim_deploy/scene/gate_check.py` (new in P4). Per demo:

1. Carry: the recorded 60 Hz TCP poses P(t) relative to the first, applied to the start pose:
   W(t) = O_T_start · inv(P_0) · P(t) (lines 51-61; position interpolated linearly, rotation by slerp).
2. Requests: from t_c = 0 every 0.1 s, a request with 12 waypoints W(t_c + k · 50 ms), k = 1..12, checked on its own
   from start = W(t_c) with "now" = t_c (lines 127-141). This assumes the previous target was reached exactly.
3. The gate port, per request (`gate_request`, lines 64-118), compared rule by rule with the C++ below.
4. Phases from the demo's grasp row g (the width rule `placement.grasp_rows`): approach [0, g), grasp [g, g + 0.5 s),
   lift [g + 0.5 s, g + 2 s), retreat after that (line 48). A request's phase is the phase of t_c.
5. Per demo also: max tilt, min/max x, y, z of the carried path on a 50 ms grid; the speed limit each 50 ms step would
   need with the bucket's slack, `v_needed` = step / (0.05 s + 0.010 s) (and the same for rotation); the distance lost
   by the grasp when following W with the speed cap only (the manager never makes a shortened distance up), and the
   smallest v (0.05 m/s steps) for which that loss is <= 1 cm (lines 142-170).
6. IK along the path every 0.1 s, warm-started from q_start (DLS `trainstart.solve_ik`, 150 iterations; a sample is
   unreachable if position error > 1 mm or angle error > 0.5 rad); the smallest margin to the Panda joint limits and
   its joint (lines 177-202).
7. Summary per dataset (lines 226-266): "28 of 30" is scaled to `ceil(n · 28/30)` = 58 of 62 for PICK_DOUBLE; a value
   "for 28/30" is the 28th smallest per-demo value. The IK p10 is the `n // 10`-th smallest margin.

**The port vs the C++** (`deployment/control_deploy/franka_control/src/safety_gate.cpp`, identical to current upstream
`~/Capstone-final-deployment/.../src/safety_gate.cpp` and `include/franka_control/safety_gate.h` @ 447910b: `diff`
empty):

| Rule | C++ (`safety_gate.cpp`) | Port (`gate_check.py`) | Same? |
|---|---|---|---|
| skip past waypoints | lines 69-72: `t_ns <= now` skipped | none: all waypoints are 50-600 ms ahead of "now" | not modelled (no latency in the port) |
| 1. finite | lines 78-80 | none | not needed (recorded data) |
| 2. max_lead | lines 82-88: lead > `max_lead_s` refused | lines 75-77: (k + 1) · 0.05 s > `max_lead_s` refused | same rule; in the port the lead is at most 0.6 s < 1.5 s, so it can never fire |
| 3. gripper width in [0, 0.08] m | lines 90-92 (`gripper_width_max_m` 0.08, `contracts/limits.yaml` line 6) | none | **not modelled**: widths are not checked (the laptop clips widths to that limit per the `limits.yaml` comment; not verified here) |
| 4. box | lines 94-122: per axis, outside by <= margin + 1e-12 -> pulled onto the face, beyond -> refused (all axes listed) | lines 78-86: any axis beyond margin + 1e-12 -> refused; else clip to the box | same outcome; the port's text names only the worst axis |
| 5. cone on the flange z | lines 124-143: tilt of (q_TCP · q_F_TCP⁻¹) z from `cone_axis`; > 60 -> turned back to 60 about z × axis (left-multiplied); > 62 + 1e-12 -> refused | lines 87-96: tilt of the TCP z; same pull-back (left-multiplied rotvec) and refusal | same: F_T_TCP turns only about z (`contracts/frames.yaml` rows 1-3), so flange z = TCP z. The C++ fallback axis for z antiparallel to the axis is missing in the port; unreachable (180 deg is refused first) |
| refusal = whole request | lines 157-167 | first refused waypoint returns (line 77, 83, 92) | same outcome |
| 6. speed, leaky bucket | lines 169-199: `allowed = min(left, v·slack) + v·dt`, on the pulled-back poses, position then rotation (slerp), tolerance 1e-9 | lines 99-117: the same formula, order, tolerance and slerp; rotation angle from `(R · R_prev⁻¹).magnitude()` vs C++ `rotationAngle` (`pose_math.cpp` lines 27-30): same angle | same |
| start of the bucket | `supervisor.cpp` line 443: start = `plan_.at(now)` (where the current plan is at that moment) or the held target | start = W(t_c), time 0 = now | **assumption**: perfect tracking and no lag from earlier shortening; the port is optimistic about speed (the "lost by the grasp" metric covers the accumulated lag) |

Verdict: the port matches the C++ rule for rule on box, cone, speed and lead; it omits the width rule, the finite
check and skipped waypoints, and it has no inference latency.

**What counts as a pass.** A pose passes for a demo when no request is refused; "pulled back" and "shortened" are
counted but do not fault. The pose ranking uses refusals without the floor (the floor refusals are the carry
artifact: a demo that started higher above the table than 16 cm gets carried into the table) and IK reachability.

## Commands to reproduce

Not re-run for this record (the CLI writes wherever `--json` points, here into `sim_evals/`). It needs the
`sim_deploy` env (numpy, scipy, mujoco for FK/IK), the built model and the two datasets; no sim, no ports, no GPU,
CPU only. Run time was not recorded. `--q` must be given as `--q="..."` because the value starts with `-`. The command
lines are not recorded in the JSONs; the flags below follow from each JSON's `gate` block and whether the demos have an
`ik` record. The `.txt` file is the command's stdout.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
PY=~/envs/sim_deploy/bin/python; OUT=sim_evals/_start_pose_sweep/gate
Q55_16="-0.5928536,0.5219462,-0.0897168,-1.5930073,0.6595725,1.5068439,-1.5301472"   # leftx55_t48_h16
Q55_22="-0.5926811,0.4637061,-0.1004182,-1.5344559,0.6613767,1.4082918,-1.4666293"   # leftx55_t48_h22
Q45_16="-0.6014899,0.3057479,-0.1787837,-1.9181251,0.6037021,1.5439553,-1.6675477"   # left_t48_h16
Q45_22="-0.5727588,0.2361335,-0.2192034,-1.8621043,0.5989295,1.4344807,-1.6045637"   # left_t48_h22
Q50_16="-0.6109992,0.4071175,-0.1158965,-1.7682436,0.6316207,1.5321963,-1.5993218"   # leftx50_t48_h16
Q50_22="-0.5876186,0.3434093,-0.1527175,-1.7115005,0.6345415,1.4258252,-1.5394546"   # leftx50_t48_h22
# default --dataset = both zips (DEFAULT_DATASETS in gate_check.py)
$PY -m sim_deploy.scene.gate_check --q="$Q55_16" --json $OUT/gate_h16.json > $OUT/gate_h16.txt
$PY -m sim_deploy.scene.gate_check --q="$Q55_16" --no-floor --no-ik --json $OUT/gate_h16_nofloor.json > $OUT/gate_h16_nofloor.txt
$PY -m sim_deploy.scene.gate_check --q="$Q55_22" --json $OUT/gate_h22.json > $OUT/gate_h22.txt
$PY -m sim_deploy.scene.gate_check --q="$Q55_22" --no-floor --no-ik --json $OUT/gate_h22_nofloor.json > $OUT/gate_h22_nofloor.txt
$PY -m sim_deploy.scene.gate_check --q="$Q45_16" --no-floor --json $OUT/gate_left_t48_h16.json > $OUT/gate_left_t48_h16.txt
$PY -m sim_deploy.scene.gate_check --q="$Q45_22" --no-floor --json $OUT/gate_left_t48_h22.json > $OUT/gate_left_t48_h22.txt
$PY -m sim_deploy.scene.gate_check --q="$Q50_16" --no-floor --json $OUT/gate_leftx50_t48_h16.json > $OUT/gate_leftx50_t48_h16.txt
$PY -m sim_deploy.scene.gate_check --q="$Q50_22" --no-floor --json $OUT/gate_leftx50_t48_h22.json > $OUT/gate_leftx50_t48_h22.txt
```

The derived CSVs in `data/` came from read-only Python over the JSONs (per demo: first refusal and its axis, the axes
refused, phases, worst values, IK) and, for `p4gate_grasp_z.csv`, the carry evaluated at each demo's grasp row:
`z = (sp.fk(q)["O_T_tcp"] @ inv(P[0]) @ P)[grasp_rows(widths)["start"], 2, 3]` with `sim_deploy.scene.start_pose.fk`,
`zarr_lowdim.load_episode` and `placement.grasp_rows` (read-only, writes nothing in the repo).

## Results

### Rules that are the same for every pose (the relative motion does not depend on the start)

Identical in all eight files (`worst`, `for_k_of_n`, `lost_at_grasp_m`, `v_peak_phase`, `tilt_peak_phase`):

| Rule / value | PICK_TEST1 (30) | PICK_DOUBLE (62) |
|---|---|---|
| max_lead refusals | 0 (cannot fire: lead <= 0.6 s) | 0 |
| max tilt (deg) | 59.5 (ep 29, retreat); 0 demos > 60, 0 > 62 | 65.2 (ep 37, retreat); 3 demos > 60 (eps 57 60.5, 50 61.5, 37 65.2), 1 > 62 |
| cone_max for 28/30 (58/62): no pull-back / no refusal | 54.0 / 52.0 deg | 57.9 / 55.9 deg |
| v needed for 28/30 (58/62) with no shortening | 1.211 m/s | 1.216 m/s |
| worst v needed (one 50 ms step, with the 10 ms slack) | 1.274 m/s (= 1.53 m/s over the bare 50 ms) | 1.277 m/s (1.53 m/s) |
| median demo's v needed; demos needing > 0.25 / > 0.5 m/s | 0.67 m/s; 30 / 22 | 0.85 m/s; 61 / 52 |
| phase of each demo's speed peak | approach 30/30 | approach 62/62 |
| distance lost by the grasp at 0.25 m/s: median / max / demos > 2 cm | 0.026 m / 0.080 m / 20 (24 > 1 cm) | 0.038 m / 0.080 m / 50 (54 > 1 cm) |
| v so that the loss is <= 1 cm for 28/30 (58/62) | 1.05 m/s | 1.05 m/s |
| w needed for 28/30 (58/62); worst | 0.576 rad/s; 0.628 (ep 8); 0 demos > 1.0 | 0.688 rad/s; 1.078 (ep 49); 2 demos > 1.0 (eps 44, 49) |
| tilt peak phase | approach 12, lift 5, retreat 13 | approach 32, lift 6, retreat 24 |

### Rule by rule, per pose (refusals, pull-backs, shortening, IK)

"refused" = demos with at least one refused request; the first refusal is split by rule, phase and axis (axis from the
`refused` detail text). Shortened by phase counts demos with a shortened request in that phase (a refused request is
not counted as shortened, which is why the grasp column moves with the floor).

**PICK_TEST1 (30):**

| File | Pose, floor | Refused | First refusal | min x (m) | Box pulled back (by phase) | Cone pulled / refused | Shortened (approach, grasp, lift, retreat) | IK unreachable demos | IK min margin (ep, joint); p10 |
|---|---|---|---|---|---|---|---|---|---|
| gate_h16 | leftx55_t48_h16, -0.005 m | 17 | box approach 14 (all z), retreat 3 (all x: eps 24, 28, 29) | 0.159 | 20 (ap 15, gr 3, li 8, re 19) | 0 / 0 | 30 (30, 1, 19, 23) | 0 | 0.041 rad (ep 28, j4); 0.21 |
| gate_h16_nofloor | leftx55_t48_h16, -1 m | 3 | box retreat 3 (x: eps 24, 28, 29) | 0.159 | 4 (re 4) | 0 / 0 | 30 (30, 9, 19, 23) | - | - |
| gate_h22 | leftx55_t48_h22, -0.005 m | 9 | box approach 6 (z), retreat 3 (x) | 0.159 | 11 (ap 7, gr 1, li 4, re 10) | 0 / 0 | 30 (30, 4, 19, 23) | 0 | 0.085 rad (ep 28, j4); 0.191 |
| gate_h22_nofloor | leftx55_t48_h22, -1 m | 3 | box retreat 3 (x) | 0.159 | 4 (re 4) | 0 / 0 | 30 (30, 9, 19, 23) | - | - |
| gate_leftx50_t48_h16 | leftx50_t48_h16, -1 m | 10 | box retreat 10 (x) | 0.109 | 14 (re 14) | 0 / 0 | 30 (30, 9, 19, 22) | 1 (ep 28) | 0.015 rad (ep 28, j4); 0.088 |
| gate_leftx50_t48_h22 | leftx50_t48_h22, -1 m | 10 | box retreat 10 (x) | 0.109 | 14 (re 14) | 0 / 0 | 30 (30, 9, 19, 22) | 1 (ep 28) | 0.001 rad (ep 28, j4); 0.069 |
| gate_left_t48_h16 | left_t48_h16, -1 m | 21 | box retreat 19, lift 2 (x) | 0.059 | 24 (re 24, li 2) | 0 / 0 | 30 (30, 9, 19, 23) | 6 (eps 6, 10, 12, 13, 22, 28) | 0.001 rad (eps 10, 13, j4); 0.002 |
| gate_left_t48_h22 | left_t48_h22, -1 m | 21 | box retreat 19, lift 2 (x) | 0.059 | 24 (re 24, li 2) | 0 / 0 | 30 (30, 9, 19, 23) | 7 (eps 4, 6, 8, 10, 12, 13, 28) | 0.001 rad (eps 4, 13, 28, j4); 0.001 |

**PICK_DOUBLE pick1-double-v2 (62):**

| File | Pose, floor | Refused | First refusal | min x (m) | Box pulled back (by phase) | Cone pulled / refused (first) | Shortened (approach, grasp, lift, retreat) | IK unreachable demos | IK min margin; p10 |
|---|---|---|---|---|---|---|---|---|---|
| gate_h16 | leftx55_t48_h16, -0.005 m | 36 | box approach 29 (z), retreat 5 (x), lift 1 (x); cone lift 1 (ep 37, 62.4 deg) | 0.142 (ep 40) | 40 (ap 31, gr 9, li 18, re 36) | 3 (li 2, re 3) / 1 | 62 (62, 10, 43, 51) | 8 (eps 35, 36, 40, 42, 46, 47, 48, 49; all day 2, j4) | 0.001 rad (eps 46, 47, j4); 0.006 |
| gate_h16_nofloor | leftx55_t48_h16, -1 m | 14 | box retreat 12, lift 1 (x); cone lift 1 (ep 37) | 0.142 | 16 (re 16, li 2) | 3 / 1 | 62 (62, 21, 43, 51) | - | - |
| gate_h22 | leftx55_t48_h22, -0.005 m | 21 | box approach 7 (z), retreat 12, lift 1 (x); cone lift 1 | 0.142 | 25 (ap 9, gr 2, li 7, re 22) | 3 / 1 | 62 (62, 16, 43, 51) | 9 (eps 35, 36, 42, 44-49) | 0.001 rad (ep 45, j4); 0.007 |
| gate_h22_nofloor | leftx55_t48_h22, -1 m | 14 | box retreat 12, lift 1; cone lift 1 | 0.142 | 16 (re 16, li 2) | 3 / 1 | 62 (62, 21, 43, 51) | - | - |
| gate_leftx50_t48_h16 | leftx50_t48_h16, -1 m | 28 | box retreat 25, lift 2; cone lift 1 | 0.092 | 36 (re 36, li 3) | 3 / 1 | 62 (62, 21, 43, 48) | 17 | 0.002 rad (ep 56, j4); 0.009 |
| gate_leftx50_t48_h22 | leftx50_t48_h22, -1 m | 28 | box retreat 25, lift 2; cone lift 1 | 0.092 | 36 (re 36, li 3) | 3 / 1 | 62 (62, 21, 43, 48) | 13 | 0.001 rad (ep 28, j4); 0.007 |
| gate_left_t48_h16 | left_t48_h16, -1 m | 49 | box retreat 40, lift 8; cone lift 1 | 0.042 | 47 (re 47, li 5) | 2 (li 2, re 1) / 1 | 62 (62, 21, 43, 43) | 30 | 0.001 rad; 0.002 |
| gate_left_t48_h22 | left_t48_h22, -1 m | 49 | box retreat 40, lift 8; cone lift 1 | 0.042 | 47 (re 47, li 5) | 2 / 1 | 62 (62, 21, 43, 43) | 29 | 0.001 rad; 0.003 |

Joint 4 is the joint of the smallest margin in every demo of every file. Every box refusal is on z (the floor, only
with floor -0.005 m) or on x (the robot-side face x 0.20 m); none on y or the top.

### DESIGN's gate table, cell by cell (`sim_deploy/DESIGN.md` lines 948-960, `gate_h16.json` / `gate_h16_nofloor.json`)

| Row | DESIGN PICK_TEST1 | Raw PICK_TEST1 | DESIGN PICK_DOUBLE | Raw PICK_DOUBLE |
|---|---|---|---|---|
| max_lead | never | 0 lead refusals | never | 0 |
| box refused (first refusal) | 17 (approach 14 on the floor, retreat 3 on x) | 17: box approach 14 (z), retreat 3 (x) | 36 (approach 29, retreat 5, lift 1) | 36 refused, of which **35 box** (approach 29 z, retreat 5 x, lift 1 x) + **1 cone** (ep 37, lift) |
| box refused without the floor | 3, retreat, x < 0.19 (min x 0.159) | 3, retreat, x (eps 24, 28, 29); min x 0.159 | 14 (retreat 12, lift 1; min x 0.142) | 14 refused = **13 box** (retreat 12, lift 1) + **1 cone**; min x 0.142 |
| cone pulled back / refused | 0 / 0 (max tilt 59.5 deg) | 0 / 0; 59.5 | 3 / 1 (lift; max 65.2 deg) | 3 / 1 (ep 37, lift, 62.4 deg at refusal); 65.2 (ep 37, retreat) |
| cone_max for 28/30 (58/62) | <= 54.0 already passes: 60 is enough | 54.0 no pull-back, 52.0 no refusal; all 30 <= 59.5 | 57.9 (no pull-back), 55.9 (no refusal) | 57.9 / 55.9 |
| v_max 0.25: demos shortened | 30 (all in the approach; also lift 19, retreat 23) | 30 (approach 30, lift 19, retreat 23, **grasp 1**; no floor: grasp 9) | 62 | 62 (approach 62, lift 43, retreat 51, grasp 10) |
| v for 28/30 with no shortening | 1.21 m/s (worst 50 ms step: 1.27) | 1.211; worst 1.274 | 1.22 | 1.216; worst 1.277 |
| distance lost by the grasp at 0.25 | median 2.6 cm, max 8.0, 20 > 2 cm | 0.026 m, 0.080 m, 20 | median 3.8 cm, 50 > 2 cm | 0.038 m, 50 (max 0.080 m) |
| v so the loss <= 1 cm for 28/30 | 1.05 m/s | 1.05 | 1.05 | 1.05 |
| w_max 1.0 | never limiting for 28/30 (needs 0.58 rad/s) | 0.576 for 28/30; worst 0.628 | worst 1.08 | 1.078 (ep 49); 0.688 for 58/62 |
| IK along the path | all reachable, min margin 0.041 rad (ep 28, joint 4), p10 0.21 | 0 unreachable; 0.041 (ep 28, j4); 0.21 | 8 with unreachable samples (day 2, eps 35-49, joint 4), p10 0.006 | 8 (35, 36, 40, 42, 46, 47, 48, 49), j4; p10 0.006; min 0.001 |

The 22 cm alternative sentence (DESIGN lines 962-963) against `gate_h22*.json`:

| Claim | Raw |
|---|---|
| cone, speed and w rows the same | yes: identical `worst` / `for_k_of_n` / `lost_at_grasp_m` (shortened per phase differs only in the grasp column: 4 vs 1) |
| box refused 9/30 (6 approach floor, 3 retreat) | 9: approach 6 (z), retreat 3 (x) |
| 21/62 | 21 refused = **20 box** (approach 7 z, retreat 12 x, lift 1 x) + **1 cone** (ep 37) |
| without the floor the same 3 / 14 | 3 / 14 (`gate_h22_nofloor`, identical to `gate_h16_nofloor` apart from z extents) |
| IK all 30 reachable (min 0.085) | 0 unreachable, min 0.085 rad (ep 28, j4), p10 0.191. Not in DESIGN: PICK_DOUBLE 9 unreachable, p10 0.007 |

### Eraser distance (48 deg; box without the floor; IK along the path)

DESIGN line 964-965: "at 48 deg / 16 cm with the eraser 0.45 / 0.50 / 0.55 m out, box refusals without the floor 21 /
10 / 3 of 30, IK unreachable demos 6 / 1 / 0 of 30". The 21 and 10 are read directly from `gate_left_t48_h16` and
`gate_leftx50_t48_h16` (run with `--no-floor`); the 3 and 0 come from `gate_h16_nofloor` (refusals) and `gate_h16`
(IK, with the floor; IK does not depend on the box).

| Eraser out (pose) | Refused without the floor, PICK_TEST1 | min x (m) | IK unreachable, PICK_TEST1 | IK min margin / p10 (rad) | Refused w/o floor, PICK_DOUBLE | IK unreachable, PICK_DOUBLE |
|---|---|---|---|---|---|---|
| 0.45 m (`left_t48_h16`) | 21 (retreat 19, lift 2) | 0.059 | 6 | 0.001 / 0.002 | 49 | 30 |
| 0.50 m (`leftx50_t48_h16`) | 10 (retreat 10) | 0.109 | 1 | 0.015 / 0.088 | 28 | 17 |
| 0.55 m (`leftx55_t48_h16`) | 3 (retreat 3) | 0.159 | 0 | 0.041 / 0.21 | 14 | 8 |
| 0.45 m, 22 cm (`left_t48_h22`) | 21 | 0.059 | 7 | 0.001 / 0.001 | 49 | 29 |
| 0.50 m, 22 cm (`leftx50_t48_h22`) | 10 | 0.109 | 1 | 0.001 / 0.069 | 28 | 13 |
| 0.55 m, 22 cm (`leftx55_t48_h22`) | 3 | 0.159 | 0 | 0.085 / 0.191 | 14 | 9 |

DESIGN's "21 / 10 / 3" and "6 / 1 / 0" match the raw files exactly. Moving the region 5 cm out moves min x by
exactly 0.050 m (0.059 -> 0.109 -> 0.159), as expected for a pure translation of the carry.

### What a different box would do (leftx55_t48_h16, PICK_TEST1 / PICK_DOUBLE)

Estimated from each demo's min x (rounded to 1 mm in the JSON) with the 10 mm margin, refused if min x < x_min - 0.01 m;
the gate files themselves give 3 / 13 at x_min 0.20 m (ep 45 is out by 0.010 m, lost in the rounding).

| Box | Source | Refused on x | Refused on the floor | Note |
|---|---|---|---|---|
| x_min 0.25 m, z_min 0.05 m | ea9f7fb copy `deployment/.../config.yaml` line 45 | 10 / 28 | 24 / 53 carried grasps below z 0.04 m (refused; 25 / 54 paths go below it somewhere); a real grasp at the eraser: all | not the current real config |
| x_min 0.20 m, z_min -0.005 m | user's real station = `control_sim.yaml` line 11 = current upstream `config.yaml` line 43 @ 447910b | 3 / 13 (gate) | 14 / 29 carried into the table (approach) | the carry artifact; 0 on the floor in closed loop is expected |
| x_min 0.16 m | - | 0 / 1 | - | 1 / 3 pulled back |
| x_min 0.15 m | - | 0 / 0 | - | 0 / 1 pulled back |

Carried TCP z at the grasp start row (`data/p4gate_grasp_z.csv`): at 16 cm, PICK_TEST1 min -0.127 m, median -0.003 m,
max 0.077 m, 24/30 below 0.04 m; PICK_DOUBLE 53/62 below 0.04 m. At 22 cm: median 0.057 m, 13/30 and 24/62 below
0.04 m. The median of -0.003 m at 16 cm says the demos' median grasp is 16 cm below their start, i.e. the chosen height
puts the median carried grasp on the table.

## Analysis

**Which rule binds.** v_max. Every demo of both datasets is shortened, every demo's speed peak is in the approach, and
the open-loop loss at the grasp is a median 2.6 cm (PICK_TEST1) / 3.8 cm (PICK_DOUBLE), up to 8.0 cm. To pass 28/30
unshortened the gate would need 1.21 m/s (4.8x 0.25); to keep the loss <= 1 cm, 1.05 m/s. At 0.5 m/s, 22/30 demos
still need more (G6's finding that 0.5 does not fix it). w_max 1.0 rad/s binds only for 2 PICK_DOUBLE demos. The cone
is fine for PICK_TEST1 (max 59.5 deg < 60) and costs one PICK_DOUBLE demo (ep 37, day 2). max_lead cannot fire in this
model.

**Box refusals have two causes.** On z: the floor artifact. A demo that started higher than 16 cm above its grasp is
carried into the table (14/30 at 16 cm, 6/30 at 22 cm, all in the approach); closed loop, the policy aims at the
eraser and would not do that, so these are not counted against a pose. On x: the retreat after the grasp moves toward
the robot and crosses the face x 0.20 m. That one is real (the policy reproduces the demos' retreat) and depends only
on how far out the region is: 21 / 10 / 3 of 30 refused at 0.45 / 0.50 / 0.55 m.

**Pose ranking.** Height changes only the floor artifact (9 vs 17 refusals with the floor for PICK_TEST1) and a
little IK margin (0.085 vs 0.041 rad); it does not change x, cone or speed. Distance out changes both the x refusals
and the IK: at 0.45 m 6/30 demos are unreachable along the path (joint 4 at its limit), at 0.55 m none. So
`leftx55_t48_h16` and the 22 cm alternative are the best of the six on everything that is real; 22 cm has 8 fewer
floor-artifact refusals and a larger PICK_TEST1 IK margin, 16 cm puts the median carried grasp on the table.

**Day 2 (PICK_DOUBLE eps 30-61) is harder** than day 1: of the 14 no-floor refusals at the chosen pose 11 are day 2
(x eps 33, 36, 39, 40, 42, 45, 46, 47, 48, 49 and cone ep 37), and all 8 IK-unreachable demos are day 2. Day 2's
retreats reach further toward the robot (min x 0.142 m, ep 40) and its poses closer to joint 4's limit.

**Docs vs raw data:**

1. doc says "box refused (first refusal) | ... | 36 (approach 29, retreat 5, lift 1)" at `sim_deploy/DESIGN.md`
   "Screen and full eval", line 951; raw says 36 refused demos, of which 35 first refused by the box (29 + 5 + 1) and
   1 by the cone (ep 37, lift, tilt 62.4 deg) at `sim_evals/_start_pose_sweep/gate/gate_h16.json`
   (`datasets."pick1-double-v2.zarr.zip".summary.first_refusal_by_rule_phase`). The row label "box refused" is off
   by the one cone refusal.
2. doc says "box refused without the floor artifact | ... | 14 (retreat 12, lift 1; ...)" at DESIGN.md line 952; raw
   says 14 refused = 13 box (12 + 1) + 1 cone (ep 37) at `gate_h16_nofloor.json`. Same cause.
3. doc says "box refused 9/30 ... / 21/62" for the 22 cm alternative at DESIGN.md line 963; raw says 21 refused = 20
   box (approach 7, retreat 12, lift 1) + 1 cone at `gate_h22.json`.
4. doc says "v_max 0.25: demos shortened | 30 (all in the approach; also lift 19, retreat 23)" at DESIGN.md line 955;
   raw also has grasp 1 (with the floor) / grasp 9 (without) at `gate_h16.json` / `gate_h16_nofloor.json`
   `shortened_demos_by_phase`. Omission only.
5. doc says "the cone, speed and w rows are the same" for 22 cm at DESIGN.md line 962 (true) and gives no 22 cm IK
   number for PICK_DOUBLE; raw says 9 unreachable (8 at 16 cm), p10 0.007 rad at `gate_h22.json`. Omission only.
6. doc says "the floor -0.005 is needed (0.05 would refuse every grasp)" at DESIGN.md line 968; raw (open-loop carry)
   says 24/30 carried grasps are below z 0.04 m (the refusal line for a 0.05 m floor with the 10 mm margin) at
   `sim-experiments/01_start_pose/data/p4gate_grasp_z.csv` (computed from the datasets and the chosen pose's FK); the
   other 6 are carried grasps 4.3-7.7 cm above the table (eps 18, 20, 21, 24, 25, 29) because their demos started
   lower than 16 cm (25/30 carried paths go below 0.04 m somewhere: ep 20 dips to 0.038 m later). "Every grasp" holds
   for a real grasp at the eraser (pinching the 12 mm eraser puts the fingertip TCP at the table, `control_sim.yaml`
   lines 14-19), not for the carry. Also: z 0.05 m is only in the ea9f7fb copy; current upstream already has -0.005 m.
7. doc says "the demos are 4-5x faster than 0.25 in short bursts" at DESIGN.md line 966; raw says 4.8x for 28/30 and
   5.1x for the worst step with the bucket's 10 ms slack (`v_needed` 1.211 / 1.274 m/s), 6.1x over the bare 50 ms step
   (1.53 m/s), and 2.7x for the median demo (0.67 m/s) at `gate_h16.json` per-demo `worst.v_needed_m_per_s`. The
   "4-5x" is the 28/30 value, not typical.
8. doc says "29 of 30 demos have 50 ms segments above 0.5 m/s before the grasp (peaks 0.6-1.5 m/s)" at DESIGN.md
   line 879 and PLAN.md §6 P3 G6; raw here says 24/30 demos have a 50 ms step above 0.5 m/s on the gate's 50 ms grid
   (all peaks in the approach, max 1.53 m/s) at `gate_h16.json`. Different sampling (G6's count is from another tool,
   see [00_integration_checks](../00_integration_checks/README.md)); not necessarily a contradiction.
9. doc says "Left heading: 30/30 reachable, margin >= 0.25 rad (0.47 at `leftx55_t48_h16`)" at DESIGN.md line 929
   (`demo_paths.py`: IK of the grasp and end pose only); raw IK along the whole path says `left_t48_h16` 6/30
   unreachable, min 0.001 rad, and `leftx55_t48_h16` min 0.041 rad (ep 28) at `gate_left_t48_h16.json` /
   `gate_h16.json`. Different measurement; the along-path one is the stricter.
10. doc says "line-by-line Python port of `safety_gate.cpp`" at DESIGN.md line 942; raw code says the port omits the
    width rule (C++ lines 90-92), the finite check and skipped waypoints, and starts each request at W(t_c) instead of
    `plan_.at(now)` (`supervisor.cpp` line 443). The rules it does model match.
11. doc says "Real robot: x 0.25, z 0.05 (upstream, "TODO: measure")" for the workspace box min and "Upstream
    final-deployment still says 150 mm" at `sim-experiments/README.md` "Sim-only settings vs the real robot" and
    "Frames and poses"; current upstream says [0.20, -0.40, -0.005] at
    `~/Capstone-final-deployment/deployment/control_deploy/franka_control/config.yaml` line 43 (447910b) and F_T_TCP
    z 0.23565 m at `~/Capstone-final-deployment/deployment/contracts/frames.yaml` line 10 (927a5ba). Those old values
    are only in the ea9f7fb copy here.
12. `sim_deploy/DIRECTIONS.md` §2c does not list `python -m sim_deploy.scene.gate_check`; the command exists only in
    DESIGN.md and the module docstring. A doc gap, not a number.

Every other DESIGN cell matches the raw value (table above).

## Conclusion

- At the chosen pose `leftx55_t48_h16` the gate, without the floor artifact, refuses 3 of 30 PICK_TEST1 demos (all on
  the retreat, x face 0.20 m) and 14 of 62 PICK_DOUBLE demos (13 box on x, 1 cone); all 30 PICK_TEST1 demos are
  reachable along the path (min margin 0.041 rad). The eraser 0.55 m out beats 0.50 m and 0.45 m on both refusals
  (3 / 10 / 21 of 30) and IK (0 / 1 / 6 unreachable); 22 cm is equal on everything real and has fewer floor-artifact
  refusals.
- For the real PC: the start pose itself is never refused. v_max 0.25 m/s is the binding rule (every demo shortened,
  1.21 m/s needed for 28/30, 0.5 m/s not enough for 22/30). The cone 60 deg is enough for PICK_TEST1 (max 59.5 deg).
  Box x 0.20 m clips 3/30 retreats; x 0.15 m would refuse none of either dataset (PICK_TEST1 min x 0.159 m, PICK_DOUBLE
  0.142 m pulled back). The floor must stay at -0.005 m (current upstream, user's station); the ea9f7fb copy's 0.05 m
  would refuse a real grasp at the eraser. The copy's x 0.25 m would refuse 10/30 retreats.
- Nothing was changed because of this record.

## Limitations / sim-vs-real caveats

- Open loop with perfect tracking: each request starts on the demo path. The real PC starts at `plan_.at(now)`, which
  lags after any shortening; the policy is closed loop and can re-aim. The floor refusals are an artifact of the carry.
- No inference latency, so no skipped waypoints and a full first 50 ms step; on the real PC the waypoints that are
  already past are skipped (C++ lines 69-72).
- Gripper width not checked: training widths go up to 0.090 m (handheld), the PC refuses > 0.08 m unless the laptop
  clips first.
- IK is the sim model's DLS (`trainstart.solve_ik`, 150 iterations, tolerance 1 mm / 0.5 rad), warm-started at 10 Hz;
  it says nothing about self-collision or table contact along the path, and "unreachable" may be a solver miss.
- The demos are handheld recordings: speed bursts may be hand jerks or SLAM tracking jumps (DESIGN line 880), so the
  "v needed" values are an upper bound for what a policy will ask.
- min x and the grasp-z values are on a 50 ms grid and rounded to 1 mm; the box estimates for other x_min values are
  from those (one demo, ep 45, sits on the 10 mm margin).

## Open questions

1. Does `policy_deploy` clip widths to 0.08 m before C3, or would the real gate refuse a 0.090 m width? Answer: read
   the copy's action-chunk code or send one chunk with a 0.09 m width to the sim PC.
2. How much of the v_max loss does a closed-loop policy recover? Answer: a learned policy run from the start pose once
   the IMX219 render is fixed ([02_camera_study](../02_camera_study/README.md)), or a gate port that starts each
   request at the lagging target (`plan_.at(now)`) with the policy's real latency.
3. Should the real box x min go to 0.15 m? Answer: measure what is behind the robot-side face on the real table; the
   data here say 0.15 m clears every retreat of both datasets.
4. Why is day 2 of PICK_DOUBLE harder (11 of 14 refusals, all 8 IK misses)? Answer: compare the two days' start and
   retreat poses relative to the eraser (episodes 30-61 vs 0-29) in the datasets.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| gate results (8 JSON, 8 TXT; 96-312 KB JSON, not copied) | `sim_evals/_start_pose_sweep/gate/gate_{h16,h16_nofloor,h22,h22_nofloor,left_t48_h16,left_t48_h22,leftx50_t48_h16,leftx50_t48_h22}.{json,txt}` |
| pose files (q of each id) | `sim_evals/_start_pose_sweep/poses/{leftx55,leftx50,left}_t48_h{16,22}.yaml`, `sim_deploy/stations/start_pose.yaml` |
| the tool | `sim_deploy/scene/gate_check.py` |
| the C++ it ports | `deployment/control_deploy/franka_control/src/safety_gate.cpp`, `src/supervisor.cpp` (line 443), `config.yaml` (lines 44-54); upstream `~/Capstone-final-deployment/deployment/control_deploy/franka_control/` @ 447910b |
| datasets | `~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip`, `~/data/umi/datasets/multi/pick1-double-v2/pick1-double-v2.zarr.zip` |
| copied into this folder | `data/p4gate_gate_*.txt` (the 8 TXT summaries, unchanged); `data/p4gate_summary.csv` (one row per file and dataset, from the JSON summaries); `data/p4gate_per_demo.csv` (one row per file, dataset and demo: first refusal and axis, refused axes, phases, worst values, IK); `data/p4gate_grasp_z.csv` (carried TCP z at each demo's grasp start / end row and the path minimum, 16 and 22 cm) |
