# G6: conventions end to end (open-loop replay of training episodes)

**Status:** done: E7 fail (superseded); P3 pass for the conventions, open-loop replay fails on fast demo segments ·
**Date:** 2026-11-20 · **Who:** agent (E7, P3), documented by agent · **Plan ref:** sim_deploy/PLAN.md §6 (G6 row,
P3 table); DESIGN.md "E7: integration" (fix 5, open item 1), "P3/P4" -> "P3 results", "Fix log (P3/P4)" 3 and 5

`$S` below = `/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad`, the session
scratchpad of the E7 and P3 agents: **temporary, not in sim_evals**. Small files from it are copied into
[data/g6/](data/g6/).

## Question

Played open loop through the whole chain (laptop `policy_deploy` with the `replay` policy -> C3 -> the sim PC's gate
and controller), does a recorded PICK_TEST1 episode move the sim TCP along the recorded path (frames, axes and
relative poses right) and close the fingers on the eraser placed as in that episode?

## Why it matters for the real robot

The learned policies output the same kind of relative action chunks as the replay. If the replay path were rotated,
mirrored or scaled, every policy would be wrong on the Franka in the same way. G6 is the check that the frames
(`F_T_TCP` 235.65 mm, the anchor convention, the width = fingertip gap) line up end to end. It also measures what the
real gate's `v_max` 0.25 m/s does to human-speed motions, which applies to the robot unchanged.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (eval.yaml `git.sim_deploy` `3633fcb3…-dirty`; + uncommitted `sim_deploy/`, `deployment/`) |
| Model | E7: `701815be08cd…` (old table top O z 0.05). P3: `166a0649571a…` (table top O z 0) in all 7 P3 G6 eval.yaml files |
| Home / start pose | E7: `--home deploy`, TCP O (0.3069, 0.0000, 0.3542). P3: `--home start`, P3 stand-in q `[-0.6419329, 0.3706554, -0.1347388, -1.8275169, 0.5918946, 1.4884701, -1.6761742]`, TCP O (0.4385, -0.1641, 0.1618), 16.2 cm above the table, tilt 49.5 deg, heading left |
| Eraser placement | the replayed episode's own placement (eval.yaml `scene.replay_placement`: the demo's grasp pose carried to the episode's start TCP) |
| Stations / overlays | E7: `stations/control_sim.yaml`, box in force min x 0.250 / z 0.050 m (from the gate's texts). P3: `control_sim.yaml` + `start_pose.yaml`, box min [0.20, -0.40, -0.005], max [0.80, 0.40, 0.70] m; the "vmax05" runs add the sim-only overlay `stations/vmax05_g6.yaml` (`safety.v_max_m_per_s: 0.5`) via `SIM_DEPLOY_SIM_PC_OVERLAY` |
| Policy | `replay:/home/ruben/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip#<i>`, i = 2 (E7); 19, 0, 28, 16 (P3: catalog `grasp_ok` episodes from the start pose are 0, 7, 14, 16, 19, 23, 28, `sim_deploy/scene/README.md`) |
| Seeds / episodes | 1 episode each, seed 0 (the placement is the replay's, not random) |
| Timeout | E7 40 s; P3 25 s |
| Sim-only settings | `v_max` 0.5 m/s in the three vmax05 runs; box min (P3); finger contact pairs; width offset 0.0 |

## Method

**Replay policy** (`deployment/policy_deploy/policy_deploy/policy/replay_policy.py`, E1, cursor fixed by E7 fix 5):
at each infer it returns the next `exec_h` steps of the recording relative to the recording's row at the cursor,
c = s + round(t * d / 0.05 s), t = time since the episode's first infer. Before fix 5 the cursor advanced exec_h
rows per infer while the manager replans every ~33 ms, so the replay ran ~12x too fast. The fixed file is dated
01:40, before every replay here.

**Pass (PLAN §6):** the TCP follows the recorded path and the grasp closes on the eraser.

**Measurements.**

- *Path error* (P3, `sim_evals/_start_pose_sweep/tools/g6_path.py EVAL_DIR`, read-only, prints JSON): expected path
  O_T(t) = O_T_start · inv(P_0) · P_(60 t) (the recording carried to the episode's first TCP, 60 Hz); actual = the
  driver's 20 Hz ground-truth TCP in `trace.json` (`tcp_O_xyz_wxyz`, P3 fix 3). A time lag in -0.3..1.0 s is fitted
  to minimise the median position error; reported: time-aligned median / p95 / max position error (mm), rotation
  error (deg), *path shape* error (each expected point up to the grasp -> nearest actual point), and the error at the
  recorded grasp (the first closing row from `placement.grasp_rows`, "start"). PLAN's "closes N mm short of the
  grasp" is this error at the recorded grasp.
- *E7 ep 2* has no TCP column in `trace.json` (added later); the same comparison from the 100 Hz C2 recording
  `$S/e7/g6_c2.jsonl`, lag 0: [data/g6/e7_ep2_c2_path.py](data/g6/e7_ep2_c2_path.py).
- *Shortened waypoints*: episodes.jsonl `c2.gate.shortened` (the PC's count over the episode); the sim_pc.log lines
  name the rule ("shortened: 0.0328 m in 0.050 s -> 0.0300 m (v_max 0.50 m/s)").
- *Lift*: episodes.jsonl `max_lift_m` (eraser centre above its settled start). Outcome words: sim-experiments/README.md.
- *Demo speeds*: [data/g6/demo_speed.py](data/g6/demo_speed.py): per demo, the TCP speed over 50 ms (3 rows at 60 Hz,
  sliding) from the start to the grasp start -> [data/g6/demo_speed.csv](data/g6/demo_speed.csv).

## Commands to reproduce

Each run needs the sim ports free and no other sim running; ~1 min each.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
Z=/home/ruben/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip
# P3, real v_max 0.25 m/s, from the start pose (now the P4 default start pose: set the P3 stand-in in
# stations/start_pose.yaml or pass --start-pose <file> to reproduce exactly)
for i in 28 19 16 0; do sim_deploy/run.sh --policy "replay:$Z#$i" --episodes 1 --tag p3_g6_vmax025 --timeout 25; done
# P3, sim-only v_max 0.5 m/s
for i in 28 19 16; do SIM_DEPLOY_SIM_PC_OVERLAY=sim_deploy/stations/vmax05_g6.yaml \
  sim_deploy/run.sh --policy "replay:$Z#$i" --episodes 1 --tag p3_g6_vmax05 --timeout 25; done
# E7 (deploy home): sim_deploy/run.sh --home deploy --policy "replay:$Z#2" --episodes 1 --tag g6 --timeout 40
```

Analysis (read-only, no sim):

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH; export PYTHONDONTWRITEBYTECODE=1
~/envs/sim_deploy/bin/python -B sim_evals/_start_pose_sweep/tools/g6_path.py sim_evals/_replay/PICK_TEST1/pick1-v1_ep019/2026-11-20_0947_p3_g6_vmax025
~/envs/sim_deploy/bin/python -B sim_evals/_start_pose_sweep/tools/g6_time.py <EVAL_DIR>      # expected vs actual at fixed times
~/envs/sim_deploy/bin/python -B sim-experiments/00_integration_checks/data/g6/demo_speed.py
~/envs/sim_deploy/bin/python -B sim-experiments/00_integration_checks/data/g6/e7_ep2_c2_path.py $S/e7/g6_c2.jsonl
```

## Results

### Outcomes and gate counts (raw: each eval's `episodes.jsonl`, `metrics.json`)

| Eval (`sim_evals/_replay/PICK_TEST1/...`) | v_max (m/s) | Outcome | Grasp t (s) | Max lift (cm) | Shortened / pulled back / rejected | Steps sent | Fault (s into the episode) |
|---|---|---|---|---|---|---|---|
| `pick1-v1_ep002/2026-11-20_0147_g6` (E7, deploy home) | 0.25 | fault | - | 0.0 | 178 / 5 / 1 | 154 | 5.58: "outside workspace box by 0.012 m on x (x = 0.238 m < min 0.250 m)" |
| `pick1-v1_ep019/2026-11-20_0947_p3_g6_vmax025` | 0.25 | **success** | 2.31 | 17.3 | 40 / 0 / 0 | 114 | - |
| `pick1-v1_ep000/2026-11-20_0954_p3_g6_vmax025` | 0.25 | **success** | 3.26 | 19.9 | 47 / 0 / 0 | 152 | - |
| `pick1-v1_ep028/2026-11-20_0946_p3_g6_vmax025` | 0.25 | fault | - | 0.0 | 169 / 46 / 1 | 166 | 5.51: "outside workspace box by 0.011 m on x (x = 0.189 m < min 0.200 m)" |
| `pick1-v1_ep016/2026-11-20_0947_p3_g6_vmax025` | 0.25 | timeout | - | 0.0 | 168 / 0 / 0 | 749 | - (25 s) |
| `pick1-v1_ep019/2026-11-20_0951_p3_g6_vmax05` | 0.5 | **success** | 2.31 | 16.2 | 14 / 0 / 0 | 114 | - |
| `pick1-v1_ep028/2026-11-20_0949_p3_g6_vmax05` | 0.5 | fault | - | 0.0 | 37 / 14 / 1 | 155 | 5.16: "outside workspace box by 0.011 m on x (x = 0.189 m < min 0.200 m)" |
| `pick1-v1_ep016/2026-11-20_0953_p3_g6_vmax05` | 0.5 | timeout | - | 0.0 | 45 / 0 / 0 | 748 | - (25 s) |

v_max in force confirmed in the sim PC logs: `_logs/2026-11-20_094921/sim_pc.log` has 22 "(v_max 0.50 m/s)" lines,
`_logs/2026-11-20_094619/sim_pc.log` 50 "(v_max 0.25 m/s)" lines. eval.yaml does not record the overlay.

### Path error (raw: `g6_path.py` output, [data/g6/g6_path_output.jsonl](data/g6/g6_path_output.jsonl))

| Episode, v_max | Lag (s) | Pos. error median / p95 / max (mm) | Rot. error median / max (deg) | Path shape to grasp median / max (mm) | Error at recorded grasp (mm) [x, y, z] | Recorded grasp at (s) | Fingers < 0.070 m at (s) |
|---|---|---|---|---|---|---|---|
| 19, 0.25 | -0.04 | 8.8 / 15.7 / 21.4 | 3.56 / 5.62 | 4.8 / 13.2 | 14.5 [2.1, -9.7, 10.5] | 1.87 | 2.06 |
| 19, 0.5 | 0.02 | 9.2 / 13.5 / 14.8 | 3.78 / 6.16 | 4.1 / 10.4 | 11.4 [2.6, -5.8, 9.4] | 1.87 | 2.10 |
| 0, 0.25 | -0.10 | 16.8 / 33.8 / 36.3 | 5.17 / 8.97 | 3.9 / 16.6 | 26.6 [-0.5, -14.1, 22.5] | 2.75 | 3.01 |
| 28, 0.25 | -0.06 | 80.7 / 152.7 / 154.3 | 10.02 / 11.80 | 7.0 / 80.7 | 81.9 [2.0, -51.0, 64.0] | 1.88 | 2.05 |
| 28, 0.5 | -0.04 | 80.1 / 152.6 / 154.9 | 10.29 / 11.79 | 9.0 / 79.7 | 80.1 [3.1, -52.2, 60.7] | 1.88 | 2.06 |
| 16, 0.25 | -0.22 | 93.6 / 132.5 / 133.9 | 6.79 / 13.78 | 18.0 / 87.8 | 106.1 [7.9, -78.2, 71.3] | 1.83 | 2.00 |
| 16, 0.5 | -0.24 | 89.8 / 127.2 / 129.8 | 6.57 / 13.88 | 15.1 / 85.6 | 106.6 [6.3, -77.7, 72.7] | 1.83 | 2.01 |

E7 ep 2 (C2, lag 0; [data/g6/e7_ep2_c2_path.py](data/g6/e7_ep2_c2_path.py)): at the recorded grasp (row 109, 1.82 s)
expected TCP O (0.2723, 0.0133, 0.0461), actual (0.2779, -0.0005, 0.1600): error 114.9 mm, of which +113.9 mm in z
(above). Fingers below 0.070 m at 1.98 s with the TCP 10.8 cm above the old table; C2 width min 0.0563 m (closed
in the air on the commanded width); lowest TCP z 0.1584 m. The fault is a rejected *waypoint* at x 0.238 m; the TCP
itself never went below x 0.2771 m.

### Demo speeds before the grasp (raw: [data/g6/demo_speed.csv](data/g6/demo_speed.csv), PICK_TEST1 pick1-v1, 30 demos)

- 29 of 30 demos have at least one 50 ms segment faster than 0.5 m/s before the grasp; only **episode 0** does not
  (peak 0.483 m/s).
- Peaks of the other 29: 0.514 (ep 29) to 1.529 m/s (ep 6); ep 19 0.665, ep 28 1.096, ep 16 1.167 m/s.
- Segments faster than the real gate's 0.25 m/s before the grasp: 3-37 per demo (ep 0: 4, ep 19: 4, ep 28: 19, ep 16: 25).
- In every demo the episode's fastest 50 ms segment lies before the grasp.

## Analysis

- **Conventions: pass.** Where the gate does not cut the motion, the replay lands where the recording says: ep 19
  follows its path to 8.8 mm median (4.8 mm path shape) and grasps 14.5 mm from the recorded grasp, ep 0 16.8 mm /
  26.6 mm; both grasp and lift (17.3 and 19.9 cm). Axes, signs, the anchor convention, the TCP offset and the width
  are right; a frame error would show as centimetres to decimetres from the first second on.
- **Open loop fails where the demo is faster than the gate.** Each chunk is relative to the manager's anchor (its
  previous target), so a waypoint the gate shortens is never made up. Ep 28 and ep 16 lose 80-107 mm by the
  recorded grasp, mostly in y and z (the TCP stays high and to one side), close on nothing, and ep 28 then backs out
  of the box (rejected waypoint x 0.189 m). Raising `v_max` to the sim-only 0.5 m/s cuts their shortened waypoints
  from 169 / 168 to 37 / 45 but leaves the error the same (80.1 / 106.6 mm): the remaining fast segments (ep 28 peak
  1.10, ep 16 1.17 m/s) and the early losses are enough. Ep 19 at 0.5 m/s improves at the grasp (14.5 -> 11.4 mm).
  Ep 0, the only demo with no segment above 0.5 m/s, succeeds at the real 0.25 m/s.
- **E7 ep 2 (deploy home, old table): fail for the same reason, plus the old box.** 178 waypoints shortened in 154
  steps, the TCP was 11.4 cm above the recorded grasp point when the fingers closed, and the replay's retreat then
  asked for x 0.238 m, outside the copy's box (min x 0.25). The deploy home also starts the demo from a pose it was
  never recorded from (fingers straight down), which P2/P3 replaced with the start pose.
- **What this means for policies.** A closed-loop policy sees where the arm actually is and can re-aim, so the
  shortening is not fatal for it; but it was trained on the same fast motions, so expect its chunks to be shortened
  too. The real `v_max` stays 0.25 m/s (the copy: "raise to 0.5 only after tuning on the arm").

### Docs vs raw data

1. PLAN §6 P3 G6 "29 of 30 demos have 50 ms segments faster than 0.5 m/s before the grasp (peaks 0.6-1.5 m/s)": raw
   peaks of those 29 are 0.514-1.529 m/s; five are below 0.6 (ep 29 0.514, ep 25 0.573, ep 22 0.580, ep 24 0.582,
   ep 21 0.584) (`data/g6/demo_speed.csv`, computed here with sliding 3-row segments; the P3 agent's own
   computation is not in sim_evals).
2. DESIGN.md "P3/P4" -> "P3 results" "28 and 16 close 8-11 cm short of their grasp": raw 80.1-81.9 mm (ep 28) and
   106.1-106.6 mm (ep 16) at the recorded grasp, i.e. 8-11 cm; agrees. The word "short" is the 3-D error at the
   recorded grasp time (mostly y and z), not a distance along the path.
3. PLAN §6 G6 (E7) "the TCP stopped 11 cm above the grasp point and closed in the air (C2 0.056 = commanded)": raw
   +113.9 mm in z at the recorded grasp, C2 width min 0.0563 m (`$S/e7/g6_c2.jsonl`; not in sim_evals).
4. PLAN §6 G6 (E7) "then left the old box (x 0.238)": the TCP did not leave the box; the gate rejected a waypoint
   at x 0.238 m (the TCP's minimum x was 0.2771 m) (`0147_g6/episodes.jsonl`, `$S/e7/g6_c2.jsonl`).
5. PLAN §6 G6 (E7) "Grasp itself checked offline: 5 N/pad at 0.054, no slip": no raw file found (`$S/grasp/` holds
   only two camera JPEGs); (from DESIGN "Fin-ray compliance (sim)", not in sim_evals).
6. sim-experiments/README.md "Each eval's eval.yaml records the merged station": eval.yaml records only the path
   `sim_deploy/build/control_sim_start.merged.yaml` (rewritten by every later run) and not the overlay; v_max 0.5 is
   visible only in `sim_evals/_logs/<time>/sim_pc.log`.
7. sim-experiments/README.md model versions "`7746a439…` (E7, D8 tags)": E7's `0147_g6/eval.yaml` says `701815be08cd…`.

## Conclusion

The conventions are right end to end: from the start pose, replays of episodes 19 and 0 follow their recorded paths
to 8.8 / 16.8 mm (median) and grasp and lift the eraser. Open-loop replay of the other demos fails because 29 of 30
demos contain 50 ms bursts faster than 0.5 m/s, the gate shortens them (at 0.25 and at 0.5 m/s), and an open-loop
replay never recovers the lost distance. G6 is therefore a pass for what it is meant to check (frames), and the
speed finding is carried to G7 and the camera study (PLAN §10 A1 uses a sim-only v_max 1.5 for replays). The E7
result (deploy home, old table, old box) is superseded by P3.

## Limitations / sim-vs-real caveats

- Contact and friction of the fin-ray pads are a stand-in (explicit contact pairs, 3.5 mm margin); a replay that
  grasps in the sim might slip on the robot and vice versa.
- Only 4 of the 7 `grasp_ok` demos were replayed in P3, one episode each.
- The 20 Hz trace limits time resolution (50 ms); the lag fit has 20 ms steps.
- The demos' fast bursts may partly be tracking jumps of the handheld rig rather than real hand motion (P3's
  guess, not checked).

## Open questions

1. Are the 0.5-1.5 m/s bursts real motion or SLAM/tracking jumps? Inspect the IMX219 frames around those rows.
2. Do the other `grasp_ok` demos (7, 14, 23) behave like 19 / 0 or like 28 / 16? Three more replays answer it.
3. How do learned policies' chunks fare under the gate (shortened counts per episode)? See [G7_policies.md](G7_policies.md)
   and the camera study.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| eval folders | `sim_evals/_replay/PICK_TEST1/pick1-v1_ep002/2026-11-20_0147_g6/` (E7); `pick1-v1_ep019/2026-11-20_0947_p3_g6_vmax025/`, `pick1-v1_ep000/2026-11-20_0954_p3_g6_vmax025/`, `pick1-v1_ep028/2026-11-20_0946_p3_g6_vmax025/`, `pick1-v1_ep016/2026-11-20_0947_p3_g6_vmax025/`, `pick1-v1_ep019/2026-11-20_0951_p3_g6_vmax05/`, `pick1-v1_ep028/2026-11-20_0949_p3_g6_vmax05/`, `pick1-v1_ep016/2026-11-20_0953_p3_g6_vmax05/` (each has `episodes/000/external.mp4`) |
| logs | `sim_evals/_logs/2026-11-20_094619/` (28), `094700/` (19), `094739/` (16), `094921/` (28, 0.5), `095138/` (19, 0.5), `095300/` (16, 0.5), `095438/` (0); E7 0147: no `_logs` folder, last sim PC session in `$S/e7/sim_logs/sim_pc.log` |
| tools | `sim_evals/_start_pose_sweep/tools/g6_path.py`, `g6_time.py`; overlay `sim_deploy/stations/vmax05_g6.yaml`; dataset `/home/ruben/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip` |
| scratchpad (temporary) | `$S/e7/g6_c2.jsonl` (6 MB, E7 C2 at 100 Hz), `$S/p34/g6_*.log`, `$S/p34/g6b_*.log` |
| copied into this folder | `data/g6/g6_*.log`, `g6b_*.log` (run.sh output, from `$S/p34/`), `g6_path_output.jsonl` (g6_path.py output), `demo_speed.py` + `demo_speed.csv`, `e7_ep2_c2_path.py` (computed here) |
