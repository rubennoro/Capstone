# 01_start_pose: the start pose for the PC key `p` (choose `q_start_rad`)

**Status:** done / settled (adopted upstream in rnoro/final-deployment `abba9e8`, to be checked on the arm) · **Date:** 2026-11-20 ·
**Who:** agent (coordinator + six part writers) · **Plan ref:** sim_deploy/PLAN.md §6 ("P3" table), §9, §10, §11; DESIGN.md "E7" -> "Start pose variant `--home trainstart`",
"P1", "P2", "P3/P4" (all subsections)

This experiment is in six parts. Each part file follows [TEMPLATE.md](../TEMPLATE.md) with its own Setup, Commands,
Results and "Docs vs raw data". This README gives the overview, the index, the overall conclusion and the
consolidated list of disagreements between the docs and the raw files. All paths are relative to `~/Capstone-main-sim/`.
Every number was taken from the raw files under `sim_evals/` (or from code) and spot-checked by the coordinator
against them.

## Question

Which `homing.q_start_rad` should the PC's `p` key move the arm to before an episode: one the real PC accepts, the
cameras see the eraser from, and the demos' relative motions can be carried from without the gate refusing them?
And can the sim rank candidate start poses by policy success?

## Why it matters for the real robot

`q_start_rad` was `null` in final-deployment when this work started (it is now set, `abba9e8`). While it is null, `p` is refused; once it is set, C3 `start` is refused
unless every joint is within 0.05 rad of it. So it decides where every real episode begins. The policies only see
motion relative to the episode start, so the start pose also decides whether the demos' motions stay inside the real
gate's box and cone and inside the arm's joint limits. The user can either paste the 7 numbers below or hand-guide the
arm and read them with `pc_probe --start-pose`.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Model | `sim_deploy/build/sim_rig/model.json` `model_sha256` `166a0649571a…` (table top = O z 0, base on the table, no pedestal); the one superseded run used `7746a439…` (table z 0.05) |
| Home / start pose | `--home start`: PC `h` -> PC `p` to `sim_deploy/stations/start_pose.yaml` (now the chosen pose `leftx55_t48_h16`) |
| Eraser placement | `sim_deploy/scene/randomization_start.yaml`, relative to the start pose (ahead 0.090-0.217 m, left -0.035..-0.003 m, yaw_rel -6.6..+0.8 deg) |
| Stations / overlays | `stations/control_sim.yaml` (sim box [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70]) + `start_pose.yaml`; screen only: overlay `stations/homing_fast_sweep.yaml` (homing 0.6 rad/s) |
| Policy / checkpoint | `PICK_TEST1/run_006_2026-11-18` (behavior prompting, `pick1-v1`, epoch 59) for G7 and the screen; recorded demos of PICK_TEST1 (30) and PICK_DOUBLE `pick1-double-v2` (62) for the offline checks |
| Inference settings | bf16, 8 diffusion steps, `exec_h` 12 (closed loop); offline first chunk fp32 on the CPU |
| Seeds / episodes | G7: 3 episodes; screen: seed 0, 1 episode per pose |
| Timeout | G7 60 s; screen 25 s |
| Sim-only settings | sim box floor -0.005 (also current upstream); `--fast-homing` (screen only); `--no-floor` in `gate_check` (offline only) |

## Method

1. Port the real `p` key into the sim's copy and wire a sim-only scene `key` command. Make the eval flow
   `h` -> `p` -> eraser -> `c`. Use a stand-in start pose from the demos ([part 1](1_p_key_and_standins.md)).
2. Generate 63 candidate start poses (tilt x height x heading, plus two shifted left layouts). Check each one offline:
   limits, box, cone, contacts, joint margin, eraser region in both cameras, the `p` path, and the real PC's rules.
   Carry the 30 demos from each feasible pose ([part 2](2_candidates_envelope.md)).
3. Screen the feasible poses with run_006 for early failures ([part 3](3_early_failure_screen.md)).
4. Push the demos from the best layouts through a Python port of the real safety gate
   ([part 4](4_gate_check.md)).
5. Write the handoff for the real PC ([part 5](5_real_pc_handoff.md)).
6. Explain why the policy screen cannot rank poses: the offline first-chunk camera swap
   ([part 6](6_camera_swap.md)).

## Commands to reproduce

Each part has its own commands. They write into `sim_evals/` or `sim_deploy/stations/`. Run the sim chain parts
(3, and G7 in 6) only when no other sim or GPU job is running (ports 5555/5610/5620/5630/5640/5650/5651/8770, 4 GB GPU).

## Results

### Index of the parts

| # | Part | Finding (one line) |
|---|---|---|
| 1 | [The key `p` and the stand-ins](1_p_key_and_standins.md) | The ported `p` (home only, joint move, fingers 0.08 m, box + cone check at the end, refused while null or if the move would take >= 30 s; C3 `start` needs every joint within 0.05 rad) runs unchanged in the sim. The stand-ins were: E7 trainstart stand-in (16.18 cm above the old z 0.05 table, 49.51 deg), then the P3 stand-in (the same IK for the z 0 table, reproduced to 7 decimals), then the chosen pose `leftx55_t48_h16`. |
| 2 | [Candidates and the feasibility envelope](2_candidates_envelope.md) | 40 of 63 candidates are feasible (26 of 45 in the grid, 14 of 18 shifted). The fails: IMX519 view at 38 deg / 10 and 16 cm (14 poses); joint 7 for heading right (5); joints 4/5 or no IK for heading away (8). With the left heading, moving the eraser out from 0.45 to 0.55 m cuts the retreat crossings from 21 to 3 of 30. |
| 3 | [Early-failure screen](3_early_failure_screen.md) | 23 of 26 grid-feasible poses screened: no grasp or approach (closest 10.5 cm), 18 faults (box 11, cone 7) and 5 timeouts, one early fault (cone, 9.77 s). The screen cannot rank poses, and the chosen pose was never screened (the `leftx*` poses were added after the screen started). |
| 4 | [The demos through the real gate](4_gate_check.md) | At `leftx55_t48_h16`, without the floor artifact, the gate refuses 3 of 30 PICK_TEST1 demos and 14 of 62 PICK_DOUBLE demos (13 box, 1 cone), and IK reaches all 30 along the path. v_max 0.25 m/s is the binding rule (1.211 m/s needed for 28/30). At 0.45 / 0.50 / 0.55 m out: 21 / 10 / 3 refusals and 6 / 1 / 0 IK-unreachable demos. |
| 5 | [Real PC handoff](5_real_pc_handoff.md) | Adopted upstream in `abba9e8`, to be checked on the arm. `q_start_rad: [-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]`, TCP O (0.531, -0.159, 0.160) m, 48.0 deg, pointing to the robot's left. It passes the real station with both TCPs; margins: joints 1.241 rad, box 0.165 m, cone 12.0 deg, `p` 14.47 s of 30 s. |
| 6 | [Camera swap](6_camera_swap.md) | run_006 rises and backs off from the P3 stand-in (G7: 3 faults). Offline, the first chunk goes +5.6 cm toward the fingertips on the training images and -1.3 cm on the sim images; +5.4 cm with only the training IMX219, -0.5 cm with only the sim IMX219. So the sim IMX219 render is what the policy rejects. Followed up in [02_camera_study](../02_camera_study/README.md). |

### The three start poses used (part 1, part 5)

| Pose | q (rad) | TCP O (m) | Height above table | Tilt | Heading | Source |
|---|---|---|---|---|---|---|
| E7 trainstart stand-in | `[-0.6225485, 0.3138391, -0.1643961, -1.7825973, 0.5934631, 1.3978677, -1.6275796]` | (0.4385, -0.1641, 0.2118), old table z 0.05 | 16.18 cm | 49.51 deg | left | `sim_deploy/scene/trainstart.py`, `scene/trainstart.yaml` |
| P3 stand-in | `[-0.6419329, 0.3706554, -0.1347388, -1.8275169, 0.5918946, 1.4884701, -1.6761742]` | z 0.1618, table z 0 | 16.18 cm | 49.51 deg | left | PLAN §9, `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/start_pose.yaml` |
| chosen pose `leftx55_t48_h16` | `[-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]` | (0.531, -0.159, 0.160) | 16.0 cm | 48.0 deg | left | `sim_deploy/stations/start_pose.yaml`, `sim_evals/_start_pose_sweep/poses/leftx55_t48_h16.yaml` |

### A1 follow-up: open-loop replays from the chosen pose (partial)

The camera study's phase A1 replayed PICK_TEST1 `pick1-v1` demos open loop from `leftx55_t48_h16`, at the real
v_max 0.25 m/s (`campA1_v025`) and at a sim-only 1.5 m/s (`campA1_v15`, overlay `stations/vmax15_campA.yaml`). It
stopped part-way when PLAN §11 paused the study. Details: [02_camera_study/A_baseline](../02_camera_study/A_baseline/README.md).
Counted here from the raw files: 30 eval folders `sim_evals/_replay/PICK_TEST1/pick1-v1_ep*/2026-11-20_*_campA1_*`,
each with one `episodes.jsonl` row and one `sim_evals/index.csv` row (30 `campA1_*` rows, none missing).

| Tag | Folders (episodes) | Run | Success = grasp = lift | Faults | Stopped | Timeouts | Not run |
|---|---|---|---|---|---|---|---|
| `campA1_v025` | 23 (eps 0-17, 25-29) | 22 | **3** (eps 0, 1, 13) | 8: 7 box floor (z -0.015 to -0.027 m < -0.005), 1 realtime (ep 27, `communication_constraints_violation (sim: physics 53.1 ms behind wall time)`) | 6 (eps 9, 10, 16, 25, 26, 28) | 5 (eps 7, 11, 12, 14, 29) | 1 (ep 17) |
| `campA1_v15` | 7 (eps 0, 1, 2, 13, 14, 15, 16) | 7 | **2** (eps 0, 13) | 2 box floor (z -0.018, -0.019 m) | 3 (eps 1, 14, 16) | 0 | 0 |

- **3/24 vs raw:** PLAN §11 ("A1 replays partly run (3/24 grasp at v 0.25 ...)") counts 24. The raw has 23 v025
  folders, 22 of them run (ep 17 `not_run`): **3 of 22** grasp (3 of 23 with the not-run one). The 2/7 at 1.5 m/s
  matches.
- **The floor faults are replay start-height artefacts:** all 9 are demos whose path, carried to the start pose,
  goes 7.2-12.8 cm below the table (`data/p4gate_grasp_z.csv` / `p4gate_per_demo.csv`, variant h16: eps 2, 3, 4, 5, 6,
  8, 15 (v15: 2, 15) at min z -0.072 to -0.128 m, all refused on box z in the approach by the gate port, [part 4](4_gate_check.md)).
  These demos started higher above their grasp than the start pose's 16 cm. An open-loop replay can't re-aim; a
  closed-loop policy would see the table.
- **Realtime faults:** two. v025 ep 27 (53.1 ms behind, 3 chains in parallel, load ~55) and v025 ep 17 (50.1 ms behind while policy_deploy started, so the episode is `not_run`; found in its `policy_deploy.log`, missing from `_logs/2026-11-20_105539/sim_pc.log`, which run.sh rewrote on restart; see [02 A_baseline](../02_camera_study/A_baseline/README.md)). The `stopped` runs are not
  physics either: A1's interim summary (`sim_evals/_camera_study/A/a1_summary.md`, INTERIM 10:54) marks v025 eps 9,
  10, 25, 26, 27, 28 and v15 eps 1, 14, 16 as infra-ended (`frames: no reply to c1.get_health/v1 within 0.1 s` or the
  libfranka violation). v025 ep 16 ended after 10:54 and is not in that summary; its `policy_deploy.log` has the same `no reply to
  c1.get_health` abort, so it is infra-ended too.
- So of the 15 v025 runs not ended by the host, 3 grasp; every miss is a floor artefact or a timeout (the gate's
  v_max shortening, [part 4](4_gate_check.md)). The pose itself caused no fault.

## Analysis

The poses were chosen on offline checks, in this order:

1. **Feasibility** (part 2): what `p` accepts, what both cameras see, and what keeps the wrist away from its limits.
   This leaves the left heading at 48 deg, plus a few diagonal and right poses with thin margins.
2. **The demos carried from the pose** (parts 2, 4): with the left heading, the demos' retreat moves toward the
   robot-side box face x 0.20. Moving the eraser out to 0.55 m fixes most of that (3 of 30 refused instead of 21), and
   IK reaches all 30 demos along the path.
3. **Height:** 16 cm against 22 cm. The 22 cm alternative is equal on everything the real robot would see. It only
   has fewer "carried into the table" artifacts (open-loop demos that started higher than the pose run into the
   table). 16 cm is closer to the demos' median start (16.6 cm above the grasp).

The closed-loop policy cannot rank the poses. At every screened pose run_006 ends rising and backing away. The
offline camera swap (part 6) puts the cause in the sim IMX219 image, not in the pose. The four right-heading poses
first move about 9 cm toward the eraser before they turn away (part 3); why is an open question.

### Docs vs raw data (consolidated)

Each item: what the doc says and where, against the raw value and its path. The raw value is the one used in this
record.

**Start pose and layout (part 1)**
1. DESIGN.md "E7" -> "Start pose variant `--home trainstart`": the demos' start tilt is "mean 49.5 deg (40-64)". Raw
   `sim_deploy/scene/trainstart.yaml`: 49.51 deg, range **38.1-64.0 deg** (per-episode `start_tilt_deg`).
2. PLAN.md §9 (P2 row), `deployment/policy_deploy/policy_deploy/manager/pc_probe.py` lines 227-228 and the comment in
   `randomization_start.yaml`: the eraser is "~17 cm below the fingertips". Raw `sim_deploy/scene/randomization_start.yaml`
   `demos.below_m` median **0.161 m**. The 166 mm figure is the height above the grasp point, a different reference.
3. DESIGN.md "P1" -> "The stand-in against the ported checks": "the Cartesian hold sags ~1 mm". FK of the probe q
   (`01_start_pose/data/p1_standins_fk.json`) gives TCP z 0.2114 against 0.2118 m: **0.4 mm** (DESIGN "P2" also says 0.4 mm).
4. DESIGN.md "P3/P4" -> "Layout change": "every P3 run made with the old table is marked `SUPERSEDED.txt`". Raw: there
   is **one** such run, `sim_evals/_baseline/hold_still/2026-11-20_0939_p3_g5_hold` (model `7746a439…`). Every other
   P3/P4 eval records `166a0649…`.
5. `sim-experiments/README.md` "Where the raw data lives": `7746a439…` is listed as the E7 model. Raw: the E7 evals of
   01:30-01:49 (`_baseline/hold_still/2026-11-20_0130_g5_hold`, `_0134_g5_hold_deploy`, `_0136_g5_hold_trainstart`,
   `PICK_TEST1/run_006_2026-11-18/2026-11-20_0149_smoke`) record **`701815be08cd…`** in `eval.yaml`. `7746a439…`
   first appears in the P1 smoke at 09:17.
6. `sim-experiments/README.md` "Frames and poses" (TCP row): "Upstream final-deployment still says 150 mm". Raw
   `~/Capstone-final-deployment/deployment/contracts/frames.yaml` line 10: z **0.23565** since commit `927a5ba`. The
   150 mm exists only in this repo's ea9f7fb copy.
7. `sim-experiments/README.md` "Sim-only settings vs the real robot" (workspace box min row): the real robot has x 0.25,
   z 0.05. Raw `~/Capstone-final-deployment/deployment/control_deploy/franka_control/config.yaml` line 43:
   **[0.20, -0.40, -0.005]** since `447910b`. The old box exists only in the ea9f7fb copy.

**Candidates and envelope (part 2)**

8. DESIGN.md "P4 candidates" ("every candidate passes the real PC's `p` rules") and `start_sweep.py` `REAL_BOX`
   (the user's box). Raw `sim_evals/_start_pose_sweep/candidates.json`: the 43 grid rows with IK carry real-PC box
   margins computed against the **old ea9f7fb box [0.25, -0.40, 0.05]** (e.g. `left_t38_h10` stores 0.1175 / 0.05 m;
   the current box gives 0.1725 / 0.105 m). Only the 18 `leftx*` rows use the current box. The `candidates.md` "real
   box margin 150 mm" column has the same mix. Pass/fail is unchanged, because the current box contains the old one.
9. DESIGN.md fail table: "joint 7 margin < 0.10 rad at right 38/16, 48/16, 48/22, 58/22" (4 poses). Raw
   `candidates.json`: **5**. `right_t38_h10` also fails on joint 7, at 0.073 rad (and on the IMX519 view).
10. DESIGN.md fail table, away row: "at 0.45-0.50 m from the base axis no solution (joint 4 at -3.07)". **Not in the
    raw files**: they hold away poses only at G 0.60. The `start_sweep.py` docstring says G 0.50 gives no solution
    (joint 4 at its limit, 39 mm residual) and mentions neither 0.45 nor -3.07. At G 0.60, `away_t48_h22` and
    `away_t58_h10` have no IK.
11. DESIGN.md fail table, left row: "table clearance = the start height". Raw `candidates.json` `path.table_clearance_m`
    is the start height **minus 1.06-1.46 cm** (8.54-20.94 cm for 10-22 cm).
12. DESIGN.md "the demos carried": "the pick depends only on the start height (14/30 at 16 cm, 6/30 at 22, 26/30 at
    10)". Raw `sim_evals/_start_pose_sweep/demo_paths.json` `box_violations_by_axis.z<`: these are the **48 deg**
    values. At 58 deg it is **12 / 2 / 21** (16 / 22 / 10 cm), and at 38 deg / 22 cm it is 10. Tilt matters too.
13. DESIGN.md "the demos carried": "the away-type headings avoid that face [x 0.20]". Raw `demo_paths.json`:
    away_left crosses **x < 0.19 in 7-8 of 30**. Away and away_right cross in 0.
14. DESIGN.md: the shifted layouts' demo numbers come from `tools/left_out.py`. Raw: `left_out.py` only adds the 18
    candidates to `candidates.json` / `.md`. The demo numbers come from `tools/demo_paths.py`, run afterwards (file
    times 10:15 / 10:17).
15. DESIGN.md: the retreat "crosses the robot-side box face x 0.20 in 21/30". The code counts **x < 0.19 m** (the box
    plus the gate's 10 mm margin). The 21 / 10 / 3 counts themselves are confirmed.
16. DESIGN.md "the demos carried": "Left heading: 30/30 reachable, margin >= 0.25 rad (0.47 at `leftx55_t48_h16`)". This
    is correct in `demo_paths.json` (min 0.25 at `left_t48_h10`, 0.474 at `leftx55_t48_h16`), but it checks **only
    the grasp and end pose**. IK along the whole path (`gate/gate_left_t48_h16.json`) has 6/30 unreachable at
    `left_t48_h16` (min 0.001 rad), and min 0.041 rad at `leftx55_t48_h16` (`gate/gate_h16.json`).
17. Not in any raw file: "the retreat moves ~25 cm to the hand's left", and "at 38 deg the corners 9 cm ahead fall
    below its image". Only the number of failing placements is stored (4 of 9 at 10 cm, 3 of 9 at 16 cm).

**Screen (part 3)**

18. DESIGN.md "Screen and full eval": "the TCP never got closer to the eraser than it started (10-28 cm)". Raw
    `episodes/000/trace.json` + `episodes.jsonl` `eraser_start_pose`: 17 of 23 poses got closer by >= 0.5 mm. The 4
    right-heading poses got **8.9-9.8 cm** closer (`right_t48_h10`: 0.197 -> 0.105 m,
    `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_1013_sweep_screen_right_t48_h10/`), and `away_t48_h16` 2.5 cm
    closer. "10-28 cm" is the range of the **closest** distances (0.105-0.275 m).
19. DESIGN.md "Screen and full eval": "the rest at 11-24 s". Raw `episodes.jsonl`: **12.50-23.71 s** (earliest
    `away_t48_h16`).
20. DESIGN.md "Screen and full eval": "every pose shows the same retreat as G7". Raw `trace.json`: every pose ends
    climbing, but the right-heading poses first move 7.8-9.0 cm forward and 3.7-5.8 cm down in the first 3 s (e.g.
    `right_t58_h16` (0.469, 0.159, 0.160) -> (0.445, 0.081, 0.114) m).

**Gate check (part 4)** (DESIGN line numbers refer to `sim_deploy/DESIGN.md` as of this record)

21. DESIGN.md gate table, PICK_DOUBLE "box refused 36". Raw `sim_evals/_start_pose_sweep/gate/gate_h16.txt`: 36
    refused = **35 box + 1 cone** (ep 37, lift, 62.4 deg).
22. DESIGN.md gate table, PICK_DOUBLE "14" without the floor. Raw `gate/gate_h16_nofloor.json`: **13 box + 1 cone**.
23. DESIGN.md "the 22 cm alternative": "21/62". Raw `gate/gate_h22.json`: **20 box** (approach 7, retreat 12, lift 1)
    **+ 1 cone**.
24. DESIGN.md gate table, shortened "(all in the approach; also lift 19, retreat 23)". Raw `gate/gate_h16.txt`: also
    **grasp 1** (with the floor), and grasp 9 without it. An omission.
25. DESIGN.md gate table, "cone_max for 28/30 ... <= 54.0 already passes". Raw `gate/gate_h16.txt` `for_28_of_30`:
    54.0 with no pull-back, **52.0** with no refusal (the second number is missing in the doc).
26. DESIGN.md "the 22 cm alternative" gives no PICK_DOUBLE IK figure. Raw `gate/gate_h22.json`: **9 unreachable**,
    p10 0.007 rad. An omission.
27. DESIGN.md "For the real PC": "0.05 would refuse every grasp". Raw (open-loop carry, `01_start_pose/data/p4gate_grasp_z.csv`):
    **24/30** grasps fall below z 0.04 m; 6 (eps 18, 20, 21, 24, 25, 29) sit 4.3-7.7 cm above the table. "Every
    grasp" holds for a real grasp at the eraser, not for the carried demos. z 0.05 exists only in the ea9f7fb copy.
28. DESIGN.md "For the real PC": "the demos are 4-5x faster than 0.25". Raw `gate/gate_h16.json`: 4.8x for 28/30
    (1.211 m/s), 5.1x for the worst step (1.274 m/s), but only **2.7x** for the median demo (0.67 m/s).
29. DESIGN.md "P3 results" (G6) and PLAN §6 G6: "29 of 30 demos have 50 ms segments above 0.5 m/s". On the gate's
    50 ms grid: **24/30**. The sampling differs, so this may not be a contradiction.
30. DESIGN.md "P3/P4" -> "Screen and full eval": `gate_check.py` is a "line-by-line port" of `safety_gate.cpp`. Raw
    code: it matches the box, cone, speed-bucket and lead rules. It **omits** the width rule, the finite check and
    skipped waypoints, and it starts each request at W(t_c) where the real PC uses `plan_.at(now)` (`supervisor.cpp`
    line 443). The lead rule cannot fire in the port (lead is at most 0.6 s against a 1.5 s limit).
31. `sim_deploy/DIRECTIONS.md` §2c does not list `gate_check`, and the gate JSONs record neither the pose id nor the
    command line. The variants' flags were inferred: the `left_*` / `leftx50_*` files are `--no-floor` with IK, and
    the h16/h22 no-floor files are `--no-ik`. A doc gap, not a number.

**Handoff (part 5)**

32. DESIGN.md "The chosen pose": the handoff is "checked by `start_sweep handoff`" against "the user's real station".
    Raw: those values are constants hard-coded in `sim_deploy/scene/start_sweep.py` lines 68-74, with
    `REAL_F_T_TCP_Z = 0.150` called the real TCP. Both the box and the 235.65 mm TCP are now committed upstream. In
    `candidates.json`, `p_accepted_real_tcp` means the old 150 mm TCP. No number changes.
33. DESIGN.md "The chosen pose": "no checkpoint succeeds yet". Raw `sim_evals/index.csv`: only run_006 was ever tried
    (G7 smoke, `p3_g7_start`, 23 screen rows, 0 successes), and **never from `leftx55_t48_h16`** (no `leftx` rows in
    `_start_pose_sweep/summary.csv`).
34. DESIGN.md "The chosen pose": ">= 24.0 cm to the robot's own links". Raw `candidates.json` `path.self_clearance_m`
    0.2397 m covers only links 6-7 / hand / rig / fingers against links 0-3, at the home end of the path (s 0.0).
35. `~/Capstone-final-deployment/deployment/control_deploy/DIRECTIONS.md` step 10 (around line 230) tells the user to
    hand-guide with the fingertips "pointing away from the robot" and the object "centred left-right". Raw
    `candidates.json` / `demo_paths.json`: heading away is infeasible or leaves 0.001-0.12 rad of joint margin, and the
    chosen heading is left. **This one matters for the real arm.** Fixed upstream since (HEAD `0c5903a`, line 272:
    "fingers pointing to the robot's left"), but that line also places the object at "about (0.53, 0.00) m" (see 44).
36. Rounding only: 2.32 rad / "about 133 deg" for joint 7 against raw 2.3155 rad = 132.7 deg; region "9-22 cm, yaw
    -7..+1" against raw 0.090-0.217 m and -6.6..+0.8 deg.

**Camera swap (part 6)**

37. DESIGN.md "P3 results" (G7) and PLAN §6 P3 G7: run_006 "rises and backs off from the first chunk" in all 3
    episodes. Raw first-chunk end z (`session/*_mock/manager/tap/ep_0001/chunk_1.npz` `chunk_eef_pos`, G7 eval):
    -0.78 / **+1.15** / -0.56 cm. Episode 001 first came 2.49 cm closer (23.15 cm at 3.06 s, `episodes/001/trace.json`).
    So the claim holds for 2 of 3 episodes.
38. DESIGN.md "P3 results" and PLAN §6 G7: "demos +5.7 cm in 0.75 s". **Not in the raw logs** (`g7_images.py`'s output
    was not saved). Recomputed read-only from `pick1-v1.zarr.zip`: median 0.057 m (range 0.026-0.099 m). The value holds.
39. DESIGN.md "P3 results": "the width (0.080 vs 0.087) changes nothing". Raw `sim_evals/_start_pose_sweep/p3/offline.log`:
    sim images give -1.3 / -1.3 cm; training images give **+5.6 (width 0.0868 m) / +6.2 cm (width 0.080 m)**. The
    sign does not change.
40. DESIGN.md "P3 results": "the same model and low-dim input". True, but the offline run is fp32 on the CPU with the
    warm-up prompt (train-split index 14 = `episode_000153`), while the closed loop ran bf16 on the GPU with
    prompt_index 0 (`episode_000138`) (`session/2026-11-20_09-57-37_mock/logs/policy.log` lines 7, 12).
41. PLAN §6 G7 and DESIGN "P3 results": the repro command gives the relative script path
    `sim_evals/_start_pose_sweep/tools/offline_first_chunk.py` with cwd `deployment/policy_deploy`. **That path does
    not resolve from that cwd**: the command as written fails. Use absolute paths (the script exists; it needs
    `p3/ep152_ref.npz` through a hard-coded absolute path). See [part 6](6_camera_swap.md) "Commands to reproduce".
42. PLAN §10 (last paragraph): "with the sim's [IMX219] it backs away (-1.3 cm)". -1.3 cm is the all-sim case. The
    matching swap (training IMX519 + sim IMX219) is **-0.5 cm** (`p3/offline_split.log`).
43. DESIGN.md "P3 results" lists table, shadows, eraser and IMX219 background as the appearance differences. The PNG
    (`figures/g7_policy_images_vs_ep152.png`) also shows a **finger colour** difference (grey against light blue).

**Added after the upstream adoption (2026-11-20, read-only checks of `~/Capstone-final-deployment`)**

44. final-deployment `abba9e8` (commit message, and the comment in `deployment/control_deploy/station.yaml` lines 7-9):
    "object ~(0.53, 0.00)". Raw `sim_evals/_start_pose_sweep/candidates.json` `leftx55_t48_h16.grasp_centre_O_m`:
    **(0.55, 0.00)** (= start TCP x 0.531 + the region's 0.019 m "right" offset, `scene/randomization_start.yaml`). 0.53
    is the TCP's x. The q in `abba9e8` is identical to ours to 7 decimals.
45. PLAN.md §11 "Progress so far" (camera study row): "A1 replays partly run (3/24 grasp at v 0.25 ...)". Raw: **3 of
    22 run** (23 v025 folders, ep 17 `not_run`), `sim_evals/_replay/PICK_TEST1/*/*campA1_v025` (A1 follow-up above).
46. Part 5's expected probe output and part 1's `p` semantics give the finger opening as 0.08 m / `80.0 mm` (this
    repo's ea9f7fb copy and the sim). Upstream since `b4030f6`: `contracts/limits.yaml` `gripper_width_max_m`
    **0.04** (the printed fingers' measured opening). On the real arm `p` opens to 0.04 m and `pc_probe` prints
    `40.0 mm`. Part 5 now says so. The copy is behind upstream (PLAN §11 W0 re-syncs it).

## Conclusion

- **Chosen pose: `leftx55_t48_h16`**, `q_start_rad: [-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725,
  1.5068439, -1.5301472]`. TCP O (0.531, -0.159, 0.160) m, 16.0 cm above the table, tilted 48.0 deg, pointing to the
  robot's left, eraser about 0.55 m out. It is the sim default (`sim_deploy/stations/start_pose.yaml`), and it was
  **adopted upstream in `abba9e8`, to be checked on the arm** (final-deployment `deployment/control_deploy/station.yaml`,
  identical q). Settled: PLAN §11 takes it as fixed. The 22 cm alternative (`leftx55_t48_h22`) is the fallback.
- **Why this pose:**
  - It is feasible with wide margins: joints 1.241 rad, box 0.165 m, cone 12.0 deg, `p` 14.47 s of 30 s.
  - Both cameras see the whole eraser region.
  - It passes the real station with the 235.65 mm and the 150 mm TCP.
  - Of the layouts tried, it has the fewest gate refusals for the demos' relative motions (3 of 30 without the floor
    artifact, against 10 and 21 at 0.50 / 0.45 m) and the fewest IK misses (0 of 30).
  - Left is the only heading that is both feasible and far from the joint limits.
- **What blocks ranking by policy success:** run_006 rejects the sim's IMX219 render. Its first chunk backs away
  with the sim IMX219 and approaches with the training IMX219, so every pose fails the same way. No checkpoint has
  succeeded in the sim, and the chosen pose itself has never been run with a policy. Ranking by success waits for
  [02_camera_study](../02_camera_study/README.md).
- **Sim success is not real success:** the pose must be checked on the arm ([part 5](5_real_pc_handoff.md) checklist,
  with `gripper width: 40.0 mm` since upstream `b4030f6`). Place the eraser at O (0.55, 0.00), not the (0.53, 0.00) in
  `abba9e8`'s comment. The open-loop A1 replays from this pose grasp in 3 of 22 runs, and the pose itself caused no
  fault (A1 follow-up).
  The real binding rule during episodes will be v_max 0.25 m/s, not the start pose.

## Limitations / sim-vs-real caveats

- Feasibility, gate and IK are checked on the sim model and on the demos played open loop. A closed-loop policy can
  make up shortened distances and will not follow the demos exactly.
- The gate port leaves out the width, finite and skip rules and models no latency (part 4).
- The eraser region assumes the demos' relative layout. The real table and its clutter are not modelled.
- The policy screen is one episode per pose, and only run_006 (part 3).
- The chosen pose was never run closed loop in the sim.

## Open questions

1. Does any checkpoint approach the eraser from `leftx55_t48_h16` once the IMX219 render is fixed? Answered by
   [02_camera_study](../02_camera_study/README.md) (phases A2, C2).
2. Why do the right-heading poses first move about 9 cm toward the eraser (part 3)? Compare their first chunks with
   the left poses' (tap `chunk_1.npz`).
3. Should the real box x min go to 0.15 m (it would refuse none of the demos' retreats)? Measure the real table
   around the base.
4. How much of the v_max loss (median 2.6 cm at the grasp) does a closed-loop policy make up? Run a policy that works
   in the sim at 0.25 m/s.
5. Does the object placement in final-deployment (`abba9e8` station.yaml comment, DIRECTIONS line 272: (0.53, 0.00))
   need to move to (0.55, 0.00)? The heading there already says left. The user's call; check the IMX219 view on the arm.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| candidates, demo paths, poses | `sim_evals/_start_pose_sweep/{candidates.md,candidates.json,demo_paths.json,poses/}` |
| screen summary | `sim_evals/_start_pose_sweep/{summary.csv,summary.md}` |
| A1 follow-up | `sim_evals/_replay/PICK_TEST1/pick1-v1_ep*/2026-11-20_*_campA1_{v025,v15}/`, `sim_evals/_camera_study/A/{a1_summary.md,a1_episodes.csv}`, `sim_evals/index.csv` (`campA1_*`) |
| screen evals | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_10*_sweep_screen_*/` (the 24th has `INTERRUPTED.txt`) |
| gate check | `sim_evals/_start_pose_sweep/gate/gate_*.{json,txt}` |
| G7 and offline first chunk | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/`, `sim_evals/_start_pose_sweep/p3/` (`ep152_ref.npz` 15.7 MB stays there) |
| tools | `sim_evals/_start_pose_sweep/tools/{demo_paths,left_out,g6_path,g6_time,g7_images,offline_first_chunk}.py`, `sim_deploy/scene/{start_sweep,start_pose,trainstart,gate_check}.py` |
| superseded run | `sim_evals/_baseline/hold_still/2026-11-20_0939_p3_g5_hold/SUPERSEDED.txt` |
| copied into this folder | `data/p1_*` (P1 logs, FK), `data/p2_*` (candidates), `data/p3screen_*` (screen), `data/p4gate_*` (gate summaries), `data/p5_*` (pose files, recomputation), `data/p6_*` (offline logs), `figures/g7_policy_images_vs_ep152.png` |
