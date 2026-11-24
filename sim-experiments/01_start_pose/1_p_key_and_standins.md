# 01_start_pose / 1: the PC key `p`, its sim wiring, and the three stand-in start poses

**Status:** done · **Date:** 2026-11-20 · **Who:** agent ·
**Plan ref:** sim_deploy/PLAN.md §6 ("P3" table), §9, §10; DESIGN.md "E7" -> "### Start pose variant `--home trainstart`",
"## P1: start-pose port", "## P2: start pose in eval/scene/UI", "## P3/P4" -> "### Layout change"; DIRECTIONS.md §2b, §2c

Part 1 of 6 of [01_start_pose](README.md). Next parts: [2_candidates_envelope.md](2_candidates_envelope.md),
[3_early_failure_screen.md](3_early_failure_screen.md), [4_gate_check.md](4_gate_check.md),
[5_real_pc_handoff.md](5_real_pc_handoff.md), [6_camera_swap.md](6_camera_swap.md).

## Question

1. What exactly does the real PC key `p` do, and what does it refuse? Does the sim run the same code and the same
   per-episode flow?
2. Which `q_start_rad` did the sim use, in which order, and where do these values come from? For each one: q, TCP in
   O, height, tilt, heading. The three are the E7 trainstart stand-in, the P3 stand-in and the chosen pose
   `leftx55_t48_h16`.
3. What did the P3/P4 layout change (table at O z 0) change, and what did it leave alone?

## Why it matters for the real robot

On the Franka, `homing.q_start_rad` is still `null`. Until someone measures it by hand-guiding the arm and reading it
with `pc_probe --start-pose`, `p` is refused. Until then every `--home start` sim result depends on a stand-in. This
part records which stand-in each sim run used and how close each one is to the demos' start. It also records that the
sim runs the real PC's `p` code unchanged: same refusals, same end-of-move box and cone check, same C3 `start` rule.
A sim result about the start pose therefore carries over to the real key. The value recommended for the real arm (the
chosen pose `leftx55_t48_h16`) and how to check it on the arm are in [5_real_pc_handoff.md](5_real_pc_handoff.md).

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Model | now `sim_deploy/build/sim_rig/model.json` `model_sha256` `166a0649571a…` (table top at O z 0, base on the table, no pedestal). Earlier: `7746a4391b50…` (table at z 0.05 on a pedestal; P1 smoke and the superseded P3 run) and `701815be08cd…` (the E7 G5/G7 evals of 01:30-01:49) |
| Home / start pose | `--home start` with the three stand-ins below. Home = copied `config.yaml` `homing.q_home_rad` `[0.0, -0.7853982, 0.0, -2.3561945, 0.0, 1.5707963, 0.7853982]` (TCP O (0.3069, 0.0, 0.3546), fingers straight down, 35.46 cm above the z 0 table) |
| Eraser placement | `sim_deploy/scene/randomization_start.yaml` (relative to the start pose, seeded) |
| Stations / overlays | `stations/control_sim.yaml` + `stations/start_pose.yaml`, merged by `sim_pc/merge_station.py` into `build/control_sim_start.merged.yaml` at each sim PC start |
| Policy / checkpoint | none of my own. The evals cited: `hold_still`, `replay:` PICK_TEST1 `pick1-v1`, `PICK_TEST1/run_006_2026-11-18` (P3 G7) |
| Inference settings | not applicable (no new runs) |
| Seeds / episodes | the cited evals: `--seed 0`, 1-3 episodes |
| Timeout | not applicable |
| Sim-only settings | sim box `workspace_min_m` [0.20, -0.40, -0.005] (was [0.20, -0.40, 0.045] for the old table); `q_start_rad` = a stand-in; scene `key` command on :5651 (sim build only) |

Box words: the **sim box** is `control_sim.yaml` over the copy's `config.yaml`: [0.20, -0.40, -0.005]..[0.80, 0.40,
0.70] m. The **real box** is upstream `rnoro/final-deployment` `deployment/control_deploy/franka_control/config.yaml`
@ `447910b` (2026-11-20 10:05): the same [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70] m, cone 60 deg. The copy in this
repo (ea9f7fb + snapshot) still has [0.25, -0.40, 0.05] in its `config.yaml`. The sim never uses that value because
`control_sim.yaml` overrides it.

## Method

There were no new sim runs. Everything below was read from code, YAML, eval folders and logs, plus two read-only CPU
computations:

1. **The real key `p`.** I read the ported franka_control (`deployment/control_deploy/franka_control/`) and the
   laptop manager (`deployment/policy_deploy/policy_deploy/manager/`), and compared them with upstream
   `rnoro/final-deployment` (`git show`, read only). The snapshot that P1 ported was committed upstream as
   `927a5ba` (2026-11-20 09:58).
2. **The sim wiring.** I read `sim_pc/src/sim_scene.cpp` (scene `key`), `eval/driver.py` (the per-episode
   `h` -> `p` -> eraser -> `c`), `scene/start_pose.py` + `scene/__main__.py` (`start-pose`) and
   `scene/randomization_start.yaml`.
3. **The stand-ins.** q values come from the files that held them: `scene/trainstart.yaml`, the `start_pose.yaml`
   copies and `home` blocks in eval folders, `stations/start_pose.yaml`, and
   `sim_evals/_start_pose_sweep/poses/*.yaml`. The pose reached after `p` comes from `episodes.jsonl`
   `start_pose`.
4. **Recomputed (read-only, CPU, no files written outside this folder):**
   - `sim_deploy.scene.start_pose.describe()` (FK + the start-pose checks) for each stand-in, the P1 `pc_probe`
     q and the home, in the current model -> `data/p1_standins_fk.json`.
   - `sim_deploy.scene.trainstart.compute()` (the E7 IK) **without** `--write`, in the current model ->
     `data/p1_trainstart_ik_z0.json`. It reproduces the P3 stand-in to all 7 printed decimals.
5. **The layout change.** I read `model.json`, `stations/sim_rig.yaml scene.table` and `control_sim*.yaml`, read the
   model sha in each P3 `eval.yaml`, and listed every `SUPERSEDED.txt` under `sim_evals/`.

A claim **passes** when the code or raw file says the same as the docs. Where they differ, the raw value is used
here and listed under "Docs vs raw data".

## Commands to reproduce

Do not run these while another agent runs the sim chain. The lines marked WRITES change tracked files in the repo.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH

# Read only: the start pose in use, with its checks (loads the MuJoCo model, ~8 s; show()/describe()/Scene.check_region
# only read files: checked in scene/start_pose.py)
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose --show

# (a) E7 trainstart stand-in. Made at E7 with the old model (table z 0.05). WRITES scene/trainstart.yaml and
#     stations/control_sim_trainstart.yaml. With today's model it would write the P3 numbers instead.
~/envs/sim_deploy/bin/python -m sim_deploy.scene.trainstart --write

# (b) P3 stand-in = the same IK against the current model (table z 0). Without --write it only prints (read only,
#     CPU, under 2 min here, not timed); its q_home_rad is the P3 q:
~/envs/sim_deploy/bin/python -m sim_deploy.scene.trainstart
#     ... then into the start-pose file. WRITES stations/start_pose.yaml (written 2026-11-20 09:41:08):
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose \
  --q "-0.6419329,0.3706554,-0.1347388,-1.8275169,0.5918946,1.4884701,-1.6761742" \
  --source "STAND-IN (P3, 2026-11-20): E7's training-like start re-solved for the table at O z 0 (scene/trainstart.py IK, 16.2 cm above the table)"

# (c) the chosen pose leftx55_t48_h16: candidate from the P4 sweep (see 2_candidates_envelope.md; WRITES
#     sim_evals/_start_pose_sweep/), then into the start-pose file. WRITES stations/start_pose.yaml (written 10:18:12):
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose \
  --q "-0.5928536,0.5219462,-0.0897168,-1.5930073,0.6595725,1.5068439,-1.5301472" \
  --source "P4 sweep 2026-11-20: leftx55_t48_h16 (...)"
#     Check without writing: add --dry. Other sources: --from-probe <PROBE_TXT> (a saved `pc_probe --start-pose`
#     printout), --from-git [REF] (REF:deployment/control_deploy/station.yaml; today it fails: "not set yet").

# Read only: FK of the three stand-ins in the current model (what data/p1_standins_fk.json holds)
MUJOCO_GL=disable ~/envs/sim_deploy/bin/python -c "
from sim_deploy.scene import start_pose as sp
for q in ([-0.6225485,0.3138391,-0.1643961,-1.7825973,0.5934631,1.3978677,-1.6275796],
          [-0.6419329,0.3706554,-0.1347388,-1.8275169,0.5918946,1.4884701,-1.6761742],
          [-0.5928536,0.5219462,-0.0897168,-1.5930073,0.6595725,1.5068439,-1.5301472]):
    print(sp.summary_line(sp.describe(q)))"
```

## Results

### R1. The real PC key `p` (ported snapshot = upstream `927a5ba`)

| Rule | Code (ported copy, `deployment/control_deploy/franka_control/`) | Refusal / log text (exact) |
|---|---|---|
| refused while `q_start_rad` is null (the `config.yaml` default, line 64: `q_start_rad: null`) | `src/supervisor.cpp:373-376` (first check in `onStartPose`) | `key p refused: homing.q_start_rad is not set: measure it and set it in control_deploy/station.yaml (franka_control: homing:)` |
| only in idle / stopped (the state machine's `kHome` check) | `supervisor.cpp:377-378`, wording from `keyRefusal` `:255-261` | e.g. `key p refused: mode is holding: the laptop is driving an episode: press s or stop it on the laptop first` |
| only from home: every joint within `homing.at_pose_tol_rad` (0.05 rad, `config.yaml:65`) of `q_home_rad` | `supervisor.cpp:379-381` | `key p refused: not at home: press h first` |
| refused when the move would last >= `homing.timeout_s` (30 s) | `startArmMove`, `supervisor.cpp:391-395` (shared with `h`) | `refused: start pose move would take <t> s > homing.timeout_s 30.0 s: move the arm closer by hand` |
| the move: `h`'s joint move (`HomingMotion`, 0.3 rad/s, min 2.0 s) to `q_start_rad` | `startArmMove(source, ArmGoal::kStartPose, *q_start_rad, gripper_open_m_)`, `:382` | `key p: start pose: <t> s joint move to homing.q_start_rad` |
| fingers to `gripper_width_max_m` 0.08 m (`contracts/limits.yaml:6`) | the finger target is set just before the arm move: `src/control_loop.cpp:294` (`setGripperTarget(*home_open_width_m_)` before `moveToJoints`); the fingers then move during the arm move | `key p: the hand opens to 0.080 m (gripper_width_max_m) during the move` |
| at the end: TCP inside the box (no margin), else `fault` | `supervisor.cpp:180-183` (also for `h`) | `start pose TCP (x, y, z) m is outside the workspace box: check homing.q_start_rad and the box` |
| at the end (p only): tilt <= `safety.cone_max_deg` with no margin, else `fault` | `supervisor.cpp:184-190` | `start pose tilt <t> deg from the approach axis > safety.cone_max_deg 60.0: check homing.q_start_rad and the cone` |
| success | `supervisor.cpp:197-198` | `start pose reached (fingers at 0.080 m): start the episode` |
| C3 `start` refused away from the start pose, once it is set (every joint within 0.05 rad) | `onStart`, `supervisor.cpp:325-328` | `refused: not at the start pose: press h, then p on the PC` |
| config load: 7 numbers inside the Panda's limits, or null / absent | `src/config_loader.cpp:496-511`; `include/franka_control/config.h:78-79` (`std::optional<Vec7> q_start_rad`, `at_pose_tol_rad = 0.05`) | `homing.q_start_rad: joint <i> (<v> rad) is outside the Panda's limits [...]` |
| the keyboard key | `src/keyboard_reader.cpp:54` | `keys: h home arm, p start pose, g home hand, s stop, q quit` |
| laptop `h` removed | `policy_deploy/manager/operator.py:19-29` (`KEY_LIST = "keys: c start  y success  n fail  s stop  q quit"`; the Key enum has no `h`) | `keyboard_operator.py:61`: `'h': not a key. keys: c start  y success  n fail  s stop  q quit` |
| `pc_probe --start-pose` | `pc_probe.py:243-259` (`start_pose_lines`), demos text `:227-228` | `q_start_rad: [...]   # paste under franka_control: homing: in the PC's control_deploy/station.yaml`, `fingertip (TCP) height: z <z> m ...`, `fingertip direction (TCP z axis) -> base (x, y, z): tilted <t> deg from straight down, pointing to the robot's left`, `gripper width: <w> mm`, `demos start: fingertips tilted ~48 deg from straight down (38-64), ~16 cm above the table (6-27), hand open; object ~16 cm ahead horizontally and ~17 cm below the fingertips` |

Upstream `rnoro/final-deployment` (read with `git show`): `927a5ba` holds this code, the TCP 235.65 mm (`frames.yaml`
z 0.23565) and the box floor z -0.005. `447910b` sets x min 0.20. `q_start_rad` is still `null` there, and
`deployment/control_deploy/station.yaml` has no `q_start_rad`.

### R2. The sim side

| Piece | What it does (verified in code) |
|---|---|
| scene `key` (sim build only) | `{"cmd": "key", "key": "h|p|g|s"}` on REP :5651 (`sim_pc/src/sim_scene.cpp:257-258, 286-300`): pushes the key into franka_control's own `KeyQueue` (the keyboard thread's path -> `Supervisor::onKey`). The reply waits up to 1.0 s (`kKeyAnswerS`, `:58`) for the first log line that starts with `key <k>:` or `key <k> refused:`. It answers `{"ok": true, "log": ...}`, `{"ok": false, "error": ...}`, or `{"ok": true, "log": null}` when no line arrives in time. Any other key gives `"key" must be one of h, p, g, s (q is not queued: ...)` |
| driver flow `--home start` (`eval/driver.py:436-493`) | per episode: `h` via the scene key (`--home-key auto` -> `pc`, because `start_pose.laptop_has_home_key()` finds no `HOME = "h"` in the copy's `operator.py`), then `p`. After each key, C2 must reach `homing` within 3 s and then `idle` with every joint within `at_pose_tol_rad` of the goal (`_pc_key_move`, `:382-411`). Then the eraser from `randomization_start.yaml`, `settle_s` 0.3 s, then the laptop's `c` |
| outcomes | a refused `p`, a C2 fault during `p`, no homing within 3 s, or no return to idle -> `start_pose_failed` (`:462`). Idle at another pose -> the eval stops (`EvalError`: the sim PC runs another start pose). `c` refused with `not at the start pose` -> `start_failed` (`:486-493`) |
| `stations/start_pose.yaml` | only `franka_control: homing: q_start_rad:` plus comments (`# source:`, `# date:`, `# in the sim:`). `run.sh` merges it at every sim PC start, including restarts after a fault. `--home deploy` merges a generated null file instead |
| `python -m sim_deploy.scene start-pose` (`scene/start_pose.py`, `scene/__main__.py:26-70`) | `--q` / `--from-probe FILE` (the last `q_start_rad: [...]` line) / `--from-git [REF]` (`GIT_OPTIONAL_LOCKS=0 git show REF:deployment/control_deploy/station.yaml`, default ref `rnoro/final-deployment`). Checks that refuse a write unless `--force` is given: Panda joint limits (the loader's table), model joint limits, TCP in the sim box (no margin), cone (no margin), no contact (eraser aside), a heading (not straight down), move time from home < `timeout_s` (`max(2.0, 1.875 max|dq| / 0.3)`). Information only: tilt and height vs the demos; with `--show` also the eraser region at that pose. `--dry` checks without writing. `--show` alone only reads |
| `scene/randomization_start.yaml` (raw values) | `ahead_m: [0.09, 0.217]`, `left_m: [-0.035, -0.003]`, `yaw_rel_deg: [-6.6, 0.8]`, `median: {ahead_m: 0.159, left_m: -0.019, yaw_rel_deg: -2.2}`, `settle_s: 0.3`; demos (30 episodes of `pick1-v1`): tilt median 48.2 deg (38.1-64.0), tip height median 0.163 m (0.062-0.271), eraser below the fingertips median 0.161 m (0.064-0.272). Matches DESIGN P2 |

Observed in the sim (raw logs, copied into `data/`):

| Situation | Raw line | Source |
|---|---|---|
| `q_start_rad: null`, scene `p` | `key p refused: homing.q_start_rad is not set: measure it and set it in control_deploy/station.yaml (franka_control: homing:)`; then C3 start accepted: `started: holding the current target` | `data/p1_fc_null.log` (P1 smoke, model 7746a439) |
| E7 trainstart stand-in, `p` from home | `key p: start pose: 15.1 s joint move to homing.q_start_rad` ... `start pose reached (fingers at 0.080 m): start the episode` after 15.09 s | `data/p1_fc_standin.log` |
| `p` again, at the start pose | `key p refused: not at home: press h first` | same; again in P3: `data/p1_p3_g5_refusal_sim_pc.log` 09:45:35 |
| `p` / `h` while an episode is holding | `key p refused: mode is holding: the laptop is driving an episode: press s or stop it on the laptop first` | `data/p1_fc_standin.log` |
| `c` away from the start pose (driver `--home deploy` against a sim PC with the P3 stand-in set) | outcome `start_failed`, note `c: the PC refused start: refused: not at the start pose: press h, then p on the PC` | `sim_evals/_baseline/hold_still/2026-11-20_0945_p3_g5_refuse_c_deploy/episodes.jsonl` (+ `NOTE.txt`) |

### R3. The three stand-ins, in order of use

| | (a) E7 trainstart stand-in | (b) P3 stand-in | (c) chosen pose `leftx55_t48_h16` |
|---|---|---|---|
| in use | E7 (`--home trainstart`, as `q_home_rad`); P1/P2 as `q_start_rad` (file written 09:26:28) | 09:41:08 to 10:18:12 (P3 G5-G7, P4 screen) | since 10:18:12 (sim default, recommendation) |
| table | O z 0.05 (pedestal) | O z 0 | O z 0 |
| q (rad, as in the raw file) | `[-0.6225485, 0.3138391, -0.1643961, -1.7825973, 0.5934631, 1.3978677, -1.6275796]` | `[-0.6419329, 0.3706554, -0.1347388, -1.8275169, 0.5918946, 1.4884701, -1.6761742]` | `[-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]` |
| TCP O (m) | (0.4385, -0.1641, 0.2118) | (0.4385, -0.1641, 0.1618) | (0.531, -0.159, 0.160) |
| height above its table | 16.18 cm (21.18 cm in today's layout) | 16.18 cm | 16.0 cm |
| tilt | 49.51 deg | 49.51 deg | 48.0 deg |
| fingertip direction (TCP z in O) | (-0.031, 0.760, -0.649) | (-0.031, 0.760, -0.649) | (0.000, 0.743, -0.669) |
| heading | 92.36 deg, "pointing to the robot's left" | 92.36 deg, "pointing to the robot's left" | 90.0 deg, "pointing to the robot's left" |
| joint margin (model limits) | 1.270 rad (joint 7) | 1.221 rad (joint 7) | 1.241 rad (joint 2) |
| `p` move from home (computed / logged) | 15.08 s / 15.1 s | 15.38 s / 15.4 s | 14.47 s / 14.52 s (driver) |
| eraser region centre (demos' median grasp) | O (0.45, 0.0) | O (0.45, 0.0) | O (0.55, 0.0) |
| how it was made | `scene/trainstart.py` IK: the demos' median start relative to the grasp, placed at the centre of `randomization_trainstart.yaml` (x 0.41-0.49, y -0.04..0.04), closing axis O +x | the same IK, same code, against model `166a0649` (reproduced: `data/p1_trainstart_ik_z0.json`, position error 0.024 mm, 0.0006 deg) | P4 candidate grid, then written with `start-pose --q` (see [2_candidates_envelope.md](2_candidates_envelope.md)) |
| source files | `sim_deploy/scene/trainstart.yaml` (`q_home_rad`, `ik`), `stations/control_sim_trainstart.yaml`, `scene/start_pose.py:52` `STAND_IN`, copy in `_baseline/hold_still/2026-11-20_0939_p3_g5_hold/eval.yaml home.start_pose` | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/start_pose.yaml` + `eval.yaml home.start_pose`, `sim_evals/_start_pose_sweep/poses/standin_z0.yaml`, `scene/start_sweep.py:67` `STAND_IN_Z0`, comment in `stations/start_pose.yaml` | `sim_deploy/stations/start_pose.yaml`, `sim_evals/_start_pose_sweep/poses/leftx55_t48_h16.yaml`, `candidates.json` |

The E7 trainstart statistics (raw `scene/trainstart.yaml`, 30 PICK_TEST1 episodes):
- Median start relative to the grasp, in the gravity-aligned grasp frame: (-11.49, -164.08, 166.12) mm, i.e. 11 mm
  along the closing axis, 164 mm behind and 166 mm above.
- Tilt of the mean start rotation: 49.51 deg. Per-episode tilt: mean 49.54 deg, median 48.2 deg, range 38.1-64.0 deg.
- Grasp tip height median: -0.0043 m. Start TCP z = table + 0.16612 - 0.0043 m = table + 0.1618 m.

The pose actually reached after `p` (C2 after the key, `episodes.jsonl start_pose`):

| Eval | Stand-in | max joint error (rad) | TCP O (m) | tilt | height | width |
|---|---|---|---|---|---|---|
| `_baseline/hold_still/2026-11-20_0939_p3_g5_hold` (SUPERSEDED) | (a), table z 0.05 | 0.000094 | (0.4385, -0.164, 0.2118) | 49.52 deg | 16.18 cm | 0.0800 m |
| `_baseline/hold_still/2026-11-20_0943_p3_g5_hold_z0` | (b) | 0.000114 | (0.4384, -0.164, 0.1617) | 49.52 deg | 16.17 cm | 0.0800 m |
| `PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start` eps 0-2 | (b) | 0.000069-0.000073 | (0.4385, -0.164, 0.1618) | 49.52 deg | 16.18 cm | 0.0800 m |
| `_replay/PICK_TEST1/pick1-v1_ep000/2026-11-20_1037_campA1_v025` | (c) | 0.000089 | (0.5309, -0.159, 0.1599) | 48.01 deg | 15.99 cm | 0.0800 m |

The P1 `pc_probe --start-pose` q, `[-0.6231, 0.3176, -0.1643, -1.7796, 0.5940, 1.3950, -1.6265]` (DESIGN P1; the
printout was not saved; the probe ran after the hand homing `g`, in the hold that follows the move), differs from the
E7 trainstart stand-in by joint 1 -0.0006, joint 2 **+0.0038**, joint 3 +0.0001, joint 4 +0.0030, joint 5 +0.0005, joint 6
-0.0029 and joint 7 +0.0011 rad. Through FK (`data/p1_standins_fk.json`) that q gives TCP O (0.4385, -0.1640, 0.2114)
and tilt 49.67 deg: 0.4 mm lower and 0.16 deg more tilt than the stand-in.

### R4. The layout change (P3/P4)

| | Before (E6/E7, P1/P2) | After (P3/P4, current) | Raw |
|---|---|---|---|
| table top | O z 0.05; the robot on a pedestal (top at z 0); table x 0.16-0.96, y +-0.50 | O z 0 = the base's mounting plane; x -0.40..1.00, y +-0.80; no pedestal | `stations/sim_rig.yaml scene.table` (`top_z_m: 0.0`, comment lines 60-64); `model.json table {'top_z_m': 0.0, 'x_m': [-0.4, 1.0], 'y_m': [-0.8, 0.8]}`; `model/build.py:422` pedestal optional |
| model sha | `7746a4391b50…` (P1 smoke, 0939 run); `701815be08cd…` in the E7 evals of 01:30-01:49 | `166a0649571ab688ee9b18704c189f7d0aea6c80814a6be52fac10ef1e17b075` | `model.json model_sha256`; `eval.yaml model.sha256` |
| sim box floor | `workspace_min_m` z 0.045 | z **-0.005** (table - 5 mm), in `control_sim.yaml` and `control_sim_trainstart.yaml` | both station files, line 11 |
| deploy home TCP | 30.46 cm above the table | 35.46 cm (TCP O z 0.3546) | `_baseline/hold_still/2026-11-20_0945_p3_g5_refuse_c_deploy` `start_pose.height_above_table_m` 0.3546; FK in `data/p1_standins_fk.json` |
| start pose | (a), 16.18 cm above the old table (21.18 cm above the new one) | (b), 16.18 cm above the new table, same xy and orientation | R3 |
| unchanged | `randomization_start.yaml` (relative to the start pose; built from the demos only, never from the table); the replay placements from the start pose (same height above the table); the eraser region's xy clip (same box xy) | | file contents; DESIGN P3/P4 |
| superseded runs | one: `sim_evals/_baseline/hold_still/2026-11-20_0939_p3_g5_hold/SUPERSEDED.txt` ("run with the old layout (table top O z 0.05, pedestal). Rerun with the table at z 0."), redone as `2026-11-20_0943_p3_g5_hold_z0`. It is the only P3 eval with the old sha. Every other P3 eval (0943-0957) has `166a0649…` | | `find sim_evals -name SUPERSEDED.txt` |

## Analysis

**The sim's `p` is the real `p`.** The sim PC is the copied franka_control with the snapshot applied, and that
snapshot is now upstream `927a5ba`. The scene `key` only pushes into franka_control's own `KeyQueue`, so every
refusal, the joint move, the opening to 0.08 m, the end-of-move box and cone check and the C3 `start` rule run the
real code. Its text reaches the driver unchanged (R2: the null, not-at-home and holding refusals). The real PC applies
the same checks against the real box. That box is identical to the sim box since upstream `447910b`. The copy's own
`config.yaml` box [0.25, -0.40, 0.05] is never used in the sim.

**The three stand-ins share an orientation but not a placement.**
- (a) and (b) are the same TCP pose relative to a table 5 cm apart. Same xy, same fingertip direction (-0.031,
  0.760, -0.649), z 0.2118 vs 0.1618. They differ in q only because the arm must reach 5 cm lower. Joint 2 goes
  0.314 -> 0.371 rad and joint 4 goes -1.783 -> -1.828 rad. The joint margin drops 1.270 -> 1.221 rad, both on joint 7.
- Re-running the E7 IK against today's model reproduces (b) exactly (`data/p1_trainstart_ik_z0.json`). So (b) is
  "(a) for the new table", not a new design.
- (c) keeps the demos' heading (fingertips to the robot's left, about +y). The tilt is rounded to 48 deg, close to the
  demos' median 48.2 deg, and the height to 16 cm (demos' median 16.3 cm). The layout moves 10 cm away from the base:
  eraser centre O x 0.55 instead of 0.45. Why: [2_candidates_envelope.md](2_candidates_envelope.md) and
  [4_gate_check.md](4_gate_check.md).

**Why the E7 trainstart stand-in had to be re-solved.** In today's layout the E7 q puts the fingertips 21.18 cm above the table,
5 cm higher than the demos' median start. The eraser region is placed relative to the start TCP projected on the table
(a horizontal offset only), so the whole scene would sit 5 cm further below the fingertips than in the demos.

**Joint 2: 0.3176 (pc_probe) vs 0.3138 (stand-in).** This is null-space drift, not a wrong value.
- After `p` the arm is within 0.0001 rad of `q_start_rad`. The driver sees 0.000094 rad in the 0939 run, and P1's
  "start pose reached" came 15.09 s after the key.
- The probe ran later, in the torque session that holds the reached pose. That session is the Cartesian impedance
  controller: 800 N/m translational stiffness, but only 0.5 Nm/rad null-space stiffness (`config.yaml` `control`;
  `impedance_controller.cpp:55-61`). It holds the 6-D TCP pose, not the joints, so the redundant 7th degree of
  freedom drifts.
- Joints 2, 4 and 6 move together (+0.0038, +0.0030, -0.0029 rad) while the TCP moves only 0.4 mm down and 0.16 deg in
  tilt (FK of the probe q).
- The 0.4 mm vertical sag matches the gravity compensation gap the sim PC logs: the end effector in the model is
  1.396 kg, compensated 1.366 kg (`data/p1_fc_standin.log`), and 0.030 kg x 9.81 m/s^2 / 800 N/m = 0.37 mm. This is
  an inference; no test isolated it.
- 0.0038 rad is far inside `at_pose_tol_rad` 0.05, so C3 `start` is accepted after the drift.
- For the real arm: a `pc_probe --start-pose` printout taken in the hold can differ from the commanded q by a few
  mrad on joints 2/4/6. That is harmless for `at_pose_tol_rad`. Two probes of the same pose are compared by TCP and
  tilt, not by joints.

### Docs vs raw data

1. Doc says the demos' start tilt is "mean tilt **49.5 deg** (40-64)" at DESIGN.md "### Start pose variant
   `--home trainstart`". Raw says tilt 49.51 deg (of the mean rotation; per-episode mean 49.54, median 48.2) and a
   per-episode range of **38.1-64.0 deg** (episode 21: 38.1, episode 20: 39.4) at `sim_deploy/scene/trainstart.yaml`
   (`start_tilt_deg`, `episodes[*].start_tilt_deg`). The lower bound is 38, not 40. PLAN §9 P4 row and
   `randomization_start.yaml demos.tilt_deg.range` say 38-64.
2. Doc says the eraser is "~17 cm below the fingertips" at PLAN.md §9 (P2 row) and in `pc_probe.py:227-228`
   `_DEMOS_START`. Raw says the eraser centre is a median **0.161 m** below the fingertips (range 0.064-0.272) at
   `sim_deploy/scene/randomization_start.yaml demos.below_m`. 166 mm (`trainstart.yaml start_rel_grasp_median_m` z
   0.16612) is the height above the grasp TCP, a different point.
3. Doc says "after the next torque session's hold 0.0038 rad (joint 2: the Cartesian hold sags **~1 mm**)" at
   DESIGN.md "## P1: start-pose port" ("The stand-in against the ported checks"). Raw says the probe q gives TCP z
   0.2114 vs 0.2118 m, a **0.4 mm** sag (tilt +0.16 deg), at `sim-experiments/01_start_pose/data/p1_standins_fk.json`
   (`P1_pc_probe_printout_q` vs `E7_trainstart_standin`). DESIGN P2 says 0.4 mm, which agrees with raw.
4. Doc says "Every P3 run made with the old table is marked `SUPERSEDED.txt`" at DESIGN.md "### Layout change". Raw
   has exactly **one** such run, `sim_evals/_baseline/hold_still/2026-11-20_0939_p3_g5_hold/SUPERSEDED.txt` (sha
   `7746a439…` in its `eval.yaml`); every other P3 eval has `166a0649…`. Not a contradiction, but the plural overstates
   it.
5. Doc says model versions "`9c75edf8…` (E6), `7746a439…` (E7, D8 tags), `166a0649…` (P3/P4)" at
   sim-experiments/README.md "### Where the raw data lives". Raw: the E7 G5/G7 evals of 2026-11-20 01:30-01:49
   (`_baseline/hold_still/2026-11-20_0130_g5_hold`, `..._0134_g5_hold_deploy`, `..._0136_g5_hold_trainstart`,
   `PICK_TEST1/run_006_2026-11-18/2026-11-20_0149_smoke`) recorded **`701815be08cd…`** in `eval.yaml model.sha256`;
   `7746a439…` first appears in the P1 smoke at 09:17 (`data/p1_fc_null.log`).
6. Doc says "Upstream final-deployment still says 150 mm" at sim-experiments/README.md "### Frames and poses" (TCP
   row), and the real-robot box is "x 0.25, z 0.05 (upstream, 'TODO: measure')" at README.md "### Sim-only settings vs
   the real robot". Raw: upstream `rnoro/final-deployment` `927a5ba` (2026-11-20 09:58) has
   `deployment/contracts/frames.yaml` z **0.23565**, and `447910b` (10:05) has `config.yaml workspace_min_m` **[0.20,
   -0.40, -0.005]** (`git show rnoro/final-deployment:...`). 150 mm and [0.25, -0.40, 0.05] survive only in the
   ea9f7fb copy here.
7. Doc says (PLAN §9, 2026-11-20) the snapshot was "uncommitted at the time". That was true when written; it is now
   upstream commit `927a5ba`. Not an error, only a state change, noted for part 5.

Checked and matching (no discrepancy):
- DESIGN's E7 q, TCP (0.439, -0.164, 0.212), 16.2 cm, joint margin 1.27 rad, and 164 / 166 mm.
- The P3 q and TCP (0.439, -0.164, 0.162) in PLAN §6 and §9.
- The chosen q and TCP (0.531, -0.159, 0.160), 48.0 deg in PLAN §9 and DESIGN.
- The region numbers in DESIGN P2.
- The deploy home 35.5 / 30.5 cm (raw 35.46 / 30.46 cm).
- The P1 smoke times: 15.1 s `p`, 14.5 s `h`.
- The refusal strings.

## Conclusion

1. The real `p` works from home only and only in idle or stopped. It makes a joint move to `homing.q_start_rad` with
   the finger target set to 0.08 m just before the move. At the end it checks the box and the cone (60 deg, no
   margin) and faults if either fails. It is refused while the value is null. Once set, C3 `start` is refused unless
   every joint is within 0.05 rad. The laptop has no `h`; `pc_probe --start-pose` prints the value to paste. The sim
   runs this code unchanged, with a sim-only scene `key` and the driver's `h` -> `p` -> eraser -> `c`.
2. The sim used three stand-ins:
   - the E7 trainstart stand-in (16.18 cm above the old z 0.05 table, 49.51 deg);
   - the P3 stand-in (the same IK for the z 0 table, 16.18 cm, 49.51 deg, reproduced exactly);
   - since 10:18 the chosen pose `leftx55_t48_h16` (TCP O (0.531, -0.159, 0.160), 16.0 cm, 48.0 deg, pointing to the
     robot's left), which is what `stations/start_pose.yaml` holds now.

   All three are reached within 0.0001 rad.
3. The layout change moved the table to O z 0 (model `166a0649…`) and the box floor to -0.005. The deploy home is now
   35.46 cm above the table. The eraser region relative to the start pose and the replay placements did not change.
   One P3 run is superseded.

## Limitations / sim-vs-real caveats

- All three are stand-ins. None is a measured real-arm pose: the real `q_start_rad` is still null upstream.
- The E7 and P3 stand-ins come from the demos' median start relative to the grasp. The handheld recordings'
  orientation is replayed exactly; the placement on the table is a free choice (the policy sees only relative poses).
- `trainstart.py` checks the TCP inside the copy's `config.yaml` box (x >= 0.25, z >= 0.05), not the sim box. It does
  not check the camera view itself. DESIGN's "the whole region in both camera views" for E7 is not verified by the
  script.
- The end-of-move box and cone `fault` paths of `p` were tested only offline (the `start-pose` refusals in DESIGN P2).
  No sim run moved to a bad start pose, because the CLI refuses to write one.
- In the sim the hold after `p` drifts in the null space by about 0.004 rad. On the real arm the drift depends on the
  real payload compensation, which is not set yet (`load` is still 0 upstream: comment in `control_sim.yaml`).
- `p` moves at the real 0.3 rad/s in every eval cited here. The P4 screen used the sim-only 0.6 rad/s
  ([3_early_failure_screen.md](3_early_failure_screen.md)).

## Open questions

1. What is the real `q_start_rad`? Answered by hand-guiding the Franka toward the chosen pose, reading
   `pc_probe --seconds 3 --start-pose`, and loading it with `scene start-pose --from-probe` or `--from-git`
   ([5_real_pc_handoff.md](5_real_pc_handoff.md)).
2. Does the real arm's hold after `p` drift in joints as the sim does? Answered by two `pc_probe --start-pose` reads
   10 s apart after `p` on the arm.
3. Should `scene/start_pose.py:52` `STAND_IN` (still the E7 value for the table at z 0.05, and not used by any code)
   and the E7 files `scene/trainstart.yaml` / `stations/control_sim_trainstart.yaml` (still `q_home_rad` = the E7
   stand-in) be updated or retired? `--home trainstart` is an alias of `start` and no longer reads them (DESIGN P2).
   The owner decides; this part changed nothing.
4. Should the E7-era evals (01:30-01:49, table z 0.05) carry a marker like the superseded P3 run? They were not redone
   under the new layout, so they are not formally superseded. Answered by the coordinator.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| real key `p` code | `deployment/control_deploy/franka_control/{config.yaml, src/supervisor.cpp, src/config_loader.cpp, src/control_loop.cpp, include/franka_control/{config,supervisor,shared_state}.h}`; `deployment/policy_deploy/policy_deploy/manager/{operator,keyboard_operator,pc_probe}.py`; upstream `rnoro/final-deployment` `927a5ba`, `447910b` (git show) |
| sim wiring | `sim_deploy/sim_pc/src/sim_scene.cpp`, `sim_deploy/eval/driver.py`, `sim_deploy/scene/{start_pose,__main__,trainstart,start_sweep}.py`, `sim_deploy/scene/randomization_start.yaml` |
| stand-in files | `sim_deploy/scene/trainstart.yaml`, `sim_deploy/stations/{control_sim_trainstart,start_pose,control_sim,sim_rig}.yaml`, `sim_evals/_start_pose_sweep/poses/{standin_z0,leftx55_t48_h16}.yaml`, `sim_evals/_start_pose_sweep/candidates.json` |
| eval folder(s) | `sim_evals/_baseline/hold_still/2026-11-20_0939_p3_g5_hold/` (SUPERSEDED), `..._0943_p3_g5_hold_z0/`, `..._0945_p3_g5_refuse_c_deploy/`, `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/` (`start_pose.yaml`, `eval.yaml home`), `sim_evals/_replay/PICK_TEST1/pick1-v1_ep000/2026-11-20_1037_campA1_v025/`; `sim_evals/index.csv` rows 2-20 |
| logs | `sim_evals/_logs/2026-11-20_094335/` (P3 G5 hold, z 0) |
| model | `sim_deploy/build/sim_rig/model.json` (`model_sha256` `166a0649…`, `table`) |
| copied into this folder | `data/p1_fc_null.log`, `data/p1_fc_standin.log` (P1 sim-PC-alone smoke, from the P1 agent's session scratchpad `p1/`), `data/p1_p3_g5_refusal_sim_pc.log` (P3 G5 refusal run, from scratchpad `p34/refusal_logs/sim_pc.log`), `data/p1_standins_fk.json` (read-only `describe()` of the stand-ins, the probe q and home in model `166a0649`, computed here), `data/p1_trainstart_ik_z0.json` (read-only `trainstart.compute()` in model `166a0649`, computed here) |
