# 03 W0: re-sync the sim's copy of the deployment with rnoro/final-deployment f632576

**Status:** done, gate passed (2026-11-21 03:35; the port and part of the gripper work by the first W0 coordinator at 20:29-20:47, cut off; finished, smoked and reviewed by the second) · **Date:**
2026-11-20/21 · **Who:** W0 coordinator (port, gripper, start pose, smoke subagents) + reviewer; this skeleton by the docs
lead · **Plan ref:** sim_deploy/PLAN.md §11 (W0 row, "Copy of deployment/" row), §6 (G1, G5, G6), §9 (start pose);
DESIGN.md "E1: as built", "E7: integration" ("D8 settled: training width = fingertip gap"), "P1"

## Question

After porting `ea9f7fb..f632576` into `deployment/` and re-applying the sim-only changes, is the copy = `f632576` +
the listed sim changes only (G1), does the chain run `hold_still` (G5) and a replay, and can the sim gripper with
upstream's 0.04 m max and `gripper_map` still grasp the 51 mm eraser (yes / no, and at what widths)? (Answer: no; the sim's eraser became 0.030 wide, see Conclusion.)

## Why it matters for the real robot

The sim is only a test bench for the arm if the laptop code in it is the laptop code on the arm. Upstream changed what
reaches the PC since `ea9f7fb`: the policy's widths now go through `gripper_map` (both kinds), waypoints are clamped
into the PC's box on the laptop before sending (`manager.workspace_clamp_m` 0.002; on the arm this turned the gate's
z -0.019 rejection into clamps, TRIALS.md), and the max opening is 0.04 m. Every W1-W4 result is about this pipeline,
so W0 comes first. The gripper question is also a real-robot one: upstream maps the robot's 0.030-0.040 m to training's
0.051-0.090 m; if the real eraser is 51 mm wide and the printed fingers open to 40 mm, the eraser cannot be grasped
across its width on the arm either.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`) |
| Upstream | `~/Capstone-final-deployment`, `rnoro/final-deployment`: base of the copy `ea9f7fb`, target `f632576` (18:04); the branch tip is now `0c5903a` (19:41, see below). Read only, `GIT_OPTIONAL_LOCKS=0` |
| Copy before W0 | `ea9f7fb` + E1/E4/E6 sim changes (G1: 18 listed files, each with a header) + P1's port of `sim_deploy/ports/start_pose_snapshot_20261120_0907.patch` (42 files, taken from the uncommitted upstream tree at 09:07) |
| Model | `model_sha256` `166a0649…` before W0; after W0 `96822113…` |
| Home / start pose | `--home start`, `stations/start_pose.yaml` = `leftx55_t48_h16` (the same 7 numbers as upstream `abba9e8`) |
| Eraser | before W0 `sim_rig.yaml eraser.size_m: [0.127, 0.051, 0.012]`; after W0 `[0.127, 0.030, 0.012]` |
| Stations | `stations/control_sim.yaml` + `start_pose.yaml`; `stations/policy_sim.yaml` (laptop) |
| Policy / checkpoint | `hold_still`; `replay:~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip#<i>` (ep 19 passed G6 from the start pose in P3); load checks: run_006 (BP), PICK_DOUBLE run_002 (plain) |
| Inference settings | load checks only: bf16, 8 steps, `exec_h` 12 (upstream's station for run_002) |

### Upstream commits to port (`git log ea9f7fb..f632576 -- deployment`: 5 commits)

| Commit | Time (2026-11-20) | Subject (shortened) | Files (under `deployment/`) | In the copy today |
|---|---|---|---|---|
| `927a5ba` | 09:58 | start pose (PC key `p`, C3 start refused away from it once `homing.q_start_rad` is set), laptop `h` removed; TCP 235.65 mm; box floor z -0.005; docs | 53 files: `franka_control` (supervisor, keyboard reader, key queue, config loader, control loop, shared state + tests), `config.yaml` (`q_start_rad` null, `at_pose_tol_rad` 0.05, `workspace_min_m` z 0.05 -> -0.005), `contracts/frames.yaml` (F_T_TCP z 0.150 -> 0.23565), `policy_deploy` manager (`episode_loop.py`, `operator.py`, `pc_probe.py --start-pose`, `status_line.py`, `failure_monitor.py`, `heartbeat_sender.py` docstring, `config/manager.yaml` comments), docs (DIRECTIONS, STRUCTURE, PLAN, PI_PAYLOAD, TEST_FRAMES) | **partly**: the 09:07 snapshot carries the key `p` and the laptop changes; 11 of 927a5ba's files are not in it (`contracts/frames.yaml` + its test, 3 `franka_control` tests, `control_deploy/tests/test_console.py`, `config/manager.yaml`, `heartbeat_sender.py`, PI_PAYLOAD.md, TEST_FRAMES.md, STRUCTURE.md). The copy has TCP 0.23565 already (E6/D7) and `q_start_rad` / `at_pose_tol_rad`, but `workspace_min_m` is still [0.25, -0.40, 0.05] (the sim station overrides it). Whether the snapshot's 42 files equal 927a5ba's versions: to be diffed by W0 |
| `447910b` | 10:05 | `safety.workspace_min_m` x 0.25 -> 0.20 | `control_deploy/franka_control/config.yaml`, `control_deploy/STRUCTURE.md`, `control_deploy/tests/test_console.py` | no (the sim station has [0.20, -0.40, -0.005] already) |
| `abba9e8` | 15:45 | `control_deploy/station.yaml`: `homing.q_start_rad` from the simulation | `control_deploy/station.yaml`, `control_deploy/tests/test_station.py`, DIRECTIONS x2, PLAN | no; the same q is in `sim_deploy/stations/start_pose.yaml` |
| `b4030f6` | 17:56 | real addresses (PC 192.168.50.186, laptop 192.168.50.227, Pi 192.168.50.59), gripper max 0.04 m, F_T_TCP settled; run_002 diffusion + gripper map | 20 files: `contracts/limits.yaml` (`gripper_width_max_m` 0.08 -> 0.04), `contracts/link.yaml` (addresses), `contracts/frames.yaml` (`settled: true`), `franka_control/config.yaml` (`fake_gripper.start_width_m` 0.04), `policy_deploy/config/policy.yaml` (`gripper_map` block; `warmup_gripper_width_m` 0.05 -> 0.04), `config/manager.yaml` (`workspace_clamp_m: 0.002`), `station.yaml` (Pi address; run_002, bf16, 8 steps, `exec_h` 12), `policy/gripper_map.py` (new, 75 lines), `policy/diffusion_policy.py` + `promptable_diffusion_policy.py` (widths through the map, both kinds), `policy/settings.py`, `manager/settings.py` (the clamp box read from `control_deploy/franka_control/config.yaml`), `manager/pose_converter.py` (the clamp), `manager/policy_step.py` (logs clamps), `manager/__main__.py`, `frames/probe.py` (`--live`), `frames/pi_collect.py` (`--show`), DIRECTIONS x2, `control_deploy/STRUCTURE.md` | no |
| `f632576` | 18:04 | DIRECTIONS: running trials (one start-up block, full policy list, per-trial recording) | `policy_deploy/DIRECTIONS.md` | no |

After the target: `0c5903a` (19:41) adds `policy_deploy/TRIALS.md`, `frames/trial_video.py` and DIRECTIONS / STRUCTURE
lines (plus the repo's AGENTS.md). Not in §11's W0 target; the coordinator decides whether to take it (docs + an
offline video tool, no pipeline change).

### What the port changes for the sim (written before W0: the pre-W0 state; outcomes in Results / Analysis)

1. **Gripper max.** The sim's `SimGripper::move` clamps the hand to `clamp(width - width_offset_m, 0, min(width_max_m,
   maxGap))` (`sim_pc/src/sim_parts.cpp:83`), with `width_max_m` = the contract's `gripper_width_max_m` and maxGap =
   `finger_joint1 + finger_joint2` at full stroke. In the MuJoCo Panda hand each finger joint has `range="0 0.04"`
   (`build/sim_rig/scene.xml`, class `hand_finger`), so **per-finger max 0.04 -> gap up to 0.080 m**
   (`sim_config.h gripper_max_gap_m = 0.08`). Upstream's `gripper_width_max_m: 0.04` is **not** a per-finger value: it
   is the whole opening (libfranka's width, the hand's `grip` reading, "max fingertip opening with the printed
   fingers", `contracts/limits.yaml`; upstream control_deploy/DIRECTIONS shows `grip 0.040 m` with the hand open).
   Ported as is, the sim's gap would top out at min(0.04, 0.08) = **0.040 m < the eraser's 0.051 m**: the fingers could
   no longer straddle the eraser across its width.
2. **`gripper_map` semantics** (`policy/gripper_map.py` at `f632576`): one straight line through two points, used both
   ways, kind `diffusion` only (so BP and plain alike; not `replay`, not `hold_still`):
   training_closed_m 0.051 <-> robot_closed_m 0.030; training_open_m 0.090 <-> robot_open_m 0.040
   (`config/policy.yaml`). Scale = (0.040 - 0.030) / (0.090 - 0.051) = 0.010 / 0.039 = 0.2564 robot m per training m.
   - `to_robot` (model -> PC): 0.051 -> 0.030; 0.054 (the sim's training hold width) -> 0.030 + 0.003 x 0.2564 =
     0.0308; 0.090 -> 0.040; then clipped to [0, 0.04] by `Policy.make_chunk`.
   - `to_model` (C4 observation -> model): the robot width is first clamped to [0.030, 0.040], so any reading >= 0.040
     goes to the model as 0.090 (fully open) and any reading <= 0.030 as 0.051 (closed on the eraser).
   - Upstream's reading: training's width is the handheld's ArUco marker distance (0.051 closed on the eraser .. 0.090
     open); the robot holding the eraser reads 0.030. The sim's reading (D8 revised by E7, `control_sim.yaml`
     `width_offset_m: 0.0`): training's width is the fingertip gap; the sim holds the eraser at 0.052-0.056 (20 N at
     about 0.047, `sim_rig.yaml`). With the sim's 51 mm eraser and the map, a held eraser (gap ~0.054) would read as
     0.090 = "fully open" to the model.
   - W0 has to settle which reading the sim follows and record it (e.g. the real eraser's width across the
     fingertips, the printed fingers' real max opening as a gap, `width_offset_m` once measured on the arm, upstream
     control_deploy/DIRECTIONS step 8 "The finger offset"), and say what the sim eraser / gripper settings become. Any
     sim-only deviation goes in the sim-only list (G1) and in the top README's "Sim-only settings".
3. **Replays.** `replay_policy.py` (sim-only, E1) sends training widths clipped to [0, `gripper_width_max_m`]
   (`base_policy.make_chunk`); the map is not applied to kind `replay`. With 0.04 every recorded width (0.051-0.090)
   becomes 0.040, so a replay would never close on the eraser. The "chain runs a replay" gate needs a decision here
   (map the replay's widths the same way, or keep the replay as a motion check and say so).
4. **The laptop clamp** reads the box from the copy's `control_deploy/franka_control/config.yaml`
   (`manager/settings.py PC_CONFIG`), while the sim PC's gate uses `stations/control_sim.yaml`. After porting
   `927a5ba` + `447910b` both are [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70]; W0 checks that they agree. Box faults seen
   so far (E7 G7: z 0.71 > 0.70; P3 G7: y -0.411) would become clamps, so W3's fault counts are not comparable with
   E7 / P3 runs.
5. **Addresses, `settled`, station.** The sim runs `franka_control --sim --local` and `policy_deploy run --mock`
   (127.0.0.1); W0 confirms the real addresses in `link.yaml` are not used then. `stations/policy_sim.yaml` stays the
   sim's laptop station (Pi 127.0.0.1, `hold_still` by default, set per eval by `policy_deploy use`); upstream's
   `station.yaml` (run_002, Pi 192.168.50.59) is not copied over it. `settled: true` removes the check the sim skipped
   under `--mock`.
6. **Start pose.** `python -m sim_deploy.scene start-pose --from-git` reads `rnoro/final-deployment:deployment/
   control_deploy/station.yaml` `franka_control.homing.q_start_rad` (default ref = the branch, i.e. `0c5903a` now; the
   file is the same at `f632576`) and writes `stations/start_pose.yaml` after the box / cone checks; with the same
   7 numbers it is a no-op apart from the file's source line.

## Method

One subagent per step, in this order; the reviewer gates the workstream.

1. **Port.** Three-way merge of `git diff ea9f7fb f632576 -- deployment` into the copy, file by file, keeping each
   file's header ("copied from ... @ <sha>; changes: ...") and updating it to `f632576`. Diff the 09:07 snapshot's 42
   files against 927a5ba's versions first. Output: the new sim-only change list (file, why) = G1's "listed changes".
2. **Gripper.** Decide and document the width semantics (above); set the sim's gripper / eraser accordingly (sim-only
   files); offline grasp check as in E7 (pad forces at the hold width, no slip) before any chain run.
3. **Start pose.** `scene start-pose --from-git --dry`, then without `--dry`; `scene check`.
4. **Rebuild + smoke.** Rebuild the sim PC (`deployment/control_deploy/franka_control/build-sim/`) and the model if
   needed; G1 (diff report + `policy_deploy check --mock` against the running sim); G5 (`hold_still`: no motion,
   start pose reached); one replay from the start pose; `policy_deploy use` + warm-up for run_006 and run_002 to see
   that both kinds load with the gripper map (no rollout needed).

Pass: G1 diff = `f632576` + the listed sim changes only; `check --mock` all PASS; `hold_still` drift within G5's
numbers (P3: 0.07 mm / 0.12 deg over 20 s); the replay runs to its end without an infrastructure failure (grasp
expected only if step 2 makes it possible).

## Commands to reproduce

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# 1. G1: the copy vs f632576 (read only on upstream)
S=$(mktemp -d); GIT_OPTIONAL_LOCKS=0 git -C ~/Capstone-final-deployment archive f632576 deployment | tar -x -C $S
diff -rq $S/deployment deployment -x build-sim -x '*.egg-info' -x __pycache__     # 19 files, all sim-only
# 2. gripper: model + offline grasp
~/envs/sim_deploy/bin/python -m sim_deploy.model build --station sim_deploy/stations/sim_rig.yaml
~/envs/sim_deploy/bin/python -m sim_deploy.scene check
~/envs/sim_deploy/bin/python sim_evals/_pipeline/W0/gripper/grasp_check.py --target 0.0      # and --target 0.0308
# 3. start pose
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose --from-git f632576 --dry
# 4. rebuild + smoke (one chain at a time; CPU load < 3: other agents' jobs broke the first hold_still)
cmake --build deployment/control_deploy/franka_control/build-sim -j8
sim_deploy/run.sh --policy hold_still --rollouts 1 --timeout 20 --tag w0_g5_hold
sim_deploy/run.sh --policy "replay:$HOME/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip#19" --rollouts 1 --timeout 60 --tag w0_replay19
# check --mock while a sim chain runs (in a pty, from deployment/policy_deploy)
~/envs/sim_policy_deploy/bin/policy_deploy check --mock --station ../../sim_deploy/stations/policy_sim.yaml
# load check, both kinds (GPU lock), from deployment/policy_deploy
flock ../../sim_evals/.gpu.lock ~/envs/sim_policy_deploy_torch/bin/python ../../sim_evals/_pipeline/W0/load_check/load_check.py \
  ~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 ~/Capstone/data/PICK_DOUBLE/policy/run_002_2026-11-19
```

## Results

| Check | Result | Pass? |
|---|---|---|
| G1 copy vs f632576 | 19 files differ, all listed sim-only (E4 `--sim` 5, P1 `Log::setTap` 2, E1 `fake_pi` 5, E1 `replay` 7 incl. STRUCTURE.md); everything else byte-identical, nothing missing. The 09:07 snapshot and D7 `frames.yaml` are no longer differences | pass |
| `check --mock` against the running sim | every row PASS (Pi reachable, terminal, `settled: true`, `gripper_width_max_m 0.04`); the PC is 127.0.0.1 (link.yaml's real address unused) | pass |
| Box | sim PC gate = laptop clamp = config.yaml [0.20, -0.40, -0.005]..[0.80, 0.40, 0.70]; no sim override left | pass |
| Start pose | `--from-git f632576` = the 7 numbers already in `start_pose.yaml` (no-op); TCP (0.531, -0.159, 0.160), 48.0 deg, all checks ok | pass |
| Model | rebuilt, sha `96822113…` (eraser 0.030 wide; finger joints 0..0.020 each) | - |
| Offline grasp (arm held, eraser hanging) | grasp to 0 (closes fully): gap 22.0 mm, 20.0 N per pad, slip 0.5 mm in 5 s; target 0.0308: gap 31.6 mm, 7.8 N per pad, slip 0.006 mm | held |
| G5 `hold_still` 20 s | timeout (= held), max TCP drift 0.071 mm (P3: 0.070, same measure), gate 4928 accepted / 0 rejected, fingers at 0.040 | pass |
| Replay PICK_TEST1 #19 from the start pose | success: grasp at 2.2 s, lift 16.5 cm; 47 waypoints shortened (v_max 0.25), 1 chunk clamped (z -0.003), no fault; the fingers closed to a 1.9 mm gap: the eraser was pinched at its end, not across its width (video) | pass (runs to the end); grasp not the demo's |
| Load: run_006 (BP) / run_002 (plain) | `use` ok; `PromptableDiffusionPolicy` 19 s build, warm-up 0.24 / 0.12 s, prompt PICK_TEST1 #14 / `DiffusionPolicy` 6 s, 0.12 / 0.10 s; both bf16, 8 steps, exec_h 12, the f632576 map; chunk widths 0.030-0.040 | pass |

## Analysis

- **Port.** Nothing was left to port: the first coordinator's rewrite at 20:29 made every non-sim file upstream's
  bytes, and every sim-only file carries the f632576 header. The only new sim behaviour is the replay's width map.
- **Gripper semantics.** Training's width is the fingertip gap (D8/E7: 0.051 = closed on the 51 mm eraser, 0.090 open)
  and upstream's `gripper_map` reads it the same way at its two anchor points. The sim follows upstream on the robot
  side (stroke 0.040 measured on the real fingers, 39-41 mm in the 2026-11-20 arm images; map; offset 0; grasp closes
  until stall). With flat parallel pads a 51 mm eraser never fits in 40 mm, so either the object or the robot has to
  differ from reality; changing the robot would hide what the arm does, so the sim's object is narrowed to upstream's
  own `robot_closed_m` 0.030. With it, every mapped close (training <= ~0.07 -> robot < 0.035 = open 0.040 - deadband)
  is a `grasp()` that stalls on the eraser, C2 reads ~0.022, and `to_model` returns 0.051 (closed) to the model.
- **Replay #19.** The replay reached and lifted the eraser, but by its end: the open-loop path from the start pose does
  not centre the eraser between the pads. As a motion/pipeline check it passes; it is not evidence that a centred
  grasp across the width works in the chain (the offline check is).
- **CPU.** With W1/W2 jobs running (load 8-14) the first `hold_still` failed on a 0.1 s frames health timeout; runs at
  load < 3 were clean. Chain runs need a quiet CPU, not only free ports.

## Conclusion

Gate passed: the copy is f632576 + 19 listed sim-only files, `check --mock` passes, `hold_still` holds as in P3, a
replay runs through the full chain, and both checkpoint kinds load with the gripper map. Gripper decision: robot side
as upstream; sim eraser 30 mm across the grasp (sim-only). **For the real robot:** the user's 51 mm eraser cannot be
grasped across its width with the 40 mm fingers; upstream's `robot_closed_m 0.030` assumes a 30 mm object.

## Limitations / sim-vs-real caveats

- The 30 mm eraser is held at a 22 mm gap: the soft pad layer is compressed ~8 mm, not physical; `to_model`'s clamp
  hides it (the model still sees 0.051). On the arm the hand would read ~0.029-0.030.
- W3: run_002 answers a closed observation with 0.0345-0.0365 (load check): within the 5 mm deadband of a typical
  close (~0.0345), but after a close at <= 0.0315 a later 0.0365 is a `move` that opens the fingers and drops the eraser.

- The printed fingers' real opening, the real eraser's width across the fingertips and `width_offset_m` are not
  measured (upstream `width_offset_m: 0.0  # TODO: measure with the printed fingers`); whatever W0 chooses is a
  stand-in until they are.
- The fin-ray compliance is modelled with explicit pad contacts (E7); hold widths under the map are not checked
  against the real fingers.

## Open questions

1. Is upstream's 0.04 m the fingertip gap or the hand reading with an unknown offset? Answered by the caliper
   measurement in upstream control_deploy/DIRECTIONS step 8 ("The finger offset").
2. How wide is the real eraser across the fingertips (upstream `robot_closed_m: 0.030  # the eraser's width`) vs the
   sim's 0.051? Answered by measuring the eraser and reading `grip` while the arm holds it.
3. Should `replay` get the gripper map too? **Yes (W0):** `replay_policy.py` maps widths with `to_robot`.
4. Take `0c5903a` as well? **No (W0):** docs + an offline video tool, no pipeline change.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| G1 diff report | `sim_evals/_pipeline/W0/G1/` (`G1_report.md`, `copy_vs_f632576.diff`, `check_mock.txt`) |
| gripper offline check | `sim_evals/_pipeline/W0/gripper/` (`grasp_check.py`, two outputs) |
| load check | `sim_evals/_pipeline/W0/load_check/` (`load_check.py`, `.json`, `.log`, station copies) |
| G5 `hold_still` | `sim_evals/_baseline/hold_still/2026-11-21_0323_w0_g5_hold/` (failed try under load: `2026-11-21_0319_w0_g5_hold/`; host for check: `2026-11-21_0332_w0_g1_check_host/`) |
| replay smoke | `sim_evals/_replay/PICK_TEST1/pick1-v1_ep019/2026-11-21_0327_w0_replay19/` |
| logs | `sim_evals/_logs/<date>_<time>/` |
| port sources | `sim_deploy/ports/start_pose_snapshot_20261120_0907.{patch,README,untracked.txt}`; upstream `ea9f7fb..f632576` |
