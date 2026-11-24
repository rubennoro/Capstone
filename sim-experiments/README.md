# sim-experiments: the record of every sim experiment

This folder is the full, readable record of every experiment run with the MuJoCo deployment sim. For each one it
gives the question, method, exact setup, commands to reproduce, results, analysis, conclusion and open questions,
so someone new can follow it and run it again. It is tracked in git. The raw results it cites stay in the
gitignored `sim_evals/`.

## What the sim is

`sim_deploy/` is a MuJoCo stand-in for the robot side of the real deployment. It plays the **Pi** (the rig's
IMX519 pinhole and IMX219 fisheye, rendered at 30 fps and streamed as `pi_stream/v1` JPEGs on :5555 with a
simulated Pi clock) and the **PC** (the copied C++ `franka_control` with a MuJoCo arm and gripper behind its
`Arm`/`Gripper` interfaces, physics stepped 1 ms per control cycle in wall-clock time: the real safety gate,
supervisor, watchdog, impedance controller and C2/C3 codec run). The laptop's `policy_deploy` is a copy of
`rnoro/final-deployment@ea9f7fb` plus the start-pose port (key `p`, P1) and the listed sim changes; it runs as
`policy_deploy run --mock` with `launcher.fake_pc: false` and `fake_pi: false`, so it talks to the sim as it talks to
the robot and skips only the hardware checks (Wi-Fi, laptop address, `settled`). An eval driver resets the scene
(Panda + Franka Hand + soft fin-ray fingers + rig, a table, a 127 x 51 x 12 mm yellow eraser), presses the PC's keys
(`h`, `p`, through the sim-only scene channel) and the laptop's (`c`, `y`/`n`), judges each episode from the sim's
ground truth and writes `sim_evals/`.
How it is built: [sim_deploy/PLAN.md](../sim_deploy/PLAN.md), [sim_deploy/DESIGN.md](../sim_deploy/DESIGN.md).
How to run it: [sim_deploy/DIRECTIONS.md](../sim_deploy/DIRECTIONS.md).

## How the experiments are organised

- One folder per experiment, numbered in the order they were started: `NN_<name>/`. A big experiment has a
  `README.md` (overview, index of its parts, the overall conclusion) and one file or subfolder per part.
- Every experiment file follows [TEMPLATE.md](TEMPLATE.md): Question; Why it matters for the real robot; Setup;
  Method; Commands to reproduce; Results; Analysis; Conclusion; Limitations / sim-vs-real caveats; Open questions;
  Raw data.
- Small results (CSV, markdown tables, figures up to ~1-2 MB) are copied into the experiment folder (`data/`,
  `figures/`). Large raw data (eval folders, videos, `.npz` arrays, big JSONs) stay in `sim_evals/` and are linked
  by path. The whole folder stays below ~20 MB.
- Numbers are taken from the raw files. Where `PLAN.md` / `DESIGN.md` say something different, the experiment's
  Analysis has a "Docs vs raw data" list; the raw value is the one used.
- A run that was redone under changed conditions (e.g. the table moved) is kept and marked **superseded** here. In
  `sim_evals/` only the P3 runs made with the old table carry a `SUPERSEDED.txt`; the E7 runs (old table too) do not,
  so check the model sha and the table height (below) before reusing an old run.

## Conventions

### Frames and poses

| Term | Meaning |
|---|---|
| O | the robot base frame (libfranka's O): x away from the robot, y to the robot's left, z up. Table and eraser poses are in O. |
| table | since P3/P4 (2026-11-20, model `166a0649…`) the base sits on the table: table top = O z 0, x -0.40..1.00, y ±0.80. Before that (E6, E7, P1/P2 and the first P3 run) the table top was at O z 0.05 (x 0.16-0.96, y ±0.50) with the robot on a pedestal (top z 0). |
| TCP | the fingertip point: `F_T_TCP` = +45 deg about flange z, **235.65 mm** out (D7; the copy's `deployment/contracts/frames.yaml`). Upstream final-deployment had 150 mm at `ea9f7fb` (the copy's base) and has 235.65 mm since `927a5ba` (2026-11-20). `settled: true` in the copy since W0 (= f632576); upstream `settled: true` since `b4030f6`. |
| tilt | angle of the fingertip direction (TCP z) from straight down, in deg (`scene/start_pose.py tcp_numbers`). With this `F_T_TCP` it equals the cone tilt the gate checks (flange z vs the cone axis straight down). |
| heading | the fingertip direction (TCP z) projected on the table. In words as `pc_probe` prints it: "to the robot's left" = +y, "to the robot's right" = -y, "forward (away from the base)" = +x, "back (toward the base)" = -x when within 25 deg of one of these, else "between A and B"; none when the tilt is near 0. Pose ids use `left` 90, `right` -90, `away` 0, `away_left` 45, `away_right` -45 deg (from +x toward +y). |
| height | fingertip (TCP) height above the table, in cm. |
| pose ids | `<heading>[x<G>]_t<tilt>_h<height>` (`scene/start_sweep.py candidates`), e.g. `leftx55_t48_h16` = heading left, tilt 48 deg, 16 cm, with the demos' median eraser centre at O (0.55, 0). Without `x<G>`: (0.45, 0) for left / right, (0.50, 0) for the diagonals, (0.60, 0) for away. `standin_z0` = the P3 stand-in. See [01_start_pose](01_start_pose/README.md). |
| homes | `--home start` (default since P2): the PC's `h`, then the PC's `p` to `homing.q_start_rad` (`sim_deploy/stations/start_pose.yaml`), eraser placed relative to the start pose. `--home deploy`: `h` only, `q_home_rad` of the copied `config.yaml`, fingers straight down with the TCP at O (0.307, 0.000, 0.355): 35.5 cm above the table at z 0 (30.5 cm above the old table at z 0.05); eraser from `scene/randomization.yaml` (x 0.27-0.34, y -0.03..0.03, yaw -8..+12 deg); `run.sh` starts the sim PC with `q_start_rad: null`. `--home trainstart`: an alias of `start` since P2; before P2 (E7 runs) a separate home at E7's training-like pose (`stations/control_sim_trainstart.yaml`). |
| ahead / left (placement) | `--home start`: eraser centre offsets in the start pose's table frame (`scene/randomization_start.yaml`): origin = the start TCP projected on the table, ahead = the fingertip direction projected on the table, left = up x ahead. Ranges = the PICK_TEST1 demos' 10th-90th percentiles: ahead 0.090-0.217 m, left -0.035..-0.003 m (i.e. right of the fingertips), yaw -6.6..+0.8 deg; the centre is clipped to the table and the box. |
| width | `gripper_width_m` = the fingertip gap (D8 revised by E7): 0 closed, about 0.054 holding the eraser (0.052-0.056 in training), 0.040 the printed fingers' full stroke on the robot and in the sim since W0 (0.080 before W0; the policy's widths are clipped to it), up to 0.090 in training (handheld). Upstream since `b4030f6`: `gripper_width_max_m` 0.04 = the printed fingers' **whole** opening (not per finger), and `policy/gripper_map.py` maps training widths, read there as the handheld marker distance (0.051 closed on the eraser, 0.090 open), linearly onto the robot's 0.030 / 0.040; `p` opens to 0.04 and `pc_probe` prints 40.0 mm. A 0.040 m gap is narrower than the 51 mm eraser: W0 keeps the robot side as upstream and narrows the sim's eraser to 0.030 ([03 W0](03_policy_pipeline/W0_resync/README.md)). |
| seeds | episode n of an eval with `--seed S` draws its eraser pose from `numpy.random.default_rng([S, n])`: the same seed gives every policy the same placements. |

Units: m / cm / mm for lengths, deg for tilts and cones, rad for joints, ms for timing; always written out.

### What the judge calls success, grasp, lift, approach

The eval driver (`sim_deploy/eval/driver.py`, `JUDGE` dict: `lift_m` 0.05, `hold_s` 1.0, `lifted_m` 0.02,
`off_table_m` 0.03, `poll_hz` 20) polls the sim's ground truth at 20 Hz during an episode:

| Word | Definition |
|---|---|
| grasp (`grasped`) | at the same poll, geoms of both fingers in contact with the eraser: names matching `^(pad\|finger_holder)_(l\|r)` (the pads, the UMI holders) |
| lift (`lifted`) | grasped and the eraser centre >= 2 cm above its settled start at the same poll |
| success | eraser centre >= 5 cm above its settled start at every poll for 1 s (no grasp test: the lift is what counts) |
| approach | screen class only (`scene/start_sweep.py classify`): no grasp, closest TCP to the settled eraser centre < 5 cm (from `trace.json`) |
| no_progress | screen class: none of the above |
| fault_early_<rule> | screen class (checked first): outcome `fault` within 10 s; rule from the fault text: box, cone, reflex, v_max, or other |

Episode outcomes (`episodes.jsonl` `outcome`):
- `success`; `timeout` (`--timeout`, default 60 s); `fail` (the eraser centre fell > 3 cm below the table top, or
  policy_deploy ended the episode itself with an outcome other than stopped / aborted);
- `fault` (C2 mode `fault`: the gate refused a waypoint, a reflex; latched as on the robot, so `run.sh` restarts the
  sim PC and Pi and resumes the eval folder, up to 5 times); `stopped` (C2 `stopped`, e.g. the watchdog ~1 s after
  the last C3, or policy_deploy ended the episode as stopped / aborted);
- `crashed` (policy_deploy exited during the episode); `sim_error` (the ground-truth request failed);
- never started: `home_failed` (`h` gave no home), `start_pose_failed` (`p` refused or did not reach the pose),
  `start_failed` (`c` not accepted within 30 s, e.g. "not at the start pose");
- `not_run` (policy_deploy did not start, e.g. the sim PC latched a fault; a resume re-runs these);
  `interrupted` (Ctrl-C during the episode).

`metrics.json` (`eval/report.py metrics`) and the `index.csv` row: success, grasp and lift rates over **every
recorded episode except `interrupted`** (the never-started and `not_run` ones count as non-successes), the success
rate with its Wilson 95 % interval; `outcomes` (counts, `interrupted` included); `faults` (episodes with a fault:
C2 fault or crashed), `stops`, `timeouts`; `t_grasp_s` / `t_lift_s` / `t_success_s` / `duration_s`; `steps`:
sent, skipped by reason, `policy_ms_p50`, `policy_ms_p95_max`, `cycle_ms_p50`, `margin_ms_min`, `margin_ms_p50`.

### Sim-only settings vs the real robot

"Real robot" = `rnoro/final-deployment` at `f632576` (2026-11-20). Since W0 the copy in `deployment/` is f632576 +
19 sim-only files ([03 W0](03_policy_pipeline/W0_resync/README.md)); the sim stations override it where listed. Everything not listed is the copy's
value.

| Setting | Real robot | Sim | Where | Used by |
|---|---|---|---|---|
| workspace box min | [0.20, -0.40, -0.005] (upstream since `927a5ba` / `447910b`; the copy's `config.yaml` too since W0) | the copy's [0.25, -0.40, 0.05] in the E7 chain runs (01:30-01:49: their gate texts say `min 0.050`), then [0.20, -0.40, 0.045] with the old table (E7 fix 12, after those runs), **[0.20, -0.40, -0.005]** since P3/P4; max [0.80, 0.40, 0.70] in both | `stations/control_sim.yaml`, `control_sim_trainstart.yaml` | every sim run; no longer sim-only |
| `v_max_m_per_s` | 0.25 | 0.25; sim-only 0.5 | overlay `stations/vmax05_g6.yaml` | P3 G6 replays only (tag `p3_g6_vmax05`) |
| `v_max_m_per_s` | 0.25 | sim-only 1.5 | overlay `stations/vmax15_campA.yaml` | camera study A1 replays only (tag `campA1_v15`) |
| homing speed (`homing.max_joint_speed_rad_per_s`) | 0.3 rad/s | sim-only 0.6 rad/s | overlay `stations/homing_fast_sweep.yaml` (`start_sweep --fast-homing`) | P4 screen only (the policy never sees homing) |
| `homing.q_start_rad` | the same 7 numbers in upstream `control_deploy/station.yaml` since `abba9e8` (from the sim; not yet checked on the arm) | stand-in, now `leftx55_t48_h16` (before: the P3 stand-in, E7's trainstart) | `stations/start_pose.yaml` | every `--home start` run |
| payload (`load`) | mass 0 (TODO: weigh the rig) | mass 0 since 2026-11-22 (= real; until then 0.6362 kg, CoM and inertia of rig + finger kits from the model, `sim_pc/payload.py`) | `control_sim.yaml franka_control.load` | runs before 2026-11-22 |
| finger contact | soft fin-ray | explicit pad-eraser contact pairs (3.5 mm margin, solref -200/-28, stiff friction) + finger servo kp 20000 N/m | `stations/sim_rig.yaml finger_contact`, `control_sim.yaml sim.gripper.kp_n_per_m` | every run since E7 |
| eraser width across the grasp | the user's eraser, 51 mm (does not fit the printed fingers' 40 mm opening) | **0.030 m** (= upstream `gripper_map.robot_closed_m`); W2's look stations keep 51 mm for images | `stations/sim_rig.yaml scene.eraser.size_m` | every run since W0 (model `96822113…`) |
| finger stroke / grasp | 0..0.040 (f632576), Franka Hand `grasp()` closes until it stalls | the same: finger joints 0..0.020 each, `sim.gripper.grasp_closes_fully: true` | `sim_rig.yaml finger_max_gap_m`, `control_sim*.yaml sim.gripper` | every run since W0; not sim-only |
| width offset | 0.0 | 0.0 (E7; wave 2 used 0.0443, wrong) | `control_sim.yaml gripper.width_offset_m` | every run since E7 |
| appearance | real lab | procedural light-wood table, directional key light with shadows + fill, kit colours, ArUco sticker textures, neutral grey floor and sky | `stations/sim_rig.yaml appearance`, `scene.lights` | every run; studied in [02_camera_study](02_camera_study/README.md) (paused), now [03 W2](03_policy_pipeline/W2_sim_look/README.md) |
| cameras | real IMX519 / IMX219 | renders: IMX519 pinhole with the nominal K (never calibrated), IMX219 fisheye with the measured lens; camera poses from the mount STEP (G3: in real handheld frames the IMX219 sits ~9 mm closer to the fingers than `rig_pi_v3`'s CAD says); no motion blur | `sim_pi/`, `sim_rig.yaml` (`rig_file` = the copy's `rig_pi_v3_640.yaml`) | every run |
| Pi stream timing | the live Pi | the live Pi's exposures (25.6 / 33.3 ms), IMX219 phase +16.3 ms, send delay 45.8 ms; Pi clock offset +1000 s, drift 20 ppm | `sim_pi/settings.py` defaults (no `sim_pi:` block in `sim_rig.yaml`) | every run |

Overlays are merged last into the sim PC's station by `run.sh` via `SIM_DEPLOY_SIM_PC_OVERLAY=<file>`
(`sim_pc/merge_station.py`); they may not set `q_start_rad`. `eval.yaml` records only the merged station's **path**
(`home.sim_pc_station`, `build/control_sim_<home>.merged.yaml`, overwritten at the next sim start), not the overlay:
an overlay shows only in the eval's tag and in `sim_pc.log` (e.g. the gate's "v_max 0.50 m/s" lines). Name it in
the tag and in the experiment's Setup.

### Where the raw data lives

```
sim_evals/index.csv                                      one row per finished eval folder (written at its end)
sim_evals/<task>/<run_folder>/<YYYY-MM-DD_HHMM>_<tag>/   a learned policy (e.g. PICK_TEST1/run_006_2026-11-18/...)
sim_evals/_baseline/hold_still/<YYYY-MM-DD_HHMM>_<tag>/  the hold_still baseline
sim_evals/_replay/<task>/<dataset>_ep<NNN>/<...>_<tag>/  an open-loop replay of a recorded episode (e.g. pick1-v1_ep019)
sim_evals/_start_pose_sweep/                             P4: candidates.{md,json}, poses/, demo_paths.json, gate/,
                                                         p3/ (offline G7), tools/, summary.{csv,md}
sim_evals/_camera_study/                                 PLAN section 10 camera study (A/, B/, ...)
sim_evals/_dry_run/                                      --dry-run checks (no sim, no index row)
sim_evals/_logs/<YYYY-MM-DD_HHMMSS>[_<tag>]/             run.sh: sim_pc.log, sim_pi.log, eval.log (+ model.log when
                                                         it built the model); _logs/ui/ the UI's logs
```

Each eval folder: `eval.yaml` (command, policy and checkpoint, seeds, judge, home and start pose, scene and
randomisation, model dir + `model.sha256`, commits), `station.yaml` (the frozen policy_deploy station), `use.log`,
`start_pose.yaml` (`--home start`: a copy of the file used), `episodes.jsonl`, `metrics.json`, `summary.html`,
`policy_deploy.log`, `session/` (policy_deploy's session folders), `episodes/<nnn>/` (`trace.json`: the 20 Hz
t, lift, grasp, finger gap and TCP pose in O; `external.mp4`; `session_episode.json`; `steps.jsonl`; `tap/` with
`--tap`). Dry runs add `check.log`, `dry_run.json`. Markers: `SUPERSEDED.txt`, `INTERRUPTED.txt`, `NOTE.txt`.
`sim_deploy/` and `deployment/` are uncommitted, so `eval.yaml` `git.sim_deploy` reads `3633fcb…-dirty` for every
run: the model sha and the date are what identify the sim version.

Model versions (`build/sim_rig/model.json` `model_sha256` = `eval.yaml` `model.sha256`, checked by the sim PC, the
sim Pi and the driver at start):

| sha | Built | Table top | Evals that used it |
|---|---|---|---|
| `cc928f83…` | E6, first build | z 0.05 | dry runs 00:43-00:47 only |
| `9c75edf8…` | E6, final | z 0.05 | dry runs 00:53 only |
| `701815be…` | E7, during integration | z 0.05 | the E7 chain runs 01:30-01:49: G5 hold_still x4, G5 fault replays (ep 3) x2, G6 replay ep 2, G7 smoke run_006 |
| `7746a439…` | end of E7 (PLAN G2 "now"), used through P2 | z 0.05 | P2 dry runs 09:23-09:30; `_baseline/hold_still/2026-11-20_0939_p3_g5_hold` (superseded) |
| `166a0649…` | P3/P4: base on the table | **z 0** | every run since 09:43 (P3, the P4 sweep, the camera study); the current `model.json` (built 2026-11-20 14:40 UTC) |

## Checkpoints

The 14 runnable policies: the run folders under `~/Capstone/data/{PICK_TEST1,PICK_DOUBLE}/policy/` with a
`checkpoints/*.ckpt` (one each; details in `~/Capstone/data/RUNS.md`). Pass the folder to the driver
(`--policy ~/Capstone/data/<FOLDER>/policy/<run>`); its results go to `sim_evals/<FOLDER>/<run>/`. BP needs
`--precision bf16 --diffusion-steps 8 --exec-h 12` on this laptop.

| Run folder | Policy | Dataset | Episodes | Checkpoint (epoch, val MSE) | In the sim so far |
|---|---|---|---|---|---|
| `PICK_TEST1/policy/run_020_2026-11-19` | BP | `pick1-ep10-v1` | 10 | 59, 0.0008353 | - |
| `PICK_TEST1/policy/run_012_2026-11-18` | BP | `pick1-ep15-v1` | 15 | 59, 0.0004078 | - |
| `PICK_TEST1/policy/run_014_2026-11-18` | BP | `pick1-ep20-v1` | 20 | 59, 0.0003294 | - |
| `PICK_TEST1/policy/run_016_2026-11-18` | BP | `pick1-ep25-v1` | 25 | 59, 0.0002300 | - |
| `PICK_TEST1/policy/run_006_2026-11-18` | BP | `pick1-v1` | 30 | 59, 0.0002007 | G7 smoke (E7), P3 G7, P4 screen |
| `PICK_TEST1/policy/run_030_2026-11-19` | plain | `pick1-ep10-v1` | 10 | 50, 0.0008616 | - |
| `PICK_TEST1/policy/run_028_2026-11-19` | plain | `pick1-ep15-v1` | 15 | 59, 0.0003967 | - |
| `PICK_TEST1/policy/run_031_2026-11-19` | plain | `pick1-ep20-v1` | 20 | 59, 0.0003717 | - |
| `PICK_TEST1/policy/run_024_2026-11-19` | plain | `pick1-ep25-v1` | 25 | 59, 0.0002694 | - |
| `PICK_TEST1/policy/run_022_2026-11-19` | plain | `pick1-v1` | 30 | 50, 0.0002123 | - |
| `PICK_DOUBLE/policy/run_003_2026-11-19` | plain | `pick1-double-ep15-v1` | 30 (15+15) | 59, 0.0002274 | - |
| `PICK_DOUBLE/policy/run_004_2026-11-19` | plain | `pick1-double-ep20-v1` | 40 (20+20) | 59, 0.0003348 | - |
| `PICK_DOUBLE/policy/run_005_2026-11-19` | plain | `pick1-double-ep25-v1` | 50 (25+25) | 50, 0.0005648 | - |
| `PICK_DOUBLE/policy/run_002_2026-11-19` | plain | `pick1-double-v2` | 62 (30+32) | 40, 0.0004976 | - |

BP = behavior prompting (`umi_policy_dunetp`: no `TRAIN_CONFIG` line in `run_config/submit_config.env`); plain =
plain diffusion (`TRAIN_CONFIG="umi_policy_dunet"`). PICK_TEST1 = one recording day; PICK_DOUBLE = PICK_TEST1 +
PICK_TEST2 (the first N of each day). `PICK_DOUBLE/run_002` is the real-robot deployment candidate.

## Index

| # | Experiment | Date | Status | Model | Finding (one line) |
|---|---|---|---|---|---|
| 00 | [Integration checks G1-G7](00_integration_checks/README.md) | 2026-11-20 | done: G1-G5 pass, G6 conventions pass, G7 policies not functional | `701815be…` (E7), `166a0649…` (P3) | Copy, geometry, cameras, stream and PC behaviour pass (G1-G5); open-loop replays get the conventions right: eps 19 and 0 grasp and lift, eps 28 and 16 close 8-11 cm short because the gate's v_max shortens fast demo segments and an open-loop replay never makes the distance up; run_006 rises and backs away from its first chunk and faults (G7: "not functional", PLAN §11). Re-run G1/G4/G5/G6 after W0. |
| 01 | [Start pose (key `p`)](01_start_pose/README.md) | 2026-11-20 | done, settled (adopted upstream in final-deployment `abba9e8`, to be checked on the arm) | `166a0649…` | 40 of 63 candidate start poses are feasible; `leftx55_t48_h16` (48 deg, 16 cm, fingertips to the robot's left, eraser 0.55 m out) is the sim default and the recommendation for the real arm, chosen offline (kinematic margins + the demos through the gate); the policy screen cannot rank poses because the policy rejects the sim's IMX219 view. A1 replays from it: 3 of 22 run grasp at v_max 0.25, 2 of 7 at sim-only 1.5; every floor fault is a replay start-height artefact; the pose itself caused none. |
| 02 | [Camera study: is it the start pose or the camera?](02_camera_study/README.md) | 2026-11-20 | paused; folded into 03 W2 | `166a0649…` | A1 (partial): 3 of 22 replays that ran grasp at v_max 0.25, 2 of 7 at sim-only 1.5 (the 4 physical pairs: same outcome), all 9 floor faults are demos that started 21.6-27.1 cm high (carried grasp <= -70.8 mm), 11 of 30 runs host-ended (9 health aborts, 2 realtime); B1 reference frames done; A2, B2-B5, C1, C2, D not run: the camera question is not answered yet. |
| 03 | [Policy pipeline: every policy runs in the sim as on the robot](03_policy_pipeline/README.md) | 2026-11-20 | running (W1, W2 started; W0, W3, W4 planned) | `166a0649…` (until W0 rebuilds) | - |
