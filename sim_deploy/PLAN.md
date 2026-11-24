# sim_deploy: the plan

A MuJoCo simulation that stands in for the **Pi** and the **PC**, so the laptop's real `policy_deploy` runs against it
exactly as it runs against the robot. It is used to evaluate our policies (plain diffusion and behavior prompting) in
simulation. Every policy's results go to `sim_evals/` (gitignored). **No benchmark runs until every check in §6 passes.**

Branch `rnoro/main-sim`, worktree `~/Capstone-main-sim`. Written 2026-11-19. `rnoro/final-deployment` is **never
modified**: its code is copied into this branch, and only the copy changes.

## 1. What the policy sees, and how the sim produces it

```
            THE SIM (this branch)                                   LAPTOP SIDE (copy of final-deployment)
 ┌──────────────────────────────────────────────┐
 │ physics: Panda + Franka Hand + soft fingers  │
 │   + rig on franka_mount_with_rig.STEP        │
 │   + table + yellow eraser                    │
 │                                              │
 │ sim Pi ── C0 pi_stream/v1, :5555 ────────────┼──► frames ──C1──► manager ◄──C4──► policy
 │   IMX519 640x480 JPEG (pinhole, nominal K)   │      (unwarp + letterbox    │  ▲      (diffusion or
 │   IMX219 640x480 JPEG (fisheye, measured     │       to 224, as training)  │  │       prompting ckpt)
 │   lens) · Pi clock, mid-exposure stamps      │                             │  │
 │                                              │                             │  │
 │ sim PC ◄── C3 set_waypoints, :5630 ──────────┼─────────────────────────────┘  │
 │   franka_control (copied C++) with a MuJoCo  │                                │
 │   arm/gripper: gate, watchdog, impedance     │                                │
 │   ── C2 state 100 Hz (O_T_TCP, width), :5620 ┼────────────────────────────────┘
 │                                              │
 │ eval driver: resets the scene, presses       │
 │   h / c / y / n, judges, writes sim_evals/   │
 └──────────────────────────────────────────────┘
```

- **Policy inputs:** two IMX519 images (T−50 ms, T) and two IMX219 images (T−100 ms, T), each 224×224, plus the
  pose relative to the anchor (`eef_pos`, `eef_rot6d`, `eef_rot6d_wrt_start`) and `gripper_width_m`. The laptop builds
  all of it from C0 and C2 with the copied code, so the sim only has to send what the Pi and PC send.
- **How it runs:** `policy_deploy run --mock` with `launcher.fake_pc: false` (the PC is the sim, on 127.0.0.1) and
  the new `launcher.fake_pi: false` (frames connects to the sim's Pi at `station.yaml pi.address` instead of replaying a
  recording). The policy runs as `mock_kind: diffusion`. Skipped under `--mock`: only the hardware checks (Wi-Fi,
  laptop address, `settled`).

## 2. Values the sim uses (2026-11-19)

| What | Value | Source |
|---|---|---|
| Rig on the arm | `franka_mount_with_rig.STEP` as modelled: 13 mm block + 3 mm plate, cover 38.1 mm out from the hand face | `~/Downloads/`, 11-17 spec page |
| Hand ↔ STEP | mount frame = hand frame turned 180° about z (z = hand axis, lip at the hand–arm joint); camera on the hand's **+x** face | STEP mount frame; deployment `contracts/frames.yaml` |
| IMX219 on the Franka | lens 137 mm out from the hand axis, 8.4 mm to the side, 27 mm past the joint, looking 15° in toward the hand | STEP: `UC-609` + lens `8.STEP` |
| IMX519 on the Franka | top board, 30° face | STEP: `B0371` / `UC-873` |
| IMU | `ABX00101` (chip `U1_1`) right of the IMX219; facing the fingers: x down, y right, z toward the viewer | the user |
| Other parts shown | Pi 5 (big holes), buttons by the vertical piece, APA102 LED strip toward the fingers | STEP + the user |
| IMX219 lens | measured: fx 449.20, fy 449.54, cx 639.86, cy 459.44 @1280×960; k = 0.0378124, −0.0030246, 0.0040943, −0.0018513; streamed 640×480 (scaled: 224.60, 224.77, 319.68, 229.47) | `~/data/umi/rigs/rig_pi_v3.lens.yaml` |
| IMX519 lens | nominal 1231.25 @1600×1200 (492.5 @640×480); never calibrated | `rig_pi_v3.yaml` |
| Policy's UW view | virtual pinhole 640×480, K (246.65, 246.65, 320.16, 239.54), then letterbox to 224 | rig file `virtual_ultrawide` |
| TCP | `F_T_TCP` = +45° about flange z, 150 mm out (`settled: false`) | copied `contracts/frames.yaml` |
| Training TCP (what the policy assumes) | IMX219 → fingertip: 8.1 right, 54.8 down, 204.9 ahead (mm), Rx(15°) | `rig_pi_v3.yaml` `T_cam_tcp` |
| Fingers | the same soft fin-ray fingers + UMI holders with ArUco tags (`iphumi_umi_soft_v1`), on the printed adapter (§5 E3) | sim kit + adapter |
| Franka finger interface | per finger 2× M4×12 (4 in all), 2.25 Nm, 6.00 mm carrier tongue, 21 mm wide, holes 12 mm apart; travel 25–105 mm; tested for 54 mm fingers | Franka Hand manual R50010 v1.2 §5.2.4 |
| Gripper width | `gripper_width_m` = the fingertip gap (E7, D8 revised: umiproc records tag distance − min_width); recorded 0.0511 (closed on the eraser) to 0.090 (handheld open), holds 0.052-0.056; the real printed fingers open **0-0.040** (f632576 `limits.yaml`; 39-41 mm measured in the 2026-11-20 real-arm images), the sim the same since W0; the laptop maps widths with upstream's `policy.gripper_map` (0.051..0.090 <-> 0.030..0.040, DESIGN W0) | `gripper_range.json` (PICK_TEST1/2), umiproc width stage |
| Object | yellow eraser, 127 × 51 × 12 mm, lying flat, gripped across its 51 mm width (fingertip gap 52-56 mm while holding: width ≈ 0.054; opens to ≈ 0.090), about 30 g. **W0: wider than the real 40 mm opening: not graspable on the robot; the sim's eraser is 30 mm wide across the grasp (= upstream `gripper_map.robot_closed_m`, sim-only stand-in, `scene/README.md` "Grasp feasibility"); W2's look stations keep 51 mm** (W2's image fit: ~102 × 50 mm; to be measured) | the user; IMX519 frames of episode_000152; PICK_TEST1 widths |
| Rig mass on the arm | estimated from the STEP volumes (printed parts) + board datasheets | E2 |

## 3. Folder layout (this branch)

```
deployment/                  copy of rnoro/final-deployment@f632576 deployment/ (W0; each changed file's header says what changed)
  contracts/                 unchanged
  policy_deploy/             changes: launcher.fake_pi (E1), `use replay:...` (E1),
                             policy kind `replay` for check G6 (E1)
  control_deploy/            changes: a `--sim` arm/gripper choice in franka_control (E4)
sim_deploy/
  PLAN.md                    this file
  DESIGN.md                  how the parts fit (written before code, one section per part)
  rig/                       STEP -> meshes + camera/IMU poses + station tooling (E2)
  adapter/                   printable finger adapter: CAD script, STL, drawing (E3)
  sim_pc/                    MuJoCo arm + gripper for franka_control, scene control channel, state export (E4)
  sim_pi/                    camera renders -> pi_stream/v1 sender (E5)
  scene/                     table + eraser, randomisation, resets (E6)
  eval/                      rollout driver, judge, report, sim_evals writer (E6)
  stations/                  station.yaml for the sim run(s)
  run.sh                     one command: start sim PC + sim Pi, then policy_deploy, run N rollouts
controlSW/franka_sim/        existing MuJoCo model code, reused for the Panda, hand, kit and scene
sim_evals/                   GITIGNORED: results (§7)
```

## 4. Decisions (to confirm before building)

| # | Decision | Recommendation |
|---|---|---|
| D1 | The sim PC | **The copied `franka_control` C++ with a MuJoCo arm and gripper** behind its existing `Arm`/`Gripper` interfaces (6 methods; MuJoCo's C library ships with the `mujoco` wheel). The real gate, supervisor, watchdog, impedance controller and C2/C3 codec run, so the sim checks the PC code too. Fallback: the Python `fake_pc` driving `franka_sim`'s MuJoCo backend. |
| D2 | A `replay` policy kind in the copy | Yes, sim only: it returns a recorded training episode's actions through the normal C4 path. It checks frames, TCP axes and the relative-pose conventions (check G6) before any learned policy runs. |
| D3 | Mount | The STEP as modelled (13 mm block), not the 4.9 mm fix. |
| D4 | Adapter hole height (Fig. 5.5: 8 / 11.40 / 15 mm) | Slotted holes, ±2 mm, until a caliper check. |
| D5 | Envs | New `~/envs/sim_policy_deploy` and `~/envs/sim_policy_deploy_torch`, installed editable from `~/Capstone-main-sim` (the existing envs point at final-deployment). Several GB; about 40 GB free. |
| D6 | The old `controlSW/franka_deploy` launcher | Left as it is, not used. |

## 5. Work packages and agents

Method: one executor per package, one reviewer over all of them and the integration, a local commit per wave after
a file-by-file walkthrough with the user, no push. Each package starts by writing its section of `DESIGN.md`. No
package changes anything outside its folder except the listed files in `deployment/`.

**Wave 0 (me, with the user):** this plan, then `DESIGN.md` (interfaces between the packages: the scene control
channel, the state export, the station file). Settle D1–D6.

**Wave 1 (three agents in parallel):**
- **E1 deploy copy.** Copy `deployment/` from `rnoro/final-deployment@ea9f7fb` (re-based from dc1ee36; ea9f7fb already carries the measured lens), header per changed file. Add
  `launcher.fake_pi` (default true = today's behaviour) and the `replay`
  kind. Build the two envs. Deliver a diff report: copy vs source, only the listed changes.
- **E2 rig from CAD.** From the STEP: simplified meshes per part, both camera poses and the IMU pose in the hand frame,
  the optical centres from the lens geometry, the rig mass and centre of mass. Write the sim station tooling (replacing
  the iPhone placeholders: `C_T_TCP`, `C_main_T_C_uw`, housing, collision boxes, payload). Check against the 11-17
  numbers (38.1 mm out, 156.7 mm to the fingertip point).
- **E3 finger adapter.** A printable per-finger adapter: clamps the 6 mm carrier tongue, 2× M4×12, holds the soft finger
  in its UMI holder (tags in the handheld position). Parametric CAD script, STL + STEP, a dimensioned drawing. The same
  part goes into the sim kit.

**Wave 2 (three agents in parallel, after wave 1):**
- **E4 sim PC.** `SimArm` / `SimGripper` in the copied `franka_control`: MuJoCo stepped 1 ms per control cycle in
  wall-clock time, torques from the real impedance controller, model terms (Jacobian, mass, Coriolis) from MuJoCo,
  gravity compensated as on the robot. Also a scene control channel (reset, set object pose, ground truth) and a
  state export (all qpos at ≥ 100 Hz) for the renderer. A `--sim` build and flag.
- **E5 sim Pi.** A process that mirrors the physics state and renders both cameras at 30 fps. The IMX519 is a pinhole
  render. The IMX219 is a wide render (or cube map) warped into the fisheye with the measured lens, keeping the
  uncovered black corners. Output is JPEG `pi_stream/v1`: 3-part messages, Pi clock with offset and drift,
  `t_mid_ns` = mid-exposure, `uw_reused` when the IMX219 frame repeats.
- **E6 scene + eval.** The table, the eraser (size, colour, mass, friction) and a seeded randomisation of its pose.
  The rollout driver runs `policy_deploy` in a pseudo-terminal and presses `h`, `c`, `y`/`n`. The judge (grasped,
  lifted, placed; safety from C2 modes and faults), the report, and the `sim_evals/` writer (§7).

**Wave 3 (integration, one agent + the reviewer):** `sim_deploy/run.sh`, then the checks in §6, in order. A failing
check goes back to its package; nothing moves on until it passes.

**Wave 4 (benchmarks, only after §6):** for each checkpoint (PICK_TEST1 scaling runs, PICK_DOUBLE runs, plain vs
prompting), N seeded rollouts with the same seeds per policy. LIBERO gets a separate plan later.

## 6. Checks the sim must pass before any benchmark

| | Check | Pass when | E7 result (2026-11-20) |
|---|---|---|---|
| G1 | Copy is faithful | the diff report shows only the listed changes; `policy_deploy check --mock` passes against the sim | **pass**: `diff -r` vs `git archive ea9f7fb deployment` = the 18 listed files (E1 12, E4 5, E6 frames.yaml), each with its header; E7 changed only the E1-new `replay_policy.py` + STRUCTURE.md §5.5. `policy_deploy check --mock` in a pty against the running sim: every row PASS (Pi reachable, terminal) |
| G2 | Geometry | rig poses match the STEP; 11-17 numbers reproduced (±1 mm); the rendered IMX219/IMX519 views show the rig parts where the STEP puts them | **pass**: E2/E3 geometry unchanged by E7 (cameras, TCP 235.65 mm, masses equal to the wave-2 model to 0.0); kit tags now 44.29 mm apart at gap 0 (D8); one sha (`7746a439…` now) checked by sim PC, sim Pi and driver |
| G3 | Cameras vs reality | sim rendered in the **handheld** pose (`rig_pi_v3`) with the real fingers: finger and tag pixel positions after policy_deploy's unwarp match a real training frame (e.g. PICK_TEST1, `episode_000152`) within a few px; straight edges stay straight after the unwarp | **pass with an explained residual**: D8 revised (gap = width, real stickers): 1.0 / 3.3 px (mean / max, 224 px) with the camera moved (−4.0, +2.5, +8.8) mm; with training's `T_cam_tcp` 5.0 / 12.0 px, tags 13-17 % too small = the camera ~9 mm closer to the fingers than rig_pi_v3 says (wave 2's D8: 25-35 px) |
| G4 | Stream and clocks | 30 packets/s, no stale or down camera, the clock map converges, the pair rule passes, `policy_deploy.frames.probe --save` images look like training's | **pass**: 30.0 pkt/s, 0 gaps / bad / stale, pair gap −16.3 ms, clock synced at once (drift 20.1 ppm fitted vs 20 set), refused 0/0/0, both cameras ok 30 fps; the four policy images saved at both homes (E7 scratchpad `g4/policy_images_both.png`) |
| G5 | PC behaviour | `hold_still` rollout = no motion; scripted waypoints tracked within a set error; out-of-box waypoint → `fault`; no C3 for 1 s → `stopped`; homing works | **pass**: hold_still 0.04 mm / 0.09° (deploy), 0.07 mm / 0.12° (trainstart) over 20 s; homing to both homes; out-of-box waypoint → `fault` (latched, as on the robot: run.sh restarts the sim and resumes); manager frozen 1.5 s → `holding` then `stopped` ~1 s after the last C3, next rollout homes and runs without a restart |
| G6 | Conventions end to end | the `replay` policy replays a training episode's actions with the eraser placed as in that episode: the TCP follows the recorded path (axes and relative poses right) and the grasp closes on the eraser | **fail**: replay ep 2 (deploy home): the gate's `v_max` 0.25 m/s shortened 178 waypoints, the TCP stopped 11 cm above the grasp point and closed in the air (C2 0.056 = commanded), then left the old box (x 0.238). Replay cursor bug fixed first (it ran 12× fast). Grasp itself checked offline: 5 N/pad at 0.054, no slip |
| G7 | Policies run | each checkpoint (plain and prompting) completes 3 rollouts with no fault, inference within the `exec_h` budget, tap images saved | **fail (smoke)**: run_006 (prompting, bf16 / 8 steps / exec_h 12), deploy home, 2 rollouts: both faulted after 6-7 s (TCP up to z 0.71 > box 0.70); policy 177-182 ms p50 (p95 198), margin min 11 ms, 0 skipped, VRAM 3.3 GB of 4 GB with the renderer. `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0149_smoke` |

**P3: G5-G7 from the start pose (key `p`), 2026-11-20, table at O z 0** (the user: the base sits on the table; DESIGN.md
"P3/P4"). Stand-in start pose re-solved for the new table: q `[-0.6419329, 0.3706554, -0.1347388, -1.8275169,
0.5918946, 1.4884701, -1.6761742]` (TCP O (0.439, -0.164, 0.162), 16.2 cm above the table, 49.5 deg, pointing to the
robot's left). All through `run.sh` (real v_max 0.25 unless stated). Runs with the old table (z 0.05) are marked
`SUPERSEDED.txt`.

| | P3 result |
|---|---|
| G5 | **pass**: `hold_still` 1 rollout: TCP drift 0.07 mm / 0.12 deg over 20 s, hand 80.0 mm, start pose reached within 0.0001 rad (tol 0.05), 0 shortened / rejected (`_baseline/hold_still/2026-11-20_0943_p3_g5_hold_z0`). Refusals via the driver: `c` away from the start pose (driver `--home deploy` against the start-pose sim PC) -> `start_failed`, "refused: not at the start pose: press h, then p on the PC"; `p` away from home (the driver's scene link, after `p`) -> "key p refused: not at home: press h first" |
| G6 | **pass for the conventions, open loop fails on fast demo segments**: replays (catalog `grasp_ok` from the start pose) ep 19: success, path error median 8.8 mm (time aligned), 14.5 mm at the recorded grasp, lift 17.3 cm; ep 0: success, 16.8 mm / 26.6 mm, lift 19.9 cm; ep 28: closes 82 mm short of the grasp (169 waypoints shortened), then box fault x 0.189 < 0.20; ep 16: 106 mm short (168 shortened), timeout. Sim-only `v_max 0.5` (`stations/vmax05_g6.yaml`): 19 success (11.4 mm at the grasp), 28 and 16 unchanged (80 / 107 mm short; 37 / 45 shortened). Cause: 29 of 30 demos have 50 ms segments faster than 0.5 m/s before the grasp (peaks 0.6-1.5 m/s); ep 0 is the one without. A shortened distance is never made up open loop (chunks are relative to the manager's anchor) |
| G7 | **fail (policy vs sim images)**: run_006 (bf16 / 8 steps / exec_h 12), 3 rollouts: 3 faults, box y -0.411 at 18.4 s, cone 62.0 deg at 16.9 s, cone 63.0 deg at 11.4 s; no grasp. The policy rises and backs off from the first chunk. Policy 270 / 270 / 188 ms p50 (p95 <= 278), margin min -140 ms, 0 skipped (another session's test suites loaded the laptop). Policy images at the start pose vs episode_000152's first frames: `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/g7_policy_images_vs_ep152.png`: geometry matches; table texture, shadows, colours and the IMX219's background do not. Offline, same low-dim input: first chunk +5.6 cm toward the fingertips on the training images (demos +5.7 cm in 0.75 s), -1.3 cm on the sim's; sim IMX519 + training IMX219 +5.4 cm, training IMX519 + sim IMX219 -0.5 cm: **the sim's IMX219 render is what the policy rejects** (reproduce: `~/envs/sim_policy_deploy_torch/bin/python sim_evals/_start_pose_sweep/tools/offline_first_chunk.py <G7 eval>`, cwd `deployment/policy_deploy`, `SPLIT=1` for the camera swap; logs and images in `sim_evals/_start_pose_sweep/p3/`) |

## 7. `sim_evals/` (gitignored)

```
sim_evals/
  index.csv                                  one row per eval: policy, kind, dataset, eps, seeds, success, faults, date
  <task>/<run_folder>/<YYYY-MM-DD_HHMM>_<tag>/
    eval.yaml                                frozen: station, seeds, scene settings, both git commits, checkpoint path
    rollouts.jsonl                           per rollout: seed, outcome, time to grasp, faults, safety flags
    metrics.json                             success rate (with CI), grasp/lift/place rates, latency, skipped steps
    summary.html                             the report
    rollouts/<n>/                            policy_deploy session folder, tap images, an external-camera video
```

`run_folder` is the training run (e.g. `PICK_TEST1/run_006_...`), so every policy's sim results sit beside its name.

## 8. Not in scope here

- LIBERO benchmarks (a separate plan once the sim checks pass).
- Changing `rnoro/final-deployment`. The lens fix there is the user's call; see §2 for the values.
- Tuning the real robot's gains, workspace box or payload.

## 9. Start pose (key `p`), 2026-11-20

The real deployment (rnoro/final-deployment, uncommitted at the time: snapshot `ports/start_pose_snapshot_20261120_0907.patch`)
adds a PC keyboard key `p`: from home only, a joint move to `homing.q_start_rad` that opens the hand to 0.08 (0.04 since b4030f6); once
`q_start_rad` is set, C3 `start` is refused unless every joint is within `homing.at_pose_tol_rad` (0.05) of it.
Update: upstream `abba9e8` set `q_start_rad` to this sim's recommended pose (see §11).

| Step | Who | What |
|---|---|---|
| P1 | agent | port the snapshot into `deployment/` (merge with the sim's listed changes); sim-only scene command `{"cmd": "key", "key": "p"}` that pushes `p` into franka_control's own keyboard queue; rebuild |
| P2 | agent | eval flow `h` → `p` → eraser placed relative to the start pose (demos: ~16 cm ahead, ~17 cm below the fingertips) → `c`; UI home = "start pose (p)" by default; `q_start_rad` in one file `stations/start_pose.yaml`; `python -m sim_deploy.scene start-pose` writes it from 7 numbers, a `pc_probe --start-pose` printout or final-deployment's `station.yaml` (`git show`), and checks box/cone |
| P3 | agent | checks with the stand-in (G5 refusals, G6 replay, G7 run_006) |
| P4 | agent | sweep of candidate start poses over the demos' range (tilt 38–64°, fingertips 6–27 cm above the table, heading) × every runnable checkpoint, few short rollouts each, real v_max 0.25 → the envelope of `q_start_rad` that doesn't fail early, the best stand-in, `sim_evals/_start_pose_sweep/` |
| — | reviewer | review all, then docs + walkthrough before any commit |

Sim default until the measured value exists (P4, 2026-11-20, table at O z 0): `[-0.5928536, 0.5219462, -0.0897168,
-1.5930073, 0.6595725, 1.5068439, -1.5301472]` (TCP O (0.531, -0.159, 0.160), 16.0 cm above the table, 48.0°,
pointing to the robot's left; DESIGN.md "P3/P4", also the recommendation for the real arm). Earlier stand-ins: P3
`[-0.6419329, 0.3706554, -0.1347388, -1.8275169, 0.5918946, 1.4884701, -1.6761742]` (E7 trainstart re-solved for the
table at z 0) and E7's `[-0.6225485, 0.3138391, -0.1643961, -1.7825973, 0.5934631, 1.3978677, -1.6275796]` (TCP O
(0.439, -0.164, 0.212), 16.2 cm above the old sim table at z 0.05, 49.5°). When 7 numbers
appear under `franka_control: homing: q_start_rad:` in final-deployment's `deployment/control_deploy/station.yaml`,
`scene start-pose --from-git` takes them over.

## 10. Is it the start pose or the camera? (2026-11-20)

Question: from the recommended start pose (`stations/start_pose.yaml`, leftx55_t48_h16), do the demos' motions and
the trained policies run? If the motions run and the policies don't, the problem is the camera view (the Franka mount
sees the fingers/scene ~3–4 % off the handheld view). No hardware changes; rnoro/final-deployment untouched; every
correction is a sim-only experiment until the end. **Recommendation only after the complete sim eval (A–C).**

| Phase | Agent | What | Runs in parallel with |
|---|---|---|---|
| A1 | chain | **pose and physics**: replay PICK_TEST1 demos from the start pose (eraser placed per demo), at the real v_max 0.25 and once with a sim-only v_max 1.5 (open-loop replays can't re-aim) → grasp/lift rate, path error, faults | B |
| A2 | chain | **policies as they are** (no correction): every runnable checkpoint × 3 rollouts from the start pose, real limits → baseline success/grasp/approach/faults, first-chunk direction | B |
| B | offline | **camera geometry and appearance**: renders of the same scene from the handheld pose vs the mount, each with the current look and with a realistic look (real table texture and eraser colour sampled from training frames, diffuse light, the PICK_TEST1 workspace mesh from `umiproc viz` as the background if usable) as separate model builds; offsets vs real training first frames; fit the IMX219 virtual-camera correction (tilt + zoom in the rig file) and the IMX519 crop/zoom (sim-only option in the copy's frame prep, default off) | A |
| C1 | chain | **attribution** (model forward passes): {handheld, mount, corrected mount} × {current, realistic look} vs the real frame → is it geometry (real-world problem) or appearance (sim-only problem) that makes the policy back away | — (after A, B) |
| C2 | chain | **policies with the fixes C1 points to** (realistic look; + IMX219 correction; + IMX519 correction): every runnable checkpoint × 3 rollouts, same seeds as A2 | — |
| D | reviewer | check A–C, one table in `sim_evals/_camera_study/`, then the recommendation | — |

P3/P4 (2026-11-20) already found: with training's IMX219 swapped in, run_006 approaches (+5.4 cm); with the sim's it backs away (−1.3 cm); its guess is appearance, not geometry (untested). One chain at a time (ports, 4 GB GPU): A1 → A2 → C1 → C2 sequentially; B is CPU-only rendering.

## 11. Goal now: every policy (behavior prompting or plain diffusion) runs in the sim as it would on the robot, easily (2026-11-20)

The start pose is settled (`abba9e8` on rnoro/final-deployment took the sim's q_start_rad). The camera study (§10) is
paused and folds into W2 below. Runnable checkpoints today (`python -m sim_deploy.ui.catalog`): 14 — prompting
PICK_TEST1 run_006/012/014/016/020; plain PICK_TEST1 run_022/024/028/030/031; plain PICK_DOUBLE run_002/003/004/005.

### Progress so far

| Part | State |
|---|---|
| Sim chain (sim PC = copied franka_control + MuJoCo, sim Pi, copied policy_deploy unchanged except listed sim-only changes) | built; G1–G5 pass (E7, P3) |
| Rig/mount/cameras from the STEP, printed finger adapter, fin-ray compliance, eraser, table at z 0 (base on the table) | built (E2, E3, E7, P3/P4) |
| Start pose key `p` (ported from the real deployment), start-pose tool, eval flow h → p → c | built (P1, P2) |
| UI (pick a policy, ▶ Play, live views, results) | built (U1–U5); not yet used for a real run |
| Start pose | settled: `[-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]` |
| Policies in the sim | **not functional**: run_006 backs away from every start pose; the camera swap points at the sim's IMX219 image (P3/P4) |
| Camera study (§10) | paused, folded into W2: A1 replays partly run (3 of 22 that ran grasped at v 0.25, 2 of 7 at 1.5; floor faults are replay start-height artefacts; 2 realtime faults from parallel chains), B1 reference frames done, a partial IMX519 fit on disk; A2, C, D not done |
| Docs `sim-experiments/` | 00, 01, 02 done (with docs-vs-raw correction lists); 03 set up |
| Copy of deployment/ | at ea9f7fb + sim changes + the 927a5ba snapshot; **behind upstream f632576**: gripper max 0.04 m + `policy/gripper_map.py`, settled TCP, real addresses, q_start_rad in station.yaml, manager/pose_converter/diffusion_policy changes |

Known issues: the sim's IMX219 look; v_max 0.25 cuts fast demo motions (closed loop should cope); replays need their
own start height; parallel sim chains overloaded the CPU once (realtime fault); other sessions' pytest suites
briefly bind 5555/5620.

### Workstreams (each a coordinator agent that splits its steps across its own subagents; a reviewer gates each)

| | Workstream | Steps (one subagent each) | Done when |
|---|---|---|---|
| W0 | **Re-sync with the real deployment** (first; everything depends on it) | port f632576 into the copy + re-apply sim-only changes; the sim gripper with the printed fingers' 0.04 m max and `gripper_map` semantics (and what it means for the 51 mm eraser grasp); start pose from upstream `station.yaml` (`scene start-pose --from-git`); rebuild + G1/G5 smoke | copy = f632576 + listed sim changes; chain runs hold_still and a replay |
| W1 | **Deploy path = training, offline, on real images** (no rendering) | harness feeding real training episodes through the copied laptop pipeline (ImagePrep, pose converter, gripper map, C4) for both kinds; prompting specifics (prompt dataset, task_name, prompt_index); per-checkpoint open-loop error vs the recorded actions (and vs the training code's own offline eval); timing bench per checkpoint → settings (precision, diffusion steps, exec_h) | every checkpoint loads and predicts like training on real images; settings file per checkpoint |
| W2 | **Sim images close enough to real** (continues §10 B/C1) | realistic look (table, eraser colour, light, workspace mesh), camera colour/noise/JPEG; attribution with W1's harness: sim vs real first-chunk response per checkpoint; IMX219/IMX519 view corrections stay an opt-in experiment | sim response within tolerance of the real-image response for every checkpoint |
| W3 | **Closed-loop functional evals** (after W0–W2) | every checkpoint × N rollouts from the start pose, real limits, one chain at a time (GPU); BP vs plain table | each checkpoint approaches/grasps as its offline behaviour predicts; failures explained |
| W4 | **Easy to run any policy** | per-checkpoint settings applied automatically (catalog + driver + UI); `eval batch` for a list or "all"; UI end-to-end on a real run; DIRECTIONS | one command / one click runs any checkpoint correctly |
| W5 | **Docs** | `sim-experiments/03_policy_pipeline/` (W0–W4), finish 00/01/02, AGENTS.md | every result reproducible from the docs |

Order: W0 → (W1 ∥ W2) → W3 ∥ W4 → W5 throughout. GPU users take turns (one policy process at a time on the 4 GB
GPU); the sim chain runs one at a time unless a coordinator proves real-time holds with more (network namespaces).
