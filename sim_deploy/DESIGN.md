# sim_deploy: design

How the parts of [PLAN.md](PLAN.md) fit together. This file fixes the **interfaces between the packages** (E1–E6) so
they can be built in parallel. Each package adds its own section at the end ("E<n>: as built") when it is done.

Decisions D1–D6 of PLAN.md §4 are taken as recommended (2026-11-19). Two more:
- **D7 TCP:** the copy's `contracts/frames.yaml` keeps deployment's rotation (+45° about flange z) but its z offset
  becomes the real fingertip point with the printed adapter (E3 computes it, E2 checks it in the model). The policy's
  poses are about the training TCP (the fingertip point), so a wrong offset would turn rotations into translations.
  The value goes in the report for the user to carry to deployment.
- **D8 gripper width (revised by E7, 2026-11-20):** the policy's width is umiproc's recorded width = the ArUco tag
  x-distance minus the calibration's `min_width` (`umiproc/dp/stages/width.py calibration_interpolator`: "closed
  reads 0"), i.e. **the fingertip gap itself** (slope 1, offset 0 +- ~1 mm). The sim PC reports and accepts widths as
  the gap: `gripper.width_offset_m: 0.0`. (Wave 2 read 0.0443 as the closed width in training units and used
  `width = gap + 0.0443`: wrong by 44 mm.) Evidence and the CAD discrepancy: "E7: integration".

## 1. Processes and ports (all on 127.0.0.1, one machine)

| Process | Started by | Binds | Connects |
|---|---|---|---|
| sim PC: `franka_control --sim` (C++, E4) | `run.sh` | C2 PUB :5620, C3 REP :5630, state PUB :5650, scene REP :5651 | — |
| sim Pi: `python -m sim_deploy.sim_pi` (E5) | `run.sh` | C0 PUB :5555 | state SUB :5650 |
| `policy_deploy run --mock` (copy, E1) | the eval driver | C1 :5610, C4 :5640 (as today) | C0 :5555, C2 :5620, C3 :5630 |
| eval driver: `python -m sim_deploy.eval` (E6) | the user | — | scene REP :5651, C2 SUB :5620 (read only), the policy_deploy terminal (pty) |

`run.sh` starts the sim PC, then the sim Pi, then the eval driver, which starts and stops `policy_deploy` itself.
The sim PC is the only C3 server and `policy_deploy`'s manager is its only C3 client (as on hardware).

## 2. One model file

`python -m sim_deploy.model build --station sim_deploy/stations/sim_rig.yaml` (E6 owns the builder; it calls E2's
`add_rig` and E3's adapter/finger placement, and reuses `controlSW/franka_sim` for the Panda, Franka Hand and finger kit)
writes `sim_deploy/build/<station id>/scene.xml` + `meshes/` + `model.json`. **Both the sim PC (MuJoCo C API,
`mj_loadXML`) and the sim Pi (Python `mujoco`) load this same file**, so qpos means the same in both. `model.json` holds
its sha256, nq/nv, and the names below; every process checks the sha at start (scene `hello`).

Names fixed in the model:

| Name | Kind | Meaning |
|---|---|---|
| `joint1` … `joint7` | hinge | Panda arm, in order |
| `finger_joint1`, `finger_joint2` | slide | Franka Hand fingers (each 0–0.020 m since W0: the printed fingers open 0–0.040 m in all, `sim_rig.yaml finger_max_gap_m`; menagerie 0–0.040 each) |
| `flange` | site | link 7 flange frame = libfranka's F frame |
| `tcp` | site | `flange` × `F_T_TCP` of the copy's `contracts/frames.yaml` |
| `cam_imx519`, `cam_imx219` | camera | lens optical centres from the STEP (E2). MuJoCo cameras look along −z with +y up; OpenCV pose `R_cv` → MuJoCo `R_cv · diag(1, −1, −1)` |
| `eraser` | body + free joint | the object; its free joint is the last 7 qpos entries |
| `table` | geom | the table top |
| groups | geom group | 0 scene, 1 arm + hand, 2 rig parts, 3 fingers + adapter + tags, 5 collision-only (hidden) |

Base frame = libfranka's O frame (robot base). Table and eraser poses are in O.

## 3. Time

- One clock: `CLOCK_MONOTONIC` of this machine (the "PC clock" and the "laptop clock" are the same here; the Pi clock
  is simulated, §5).
- The sim PC steps physics 1 ms per step, paced to `CLOCK_MONOTONIC`. It never runs ahead of wall time. If it falls
  more than 5 ms behind it logs `slow` and counts it; more than 50 ms behind during torque control → treat as a
  libfranka realtime error (`ArmError kRealtime`), as the real robot would.
- Physics runs **all the time**, also outside `runTorqueControl` (between rollouts, during homing, while idle). With no
  torque session the arm holds its joints like the real robot's brakes/controller (a stiff joint hold), so the eraser
  can still fall, slide and be reset.

## 4. State export (sim PC → sim Pi), ZMQ PUB :5650

One single-part JSON message every 5 ms of physics time:
`{"schema": "sim.state/v1", "t_ns": <int, CLOCK_MONOTONIC of that physics instant>, "step": <int>, "model_sha256": "...",
"qpos": [<nq floats>]}`. Subscribers keep the newest one. The renderer may interpolate two messages to a render time;
it never extrapolates more than 10 ms.

## 5. Scene control (eval driver ↔ sim PC), ZMQ REP :5651

JSON request → JSON reply, every reply has `"ok": true|false` and `"error"` when false.

| Request | Reply | Notes |
|---|---|---|
| `{"cmd": "hello"}` | `model_sha256`, `nq`, `names`, `t_ns` | |
| `{"cmd": "set_object", "name": "eraser", "pose": [x, y, z, qw, qx, qy, qz]}` | `t_ns` | applied between two physics steps; velocity zeroed |
| `{"cmd": "ground_truth"}` | `t_ns`, `objects: {eraser: {pose, lin_vel}}`, `tcp: pose`, `flange: pose`, `finger_gap_m`, `contacts: [{a, b, force_n}]` | contacts between named geoms, finger–eraser and anything–table except the eraser |
| `{"cmd": "set_arm", "q": [7 floats]}` | `t_ns` | refused (`ok: false`) while a torque session or homing runs; for fast resets only |
| `{"cmd": "key", "key": "h"\|"p"\|"g"\|"s"}` (**sim only**, P1) | `key`, `log` (the supervisor's line; `null` if none within 1 s); refused: `ok: false`, `error` = `key p refused: ...` | pushed into franka_control's own `KeyQueue` = exactly a key typed on the PC (also without `--keyboard`); the move's outcome shows in C2 / the log |

The scene control channel is never used by `policy_deploy`. Only `key` changes the arm's mode, and only as the PC's
own keyboard would (P1).

## 6. Sim Pi (E5)

- A live version of `policy_deploy/frames/fake_pi_sender.py` (copy): same `pi_stream/v1` 3-part messages, same Pi
  clock model (`t_pi = t_mono + offset` with drift, e.g. +1000 s and 20 ppm), packets at 30 Hz, `t_mid_ns` = the
  render instant (mid-exposure), `t_sensor_ns` = `t_mid_ns + exposure/2` (end of exposure, as the real Pi), fixed
  exposure e.g. 5 ms, `uw_reused` false unless a render is skipped.
- IMX519 (`main`): pinhole render at 640×480 with the nominal K (492.5, 492.5, 319.5, 239.5), its tiny radtan
  distortion applied. IMX219 (`ultrawide`): rendered as a **fisheye** at 640×480 with the measured lens scaled to 640
  (224.6009, 224.7683, 319.6793, 229.4701; k 0.0378124, −0.0030246, 0.0040943, −0.0018513), from a cube map (or
  several pinholes) around the camera, black outside the lens circle and the rendered field. JPEG quality 90.
- What the laptop does with these (unwarp, letterbox, pairing) stays in the copied policy_deploy; the sim Pi must not
  pre-process anything the real Pi does not.

## 7. Policy side (E1)

- `sim_deploy/stations/policy_sim.yaml` = deployment's `station.yaml` with: python paths to
  `~/envs/sim_policy_deploy{,_torch}`, `pi.address: 127.0.0.1`, `launcher: {fake_pc: false, fake_pi: false}`,
  `session_root: ~/sim_deploy_sessions`, `policy.kind: hold_still` and `policy.mock_kind` set per eval.
- `launcher.fake_pi` (new, default true = today's behaviour): false → under `--mock` frames does not start
  `FakePiSender` and connects to `pi.address:pi.port`.
- policy kind `replay` (new): `policy.replay_episode` = a packaged episode (zarr path + episode index). Each `infer`
  returns the next 16 relative steps of that episode's recorded TCP poses and widths, about the current anchor,
  advancing by `exec_h` per call. No torch. Used for check G6.
- The measured UW lens: already in upstream ea9f7fb (`contracts/rigs/rig_pi_v3_640.yaml`, generated from training's rig_pi_v3 + lens file), so no change here.

## 8. Sim PC (E4)

- `Arm` and `Gripper` implementations backed by one MuJoCo `mjModel/mjData` owned by a physics thread.
- `runTorqueControl(step)`: each physics step calls `step(reading)`; `reading` from MuJoCo: `q`, `dq`, `O_T_F` (site
  `flange`), `J_F` (6×7 at `flange`, [linear; angular] in O), `coriolis` (MuJoCo `qfrc_bias` minus gravity), `tau_J_d`.
  The returned `tau` is applied **on top of gravity compensation** (libfranka's convention: the user torque excludes
  gravity). Torque and rate limits as libfranka's; violations → `ArmError kCommand`/`kReflex` like the real arm.
  Contact force on the arm (not the fingers) above `collision_torque_thresholds_nm` → `kReflex`.
- `moveToJoints(HomingMotion)`: follow the motion's joint profile with a stiff joint PD in MuJoCo.
- `Gripper`: finger position actuators; `moveTo` closes until the target or until stalled on an object (as libfranka
  `move`/`grasp`); `widthM()` = fingertip gap.
- Build: a `build-sim/` CMake build of the copied `franka_control` without libfranka, linking MuJoCo 3.3.0 from the
  `mujoco` wheel in `~/envs/sim_deploy` (`include/`, `libmujoco.so.3.3.0`). Flag `--sim` (with `--local`) picks
  SimArm/SimGripper; the model path and ports come from a `sim:` block in the sim station file.

## 9. Eval driver and results (E6)

- `python -m sim_deploy.eval run --policy <RUN_FOLDER | hold_still | replay:<zarr>#<episode>> --rollouts N --seed S
  [--tag T]`.
- Per eval: write `policy.mock_kind` (and checkpoint settings, via the copy's `policy_deploy use`) into a frozen copy
  of `policy_sim.yaml`; start `policy_deploy run --mock --station <frozen>` in a pty; press `h`, wait for `idle`.
- Per rollout: `set_arm` to home if needed, `set_object` eraser at the seeded pose, press `c`, watch ground truth and
  C2 until success (eraser ≥ 5 cm above its start for 1 s), timeout (default 60 s), or a fault/stop; press `y`/`n`.
- After the last rollout press `q`, collect the session folder, write `sim_evals/` (PLAN §7) and append `index.csv`.
- Environments: `~/envs/sim_deploy` (python 3.10, mujoco 3.3.0, pyzmq, opencv-python-headless, numpy, pyyaml, pexpect,
  plus editable `controlSW/franka_sim`, `deployment/contracts` and `sim_deploy` of this repo).

## 10. Rules for every package

- Change nothing outside your folder, except the files PLAN §3 lists for `deployment/`. Never write to
  `~/Capstone-final-deployment` (read it, copy from it). Never commit or push: the user reviews first.
- Copied files keep a header: `# copied from rnoro/final-deployment:<path> @ f632576` (ea9f7fb before W0) + `# changes: ...`.
- Don't run the existing test suites unless told; do run what you build (a smoke run proving it works).
- Ports 5555, 5610, 5620, 5630, 5640, 5650, 5651: one user of each at a time. Check nothing else is listening first.

## E1: as built (2026-11-19)

**Copy.** `deployment/` = `git archive ea9f7fb deployment` (rnoro/final-deployment, 2026-11-19 21:46; tracked files
only: no build dirs, egg-info or caches). It replaced a first copy at dc1ee36. `deployment/` needs nothing else from
that branch: `_dataproc` is vendored in `policy_deploy/frames/_dataproc/`, the C++ reads `../../contracts`, and the
only outside package is `behavior_prompting` (torch env).

**Camera constants: upstream's, unchanged.** ea9f7fb's `frames.rig_file` is `../contracts/rigs/rig_pi_v3_640.yaml`
(generated by `frames/deploy_rig.py` from training's `rig_pi_v3.yaml` + `rig_pi_v3.lens.yaml`): ultrawide 640×480
`[224.60087, 224.76831, 319.67932, 229.47013]`, k `[0.0378124, -0.0030246, 0.0040943, -0.0018513]` (provenance
`calibrated`), `virtual_ultrawide` `[246.64919, 246.64919, 320.16495, 239.53679]`, identity rotation: the measured
lens and training's virtual camera. The rig-file change planned for E1 is dropped (§7's last bullet is superseded).

**Changes vs ea9f7fb** (each file has `# copied from rnoro/final-deployment:<path> @ ea9f7fb` + `# changes:`;
`diff -r` against the archive shows only these 12 files; under `deployment/policy_deploy/`):

| Change | Files |
|---|---|
| `launcher.fake_pi` (default true = as before). false under `--mock`: frames starts no `FakePiSender` and connects to `pi.address:pi.port` (then required; `mock_episode` not); `check` shows the real-run "Pi reachable" row instead of "mock episode"/"fake Pi port" | `config/launcher.yaml`, `policy_deploy/launcher/launcher.py` (`LauncherSettings.fake_pi`, checked like `fake_pc`, passed to `FramesProcess`), `policy_deploy/launcher/processes.py` (`FramesProcess(fake_pi=)`), `policy_deploy/frames/settings.py` (`Settings.fake_pi` from the station's launcher section, so `python -m policy_deploy.frames --mock` alone behaves the same), `policy_deploy/frames/app.py` |
| Policy kind `replay` (no torch, no zarr package): `policy.replay_episode: {dataset, index}`, reads `manager.exec_h`; math in the class docstring and STRUCTURE.md §5.5 | `policy_deploy/policy/replay_policy.py` (new), `policy_deploy/policy/policy_factory.py`, `policy_deploy/policy/settings.py`, `policy_deploy/policy/self_check.py` ("replay episode" row), `config/policy.yaml`, `STRUCTURE.md` (§5.2, §5.3 rows, §5.5) |
| `policy_deploy use replay:<dataset>#<index>` (kind/mock_kind replay + `replay_episode`; index checked against `meta/episode_ends`); `replay_episode` is a policy-choice key, so every other target removes it. `cli.py` unchanged (it passes the target through; its help still says `RUN_FOLDER\|hold_still`) | `policy_deploy/launcher/use_policy.py` |
| the "policy.kind is not set" hint names replay | `policy_deploy/launcher/processes.py` |

Replay math: rows `[s, e)` of the episode at 60 Hz, `d = 3` rows per 50 ms step; cursor `c = s` on reset; each infer
returns `T_k = inv(P_c) · P_min(c+3k, e−1)` (position + first two rows of R) and `w_min(c+3k, e−1)`, k = 0..15, then
`c += 3·exec_h` (clamped; **superseded by E7 fix 5**: the cursor follows the time since the first infer; W0: widths go through `policy.gripper_map.to_robot`): the training sample's own relative action at row c, so the arm repeats the recorded motion
from wherever it is and holds at the end.

**Station.** `sim_deploy/stations/policy_sim.yaml` (§7), refreshed against ea9f7fb's `station.yaml` (`frames: {}`: the
rig file and `mock_episode` = episode_000152 are now the config defaults). Per eval the driver copies it and runs
`policy_deploy use RUN_FOLDER | hold_still | replay:<dataset>#<index> --station <copy>`. RUN_FOLDER sets
`kind`/`mock_kind: diffusion`, `checkpoint`, and for a prompting run `prompt_dataset` + `manager.task_name`/
`prompt_index`. `use` writes `.<name>.use.tmp` beside the copy and checks it with `Station.load` only (since ea9f7fb the
policy values are checked by `policy_deploy check`). Paths in the copy must be absolute or `~`.

**Envs** (`sim_deploy/setup_envs.sh [light] [torch] [sim]`, idempotent, pinned): `~/envs/sim_policy_deploy` and
`~/envs/sim_policy_deploy_torch` have exactly the `pip freeze` of `~/envs/policy_deploy{,_torch}` (Python 3.10.12),
with `deploy_contracts`, `policy_deploy` (and `behavior_prompting`) editable from this checkout (the re-export kept
the egg-info folders, so the installs still resolve here). `behavior_prompting` here is identical to
final-deployment's (`diff -rq` empty), so it was installed as is. `~/envs/sim_deploy`: Python 3.10, mujoco 3.3.0, numpy
2.2.6, scipy, pyzmq, msgpack, pyyaml, opencv-python-headless, pexpect, `deploy_contracts` editable, and
`controlSW/franka_sim` on `sys.path` by a `.pth` file (it has no pyproject; it adds `controlSW/franka_deploy` itself).

**Checks run** (from `deployment/policy_deploy`, `unset PYTHONPATH`):
- `~/envs/sim_policy_deploy/bin/python -m policy_deploy.frames.rig_file ../contracts/rigs/rig_pi_v3_640.yaml` → ok.
- `~/envs/sim_policy_deploy/bin/policy_deploy check --mock --station ../../sim_deploy/stations/policy_sim.yaml`: every
  row passes except `frames Pi reachable` (no sim Pi on 127.0.0.1:5555 yet) and `manager terminal` (no tty in the
  agent shell; passes in a terminal or pty). Under `--mock` the PC is not probed (`manager PC` is an info row), so a
  missing sim PC shows only at `run`.
- frames under `--mock` with `fake_pi: false` against a stand-in Pi (`frames.fake_pi` bound on :5555): ready, 30 fps both
  cameras; no FakePiSender started.
- `replay` on PICK_TEST1 `pick1-v1` index 14 (= `pi-2026-11-18_episode_000152`, rows 8217–8806): chunks equal
  `inv(P_c) P_j` to 8e-9; past the end identity; over C4 (`policy.probe`) infer ≈ 3 ms. `use replay:…#14` writes it,
  `#30` is refused (30 episodes), `use hold_still` removes it again.

**Image check** (copy's `ImagePrep`, episode_000152 delivered as the Pi streams it: 640×480 INTER_AREA, JPEG q90;
training = the package's 98 ultrawide images of index 14; frames matched with umiproc's own 10 Hz selection from
`grid.npy`/`piece.json` and mid-exposure times, confirmed by content: ±1-3 neighbouring frames are 1.3-6× worse):

| Rig file | MAD vs training (all px / shown px, 0-255) | median optical flow | ArUco tag corner offset (224 px) |
|---|---|---|---|
| dc1ee36 `rigs/rig_pi_v2_640.yaml` (nominal 286.45, k 0.02 …) | 26.8 / 35.7 | 6.7 px (an underestimate: the views differ in scale) | 36.4 px (tags lost in 4 of 6 cases) |
| ea9f7fb `contracts/rigs/rig_pi_v3_640.yaml` (measured lens) | 1.03 / 1.38 | 0.01 px | 0.05 px (max 0.25) |

The nominal rig shows a much wider view (rig board and cables at the bottom, smaller fingers); ea9f7fb's matches
training to JPEG noise. Side-by-sides: `/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad/e1/uw_compare_{1,2,3}_*.png`.

**For the other packages.** E4: the settled rule refuses any kind but `hold_still` unless `hello.mock` is true, and
`compare_hello` under `--mock` expects `mock: true`, so the sim PC must answer `hello.mock: true`. The policy clips
every width to `limits.yaml gripper_width_max_m` 0.08 (`make_chunk`; 0.04 since W0/f632576), and episode 152's recorded widths reach 0.088
(training units, D8): open widths above 0.08 never reach the PC. E6/G6: the replay cursor assumes every chunk is
executed; a skipped step puts the replay `exec_h` steps ahead of the arm.

## E2: as built

`sim_deploy/rig/` (README there): `extract.py` (STEP -> `assets/` + `rig.yaml`, CAD venv), `add_rig.py`
(`add_rig(spec, hand_body, rig_dir, intrinsics=None)`, env_sim), `smoke_render.py`. STEP sha256 `5ce17c8b…6e63`.

**Frames.** hand (mm) = R·STEP + t with x_h = z_S − 1214.765, y_h = x_S − 699.8003, z_h = y_S − 925.6717 (a proper
rotation; the `Generic_Camera_Mount` frame is the hand turned 180° about z, origin at hand z = +7.5 mm, lip from hand
z 0 to 7.5). flange_T_hand = Rz(−45°), no translation (menagerie; the compiled model agrees). Consistency with
`contracts/frames.yaml`: F_T_TCP = Rz(+45°) = Rz(−45°)·Rz(+90°), so TCP x = hand +y (finger closing = image right),
TCP y = hand −x; the camera sits on the TCP's −y side = hand **+x**, which is where the STEP puts the cameras
(x_h 137–165 mm). Both statements hold together.

**Cameras** (hand frame, OpenCV: x = hand +y, y = (−cos t, 0, −sin t), z = (−sin t, 0, cos t); MuJoCo R_mj = R_cv·diag(1,−1,−1)):

| | optical centre (mm) | tilt toward the hand | ± along axis / lateral / rotation |
|---|---|---|---|
| `cam_imx219` | (137.14, −8.38, 26.82): 2.5 mm behind the barrel front (8.STEP r 7 cap; board 13.8 mm behind) | 15.000° | 2.5 mm / 0.5 mm / 1° |
| `cam_imx519` | (165.21, −8.18, 30.93): 0.8 mm behind the lens front (UC-873 r 2.75 barrel; board 6.0 mm behind) | 30.000° | 0.8 mm / 0.5 mm / 1° |

Image axes are a choice, not CAD (no sensor die modelled): fingers at the bottom, fingertip right of the IMX219, as in
training. IMU (U1_1 centre) at (129.4, −37.6, 9.3) mm, axes per the user: x = IMX219 image down, y = image left,
z = IMX219 optical axis (chip face parallel to the 15° face to 0.0002°); imu_T_cam_imx219 rotation = rig_pi_v3's
corrected one, t = (−12.0, −29.2, 14.9) mm.

**Camera -> fingertip point** (0, 0, z_tip) with TCP axes, z_tip in mm: IMX219 t = (8.38, 139.41 − 0.2588 z_tip,
9.59 + 0.9659 z_tip), R = Rx(15°); IMX519 t = (8.18, 158.54 − 0.5 z_tip, 55.82 + 0.8660 z_tip), R = Rx(30°). At the
old kit's z_tip = 223.9: IMX219 (8.4, 81.5, 225.9) vs training (8.1, 54.8, 204.9): same rotation (0.001°), 33.9 mm
apart (+26.6 down, +21.0 ahead); IMX519 (8.2, 46.6, 249.7) vs training (8.2, 26.1, 216.1) (rig_pi_v3 chain): 39.4 mm
apart. E3's real z_tip goes into the same formulas (`add_rig.cam_T_tip`).

**11-17 reproduced:** cover base x_h = 56.0 mm = 38.1 mm out from the spec page's hand-face reference x_h = 17.9 mm
(the mount's own contact faces are at x_h 15.0 / 20.2); cover front edge z_h = 67.15 mm = 156.75 mm before 223.9;
tilt 0.000°, sideways 0.0 mm.

**Mass** (rig only): 391 g ± 55 g: cover 189, mount 73, plate 43 (PLA 1.24 g/cm³ × fill 0.85 / 0.60 / 0.95 of the
CAD volume 179.5 / 98.3 / 36.8 cm³), Pi 5 46, IMX219 + lens 8, IMX519 4, buttons 3, IMU 2, LEDs 2, unmodelled
screws/ribbons 20 ± 10. COM hand (75.1, 4.3, 22.6) mm = flange (56.1, −50.1, 22.6) mm; inertia about the COM (hand)
diag ≈ (7.50, 7.80, 11.90)·10⁻⁴ kg m² (full tensors in `rig.yaml`). A kitchen scale on the real cover would replace
the largest guess (the fill factors).

**Model.** `add_rig` puts body `rig` (explicit inertial) under the hand with 9 visual meshes (group 2, ≤ 20k faces;
the printed parts are watertight, the boards not) and 4 convex hulls (mount, plate, cover, Pi 5; group 5,
contype = conaffinity = 1), and the cameras on the hand body. Joint-free bodies share link 7's weld, so the rig never
collides with the hand, fingers, link 7 or link 6 (filterparent); it does with links 0–5, the table and the eraser.
Smoke run (env_sim, `sim-rig` + rig): compiles, no rig contacts at home, cameras read back = rig.yaml (< 0.002°),
fingertips at v ≈ 317 (IMX219, f 250) and 298 (IMX519, f 400) of 480 = lower half, centred. A 142° pinhole at the
IMX219 sees the Pi 5 at the bottom right (11 %), the cover (6 %) and the LED Modulino at the bottom left (4 %).

**For the other packages.** E6: menagerie's hand visuals are already in geom group 2 (DESIGN says 1 for the hand);
franka_sim's `tool` body still carries the iPhone housing/cradle boxes and `cam_main`/`cam_uw` (the smoke run deletes
the geoms); `add_rig` references the STLs by absolute path (`info["mesh_files"]`) for the `meshes/` copy.
`rig.yaml station_tooling` has F_T_C for both cameras, C_imx519_T_C_imx219 (the old C_main_T_C_uw), the four parts'
boxes and the rig payload (flange frame; E3's fingers and adapters add to it). E5: render with the names above and
the intrinsics argument (same +0.5 px convention as franka_sim).

## E3: as built (2026-11-19)

The finger adapter is in `sim_deploy/adapter/`. See [adapter/README.md](adapter/README.md) for printing, hardware,
assembly and the caliper checks, and [adapter/adapter.yaml](adapter/adapter.yaml) for every number with its source.
`adapter.py` (cadquery, env `~/envs/sim_cad`, python 3.11) writes the STEP/STL files, the drawing, `adapter.yaml` and
the sim kit.

- **Part.** One part per side, the two mirror images (left = tag-0 holder at hand -y, right = tag-1 holder at hand +y,
  camera on hand +x). It is a cap over the carrier tongue (6.00 x 21 mm). The pocket ceiling rests on the tongue top
  (height datum) and the outer wall on the tongue's outer face. 2 M4x12 go through slots (+-2 mm along z, D4). The
  holder sits on a pad that stands in for the handheld's MGN9C carriage: its own 4 holes (15 x 10 mm), M3x16 screws
  in 2.5 mm pilots. The part is 41.3 mm tall, 19.7 cm3, about 25 g as solid PETG. It prints on its back face with no
  supports.
- **Interface reading (Franka manual R50010 Fig. 5.5).** The M4 holes are 7.0 mm below the tongue top (15 - 8). The
  tongue top is 11.40 mm above the housing top. Franka's hand meshes in franka_sim agree to 0.1 mm: housing 65.9,
  holes 70.3-70.4, tongue top 77.3 (hand z). The tongue inner faces are 25 mm apart at the closed stop (Fig. 5.4).
- **D7: TCP z = 0.23565 m from the flange** (`F_T_TCP` translation (0, 0, 0.23565); rotation unchanged). Hand frame =
  flange turned -45 deg about z, origin at the flange. The fingertip point is (0, 0, 235.65 mm) at every gap. The sum:
  tongue top 77.3 + 1.0 rack clearance + 157.35 from the holder back end to the soft fingertip. The holder's rack spans
  the tongue, so the holder cannot sit lower without changing the holder. The rig placement (spec page,
  `franka_mount_with_rig.STEP`) assumed 223.9 mm, so **the fingertips are 11.8 mm further from the cameras than the
  rig study assumed**. E2/G3 should judge the image match with 235.65.
- **D8: width offset.** In the CAD, the tag centre distance is gap + 40.58 mm (slope 1). The handheld measured
  0.0443 / 0.0439 m with the fingers touching. Recommended for the sim PC: `width = gap + 0.0443`, because the
  measurement defines the policy's units. The 9 % difference is unexplained: possibly the printed tag size (about
  14.65 mm instead of 16 mm would give a slope of about 1.09) or the sticker placement. Check on the robot.
- **Sim (for E6).** Use the kit `adapter/kit_franka_v1` (`finger_kit/v1`, id `iphumi_umi_soft_franka_v1`). It holds
  the `iphumi_umi_soft_v1` fingers and holders unchanged, plus `adapter_{left,right}.stl` in the same CAD coordinates.
  `kit_T_M` = the menagerie jaw mount J, so set tooling `finger_adapter: {mode: kit, kit:
  ../sim_deploy/adapter/kit_franka_v1, kit_sha256: <adapter.yaml sim_kit.kit_sha256>, adapter_length_m: 0.0,
  lateral_offset_m: 0.0, adapter_yaw_deg: 0.0}` and `tcp_x_along_hand: "+y"`. `model._mount_kit` then creates these
  geoms on `left_finger` / `right_finger` (group 0; E6 moves them to group 3 per §2): `pad_{l,r}` (soft fingers,
  contact), `finger_holder_{l,r}` (holders), `finger_holder_{l,r}_1` (adapters), `tag_{l,r}` (+ `_ink`). On the
  `+y` station, `tag_l` = id 1. It also sets the `tcp` site at z 235.65 mm and the finger-body inertials (0.138 kg
  each, including the stock finger's 15 g). The menagerie stock finger mesh is hidden and does not collide; the carrier
  tongue itself is not modelled (it is inside the adapter pocket).
- **Smoke test (`adapter/smoke_sim.py`, env_sim, mujoco 3.3.0): OK.** It compiles. F_T_TCP z is 235.648 mm. At gap 0
  the gripping faces sit at hand |y| < 0.0001 mm and the pad hulls are at distance 0 (touching). The tags are 40.58 mm
  apart. Over gaps 0 to 0.08, rays from the adapter vertices to the hand's visual meshes show at least 1.24 mm of
  housing below, 0.95 mm to the slot sides and 2.2 mm to the slot ends. hand.xml excludes hand-finger contacts, so
  MuJoCo would not report an overlap; that is why the smoke test uses rays. Renders: `sim_hand_gap000.png`,
  `sim_hand_gap080.png` in the E3 scratchpad.
- **Open (caliper on the real hand).** Tongue top to the holes (7.0). Flange to the tongue top (77.3; this sets D7).
  Closed tongue gap (25.0). Printed tag size and tag distance at contact (D8). Weigh the printed parts for the payload
  (about 0.26 kg of fingers in all, against 0.03 kg stock).
- **Loads.** The fingertips sit 165 mm beyond the carrier screws, about 3 times the 54 mm finger length Franka tested
  (tilting loads). Keep the grasp force at or below about 20 N. Tighten the M4 screws to 1.2 Nm in the printed part,
  not the manual's 2.25 Nm.

## E4: as built (2026-11-20)

`sim_deploy/sim_pc/` (README there: build, run, scene requests). `franka_control --sim --local --station
sim_deploy/stations/control_sim.yaml` = the copied franka_control with `SimArm`/`SimGripper` behind its `Arm`/`Gripper`
interfaces; the supervisor, gate, watchdog, impedance controller, homing, keyboard and C2/C3 codec are unchanged.

**Changes in the copy** (each file has the `copied from rnoro/final-deployment:<path> @ ea9f7fb` + `changes:` header,
under `deployment/control_deploy/franka_control/`): `CMakeLists.txt` (option `FRANKA_CONTROL_SIM`, default OFF: the
sim_pc sources go into `franka_control_core` with `FRANKA_CONTROL_HAS_SIM=1`, MuJoCo 3.3.0 from the wheel, rpath),
`include/franka_control/command_line.h` + `src/command_line.cpp` (`--sim`; needs `--local`, refuses `--mock`),
`src/app.cpp` (`--sim` → `sim_pc::makeSim`, `hello.mock = mock || sim`), `.gitignore` (`build-sim/`). The `sim:`
block is read by sim_pc's own loader from the same station file: franka_control's loader ignores unknown top-level
station keys, so `config_loader.cpp` is unchanged. Build: `cmake -S . -B build-sim -DFRANKA_CONTROL_SIM=ON
-DFRANKA_CONTROL_WITH_LIBFRANKA=OFF -DFRANKA_CONTROL_BUILD_TESTS=OFF` (no warnings).

**Physics (section 3).** One thread steps 1 ms per step with `mj_step1` (state, reading) → torques → `mj_step2` →
reflex check → scene jobs, paced to `CLOCK_MONOTONIC` and never ahead. More than 5 ms behind counts as `slow`; more
than 50 ms ends a running control call with `ArmError kRealtime` and physics time skips ahead (resync). It runs from
start-up to shutdown. Each step serves one of:
- a **torque call** (`runTorqueControl`): `step(reading)` is called on the physics thread (the control thread waits in
  `runTorqueControl`, as libfranka calls its callback). The reading: `q`, `dq`, `O_T_F` = site `flange`, `J_F` =
  `mj_jacSite` at `flange` [linear; angular], `coriolis` = `qfrc_bias` − true gravity, `tau_J_d` = the last command.
  The returned tau goes through libfranka's 100 Hz low-pass and 1000 Nm/s rate limiter (or, with
  `libfranka_limit_rate: false`, a `controller_torque_discontinuity` reflex), is saturated at 87/12 Nm, and
  **libfranka's gravity model** is added: the arm links (link0–link7 of the model) + Desk's Franka Hand (0.73 kg at
  (−0.01, 0, 0.03) m, flange) + the station's `load`. Not the model's true end effector: a wrong `load` sags as on
  hardware (measured below). `finished` ends the call after that step.
- a **joint call** (`moveToJoints`): the `HomingMotion` profile restarted from the measured q, a joint PD with
  K = `control.joint_impedance`, D = 2ζ√(K·M_ii), + the gravity model.
- otherwise a **joint hold** with the same PD where the arm is (the robot's own controller / brakes).

`requestStop()` is a per-call epoch (as `LibfrankaArm`): it ends only the call running then, with `kStopRequested`.
**Reflex:** contact forces on robot bodies (not the finger bodies) as J^T F on the arm joints, plus the gravity-model
error (true − compensated), low-pass 20 Hz (libfranka's `tau_ext_hat_filtered`); above
`control.collision_torque_thresholds_nm` the call ends with `kReflex` ("Motion aborted by reflex! ["joint_reflex"]
(sim: joint n ... contacts: a/b)"), `status()` stays `kReflex` and the arm holds until restart. The model's
actuators on robot dofs (franka_sim's 8 motors) are switched off: franka_control drives every robot dof.

**Gripper (D8).** `SimGripper` mirrors `FrankaHandGripper`: `widthM()` = gap + `gripper.width_offset_m`;
`moveTo(w)`: gap = clamp(w − offset, 0, min(`gripper_width_max_m`, 0.08)) (W0: min(limits 0.04, the model's stroke 0.04); with `sim.gripper.grasp_closes_fully` a grasp targets 0, see W0); closing by more than the deadband is a
grasp (`grasp_force_n` per finger, held on the object after it stalls), otherwise a move (40 N); `stop()` aborts a move
(it throws, as libfranka); `home()` closes fully and reopens fully. The fingers are a force-limited PD per finger
joint (the model's equality keeps them together); stall = the gap moved < 0.2 mm in 0.1 s.
Gap = `finger_joint1 + finger_joint2` (E3: the faces touch at q = 0).

**How the real code maps the width** (`src/franka_hand_gripper.cpp`): C2 width = libfranka `readOnce().width +
width_offset_m`; a commanded width w moves the hand to `clamp(w − width_offset_m, 0, min(limits gripper_width_max_m,
max_width))` (grasp when closing, else move). The sim station sets `width_offset_m: 0.0443`, so C2 reports gap +
0.0443 and w moves the gap to w − 0.0443. **Side finding for the real deployment (not changed):** its
`width_offset_m: 0.0` makes C2 report the bare hand opening (the fingertip gap) and execute the policy's width as the
hand opening. The policy works in training units (gap + 0.0443): it would see a closed hand as 0.000 (training never
went below 0.0439) and its "closed" command 0.0443 would stop the fingers 44 mm apart, so nothing narrower than
~44 mm is ever gripped. It should be 0.0443 there too (calibrate on the robot: tags in the camera vs C2).
**Second finding:** `limits.yaml gripper_width_max_m: 0.08` caps every width in training units (the gate rejects
more, the policy clips to it), so with the offset the fingers open at most **35.7 mm**. E6's eraser is grasped across
51 mm: it cannot be grasped. Training opened to 0.134 (gap ≈ 90 mm); the hand's full stroke is 0.08 + 0.0443 =
0.1243 in training units. Raising `gripper_width_max_m` is a contract change (shared, `make_examples`) for the user.
The hand homes fully open, so C2 starts at 0.1243 (inside training's range); the console clips its start to 0.08.

**Payload.** `sim_pc/payload.py` from the model: rig 0.3907 kg (body `rig`, = rig.yaml 0.39066) + 2 finger kits
0.1378 kg − 2 stock fingers 0.015 kg = **0.6362 kg, COM (0.0347, −0.0310, 0.0631) m, inertia about the COM
[[3.381e-3, 2.38e-4, 9.31e-4], [2.38e-4, 3.407e-3, −8.69e-4], [9.31e-4, −8.69e-4, 2.454e-3]] kg m²** (flange frame,
gap 0.04), was in `control_sim.yaml franka_control.load` until 2026-11-22. Assumes Desk's 0.73 kg includes the stock
fingers (the model's hand is 0.73 + the fingers, so the sim kept a 30 g residual: 0.36 mm sag). **Since 2026-11-22
the sim PC's load is the real PC's: mass 0, no setLoad** (user decision: mirror the real arm, whose rig was never
weighed; the model still carries the rig, so the sim now sags as the real arm, ~8 mm, and the reflex sees the same
model error). When the rig is weighed and its load set on the real PC, set the same values in `control_sim.yaml`;
the model estimate above stays there as a comment. Nothing else reads the value (`sim_world.cpp` adds `load` to the
gravity compensation only when its mass is > 0).

**State export / scene control (sections 4, 5).** As specified, plus: `hello` also returns `timestep_s`,
`gripper_width_offset_m`, the joint addresses and stats; `ground_truth` also `gripper_width_m` (training units) and
`arm` {q, call, reflex, tau_ext_nm}; contacts are `{a, b, force_n}` with geom names (body name for an unnamed geom),
every active contact except eraser–table, `force_n` = |normal force|; a new `stats` request. `model_sha256` is checked
at start with E6's method (`sha256` of "sha  relpath" lines over scene.xml + meshes/*); a mismatch is a FATAL.

**Deviations.** `set_arm` is always refused once franka_control is ready: it holds the arm in a torque session from
start-up to shutdown, also in idle (a teleport would be pulled back by the held target). E6: reset with `h` / C3
`home` (2.0 s from the moved pose). The Cartesian collision force thresholds are not modelled (joint torques only);
`load` inertia is used for gravity only. The control_deploy wrapper is not used with `--sim` (its strict station
loader refuses `sim:`; run the binary). F_T_TCP is read from the copy's `contracts/frames.yaml` at run time (z 0.23565).

**Smoke run (2026-11-20, E6's model sha cc928f83…, logs in the E4 scratchpad `e4/`):**
- hello: `mock: true`, `compare_hello(mock=True)` empty (smoke driver, pc_probe, console); commit `3633fcb3…`.
- C2 100.0 Hz, t_ns steps 10.00 ms, 0 seq gaps; pc_probe RTT p50 1.0 ms. Home: C2 `O_T_TCP` = the model's `tcp` site
  within 0.000 mm / 0.0001°; (0.3069, 0.0, 0.3542) m, TCP x → base −y, y → base −x, z down.
- State PUB 200.5 msg/s, t_ns every 5.00 ms, schema/sha/nq right.
- Console (the copy's steps, all PASS): hold drift 0.00 mm; move 1.5 cm tracking 1.78 mm; rotate 2° tracking 0.89°;
  gripper 4.3 mm; watchdog stopped after 1.016 s; out_of_box rejected → `fault`.
- Smoke driver: a 9.4 cm + 10° move in 2 s: 78 waypoints accepted, none shortened; tracking |p − p_target| max 8.25
  mm, mean 2.87 mm, rotation lag max 4.6° (40 Nm/rad); at rest 0.21 mm / 0.75° from the goal. Gripper widths in
  training units: commanded 0.0443 → C2 0.0444 (gap 0.0001), 0.0800 → 0.0800 (gap 0.0357), 0.0643 → 0.0672 (gap
  0.0229: the copy's 5 mm deadband on a ramped width). Grasp of the eraser on its side (25 mm): C2 0.0693 (gap
  0.0250), 20 N per pad; lifted 13 cm with it; released, it falls. No C3 for 1 s → `stopped` after 0.983 s. C3 `home`
  ok in 2.01 s, |q − q_home| 0.0009 rad. Keys: `h` then `s` mid-move → `stopped`; `h` → `idle`; `q` → clean exit.
- Load: right → idle TCP 0.36 mm below its target; `load.mass_kg: 0` → 8.17 mm sag, tau_ext 3.4 Nm (joint 4).
- Reflex: an object dropped on the rig cover with thresholds lowered to 0.3 Nm → `fault: libfranka: Motion aborted by
  reflex! ["joint_reflex"] (sim: joint 4 ... contacts: eraser/rig_cover_col)`.
- Pacing: real-time factor 1.0000 in every run; slow steps 0–3 per run (max lag 6.1 ms, at homing), 0 resyncs; step
  cost mean 90–150 µs, max 0.86 ms; the 1 kHz loop reports ~1 late cycle per 10 s (no RT kernel).

## E6: as built (2026-11-20)

**Package.** `sim_deploy/pyproject.toml` maps this folder to the package `sim_deploy` (only the top package is listed;
subpackages rig, adapter, model, scene, eval, sim_pi resolve inside it). Installed editable in `~/envs/sim_deploy`
(`pip install -e sim_deploy`); `sim_deploy.ROOT` / `sim_deploy.REPO` are the two folders.

**Model** (`model/`, `stations/sim_rig.yaml`). `python -m sim_deploy.model build [--station ...]` →
`build/sim_rig/{scene.xml, meshes/ (86 files), model.json}`; `python -m sim_deploy.model info` prints the summary,
`python -m sim_deploy.model.inspect scene.xml` loads it in a fresh process (what the sim PC / Pi see).
- Chain: franka_sim `build_model` (station `panda` + `franka_hand_iphumi_rig1` + `rig1_wifi` + `pick_place_table`,
  run `sim/physics`, scene off) with the overrides `finger_adapter.mode kit, kit ../sim_deploy/adapter/kit_franka_v1`
  (relative to controlSW/, franka_sim's convention), `kit_sha256 ee6328eb…`, `adapter_length_m 0`,
  `lateral_offset_m 0`, `adapter_yaw_deg 0`, `tcp_x_along_hand "+y"` → the `tool` body's housing, cradle,
  `cam_main`/`cam_uw` and `ee` site deleted, the menagerie spotlight and the stock finger geoms deleted → `add_rig(spec,
  "hand/hand", rig/)` with intrinsics from the copy's rig file → table, pedestal, floor, skybox, eraser, `cam_external`,
  two directional lights → geom groups → key `home` → meshes copied (kit meshes written as binary STL from franka_sim's
  inline vertices) → `scene.xml` with the attached hand's `hand/` prefix dropped (in the XML: bodies `hand`,
  `left_finger`, `right_finger`, joints `finger_joint1/2`, tendon `split`; clashing assets and classes become
  `hand_…`) → reloaded in a fresh interpreter and compared with the in-memory build (counts, body/camera positions,
  masses, geom sizes; MuJoCo's in-process mesh cache gave a false OBJ error on an in-process reload) → `model.json`,
  written last via a temp file (deleted first on a rebuild).
- `model_sha256` = sha256 of the `sha256sum`-style lines of `scene.xml` + `meshes/*` sorted by path (C locale):
  `cd build/sim_rig && LC_ALL=C sha256sum scene.xml meshes/* | LC_ALL=C sort -k2 | sha256sum`. Builds are
  byte-identical (same sha from any output folder). Current: `9c75edf8…6a0e`.
- nq 16 / nv 15 / nu 8: qpos `joint1..7` 0-6, `finger_joint1/2` 7-8 (0-0.04 each), `eraser` free joint 9-15.
  Actuators (franka_sim): `motor_j1..7` (torque, ctrlrange ±tau_max), `motor_hand` (tendon `split`, ±140 N); equality
  couples the fingers. Timestep 1 ms, implicitfast, elliptic cones, impratio 10, multiccd on. Sites `flange` (link 7 +
  107 mm), `tcp`, plus menagerie's `attachment_site`. Cameras `cam_imx519` (rig file `cameras.main` pinhole 492.5 @
  640×480), `cam_imx219` (the policy's virtual pinhole 246.65 as a stand-in: E5 renders the fisheye), `cam_external`
  (fixed, fovy 50, from (1.20, −0.75, 0.80) at (0.36, 0, 0.24)). Offscreen buffer 1600×1600, znear 0.0005·extent.
- Groups: 0 scene (table, eraser, pedestal, floor), 1 arm + hand visuals (menagerie's group 2 moved), 2 rig visuals
  (E2), 3 kit geoms with rgba alpha > 0 (pads, holders, adapters, tags + ink: 10), 5 collision-only (menagerie
  collision class + the 4 rig hulls: 15), 4 sites. The stock finger meshes are gone (kit mode had them contact-free).
- `tcp` = flange × the copy's `contracts/frames.yaml` F_T_TCP (read at build time); franka_sim's kit-derived TCP
  differs by 0.002 mm / 0°. Home key: `q_home_rad` of the copy's `franka_control/config.yaml` (7 digits), gap 0.080,
  eraser at its default pose; home TCP O (0.3069, 0.0000, 0.3546), flange 0.591 high.
- Masses: robot subtree 18.09 kg; hand 0.73 + fingers 2 × 0.1378 (E3 kit incl. stock finger) + rig 0.391 (E2) =
  hand subtree 1.396 kg; eraser 0.030 kg; the table is static (its 600 kg is never simulated).
- Scene (in O): table top z 0.050 (= the gate's `workspace_min_m` z: every fingertip height above the table is in the
  box, a TCP up to 10 mm into it is pulled back, not refused), x 0.16-0.96, y ±0.50, down to a floor at −0.70; the
  robot on a visual pedestal (x −0.20..0.14, top z 0); `table`–`link0/1` contacts excluded. Eraser: `size_m`
  [length, width, thickness] 127 × 51 × 12 mm with `resting: flat` (the user, 2026-11-20; `on_long_edge`, `on_end`
  also possible, both parameters), 30 g, friction (1.0, 0.01, 0.0001), condim 4, solref (0.002, 1) / solimp franka_sim's grip contact, yellow,
  default pose (0.29, −0.02) yaw 0 (long axis along O x). Lights: directional key from above with shadows
  (diffuse 0.5) + a fill (0.2), headlight ambient 0.2 / diffuse 0.12; neutral grey table, floor and skybox.
- Smoke: compiles from disk; no contacts at home; the eraser settles 0.05 mm in 1 s; renders at home
  (`$S/e6/cam_imx219_home.png`, `cam_imx519_home.png`, `cam_external_home.png`, `cam_imx219_wide110_home.png`, with
  $S the session scratchpad): the eraser between the fingertips in both rig cameras (IMX219 at (334, 249) px, IMX519
  at (346, 154) px), the fingers at the bottom as in training. **For E5:** the wide (f = 110 px) IMX219 render shows a
  black ring at about 66° off-axis: the IMX219 board mesh's lens barrel (E2 `rig_imx219_board`) seen from the optical
  centre 2.5 mm behind its front; it would show inside a 160° fisheye. Hide the camera's own board for that camera
  (or check it against a real frame at G3).

**D7.** The copy's `deployment/contracts/frames.yaml`: F_T_TCP translation z 0.150 → 0.23565, rotation and
`settled: false` unchanged, the two-line copy header; its comment line now says 235.65 mm. Readers: Python
`deploy_contracts.frames.TcpFrame` (laptop: compare_hello, settled rule), C++ `contract_frame.cpp` via
`CONTRACTS_FRAMES_FILE` (a path compiled in; the value is read at run time, no rebuild), control_deploy's station /
checks (path only), and this builder. `contracts/examples/` needs no regeneration: `examples.py` uses literal values
("never read from the yaml files"; its hello F_T_TCP is a made-up 0.6/0.8 rotation with z 0.15). Not run, would now
fail: `contracts/tests/test_frames.py::test_loads_the_real_file` (expects 0.150) and franka_control
`tests/test_pose_math.cpp` (testConfig loads the real frames.yaml, expects 0.150). `STRUCTURE.md` section 4 still
quotes the old file. Other tests use their own literal 0.15 matrices.

**Scene** (`scene/`, README there). `randomization.yaml`: x 0.26-0.34, y −0.06..0.02, yaw −8..+12°, settle 0.3 s;
rollout n of seed S uses `default_rng([S, n])`. The region is the training data's: `python -m sim_deploy.scene
replay-all <zarr>` carries each recorded grasp (relative to the episode's first pose) to the sim's home TCP; PICK_TEST1
gives x 0.238-0.332, y −0.057..0.016, yaw −1..5°. `check` confirms every corner is on the table, in the box and in the
IMX219 (and IMX519) view at home. Replay placement: `O_T_grasp = O_T_TCP_home · inv(P_s) · P_g` (g = the width's
closing start), eraser centre half its height behind the tip on the approach axis (the recordings' frame has the
table at z ≈ 0 and every grasp tip at −1..+1 cm), its body y along the grasp closing axis; `height_error_m` reports
how far above (+) the resting eraser the recorded close happens. Episode 14: +11.6 cm (G6 there checks the path,
not the grasp); episodes 2 (−0.4 cm), 8 (+1.6), 3 (−1.4) suit a grasp check. **Start pose:** every recording starts
with the approach axis 40-51° from vertical (camera looking ahead) at a 12-24 cm tip height and grasps at 28-40°; the
sim's home points straight down at 30.5 cm. The replay keeps the relative motion (it grasps at 10-20° in the sim); a
learned policy's first views differ from training's. A training-like start needs another home pose (E4: `set_arm`
is always refused).

**Grasp feasibility (open, for the user).** The user: flat, pinched across the ~12 mm thickness. The recordings: at
every grasp the closing axis is within 4-6° of horizontal with the tips at the table, so over a flat eraser it meets
the 51 mm width; PICK_TEST1 holds at 0.052-0.056 (D8: gap 8-12 mm) and opens to at most 0.0899 (gap 46 mm), and the
policy clips widths to 0.08 (gap 36 mm). Physics smoke (joint PD + gravity compensation, 20 N per finger, the full
80 mm stroke): the tips reach the table at x 0.26-0.34 (vertical, or tilted up to 35° tips toward the base); at a
6 mm tip height only the pads are near the table (holders/adapters ≥ 59 mm above; pad corners touch at 35° tilt,
clear by 1.6 mm at 20°); a flat eraser grasped across 51 mm lifts 10 cm, and so does one `on_long_edge` across 12 mm
(gap 11.9 mm = width 0.056, the training hold). Through the policy, a flat one cannot be opened around (36 mm gap
cap). `resting: on_long_edge` fits the data; the default stays `flat` pending the user's check.

**Eval** (`eval/`, README there). `python -m sim_deploy.eval run --policy … --rollouts N --seed S [--tag T]
[--timeout 60] [--tap] [--dry-run]`: `target.py` (run folder / hold_still / replay → task, folder, checkpoint, training
commit from `train_meta.json`), `driver.py` (freeze + `use`, eval.yaml, the rollout loop, collection, results),
`pd_session.py` (pexpect pty; parses the status line's head `[idle]` / `[homing]` / `[ep N running …]` and the message
lines "policy_deploy: all N processes ready (session: …)", "episode N started", "episode N: <outcome> (<reason>), …",
"home pose reached", key refusals `c: …` / `h: …`), `sim_link.py` (scene REQ, C2 watcher with per-rollout mode/error/
gate windows, state mirror, video thread), `report.py` (Wilson CI, summary.html, index.csv). The frozen station's
`session_root` is the eval folder, so the session lands in `<eval>/session/`. Success = eraser centre ≥ 5 cm above its
settled start for 1 s, grasp not required (user decision 2026-11-22; rule and values in `evaltool/constants.py`, which
the live judge imports); grasp = pads or holders of both fingers in contact with the eraser (E4's ground-truth contact
names); lift = grasped and ≥ 2 cm. Videos: `cam_external` at up to 10 fps, mp4v (no H.264 encoder in
opencv-headless), written at the rendered rate (about 5.6 fps on this laptop with shadows: the Intel iGPU via EGL).
Dry runs (no sim needed) go to `sim_evals/_dry_run/` without an index row; run on 2026-11-20 for `hold_still`,
`replay:…pick1-v1.zarr.zip#14` and `PICK_TEST1/policy/run_006_2026-11-18` (prompting): all OK; `policy_deploy check
--mock` fails only "Pi reachable" and "manager terminal" (no sim Pi, no tty), as E1 found. Smoke-tested offline: the
pty parser on synthetic manager output, the state mirror + video on a fake :15650 publisher, the metrics.

**run.sh.** Builds the model if `model.json` is missing; refuses if a sim port is busy; starts
`franka_control --sim --local --station stations/control_sim.yaml` (cwd `franka_control/`; flags from E4's
command_line.cpp) and waits for a scene `hello` with model.json's sha (60 s); starts `python -m sim_deploy.sim_pi
--station stations/sim_rig.yaml` (E5's settings take the model from `build/<id>/`) and waits for a :5555 packet
(30 s); runs the driver with the user's arguments (`--dry-run` skips the sim). Logs in `sim_evals/_logs/<time>/`;
SIGINT then SIGKILL to both on exit/Ctrl-C. `bash -n` and a `--dry-run` pass; the full chain is wave 3's.

**Open for integration.** (1) The grasp feasibility above. (2) No `set_arm` is used: every rollout starts with `h`
(a real homing, a few seconds). (3) After a fault the driver restarts policy_deploy; whether the sim PC leaves
`fault` on the next home is E4's behaviour, untested. (4) The `[ep N running]` status line is not parsed for
counters: skipped steps, margins and policy/cycle times come from the session's `steps.jsonl` after the eval.
(5) Replay G6 assumes every chunk executes (E1); `skips` in the rollout record show when not.

## E5: as built (2026-11-20)

`sim_deploy/sim_pi/` (README there): `python -m sim_deploy.sim_pi --station stations/sim_rig.yaml` (model
`build/<id>/scene.xml`, waits for `model.json`; state SUB :5650; PUB :5555), `state_stub` (smoke runs),
`handheld_check` (G3). Env `~/envs/sim_deploy`; the copied `deployment/policy_deploy` is put on `sys.path`
(`_deploy.py`) so the rig loader, camera models and the `pi_stream/v1` contract are the laptop's own code.

**Rendering.** One EGL context on the NVIDIA GPU (the shell's `__EGL_VENDOR_LIBRARY_DIRS` points at Anaconda's
Mesa, which put EGL on the Intel iGPU at ~3x the cost; `render.py` sets `__EGL_VENDOR_LIBRARY_FILENAMES` to the NVIDIA
vendor file before mujoco is imported, `SIM_PI_EGL_VENDOR=default` keeps the shell's choice). Views are pinholes with
asymmetric frusta set directly on the scene's `mjvGLCamera` (checked with marker spheres: < 0.2 px). IMX519: one
1621x1217 undistorted render, radtan by remap to the 1600x1200 native image; IMX219: 5 cube-face rectangles (2.45 Mpx,
f 449.4) remapped into the 1280x960 fisheye with the measured lens; both INTER_AREA to 640x480, JPEG q90. The lens
polynomial is used past the copy model's 89° cap (the sensor corners are at 98°; monotonic to 106.7°): the real
IMX219 frames show the scene to the corners, no black circle, and so does the sim. Each camera's own board/barrel is
hidden in its views (E6's note). The model's lights stay (made directional); the headlight keeps only its ambient
part (otherwise seams between cube faces). The render model is a private compile of `scene.xml`: lights, offscreen
size, shadow extent only; nq and joint addresses are checked equal to the file's.

**Stream.** As DESIGN section 6 with the live Pi's timing (PI_PAYLOAD 0.6) rather than "e.g." values: exposures
25 640 / 33 287 us, the ultrawide's mid-exposure 16.3 ms after the main's (pair gap −16.3 ms), each packet sent 45.8 ms
after its main `t_sensor_ns`; Pi clock = monotonic + 1000 s + 20 ppm; `skew_ns` with the live sign. Each camera is
rendered at its own mid-exposure from two interpolated state messages (quaternions normalised), never extrapolated;
a state held > 10 ms counts as `stale` (the frame is still sent, as the real Pi never stops). Overruns: > 1/2 period
late → main only, ultrawide repeated (`uw_reused`, never twice in a row); > 1 period → the tick is skipped (packet_no
and frame_no gap). Every packet passes `PiStreamPacket.from_parts` before sending.

**Smoke run** (E6's model sha 9c75edf8…, `state_stub` on :5660 with joints 1/5 swinging and the fingers moving, sim Pi
90 s, the copy's frames under `--mock` with `policy_sim.yaml`, `pi_collect` 30 s, `probe --save` twice; logs and
images in `$S/e5/`, $S the session scratchpad):
- 2694 packets in 90 s = 30.0 pkt/s; 0 skipped, 0 stale, 3 `uw_reused` (while the other processes started), 0 packet
  gaps; `pi_collect`: 901 packets, 0 bad, pair gap −16.3 ms (2 reuse packets at +17.0 ms), longest same-pair run 2.
- frames: clock synced at once (offset 1000.000 s, drift fitted 20.2-21.5 ppm vs 20 set), both cameras `ok` at 30.0 fps,
  latency (arrival − mapped main mid-exposure) 59-60 ms mean, pair rule refused 0/0/0 (old/new/reused), ultrawide
  `drop 3` = the 3 repeats (harmless frame_no gaps), probe: main ok 30.0 fps, ultrawide ok 30.0 fps.
- GL per tick 23.4 ms mean (p95 27.8): main 6.5 ms, ultrawide 16.9 ms (5 views with shadows); encode (remap + INTER_AREA
  + JPEG, 2 threads) 9.4 ms; send at 59.0 ms after main mid-exposure (target 58.6), 31 of 2694 more than 2 ms late.
- VRAM 243 MiB (nvidia-smi, the sim Pi process). Not measured with the policy on the GPU at the same time.

**G3** (`handheld_check`, 64 frames of `pick1-v1` index 14 with both tags in training's 224 image; tags DICT_4X4_50
rendered as real patterns; corners in 224 px; the training reference via the torch env):

| Placement | tag 0 corners mean / max | tag 1 corners mean / max | tag size sim/real |
|---|---|---|---|
| D8 as assumed (fingertip gap = width − 44.3 mm), camera = training `T_cam_tcp` | 24.5 / 30.8 | 34.7 / 42.2 | 0.83 / 0.78 |
| tag distance fitted (+60.6 mm vs D8), camera = training | 7.2 / 9.8 | 6.8 / 11.8 | 0.88 / 0.84 |
| tag distance fitted (+49.6 mm) + camera moved (−4.2, +1.7, +7.8) mm in IMX219 axes | **1.16 / 3.0** | **1.07 / 2.75** | 0.98 / 0.96 |

The same fits by pure projection through the copy's virtual camera give 1.01 px (translation) and 0.75 px (pose,
rotation −3.1/−1.9/+1.9°), so rendering + lens + ImagePrep agree with the projection model to ~0.1-0.2 px, and the
tag-size freedom fits 16.2 mm (the "smaller printed tags" explanation of D8's 9 % is not supported). Independently of
the tags, the kit fingertips projected with gap = width + 5…6 mm land on the real fingertips, with D8's gap they land
~10 px inside. **Findings for the user:** (1) in these training frames the recorded width is close to the fingertip
gap itself (gap ≈ width + 5 mm), not gap + 44.3 mm; D8's sim width offset (E4: `width = gap + 0.0443`) would send the
sim fingers ~50 mm narrower than the policy means. To confirm before G6/G7 (how umiproc defines width; a caliper on the
handheld). (2) Training's `T_cam_tcp` is ~9 mm off the image evidence, mostly along the IMX219 optical axis (the
fingers/tags sit ~8 mm closer to the lens than the CAD chain says, or the lens scale differs). (3) Training's IMX519
relative to the IMX219 (`T_imu_cam`) and E2's CAD differ by 6.4 mm (rotation 0.003°). The G3 pass ("a few px") holds
only with the fitted width mapping and camera shift; with training's values as they are it is 25-35 px.

**What differs from the real lab** (E6's scene): plain grey table (real: light wood), the finger kit's grey holders
and dark pads (real: blue holders, white fin-ray ribs with black outer faces), the Franka model's tags are plain
black squares (real: ArUco patterns; `handheld_check` shows how to texture them), no lab clutter (monitors,
keyboards), no LED glow, no motion blur, no vignetting or sensor noise, no auto-exposure/white-balance (the real frames
are pink-tinted). Shadows are from two directional lights, not area light.

**For the others.** E6: tag textures (ArUco ids 0/1, 16 mm marker on 20 mm white, `handheld_check.tag_texture`) and
kit colours would make the sim images closer to training. E4/D8: see finding (1). Wave 3: the sim Pi checks the model
sha only at start and stops with `sim_pi: FAILED` on the first state message of another model (rebuild → restart
both).

## E7: integration (2026-11-20)

### D8 settled: training width = fingertip gap

- **umiproc's definition.** `fit_gripper_range` takes the tag-to-tag x distance (`right.tvec.x - left.tvec.x`, PnP
  with 16 mm markers on the virtual pinhole) of every calibration frame; `min_width` / `max_width` are its extremes.
  Every episode width then goes through `calibration_interpolator`: `w - min_width`, clamped to `[0, max - min]`.
  So a recorded width is the tag distance **minus the closed tag distance**: 0 = fully closed. PICK_TEST1:
  `min_width` 0.04429, `max_width` 0.13421 -> recorded 0 .. 0.0899 (training's "open <= 0.090" is the full opening,
  its holds 0.052-0.056 are a 52-56 mm gap around the 51 mm eraser).
- **The calibration recording** (`PICK_TEST1/calibration/pi-2026-11-18_episode_000179`, 704 IMX219 frames,
  remapped with the copy's 640 rig + virtual camera, umiproc's detector settings; my reproduction: min 0.04451, max
  0.13455 vs umiproc 0.04429 / 0.13421): in the fully closed frames the soft fingertips **touch** (holder seam
  <= ~1 mm wide), so closed = gap 0 and width = gap. The tags slide on a straight rail (PnP positions along the
  rail: direction (1, 0.00, -0.00), spread across it < 1.3 mm), so the slope is 1 (tag scale: E5's fit 16.16 mm).
- **Why the CAD said 40.6 mm.** In the closed frames each tag centre sits 22.25 / 22.51 mm from the holder seam
  (measured in the tags' own PnP plane, independent of the camera pose); the CAD sticker outline centres are 20.0 /
  20.6 mm from the contact plane. **The real stickers are ~2 mm further out on each holder** than the engraved
  outline. With E5's camera shift, the PnP tag positions match the CAD's other two coordinates (asm y 87.4-88.8 vs
  87.5, z 15.3-20.0 vs 16.6). Not the tag size.
- **Sim.** `adapter.py` `Holder.tag_outward_real = 1.855` mm moves the kit's tags outward (tag distance at gap 0 =
  44.29 mm = training's `min_width`); kit `kit_franka_v1` sha `88fc5453...` (STLs unchanged, kit.yaml + SOURCE.md),
  `sim_rig.yaml` pinned to it; model `finger_kit.tag_separation_closed_m` 0.04429. `control_sim.yaml
  gripper.width_offset_m: 0.0`. The real deployment's `width_offset_m: 0.0` is **right** (E4's side finding
  reversed), provided the hand reads ~0 with the fingertips touching (E3's adapter: they touch at the closed stop):
  check on the robot that the closed hand reads C2 ~0.000.

### Fin-ray compliance (sim)

Rigid pads cannot hold a 51 mm eraser at the training hold width (gap 52-56 mm). `sim_rig.yaml finger_contact`
(`model/build.py _finger_contact`): explicit `pad_{l,r}`-`eraser` contact pairs (they replace the dynamic ones) with
`margin` 3.5 mm (a soft layer: contact from a 58 mm gap on), `solref [-200, -28]` (direct stiffness/damping; with the
eraser's mass about 2.9 N/mm per pad), `solimp [0.99, 0.99, 0.001, 0.5, 2]`, stiff friction `solreffriction
[0.002, 1]` (with the soft default the eraser crept out at ~0.05 mm/s), friction 1, condim 4. And the finger servo
`control_sim.yaml sim.gripper.kp_n_per_m` 2000 -> **20000** (with 2000 the layer pushed the fingers 3-4 mm wider than
commanded; the hand's fingers are geared and stiff). Measured (arm held by a PD at home, eraser between the tips,
the SimGripper servo law, 5 s): commanded 0.052 / 0.054 / 0.056 -> gap 52.8 / 54.5 / 56.3 mm, 7.5 / 5.0 / 2.5 N per
pad, no slip (< 0.01 mm); 0.045 -> 20 N cap. A first harness that teleported the arm each step showed a fake 9 mm/s
slip (the arm fell g*dt per step and dragged the eraser): hold the arm with forces, not qpos writes.

### Start pose variant `--home trainstart`

`python -m sim_deploy.scene.trainstart --write`: every PICK_TEST1 episode's start pose in a gravity-aligned frame at
its grasp (x = the closing axis on the table): median 11 mm along the closing axis, **164 mm behind, 166 mm above**,
mean tilt **49.5 deg** (40-64). Placed with the grasp at the centre of `scene/randomization_trainstart.yaml` (x
0.41-0.49, y -0.04..0.04, yaw 82-102 deg: the same spread, turned 90 deg and moved out) approached from the robot's
right (closing axis O +x): start TCP O (0.439, -0.164, 0.212), tip 16.2 cm above the table. Approached from the robot
side (as the deploy home faces) the start sits 12 cm from the base axis and has no IK solution; the policy only sees
poses relative to the episode start, so the layout's yaw is free. IK (damped least squares, 24 seeds, cost = distance
to the deploy home in joints 1-6): q = (-0.623, 0.314, -0.164, -1.783, 0.593, 1.398, -1.628), error 0.02 mm /
0.0006 deg, joint margin 1.27 rad, no contacts, in the box, the whole region in both camera views.
`stations/control_sim_trainstart.yaml` = `control_sim.yaml` + `franka_control.homing.q_home_rad` (a station override
of config.yaml, merged by franka_control's loader). `run.sh --home trainstart` starts the sim PC with it; the driver
(`--home`) checks the TCP after the first homing (within 1 cm) and uses the variant's region and home TCP (replay
placement, in-view checks); `eval.yaml home` records it. Homing from the key pose to it: no reflex, no contact.

### Appearance

`sim_rig.yaml appearance` (`_appearance`): pads white (0.93), holders blue (0.22, 0.30, 0.62), adapters dark grey;
`tag_l` / `tag_r` are textured slabs of the real stickers (DICT_4X4_50 ids from the kit, 16 mm marker on 20 mm
white; the ink boxes removed): detected by OpenCV in the model's IMX219 view with the right ids and orientation
(marker top toward the fingertips); a procedural light-wood table texture. Textures live in `meshes/` (in the sha).

### Workspace box (sim stations)

The training motions leave the copy's box (`workspace_min_m` x 0.25, z 0.05, "TODO: measure around the table"):
replayed from the deploy home, 10 of 30 PICK_TEST1 episodes reach x < 0.25 (min 0.198, after the grasp, moving
back toward the robot), and pinching the 12 mm eraser puts the fingertip TCP at the table. `control_sim*.yaml`:
`safety.workspace_min_m: [0.20, -0.40, 0.045]` (user's values). Finding for the real box: the same. (W0: upstream took x 0.20 and z -0.005 into config.yaml; the sim override is gone.)

### The eraser region

By the user's choice the deploy region is centred under the home TCP: `randomization.yaml` x 0.27-0.34, y
-0.03..0.03 (training median y -0.021), eraser `default_pose` (0.30, 0.00).

### Fix log

| # | Where | What was wrong -> fix |
|---|---|---|
| 1 | D8 (adapter, stations, sim_pi/handheld_check) | width offset 0.0443 -> 0; kit tags +1.855 mm outward; G3 variants E7 / E7+camera / D8_old |
| 2 | `run.sh` | the port check was inverted (refused when every port was free) |
| 3 | `eval/pd_session.py` | a message printed right after the status line's erase code (`...\x1b[Kpolicy_deploy: all 3 processes ready`) was taken as status: the session folder was never found, steps/tap never collected -> split at `\x1b[K` |
| 4 | `eval/driver.py` | `[idle]` arrives before the ready line: wait up to 5 s for the session folder |
| 5 | `deployment/policy_deploy/policy_deploy/policy/replay_policy.py` (+ STRUCTURE.md 5.5) | the cursor advanced exec_h steps per infer, but the manager replans at every frame (~33 ms) and replaces the chunk: the replay ran ~12x too fast and the arm moved ~1/12 of the path -> cursor = time since the first infer |
| 6 | `eval/driver.py`, `pd_session.py` | a FAILED start (the PC in fault) was not recognised: waited 180 s -> `FAILED` parsed, exit early, exit code 3 when the sim PC latched a fault |
| 7 | `run.sh`, `eval/driver.py --resume`, `report.append_index` | after a latched fault (needs a PC restart, as on the robot) the eval stopped -> run.sh restarts the sim PC + Pi and resumes the same folder (<= 5 times); one index row per eval folder |
| 8 | `run.sh` (`set +e` around the driver) | with `pipefail` + `set -e` the script ended at the driver's exit code before reading it |
| 9 | `eval/driver.py` (UI runner) | SIGINT dropped the rollout in progress -> recorded as `interrupted` (in no rate: `report.metrics` keeps it in outcomes / `interrupted`); `log()` survives a closed stdout; `run.sh` uses `tee -i`; the stable line is `[eval] eval folder: <path>` |
| 10 | `eval/driver.py`, `scene/placement.py`, `scene/__main__.py`, `run.sh` | `--home deploy|trainstart` (new `scene/trainstart.py`, `randomization_trainstart.yaml`, `stations/control_sim_trainstart.yaml`) |
| 11 | `model/build.py`, `sim_rig.yaml` | appearance + fin-ray contact pairs; `control_sim.yaml` finger kp 20000 |
| 12 | `control_sim.yaml` | `safety.workspace_min_m [0.20, -0.40, 0.045]`; eraser region / default pose centred (user) |

### Checks (PLAN section 6): see the results column there. Open items

1. **G6/G7 speed:** the gate's `v_max_m_per_s` 0.25 shortens the human-speed motions (replay ep 2: 178 waypoints
   shortened in 154 steps). Each chunk is relative to the manager's anchor (its previous target), so a shortened
   distance is never made up: the replayed TCP stopped 11 cm above its grasp point and closed in the air. The
   learned policies are trained on the same speeds. Decide: raise `v_max` (the copy says "raise to 0.5 only after
   tuning on the arm") in the sim stations, or slow the replay.
2. **G7:** from the deploy home run_006 drove up out of the box (z 0.715 > 0.70) within 7 s: its first views (the
   camera looking back at the robot base) are unlike training's. Try `--home trainstart`.
3. The gate's cone (`cone_max_deg` 60 about straight down): training grasps tilt 28-40 deg and starts up to 64 deg.
4. G3's residual with training's `T_cam_tcp` (5 / 12 px): the camera sits ~9 mm closer to the fingers along its axis
   than `rig_pi_v3`'s CAD chain says (E5's finding; a 1.0 / 3.3 px fit with (-4.0, +2.5, +8.8) mm). Training files are
   not changed; the user's call.
5. Pads look grey in the rig's shadow (directional light from above); the real lab light is diffuse.
6. Not re-run after E7: E2/E3 smoke scripts (geometry unchanged: cameras, TCP and masses equal to the wave-2 model
   to 0.0), G6 on episodes 8 and 3, G7 on other run folders (fast-track).

## P1: start-pose port (2026-11-20)

**Port.** `git apply sim_deploy/ports/start_pose_snapshot_20261120_0907.patch` (the read-only `git diff ea9f7fb --
deployment` of final-deployment's working tree, no untracked files) onto `deployment/`: it applied **cleanly**, no
textual conflict. The only file both sides changed is `policy_deploy/STRUCTURE.md`; its 4 hunks landed 26 lines
lower (below the sim's §5.5 replay text, which is kept). Each of the 42 files the patch touches has, after its
provenance header, `copied from rnoro/final-deployment:<path> @ ea9f7fb` (where the sim had not added it before) and
`changes: + start pose (key p) from final-deployment working tree snapshot 20261120_0907` (`<!-- -->` in .md, before
a leading docstring in .py). The snapshot's own header lines (`+ ...`) are kept as upstream text.

| Group | Files (under `deployment/`) |
|---|---|
| docs | `DIRECTIONS.md`, `PLAN.md`, `control_deploy/{DIRECTIONS,STRUCTURE}.md`, `policy_deploy/{DIRECTIONS,STRUCTURE}.md` |
| control_deploy (Python) | `control_deploy/{cli,runner,status_line}.py`, `tests/test_{runner,status_line}.py` |
| franka_control | `config.yaml`; `include/franka_control/{config,control_loop,key_queue,keyboard_reader,shared_state,supervisor}.h`; `src/{config_loader,control_loop,key_queue,keyboard_reader,supervisor}.cpp`; `tests/test_{config_loader,control_loop,fake_arm_run,keyboard_reader,supervisor}.cpp` |
| policy_deploy manager | `policy_deploy/manager/{episode_loop,failure_monitor,operator,pc_probe,status_line}.py` |
| policy_deploy tests | `tests/integration/test_{cross_franka_control,run_mock}.py`, `tests/manager/test_{episode_loop,failure_rows_pc,flow,main,operator,pc_probe,status_line}.py` (updated by the patch, not run) |

What it does (final-deployment's design, unchanged here): PC key `p` = from home only (every joint within
`homing.at_pose_tol_rad` 0.05 of `q_home_rad`), `Supervisor::startArmMove` (the `h` path) to `homing.q_start_rad`
with the finger target set to `gripper_width_max_m` (0.08; 0.04 since f632576) just before the move; at the end the TCP must be inside the
box and tilt <= `cone_max_deg` (no margin), else `fault`; refused while `q_start_rad` is null (config.yaml default).
Once set, C3 `start` is refused unless every joint is within 0.05 of it. **The laptop's `h` is removed** (no laptop
key moves the arm; C3 `home` stays in the contract). `pc_probe --start-pose` prints the C2 joints as a
`q_start_rad:` line, the TCP height, tilt and direction, the width.

**The sim's own changes vs the snapshot (no overlap, nothing to resolve):** `launcher.fake_pi` and the `replay` kind
(E1; replay cursor fix E7), D7 `contracts/frames.yaml` z 0.23565, `--sim` (CMakeLists, command_line, app.cpp,
`.gitignore`). `SimArm::moveToJoints` serves p's joint goal as it serves h's (same `HomingMotion`), and the fingers
open through franka_control's gripper thread (`SimGripper::moveTo`): no sim_pc change was needed for the move. The
copy now differs from `git archive ea9f7fb deployment` in 61 files: the 42 above + the 18 earlier sim files (one,
`policy_deploy/STRUCTURE.md`, is in both) + `log.h` / `log.cpp` (below).

**Consequence for the eval (P2):** `eval/driver.py` (line ~314) homes by pressing `h` in the policy_deploy pty; the
ported laptop has no `h` (it now prints `'h': not a key. keys: c start ...`, and HOME_DONE never comes), so homing must go through the scene key
`h` (then `p`).

**Scene key (sim only).** `{"cmd": "key", "key": "h|p|g|s"}` on :5651 (§5 table). Wiring, all under
`FRANKA_CONTROL_HAS_SIM`: `SimParts::attach_keys` (a weak_ptr hook); `app.cpp` attaches the messaging thread's
`KeyQueue` right after creating it and detaches it (RAII) before it goes away. The messaging loop drains that queue
on every iteration whether or not `--keyboard` is given, so a scene key takes exactly the keyboard thread's path
(`KeyQueue::push` -> `Supervisor::onKey`). For the reply, `Log::setTap` (new, sim build only, in the copied
`log.h`/`log.cpp`) passes each printed line to the sim, which waits (up to 1 s, the REP held, the state PUB not
blocked) for the supervisor's one line starting `key <k>:` or `key <k> refused:`. Files: `sim_pc/include/sim_pc/
{sim_world,sim_parts}.h`, `sim_pc/src/{sim_scene,sim_parts,sim_world}.cpp`, `deployment/.../src/app.cpp`,
`include/franka_control/log.h`, `src/log.cpp`. Build: `build-sim` reconfigured and rebuilt, no warnings.

**`stations/start_pose.yaml`** holds only `franka_control: homing: q_start_rad:` (null or 7 numbers) + a comment
block (source, date, the pose in the sim layout); initially PLAN §9's stand-in. franka_control loads **one**
`--station` file (its loader merges that file's `franka_control:` over `config.yaml`), so `run.sh` now runs
`sim_pc/merge_station.py BASE start_pose.yaml build/<base>.merged.yaml` before every sim PC start (also each
restart after a fault, so an edited start pose is picked up) and starts the sim PC with the merged file
(`build/control_sim.merged.yaml`; gitignored). The merger refuses any other key in start_pose.yaml, merges maps key
by key (franka_control's rule) and makes `sim.model`/`model_json` absolute.
`stations/control_sim_trainstart.yaml` is **kept working**, not retired: `run.sh --home trainstart` merges it the
same way; its home already is the stand-in, so `p` there is a 2.0 s no-op move and C3 `start` is accepted at home.
It is superseded by `h` -> `p` and can go once P2's flow uses the start pose (`scene/trainstart.py --write`
regenerates it from control_sim.yaml and does not know start_pose.yaml).

**The stand-in against the ported checks:** joints inside the Panda's limits (the loader accepted it; min margin
1.27 rad); reached pose TCP O (0.4385, -0.1640, 0.2114) inside the sim box (x 0.20..0.80, y ±0.40, z 0.045..0.70,
the E7 override) and tilt 49.66 deg <= `cone_max_deg` 60 (no `fault`); 15.1 s from the deploy home < `homing.
timeout_s` 30; at the end |q - q_start| 0.0001 rad, after the next torque session's hold 0.0038 rad (joint 2: the
Cartesian hold sags ~1 mm), well inside `at_pose_tol_rad` 0.05.

**Smoke (sim PC alone, no `--keyboard`, ports free before and after; logs in the P1 scratchpad `p1/fc_*.log`):**
- `q_start_rad: null` (temp station): scene `p` -> `ok: false`, `key p refused: homing.q_start_rad is not set: ...`;
  C3 `start` accepted anywhere (unchanged).
- Stand-in (`build/control_sim.merged.yaml`): C3 `start` at home -> `refused: not at the start pose: press h, then p
  on the PC`. Scene `p` -> `{"ok": true, "log": "key p: start pose: 15.1 s joint move to homing.q_start_rad"}` (3-6 ms
  reply); C2 `homing` -> `idle` after 15.12 s, log `start pose reached (fingers at 0.080 m)`; |q - q_start| 0.0001
  rad; C2 width 0.0800, finger gap 0.0800; no contacts. `p` again -> `not at home: press h first`. `g` accepted.
  `pc_probe --start-pose`: `q_start_rad: [-0.6231, 0.3176, -0.1643, -1.7796, 0.5940, 1.3950, -1.6265]`, z 0.211 m,
  tilted 49.7 deg, "pointing to the robot's left", 80.0 mm, compare_hello all equal. C3 `start` -> ok (`holding`);
  scene `p` / `h` while holding -> refused (`mode is holding: the laptop is driving an episode ...`). 28 waypoints
  +8 cm in z accepted (0 shortened), stop -> `stopped`, |q - q_start| 0.1385 -> C3 `start` refused (not at the start
  pose), `p` refused (not at home). Scene `h` -> 14.5 s move, |q - q_home| 0.0001 -> C3 `start` refused. `s` ->
  `key s: stopped (was idle)`. `q` / unknown key -> `ok: false`. SIGTERM -> exit 0.
- `control_sim_trainstart.merged.yaml`: `p` (2.0 s), `start pose reached`, `start` accepted; exit 0.
- Physics: rtf 1.0000, 0 resyncs, 0 reflexes; 36 slow steps (max lag 8.9 ms), step cost mean ~410 us (E4: 90-150 us;
  other work was running on the machine). The 1 kHz loop's late-cycle warnings are E4's (no RT kernel).

## P2: start pose in eval/scene/UI (2026-11-20)

**Flow (`--home start`, the default of the driver, `run.sh`, the scene CLI and the UI).** Per rollout, as on the
robot: the PC's `h`, the PC's `p`, then the laptop's `c`. The ported laptop has no `h`, so the driver presses both PC
keys through P1's scene `key` (`--home-key auto` takes the laptop's `h` only if the copy's `operator.py` still has
`HOME = "h"`). After each key: C2 must show `homing` within 3 s (else the key had no effect), then `idle` with every
joint within `homing.at_pose_tol_rad` (0.05, from config.yaml + control_sim.yaml) of the goal (`q_home_rad`, then
`start_pose.yaml`'s q). A refused key (`ok: false`, e.g. `not at home: press h first`) ends the rollout as
`start_pose_failed`; idle at another pose stops the eval (the sim PC runs another start pose: `run.sh` merges
`start_pose.yaml` only at a sim PC start). Then the eraser (below), `settle_s`, `c`. A `not at the start pose` refusal
of `c` ends the rollout as `start_failed` with a hint instead of retrying for 30 s.
`--home deploy`: `h` only, `scene/randomization.yaml`; `run.sh --home deploy` merges a generated
`build/start_pose_null.yaml` (`q_start_rad: null`) so the sim PC, like the real one, accepts `c` anywhere.
`run.sh` writes `build/control_sim_<home>.merged.yaml` with P1's `merge_station.py` and exports its path
(`SIM_DEPLOY_SIM_PC_STATION`, recorded in `eval.yaml`). `trainstart` is an alias of `start` everywhere (its IK
solution is the stand-in); `control_sim_trainstart.yaml`, `scene/trainstart.yaml` and
`randomization_trainstart.yaml` are no longer read (kept; `scene/trainstart.py` still regenerates them).

**Records.** `eval.yaml home`: variant, `home_key`, `q_home_rad`, `at_pose_tol_rad`, `rollout_start_O_T_tcp` (v1: `episode_start_O_T_tcp`),
`sim_pc_station`, and for `start` the start pose used (q, the file's text, TCP, fingertip direction, tilt, heading in
words, height above the table, cone tilt, move time, checks, vs the demos); `<eval>/start_pose.yaml` is a copy. Per
rollout: `home`, `t_home_s`, `t_start_pose_s`, `start_pose` (C2 after the last key: q, TCP, tilt, heading, height above
the table, largest joint error, width), `placement` (ahead, left, yaw_rel, `clipped`). `summary.html` has a "rollout
start" row.

**The start pose file.** `stations/start_pose.yaml` (P1's format: only `franka_control.homing.q_start_rad`, comments
allowed) is written by `python -m sim_deploy.scene start-pose --q ... | --from-probe FILE | --from-git [REF]`
(`scene/start_pose.py`); `--show` prints it. Checks in the sim model (refused without `--force`): Panda joint limits
(the loader's table) and the model's, no contact, TCP in the sim box, tilt <= `cone_max_deg` with no margin, the `p`
move from home < `homing.timeout_s` (`HomingMotion`: `max(min_duration_s, 1.875 max|dq| / max_joint_speed)`), a
heading (not straight down). Information: tilt / height vs the demos, the eraser region at that pose. `--from-git`
reads the committed `REF:deployment/control_deploy/station.yaml` with `GIT_OPTIONAL_LOCKS=0 git show` (today: "not set
yet"). The stand-in in the model: TCP O (0.4385, -0.1641, 0.2118), 16.2 cm above the table, 49.5 deg, pointing to the
robot's left (P1 reached (0.4385, -0.1640, 0.2114), 49.66 deg: 0.4 mm of sag).

**The eraser relative to the start pose.** `scene/randomization_start.yaml` (schema `sim_deploy.scene_random_start/v1`,
written by `scene start-data --write`): frame origin = the start TCP on the table, ahead = TCP z on the table, left =
up × ahead; ranges = the PICK_TEST1 demos' 10th-90th percentiles: ahead 0.090-0.217 m (median 0.159), left
-0.035..-0.003 (median -0.019), yaw_rel -6.6..+0.8 deg (median -2.2); the centre clipped to the table and the box
(xy). `Scene.load(home="start")` runs the FK once (the MuJoCo model load ~8 s, cached per process); `home_tcp()` is then
the start TCP, so the replay placement is carried from the start pose too: all 30 PICK_TEST1 replays grasp within
2 cm of the resting eraser (median +0.2 cm; +7.5 cm from the deploy home). `in_workspace` now uses the sim PC's
effective box (control_sim.yaml over config.yaml, x >= 0.20) instead of config.yaml's (x >= 0.25). (W0: config.yaml itself now has x 0.20 / z -0.005 and control_sim.yaml no longer overrides the box.)

**UI.** `RunRequest.home` = `start` (default, "start pose (p)") | `deploy` ("deploy home (h only)"); `trainstart` is
accepted as `start` (server and runner); the runner now passes `--home`. `GET /api/start_pose` = `start_pose.show()`
trimmed, cached on the mtimes of start_pose.yaml and model.json; the page shows it under the settings (click =
reload). The catalog's replay `grasp_ok` is carried from the start pose when one is set.

**Smoke (offline, no sim, no sim ports; 2026-11-20).** Dry runs `--home start` and `--home deploy` for `hold_still`
and `PICK_TEST1/run_006` (bf16, 8 steps, exec_h 12): `dry run: OK` (the start pose and "key p in the copy" pass;
`policy_deploy check --mock` fails only Pi reachable + terminal, and for run_006 also `self_check` timed out after
20 s, both homes alike, while other work ran); `--home trainstart` -> start. `scene start-pose`: `--show`, `--q`
(stand-in written), `--from-probe` (synthetic printout), `--from-git` ("not set yet", exit 1), refusals (straight down:
heading; into the table: box + contact; joint 4 at 0: limits, box, cone); P1's `merge_station.py` accepts the written
file and the null file. `scene check` (start and deploy): OK. Driver key logic with fake C2/scene: accepted 0.7 s,
other pose, refused, no reaction (3 s). UI `--stub` on :8794: page has both options and the start-pose line,
`/api/start_pose` (first 9.8 s, then 30 ms), `trainstart` -> start, `bogus` -> 400; runner maps start / deploy /
trainstart / missing -> `--home start|deploy`. Not run against the sim chain.

## P3/P4: start-pose checks and sweep (2026-11-20)

### Layout change (the user, before the sweep)

The robot base is mounted directly on the table: **table top = the base's mounting plane = O z 0**, flat, all around
the base. `stations/sim_rig.yaml scene.table`: `top_z_m` 0.0, x -0.40..1.00, y ±0.80; the pedestal is gone
(`model/build.py`: `scene.pedestal` optional); link0 and link1 stay excluded from table contact (link0 sits on it),
every other link, the hand, rig and fingers collide with it. Sim box floor `safety.workspace_min_m` z **-0.005** (table
- 5 mm, as E7's 0.045 was for the old table) in `control_sim.yaml` and `control_sim_trainstart.yaml`; `ui/live.py`
`table_top_z_m` 0.0. Model rebuilt: sha `166a0649571a...`; `scene check` (start and deploy) OK. `start-data` (the region
relative to the start pose) is unchanged (it never depended on the table); the replay placements from the start pose
are unchanged (same height above the table); the deploy home is now 35.5 cm above the table. The P3 stand-in was
re-solved with E7's trainstart IK for the new table (16.2 cm above it). Every P3 run made with the old table is marked
`SUPERSEDED.txt` in its eval folder (summary/index rows stay).

### P3 results (PLAN section 6, "P3" table)

- **G5 pass.** hold_still from the start pose: 0.07 mm / 0.12 deg over 20 s, hand 80.0 mm, start pose within 0.0001
  rad; the refusals through the driver (`c` away from the start pose -> `start_failed`; `p` away from home -> `not at
  home: press h first`).
- **G6: the conventions pass, open-loop replay fails where the demos are faster than the gate.** Replays 19 and 0 grasp
  and lift (path error median 8.8 / 16.8 mm, 14.5 / 26.6 mm at the recorded grasp). 28 and 16 close 8-11 cm short of
  their grasp, at v_max 0.25 and at a sim-only 0.5 alike (`stations/vmax05_g6.yaml`): 29 of 30 demos have 50 ms
  segments above 0.5 m/s before the grasp (peaks 0.6-1.5 m/s, likely hand jerks or tracking jumps; episode 0 is the
  exception), the gate shortens them, and an open-loop replay never makes the distance up. A closed-loop policy can.
- **G7 fail: run_006 rejects the sim's IMX219 images.** From the stand-in, all 3 rollouts rise and back off from the
  first chunk and fault after 11-18 s (box y, cone 62-63 deg). Offline (torch env, CPU, `sim_evals/_start_pose_sweep/tools/offline_first_chunk.py <G7 eval>`,
  `SPLIT=1` for the camera swap; logs `sim_evals/_start_pose_sweep/p3/offline*.log`), the same model and low-dim input: training episode_000152's first images -> first chunk
  +5.6 cm toward the fingertips (the demos: +5.7 cm in 0.75 s); the sim's -> -1.3 cm; sim IMX519 + training IMX219
  +5.4 cm; training IMX519 + sim IMX219 -0.5 cm; the width (0.080 vs 0.087) changes nothing. The geometry matches
  (eraser, fingers, tags where training has them: `g7_policy_images_vs_ep152.png` in the G7 eval folder); what differs
  is appearance: the orange striped table texture vs the pale lab table, the rig's and arm's hard shadows, the
  saturated eraser, the IMX219's empty table to the horizon and grey sky vs the lab clutter. **This is the open item
  that blocks any benchmark** (E5/E7 appearance, `sim_rig.yaml appearance` + the sim Pi's IMX219 render).
- Inference in G7: 270 / 270 / 188 ms p50, margin down to -140 ms, 0 skipped: another session's pytest suites
  (policy_deploy, teleop_collect, control_deploy) ran on the laptop at the same time (load ~8); E7 had 177-182 ms.
  Those suites also briefly bind :5620 / :5555 (their fake PC / Pi): two sim starts failed with "cannot bind the C2
  endpoint ... Address already in use" (fix 2 below).

### P4 candidates and the envelope of q_start_rad

`python -m sim_deploy.scene.start_sweep candidates` (`scene/start_sweep.py`): tilt {38, 48, 58} deg x fingertip height
{10, 16, 22} cm above the table x heading {left, away, right, away-left, away-right}; orientation = the stand-in's
(the demos' mean start orientation) turned to the heading and tilt; position = the demos' median eraser centre at a
fixed point G (left/right (0.45, 0), diagonals (0.50, 0), away (0.60, 0)); IK = trainstart's DLS, 24 seeds. Then the
left heading with the layout moved out, G (0.50, 0) and (0.55, 0) (`leftx50_*`, `leftx55_*`). Checks: describe()
(limits, sim box, cone, contact, heading, move time), joint margin >= 0.10 rad, the region (every corner and the median
on the table, in the box, in BOTH cameras' views), the p path (linear joint interpolation from home, 121 samples:
no contact, >= 2 cm to the table and between hand/rig/fingers/links 6-7 and links 0-3), and the real PC (below).
Results: `sim_evals/_start_pose_sweep/candidates.{md,json}`, pose files `poses/<id>.yaml`.

**Feasible: 40 of 63** (26 of the first 45, 14 of 18 shifted). Why the others fail = the limits on q_start_rad:

| Fails | Where | Why |
|---|---|---|
| eraser region out of the IMX519 view | tilt 38 at 10 and 16 cm (every heading) | the IMX519 looks over the near region: at 38 deg the corners 9 cm ahead fall below its image; at 22 cm they are in view |
| no IK / joint 4 or 5 at its limit, self clearance 0.7 cm | heading away (fingertips pointing away from the base) | the flange sits behind the TCP toward the base; with the eraser at 0.45-0.50 m from the base axis no solution (joint 4 at -3.07); at 0.60 m only 48 deg / 16 cm is feasible, joint margin 0.12 rad, self clearance 7 cm |
| joint 7 margin < 0.10 rad | heading right at 38/16, 48/16, 48/22, 58/22 | the hand roll the camera-up orientation needs at the robot's right is near joint 7's limit |
| (none) | heading left, every tilt and height except 38 deg at 10/16 cm | joint margin 1.0-1.4 rad, self clearance 22-24 cm, table clearance = the start height |

Every candidate passes the real PC's `p` rules (below); the cone margin at 58 deg is only 2 deg (the demos start up to
64 deg: such starts would be refused by the PC's cone 60).

**The demos carried from each pose** (`sim_evals/_start_pose_sweep/tools/demo_paths.py` -> `sim_evals/_start_pose_sweep/demo_paths.json`; the shifted
layouts: `tools/left_out.py`):
each of the 30 PICK_TEST1 recorded paths applied relative to the start pose (as the policy's relative actions are):
inside the real box with the gate's 10 mm margin and the cone + 2 deg; and IK of its grasp and end pose seeded from the
start q. Findings: the **pick** (start to 1.5 s after the grasp) depends only on the start height (demos that started
higher than the start carry into the table: 14/30 at 16 cm, 6/30 at 22 cm, 26/30 at 10 cm) and on the tilt (58 deg: 9/30
over the cone); the **retreat** after the grasp moves ~25 cm to the hand's left: with the left heading at the stand-in's
layout it crosses the robot-side box face x 0.20 in 21/30 demos, with the layout 5 / 10 cm further out in 10 / 3. The
away-type headings avoid that face but reach the demos' grasp and end poses with 0.001-0.09 rad joint margin (some not
at all: away-left 11-21/30). Left heading: 30/30 reachable, margin >= 0.25 rad (0.47 at `leftx55_t48_h16`).

### Screen and full eval

**Screen** (`start_sweep run --stage screen --poses feasible`, run_006, 1 rollout, `--timeout 25`, seed 0, sim-only
`--fast-homing`, no video; `sim_evals/_start_pose_sweep/summary.md`): 23 of the 26 first feasible poses (the shifted `leftx*` were added
after it started; stopped on the coordinator's change of plan, the 24th eval marked `INTERRUPTED.txt`). No grasp, no
approach anywhere: 18 faults (box 11, cone 7), 5 timeouts; one early fault (< 10 s: cone, 9.8 s at away_left 58 deg /
10 cm), the rest at 11-24 s; the TCP never got closer to the eraser than it started (10-28 cm). Every pose shows the
same retreat as G7, so the screen cannot rank poses until the IMX219 render is fixed (G7). **The full eval was not
run** (coordinator: it cannot rank poses either, and it competes for the GPU). The ranking is offline instead:

**The demos through the real gate** (`python -m sim_deploy.scene.gate_check --q ...`, new `scene/gate_check.py`: a
line-by-line Python port of `safety_gate.cpp` (max_lead; box pull-back within the 10 mm margin, refused beyond; cone on
the flange z axis, pulled back to 60, refused beyond 62; the per-request leaky bucket for v_max 0.25 m/s and w_max 1.0
rad/s with 10 ms slack), one request per 0.1 s with exec_h 12 waypoints at 50 ms, start = the anchor; IK along the
path at 10 Hz, warm-started from q_start; `--no-floor` puts the box floor at -1 m to separate the carried-into-the-table
artifact). Results `sim_evals/_start_pose_sweep/gate/gate_*.json`. PICK_TEST1 (30) / PICK_DOUBLE pick1-double-v2 (62):

| leftx55_t48_h16 (the chosen pose) | PICK_TEST1 | PICK_DOUBLE |
|---|---|---|
| max_lead | never | never |
| box refused (first refusal) | 17 (approach 14 on the floor, retreat 3 on x) | 36 (approach 29, retreat 5, lift 1) |
| box refused without the floor artifact | 3, retreat, x < 0.19 (min x 0.159) | 14 (retreat 12, lift 1; min x 0.142) |
| cone pulled back / refused | 0 / 0 (max tilt 59.5 deg) | 3 / 1 (lift; max 65.2 deg) |
| cone_max for 28/30 (58/62) to pass | <= 54.0 already passes: 60 is enough | 57.9 (no pull-back), 55.9 (no refusal) |
| v_max 0.25: demos shortened | 30 (all in the approach; also lift 19, retreat 23) | 62 |
| v for 28/30 with no shortening | 1.21 m/s (worst 50 ms step: 1.27) | 1.22 |
| distance lost by the grasp at 0.25 (never made up) | median 2.6 cm, max 8.0, 20 > 2 cm | median 3.8 cm, 50 > 2 cm |
| v so that the loss is <= 1 cm for 28/30 | 1.05 m/s | 1.05 |
| w_max 1.0 | never limiting for 28/30 (needs 0.58 rad/s) | worst 1.08 |
| IK along the path | all reachable, min margin 0.041 rad (ep 28, joint 4), p10 0.21 | 8 with unreachable samples (day 2, eps 35-49, joint 4), p10 0.006 |

The 22 cm alternative: the cone, speed and w rows are the same (the relative motion does not change); box refused 9/30
(6 approach floor, 3 retreat) / 21/62, without the floor the same 3 / 14; IK all 30 reachable (min 0.085). The layout:
at 48 deg / 16 cm with the eraser 0.45 / 0.50 / 0.55 m out, box refusals without the floor 21 / 10 / 3 of 30, IK
unreachable demos 6 / 1 / 0 of 30: 0.55 is the best of the three on both. **For the real PC:** the start pose is never
refused; during rollouts the binding rule is v_max (the demos are 4-5x faster than 0.25 in short bursts; 0.5 does not
fix it, G6); the cone 60 and the joint limits are fine for PICK_TEST1-like motions; the box x 0.20 clips 3/30 retreats
(an x min of 0.15 would pass all of them); the floor -0.005 is needed (0.05 would refuse every grasp).

### The chosen pose (sim default and recommendation for the real arm)

`leftx55_t48_h16`, written with `scene start-pose --q` (`stations/start_pose.yaml`, the old stand-ins in its source):

```
q_start_rad: [-0.5928536, 0.5219462, -0.0897168, -1.5930073, 0.6595725, 1.5068439, -1.5301472]
```

**Real PC handoff** (checked by `start_sweep handoff`; the user's real station: box [0.20, -0.40, -0.005]..[0.80, 0.40,
0.70], F_T_TCP 235.65 mm, cone 60, v_max 0.25, at_pose_tol 0.05, homing 0.3 rad/s, timeout 30 s):

| | |
|---|---|
| TCP (fingertip) in base | (0.531, -0.159, 0.160) m: 16.0 cm above the table (table = base plane, z 0) |
| fingertip direction | (+0.00, +0.74, -0.67): tilted 48.0 deg from straight down, pointing to the robot's left |
| expected `pc_probe --start-pose` | `q_start_rad: [-0.5929, 0.5219, -0.0897, -1.5930, 0.6596, 1.5068, -1.5301]`, `z 0.160 m`, `(+0.00, +0.74, -0.67): tilted 48.0 deg from straight down, pointing to the robot's left`, `80.0 mm` (with the old 150 mm TCP: z 0.217) |
| sim layout it assumes | eraser centre (demos' median) at O (0.55, 0.00): 15.9 cm ahead of the fingertips (+y), 1.9 cm to their right (+x); region 9-22 cm ahead, 0.3-3.5 cm right, yaw -7..+1 deg |
| margins | joints 1.24 rad (joint 2); box 0.165 m (z); cone 12.0 deg; p move 14.5 s of 30 s (joint 7 turns 2.32 rad) |
| p path from q_home | no contact; >= 14.7 cm to the table (left finger, at the end); >= 24.0 cm to the robot's own links |
| upstream config @ ea9f7fb | also passes the old box [0.25, -0.40, 0.05] at the start pose; but that box's z 0.05 faults every grasp of the 12 mm eraser with the fingertip TCP on a z 0 table: keep the user's -0.005 (and x 0.20 for the retreats) |
| alternative | the same 22 cm up: `[-0.5926811, 0.4637061, -0.1004182, -1.5344559, 0.6613767, 1.4082918, -1.4666293]`, z 0.220 (fewer carried demos into the table, 6/30), joint margin 1.30 |

**When hand-guiding the real arm:** aim for 45-50 deg tilt, 15-20 cm fingertip height, fingertips pointing to the
robot's left (+y) with the eraser about 0.55 m out from the base axis, i.e. the fingertip point at about (0.53, -0.16).
Outside that: below ~40 deg the IMX519 does not see the near eraser positions; above ~58 deg the cone (60) has no
room for the demos' tilting; pointing away from the base needs the eraser >= 0.6 m out and leaves the wrist near its
limits (the demos' grasps unreachable or at 0.001 rad); pointing to the robot's right runs joint 7 into its limit;
the stand-in's layout (eraser 0.45 m out) sends the demos' retreat through the robot-side box face x 0.20.

**Check on the real arm:** (1) `h`, then `p` slowly with a hand on the user stop (14.5 s; the hand rolls about 133 deg);
(2) `pc_probe --seconds 3 --start-pose`: z 0.160 +- 5 mm, tilt 48.0 +- 1 deg, "pointing to the robot's left", 80 mm,
joints within 0.05 rad; (3) the eraser ~16 cm ahead / 2 cm right of the fingertips; (4) save the four policy images
(`policy_deploy.frames.probe --save`) and compare them with a training first frame (episode_000152: the eraser
centred, about a third up from the fingertips in the IMX219); (5) then `c`. **Sim success is not real success:** in
this sim no checkpoint succeeds yet, because of the IMX219 visual gap above; the real policy has to be judged on the
arm.

### Fix log (P3/P4)

| # | Where | What was wrong -> fix |
|---|---|---|
| 1 | `stations/sim_rig.yaml`, `model/build.py`, `control_sim*.yaml`, `ui/live.py` | the layout (base on the table, z 0): table, no pedestal, box floor -0.005 |
| 2 | `run.sh` | the sim PC can die after its scene hello (C2 bind refused while another program held :5620); run.sh waited 30 s and said "no pi_stream packet" -> wait for `ready: C2 on` (print the FATAL line), retry the sim start (`SIM_DEPLOY_START_TRIES`, `SIM_DEPLOY_START_WAIT_S`, default 3 x 3 s); a busy port is retried too |
| 3 | `eval/driver.py` | no TCP path in the results (G6 needs it) -> `trace.json` gets `tcp_O_xyz_wxyz` (the 20 Hz ground truth the driver already reads) |
| 4 | `eval/driver.py`, `run.sh` | one start pose file only -> `--start-pose FILE` (both; default `stations/start_pose.yaml`) |
| 5 | `sim_pc/merge_station.py`, `run.sh` | no way to add sim-only settings for one run -> optional OVERLAY files (`SIM_DEPLOY_SIM_PC_OVERLAY`), refused if they set `q_start_rad`; `stations/homing_fast_sweep.yaml` (sweep), `stations/vmax05_g6.yaml` (G6) |
| 6 | `scene/README.md` | "all 30 replays grasp within 2 cm from the start pose": only the median is; 7 are (catalog `grasp_ok`) |
| 7 | new `scene/start_sweep.py` | candidates, real-PC checks, p path clearance, the sweep runner (one sim per pose), the summary |
| 8 | new `scene/gate_check.py` | the demos carried to a start pose through a Python port of the real gate + IK along the path |

Not changed: the copied `deployment/` (nothing in it blocked P3/P4).

## W0: re-sync with rnoro/final-deployment f632576 (2026-11-20/21)

**Copy.** `deployment/` = f632576's `deployment/` + 19 sim-only files (`diff -rq` against `git archive f632576
deployment`, excluding `build-sim/`, `*.egg-info`, `__pycache__`); every other file is upstream's bytes, so the
09:07 snapshot (P1) and D7's `frames.yaml` are no longer differences (upstream has both). The 19: E4's `--sim`
(`CMakeLists.txt`, `.gitignore`, `command_line.{h,cpp}`, `app.cpp`), P1's `Log::setTap` (`log.{h,cpp}`), E1's
`fake_pi` (`launcher.yaml`, `frames/{app,settings}.py`, `launcher/{launcher,processes}.py`) and kind `replay`
(`use_policy.py`, `policy.yaml`, `policy_factory.py`, `replay_policy.py`, `self_check.py`, `policy/settings.py`,
`STRUCTURE.md`). Each has `copied from rnoro/final-deployment:<path> @ f632576` + `changes:`. New in W0: the replay's
widths go through `policy.gripper_map.to_robot` like `DiffusionPolicy`'s (else every recorded width 0.051-0.090
clips to 0.040 and a replay never closes). `0c5903a` (TRIALS.md, `frames/trial_video.py`) is not taken.
**20th file (2026-11-22, user decision, not yet upstream):** `policy/base_policy.py` `make_chunk` orthonormalizes
the model's rot6d (Gram-Schmidt via the manager's own `pose_math.matrix_from_rot6d` = behavior_prompting/UMI
`rot6d_to_mat`) before the C4 check; before it, a raw row > ~5.7° off perpendicular (`rot6d_dot_max` 0.1) was a
PolicyFatal although the manager would have Gram-Schmidted it anyway. Robot motion is unchanged for every chunk the
raw check accepted; broken output (zero / parallel / non-finite) is still refused. Handoff for the real branch, with
the sim results: `sim_deploy/ports/rot6d_gram_schmidt.md`. Until it is ported, sim and real differ here.
Report: `sim_evals/_pipeline/W0/G1/G1_report.md`.

**Box and stations.** The sim stations no longer override the box: config.yaml's [0.20, -0.40, -0.005]..[0.80,
0.40, 0.70] is both the sim PC gate's and the laptop clamp's (`manager/settings.py` reads the copy's config.yaml).
`stations/policy_sim.yaml` stays the laptop station (127.0.0.1, `fake_pc`/`fake_pi` false, bf16 / 8 / `exec_h` 12 as
upstream's); upstream's real addresses in `link.yaml` are unused under `--sim --local` / `--mock`. Start pose:
`scene start-pose --from-git f632576` = the same 7 numbers (no-op).

**Gripper.** Robot side = upstream: stroke 0.040 (`sim_rig.yaml finger_max_gap_m`, finger joints 0..0.020 each;
`control_sim*.yaml sim.gripper.max_gap_m` 0.04), `width_offset_m` 0, the laptop's `gripper_map` (training
0.051/0.090 <-> robot 0.030/0.040), and `sim.gripper.grasp_closes_fully: true` (a grasp closes until the fingers
stall, as the Franka Hand's `grasp()` with nothing between them closes fully). The 51 mm eraser cannot fit in a
40 mm gap (flat parallel pads), so the sim's eraser is **0.030 m** across the grasp (`scene.eraser.size_m [0.127,
0.030, 0.012]`) = upstream's `robot_closed_m`. Offline (`sim_evals/_pipeline/W0/gripper/grasp_check.py`, arm held
by a PD at home, eraser hanging between the tips): grasp from 0.040 -> holds at a 22.0 mm gap, 20.0 N per pad,
0.5 mm slip in 5 s; with a 0.0308 target instead (the mapped training hold 0.054) 31.6 mm, 7.8 N per pad, 0.006 mm.
C2 then reads 0.022, `to_model` clamps it to 0.030 -> 0.051 (closed). W2's look stations keep 51 mm for images.
Model sha `96822113…` (was `166a0649…`).

**Smoke (2026-11-21, real limits).** `check --mock` against the running sim: all PASS
(`sim_evals/_pipeline/W0/G1/check_mock.txt`). `hold_still` 20 s from the start pose: max TCP drift 0.071 mm (P3
0.070 by the same measure), gate 4928 accepted, 0 rejected (`_baseline/hold_still/2026-11-21_0323_w0_g5_hold`; a
first try at 03:19 ended after 5.4 s when frames missed a 0.1 s health reply with other agents loading the CPU, load
13). Replay PICK_TEST1 pick1-v1 #19: success (grasp 2.2 s, lift 16.5 cm), 47 waypoints shortened, 1 chunk clamped at
z -0.003, no fault; but the fingers closed to a 1.9 mm gap: the eraser was pinched at its end, not across its width
(`_replay/PICK_TEST1/pick1-v1_ep019/2026-11-21_0327_w0_replay19`). Load: run_006 (BP, `PromptableDiffusionPolicy`)
and PICK_DOUBLE run_002 (plain) build, warm up and infer with the map (bf16, 8 steps; chunk widths 0.030-0.040;
`sim_evals/_pipeline/W0/load_check/`).

## Eval split: record, score, batch, real import (2026-11-21/22)

Record format and scoring rules: [evaltool/SCHEMA.md](evaltool/SCHEMA.md) (`sim_deploy.rollout/v2`, before 2026-11-22 `sim_deploy.episode/v1`); not repeated here.

- **Recorder (driver).** `eval/driver.py` runs rollouts; its live judge only decides when to stop one.
  `eval/gt_recorder.py` writes, per eval, `eval.yaml` `record_schema` / `launcher` (env `SIM_DEPLOY_LAUNCHER`) /
  `batch_id` (env `SIM_DEPLOY_BATCH_ID`) / `protocol` + `protocol_sha`, and per rollout `gt.jsonl` (20 Hz ground
  truth from the scene REP :5651 and the state export :5650) and `rollout.json` (placement, stop reason, clamps, C2
  deltas, timing, load). `--only-rollouts LIST` runs chosen indices with their full-run placements.
- **Eval tool (reader).** `evaltool/` reads the record and never writes into it except `score.json` (`score`) and
  `frames.jpg` (`frames`); `report` writes only its `--out` folder. Rule values once in `evaltool/constants.py`, copied
  into each `score.json`. Folders without `gt.jsonl` are scored from `trace.json` with the metrics it allows.
- **Batch.** `eval/batch.py`: a plan YAML (`eval/batches/`) of checkpoints x looks, one `run.sh` at a time under
  `sim_evals/.gpu.lock`, gated on the UI being idle, free sim ports and load < `load_max`. Append-only
  `sim_evals/_batches/<id>/manifest.jsonl` makes `--resume` replay it. After each run `evaltool score`; invalid
  rollouts are re-run (`invalid_retries`) and the report counts each index once from the highest retry. At the end
  `evaltool report --batch <id>` into `sim_evals/_reports/<id>/`.
- **Real import.** `evaltool/import_real.py` turns real-arm probe trials (`<run>/deploy/*_probe_live/` + the
  `policy_deploy` session) into eval-like folders under `sim_evals/_real/` with `source: real`; outcome comes only from
  a confirmed `labels.yaml` (`labels_template.yaml`), so sim and real share one report.
- **Rollout, not episode (2026-11-22, user decision).** An eval trial is a *rollout* in every name sim_deploy owns
  (CLI `--rollouts` / `--only-rollouts`, `rollouts/NNN/`, `rollouts.jsonl`, `rollout.json`, `session_rollout.json`,
  keys, UI text); "episode" stays for training demonstrations and for policy_deploy's own names, which are parsed
  unchanged (`manager/episodes.jsonl`, `episode_id`, "episode N started"). Record v2, eval.yaml / score.json v2
  (SCHEMA.md "Versions"); the existing `sim_evals/` was converted once by `evaltool migrate-rollouts`.
- **Stop path (2026-11-22).** pexpect gives policy_deploy's launcher a session of its own and the launcher starts
  frames and policy with `start_new_session=True`, so none of them is in run.sh's process group: a group SIGINT
  (batch stop, Ctrl-C) never reaches them, only the launcher's own stop ends frames / policy. pexpect's
  `terminate()` (SIGHUP first) or the pty closing (driver killed) ended the launcher without that stop and left frames
  running on :5610. Now: `pd_session.quit` sends `q`, then SIGINT to the launcher's group (its Ctrl-C path), then
  SIGKILL, and stops whatever was under the launcher and still runs; run.sh exports `SIM_DEPLOY_RUN_TAG` and its EXIT
  trap stops every process still carrying it; batch.py sets one tag per run.sh call and sweeps it after the call
  (covers a run.sh killed after the 60 s grace) (`eval/procs.py`).

## Start set (2026-11-22, W3; replaces the per-dataset demo starts of the same day)

User decisions: W3 evals run only the `robot` look, from the real start pose, with **one fixed list of eraser starts
for every checkpoint**, so a sim rate predicts the real cell and checkpoints are compared on the same placements.
Code `eval/start_set.py` (+ `eval/split_probe.py` in the torch env), file `eval/start_sets/w3.yaml` (committed),
driver `--starts set[:FILE]`, batch plan `starts: set`. How to use: `eval/README.md` "Start set"; record fields:
`evaltool/SCHEMA.md`. The first design (one file of ten training demos per dataset, `--starts demos`) was replaced
before any real run; its code and files are kept in `sim_evals/_logs/retired_demo_starts_2026-11-22/`.

- **Rollouts 0-9: the shared demos.** All 9 training datasets start with the same 10 demos (the whole of
  pick1-ep10-v1: PICK_TEST1 episodes 138-147 of 2026-11-18); `starts build` checks names, row counts, poses and widths
  of every dataset's first 10 against pick1-ep10-v1 (2026-11-22: identical in 9/9). Every checkpoint held exactly one
  of them out for validation (its split read from the checkpoint, `_extra_training_split_info`, CPU
  `torch.load(mmap=True)`, cross-checked with behavior_prompting's `get_val_mask` recomputed on the local zarr; all 14
  agree): run_006, run_022, run_016, run_024, PICK_DOUBLE run_003: demo 2; run_012, run_028, run_014, run_031: 1;
  run_020, run_030: 0; PICK_DOUBLE run_002: 5, run_005: 4, run_004: 3. That rollout is recorded `held_out: true`, and
  the report gives the 9 trained demos and the 1 held-out demo apart (a val demo is a generalisation test).
- **Rollouts 10-17: the real-trial poses.** The eraser poses fitted in W2 to the IMX219 / IMX519 images of the 8
  real-arm trials of 2026-11-20 (`sim_evals/_pipeline/W2/sets/robot_trials_fit.json`, 0.6-1.0 px rms; trial
  run_022 trial4 had another object and is not fitted), with the arm at q_start as on the real arm. The real object
  was the 127 x 51 x 12 mm eraser; the sim lays its own 30 mm wide eraser (user decision: the real arm will use a
  <= 38 mm object) flat at the fitted centre x, y and yaw. The values are copied into the file with the fit's sha256,
  so a re-run of W2 does not move them. These lie 3-8 cm ahead of the start fingertip; the demo starts 12-22 cm.
- **Demo placement (assumptions, all reported per start).** (1) The transfer is the replay placement's:
  `O_T_grasp = O_T_TCP_start . inv(P_s) . P_g` in the TCP frame, so the demo's start-to-grasp motion is kept relative
  to the camera; the sim start (16 cm above the table, 48 deg tilt) is not the demo's, so the carried-over grasp lands
  above or below the table (`height_error_m`, -12.8 to +0.1 cm over the 10) and the eraser is laid on the table
  straight below / above it. (2) The eraser centre is half its resting height (6 mm) behind the grasp fingertip point
  along the approach axis. (3) Its width lies along the grasp's closing axis (TCP x) projected on the table. (4) The
  recorded eef pose is the same physical point and axes as the sim TCP. (5) Flat on the table, no roll / pitch.
- **Checks** (required; a failing start is reported, not substituted): whole eraser on the table, centre and
  footprint in the PC's workspace box, all 8 corners in the IMX219 pinhole at the start pose (no occlusion test).
  Information: IMX519 view, ahead / left / yaw_rel relative to the start. 2026-11-22: all 18 pass every check.
- **Reproducibility.** The driver recomputes each start with its own model and start pose (0.5 mm / 0.1 deg) and
  refuses a file that no longer holds, or a checkpoint whose split the file was not built with. eval.yaml freezes the
  file (`scene.start_set`, with this checkpoint's `held_out_rollouts`); `protocol.seed_rule` hashes the sources' sha256,
  q_start and every rollout's start and pose (not the held-out demo), so all 14 checkpoints share one `protocol_sha`.
- **Known gaps (reported, not fixed).** The sim's control cycle is ~200 ms (p50; policy ~170 ms on the laptop GPU)
  against ~110-130 ms on the real arm (2026-11-20 trials, `sim_evals/_real` steps): the policy replans less often in
  the sim (user decision: report it; the W3 report has a timing table). The object is 30 mm wide, not the trials' 51 mm.
  For the 5 prompting checkpoints the prompt is `task_name: PICK_TEST1, prompt_index: 0` of their prompt dataset
  (`policy_deploy use`), i.e. index 0 of the checkpoint's own train split (use_policy.py + the training split mask):
  shared demo 0 = rollout 0 for run_006 / 012 / 014 / 016; run_020 holds demo 0 out, so its prompt is demo 1 =
  rollout 1. That rollout of a BP checkpoint starts where its own prompt grasped.
- **Success also needs the eraser clear of the table** (score v4, user decision 2026-11-22): in every row of the 1 s
  hold the eraser box's lowest corner >= table top + 0.01 m, besides the centre >= settled + 0.05 m; one function
  (`evaltool/constants.py clear_of_table`) used by `evaltool score` and the driver's early stop on the same rows.
- **GPU-fault invalidation (B1).** A kernel `NVRM: Xid` found by the batch after a run.sh call writes
  `<eval>/gpu_fault.json` (first Xid, local time): the start of a fault window `[xid_time, cleared)`. When a batch
  starts (`--resume` after the reboot) and its GPU check passes, it writes `cleared` into every open gpu_fault.json the
  batch recorded: the start of the boot after the Xid (`/proc/stat btime`), or the check time if there was no reboot
  (naive local time, like `wall_start`). `evaltool score` marks every rollout whose `[wall_start, wall_end]` overlaps a
  window invalid (`gpu_fault`; open-ended until `cleared` is written; no `wall_end`: overlaps when it started before
  the window end), and the batch's resume re-runs them. So a rollout running at the Xid is invalid too, and the
  rollouts run after the reboot in the same eval folder stay valid. A later Xid after a cleared window opens a new one
  (the old one is kept under `earlier`).
- **Pieces and GPU cool-downs (2026-11-22, user decision).** The laptop RTX 3050 Ti crashed twice the same day (kernel
  `NVRM: Xid 62`, firmware), both times deep into a batch: one run.sh call per checkpoint kept policy inference + EGL
  camera rendering going ~20+ min without a break at 70-80 C. The fix only slows down *between* rollouts, never inside
  one (a rollout's timing must stay real-time): the driver's `--max-new-rollouts K` runs at most K not-yet-run
  rollouts and ends (exit 0, eval incomplete; not in `eval.yaml`'s protocol, so `protocol_sha` is unchanged); run.sh
  counts K over its own restarts after a sim PC fault (exit 3). The batch (`piece_rollouts`, default 6) calls run.sh
  again with `--resume <eval folder>` until the run is complete; these planned continuations do not count toward
  `MAX_RESUMES`, only "short" calls (incomplete with fewer new rollouts than the piece). Before every call, inside the
  GPU lock: the GPU check, then wait for `temperature.gpu < gpu_cool_c` (65 C) and `gpu_pause_s` (60 s) since the
  previous call, at most `gpu_cool_max_s` (900 s, then WARN and go ahead); manifest `gpu_cool` per call (start / end
  temperature, waited s). The GPU check also runs after every call, so a fault stops the batch after the piece it hit.
  Results: the same eval folder, the same rollout indices and starts as one call; each piece adds a sim start (~90 s)
  plus the cool-down. The defaults live in batch.py, so a batch frozen before this resumes with them.
  Review fixes (same day): a run whose folder already holds every rollout while its last manifest event says
  incomplete (Ctrl-C on the final piece's last rollout, or run.sh finishing after the batch died) is recorded complete
  (`found_complete`) instead of calling run.sh with nothing left, which would start a fresh empty folder. And the driver
  now treats "key h refused: mode is fault" (the sim PC latched a fault between rollouts, e.g. physics behind wall time
  under load: `communication_constraints_violation`) like a fault at start-up: that rollout and the rest of the call
  stay `not_run` and it exits 3, so run.sh restarts the sim and resumes them (before: each was used up as
  `home_failed`, as run_012's rollouts 15-17 in `w3_2026-11-22_1834`).
