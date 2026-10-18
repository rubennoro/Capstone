# Track A design: sim core (MuJoCo model, kinematics, Stage 1, backend, Franka Hand, scene, render, Stage 1b)

This is the design produced in planning (2026-10-18). Resolved against PLAN.md:
- the gripper type is `franka_hand` (not `bridge_hand`);
- the hand_server port is 5566;
- the width map default is b = 0.008;
- the success rule is 0.05 m / 2.0 s / 60 s / slip 0.002;
- all visual meshes for both arms are vendored.

## Findings that shape the design

1. **`PandaPyBackend.error` is never assigned** (backends.py; only set to None in `__init__`/`recover`). Real reflexes would never reach the bridge → controller fault → task hold chain. Fix (A4): derive `error` from `robot_mode == Reflex` / `last_motion_errors` / `current_errors` and from a dead panda-py controller. Test with a fake `panda_py` module, and add a line to panda_py_checklist.md.
2. **The EE convention is unmodelled.** `O_T_EE = O_T_F · F_T_NE · NE_T_EE`. With Desk EE = "Franka Hand", F_T_NE is a 0.1034 m z offset rotated −45° about z. The bridge assumes O_T_EE = flange (`ee_is_tcp: false`), and its safety box/cone apply to that frame. The robot config carries `desk_end_effector`, `F_T_NE` and `NE_T_EE`; `EE_T_TCP = inv(F_T_NE·NE_T_EE)·F_T_TCP`. This becomes a hardware-day check.
3. **Camera clearance.** The camera origin in TCP coordinates is ≈ (0, −0.117, −0.1875). With a fingertip TCP about 0.13–0.15 m from the flange, the phone lens sits about 5 cm behind the flange plane and 12 cm off-axis, next to links 6/7. A1 tests and reports the clearance.
4. **Import path.** The launcher sets `PYTHONPATH = controlSW/franka_deploy` only, so `controlSW/franka_sim` must be added (B6 does this in the launcher; A's tests use their own conftest).
5. **Menagerie actuators are position servos** (kp 4500/3500/2000). Replace them with torque motors. FR3 has `frictionloss` 0.25–1.14 Nm, scaled to 0 by default (FCI torque runs over internal friction compensation). FR3 has no hand: attach the Panda `hand.xml` `hand` body at `attachment_site` (z 0.107 in fr3_link7) with quat (0.9238795, 0, 0, −0.3826834).
6. **Assets.** Panda visual OBJs are 34.2 MB and FR3 visuals 57.3 MB; collision STLs are about 0.1 MB per robot. The user chose to vendor everything.
7. **Success rule.** `annotate.py` uses 0.03 m; hardware_day §10 says 0.05 m / 2 s / 60 s. The criteria use §10, and annotate reads the criteria (B2 owns annotate).
8. **Joint limits differ.** Panda q6 ∈ [−0.0175, 3.7525]; FR3 q6 ∈ [0.4398, 4.6216], q2 ±1.8361. Run Stage 1 on both.
9. **gonogo's loop p99.9 gate is meaningless for a Python sim.** Use only the tracking outputs (`lag_ms`, `steady_err_mm_*`); add `--no-loop-gate`.

## Package layout

```
controlSW/franka_sim/
  README.md  pyproject.toml (optional; tests use conftest sys.path)
  franka_sim/
    __init__.py            # never imports mujoco at import time
    cli.py                 # register(name, add_parser_fn, main_fn); __main__.py dispatches
    assets/franka_emika_panda/ (panda.xml, hand.xml, assets/*, LICENSE, README.md)
    assets/franka_fr3/        (fr3.xml, assets/*, LICENSE)
    assets/SOURCE.md          # URL, pinned commit, date, sha256 per original file
    configs/station_fixture.yaml   # temporary until B1 (same group layout as controlSW/configs/station)
    model.py kinematics.py frames.py simcore.py hand_sim.py backend.py scene.py ground_truth.py
    stage1_replay.py campaign.py render.py camera_check.py viz.py
    tools/vendor_menagerie.py tools/check_assets.py
  tests/ conftest.py (inserts controlSW/franka_sim and controlSW/franka_deploy; importorskip("mujoco") where needed)
```

Dependency direction: franka_sim imports franka_deploy (Backend base, `SafetyFilter`, `cycle_hist`, `replay_server.load_episode`, `eval.metrics.phases`, `station`). franka_deploy imports franka_sim only inside `make_backend` when `type == "mujoco"`.

## A0: environment and assets

- mujoco 3.3.0 is already installed in env_realenv by the orchestrator.
- `tools/vendor_menagerie.py --commit <sha|main> --out <scratch>`:
  - sparse-clones `google-deepmind/mujoco_menagerie` into a fresh, empty scratch directory (untrusted download: read the files, don't execute anything from it);
  - copies `franka_emika_panda/` and `franka_fr3/` (XML, all assets, LICENSE, README) into `assets/`;
  - writes SOURCE.md with the full commit sha and the sha256 of each file.
- `tools/check_assets.py` verifies against SOURCE.md.
- `cli.py` registry and `__main__.py` with `check-assets`.
- Tests:
  - both XMLs compile;
  - sha256 matches;
  - `import franka_sim` works in env_bp, which has no mujoco.

## A1: model.py

```python
@dataclass
class SimModel:
    model; spec; names: dict   # canonical -> MJCF names: j1..j7, finger_l/r, sites flange/ee/tcp, cams cam_main/cam_uw,
                               # bodies hand/object/table, tendon, actuator ids
    F_T_TCP: np.ndarray; EE_T_TCP: np.ndarray; limits: dict
def build_model(station, sim=None, with_scene=True, visual_mode="full") -> SimModel
def export_frames(sm) -> dict  # F_T_TCP, EE_T_TCP, F_T_cam_main, F_T_cam_uw, tag poses
```

Build steps (MjSpec, mujoco 3.3.0):
1. Load the robot. For FR3, attach the hand (finding 5).
2. Delete the stock actuators. Add `motor` actuators on joints 1–7 with ctrlrange ±[87,87,87,87,12,12,12], and a motor on the finger tendon `split`. Scale `frictionloss` by `joint_frictionloss_scale` (default 0).
3. Add site `flange` (link 7, z 0.107, no rotation) and site `ee` = flange·F_T_NE·NE_T_EE.
4. Disable the stock fingertip pad collision. Add adapter boxes (`finger.extension_m`, `pad_size_m`) with the inner pad face `b/2` further out, so pad separation = gap + b. Use `condim=4`, configured friction, and 16 mm tag patches.
5. Add site `tcp` at the pad-centre midpoint at the fingertip. `F_T_TCP` is derived (or the station's calibrated value, with the deviation reported).
6. `cam_main` at TCP·inv(C_T_TCP) with the OpenCV → MuJoCo flip `R_mj = R_cv·diag(1,−1,−1)`, intrinsics from the stream K (3.x camera `resolution`/`focalpixel`/`principalpixel`); `cam_uw` = cam_main·main_T_uw.
7. Phone or rig housing box with the main lens at `cam_main`; cradle boxes; masses.
8. Scene: table and eraser (`scene.py`); `implicitfast`, elliptic cone, `impratio` 10, all configurable.

Tests (both robots):
- C_T_TCP to 1e-9;
- F_T_TCP;
- hand-in-flange identical for Panda and FR3;
- pad separation = gap + b at 0, 0.04, 0.08;
- MJCF joint ranges within the datasheet limits;
- tooling mass sum;
- no contacts at the home keyframe;
- **minimum distance from phone/cradle to links 5–7 over a q6 × q7 sweep ≥ `min_clearance_m` (0.01), else flagged with the worst pose printed**;
- EGL render smoke test.

## A2: kinematics.py, frames.py

```python
class Kin:
    def __init__(self, sm, frame="flange"); def fk(self, q); def jac(self, q)   # 6x7 world [v; w]
    def ik(self, T, q0, *, q_rest=None, tol_p=1e-4, tol_r=1e-3, iters=100, damping=1e-2, limit_margin=0.02, w_null=0.1)
    def ik_multistart(self, T, seeds, **kw); def contacts(self, q, margin=0.0, groups=("self","table","tooling"))
    def min_distance(self, q, pairs); def manipulability(self, q)
def check_traj(T_seq, station, q_seed) -> dict
frames.py (pure numpy): arkit_to_zup(T), placement_T(gx, gy, yaw, z), anchor_placement(T_grasp_arkit, g_xyz, yaw), rot6d helpers
```

Tests:
- FK against Franka modified DH (a = [0,0,0,.0825,−.0825,0,.088,0], d = [.333,0,.316,0,.384,0,0,.107], α = [0,−π/2,π/2,π/2,−π/2,π/2,π/2,0]), 200 random q, 1e-6 m, both arms;
- IK ≥ 99 % of 500 reachable poses, within limits;
- singular poses give `ok=False`;
- frames gravity consistent with `T_YUP_ZUP`;
- IK ≤ 1.5 ms per warm-started frame.

## A3: Stage 1 (stage1_replay.py)

CLI: `python -m franka_sim stage1 --zarr Z --station <ids or deploy profile> --robot panda|fr3 [--grid run/sim/stage1.yaml] --out DIR [--jobs N] [--demos ...]`

1. `load_episode` (low-dim) for each demo. The grasp index comes from `metrics.phases(width, height_y, t)`, and the grasp point is the TCP at that index.
2. Placement grid (gx, gy, yaw). Anchor each demo's grasp TCP at (gx, gy, table_top + grasp_tcp_height_m). Gravity is kept and only yaw is free.
3. base_T_tcp → base_T_EE = base_T_tcp·inv(EE_T_TCP).
4. IK, warm-started; frame 0 uses multistart with q_rest = the ready posture. A frame fails if the residual is > 1 mm or > 0.5°.
5. Metrics per (demo, placement):
   - reach fraction;
   - minimum joint margin;
   - minimum manipulability;
   - peak |dq|/max, |ddq|/max, jerk/max (Savitzky–Golay at 60 Hz; jerk flagged as noise-dominated, also computed after retiming);
   - collisions and minimum distance (self, table, tooling↔arm);
   - safety box violations and cone angle;
   - **clamp statistics from the real `SafetyFilter(SafetyConfig.from_dict(bridge.safety))` at 200 Hz**: clamp fraction, maximum backlog, slowdown;
   - Cartesian speed against the datasheet;
   - width map: gap = (w − b)/a outside [0, 0.08];
   - start-pose feasibility.
6. Coarse-to-fine search: coarse at 10 Hz on a 7×7×8-yaw grid with multiprocessing, then 60 Hz on the top 5. Score = (number of fully feasible demos, then worst joint margin, then clamp fraction).
7. Outputs: `stage1.json` (placements, per-demo metrics, recommendation, start q/pose per demo, recommended home), `summary.txt`, optional plots.

Tests:
- synthetic straight line → 100 % reach, 0 clamps;
- an over-speed segment → clamps > 0 and slowdown > 1 (checked by hand);
- an out-of-reach demo is flagged;
- the result is deterministic.

Acceptance: 45 demos × 2 arms in under 30 min with 4 jobs.

## A4: SimCore + MuJoCoBackend

```python
class SimCore:   # no threads, deterministic
    def __init__(self, sm, cfg, q_init); def set_target(self, O_T_EE); def start(self); def stop(self); def recover(self)
    def begin_joint_move(self, q_goal, speed) -> float; def step(self, n=1); def state(self) -> dict; def inject_reflex(self, msg)
class MuJoCoBackend(Backend):
    @classmethod
    def from_config(cls, backend_cfg, home_O_T_EE); extra_topics = (...); def handle_extra(self, topic, data)
```

**Control law (verify against the pinned panda-py source):**
- Target filter per ms: `p_d ← fc·p_t + (1−fc)·p_d`, with slerp for orientation.
- Errors: position e = p − p_d. Orientation: sign-flip the quaternion if needed, `e_q = q⁻¹·q_d`, `e_r = −R·vec(e_q)`.
- Torques: τ_task = Jᵀ(−K e − D J dq) with D = 2ζ√K (diagonal); τ_ns = (I − Jᵀ J̄ᵀ)(k_ns(q_ns − q) − 2√k_ns dq).
- Limits: rate limit ≤ 1 Nm/ms; saturation ±τ_max.
- Robot-side gravity: `qfrc_bias` minus Coriolis, plus a payload-mismatch term J_comᵀ(m_cfg − m_true)g.
- Parameters come from `backend.panda_py` (impedance, damping_ratio, nullspace_stiffness, filter_coeff, collision) and can be overridden by `backend.mujoco`.

**Modes:**
- `idle`/`stopped`: joint PD hold.
- `running`: impedance.
- `reflex`: hold; targets ignored until `recover()`.
- `move_joints`: quintic trajectory that blocks the REP, as on hardware.

**Reflexes:**
- |τ_ext| or |F_ext| above the thresholds. τ_ext = arm-DOF `qfrc_constraint` plus applied forces; F_ext = pinv(Jᵀ)τ_ext. Thresholds default to libfranka τ [20,20,18,18,16,14,12] and F [20,20,20,25,25,25].
- Position limits, |dq| > max, self-contact.
- Optional: torque discontinuity, Cartesian speed.

**Thread:**
- 1 kHz, `due = floor((now − t0)/dt) − k`, stepping min(due, max_catchup_steps).
- More than 50 ms behind → resync and count it.
- `get_state` returns a cached snapshot.
- `loop_stats` adds `sim_lag_ms_max`, `steps_per_wake_max`, `rt_resyncs`, `realtime_factor`.

**franka_deploy integration (additive):**
- `make_backend("mujoco")`.
- Bridge routing of `extra_topics`.
- A commented `backend.mujoco` block in `configs/bridge.yaml` (B owns the file: coordinate through the orchestrator).
- `check_config` allows `mujoco` (B owns the launcher: B6 adds it).
- gonogo: retry readiness for up to 10 s, plus `--no-loop-gate`.
- Characterization table `franka_sim/configs/characterization_baseline.json`.

**Tests:**
- hold drift < 0.5 mm over 5 s;
- 2 cm step overshoot ≤ 5 %, steady-state ≤ 1 mm;
- payload error 0.3 kg → z offset ≈ mg/K_z within 30 %;
- rate limiter respected;
- joint-limit reflex;
- table push → cartesian_reflex;
- move_joints accurate to 1e-3;
- both arms;
- real time: p99 period ≤ 2 ms, 0 resyncs on an idle laptop, get_state ≤ 0.2 ms;
- a stack test (copy of test_franka_stack Stack with mujoco, ports + offset): move + inject_reflex → controller fault;
- gonogo steps run: lag in [5, 80] ms, steady-state ≤ 2 mm.

## A5: Franka Hand

`franka_deploy/gripper/hand_logic.py` (pure):

```python
@dataclass
class WidthMap: a=1.0; b=0.008; gap_min=0.0; gap_max=0.08; def gap(self, w_train); def train(self, gap)
@dataclass
class HandCmd: kind; width; speed; force=0; eps_inner=0; eps_outer=0
@dataclass
class HandPolicyConfig: close_below_train=0.06; open_above_train=0.066; move_deadband_m=0.004; min_cmd_interval_s=0.25;
                        speed_mps=0.1; grasp_force_N=30.0; eps_inner_m=0.01; eps_outer_m=0.08
class HandCommandPolicy: def update(self, w_train, now, hand_state) -> HandCmd | None
```

- **States:**
  - OPEN, and the command drops below `close_below` → `grasp`, entering GRASPING.
  - In OPEN, a change larger than the deadband, with the minimum interval elapsed and the hand not busy → `move`.
  - In GRASPING, the command rises above `open_above` → `move`, back to OPEN.
  - Latest command wins; force ≤ 70 N.
- **Driver:** `franka_deploy/gripper/franka_hand.py` `FrankaHandDriver(GripperDriver)`.
  - Non-blocking: a background thread with its own REQ socket at 50 Hz.
  - `set_width` enqueues the latest command; `get_width` returns cached state (stale for more than 0.2 s → the last value plus a counter).
  - `status()`; `cal` built so existing `.cal` users still work.
  - `make_gripper("franka_hand")`.
- **`franka_deploy/gripper/hand_server.py`:** `LibfrankaHand` (no coverage; worker thread on `panda_py.libfranka.Gripper`) and `FakeHand`, on 5566. Checklist items: whether panda-py releases the GIL during gripper calls, and whether `stop()` works from another thread.
- **`franka_sim/hand_sim.py`:**
  - `move`: ramp at the commanded speed with a capped PD (2×70 N); blocked > 0.2 s with error > 2 mm → failure.
  - `grasp`: close until stall (|v| < 2 mm/s for 50 ms), then hold force −2F; `grasped` = within the epsilons.
  - `stop`.
  - `MuJoCoBackend.extra_topics += ("hand_cmd", "hand_state")`.
- **Tests:**
  - logic tests in both envs;
  - the driver against `hand_server --backend fake` never blocks for more than 1 ms;
  - SimCore grasp of a 45 mm box;
  - stack: the controller with `franka_hand` pointed at the mujoco bridge runs open → close → open, with the 200 Hz loop p99 within +0.2 ms.

## A6: scene, resets, ground truth

- **`scene.py`:** table; eraser as a free-joint box (`condim=4`).
- **`extra_topics += ("sim_reset", "sim_state")`.** Reset is allowed only with `--allow-sim-reset`/`allow_inject`, refused while grasping, and resets objects and the hand only.
- **State PUB** on 5570 (I-SIMSTATE).
- **Bridge tick:** if `hasattr(backend, "sim_record")`, write `{"t","kind":"sim","obj","gap","grasped","contacts","F_ext_max"}` to the jsonl at `sim_log_hz`.
- **`ground_truth.evaluate(events, criteria, t_start)`:**
  - success = lift ≥ 0.05 held ≥ 2.0 s within 60 s;
  - also grasp_t, max_lift, slip, knocked_over (tilt > 30°), arm–table contact;
  - `evaluate_episode(ep_dir, criteria)`.
- **`eval/report.py`:** additive `sim_gt`; `sim_success_rate` reported separately; human labels win.
- **Tests:**
  - synthetic streams: 5 cm held 2.1 s → success; 4.9 cm → fail; a drop → slip;
  - scripted pick-and-lift succeeds on ≥ 9/10 seeds; if contacts are unstable, use `substeps: 2`;
  - reset is refused while grasped.

## A7: render + camera check + render sender

- **`render.py`:** `Renderer(sm, station).render(qpos) -> {main, ultrawide}`, BGR 960×720 at the stream K.
- **`camera_check.py`** (`python -m franka_sim camera-check`):
  - renders at the ARKit training intrinsics (main 1920×1440 f 1406.34; UW 640×480 f 259.02, plus the second variant);
  - overlays on real frames (`sensor_rig.sim.iphone_reader.IphoneEpisode` from data/raw/2026-10-01);
  - projects the tag centres and compares them with ArUco detections on real UW frames (cv2.aruco DICT_4X4_50; iPhUMI code is protected, so it is not imported);
  - reports per-tag pixel error and the hand-body visibility mask;
  - writes PNG overlays to scratch.
- **`franka_sim/senders/render_sender.py`:**
  - subscribes to the 5570 PUB;
  - renders at render_fps;
  - JPEG-encodes at the sensor quality;
  - sends with the sensor protocol (B4's StreamClient): v1 without stamps, v2 with `t_mono` stamps;
  - writes truth lines.

## A8: Stage 1b campaign

- **Profile `sim` overlays** (the B6 profiles `sim-iphone`/`sim-rig` with `policy: demo_replay`).
- **`python -m franka_sim campaign --stage1 stage1.json --config deploy.yaml --profile sim-iphone --demos all --speed 0.5,1.0 --out DIR`.** Per demo × speed:
  1. Write an overlay: q_init = q_start[d]; object at the demo's grasp location; hand gap = gap(w0); home = start TCP; replay episode d at the given speed.
  2. Run `franka_deploy run --no-tty --script "3:c,<T>:y,<T+3>:q"`.
  3. Collect: `sim_gt`, TCP path against the demo (≤ 1 cm / 3° over 5 s), clamps, settle timeouts, reflexes, hand statistics, peak F_ext.
- **Output:** `campaign.json` with Wilson CIs (`eval.metrics.wilson`).
- **Acceptance:**
  - all 45 demos at speed 0.5 run without system faults;
  - the tracking metric holds on ≥ 90 % of runs;
  - sessions pass `validate` and `eval`.

## Risks

- **Real time in Python at 1 kHz.** Catch-up stepping and cached state; fallback rate_hz 500 with substeps 2.
- **Grasp contact stability.** Elliptic cone, impratio, condim 4, substeps.
- **Sim–real impedance gap.** Tuning knobs, fitted after hardware day.
- **F_T_NE convention.**
- **Clearance at link 7.**
- **Unverified panda-py details.**
- **Hand bandwidth.**
- **ARKit noise inflating jerk.**
- **Grasp height in ARKit coordinates.** Sweep ±1 cm.
- **Disk space.**
- **MjSpec API churn.** Pinned to 3.3.0.
- **Launcher startup cost per campaign run.**
