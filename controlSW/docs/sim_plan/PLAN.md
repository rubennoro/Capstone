# Plan: Franka simulation evaluator (iPhone + sensor rig), configurable station, agent-team execution

## 1. Context

The policy (`block_raise`, Behavior Prompting run_001) was trained on handheld iPhUMI data. Deployment target:

- **Arm and gripper:** a Franka arm (Panda or FR3, still undecided) with the stock **Franka Hand**, fitted with 3D-printed copies of the iPhUMI fingers.
- **Sensor mount:** the sensor sits in the iPhUMI phone cradle at the same pose relative to the fingertips as on the handheld gripper. The sensor is either the **iPhone 16 Pro** (existing iPhUMI app) or later the **custom sensor rig**, which reproduces the iPhone's camera poses.
- **Topology:** the sensor streams **wirelessly** to a policy host (PC or Jetson). That host runs the camera receiver, task, policy, controller and 1 kHz bridge, and is **wired directly to the Franka control box**.

Before touching the real arm, we need a simulator that answers four questions:

1. Can this Franka + Hand + mount physically do what the demos did?
2. Can it carry out what *any* policy outputs (BP, plain diffusion policy, simpler ones) through our real control code?
3. Does the whole deployment program, including the wireless sensor link, behave correctly for both sensors?
4. Are all constants in one configurable place, so that unknowns (arm model, CAD, calibration, latency) can be swapped in later?

Out of scope: proving real-world task success. The policy's input is real images, so that is decided on the arm at low speed. Renders are used only to test plumbing, safety and timing.

## 2. Locked decisions

| Topic | Decision |
|---|---|
| Arm | Panda and FR3 both modelled; choose in config (Maral/Kian on Monday) |
| Gripper | Franka Hand + printed iPhUMI fingers (shape, color, ArUco 4X4_50 16 mm IDs 0/1) |
| Width map | training_width = a·hand_gap + b, default a=1, b=+0.008 m. Training range 0.0447–0.0844 maps to gap 0.0367–0.0764, inside the 0–0.080 stroke. Fitted on hardware day with `fdcal width` |
| Mount | Reuse the iPhUMI cradle; the camera→TCP target is C_T_TCP (R_x≈15°, t=(0, 0.0644, 0.2114)) |
| Sensor | iPhone 16 Pro now; the rig reproduces the iPhone camera poses later |
| Topology | Sensor → Wi-Fi → policy host (PC or Jetson) → Ethernet → Franka control box |
| Simulator | MuJoCo 3.3.0 (same as the cluster bp env); official mujoco_menagerie Panda/FR3 + Franka Hand |
| Arm models in repo | **Everything for both arms**: XMLs, collision and visual meshes, LICENSE (Apache-2.0), SOURCE.md with pinned commit and sha256 |
| Policy boundary | The existing robotmq inference-server protocol. Any policy = a server that passes the contract test |
| Closed-loop visual realism | Out of scope; render-based closed loop is labelled "plumbing only" |
| Success rule | Eraser lifted ≥ 0.05 m, held ≥ 2.0 s, within 60 s, slip ≤ 2 mm (hardware_day.md §10). `annotate.py` is updated to read it |
| Code rules | Additive-only for protected paths (`iPhUMI/`, `behavior_prompting/`); regression guard before every commit |
| Commits | New local branch `sim-eval`; one local commit per passed wave; no push without asking |
| Agents | 2 executors + 2 reviewers, **up to 4 running at once**, with the rate-limit rules in §10 |

## 3. Architecture

```
REAL
 iPhone app (Socket.IO "Ethernet" mode, v1)  ─┐ Wi-Fi
 rig Jetson: sensor-rig stream-send (v2)     ─┘──────▶ POLICY HOST (PC | Jetson)
                                                      WirelessCamera receiver (ports 5555/5556) ─shm─▶ task ⇄ policy server
                                                      task → controller (200 Hz, F_T_TCP, franka_hand driver)
                                                      controller → bridge (1 kHz, tcp 5565) → PandaPyBackend ─Ethernet─▶ Franka box
                                                      franka_hand driver → hand_server (tcp 5566, libfranka Gripper) ─Ethernet─▶ box
SIM (laptop, same processes)
 image source: render_sender (MuJoCo renders) | iphone_replay_sender (raw demo video) | rig_replay_sender (raw rig episode)
   → LinkEmulator (latency, jitter, stalls, bandwidth, clock skew) → same WirelessCamera receiver → same task/policy/controller
   → same bridge → MuJoCoBackend (1 kHz real time; impedance; reflexes; fingers; table + eraser); hand topics served by the bridge
```

- Senders imitate the device's exact wire behaviour, so the receiver config is identical in sim and real.
- **The iPhone sends no capture stamps.** Its frames are stamped on arrival minus a measured latency.
- **The rig sends sender stamps.** Its clock offset is estimated by ping over the same socket. chrony alone does not sync `CLOCK_MONOTONIC`.

## 4. Configuration system: one home for every constant

**Root:** `controlSW/configs/`.

**Loader:** `controlSW/franka_deploy/franka_deploy/station.py`. It must not depend on mujoco or robotmq, because it runs on the Jetson.

```
controlSW/configs/
  station/robot/      panda.yaml, fr3.yaml          joint/Cartesian/torque limits, home q, impedance, collision thresholds,
                                                    desk end-effector convention (F_T_NE, NE_T_EE), MJCF path, FCI versions
  station/tooling/    franka_hand_iphumi_iphone16pro.yaml, franka_hand_iphumi_rig1.yaml
                                                    F_T_TCP, C_T_TCP, F_T_C_main, C_main_T_C_uw, finger adapter (extension, pad
                                                    offset b/2, friction), width_map {a,b,gap_min,gap_max,coverage_required_m},
                                                    hand speed/force/epsilons/home width, payload mass/COM/inertia, collision
                                                    boxes (cradle, phone or rig housing), sensor_compat
  station/sensor/     iphone16pro_wifi.yaml, rig1_wifi.yaml
                                                    protocol v1|v2, ports, required app settings (Ethernet, High, 30 fps, host IP),
                                                    stream size/physical fps/JPEG quality/K, both training-intrinsics variants
                                                    (39 demos f=1406.34/259.02, 6 demos f=1278.45/246.65), publish fps 60
                                                    fixed_rate, clock mode + latency, health (max_age_s), task camera_latency_s,
                                                    max_image_age_s
  station/scene/      block_raise_table.yaml        table, eraser size/mass/friction, grasp TCP height, workspace box, placement grid
  station/host/       jetson_orin_nano.yaml, pc_laptop.yaml   python env paths, work_root, device, checkpoint root
  run/policy/         bp_run001.yaml, dp_template.yaml, demo_replay.yaml, fake.yaml   (kind: bp_server|replay|fake|command)
  run/link/           ideal.yaml, wifi_good.yaml, wifi_busy.yaml, wifi_bad.yaml (+ *_skew variants for the rig)
  run/criteria/       block_raise.yaml              gates for every stage + success rule + gonogo criteria
  run/sim/            physics.yaml (timestep, substeps, friction scales, reflex toggles), render.yaml, video_replay.yaml,
                      rig_raw_replay.yaml, stage1.yaml (grid, thresholds, downsample, top-k)
  run/sweeps/         block_raise_default.yaml
```

**Conventions (enforced by the loader and by tests):**
- Every file has `schema: <group>/v1` and `id`.
- **Units are in key names.** Suffixes: `_m _rad _deg _s _hz _kg _n _nm _m_s _rad_s _px _ppm`. Matrices are `T_*` (4×4) or `K`.
- **Provenance is machine-readable.** A `provenance:` map gives every leaf `{status: measured|placeholder|hw|datasheet|derived|decision|fitted, source}`.
- **A `critical:` list per file.** Real profiles refuse placeholder or hw values in these paths; sim profiles only warn. Overrides go through `allow_placeholders`, and are logged.
- **Interpolation:** `${group.path}` only. Overrides use `--set path=value` (used by sweeps). Precedence: group file < `station_set` < explicit `bridge.override`/`controller.override`. Contradictions are recorded in the snapshot.
- **`deploy.yaml` profiles pick ids:** `station: {robot, tooling, sensor, scene, host}` + `run: {policy, link, criteria, sim}`.
  - Profiles **without** `station:` behave byte-identically to today (golden test).
  - New profiles: `sim-iphone`, `sim-rig`, `sim-iphone-fake`, `sim-rig-fake` (FakeBackend, no MuJoCo), `real-iphone`, `real-rig`.
- **The launcher snapshot writes derived configs:**
  - `bridge.yaml` (impedance, limits, payload, safety box from the scene);
  - `franka_cartesian.yaml` (F_T_TCP / EE_T_TCP, gripper block);
  - task args;
  - `camera.yaml`;
  - `sender.yaml`;
  - `station_resolved.yaml` with provenance and a fingerprint.

**Hardcoded constants to lift (same defaults, so existing tests stay green):**

| Where today | Constant | Goes to |
|---|---|---|
| `tasks/run_task.py:57-61` | lookahead 0.1, home gripper 0.08, camera_latency_s 0.08, policy_latency_s 0.1 | CLI flags fed from `station.task_args()` |
| `configs/task_franka.yaml` (read by no code) | anchor/state-age values | deleted; values live in deploy.yaml task / run_task defaults |
| `fdcal/handeye.py:24-25` | C_T_TCP, K_MAIN_960 | tooling.C_T_TCP, sensor.streams.main.K (`--station`) |
| `fdcal/width.py:23,59,85` | K_UW_960, 0.085 coverage | sensor.streams.ultrawide.K, tooling.width_map.coverage_required_m (`--min-coverage`, `--station`) |
| `eval/annotate.py:28` | 0.03 m / 2 s / 2 mm | criteria.success (0.05 / 2.0 / 60 / 0.002), `--criteria` |
| `tools/gonogo.py:48` | CRITERIA | criteria.gonogo (+ `--no-loop-gate` for sim) |
| `bridge.yaml` home, safety box/cone, impedance | | robot / scene / topology via station fragments |
| `franka_cartesian.yaml` F_T_TCP placeholder, gripper calibration | | tooling (model-derived F_T_TCP exported by the sim builder) |
| `deploy.yaml profiles.mock.home` | hand-composed pose | derived from home + F_T_TCP |
| `gripper/driver.py` max width 0.11, FakeGripper start 0.08 | | tooling |
| Bridge port 5555 (deploy.yaml:38, bridge.yaml:2, franka_cartesian.yaml:2, franka_bridge.py:198, fdcal/capture.py:155, harness/dryrun.py:203) | **clashes with the iPhone app's hardcoded 5555** | moved to **5565** everywhere |

## 5. Components

### Track A: simulator core (`controlSW/franka_sim/`, new sibling package)

`franka_deploy` imports it only inside `make_backend` when `type == "mujoco"`, so the Jetson never needs MuJoCo.

- **`franka_sim/assets/`:** menagerie `franka_emika_panda/` (panda.xml, hand.xml, collision and visual meshes) and `franka_fr3/`, each with LICENSE, plus `SOURCE.md` (URL, pinned commit, sha256 per file). `tools/vendor_menagerie.py` fetches into an isolated scratch directory, verifies and copies. `tools/check_assets.py` verifies.
- **`model.py`:** `build_model(station, sim_cfg, with_scene=True) -> SimModel`, using MjSpec.
  - Replace the position-servo actuators with torque motors (±87/12 Nm) and a hand tendon motor. FR3 frictionloss is scaled to about 0 by default.
  - Attach `hand.xml` to FR3 at link 7, z = 0.107, −45° yaw.
  - Add sites: `flange`, `ee` (flange·F_T_NE·NE_T_EE), `tcp`.
  - Add printed-finger adapter boxes (pad offset gives separation = gap + b) and ArUco tag patches.
  - Add the cradle and the phone (or rig housing) boxes with masses; cameras `cam_main` / `cam_uw` at TCP·inv(C_T_TCP) (OpenCV→MuJoCo flip) with stream intrinsics; table and eraser.
  - `export_frames()` writes F_T_TCP, EE_T_TCP, camera and tag poses.
- **`kinematics.py`, `frames.py`:**
  - FK and Jacobian; damped-least-squares IK with limits and nullspace posture; multistart; contacts and min-distance; manipulability.
  - `frames.py` is pure numpy: ARKit y-up → base z-up, placement transforms, the rot6d (rows) convention.
- **`stage1_replay.py`** (Stage 1, offline):
  1. For all 45 demos: grasp index from `eval.metrics.phases`.
  2. Coarse-to-fine placement search (gx, gy, yaw), anchoring each demo's grasp TCP at the table grasp height.
  3. TCP → EE → IK.
  4. Metrics: reach %, joint margin, manipulability, dq/ddq/jerk against the datasheet (Savitzky–Golay filtered); collisions (self, table, tooling↔arm); safety box and cone; **clamp statistics from the real `bridge.safety.SafetyFilter`** at 200 Hz (clamp fraction, backlog, slowdown); Cartesian speed; width-map fit to the stroke.
  5. Outputs: recommended placement, per-demo start q and pose, recommended home, `stage1.json` + summary. Both arms.
- **`simcore.py` + `backend.py`:**
  - **`SimCore`:** deterministic and steppable; reproduces panda-py's CartesianImpedance law (target filter, Jᵀ(−Ke − DJq̇), nullspace term, Coriolis, ≤ 1 Nm/ms rate limit, torque saturation, gravity on the robot side with a payload-mismatch term). Modes: idle, running, stopped, reflex. Quintic `move_joints`. libfranka-style reflexes: contact thresholds (cartesian/joint), joint velocity, position limits, self-collision, tau range, optional Cartesian speed.
  - **`MuJoCoBackend(Backend)`:** a 1 kHz wall-clock thread with catch-up stepping and a cached state snapshot. `loop_stats` adds sim lag, resyncs and real-time factor. `inject_reflex`. `extra_topics` routed by the bridge: `hand_cmd`, `hand_state`, `sim_reset`, `sim_state`. A ZMQ PUB of the sim state on 5570 at ≥ 100 Hz (for the judge and the render sender). Sim ground-truth lines go to `bridge.jsonl`. Optional passive viewer and offscreen video.
  - Bridge sim characterization with `tools/gonogo.py`: lag, steady-state error, a knob table.
- **Franka Hand:**
  - `franka_deploy/gripper/hand_logic.py` (pure): WidthMap; HandCommandPolicy with grasp-below/open-above hysteresis, deadband, minimum interval, latest-wins, speed cap and force ≤ 70 N.
  - `franka_deploy/gripper/franka_hand.py`: `FrankaHandDriver(GripperDriver)`, **non-blocking** (background thread, own REQ socket). This matters because the controller calls `set_width` inside its 200 Hz loop.
  - `franka_deploy/gripper/hand_server.py`: real `LibfrankaHand` worker in its own process (port 5566; `Gripper.move`/`grasp` block for 0.3–1.5 s and must not sit in the RT bridge) and a `FakeHand`.
  - `franka_sim/hand_sim.py`: libfranka move/grasp semantics on the finger tendon, served by the bridge in sim. The driver code is identical in sim and real; only the endpoint differs.
- **`scene.py`, `ground_truth.py`:** table and eraser; `sim_reset` resets objects and the hand only, never teleports the arm, and is refused while grasping. `evaluate()` computes success, lift, slip, knock-over and arm–table contact from the sim events. `eval/report.py` gains an additive `sim_gt` field.
- **`render.py` + camera-view check:**
  - `Renderer(model, station).render(qpos) -> {main, ultrawide}` BGR 960×720 at the deploy intrinsics.
  - `python -m franka_sim camera-check` renders the fingers, hand and tags at the ARKit training intrinsics (main 1920×1440, UW 640×480) and overlays them on real firstreal frames (`IphoneEpisode`).
  - It projects the tag centres and compares them with `detect_aruco` on real frames: a pixel error per tag, plus whether the hand body is visible. This checks the mount design before printing.
- **`campaign.py`** (Stage 1b): per demo × speed, a full-stack run with `policy: demo_replay`. The arm starts at the demo's start pose mapped through the placement, and the eraser sits at the demo's grasp location. Collected per run: `sim_gt`, replay-tracking error, clamps, settle timeouts, reflexes, hand statistics. Success rate with a Wilson CI.
- **Other fixes Track A owns:**
  - **`PandaPyBackend.error` is never set,** so real reflexes would not propagate. Derive it from the robot mode and the errors, testable with a fake panda_py module, and add it to `panda_py_checklist.md`.
  - **The Desk EE convention (F_T_NE) goes into the robot config.** It also becomes a hardware-day check item.

### Track B: config, wireless sensor path, policy slot, evaluation runner

- **`station.py`** + all YAMLs (§4); constant lifting; bridge port 5565; launcher snapshot from the station; preflight `station_valid` / `station_placeholders` / `tooling_sensor_compat`.
- **Policy slot:**
  - launcher `start_policy(kind)`: bp_server, replay, fake (`fake_policy_server --ready-file`), or command template;
  - `tools/policy_contract.py`, which reuses `analyze_tap.CONTRACT_OBS/REPLY` and checks: config keys, reset < 0.5 s ×5, K=16, shapes, rot6d sanity, the prompt/ERROR semantics, `export_recorded_data`;
  - `docs/adding_a_policy.md`.
- **Wire protocol** `sensor_rig/sensor_rig/stream/protocol.py` (pure; the rig must not depend on franka_deploy):
  - **v1** = the iOS app: event `rgb`, base64 JPEG per stream port; `validate` must return a non-empty value.
  - **v2** = the rig and the sim senders: event `packet` with binary JPEGs, `t_main_ns`, `t_uw_ns`, `seq`, `clock`; plus `clock_ping` and `hello`.
- **`franka_deploy/stream/`:**
  - `link.py` LinkEmulator: deterministic, in order, TCP-style stalls (no frame drops), token-bucket bandwidth with an unbounded FIFO, clock offset and drift;
  - `client.py` StreamClient: reproduces the app's validate loop;
  - `iphone_replay_sender.py`: `IphoneEpisode` → 960×720 with the half-pixel convention, at the episode's own frame times, writing a truth file;
  - `rig_replay_sender.py`.
- **Receiver** `franka_deploy/peripherals/wireless_camera.py`, a `BaseCamera` subclass:
  - one Flask-SocketIO server per port, threading mode, websocket required (`simple-websocket`);
  - cv2 decode; newest frame per stream; fixed-rate 60 Hz publish (fps % 20);
  - `BaseCamera.camera_latency_s = 0`, with latency applied exactly once in `clock_map.py`. Modes: `arrival_minus_latency`, `arrival_dejitter`, `sender_stamp_ping`, `sender_stamp_realtime`.
  - stale guard (stop publishing past `max_age_s`); a clear error on size mismatch (the app's Low setting);
  - health JSON (fps, inter-arrival, age, decode ms, reconnects, transport, seq gaps, clock offset/RTT/drift);
  - a sim truth log for measuring timestamp error;
  - fallback if the robotmq GIL adds > 5 ms jitter: split into a socket process + camera process over shared memory (built only if B5's measurement says so).
- **Task guard:** `--max-image-age-s` (default None = today's behaviour). It raises a retryable `FrameSelectionError` on old images.
- **Launcher wireless integration:**
  - `camera.kind: rig_local|wireless`;
  - in sim profiles a `sender` child process; teardown order;
  - preflight `wireless_health_ok` (websocket, fps ≥ 0.9·physical, age p95, no reconnects) and `clock_ok` (v2);
  - receiver and state ports in `ports_free`;
  - launcher support for Track A's needs: `PYTHONPATH` += franka_sim, the `hand` child when `hand.endpoint != bridge`, a `sim_reset[:json]` script action.
- **Rig sender** `sensor_rig/sensor_rig/stream/sender.py`: `RigFrameSource` → virtual-iPhone images → v2 with capture stamps; drop-oldest queue; `sensor-rig stream-send`.
- **Render sender** `franka_sim/senders/render_sender.py`: subscribes to the sim state PUB, renders at 30 fps, and speaks the sensor's protocol (v1 without stamps for the iPhone, v2 for the rig).
- **Eval runner** `franka_sim/eval/`, run as `python -m franka_sim eval --deploy-config D --profile P --stage 1|1b|2a|2b|3|all [--policy] [--sweep] [--set] --out DIR` (`--stage all` = 1, 1b, 2a, 2b; it includes the 1b full-stack campaign, about 1 h for the 45 demos; Stage 3 only when asked for):
  - `criteria.py` (pure);
  - `report.py`: reuses `franka_deploy.eval.report.episode_metrics/summarize`, adds gates, verdict, `image_source`, `policy_id`, station fingerprint, `criteria_version`, schema `sim_report/v1`, and an HTML banner for render runs;
  - `judge.py`: subscribes to the sim state and writes `y`/`n` to `commands.txt`;
  - `sweeps.py`: cartesian axes, max_runs, seed, resume;
  - `franka_deploy/tools/predict_demos.py` (env_bp) for Stage 2a.
- **Hardware-day tools:**
  - `tools/wireless_latency.py`: QR-code latency per stream;
  - `fdcal stream-intrinsics`: ChArUco calibration of the deployment stream, compared with both training variants.
- **Docs:** the hardware_day.md iPhone/Franka Hand path (app settings, dedicated 5 GHz AP, firewall 5555/5556, latency, intrinsics, reconnect check); README; PURCHASING.md (the Franka Hand is now chosen); ROADMAP; TODO; AGENTS.

## 6. Evaluation stages and what each proves

All stages run for **iPhone** (`sim-iphone`) and **rig** (`sim-rig`), **Panda** and **FR3**, and **any policy** that passes the contract.

| Stage | What runs | What it proves | Key gates (criteria/block_raise.yaml) |
|---|---|---|---|
| 1 | Offline IK replay of 45 demos + placement search | The embodiment can do the demos; where to put the eraser; the cradle clears link 6/7 | ≥ 95 % reachable at the chosen placement, ≥ 10° joint margin, 0 collisions, mapped width p99 ≤ stroke |
| 1b | Full stack, `demo_replay` policy, MuJoCo physics, Franka Hand grasp/lift | The control stack + impedance + hand can execute the demos and lift the eraser | replay tracking ≤ 1 cm / 3° on ≥ 90 % of runs, 0 reflexes, sim success reported per speed |
| 2a | Offline policy predictions on held-out demos → base frame → IK | The policy's outputs are feasible on this arm, and how close they are to the demo | chunk reachable ≥ 95 %, margin ≥ 10°, pos err p95 k4/k8/k15 ≤ 15/25/40 mm |
| 2b | Live stack, demo **video** through the sender + emulated Wi-Fi, teacher-forced | The live wireless pipeline gives the same predictions as offline; timestamp quality | live-vs-offline p95 ≤ 5 mm, 0 frame-selection errors, stamp err p95 ≤ 30 ms (v1) / 5 ms (v2) |
| 3 | Closed loop with **rendered** images (plumbing), any policy, with sweeps | Safety, timing, holds/stops, hand, link failures end safely | 0 reflexes, 0 non-finger collisions, tracking p95 ≤ 5 mm, width reached within 1 step, inference p95 ≤ 0.45 s, receiver age p95 ≤ 0.15 s, 0 stale-image errors; success informational only |
| Camera check | Renders vs real frames (tags, fingers, hand body) | The mount/finger design reproduces the training view | tag reprojection error and hand-visibility reported for mount design |

Sweeps: camera latency assumed 0.05–0.15 s (true latency fixed); v_max 0.25/0.5; link ideal/good/busy/bad; Panda/FR3; iPhone/rig.

## 7. Phases (each = one executor task + one reviewer pass)

Per-phase spec files go in `controlSW/docs/sim_plan/phases/<id>.md`. Each spec has: goal, exact files, interfaces, test commands with env, a numbered acceptance checklist, out of scope, and "read only these files". Shared interfaces live in `controlSW/docs/sim_plan/INTERFACES.md`.

**Track A**

| Phase | What it delivers | Acceptance (summary) |
|---|---|---|
| A0 | mujoco 3.3.0 into env_realenv; vendor menagerie (both arms, full); check-assets | both XMLs compile; sha256 match; `import franka_sim` works without mujoco |
| A1 | `model.py` + export_frames (consumes the station; fixture until B1 lands) | camera→TCP == C_T_TCP (1e-9); flange→TCP == F_T_TCP; pad separation = gap+b; hand pose same on Panda/FR3; no contacts at home; **cradle↔link 5–7 clearance report (≥ 10 mm or flagged)**; EGL render smoke |
| A2 | `kinematics.py`, `frames.py` | FK vs Franka DH (1e-6, 200 q, both arms); IK ≥ 99 % on 500 poses; ≤ 1.5 ms/frame warm |
| A3 | Stage 1 | 45 demos × 2 arms < 30 min with 4 jobs; clamp fraction consistent with the known 7 % above 0.25 m/s; deterministic; synthetic fixtures pass |
| ⏸ | **Checkpoint: Stage 1 + clearance results go to the user** (may change the cradle design or the arm choice) | |
| A4 | SimCore + MuJoCoBackend + bridge `extra_topics` + make_backend `mujoco` + gonogo characterization + PandaPyBackend error fix | hold drift < 0.5 mm/5 s; 2 cm step overshoot ≤ 5 %, steady ≤ 1 mm; payload-error z offset ≈ mg/K; reflexes fire correctly; real-time p99 period ≤ 2 ms; stack test with mujoco passes; gonogo lag 5–80 ms, steady ≤ 2 mm |
| A5 | Franka Hand: hand_logic, FrankaHandDriver, hand_sim, hand_server, make_gripper `franka_hand` | logic tests in both envs; `set_width`/`get_width` never block > 1 ms; 45 mm box grasp succeeds; controller loop p99 unchanged (≤ +0.2 ms) |
| A6 | scene, sim_reset/sim_state, state PUB, ground_truth, report `sim_gt` | scripted pick-and-lift ≥ 9/10 seeds; synthetic GT streams classified correctly; reset refused while grasped |
| A7 | Renderer + camera-check + render_sender | render K verified by a projected point; overlay report on 5 firstreal episodes; render sender v1 sends no stamps, v2 sends stamps |
| A8 | Stage 1b campaign | 45 demos at speed 0.5 without system faults; tracking metric on ≥ 90 %; sessions pass `franka_deploy validate` + `eval` |

**Track B**

| Phase | What it delivers | Acceptance (summary) |
|---|---|---|
| B1 | `station.py` + all YAMLs | all load and validate; unit/provenance/cycle/compat errors caught; derived fragments equal today's configs; fingerprint stable |
| B2 | constant lifting, port 5565, launcher snapshot from station, check_config, fdcal `--station`/`--min-coverage`, annotate `--criteria` | golden snapshot of jetson/mock profiles unchanged (except the port); run_task defaults identical; real profile fails on critical placeholders, sim warns |
| B3 | policy slot + contract tool + doc | contract passes against fake and replay servers; fails on a deliberately broken one (K=8, 0.7 s reset); mock run with `policy: fake` reaches READY without a GPU |
| B4 | protocol, LinkEmulator, StreamClient, iPhone replay sender, requirements-wireless.txt | deterministic schedules; in order; backlog grows past the bandwidth cap; stalls delay later frames; v1 wire format matches the app; real-time pace within 2 % |
| B5 | WirelessCamera + ClockMap | v1 stamp == arrival − latency within 3 ms (ideal link); v2 with 12.345 s offset + 30 ppm drift within 2 ms after 5 s; dejitter beats arrival on wifi_busy; size-mismatch error; stale guard; non-empty validate; websocket transport; GIL jitter measured |
| B6 | launcher wireless + sender child + profiles + task image-age guard + Track A launcher needs | `sim-iphone-fake` e2e scripted run; `kill:sender` → hold + stale counter; mock profile and test_franka_stack unchanged |
| B7 | rig sender + `sim-rig-fake` | v2 stamps == Packet.t_capture_ns; drop-oldest under a slow link; simulated rig episode through wifi_good + skew has stamp error p95 ≤ 2 ms; all 400 sensor_rig tests pass |
| B8 | eval runner, criteria, Stage 1 wrapper, Stage 2a (`predict_demos`), Stage 2b | `report.json` validates `sim_report/v1`; 2a with demo_replay gives 0 error; 2b with the fake policy produces gates; reproducible with a seed |
| B9 | Stage 3 (judge, between-episode reset, `sim-iphone`/`sim-rig` finalized) + sweeps + summary | demo_replay 3 episodes × 3 grid poses with verdicts; 2×2 sweep resumes; a summary compares fake vs demo_replay |
| B10 | hardware-day tools + docs | QR decode test; stream-intrinsics recovers K within 0.5 % on synthetic renders; every documented command runs `--help` |

## 8. Waves and dependencies

| Wave | Track A | Track B | Notes |
|---|---|---|---|
| W0 | orchestrator: branch `sim-eval`, env installs (mujoco, simple-websocket, websocket-client, flask-socketio into env_realenv), INTERFACES.md, all phase specs | | no agents |
| W1 | A0 | B1 | |
| W2 | A1 | B2 | A1 switches from the fixture to the real station loader |
| W3 | A2 | B3 | |
| W4 | A3 (Stage 1) | B4 | **user checkpoint after A3** |
| W5 | A4 | B5 | |
| W6 | A5 | B6 | |
| W7 | A6 | B7 | |
| W8 | A7 | B8 | B8 needs A2 (IK API) + B6/B7 |
| W9 | A8 | B9 | B9 needs A4–A7 + B8 |
| W10 | system review (A reviewer: whole stack; B reviewer: docs vs code) | B10 | final end-to-end runs (§11), AGENTS.md update |

**File ownership:**
- **Track A:** `controlSW/franka_sim/**` except `eval/` and `senders/`; also `bridge/backends.py`, `bridge/franka_bridge.py` (extra_topics/tick), `gripper/**`, `eval/report.py` (sim_gt), `tools/gonogo.py`, `docs/panda_py_checklist.md`.
- **Track B:** `controlSW/configs/**`, `station.py`, `stream/**`, `peripherals/wireless_camera.py` + `clock_map.py`, `launch/**`, `tasks/**`, `fdcal/**`, `eval/annotate.py`, `mocks/**`, `tools/policy_contract.py` + `predict_demos.py` + `wireless_latency.py`, `harness/dryrun.py`, `configs/*.yaml`, `__main__.py`, `sensor_rig/sensor_rig/stream/**`, `franka_sim/eval/**`, `franka_sim/senders/**` (render_sender is written by A7 under B's directory with B's agreement, listed in A7's spec), docs.

Touching a file outside your own track's list is an automatic review FIX. Never touched by anyone: `iPhUMI/**`, `behavior_prompting/**`.

## 9. Environments and disk

- Reuse the scratch envs in `/tmp/claude-1000/-home-ruben-Capstone/5ee0411f-…/scratchpad/`:
  - env_realenv (+ mujoco 3.3.0, simple-websocket, websocket-client, flask-socketio/python-socketio matching real-env pins);
  - env_bp (inference, eval, predict_demos);
  - env_iphumi (sensor_rig tests, regression pipeline check).
  - If they are gone, recreate them from the existing env specs.
- `MUJOCO_GL=egl` for headless rendering.
- **Disk is about 3.6 GB free.** Full arm assets are about 90 MB in the tree plus git objects. Campaign and eval output goes under `work_root` outside the repo, with `video: false` by default. Orchestrator checks `df` at each wave gate and stops below 1.5 GB.

## 10. Agent orchestration and rate-limit rules

**Roles:**
- Per track, one **executor** (general-purpose agent; edits code; runs only its phase's tests).
- Per track, one **reviewer** (general-purpose agent, told never to edit; verifies against the spec).
- The orchestrator (me) writes specs, gates waves, merges and commits locally, and reports to you.

**Concurrency (up to 4 at once, pipelined):** wave N's reviewers run while wave N+1's executors build. To keep this safe:
1. **Stagger launches by about 60–90 s.** Never send 4 launches in one message.
2. **At most 2 Opus agents running at the same time.**
   - Executors on Opus for the hard phases: A4, A5, A6, B5, B6, B9.
   - Sonnet for the mechanical phases: A0, A2, A3, B1, B2, B3, B4, B7, B8 skeleton, B10.
   - Reviewers: Opus for the hard phases, Sonnet otherwise.
   - Mixing model tiers spreads load across rate-limit pools.
3. **Hard caps on what goes in and out.**
   - Executor prompt: about 30 lines (role, spec path, ownership rule, envs, report format). It never pastes this plan or code.
   - Executor final report: ≤ 40 lines (files, test commands with pass/fail counts, deviations, questions).
   - Reviewer: `REVIEW.md` ≤ 60 lines plus one paragraph.
   - Specs: ≤ 150 lines, with a "read only these files" list, so agents don't re-explore the repo.
4. **On 429/overload errors.**
   - Never respawn. Back off exponentially (60 → 120 → 240 → 480 s, cap 10 min, ±20 % jitter), then **resume the same agent with SendMessage**: "continue from your WORKLOG".
   - Each phase keeps an append-only `<id>.WORKLOG`, so a lost agent can be replaced from the spec plus the worklog.
   - **Never two agents on one phase.**
   - After 2 rate-limit events in one wave, automatically drop to max 2 concurrent for the rest of that wave.
5. **Shared machine resources.**
   - Process and stack tests use fixed ports and `/dev/shm`. Full-suite and e2e runs are serialized with `flock <scratch>/suite.lock`.
   - New tests take ports from `FD_TEST_PORT_OFFSET`: Track A +1000, Track B +2000.
   - Long runs (Stage 1, campaign) go in the background and are waited on by notification, not polling.
6. **Reviewers verify without re-exploring.**
   - Read `git diff <wave_base> --stat`, the changed files, and the direct callers of changed symbols (grep).
   - Re-run the spec's test commands plus the regression guard.
   - Tick the acceptance checklist with evidence (command output or file:line).
   - Check the ownership map, and run the B1 provenance validator on new YAML.
   - Verdict PASS / FIX / BLOCK. FIX goes to the same executor through SendMessage. If that executor is mid-way through the next phase, it applies the fix first. The reviewer then re-checks only the fix.
7. **Wave gate (orchestrator) before the local commit `sim-eval: wave N`:**
   1. both REVIEW.md files PASS;
   2. phase tests pass;
   3. the full franka_deploy suite passes in env_realenv (+ the env_bp subset if touched);
   4. the sensor_rig suite passes if touched;
   5. `scripts/regression/check_protected.sh` and `test_iphone_deploy_unchanged.py` pass;
   6. `iphone_pipeline_check.py` passes, run at W6 and W10;
   7. at least 1.5 GB of disk is free.
   - No push. You get a short status line per wave.

## 11. Verification (end to end, at W10)

1. **Tests.** `pytest controlSW/franka_deploy/tests controlSW/franka_sim/tests` (env_realenv), the env_bp subset, the sensor_rig suite, and the 3 regression checks. All green.
2. **Assets and config.** `python -m franka_sim check-assets`. Then `python -m franka_deploy check-config --config controlSW/franka_deploy/configs/deploy.yaml --profile sim-iphone|sim-rig|real-iphone|real-rig` (from `controlSW/franka_deploy`: `--config configs/deploy.yaml`) shows the expected placeholders (real-* FAIL on critical placeholders, as designed).
3. **Stage 1 and the camera check.** `python -m franka_sim eval --profile sim-iphone --stage 1 --out $S/s1` gives the placement plus clearance. For both arms: `python -m franka_sim stage1 --zarr $ZARR --ids robot=panda,tooling=franka_hand_iphumi_iphone16pro,sensor=iphone16pro_wifi,scene=block_raise_table --robots panda,fr3 --out $S/s1_both`. Then `python -m franka_sim camera-check --zarr $ZARR --stage1 $S/s1_both/<robot>_ext<e>_<side>/stage1.json --out $S/camcheck` (one combo dir per robot/extension/side) gives the overlay report.
4. **Stages 1b, 2a, 2b, 3.** `python -m franka_sim eval --profile sim-iphone --stage all --policy demo_replay --out DIR` (includes Stage 1b: about 1 h for 45 demos), then `--policy bp_run001` (real checkpoint, GPU). Repeat with `--profile sim-rig` (and `--out DIR` on every eval call). Each run writes `report.json` + `report.html` with PASS/FAIL per gate.
5. **Sweep.** Latency × link × v_max with summary.html comparing runs.
6. **Policy contract.** `python -m franka_deploy policy-contract --endpoint tcp://localhost:18765 [--expect run/policy/<id>.yaml]` against bp, fake and replay servers.
7. **Manual look.** A Stage 3 run with `backend.mujoco.viewer: true` (needs a display; set it in a copy of deploy.yaml as `profiles.sim-iphone.bridge.override.backend.mujoco.viewer: true`, since `--set` is station-only) to watch the arm, hand and eraser.

## 12. Hardware implications to surface as they appear

- **The cradle/phone may come within centimetres of link 6/7** (camera about 0.19 m behind the TCP, 0.12 m off-axis). Found at A1; may need longer finger adapters or a different cradle angle.
- **The Desk end-effector setting must match the config.** A 0.1034 m / −45° mismatch between sim and real otherwise.
- **The iPhone app must be on Ethernet mode, High resolution, 30 fps** (its default is Low). Use a dedicated 5 GHz AP. Expect about 100 ms latency, to be measured.
- **Deployment ultrawide intrinsics are unverified against training** (ARKit UW), and training itself mixes two intrinsics variants. Checked with `fdcal stream-intrinsics`.
- **PURCHASING.md currently says not to buy the Franka Hand.** To be updated.
- **`PandaPyBackend` never reports reflexes today.** Fixed in A4.

## 13. Out of scope

- Photoreal or closed-loop visual evaluation (Gaussian splats and the like).
- Training a plain diffusion policy (only its server template + contract).
- Anything on the cluster.
- Pushing to GitHub without asking.
