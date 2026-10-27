# Cross-track interfaces (frozen; changes only through the orchestrator)

## 1. File ownership

Editing a file outside your own track's list is an automatic review FIX. Nobody ever touches `iPhUMI/**` or `behavior_prompting/**`; the regression guard enforces this.

**Track A** (sim core):
- `controlSW/franka_sim/**`, except `franka_sim/eval/**` and `franka_sim/senders/**`.
- In franka_deploy:
  - `franka_deploy/bridge/backends.py`, `franka_deploy/bridge/franka_bridge.py` (extra_topics routing, tick sim logging);
  - `franka_deploy/gripper/**`;
  - `franka_deploy/eval/report.py` (additive `sim_gt`);
  - `franka_deploy/tools/gonogo.py`;
  - `controlSW/franka_deploy/docs/panda_py_checklist.md`;
  - tests named `tests/test_sim_*.py`, `tests/test_hand_*.py`, `tests/test_gonogo*.py` changes.

**Track B** (config, wireless, policy, eval):
- `controlSW/configs/**`.
- In `franka_deploy/`:
  - `station.py`, `stream/**`, `peripherals/wireless_camera.py`, `peripherals/clock_map.py`;
  - `launch/**`, `tasks/**`, `fdcal/**`, `eval/annotate.py`, `mocks/**`, `harness/dryrun.py`, `__main__.py`;
  - `tools/policy_contract.py`, `tools/predict_demos.py`, `tools/wireless_latency.py`.
- `controlSW/franka_deploy/configs/*.yaml`, `controlSW/franka_deploy/requirements-wireless.txt`.
- `sensor_rig/sensor_rig/stream/**` (plus the one-line CLI registration in `sensor_rig/sensor_rig/cli.py`).
- `controlSW/franka_sim/franka_sim/eval/**`, `controlSW/franka_sim/franka_sim/senders/**`. Exception: `render_sender.py` is written by A7.
- Docs: `controlSW/franka_deploy/docs/hardware_day.md`, `controlSW/franka_deploy/README.md`, `controlSW/docs/adding_a_policy.md`, `PURCHASING.md`, `ROADMAP.md`, `TODO.md`.
- Tests named `tests/test_station*.py`, `test_stream_*.py`, `test_wireless_*.py`, `test_policy_contract.py`, `test_launcher_*.py`, `test_clock_map.py`.

**Shared, orchestrator only:** `controlSW/docs/sim_plan/**` (agents may append only to their own WORKLOG/REVIEW) and `AGENTS.md`.

## 2. Ports (production defaults; tests add FD_TEST_PORT_OFFSET)

| Port | Use |
|---|---|
| 5555 / 5556 | WirelessCamera receiver (main / ultrawide). Hardcoded in the iPhone app; do not move. |
| **5565** | bridge REQ/REP. Moved from 5555 in B2; until then the existing 5555 stays. |
| 5566 | hand_server (real Franka Hand, libfranka Gripper) |
| 5570 | MuJoCoBackend sim-state PUB |
| 16702 | camera robotmq server |
| 55556 | video logger |
| 26010 | controller |
| 18765 | inference |
| 15557 | spacemouse |

## 3. Conventions

- **Poses:** 7-vectors are `[x, y, z, qw, qx, qy, qz]`; 4×4 matrices otherwise. scipy uses xyzw order: convert with `[1,2,3,0]` / `[3,0,1,2]`.
- **rot6d:** the first two **rows** of R (behavior_prompting `pose_util.py:95-98`). Reuse `franka_deploy.eval.report.rot6d_to_R` (or the function in `tools/replay_server.py`). Do not reimplement.
- **Frames:**
  - `O` = robot base (z up).
  - `F` = flange (libfranka F, link 8 = link 7 + 0.107 z).
  - `NE`/`EE` = libfranka nominal / end effector: `O_T_EE = O_T_F · F_T_NE · NE_T_EE`.
  - `TCP` = training TCP: z approach, x right, y down.
  - `C` = OpenCV camera frame of the (virtual) iPhone main camera.
  - The ARKit world is gravity aligned with **+y up** (`sensor_rig/sensor_rig/transforms.py` `T_YUP_ZUP`, `D_CV_ARKIT`).
- **Camera:** `C_T_TCP` = R [[1,0,0],[0,0.966,-0.259],[0,0.259,0.966]], t (0, 0.0644, 0.2114). Source: gripper_calibration.yaml iPhone17,1 via `fdcal/handeye.py:24`.
- **Gripper width units:** "training width" = tag separation minus the closed value; training range 0.0447–0.0844 m. The Franka Hand gap ranges 0–0.080 m. `train = a·gap + b`, with defaults a = 1.0 and b = 0.008.
- **Clock:** every process uses `time.monotonic()`. The simulator runs in real time.
- **Units in config key names:** `_m _rad _deg _s _hz _kg _n _nm _m_s _rad_s _px _ppm`.

## 4. APIs

**I-STATION (B1, used by everyone).** `franka_deploy/station.py`. Imports only the stdlib, numpy and yaml: no mujoco, no robotmq.

```python
STATION_ROOT = REPO_ROOT / "controlSW" / "configs"
def load_station(spec: dict, root: Path = STATION_ROOT, sets: list[str] = ()) -> Station
    # spec = {"station": {robot, tooling, sensor, scene, host}, "run": {policy, link, criteria, sim}} (any subset)
def station_from_deploy_cfg(cfg: dict) -> Station | None      # None when cfg has no "station" key
@dataclass(frozen=True)
class Station:
    groups: dict[str, dict]; ids: dict[str, str]; provenance: dict; critical: tuple[str, ...]; overrides: tuple[str, ...]
    def get(self, path: str, default=...)        # "tooling.width_map.b"
    def prov(self, path) -> Prov; def placeholders(self, critical_only=False) -> list
    def fingerprint(self) -> str; def to_yaml(self) -> str
    def bridge_fragment(self) -> dict; def controller_fragment(self) -> dict; def task_args(self) -> dict
    def camera_config(self, endpoint, logger_endpoint) -> dict; def sender_config(self) -> dict
```

Until B1 lands, Track A reads a fixture dict with the same group layout (`franka_sim/configs/station_fixture.yaml`). It switches to `load_station` in A1 or A2.

**I-MODEL (A1).**
- `franka_sim.model.build_model(station, sim_cfg=None, with_scene=True) -> SimModel`.
- `SimModel` fields: `model`, `spec`, `names`, `F_T_TCP`, `EE_T_TCP`, `limits`.
- `export_frames(sm) -> dict`.

**I-KIN (A2).**
- `franka_sim.kinematics.Kin(sm, frame="flange")`: `.fk(q)`, `.jac(q)`, `.ik(T, q0, **kw) -> (q, info)`, `.contacts(q)`, `.min_distance(q, pairs)`.
- `check_traj(sm, T_seq, q_seed, frame="ee", *, tol_p=1e-3, tol_r=deg2rad(0.5), collision=True) -> {reachable[N], q[N,7], err_p[N], err_r[N], joint_margin_rad[N,7], joint_margin_deg[N], collisions[N], manip[N]}` (as built in A2).
- `frames.arkit_to_zup(T)` = `inv(T_YUP_ZUP) @ T` (sensor_rig's T_YUP_ZUP maps z-up to y-up)..

**I-STAGE1 (A3).** `python -m franka_sim stage1 ...` writes `stage1.json`, containing `placement: {base_T_table|grid point, yaw}`, `home_q`, the start q/pose for each demo, and `score`.

**I-BACKEND (A4).**
- `make_backend` type `mujoco` returns `franka_sim.backend.MuJoCoBackend.from_config(cfg, home_O_T_EE)`.
- The bridge routes any unknown topic found in `backend.extra_topics` to `backend.handle_extra(topic, data)`.

**I-HAND (A5).** The same REQ/REP topics are served by the bridge in sim and by `hand_server` (5566) on real hardware:
- `hand_cmd {kind: move|grasp|stop|homing, width_m, speed_m_s, force_n, eps_inner_m, eps_outer_m, seq}` → `{accepted, seq}`. Latest wins.
- `hand_state` → `{gap_m, gap_vel_m_s, busy, mode, grasped, last_result, seq_done, t}`.
- The controller gripper type is `franka_hand`, with key `endpoint` (the bridge endpoint in sim, 5566 on real hardware).

**I-SIMSTATE (A6).**
- ZMQ PUB on `tcp://127.0.0.1:5570`, at least 100 Hz, robotmq-serialized dicts: `{t_mono, qpos, qvel, eraser_T_base (4x4), hand_gap_m, grasped, contacts: [{a, b, force_n}], reflex: str|None, tau_ext}`.
- Bridge topic `sim_reset {object_pose?|eraser_xy_yaw?, hand_gap_m?}`. It is only allowed with `--allow-sim-reset` or allow_inject, is refused while grasping, and never moves the arm.
- Bridge topic `sim_state` returns the latest snapshot.
- `bridge.jsonl` lines `{"kind":"sim", ...}` at 10 Hz.

**I-RENDER (A7).** `franka_sim.render.Renderer(sm, station).render(qpos_full) -> {"main": BGR(720,960,3), "ultrawide": BGR(720,960,3)}` at the sensor stream K.

**I-CLI (A0).** `franka_sim/__main__.py` holds a subcommand registry, `franka_sim.cli.register(name, add_parser_fn, main_fn)`. Track B registers `eval` without editing A's files (the same pattern as `sensor_rig/sim/cli.py`).

**I-POLICY (existing).** robotmq server topics `policy_config`, `policy_reset` (< 0.5 s), `policy_prompt`, `policy_inference` (K = 16 steps of pos3 + rot6d + width, relative to the last observed pose) and `export_recorded_data`. The contract test is `tools/policy_contract.py` (B3).

**I-CAMPAIGN (A8).** `franka_sim/campaign.py` (RE env; launches full sim stacks).
```python
def run_campaign(config, profile, stage1_json, demos, speeds, out_dir, sets=(), jobs=1) -> dict
    # demos "all" | "0,3,7" | [int]; speeds "0.5,1.0" | [float]; sets: extra station overrides for every run.
    # Returns the campaign.json content (also written to out_dir/campaign.json + campaign.txt). Resumable: rows in
    # out_dir/rows/<run>.json with the same fingerprint are reused.
```
- `campaign.json` (schema `sim_campaign/v1`): `fingerprint`, `overrides` (stage1 station_sets + `tooling.F_T_TCP` = model TCP
  + sets), `criteria` (= `criteria.stage1b`), `aggregates` per speed (`"0.5"`: n, system_faults, `track_ok {k, n, p, ci95}`
  Wilson 95 %, reflexes, pad_table, sim_gt_success, causes, failing), `sessions`, `rows`.
- Row (flat keys read by `eval/stage1b.py`): `run, demo, speed, stage1_infeasible, system_fault (bool), fault_reason,
  track_ok, track_fraction, reflexes [str], pad_table (int onsets), pad_table_max_force_n, cause
  (system_fault|reflex|pad_table|safety_clamp|hand|overshoot|null), session, episode, sim_gt {...}`, plus `track {...}`,
  `plan {...}`, `hand`, `rt`, `clamp_count`, `settle_timeouts`, `F_ext_max`, `wall_s`.
- CLI: `python -m franka_sim campaign --stage1 S --config deploy.yaml --profile sim-iphone --demos all --speed 0.5 --out DIR [--jobs N] [--set k=v]`.
  Ports: `FD_TEST_PORT_OFFSET` (default 1000) + 100 x job slot.

**I-TCP (W10A fix, Track A).** `franka_sim.model.derived_tcp(station) -> np.ndarray (4x4 F_T_TCP)`: the single sim TCP (flange -> hand -> extended pads, pad-centre midpoint; honours `extension_m`, `pad_size_m`, `tcp_x_along_hand`; ignores station `F_T_TCP`/`F_T_C_main`). Cached per `station.fingerprint()`; one hand-only compile. Track B's launcher (W10B, `launcher.sim_tcp_fill` in `load_config`): for `run_mode: sim` on the MuJoCo backend, when `tooling.F_T_TCP` is null (both tooling YAMLs, status `hw`) or placeholder/hw and no explicit `tooling.F_T_TCP=` is in `station_set`/`--set`, it appends `tooling.F_T_TCP=<launcher.sim_tcp(station)>` as a station override (derived_tcp, or F_T_C_main @ C_T_TCP when F_T_C_main is measured), so station_resolved.yaml, the controller cfg, `station_home` and the backend check see one value; recorded as provenance `derived (sim)` in `station_overrides` + `summary.sim_tcp`. Preflight `sim_tcp` (run + check-config) refuses a sim MuJoCo run whose controller TCP differs from `sim_tcp` by > 1 mm / 0.1 deg; the backend's mismatch WARNING is copied into the launcher output and `summary.warnings`. FakeBackend sims and real profiles never import franka_sim; with F_T_TCP null their controller keeps `franka_cartesian.yaml`'s placeholder, and real check-config still FAILs on the critical `tooling.F_T_TCP`. `MuJoCoBackend.from_config` prints a WARNING when the config's controller F_T_TCP differs by > 1 mm / 0.1 deg (`sm.info["controller_F_T_TCP_deviation"]`). `SimModel.info` adds `F_T_TCP_derived`, `F_T_C_main_source` (`station` | `derived ...`), `F_T_TCP_vs_camera_deviation`. A non-null `tooling.F_T_C_main` whose provenance is not `placeholder`/`hw` wins: cameras use it and F_T_TCP = F_T_C_main @ C_T_TCP (source `F_T_C_main`). `loop_stats` adds `rt_resyncs_window` (per window); `rt_resyncs` stays cumulative. Campaign: `process_tree(pid)`, `kill_cleanup_shm(pids)`.
