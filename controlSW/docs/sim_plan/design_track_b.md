# Track B design: station config, wireless sensor path, policy slot, eval runner

This is the design produced in planning (2026-10-18). Resolved against PLAN.md:
- the gripper type is `franka_hand` with an `endpoint` key (not `bridge_hand`);
- the width map default is b = 0.008;
- the success rule is 0.05 / 2.0 / 60 / slip 0.002 (hold 2.0 s, not 1.0);
- the hand_server port is 5566;
- the sim state PUB port is 5570.

## Findings that shape the design

1. **Port clash.** The iPhone app hardcodes ports 5555 (main) and 5556 (ultrawide); the bridge uses 5555. Move the bridge to **5565** everywhere:
   - deploy.yaml:38, bridge.yaml:2, franka_cartesian.yaml:2;
   - the franka_bridge.py:198 default, the fdcal/capture.py:155 default, harness/dryrun.py:203.
   - No test uses 5555.
2. **Closed loop needs renders.** In closed loop the images must follow the sim. Video replay is only for Stage 2b (teacher-forced, `--history-source replay_gt`) and for transport tests. Render-based Stage 3 tests plumbing, safety and timing, not the real success rate. Report `image_source: render` with a banner.
3. **No replay "sensors".** A sensor yaml describes the real device; sim senders imitate that device's wire behaviour from the same yaml. The image source (`render|video_replay|rig_raw_replay`) is a `run.sim` setting.
4. **Stale images aren't detected.** `get_observations` (franka_task.py:366-372) never checks image age, and robotmq retains messages for 1 s. Two guards: the receiver stops publishing past `max_age_s`, and a new task flag `--max-image-age-s` (default None) raises a retryable `FrameSelectionError`.
5. **Split station from run.** `station: {robot, tooling, sensor, scene, host}`, `run: {policy, link, criteria, sim}`. `host` replaces the Jetson paths hardcoded in deploy.yaml.
6. **chrony does not sync CLOCK_MONOTONIC.** The `sender_stamp_realtime` mode converts through realtime on both ends. The default for the rig is NTP-style ping over the same socket (minimum-RTT sample).
7. **Frames go over TCP.** Loss shows up as stalls, and the app never drops frames (emit queues). So the LinkEmulator delivers in order, models loss as stalls, and keeps an unbounded FIFO behind a bandwidth token bucket.
8. **The receiver must answer `validate` with a non-empty value** (SocketClient.swift:89-96 reconnects on an empty ack). Count reconnects in health.
9. **Training intrinsics come in two variants.** 39 demos use main f 1406.3428 at 1920×1440 and UW 259.01538 at 640×480; 6 demos use 1278.4507 / 246.64919 (data/raw/2026-10-01/*/*.json). The sensor yaml records both; the replay sender uses each episode's own K.
10. **Websocket needs `simple-websocket`.** Without it engineio falls back to long-polling. Python senders need `websocket-client`. Both are installed in env_realenv. Health reports the transport, and preflight fails on polling.
11. **Width coverage.** fdcal width requires 0.085 m, but the Franka Hand strokes 0.080 m. Make the coverage a tooling value, and add a criteria check that the demo width p99, after the width map, fits the stroke.

## Code facts relied on

- The launcher start order is bridge, controller, camera, inference, spacemouse, task.
  - The camera command is hardcoded to `real_env.peripherals.rig_camera` (launcher.py:275-283).
  - Inference is either `inference.server` or `replay_server` (290-305).
  - `check_config` only accepts `camera.config in (wrist_right, wrist_mock)` (531).
  - `snapshot()` (173-215) writes the merged `bridge.yaml` and `franka_cartesian.yaml`.
- run_task.py:57-61 has literals: lookahead 0.1, home gripper 0.08, camera_latency_s 0.08, policy_latency_s 0.1. The task's `camera_latency_s` only extends the pause wait (franka_task.py:557); it is not a stamp correction.
- `BaseCamera.run` asserts the frame shape equals `camera_configs` and subtracts `camera_latency_s` once. It publishes main, ultrawide, then a timestamp, and skips publishing if the frames dict is empty.
- `frame_select` tolerance is max(2/fps, 2·dt), so a fixed-rate republish at 60 Hz over a 30 fps physical rate works (RigCamera already does this).
- The app's High setting gives 960×720, Low gives 320×240 (the default!). Messages are `emit("rgb", base64 JPEG)`, JPEG quality 0.9, with no stamps. The Python server answers `validate` with `"server got it!"`.
- `RigFrameSource` has a `sources=` injection hook. `Packet` carries `t_capture_s`, `t_capture_ns`, `seq`, `skew_s`. `MockCamera.replay` and `simulate_raw_from_iphone` exist.
- The policy contract is already in `tools/analyze_tap.py` (`CONTRACT_OBS`, `CONTRACT_REPLY`).

## Config system

Layout (see PLAN.md §4):
- `controlSW/configs/station/{robot,tooling,sensor,scene,host}/`;
- `controlSW/configs/run/{policy,link,criteria,sim,sweeps}/`.

**Conventions:**
- Each file has `schema: <group>/v1` and `id: <stem>`.
- **Units in key names.** The loader rejects numeric leaves without a recognised suffix, except under the allow-listed keys `K`, `T_*`, `F_T_*`, `C_T_*`, `*_T_*`, `width_map` (a, b), `inertia_kgm2`, `impedance` matrices, ids, versions and counts (`n_*`, `seed`, `version`).
- **Provenance.** A `provenance:` map keyed by dotted path, prefix-inherited: `{status: measured|placeholder|hw|datasheet|derived|decision|fitted, source, note?}`. Every leaf must resolve.
- **`critical:` paths.** Real profiles refuse placeholder/hw values there; sim profiles warn. `allow_placeholders: [...]` overrides, and the override is logged.
- **Interpolation:** `${group.path}`, resolved after load; cycles are errors.
- **Overrides:** `path=value`, YAML-parsed.
- **Precedence:** group file < `station_set` < explicit `bridge.override`/`controller.override` (contradictions recorded).

### Example: robot/panda.yaml

```yaml
schema: station.robot/v1
id: panda
model: {mjcf: franka_sim/assets/franka_emika_panda/panda.xml, flange_site: attachment_site}
desk_end_effector: none              # none | franka_hand  (must match Desk; hardware_day check)
F_T_NE: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
NE_T_EE: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
joints:
  q_min_rad: [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
  q_max_rad: [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
  dq_max_rad_s: [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
  ddq_max_rad_s2: [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
  dddq_max_rad_s3: [7500, 3750, 5000, 6250, 7500, 10000, 10000]
  tau_max_nm: [87, 87, 87, 87, 12, 12, 12]
  dtau_max_nm_s: [1000, 1000, 1000, 1000, 1000, 1000, 1000]
cartesian: {v_max_m_s: 1.7, a_max_m_s2: 13.0, j_max_m_s3: 6500.0, w_max_rad_s: 2.5}
home_q_rad: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
fci: {hostname: 172.16.0.2, libfranka_version: null, system_version: null, panda_py_version: null}
impedance: {translational_n_m: 800.0, rotational_nm_rad: 40.0, damping_ratio: 1.0, nullspace_stiffness: 0.5,
            filter_coeff: 1.0, collision_thresholds: null}
provenance:
  joints: {status: datasheet, source: "Franka Panda datasheet"}
  cartesian: {status: datasheet, source: "Franka Panda datasheet"}
  home_q_rad: {status: decision, source: "ready pose; replaced by stage1 recommendation"}
  fci: {status: hw, source: "bridge.yaml:19-22"}
  impedance: {status: hw, source: "bridge.yaml:23-27 (tune in B3-hw)"}
  model: {status: derived, source: "vendored menagerie"}
  desk_end_effector: {status: hw, source: "Desk setting"}
  F_T_NE: {status: hw, source: "Desk setting"}
  NE_T_EE: {status: hw, source: "Desk setting"}
critical: [fci.libfranka_version, impedance.collision_thresholds, desk_end_effector]
```

**FR3** (`fr3.yaml`, all `status: datasheet` to verify):
- q_min [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
- q_max [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
- dq_max [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
- ddq 10 ×7, dddq 5000 ×7
- cartesian: v 3.0, a 9.0, j 4500, w 2.5
- Cross-check against the vendored MJCF ranges (e.g. FR3 q6 0.4398–4.6216 in MJCF).

### Example: tooling/franka_hand_iphumi_iphone16pro.yaml

```yaml
schema: station.tooling/v1
id: franka_hand_iphumi_iphone16pro
sensor_compat: [iphone16pro_wifi]
hand: {type: franka_hand, stroke_m: 0.080, speed_m_s: 0.10, force_n: 30.0, force_max_n: 70.0,
       grasp_epsilon_inner_m: 0.01, grasp_epsilon_outer_m: 0.08, home_width_m: 0.08}
finger_adapter: {printed: iphumi_finger_v1, extension_m: 0.0, pad_size_m: [0.02, 0.008, 0.03], friction: 1.0}
width_map: {a: 1.0, b: 0.008, gap_min_m: 0.0, gap_max_m: 0.080, coverage_required_m: 0.080,
            train_closed_m: 0.0431, train_min_m: 0.0447, train_max_m: 0.0844}
F_T_TCP: [[1,0,0,0],[0,0.966,-0.259,0.0644],[0,0.259,0.966,0.3114],[0,0,0,1]]     # placeholder (franka_cartesian.yaml)
C_T_TCP: [[1,0,0,0],[0,0.966,-0.259,0.0644],[0,0.259,0.966,0.2114],[0,0,0,1]]
F_T_C_main: null          # CAD then fdcal handeye; null -> derived F_T_TCP @ inv(C_T_TCP)
C_main_T_C_uw: null       # iPhone main->UW extrinsic; HW/CAD (rig placeholder: 12 mm offset, parallel axes)
payload: {mass_kg: 1.15, com_flange_m: [0.0, 0.0, 0.06], inertia_kgm2: [[0.003,0,0],[0,0.003,0],[0,0,0.002]]}
collision_boxes:          # flange frame
  - {name: cradle, center_m: [0.0, 0.07, 0.08], half_extents_m: [0.04, 0.01, 0.08], mass_kg: 0.10}
  - {name: phone, center_m: [0.0, 0.085, 0.10], half_extents_m: [0.036, 0.004, 0.075], mass_kg: 0.199}
provenance: {...every leaf...}
critical: [F_T_TCP, width_map, payload.mass_kg]
```

`franka_hand_iphumi_rig1.yaml` has the same keys, `sensor_compat: [rig1_wifi]`, and a rig housing box and payload.

### Example: sensor/iphone16pro_wifi.yaml

```yaml
schema: station.sensor/v1
id: iphone16pro_wifi
device: {model: "iPhone 16 Pro", identifier: "iPhone17,1", sender: "iPhUMI app Deployment, Ethernet mode"}
app_settings_required: {connection: Ethernet, stream_resolution: High, camera_fps_hz: 30, host_ip: "<policy host IP>"}
transport: {protocol: iphumi_socketio_v1, bind_host: 0.0.0.0, ports: {main: 5555, ultrawide: 5556},
            validate_ack: "server got it!", require_websocket: true, max_http_buffer_bytes: 10485760}
streams:
  main: {width_px: 960, height_px: 720, physical_fps_hz: 30, jpeg_quality: 0.9,
         K: [[703.1714, 0, 476.3752], [0, 703.1714, 361.3600], [0, 0, 1]]}
  ultrawide: {width_px: 960, height_px: 720, physical_fps_hz: 30, jpeg_quality: 0.9,
              K: [[388.5231, 0, 480.6903], [0, 388.5231, 359.7618], [0, 0, 1]]}
training_intrinsics_variants:
  - {main_1920_f_px: 1406.3428, main_c_px: [953.2503, 723.22003], uw_640_f_px: 259.01538, uw_c_px: [320.2935, 239.67455], n_demos: 39}
  - {main_1920_f_px: 1278.4507, main_c_px: [954.6028, 723.5324], uw_640_f_px: 246.64919, uw_c_px: null, n_demos: 6}
publish: {fps_hz: 60, mode: fixed_rate}
on_size_mismatch: error
clock: {mode: arrival_minus_latency, latency_s: {main: 0.08, ultrawide: 0.08}, dejitter_window_s: 2.0}
health: {max_age_s: 0.30, min_fps_ratio: 0.9, stats_period_s: 1.0}
task: {camera_latency_s: 0.08, max_image_age_s: 0.35}
camera_server: {name: right_wrist_camera_0, video_codec: libx264}
critical: [clock.latency_s, streams.main.K, streams.ultrawide.K]
```

**rig1_wifi.yaml** differences:
- `protocol: iphumi_socketio_v2` (event `packet`, main port only);
- `rig_config_path: /home/jetson/rig/rig_calibrated.yaml`;
- `clock: {mode: sender_stamp_ping, ping_hz: 2.0, ping_window_s: 20.0, residual_latency_s: {main: 0, ultrawide: 0}}`;
- `sender: {jpeg_quality: 90, max_queue: 4, drop_policy: drop_oldest}`;
- `critical: [rig_config_path]`.

### Example: scene/block_raise_table.yaml

```yaml
schema: station.scene/v1
id: block_raise_table
table: {size_m: [1.2, 0.8, 0.75], top_z_base_m: 0.0, friction: 0.8}
robot_base_T_table: null      # from stage1.json or measured
eraser: {size_m: [0.060, 0.020, 0.012], mass_kg: 0.015, friction: [0.9, 0.005, 0.0001]}
grasp_tcp_height_m: 0.01
workspace: {min_m: [0.25, -0.40, 0.05], max_m: [0.80, 0.40, 0.70], cone_axis: [0.0, 0.0, -1.0], cone_max_deg: 60.0}
placement_grid: {x_m: [0.40, 0.55, 4], y_m: [-0.10, 0.10, 3], yaw_deg: [-30, 30, 3], seed: 0}
```

### host/, run/

- **host/pc_laptop.yaml:** `work_root`, `python: {bp, realenv}`, `extra_path`, `inference_device`, `checkpoint_root`. **jetson_orin_nano.yaml** holds today's deploy.yaml base values.
- **run/link/\*.yaml:**
  - fields: `latency_s`, `jitter_s` (gamma), `stall: {rate_hz, duration_s: [lo, hi]}`, `bandwidth_mbit_s`, `clock: {offset_s, drift_ppm}`, `seed`;
  - `ideal` = all zeros;
  - `wifi_good` = 0.008 / 0.004 / 0.02 Hz / 400;
  - `wifi_busy` = 0.020 / 0.015 / 0.2 Hz (0.05–0.25 s) / 150;
  - `wifi_bad` = 0.040 / 0.040 / 1 Hz (≤ 0.5 s) / 80;
  - `*_skew` variants add `clock: {offset_s: 12.345, drift_ppm: 30}`.
- **run/policy/\*.yaml:**
  - `kind: bp_server|replay|fake|command`;
  - `bp_run001`: checkpoint, prompt_zarr, prompt_index 0, python bp, ready_timeout_s 300, `contract: {K: 16, exec_h: 8, uses_prompt: true, reset_max_s: 0.5}`;
  - `dp_template`: `kind: command`, with a `cmd` template using `{python} {endpoint} {ready_file} {work_dir}`;
  - `demo_replay`: `kind: replay, episode, speed, start_frame`;
  - `fake`: `kind: fake, delay_s`.
- **run/criteria/block_raise.yaml:**
  - `success: {lift_min_m: 0.05, hold_s: 2.0, timeout_s: 60, slip_tol_m: 0.002}`;
  - `stage1: {reachable_fraction_min: 0.95, joint_margin_deg_min: 10, collisions_max: 0, width_p99_mapped_le_stroke: true}`;
  - `stage1b: {track_pos_m: 0.01, track_rot_deg: 3, track_ok_fraction_min: 0.9, reflex_max: 0}`;
  - `stage2: {chunk_reachable_fraction_min: 0.95, chunk_joint_margin_deg_min: 10, pred_vs_demo_pos_err_mm_p95_max: {k4: 15, k8: 25, k15: 40}, live_vs_offline_pos_err_mm_p95_max: 5, frame_selection_errors_max: 0, cam_stamp_err_ms_p95_max: {v1: 30, v2: 5}}`;
  - `stage3: {collisions_max: 0, reflex_max: 0, tracking_err_mm_p95_max: 5, gripper_width_reached_within_steps: 1, inference_latency_s_p95_max: 0.45, receiver_age_s_p95_max: 0.15, stale_image_errors_max: 0, obs_motion_mm_p95_max: 3}`;
  - `gonogo: {reflexes: 0, min_success_rate: 0.99, loop_p999_ms: 0.5}`.
- **run/sim/:**
  - `physics.yaml`: timestep_s 0.001, substeps 1, joint_frictionloss_scale 0, damping_scale 1, impratio 10, cone elliptic, payload_error_kg 0, reflex toggles;
  - `render.yaml` (`image_source: render, render_fps_hz: 30`);
  - `video_replay.yaml`;
  - `rig_raw_replay.yaml`;
  - `stage1.yaml` (grid 7×7, yaw step 45°, coarse 10 Hz, fine top-k 5, IK tolerances 1 mm / 0.5°).

### station.py API

See INTERFACES.md I-STATION. Derived fragments, which must equal today's configs when the station holds today's values:
- `bridge_fragment()`: `backend.panda_py.impedance` as a 6×6 diagonal matrix from translational/rotational; damping_ratio, nullspace_stiffness, filter_coeff, hostname; `safety.workspace_min/max/cone` from the scene; `v_max`/`w_max` from the task/run; `home_O_T_EE_xyz_wxyz`.
- `controller_fragment()`: `F_T_TCP` (EE_T_TCP when desk_end_effector ≠ none), and `gripper: {type: franka_hand, width_map, policy...}` when tooling.hand.type == franka_hand.
- `task_args()`: `{lookahead_s: 0.1, home_gripper_m: hand.home_width_m, camera_latency_s: sensor.task.camera_latency_s, policy_latency_s: 0.1, max_image_age_s}`.

### deploy.yaml

- Profiles without `station:` are unchanged (golden test).
- New profiles: `sim-iphone`, `sim-rig`, `sim-iphone-fake`, `sim-rig-fake`, `real-iphone`, `real-rig` (examples in PLAN.md §4).

## Wireless path

- **Protocol** (`sensor_rig/sensor_rig/stream/protocol.py`, pure):
  - v1: event `rgb`, a base64 JPEG `str`; `validate` returns non-empty; `depth` is ignored and counted.
  - v2: event `packet` `{v: 2, seq, main_jpeg: bytes, uw_jpeg: bytes, t_main_ns, t_uw_ns, clock: sender_monotonic|sender_realtime, w, h}` with binary attachments; `clock_ping` (server→client call: `{t0_ns}` → `{t1_ns, t2_ns}`); `hello` `{sender, protocol, streams, fps_hz, K}`.
  - Helpers: `encode_v1`, `decode_v1`, `make_packet`, `check_packet`.
- **Receiver** (`franka_deploy/peripherals/wireless_camera.py`), `class WirelessCamera(BaseCamera)`:
  - `__init__(transport, clock, health, stats_path=None, truth_path=None, on_size_mismatch="error", video_codec="libx264", **base_kwargs)`; `get_images_bgr_hwc_and_timestamp()` at a fixed rate of `camera_configs.main.fps`; `main(argv)` with `--config camera.yaml`.
  - One Flask-SocketIO app per port, `async_mode="threading"`, in a daemon thread.
  - cv2.imdecode; the newest frame per stream, held as `(bgr, t_arrival, t_sender|None, seq|None)`.
  - Packet assembly: main = newest main (a repeated frame keeps its stamp); UW = newest UW (v1) or the packet UW (v2).
  - Before the first frame, and when stale, return `({}, 0.0)`.
  - Size mismatch → clear error.
  - Health JSON is atomically replaced every `stats_period_s`.
- **ClockMap** (`peripherals/clock_map.py`, pure):
  - `arrival_minus_latency`;
  - `arrival_dejitter` (lower-envelope fit of `t_arrival_k ≈ a + k/fps` over a window; gaps from phone-side drops via rounding);
  - `sender_stamp_ping` (offset θ from the minimum-RTT sample, `θ = ((t1−t0)+(t2−t3))/2`, linear drift over the window);
  - `sender_stamp_realtime`;
  - output clamped monotonic, with a clamp counter.
- **GIL risk:** robotmq holds the GIL. Measure the jitter the receiver adds. If p95 > 5 ms, split into a socket process + camera process over shared memory.
- **Senders:**
  - `franka_deploy/stream/client.py` `StreamClient(host, ports, protocol, link=None, truth_path=None)`: `send_v1(stream, jpeg, t_true)`, `send_v2(packet, t_true)`; reproduces the app's 1 s validate loop.
  - `franka_deploy/stream/link.py` `LinkEmulator(latency_s, jitter_s, stall, bandwidth_mbit_s, clock, seed)`:
    - `deliver_t = max(prev_deliver_t, send_t + latency + jitter, bucket_free_t + size/bw) + stall`;
    - a delivery thread;
    - `sender_clock_ns()`;
    - pure `schedule(sizes, send_times)`.
  - `iphone_replay_sender.py`: `--episode DIR --protocol v1 --fps 30 --size 960x720 --jpeg-quality 0.9 --link link.yaml --truth T --ready-file F [--loop] [--start-now-ns N]`; reads with `IphoneEpisode`; half-pixel resize; uses the episode's own frame times.
  - `rig_replay_sender.py` wraps `sensor_rig.stream.sender.RigStreamSender` with the mock backend.
  - `sensor_rig/sensor_rig/stream/sender.py` `RigStreamSender(rig, host, ports, fps_hz, jpeg_quality, backend, backend_opts, pairing, max_queue, drop_policy, link=None)`, registered as `sensor-rig stream-send`.
- **Task guard:** `--max-image-age-s`.

## Launcher

- `self.station = station_from_deploy_cfg(cfg)`.
- `snapshot()` merges `station.bridge_fragment()` / `controller_fragment()` before the overrides and writes `station_resolved.yaml`, `camera.yaml`, `sender.yaml`. With no station: byte-identical output.
- `camera.kind: rig_local` (default, existing) or `wireless` (`franka_deploy.peripherals.wireless_camera --config`).
- In sim profiles, a `sender` child starts after the camera:
  - `render` → `python.realenv -m franka_sim.senders.render_sender`;
  - `video_replay` → `iphone_replay_sender`;
  - `rig_raw_replay` → `rig_replay_sender`.
- `TEARDOWN_ORDER = (task, controller, sender, camera, inference, spacemouse, hand, bridge)`.
- `start_policy(kind)` handles bp_server, replay, fake (`fake_policy_server --ready-file`), command. Task flags come from `task_args()`.
- Preflight additions: `station_valid`, `station_placeholders`, `ports_free` (incl. receiver/state/hand ports), `tooling_sensor_compat`, `websocket_available`, `wireless_health_ok`, `clock_ok` (v2).
- `check_config` accepts the `camera.kind` values.
- Track A needs: add `controlSW/franka_sim` to PYTHONPATH; a `hand` child when `hand.endpoint != bridge`; the `sim_reset[:json]` script action; backend `mujoco` allowed.
- fdcal: `handeye`/`width` `--station`; `width --min-coverage` (default 0.085 unchanged); `capture` port 5565; new `stream-intrinsics`.

## Tools

- `tools/policy_contract.py` (`python -m franka_deploy policy-contract --endpoint E [--prompt-index 0] [--obs dummy|tap:DIR] [--expect policy.yaml]`) checks:
  - the config keys;
  - reset < 0.5 s ×5;
  - inference before a prompt → ERROR when `use_prompting`;
  - prompt OK; a second prompt without a reset → ERROR;
  - inference reply matches `CONTRACT_REPLY` (finite; rot6d norm in 0.9–1.1; width in [0, 0.12]; K; action 0 near identity is a warning);
  - `export_recorded_data` exists.
  - Writes contract.json; reuses `analyze_tap` constants.
- `tools/wireless_latency.py`: shows a QR code of the monotonic time, decodes it with cv2.QRCodeDetector, and writes per-stream latency p50/p95.

## Eval runner (`franka_sim/eval/`)

- **CLI:** `python -m franka_sim eval --deploy-config D --profile P --stage 1|1b|2a|2b|3|all [--policy id] [--episodes N] [--sweep file] [--set k=v] --out DIR`. `--stage all` runs 1, 1b, 2a, 2b (1b, the full-stack campaign, takes about 1 h for 45 demos).
- **Run layout:** `DIR/<stage>_<policy>_<fingerprint>_<sha8(sets)>/`, holding `station_resolved.yaml`, `criteria.yaml`, `report.json`, `report.html`.
- **Modules:**
  - `runner.py`: per-env subprocesses; the sim side never imports torch and the bp side never imports mujoco.
  - `stage1.py`: wraps A3 and evaluates the gates (demo width p99 mapped vs stroke, read in a bp subprocess).
  - `stage2a.py`: `predict_demos.py` (env_bp) predicts K=16 chunks every `exec_h` frames on val demos into `pred.npz`; then maps them to the base frame via the stage1 placement, runs `check_traj`, and computes error at k = 4/8/15.
  - `stage2b.py`: launcher with `sim: video_replay`, `--history-source replay_gt --replay-npz --replay-start-now-ns` (from the sender ready file); `analyze_tap`; live vs offline; stamp error from the truth files.
  - `stage3.py`: launcher `--no-tty --script`; `judge.py` (subscribes to the sim state and writes y/n to commands.txt); between episodes `r`, `sim_reset` with the next grid pose, then `c`; `sim_truth.json` per episode.
  - `sweeps.py`: cartesian axes, `max_runs`, seed, resume.
  - `report.py`: reuses `franka_deploy.eval.report.episode_metrics/summarize`, adds gates/verdict/`image_source`/`policy_id`/fingerprint/`criteria_version`/schema `sim_report/v1`, and a banner for render runs.
  - `criteria.py`: pure `evaluate(criteria, metrics, stage)`.

## Risks

- **Wi-Fi bandwidth.** Use 30 fps, a dedicated 5 GHz AP, and the host wired to the AP.
- **Arrival-time stamps.** Measure stamp error; dejitter mode as mitigation.
- **GIL contention in the receiver.**
- **JPEG decode CPU on the Jetson.**
- **Deployment UW intrinsics.**
- **Width coverage.**
- **Visual sim gap.** Label render runs.
- **App reconnect behaviour.**
- **launcher.py merge conflicts.** B2, B3 and B6 run in sequence within Track B; Track A never edits the launcher.
