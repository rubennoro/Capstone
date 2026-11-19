# Deployment pipeline: reference tables

Condensed from `deployment/STRUCTURE.md`, `PLAN.md` and `policy_deploy/PI_PAYLOAD.md` on `rnoro/final-deployment`
@ `2ffe641`. Go back to those files for the exact rules.

## 1. Contracts and where they are defined

| Contract | Between | Schemas | Clock | Defined in |
|---|---|---|---|---|
| C0 | Pi → frames | `pi_stream/v1` (3-part: JSON header, main JPEG, ultrawide JPEG). Converted internally to `c0.packet/v1` | Pi | `policy_deploy/PI_PAYLOAD.md`, `policy_deploy/contracts/pi_stream.py`, `c0_packet.py` |
| C1 | frames ↔ manager | `c1.get_frames/v1`, `c1.frames/v1`, `c1.get_health/v1`, `c1.health/v1` | laptop | `policy_deploy/contracts/c1_frames.py` |
| C4 | manager ↔ policy | `c4.config/v1`, `c4.config.reply/v2` (+`kind`), `c4.reset/v1`, `c4.infer/v1`, `c4.chunk/v1`, `c4.prompt/v1` | none | `policy_deploy/contracts/c4_policy.py`, `c4_limits.yaml` |
| C2 | PC → laptop (PUB 100 Hz) | `c2.state/v1` | PC | `contracts/deploy_contracts/c2_state.py` |
| C3 | laptop → PC (REQ/REP) | `c3.hello/v1`, `c3.hello.reply/v1`, `c3.get_state/v1`, `c3.heartbeat/v1`, `c3.start/v1`, `c3.stop/v1`, `c3.home/v1`, `c3.set_waypoints/v1`, `c3.set_waypoints.reply/v1`, `ack/v1`, `error/v1` | PC | `contracts/deploy_contracts/c3_commands.py`, `common.py` |

`SHARED_SCHEMAS` (`contracts/deploy_contracts/shared.py`) is exactly the 12 C2/C3 schemas. franka_control's codec
knows exactly these, so a laptop-only schema sent to the PC gets `unknown_schema`.

**Wire format:** one ZeroMQ message = one msgpack map, `schema` first, then keys in field-table order. An array is
`{"__nd__": true, "dtype", "shape", "data": <bin>}`, C order, little-endian. 4×4 poses are **row-major** (Eigen and
libfranka are column-major, and a transposed pose fails the rigid check). Arrays are never cast. Decode error codes,
in check order: `not_msgpack`, `not_a_map`, `no_schema`, `unknown_schema`, `schema_version`, `missing`,
`unknown_field`, `type`, `dtype`, `shape`, `value`, `unexpected_message`. Python errors: `ContractError`,
`TransportTimeout`, `TransportCancelled`, `RemoteError`. Transport helpers: `RpcClient`, `RpcServer(schemas=...)`,
`Publisher`, `Subscriber` (linger 0; lazy pirate: a timeout reopens the socket).

### C0 header (`pi_stream/v1`)

`version` 1, `packet_no`, `t_send_ns` (Pi clock), `clock_offset_ns` (unused), `main` / `ultrawide` frame objects
(`frame_no`, `t_sensor_ns`, `exposure_us`, `t_mid_ns = t_sensor − exposure/2`), `skew_ns` (**the live Pi sends the
opposite sign: never use it**), `uw_reused`. Both JPEGs are 640×480. On the live Pi the ultrawide is about 16 ms newer
than the main, and a packet leaves about 46 ms after the main frame's `t_sensor_ns`.

### C2 `ArmState`

`t_ns` int64 (PC monotonic, the start of the 1 kHz cycle) · `seq` uint64 (+1 per broadcast, starts at 1; going back means a
restart) · `O_T_TCP`, `O_T_TCP_target` float64 4×4 · `q`, `dq` float64 (7,) · `gripper_width_m` · `mode` ∈ {idle,
running, holding, stopped, homing, fault} · `error` · `gate` {accepted, shortened, pulled_back, rejected}, counted since
franka_control started · `loop_late`. C2 never pauses (also in fault and homing). `t_ns` repeats after a fault,
around homing, and for up to 1 s after a stopped homing.

### C3 `SetWaypoints` / reply

Request: `chunk_id` uint64, `t_ns` int64 (N,) strictly increasing, PC clock; `O_T_TCP` float64 (N,4,4); `gripper_width_m`
(N,); 1 ≤ N ≤ `max_waypoints` (64). Reply: `chunk_id`, `accepted`, `shortened`, `skipped`, `rejected`
[{index, reason}], with `accepted + skipped + len(rejected) == N`. Any rejection means accepted 0, the PC goes to
`fault`, and the other waypoints say `not checked: ...`.

### C3 `HelloReply`

`t_pc_ns` (stamped right before encoding), `commit` (40 hex or `unknown`), `mock`, `schemas`, `F_T_TCP`,
`rotation_tol`, `gripper_width_max_m`, `max_waypoints`, `watchdog_limit_s`, `homing_timeout_s`, `max_lead_s`.
`compare_hello()` (`hello_check.py`) checks schemas, mock, F_T_TCP (1e-9), the limits and the watchdog. The laptop
also checks: worst heartbeat gap 0.57 s < watchdog 1.0 s, `timeouts.home_s` 90 > `homing_timeout_s` 30, longest
waypoint lead < `max_lead_s` 1.5, map vs hello within 5 ms, and the commit (a difference is only a warning).

## 2. C3 replies per mode

| Request | idle | holding | running | stopped | homing | fault |
|---|---|---|---|---|---|---|
| hello / get_state / heartbeat | reply | reply | reply | reply | reply | reply |
| start | ok → holding | refused | refused | ok → holding | refused | refused |
| stop | ok → stopped | ok → stopped | ok → stopped | ok | ok, aborts the move → stopped (or idle if it finished) | ok, stays fault |
| home | → homing, reply at the end | refused | refused | → homing | refused | refused |
| set_waypoints | error/v1 | gate | gate | error/v1 | error/v1 | error/v1 |

- `home` blocks the REP socket until the move ends. A home stopped by the PC's `s` replies `ok:false` with a reason
  starting `stopped: `, and the laptop then stays idle. Any other not-ok home is a conclude.
- Watchdog: fed by every decodable C3 request, armed only in holding/running, limit `link.yaml timing.watchdog_limit_s`
  1.0 s → `stopped`. A backstop in the 1 kHz loop fires at the limit + 0.1 s.
- Rules: **H1**: hellos only from the manager's main loop, one keep-alive per cycle. **H2**: no C3 while C2 shows
  running/holding (wait 2× watchdog, then refuse with "another C3 client").

## 3. Safety gate (`safety_gate.cpp`), per waypoint, in order

1. Time already passed → skipped (not checked).
2. Not finite → reject.
3. More than `max_lead_s` (1.5 s) ahead → reject (this catches a waypoint sent on the laptop clock).
4. Width outside [0, `gripper_width_max_m` 0.08] → reject.
5. Workspace box on the TCP position: within `box_margin_m` 1 cm outside → pulled back, beyond that → reject.
6. Tilt cone of the flange z axis vs `cone_axis` [0,0,−1], `cone_max_deg` 60: within 2° outside → pulled back, beyond
   that → reject.
7. Speed (leaky bucket from the current target): shortened to `v_max` 0.25 m/s and `w_max` 1 rad/s, with
   `slack_s` 10 ms.

Any reject → the whole request is refused → `fault` (never cleared remotely: Ctrl-C and restart `control_deploy run`).

## 4. Config files

| File | Machine | Holds | Overridden by |
|---|---|---|---|
| `contracts/limits.yaml` | both, C++ too | `rotation_tol` 1e-6, `gripper_width_max_m` 0.08, `max_waypoints` 64 | nothing |
| `contracts/frames.yaml` | both, C++ too | `F_T_TCP` (row-major), `settled` (laptop only) | nothing |
| `contracts/link.yaml` | both, C++ too | `pc.address`, `laptop.addresses`, `ports.state` 5620 / `commands` 5630, timing (heartbeat 0.02, watchdog 1.0, c3_timeout 0.3, state_stale 0.1, state_lost 2.0), `clock.estimator` | nothing (only `--local` / `--mock` switch to 127.0.0.1) |
| `policy_deploy/station.yaml` | laptop | session root, `python.light` / `python.torch`, `pi.address`/`port` 5555, `wifi_ssid`, local ports 5610/5640/5600, and the `frames:` / `policy:` / `manager:` / `fake_pc:` / `launcher:` override sections | — |
| `policy_deploy/config/{frames_process,policy,manager,fake_pc,launcher}.yaml` | laptop | every setting with its default; an unknown key in the station is refused | station sections |
| `control_deploy/station.yaml` | PC | session root, the `franka_control:` override section (e.g. `realtime: ignore`), the `console:` step sizes | — |
| `control_deploy/franka_control/config.yaml` | PC | robot_ip 172.16.0.2, control gains, load, safety (the gate), homing (q_home, 30 s timeout), gripper, messaging, fake_arm | station `franka_control:` key by key (not `libfranka_version`) |

Key manager settings: `exec_h` 8, `lookahead_ms` 0, `pose_source` target, `odd_chunk_max_pos_m` 0.02 / `_rot_deg` 5,
`anchor_fallback_m` 0.02, `max_image_age_ms` 250, `timeouts.policy_hang_s` 5, `task_name` / `prompt_index`,
`operator_script` (a scripted operator for tests and mock), `save_tap`.

## 5. Failure table (PLAN §7)

| Problem | Kind |
|---|---|
| A camera stale (0.35 s), a missed frame pick, an ultrawide pair refused | slow: skip the step |
| A camera down (2 s), the Pi stream restarted | conclude |
| Policy slower than its waypoints | slow: the arm holds |
| An odd chunk | slow: skip, count |
| Policy hung (5 s), an error, a broken chunk | conclude |
| A C2 seq gap, or the newest C2 older than 100 ms (mapped) | slow: skip |
| No C2 for 2 s (by arrival), a C3 timeout | conclude: "PC not answering" |
| C2 seq going back | conclude: "franka_control restarted" |
| Clock mismatch, a hello mismatch at start-up | conclude |
| PC `fault` | conclude (the PC holds; restart `control_deploy run`) |
| PC `stopped` during an episode without the laptop asking (watchdog, PC `s`) | conclude |
| PC homing / stopped between episodes (its keys) | normal: the laptop follows C2 and shows a hint |
| A contract violation | conclude, naming the message and field |
| Operator `s` | the episode ends as stopped |
| Operator `q` / Ctrl-C | conclude (normal, exit 0): stop sent, the PC keeps holding |

Problem codes: `policy_deploy/manager/problems.py`. Classification: `failure_monitor.py`.
