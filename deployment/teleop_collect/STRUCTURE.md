# teleop_collect: structure

Written from `PLAN.md`; where the two disagree, PLAN.md wins and this file is fixed. This file fixes **everything that
crosses between two executors** (§12.1 of PLAN.md): the tree and the owner of every file, the laptop contracts, the
config and station keys, the Python interfaces one part calls in another, and each executor's tests. Inside its own
files an executor decides the rest, and lists every deviation from this file in its report.

`fd:<path>` = `rnoro/final-deployment` @ `2ffe641`, at `~/Capstone-final-deployment/deployment/<path>`, read-only.
Re-synced to `ea9f7fb` (2026-11-19): `deployment/contracts/` (new `F_T_TCP`, `rigs/rig_pi_v3_640.yaml`), the rig file
setting and `frames/_dataproc/config.py`; the files taken from `ea9f7fb` name it in their header.
`pd:` = `fd:policy_deploy/policy_deploy/`. Owners: **E1** foundation, **E2** frames, **E3** teleop link, **E4** teleop
core, **E5** ui, **E6** launcher and fake PC, **E7** integration and documents. **L** = the lead.

## 1. Conventions

- **Copied files** start with a header comment, then the module docstring:
  - `# copied from rnoro/final-deployment:deployment/<path> @ 2ffe641` (or `@ ea9f7fb`), plus `# changes: imports only` when only the
    imports changed;
  - or `# adapted from ... @ 2ffe641`, with `# changes: <every change, in a few words>`;
  - a file built from several sources names each one.
- **Import paths:**
  - `policy_deploy.contracts` → `teleop_collect.contracts`;
  - `policy_deploy.frames` → `teleop_collect.frames`;
  - `policy_deploy.manager` → `teleop_collect.teleop`;
  - `policy_deploy.fake_pc` → `teleop_collect.fake_pc`;
  - `policy_deploy.launcher` → `teleop_collect.launcher`.

  User-facing text that named `policy_deploy` / `manager` now says `teleop_collect` / `teleop`.
- **Time:** integer ns. `_pc_ns` = PC `CLOCK_MONOTONIC`, `_pi_ns` = Pi clock, `_ns` = laptop `CLOCK_MONOTONIC`
  (fd:STRUCTURE.md §6.1). Every clock is injectable (`clock: Callable[[], int] = time.monotonic_ns`), as in pd:.
- **Errors:** one line naming the process, what failed and what to check (pd:contracts/process_args.py `print_fatal`).
- **Python 3.10**, `from __future__ import annotations`, dataclasses, type hints. No new dependency beyond §3.1.
- **Tests:** `unset PYTHONPATH; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/teleop_collect/bin/python -m pytest <paths> -q`,
  run from `deployment/teleop_collect/`. Free ports only (`tcp://127.0.0.1:*`, HTTP port 0) until wave 3.

## 2. Tree and owners

```
deployment/
  contracts/                                  L (wave 0, done): byte-for-byte copy of fd:contracts/ (@ ea9f7fb), never edited
  control_deploy/DIRECTIONS.md                E7  the PC's directions (documents only, PLAN §12.5)
  teleop_collect/
    PLAN.md  STRUCTURE.md                     L
    DESIGN.md  DIRECTIONS.md                  E7
    pyproject.toml  station.yaml              E1
    config/                                   E1  (all five files, keys and defaults exactly as §3.3)
      frames_process.yaml  teleop.yaml  ui.yaml  fake_pc.yaml  launcher.yaml
    teleop_collect/
      __init__.py  __main__.py                E1  (__main__ calls teleop_collect.launcher.cli:main)
      common/                                 E1
        __init__.py  pose_math.py             copied from pd:manager/pose_math.py
      contracts/                              E1  (§4)
        __init__.py  DESIGN.md
        paths.py  station.py  config_merge.py  process_args.py        adapted from pd:contracts/
        pi_stream.py  c0_packet.py  clock_fit.py                       copied from pd:contracts/
        health.py                             CameraHealthReport, HealthReport, from pd:contracts/c1_frames.py
        r1_recorder.py  u1_teleop.py  u2_preview.py  keys.py           new
        examples.py                           new: in-memory examples of R1/U1/U2 (fd:contracts examples helpers)
      frames/                                 E2  (§5)
        __init__.py (NAME = "frames")  __main__.py  app.py  settings.py  errors.py  DESIGN.md
        receiver.py  pi_stream_source.py  packet_source.py  recording_packet_source.py
        camera_health.py  pair_rule.py  packet_pairer.py  ultrawide_store.py  rig_file.py
        fake_pi_sender.py  fake_pi.py  pi_recording.py  pi_collect.py
        _dataproc/__init__.py  _dataproc/config.py  _dataproc/paths.py   copied from pd:frames/_dataproc/
        recorder.py  mkv_writer.py  preview.py  servers.py              new
      teleop/                                 E3  (§6)
        __init__.py (NAME = "teleop")  __main__.py  app.py  settings.py  errors.py  problems.py  DESIGN.md
        pc_client.py  pc_listener.py  pc_clock.py  heartbeat_sender.py  link_stats.py  state_history.py
        startup_sequence.py  startup_checks.py  failure_monitor.py  session_log.py  pc_probe.py
        frames_client.py  u1_server.py  teleop_loop.py  episode_flow.py
        motion/                               E4  (§7)
          __init__.py  DESIGN.md  keys.py  twist.py  planner.py  controller.py  home.py
        episode/                              E4  (§8)
          __init__.py  DESIGN.md  store.py  recorders.py  meta.py
      ui/                                     E5  (§9)
        __init__.py (NAME = "ui")  __main__.py  server.py  bridge.py  settings.py  scripted_page.py  DESIGN.md
        static/index.html  static/app.js  static/style.css
      launcher/                               E6  (§10)
        __init__.py  cli.py  launcher.py  managed_process.py  processes.py  check_helpers.py  check_result.py
        session.py  errors.py  DESIGN.md
      fake_pc/                                E6  (§10)
        __init__.py  __main__.py  fake_pc.py  settings.py  drifting_clock.py  fake_gate.py  plan_follower.py
        DESIGN.md
    tests/
      conftest.py  station_helpers.py  test_import_rules.py            E1
      contracts/                              E1
      frames/  frames_helpers.py              E2  (test rigs in tests/frames/test_rigs/: the root .gitignore drops data/)
      teleop/  teleop_helpers.py              E3
      motion/  episode/  core_helpers.py      E4
      ui/  ui_helpers.py                      E5
      launcher/  fake_pc/  launcher_helpers.py  E6  (the fake process in tests/launcher/fake_bin/)
      integration/  integration_helpers.py    E7
```

Test folders have no `__init__.py`. Shared helper modules sit in `tests/` with a unique name, as pd:'s tests do
(`--import-mode=importlib`, `pythonpath = ["tests"]`).

## 3. Environment, package, station, config

### 3.1 Environment and package (E1)

- **Environment:** `~/envs/teleop_collect` (Python 3.10, created by L) has `deployment/contracts[test]` installed
  editable, plus `opencv-python-headless==4.12.0.88`, `scipy==1.15.3`, `av==17.1.0` and `aiohttp==3.10.10`. E1 adds
  this package with `~/envs/teleop_collect/bin/pip install -e "deployment/teleop_collect[test]"`.
- **`pyproject.toml`:**

  | Key | Value |
  |---|---|
  | `name` | `teleop-collect` |
  | `version` | `1.0.0` |
  | `requires-python` | `>=3.10,<3.11` |
  | `dependencies` | `numpy>=1.26,<2.3`, `pyyaml==6.0.3`, `opencv-python-headless==4.12.0.88`, `scipy==1.15.3`, `av==17.1.0`, `aiohttp==3.10.10` |
  | `test` extra | `pytest==8.3.5` |
  | script | `teleop_collect = "teleop_collect.launcher.cli:main"` |
  | package data | `teleop_collect.ui` gets `static/*` |
  | pytest | `testpaths ["tests"]`, `pythonpath ["tests"]`, `addopts "--import-mode=importlib"` |

  As in fd, `deploy-contracts` is not listed as a dependency: it is installed editable first.

### 3.2 `station.yaml` (E1; loader `teleop_collect.contracts.station.Station`)

```yaml
station_version: 1
session_root: ~/teleop_collect_sessions   # logs of every run (one folder per run)
data_root: ~/CapstoneTeleop               # the episodes: <data_root>/<task>/episode_NNN/
mock_data_root: ~/CapstoneTeleop_mock     # used instead under --mock, so fake episodes never mix with training data
pi:
  address: 192.168.50.30                  # PLACEHOLDER: the Pi's ASUS_5g address
  port: 5555                              # fixed by the Pi
wifi_ssid: ASUS_5g
ports:                                    # all on 127.0.0.1
  mock_pi: 5650                           # --mock: the fake Pi inside frames
  r1: 5651                                # frames: R1 (teleop -> frames)
  preview: 5652                           # frames: U2 (ui -> frames)
  u1: 5653                                # teleop: U1 (ui -> teleop)
  http: 8765                              # ui: the page, http://127.0.0.1:8765
frames: {}                                # overrides of config/frames_process.yaml
teleop: {}                                # overrides of config/teleop.yaml
ui: {}
fake_pc: {}
launcher: {}
```

- **Copied from pd:contracts/station.py**, with these changes:
  - `PROCESS_SECTIONS = ("frames", "teleop", "ui", "fake_pc", "launcher")`;
  - no `python` section (one environment);
  - `data_root` is required;
  - `PORT_KEYS = ("mock_pi", "r1", "preview", "u1", "http")`, which must be distinct;
  - endpoints `r1_endpoint()`, `preview_endpoint()`, `u1_endpoint()`, `http_url()`, `pi_endpoint()`, `mock_pi_endpoint()`;
  - `data_root` is a `Path` attribute (`~` expanded).
- **Keys refused anywhere in the station:** the keys of the three contract files, as pd: does. Their addresses, ports
  and timing come only from `deployment/contracts/link.yaml`.

### 3.3 Config files (E1 writes them; each process's `settings.py` validates its own)

Merged with the station section as in pd:contracts/config_merge.py: maps merge key by key, and a key that is not in the file
is refused.

**`config/frames_process.yaml`** (pd:'s `frames_process.yaml` minus `select.main_spacing_ms`,
`select.ultrawide_spacing_ms`, `select.tolerance_frames` and `probe`, plus `record`):

```yaml
rig_file: ../contracts/rigs/rig_pi_v3_640.yaml     # the deployment's rig (fd:contracts/rigs/, generated)
mock_episode: ~/Downloads/data/incoming/episode_000152
default_exposure_ns: 5000000
receive: {timeout_s: 0.1, recv_hwm: 60}
buffer: {keep_s: 1.0}                     # ultrawide frames kept for pairing
select: {pair_max_age_ms: 60.0, pair_max_lead_ms: 25.0, pair_max_reuse: 2}
health: {stale_s: 0.35, down_s: 2.0, fps_window_s: 1.0}
clock: {window_s: 60.0, best_fraction: 0.3, drift_min_span_s: 10.0, max_drift_ppm: 500.0, max_sync_age_s: 10.0}
server: {poll_s: 0.02, join_timeout_s: 2.0}
log: {stats_every_s: 10.0}
record:
  queue_max_packets: 90                   # 3 s of packets waiting for the writer; beyond that the recording fails
  clock_log_every_s: 1.0                  # clock_pi.csv row period
  playback_fps: 30                        # the .mkv timestamps (playback only; true times are in the CSV)
mock: {max_seconds: 10.0, jpeg_quality: 90, pi_clock_offset_s: 1000.0, drift_ppm: 20.0, send_hwm: 30}
check: {reachable_timeout_s: 1.0, subprocess_timeout_s: 10.0}
```

**`config/teleop.yaml`:**

```yaml
motion:                                   # PLAN 5.2-5.3
  v_max_m_s: 0.05                         # per axis, each held translation key
  w_max_deg_s: 10.0                       # the twist, and the tilt (both tilt axes together, normalised)
  slow_factor: 0.25                       # Shift
  waypoints: 2                            # K
  spacing_ms: 50                          # Delta; the lead is K x Delta = 100 ms
  key_stale_ms: 150                       # no key state for this long = all keys released
  state_max_age_ms: 50                    # no motion while the newest C2 is older (mapped)
  assumed_pc_box_margin_m: 0.01           # the PC's safety.box_margin_m (fd:control_deploy/franka_control/config.yaml)
  assumed_pc_cone_margin_deg: 2.0         # the PC's safety.cone_margin_deg
  margin_use: 0.8                         # settings are refused if the worst step exceeds this share of a margin
home:
  on_start: true                          # PLAN 8: home the whole arm once at start-up
  tolerance_rad: 0.01                     # every joint within this of the home reference
  tolerance_m: 0.005                      # and the TCP target position within this (the fake PC reports q = 0)
history_s: 1.0
state_recv_hwm: 1000
listener_poll_s: 0.02
link_jitter_allowance_s: 0.2
health_poll_ms: 500
u1_max_per_cycle: 4                       # U1 requests served per cycle
at_limit_hold_ms: 500                     # "at workspace limit" shown this long after gate.pulled_back rises
messages_kept: 200                        # page messages kept in memory
timeouts:
  health_s: 0.05                          # R1 get_health (a term of the worst C3 gap)
  frames_s: 4.0                           # R1 start_recording / stop_recording (frames' stop drains up to 1 + 2 s); 1.0 s on aborts
  home_s: 90.0                            # C3 home; above the PC's homing_timeout_s (hello)
  startup_s: 20.0                         # start-up, not counting the waits while C2 shows homing or the home itself
  pc_homing_wait_s: 60.0
clock:                                    # pd:'s config/manager.yaml clock block, unchanged
  hello_period_s: 0.5
  startup_hello_period_s: 0.02
  round_trip: {window_s: 10.0, best_fraction: 0.3}
  one_way: {window_s: 5.0, best_fraction: 0.3}
  drift_min_span_s: 5.0
  max_drift_ppm: 200.0
  min_samples_s: 2.0
  min_samples: 10
  max_sample_age_s: 5.0
  startup_agree_ms: 5.0
  jump_log_ms: 1.0
  mismatch_sets: 3
  mismatch_min_lead_ms: 60.0              # teleop's lead is 100 ms (pd: 100 ms for a policy's 350 ms)
  future_state_ms: 20.0
  log_every_s: 10.0
link_stats: {window_s: 10.0}
```

**`config/ui.yaml`:**

```yaml
status_hz: 10            # status pushed to the page
preview_hz: 15           # previews pulled from frames and pushed to the page
keys_resend_ms: 50       # the page resends its key state this often while it has focus
keys_stale_ms: 150       # the server sends "no keys" to teleop when the page is silent this long
rpc_timeout_s: 0.5       # U1 / U2 request timeout
open_browser: true       # teleop_collect run opens the page (also --no-browser)
```

**`config/fake_pc.yaml`:** copied from fd:policy_deploy/config/fake_pc.yaml, unchanged.

**`config/launcher.yaml`:**

```yaml
stop_timeout_s: 5.0
poll_s: 0.1
fake_pc: true                       # --mock: start the fake PC (false for the cross test)
ready_timeout_s: {fake_pc: 10.0, frames: 30.0, teleop: 150.0, ui: 15.0}
                                    # teleop: startup 20 + pc_homing_wait 60 + home_s 90 cannot all happen, the
                                    # longest legal is 20 + 60 + 30 (the PC's homing timeout) + 5 = 115 s
```

## 4. Laptop contracts (E1): `teleop_collect.contracts`

- **Copied with only the import changes:** `pi_stream.py`, `c0_packet.py` and `clock_fit.py`.
- **`health.py`:** `CameraHealthReport` and `HealthReport` (Records) from pd:contracts/c1_frames.py, without the C1
  messages.
- **`paths.py`:** `folder()` (`deployment/teleop_collect/`), `config_file(name)`, `rigs_dir()`, `default_station()`,
  `deployment_dir()`, with the install hint `teleop_collect must be installed editable from deployment/teleop_collect`.
- **`process_args.py`:** `PROCESS_NAMES = ("frames", "teleop", "ui", "fake_pc")`.
- **`__init__.py`:** re-exports every public name below and registers every schema, so importing
  `teleop_collect.contracts` registers R1/U1/U2 (as pd: does for C1/C4).

All new messages use `deploy_contracts.Message` / `Record` and its field helpers. Commands are answered with `ack/v1`
(`deploy_contracts.Ack`), or `error/v1` when broken.

### 4.1 `keys.py`

```python
MOTION_KEYS = ("x+", "x-", "y+", "y-", "z+", "z-", "twist+", "twist-",
               "tilt_fwd", "tilt_back", "tilt_left", "tilt_right")
GRIPPER_COMMANDS = ("close", "open")
OUTCOMES = ("success", "fail", "discard")
TASK_NAME_RE = r"^[a-z0-9_]{1,64}$"
EPISODE_NAME_RE = r"^episode_[0-9]{3,}$"
```

The page's keyboard map (the ui's alone; teleop only sees these names):

| Keys | Name |
|---|---|
| W / S | `x+` / `x-` |
| A / D | `y+` / `y-` |
| R / F | `z+` / `z-` |
| U / O | `twist+` / `twist-` |
| I / K | `tilt_fwd` / `tilt_back` |
| J / L | `tilt_left` / `tilt_right` |
| N / M | gripper `close` / `open` |
| Shift | `slow` |
| Esc | `stop_arm` |

### 4.2 R1: teleop → frames (`r1_recorder.py`), REP on `station.r1_endpoint()`

| Schema | Fields | Reply |
|---|---|---|
| `r1.start_recording/v1` | `episode_dir: str` (absolute, existing, empty of frames files) | `ack/v1`: ok, or refused (`already recording`, `<camera> camera <status>`, `Pi clock not synced`, `<dir>: not a folder`, `<dir>: already has recording files`) |
| `r1.stop_recording/v1` | none | `r1.summary/v1` (refused as `error/v1` when not recording) |
| `r1.get_health/v1` | none | `r1.health/v1` |

`CameraSummary` (Record):

| Field | Type |
|---|---|
| `frames` | uint64, frames written |
| `frame_no_gaps` | uint64 |
| `bytes` | uint64 |
| `first_t_mid_ns`, `last_t_mid_ns` | optional int64, laptop |

`RecordingSummary`, `r1.summary/v1`:

| Field | Type |
|---|---|
| `main`, `ultrawide` | `CameraSummary` |
| `packets` | uint64 |
| `packet_no_gaps` | uint64 |
| `uw_reused` | uint64 |
| `pair_refused` | uint64 |
| `error` | str, `""` unless the writer failed |
| `pi_clock_offset_ns` | optional int64 |
| `pi_clock_drift_ppm` | optional float64 |

`FramesHealth`, `r1.health/v1`:

| Field | Type |
|---|---|
| `health` | `HealthReport` |
| `synced` | bool, the Pi clock map is fresh |
| `recording` | bool |
| `recording_error` | str, `""` or why the writer failed (queue full, disk) |
| `frames_main`, `frames_ultrawide` | uint64, written in this recording |

### 4.3 U2: ui → frames (`u2_preview.py`), REP on `station.preview_endpoint()`

| Schema | Fields | Reply |
|---|---|---|
| `u2.get_preview/v1` | `main_after: int64 (min -1)`, `ultrawide_after: int64 (min -1)`: the newest `frame_no` the ui already has | `u2.preview/v1` |

`Preview`, `u2.preview/v1`:

| Field | Type |
|---|---|
| `main_jpeg`, `ultrawide_jpeg` | optional bytes: the newest JPEG exactly as received, only when its `frame_no` is above `*_after` |
| `main_frame_no`, `ultrawide_frame_no` | int64, the newest, -1 if none yet |
| `health` | `HealthReport` |

### 4.4 U1: ui → teleop (`u1_teleop.py`), REP on `station.u1_endpoint()`

| Schema | Fields | Reply |
|---|---|---|
| `u1.get_status/v1` | `after_message: uint64` | `u1.status/v1` |
| `u1.keys/v1` | `held: str_list(MOTION_KEYS)`, `slow: bool`, `focus: bool` | `ack/v1` (always ok; ignored outside an episode) |
| `u1.gripper/v1` | `command: str (GRIPPER_COMMANDS)` | `ack/v1` |
| `u1.select_task/v1` | `name: str` | `ack/v1` |
| `u1.create_task/v1` | `name: str`, `description: str` | `ack/v1` (also selects it) |
| `u1.home/v1` | none | `ack/v1`: ok = homing started (the outcome shows in status) |
| `u1.start_episode/v1` | none | `ack/v1` |
| `u1.stop_episode/v1` | `outcome: str (OUTCOMES)` | `ack/v1` |
| `u1.delete_episode/v1` | `episode: str (EPISODE_NAME_RE)` | `ack/v1` |
| `u1.stop_arm/v1` | none | `ack/v1` |

Records:

| Record | Fields |
|---|---|
| `EpisodeInfo` | `name: str`, `outcome: str (success, fail)`, `duration_s: float64`, `frames_main: uint64` |
| `UiMessage` | `id: uint64`, `level: str (info, warn, error)`, `text: str` |

`TeleopStatus`, `u1.status/v1`:

| Field | Type | Meaning |
|---|---|---|
| `phase` | str: `starting`, `idle`, `recording`, `homing`, `stopping`, `done` | |
| `ready` | bool | start-up finished |
| `task`, `task_description` | str | `""` when none selected |
| `tasks` | str_list | every task under `data_root` |
| `episodes` | list of `EpisodeInfo` | the selected task's complete episodes, oldest first |
| `recording_episode` | str | `""` unless recording |
| `episode_s` | float64 | time since Start, 0 when not recording |
| `arm_mode` | str | newest C2 mode, `""` before any |
| `tcp_m` | optional float64 (3,) | newest C2 `O_T_TCP` position |
| `gripper_width_m` | optional float64 | newest C2 width |
| `commanded_width_m` | float64 | |
| `home_known`, `at_home` | bool | |
| `start_allowed` | bool | |
| `start_refusal` | str | why not, `""` when allowed |
| `at_limit` | bool | gate `pulled_back` rose within `at_limit_hold_ms` |
| `moving` | bool | a motion key is in effect |
| `slow` | bool | |
| `cameras` | optional `HealthReport` | the newest R1 health |
| `c2_hz` | float64 | |
| `rtt_ms` | optional float64 | |
| `clock_ok` | bool | |
| `pc_mock` | bool | |
| `messages` | list of `UiMessage` | those with `id > after_message`, at most 50, oldest first |
| `final_line` | str | the one-line reason when `phase` is `done` |

Validation adds: `episode_s >= 0`; `messages` ids strictly increasing; `start_refusal == ""` exactly when `start_allowed`.

### 4.5 `examples.py`

`build_examples() -> list[Example]` with the helpers of `deploy_contracts.examples`. It holds at least one good example per
schema of §4.2–4.4, and broken ones for:
- an unknown key name;
- an unknown outcome;
- a bad episode name;
- `main_after < -1`;
- non-increasing message ids;
- `start_refusal` set while allowed.

They are built in memory, never written to `deployment/contracts/examples/`.

## 5. frames (E2)

### 5.1 Threads and objects

| Thread | Runs |
|---|---|
| receiver | `Receiver.run`: `PiStreamPacketSource` (C0 SUB, `OneWayClockMap`) → checks → `CameraHealth` → `LatestFrames` (preview) → `FrameRecorder.enqueue` while recording |
| writer | `FrameRecorder`'s writer: takes `RecordItem`s from a bounded queue (`record.queue_max_packets`) and writes the two `.mkv` files, `packets.csv`, `main_frames.csv`, `ultrawide_frames.csv`, `clock_pi.csv` |
| main | `FramesServers`: polls the R1 and the U2 REP sockets every `server.poll_s`; the ready file; the stats line; watches for a FATAL |
| fake-pi (`--mock`) | `FakePiSender` on `station.mock_pi_endpoint()` |

**Adapted receiver** (pd:frames/receiver.py). Images are never decoded:
- `_take_frame` keeps the frame-gap counting and returns the `EncodedFrame`;
- the size check uses the JPEG header (pd:contracts/pi_stream.py `jpeg_size`);
- `UltrawideStore` holds `EncodedFrame`s;
- `PacketBuffer`, `PacketSelector`, `ImagePrep` and the decoders are dropped. The out-of-order check compares with the
  last accepted packet time.

Each accepted packet produces one `RecordItem`, handed to the preview and, while recording, to the recorder:

```python
@dataclass(frozen=True)
class RecordItem:
    packet_no: int; t_send_pi_ns: int; arrival_ns: int
    main: EncodedFrame; main_t_mid_pi_ns: int; main_t_mid_ns: int
    ultrawide: EncodedFrame | None          # only when its frame_no is new (a reused one is None)
    ultrawide_t_mid_pi_ns: int | None; ultrawide_t_mid_ns: int | None
    ultrawide_frame_no: int; uw_reused: bool; pair_used: bool; pair_gap_ms: float | None
```

Packets before the Pi clock is synced are not recorded (start is refused until it is), and still update health.

### 5.2 Recorder (`recorder.py`, `mkv_writer.py`)

```python
class MkvWriter:                       # one camera's .mkv; JPEG bytes muxed as av.Packet, never re-encoded
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None
    def write(self, jpeg: bytes) -> int        # returns the video_index (0, 1, ...); pts = index x 1000/fps ms
    def close(self) -> None
class FrameRecorder:
    def __init__(self, queue_max: int, clock_log_every_s: float, sizes: Mapping[str, tuple[int, int]], fps: int,
                 clock_map: ClockMap, clock: Callable[[], int] = time.monotonic_ns) -> None
    def start(self, episode_dir: Path) -> None          # opens files, starts the writer; RecordingRefused on errors
    def enqueue(self, item: RecordItem) -> None          # never blocks; a full queue sets error and stops accepting
    def stop(self) -> RecordingSummary                   # drains, closes every file, joins the writer
    @property
    def recording(self) -> bool
    @property
    def error(self) -> str                               # "" or the first failure
```

- The stream is `mjpeg`, `yuvj420p`, at the rig size, with `time_base` 1/1000. This was checked to round-trip the bytes exactly.
- **Column order is exactly PLAN §6.2.**
  - `main_frames.csv` gets one row per main frame, and `ultrawide_frames.csv` one row per *new* ultrawide frame.
    `video_index` is the frame's position in its `.mkv`.
  - `packets.csv` gets one row per recorded packet.
  - `clock_pi.csv` gets one row per `clock_log_every_s`: `t_ns`, `offset_ns`, `drift_ppm`, `samples`, `used`.
    The raw one-way samples are in `packets.csv` (`t_send_pi_ns`, `arrival_ns`).
- Files are flushed at least once per second.

### 5.3 Preview and servers

- **`preview.py`:** `LatestFrames.update(item)` and `LatestFrames.get(main_after, ultrawide_after) -> Preview`,
  under a lock.
- **`servers.py`:** `FramesServers(r1_endpoint, preview_endpoint, recorder, latest, health, clock_map)`, two
  `deploy_contracts.RpcServer`s with `serve_once`.
  - `start_recording` refuses if:
    - a recording is already running;
    - either camera is not `ok`;
    - the clock map is not fresh;
    - the folder is missing or already has recording files.
  - `get_health` is always answered.
- **Ready** (the ready file): both cameras `ok` once, as in pd:.
- **Stats line:** every `log.stats_every_s`, as in pd:, plus `recording` and `queue`.

### 5.4 Entry points

- `python -m teleop_collect.frames [--station] [--session] [--ready-file] [--mock]` runs the process.
- `python -m teleop_collect.frames.pi_collect --station station.yaml --seconds 30`: pd:'s tool, adapted.
- `python -m teleop_collect.frames.rig_file --json RIG`: pd:'s, for the launcher check.

## 6. teleop (E3)

### 6.1 Threads

| Thread | Runs | Owns |
|---|---|---|
| main | `TeleopLoop`: one cycle every `heartbeat_period_s`. All decisions, **every C3 request** (rule H1), R1 calls, U1 serving | `PcClient` (C3 REQ), `FramesClient` (R1 REQ), `U1Server` (U1 REP) |
| pc-listener | `PcListener`: every C2 into `StateHistory`, `PcClock`, `LinkStats`, and, while recording, `StateRecorder.write` | C2 SUB |

`PcListener` (adapted pd:) gets `set_recorder(recorder: StateRecorder | None)`. The hook is swapped under a lock and
called with `(state, arrival_ns, t_ns_mapped | None)`. The mapped laptop time comes from
`PcClock.snapshot().to_laptop_ns(t_pc_ns, arrival_ns)`, or `None` if not synced.

### 6.2 Copied and adapted (from pd:manager/)

| File | Changes |
|---|---|
| `pc_client.py`, `pc_clock.py`, `link_stats.py`, `state_history.py`, `heartbeat_sender.py`, `errors.py` | imports, process name `teleop` |
| `pc_listener.py` | the recorder hook (§6.1) |
| `settings.py` | `config/teleop.yaml` + `link.yaml` (§3.3); `MotionSettings`, `HomeSettings` objects passed to E4 (§7); the margin rule (§7.3) refused at load |
| `problems.py`, `failure_monitor.py` | the kinds and codes of §6.5 |
| `startup_checks.py` | no `settled` rule, no policy; `waypoint_lead_ms` = K × Δ; `worst_gap` as pd: |
| `startup_sequence.py` | PLAN §8 (§6.4) |
| `session_log.py` | session.yaml, events.jsonl, clock.jsonl, link.jsonl under `<session>/teleop/`; no steps, tap or policy |
| `pc_probe.py` | unchanged behaviour, imports; `--check --json` for the launcher |

### 6.3 One cycle (`teleop_loop.py`)

The order is fixed:
1. Signal flags (Ctrl-C → conclude STOP; SIGTERM → launcher stop).
2. State rules (`FailureMonitor.check_state`) and listener errors.
3. Key state: `KeyState.current(now)` (§7.1).
4. **Exactly one C3 request:**
   - `hello`, if `HeartbeatSender.hello_due(now)`;
   - otherwise the set from `MotionController.step(...)` (§7.4), when it returns one (`PcClient.set_waypoints`);
   - otherwise a heartbeat.
   - A set's reply is checked as pd: does (`chunk_id`, the sum, `rejected` → conclude `GATE_REJECTED`). The clock-mismatch
     guard uses `clock.mismatch_*` (§3.3).
5. R1 health every `health_poll_ms`.
6. Up to `u1_max_per_cycle` U1 requests (§6.6).
7. Status bookkeeping (`at_limit` from `gate.pulled_back`) and periodic session-log rows.

`chunk_id` counts up from 1 for the whole run.

### 6.4 Phases, start-up and the episode flow (`startup_sequence.py`, `episode_flow.py`)

**Phases:** `STARTING → IDLE ⇄ RECORDING`, `IDLE → HOMING → IDLE`, `RECORDING → STOPPING → IDLE`, any → `DONE`.

**Start-up** (PLAN §8):
1. `TaskStore.clean_incomplete()`, each deletion logged.
2. C2 arriving.
3. The H2 wait.
4. Hellos until the clock map is ready.
5. `compare_hello` and the budget checks.
6. `get_state` shows `idle` or `stopped`.
7. Heartbeats.
8. If `home.on_start`: the C3 home (§6.7), then hellos every cycle until the map is ready again.
9. R1 health shows both cameras `ok`.
10. `data_root` writable.
11. `session.yaml`, the ready file, phase IDLE.

`timeouts.startup_s` covers steps 2–10, not counting the H2 / homing waits or the home itself.

**Start allowed** (`start_refusal` text in brackets, the first that applies):
- phase IDLE, not done (`starting up` / `recording` / `homing`);
- a task selected (`select or create a task first`);
- C2 mode `idle` or `stopped` (`the PC is <mode>`);
- home known and at home (`home the arm first (Home, or h on the PC)`);
- both cameras `ok` (`<camera> camera <status>`);
- the clock map ready (`waiting for the PC clock map`).

**Start:**
1. `Task.new_episode()`.
2. Open `EpisodeRecording` (§8.2).
3. R1 `start_recording(dir)`. A refusal discards the folder and shows the reason.
4. C3 `start`. A refusal: R1 `stop_recording`, discard, show the reason.
5. Commanded width = newest C2 width.
6. Phase RECORDING.

**Stop** (U1 `stop_episode(outcome)`):
1. Phase STOPPING.
2. `MotionController.stop()`.
3. C3 `stop`.
4. R1 `stop_recording` → summary.
5. Close the recorders.
6. `outcome == discard` → `EpisodeDir.discard()`; otherwise `EpisodeDir.finish(build_meta(...))`.
7. Phase IDLE.

**Abort** (an EPISODE problem, §6.5): as Stop, with the folder discarded. The reason goes to the page messages and
`events.jsonl` of the session log.

**Conclude:** C3 `stop` (if heartbeats started). An open episode is aborted (discarded). Then the FATAL line, exit 1, or 0
for STOP.

### 6.5 Problem kinds (adapted `problems.py`)

`Kind`: `NORMAL`, `SLOW`, **`EPISODE`** (new: abort the open episode, stay up), `CONCLUDE`, `STOP`.
- **Kept, with pd:'s kinds:**
  - `STATE_GAP`, `STATE_OLD`, `CLOCK_STALE` (SLOW);
  - `PULLED_BACK`, `SHORTENED`, `LOOP_LATE`, `HOME_STOPPED` (NORMAL);
  - `PC_NOT_ANSWERING`, `STATE_LOST`, `PC_RESTARTED`, `ARM_FAULT`, `GATE_REJECTED`, `CLOCK_MISMATCH`,
    `HELLO_MISMATCH`, `PC_BUSY`, `CONTRACT_VIOLATION`, `HOMING_FAILED`, `STARTUP`, `NO_ANSWER`, `INTERNAL_ERROR`
    (CONCLUDE);
  - `OPERATOR_STOP`, `LAUNCHER_STOP` (STOP).
- **Changed or new:**
  - `CAMERA_STALE` (SLOW);
  - `CAMERA_DOWN` (**EPISODE** while recording, NORMAL otherwise);
  - `PC_STOPPED` (**EPISODE**);
  - `OPERATOR_ABORT` (EPISODE, Stop arm / Esc);
  - `RECORDING_FAILED` (EPISODE, R1 `recording_error`, or a recorder write error);
  - `SET_SKIPPED` (SLOW, a set skipped entirely).
- **Dropped:** the policy codes and `TCP_UNSETTLED`.

### 6.6 U1 server (`u1_server.py`)

A `deploy_contracts.RpcServer` on `station.u1_endpoint()`, served from the main loop only (`serve_once(0)`, up to
`u1_max_per_cycle` per cycle). The handlers are methods of `TeleopLoop` / `EpisodeFlow`.

| Request | Handling |
|---|---|
| `keys` | goes into `KeyState` |
| `gripper` | queued for the next cycle's `MotionController` |
| `home` | starts the home in the next cycle |
| `stop_arm` | Abort with `OPERATOR_ABORT` when recording, otherwise C3 `stop` |
| `get_status` | builds `TeleopStatus` |

Refusals are `ack(ok=false, reason)` in plain words.

### 6.7 Home

- **Sending it:** `PcClient.home(interrupted)` waits in `heartbeat_period_s` slices. The `interrupted` callback E3
  passes also serves U1:
  - `get_status` (phase HOMING);
  - `keys` (ignored);
  - everything else refused with `homing: wait, or press s on the PC (the user stop is always there)`.

  It returns the Ctrl-C/SIGTERM flag.
- **Outcomes:** as pd:, i.e. ok, `stopped:` (NORMAL), refused because the PC is homing (hint), anything else concludes
  `HOMING_FAILED`.
- **After any home:** hellos every cycle until the map is ready again. A successful home sets
  `HomeReference.set_from(q, T_target)` from the first `idle` C2 after it (§7.5).

### 6.8 Who E3 calls in E4

`KeyState`, `MotionController`, `HomeReference` (§7) and `TaskStore`, `EpisodeRecording`, `build_meta` (§8). Exactly
the signatures below. In its own tests E3 may use the real E4 classes once they exist, or small fakes with the same
signatures.

## 7. teleop core: motion (E4), pure logic, no sockets, no threads

### 7.1 `keys.py`

```python
@dataclass(frozen=True)
class HeldKeys:
    held: frozenset[str]          # subset of MOTION_KEYS
    slow: bool
class KeyState:
    def __init__(self, stale_ms: float) -> None
    def update(self, held: Sequence[str], slow: bool, focus: bool, now_ns: int) -> None
    def current(self, now_ns: int) -> HeldKeys          # empty when not focus, or no update for stale_ms
    def changes(self) -> list[tuple[int, HeldKeys, bool]]   # (t_ns, keys, focus) since the last call, for keys.csv
```

### 7.2 `twist.py`

```python
@dataclass(frozen=True)
class Twist:
    v_m_s: np.ndarray             # (3,) base frame
    w_twist_rad_s: float          # about the TCP's own z axis (the approach axis)
    w_tilt_rad_s: np.ndarray      # (2,) about the base x and y axes, through the TCP point
    def is_zero(self) -> bool
def twist_from_keys(keys: HeldKeys, s: MotionSettings) -> Twist
```

Each axis is ±`v_max` per held key, and opposite keys cancel. Slow multiplies everything by `slow_factor`.
- **Tilt by effect on the TCP z axis** (pointing down at home):
  - `tilt_fwd` moves it toward +x, i.e. negative rotation about base y;
  - `tilt_back` is the opposite;
  - `tilt_left` moves it toward +y, i.e. positive rotation about base x;
  - `tilt_right` is the opposite.
- **The tilt vector is normalised to at most `w_max`** when both tilt axes are held, so the cone bound holds.
- `twist+` is a positive (right-hand) rotation about the TCP z.

### 7.3 `planner.py`

```python
@dataclass(frozen=True)
class Waypoint:
    t_pc_ns: int; O_T_TCP: np.ndarray; width_m: float     # (4, 4) row-major
def advance(T_base: np.ndarray, twist: Twist, tau_s: float) -> np.ndarray
    # p = p_base + v * tau;  R = Rot(base, [w_tilt_x, w_tilt_y, 0] * tau) @ R_base @ Rot_z(w_twist * tau)
class WaypointPlanner:
    def __init__(self, s: MotionSettings) -> None
    def move(self, T_base, t_base_pc_ns, twist, now_pc_ns, width_m) -> list[Waypoint]
        # K waypoints at now_pc + k*Delta (k = 1..K), each advance(T_base, twist, (t_k - t_base)/1e9)
    def release(self, T_base, t_base_pc_ns, last_twist, now_pc_ns, width_m) -> list[Waypoint]
        # one waypoint at now_pc + Delta: advance(T_base, last_twist, (now_pc - t_base)/1e9)
    def hold(self, T_base, now_pc_ns, width_m) -> list[Waypoint]
        # one waypoint at now_pc + Delta at T_base (a finger command with no motion)
def worst_step(s: MotionSettings) -> tuple[float, float]
    # (m, deg): v_max * (K*Delta + state_max_age), w_max * (same), unscaled (slow only makes it smaller)
```

`MotionSettings` (built by E3's `settings.py`, defined here in `planner.py`): the `motion:` keys of §3.3, plus
`heartbeat_period_s`. Its constructor refuses settings where `worst_step` exceeds `margin_use` × the assumed PC margins.

### 7.4 `controller.py`

```python
@dataclass(frozen=True)
class BaseState:
    T_target: np.ndarray; t_pc_ns: int; width_m: float; mode: str   # from the newest C2
@dataclass(frozen=True)
class MotionCommand:
    kind: str                     # "move" | "release" | "fingers"
    waypoints: tuple[Waypoint, ...]
    twist: Twist
    slow: bool
class MotionController:
    def __init__(self, s: MotionSettings, width_open_m: float) -> None
    def begin(self, commanded_width_m: float) -> None      # at Start
    def stop(self) -> None                                 # at Stop / Abort: no more commands
    def gripper(self, command: str) -> None                # "close" -> 0.0, "open" -> width_open_m
    def step(self, now_pc_ns: int | None, base: BaseState | None, keys: HeldKeys) -> MotionCommand | None
    @property
    def commanded_width_m(self) -> float
    @property
    def moving(self) -> bool
```

`step` rules, in order:
1. Not begun / stopped → `None`.
2. `now_pc_ns` or `base` missing, or `now_pc − base.t_pc > state_max_age_ms` → `None`. Nothing is sent; the arm ends
   its last plan within the lead and holds. *(This refines PLAN §5.3's "holding set": a release built on a stale
   state could stick out further than the margin, so nothing is sent at all.)*
3. A non-zero twist → `move`.
4. The twist became zero after a move → one `release`.
5. A pending finger command → `fingers` (`hold`), with no motion key held.
6. Otherwise `None`.

Also: no command unless `base.mode` is `running` or `holding` (the PC refuses `set_waypoints` otherwise); a
`release` never advances past the end of the last plan sent (`WaypointPlanner.release(..., until_pc_ns)`); `begin()`
clips the start width into [0, `width_open_m`].

When a finger command and motion come together, the new width rides on the move's waypoints.

### 7.5 `home.py`

```python
class HomeReference:
    def __init__(self, s: HomeSettings) -> None
    def set_from(self, q: np.ndarray, T_target: np.ndarray) -> None   # after a home that ended in idle
    def observe(self, mode: str, q: np.ndarray, T_target: np.ndarray) -> None
        # a C2 homing -> idle transition (the PC's h) sets it, only while none is known and only if the arm moved
        # (joints > tolerance_rad or TCP > tolerance_m) during that homing, so the PC's g never counts
    @property
    def known(self) -> bool
    def at_home(self, q: np.ndarray, T_target: np.ndarray) -> bool
        # all |q - q_ref| <= tolerance_rad and |p - p_ref| <= tolerance_m
```

## 8. teleop core: episode (E4), files only

### 8.1 `store.py`

```python
class TaskError(ValueError): ...
@dataclass(frozen=True)
class EpisodeInfo:
    name: str; outcome: str; duration_s: float; frames_main: int
class TaskStore:
    def __init__(self, data_root: Path) -> None
    def tasks(self) -> list[str]                           # sorted names of folders with a valid task.yaml
    def create(self, name: str, description: str, settings: Mapping) -> Task   # TaskError: bad name, exists
    def open(self, name: str) -> Task                      # TaskError: missing
    def clean_incomplete(self) -> list[Path]               # deletes every episode_* folder without COMPLETE
class Task:
    name: str; description: str; dir: Path
    def episodes(self) -> list[EpisodeInfo]                # complete ones, by number
    def new_episode(self) -> EpisodeDir                    # next number from task.yaml next_episode (never reused)
    def delete(self, episode: str) -> None                 # TaskError: missing or not complete
class EpisodeDir:
    name: str; path: Path
    def finish(self, meta: Mapping) -> None                # writes meta.yaml, then COMPLETE (fsync both)
    def discard(self) -> None                              # removes the folder; refuses any path outside its task
```

- **`task.yaml`:** `name`, `description`, `created` (ISO time), `next_episode` (int, updated atomically: write a temp
  file, then rename), `settings`.
- **Episode names:** `episode_%03d`, more digits beyond 999.

### 8.2 `recorders.py`

```python
class EpisodeRecording:                 # opened by Start, closed by Stop/Abort; owns the files below
    def __init__(self, episode: EpisodeDir, waypoints: int = 2) -> None   # K sizes commands.csv's slots
    states: StateRecorder               # robot_states.csv; write(state: ArmState, arrival_ns, t_ns | None); thread-safe
    commands: CommandRecorder           # commands.csv; write(t_send_ns, t_reply_ns, request, msg, reply, twist, slow)
    keys: KeyRecorder                   # keys.csv; write(t_ns, keys: HeldKeys, focus)
    pc_clock: PcClockRecorder           # clock_pc.csv; hello(sample) per hello, fit(t_ns, offset_ns, drift_ppm, ...)
    events: EventRecorder               # events.jsonl; write(t_ns, kind, detail)
    def counts(self) -> dict[str, int]  # c2_messages, c2_seq_gaps, sets_sent, sets_all_skipped, shortened, ...
    def close(self) -> None             # idempotent
```

- **Columns:** exactly PLAN §6.2, row-major 4×4 as 16 columns named `O_T_TCP_00` … `O_T_TCP_33`, then `q_0..q_6`,
  `dq_0..dq_6`. `commands.csv` gets one row per request, the waypoints flattened `wp<k>_t_pc_ns`, `wp<k>_00..33`,
  `wp<k>_width_m` for k < K.
- Files are flushed at least once per second.

### 8.3 `meta.py`

```python
def build_meta(*, task: Task, episode: EpisodeDir, outcome: str, times: Mapping[str, int], frames: RecordingSummary,
               counts: Mapping[str, int], hello: HelloReply, laptop_commit: str, laptop_dirty: bool,
               rig_file: Path, motion: MotionSettings, pi_fit: Mapping, pc_fit: Mapping) -> dict
```

It returns the PLAN §6.3 content as plain YAML-able types. `hello` comes complete, `F_T_TCP` as a 4×4 list.

## 9. ui (E5)

### 9.1 Server

- **`server.py`:** an aiohttp app bound to `127.0.0.1:<ports.http>` only (never another host).
  - `GET /` serves `static/index.html`, plus the static files.
  - `GET /ws` is the WebSocket.
- **`bridge.py`** owns the two REQ clients, a `U1Client` and a `U2Client` (`deploy_contracts.RpcClient`, timeout
  `rpc_timeout_s`), on worker threads (asyncio `run_in_executor`). The event loop never blocks.
- **The deadman:** when the page is silent for `keys_stale_ms`, or its socket closes, the server sends
  `keys(held=[], slow=false, focus=false)`. teleop has its own `key_stale_ms` too.
- **Ready** (the ready file): bound and U1 answered `get_status` once.
- **A U1 timeout or error** shows `teleop not answering` on the page. The ui never exits on its own; it ends with the
  launcher.

### 9.2 WebSocket protocol (internal to E5, used by `ScriptedPage`)

| Direction | Message |
|---|---|
| server → page | `{"type": "status", ...TeleopStatus fields...}` at `status_hz` |
| server → page | `{"type": "preview", "camera": "main"\|"ultrawide", "frame_no": N}` followed by one binary message (the JPEG) |
| page → server | `{"type": "keys", "held": [...], "slow": b, "focus": b}` on every change and every `keys_resend_ms` |
| page → server | `{"type": "gripper", "command": "close"\|"open"}` on key down (no repeat) |
| page → server | `{"type": "command", "name": "home"\|"start_episode"\|"stop_arm"\|"select_task"\|"create_task"\|"stop_episode"\|"delete_episode", "args": {...}, "id": n}` (the arguments in their own map, so a task `name` cannot clash with the command `name`) |
| server → page | `{"type": "hello", ...}` on connect (the rates and key names), `{"type": "error", "text": ...}` for a message it cannot use |
| server → page | `{"type": "ack", "id": n, "ok": b, "reason": str}` |

The server accepts a request only with a `Host` of `127.0.0.1:<port>` or `localhost:<port>`, and a WebSocket only from that
same `Origin`. That way another web site open in the browser cannot reach the arm through the page.

### 9.3 The page (`static/`)

- Plain HTML, CSS and JS, no build step and no CDN: it must work offline in the lab.
- The layout of PLAN §4.3, with buttons Home, Start episode, Stop arm, and Save success / Save fail / Discard after Stop.
- A delete with a confirm, the task picker and New task.
- **The keyboard map of §4.1.** `keydown` / `keyup` maintain the held set, ignoring auto-repeat.
- **Losing the keys:** `blur` and `visibilitychange` send an empty held set with `focus:false`.
- Keys work only while the page has focus and the phase is `recording`.
- Shows `REC episode_NNN m:ss`, the `start_refusal` text, the messages, and `at workspace limit`.

### 9.4 `scripted_page.py` (used by E7)

```python
class ScriptedPage:                     # speaks the WebSocket protocol exactly as app.js does
    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None
    def connect(self) -> None; def close(self) -> None
    def status(self) -> dict                                   # the newest status message
    def wait_for(self, pred: Callable[[dict], bool], timeout_s: float, what: str) -> dict
    def command(self, name: str, **args) -> dict               # sends, returns the ack {"ok", "reason"}
    def hold(self, keys: Sequence[str], seconds: float, slow: bool = False) -> None   # key frames like the page
    def gripper(self, command: str) -> None
    def blur(self) -> None
    def previews(self) -> dict[str, int]                       # newest frame_no received per camera
```

## 10. launcher and fake PC (E6)

### 10.1 Launcher (adapted from pd:launcher/)

- **Command:** `teleop_collect check | run [--mock] [--station FILE] [--no-browser]`.
- **`ORDER`:** `("fake_pc", "frames", "teleop", "ui")`; `fake_pc` only with `--mock` and `launcher.fake_pc`.
- **Each process** is started as `python -m teleop_collect.<name> --station S --session D --ready-file F [--mock]` with
  `sys.executable`. Its log goes to `<session>/logs/<name>.log`.
- **On ready:** `teleop_collect: ready: open <station.http_url()>`, plus `webbrowser.open` unless `--no-browser` or
  `ui.open_browser: false`.
- **Failures:** one line, `teleop_collect: FAILED <process> ...: <FATAL line>`.
- **Station rows:**
  - `station.yaml` loads;
  - `deploy_contracts` comes from this checkout's `deployment/contracts` (otherwise FAIL with the install hint);
  - the three contract files load;
  - the laptop address is in `link.yaml laptop.addresses` (not under `--mock`);
  - the Wi-Fi SSID (WARN);
  - the commit (INFO), the PC's commit is a WARN only if it differs and the PC answered;
  - free disk under `data_root` (WARN under 2 GB);
  - `data_root` writable.
- **Process rows:**
  - fake_pc: ports free;
  - frames: the rig file subprocess check, Pi reachable (not `--mock`), mock episode present (`--mock`), R1/preview
    ports free;
  - teleop: the U1 port free, `python -m teleop_collect.teleop.pc_probe --check --json` (not `--mock`);
  - ui: the HTTP port free.

### 10.2 Fake PC

Copied from pd:fake_pc/ with only the import and process-name changes. `config/fake_pc.yaml` unchanged. It already
implements:
- the C3 home (the REP is blocked until the move ends);
- the keyboard hooks;
- the drifting clock;
- `schemas=SHARED_SCHEMAS`.

## 11. Integration and documents (E7)

- **`tests/integration/test_run_mock.py`:** `teleop_collect run --mock --no-browser` as a subprocess, driven by
  `ScriptedPage`:
  - create a task;
  - home, start, hold `x+` for 1 s, `N`, `M`;
  - stop as success;
  - check the episode folder: every file, both `.mkv` files demux to JPEGs, row counts plausible, `COMPLETE` last;
  - start and discard;
  - start, then Stop arm (folder deleted);
  - delete an episode;
  - Ctrl-C: exit 0, no incomplete folder.
- **`tests/integration/test_cross_franka_control.py`:** the same flow against
  `fd:control_deploy/franka_control/build-nolibfranka/franka_control --mock --local` (run from there, never rebuilt;
  skipped if the binary is missing), with `launcher.fake_pc: false`.
- **Multi-process rows of PLAN §7:**
  - a camera down mid-episode (the fake Pi paused) → the episode is aborted, the app stays up;
  - the fake PC `restart` → conclude;
  - the fake PC `inject_fault` → conclude;
  - the page blurred → no motion.
- **Seam fixes** in any folder, each listed in the report.
- **Documents** (PLAN §12.5): `teleop_collect/DIRECTIONS.md`, `deployment/control_deploy/DIRECTIONS.md`,
  `teleop_collect/DESIGN.md`.

## 12. Tests per executor (all without hardware)

| Owner | Tests |
|---|---|
| E1 | station (keys, refused contract keys, ports distinct, data_root); config files load and merge; R1/U1/U2 examples (good pass, broken fail with their field and problem); the copied C0 and clock-fit tests (from pd:tests/contracts/); `test_import_rules.py`: no `policy_deploy` / `control_deploy` import anywhere, and `frames`, `teleop`, `ui`, `fake_pc` import only `teleop_collect.contracts`, `teleop_collect.common`, `deploy_contracts` and themselves |
| E2 | the copied receiver, pair-rule, health, Pi-stream, fake-Pi and pi_collect tests, adapted; `MkvWriter` byte round trip; `FrameRecorder`: rows per camera, reused ultrawide written once, `video_index` = row, queue overflow → error, stop drains; R1 refusals; U2 only newer frames; the frames process against the fake Pi (free ports) |
| E3 | the copied PC client, clock, listener, heartbeat, state-history, startup and failure-monitor tests, adapted; `TeleopLoop` against the fake PC (free ports) with a fake R1 and fake U1 requests: one C3 request per cycle, hello priority, start / stop / discard / abort / conclude, home refusals during homing, start refusals, the clock guard |
| E4 | twist signs and normalisation; `advance` math (identity at tau 0, tilt and twist directions by effect, the pivot keeps the TCP position); `worst_step` and the settings refusal; the controller rules of §7.4 including the stale-state guard and release-once; `HomeReference`; the store (names, numbering never reused, delete, discard refuses outside paths, clean_incomplete); recorders' columns; `build_meta` |
| E5 | server binds 127.0.0.1 only; the WebSocket protocol against fake U1/U2 servers; the deadman (silence → empty keys); `ScriptedPage` against the server; the page's JS key handling checked by reading `app.js` for the map of §4.1 (no browser in the tests) |
| E6 | the copied launcher tests, adapted (order, ready waits, stop in reverse, FATAL line, check rows); the copied fake-PC tests with the shared examples |
| E7 | §11 |
