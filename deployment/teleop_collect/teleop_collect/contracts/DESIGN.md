# teleop_collect.contracts: design

## What it does, and does not

The laptop-only contracts and the conventions every laptop process shares. It is the one subpackage everything else
imports (with `teleop_collect.common`), so it imports no other part of `teleop_collect` (PLAN §2 rule 5,
`tests/test_import_rules.py`).

- **Does:** the Pi's C0 wire format and the internal packet form; the clock fit; the station file and the config
  merge; the common process arguments and the FATAL line; the package paths; the three new local contracts R1, U1
  and U2 (STRUCTURE §4) with their in-memory examples. Importing `teleop_collect.contracts` registers every R1/U1/U2
  schema (and `c0.packet/v1`) in `deploy_contracts`' registry; `SHARED_SCHEMAS` stays unchanged.
- **Does not:** open sockets, run threads (except the clock fit's lock), or talk to the PC. C2/C3 stay in
  `deploy_contracts` (`deployment/contracts/`, never edited here). Whether a folder exists, a task name is free, or a
  camera is ok is checked by the process that owns it, and refused with an `ack/v1`, not by the contract.

## Modules and classes

| Module | Holds |
|---|---|
| `pi_stream.py` | `PiStreamPacket`, `PiStreamHeader`, `PiFrameStamp`, `jpeg_size`: C0 `pi_stream/v1` as the Pi sends it |
| `c0_packet.py` | `Packet`, `PacketHeader`, `FrameInfo`, `EncodedFrame`, `NO_PAIR`: the internal packet form |
| `clock_fit.py` | `ClockMap` (round trips, the PC), `OneWayClockMap` (one-way stamps, the Pi), `ClockFit`, `ClockSample`, `ClockStats`, `NotSynced` |
| `station.py` | `Station` (`station.yaml`: roots, the Pi, the five ports and their endpoints, one section per process), `StationError` |
| `config_merge.py` | `load_config(name, station)`, `resolve_path`: `config/<name>.yaml` with the station's section merged over it |
| `process_args.py` | `ProcessArgs` (`--station --session --ready-file --mock`), `fatal_line`, `print_fatal`, `setup_logging` |
| `paths.py` | `folder()`, `config_file(name)`, `rigs_dir()`, `default_station()`, `deployment_dir()` |
| `health.py` | `CameraHealthReport`, `HealthReport` (Records, carried by R1 and U2), `CAMERAS` |
| `keys.py` | `MOTION_KEYS`, `GRIPPER_COMMANDS`, `OUTCOMES`, `SAVED_OUTCOMES`, `TASK_NAME_RE`, `EPISODE_NAME_RE`, `is_task_name`, `is_episode_name` |
| `r1_recorder.py` | R1 teleop -> frames: `StartRecording`, `StopRecording`, `GetFramesHealth`; replies `RecordingSummary` (with two `CameraSummary`), `FramesHealth` |
| `u2_preview.py` | U2 ui -> frames: `GetPreview`; reply `Preview`; `NO_FRAME = -1` |
| `u1_teleop.py` | U1 ui -> teleop: `GetStatus`, `KeysUpdate`, `GripperCommand`, `SelectTask`, `CreateTask`, `HomeRequest`, `StartEpisode`, `StopEpisode`, `DeleteEpisode`, `StopArm`; reply `TeleopStatus` (with `EpisodeInfo`, `UiMessage`); `PHASES`, `MESSAGE_LEVELS`, `STATUS_MAX_MESSAGES` |
| `examples.py` | `build_examples()`: good and broken R1/U1/U2 examples; `health()`, `recording_summary()`, `teleop_status(**change)` for tests |

Every message is a frozen `deploy_contracts.Message` built from its field helpers, so a message that exists is valid.
Commands are answered with `ack/v1`; a broken request gets `error/v1` from `RpcServer`.

## Decisions

- **Local REQ/REP on 127.0.0.1** for R1, U1, U2, ports from the station (PLAN §3). One port per server, all five
  distinct (`PORT_KEYS`), checked at load; the HTTP port is the page's.
- **Rules a message can check alone are in the contract** (STRUCTURE §4.4-4.5): key names, outcomes, the
  `delete_episode` name pattern, `main_after >= -1`, strictly increasing message ids, `start_refusal == ""` exactly
  when start is allowed, `episode_s >= 0`. Added because they follow from the field meanings: no key twice in
  `held`; `arm_mode` is `""` or a C2 mode; at most `STATUS_MAX_MESSAGES` (50) messages; `CameraSummary` times set
  exactly when `frames > 0` and `last >= first`; a `Preview` JPEG is non-empty and has a `frame_no`;
  `StartRecording.episode_dir` is absolute.
- **Task names are not a contract rule:** a bad name typed in the page must come back as a readable `ack` refusal,
  not as a contract error (PLAN §7: a contract failure concludes). `is_task_name` is there for teleop to use.
- **Images stay bytes:** U2 carries the JPEG exactly as received (PLAN §6.1); nothing here decodes it.
- **Two data roots:** `data_root` (real episodes) and `mock_data_root` (`--mock` runs), both required and refused when
  they are the same folder. `Station.data_root` is the one for this run (`mock_data_root` when loaded with
  `mock=True`), so every process that writes episodes follows `--mock` without knowing about it.
- **The station refuses every contract key** at any depth (PLAN §2 rule 5): addresses, ports and timing of the PC
  come only from `deployment/contracts/link.yaml`. `Station.load` also loads each section through `load_config`, so an
  override key that is not in the config file is refused at load, with every problem in one `StationError`.

## Copied from fd: (`rnoro/final-deployment` @ 2ffe641) and what changed

| File | From `policy_deploy/policy_deploy/` | Changes |
|---|---|---|
| `pi_stream.py`, `c0_packet.py`, `clock_fit.py` | `contracts/` | none but the module paths in docstrings |
| `../common/pose_math.py` | `manager/pose_math.py` | none |
| `station.py` | `contracts/station.py` | sections frames/teleop/ui/fake_pc/launcher; no `python`; `data_root`; ports mock_pi/r1/preview/u1/http; the new endpoints |
| `config_merge.py`, `process_args.py`, `paths.py` | `contracts/` | names: teleop_collect, its folder, its processes and configs |
| `health.py` | `contracts/c1_frames.py` | the two health Records only; + `CAMERAS` |
| `__init__.py` | `contracts/__init__.py` | the re-exports of this package |
| `../../config/fake_pc.yaml`, `../../rigs/rig_pi_v2_640.yaml` | `config/`, `rigs/` | none (byte-identical, no header, so a diff stays empty) |

New: `keys.py`, `r1_recorder.py`, `u1_teleop.py`, `u2_preview.py`, `examples.py`. Each copied file's header lists its
changes (STRUCTURE §1).

## Tests

`tests/contracts/` (run from `deployment/teleop_collect/`):
`unset PYTHONPATH; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/teleop_collect/bin/python -m pytest tests/contracts tests/test_import_rules.py -q`

- `test_c0.py`, `test_pi_stream.py`, `test_clock_fit.py`: copied from fd:, imports only.
- `test_station.py`: the example station, the helper station, refused contract keys at any depth, distinct ports,
  required roots, sections checked against the config files.
- `test_config_files.py`: every config has exactly the keys of STRUCTURE §3.3 and its defaults; merge and
  `resolve_path`; `fake_pc.yaml` and the rig byte-identical to fd:; `ProcessArgs`.
- `test_examples.py`: every R1/U1/U2 schema has a good example, broken ones fail with their field and problem.
- `test_r1_u1_u2.py`, `test_health.py`: names, round trips, each rule above, one request over a real REQ/REP pair.
- `tests/test_import_rules.py`: the import rules for the whole package (whatever files exist).

`tests/station_helpers.py` gives every executor `free_port()`, `write_station(tmp_path, *, sections, ports,
pi_address)` and `contract_error(fn, field, problem)`.

## How to extend

- **A new field:** add it with a field helper, bump nothing if both sides change together (one package), add it to a
  good example's `values` and, if it has a rule, a broken example.
- **A new request:** a `@register`ed `Message` with a new schema `<contract>.<name>/v1`, its re-export in
  `__init__.py`, a good example (the test fails without one), and the handler in the server that owns it.
- **A new config key:** add it to `config/<name>.yaml` with its default, to `KEYS` in `test_config_files.py`, and
  validate it in that process's `settings.py`. A station override of it then works with no change here.
- **A new port:** `PORT_KEYS`, an endpoint method, `station.yaml`, and `PORT_KEYS` in `tests/station_helpers.py`.
