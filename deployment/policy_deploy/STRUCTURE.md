# policy_deploy/: the laptop

The design of everything that runs on the laptop: the frames, policy and manager processes, their launcher, the
laptop-only contracts (C0, C1, C4), the fake PC and the tests. Written from `../PLAN.md` (§3, §4, §4.1, §5, §5.6, §6,
§7, §8.2, §9, §12, §13) and `../STRUCTURE.md` (the shared contracts, communication, clocks, decisions). Where this
file and either of them disagree, they win and this file is fixed. What was settled is in §14 (Decisions).

Source references `old:<path>:<line>` point into `rnoro/deployment` (`/home/ruben/Capstone-deployment/deployment/`),
where every copied file's tests pass. `TOP §n` is `../STRUCTURE.md`, `PLAN §n` is `../PLAN.md`.

## 1. Role

The laptop receives the Pi's frames, runs the policy and drives the PC over the network. It owns:

- **frames**: the Pi stream (C0) in, the Pi → laptop clock map, camera health, the 224×224 images (C1 out).
- **policy**: the model on the GPU (C4). It knows nothing about the arm, absolute poses or the PC.
- **manager**: the episode loop and keyboard; C2 in, C3 out; the laptop → PC clock map; the pose conversions; the
  slow/conclude rule (PLAN §7).
- **launcher**: `policy_deploy check | run [--mock]` for those three processes only.
- **fake_pc**: the Python stand-in for franka_control under `--mock`.

It never starts, stops, configures or reads files of the PC (PLAN rule 2). The PC is started first by the operator
and stays up across laptop runs (PLAN §8.1); the laptop only meets it through C2 and C3.

Terms (as in old:message_comms/STRUCTURE.md §1): **anchor** (the TCP pose at the newest main image time), **chunk**
(16 relative steps, 50 ms apart), **waypoint** (`(t_ns, O_T_TCP, gripper_width_m)` on the PC clock), **`exec_h`**
(waypoints sent per step), **rot6d** (first two **rows** of R), **heartbeat**, **watchdog**. New here:

- **PC clock map**: the manager's estimate of `PC CLOCK_MONOTONIC − laptop CLOCK_MONOTONIC` as a line in laptop time.
- **H2 wait**: TOP §3.7.3 rule H2, no C3 request while C2 shows another client's episode.
- **Hello sample**: `(t_send_ns, t_pc_ns, t_recv_ns)` from one `hello` round trip.
- **Worst heartbeat gap**: the longest the manager can go between two C3 requests (PLAN §6).

Variable names follow TOP §6.1: PC times end in `_pc_ns`, Pi times in `_pi_ns`, laptop times in `_ns`.

## 2. Layout, environments, station

### 2.1 Tree

One distribution, `policy-deploy` (import package `policy_deploy`, version 1.0.0). One class per file, named after the
class; small value types share the file of the class that makes them (old convention, old:deploy/STRUCTURE.md §7.6).

```
deployment/policy_deploy/
  STRUCTURE.md                 this file
  DIRECTIONS.md                from scratch on the laptop (outline in §13)
  PI_PAYLOAD.md                C0: old:contracts/PI_PAYLOAD.md §0 unchanged, plus appendix §3-4 (the internal packet
                               form) and one note: the live Pi's `skew_ns` has the opposite sign to §0.3's definition
                               (§0.6), so nothing may use `skew_ns` (§4.2); the c0 msgpack sender and time_sync parts
                               are dropped
  pyproject.toml               distribution "policy-deploy" (§2.3)
  station.yaml                 the laptop station file (§2.4)
  config/                      every setting of each process, with its default (§2.5)
    frames_process.yaml  policy.yaml  manager.yaml  fake_pc.yaml  launcher.yaml
  rigs/
    rig_pi_v2_640.yaml         copied unchanged from old:process_frames/rigs/rig_pi_v2_640.yaml (its tcp: block is
                               ignored here; the TCP is contracts/frames.yaml, PLAN §9)
  policy_deploy/
    __init__.py                docstring only: importing the package imports nothing
    __main__.py                python -m policy_deploy check | run  (= launcher.cli.main)
    contracts/                 laptop-only contracts and laptop conventions, shared by the subpackages (§3)
      __init__.py              re-exports
      paths.py                 folder(), config_file(name), rigs_dir(), default_station(); config_file("frames") is
                               config/frames_process.yaml, every other name config/<name>.yaml
      station.py               Station: station.yaml (§2.4)
      config_merge.py          load_config(name, station): config/<name>.yaml + the station section (§2.5)
      process_args.py          ProcessArgs: the common arguments, session folder, ready file, fatal_line() (§2.6)
      pi_stream.py             C0 wire format pi_stream/v1                          copied (old:contracts/contracts/pi_stream.py)
      c0_packet.py             FrameInfo, PacketHeader, EncodedFrame, Packet        adapted (old:contracts/contracts/c0_frames.py)
      c1_frames.py             C1 messages                                          copied (old:contracts/contracts/c1_frames.py)
      c4_policy.py             C4 messages; config reply v2                         adapted (old:contracts/contracts/c4_policy.py)
      c4_limits.py, c4_limits.yaml   C4Limits: the three C4 tolerances              adapted (old:contracts/contracts/limits.py)
      sizes.py                 OBS_STEPS, ACTION_STEPS, STEP_MS, IMAGE_SIZE         copied (old:contracts/contracts/sizes.py)
      _rot6d.py                rot6d_problem()                                      copied
      clock_fit.py             ClockSample, ClockStats, ClockMap, OneWayClockMap    moved (old:process_frames/process_frames/clock_map.py)
      examples.py              in-memory C0/C1/C4 examples                          adapted (old:contracts/contracts/examples.py:402-694)
    launcher/                  check | run (§9)
      cli.py errors.py session.py check_result.py check_helpers.py managed_process.py processes.py launcher.py
    frames/                    process "frames" (§4)
      __main__.py app.py settings.py errors.py rig_file.py packet_source.py pi_stream_source.py
      recording_packet_source.py frame_decoder.py jpeg_decoder.py ultrawide_store.py pair_rule.py packet_buffer.py
      packet_selector.py camera_health.py receiver.py frames_server.py pi_recording.py packet_pairer.py
      fake_pi_sender.py fake_pi.py probe.py pi_collect.py _dataproc/
    policy/                    process "policy" (§5)
      __main__.py app.py settings.py errors.py policy_server.py policy_factory.py base_policy.py
      promptable_policy.py hold_still_policy.py key_map.py neutral_observation.py self_check.py probe.py
      diffusion_policy.py model_runner.py checkpoint_loader.py light_workspace.py bench.py
      promptable_diffusion_policy.py prompt_source.py                                     (last seven: torch)
    manager/                   process "manager" (§6, §8)
      __main__.py settings.py errors.py frames_client.py policy_client.py pc_client.py heartbeat_sender.py
      pc_listener.py state_history.py pc_clock.py link_stats.py startup_checks.py startup_sequence.py
      pose_math.py pose_converter.py policy_step.py step_worker.py problems.py failure_monitor.py episode_loop.py
      operator.py keyboard_operator.py scripted_operator.py status_line.py session_log.py pc_probe.py self_check.py
    fake_pc/                   process "fake_pc", --mock only (§10)
      __main__.py settings.py fake_pc.py drifting_clock.py fake_gate.py plan_follower.py
  tests/                       one folder per subpackage (§12)
    conftest.py  contracts/  frames/  policy/  manager/  fake_pc/  launcher/  integration/  test_import_rules.py
```

### 2.2 Import rules

| Module | May import | Never |
|---|---|---|
| `policy_deploy.contracts` | stdlib, numpy, yaml, msgpack, zmq, `deploy_contracts` | any other `policy_deploy` subpackage |
| `policy_deploy.frames`, `.policy`, `.manager`, `.fake_pc` | stdlib, their own third-party libraries, their own subpackage, `policy_deploy.contracts`, `deploy_contracts` | another process subpackage, `policy_deploy.launcher`, `control_deploy` |
| `policy_deploy.launcher` | stdlib, yaml, `policy_deploy.contracts`, `deploy_contracts` | any process subpackage: it runs them as subprocesses (`python -m ...`), so torch, OpenCV or PyAV never load in the launcher |
| `policy_deploy.policy` torch rule | torch, dill, hydra, timm, diffusers, behavior_prompting only in `diffusion_policy.py`, `model_runner.py`, `checkpoint_loader.py`, `light_workspace.py`, `bench.py`, `promptable_diffusion_policy.py`, `prompt_source.py` | at package import (`hold_still` runs without torch) |

`tests/test_import_rules.py` reads every module with `ast` (nothing is imported) and enforces the table. It also
checks that no module name of `rnoro/deployment` (`contracts`, `deploy`, `process_frames`, `policy`, `message_comms`)
is imported, so nothing clashes with the old editable installs in `~/envs/deploy`.

The station schema, the merge rule and the clock fit live in `policy_deploy.contracts` because more than one process
subpackage uses them and the import rule allows nothing else (PLAN §8.2). They are laptop conventions, not wire
messages; `deploy_contracts` holds no clock code (TOP §6.1).

### 2.3 Environments and install

Two new uv venvs, Python 3.10 (TOP §1.4). The old `~/envs/deploy` belongs to `rnoro/deployment` and is never touched.

| Env | Path | Installs (all editable) | Runs |
|---|---|---|---|
| light | `~/envs/policy_deploy` | `deployment/contracts`, `deployment/policy_deploy[light,test]` | launcher, frames, manager, fake_pc, policy with `hold_still`, every test except torch ones |
| torch | `~/envs/policy_deploy_torch` | torch 2.7.1 + torchvision 0.22.1 (`--index-url https://download.pytorch.org/whl/cu118`, driver 535 is CUDA 12.2), behavior_prompting's pip list (`behavior_prompting/environment.yml`, numpy 1.26.4), `-e behavior_prompting`, `deployment/contracts`, `deployment/policy_deploy[model,test]` | policy with `diffusion`, `policy.bench`, `policy.self_check`, the policy and contracts tests |

```
uv venv ~/envs/policy_deploy --python 3.10
uv pip install --python ~/envs/policy_deploy/bin/python -e deployment/contracts -e "deployment/policy_deploy[light,test]"
```

`pyproject.toml`:

| Key | Value |
|---|---|
| `requires-python` | `>=3.10,<3.11` |
| dependencies | `numpy>=1.26,<2.3`, `pyyaml==6.0.3`; `deploy-contracts` is **not** listed: it is installed editable in the same command (the rule of TOP §1.4 for both folders; `check` fails without it, §9.2) |
| extra `light` | `opencv-python-headless==4.12.0.88` (frames: JPEG decode, image prep), `scipy==1.15.3` (the rig loader; the manager's slerp), `av==17.1.0` (only to read the recorded episode for the fake Pi) |
| extra `model` | `torch`, `dill`, `hydra-core`, `timm`, `diffusers` (unpinned: versions from `behavior_prompting/environment.yml`) |
| extra `test` | `pytest==8.3.5` |
| packages | `policy_deploy*` (incl. `policy_deploy.frames._dataproc*`); package data `contracts/c4_limits.yaml` |
| console script | `policy_deploy = policy_deploy.launcher.cli:main` (so `policy_deploy check \| run` works, as PLAN and DIRECTIONS write it) |
| pytest | `testpaths = ["tests"]`, `pythonpath = ["tests"]`, `addopts = "--import-mode=importlib"` |

The torch env never gets the `light` extra: OpenCV-headless would clash with behavior_prompting's `opencv-python`
(the reason the old design had one pyproject per folder, old:deploy/STRUCTURE.md §7.3). `config/`, `rigs/` and
`station.yaml` are found through `paths.folder()` = `Path(policy_deploy.__file__).resolve().parents[1]`, so the
install must be editable (as `deploy_contracts`, TOP §1.4); otherwise `paths.folder()` raises with the install hint.

### 2.4 `station.yaml`

One file for real and `--mock` runs (PLAN rule 1). It may **not** override any key of the three contract yaml
files (TOP §1.3 rule 4): the PC's address, ports and timing come only from `deployment/contracts/link.yaml`.

```yaml
station_version: 1
session_root: ~/policy_deploy_sessions
python:
  light: ~/envs/policy_deploy/bin/python
  torch: ~/envs/policy_deploy_torch/bin/python      # used only for policy.kind diffusion (and bench)
pi:
  address: 192.168.50.30          # PLACEHOLDER: the Pi's reserved ASUS_5g address (PLAN §15)
  port: 5555                      # fixed by the Pi (PI_PAYLOAD.md §0.1)
wifi_ssid: ASUS_5g                # check: the laptop's Wi-Fi network
ports:                            # all on 127.0.0.1 (TOP §2.2)
  frames: 5610                    # C1
  policy: 5640                    # C4
  mock_pi: 5600                   # --mock: the fake Pi inside frames
frames: {}                        # overrides of config/frames_process.yaml
policy:
  kind: hold_still                # hold_still until a rig-trained diffusion checkpoint exists (PLAN §1)
manager: {}
fake_pc: {}
launcher: {}                      # e.g. fake_pc: false (the cross test, §12.3)
```

`Station.load(path, mock)` (adapted from old:deploy/deploy/station.py) collects every problem with its key path into
one `StationError`. Endpoints: `pi_endpoint()` = `tcp://<pi.address>:<pi.port>` (or `tcp://127.0.0.1:<mock_pi>`
under `--mock`), `frames_endpoint()`, `policy_endpoint()`. `pi.address` is required except under `--mock`;
`python.torch` only when `policy.kind` is `diffusion`.

**Contract keys refused** (TOP §1.3 rule 4), by an explicit list, never by generic names (the station's own `ports`,
`pi.address` and `frames` are legal):

| Refused | Where | Belongs to |
|---|---|---|
| `watchdog_limit_s`, `heartbeat_period_s`, `c3_timeout_s`, `state_stale_s`, `state_lost_s`, `estimator` | at any level | `contracts/link.yaml` |
| `F_T_TCP`, `settled` | at any level | `contracts/frames.yaml` |
| `rotation_tol`, `gripper_width_max_m`, `max_waypoints` | at any level | `contracts/limits.yaml` |
| `pc`, `laptop`, `timing`, `clock` | top level | `contracts/link.yaml` |

The message names the key path and the file, e.g. `station.yaml: manager.c3_timeout_s: belongs to
deployment/contracts/link.yaml`. The leaf names are checked against every merged config file too, so no
`config/*.yaml` may define them.

### 2.5 Config files and the merge rule

`config_merge.load_config(name, station)`: `paths.config_file(name)` (`config/<name>.yaml`; for frames
`config/frames_process.yaml`), then the station's `<name>:` section merged over it
(maps key by key, other values replaced, an override key not in the file is an error with its path). The same rule as
old:deploy/STRUCTURE.md §6.3, now written once in `policy_deploy.contracts` and used by every process and the
launcher. Relative paths: a station value against the station file's folder, a default against `policy_deploy/`.

### 2.6 Common arguments, session folder, FATAL line

Every process's `__main__` takes `--station FILE [--session DIR] [--ready-file FILE] [--mock]`
(old:deploy/STRUCTURE.md §4.2). Alone (no `--session`), it writes into
`<session_root>/<YYYY-MM-DD_HH-MM-SS>_<name>_alone/<name>/`, the same layout as a run. A process that exits because
of a failure writes, as its last stderr line, `FATAL <process>: <what failed>: <what to check>` and exits 1; a normal
stop (`q`, SIGINT, SIGTERM) is exit 0 with no FATAL line. The process names are `frames`, `policy`, `manager`,
`fake_pc`. When the cause is on the other machine, `<what failed>` names it: `PC not answering: ...`,
``the PC stopped the arm (watchdog or the PC's `s`): ...``, `franka_control restarted on the PC: ...`.

## 3. The laptop contracts (`policy_deploy.contracts`)

Built on `deploy_contracts`' `Message`, `Record`, field helpers, codec, registry, errors, `Ack`/`ErrorReply`,
`RpcClient`/`RpcServer` and example helpers (TOP §3.12). Importing `policy_deploy.contracts` registers its schemas
into the shared registry; `SHARED_SCHEMAS` does not change (TOP §3.4).

### 3.1 C0: Pi → frames

`pi_stream/v1`, exactly PI_PAYLOAD.md §0 (copied file; PLAN §5.1). `PiStreamPacket.from_parts(parts)` checks the 3
parts, the JSON header keys and types, `version` 1 and non-empty JPEGs; `jpeg_size()` reads width and height from the
SOF marker. Nothing is sent back to the Pi. The c0 msgpack packets, `TimeSyncRequest`/`TimeSyncReply` and the C0
multipart helpers (`PacketPublisher`, `PacketSubscriber`) are not carried over.

**Internal packet form** (`c0_packet.py`). The receiver works on the old C0 packet (`Packet`, `PacketHeader`
`c0.packet/v1`, `FrameInfo`, `EncodedFrame`, old:contracts/contracts/c0_frames.py:23-139), as in `rnoro/deployment`.
It never crosses a socket now; `to_parts`/`from_parts` stay for the tests. `c0.packet/v1` is a laptop-only schema.

### 3.2 C1: frames ↔ manager (laptop clock)

Unchanged from old:contracts/contracts/c1_frames.py, except every time is now **laptop** `CLOCK_MONOTONIC`.

| Request → reply | Fields |
|---|---|
| `c1.get_frames/v1` → `c1.frames/v1` | `main` uint8 (2, 224, 224, 3) or nil; `main_t_ns` int64 (2,) or nil; `ultrawide` uint8 (2, 224, 224, 3) or nil; `ultrawide_t_ns` int64 (2,) or nil; `health` |
| `c1.get_health/v1` → `c1.health/v1` | `health` |

`health` = `{main, ultrawide}`, each `{status: ok|stale|down, last_frame_age_ms ≥ 0, fps ≥ 0, dropped uint64}`.
Images and times come together, only when both cameras are `ok`; times strictly increasing, oldest first. A main or
ultrawide time is the **packet time**: the main frame's mid-exposure mapped to the laptop clock (both cameras of a
packet share it, as in iPhUMI).

### 3.3 C4: manager ↔ policy

As old:contracts/contracts/c4_policy.py, with one change: the config reply gains `kind`.

| Request → reply | Fields |
|---|---|
| `c4.config/v1` → **`c4.config.reply/v2`** | `obs_steps` 2, `action_steps` 16, `step_ms` 50, `image_size` 224 (int ≥ 1); `inputs` (str list of Observation fields the model uses); `uses_prompt` bool; **`kind`** str, non-empty (`hold_still`, `diffusion`, ...) |
| `c4.reset/v1` → `ack/v1` | |
| `c4.prompt/v1` → `ack/v1` | `task_name` str, `prompt_index` int ≥ 0; only when `uses_prompt`, once after each reset |
| `c4.infer/v1` (Observation) → `c4.chunk/v1` | below |

`Observation` (`c4.infer/v1`): `eef_pos` float32 (2, 3); `eef_rot6d` float32 (2, 6); `eef_rot6d_wrt_start` float32
(2, 6); `gripper_width_m` float32 (2, 1); `main`, `ultrawide` uint8 (2, 224, 224, 3). Checks: finite; last rows 0 and
`[1,0,0,0,1,0]` within `anchor_row_tol`; rot6d unit within `rot6d_norm_tol` and perpendicular within `rot6d_dot_max`;
widths ≥ 0.

`ActionChunk` (`c4.chunk/v1`): `eef_pos` float32 (16, 3); `eef_rot6d` float32 (16, 6); `gripper_width_m` float32
(16, 1) in `[0, gripper_width_max_m]` (the shared `limits.yaml`, TOP §4.1); `width_clipped` uint64 ≤ 16. Checks:
finite, rot6d as above, widths in range compared in float64. Step 0 near the anchor is not a contract rule (the
manager's odd-chunk check, §7.4).

`kind` exists so the manager can apply PLAN §8.2 ("a trained policy is refused on the real arm while `frames.yaml`
says `settled: false`") without reading the policy's station section (§14, decision 3). The bump is laptop-only:
both ends are in this folder.

`c4_limits.yaml` (moved from old:contracts/contracts/limits.yaml, TOP §4.1): `anchor_row_tol: 1.0e-5`,
`rot6d_norm_tol: 0.1`, `rot6d_dot_max: 0.1`. `C4Limits.load()` is strict like `ContractLimits` (missing, unknown,
non-positive refused) and cached.

### 3.4 `clock_fit.py`

Moved from old:process_frames/process_frames/clock_map.py:63-176 so frames and the manager share one fit. Changes:
names say **local** and **remote** instead of PC and Pi (`to_local_ns(t_remote_ns, at_local_ns)`, new
`to_remote_ns(t_local_ns)` = `t_local_ns + offset_ns(at_local_ns=t_local_ns)`, `offset_ns` = remote − local); the
window, best fraction, minimum drift span, drift cap and freshness are constructor arguments; `stats()` adds the
newest sample's age and the sample span. The math is unchanged: the best samples of the window (`ClockMap`: the
shortest round trips; `OneWayClockMap`: the least-delayed stamps of each 1 s slice), a least-squares line once the
best samples span `drift_min_span_s`, drift set to 0 above `max_drift_ppm`, one lock. `OneWayClockMap` refits at
most every 0.25 s; `ClockMap` refits on every sample (as old).

| Used by | Class | local / remote | Samples |
|---|---|---|---|
| frames (Pi → laptop) | `OneWayClockMap` | laptop / Pi | `add_one_way(t_send_ns of each packet, laptop arrival)` |
| manager, `round_trip` | `ClockMap` | laptop / PC | `add_sample(ClockSample(t_send_ns, t_pc_ns, t_recv_ns))` per hello |
| manager, `one_way` | `OneWayClockMap` | laptop / PC | `add_one_way(C2 t_ns, laptop arrival)` when `t_ns` advanced |

### 3.5 Examples

`examples.py` builds the C0/C1/C4 examples in memory with `deploy_contracts`' public helpers (TOP §5.1), as
old:contracts/contracts/examples.py:402-694 did, plus `c4.config.reply/v2` good and broken (`kind` missing, `kind`
empty). None are stored. The C0 examples are `pi_stream/v1` messages (good, wrong part count, header not JSON,
missing key, boolean `packet_no`, version 2, empty JPEG) plus the internal packet ones the receiver tests use.

## 4. Process: frames

Copied from old:process_frames (the `pi_v1` path), with the frozen-ultrawide fix (PLAN §7) and the c0/H.264/time-sync
code dropped (the Pi contract is final, PLAN §8.2).

### 4.1 Responsibilities and threads

Receive every Pi packet, check it against C0 and the rig file, decode both JPEGs, map Pi times to the laptop clock,
keep the last second, track each camera's health, and on request pick 2 main + 2 ultrawide packets and prepare them
exactly as in training (C1). It reports; the manager decides (old:process_frames/STRUCTURE.md §9).

| Thread | Runs | Owns |
|---|---|---|
| receiver | `Receiver.run`: `PiStreamPacketSource` → checks → `JpegDecoder` → `UltrawideStore` → `OneWayClockMap` → `PacketBuffer`, `CameraHealth` | the SUB socket to the Pi |
| main | `FramesServer` (C1 REP), ready file, stats line every 10 s, watching for a FATAL from the receiver | the C1 REP socket |
| fake-pi-frames (`--mock`) | `FakePiSender` | the fake Pi PUB on `127.0.0.1:<ports.mock_pi>` |

Shared objects and locks as old:process_frames/STRUCTURE.md §4 (one lock each on `PacketBuffer`, each `CameraHealth`,
the clock map; never two at once; packets immutable once buffered). There is no clock-sync thread.

### 4.2 Classes

| Class (file) | One job | From | Changes |
|---|---|---|---|
| `Settings` (`settings.py`) | `config/frames_process.yaml` + station section, validated | old:process_frames/process_frames/settings.py | station schema of §2.4; drops `stream`, `decoder`, `clock.interval_s/timeout_s/startup_exchanges`, `h264.*`, `mock.keyframe_interval`, every `pi_time_sync` key; adds `select.pair_max_age_ms`, `select.pair_max_lead_ms`, `select.pair_max_reuse` |
| `load_live_rig`, `rig_summary`, `main` (`rig_file.py`) | the rig file for live use; `python -m policy_deploy.frames.rig_file FILE --json` | copied | module path |
| `PacketSource` (`packet_source.py`) | interface: `next_packet(timeout_s) -> Packet | None`, `close()` | copied | none |
| `PiStreamPacketSource`, `PiStreamConverter` (`pi_stream_source.py`) | SUB to the Pi; each message → `PiStreamPacket` → internal `Packet`; feeds the clock map from `t_send_ns` | old:process_frames/process_frames/pi_stream_source.py:37-92 | the converter also counts `uw_reused` headers (stats); `Packet` carries the ultrawide frame only when its `frame_no` is new (unchanged) |
| `RecordingPacketSource` | test stand-in: packets from memory | copied | none |
| `FrameDecoder`, `JpegDecoder` (`frame_decoder.py`, `jpeg_decoder.py`) | decode one frame to RGB | copied | `make_decoder` knows only `jpeg`; `H264Decoder` dropped (the interface stays for a swap) |
| `UltrawideStore`, `DecodedUltrawide` (`ultrawide_store.py`) | decoded ultrawide frames by `frame_seq` | old:process_frames/process_frames/ultrawide_store.py | adds `evict_before(t_pi_ns)`, called on every packet (fix 3); `DecodedUltrawide` adds `exposure_ns` |
| `PairRule` (`pair_rule.py`) | is this pair usable: captured within the window around its main frame, and not the pair of too many consecutive packets? | new | §4.3 fix 1 |
| `PacketBuffer`, `BufferedPacket` (`packet_buffer.py`) | the last `keep_s` of packets | copied | field `t_pc_ns` renamed `t_ns` (laptop clock); `pair_gap_ns` is now mid-exposure based |
| `PacketSelector`, `Selection` (`packet_selector.py`) | iPhUMI's pick rule | copied | `t_pc_ns` → `t_ns` |
| `CameraHealth` (`camera_health.py`) | ok / stale / down per camera | copied | `record_frame(t_ns)` naming; callers change (fix 2) |
| `Receiver`, `CameraCounters`, `PacketCounters` (`receiver.py`) | one packet at a time (§4.3) | old:process_frames/process_frames/receiver.py:92-148 | steps 6-11 of §4.3; counters add `uw_reused`, `pair_too_old`, `pair_too_new`, `pair_reused` |
| `FramesServer` (`frames_server.py`) | C1 REP: `get_frames`, `get_health` | copied | none |
| `PiRecording`, `RecordedCamera`, `CaptureEvent`, `JpegEncoder` (`pi_recording.py`) | a recorded Pi episode as frames, for the stand-ins | copied | `H264Encoder` dropped |
| `PacketPairer` (`packet_pairer.py`) | the fake Pi's pairing, as the live Pi does it | old:process_frames/process_frames/packet_pairer.py | **nearest-frame pairing**: `pair_seq` is the added ultrawide frame whose capture `t_pi_ns` is nearest the main frame's; a tie goes to the older frame; no age or lead cut (the receiver's `PairRule` judges); `NO_PAIR` only before the first ultrawide frame. The old "latest frame ≤ main + 5 ms" rule (`PAIR_AHEAD_NS`) and its `max_age_ns` cut are removed. Rule 2 (every encoded frame sent once, in the next packet) unchanged |
| `FakePiSender` (`fake_pi_sender.py`) | stand-in Pi, `pi_v1` messages | old:process_frames/process_frames/fake_pi_sender.py:197-240 | `pi_v1` only (no c0, no time_sync thread); pairs through the nearest-frame `PacketPairer`: before sending a main frame's packet it adds every ultrawide frame of the recording captured up to half an ultrawide frame period after it (the recording knows every capture time), so the nearest frame is known; like the live Pi (PI_PAYLOAD.md §0.6) each packet leaves 45.8 ms after its main frame's `t_sensor_ns` (`SEND_DELAY_NS`), and `skew_ns` has the live sign (`ultrawide.t_mid − main.t_mid`, positive when the ultrawide is newer); `uw_reused` and `skew_ns` follow the pair; adds `repeat_ultrawide(seconds)` (fix 4) |
| `FramesApp` (`app.py`) | builds, runs, ready file, stats, shutdown | old:process_frames/process_frames/app.py:39-290 | no `ClockSync`; always `PiStreamPacketSource` + `OneWayClockMap` |
| `fake_pi.py`, `probe.py`, `pi_collect.py` | fake Pi alone; ask a running frames once; receive-only collector | copied | `pi_collect` adds the pair-gap report: `main.t_mid_ns − ultrawide.t_mid_ns` computed from the two `t_mid_ns` (never from `skew_ns`, whose live sign is reversed, PI_PAYLOAD.md §0.6), min/median/max, a histogram, and the longest run of consecutive packets with the same pair; it is what confirms `pair_max_lead_ms` (§14, decision 1) |
| `_dataproc/` | image prep, rig loader, camera model | copied unchanged from `rnoro/data-work` (old:process_frames/STRUCTURE.md §3) | none; `test_copies.py` keeps them byte-identical |

Dropped: `clock_sync.py`, `pi_packet_source.py`, `h264_decoder.py`, `launch.py` (the launcher owns that, §9).

### 4.3 One packet (receiver), with the frozen-ultrawide fix

1. `source.next_packet(receive.timeout_s)`; a C0 violation → FATAL `C0 packet breaks its contract: <field>: check
   the Pi sender version`. `arrival_ns = clock()`; the source has fed `(t_send_ns, arrival_ns)` to the clock map.
2. After a Pi restart (step 4), count `ignored` and return.
3. Each frame map against the rig file (sensor, 640×480, `jpeg`) → FATAL naming the camera (old step 3).
4. Restart check (`packet_seq`, a `frame_seq` or a `t_pi_ns` going back) → both cameras latched `down` (old step 4).
5. Gaps and decode: main frame; the ultrawide frame only when the packet carries it (a new `frame_no`). Each decoded
   ultrawide frame goes into the store with its `t_pi_ns` and `exposure_ns`.
6. **Fix 3, eviction by time:** `store.evict_before(main.t_pi_ns − buffer.keep_s)` on every packet, so a frame the Pi
   keeps repeating does not stay findable forever (old:process_frames/process_frames/ultrawide_store.py:24-29 evicted
   only on `add`).
7. Clock not fresh (`clock.max_sync_age_s`) → `unsynced`, return.
8. Packet time `t_ns = clock_map.to_local_ns(rig.camera_time_ns("main", main.t_pi_ns, exposure), arrival_ns)`.
9. **Fix 2, ultrawide health from accepted pairs at their own time** (done after step 11, once the packet is
   buffered): when the packet's pair was accepted by step 10 and is a newer `frame_seq` than any counted before,
   `health["ultrawide"].record_frame(to_local_ns(pair_mid_pi, arrival))`. A repeated frame records nothing, so a
   stalled IMX219 ages to `stale` (0.35 s) and `down` (2 s); and, as in the old receiver, a frame that is never
   accepted as a pair records nothing either, so a pair always refused for timing (e.g. always `pair_too_new`) makes
   the ultrawide `stale`, then `down`, and the episode concludes (PLAN §7 "slow, then conclude") instead of every step
   skipping as SLOW for ever. The old receiver refreshed ultrawide health with the main packet time
   (old:process_frames/process_frames/receiver.py:144-147), so a frozen ultrawide never went stale.
10. **Fix 1, pair rule** (PLAN §7): `pair = store.get(pair_seq)`; then `PairRule.check(pair_seq, packet_seq,
    main_mid_pi, pair_mid_pi)`. The pair is usable only if **both**:
    - **window:** `gap = main_mid − pair_mid` (Pi clock, mid-exposures) satisfies `−pair_max_lead_ms ≤ gap ≤
      pair_max_age_ms` (defaults: lead 25 ms, age 60 ms; the lead confirmed from `pi_collect`'s pair-gap report,
      §14 decision 1);
      else counted `pair_too_new` / `pair_too_old`;
    - **reuse:** `PairRule` keeps the previous packet's `pair_seq` and a run count (consecutive packets, this one
      included, whose pair is that same `frame_seq`, counted by `packet_seq`, so a lost packet in the run counts as
      one of it: it neither resets the run nor lets a second repeat through; reset to 1 on a new `pair_seq`, and on a
      Pi restart). Usable iff
      the run count ≤ `pair_max_reuse` (2). A third consecutive packet with the same pair is unusable whatever its
      timing (counted `pair_reused`), and so is every further one until the pair changes.
    An unusable pair: the packet has no ultrawide image. The live Pi pairs the nearest frame (gap about −16 ms), so one
    repeat (gap + 33 ms ≈ +17 ms) is served and a second repeat (≈ +50 ms, inside the 60 ms window) is refused by the
    reuse rule; the window is a sanity bound that does not depend on the phase being right.
11. Order check; `buffer.push(BufferedPacket(...))`; `health["main"].record_frame(t_ns)` when the main image is
    usable; latency (`arrival − t_ns`) and pair-gap statistics.

**Fix 4, the fake Pi's repeat mode:** `FakePiSender.repeat_ultrawide(seconds)` adds no new ultrawide frame for that
long, so the nearest-frame pairer keeps returning the last one (same `frame_no`, `uw_reused: true`), as a stalled
IMX219 would on the live Pi.

Serving (C1, main thread) is unchanged (old:process_frames/STRUCTURE.md §6.2): both-or-nothing; window of the newest 4
packets; anchor T = newest packet time; main targets `[T − 50 ms, T]`, ultrawide `[T − 100 ms, T]`; nearest packet,
ties to the earlier one; a miss beyond 2 frame periods, a packet without a usable image for that camera, or the same
packet twice → no images and that camera `stale` in that reply only. At 30 FPS the main images are the packets at
T − 66.7 ms and T, the ultrawide at T − 100 ms and T.

### 4.4 Config (`config/frames_process.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `rig_file` | `rigs/rig_pi_v2_640.yaml` | both cameras 640×480 @ 30 |
| `mock_episode` | `~/Downloads/episode_000045` | the recorded Pi episode the fake Pi plays (`--mock` only) |
| `default_exposure_ns` | `5000000` | when `exposure_us` is 0, capped at half the frame period |
| `receive.timeout_s`, `receive.recv_hwm` | `0.1`, `60` | as old |
| `buffer.keep_s` | `1.0` | packets and decoded ultrawide frames kept |
| `select.main_spacing_ms`, `ultrawide_spacing_ms`, `tolerance_frames` | `50`, `100`, `2.0` | as old |
| `select.pair_max_age_ms` | `60` | fix 1: the pair at most this much older than its main frame (mid-exposure) |
| `select.pair_max_lead_ms` | `25` | fix 1: the pair at most this much newer; comment in the file: `set from pi_collect's pair-gap report` (§14 decision 1) |
| `select.pair_max_reuse` | `2` | fix 1: the same ultrawide frame may be the pair of at most this many consecutive packets |
| `health.stale_s`, `down_s`, `fps_window_s` | `0.35`, `2.0`, `1.0` | PLAN §7 |
| `clock.window_s`, `best_fraction`, `drift_min_span_s`, `max_drift_ppm`, `max_sync_age_s` | `60`, `0.3`, `10`, `500`, `10` | as old (one-way) |
| `server.poll_s`, `server.join_timeout_s` | `0.05`, `2.0` | as old |
| `log.stats_every_s` | `10` | stats line and `stats.jsonl` |
| `mock.max_seconds`, `jpeg_quality`, `pi_clock_offset_s`, `drift_ppm`, `send_hwm` | `10`, `90`, `1000`, `20`, `30` | the fake Pi |
| `check.reachable_timeout_s`, `subprocess_timeout_s` | `1.0`, `10.0` | launcher checks |
| `probe.timeout_s` | `2.0` | `frames.probe` |

### 4.5 Start-up, ready, shutdown

`FramesApp.run()`: rig file (FATAL if invalid), `ImagePrep`, health, buffer, selector, bind C1 (FATAL if busy), with
`--mock` start the fake Pi on `127.0.0.1:<ports.mock_pi>` playing `mock_episode` at the rig's sizes, then the receiver
thread. **Ready** (ready file) when C1 answers and both cameras have been `ok` at the same time once. Every 5 s while
not ready it logs what it waits for (`no packets yet from tcp://<pi>:5555: is the Pi on ASUS_5g and its sender
running?`, `ultrawide stale`). SIGINT/SIGTERM: stop flag, join threads, stop the fake Pi, close C1, exit 0.
Outputs: `<session>/frames/stats.jsonl`.

## 5. Process: policy

Copied from old:policy unchanged in behaviour (old:policy/STRUCTURE.md), plus `kind` in the config reply, a
`precision` setting, GPU memory in `startup.json`, the timing tool `bench`, and prompting checkpoints
(`PromptableDiffusionPolicy`, still kind `diffusion`; the prompt parts of detached_inference).

### 5.1 Responsibilities and threads

Single-threaded (old §3.2): load, warm up, bind C4 (`127.0.0.1:<ports.policy>`), then `RpcServer.serve_once` until
stopped. Bad requests get `ErrorReply` and serving goes on; its own failures (load, CUDA, a reply breaking C4) are
FATAL after the `ErrorReply` (old §3.5).

### 5.2 Classes

| Class (file) | From | Changes |
|---|---|---|
| `Policy` (`base_policy.py`), `make_chunk` (width clip to the largest float32 ≤ `gripper_width_max_m`) | old:policy/policy/base_policy.py | `config()` fills `kind = KIND` |
| `PromptablePolicy`, `HoldStillPolicy`, `KeyMap`, `neutral_observation`, `PolicyServer`, `build_policy`, errors | copied | imports only |
| `DiffusionPolicy` (`diffusion_policy.py`) | copied | passes `precision` to the runner; `from_settings`: a prompting checkpoint → `PromptableDiffusionPolicy` (needs `prompt_dataset`), `prompt_dataset` with a plain one refused |
| `PromptableDiffusionPolicy` (`promptable_diffusion_policy.py`, torch) | detached_inference.py `policy_prompt`/`policy_reset` | a `PromptablePolicy` over `DiffusionPolicy` (same `ModelRunner`): `prompt()` → `model.prompt()` under `inference_mode`, fp32 (also with `bf16`); `reset()` clears it; warm-up prompt = the longest demo |
| `PromptSource` (`prompt_source.py`, torch) | detached_inference.py:164-192, 376-395 | the checkpoint's `task.dataset` with `only_prompt`, `dataset_path` = `prompt_dataset`, `val_ratio` 0, `cache_dir` None (zip read in place), split `prompt_split`; unknown task / index → `BadRequest`; longest demo from the replay buffer's `task_lengths` |
| `ModelRunner` (`model_runner.py`) | old:policy/policy/model_runner.py:43-77 | `precision: bf16` wraps `predict_action` in `torch.autocast("cuda", dtype=torch.bfloat16)` (training's `mixed_precision: bf16`); weights stay fp32; output `.float()` as before |
| `CheckpointLoader`, `LoadedModel` (`checkpoint_loader.py`) | old:policy/policy/checkpoint_loader.py:31-179 | `LoadedModel.dataset_cfg` (the resolved `cfg.task.dataset`, for `PromptSource`) |
| `light_workspace.py` | copied | none |
| `Settings` (`settings.py`) | old:policy/policy/settings.py | station schema of §2.4; adds `precision`, `mock_kind`; endpoint from `ports.policy` |
| `PolicyApp` (`app.py`) | old:policy/policy/app.py:65-140 | `startup.json` adds `torch.cuda.max_memory_allocated/reserved` after warm-up, `nvidia-smi` used/total memory, GPU name, power limit, `precision`; a `PromptablePolicy` is prompted with its `warm_up_prompt()` before the warm-up inferences (`warmup_prompt`) |
| `self_check.py` | old:policy/policy/self_check.py | adds rows: free GPU memory and other GPU processes (`nvidia-smi --query-compute-apps`), `precision`, `prompt dataset` (when set; zarr, numcodecs, imagecodecs in imports), `exec_h` is the manager's and not shown here |
| `probe.py` | old:policy/policy/probe.py:29-115 | prints `kind`; `--infers N` reports mean/p50/p95/max |
| `bench.py` (new, torch) | new | §11: times `predict_action` without a socket or a checkpoint of the right kind; a prompting checkpoint with `--prompt-dataset` |

### 5.3 Config (`config/policy.yaml`)

As old:policy/config.yaml, plus:

| Key | Default | Meaning |
|---|---|---|
| `kind` | `null` (required unless `--mock`) | `hold_still` \| `diffusion` |
| `checkpoint` | `null` | required for `diffusion` |
| `device` | `cuda` | PLAN §13 device backup (`cpu`) |
| `diffusion_steps` | `null` (the checkpoint's 16) | PLAN §13: third backup |
| `precision` | `fp32` | PLAN §13: second backup `bf16` (autocast) |
| `prompt_dataset` | `null` | a prompting checkpoint only (refused for a plain one): the processed `.zarr.zip` of the demos, resolved like `checkpoint`; `manager.task_name` / `prompt_index` pick the demo |
| `prompt_split` | `train` | `train` (the demos the model trained on) \| `val` (held out) \| `all` (every demo: a dataset the model never trained on) |
| `mock_kind` | `hold_still` | the kind that runs under `--mock`; `diffusion` times the real model against the fake Pi and fake PC (needs `checkpoint` and the torch env) |
| `warmup_inferences`, `warmup_gripper_width_m`, `reset_max_s`, `poll_s`, `self_check_timeout_s`, `training_timing.*`, `key_map.*` | as old | |

The interpreter follows the kind that runs: `diffusion` → `python.torch`, everything else → `python.light`
(old §6).

### 5.4 Start-up, ready, shutdown

As old §3.1: settings, session folder, `build_policy` (load + checks), warm-up (`warmup_inferences` infers on
`neutral_observation`, each timed, then `reset`), bind C4, `startup.json`, ready file, serve. Ready means the model is
on the device, warmed up, and C4 answers. The launcher allows 300 s. SIGINT/SIGTERM: exit 0. Outputs:
`<session>/policy/startup.json`; per-episode inference stats in its log at each `reset`.

## 6. Process: manager

Derived from old:message_comms. Its role is the same (old:message_comms/STRUCTURE.md §1); what changes is that C2/C3
cross the network to a PC on another clock, that it no longer reads the PC's config (old:message_comms/launch.py:54-113
read `franka_control/config.yaml`), and that it must coexist with a PC that outlives it.

### 6.1 Threads

| Thread | Runs | Owns (sockets) | Blocks at most |
|---|---|---|---|
| main | `EpisodeLoop`: rhythm, keys, **every C3 request** (rule H1), health polls, failure decisions, logging, status line | `PcClient` (C3 REQ), a `FramesClient` for `get_health` | one C3 wait (`c3_timeout_s` 0.3 s) + one `get_health` (`timeouts.health_s` 0.05 s) per cycle; `home` waits in heartbeat-period slices |
| step worker | `StepWorker`: one job at a time (`config`, `reset` + `prompt`, or `PolicyStep.run`) | a `FramesClient` for `get_frames`, `PolicyClient` (C4) | `timeouts.policy_hang_s` (5 s) |
| pc listener | `PcListener`: every C2 into `StateHistory`, `PcClock.add_state`, `LinkStats` | the C2 SUB | `listener_poll_s` per receive |
| keyboard | `KeyboardOperator` reader | none | forever (daemon) |

Shared data as old §3 (`StateHistory` lock; worker queues; key queue; signal flags; listener error), plus the
`PcClock` (its own lock: the listener writes `one_way` samples, the main thread writes hello samples, the worker and
main read) and `LinkStats` (one lock). All decisions are made on the main thread.

### 6.2 Classes

| Class (file) | One job, main methods | From | Changes |
|---|---|---|---|
| `Settings`, `Timeouts` (`settings.py`) | `config/manager.yaml` + station section + `LinkSettings` + `TcpFrame` | old:message_comms/message_comms/settings.py | link values (`heartbeat_period_s`, `c3_timeout_s`, `state_stale_s`, `state_lost_s`, `clock_estimator`, endpoints) come only from `link.yaml`; the old `loop_ms`, `max_state_age_ms`, `state_lost_s`, `timeouts.commands_s` keys are gone; adds the keys of §6.6 |
| `FramesClient` (`frames_client.py`) | `get_frames()`, `get_health()` | copied | process name `frames` |
| `PolicyClient` (`policy_client.py`) | `config()`, `reset()`, `prompt()`, `infer()` | copied | expects `c4.config.reply/v2` |
| `PcClient` (`pc_client.py`) | every C3 request (§8.1) | old:message_comms/message_comms/franka_client.py:14-62 | adds `hello()`; timeouts from `link.yaml`; process name `PC` |
| `HeartbeatSender` (`heartbeat_sender.py`) | `beat(now_ns)`: the cycle's one C3 keep-alive, a hello when due else a heartbeat; feeds `PcClock` and `LinkStats` | new | §8.2 |
| `PcListener` (`pc_listener.py`) | C2 thread: `start()`, `stop()`, `error()`, `received()` | old:message_comms/message_comms/state_listener.py:13-51 | `PcClock.add_state(t_ns, arrival)` when `t_ns` advanced; arrival and gap stats to `LinkStats`; process name `PC` |
| `StateHistory`, `PoseSample`, `LookupMiss` (`state_history.py`) | last `history_s` of C2, interpolation (PC times) | old:message_comms/message_comms/state_history.py | `seq` going back raises `PcRestarted`; equal `seq` or `t_ns` going back stays a `ContractError` |
| `PcClock`, `RoundTripPcClock`, `OneWayPcClock`, `make_pc_clock()` (`pc_clock.py`) | laptop ↔ PC map behind one interface (§8.3) | new, over `contracts.clock_fit` | |
| `LinkStats` (`link_stats.py`) | C3 round trips (p50/p95/max of a window), C2 rate, `seq` gaps, newest arrival; for the status line, `link.jsonl`, `pc_probe` | new | |
| `StartupChecks`, `CheckFailure` (`startup_checks.py`) | the hello and budget checks of §8.4 | new | |
| `StartupSequence` (`startup_sequence.py`) | the start-up order of §6.3 | new; replaces old:message_comms/message_comms/episode_loop.py:166-247 | |
| `pose_math`, `PoseConverter`, `Anchor` | conversions (§7) | copied (old:message_comms/message_comms/pose_math.py, pose_converter.py) | `Anchor` carries both times of the anchor image: `t_ns` (laptop, for `NO_NEW_FRAME` and image age) and `t_pc_ns` (PC, for waypoint times, the margin and `StepRecord`); `to_waypoints` uses `t_pc_ns` |
| `PolicyStep`, `StepJob`, `StepDone`, `StepSkipped`, `NoNewFrame`, `StepTiming` (`policy_step.py`) | one step on the worker (§7) | old:message_comms/message_comms/policy_step.py:81-182 | image times mapped to PC time with one fit snapshot; state age on mapped "now"; `CLOCK_STALE` skip |
| `StepWorker`, `WorkerResult` | the worker thread | copied | none |
| `Kind`, `ProblemCode`, `Problem`, `Verdict` (`problems.py`); `FailureMonitor` (`failure_monitor.py`) | the slow/conclude table, one dict `RULES` | old:message_comms/message_comms/problems.py:8-61, failure_monitor.py:20-135 | new codes and texts (§8.5); `check_state` takes PC "now" |
| `EpisodeLoop`, `Phase`, `LoopProgress` (`episode_loop.py`) | the loop (§6.4) | old:message_comms/message_comms/episode_loop.py | start-up moved to `StartupSequence`; `HeartbeatSender` instead of a bare heartbeat; margins in PC time; `CLOCK_MISMATCH` counter; key dispatch table; `c`/`h` refused while C2 shows `homing`; `HOME_STOPPED`; the `start` ack time kept as `start_ack_pc_ns` (mapped, §8.5); `last_anchor_t_ns` stays laptop time (old line 502) |
| `Operator`, `Key`, `KeyboardOperator`, `ScriptedOperator`, `ScriptEntry` | keys and status line | copied | `Key` lookup through a table (§6.5) |
| `StatusLine`, `StatusSnapshot` | one line ≤ 120 chars | old:message_comms/message_comms/status_line.py | adds link RTT, C2 rate, clock state, `PC mock` |
| `SessionLog` and records | the session files (§6.7) | old:message_comms/message_comms/session_log.py | adds `clock.jsonl`, `link.jsonl`, the hello reply and both commits in `session.yaml`, `c3_rtt_ms` in `StepRecord` |
| `pc_probe.py`: `PcProbe`, `PcCheck` | read-only link probe; `--check` prints check lines for the launcher (§8.6) | new | |
| `self_check.py` | settings and the local budget rule, as JSON check lines (§9.2) | new; replaces old:message_comms/launch.py checks | |

### 6.3 Start-up (phase `STARTING`), in order

The decided order (PLAN §8.2, TOP §8.2). Heartbeats are sent from the moment hellos start; nothing is sent on C3
before that.

| # | Step | Done when | Fails with (FATAL manager: ...) |
|---|---|---|---|
| 1 | settings, `SessionLog`, start `PcListener` and `StepWorker` | | `config: <problems>: check station.yaml manager: and config/manager.yaml` |
| 2 | **C2 arriving**: a valid C2 message | first message | `PC not answering: no C2 state from tcp://<pc>:5620 within 20 s: check that control_deploy run is running on the PC, ASUS_5g, and link.yaml pc.address` |
| 3 | **H2 wait**: while the newest C2 mode is `running` or `holding`, send nothing on C3, up to `2 × link.watchdog_limit_s` (laptop time since the first C2); also while it is `homing`, up to `timeouts.pc_homing_wait_s` (a `home` started over C3 by an earlier laptop session blocks the PC's REP until it ends, TOP §3.7.2, so a hello would time out) | mode is `idle` or `stopped` | `running`/`holding` past the limit: `the PC is driving an episode for another C3 client: stop the other laptop session, pc_probe or console`; `fault`: `the PC is in fault: <error>: Ctrl-C and restart control_deploy run on the PC`; `homing` past its limit: `the PC is still homing after 60 s: check the PC terminal` |
| 4 | **hellos start**: one hello every cycle (`clock.startup_hello_period_s`) until the map has `clock.min_samples_s` (2 s) and `clock.min_samples` (10) of samples; then one every `clock.hello_period_s` (0.5 s), heartbeats in the other cycles. With `one_way`, the C2 samples since step 2 count too. | clock ready | C3 timeout: `PC not answering: no reply to c3.hello/v1 within 0.3 s: ...` |
| 5 | **`compare_hello` + laptop checks** on the newest hello reply (`StartupChecks`, §8.4): mismatches, worst gap, home timeout, waypoint lead, 5 ms clock agreement, commit (warning only) | no failure | `hello mismatch: <field>: PC <value>, laptop <value> (<source>): <what to check>`; `clock mismatch at start-up: ...` |
| 6 | **mode via `get_state`**: `idle`/`stopped` → go on; `homing` → keep beating, ask again every 0.5 s, up to `hello.homing_timeout_s + 5 s`; `running`/`holding` → another client; `fault` → refuse | `idle` or `stopped` | as step 3 |
| 7 | **heartbeats** run every cycle from here (they already did since step 4; this is where the old design started them) | | |
| 8 | **policy config** (C4 v2, `worker.submit_config()`, the loop keeps cycling): sizes equal `sizes.py`; `exec_h ≤ action_steps`; `uses_prompt` needs `task_name`; **settled rule**: `kind != hold_still` and `TcpFrame.settled` false and the PC's `hello.mock` false → refuse | reply checked | `policy config: step_ms is 100, manager expects 50: check the checkpoint`; `trained policy refused on the real arm: contracts/frames.yaml settled is false (PLAN §9): use policy.kind hold_still, or settle the TCP` |
| 9 | **cameras**: `get_health` every `health_poll_ms` until both `ok`; `down` → conclude | both `ok` | `startup: no camera ok within 20 s (ultrawide stale): check frames` |
| 10 | `write_session` (settings, hello reply, both commits, clock fit), operator start, key list, **ready file**, phase `IDLE` | | |

Steps 2-9 together must finish within `timeouts.startup_s` (20 s), not counting time spent waiting while C2 shows
`homing` (steps 3 and 6). The longest legal start-up is therefore `startup_s + pc_homing_wait_s +
(homing_timeout_s + 5 s)` = 20 + 60 + 35 = **115 s** (the policy is already ready when the manager starts). The
launcher's ready timeout for the manager is **130 s** (§9.1), above that, so the manager's own FATAL line always comes
first. Throughout start-up every cycle also checks the signal flags and, once hellos run, the state rules of §8.5
(`fault`, `seq` going back, no C2 for `state_lost_s` by laptop arrival).

### 6.4 The loop (phases `IDLE`, `RESETTING`, `RUNNING`, `HOMING`, `DONE`)

As old:message_comms/STRUCTURE.md §4.8 and §5, with these changes. One `cycle()`, every `link.heartbeat_period_s`:

1. Interrupt flag (Ctrl-C) → quit; terminate flag (SIGTERM from the launcher) → launcher stop.
2. `monitor.check_state(history.latest(), pc_clock.now_pc_ns(), ...)`; then `listener.error()` (a `PcRestarted`
   becomes `PC_RESTARTED`, any other error `CONTRACT_VIOLATION`).
3. `heartbeats.beat(now)` (a hello or a heartbeat), except in `HOMING`.
4. A key → its handler (§6.5).
5. Every `health_poll_ms`: `get_health` → camera `down` concludes.
6. `RUNNING`, worker idle, nothing submitted this cycle → submit the next step.
7. Every `status_ms`: the status line.

Between cycles `run()` waits on the worker; a `StepDone` is handled at once (old §5.4 steps 9-11): drop it if its
episode ended; else `set_waypoints` (C3, main thread), reply checks (§8.1), then **submit the next step**, then log the
step and the tap. A cycle may therefore carry two C3 requests (its keep-alive and a `set_waypoints`); both feed the
watchdog, so the worst gap of §8.4 still holds.

Episode start (`c`, old §5.3): accepted in `IDLE` with the worker idle, **the clock map ready** (after a long `home`
the `round_trip` map has had no samples; hint `c: waiting for the PC clock map`) and **the newest C2 mode not
`homing`** (the PC's own `h`/`g`; hint `c: the PC is homing (its h or g key): wait for idle` through the old
`_refuse` path; the laptop stays `IDLE` and sends nothing). Then `reset` (+ `prompt`), `start`, start pose =
`history.latest()` in `pose_source`, phase `RUNNING`. A `start` the PC refuses (`Ack ok: false`, e.g. it began homing
in between) → back to `IDLE` with the hint `c: the PC refused start: <reason>`, as old
(old:message_comms/message_comms/episode_loop.py:371-376); never a conclude. The `start` ack's laptop time is mapped
at once, `start_ack_pc_ns = pc_clock.snapshot().to_pc_ns(start_ack_ns)`: that is the time compared with C2 `t_ns` by
the `PC_STOPPED` rule (§8.5; old code compared a laptop time with C2 `t_ns`).

Home (`h`, old §5.7): refused locally while the newest C2 mode is `homing` (hint `h: the PC is already homing (its h
or g key)`; stays `IDLE`, nothing sent, never a conclude). Otherwise heartbeats and hellos pause; `pc.home(interrupted)`
waits up to `timeouts.home_s` in heartbeat-period slices. The outcome:

| Reply | Outcome | Kind | Then |
|---|---|---|---|
| `ok: true` | home reached | normal | `IDLE` |
| `ok: false`, reason starts with `stopped:` (TOP §3.7.2) | `HOME_STOPPED`: the PC's `s` stopped the move; C2 shows `stopped` | normal | the reason on the status line and in `events.jsonl`; `IDLE` |
| `ok: false`, reason `refused: mode is homing` (the PC's `h`/`g` started in the ~10 ms between the C2 check and this request) | home refused: the PC is already homing | normal | the hint on the status line, like a refused `start`; `IDLE` |
| any other `ok: false`, an `error/v1`, or no reply within `home_s` | `HOMING_FAILED` | conclude | §8.5 |

After every non-concluding outcome the manager puts `heartbeats.startup_mode(True)` (a hello every cycle) until
`pc_clock.ready()`, then back to one hello every `hello_period_s`: the map had no samples during the home. After a
home longer than the 10 s window, `ready()` needs 10 new samples spanning 2 s: about 2 s at a hello every cycle,
instead of about 4.5 s at one per 0.5 s.

Ctrl-C or SIGTERM during `home` drops the reply (socket closed, linger 0) and the manager exits 0 without
`stop` (the PC is busy homing and cannot answer). **The PC is not stopped by the launcher any more**: it finishes the
homing move and goes `idle`; the next laptop run waits for it (§6.3 step 3). The immediate stops are the robot's
external stop button and the PC keyboard `s`.

`s`, `y`, `n`, `q`, Ctrl-C, SIGTERM: as old §5.6, §5.8 (`stop` first, the episode recorded `stopped`/`success`/
`fail`/`aborted`). Conclude (old §5.9): `stop` once with `c3_timeout_s` (failure logged, not retried), record the
episode `aborted` with the reason, stop the threads, FATAL line, exit 1. **The PC keeps running**, holding the arm
(PLAN §7).

### 6.5 Keyboard

`c` start, `y` success, `n` failure, `s` stop, `h` home (C3 `home`), `q` quit; same meanings and phase rules as
old:message_comms/STRUCTURE.md §4.9. A key that doesn't fit the phase gets a one-line hint. New (PLAN §7): while the
newest C2 mode is `homing` (the PC's `h` or `g`), `c` and `h` are refused locally with a hint on the status line; the
manager stays `IDLE` and never concludes for it (§6.4).

**Extension point for jog keys (PLAN §14, later):** `EpisodeLoop` dispatches through one table
`KEY_HANDLERS: dict[str, Callable[[EpisodeLoop], None]]`, and `Key` values are looked up in it, so a later jog key is
one entry plus one `Key` member. A jog command would be a one-waypoint `set_waypoints` built by a future `JogCommand`
class in `IDLE` after `start`; nothing for it is built now.

### 6.6 Config (`config/manager.yaml`)

Kept from old:message_comms/config.yaml with their defaults: `exec_h` 8, `lookahead_ms` 0, `pose_source` target,
`odd_chunk_max_pos_m` 0.02, `odd_chunk_max_rot_deg` 5.0, `anchor_fallback_m` 0.02, `status_ms` 250,
`health_poll_ms` 500, `history_s` 1.0, `max_image_age_ms` 250, `state_recv_hwm` 1000, `timeouts.frames_s` 1.0,
`policy_hang_s` 5.0, `config_s` 5.0, `reset_s` 1.0, `prompt_s` 60.0, `startup_s` 20.0, `task_name` null,
`prompt_index` 0, `operator_script` null, `operator_script_wait_s` 30, `skip_warn_s` 1.0, `log_steps` true,
`save_tap` false, `tap_images_every` 10.

Changed or new:

| Key | Default | Meaning |
|---|---|---|
| `timeouts.health_s` | `0.05` | C1 `get_health` on the main thread (localhost); a term of the worst gap |
| `timeouts.home_s` | `90.0` | C3 `home`; must exceed `hello.homing_timeout_s` (§8.4) |
| `timeouts.pc_homing_wait_s` | `60.0` | start-up: longest wait without C3 while C2 shows `homing` (§6.3 step 3) |
| `link_jitter_allowance_s` | `0.2` | worst gap: allowance for uplink delay variance and local work (PLAN §6) |
| `listener_poll_s` | `0.02` | C2 receive timeout (how fast the listener sees its stop flag) |
| `clock.hello_period_s` | `0.5` | hello rate after start-up (PLAN §5.6) |
| `clock.startup_hello_period_s` | `0.02` | hello rate during start-up step 4 (every cycle) |
| `clock.round_trip.window_s`, `best_fraction` | `10.0`, `0.3` | PLAN §5.6 |
| `clock.one_way.window_s`, `best_fraction` | `5.0`, `0.3` | PLAN §5.6 |
| `clock.drift_min_span_s`, `clock.max_drift_ppm` | `5.0`, `200.0` | drift fitted once the best samples span 5 s; a larger fitted drift is noise (worst extrapolation error 200 ppm × 5 s = 1 ms). With `one_way` (window 5 s = this span) drift is practically never fitted; harmless (30 ppm × 2.5 s = 0.08 ms) |
| `clock.min_samples_s`, `clock.min_samples` | `2.0`, `10` | the map is ready only with this span and count (PLAN §5.6 guard 1) |
| `clock.max_sample_age_s` | `5.0` | the map is not ready when its newest sample is older (after a long `home`) |
| `clock.startup_agree_ms` | `5.0` | start-up guard 2 (§8.4) |
| `clock.jump_log_ms` | `1.0` | a refit that moves "now" by more than this is logged (guard 3) |
| `clock.mismatch_sets`, `clock.mismatch_min_lead_ms` | `3`, `100` | guard 4 (§7.5) |
| `clock.future_state_ms` | `20.0` | guard 4, the other side: a set sent while the newest C2 `t_ns` is more than this ahead of the mapped "now" counts toward `CLOCK_MISMATCH` (a map running behind, §7.5) |
| `clock.log_every_s` | `10.0` | `clock.jsonl` period |
| `link_stats.window_s` | `10.0` | RTT percentiles window |

The worst-gap rule is written as a comment in the file: `heartbeat_period_s + c3_timeout_s + timeouts.health_s +
link_jitter_allowance_s` = 0.02 + 0.3 + 0.05 + 0.2 = **0.57 s** `< watchdog_limit_s` (1.0 s, PLAN §6).

### 6.7 Outputs (`<session>/manager/`)

| File | Content |
|---|---|
| `session.yaml` | merged settings, the `contracts/link.yaml` and `contracts/frames.yaml` values used, the hello reply, laptop and PC commits, policy config, the clock estimator and its fit at ready |
| `episodes.jsonl`, `steps.jsonl`, `events.jsonl`, `tap/` | as old §4.10; `StepRecord` adds `c3_rtt_ms`, `t_anchor_pc_ns`, `margin_ms` (PC clock) |
| `clock.jsonl` | every `clock.log_every_s`: estimator, offset (PC − laptop), drift ppm, samples used, shortest RTT, newest sample age; also every logged fit jump. Lets PC logs be lined up with laptop logs |
| `link.jsonl` | every 10 s: C3 RTT p50/p95/max, C2 rate, `seq` gaps, newest C2 age (mapped) |

## 7. The policy inputs, end to end (PLAN §4.1)

### 7.1 The chain for one step

```
Pi: t_sensor_ns, exposure_us  ──frames──►  T_ns (laptop)  ──C1──►  main_t_ns[0..1] (laptop)
                                                                          │ PolicyStep: one PcClock snapshot
                                                                          ▼
C2 t_ns (PC) ──PcListener──► StateHistory ──pose_at(main_t_pc[i])──► T_i (target or measured), w_i
                                                                          │ PoseConverter.to_observation (anchor T_1)
                                                                          ▼
                                                     Observation (C4) ──policy──► ActionChunk (C4)
                                                                          │ odd-chunk check, to_waypoints
                                                                          ▼
                                     SetWaypoints t_ns = main_t_pc[1] + k·50 ms (PC) ──C3──► PC gate
```

### 7.2 Field by field

| C4 field | Produced by | From | At which time | Clock |
|---|---|---|---|---|
| `main` (2, 224, 224, 3) | frames: `JpegDecoder` → `ImagePrep.prep_main` (letterbox) in `FramesServer.get_frames` | the packets picked nearest T − 50 ms and T | their packet times `main_t_ns` | Pi → laptop |
| `ultrawide` (2, 224, 224, 3) | frames: `ImagePrep.prep_ultrawide` (fisheye remap to the virtual pinhole, letterbox) | each picked packet's accepted pair (§4.3 fix 1), packets nearest T − 100 ms and T | served with the packet times; the pair's own capture is within 25 ms after to 60 ms before it, and it was the pair of at most 2 consecutive packets | Pi → laptop |
| `eef_pos` (2, 3) | manager: `PoseConverter.to_observation` | `StateHistory.pose_at(main_t_pc[i], source)`; `R_aᵀ(p_i − p_a)`; row 1 exactly 0 | `main_t_pc[0]`, `main_t_pc[1]` | PC |
| `eef_rot6d` (2, 6) | same | `rot6d(R_aᵀ R_i)`, first two **rows**; row 1 `[1,0,0,0,1,0]` | same | PC |
| `eef_rot6d_wrt_start` (2, 6) | same | `rot6d(R_sᵀ R_i)`, `R_s` = the episode start pose (`history.latest()` at `start`, same source) | same | PC |
| `gripper_width_m` (2, 1) | same | C2 `gripper_width_m` interpolated linearly at the same times (absolute; read by the PC at 50 Hz, so up to 20 ms older than `t_ns`, TOP §3.6 obligation 5) | same | PC |

`source` is `O_T_TCP_target` (`pose_source: target`), falling back to `O_T_TCP` for the whole step when they differ
by more than `anchor_fallback_m` at the anchor (old:message_comms/STRUCTURE.md §6.7). The anchor is
`Anchor(T_a = T_1, t_ns = main_t_ns[1], t_pc_ns = main_t_pc[1])`: both clocks, so `NO_NEW_FRAME` and the image age
compare laptop times and the waypoints use PC times; the same `Anchor` object goes into `to_waypoints` (old §4.4). Interpolation
(`StateHistory.pose_at`, old §6.1): the two neighbouring C2 samples with consecutive `seq`, linear position, slerp
rotation, linear width; repeated `t_ns` add no sample; a `seq` gap around the time → `STATE_GAP`; before the oldest or
after the newest → `STATE_OLD`; never extrapolated. `get_state` replies never enter the history (TOP §3.6 obligation 3).

### 7.3 The two clock maps and their error

| Map | Where | Samples | Error of a mapped time |
|---|---|---|---|
| Pi → laptop | frames, `OneWayClockMap` (60 s window, least-delayed 30 % per 1 s slice, drift fitted after 10 s) | every packet's `t_send_ns` vs laptop arrival | laptop times come out **late** by the minimum Pi → laptop delay δ₁ (stamp-to-wire + Wi-Fi + receiver wake-up), a few ms; constant, so it does not move between steps |
| laptop → PC, `round_trip` (default) | manager, `RoundTripPcClock` (10 s window, shortest-RTT 30 %) | hello samples: `t_pc_ns` vs `(t_send + t_recv) / 2` | ± half the round-trip asymmetry of the best samples: well under 1 ms wired, 1-3 ms on Wi-Fi; plus the hello encode/decode around the stamps (~0.1 ms) |
| laptop → PC, `one_way` (backup) | manager, `OneWayPcClock` (5 s window) | C2 `t_ns` vs laptop arrival, only when `t_ns` advanced | PC times come out **early** by the minimum of (PC publish lag 0-6 ms + PC → laptop delay): ~1.5 ms wired, a few ms on Wi-Fi |

Both PC estimators also extrapolate drift: at most `max_drift_ppm × window/2` = 1 ms. `PolicyStep` takes **one
snapshot** of the PC fit and maps both main times and "now" with it, so a refit between the two lookups cannot shift
one row against the other.

The pose-to-image error is the sum: `round_trip`: δ₁ ± asymmetry/2, about 1-5 ms on Wi-Fi, ≤ 1.25 mm at 0.25 m/s;
`one_way`: δ₁ − δ_min. The same constant makes the waypoints run δ₁ late on the PC (the arm a few ms behind the plan).
Both are small against a 33 ms frame and a 50 ms step. `pc_probe` shows both PC estimators side by side (§8.6).

### 7.4 Output path

1. `policy.infer(obs)` → `ActionChunk`, validated on decode (shapes, finite, unit and perpendicular rot6d, widths in
   `[0, 0.08]`). A broken chunk is `INVALID_CHUNK`, conclude.
2. **Odd-chunk check** (`PolicyStep`, old §5.4 step 7b): `|eef_pos[0]| ≤ 0.02 m` and the angle of
   `matrix_from_rot6d(eef_rot6d[0]) ≤ 5°`. Otherwise `ODD_CHUNK`: nothing sent, counted, slow.
3. `PoseConverter.to_waypoints` for `k = 0 … exec_h − 1`: `R_rel,k` = Gram-Schmidt on the rot6d rows (float64),
   `T_abs,k = T_a · [[R_rel,k, eef_pos[k]], [0,0,0,1]]`, width passed through, and
   `t_ns[k] = anchor.t_pc_ns + k × 50 ms + lookahead_ms` (int64, **PC clock**). `SetWaypoints` validates itself (increasing,
   rigid within `rotation_tol`, `N ≤ max_waypoints`).
4. Main thread: `pc.set_waypoints(msg)`; reply checks (chunk id echo, `accepted + skipped + len(rejected) == N`);
   `rejected` → `GATE_REJECTED`; `skipped == N` → `POLICY_LATE` or a clock-mismatch count (§7.5); margin.

### 7.5 Margin and the clock-mismatch guard

`margin_ms` = (last waypoint time of the previous set) − `pc_clock.now_pc_ns()` at the send of the new set, both PC
clock. A negative margin is how long the arm held (`POLICY_LATE`, slow). A set the PC skipped entirely (`skipped == N`)
while the laptop's own lead (last waypoint − mapped send time) was at least `clock.mismatch_min_lead_ms` (100 ms, well
above Wi-Fi spikes) counts toward `CLOCK_MISMATCH`. So does a set sent while the newest C2 `t_ns` is more than
`clock.future_state_ms` (20 ms) ahead of `pc_clock.now_pc_ns()`: a C2 reading "from the future" means the map runs
behind, so the waypoints would be accepted but late, which the skipped-set rule cannot see. `clock.mismatch_sets` (3)
counted sets in a row conclude. A set that is neither (it has an accepted waypoint and no future C2) resets the
count. One cause per set: an entirely skipped set that counts toward `CLOCK_MISMATCH` (for either reason) is not also
counted as `POLICY_LATE`. Gross errors are also caught by the PC: a waypoint more than `max_lead_s`
(1.5 s) ahead is rejected (a fault).

### 7.6 Timing budget with the network hop

The step period is `C' = frames + conversions + inference + RTT(set_waypoints)`, because the next step is submitted
after the `set_waypoints` reply (old order, kept). A set reaches the PC one uplink delay `d_up` after it is sent. The
arm never holds if

```
L + 2·C' + d_up ≤ (exec_h − 1) × 50 ms          exec_h 8: 350 ms   10: 450 ms   12: 550 ms
```

(PLAN §4's condition with the hop written out; PLAN counts the C3 round trip inside C.)

| Term | Typical | Measured and logged where |
|---|---|---|
| exposure/2 + Pi readout, encode, send | ~58 ms (live: the packet left 45.8 ms after `t_sensor`, which is 12.8 ms after `t_mid`, PI_PAYLOAD.md §0.6) | frames stats `lat` = arrival − packet time (`stats.jsonl`) |
| Wi-Fi Pi → laptop | 2-10 ms | inside `lat` |
| waiting for the step + C1 pick | 0-33 ms | — |
| **L** (main mid-exposure → C1 reply) | 75-110 ms | `steps.jsonl image_age_ms` (laptop clock, no map involved) |
| C1 call + prep (640×480 → 224) | 3-8 ms | `frames_ms` |
| conversions + C4 encode/decode | 2-5 ms | inside `policy_ms` |
| inference | 150-220 ms estimated, fp32, 16 steps (PLAN §8.2) | `policy_ms`; the policy log per episode; `bench` (§11) |
| C3 `set_waypoints` round trip | < 1 ms wired, 5-20 ms Wi-Fi | `steps.jsonl c3_rtt_ms`; `link.jsonl`; status line `rtt` |
| d_up | ≈ RTT/2 | not measured directly; its effect shows in `margin_ms` and the PC's `skipped` |
| result | | `margin_ms` (PC clock) on the status line and in `steps.jsonl`; `late` count |

Hold still costs `C'` ≈ 10-15 ms and is far inside the budget. With the estimated inference, exec_h 8 likely holds
briefly each step ("slow"); §11 decides `exec_h`.

## 8. The PC as the laptop sees it

### 8.1 `PcClient` (C3, main thread only)

One `deploy_contracts.RpcClient("PC", link.commands_endpoint(local=mock))`; lazy pirate on timeout (TOP §3.11).
Errors pass through (`TransportTimeout`, `RemoteError`, `ContractError`); `Ack(ok=false)` → `Refused` (except
`start` and `home`, whose `Ack` is returned: the loop decides by it, §6.4); a cancelled `home` → `RequestCancelled`.
`PcClient.home_stopped(ack)` is the one place that reads the contract prefix: `not ack.ok and
ack.reason.startswith("stopped:")` (TOP §3.7.2).

| Method | Request → reply | Timeout | Sent when |
|---|---|---|---|
| `hello() -> HelloSample` | `c3.hello/v1` → `c3.hello.reply/v1`; stamps `t_send_ns` just before the request and `t_recv_ns` just after the reply; a plain `RpcClient.request` **without `poll_s` or `cancel`** (a sliced poll adds up to `poll_s` on the receive side only: an asymmetry the estimator would read as offset, old rpc.py:75-76) | `c3_timeout_s` | by `HeartbeatSender`: every cycle in start-up step 4, then every `hello_period_s`, in place of that cycle's heartbeat |
| `heartbeat()` | `c3.heartbeat/v1` → `ack` | `c3_timeout_s` | every other cycle from start-up step 4, except in `HOMING` |
| `get_state() -> ArmState` | → `c2.state/v1` | `c3_timeout_s` | start-up step 6 (mode only; never a C2 sample) |
| `start() -> Ack` | → `ack`; not ok → `IDLE` with a hint (§6.4) | `c3_timeout_s` | `c`, unless C2 shows `homing` |
| `stop()` | → `ack` (never refused) | `c3_timeout_s` | `y`, `n`, `s`, `q`, Ctrl-C, SIGTERM, conclude |
| `home(interrupted) -> Ack` | → `ack` at the end of the move: ok, `stopped:` (→ `HOME_STOPPED`), or other not ok (→ `HOMING_FAILED`) | `timeouts.home_s`, in heartbeat-period slices | `h` in `IDLE`, unless C2 shows `homing` (§6.4) |
| `set_waypoints(msg)` | → `c3.set_waypoints.reply/v1`; checks the `chunk_id` echo and the sum | `c3_timeout_s` | each `StepDone` |

No other thread or socket sends C3 (rule H1), so a hung main loop starves the watchdog and the PC stops the arm.

### 8.2 `HeartbeatSender`

```
class HeartbeatSender:
    def __init__(self, pc: PcClient, pc_clock: PcClock, stats: LinkStats, hello_period_s: float,
                 startup_hello_period_s: float, clock: Callable[[], int]) -> None
    def startup_mode(self, on: bool) -> None          # start-up step 4: a hello every cycle
    def beat(self, now_ns: int) -> HelloReply | None  # one C3 request: a hello when due (returns its reply), else a heartbeat
    def last_hello(self) -> HelloReply | None
```

Each hello's sample goes to `pc_clock.add_hello(...)` and its RTT to `stats`; each heartbeat's RTT to `stats`. A later
hello whose `t_pc_ns` is below the previous one means **the PC rebooted** (`CLOCK_MONOTONIC` survives a franka_control
restart; a restart shows as C2 `seq` going back): `PC_RESTARTED` with the reboot text (§8.5). The newest reply is kept
for `StartupChecks`. `startup_mode(True)` is used in start-up step 4 and again after every `HOMING` that did not
conclude, until `pc_clock.ready()` (§6.4).

### 8.3 `PcListener` and the PC clock classes

`PcListener._run`: `Subscriber("PC", link.state_endpoint(local=mock), ArmState, state_recv_hwm).receive(poll)`; per
message: `arrival = clock()`; `history.add(state)` (raises `PcRestarted` on `seq` going back, `ContractError` on equal
`seq` or `t_ns` going back); if `t_ns` advanced, `pc_clock.add_state(t_ns, arrival)`; `stats.c2(arrival, seq)`. The
first exception is stored and the thread ends; the main thread concludes on it.

```
class PcClock(ABC):                                          # pc_clock.py
    def add_hello(self, s: HelloSample) -> None             # every hello (both estimators keep the newest for checks)
    def add_state(self, t_pc_ns: int, arrival_ns: int) -> None
    def ready(self, now_ns: int) -> bool                     # span >= min_samples_s, count >= min_samples, newest <= max_sample_age_s
    def snapshot(self) -> PcClockFit                         # frozen: to_pc_ns(t_ns), offset_ns, drift_ppm, samples
    def now_pc_ns(self) -> int                               # snapshot().to_pc_ns(clock())
    def agreement_ms(self, s: HelloSample) -> float          # t_pc_ns - to_pc_ns(midpoint), in ms
    def stats(self, now_ns: int) -> PcClockStats             # for the status line, clock.jsonl, pc_probe
class RoundTripPcClock(PcClock)   # a contracts.clock_fit.ClockMap fed by add_hello; add_state ignored
class OneWayPcClock(PcClock)      # a contracts.clock_fit.OneWayClockMap fed by add_state; add_hello kept for checks
def make_pc_clock(estimator: str, settings: ClockSettings, clock) -> PcClock   # link.yaml clock.estimator
```

Every refit is compared with the previous one at "now"; a move above `jump_log_ms` is an event and a `clock.jsonl`
line (guard 3). The fake PC and the tests run the PC clock as `laptop × (1 + drift) + offset` (TOP §6.3).

### 8.4 Start-up checks (`StartupChecks`)

Run at start-up step 5 on the newest hello reply, by the manager, `pc_probe --check` and so by `policy_deploy check`.

| Check | Rule | On failure |
|---|---|---|
| equality | `deploy_contracts.compare_hello(reply, mock=<own --mock>)`: `schemas`, `mock`, `F_T_TCP`, `rotation_tol`, `gripper_width_max_m`, `max_waypoints`, `watchdog_limit_s` (TOP §3.9) | refuse, one line per `HelloMismatch` |
| worst gap | `heartbeat_period_s + c3_timeout_s + timeouts.health_s + link_jitter_allowance_s < reply.watchdog_limit_s` (0.57 < 1.0) | refuse: `worst heartbeat gap 0.57 s is not below the PC's watchdog 0.50 s: raise watchdog_limit_s in link.yaml or lower the laptop timeouts` |
| home timeout | `timeouts.home_s > reply.homing_timeout_s` | refuse |
| waypoint lead | `(exec_h − 1) × 50 ms + lookahead_ms < reply.max_lead_s × 1000` (the lead at send time is smaller still) | refuse |
| clock agreement | once the map is ready: `|agreement_ms|` of the shortest-RTT hello of the start-up burst `≤ clock.startup_agree_ms` (5 ms) (PLAN §5.6 guard 2) | refuse: `clock mismatch at start-up: map and hello differ by 7.2 ms: run pc_probe; try clock.estimator round_trip` |
| commit | `reply.commit` vs `git -C <deployment> rev-parse HEAD` (`unknown` if git fails; `+dirty` noted) | **warning** only (TOP §8.1) |
| settled | step 8, with the policy's `kind` | refuse (above) |

### 8.5 The failure table as the manager implements it

One dict `RULES` in `failure_monitor.py` (old:message_comms/message_comms/failure_monitor.py:20). "Skip" = no
waypoints from this step, the PC finishes its plan and holds, heartbeats go on. "Conclude" = §6.4.

| Code | Condition | Detected by | Kind | Message |
|---|---|---|---|---|
| `CAMERA_STALE` | a camera `stale`, or a failed pick (C1 reports `stale` for that reply) | C1 `get_frames` | slow | skip; `SKIPPING <s>: <reason>` after `skip_warn_s` |
| `NO_NEW_FRAME` | newest main image already used (laptop `main_t_ns[1]` ≤ last anchor's) | `PolicyStep` | none | try next cycle |
| `IMAGE_OLD` | `now − main_t_ns[1] > max_image_age_ms` (laptop clock) | `PolicyStep` | slow | skip |
| `CLOCK_STALE` | PC map not ready (newest sample older than `max_sample_age_s`) | `PolicyStep` | slow | skip |
| `STATE_GAP` | C2 `seq` gap around an image time | `StateHistory.pose_at` | slow | skip |
| `STATE_OLD` | `now_pc − newest t_ns > state_stale_s` (100 ms, mapped), or `TOO_OLD`/`TOO_NEW`/`EMPTY` | `PolicyStep` | slow | skip |
| `ODD_CHUNK` | step 0 > 2 cm / 5° from the anchor | `PolicyStep` | slow | skip, count |
| `POLICY_LATE` | `skipped == N` (not counted toward `CLOCK_MISMATCH`, §7.5) or margin < 0 (a set not entirely skipped) | `set_waypoints` reply | slow | count `late` |
| `SHORTENED`, `PULLED_BACK`, `WIDTH_CLIPPED`, `LOOP_LATE`, `ANCHOR_FALLBACK` | normal operation | C3 reply, C2 deltas, C4 `width_clipped`, anchor rule | normal | counted, status line |
| `CAMERA_DOWN` | a camera `down` (2 s), the Pi lost or restarted | any C1 reply | conclude | `<camera> camera down: check the Pi and ASUS_5g` |
| `POLICY_HUNG` | no `infer` reply in `policy_hang_s` | C4 timeout | conclude | `policy: no reply to infer within 5.0 s: check the policy log and the GPU` |
| `POLICY_ERROR` | `ErrorReply` from policy, `Refused` reset/prompt | C4 | conclude | `policy: <error>: check the policy log` |
| `INVALID_CHUNK` | chunk fails C4 | C4 decode | conclude | `policy: invalid chunk: <field>: <detail>: check the checkpoint and policy.kind` |
| `NO_ANSWER` | a laptop process stops answering (C1, C4 config/reset/prompt timeout) | timeout | conclude | `<process>: no reply to <request> within <t> s: check that <process> is running (see its log)` |
| `PC_NOT_ANSWERING` | any C3 timeout except `home` | `TransportTimeout` on `c3.*` | conclude | `PC not answering: no reply to <request> within 0.3 s: check franka_control on the PC and the link` |
| `STATE_LOST` | no C2 for `state_lost_s` (2 s), by laptop **arrival** time: liveness, since `t_ns` can repeat (TOP §3.6 obligation 1); staleness is `STATE_OLD`, by mapped `t_ns` | `check_state` | conclude | `PC not answering: no C2 state for 2.0 s: check franka_control on the PC and the link` |
| `PC_RESTARTED` (new) | C2 `seq` goes back (franka_control restarted), or a hello's `t_pc_ns` goes back (the PC rebooted) | `StateHistory.add`, `HeartbeatSender` | conclude | `franka_control restarted on the PC: restart policy_deploy run`; for the hello case `the PC rebooted (its clock went back): start control_deploy run on the PC, then policy_deploy run` |
| `ARM_FAULT` | C2 `mode == fault` (any phase) | `check_state` | conclude | `the PC reports fault: <error>: the PC holds; Ctrl-C and restart control_deploy run on the PC (release the user stop if it was a reflex)` |
| `GATE_REJECTED` | `rejected` non-empty | C3 reply | conclude | `the PC's gate rejected waypoint <i>: <reason>: check the workspace box and the policy` (`<i>: <reason>` from `gate_rejection_detail`: the first rejected waypoint whose reason does not start with `not checked`, TOP §3.7.1) |
| `PC_STOPPED` (was `ARM_STOPPED`) | C2 `stopped` in `RUNNING` without our `stop`; C2 messages with `t_ns < start_ack_pc_ns` (the `start` ack's laptop time mapped to PC time, §6.4) are ignored | `check_state` | conclude | ``the PC stopped the arm (watchdog or the PC's `s`): check the link and the PC`` (PLAN §7) |
| `PC_REFUSED` (new) | `error/v1` without a field from the PC (e.g. `set_waypoints` refused by the mode) | `RemoteError` on `c3.*` | conclude | `the PC refused <request>: <error>: check the PC terminal` |
| `CLOCK_MISMATCH` (new) | three sets in a row that were entirely skipped while the laptop thought them ≥ 100 ms ahead, or sent while the newest C2 `t_ns` was > 20 ms ahead of the mapped "now" | §7.5 | conclude | `clock mismatch: the PC skipped 3 waypoint sets the laptop timed ahead: run pc_probe; check clock.estimator` |
| `HELLO_MISMATCH` (new) | §8.4 at start-up, incl. the 5 ms clock agreement | `StartupChecks` | conclude | `hello mismatch: <field>: PC <v>, laptop <v> (<source>): <check>` |
| `PC_BUSY` (new) | C2 `running`/`holding` after the H2 wait, or at step 6 | `StartupSequence` | conclude | `the PC is driving an episode for another C3 client: ...` |
| `TCP_UNSETTLED` (new) | trained policy, `settled: false`, real PC | start-up step 8 | conclude | as §6.3 step 8 |
| `CONTRACT_VIOLATION` | any message failing its contract (C1, C2, C3 incl. `error/v1` with a field, non-chunk C4; the client's extra checks; policy config sizes) | `ContractError`, `RemoteError` with field | conclude | `<message>: field '<f>': <detail>: check that both machines run the same commit` |
| `HOMING_FAILED` | `home` answered not ok **without** the `stopped:` prefix (refused, failed, timed out on the PC), an `error/v1`, or no reply in `home_s` | `pc.home` | conclude | `the PC's homing failed: <reason>: check the arm's path to the home pose` |
| `HOME_STOPPED` (new) | `home` answered not ok with the `stopped:` prefix (the PC's `s` stopped the move; TOP §3.7.2) | `pc.home` | normal | status line and event `home stopped on the PC: <reason>: the arm holds where it stopped`; `IDLE`; clock map re-warmed (§6.4) |
| `STARTUP` | §6.3 not done in time | `StartupSequence` | conclude | `startup: <what> within 20 s (<why>): check <process>` |
| `OPERATOR_SCRIPT`, `INTERNAL_ERROR` | as old | | conclude | as old |
| `EPISODE_STOP` | `s` | key | episode ends | `stop`, record `stopped`, back to `IDLE` |
| `OPERATOR_STOP` | `q`, Ctrl-C | key, SIGINT | stop | `stop`, record `stopped`, exit 0 |
| `LAUNCHER_STOP` (was `DEPLOY_STOP`) | SIGTERM from the launcher | signal | stop | `stop`, record `aborted` (`stopped by the launcher: another process exited`), exit 0 |

Not problems (no code, never a conclude): `c` or `h` refused locally while the newest C2 mode is `homing` (the old
`_refuse` path: a hint on the status line, and a scripted operator retries the key until `operator_script_wait_s`);
a `start` the PC refuses (event `start_refused`, a hint, back to `IDLE`, as old). Both are in §6.4.

`from_exception` maps as old §4.7, with: `TransportTimeout` on `c3.*` (not `home`) → `PC_NOT_ANSWERING`;
`RemoteError` from `PC` with `field` → `CONTRACT_VIOLATION`, without → `PC_REFUSED`; `PcRestarted` →
`PC_RESTARTED`. The PC side of "the laptop dies, sleeps or the link drops" needs nothing here: the PC's watchdog
stops following and holds, and the next laptop run reconnects (§6.3).

### 8.6 `pc_probe` (read-only)

`python -m policy_deploy.manager.pc_probe --station FILE [--seconds 30] [--local] [--check] [--json]`. A C3 client,
so never alongside a manager (TOP §2.4). It follows rule H2 first (waits while C2 shows `running`/`holding`/`homing`;
refuses after the limit). Then for `--seconds`: C2 rate, `seq` gaps, `t_ns` repeats, newest-state age; a hello every
0.5 s (a burst of 2 s first) and heartbeats in between, with RTT p50/p95/max; **both** PC estimators fed side by side
(a `RoundTripPcClock` and an `OneWayPcClock`) and their offsets, drifts and difference printed every second; the hello
reply and `compare_hello`. It never sends `start`, `stop`, `home` or `set_waypoints`. `--check` runs 2 s and prints
one JSON check line per §8.4 row and per link fact (C2 rate, RTT, clock agreement), for the launcher (§9.2). Its
clock agreement is judged with the map of `link.yaml clock.estimator` (the one_way map when so configured), as `run`
judges it at start-up step 5; with round_trip the row would pass by construction, since that map is fitted from the
same hellos.

## 9. The launcher (`policy_deploy.launcher`)

Copied from old:deploy (managed processes, session, checks, table), for the laptop's processes only. The geometry
tool, `gripper.yaml`, the TCP-triangle check (old:deploy/deploy/launcher.py:13) and the `franka_control` process are
not carried over (PLAN §8.2, §9).

### 9.1 Classes

| Class (file) | From | Changes |
|---|---|---|
| `CheckResult`, `print_table`, `exit_code` (`check_result.py`) | copied | none |
| check helpers (`check_helpers.py`): `file_exists`, `executable`, `port_free`, `python_imports`, `host_reachable`, `json_check_lines(cmd)` | old:deploy/deploy/check_helpers.py | adds `json_check_lines` (runs a `--json` self-check subprocess and turns lines into rows) |
| `Session` (`session.py`) | old:deploy/deploy/session.py | also copies the three contract files into the session's `contracts/` folder (never next to the `frames/` folder, so `frames.yaml` is never mistaken for the frames process's settings) |
| `ManagedProcess` (`managed_process.py`) | old:deploy/deploy/managed_process.py | `program()`/`checks()` given by `processes.py`; cwd = `deployment/policy_deploy/` |
| `FakePcProcess`, `FramesProcess`, `PolicyProcess`, `ManagerProcess` (`processes.py`) | old `launch.py` files of each folder | one class per process in one file; their checks call subprocesses (`frames.rig_file --json`, `policy.self_check`, `manager.self_check`) |
| `Launcher` (`launcher.py`) | old:deploy/deploy/launcher.py | `ORDER = ("fake_pc", "frames", "policy", "manager")`, `fake_pc` only with `--mock` and `launcher.fake_pc: true` (the default); no geometry check |
| `cli.py`, `errors.py` | copied | command name `policy_deploy` |

Ready files and timeouts (`config/launcher.yaml`, overridable in `station.yaml launcher:`):

| Process | Ready when | `ready_timeout_s` | Interpreter |
|---|---|---|---|
| `fake_pc` (`--mock`) | C2 and C3 bound on `127.0.0.1`, first C2 published | 10 | light |
| `frames` | C1 answers, both cameras `ok` once | 30 | light |
| `policy` | model loaded, warmed up, C4 answers | 300 | torch for `diffusion`, else light |
| `manager` | §6.3 done (`USES_TERMINAL`: keyboard and status line) | 130 (above the longest legal start-up, 115 s: `startup_s` 20 + `pc_homing_wait_s` 60 + `homing_timeout_s` 30 + 5, §6.3) | light |

`stop_timeout_s` 5.0, `poll_s` 0.1. `fake_pc: true`: under `--mock` the launcher starts the fake PC; `false` is only
for the cross test (§12.3), where a real `franka_control --mock --local` already listens on `127.0.0.1` with the
`link.yaml` ports and the launcher starts frames, policy and manager exactly as under `--mock`. Outside `--mock` the
key has no effect (the PC is never started by the laptop).

### 9.2 `policy_deploy check [--station FILE] [--mock]`

Starts nothing that moves; prints one table; exit 2 if any `fail`.

| Process | Row | How | Fail when |
|---|---|---|---|
| station | `station.yaml` | `Station.load` | any problem (all listed) |
| station | `deploy_contracts` | `deploy_contracts.paths.contracts_dir()` is this checkout's `deployment/contracts` | missing or another checkout: `uv pip install -e deployment/contracts into ~/envs/policy_deploy` (TOP §1.4) |
| station | `contracts/link.yaml`, `frames.yaml`, `limits.yaml` | `LinkSettings.load`, `TcpFrame.load`, `ContractLimits.load` | invalid |
| station | `laptop address` (not mock) | one of `link.yaml laptop.addresses` is on a laptop interface | missing (the PC's firewall would drop the laptop) |
| station | `Wi-Fi` (not mock) | `nmcli -t -f active,ssid dev wifi` | **warn** when not on `wifi_ssid` (wired runs may be fine) |
| station | `commit` | `git rev-parse HEAD` | info |
| station | `disk` | free space under `session_root` | **warn** below 2 GB |
| frames | `rig file` | subprocess `frames.rig_file <file> --json` | invalid |
| frames | `imports` | `python_imports(light, [cv2, scipy, zmq, av])` | missing |
| frames | `C1 port` | `port_free(127.0.0.1:5610)` | busy |
| frames | `Pi reachable` (not mock) | TCP connect to `pi.address:5555`, 1 s | `no TCP connection to <pi>:5555: is the Pi on ASUS_5g and its sender running? (station.yaml pi.address)` |
| frames | `mock episode`, `fake Pi port` (mock) | folder with `meta.json`; `port_free(5600)` | missing, busy |
| policy | `kind` | config | unset (not mock) |
| policy | `settled` | `contracts/frames.yaml settled` vs `kind` (not mock) | `kind != hold_still` and `settled: false` |
| policy | `torch env` (diffusion) | `python.torch` exists and is executable | missing |
| policy | self-check rows | subprocess `policy.self_check --json` in the kind's env: settings, checkpoint exists, prompt dataset exists (when `prompt_dataset` is set), imports (torch, behavior_prompting), CUDA, GPU name, free VRAM, other GPU processes (**warn** if > 300 MB used by others), `device`, `precision`, `diffusion_steps` | as old §8.2 |
| policy | `C4 port` | `port_free(5640)` | busy |
| manager | self-check rows | subprocess `manager.self_check --json`: settings in range, operator script or a terminal, the worst-gap rule against `link.yaml watchdog_limit_s`, `exec_h` (info, PLAN §13 backup), `clock.estimator` (info) | as old |
| manager | PC rows (not mock) | subprocess `manager.pc_probe --check --json`: C2 arriving (rate), H2 state, hello answered (RTT), every §8.4 row, PC commit vs laptop commit (**warn** on difference), `PC mock` | no C2, PC busy, hello mismatch, budget |
| fake_pc | `ports` (mock, `fake_pc: true`) | `port_free(127.0.0.1:5620/5630)` | busy |

Under `--mock` the PC rows are one `info` (`fake PC on 127.0.0.1, started by run`, or with `fake_pc: false`
`PC on 127.0.0.1 started outside the launcher (cross test)`). `check` is a C3 client for about 2 s
(the PC rows); never run it while a laptop run is going.

### 9.3 `policy_deploy run [--station FILE] [--mock]`

1. `Station.load`; all checks; any `fail` → `policy_deploy: not started: N check(s) failed (see above)`, exit 2.
2. `Session.create`: `<session_root>/<YYYY-MM-DD_HH-MM-SS>[_mock]/`.
3. In `ORDER`: start, wait for its ready file, then the next. Never anything on the PC.
4. `policy_deploy: all N processes ready (session: <path>)`; the manager owns the terminal.
5. Wait. The manager exiting 0 → stop the rest in reverse, `policy_deploy: done`, exit 0. Ctrl-C → the manager gets
   it too (it sends `stop`, records `stopped`, exits 0); the launcher waits for it, stops the rest, exit 130. Any
   process exiting otherwise → stop all in reverse (the manager gets SIGTERM: `stop`, episode `aborted`), exit 1 with
   one line.
6. Reverse order: `manager`, `policy`, `frames`, `fake_pc`. The PC keeps holding the arm.

One-line messages (old:deploy/STRUCTURE.md §5.3):

```
policy_deploy: FAILED manager (exit 1): FATAL manager: PC not answering: no reply to c3.heartbeat/v1 within 0.3 s: check franka_control on the PC and the link  [log: <session>/logs/manager.log]
policy_deploy: FAILED frames during start-up: not ready after 30 s (waiting for: frames from both cameras; check the Pi on ASUS_5g and station.yaml pi.address)  [log: ...]
```

Session folder:

```
<session_root>/<YYYY-MM-DD_HH-MM-SS>[_mock]/
  station.yaml                                              copy, as run
  contracts/  link.yaml  frames.yaml  limits.yaml           copies of the contract files, as run
  logs/  launcher.log  fake_pc.log  frames.log  policy.log  manager.log
  ready/ <name>.ready
  fake_pc/  commands.jsonl                                  every C3 request and reply (mock)
  frames/   stats.jsonl
  policy/   startup.json
  manager/  session.yaml  episodes.jsonl  steps.jsonl  events.jsonl  clock.jsonl  link.jsonl  tap/
```

## 10. `fake_pc` (the PC under `--mock`)

`python -m policy_deploy.fake_pc --station FILE [--session] [--ready-file] --mock`, also runnable alone. Binds C2 PUB
and C3 REP on `127.0.0.1` with the `link.yaml` ports (TOP §2.3), so the manager connects exactly as to the PC. A
stand-in for the messages and modes of franka_control, not for its physics.

### 10.1 What it implements

| Part | Class (file) | Behaviour |
|---|---|---|
| clock | `DriftingClock` (`drifting_clock.py`) | `t_pc = laptop_monotonic × (1 + drift_ppm·1e-6) + offset_s` (defaults +1000 s, 30 ppm); used for C2 `t_ns`, `hello.t_pc_ns` and its gate's "now" |
| C2 | `FakePc` thread "fake-pc-state" | a `deploy_contracts.Publisher` at `state_rate_hz` (100): `t_ns` (cycle start), `seq` from 1, poses, width, mode, error, gate counters, `loop_late` 0, `q`/`dq` constant; `t_ns` frozen once the 1 kHz loop has ended (`inject_fault`, `start_mode: fault`), as on the PC; after a gate rejection the real loop keeps holding, so `t_ns` keeps advancing in that `fault` |
| C3 | `FakePc` thread "fake-pc-commands" | a `deploy_contracts.RpcServer(..., schemas=SHARED_SCHEMAS)` with handlers for every C3 request; replies per mode exactly as TOP §3.7.2; broken requests get `error/v1` naming the field (the `RpcServer` does this); a laptop-only schema gets `error/v1` `unknown_schema` through the schema filter (TOP §3.11), although this process registers C1/C4 too; every request that reaches a handler feeds the watchdog (broken ones never reach one, as on the PC); refusals carry the real franka_control texts (`StateMachine::check`): `refused: mode is <mode>`, then `: <fault detail>` in `fault` or ` (<cause>)` for a `stopped` not caused by a stop request (a watchdog, a stopped homing), then `: <why>` (`already started`, `send stop first`, `send start first`, `homing in progress`); a refused `set_waypoints` is `error/v1` `set_waypoints refused: …` |
| hello | | `t_pc_ns` read immediately before the reply; `commit` = the laptop's `git rev-parse HEAD` or `unknown`; `mock: true`; `schemas` = `SHARED_SCHEMAS`; `F_T_TCP` from `contracts/frames.yaml`; the three limits; `watchdog_limit_s` from `link.yaml`; `homing_timeout_s` and `max_lead_s` from `config/fake_pc.yaml` (30, 1.5) |
| start / stop | | `idle`/`stopped` → `holding` at the current target; `stop` → `stopped`; a C3 `stop` during a keyboard homing (`keyboard_home()`, `keyboard_hand_home()`) aborts it: the target stays where the ramp was, mode `stopped`; if the ramp already ended, `idle` (TOP §3.7.2) |
| home | | refused unless `idle`/`stopped`; `homing` for `home_duration_s` (1.0) moving the target linearly to `home_pose`; the REP answers nothing else meanwhile (as the PC); then `idle` and the reply `ok`. A `keyboard_stop()` meanwhile ends it: `stopped` where the ramp was, and the reply `ack {ok: false, reason: "stopped: homing stopped by the PC keyboard (fake PC)"}` with the contract prefix |
| gate | `FakeGate` (`fake_gate.py`) | time passed → skipped; more than `max_lead_s` ahead, width outside `[0, gripper_width_max_m]` → rejected, `fault`; accepted ones replace the plan from the current target; mode `running`; every waypoint skipped (none rejected): the plan becomes the target at that moment, mode `holding` (as the PC: the arm freezes, it does not finish the previous plan); no box, cone or speed rules |
| follower | `PlanFollower` (`plan_follower.py`) | target = linear position / slerp rotation between plan waypoints at "now", holding after the last (mode back to `holding`); measured = target + `measured_offset_m` (0 by default; a few mm in anchor tests); width moves toward the target width at `width_speed_m_s` |
| watchdog | | armed in `holding`/`running`; no C3 request for `watchdog_limit_s` → `stopped`; paused while homing |
| log | | every request and reply to `<session>/fake_pc/commands.jsonl` |

**Test hooks** (methods on `FakePc`, used in-process by tests; `config/fake_pc.yaml start_mode` sets the first mode
for subprocess tests): `inject_fault(text)`, `restart()` (`seq` back to 1), `stall(seconds)` (neither C2 nor C3),
`delay_states(dist)` (uneven C2 delays), `burst_after_stall()`, `drop_states(n)` (a `seq` gap), `repeat_t_ns(n)`,
`set_mode(mode)` (e.g. an orphaned `holding` for the H2 wait), `shift_clock(seconds)` (to provoke a clock mismatch),
`report_shortened(n)` / `report_pulled_back(n)`, and the PC keyboard:

| Hook | As the PC's key | Does |
|---|---|---|
| `keyboard_home()` | `h` | refused unless `idle`/`stopped` (returns the refusal); else `homing`, the target ramps to `home_pose` over `home_duration_s` while the REP keeps answering (`start`/`home` refused, `stop` aborts), then `idle` |
| `keyboard_hand_home()` | `g` | refused unless `idle`/`stopped`; else `homing`, the arm target held, the width ramps to 0 and back to the commanded width at `width_speed_m_s`, the REP keeps answering, then `idle` |
| `keyboard_stop()` | `s` | `idle`/`holding`/`running` → `stopped`; in `stopped` and `fault` no change; during any homing it aborts it → `stopped` (during a pending C3 `home`, that `home`'s deferred reply is `ok: false` with the `stopped:` prefix) |

### 10.2 What it deliberately does not do

No libfranka, no dynamics or impedance, no workspace box, tilt cone or speed shortening (only counters via hooks), no
joint motion, no gripper physics beyond a width ramp, no real-time loop or `loop_late`, no keyboard (hooks only), no
bind-address or firewall rules, no `--local`. The real behaviour is tested by `control_deploy` and by the cross test
(§12.3).

## 11. Measuring the policy on this laptop (PLAN §8.2, §12 step 2)

`run_001` is a **prompting** checkpoint (`use_prompting: true`, a prompt transformer over 512 tokens,
`results/block_raise/run_001_2026-10-05/run_config/config.yaml:89`). It is served by `PromptableDiffusionPolicy`
(kind `diffusion`, with `prompt_dataset`), but its encoder differs from the plain diffusion policy that will run
(PLAN §1). So timing uses `policy.bench` (§14 decision 2).

### 11.1 `policy.bench` (torch env)

```
python -m policy_deploy.policy.bench [--config umi_policy_dunet | --checkpoint PATH] [--steps 16,8,4]
       [--precision fp32,bf16] [--infers 500] [--warmup 3] [--device cuda] [--out DIR]
```

- `--config umi_policy_dunet` (default): composes behavior_prompting's plain UMI config (`model: diffusion_unet`:
  `TimmObsEncoder` with `vit_base_patch16_clip_224.openai`, UNet `[256, 512, 1024]`, DDIM 16 steps), with
  `pretrained: false` (no download), instantiates `cfg.policy` at random weights, `eval().to(device)`, and feeds
  `neutral_observation()` converted by `ModelRunner`'s own conversion. Timing does not depend on the weights.
- `--checkpoint PATH`: loads through `CheckpointLoader`; reports load time, host RAM peak and VRAM of the real
  weights; times inference for a plain checkpoint, or for a prompting one with `--prompt-dataset ZIP
  [--prompt-split train]` (prompted once with the longest demo first, as the warm-up).
- For each `steps × precision`: load s, warm-up times, mean/p50/p95/max over `--infers` sustained calls,
  `torch.cuda.max_memory_allocated/reserved`, `nvidia-smi` used memory, power draw, SM clock and temperature at the
  end (throttling at the 35 W cap). One line each, and `bench.jsonl` in `--out`.

### 11.2 Procedure

1. Laptop on AC, `powerprofilesctl set performance`, browser closed (RAM: ~6 GB free today; loading needs 5-8 GB).
   `nvidia-smi`: no process on the dGPU, 4096 MiB, note the power limit.
2. Torch env installed (§2.3); `python -m policy_deploy.policy.self_check --station station.yaml` in it: CUDA, GPU.
3. `bench --config umi_policy_dunet --steps 16,8 --precision fp32,bf16 --infers 500`.
4. `bench --checkpoint .../run_001_2026-10-05/checkpoints/best_slim_epoch=0050-...ckpt` (load time, RAM, VRAM).
5. Under load: repeat step 3 with `--infers 200` while `python -m policy_deploy.frames` runs on the live Pi in another
   terminal (CPU and memory-bus contention from JPEG decode).
6. Once a plain checkpoint exists: the real path, `python -m policy_deploy.policy` (`startup.json`: warm-up and VRAM),
   `policy.probe --infers 500`, then `policy_deploy run --mock` with `policy.mock_kind: diffusion` for `L`, `C'` and
   margins from `steps.jsonl`.
7. Record the table in DIRECTIONS.md and set `manager.exec_h` (and `policy.precision` / `diffusion_steps` if needed)
   in `station.yaml`.

### 11.3 Decision rule for `exec_h`

With `I95` = p95 inference of step 5, `C' = 8 ms (frames + conversions) + I95 + RTT95`, `L95` = 110 ms until measured
live (then p95 of `image_age_ms`), and a 20 ms reserve:

1. The smallest `exec_h` in `{8, 10, 12}` with `L95 + 2·C' + RTT95/2 + 20 ≤ (exec_h − 1) × 50` (fp32, checkpoint
   steps). Training's runner used 12 (`config.yaml:219`).
2. None fits → `precision: bf16` and repeat 1.
3. Still none → `diffusion_steps: 8` and repeat 1 (it changes the model's behaviour: check episodes).
4. VRAM: peak reserved above 3.6 GB of 4 GB, or an OOM → `bf16` first; report if still over (a weight-cast mode would
   be a new setting).

Example: `I95` = 180 ms, RTT 10 ms: `C'` = 198 ms; exec_h 8 needs ≤ 350 (fails: 531); 10 needs ≤ 450 (fails); 12
needs ≤ 550 (passes: 531). So `exec_h` 12, fp32.

## 12. Tests

All on the laptop with no Pi, arm or GPU; sockets on `tcp://127.0.0.1:*`; no sleeping to wait (fake clocks,
`ManualWorker`, condition timeouts). From `deployment/policy_deploy/`:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/policy_deploy/bin/python -m pytest tests            # torch tests skip
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/policy_deploy_torch/bin/python -m pytest tests/contracts tests/policy
```

### 12.1 Carried over (adapted to the new names, clocks and station)

| Folder | Old files |
|---|---|
| `tests/contracts` | old:contracts/tests/test_c1.py, test_c4.py (+ `kind`), test_pi_stream.py, test_c0.py (internal packet only), the C0/C1/C4 part of test_examples.py; old:process_frames/tests/test_clock_map.py (now `clock_fit`, plus `to_remote_ns`) |
| `tests/frames` | old:process_frames/tests/ test_imageprep.py, test_vendored_camera_model.py, test_imageprep_live.py, test_copies.py, test_rig_file.py, test_settings.py, test_frame_times.py, test_packet_pairer.py (rewritten for nearest pairing, §12.2), test_packet_buffer.py, test_packet_selector.py, test_camera_health.py, test_receiver.py (JPEG only), test_pi_recording.py (JPEG only), test_fake_pi_sender.py (`pi_v1` only), test_frames_server.py, test_pipeline.py (`pi_v1`), test_real_episode.py (skipped without `~/Downloads/episode_000045`), test_pi_v1.py, test_pi_collect.py, test_main.py, test_decoders.py (JPEG part); dropped: test_clock_sync.py, the H.264 tests, test_launch.py (moves to `tests/launcher`) |
| `tests/policy` | every file of old:policy/tests (test_settings, test_make_chunk, test_key_map, test_hold_still_policy, test_policy_server, test_neutral_observation, test_policy_factory, test_import_without_torch, test_app, test_warm_up, test_self_check, test_probe, test_diffusion_policy, test_checkpoint_loader); test_launch.py moves to `tests/launcher`; new test_promptable_diffusion_policy (`POLICY_TEST_CHECKPOINT` + `POLICY_TEST_PROMPT_DATASET` add the real run_001 check) |
| `tests/manager` | old:message_comms/tests/ test_pose_math.py, test_pose_converter.py, test_state_history.py, test_state_listener.py (now `PcListener`), test_clients.py (now `PcClient` incl. hello), test_policy_step.py, test_step_worker.py, test_episode_loop.py, test_failure_monitor.py, test_operator.py, test_status_line.py, test_session_log.py, test_settings.py, test_main.py, test_flow.py (against `FakePc` instead of `FakeFrankaControl`); helpers `fakes.py`, `harness.py`, `raw_server.py`, `reference_pose_utils.py` |
| `tests/launcher` | old:deploy/tests/ test_station.py (new schema), test_config_merge.py, test_session.py, test_check_result.py, test_check_helpers.py, test_managed_process.py, test_launcher.py (new order), test_check.py, test_mock.py, test_cli.py, and each folder's test_launch.py as process-class tests; dropped: test_geometry.py, test_gripper_file.py, test_tcp_triangle.py |

### 12.2 New

| File | Checks |
|---|---|
| `tests/test_import_rules.py` | §2.2, by `ast`; no old package names imported |
| `tests/contracts/test_c4_v2.py` | `kind` required, non-empty; v1 config reply refused (`schema_version`) |
| `tests/contracts/test_station.py` | the example station loads (its `ports`, `pi.address` and `frames:` are legal); each refused name of §2.4 refused at any level (e.g. `manager.c3_timeout_s`, `frames.F_T_TCP`) and each top-level one (`pc`, `laptop`, `timing`, `clock`) at the top, naming the contract file; `manager.clock` stays legal; `pi.address` required except mock; `python.torch` only for `diffusion`; `paths.config_file("frames")` is `config/frames_process.yaml` |
| `tests/frames/test_frozen_ultrawide.py` | with `FakePiSender.repeat_ultrawide` and with hand-made packets: one repeat (the same pair for 2 consecutive packets) still served; a second repeat (3rd consecutive packet) → unpaired, `pair_reused`, **at the live phase** (gap −16.3 ms, so the 3rd packet's gap is +50.4 ms, inside the window) and at the opposite phase (+17 ms); the run count resets on a new `pair_seq` and after a Pi restart; a repeat lasting 0.4 s → ultrawide `stale` (steps get no images), 2.1 s → `down`; ultrawide health ages from the frame's own mapped capture time, not the packet time; the store evicts by time on every packet (a repeated old `frame_no` is not found after `keep_s`); a pair newer than `pair_max_lead_ms` → `pair_too_new`, older than `pair_max_age_ms` → `pair_too_old`; edges exactly at +60 ms and −25 ms accepted, 1 ns beyond either refused; new ultrawide frames whose pair is always refused (`pair_too_new`) → `stale`, then `down`; the reuse run counts `packet_seq` (a lost first repeat does not let the second through) |
| `tests/frames/test_packet_pairer.py` (replaces the old file) | nearest-frame pairing: the nearer of the frames before and after the main frame, a tie to the older; a later frame within half a period wins (the live case, ultrawide 16 ms newer); no age cut (a stalled stream keeps pairing its last frame); `NO_PAIR` only before the first frame; rule 2 (each frame sent once, in order) as old. `FakePiSender` pi_v1 messages have the nearest pair and `skew_ns` = `ultrawide.t_mid − main.t_mid` (the live sign) |
| `tests/frames/test_pi_collect.py` (added cases) | the pair-gap report computes from the two `t_mid_ns`, not `skew_ns` (a message with a reversed `skew_ns`, as the live Pi sends, gives the same report); min/median/max, histogram, longest same-pair run |
| `tests/manager/test_pc_clock.py` | each estimator with a fake PC clock offset +1000 s and drift ±30 ppm: error < 0.5 ms with symmetric delays; **uneven delays** (each direction 1 ms + exponential jitter, uneven): `round_trip` error < half the minimum RTT + 0.5 ms; **one-way bias** equals the minimum delay; outliers (5 % delayed 50-300 ms, and a burst) never move the fit; **repeated `t_ns`** never become `one_way` samples; `get_state` replies never enter; `ready()` needs 2 s and 10 samples and turns false after `max_sample_age_s`; drift above `max_drift_ppm` → 0; one snapshot maps both rows and "now" consistently across a refit; a refit jump is reported; thread safety |
| `tests/manager/test_startup_checks.py` | an equal hello passes; each `compare_hello` field differing refuses naming it; worst gap 0.57 vs watchdog 1.0 passes, vs 0.5 refuses with both values; `home_s ≤ homing_timeout_s` refuses; `exec_h` 16 with `max_lead_s` 0.5 refuses; clock agreement 4.9 ms passes, 5.1 ms refuses (shortest-RTT hello used); commit differing → warning only, `unknown` → warning; settled rule: `diffusion` + `settled: false` + real PC refuses, + mock PC passes, `hold_still` always passes |
| `tests/manager/test_startup_sequence.py` | with `FakePc` in-process and `FakeClock`: the launcher's manager timeout (130 s) exceeds `startup_s + pc_homing_wait_s + homing_timeout_s + 5` from the default configs; the exact order of §6.3 (the fake's request log shows no C3 before the H2 wait ends, then hellos, `get_state`, heartbeats); **H2**: an orphaned `holding` turns `stopped` by the fake's watchdog within 2 × watchdog and start-up continues; a `holding` that stays (hellos from another "client" hook) → `PC_BUSY` after the limit, nothing sent; `homing` at start → no C3 until `idle`; `homing` at step 6 → waits, then goes on; `fault` → refused at once with the error text; no C2 → `PC not answering` after `startup_s`; hello mismatch → refused; ready file only after cameras ok and config checked |
| `tests/manager/test_heartbeat_sender.py` | one C3 request per `beat`; a hello every `hello_period_s` replacing the heartbeat; every cycle in start-up mode; samples reach the clock; RTTs reach `LinkStats`; a hello `t_pc_ns` going back → `PC_RESTARTED` with the reboot text; hello uses no `poll_s`/`cancel` |
| `tests/manager/test_failure_rows_pc.py` | each new row end to end against `FakePc`: `restart()` → `PC_RESTARTED`; `stall(0.4)` → C3 timeout `PC_NOT_ANSWERING`; C2 stall 2.1 s (C3 still answering) → `STATE_LOST`; `repeat_t_ns` for 1 s (as after a stopped homing) with C2 still arriving → no `STATE_LOST`; a burst of old states after a stall → `STATE_OLD` skips, never fresh by arrival; watchdog trip → `PC_STOPPED` with PLAN §7's text, and the same for `keyboard_stop()` during an episode; a `stopped` C2 older than the mapped `start` ack is ignored (C2 delayed 30 ms until the ack is mapped, then no PC_STOPPED and still running after 3 steps; fake PC clock +1000 s); `shift_clock(+2 s)` → three skipped sets → `CLOCK_MISMATCH`; `shift_clock(+0.1 s)` with the laptop's refit held (the map 100 ms behind: waypoints accepted, late) → three sets with a future C2 → `CLOCK_MISMATCH`; one skipped set → only `POLICY_LATE`; `h` then `keyboard_stop()` mid-home → `HOME_STOPPED`, `IDLE`, no conclude, the map re-warmed and `c` accepted afterwards; a home refused or timed out → `HOMING_FAILED`; `keyboard_home()` / `keyboard_hand_home()` in `IDLE` → `c` and `h` refused locally (no C3 sent, no conclude), accepted again after `idle`; a `start` refused by the PC → `IDLE` with the hint; `error/v1` without field → `PC_REFUSED`; gate rejection → `GATE_REJECTED`; `stop` sent first in every conclude, the episode `aborted` |
| `tests/manager/test_pc_probe.py` | against `FakePc`: reads C2, sends only hello/heartbeat (request log), prints both estimators, refuses while `holding` past the H2 limit; `--check --json` lines; with every C2 50 ms late, `--check`'s clock agreement passes with `estimator: round_trip` and fails with `one_way` |
| `tests/fake_pc/test_fake_pc.py` | modes and replies exactly TOP §3.7.2 (one parametrized case per cell, the `homing` column reached through `keyboard_home()`); a C3 `stop` during `keyboard_home()` → `stopped`, during a finished ramp → `idle`; `keyboard_stop()` during a C3 `home` → reply `ok: false` starting `stopped:`, mode `stopped`; `keyboard_hand_home()` → `homing`, REP answering, width back to the commanded width, `idle`; hello fields and `t_pc_ns` on the drifting clock; watchdog arms only in `holding`/`running`, trips to `stopped`, paused while homing; C3 `home` blocks other requests; `seq` from 1, +1, gap on `drop_states`; `t_ns` frozen after `inject_fault` (a loop end), advancing after a gate rejection; follower reaches waypoints at their times, holds after the last; past waypoints skipped, beyond `max_lead_s` rejected with `fault`; an all-skipped set freezes the target, mode `holding`; refusals with the real franka_control texts; replies pass the shared validators |
| `tests/fake_pc/test_shared_examples.py` | every good C3 request of `deploy_contracts.load_index()` gets a valid reply of the right schema; every broken C3 request gets `error/v1` with its expected field; laptop-only schemas (`c1.frames/v1`, `c4.chunk/v1`), with `policy_deploy.contracts` imported in the test process, → `error/v1` `unknown_schema` (TOP §3.4, §3.11 schema filter) |
| `tests/launcher/test_check_rows.py` | every row of §9.2 with a passing and a failing input; PC rows parsed from `pc_probe --check --json`; `--mock` skips Pi and PC rows |

### 12.3 End to end and across the machines

| File | What |
|---|---|
| `tests/integration/test_run_mock.py` | `python -m policy_deploy run --mock` as a subprocess with a temporary station (free ports, temporary `session_root`, `manager.operator_script`: `c`, `wait_steps 20`, `y`, `c`, `wait_steps 5`, `s`, `h`, `q`). Exit 0, `policy_deploy: done`, two episodes in `episodes.jsonl` (`success`, `stopped`), increasing `chunk_id`s accepted by the fake PC on its drifting clock (none `rejected`, `max_lead_s` respected), `clock.jsonl` offset ≈ +1000 s, `stop` the last C3 request. A second case with the fake PC starting in `holding` exercises the H2 wait. Skipped without `~/Downloads/episode_000045` |
| `tests/integration/test_cross_franka_control.py` | PLAN §10 "across the two", **on the laptop** (TOP §8.10): the laptop's processes against the **real** franka_control code with the fake arm. The binary comes from `FRANKA_CONTROL_BIN` (the libfranka-less build of control_deploy/STRUCTURE.md §8.3); the test is skipped when it is unset, and without `~/Downloads/episode_000045`. Steps: (1) write a temporary franka_control station with only `session_root: <tmp>`; (2) start `$FRANKA_CONTROL_BIN --station <tmp>/fc_station.yaml --mock --local --ready-file <tmp>/fc.ready`, wait for the ready file (30 s), skip if 5620/5630 are busy; (3) run `python -m policy_deploy run --mock` with a temporary laptop station: `launcher.fake_pc: false` (the launcher starts frames with the fake Pi, policy `hold_still`, and the manager, which connects to `127.0.0.1` with the `link.yaml` ports, §9.1), `manager.operator_script`: `c`, `wait_steps 20`, `y`, `h`, `q`; (4) SIGTERM the binary, expect exit 0. Checks: start-up passes `compare_hello` against the C++'s hello (`mock` true, same schemas, `F_T_TCP`, limits, watchdog); one episode of 20 steps with `chunk_id`s accepted on the PC's real clock; `home` answered ok; `stop` sent; the laptop exits 0. Proves the laptop and the C++ speak the same C2/C3, including `hello`, the mode table and `home`. A second case (no episode needed): with the laptop's `PcClient`, `start`, then a `set_waypoints` of 5 whose waypoint 3 is 0.5 m below the box; the real reply lists all 5, the other four with the contract's `not checked: …` text, and `gate_rejection_detail` names `3: outside workspace box by … on z …`; C2 `fault`; SIGTERM exit 0 |

## 13. DIRECTIONS.md outline (the laptop, from scratch)

1. What runs where: PC first (`control_deploy run`), then the laptop; the laptop never touches the PC.
2. Prerequisites: Ubuntu, NVIDIA driver 535, `uv`, git checkout of `rnoro/final-deployment` (same commit as the PC),
   ~10 GB free disk.
3. Light env: commands of §2.3; run the light tests.
4. Torch env: torch cu118, behavior_prompting's pip list, `-e behavior_prompting`; run the policy tests; `self_check`.
5. Network: join **ASUS_5g**; reserved addresses (PLAN §15); find the Pi's address; put it in `station.yaml pi.address`;
   put the PC's address in `link.yaml pc.address` and the laptop's in `laptop.addresses` (commit, so the PC gets it).
   Going wired later: USB-C adapter, new address in `link.yaml`.
6. `station.yaml` walk-through: `session_root`, `python.*`, `policy.kind`, `manager.exec_h`.
7. Mock first: `policy_deploy check --mock`, `policy_deploy run --mock` (keys `c`, `y`, `n`, `s`, `h`, `q`).
8. Frames alone against the live Pi: `python -m policy_deploy.frames`, `frames.probe --save`, `frames.pi_collect`
   (incl. the pair-gap report).
9. Link alone: `python -m policy_deploy.manager.pc_probe` with the PC running; reading the RTT, the two estimators.
10. Policy timing: §11, the table, choosing `exec_h`.
11. Real run with hold still: `policy_deploy check`, `policy_deploy run`; status line fields; session folder.
12. Troubleshooting: every FATAL line family and what to check; sleep/suspend off during runs; a fault needs a restart
    of `control_deploy run` on the PC.
13. The cross test (PLAN §10), on this laptop: build franka_control without libfranka as control_deploy/STRUCTURE.md
    §8.3 says (apt packages, `cmake -DFRANKA_CONTROL_WITH_LIBFRANKA=OFF` into `build-nolibfranka/`, internet once for
    msgpack-cxx; rebuild after every pull), then in the light env set `FRANKA_CONTROL_BIN` to
    `deployment/control_deploy/franka_control/build-nolibfranka/franka_control` (absolute) and run
    `pytest tests/integration/test_cross_franka_control.py`.

## 14. Decisions (were open questions)

Settled by the lead on 2026-11-13, with PLAN.md and TOP updated to match. Nothing is open.

1. **The ultrawide pair rule** (PLAN §7, TOP §8.8). A pair is usable only within [main − 60 ms, main + 25 ms]
   (mid-exposures) and when the same ultrawide frame was the pair of at most 2 consecutive packets
   (`pair_max_age_ms` 60, `pair_max_lead_ms` 25, `pair_max_reuse` 2; §4.3). The live Pi pairs the nearest frame
   (16.3 ms newer in PI_PAYLOAD.md §0.6), so the old 5 ms lead is gone; `pi_collect`'s pair-gap report confirms the
   lead before the laptop-alone bring-up. The reuse rule catches a frozen IMX219 whatever the phase. `PacketPairer`
   and the fake Pi pair the nearest frame, like the live Pi; the fake Pi keeps its repeat mode.
2. **Timing the policy without a plain checkpoint:** `policy.bench` on the plain `umi_policy_dunet` architecture at
   random weights for speed and VRAM; `run_001` (a prompting checkpoint) only for load time, host RAM and the VRAM of
   real weights (§11).
3. **C4 config reply v2 with `kind`** (TOP §3.12, §8.7), laptop-only, so the settled rule needs no station reading.
4. **Waiting without C3 while C2 shows `homing` at start-up**, up to `pc_homing_wait_s` (60 s), as well as the H2
   wait (§6.3 step 3; PLAN §8.2, TOP §8.2).
5. **`precision: bf16` is autocast** (weights stay fp32); a weight-cast mode only if §11 shows VRAM is short.
6. **A C3 `home` stopped by the PC's `s`** replies `ok: false` with the `stopped:` prefix (TOP §3.7.2): `HOME_STOPPED`,
   shown, phase `IDLE`, clock map re-warmed. A `home` refused with `refused: mode is homing` (a race with the PC's own keys) is a hint, like a refused
   `start`. Any other not-ok reply or no reply is `HOMING_FAILED` (conclude).
7. **`c` and `h` are refused locally while C2 shows `homing`** (the PC's `h`/`g`): a hint, `IDLE`, never a conclude
   (PLAN §7). A `start` refused by the PC is a hint too.
8. **Clocks:** `Anchor` carries both `t_ns` (laptop) and `t_pc_ns` (PC); the `start` ack time is mapped to PC time
   before it is compared with C2 `t_ns`; `STATE_LOST` is judged by arrival (liveness), `STATE_OLD` by mapped `t_ns`;
   a C2 more than 20 ms ahead of the mapped "now" counts toward `CLOCK_MISMATCH` (a map running behind); after a
   `home` the map re-warms with a hello every cycle; a hello `t_pc_ns` going back means the PC rebooted.
9. **The launcher's manager ready timeout is 130 s**, above the longest legal start-up (115 s, §6.3, §9.1).
10. **The station refuses an explicit list of contract key names** (§2.4), never generic ones.
11. **The unrequested-`stopped` code is `PC_STOPPED`** (was `ARM_STOPPED`, then `PC_WATCHDOG`), with PLAN §7's text.
12. **The cross test runs on the laptop** against the libfranka-less build of control_deploy/STRUCTURE.md §8.3, with
    `launcher.fake_pc: false` (§9.1, §12.3; TOP §8.10).
13. **The fake PC** answers laptop-only schemas `unknown_schema` through `RpcServer`'s schema filter (TOP §3.11),
    lets a C3 `stop` abort a keyboard homing, and has hooks for the PC's `h`, `g` and `s` (§10.1).
14. **Packaging:** the console script `policy_deploy`; `deploy-contracts` is installed editable alongside, never a
    pyproject dependency (TOP §1.4); the frames process's config file is `config/frames_process.yaml`, and the session
    copies the contract files into `contracts/`, so `frames.yaml` always means the TCP.
