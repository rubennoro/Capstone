# deployment/: communication, shared contracts, codebase map

Written from PLAN.md (§2, §3, §5, §6, §8, §10). PLAN.md is the source of truth: where this file and PLAN.md disagree,
PLAN.md wins and this file is fixed. `control_deploy/STRUCTURE.md` and `policy_deploy/STRUCTURE.md` are written from
this file; everything that crosses between the two machines is defined here, completely, so the two sides can be built
without talking to each other.

Source references `old:<path>:<line>` point into `rnoro/deployment` (`/home/ruben/Capstone-deployment/deployment/`).

## 1. Codebase map

### 1.1 Tree

```
deployment/
  PLAN.md                      the approved plan
  STRUCTURE.md                 this file
  contracts/                   ONLY what crosses between the laptop and the PC (PLAN §5.5)
    STRUCTURE.md               not written: this file is the contracts' design
    pyproject.toml             distribution "deploy-contracts", import package deploy_contracts (§1.4)
    limits.yaml                C2/C3 validator limits, read by Python and C++ (§4.1)
    frames.yaml                F_T_TCP, the one definition of the TCP, read by Python and C++ (§4.2)
    link.yaml                  addresses, ports, timing of the laptop <-> PC link, read by Python and C++ (§4.3)
    rigs/                      the camera constants (laptop only; the PC never reads them)
      rig_pi_v3_640.yaml       GENERATED: training's rig (~/data/umi/rigs/rig_pi_v3.yaml + its measured lens) scaled
                               to the 640x480 stream; `python -m policy_deploy.frames.deploy_rig` (policy_deploy/
                               DIRECTIONS.md "Camera constants"). Read by the frames process (frames_process.yaml)
    deploy_contracts/          the Python package (§3.1)
    examples/                  generated C2/C3 example files, read by the Python and the C++ tests (§5)
      index.yaml
      c2/*.msgpack
      c3/*.msgpack
    tests/                     the contracts' own tests (§7.3)
  policy_deploy/               the laptop (PLAN §8.2); its own STRUCTURE.md
    STRUCTURE.md  DIRECTIONS.md  station.yaml  pyproject.toml
    PI_PAYLOAD.md              C0, moved from old:contracts/PI_PAYLOAD.md (PLAN §5.1)
    config/                    each laptop process's settings; the frames process's file is frames_process.yaml,
                               so it is never confused with contracts/frames.yaml (the TCP)
    policy_deploy/
      contracts/               laptop-only contracts C0, C1, C4, built on deploy_contracts (§3.12)
      frames/                  process "frames"
      policy/                  process "policy"
      manager/                 process "manager" (C2 subscriber, the only C3 client), incl. pc_probe
      fake_pc/                 the laptop's stand-in for franka_control (--mock)
      launcher/                check | run for the three laptop processes
    tests/
  control_deploy/              the PC (PLAN §8.1); its own STRUCTURE.md
    STRUCTURE.md  DIRECTIONS.md  station.yaml  pyproject.toml
    control_deploy/            the Python wrapper: check | run | console | firewall
    franka_control/            the C++ process (copied from old:franka_control/ at 91b178b)
      CMakeLists.txt  build.sh  config.yaml  .gitignore  cmake/  include/  src/  tests/
    tests/
```

The inside of `policy_deploy/` and `control_deploy/` below the package level belongs to their own STRUCTURE.md.

### 1.2 Who owns what

| Folder | Owns | Never |
|---|---|---|
| `contracts/` | C2 and C3 message classes and validators, the wire format, schema names, the transport helpers, the three yaml files, the C2/C3 example files | anything only one machine uses (C0, C1, C4, clock maps, gate rules, config of a process); a process of its own |
| `policy_deploy/` | everything on the laptop: frames, policy, manager, fake Pi, fake PC, both clock maps, C0/C1/C4, its station and configs | reading `control_deploy/` files; starting, stopping or configuring the PC |
| `control_deploy/` | everything on the PC: franka_control, the wrapper, the console, its station and `franka_control/config.yaml` | reading `policy_deploy/` files; any clock code |

### 1.3 Dependency rules

1. `policy_deploy` and `control_deploy` never import, include or read files from each other (PLAN rule 3).
2. Both may use `deployment/contracts/`. Python code imports `deploy_contracts`; the C++ code reads only
   `limits.yaml`, `frames.yaml`, `link.yaml` and `examples/` (it never runs or imports Python).
3. `deploy_contracts` imports only the standard library, `numpy`, `msgpack`, `zmq` and `yaml`. It never imports
   `policy_deploy` or `control_deploy`. A test enforces it (§7.3).
4. No station or config file on either machine can override a key of the three contract yaml files. Each folder's
   station loader refuses those key names explicitly (the lists are in the folder STRUCTURE.md files). The one exception
   is the loopback switch (§2.3), which is a command-line flag, never a file.

### 1.4 Installing and finding `contracts/`

**Python package name: `deploy_contracts`** (distribution `deploy-contracts`, version 2.0.0). The old package was
`contracts`, installed editable from `~/Capstone-deployment` into `~/envs/deploy`. A second package named `contracts`
would shadow or be shadowed by it in any env that has both, and `contracts` is a generic name other libraries use.
`deploy_contracts` is unique among every package of both branches (`contracts`, `deploy`, `process_frames`, `policy`,
`message_comms`, `policy_deploy`, `control_deploy`) and says what it is. The folder stays `deployment/contracts/`
(PLAN §3).

`pyproject.toml` (copied from old:contracts/pyproject.toml, renamed):

| Key | Value |
|---|---|
| `requires-python` | `>=3.10,<3.11` (as before; on the PC create the env with `uv`, as on the Jetson) |
| dependencies | `numpy>=1.26,<2.3`, `msgpack==1.1.0`, `pyzmq==27.1.0`, `pyyaml==6.0.3` |
| extra `test` | `pytest==8.3.5` |
| packages | `["deploy_contracts"]`; no package data (the yaml files sit next to the package, not inside it) |

**Install:** every Python env that runs or tests C2/C3 code installs it **editable**:
`uv pip install -e deployment/contracts` (laptop: every env of `policy_deploy`, the torch env included, because C4
builds on `deploy_contracts`; PC: the wrapper's env). Never install it into the old `~/envs/deploy`. Env names are
the folder STRUCTURE.md files' choice. **One rule for both folders:** neither `policy_deploy`'s nor `control_deploy`'s
`pyproject.toml` lists `deploy-contracts` as a dependency (it is a local editable install, not an index package);
each folder's DIRECTIONS installs it first, in the same `uv pip install` command, and each folder's `check` fails when
it is missing or comes from another checkout.

**How Python finds the yaml and example files:** `deploy_contracts.paths.contracts_dir()` is
`Path(deploy_contracts.__file__).resolve().parents[1]`, i.e. `deployment/contracts/`. This only works with the editable
install (as old:contracts/contracts/examples.py:83-85 already required). If `limits.yaml` is not there, it raises
`FileNotFoundError("deploy_contracts must be installed editable from deployment/contracts: uv pip install -e ...")`.

**How the C++ build finds them** (replaces old:franka_control/CMakeLists.txt:39-41, 112-117):

```cmake
set(CONTRACTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../contracts" CACHE PATH "deployment/contracts")
# derived, baked into franka_control_core as compile definitions (absolute paths):
#   CONTRACTS_LIMITS_FILE  = ${CONTRACTS_DIR}/limits.yaml
#   CONTRACTS_FRAMES_FILE  = ${CONTRACTS_DIR}/frames.yaml
#   CONTRACTS_LINK_FILE    = ${CONTRACTS_DIR}/link.yaml
#   CONTRACTS_EXAMPLES_DIR = ${CONTRACTS_DIR}/examples
```

- Configure fails (`FATAL_ERROR`) if any of the four is missing.
- The binary reads the three yaml files **at run time** from the baked paths. Editing a value needs a restart of
  franka_control, not a rebuild. Moving the checkout needs a rebuild (the paths are absolute), as PLAN §8.1 says.
- The tests read `examples/` from `CONTRACTS_EXAMPLES_DIR` (as old:franka_control/tests/support/test_support.cpp:38).

## 2. Communication map

### 2.1 Diagram

```
                 all on the lab router ASUS_5g (Wi-Fi; laptop and PC may later be cabled to its LAN ports)

  Pi 5 (fixed sender)
    PUB bind :5555 ──────────────── C0 pi_stream/v1, 3-part messages, 30/s ───────────┐
                                                                                     ▼
  LAPTOP  policy_deploy/                                                       frames SUB connect tcp://<pi>:5555
    frames   REP bind tcp://127.0.0.1:5610  ◄── C1 ──  manager REQ
    policy   REP bind tcp://127.0.0.1:5640  ◄── C4 ──  manager REQ
    manager  SUB connect tcp://<pc.address>:5620          REQ connect tcp://<pc.address>:5630
                     ▲                                          │
                     │ C2 c2.state/v1, 100 Hz                   │ C3 c3.* requests (heartbeat every 20 ms,
                     │                                          │    hello every 0.5 s, waypoints, ...)
                     │                                          ▼
  PC  control_deploy/
    franka_control  PUB bind tcp://<pc.address>:5620     REP bind tcp://<pc.address>:5630
                    FCI (libfranka, own NIC) ──► 172.16.0.2  Franka Panda + Franka Hand
```

### 2.2 Every socket

| Contract | Socket | Binds | Connects | Pattern | Address from | Port |
|---|---|---|---|---|---|---|
| C0 `pi_stream/v1` | Pi sender | Pi (all its interfaces) | laptop frames | PUB/SUB | `policy_deploy/station.yaml` (the Pi's address) | 5555 (fixed by the Pi) |
| C0, `--mock` | fake Pi | laptop, `127.0.0.1` | laptop frames | PUB/SUB | `policy_deploy/station.yaml` | 5600 (as old station) |
| C1 | frames | laptop, `127.0.0.1` | manager | REQ/REP | `policy_deploy/station.yaml` | 5610 |
| C4 | policy | laptop, `127.0.0.1` | manager | REQ/REP | `policy_deploy/station.yaml` | 5640 |
| C2 `c2.state/v1` | franka_control state | PC, `pc.address` | manager; also `pc_probe` (laptop), `console` and `control_deploy run`'s status line (PC) | PUB/SUB | `contracts/link.yaml` | `ports.state` 5620 |
| C3 `c3.*` | franka_control commands | PC, `pc.address` | manager, **the only client** (§3.7.3) | REQ/REP | `contracts/link.yaml` | `ports.commands` 5630 |
| C2, C3, `--mock` | fake PC | laptop, `127.0.0.1` | manager | as above | host `127.0.0.1`, ports from `link.yaml` | 5620, 5630 |
| FCI | libfranka | (not ZeroMQ) | franka_control → robot | libfranka UDP/TCP | `franka_control/config.yaml` `robot_ip` | libfranka's |

Ports 5610/5620/5630/5640/5600 are those of old:station.yaml:11-18.

### 2.3 Where addresses come from

| Mode | franka_control (or the fake PC) binds | The laptop connects to |
|---|---|---|
| Real run (`control_deploy run`, `policy_deploy run`) | `tcp://<link.yaml pc.address>:<port>` | the same |
| PC alone on one machine (`control_deploy run --local`, also `--mock --local`) | `tcp://127.0.0.1:<port>` | a client on the same machine: `console` |
| Cross test (PLAN §10), on the laptop | a libfranka-less `franka_control --mock --local` (control_deploy/STRUCTURE.md §8.3): `tcp://127.0.0.1:<port>` | the laptop's processes, as under `--mock`, with the fake PC not started (policy_deploy/STRUCTURE.md §12.3) |
| Laptop alone (`policy_deploy run --mock`) | fake PC on the laptop: `tcp://127.0.0.1:<port>` | `tcp://127.0.0.1:<port>` |
| Unit tests | `tcp://127.0.0.1:*` (a free port; the helper reports the real endpoint) | that endpoint, passed in directly |

Ports always come from `link.yaml` (except unit tests). `pc.address` is an IPv4 literal of the PC's ASUS_5g interface
(Wi-Fi, or later its wired port), never `0.0.0.0` or `*` (PLAN §6, "Who can command the arm"). Going wired means
changing `pc.address` and adding the laptop's wired address to `laptop.addresses`; with addresses reserved on the
router, nothing (PLAN §6).

### 2.4 Socket rules

- Servers and publishers bind; clients and subscribers connect (PLAN §5). The PC binds both C2 and C3; the laptop only
  connects, so the start order does not matter at the socket level (the operator still starts the PC first, PLAN
  §8.1).
- **C2 may have any number of subscribers** (they are read-only). **C3 has one client at a time**: the laptop's
  manager in a run; `console` (PC) or `pc_probe` (laptop) only when no manager runs. ZeroMQ REP cannot tell clients
  apart, so this is enforced by the start-up rule of §3.7.3, not by the socket.
- Every socket uses linger 0. High-water marks come from each side's own config (old values: C2 send HWM 100 on the
  PC, old:franka_control/config.yaml:77).
- C2/C3 have no authentication. The PC's firewall lets only `link.yaml laptop.addresses` and the PC itself (loopback:
  the status line and `console` connect to `pc.address` from the PC, which Linux delivers over `lo`) reach
  `ports.state` and `ports.commands` (printed by `control_deploy firewall`, checked by `control_deploy check`).
  Encryption is later (PLAN §14).

## 3. The shared contracts: `deploy_contracts`

### 3.1 The package

One concern per file. Copied from old:contracts/contracts/ where marked; the C0, C1, C4 files, `sizes.py`, `_rot6d.py`
and the C0 multipart helpers move to `policy_deploy/policy_deploy/contracts/` (§3.12).

```
deploy_contracts/
  __init__.py        re-exports the public names below
  errors.py          ContractError, TransportTimeout, TransportCancelled, RemoteError, PROBLEMS     (copied)
  schema.py          SchemaName                                                                     (copied)
  fields.py          scalar(), array(), nested(), list_of(), str_list(), optional()                 (copied)
  message.py         Message, Record                                                                (copied)
  registry.py        register, lookup, registered                                                   (copied)
  codec.py           encode, decode, pack_obj, unpack_obj, pack_array, unpack_array                 (copied)
  paths.py           contracts_dir(), limits_file(), frames_file(), link_file(), examples_dir()     (new)
  limits.py          ContractLimits: limits.yaml (3 keys now)                                       (adapted)
  frames.py          TcpFrame: frames.yaml                                                          (new)
  link.py            LinkSettings: link.yaml                                                        (new)
  common.py          Ack, ErrorReply                                                                (copied)
  c2_state.py        ArmState, GateCounters, MODES, rigid_problem()                                 (copied)
  c3_commands.py     SetWaypoints, SetWaypointsReply, WaypointIssue, Heartbeat, Start, Stop, Home, GetState,
                     Hello, HelloReply                                                              (adapted: + hello)
  shared.py          SHARED_SCHEMAS: the fixed, sorted tuple of the 12 shared schema strings (§3.4)  (new)
  hello_check.py     HelloMismatch, compare_hello()                                                 (new, §3.9)
  rpc.py             RpcClient, RpcServer                                          (copied; + schemas filter, §3.11)
  pubsub.py          Publisher, Subscriber                                                          (copied, minus C0)
  examples.py        Example, build_examples(), load_index(), the builder helpers                  (adapted)
  make_examples.py   python -m deploy_contracts.make_examples                                       (adapted)
```

### 3.2 Base classes and field helpers

Unchanged from old:contracts/STRUCTURE.md §3 (old:contracts/contracts/fields.py, message.py). In short:

- Each message is a frozen dataclass subclassing `Message` (`SCHEMA: ClassVar[str]`); sub-messages subclass `Record`
  (no `schema` key). The class body is the field table; the validator reads it.
- Helpers: `scalar(kind, choices=None, min=None, max=None)` with `kind` in `int64, uint64, uint32, int, float64,
  bool, str, bytes`; `array(dtype, shape)` with dtype in `uint8, int64, uint64, float32, float64` and `None` = any
  length; `nested(cls)`; `list_of(cls)`; `str_list(choices=None)`; `optional(spec)`. All kinds stay, because the
  laptop's C0/C1/C4 use them.
- Building validates (`__post_init__` → `validate()` → each field in table order, then `check_values()`). **A message
  object that exists is valid.** `to_dict()` (schema first, table order), `from_dict()` (schema, missing keys,
  unknown keys, then build).
- Every field is always present; `optional` is sent as `nil`. Arrays are never cast (`float32` where `float64` is
  required is an error). Value checks on float arrays run in float64. `bool` is not an int; an int is not a
  `float64`. `min`/`max` of `scalar` are inclusive.

### 3.3 Wire format

Unchanged from old:contracts/STRUCTURE.md §5 (old:contracts/contracts/codec.py). This is what C++ reads and writes:

- One ZeroMQ message = one msgpack **map** = one message. Keys are strings in field-table order, `schema` first.
- Integers in the smallest msgpack form; `float64` fields always as msgpack float64; `bool` as bool; `str` as UTF-8
  str (the PC sends printable ASCII only, old:franka_control/src/command_server.cpp:11-19); `bytes` as bin; `None` as
  nil; lists as arrays.
- An array is the map `{"__nd__": true, "dtype": "<name>", "shape": [d0, ...], "data": <bin>}`, keys in this order,
  `data` C order (row-major), little-endian, exactly `prod(shape) × itemsize` bytes. 4×4 poses are **row-major**
  (libfranka and Eigen are column-major; a transposed pose fails the rigid check).
- Python: `msgpack.packb(d, use_bin_type=True)`, `msgpack.unpackb(b, raw=False, strict_map_key=True)`. Trailing bytes
  are `not_msgpack`.

Decoding checks run in this fixed order; the first failure raises, so every broken example has one expected error:
(1) msgpack (`not_msgpack`), a map (`not_a_map`); (2) `schema` present (`no_schema`), known name (`unknown_schema`),
supported version (`schema_version`); (3) missing keys (`missing`), unknown keys (`unknown_field`); (4) each field in
table order: `type`, `dtype`, `shape`, `value`; (5) `check_values()` (`value`); (6) with `decode(expect=...)`, a valid
message of another type (`unexpected_message`; `ErrorReply` always passes).

Problem codes, exactly: `not_msgpack`, `not_a_map`, `no_schema`, `unknown_schema`, `schema_version`, `missing`,
`unknown_field`, `type`, `dtype`, `shape`, `value`, `unexpected_message`.

### 3.4 Schemas

`SHARED_SCHEMAS` (in `shared.py`), sorted by byte order, exactly:

```
ack/v1  c2.state/v1  c3.get_state/v1  c3.heartbeat/v1  c3.hello.reply/v1  c3.hello/v1  c3.home/v1
c3.set_waypoints.reply/v1  c3.set_waypoints/v1  c3.start/v1  c3.stop/v1  error/v1
```

- Importing `deploy_contracts` alone registers exactly these. `policy_deploy.contracts` registers its own (C1, C4,
  ...) into the same registry when imported; `SHARED_SCHEMAS` is a fixed tuple and does not grow with it.
- franka_control's codec knows **exactly** `SHARED_SCHEMAS` (old:franka_control/src/codec.cpp:25-35 listed C0–C4 as
  well; those entries are removed). A laptop-only message sent to the PC is then `unknown_schema`, not
  `unexpected_message` (old:franka_control/tests/test_codec.cpp `MessagesOfOtherContractsAreUnexpected` changes to
  expect `unknown_schema`).
- `c2.state/v1` and every existing C3 message stay at v1: none of their fields, types, units, frames or clocks change
  (PLAN §5.3, §5.4). `c3.hello/v1` and `c3.hello.reply/v1` are new.

### 3.5 Common replies (`common.py`)

`Ack`, `ack/v1`: the reply to requests whose answer is "ok" or "refused".

| Field | Type | Meaning |
|---|---|---|
| `ok` | bool | true: done; false: refused |
| `reason` | str | `""` when ok; else one line, e.g. `refused: mode is running: send stop first` |

`ErrorReply`, `error/v1`: sent instead of the normal reply when the server cannot answer it: the request broke its
contract, is unknown, is not a C3 request, is refused because of the mode (`set_waypoints` only), or the handler
failed.

| Field | Type | Meaning |
|---|---|---|
| `request` | str | the request's schema, `""` if unreadable |
| `error` | str | one line, e.g. `c3.set_waypoints/v1: field 'O_T_TCP': not a rigid transform at [3]: ...` |
| `field` | str | the field at fault for a contract error, else `""` |

### 3.6 C2: PC → laptop, `c2.state/v1` (`c2_state.py`)

Copied from old:contracts/contracts/c2_state.py. Broadcast on the PC's PUB at 100 Hz; also the reply to `get_state`.

`GateCounters` (Record), each uint64, waypoints **since franka_control started** (so across laptop runs; the laptop
shows differences):

| Field | Meaning |
|---|---|
| `accepted` | accepted by the gate, shortened and pulled-back ones included |
| `shortened` | accepted, shortened to the speed limit |
| `pulled_back` | accepted, pulled back onto the box or cone (outside by at most the margin) |
| `rejected` | refused by the gate (franka_control went to `fault`) |

`ArmState`:

| Field | Type / shape | Unit | Frame / clock | Validation |
|---|---|---|---|---|
| `t_ns` | int64 | ns | **PC `CLOCK_MONOTONIC`**: start of the 1 kHz cycle the reading came from | int64 |
| `seq` | uint64 | | | uint64 |
| `O_T_TCP` | float64 (4, 4) | m | measured TCP (frames.yaml) in base frame O, row-major | finite; rigid |
| `O_T_TCP_target` | float64 (4, 4) | m | the target the 1 kHz loop tracks now, same frame | finite; rigid |
| `q` | float64 (7,) | rad | | finite |
| `dq` | float64 (7,) | rad/s | | finite |
| `gripper_width_m` | float64 | m | opening between the fingertips, absolute | finite, ≥ 0 |
| `mode` | str | | | one of `idle`, `running`, `holding`, `stopped`, `homing`, `fault` |
| `error` | str | | | any; `""` unless `fault` (the PC's convention, not checked) |
| `gate` | `GateCounters` | | | `shortened ≤ accepted`, `pulled_back ≤ accepted` |
| `loop_late` | uint64 | | late 1 kHz cycles since start | uint64 |

**Rigid** (`rigid_problem`, old:contracts/contracts/c2_state.py:16-32): finite; bottom row exactly `[0, 0, 0, 1]`;
`max |RᵀR − I| ≤ rotation_tol` (limits.yaml); `det R > 0`. Computed in float64.

**PC obligations on C2** (behaviour the laptop relies on; from old:franka_control/STRUCTURE.md §8.1):

1. `t_ns` never decreases within one franka_control run. It **repeats unchanged** after a libfranka fault (the loop
   has ended), for a few ms between a torque session and a homing move, and for up to `control.settle_timeout_s`
   (1 s) after a stopped arm homing, while the control thread waits for the robot to leave its move mode; it advances
   during the homing move and during the hand's homing (PLAN §5.3).
2. `seq` is +1 per broadcast, the first broadcast is 1 (old:franka_control/src/state_publisher.cpp:27-29). A message
   dropped by a full queue still uses its number (subscribers see the gap). A restart of franka_control starts again
   at 1, so `seq` goes backwards.
3. A **`get_state` reply carries the seq of the latest broadcast (not +1)** and the newest snapshot, so it can repeat
   a broadcast's `seq` with a newer `t_ns`. The laptop uses `get_state` replies only for mode checks, never as C2
   samples (history, `seq` gaps, clock map).
4. C2 never pauses: a message every 10 ms in every mode, including `fault` and `homing`.
5. `gripper_width_m` is read by its own thread at 50 Hz, so it can be up to 20 ms older than `t_ns`.
6. A reading that would break the contract (non-finite) is not sent; franka_control goes to `fault` and C2 carries the
   last valid values with the error.

### 3.7 C3: laptop → PC, commands (`c3_commands.py`)

#### 3.7.1 Requests and replies

| Request | Schema | Fields | Reply |
|---|---|---|---|
| hello | `c3.hello/v1` | none | `c3.hello.reply/v1` (§3.8) |
| get_state | `c3.get_state/v1` | none | `c2.state/v1` |
| heartbeat | `c3.heartbeat/v1` | none | `ack/v1` ok |
| start | `c3.start/v1` | none | `ack/v1` ok, or refused with the reason |
| stop | `c3.stop/v1` | none | `ack/v1` ok; never refused |
| home | `c3.home/v1` | none | `ack/v1`, sent when the move ends: ok, or not ok on refusal, failure, timeout, or a stop (reason starts with `stopped:`, §3.7.2) |
| set_waypoints | `c3.set_waypoints/v1` | below | `c3.set_waypoints.reply/v1`; `error/v1` when refused by the mode (not a fault) |
| anything broken, unknown, or not a C3 request | | | `error/v1` naming the field and problem |

`SetWaypoints` (copied from old:contracts/contracts/c3_commands.py:20-50). Waypoint i is `(t_ns[i], O_T_TCP[i],
gripper_width_m[i])`.

| Field | Type / shape | Unit | Frame / clock | Validation |
|---|---|---|---|---|
| `chunk_id` | uint64 | | which policy step produced these; echoed and logged | uint64 |
| `t_ns` | int64 (N,) | ns | **PC `CLOCK_MONOTONIC`**: when each waypoint is to be reached | `1 ≤ N ≤ max_waypoints`; strictly increasing |
| `O_T_TCP` | float64 (N, 4, 4) | m | absolute TCP pose in base frame O, row-major | N rows; finite; each rigid |
| `gripper_width_m` | float64 (N,) | m | absolute fingertip opening | N values; finite; ≥ 0 |

Whether a waypoint is safe is not a contract question; it is the gate's (PLAN §5.4): time passed → skipped; more than
`max_lead_s` ahead, width outside `[0, gripper_width_max_m]`, outside the box or cone beyond the margin, non-finite →
rejected, `fault`. Accepted waypoints replace every waypoint not yet reached.

`WaypointIssue` (Record): `index` uint64 (position in the request), `reason` str.

`SetWaypointsReply`, `c3.set_waypoints.reply/v1`:

| Field | Type | Meaning | Validation |
|---|---|---|---|
| `chunk_id` | uint64 | echo of the request | |
| `accepted` | uint64 | accepted, shortened and pulled-back ones included | |
| `shortened` | uint64 | accepted ones shortened to the speed limit | `≤ accepted` |
| `skipped` | uint64 | their time had already passed | |
| `rejected` | list of `WaypointIssue` | refused by the gate, each with its reason | indices unique |

PC obligation (the validator cannot check it, it doesn't know N): `accepted + skipped + len(rejected) == N`. With any
rejection the PC accepts nothing: `accepted 0`, and every non-skipped waypoint is listed (the failing one with its
reason, the others `not checked: another waypoint of this request was rejected`). The laptop checks the sum against the
N it sent and concludes if it differs.

#### 3.7.2 Replies per mode

| Request | `idle` | `holding` | `running` | `stopped` | `homing` (from the PC's `h`/`g`, or a C3 `home`) | `fault` |
|---|---|---|---|---|---|---|
| `hello` | reply | reply | reply | reply | reply | reply |
| `get_state` | reply | reply | reply | reply | reply | reply |
| `heartbeat` | ok | ok | ok | ok | ok | ok |
| `start` | ok → `holding` | refused | refused | ok → `holding` | refused | refused |
| `stop` | ok → `stopped` | ok → `stopped` | ok → `stopped` | ok | ok; aborts the move → `stopped`; if the move finished first → `idle` | ok, stays `fault` |
| `home` | → `homing`, reply at the end | refused | refused | → `homing`, reply at the end | refused | refused |
| `set_waypoints` | `error/v1` | gate | gate | `error/v1` | `error/v1` | `error/v1` |

From old:franka_control/STRUCTURE.md §4.3, plus the new `hello` row and the `homing` column. The column exists now
because the PC keyboard can home while C3 is free (PLAN §8.1); when homing was started by a C3 `home`, the REP socket
answers nothing until that `home` reply is sent (old:franka_control/src/command_server.cpp:42-50), and the client's REQ
socket cannot send meanwhile anyway, so the cells of that column are reached over C3 only during a keyboard homing.
Refusal reasons are one line naming the mode (e.g. `refused: mode is homing`). A successful `home` ends in `idle`;
its timeout is `homing_timeout_s`.

**A stopped homing (contract).** The PC's `s`, or a C3 `stop` during a keyboard homing, aborts the move: the arm (or
the fingers) holds where it stopped and the mode becomes `stopped`. If the move finished first, the normal end wins
(`idle`). A C3 `home` aborted this way (only the PC's `s` can do it, since the REP socket is busy) replies
`ack {ok: false, reason}` where `reason` starts with the fixed prefix **`stopped:`** (exactly these 8 characters, then
one space and the text), and C2 shows `stopped` from then on. No other `home` reply starts with `stopped:`: a refusal
starts with `refused:`, a failure or timeout with any other text, and the PC is then in `fault` or unchanged. The
laptop tells the cases apart by this prefix alone.

#### 3.7.3 Watchdog, hello and the one-client rule

- The watchdog limit is `link.yaml timing.watchdog_limit_s` (1.0 s), read by franka_control (PLAN §6).
- It is fed by **every C3 request that decodes and is a C3 request**, `hello` included (as now,
  old:franka_control/src/command_server.cpp:77-79). A broken request does not feed it.
- Armed in `holding` and `running` only; paused during `homing` and restarted when homing ends. A trip goes to
  `stopped` (not `fault`) and holds (PLAN §5.4).
- **Rule H1: `hello` is sent only by the manager's main loop, on the same REQ socket as the heartbeats.** Each cycle
  sends exactly one keep-alive (a hello or a heartbeat; a hello replaces the heartbeat of the cycle it falls in), and a
  `set_waypoints` in addition when a step is done. A hello from another thread or socket would keep feeding the
  watchdog while the main loop hangs.
- **Rule H2: no C3 request before C2 shows a mode other than `running`/`holding`.** A client (manager, `pc_probe`,
  `console`) first reads C2. If the mode is `running` or `holding`, it sends nothing for up to
  `2 × watchdog_limit_s`: a dead previous client's episode then trips the watchdog to `stopped`. If C2 still shows
  `running`/`holding` after that, another client is driving: refuse, "the PC is driving an episode for another C3
  client" (PLAN §7, last row). Without this rule, a new laptop's hellos (every 0.5 s, below the 1.0 s watchdog) would
  keep an orphaned episode `holding` forever.
- `pc_probe` and `console` are C3 clients: never run alongside a manager. H2 makes them refuse when an episode is
  running.

### 3.8 `hello` (new)

`Hello`, `c3.hello/v1`: no fields. Read-only: it changes no state on the PC (it feeds the watchdog, §3.7.3).

`HelloReply`, `c3.hello.reply/v1`. Fields in wire order:

| Field | Type / shape | Unit | Source on the PC | Validation |
|---|---|---|---|---|
| `t_pc_ns` | int64 | ns | **PC `CLOCK_MONOTONIC`** (the clock of C2 `t_ns`), read while answering (below) | int64 |
| `commit` | str | | the git commit the binary was built from: 40 lowercase hex, or `unknown` | matches `^([0-9a-f]{40}\|unknown)$` |
| `mock` | bool | | `--mock` (fake arm and gripper) | bool |
| `schemas` | list of str | | every schema the PC's codec knows, sorted by byte order | each a well-formed `name/vN`; strictly increasing |
| `F_T_TCP` | float64 (4, 4) | m | frames.yaml as franka_control uses it (after its rotation re-normalisation) | finite; rigid |
| `rotation_tol` | float64 | | limits.yaml | finite, > 0 |
| `gripper_width_max_m` | float64 | m | limits.yaml | finite, > 0 |
| `max_waypoints` | uint64 | | limits.yaml | ≥ 1 |
| `watchdog_limit_s` | float64 | s | link.yaml `timing.watchdog_limit_s` | finite, > 0 |
| `homing_timeout_s` | float64 | s | `franka_control/config.yaml` `homing.timeout_s` | finite, > 0 |
| `max_lead_s` | float64 | s | `franka_control/config.yaml` `safety.max_lead_s` | finite, > 0 |

`check_values()` runs its checks in table order: `commit`, `schemas` (a malformed item is field `schemas[i]`; order or
duplicates is field `schemas`), `F_T_TCP` (rigid, with the **receiver's** `rotation_tol`), then the five floats.
`max_waypoints` uses `scalar("uint64", min=1)`.

Fields beyond PLAN §5.4's list, and why:

- `schemas`: a version mismatch would otherwise show only when that message is first exchanged (a `set_waypoints` at
  the first step, mid-episode). With it the laptop refuses at start-up, naming the differing schemas.
- `rotation_tol`: PLAN §5.4 asks that "the limits match"; franka_control reads three limits keys
  (old:franka_control/src/contract_limits.cpp:43-52), so all three are reported. A different `rotation_tol` would turn
  waypoints the laptop considers rigid into `error/v1` replies.

PC obligations for `hello`:

1. Answered in every mode (§3.7.2), within the normal C3 service time.
2. `t_pc_ns` is read **after the request is received and decoded, immediately before encoding the reply**, with nothing
   in between (no publishing, no waiting). The round-trip estimator assumes the stamp sits at the midpoint of the
   round trip; time spent between receive and stamp, or stamp and send, is error.
3. `commit` is the commit of the source the running binary was built from. A configure-time capture
   (old:franka_control/CMakeLists.txt:43-47) goes stale when only `cmake --build` is rerun; the build must refresh it
   on every build (how: `control_deploy/STRUCTURE.md`). It is the full 40-character hash, not `--short`.
4. `schemas` is the codec's known list (= `SHARED_SCHEMAS` when both sides are on one commit).

### 3.9 Checking a hello reply (`hello_check.py`)

The equality part of the start-up check lives here, so the laptop (manager, `check`, `pc_probe`) and the PC's
`console` use one definition:

```python
@dataclass(frozen=True)
class HelloMismatch:
    field: str        # "schemas", "mock", "F_T_TCP", "rotation_tol", "gripper_width_max_m", "max_waypoints",
                      # "watchdog_limit_s"
    pc: str           # the PC's value, printable
    expected: str     # this side's value and its source, e.g. "1.0 (link.yaml timing.watchdog_limit_s)"

def compare_hello(reply: HelloReply, *, mock: bool, limits: ContractLimits | None = None,
                  frame: TcpFrame | None = None, link: LinkSettings | None = None) -> list[HelloMismatch]
    # None: load the file. Returns every mismatch (empty list = all equal), in table order.
```

| Field | Compared with | Rule |
|---|---|---|
| `schemas` | `SHARED_SCHEMAS` | equal; `pc`/`expected` name only the differing entries |
| `mock` | the caller's own mock flag | equal (PLAN §5.4) |
| `F_T_TCP` | `frames.yaml` | every element within 1e-9 absolute |
| `rotation_tol`, `gripper_width_max_m` | `limits.yaml` | `abs(a − b) ≤ 1e-9 × max(abs(a), abs(b))` |
| `max_waypoints` | `limits.yaml` | equal |
| `watchdog_limit_s` | `link.yaml timing.watchdog_limit_s` | relative 1e-9 as above |

The tolerances exist only because franka_control re-normalises `F_T_TCP`'s rotation; both sides parse the same decimal
text, so scalars are normally bit-equal.

The rest of the start-up check is the laptop's (policy_deploy/STRUCTURE.md), listed here so nothing falls between the
documents: `commit` against its own `git rev-parse HEAD` (shown by `policy_deploy check`, PLAN §5.5; a difference is
a warning, §8.1); its worst gap between C3 requests (PLAN §6: 0.57 s) `< watchdog_limit_s`; its `home` request timeout
`> homing_timeout_s`; its longest planned waypoint lead `< max_lead_s`; `t_pc_ns` against its clock map within 5 ms
(PLAN §5.6).

### 3.10 Errors (`errors.py`)

Copied from old:contracts/contracts/errors.py.

```python
class ContractError(ValueError):          # schema, field (dotted path, "" if none), problem (§3.3), detail
    # str(e) == "c2.state/v1: field 'O_T_TCP': shape (3, 4), expected (4, 4)"
class TransportTimeout(TimeoutError):     # process, endpoint, request, timeout_s
class TransportCancelled(RuntimeError):   # process, endpoint, request, waited_s
class RemoteError(RuntimeError):          # process, reply: ErrorReply
```

The helpers never retry or hide one. The laptop concludes on `ContractError`, `RemoteError` and `TransportTimeout`
from C2/C3 (PLAN §7). `TransportCancelled` only happens when the caller asked (e.g. Ctrl-C while waiting on `home`).
The C++ side has its own `ContractError` with the same `schema`/`field`/`problem` and the same text
(old:franka_control/src/contract_error.cpp).

### 3.11 Transport helpers (`rpc.py`, `pubsub.py`)

Copied from old:contracts/contracts/rpc.py and pubsub.py (signatures in old:contracts/STRUCTURE.md §6);
`PacketPublisher`/`PacketSubscriber` (C0 multipart) move to `policy_deploy`.

```python
class RpcClient:
    def __init__(self, name: str, endpoint: str) -> None
    def request(self, msg, expect, timeout_s, poll_s=None, cancel=None)   # sent once, never resent
    def close(self) -> None
class RpcServer:
    def __init__(self, endpoint: str, handlers: Mapping[type[Message], Callable[[Message], Message]],
                 schemas: Collection[str] | None = None) -> None                # new: the schema filter below
    endpoint: str
    def serve_once(self, timeout_s: float) -> bool
    def serve_forever(self, stop: threading.Event, poll_s: float) -> None
    def close(self) -> None
class Publisher:
    def __init__(self, endpoint: str, send_hwm: int) -> None
    def publish(self, msg: Message) -> None                                # never blocks; drops when full
class Subscriber:
    def __init__(self, name: str, endpoint: str, expect: type[M], recv_hwm: int) -> None
    def receive(self, timeout_s: float) -> M | None                        # None on timeout; every message, no conflation
```

Rules (unchanged): every timeout is the caller's (from `link.yaml` or its own config); on a timeout or cancel the
client closes its socket with linger 0 and opens a new one (lazy pirate), so a late reply is dropped; one socket per
thread; `tcp://127.0.0.1:*` binds a free port.

**Schema filter (new).** `RpcServer(..., schemas=S)` answers a request whose schema is not in `S` with `error/v1`,
problem `unknown_schema`, even when that schema is registered in this process: exactly what franka_control's codec does
for a laptop-only message (§3.4). It sits in the decode order of §3.3 at step 2, after `no_schema` and before the
registry lookup, so every other problem code is unchanged. Without `schemas` (the default) nothing changes. The fake
PC passes `schemas=SHARED_SCHEMAS`, because its process also imports `policy_deploy.contracts`, which registers C1/C4
in the same registry; without the filter it would answer `unexpected_message`, unlike the PC. Only requests that pass
decoding reach a handler.

| Helper | Laptop (`policy_deploy`) | PC (`control_deploy`) |
|---|---|---|
| `RpcClient` | manager → C3 (and → C1, C4 locally); `pc_probe` → C3 | `console` → C3 |
| `RpcServer` | fake PC (C3, with `schemas=SHARED_SCHEMAS`); frames (C1); policy (C4) | not used: franka_control has its own C++ `CommandServer` |
| `Publisher` | fake PC (C2) | not used: C++ `StatePublisher` |
| `Subscriber` | manager (C2); `pc_probe` (C2) | `console` (C2); `control_deploy run`'s status line (C2) |

### 3.12 Laptop-only contracts (summary)

Defined and tested in `policy_deploy/policy_deploy/contracts/` (full tables in `policy_deploy/STRUCTURE.md`), built on
`deploy_contracts`' `Message`, field helpers, codec, registry, errors, `Ack`/`ErrorReply`, `RpcClient`/`RpcServer` and
the example helpers.

| Contract | Schemas | Clock | Notes |
|---|---|---|---|
| C0 Pi → frames | `pi_stream/v1` (JSON header + 2 JPEGs; PI_PAYLOAD.md §0, unchanged) | Pi | multipart helpers move with it |
| C1 frames ↔ manager | `c1.get_frames/v1`, `c1.frames/v1`, `c1.get_health/v1`, `c1.health/v1` | **laptop** `CLOCK_MONOTONIC` (was "PC" in old:contracts) | |
| C4 manager ↔ policy | `c4.config/v1`, `c4.config.reply/v2` (adds the policy's `kind`, PLAN §5.2), `c4.reset/v1`, `c4.infer/v1`, `c4.chunk/v1`, `c4.prompt/v1` | no times | `ActionChunk` widths use the shared `gripper_width_max_m`; `anchor_row_tol`, `rot6d_norm_tol`, `rot6d_dot_max` and `sizes.py` move to policy_deploy |

Their examples are built in memory with the shared helpers (§5) and never stored in `deployment/contracts/examples/`.

## 4. The yaml files

All three sit in `deployment/contracts/`. Units are in the key names.

### 4.1 `limits.yaml`

| Key | Type | Value | Used by |
|---|---|---|---|
| `rotation_tol` | float > 0 | `1.0e-6` | C2/C3/hello rigid checks (Python and C++); frames.yaml check |
| `gripper_width_max_m` | float > 0 | `0.08` | the PC's gate (width range); the laptop policy's clipping and C4 `ActionChunk` check. Comment kept: `TODO: measure the max fingertip opening with the printed fingers` |
| `max_waypoints` | int ≥ 1 | `64` | C3 `SetWaypoints` (Python and C++); the C++ refuses a value above its compiled plan capacity |

The three C4-only keys of old:contracts/contracts/limits.yaml (`anchor_row_tol`, `rot6d_norm_tol`, `rot6d_dot_max`)
move to policy_deploy (PLAN §5.5: only what crosses).

### 4.2 `frames.yaml`

| Key | Type | Value | Used by |
|---|---|---|---|
| `F_T_TCP` | 4×4 list of floats, row-major, metres | below | franka_control: `O_T_TCP = O_T_F × F_T_TCP` for C2 and the gate; laptop: `compare_hello` |
| `settled` | bool | `false` | laptop only (the C++ ignores it): while false, `policy_deploy check`/`run` refuse a trained policy on the real arm; hold still is allowed. Set to `true` when PLAN §9 is done. |

```yaml
# The one definition of the TCP (PLAN §9). The TCP on the wire (C2, C3) is the training TCP.
# Training's TCP axes (PLAN §9): x = finger closing (image right), y = image down, z = toward the fingertips.
# The camera is on the Franka Hand's +x face (away from the base at home), so this is the Franka Hand TCP turned
# +90 deg about z: +45 deg about flange z, fingertips 150 mm out. Check on the arm before settled: true.
F_T_TCP:
  - [0.7071067811865476, -0.7071067811865476, 0.0, 0.0]
  - [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.150]
  - [0.0, 0.0, 0.0, 1.0]
settled: false     # true once PLAN §9 is done and checked on the arm
```

Value copied from old:franka_control/config.yaml:37-42. The `tcp:` block leaves `franka_control/config.yaml`, so a PC
station can no longer override it (old:franka_control/src/config_loader.cpp:401-409 read it from there). Validation on
both sides: 4 rows of 4 finite numbers, rigid within `rotation_tol`.

### 4.3 `link.yaml`

```yaml
# The laptop <-> PC link (PLAN §6). One set of timing values, safe on Wi-Fi, used wired too.
pc:
  address: 192.168.50.10          # PLACEHOLDER: the PC's reserved ASUS_5g address (PLAN §15). franka_control binds it.
laptop:
  addresses: [192.168.50.20]      # PLACEHOLDER: the laptop's Wi-Fi address; add its wired one when cabled
ports:
  state: 5620                     # C2
  commands: 5630                  # C3
timing:
  heartbeat_period_s: 0.02        # the manager's loop: one keep-alive (hello or heartbeat) per cycle
  watchdog_limit_s: 1.0           # PC: no C3 request for this long while holding/running -> stopped
  c3_timeout_s: 0.3               # laptop: no reply within this -> conclude (not for home)
  state_stale_s: 0.1              # laptop: newest C2 older than this (mapped t_ns) -> skip the step
  state_lost_s: 2.0               # laptop: no C2 for this long (by arrival) -> conclude
clock:
  estimator: round_trip           # laptop's PC clock map: round_trip | one_way (PLAN §5.6)
```

| Key | Type | Validation | Read by |
|---|---|---|---|
| `pc.address` | str | IPv4 dotted literal, not `0.0.0.0` | franka_control (bind); manager, `pc_probe` (connect); `console`; `control_deploy check` (present on a PC interface) |
| `laptop.addresses` | list of str, ≥ 1 | IPv4 literals, unique | `control_deploy firewall` and `check` (the firewall rule: these addresses plus loopback, §2.4); `policy_deploy check` (one of them present on the laptop) |
| `ports.state`, `ports.commands` | int | 1024–65535, different | franka_control; manager; fake PC; `pc_probe`; `console` |
| `timing.heartbeat_period_s` | float | > 0, `< watchdog_limit_s` | manager; `console` |
| `timing.watchdog_limit_s` | float | > 0 | franka_control (its watchdog; it keeps its own floor of 0.5 s, old:franka_control/src/config_loader.cpp:17); laptop (`compare_hello`, budget rule) |
| `timing.c3_timeout_s` | float | > 0, `< watchdog_limit_s` | manager; `pc_probe`; `console` |
| `timing.state_stale_s` | float | > 0, `< state_lost_s` | manager |
| `timing.state_lost_s` | float | > 0 | manager |
| `clock.estimator` | str | `round_trip` or `one_way` | manager |

The estimator's internals (hello period 0.5 s, windows, minimum samples, the 5 ms start-up agreement, PLAN §5.6) and
the laptop's local timeouts (health, jitter allowance, `home`) are laptop settings in `policy_deploy`'s config.
`watchdog.backstop_extra_s` stays in `franka_control/config.yaml`; its `watchdog.limit_s` key is removed (it now comes
from here).

**Not in link.yaml:** `max_lead_s` (1.5 s, PLAN §5.4) and `homing_timeout_s` (30 s) stay in
`franka_control/config.yaml` (`safety.max_lead_s`, `homing.timeout_s`): they are gate and homing rules of the PC,
like every other gate setting. The laptop learns them from `hello` and checks against them (§3.9).

### 4.4 Loaders

| Side | File | Class / code | Strictness |
|---|---|---|---|
| Python | limits.yaml | `ContractLimits.load(path=None)` (fields `rotation_tol`, `gripper_width_max_m`, `max_waypoints`); read at import and cached, so a broken file fails every test (old:contracts/contracts/limits.py:64) | missing or unknown key, wrong type, bad value → `ValueError` naming file and key |
| Python | frames.yaml | `TcpFrame.load(path=None)` → `F_T_TCP` float64 (4, 4), read-only, and `settled` (bool); cached | same |
| Python | link.yaml | `LinkSettings.load(path=None)` (fields as the keys, flattened: `pc_address`, `laptop_addresses`, `state_port`, `commands_port`, `heartbeat_period_s`, `watchdog_limit_s`, `c3_timeout_s`, `state_stale_s`, `state_lost_s`, `clock_estimator`); `state_endpoint(local=False)`, `commands_endpoint(local=False)` → `tcp://<pc.address or 127.0.0.1>:<port>`; cached | same, plus the cross-key rules above |
| C++ | limits.yaml | `ContractLimits::load` (unchanged, old:franka_control/src/contract_limits.cpp) | reads its 3 keys; missing, non-numeric, ≤ 0 → `ConfigError` (FATAL at start, listed by `--check-config`) |
| C++ | frames.yaml | `F_T_TCP` (rigid check as old:franka_control/src/config_loader.cpp:402-408) | reads its key; same errors |
| C++ | link.yaml | `pc.address`, `ports.state`, `ports.commands`, `timing.watchdog_limit_s` | reads its 4 keys, with the same validation as Python for them; ignores the others |

Python is strict (unknown keys refused) and runs on both machines, so a structural change is caught on whichever side
was not updated; C++ reads only the keys it uses (as now for limits.yaml).

### 4.5 Contract change or tunable

| Change | Kind | What to do |
|---|---|---|
| A field of a C2/C3 message added, removed, renamed; its type, dtype, shape, unit, frame definition, clock or meaning | **contract change** | bump that schema (§7.1), regenerate examples, update the C++ codec until its tests pass |
| A key added, removed or renamed in any of the three yaml files | **contract change** (no schema) | update the Python loader and the C++ reader in the same commit |
| A value in `limits.yaml` or `frames.yaml` (e.g. the PLAN §9 fix of `F_T_TCP`) | **shared data** | same commit on both machines; rerun `make_examples` (some broken examples depend on `max_waypoints`); `hello` compares the PC's values with the laptop's at every start. No schema bump: the definition ("the training TCP", "the Hand's opening") does not change, only its measured value |
| A value in `link.yaml` | **tunable** | edit, restart both sides; `hello` checks `watchdog_limit_s`, the laptop's budget rule checks the rest |

This replaces old:contracts/STRUCTURE.md §7's "a limits.yaml change is a version bump": with `hello` comparing the
values directly, a bump would add work without catching anything more.

## 5. Examples

### 5.1 How they are made and stored

- `deploy_contracts.examples.build_examples() -> list[Example]` builds every shared example in code: good ones from the
  message classes, broken ones by changing one thing in a good dict and packing it with `pack_obj`
  (old:contracts/contracts/examples.py:109-139). All of them are C2/C3 (C3 includes `ack/v1`, `error/v1` and the
  reply messages) and all are stored.
- `python -m deploy_contracts.make_examples [--out DIR]` writes them to `deployment/contracts/examples/` with
  `index.yaml` (format 1, unchanged layout, old:contracts/STRUCTURE.md §8), removing stale files. Never edited by hand.
- Stored numbers use only `+ − × /` on literals (no trigonometry), so the bytes are identical with any numpy. Poses use
  the exact rotation `(cos, sin) = (0.6, 0.8)` (old:contracts/contracts/examples.py:141-147). Hello examples use
  literal values, never values read from the yaml files, so tunables never change example bytes.
- The helpers are public for policy_deploy's in-memory C0/C1/C4 examples: `Example`, `good_example(name, msg,
  values=None, note="")`, `broken_example(name, base, change, field, problem, note, schema=None)`, `raw_example(name,
  data, schema, field, problem, note)`, `set_key(key, value)`, `drop_key(*path)`, `resolve_path(obj, path)`.
- The expected error of every stored example assumes only that its schema name is unknown or known as listed; none
  depends on laptop-only schemas being registered, so policy_deploy's tests may also load them.

### 5.2 The list

C2 (old:contracts/contracts/examples.py:164-241, plus one):

| Good | Broken (field, problem) |
|---|---|
| `state_running`, `state_holding`, `state_homing`, `state_fault`, `state_idle_zero`, **`state_stopped`** (new: mode `stopped`, the watchdog case) | `bad_pose_transposed` (O_T_TCP, value), `bad_pose_not_orthonormal` (O_T_TCP, value), `bad_mode` (mode, value), `missing_loop_late` (loop_late, missing), `missing_gate_pulled_back` (gate.pulled_back, missing), `wrong_shape_q` (q, shape), `wrong_dtype_q` (q, dtype), `bad_width_negative` (gripper_width_m, value), `bad_width_int` (gripper_width_m, type), `bad_gate_shortened` (gate.shortened, value), `bad_dq_nan` (dq, value), `bad_type_seq` (seq, type), `bad_seq_negative` (seq, value), `bad_array_byte_count` (O_T_TCP, shape), `unknown_field` (temperature, unknown_field), `schema_version` ("", schema_version) |

C3, existing (old:contracts/contracts/examples.py:255-396, unchanged):

| Good | Broken |
|---|---|
| `set_waypoints_8`, `set_waypoints_1`, `heartbeat`, `start`, `stop`, `home`, `get_state`, `set_waypoints_reply_ok`, `set_waypoints_reply_rejected`, `ack_ok`, `ack_refused`, `error_set_waypoints_refused`, `error_contract` | `set_waypoints_times_not_increasing`, `_lengths_differ`, `_pose_not_rigid`, `_pose_nan`, `_too_many` (max_waypoints + 1), `_empty`, `_width_negative`, `_width_nan`, `_wrong_shape`, `_wrong_dtype_t`, `_wrong_dtype_width`, `_bad_array_map`, `_missing_width`, `_unknown_field`, `_type_chunk_id`, `_chunk_id_negative`, `_schema_version`; `heartbeat_unknown_field`, `unknown_schema` (c3.jump/v1), `no_schema`, `not_a_map`, `not_msgpack`, `trailing_bytes`; `set_waypoints_reply_shortened_gt_accepted`, `_duplicate_index`, `_missing_skipped`, `_missing_reason`; `ack_ok_not_bool`, `ack_missing_reason`, `ack_schema_version`; `error_missing_field` |

C3, new for `hello`:

| Name | Valid | Content / expected error |
|---|---|---|
| `c3/hello` | good | `{schema: c3.hello/v1}` |
| `c3/hello_reply` | good | `t_pc_ns` 123461789000000, `commit` `0123456789abcdef0123456789abcdef01234567`, `mock` false, `schemas` = `SHARED_SCHEMAS`, `F_T_TCP` `[[0.6, 0.8, 0, 0], [-0.8, 0.6, 0, 0], [0, 0, 1, 0.15], [0, 0, 0, 1]]`, `rotation_tol` 1.0e-6, `gripper_width_max_m` 0.08, `max_waypoints` 64, `watchdog_limit_s` 1.0, `homing_timeout_s` 30.0, `max_lead_s` 1.5; `values` list every scalar plus `F_T_TCP[0][1]`, `F_T_TCP[2][3]` |
| `c3/hello_reply_mock` | good | as above with `mock` true, `commit` `unknown` |
| `c3/hello_unknown_field` | broken | request with `commit: "x"`: (commit, unknown_field) |
| `c3/hello_schema_version` | broken | `c3.hello/v2`: ("", schema_version) |
| `c3/hello_reply_missing_max_lead_s` | broken | (max_lead_s, missing) |
| `c3/hello_reply_t_pc_ns_float` | broken | `t_pc_ns` as a float: (t_pc_ns, type) |
| `c3/hello_reply_mock_int` | broken | `mock` 1: (mock, type) |
| `c3/hello_reply_commit_short` | broken | `commit` `0123456`: (commit, value) |
| `c3/hello_reply_schemas_unsorted` | broken | two entries swapped: (schemas, value) |
| `c3/hello_reply_schemas_item_int` | broken | `schemas[3]` = 7: (schemas[3], type) |
| `c3/hello_reply_F_T_TCP_transposed` | broken | (F_T_TCP, value) |
| `c3/hello_reply_F_T_TCP_shape` | broken | (3, 4): (F_T_TCP, shape) |
| `c3/hello_reply_watchdog_zero` | broken | `watchdog_limit_s` 0.0: (watchdog_limit_s, value) |
| `c3/hello_reply_max_waypoints_zero` | broken | (max_waypoints, value) |
| `c3/hello_reply_schema_version` | broken | `c3.hello.reply/v2`: ("", schema_version) |

### 5.3 How the tests read them

**Python** (`deployment/contracts/tests/test_examples.py`, run on both machines): every example from
`build_examples()` decodes (good, `values` match via `resolve_path`) or fails with exactly its `field` and `problem`
(broken); every file in `examples/` is in `index.yaml` and vice versa; the folder equals a fresh `make_examples` output
(a stale file fails); `encode(decode(b)) == b` for every good one.

**C++** (`control_deploy/franka_control/tests/test_codec.cpp`, from `CONTRACTS_EXAMPLES_DIR/index.yaml`, as
old:franka_control/tests/test_codec.cpp):

1. every good example decodes and its `values` match (now including the hello request and replies);
2. every broken example is refused with the same `field`, `problem` and `schema`;
3. every good example re-encodes **byte for byte** (C2, replies, `ack`, `error`, `hello.reply`): this proves the C++
   encoder writes the canonical form;
4. new: the decoded `c3/hello_reply` `schemas` equals the codec's known-schema list in order, which proves the PC knows
   exactly `SHARED_SCHEMAS`;
5. limits, frames and link values come from the contract files (`CONTRACTS_*_FILE`).

**policy_deploy** reads the same folder (through `deploy_contracts.load_index()`) in its fake-PC tests, so the fake PC
decodes and answers what franka_control does.

## 6. Clocks

### 6.1 The three clocks

| Clock | Where it exists | Carried by | Mapped by |
|---|---|---|---|
| Pi clock (the clock of libcamera's `SensorTimestamp`) | the Pi; inside frames only | C0 `t_sensor_ns`, `t_mid_ns`, `t_send_ns` | frames: Pi → laptop, one-way from `t_send_ns` (PLAN §5.6) |
| laptop `CLOCK_MONOTONIC` | the laptop | C1 times; arrival stamps; laptop logs | — |
| PC `CLOCK_MONOTONIC` | the PC | **every time on C2 and C3**: C2 `t_ns`, C3 `set_waypoints.t_ns`, `hello.reply.t_pc_ns` | manager: laptop → PC, `round_trip` or `one_way` (PLAN §5.6) |

All in integer nanoseconds. **A time crosses into another clock only through a map** (PLAN §5). Both maps live in
`policy_deploy/`; `deploy_contracts` has no clock code. The wire names stay `t_ns` (C2, C3: no schema change); in
laptop code, variables holding PC times end in `_pc_ns`, Pi times in `_pi_ns`, laptop times in `_ns`.

### 6.2 What the PC guarantees

- One clock, `CLOCK_MONOTONIC`, for C2 `t_ns`, `hello.t_pc_ns`, the gate's "now" (skip a passed waypoint; reject one
  more than `max_lead_s` ahead) and the watchdog. The PC never changes its clock and has no clock code (PLAN §5.6).
- C2 `t_ns` is the reading time (start of the 1 kHz cycle), not the send time: a state goes out 0 to about 6 ms after
  its `t_ns` (the messaging thread's poll, old:franka_control/config.yaml:78). It repeats as in §3.6 obligation 1.
- `hello.t_pc_ns` is stamped as in §3.8 obligation 2.
- `--mock` (C++ fake arm) uses the same real `CLOCK_MONOTONIC`.

### 6.3 What the laptop must do (for policy_deploy/STRUCTURE.md)

- Keep every arm-related time on the PC clock; convert only laptop times into PC time (the two main image times, and
  "now" for freshness). Send waypoint times already on the PC clock (PLAN §4, §5.6).
- Judge C2 freshness from mapped `t_ns`, never from arrival time (PLAN §5.3). Judge liveness ("no C2 for
  `state_lost_s`") from arrival time, because `t_ns` can repeat (§3.6 obligation 1).
- Ignore repeated `t_ns` for interpolation and for the `one_way` map; never use `get_state` replies as samples
  (§3.6 obligation 3).
- `round_trip` samples: stamp the laptop clock just before sending `hello` and just after its reply; sample =
  (`t_pc_ns`, midpoint). While a C3 `home` is pending (up to `homing_timeout_s`) no hello can be sent, so the map gets
  no samples for that time; after it the laptop sends a hello every cycle again until the map is ready.
- A hello `t_pc_ns` going back means the PC rebooted (`CLOCK_MONOTONIC` survives a franka_control restart; a restart
  shows as C2 `seq` going back).
- The guards of PLAN §5.6 (2 s of samples, 5 ms start-up agreement with a hello, fit jumps logged, `clock mismatch`
  after three waypoint sets that were entirely skipped, or sent while the newest C2 `t_ns` was more than 20 ms ahead
  of the mapped "now").
- The fake PC runs on `laptop CLOCK_MONOTONIC × (1 + drift) + offset` (e.g. +1000 s and 30 ppm) for C2 `t_ns`,
  `t_pc_ns` and its own gate, so every laptop test exercises the map (PLAN §8.2).

## 7. Versioning, same commit, tests

### 7.1 Versioning

- A schema is `<name>/v<N>`. Only one version of each is supported; old ones are not kept (one repo, deployed
  together).
- On a mismatch (known name, other version): `decode` raises `ContractError(problem="schema_version")`; a server
  answers `error/v1`; a client raises. The laptop concludes, naming the message. The C++ side does the same.
- `hello.schemas` exposes a mismatch at start-up (§3.8).
- After a bump: update the class; `python -m deploy_contracts.make_examples`; update the C++ codec and tests until
  `test_codec` passes; same commit.

### 7.2 Same commit

- Both machines check out the same commit of `rnoro/final-deployment` (PLAN §5.5).
- The PC **rebuilds franka_control after every pull** (baked paths, `commit` in `hello`).
- `hello.commit` is shown next to the laptop's commit by `policy_deploy check` and at `run` start-up (PLAN §5.5).
- The real guards are content checks, independent of the commit: `schemas`, the limits, `F_T_TCP`, the watchdog
  (`compare_hello`), and the example files tested on both machines.

### 7.3 The contracts' own tests

All in `deployment/contracts/tests/`, run on both machines with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest deployment/contracts/tests`. No Pi, arm or GPU.

| File | Checks |
|---|---|
| `test_schema.py`, `test_fields.py`, `test_message.py`, `test_codec.py`, `test_registry.py` | as old:contracts/STRUCTURE.md §10 (C2/C3-sized); registry: importing `deploy_contracts` alone registers exactly `SHARED_SCHEMAS`; duplicate registration refused |
| `test_shared.py` | `SHARED_SCHEMAS` sorted, unique, equal to the registered shared classes' schemas |
| `test_paths.py` | the four paths resolve next to the package; a missing `limits.yaml` gives the install hint |
| `test_limits.py` | loads; 3 keys exactly; missing, unknown, wrong type, ≤ 0 refused |
| `test_frames.py` | loads the real file; not 4×4, non-finite, transposed, reflection refused; result read-only |
| `test_link.py` | loads the real file; every key's type and range; `0.0.0.0` and hostnames refused; equal ports refused; the cross-key rules; unknown estimator refused; endpoints with and without `local` |
| `test_common.py`, `test_c2.py`, `test_c3.py` | each message's rules (old list), plus `Hello` (no fields) and every `HelloReply` rule of §3.8 |
| `test_hello_check.py` | an equal reply gives no mismatch; each field differing gives exactly that field; `F_T_TCP` within 1e-9 passes, 1e-6 off fails; `schemas` lists only the differing entries |
| `test_examples.py` | §5.3 |
| `test_rpc.py`, `test_pubsub.py` | as old (C0 multipart cases move to policy_deploy); `RpcServer(schemas=SHARED_SCHEMAS)` answers a registered non-shared schema with `unknown_schema` and leaves every other problem code unchanged; without `schemas`, behaviour as old |
| `test_import_rules.py` | `deploy_contracts` imports only stdlib, numpy, msgpack, zmq, yaml; nothing from `policy_deploy` or `control_deploy` |

## 8. Decisions (were open questions)

Settled by the lead on 2026-11-13, with PLAN.md updated to match:

1. **Commit mismatch between the machines: warn**, in `check` and at `run` start-up. Refuse only on what
   `compare_hello` and the laptop's budget checks find.
2. **The laptop's start-up order:** C2 arriving → wait while C2 shows `running`/`holding` (rule H2, no C3, up to
   2 × watchdog), and likewise while it shows `homing` (up to 60 s: the PC's REP doesn't answer during a C3 home) →
   hellos start (2 s of samples) → `compare_hello` and the laptop's checks → mode `idle`/`stopped` via
   `get_state` (wait while `homing`; refuse in `fault`) → heartbeats → policy config → cameras ok → ready (PLAN §8.2).
3. **`settled: false` in `frames.yaml`** (§4.2): `policy_deploy` refuses a trained policy on the real arm while it is
   false.
4. **The PC keyboard** (PLAN §8.1): `h` homes the arm, `g` homes the hand (C2 shows `homing`; afterwards the fingers
   return to the commanded width), `s` stops and holds, `q` quits. `h` and `g` are refused in `holding`/`running`
   (the laptop is driving), `homing` and `fault`; `s` and `q` are never refused. Keys never feed the watchdog.
5. **Stop during homing** (§3.7.2): a C3 `stop` or the PC's `s` aborts any homing move → `stopped`, holding where it
   stopped; if the move finished first → `idle`. A C3 `home` stopped this way replies `ok: false` with the `stopped:`
   prefix; the laptop shows it and stays idle. Any other not-ok `home` reply concludes on the laptop.
6. **Between episodes** the PC's keys may move it to `stopped` or `homing`; the laptop follows C2: `c` from
   `idle`/`stopped`; while C2 shows `homing` the laptop refuses `c` and `h` itself with a hint and stays idle.
7. **C4 config reply v2** (§3.12): it carries the policy's `kind`, so the settled rule needs no station reading.
8. **The ultrawide pair rule** (PLAN §7): a pair is usable only within [main − 60 ms, main + 25 ms] (mid-exposures;
   the lead confirmed from `pi_collect`) and when the same ultrawide frame was the pair of at most 2 consecutive
   packets. The fake Pi pairs the nearest frame, like the live Pi. Laptop-only (policy_deploy/STRUCTURE.md §4.3).
9. **The fake PC's schema filter** (§3.11): it answers laptop-only schemas `unknown_schema`, like franka_control.
10. **The cross test** (PLAN §10) runs on the laptop against a libfranka-less `franka_control --mock --local`
    (control_deploy/STRUCTURE.md §8.3 builds it; policy_deploy/STRUCTURE.md §12.3 owns the test).
11. **`LibfrankaArm`'s stop flag becomes a per-call stop epoch** (control_deploy/STRUCTURE.md §3.3), so the process
    survives a stopped homing and a later reflex is still a reflex. It is on the verified path: tested at bring-up.
12. **`deploy-contracts` is never a pyproject dependency** of either folder; it is installed editable first (§1.4).
