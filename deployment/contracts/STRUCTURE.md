# contracts/

The design of the shared contracts package. Written from PLAN.md sections 2, 4, 5 and 7. PLAN.md is the source of
truth: where this file and PLAN.md disagree, PLAN.md wins and this file is fixed.

## 1. Role

`contracts/` is **the only code shared between processes** (PLAN rule 3). It holds:

1. **The message definitions** for C0 to C4: one Python class per message, with exactly the fields, types, shapes and
   units of PLAN section 4.
2. **The validators**: every message is checked when it is built and when it is received. A message that breaks its
   contract raises one error that names the message and the field.
3. **The encoding**: how a message becomes bytes on the network (msgpack, with arrays as raw bytes plus shape and
   dtype), and back.
4. **The schema names and versions** (`c2.state/v1`, ...), and what happens on a mismatch.
5. **Example messages**, good ones and deliberately broken ones, for every contract. The C2 and C3 examples (the ones
   C++ needs) are **committed as files** that the Python tests and the C++ tests of `franka_control/` both read. That
   is how Python and C++ are proven to agree without C++ ever importing Python. The C0, C1 and C4 examples are
   Python-only, so they are generated in memory during the tests and not stored.
6. **Tiny transport helpers** over ZeroMQ: a request/reply client, a request/reply server, a publish/subscribe
   pair, and a publish/subscribe pair for C0's multipart packets, all with timeouts (section 6 says why these live
   here).

It knows nothing about cameras, policies or the arm beyond the shape of the messages. It has no process of its own and
no `launch.py`.

Terms used below:

- **msgpack**: a compact binary format, like JSON but in bytes. It has maps, lists, strings, integers, floats, booleans,
  `nil` (nothing) and `bin` (raw bytes). There are libraries for it in Python (`msgpack`) and C++ (`msgpack-cxx`).
- **ZeroMQ** (`pyzmq` in Python, `libzmq` + `cppzmq` in C++): a messaging library. A *socket* sends whole messages
  (blobs of bytes) to another socket. Two patterns are used:
  - **request/reply** (`REQ`/`REP` sockets): the client sends one request and waits for exactly one reply.
  - **publish/subscribe** (`PUB`/`SUB` sockets): the publisher sends messages to whoever is listening and never waits.
- **dtype**: the element type of a numpy array (`uint8`, `int64`, `float32`, ...).
- **dataclass**: a Python class whose fields are declared in its body; Python writes the constructor.
- **schema**: the name and version of a message, carried in the message itself, e.g. `c2.state/v1`.

## 2. File tree

One concern per file. A "concern" is one contract (C0, C1, ...) for the message files: each contract's messages are
small and always change together, so they share a file (25 one-class files would be harder to read, not easier).

```
contracts/
  STRUCTURE.md
  pyproject.toml            package "contracts"; deps: numpy, msgpack, pyzmq, pyyaml (see deploy/STRUCTURE.md §7)
  contracts/                the Python package
    __init__.py             re-exports the public names below (messages, encode/decode, errors, transport)
    errors.py               ContractError, TransportTimeout, TransportCancelled, RemoteError
    schema.py               SchemaName: parse "c2.state/v1", compare names and versions
    fields.py               the field declarations: scalar(), array(), nested(), list_of(), str_list(), optional()
    message.py              Message: the base class (validate on build, to_dict, from_dict)
    registry.py             schema string -> message class, filled by the @register decorator
    codec.py                encode / decode, pack_array / unpack_array (the wire format, section 5)
    limits.py               ContractLimits: loads limits.yaml once
    limits.yaml             the numeric tolerances the validators use (section 4.7)
    sizes.py                the fixed C1/C4 sizes as named constants: OBS_STEPS, ACTION_STEPS, STEP_MS, IMAGE_SIZE
    _rot6d.py               private: rot6d row checks (unit length, perpendicular), only for validators
    common.py               Ack, ErrorReply
    c0_frames.py            PacketHeader, FrameInfo, Packet, EncodedFrame, TimeSyncRequest, TimeSyncReply
    c1_frames.py            GetFramesRequest, FramesReply, GetHealthRequest, HealthReply, HealthReport,
                            CameraHealthReport
    c2_state.py             ArmState, GateCounters
    c3_commands.py          SetWaypoints, SetWaypointsReply, WaypointIssue, Heartbeat, Start, Stop, Home, GetState
    c4_policy.py            PolicyConfigRequest, PolicyConfig, Reset, Observation, ActionChunk, Prompt
    rpc.py                  RpcClient, RpcServer (request/reply)
    pubsub.py               Publisher, Subscriber (publish/subscribe); PacketPublisher, PacketSubscriber (C0 multipart)
    examples.py             Example, build_examples(), examples_dir(), load_index()
    make_examples.py        `python -m contracts.make_examples`: writes the C2/C3 examples to examples/
  examples/                 the shared C2/C3 example files (data, checked into git; section 8)
    index.yaml
    c2/  c3/                one .msgpack file per example (c3/ includes the ack/v1 and error/v1 replies)
  tests/
    ...                     section 10
```

## 3. How a message class is written

Each message is a **frozen dataclass** (its fields can't be reassigned after it is built) that subclasses `Message`.
The class body *is* the field table: each field is declared with a helper from `fields.py` that records its type,
dtype, shape or allowed values. The validator reads those declarations, so the table and the checks can never drift
apart.

`fields.py` helpers (each returns a dataclass field carrying the spec):

| Helper | Meaning | Python value |
|---|---|---|
| `scalar(kind, choices=None, min=None, max=None)` | one value; `kind` is `int64`, `uint64`, `uint32`, `int`, `float64`, `bool`, `str` or `bytes` | `int`, `float`, `bool`, `str`, `bytes` |
| `array(dtype, shape)` | numpy array of exactly this dtype; `shape` is a tuple where `None` means "any length" | `np.ndarray` |
| `nested(cls)` | a sub-message (a map on the wire) that has no `schema` of its own | an instance of `cls` |
| `list_of(cls)` | a list of sub-messages | `tuple[cls, ...]` |
| `str_list(choices=None)` | a list of strings, each one of `choices` when given (e.g. `PolicyConfig.inputs`) | `tuple[str, ...]` |
| `optional(spec)` | the field may be `None` (msgpack `nil`); it is never left out | value or `None` |

Every field is always present on the wire (an optional one is sent as `nil`). Leaving a field out is a contract error.

`Message` base class (`message.py`):

| Member | What it does |
|---|---|
| `SCHEMA: ClassVar[str]` | e.g. `"c2.state/v1"` |
| `__post_init__()` | calls `validate()`. **A `Message` object that exists is valid.** |
| `validate() -> None` | checks every field against its spec, in declaration order, then calls `check_values()`. Raises `ContractError` at the first problem. |
| `check_values() -> None` | hook for checks that involve several fields or numbers (e.g. "times increasing"). Default: nothing. |
| `to_dict() -> dict` | plain dict in field order, `schema` first; sub-messages become dicts, arrays stay arrays |
| `from_dict(d: dict) -> Self` (classmethod) | checks `schema`, missing and unknown keys, then builds (and so validates) the object |

Sub-messages (`nested`, `list_of`) subclass `Record`, a lighter base in `message.py`: same field helpers and
validation, no `schema` key.

Arrays given to a constructor are not copied or converted: a `float64` array where `float32` is required is an error,
not silently cast. This catches dtype mistakes in the sender instead of the receiver.

**Value checks on float arrays compare in float64.** Every range or tolerance check on a float array (widths against
`[0, gripper_width_max_m]`, anchor rows, rot6d norms and dot products, rigid-pose checks) first casts the values to
float64 (a private copy used only for the check; the message keeps its own array) and compares there. Under numpy 2.x,
comparing a float32 array with a Python float runs in float32, so `float32(0.07)` = 0.0700000003 would pass a 0.07
maximum; in float64 it is correctly refused.

## 4. The messages

Field tables copied from PLAN section 4. Order matters: it is the order on the wire (section 5) and the order the
validator checks in. Every top-level message starts with `schema` (str), not repeated in the tables below.

Conventions (PLAN §4): `t_ns` is int64 nanoseconds on the PC's `CLOCK_MONOTONIC` (Python `time.monotonic_ns()`);
metres, radians, seconds; images uint8 RGB `(H, W, 3)`; absolute poses are 4×4 float64 `O_T_TCP` (TCP in the robot
base frame), stored **row-major** (`M[row, col]`, the numpy default).

### 4.1 Common replies (`common.py`)

`Ack`, schema `ack/v1`. The reply to requests whose answer is "ok" or "refused".

| Field | Type | Meaning |
|---|---|---|
| `ok` | bool | true: done. false: refused |
| `reason` | str | empty when `ok`; otherwise one line, e.g. `"refused: mode is fault"` |

`ErrorReply`, schema `error/v1`. What a server sends instead of its normal reply when it can't answer: the request
broke its contract, the request type is unknown to this server, or the handler raised.

| Field | Type | Meaning |
|---|---|---|
| `request` | str | the schema of the request, or `""` if it couldn't be read |
| `error` | str | one line, e.g. `"c4.infer/v1: field 'main': shape (2, 224, 3), expected (2, 224, 224, 3)"` |
| `field` | str | the field at fault when it is a contract error, else `""` |

### 4.2 C0: Pi 5 → `process_frames` (`c0_frames.py`)

The exact contract is [PI_PAYLOAD.md](PI_PAYLOAD.md), written for whoever implements the Pi sender; this section is
its Python form. The Pi sends **one packet per main (IMX519) frame**: one ZeroMQ **multipart** message on a PUB socket
the Pi binds. A multipart message arrives whole or not at all, so a packet is never half-received.

| Part | Content |
|---|---|
| 0 | the header, a `PacketHeader` message (msgpack, `c0.packet/v1`) |
| 1 | the main frame: one encoded frame, raw bytes |
| 2 … 1+N | the N ≥ 0 ultrawide (IMX219) frames captured since the previous packet, raw bytes each, in capture order, matching `header.ultrawide` one to one |

The part count is exactly `2 + len(header.ultrawide)`. A packet usually carries one ultrawide frame; timing jitter
gives 0 or 2. Every encoded ultrawide frame travels exactly once, in capture order, even one that is nobody's pair,
because each H.264 frame depends on the ones before it.

`FrameInfo` (sub-message, the "frame map"; the same keys for the main frame and each ultrawide frame):

| Field | Type | Meaning / check |
|---|---|---|
| `sensor` | str, one of `imx519`, `imx219` | the sensor; `imx519` for the main frame, `imx219` for every ultrawide frame (checked by `PacketHeader`, which catches swapped cameras); `process_frames` also checks it against the rig file |
| `frame_seq` | uint64 | per camera, counts **captured** frames from 0 (+1 per frame), so a gap means that camera lost frames, including ones the Pi dropped before encoding |
| `t_pi_ns` | int64 | libcamera `SensorTimestamp` verbatim: the end of the first row's exposure, on the Pi's `CLOCK_BOOTTIME` |
| `exposure_ns` | int64, ≥ 0 | libcamera `ExposureTime` (µs) × 1000; 0 if unknown, in which case `process_frames` uses `default_exposure_ns` from its own `config.yaml` |
| `width`, `height` | uint32 | encoded frame size; `process_frames` checks it against the rig file |
| `encoding` | str, one of `h264`, `jpeg` | `h264` default, `jpeg` backup; the same for every frame of a camera during a run |
| `keyframe` | bool | H.264: true for a frame that decodes on its own (an IDR frame). JPEG: must be true |

`PacketHeader`, schema `c0.packet/v1`:

| Field | Type | Meaning / check |
|---|---|---|
| `packet_seq` | uint64 | counts **main frames captured** since the sender started, from 0 (+1 per packet; a main frame the Pi drops still counts). A gap means lost packets; going backwards means the sender restarted |
| `main` | `FrameInfo` | the frame of part 1 |
| `ultrawide` | list of `FrameInfo` | the frames of parts 2 … 1+N, in capture order; may be empty |
| `pair_seq` | int64, ≥ −1 | the `frame_seq` of this packet's paired ultrawide frame (in this packet or an earlier one), or −1 if there is no pair |

`check_values()`: `main.sensor` is `imx519` and every `ultrawide[i].sensor` is `imx219`; the ultrawide frames'
`frame_seq` strictly increase (capture order; a Pi-side drop may leave a gap) and they share one `encoding`; when the
packet carries ultrawide frames, `pair_seq` is not newer than the newest of them. A pair from an earlier packet can't
be checked by the contract (the header doesn't list earlier frames): `process_frames` looks it up among the frames it
has received.

**The pairing rule** (the Pi's, PI_PAYLOAD.md §5 rule 3): `pair_seq` is the ultrawide frame with the latest capture
time not later than the main frame's + 5 ms (comparing `t_pi_ns`), at most 50 ms older than the main frame; −1 if none
qualifies. With both cameras at 30 FPS a pair is normally 0–33 ms apart. The contract can't check it (it needs the
frames of earlier packets); `process_frames`' `PacketPairer` implements it for the fake Pi and the tests.

`EncodedFrame` (a plain frozen dataclass, not a message): `info: FrameInfo`, `data: bytes`: one frame as a decoder
and a sender handle it.

`Packet` (a plain frozen dataclass, not a message; **a `Packet` object that exists is valid**):

```
@dataclass(frozen=True)
class Packet:
    header: PacketHeader
    main_data: bytes                      # part 1
    ultrawide_data: tuple[bytes, ...]     # parts 2 ... 1+N
    @classmethod
    def build(cls, packet_seq, main: EncodedFrame, ultrawide: Sequence[EncodedFrame], pair_seq) -> Packet
    main: EncodedFrame                    # property
    ultrawide: tuple[EncodedFrame, ...]   # property
    def to_parts(self) -> list[bytes]     # [encode(header), main_data, *ultrawide_data]
    @classmethod
    def from_parts(cls, parts: Sequence[bytes]) -> Packet    # decode(parts[0], PacketHeader), then the part checks
```

The part checks raise `ContractError` with schema `c0.packet/v1`: a wrong part count is field `parts`, problem
`value` ("2 parts, the header lists 1 ultrawide frames: expected 3"); an empty part is field `parts[i]`, problem
`value`; a part that isn't bytes is `parts[i]`, `type`. A broken header raises from `decode` as any message does
(e.g. field `main.width`, `ultrawide[0].sensor`).

**Requirements on the Pi sender** that the contract can't check (PI_PAYLOAD.md §5; `process_frames`' decoder and
tests check them, and its `FakePiSender` follows all of them):

- one packet per main frame, sent as soon as that main frame is encoded;
- every encoded ultrawide frame sent exactly once, in the first packet after it was encoded, in capture order;
- H.264: the recording's encoder settings (main 15 Mbit/s, ultrawide 10 Mbit/s), Constrained Baseline (no
  B-frames), one access unit per frame in **Annex-B** framing (each NAL unit starts with a `00 00 01` /
  `00 00 00 01` start code), SPS/PPS before every keyframe (in the same part), a keyframe every 30 frames (1 s);
- the Pi's PUB `SNDHWM` is 30 messages (about 1 s), so a PC that falls behind loses whole packets;
- `packet_seq` and each `frame_seq` start at 0 and never reset during a run; both cameras run at 30 FPS.

`TimeSyncRequest`, schema `c0.time_sync_request/v1`: no fields. Sent by `process_frames` to the Pi's REP socket
about once a second (10 back to back at start-up).

`TimeSyncReply`, schema `c0.time_sync_reply/v1`:

| Field | Type | Meaning |
|---|---|---|
| `t_pi_ns` | int64 | the Pi's `CLOCK_BOOTTIME` in ns, read as close as possible to sending the reply |

**Status:** the Pi sender isn't written yet (PLAN §10). It may use this package to build and check the messages, or
write the msgpack itself; either way the bytes must match PI_PAYLOAD.md.

### 4.3 C1: `process_frames` ↔ `message_comms` (`c1_frames.py`)

`GetFramesRequest`, schema `c1.get_frames/v1`: no fields. `GetHealthRequest`, schema `c1.get_health/v1`: no fields.

`CameraHealthReport` (sub-message; named so it doesn't clash with `process_frames`' own `CameraHealth` class):

| Field | Type | Meaning |
|---|---|---|
| `status` | str, one of `ok`, `stale`, `down` | |
| `last_frame_age_ms` | float64, ≥ 0 | age of the newest frame, PC time, ms |
| `fps` | float64, ≥ 0 | measured frame rate, Hz |
| `dropped` | uint64 | frames dropped (`seq` gaps) since `process_frames` started |

`HealthReport` (sub-message): `main`: `CameraHealthReport`, `ultrawide`: `CameraHealthReport`.

`FramesReply`, schema `c1.frames/v1`:

| Field | Type / shape | Meaning |
|---|---|---|
| `main` | optional uint8 (`OBS_STEPS`, `IMAGE_SIZE`, `IMAGE_SIZE`, 3) = (2, 224, 224, 3) | 2 main frames, oldest first, 50 ms apart, prepared as in training |
| `main_t_ns` | optional int64 (`OBS_STEPS`,) | PC mid-exposure time of each main frame |
| `ultrawide` | optional uint8 (`OBS_STEPS`, `IMAGE_SIZE`, `IMAGE_SIZE`, 3) | 2 ultrawide frames, oldest first, 100 ms apart, remapped and prepared as in training |
| `ultrawide_t_ns` | optional int64 (`OBS_STEPS`,) | PC mid-exposure time of each |
| `health` | `HealthReport` | |

`check_values()`: **both cameras or nothing.** All four image and time fields are present **if and only if both**
cameras' `status` is `ok`. If either camera is not `ok`, all four are `nil` and only `health` is meaningful (PLAN: "no
images" when a camera can't supply valid frames; never old images as new). Times strictly increase.

`HealthReply`, schema `c1.health/v1`: `health`: `HealthReport`.

The spacings (50/100 ms) are not checked by the contract: they come from `process_frames`' config, and
`message_comms` checks them itself if it wants to.

### 4.4 C2: `franka_control` → `message_comms` (`c2_state.py`)

`GateCounters` (sub-message), each uint64, waypoints since `franka_control` started:

| Field | Meaning |
|---|---|
| `accepted` | accepted by the gate (shortened and pulled-back ones included) |
| `shortened` | accepted, shortened to the speed limit |
| `pulled_back` | accepted, pulled back onto the workspace box or tilt cone because they were outside it by at most the margin |
| `rejected` | refused (outside by more than the margin, width out of range, non-finite): `franka_control` goes to `fault` |

`ArmState`, schema `c2.state/v1`. Broadcast on a PUB socket at ~100 Hz; also the reply to C3 `get_state`.

| Field | Type / shape | Meaning |
|---|---|---|
| `t_ns` | int64 | PC time the state was read from the arm |
| `seq` | uint64 | +1 per message; a gap means a missed message |
| `O_T_TCP` | float64 (4, 4) | measured hand pose at the TCP, base frame |
| `O_T_TCP_target` | float64 (4, 4) | the target the 1 kHz loop is tracking now |
| `q` | float64 (7,) | joint angles, rad |
| `dq` | float64 (7,) | joint speeds, rad/s |
| `gripper_width_m` | float64, ≥ 0 | opening between the fingertips, m |
| `mode` | str, one of `idle`, `running`, `holding`, `stopped`, `homing`, `fault` | |
| `error` | str | empty, or the libfranka or safety error text |
| `gate` | `GateCounters` | |
| `loop_late` | uint64 | 1 kHz cycles that ran late since start (status line; mainly for the ignore-real-time backup) |

`check_values()`: `gate.shortened ≤ gate.accepted` and `gate.pulled_back ≤ gate.accepted`; all floats finite; both poses are **rigid transforms**: bottom row exactly `[0, 0, 0, 1]`, rotation
part orthonormal (`max |RᵀR − I| ≤ rotation_tol`) with determinant > 0. This check matters for C++: libfranka gives
poses as 16 numbers in **column-major** order. A matrix sent without transposing puts the translation in the bottom
row, and this check rejects it immediately.

### 4.5 C3: `message_comms` → `franka_control` (`c3_commands.py`)

`SetWaypoints`, schema `c3.set_waypoints/v1`. The waypoints are three parallel arrays of the same length N
(waypoint i is `(t_ns[i], O_T_TCP[i], gripper_width_m[i])`): one msgpack array each instead of N small maps, which is
smaller, faster, and simpler to decode in C++.

| Field | Type / shape | Meaning |
|---|---|---|
| `chunk_id` | uint64 | which policy step produced these; echoed in the reply and logs |
| `t_ns` | int64 (N,) | the time each waypoint should be reached, PC monotonic |
| `O_T_TCP` | float64 (N, 4, 4) | the absolute TCP pose of each waypoint, base frame |
| `gripper_width_m` | float64 (N,) | the absolute opening of each waypoint, m, ≥ 0 |

`check_values()`: `1 ≤ N ≤ max_waypoints`; the three lengths agree; `t_ns` strictly increasing; all finite; every
pose a rigid transform (as in C2). Whether a waypoint is safe is **not** a contract question: that is the safety gate's
job in `franka_control`.

`WaypointIssue` (sub-message): `index` uint64 (position in the request), `reason` str.

`SetWaypointsReply`, schema `c3.set_waypoints.reply/v1`:

| Field | Type | Meaning |
|---|---|---|
| `chunk_id` | uint64 | echo of the request |
| `accepted` | uint64 | how many waypoints were accepted, **shortened ones included** |
| `shortened` | uint64 | how many of the accepted ones were shortened to the speed limit |
| `skipped` | uint64 | how many were skipped because their time had already passed |
| `rejected` | list of `WaypointIssue` | refused by the gate, each with its reason (any rejection makes `franka_control` go to `fault`, PLAN §5) |

`check_values()`: `shortened ≤ accepted`; `rejected` indices unique. Waypoints pulled back to the box or cone are
accepted and counted in C2's `gate.pulled_back`.

`Heartbeat` (`c3.heartbeat/v1`), `Start` (`c3.start/v1`), `Stop` (`c3.stop/v1`), `Home` (`c3.home/v1`),
`GetState` (`c3.get_state/v1`): no fields. Replies: `Ack` for `heartbeat`, `start` (ok or refused), `stop`, `home`
(sent when the home pose is reached); `ArmState` for `get_state`.

**Requirements on `franka_control` (state rules that shape the replies):**

- **`set_waypoints` refused because of the state** (e.g. `idle`, `stopped`, `homing` or `fault`, where no waypoints
  are followed) gets an **`ErrorReply`** saying why (`"set_waypoints refused: mode is idle"`). This is **not** a fault:
  the arm's state does not change. (`message_comms` treats the `RemoteError` as its own logic error and concludes.)
- **`start`** is allowed from `idle` or `stopped` (`Ack(ok=true)`); from any other mode it is refused with
  `Ack(ok=false, reason=...)`.
- **`home`** is allowed only from `idle` or `stopped`, and otherwise refused with `Ack(ok=false, ...)`. While it runs
  the mode is `homing`; on success it ends in **`idle`**. Homing has **its own timeout of 30 s** (`homing.timeout_s` in
  `franka_control/config.yaml`); exceeding it ends homing with `Ack(ok=false, reason=...)`.
- `home` blocks its reply until the home pose is reached, so `message_comms` cannot send heartbeats meanwhile (a REQ
  socket must get its reply before it sends again). So the **watchdog pauses while homing** and **resumes** as soon as
  homing ends, either way. `message_comms` uses a `home` request timeout longer than 30 s (its config).
- **The watchdog is fed by any valid C3 request**, not only `heartbeat`. A request that breaks its contract does not
  feed it.

### 4.6 C4: `message_comms` ↔ `policy` (`c4_policy.py`)

`PolicyConfigRequest` (`c4.config/v1`) and `Reset` (`c4.reset/v1`): no fields. `Reset` is answered with `Ack`.

`PolicyConfig`, schema `c4.config.reply/v1`:

| Field | Type | Meaning |
|---|---|---|
| `obs_steps` | int, > 0 | `OBS_STEPS` (2) |
| `action_steps` | int, > 0 | `ACTION_STEPS` (16) |
| `step_ms` | int, > 0 | `STEP_MS` (50) |
| `image_size` | int, > 0 | `IMAGE_SIZE` (224) |
| `inputs` | list of str, declared `str_list(choices=<the Observation field names>)` | which inputs the model uses |
| `uses_prompt` | bool | false for a plain diffusion policy |

**The sizes are defined once, in `contracts/contracts/sizes.py`:** `OBS_STEPS = 2`, `ACTION_STEPS = 16`,
`STEP_MS = 50`, `IMAGE_SIZE = 224`, exported from `contracts`. The C1 and C4 shapes in this file are written with
them, and `policy` and `message_comms` **import them** rather than writing the numbers again. They are fixed by this
contract version (changing one is a version bump, section 7), so they are constants, not config. The validator checks
`PolicyConfig`'s types; `message_comms` compares its four values with the constants and concludes if they differ.

`Prompt`, schema `c4.prompt/v1`, answered with `Ack`: `task_name` str, `prompt_index` int (≥ 0).

`Observation`, schema `c4.infer/v1`. Relative to the anchor (the hand pose at the newest main image time), oldest
step first:

| Field | Type / shape | Meaning |
|---|---|---|
| `eef_pos` | float32 (`OBS_STEPS`, 3) | position relative to the anchor, m; last row always 0 |
| `eef_rot6d` | float32 (`OBS_STEPS`, 6) | orientation relative to the anchor as rot6d (first two rows of the rotation matrix); last row always `[1,0,0,0,1,0]` |
| `eef_rot6d_wrt_start` | float32 (`OBS_STEPS`, 6) | orientation relative to the episode start pose |
| `gripper_width_m` | float32 (`OBS_STEPS`, 1) | absolute opening, m |
| `main` | uint8 (`OBS_STEPS`, `IMAGE_SIZE`, `IMAGE_SIZE`, 3) | straight from C1 |
| `ultrawide` | uint8 (`OBS_STEPS`, `IMAGE_SIZE`, `IMAGE_SIZE`, 3) | straight from C1 |

`check_values()`: finite; the last rows of `eef_pos` and `eef_rot6d` equal `0` and `[1,0,0,0,1,0]` within
`anchor_row_tol`; every rot6d row pair of unit length and perpendicular (same limits as the chunk); widths ≥ 0.

`ActionChunk`, schema `c4.chunk/v1`, the reply to `c4.infer/v1`:

| Field | Type / shape | Meaning |
|---|---|---|
| `eef_pos` | float32 (`ACTION_STEPS`, 3) = (16, 3) | step k's position relative to the anchor, m, `STEP_MS` apart |
| `eef_rot6d` | float32 (`ACTION_STEPS`, 6) | step k's orientation relative to the anchor |
| `gripper_width_m` | float32 (`ACTION_STEPS`, 1) | absolute opening, in `[0, gripper_width_max_m]` |
| `width_clipped` | uint64 | how many of this chunk's widths the policy clipped into range; always present, 0 when none |

`check_values()` (PLAN §4 "a valid chunk"): the shapes and types above, finite values, each rot6d's two 3-vectors of
length within `1 ± rot6d_norm_tol` and with `|cos angle| ≤ rot6d_dot_max`, and `0 ≤ width_clipped ≤ ACTION_STEPS`.
A chunk that fails this is **broken** and ends the run (PLAN §5).

**Step 0 near the anchor is not a contract rule.** Whether step 0 is within 2 cm / 5° of the anchor is checked by
`message_comms`, with its own limits (`odd_chunk_max_*`) in its `config.yaml`. A chunk that fails only that check is **odd, not broken**:
`message_comms` skips that step (the arm holds), counts it, and shows the count on the status line (PLAN §4, §5:
"slow").

**The validator stays strict on the width; the policy clips.** A model can predict a width slightly outside
`[0, gripper_width_max_m]`. That is normal operation (PLAN §5), so `policy` clips each width into that range *before* building the `ActionChunk`
(in `Policy.make_chunk`), reading the same `gripper_width_max_m` from `ContractLimits`, and puts the number of clipped
widths in the chunk's `width_clipped`. `message_comms` shows it on the status line. An unclipped chunk with a width out of range is therefore a policy bug, and the validator refuses it.
`franka_control` checks the width again in its gate.

If the policy fails, the server replies `ErrorReply` instead of `ActionChunk`.

### 4.7 Limits (`limits.yaml`, `limits.py`)

The numbers the validators use live in a data file, not in code (PLAN rule 5). They are part of the contract: changing
one changes what both sides accept, so it is versioned like a field (section 7).

| Key | Value | Used by |
|---|---|---|
| `rotation_tol` | 1.0e-6 | C2, C3: rotation part of a pose orthonormal within this |
| `anchor_row_tol` | 1.0e-5 | C4 `Observation`: last rows equal 0 / identity rot6d |
| `rot6d_norm_tol` | 0.1 | C4: rot6d vectors of length 1 ± this (from main-sim's `policy_contract.py`) |
| `rot6d_dot_max` | 0.1 | C4: rot6d vectors perpendicular within this |
| `gripper_width_max_m` | 0.08 | C4 `ActionChunk` (and `policy`'s clipping): max opening, the Franka Hand limit. `# TODO: measure the max fingertip opening with the printed fingers` (a comment in `limits.yaml`) |
| `max_waypoints` | 64 | C3 `SetWaypoints`: most waypoints in one request |

`ContractLimits.load() -> ContractLimits` (cached; a frozen dataclass with these fields). A missing or unknown key is an
error at import time, so a broken limits file fails every test.

## 5. Encoding (the wire format)

`codec.py`:

| Function | What it does |
|---|---|
| `encode(msg: Message) -> bytes` | `msg.to_dict()` → msgpack bytes, in the canonical form below |
| `decode(data: bytes, expect: type[M] \| tuple[type[Message], ...] \| None = None) -> Message` | msgpack → dict → finds the class by `schema` in the registry → `from_dict` (validates). With `expect`, a valid message of another type raises `ContractError(problem="unexpected_message")`. `ErrorReply` is always allowed through, so callers can raise `RemoteError`. |
| `pack_obj(obj) -> bytes`, `unpack_obj(data) -> Any` | the low-level packer and unpacker (dicts, lists, arrays). Used by the example builder to write broken messages that the classes would refuse to build. |
| `pack_array(a: np.ndarray) -> dict`, `unpack_array(d: dict) -> np.ndarray` | one array ↔ its map form |

**Canonical form** (this is what C++ must read and write; one way to encode each message, so encoded bytes can be
compared exactly):

- A message is a msgpack **map**. Keys are strings, in the order of the field tables, `schema` first.
- Integers in the smallest msgpack integer form (what both libraries do by default). `float64` fields as msgpack
  float64 (never float32). `bool` as msgpack bool. `str` as msgpack str (UTF-8). `bytes` as msgpack bin.
  `None` as `nil`. Lists as msgpack arrays.
- A **numpy array** is a map with exactly these four keys in this order:
  `{"__nd__": true, "dtype": "<name>", "shape": [d0, d1, ...], "data": <bin>}`.
  - `dtype` is one of `uint8`, `int64`, `uint64`, `float32`, `float64`.
  - `data` is the elements in C order (row-major), **little-endian**, `prod(shape) × itemsize` bytes.
  - `pack_array` makes a C-contiguous little-endian copy if needed; `unpack_array` checks the byte count and returns a
    read-only array that shares the received bytes (no copy; call `.copy()` to modify).
- Python: `msgpack.packb(d, use_bin_type=True)` and `msgpack.unpackb(b, raw=False, strict_map_key=True)`.

Decoding checks run in a fixed order, and the first failure raises. The order is fixed so each broken example has one
known expected error:

1. bytes are msgpack (`not_msgpack`), and a map (`not_a_map`);
2. `schema` present (`no_schema`), a known name (`unknown_schema`), the supported version (`schema_version`);
3. missing keys (`missing`), then unknown keys (`unknown_field`);
4. each field in table order: type (`type`), dtype (`dtype`), shape (`shape`), allowed values and ranges (`value`);
5. `check_values()` (`value`);
6. with `decode(..., expect=...)`: a valid message of a type the caller did not expect (`unexpected_message`).

The problem codes are exactly these: `not_msgpack`, `not_a_map`, `no_schema`, `unknown_schema`, `schema_version`,
`missing`, `unknown_field`, `type`, `dtype`, `shape`, `value`, `unexpected_message`.

## 6. Transport helpers

**Why in `contracts/`:** every Python process needs the same three behaviours: send a request once with a timeout,
answer requests by type, publish/subscribe with a timeout. Rule 3 says the only shared Python code is `contracts/`, so
either they live here or each process writes its own copy (and the copies drift: a resend in one of them would compute
an inference twice, as main-sim's `rmq_rpc.py` warns). They are small (about 50 lines each), they only move
`Message` objects, and they validate both directions, which is the contract's job anyway.

Rules for every helper:

- **Servers and publishers bind; clients and subscribers connect.** On C0 the Pi is the server and the publisher, so
  it binds.
- **Every timeout is passed in by the caller** (from its config). No hidden default timeouts.
- **A request is sent once and never resent.** On a timeout, the client closes its socket (with linger 0, so nothing
  is left queued) and opens a new one, then raises `TransportTimeout`. A late reply to the old socket is thrown away.
  (This is ZeroMQ's "lazy pirate" pattern, as in main-sim's `bridge/link.py`.)
- One ZeroMQ message = one msgpack blob = one `Message`, except a C0 packet: one multipart message = one `Packet`
  (its header is one `Message`, its other parts raw frame bytes).
- Sockets are not shared between threads: each helper is used by one thread.
- An endpoint `tcp://127.0.0.1:*` binds a free port; the helper's `.endpoint` gives the real address. Unit tests use
  this so they never collide with each other or with a running station.

`rpc.py`:

```
class RpcClient:
    def __init__(self, name: str, endpoint: str) -> None          # name: the server's process name, for messages
    def request(self, msg: Message, expect: type[M] | tuple[type[Message], ...], timeout_s: float,
                poll_s: float | None = None, cancel: Callable[[], bool] | None = None) -> M
    def close(self) -> None

class RpcServer:
    def __init__(self, endpoint: str, handlers: Mapping[type[Message], Callable[[Message], Message]]) -> None
    endpoint: str                                                  # the bound address
    def serve_once(self, timeout_s: float) -> bool                 # handle at most one request; False if none came
    def serve_forever(self, stop: threading.Event, poll_s: float) -> None
    def close(self) -> None
```

- `request` raises `TransportTimeout` (no reply in time), `TransportCancelled` (cancelled, below), `RemoteError`
  (the server sent `ErrorReply`) or `ContractError` (the reply broke its contract, or is not one of `expect`).
- **Waiting in slices, cancellable.** By default `request` waits for the reply in one piece, up to `timeout_s`. With
  `poll_s` and `cancel` (both given, or neither), it waits in slices of `poll_s` and calls `cancel()` after each
  slice. If `cancel()` returns true, the client closes the socket with linger 0, opens a new one (as after a timeout,
  so a late reply is thrown away) and raises `TransportCancelled`. `timeout_s` still applies in total. This is for
  long requests the operator may interrupt, e.g. `message_comms` waiting on `home` (up to 30 s) when Ctrl-C arrives. Message format, e.g.
  `policy (tcp://127.0.0.1:5640): c4.infer/v1: no reply within 5.000 s (sent once, not resent)`.
- `RpcServer` decodes the request, finds its handler by class, and sends the handler's reply. It always replies (a REP
  socket must): a request that breaks its contract, or has no handler, gets `ErrorReply` naming the field; a handler
  that raises gets `ErrorReply` with the exception text, and the exception is logged. The server itself keeps
  running: deciding to conclude is the client's job (PLAN §5).
- A handler must return a `Message`; the server does not re-validate it (it is valid by construction, section 3).

`pubsub.py`:

```
class Publisher:
    def __init__(self, endpoint: str, send_hwm: int) -> None
    endpoint: str
    def publish(self, msg: Message) -> None                        # never blocks; drops if the queue is full
    def close(self) -> None

class Subscriber:
    def __init__(self, name: str, endpoint: str, expect: type[M], recv_hwm: int) -> None
    def receive(self, timeout_s: float) -> M | None                # None on timeout; ContractError on a bad message
    def close(self) -> None
```

```
class PacketPublisher:                                             # C0 packets, one multipart message each
    def __init__(self, endpoint: str, send_hwm: int) -> None
    endpoint: str
    def publish(self, packet: Packet) -> None                      # never blocks; drops the whole packet if full
    def close(self) -> None

class PacketSubscriber:
    def __init__(self, name: str, endpoint: str, recv_hwm: int) -> None
    def receive(self, timeout_s: float) -> Packet | None           # None on timeout; ContractError on a bad packet
    def close(self) -> None
```

`send_hwm` / `recv_hwm` (ZeroMQ's "high-water mark": how many messages may wait in the queue before new ones are
dropped) come from the caller's config. Subscribers get every message (no "keep only the newest" option): C0 needs
every packet to see `packet_seq` / `frame_seq` gaps and keyframes, and C2 needs every state for its `seq` check.
`PacketSubscriber.receive` takes one multipart message and parses it with `Packet.from_parts`.

Errors (`errors.py`):

```
class ContractError(ValueError):
    schema: str      # e.g. "c2.state/v1" ("?" if unreadable)
    field: str       # dotted path, e.g. "O_T_TCP", "health.main.status", "rejected[2].reason"; "" if none
    problem: str     # one of the codes in section 5
    detail: str      # e.g. "shape (3, 4), expected (4, 4)"
    # str(e) == "c2.state/v1: field 'O_T_TCP': shape (3, 4), expected (4, 4)"

class TransportTimeout(TimeoutError):   process: str, endpoint: str, request: str, timeout_s: float
class TransportCancelled(RuntimeError): process: str, endpoint: str, request: str, waited_s: float
class RemoteError(RuntimeError):        process: str, reply: ErrorReply
```

Every one of these is raised to the caller; the helpers never retry or hide one. `message_comms` concludes on each of
them (PLAN §5): a `ContractError` or `RemoteError`, and every `TransportTimeout` (its request timeouts are its hang
limits, set in its config). `TransportCancelled` is not a failure: it only happens when the caller asked for it (its
own `cancel()`), and the caller goes on with what caused the cancel (e.g. the operator's Ctrl-C).

## 7. Versioning

- A schema is `<name>/v<N>`, e.g. `c3.set_waypoints/v1`. Every top-level message carries one.
- **Any** change to a message bumps its version: a field added, removed or renamed; a type, dtype, shape, unit, frame
  or meaning changed; a limit in `limits.yaml` that applies to it changed.
- Only one version of each message is supported at a time. Old versions are not kept: everything lives in one repo and
  is deployed together, so supporting two versions would only hide a half-done change.
- **On a mismatch** (a known name, another version): `decode` raises
  `ContractError(problem="schema_version")`, e.g. `c2.state/v2: version v2 not supported (this side knows v1)`. A
  server answers it with `ErrorReply`; a client raises. Either way this is PLAN §5's "a message that fails its
  contract": **conclude**, naming the message. The C++ side does the same (refuses, replies `ErrorReply`, and its tests
  check it with the `schema_version` broken examples).
- After a bump: update the class, run `python -m contracts.make_examples`, and update the C++ code until the
  `franka_control` contract tests pass again.

## 8. Example messages (C2/C3 files shared with C++; C0/C1/C4 in memory)

**Made by** `examples.py`: `build_examples() -> list[Example]` builds every example of every contract in code: good
ones from the message classes, broken ones by taking a good dict and changing one thing, packed with `pack_obj`. An
`Example` holds `name`, `schema`, `valid`, the expected `field` and `problem` for a broken one, a few `values` for a
good one, and the encoded `data` bytes.

**What is stored, what is not:**

- **C2 and C3 are committed as files**, because the C++ tests of `franka_control` need them. "C3" includes the
  replies C3 uses: `c3.set_waypoints.reply/v1`, `ack/v1` and `error/v1`. `python -m contracts.make_examples` writes
  exactly these to `contracts/examples/`. The files are never edited by hand; a test checks that the files in git equal
  what the builder makes, so a stale file fails the tests.
- **C0, C1 and C4 are not stored.** They are Python-only (no C++ side reads them), and the images would make them
  large. The Python tests call `build_examples()` and use them in memory, with the same good/broken checks.

**Where:** `contracts/examples/`, checked into git. `examples_dir() -> Path` finds it from the installed package
(`Path(contracts.__file__).parents[1] / "examples"`; the packages are installed in editable mode, see
deploy/STRUCTURE.md §7). `load_index() -> list[Example]` reads it back.

**Content:** each file holds the exact bytes that would travel on the wire, one message per file
(`c2/state_running.msgpack`). `index.yaml` describes them all:

```yaml
format: 1
examples:
  - file: c2/state_running.msgpack
    schema: c2.state/v1
    valid: true
    values: {seq: 42, mode: running, gripper_width_m: 0.05, loop_late: 3, gate.pulled_back: 1}   # scalars a decoder test can compare
  - file: c2/bad_pose_transposed.msgpack
    schema: c2.state/v1
    valid: false
    field: O_T_TCP
    problem: value
    note: column-major matrix sent without transposing
```

**Which examples** (stored or in memory): for every message, at least one good one; and for each contract the broken
ones PLAN §7 asks for: wrong shape, wrong dtype, a value out of range (or a wrong unit, e.g. width in mm), a missing
field, an unknown field, a wrong schema version. Plus, per contract: C0 good packet headers with one, zero (a pair
from an earlier packet) and two ultrawide frames, with no pair (−1) and in JPEG, PI_PAYLOAD.md §8's example among them,
and broken ones with swapped sensors, a `pair_seq` below −1 or newer than the packet's newest ultrawide frame,
ultrawide frames out of capture order, two encodings in one camera, a negative or too big `uint32` size, a JPEG
non-keyframe, and an unknown frame-map key (the multipart part checks have no single-blob example: `test_c0.py` checks
them); C1 a reply with one camera `stale` (no images at
all) and a broken one with images while a camera is `down`; C2 good states in `running`, `holding` and `homing`
with non-zero `loop_late` and `gate.pulled_back`, and broken ones with the transposed pose, an unknown `mode`, a missing
`loop_late` and a missing `gate.pulled_back`; C3 non-increasing times, mismatched lengths, a non-rigid pose,
`shortened > accepted`, and the `error/v1` reply to a `set_waypoints` refused because of the state; C4 a chunk with a NaN, a non-unit rot6d, non-perpendicular rot6d rows, a wrong shape, and `width_clipped`
missing or greater than `ACTION_STEPS`; plus a **good** chunk whose step 0 is 5 cm from the anchor (an odd chunk is
still valid for the contract); `error/v1` and `ack/v1` replies. The stored C2/C3 files are small (no images).

**How C++ uses them** (no Python involved):

- Library proposal: **msgpack-cxx 6.1.1** (the C++ half of msgpack-c; header-only), built with `MSGPACK_NO_BOOST`
  so it doesn't need Boost. ZeroMQ through **libzmq 4.3** (already on this machine) and **cppzmq 4.10.0**. The
  `index.yaml` is read with **yaml-cpp 0.8.0**, which `franka_control` needs anyway for its `config.yaml` and
  `station.yaml`. All fetched at fixed versions by CMake; the final choice is `franka_control/STRUCTURE.md`'s.
- `franka_control`'s CMake gets the folder as a compile definition, `CONTRACTS_EXAMPLES_DIR`, defaulting to
  `${CMAKE_CURRENT_SOURCE_DIR}/../contracts/examples`.
- Its contract tests, for the C2, C3, `ack/v1` and `error/v1` examples in `index.yaml`:
  1. **Every good C3 request** decodes; its `values` match.
  2. **Every broken C3 request** is refused by the C++ decoder, naming the same `field` (and the same `problem`
     code).
  3. **Round trip, byte for byte**: decode each good C2 / C3-reply / `ack` / `error` example into the C++ struct,
     encode it again, and compare with the file's bytes. Equal bytes prove the C++ encoder writes exactly the
     canonical form the Python decoder expects.
  4. Every good C2 example decodes too, so `get_state` replies and broadcasts are the same struct.
- Two limits apply to messages C++ decodes: `max_waypoints` and `rotation_tol` (C3 `set_waypoints`). The C++ code
  reads them from `contracts/contracts/limits.yaml` (path given to CMake the same way, `CONTRACTS_LIMITS_FILE`), so
  the numbers are written in one place only.

## 9. Use by each process (summary)

| Process | Sends | Receives |
|---|---|---|
| `process_frames` | `TimeSyncRequest` (RpcClient to the Pi); `FramesReply`, `HealthReply` (RpcServer) | `Packet` (PacketSubscriber); `TimeSyncReply` |
| `policy` | `PolicyConfig`, `Ack`, `ActionChunk`, `ErrorReply` (RpcServer) | `PolicyConfigRequest`, `Reset`, `Observation`, `Prompt` |
| `message_comms` | C1/C3/C4 requests (three RpcClients) | `ArmState` (Subscriber) and every reply |
| `franka_control` (C++) | `ArmState` (PUB), C3 replies | C3 requests; its own C++ codec, proven by the example files |
| fake Pi (in `process_frames`) | `Packet` (PacketPublisher), `TimeSyncReply` (RpcServer) | `TimeSyncRequest` |
| `deploy` | nothing: no `launch.py` sends a contract message. Each process checks its own peers itself at start-up (e.g. `process_frames` its `time_sync` to the Pi) and reports a failure with its FATAL line | nothing |

## 10. Unit tests

All under `contracts/tests/`; run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest contracts/tests` (see deploy/STRUCTURE.md §7).

| File | Checks |
|---|---|
| `test_schema.py` | parsing `name/vN`; refusing malformed strings; unknown name vs version mismatch give different `problem` codes and readable text |
| `test_fields.py` | each field helper: right type passes; wrong type, dtype, shape, choice, range each fail with the right `problem`; `optional` accepts `None` but not a missing key; `str_list` refuses a non-string item and an item outside `choices`; `bool` is not accepted as an int |
| `test_codec.py` | array round trip for every allowed dtype and for 0-d / 1-d / 4-d shapes; non-contiguous and big-endian inputs are normalised; byte count mismatch refused; decoded arrays are read-only; `encode(decode(b)) == b` for every good example, stored and in memory (canonical form); not-msgpack and not-a-map inputs |
| `test_message.py` | building validates; `to_dict` order is the field order with `schema` first; `from_dict` refuses missing and unknown keys; frozen; a float64 array is not cast to float32 |
| `test_registry.py` | every schema maps to exactly one class; duplicate registration is refused; `decode(expect=...)` refuses another valid type but lets `ErrorReply` through |
| `test_limits.py` | `limits.yaml` loads; a missing or unknown key is refused |
| `test_c0.py`, `test_c1.py`, `test_c2.py`, `test_c3.py`, `test_c4.py`, `test_common.py` | each message's own rules: enums, JPEG keyframe, C0 sensors per role, `uint32` sizes and `uint64` counters, ultrawide capture order and one encoding, `pair_seq` −1 or not newer than the packet's newest ultrawide frame, multipart round trip with 0/1/2 ultrawide frames, a wrong part count, an empty part and a broken header refused with their field, C1 images only when both cameras are ok (none at all otherwise), rigid-pose check (incl. a transposed matrix), increasing times, length agreement, `max_waypoints`, C2 `homing` mode, `loop_late`, `gate.pulled_back` and its bounds, C3 reply `shortened ≤ accepted` and `skipped`, the `sizes.py` constants used in every C1/C4 shape, anchor rows, rot6d norm/perpendicular, an odd step 0 accepted (not a contract rule), width range (a width of `gripper_width_max_m` + 1 mm is refused: the validator does not clip), float array checks in float64 (with a 0.07 m maximum, a float32 width of 0.07, which is 0.0700000003, is refused), `width_clipped` in `[0, ACTION_STEPS]`, NaN/inf |
| `test_examples.py` | for every example from `build_examples()` (all five contracts): good ones decode and their `values` match, broken ones fail with exactly their `field` and `problem`. For the stored files: every file in `examples/` is listed in `index.yaml` and vice versa; only C2/C3 (and their `ack`/`error` replies) are stored; the folder equals a fresh `make_examples` output (no stale files) |
| `test_rpc.py` | request/reply over `tcp://127.0.0.1:*`; with `poll_s` and `cancel`, a reply that arrives normally is returned, a `cancel()` turning true raises `TransportCancelled` within about one `poll_s`, the next request on the same client works (new socket) and the late reply is dropped, `timeout_s` still raises `TransportTimeout` when `cancel()` stays false, and giving only one of `poll_s`/`cancel` is refused; a timeout raises `TransportTimeout` after about the timeout, the request was sent once, and the next request works (new socket); a late reply is dropped; a broken request gets `ErrorReply` naming the field; an unknown request type gets `ErrorReply`; a raising handler gets `ErrorReply` and the server keeps serving; a wrong reply type raises `ContractError`; `RemoteError` carries the text |
| `test_pubsub.py` | messages arrive in order; `receive` returns `None` after the timeout; a broken message raises `ContractError`; a message of another type raises; C0 packets arrive whole and in order with 0, 1 or 2 ultrawide frames, a packet one part short raises and the subscriber keeps working, publishing never blocks |
