# process_frames/

The design of the frames process. Written from PLAN.md §6.1 (its plan), §4 (contracts C0 and C1), §5 (the failure
rule) and §9 (backups), following `contracts/PI_PAYLOAD.md` (the exact C0 contract), `contracts/STRUCTURE.md` and the
conventions of `deploy/STRUCTURE.md` §7. PLAN.md is
the source of truth: where this file and PLAN.md disagree, PLAN.md wins and this file is fixed.

> **2026-11-13: the Pi sends `pi_stream/v1`, not the C0 packets described below.** The default is now
> `stream: pi_v1`: 3-part JSON + JPEG messages, both cameras at 640×480, and no time sync (PI_PAYLOAD.md §0).
> §5.4a and §5.6a describe the two pieces that differ, the packet source and the one-way clock. Everything after
> the source (receiver, buffer, frame picking, C1) is unchanged. The C0 path below stays available as `stream: c0`.

## 1. Role

A process that receives the live camera packets from the Pi 5 and turns them into the policy's 224×224 images.

It has two sides, running at the same time:

- **Receiving side:** always on, paced by the Pi. It listens to the Pi directly (contract C0,
  [contracts/PI_PAYLOAD.md](../contracts/PI_PAYLOAD.md)); packets never pass through `message_comms/`.
- **Serving side:** answers `message_comms/` only when asked (contract C1).

### Owns

- **Receiving side:**
  - receiving every C0 packet from the Pi over Wi-Fi: one per IMX519 (main) frame, 30 per second, each carrying that
    main frame, every IMX219 160° (ultrawide) frame captured since the previous packet, and which ultrawide frame is
    the packet's pair;
  - checking each packet against the contract and the rig file;
  - decoding every frame of both cameras, in order (H.264 by default, JPEG as the backup);
  - clock sync with the Pi, and the packet time: the main frame's mid-exposure time on the PC clock;
  - keeping the last ~1 s of packets, and the decoded ultrawide frames of the last ~1 s (to find a pair sent in an
    earlier packet);
  - tracking each camera's health (`ok` / `stale` / `down`).
- **Serving side:**
  - picking the packets the policy needs with iPhUMI's rule: main images from the packets nearest T − 50 ms and T,
    ultrawide images from the packets nearest T − 100 ms and T, T being the newest packet's time;
  - shrinking them to 224×224 with the exact image-prep code that made the training images.
- **The rig file** (camera sensors, resolutions, frame rates, lens models): `rigs/rig_pi_v2.yaml`, the real one (main
  1600×1200 @ 30, ultrawide 1280×960 @ 30), the only rig file in the folder. It is kept verbatim apart from two
  stated changes (§3); its `tcp:` block is written by `deploy`'s geometry tool, never by hand.
- **The Pi's pairing rule** as code (`PacketPairer`), so the fake Pi and the tests follow exactly the rule the Pi
  must.
- **The fake Pi** (`FakePiSender`), a stand-in for the Pi used only by the unit tests and `--mock`: a thread inside
  this process that sends real C0 packets over a local port. Nothing in this process replays recordings on the real
  arm; with the real Pi, every packet comes live from the Pi.

### Doesn't own

- Anything about the robot or the policy (poses, state, waypoints).
- Deciding whether the episode goes on: it **reports** health; `message_comms` decides "slow" vs "conclude" (§9).
- The Pi's sending side (PLAN §10 open item). What it must send is
  [contracts/PI_PAYLOAD.md](../contracts/PI_PAYLOAD.md); §14 sums up what this design relies on.

### Talks to

- The Pi 5 (C0): subscribes to its packet stream; sends `time_sync` requests about once a second.
- `message_comms/` (C1): on request, returns the selected 224×224 images, the PC time each one was taken, and health.
- `deploy/`: through `launch.py` (checks, ready file) and the common arguments.

### Terms used below

- **Packet:** one C0 multipart message: a header, the main frame, and the ultrawide frames captured since the
  previous packet (PI_PAYLOAD.md §3). One per main frame.
- **Pair:** the ultrawide frame the Pi names in a packet's `pair_seq`: the newest one captured not later than the main
  frame + 5 ms, at most 50 ms older (PI_PAYLOAD.md §5). It may have come in an earlier packet. −1 means none.
- **Packet time:** the main frame's mid-exposure time, mapped to the PC clock. Both of a packet's images are served
  with this time (as in iPhUMI, where a packet's images share one timestamp).
- **Ring buffer:** a list of the most recent packets; when a new one comes in, those older than `buffer.keep_s` are
  dropped from the front.
- **Keyframe:** an H.264 frame that can be decoded on its own (an IDR frame). The frames between keyframes only store
  differences to earlier frames, so after a lost frame they can't be decoded correctly until the next keyframe.
- **Mid-exposure time:** the middle of the time the sensor was collecting light for a frame. Training pairs every image
  with the pose at this time, so live frames must be stamped the same way.
- **Clock offset / drift:** the Pi's clock and the PC's clock count from different starting points (offset) and tick
  at very slightly different speeds (drift, in parts per million, ppm: 20 ppm = 20 µs per second).
- **Round trip (RTT):** the time from sending a `time_sync` request to getting its reply.
- **Anchor:** the time all four picked images are measured back from: the newest packet's time T (§6.2).
- **Bit-exact:** the same bytes, not just similar images.

## 2. File tree

```
process_frames/
  STRUCTURE.md
  launch.py                    ProcessFramesProcess (section 11); not part of the package
  config.yaml                  every setting, with its default (section 10)
  pyproject.toml               package "process_frames"; pinned dependencies (section 15)
  rigs/
    rig_pi_v2.yaml             the rig file, the user's file verbatim with two stated changes (section 3)
  process_frames/              the Python package
    __init__.py                FOLDER (the process_frames/ folder, for config.yaml and rigs/), NAME = "process_frames"
    __main__.py                `python -m process_frames`: common arguments, logging, FATAL line, exit code (section 12)
    app.py                     FramesApp: builds everything, runs the threads, ready file, stats, shutdown
    settings.py                Settings: config.yaml + the station's process_frames section + endpoints
    errors.py                  FatalError, DecodeError, SettingsError
    rig_file.py                load_live_rig(), rig_summary(); `python -m process_frames.rig_file FILE --json`
    packet_source.py           PacketSource (interface)
    pi_packet_source.py        PiPacketSource: C0 packet subscriber (the real Pi, or the fake Pi)
    recording_packet_source.py RecordingPacketSource: test stand-in for the Pi: C0 packets from memory, no network
    frame_decoder.py           FrameDecoder (interface), make_decoder()
    h264_decoder.py            H264Decoder (PyAV; the default)
    jpeg_decoder.py            JpegDecoder (OpenCV; the backup)
    clock_map.py               ClockMap, ClockSample, ClockStats   ADAPTED from main-sim clock_map.py
    clock_sync.py              ClockSync: the time_sync exchanges
    ultrawide_store.py         UltrawideStore, DecodedUltrawide: decoded ultrawide frames by frame_seq
    packet_buffer.py           PacketBuffer, BufferedPacket
    packet_selector.py         PacketSelector, Selection           ADAPTED from iPhUMI's get_downsampled_camera_images
    camera_health.py           CameraHealth
    receiver.py                Receiver, CameraCounters, PacketCounters: the receiving side
    frames_server.py           FramesServer: the serving side (C1)
    pi_recording.py            PiRecording, RecordedCamera, CaptureEvent, H264Encoder, JpegEncoder: a recorded Pi
                               episode as C0 frames and packets (for the two stand-ins)
    packet_pairer.py           PacketPairer: the Pi's pairing rule and "every ultrawide frame once"
    fake_pi_sender.py          FakePiSender: stand-in Pi for tests and --mock
    fake_pi.py                 `python -m process_frames.fake_pi`: run the fake Pi alone
    probe.py                   `python -m process_frames.probe`: ask a running process_frames once, save the images
    _dataproc/                 COPIED unchanged from rnoro/data-work (section 3); never edited by hand
      __init__.py              new, one docstring line: "copied from rnoro/data-work; see STRUCTURE.md §3"
      imageprep.py             ImagePrep, MainPrep, UltrawidePrep, letterbox  (PLAN's ImagePrep)
      config.py                RigConfig, load_rig, ConfigError              (PLAN's RigConfig)
      paths.py                 NAME_RE only
      geometry/__init__.py
      geometry/camera_model.py camera models and the fisheye -> virtual pinhole remap table
      geometry/transforms.py   only needed by the copied camera-model test
      _vendor/__init__.py      new, one docstring line
      _vendor/cv_util.py       iPhUMI's letterbox: the reference the tests compare against
  tests/
    conftest.py                fixtures and test-only fakes (section 13.1)
    reference_umi_utils.py     COPIED: iPhUMI's get_downsampled_camera_images, the reference for PacketSelector
    data/
      rig_v1.yaml              COPIED unchanged (used by the copied test_imageprep.py)
      rig_small.yaml           new: tiny resolutions (main 160x90, ultrawide 128x96) for fast pipeline tests
    test_*.py                  section 13
```

One class per file, named after the class, except small value types that live with the class that uses them
(`BufferedPacket`, `DecodedUltrawide`, `Selection`, `ClockSample`, `ClockStats`, `RecordedCamera`, `CaptureEvent`,
`CameraCounters`, `PacketCounters`), the two small encoders the recording needs (`H264Encoder`, `JpegEncoder`, in
`pi_recording.py`), and the copied `_dataproc/` files, which keep their own layout so they stay byte-identical to their
source.

`_dataproc/` uses relative imports (`from .config import RigConfig`) because that is how the source is written; it is
the one exception to "absolute imports only" (deploy/STRUCTURE.md §7.2), allowed so the copies stay unchanged.

## 3. Copied and adapted code

Every copied file starts with the provenance header of deploy/STRUCTURE.md §7.6 (`# from <branch>:<path> @ <commit>`,
`# changes: ...`). The commit is the one `git log -1 --format=%h <branch> -- <path>` gives at copy time; the values
below are today's, and the executor re-runs the command.

**Copied (images must match training: PLAN §6.1 "copied unchanged"):**

| File here | Source (`rnoro/data-work:data-processing/...`) | Commit | Changes |
|---|---|---|---|
| `_dataproc/imageprep.py` | `dataproc/imageprep.py` | `ce9ba42` | none |
| `_dataproc/config.py` | `dataproc/config.py` | `5330902` | one import line: `from dataproc.paths import NAME_RE` → `from .paths import NAME_RE` |
| `_dataproc/paths.py` | `dataproc/paths.py` | `a1ddbbc` | only the `NAME_RE` line (and its imports) kept |
| `_dataproc/geometry/__init__.py` | `dataproc/geometry/__init__.py` | `3fb0ead` | none |
| `_dataproc/geometry/camera_model.py` | `dataproc/geometry/camera_model.py` | `3fb0ead` | none |
| `_dataproc/geometry/transforms.py` | `dataproc/geometry/transforms.py` | `3fb0ead` | none |
| `_dataproc/_vendor/cv_util.py` | `dataproc/_vendor/cv_util.py` | `ce9ba42` | none |
| `tests/data/rig_v1.yaml` | `configs/rigs/rig_v1.yaml` | `5330902` | none |
| `tests/test_imageprep.py` | `tests/test_imageprep.py` | `2f6d3b1` | imports `dataproc.*` → `process_frames._dataproc.*`; `RIG_PATH` → `tests/data/rig_v1.yaml`; the iPhUMI path is `parents[3]`; the vendored-file comparison skips 3 header lines instead of 1 |
| `tests/test_vendored_camera_model.py` | `tests/test_vendored_camera_model.py` | `3fb0ead` | imports `dataproc.geometry.*` → `process_frames._dataproc.geometry.*` |

**The rig file** `rigs/rig_pi_v2.yaml` is the user's file exactly as given (added in commit `196eae4`, "verbatim"),
with two stated changes:

1. `gripper.max_opening_m: 0.110` → `0.080`, with a comment saying why (the Franka Hand's 80 mm stroke, PLAN §1);
2. the `tcp:` block. The given nominal CAD value (15° about x, `[0, 0.060, 0.200]` m, from the iPhone era) is replaced
   by the `T_cam_tcp` that `python -m deploy.geometry` computes from `deployment/gripper.yaml` and franka_control's
   `tcp.F_T_TCP` (deploy/STRUCTURE.md §6.6): a 4×4 matrix, translation `[0, 0.016891, 0.149465]` m, under the tool's
   two-line `COMPUTED` header comment. It changes only when `gripper.yaml` or `F_T_TCP` changes and the tool is rerun.

It has no provenance header (the user's header comments are kept as they are). `test_copies.py` checks the file against
`196eae4` with exactly those changes, the matrix numerically (within 1e-9) against the computed value.

**Copied from this branch** (`rnoro/deployment`), as the reference for a test:

| File here | Source | Commit | Changes |
|---|---|---|---|
| `tests/reference_umi_utils.py` | `behavior_prompting/deps/real-env/real_env/utils/umi_utils.py` | `d0298b9` | only `get_downsampled_camera_images` kept, unchanged, with the two imports it uses (`typing.Any`, `numpy`); a test checks the function text equals the source's |

**Adapted (new code following a reference; header says `# adapted from ...` and what differs):**

| File here | Reference | What is kept / what differs |
|---|---|---|
| `clock_map.py` | `rnoro/main-sim:controlSW/franka_deploy/franka_deploy/peripherals/clock_map.py` @ `935c0b3` | Only the idea of the `sender_stamp_ping` mode: use the shortest-round-trip exchanges of a window, with a max-drift fallback. Differs: a plain least-squares line instead of main-sim's Theil–Sen fit (simpler; the round-trip filter already removes the Wi-Fi outliers), integer ns, one Pi stamp per reply (C0's `time_sync` reply has one `t_pi_ns`), a minimum span before drift is used, a freshness limit, a lock (two threads use it), all numbers from config. The arrival-time modes are dropped. |
| `packet_selector.py` | `behavior_prompting/deps/real-env/real_env/utils/umi_utils.py` @ `d0298b9`, `get_downsampled_camera_images` | Kept: only the newest `max(frame_num)` packets, anchor on the newest, step back by `1 / target_freq`, nearest packet (first on ties), refuse beyond 2 frame periods. Differs: one call over packets gives both cameras, integer ns, spacings and the tolerance in frame periods from config, a miss returns a reason instead of raising, and a chosen packet without a usable image for that camera, or the same packet for both targets, is a miss too (§6.2). |
| `h264_decoder.py`, `pi_recording.py` | `rnoro/data-work:data-processing/dataproc/raw_episode.py` @ `96ff703`, `read_video_frames` | The same PyAV decode and the same `frame.to_ndarray(format="rgb24")` conversion. `H264Decoder` decodes one C0 frame at a time instead of a file. |
| `pi_recording.py` | `rnoro/data-work:data-processing/docs/PI_ADAPTER.md` and `dataproc/pi_format.py` @ `aac065c` | The Pi's file layout (`meta.json` camera list with `bitrate`, `<stem>.mp4`, `<stem>_frames.csv` with `frame_index,t_ns` and optionally `exposure_ns` and `in_episode` (the real Pi writes `in_episode`; it is ignored), roles by sensor model: `imx519` = main, `imx219` = ultrawide). Not copied: `pi_format.py` writes the format and depends on `RawEpisode`; we only read it. |

`tests/test_copies.py` (§13) checks that each "copied" file equals its source at that commit, apart from the header and
the listed changes, and `rig_pi_v2.yaml` as above, so a drifted copy fails the tests.

## 4. Threads and how they share data

```
 thread "receiver"    PiPacketSource (SUB socket, multipart) -> checks -> decoders (every frame, in order)
                      -> UltrawideStore -> ClockMap.to_pc_ns -> PacketBuffer.push, CameraHealth
 thread "clock-sync"  RpcClient (REQ socket to the Pi) -> ClockMap.add_sample              about once a second
 main thread          FramesServer: RpcServer (REP socket, C1) -> PacketBuffer.snapshot -> PacketSelector -> ImagePrep
                      + the ready file, the stats line every 10 s, watching for a FATAL from the other threads
 --mock only:         FakePiSender: thread "fake-pi-frames" (PUB socket), thread "fake-pi-time" (REP socket)
```

**Rules that keep this safe and simple:**

1. **Each socket is created and used by exactly one thread** (contracts/STRUCTURE.md §6). Classes open their sockets
   inside the thread that uses them (in `run()` or on first use), never in the constructor. The one exception is the
   C1 `RpcServer`, which the main thread binds at start-up and also uses, so a busy port fails at once.
2. **Three shared objects, each with its own lock:** `PacketBuffer` (one), `CameraHealth` (one per camera),
   `ClockMap` (one). A lock is held only for a few list or number operations, **never** during decoding, image prep or
   network I/O. No code ever holds two of these locks at once, so they can't deadlock. The `UltrawideStore` belongs to
   the receiver thread alone and has no lock.
3. **Packets are immutable once buffered.** A `BufferedPacket` is a frozen dataclass and its RGB arrays are marked
   read-only (`flags.writeable = False`) before it is pushed. `PacketBuffer.snapshot()` copies only the list of
   references (about 30) under the lock; the serving side then picks and preps outside the lock. The receiver can keep
   pushing meanwhile: the snapshot is unaffected, and a packet dropped from the buffer stays alive as long as the
   snapshot uses it. A paired ultrawide image is shared by reference between the packets that name it.
4. **One stop flag** (`threading.Event`) for all threads. A thread that hits an unrecoverable problem stores the first
   `FatalError` in `FramesApp` and sets the flag; the main thread then shuts everything down and prints the FATAL
   line (§12). Every blocking call has a timeout (`receive.timeout_s`, `clock.timeout_s`, `server.poll_s`), so each
   thread sees the flag within a fraction of a second.

## 5. Classes

Signatures only. Every class that uses time takes `clock: Callable[[], int]` (returns PC monotonic ns, default
`time.monotonic_ns`) so tests can drive time. The contract classes (`Packet`, `PacketHeader`, `FrameInfo`,
`EncodedFrame`, `TimeSyncRequest`, `TimeSyncReply`, `GetFramesRequest`, `FramesReply`, `GetHealthRequest`,
`HealthReply`, `HealthReport`, `CameraHealthReport`, `OBS_STEPS`) and the transport helpers (`PacketSubscriber`,
`PacketPublisher`, `RpcClient`, `RpcServer`) come from `contracts`.

### 5.1 Errors (`errors.py`)

```
class FatalError(RuntimeError):     # this process's own unrecoverable problem: exit 1 with a FATAL line
    what: str                       # e.g. "main stream sensor is imx519, rig file says imx219"
    check: str                      # e.g. "check which camera is on which Pi port, or process_frames.rig_file"
    # str(e) == f"{what}: {check}"; __main__ prints f"FATAL process_frames: {e}"

class DecodeError(Exception):       camera: str, seq: int, detail: str     # raised by a FrameDecoder (seq: frame_seq)
class SettingsError(ValueError):    problems: list[str]                    # every bad key, each with its path
```

`ConfigError` (a bad rig file) comes from the copied `_dataproc/config.py`.

### 5.2 `Settings` (`settings.py`)

Loads `process_frames/config.yaml`, merges the station's `process_frames:` section over it with the rule of
deploy/STRUCTURE.md §6.3 (maps merge key by key, other values are replaced, **an override key that isn't in
`config.yaml` is an error**), checks every type and range, and reads the station's shared keys it needs.

```
@dataclass(frozen=True)
class Settings:
    rig_file: Path                   # absolute
    decoder: str                     # "h264" | "jpeg"
    mock: bool
    mock_episode: Path | None        # required (and must exist) when mock
    pi_frames: str                   # endpoints.pi_frames, or endpoints.mock_pi_frames when mock
    pi_time_sync: str                # endpoints.pi_time_sync, or endpoints.mock_pi_time_sync when mock
    frames: str                      # endpoints.frames (C1, this process binds it)
    session_root: Path
    default_exposure_ns: int
    receive: ReceiveSettings; buffer: BufferSettings; select: SelectSettings; health: HealthSettings
    clock: ClockSettings; h264: H264Settings; server: ServerSettings; log: LogSettings
    mock_settings: MockSettings; check: CheckSettings; probe: ProbeSettings   # one small frozen dataclass per section

    @classmethod
    def load(cls, station_path: Path, mock: bool) -> "Settings"    # raises SettingsError listing every problem
```

Relative paths: `config.yaml` defaults are taken from the `process_frames/` folder; station overrides from the station
file's folder (deploy/STRUCTURE.md §6.2). With `mock`, the `pi_*` fields hold the mock endpoints, so the rest of the
code connects the same way to the real or the fake Pi.

### 5.3 Rig file (`rig_file.py`, plus the copied `_dataproc/config.py`)

PLAN's `RigConfig` is the copied dataproc class (`_dataproc/config.py`: `RigConfig`, `load_rig`, `ConfigError`). It
already checks every key, type and range and rejects unknown keys, a wrong `schema`, a `rig_id` that differs from the
file name, an ultrawide that isn't a fisheye, and a principal point outside the image. `rig_file.py` adds what live
use needs, without touching the copy:

```
def load_live_rig(path: Path) -> RigConfig      # load_rig(path), then: both cameras' sensor is one C0 allows
                                                #   (imx519 / imx219). Raises ConfigError naming the file and key.
def rig_summary(rig: RigConfig) -> str          # "rig_pi_v2: main imx519 1600x1200@30, ultrawide imx219 1280x960@30"
def main(argv: Sequence[str] | None = None) -> int
    # python -m process_frames.rig_file FILE [--json]
    # prints the rig summary (or, with --json, one line:
    #   {"ok": true, "rig_id": ...,
    #    "cameras": {"main": {"sensor": "imx519", "width": 1600, "height": 1200, "fps": 30.0}, "ultrawide": {...}}}
    #   or {"ok": false, "error": "<ConfigError text>"}); exit 0 if valid, else 1
```

`launch.py` runs this module in a subprocess (it may not import the package), so the rig is parsed by exactly one
loader everywhere.

### 5.4 `PacketSource` (interface, `packet_source.py`) and its two implementations

```
class PacketSource(ABC):
    @abstractmethod
    def next_packet(self, timeout_s: float) -> Packet | None   # the next C0 packet, or None if none within timeout_s
                                                                # (RecordingPacketSource: None at the end).
                                                                # ContractError if a packet breaks C0.
    @abstractmethod
    def close(self) -> None

class PiPacketSource(PacketSource):                             # pi_packet_source.py
    def __init__(self, endpoint: str, recv_hwm: int) -> None    # opens nothing yet
    # next_packet: on first call, opens a contracts PacketSubscriber(name="pi", endpoint, recv_hwm) in the calling
    #              thread; then PacketSubscriber.receive(timeout_s) (one multipart message -> Packet.from_parts)

class RecordingPacketSource(PacketSource):                      # recording_packet_source.py; test stand-in only
    def __init__(self, packets: Iterable[Packet]) -> None       # given order
    @classmethod
    def from_recording(cls, recording: PiRecording) -> "RecordingPacketSource"   # recording.packets(): one pass,
                                                                # paired by PacketPairer, the recorded times
    exhausted: bool
```

`PiPacketSource` is the source `FramesApp` uses with `stream: c0`: on the real Pi, and under `--mock` on
`FakePiSender` (it connects to the mock endpoints over `127.0.0.1` exactly as to the Pi). With `stream: pi_v1` it uses
`PiStreamPacketSource` instead (§5.4a). `RecordingPacketSource` is a stand-in for the
Pi in unit tests only (the receiver's logic without sockets); it is never selectable from config or `station.yaml`. A
change to C0 stays inside `PacketSource`, `PacketPairer` and the contract (PLAN §6.1).

### 5.4a `PiStreamPacketSource` (`pi_stream_source.py`, `stream: pi_v1`)

- **What it does:** a SUB socket on `endpoints.pi_frames`. Each 3-part message is checked by
  `contracts.PiStreamPacket` (a `ContractError` means `FATAL`), then converted by `PiStreamConverter` into a C0
  `Packet`, so the receiver doesn't change.
- **The ultrawide frame:** passed on only when its `frame_no` differs from the last one passed on. A repeat
  (`uw_reused`) gets an empty ultrawide list, with `pair_seq` naming the earlier frame, which is still in the
  `UltrawideStore`.
- **The clock:** each message's `t_send_ns` and its arrival time go to `OneWayClockMap.add_one_way` (§5.6a).
  `FramesApp` starts no `ClockSync` thread.
- **The fake Pi:** under `--mock`, `FakePiSender(stream="pi_v1")` sends the same messages. Each packet carries its
  main frame's pair, re-sent with `uw_reused` when repeated. There's no time-sync socket. `PiRecording.load(sizes=...)`
  delivers the recording at the rig file's sizes.

### 5.5 `FrameDecoder` (interface, `frame_decoder.py`) and its implementations

One decoder object **per camera** (H.264 keeps per-stream state). A frame is the contract's `EncodedFrame`: one
packet part (`data`) and its frame map (`info`).

```
class FrameDecoder(ABC):
    ENCODING: ClassVar[str]                                    # the C0 `encoding` it accepts: "h264" or "jpeg"
    @abstractmethod
    def decode(self, frame: EncodedFrame) -> np.ndarray | None # (height, width, 3) uint8 RGB;
                                                               # None = unusable (after a gap, before the next keyframe);
                                                               # DecodeError if the data can't be decoded
    def notify_gap(self) -> None                               # a frame was lost before this one. Default: nothing.

def make_decoder(kind: str, camera: str, h264: H264Settings) -> FrameDecoder     # "h264" | "jpeg"

class H264Decoder(FrameDecoder):                               # h264_decoder.py, the default
    ENCODING = "h264"
    def __init__(self, camera: str, thread_type: str) -> None  # "SLICE" or "NONE" (never FRAME: it delays output)

class JpegDecoder(FrameDecoder):                               # jpeg_decoder.py, the backup (PLAN §9)
    ENCODING = "jpeg"
    def __init__(self, camera: str) -> None
```

**`H264Decoder`.** Starts in "waiting for a keyframe". Non-keyframes while waiting return `None` without being fed to
the decoder. A keyframe while waiting creates a **fresh** PyAV codec context (`av.CodecContext.create("h264", "r")`,
FFmpeg's low-delay flag set, `thread_type` from config), so nothing from before the gap can leak into the new images;
then decoding resumes. Each frame must give exactly one picture, converted with `frame.to_ndarray(format="rgb24")`,
the same call dataproc's `read_video_frames` uses on the training videos. Zero pictures (the decoder held one back)
or more than one, a B picture (`frame.pict_type`: the stream has B-frames, which C0 forbids), or any
`av.error.FFmpegError`, raise `DecodeError`.
`notify_gap()` returns to "waiting for a keyframe".

Why the images match training: the same FFmpeg build (PyAV 17.1.0, the version dataproc pins) decodes the same kind of
stream (the Pi's encoder at the same settings as when recording, PI_PAYLOAD.md §5 rule 4) with the same RGB
conversion, and H.264 decoding gives the same output with or without threads. Live and recorded streams are separate
encodings of different moments, so "match" means the same decode and prep path, proven bit-exact in tests (§13).

**`JpegDecoder`.** `cv2.imdecode(..., IMREAD_COLOR)` then `cv2.cvtColor(BGR2RGB)`. A `None` result raises
`DecodeError`. `notify_gap()` does nothing: every JPEG frame stands alone (C0 requires `keyframe: true` for JPEG).
JPEG quality is the Pi's choice, not part of C0; at quality 90 the two streams need roughly 100–190 Mbit/s, so the Pi
may have to lower it if this backup is ever used over Wi-Fi.

**`RawDecoder`** (PLAN's test decoder) lives in `tests/conftest.py`, not the package: `data` is raw RGB bytes of
`height × width × 3`. It reports `ENCODING = "h264"`, copies H.264's keyframe rule (unusable after `notify_gap()`
until a `keyframe: true` frame), and records every `frame_seq` it is given, so the receiver's gap handling and
"every frame decoded exactly once, in order" can be tested without PyAV.

### 5.6 `ClockMap` (`clock_map.py`)

Pi time → PC time, from the `time_sync` samples. Pure arithmetic plus a lock; no sockets. Algorithm in §7.

```
@dataclass(frozen=True)
class ClockSample:
    t_send_ns: int         # PC monotonic, just before the request was sent
    t_pi_ns: int           # the Pi's CLOCK_BOOTTIME from the reply
    t_recv_ns: int         # PC monotonic, just after the reply arrived
    rtt_ns: int            # property: t_recv_ns - t_send_ns
    offset_ns: int         # property: t_pi_ns - (t_send_ns + t_recv_ns) // 2     (Pi minus PC)

@dataclass(frozen=True)
class ClockStats:
    synced: bool; samples: int; used: int; offset_ns: int | None; drift_ppm: float | None
    rtt_min_ms: float | None; last_sync_age_s: float | None

class NotSynced(RuntimeError): ...

class ClockMap:
    def __init__(self, window_s: float, best_fraction: float, drift_min_span_s: float, max_drift_ppm: float,
                 max_sync_age_s: float) -> None
    def add_sample(self, s: ClockSample) -> bool        # False: impossible sample (t_recv < t_send), ignored
    def is_fresh(self, now_ns: int) -> bool              # at least one sample, and the newest is <= max_sync_age_s old
    def offset_ns(self, at_pc_ns: int) -> int            # estimated (Pi - PC) at that PC time; NotSynced if no sample
    def to_pc_ns(self, t_pi_ns: int, at_pc_ns: int) -> int    # t_pi_ns - offset_ns(at_pc_ns)
    def stats(self, now_ns: int) -> ClockStats
```

### 5.6a `OneWayClockMap` (`clock_map.py`, `stream: pi_v1`)

A `ClockMap` fed with one-way stamps (`add_one_way(t_pi_send_ns, t_pc_arrival_ns)`), stored as samples where the
send and receive PC times are both the arrival time.

- **Each stamp gives** `offset − latency`. In each 1 s slice of the window, the best_fraction with the largest value
  (the least delayed) are fitted with the same line as `ClockMap`.
- **Result:** the offset is the true offset minus the minimum one-way latency, so PC times are late by a few ms on
  the Wi-Fi.
- **Cost:** it refits at most every 0.25 s; a full 60 s window costs about 2 ms.
- **Fresh** while packets arrive within `clock.max_sync_age_s`. `stats().rtt_min_ms` is `None`, and the stats line
  says "one-way".

### 5.7 `ClockSync` (`clock_sync.py`), thread "clock-sync"

```
class ClockSync:
    def __init__(self, endpoint: str, clock_map: ClockMap, settings: ClockSettings,
                 clock: Callable[[], int] = time.monotonic_ns) -> None
    def sync_once(self) -> bool          # one exchange: t_send = clock(); RpcClient.request(TimeSyncRequest(),
                                         #   TimeSyncReply, timeout_s); t_recv = clock(); add the sample.
                                         # False on TransportTimeout (counted, logged at most once per 10 s).
                                         # FatalError on ContractError or RemoteError (the Pi broke C0).
    def run(self, stop: threading.Event) -> None    # startup_exchanges back to back, then one every interval_s
    exchanges: int; timeouts: int
```

The `RpcClient` (contracts' lazy-pirate client: sent once, never resent; a new socket after a timeout) is opened on the
first `sync_once` call, in the thread that calls it.

### 5.8 `UltrawideStore` and `DecodedUltrawide` (`ultrawide_store.py`)

The decoded ultrawide frames of the last ~1 s, by `frame_seq`, so a packet's pair is found even when it came in an
earlier packet. Used by the receiver thread only (no lock).

```
@dataclass(frozen=True)
class DecodedUltrawide:
    frame_seq: int
    t_pi_ns: int                    # its SensorTimestamp, as sent
    rgb: np.ndarray | None          # read-only; None: unusable (decoded while waiting for a keyframe)

class UltrawideStore:
    def __init__(self, keep_s: float) -> None              # buffer.keep_s
    def add(self, f: DecodedUltrawide) -> None             # keep it; forget frames captured > keep_s before it
    def get(self, frame_seq: int) -> DecodedUltrawide | None   # None: never received (lost in a gap) or too old
    def __len__(self) -> int
```

Every decoded ultrawide frame is added, usable or not, so a pair that falls in a keyframe wait is found and reported
as "no usable image" rather than "missing".

### 5.9 `PacketBuffer` and `BufferedPacket` (`packet_buffer.py`)

```
@dataclass(frozen=True)
class BufferedPacket:
    packet_seq: int
    t_pc_ns: int                        # the packet time: main mid-exposure on the PC clock (C1 reports it for both)
    t_pi_mid_ns: int                    # the same on the Pi clock (logs and tests)
    arrival_ns: int                     # PC time the packet was taken from the socket
    main_seq: int                       # the main frame's frame_seq
    main_t_pi_ns: int                   # the main frame's SensorTimestamp, as sent
    main_rgb: np.ndarray | None         # (H, W, 3) uint8 at the rig resolution, read-only; None: unusable
    pair_seq: int                       # the header's pair_seq; -1: no pair
    pair_t_pi_ns: int | None            # the pair's own SensorTimestamp (logs, pair-gap statistic); None: not found
    ultrawide_rgb: np.ndarray | None    # the pair's image, read-only; None: no pair, or it is unusable or not received
    def image(self, camera: str) -> np.ndarray | None      # the usable image of that camera
    pair_gap_ns: int | None                                # property: main_t_pi_ns - pair_t_pi_ns

class PacketBuffer:
    def __init__(self, keep_s: float) -> None
    def push(self, p: BufferedPacket) -> None    # append; then drop packets with t_pc_ns < p.t_pc_ns - keep_s.
                                                 # ValueError if p.t_pc_ns <= the newest's (the receiver checks first)
    def snapshot(self) -> tuple[BufferedPacket, ...]   # oldest first; a copy of the list, taken under the lock
    def newest(self) -> BufferedPacket | None
    def __len__(self) -> int
```

A packet is buffered even when one of its images is unusable: the anchor is always the newest packet (iPhUMI's rule),
and a chosen packet without a usable image for a camera makes that camera's pick fail (§6.2). Memory at the v2 sizes:
1 s × 30 packets × (1600×1200×3 + 1280×960×3) bytes ≈ 280 MB, plus the same ultrawide images shared with the
`UltrawideStore`.

### 5.10 `PacketSelector` and `Selection` (`packet_selector.py`)

Pure: no lock, no clock. Rule in §6.2.

```
@dataclass(frozen=True)
class Selection:
    packets: tuple[BufferedPacket, ...] | None   # oldest first; None if the pick failed
    reason: str                                  # "" when ok, e.g. "nearest packet 99.9 ms from its target (max
                                                 #   66.7 ms = 2 frame periods)"

class PacketSelector:
    def __init__(self, fps: Mapping[str, float], spacing_ns: Mapping[str, int], count: int,
                 tolerance_frames: float) -> None
        # fps: each camera's rig fps; spacing_ns: 50 / 100 ms (config); count: OBS_STEPS; tolerance_frames: 2 (config)
    window: int                                  # iPhUMI's max(frame_num): how many of the newest packets it looks at
    def targets(self, anchor_ns: int, camera: str) -> np.ndarray     # int64, oldest first: anchor - (count-1-i)*spacing
    def pick(self, packets: Sequence[BufferedPacket]) -> dict[str, Selection]     # per camera
```

### 5.11 `CameraHealth` (`camera_health.py`)

One per camera. Written by the receiver, read by the server and the main loop; its own lock. Rules in §8.

```
class CameraHealth:
    def __init__(self, camera: str, stale_s: float, down_s: float, fps_window_s: float,
                 clock: Callable[[], int] = time.monotonic_ns) -> None    # start time = clock() now
    def record_frame(self, t_pc_ns: int) -> None     # a packet with a usable image of this camera was buffered
    def record_dropped(self, n: int) -> None         # n frames lost (a frame_seq gap)
    def mark_down(self, reason: str) -> None         # latched for the rest of the run (Pi stream restart, §6.1 step 4)
    def status(self, now_ns: int | None = None) -> str                       # "ok" | "stale" | "down"
    def report(self, now_ns: int | None = None) -> CameraHealthReport       # the C1 sub-message
    dropped: int
```

Every status change is logged once at INFO, e.g. `ultrawide: ok -> stale (newest frame 352 ms old)`.

### 5.12 `Receiver` (`receiver.py`), thread "receiver"

```
class Receiver:
    def __init__(self, source: PacketSource, decoders: Mapping[str, FrameDecoder], rig: RigConfig,
                 clock_map: ClockMap, buffer: PacketBuffer, health: Mapping[str, CameraHealth],
                 default_exposure_ns: int, keep_s: float, clock: Callable[[], int] = time.monotonic_ns) -> None
                 # keep_s: buffer.keep_s, for its UltrawideStore
    def receive_one(self, timeout_s: float) -> bool    # handle at most one C0 packet (§6.1); False if none came.
                                                       # FatalError for this process's own unrecoverable problems.
    def run(self, stop: threading.Event, timeout_s: float) -> None   # receive_one until stop
    def default_exposure_ns(self) -> int               # min(default_exposure_ns, main period // 2)
    counters: Mapping[str, CameraCounters]
        # per camera: received (frames decoded), usable (buffered packets with a usable image of it), dropped
        # (frame_seq gaps), unusable (decoded while waiting for a keyframe); ultrawide only: unpaired (pair_seq -1),
        # pair_missing (a pair_seq not among the decoded frames kept)
    packets: PacketCounters
        # received, buffered, lost (packet_seq gaps), unsynced (no fresh clock), out_of_order, ignored (after a
        # restart); latency_ms (arrival - packet time) and pair_gap_ms per stats period (take_stats())
```

Tests call `receive_one` directly, with no thread.

### 5.13 `FramesServer` (`frames_server.py`), main thread

```
class FramesServer:
    def __init__(self, endpoint: str, buffer: PacketBuffer, health: Mapping[str, CameraHealth],
                 selector: PacketSelector, prep: ImagePrep, clock: Callable[[], int] = time.monotonic_ns) -> None
                 # binds a contracts RpcServer on endpoint with handlers for GetFramesRequest and GetHealthRequest
    endpoint: str                                          # the bound address
    def get_frames(self, req: GetFramesRequest) -> FramesReply      # §6.2
    def get_health(self, req: GetHealthRequest) -> HealthReply
    def serve_once(self, timeout_s: float) -> bool
    def close(self) -> None
    served: int; misses: int                               # replies with images / replies without (and why, in the log)
```

`prep` is the copied `ImagePrep`, built once at start-up (the ultrawide remap table is the expensive part).

### 5.14 `PiRecording`, `RecordedCamera` and `CaptureEvent` (`pi_recording.py`)

Reads a recorded Pi episode (the folder the Pi writes, PI_ADAPTER.md "Pi format") and turns it into C0 frames and
packets. Used only by the two stand-ins for the Pi, `RecordingPacketSource` (unit tests) and `FakePiSender` (unit
tests and `--mock`); it never feeds the real arm.

```
@dataclass(frozen=True)
class RecordedCamera:
    camera: str; sensor: str; width: int; height: int; fps: float; bitrate: int      # bitrate: meta.json's
    frames: tuple[EncodedFrame, ...]  # encoded, frame_seq from 0, t_pi_ns and exposure_ns as recorded (0 if no column)

@dataclass(frozen=True)
class CaptureEvent:                   # one step of the Pi's work, in the order the Pi does it
    camera: str
    frame_seq: int                    # counts on over loops of the clip
    capture_ns: int                   # the recorded t_pi_ns, plus one clip duration per loop
    due_ns: int                       # ultrawide: its capture (it is encoded); main: capture + 5 ms (packet sent)
    frame: EncodedFrame               # the recorded frame

class PiRecordingError(ValueError): ...      # names the file and the problem

class PiRecording:
    @classmethod
    def load(cls, path: Path, encoding: str, max_seconds: float, keyframe_interval: int,
             jpeg_quality: int) -> "PiRecording"
    path: Path
    cameras: Mapping[str, RecordedCamera]    # "main", "ultrawide"
    duration_ns: int                         # from the first frame to one period after the last
    start_ns: int                            # property: the first frame of either camera
    def events(self) -> Iterator[CaptureEvent]    # forever, looping the clip
    def packets(self) -> list[Packet]             # one pass, as the Pi would send it, with the recorded times

def with_seq_and_time(frame: EncodedFrame, frame_seq: int, t_pi_ns: int) -> EncodedFrame
```

Loading: read `meta.json` (`cameras[]`: `file`, `model`, `size`, `fps`, `frames`, `bitrate`); role by `model`
(`imx519` = main, `imx219` = ultrawide; the Pi's own `cam0`/`cam1` names are not trusted); read `<stem>_frames.csv`
(header `frame_index,t_ns`, then optionally `exposure_ns` and the real Pi's `in_episode`; `in_episode` is ignored, so
the lead-in frames play too); keep the frames captured within `max_seconds` of the first frame of either camera;
decode `<stem>.mp4` with PyAV as `read_video_frames` does, and encode each frame as it is decoded (so only one decoded
image is held at a time) the way the live Pi sends it:

- `h264` (`H264Encoder`): libx264 through PyAV, `profile=baseline` (x264's baseline is Constrained Baseline),
  `tune=zerolatency` (no B-frames, no look-ahead), `preset=ultrafast`, the camera's `bitrate` from `meta.json` (main
  15 Mbit/s, ultrawide 10 Mbit/s on the real Pi), a keyframe every `keyframe_interval` frames and at frame 0
  (`keyint = min-keyint`, no scene cuts), SPS/PPS repeated before every keyframe, Annex-B output, one access unit per
  frame;
- `jpeg` (`JpegEncoder`): `cv2.imencode(".jpg", BGR, quality)`; every frame `keyframe: true`.

The recording is re-encoded rather than forwarding its mp4 packets, because a recording's keyframe spacing and B-frame
settings are not guaranteed, and the fake must behave like the live encoder (PI_PAYLOAD.md §5 rule 4). Images under
`--mock` are therefore a second-generation encode of the recording, which is fine for a stand-in. Problems refused with
`PiRecordingError`: no `meta.json`, a camera missing, an unknown sensor model, a camera without `bitrate`, a CSV with
the wrong header or a row count different from `frames`, times not increasing, a video that can't be decoded or is
shorter than the frames used.

**`events()`** is the order the Pi works in: each ultrawide frame at its capture time (it is then encoded and waits
for the next packet), and each main frame's packet 5 ms after the main capture, so every ultrawide frame that can be
its pair under PI_PAYLOAD.md §5 rule 3 is already encoded; ties go to the ultrawide frame. **`packets()`** runs one
pass of `events()` through a `PacketPairer`: one packet per main frame, with `packet_seq` = the main `frame_seq`;
ultrawide frames captured after the last main frame are not sent.

### 5.15 `PacketPairer` (`packet_pairer.py`)

The Pi sender's rules 2 and 3 of PI_PAYLOAD.md §5 as a small class, so `FakePiSender`, `PiRecording.packets()` and
the tests all use exactly the rule the Pi must follow.

```
PAIR_AHEAD_NS = 5_000_000           # a pair may be captured up to 5 ms after the main frame ...
PAIR_MAX_AGE_NS = 50_000_000        # ... and at most 50 ms before it

class PacketPairer(Generic[T]):
    def __init__(self, ahead_ns: int = PAIR_AHEAD_NS, max_age_ns: int = PAIR_MAX_AGE_NS) -> None
    def add_ultrawide(self, frame_seq: int, t_pi_ns: int, item: T) -> None
        # an encoded ultrawide frame, in capture order (ValueError otherwise); it goes out with the next packet
    def take(self, main_t_pi_ns: int) -> tuple[list[T], int]
        # the next packet's ultrawide frames (every pending one, in order, each exactly once) and its pair_seq:
        # the frame with the latest t_pi_ns <= main + 5 ms among those seen, at most 50 ms older; -1 if none
    pending: int
```

The two numbers are the contract's (PI_PAYLOAD.md §5), not settings: changing them changes C0.

### 5.16 `FakePiSender` (`fake_pi_sender.py`), threads "fake-pi-frames" and "fake-pi-time"

The stand-in Pi, used **only** by the unit tests and `--mock` (PLAN §9 "Camera source" backup; there is no replay
anywhere on the real arm). It plays back a short recorded clip just to produce realistic C0 traffic, and speaks
exactly C0 (PI_PAYLOAD.md): it **binds** a PUB socket for packets (contracts' `PacketPublisher`) and a REP socket for
`time_sync` (the Pi side binds, contracts/STRUCTURE.md §6).

```
class FakePiSender:
    def __init__(self, recording: PiRecording, frames_endpoint: str, time_sync_endpoint: str, *,
                 pi_clock_offset_ns: int, drift_ppm: float, send_hwm: int,
                 clock: Callable[[], int] = time.monotonic_ns) -> None
    frames_endpoint: str; time_sync_endpoint: str     # the bound addresses, valid after start()
    def start(self) -> None          # starts both threads; each binds its own socket; returns once both are bound
                                     # (a bind error is raised here)
    def stop(self) -> None           # stops both threads and closes both sockets: the Pi is "lost"
    def pause(self, camera: str | None = None) -> None    # the Pi stops encoding that camera (None: both); its
                                     # frame_seq numbers are skipped. Main paused: no packets at all (packet_seq still
                                     # counts). Ultrawide paused: packets without ultrawide frames, soon unpaired.
    def resume(self, camera: str | None = None) -> None
    def drop_frames(self, camera: str, n: int = 1) -> None  # the Pi drops that camera's next n frames before encoding
                                     # (a frame_seq gap; main: no packet for them, a packet_seq gap)
    def drop_packets(self, n: int = 1) -> None   # the next n packets are lost on the way (as when ZeroMQ's queue is
                                     # full): gaps in packet_seq and both cameras' frame_seq
    def pi_clock_ns(self, pc_ns: int) -> int              # pc_ns + offset + drift_ppm*1e-6*(pc_ns - start)
    sent: Mapping[str, int]                               # "packets", "ultrawide" (frames sent in them)
```

Sending: the recording's `events()` are played at their recorded spacing, starting at `start()`. When an event is due,
a dropped or paused frame is skipped (its number is never sent); otherwise the frame gets `frame_seq` from the event
(counting on over loops) and `t_pi_ns = pi_clock_ns(its capture time)`. An ultrawide frame goes into the
`PacketPairer`; a main frame makes a packet (`packet_seq` = the main `frame_seq`: both count main frames captured)
with the pairer's pending ultrawide frames and `pair_seq`, which is published unless it is to be lost. At the end of
the clip it starts again from frame 0 (a keyframe), with every counter and the clock still counting up, so a long mock
episode never sees time jump back. `time_sync` is answered with `pi_clock_ns(now)`. The offset (1000 s by default) and
drift (20 ppm) make the clock sync do real work under `--mock`.

`fake_pi.py`: `python -m process_frames.fake_pi --station FILE` runs the fake Pi alone, bound to the station's
`endpoints.mock_pi_*`, playing back `process_frames.mock_episode`, until Ctrl-C. With a station whose `endpoints.pi_*`
point at those addresses, a normal `python -m process_frames` (no `--mock`) can then be tested against it in two
terminals.

### 5.17 `FramesApp` (`app.py`), main thread

Builds everything from `Settings`, runs it, and shuts it down.

```
class FramesApp:
    def __init__(self, settings: Settings, ready_file: Path | None, out_dir: Path,
                 clock: Callable[[], int] = time.monotonic_ns) -> None
    def run(self) -> None            # returns on stop (SIGINT or SIGTERM); raises FatalError for an unrecoverable problem
    def request_stop(self) -> None   # called by the SIGINT and SIGTERM handlers
    frames_endpoint: str | None      # the bound C1 address; with rig, buffer, health, receiver, fake_pi: for tests, logs
```

`run()`, in order:

1. `load_live_rig(settings.rig_file)`; a `ConfigError` becomes `FatalError("rig file invalid: <error>", "check
   process_frames.rig_file")`. Log the rig summary.
2. `ImagePrep(rig)` (builds the remap table).
3. Two `CameraHealth`s, one `PacketBuffer`, a `PacketSelector` (the rig's fps, the `select.*` spacings and
   tolerance, `OBS_STEPS`), then `FramesServer(settings.frames, ...)`: binds C1. A bind error is FATAL ("is another
   process_frames running?").
4. With `--mock`: `PiRecording.load(mock_episode, encoding=settings.decoder, ...)`, then `FakePiSender(...).start()`
   on `settings.pi_frames` / `settings.pi_time_sync` (the mock endpoints).
5. One `ClockMap`, two decoders (`make_decoder(settings.decoder, ...)`).
6. Start thread "clock-sync" (`ClockSync.run`) and thread "receiver" (`Receiver.run` with a `PiPacketSource`). Each
   thread body is wrapped: a `FatalError` is stored (first one wins) and sets the stop flag; any other exception
   becomes `FatalError("internal error in thread <name>: <exception>", "see the traceback above")`, also logged with
   its traceback.
7. Main loop until the stop flag: `server.serve_once(server.poll_s)`; then
   - if no ready file has been written yet and both cameras are `ok`: create the ready file, log `ready`;
   - every `log.stats_every_s`: log the stats line and append it to `<out_dir>/stats.jsonl` (section 12);
   - every 5 s while not ready: log what it is waiting for ("no time_sync reply yet", "no packets yet",
     "ultrawide stale", ...).
8. Shutdown (always, also after a FATAL): set the stop flag, join the threads (`server.join_timeout_s` each), stop the
   fake Pi, close the server. Then return, or re-raise the stored `FatalError`.

## 6. Data flow, step by step

### 6.1 Receiving side (thread "receiver", one C0 packet at a time)

1. **Take** the next packet: `source.next_packet(receive.timeout_s)`. Nothing within the timeout → return (the loop
   checks the stop flag). A packet that breaks C0 (a wrong part count, an empty part, a wrong schema or version, a
   missing field, a wrong type, swapped sensors, a JPEG frame with `keyframe: false`, a `pair_seq` newer than the
   packet's newest ultrawide frame, ...) raises `ContractError` in the subscriber → **FATAL** "C0 packet breaks its
   contract: <field>: check the Pi sender version". `arrival_ns = clock()`.
2. **Ignore after a restart.** If the Pi stream restarted (step 4), count the packet as `ignored` and drop it.
3. **Check every frame map against the rig file and config** (cheap, every packet). Any mismatch is **FATAL**, naming
   the camera:
   - `sensor` ≠ `rig.cameras[camera].sensor` → "main stream sensor is imx519, rig file says imx219: check which camera
     is on which Pi port, or process_frames.rig_file" (the contract already ties `imx519` to main and `imx219` to
     the ultrawide, so this catches a rig file with the sensors the wrong way round);
   - `(width, height)` ≠ `rig.cameras[camera].resolution` → "ultrawide frame size 1640x1232, rig file says 1280x960:
     check the Pi stream settings or process_frames.rig_file";
   - `encoding` ≠ the decoder's `ENCODING` → "main stream is jpeg, process_frames.decoder is h264: set
     process_frames.decoder, or fix the Pi sender". This also keeps each camera on one encoding for the whole run.
4. **Restart check** (before anything changes state): `packet_seq ≤` the last one, or for either camera a
   `frame_seq ≤` its last one or a `t_pi_ns ≤` its last one (also between the ultrawide frames of one packet), means
   the Pi's sender restarted (new numbering, maybe a new clock). **Both cameras are latched `down`**
   (`health.mark_down("Pi stream restarted: packet_seq 3 after 1201")`) for the rest of the run, and every later
   packet is ignored (PLAN §5). `message_comms` sees `down` and concludes (§9).
5. **Gaps and decoding**, in part order, so each camera's H.264 chain sees every frame exactly once and in capture
   order:
   - `packet_seq` > last + 1: count the missing packets as `lost` (their frames show up as `frame_seq` gaps too);
   - for each frame, main first, then the ultrawide frames in order: if its `frame_seq` > that camera's last + 1, the
     frames in between were lost: `health.record_dropped(n)` and `decoder.notify_gap()`; then **decode** it (always,
     even if the clock isn't synced yet, so the reference chain stays intact): `rgb = decoder.decode(frame)`.
     - `None`: the frame comes after a gap and before that camera's next keyframe. Count it as `unusable`. **It is
       never served.**
     - `DecodeError` → **FATAL** "main H264 decode failed at frame_seq 1234 (<detail>): check the Pi encoder settings
       (no B-frames, SPS/PPS before every keyframe)".
     - `rgb.shape != (height, width, 3)` → **FATAL** (the stream lies about its size).
   - every decoded ultrawide frame (usable or not) goes into the `UltrawideStore` under its `frame_seq`.
6. **The pair.** `pair_seq == −1`: no pair (`unpaired`). Otherwise `store.get(pair_seq)`: the frame from this packet or
   an earlier one; not found (lost in a gap, or older than `buffer.keep_s`) counts as `pair_missing`. The pair's image
   is its `rgb` (None if it was decoded while waiting for a keyframe).
7. **Packet time.** If `clock_map.is_fresh(arrival_ns)` is false (no sync yet, or none for `clock.max_sync_age_s`),
   count as `unsynced` and stop here: a packet whose time can't be trusted is never served. Otherwise:
   - `exposure = main.exposure_ns` (libcamera `ExposureTime` × 1000, PI_PAYLOAD.md §4), or, if it is 0,
     `min(default_exposure_ns, period // 2)` with `period = round(1e9 / rig main fps)`. The default lives in
     `config.yaml` (5 ms), not in the rig file, whose schema has no such key (the copied loader rejects unknown keys);
     value and cap are dataproc `pi_adapter`'s rule for training data without an exposure column;
   - `t_pi_mid = rig.camera_time_ns("main", main.t_pi_ns, exposure)`: the copied training formula
     `t = t_pi_ns − exposure_ns // 2 + time_offset_ns` (the sign comes from the rig's `timestamp_ref:
     end_of_exposure`);
   - `t_pc = clock_map.to_pc_ns(t_pi_mid, at_pc_ns=arrival_ns)`: the packet time. The pair's own `t_pi_ns` is kept
     (`pair_t_pi_ns`) for logs and the pair-gap statistic, not used for serving.
8. **Order.** If `t_pc <= buffer.newest().t_pc_ns`, count `out_of_order`, log a warning and stop here. This can only
   happen if a clock re-fit moved the offset by more than a frame period (§7), which in steady state it does not.
9. **Buffer.** `buffer.push(BufferedPacket(...))` with the main image and the pair's image (either may be `None`).
   For each camera with a usable image, `health[camera].record_frame(t_pc)`. Add `arrival_ns − t_pc` to the latency
   statistics and the pair gap (`main.t_pi_ns − pair.t_pi_ns`) to the pair-gap statistics.

Every frame of both cameras is decoded as it arrives, whether or not anyone asked for images (PLAN §6.1). After a gap
in one camera, that camera has no usable image until its next keyframe, so a corrupted image is never served; the
other camera is unaffected.

### 6.2 Serving side (main thread, one `get_frames` request)

The rule is iPhUMI's `get_downsampled_camera_images` (`behavior_prompting/deps/real-env/real_env/utils/umi_utils.py`)
over packets, as iPhUMI applies it to its iPhone packets.

1. `now = clock()`. `report = {cam: health[cam].report(now)}`.
2. **Both or nothing.** If either camera's status is not `ok` → reply `FramesReply(main=None, main_t_ns=None,
   ultrawide=None, ultrawide_t_ns=None, health=report)` (contract C1: images only when both cameras are `ok`).
3. `packets = buffer.snapshot()`.
4. **The window.** Only the newest `window` packets are looked at: iPhUMI's `frame_num = round((count − 1) × fps /
   target_freq) + 1` per camera (main: `round(1.5) + 1 = 3` at 30 FPS and 20 Hz; ultrawide: `3 + 1 = 4` at 10 Hz),
   and `window = max(frame_num)` = 4, the same packets for both cameras.
5. **Anchor** T = the newest packet's time.
6. **Pick** (`PacketSelector.pick`), per camera:
   - main: targets `[T − 50 ms, T]` (`select.main_spacing_ms`); ultrawide: targets `[T − 100 ms, T]`
     (`select.ultrawide_spacing_ms`);
   - for each target, the packet in the window with the smallest `|t_pc_ns − target|`; ties go to the earlier packet
     (numpy's `argmin`, as in iPhUMI);
   - the pick fails if any chosen packet is more than `select.tolerance_frames` (2) frame periods of that camera
     (`2 / fps` = 66.7 ms at 30 FPS) from its target (iPhUMI's `RuntimeError`), if any chosen packet has no usable
     image for that camera (main: unusable; ultrawide: no pair, a pair not found, or an unusable one), or if the same
     packet is chosen for both targets.

   At 30 FPS with no drops, the main images are the packets at T and T − 2 periods (16.7 ms from T − 50 ms; a tie,
   so the earlier one), and the ultrawide images the packets at T and T − 3 periods (0 ms off).
   With dropped packets, T − 50 ms can fall back on T itself (50 ms ≤ 66.7 ms): iPhUMI would return that packet
   twice, but C1 needs strictly increasing times, so here it is a failed pick.
7. **A pick fails** (a drop, jitter, a keyframe wait, a packet without a pair): reply with no images, and that
   camera's status set to `stale` **in this reply only** (its age, fps and dropped count unchanged). This keeps C1's
   rule "images if and only if both are `ok`" and gives the same reaction as `stale`: `message_comms` skips the step
   ("slow"). `misses` is counted; the reason is logged at most once per second. `message_comms` relies on exactly
   this: it never sees images without both cameras `ok`, and treats `stale` as "skip the step".
8. **Prep** (the copied code, unchanged): `np.stack([prep.prep_main(p.main_rgb) for p in main])` and
   `np.stack([prep.prep_ultrawide(p.ultrawide_rgb) for p in uw])`: (2, 224, 224, 3) uint8 each. Times: the packets'
   times, `np.array([p.t_pc_ns for p in ...], dtype=np.int64)`, for the ultrawide too (as in iPhUMI, a packet's images
   share its time).
9. Reply `FramesReply(main, main_t_ns, ultrawide, ultrawide_t_ns, health=report)`. The contract class validates it as
   it is built (shapes, dtypes, times increasing, both or nothing).

`get_health` returns `HealthReply(health=report)` built as in step 1. Neither request ever waits for a new packet.
Before ready, both are answered truthfully (cameras not `ok`, no images), so nothing is ever answered "as if ready"
(deploy/STRUCTURE.md §4.4). Image prep costs about 5–10 ms per request (two letterboxes, two remaps plus letterboxes).

## 7. Clock sync (`ClockSync` + `ClockMap`), in plain words

The Pi stamps frames with its own clock (`CLOCK_BOOTTIME`). The PC needs them on its `CLOCK_MONOTONIC`. Both clocks
run steadily, but from different starting points (the **offset**) and at very slightly different speeds (the
**drift**, a few tens of ppm for normal crystals). So `Pi time − PC time` is almost a straight line over time, and the
job is to estimate that line.

**One exchange.** The PC notes `t_send`, sends `time_sync`, the Pi replies with its clock reading `t_pi`, and the PC
notes `t_recv`. The Pi read its clock somewhere between `t_send` and `t_recv`; assuming the middle gives the sample
`offset = t_pi − (t_send + t_recv) / 2`. If the trip there and back took the same time, this is exact; at worst it is
off by half the round trip (`RTT = t_recv − t_send`).

**Wi-Fi makes round trips uneven.** Most are a few ms, but some are delayed tens or hundreds of ms (retries, queues,
the radio sleeping), usually in one direction only. Those samples can be off by half their RTT. So:

1. Keep only the samples of the last `clock.window_s` (60 s; at one per second, about 60 samples).
2. Use only the **best ones**: the `clock.best_fraction` (30 %) of the window's samples with the shortest round trips
   (at least one). A delayed exchange has a long RTT, so it is simply never used. This round-trip filter is what makes
   the estimate robust to Wi-Fi; it is the only outlier rejection needed.
3. **Offset:** the mean of the best samples' offsets, each `t_pi − (t_send + t_recv) / 2`.
4. **Drift:** once the best samples span at least `clock.drift_min_span_s` (10 s), a plain **least-squares line** is
   fitted through them (their offsets against their PC time `(t_send + t_recv) / 2`). Its slope is the drift, and the
   offset at any PC time is read from the line. Before that span exists (the first ~10 s), the drift is 0 and the
   offset is the mean of step 3. A fitted drift above `clock.max_drift_ppm` (500 ppm) is noise (real crystals are
   under 100 ppm), so the drift is set to 0 and the step 3 mean is used.
5. **Mapping a frame:** `t_pc = t_pi − offset(at the frame's arrival time)`. Using the arrival time instead of the
   unknown capture time costs nothing measurable (100 ppm × 0.1 s delay = 10 µs).

With the round-trip filter, the best samples are each within about half the shortest round trip of the truth (1–2 ms
on a quiet network), and averaging tens of them over a 60 s window brings the random part well below that. A constant
one-way difference between the two directions can't be seen by any ping method; it stays as
a small bias, at most half the shortest round trip (about 1–2 ms on a quiet network), small against the 33 ms frame
period and the 50 ms policy step.

**Rhythm.** At start-up `clock.startup_exchanges` (10) exchanges back to back give a first offset within about a
second; then one exchange every `clock.interval_s` (1 s, PLAN §4). A timed-out exchange (`clock.timeout_s`, 0.5 s) is
just lost: counted, logged at most once per 10 s, never resent (contracts' rule), and not an error.

**Before the first sync,** and whenever the newest good sample is older than `clock.max_sync_age_s` (10 s), packets are
decoded but **not buffered** (`unsynced`): their PC time would be a guess. The cameras then show `stale` and, after
`down_s`, `down`, and the log says "no time_sync reply for 10 s". At start-up this only delays the ready file; during a
run, this is PLAN §5's "clock sync with the Pi lost for > 10 s": `message_comms` concludes on `down` (§9). The last
fit is deliberately not kept in use: a packet whose time can't be trusted is never served.

**Steps between re-fits.** Each new sample can move the line a little (sub-millisecond in steady state). Frames
already in the buffer keep the time they were given; the next packets use the new line. A jump bigger than a frame
period would make a packet look older than the one before it, and §6.1 step 8 drops it.

## 8. Health (`CameraHealth`)

Per camera, from the packet time of the newest buffered packet with a **usable image of that camera** (main: a
decoded main frame; ultrawide: a found, decoded pair):

`age = now − t_pc(newest usable)`, or `now − start` before the first one, clamped at ≥ 0 (a packet can look a few ms
"in the future" if the clock estimate is slightly off).

| Status | When |
|---|---|
| `ok` | a usable image has been buffered, and `age ≤ health.stale_s` (0.35 s) |
| `stale` | `health.stale_s < age ≤ health.down_s` (2 s), or none buffered yet and `age ≤ health.down_s` |
| `down` | `age > health.down_s`, or latched by `mark_down` (a Pi stream restart latches both cameras) |

- `last_frame_age_ms` = age in ms. `fps` = `(n − 1) / (t_newest − t_oldest)` over the usable images of the last
  `health.fps_window_s` (1 s), 0 with fewer than 2. `dropped` = frames lost to `frame_seq` gaps since start.
- The status comes from age only. The rate is reported, not judged: missing images already show up as failed picks
  (§6.2 step 7) and, if they last, as `stale`.
- `down` is not latched (except after a restart): if images come back, the camera returns to `ok`. That is harmless,
  because `message_comms` concludes the first time it sees `down`.
- Frames decoded while waiting for a keyframe, and packets that are unsynced, give no usable image, so they let the
  age grow: a lost H.264 frame makes that camera `stale` until its next keyframe (≤ 1 s, PI_PAYLOAD.md §5) if the
  wait is longer than 0.35 s. The packets from just before the loss stay servable, with their true times, until they
  turn `stale`; no image from after the loss is ever served before the next keyframe.
- Packets that keep arriving without a paired ultrawide frame (the ultrawide stopped, PLAN §5) let the ultrawide's age
  grow while the main stays `ok`: `stale` after 0.35 s (steps skipped), `down` after 2 s.

## 9. Failure handling: who decides what

PLAN §5: **process_frames reports; `message_comms` decides** "slow" (skip the step) or "conclude" (stop the episode and
shut down). process_frames exits by itself only for **its own** unrecoverable problems.

| Situation | What process_frames does | What `message_comms` does |
|---|---|---|
| A camera `stale` (no usable image for > 0.35 s) | reports `stale`; replies with no images; keeps running | slow: skips steps until both are `ok` |
| A dropped packet (`packet_seq` gap) or a `frame_seq` gap in one camera | counts it; H.264: that camera has no usable image until its next keyframe; older packets stay servable until they turn stale | slow: skips a step whenever no images come |
| No packet within 2 frame periods of a target, or a chosen packet without a usable image (§6.2) | no images; that camera `stale` in that reply | slow: skips that step |
| Packets without a paired ultrawide frame for longer than the stale limit | the ultrawide `stale`, then `down` past the down limit (§8) | slow, then concludes on `down` |
| A camera `down` (> 2 s), or the Pi lost (packets stop, `time_sync` times out) | reports `down`; keeps running, keeps trying | **concludes**: "main camera down: check the Pi stream and cable" |
| Clock sync lost: no good `time_sync` reply for > 10 s (PLAN §5) | stops buffering packets (unsynced) → `stale` → `down`; logs why | concludes on `down` |
| A Pi stream restart: `packet_seq`, a `frame_seq` or a `t_pi_ns` went back (PLAN §5) | latches both cameras `down`, ignores every later packet, logs why | concludes on `down` |
| Rig file invalid, or the stream doesn't match it (sensor, size) | **FATAL**, exit 1 | `deploy run` sees the exit and shuts all down, quoting the FATAL line |
| Stream encoding ≠ `process_frames.decoder` | **FATAL** | as above |
| Decoder failure (H.264 or JPEG data that can't be decoded, wrong picture size, a held-back picture) | **FATAL** | as above |
| A C0 packet (part count, header, sizes, sensors, encodings, `pair_seq`) or `time_sync` reply breaks its contract | **FATAL**, naming the message and field | as above |
| A C1 request that breaks its contract | `RpcServer` answers `ErrorReply` and keeps serving (contracts §6) | concludes on the `RemoteError` |
| Config or station section invalid; C1 port busy; an exception in any thread | **FATAL** | as above (or `deploy check` / start-up refuses) |
| SIGTERM (from `deploy run`'s stop) or SIGINT (Ctrl-C when alone) | clean stop, exit 0, no FATAL line | — |

## 10. `config.yaml`

Every key, with its default. The station's `process_frames:` section may override any of them, and nothing else
(deploy/STRUCTURE.md §6.2–6.3). Durations in seconds unless the name says `_ms` or `_ns`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rig_file` | path | `rigs/rig_pi_v2.yaml` | the rig file (relative to `process_frames/`) |
| `decoder` | `h264` \| `jpeg` | `h264` | PLAN §9 frame-encoding backup; also what `FakePiSender` sends under `--mock` |
| `mock_episode` | path \| null | `null` | a recorded Pi episode folder whose clip `FakePiSender` plays back as fake C0 traffic; used and required only with `--mock` |
| `default_exposure_ns` | int ≥ 0 | `5000000` | exposure used when C0's `exposure_ns` is 0, capped at half the rig frame period (dataproc `pi_adapter`'s rule and value) |
| `receive.timeout_s` | float > 0 | `0.1` | how long the receiver waits for one C0 packet before checking the stop flag |
| `receive.recv_hwm` | int > 0 | `60` | C0 packets ZeroMQ may queue (2 s); beyond that it drops whole packets, which shows as a `packet_seq` gap |
| `buffer.keep_s` | float > 0 | `1.0` | packets kept, and decoded ultrawide frames kept for finding a pair (PLAN: ~1 s) |
| `select.main_spacing_ms` | float > 0 | `50` | spacing of the 2 main frames (20 Hz) |
| `select.ultrawide_spacing_ms` | float > 0 | `100` | spacing of the 2 ultrawide frames (10 Hz) |
| `select.tolerance_frames` | float > 0 | `2.0` | a picked packet must be within this many frame periods of its target (iPhUMI's `2.0 / fps`: 66.7 ms at 30 FPS) |
| `health.stale_s` | float > 0 | `0.35` | age above which a camera is `stale` (PLAN §5 proposal) |
| `health.down_s` | float > `stale_s` | `2.0` | age above which a camera is `down` (PLAN §5 proposal) |
| `health.fps_window_s` | float > 0 | `1.0` | window for the measured fps |
| `clock.interval_s` | float > 0 | `1.0` | one `time_sync` exchange this often |
| `clock.timeout_s` | float > 0 | `0.5` | wait for one `time_sync` reply |
| `clock.startup_exchanges` | int ≥ 1 | `10` | exchanges back to back at start-up |
| `clock.window_s` | float > 0 | `60.0` | samples older than this are forgotten |
| `clock.best_fraction` | float in (0, 1] | `0.3` | share of the window's samples (shortest round trips) used for offset and drift |
| `clock.drift_min_span_s` | float > 0 | `10.0` | the best samples must span this long before drift is fitted |
| `clock.max_drift_ppm` | float > 0 | `500.0` | a fitted drift above this is treated as noise (set to 0) |
| `clock.max_sync_age_s` | float > 0 | `10.0` | no good sample for this long → frames are not buffered (unsynced) |
| `h264.thread_type` | `SLICE` \| `NONE` | `SLICE` | PyAV decoder threading; frame threading is not allowed (it delays every picture) |
| `server.poll_s` | float > 0 | `0.05` | main loop period: C1 poll, ready check |
| `server.join_timeout_s` | float > 0 | `2.0` | wait per thread at shutdown |
| `log.stats_every_s` | float > 0 | `10.0` | how often the stats line is written |
| `mock.max_seconds` | float > 0 | `10.0` | how much of `mock_episode` is loaded and looped |
| `mock.keyframe_interval` | int ≥ 1 | `30` | fake H.264: a keyframe every this many frames (1 s at 30 FPS, PI_PAYLOAD.md §5); the bit rate is each camera's from the recording's `meta.json` |
| `mock.jpeg_quality` | int 1–100 | `90` | fake JPEG quality |
| `mock.pi_clock_offset_s` | float | `1000.0` | fake Pi clock = PC clock + this ... |
| `mock.drift_ppm` | float | `20.0` | ... running this much fast |
| `mock.send_hwm` | int > 0 | `30` | fake Pi PUB queue, in packets (the Pi's `SNDHWM`, PI_PAYLOAD.md §2) |
| `check.reachable_timeout_s` | float > 0 | `1.0` | `deploy check`: the TCP connect to the Pi |
| `check.subprocess_timeout_s` | float > 0 | `10.0` | `deploy check`: the rig-file and import subprocesses |
| `probe.timeout_s` | float > 0 | `2.0` | `python -m process_frames.probe`: wait for each reply |

The C1 shapes (2 frames per camera, 224×224) are fixed by the contract, not settings. The camera settings (sensor,
size, fps, lens models) come **only** from the rig file, so a new rig file needs no code change.

Read from `station.yaml` (shared keys, deploy/STRUCTURE.md §6.1): `endpoints.pi_frames`, `endpoints.pi_time_sync`,
`endpoints.mock_pi_frames`, `endpoints.mock_pi_time_sync`, `endpoints.frames`, `session_root`.

## 11. `launch.py`

```
from deploy import ManagedProcess, CheckResult, check_helpers

class ProcessFramesProcess(ManagedProcess):
    NAME = "process_frames"
    READY_HINT = "frames from both cameras; check the Pi stream, Wi-Fi and process_frames.rig_file"
    def program(self) -> list[str]           # [self.station.python("light"), "-m", "process_frames"]
    def checks(self) -> list[CheckResult]

PROCESS_CLASS = ProcessFramesProcess
```

Imports only the standard library, `yaml`, `deploy` and `contracts` (deploy/STRUCTURE.md §4). No `extra_env`;
`is_ready` is the default (the ready file).

**Ready** (written by `FramesApp`, §5.17 step 7): C1 is answering, and both cameras have been `ok` at the same time at
least once (so the clock is synced and both streams decode and match the rig file).

**`checks()`**, in this order (all quick; a few seconds in total, most of it the two subprocesses):

| Name | How | Result |
|---|---|---|
| `rig file` | `file_exists`, then a subprocess `python.light -m process_frames.rig_file <rig_file> --json` | pass: `rig_pi_v2: main imx519 1600x1200@30, ultrawide imx219 1280x960@30`; fail: the loader's error |
| `imports` | `python_imports(python.light, ["process_frames", "av", "cv2", "scipy", "zmq"])` | fail names the missing module (PyAV missing → "pip install av==17.1.0") |
| `decoder` | config | **info**: `h264 (default)`, or `jpeg (backup, PLAN §9)` |
| `C1 port` | `port_free(endpoints.frames)` | fail: busy |
| `Pi reachable` (not mock) | `host_reachable` on the host and port of `endpoints.pi_frames`, `check.reachable_timeout_s` | fail: "no TCP connection to 192.168.1.50:5600: is the Pi on, on this Wi-Fi, and its sender running?" |
| `mock episode` (mock) | the folder exists and has `meta.json` | fail: "process_frames.mock_episode is not set" / not found |
| `mock ports` (mock) | `port_free` on `endpoints.mock_pi_frames` and `endpoints.mock_pi_time_sync` | fail: busy |
| `camera source` | — | **info**: `Pi at <host>` or `FakePiSender, stand-in Pi from <episode> (mock)` |

`launch.py` sends and receives no contract messages (deploy/STRUCTURE.md §4.1). Whether the Pi answers `time_sync`
correctly and streams packets that match the rig file is checked by the process itself at start-up: until it does,
the process is not ready, and a mismatch is its FATAL line (§6.1, §9).

## 12. Running it: `__main__.py`, alone, and what it prints

```
python -m process_frames --station FILE [--session DIR] [--ready-file FILE] [--mock]
```

The common arguments of deploy/STRUCTURE.md §4.2, nothing else. `main(argv: Sequence[str] | None = None) -> int`:

1. Logging to stderr in the shared format (`%(asctime)s %(name)s %(levelname)s %(message)s`).
2. `Settings.load(station, mock)`; a `SettingsError` → FATAL "invalid settings: <every problem>: check
   station.yaml's process_frames section".
3. The output folder: `--session DIR` → `DIR/process_frames/`; alone → `<session_root>/<YYYY-MM-DD_HH-MM-SS>_process_frames_alone/process_frames/`, the same layout as in a run
   (deploy/STRUCTURE.md §4.2).
4. SIGINT and SIGTERM call `app.request_stop()`: the launcher stops processes with SIGTERM, Ctrl-C sends SIGINT; both
   are a clean stop.
5. `FramesApp(settings, ready_file, out_dir).run()`. Normal return → exit 0. `FatalError` → last stderr line
   `FATAL process_frames: <what>: <check>`, exit 1.

**Troubleshooting alone**, in its own terminal:

- `python -m process_frames --station station.yaml` against the real Pi;
- `python -m process_frames --station station.yaml --mock` with the fake Pi inside;
- `python -m process_frames.fake_pi --station station.yaml` for a fake Pi alone (§5.14);
- `python -m process_frames.probe --station station.yaml [--save DIR]`: sends one `get_health` and one `get_frames`
  to a running process_frames (`endpoints.frames`), prints the health table, the four image times, their ages and
  spacings, and with `--save` writes the four 224×224 images as PNG, to check by eye that they look like the training
  images.

**What it prints** (log lines, abbreviated):

```
rig rig_pi_v2: main imx519 1600x1200@30, ultrawide imx219 1280x960@30
decoder h264; C1 on tcp://127.0.0.1:5610; Pi frames tcp://192.168.1.50:5600, time_sync tcp://192.168.1.50:5601
clock synced: offset 412.337815 s, rtt min 2.1 ms (10 exchanges)
main: stale -> ok; ultrawide: stale -> ok
ready
stats main ok 30.0fps age 41ms drop 0 unusable 0 | ultrawide ok 30.0fps age 41ms drop 0 unusable 0 unpaired 0 | packets lost 0 unsynced 0 lat 38/52ms pair gap 12/31ms | clock offset 412.337815s drift 19.8ppm rtt 2.1ms sync 0.4s ago | served 25 misses 0
```

The stats line is also appended as JSON to `<out_dir>/stats.jsonl` (per camera: status, fps, age, dropped, received,
usable, unusable, and for the ultrawide unpaired and pair_missing; per packet: received, buffered, lost, unsynced,
out_of_order, ignored, latency and pair-gap mean/max of the period; the clock; served and misses), the only file this
process writes besides the ready file, so Wi-Fi drops, latency and pairing can be looked at after a run.

## 13. Unit tests

Run from `deployment/` in the light environment:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest process_frames/tests`. No Pi, arm, GPU, or network beyond
`127.0.0.1`; sockets bind `tcp://127.0.0.1:*`; no sleeps (wait for a condition with a timeout, or an injected clock).
Socket tests use short limits (e.g. `stale_s` 0.2, `down_s` 0.5) so they run fast. Recordings are tiny and made in
the tests, except `test_real_episode.py`, which reads the real Pi recording `~/Downloads/episode_000045` when it is
present (read only) and is skipped otherwise.

### 13.1 Fixtures and fakes (`conftest.py`)

- `RawDecoder` (§5.5); `raw_frame(...)` (a valid `EncodedFrame` with raw RGB data), `raw_packet(...)` (one packet by
  hand) and `raw_packets(main_t, uw_t, ...)` (packets as the Pi sends them from capture times, paired by
  `PacketPairer`, with chosen keyframes and frames the Pi drops).
- `fake_clock`: a settable ns clock.
- `small_rig`, `v2_rig`: loaded rig files (`tests/data/rig_small.yaml`, `rigs/rig_pi_v2.yaml`).
- `pi_episode(tmp_path, ...)`: writes a tiny Pi-format episode (`meta.json` with per-camera `bitrate`, `cam0.mp4` =
  imx219 and `cam1.mp4` = imx519 encoded with PyAV's libx264, `cam*_frames.csv` with or without `exposure_ns`, and
  with the real Pi's `in_episode` column if asked) at the small rig's sizes, with a moving pattern so every frame
  differs; `v2_episode`: the same at the v2 sizes (main 1600×1200, ultrawide 1280×960), 15 frames per camera.
- `wait_for(condition, timeout_s)`.

`tests/reference_umi_utils.py` holds the copied iPhUMI function (§3).

### 13.2 Test files

| File | Checks |
|---|---|
| `test_imageprep.py` | COPIED (§3): letterbox bit-identical to iPhUMI's `cv_util` on 9 sizes; the vendored `cv_util` equals the iPhUMI source; main prep; **the remap puts a known world point at the expected virtual and 224 pixel** (0° and 12° `R_phys_virt`); prep composition and input validation |
| `test_vendored_camera_model.py` | COPIED (§3): camera models against OpenCV, round trips, remap tables |
| `test_imageprep_live.py` | **224 images identical to `cv_util` at the v2 rig's sizes** (main 1600×1200; ultrawide: the remap output 640×480 letterboxed) and at other sizes (1600×900 among them); `ImagePrep(v2_rig)` builds; the remap puts a known point at the expected pixel for the v2 rig's lens |
| `test_copies.py` | each COPIED file equals its source (`git show <branch>:<path>`) apart from the provenance header and the listed changes; **`rig_pi_v2.yaml` equals the user's file at `196eae4` except `gripper.max_opening_m: 0.080` with its comment and the `tcp:` block**, which has the geometry tool's `COMPUTED` header and a matrix equal to the computed `T_cam_tcp` within 1e-9 (the expected value is written in the test, so `deploy` is not imported); v2 parses at 1600×1200 / 1280×960 @ 30. Each check is skipped when git or its source is not available |
| `test_rig_file.py` | the v2 rig file loads with main 1600×1200 and ultrawide 1280×960, both at 30 FPS, sensors imx519 / imx219; **broken rig files are refused** with the key named: invalid YAML, missing key, unknown key, wrong `schema`, `rig_id` ≠ file name, non-positive size, principal point outside the image, ultrawide not a fisheye, a sensor C0 doesn't allow; `python -m process_frames.rig_file --json` prints the expected JSON and exit codes |
| `test_settings.py` | `config.yaml` loads and every key in §10 exists with its type (the default rig file is v2); an override merges; **an unknown override key is refused with its path**; a wrong type or range is refused; relative paths resolve from the right folder; `--mock` swaps in the mock endpoints and requires `mock_episode`; every problem is listed at once |
| `test_frame_times.py` | **mid-exposure times match the formula**: `t_pi_ns − exposure_ns // 2 + time_offset_ns`; `exposure_ns = 0` uses `min(default, period // 2)`; a `start_of_exposure` rig gives `+`; the buffered packet time = the main frame's formula − the clock offset, and the pair's own capture time is kept |
| `test_clock_map.py` | pure, seeded, no sockets: **offset with uneven Wi-Fi round trips** (each direction 1 ms + exponential jitter, uneven between directions): after 60 s at 1 Hz the mapping error is below half the minimum RTT + 0.5 ms; **drift recovery**: a true drift of ±50 ppm is found within a few ppm once 10 s of best samples exist, and the mapping error stays within the same bound; drift is 0 before `drift_min_span_s`; **outlier rejection by round trip**: 5 % of exchanges (and a burst of several in a row) delayed 50–300 ms one way are never among the best samples and don't move offset or drift; a fitted drift above `max_drift_ppm` gives 0 and the mean offset; old samples leave the window; an impossible sample is ignored; `NotSynced` before the first sample; `is_fresh` turns false after `max_sync_age_s`; thread safety (one thread adds, one maps) |
| `test_clock_sync.py` | against a `FakePiSender` with an offset: `startup_exchanges` then `interval_s` rhythm (fake clock); the offset found within 1 ms; a server that doesn't answer gives timeouts counted, not an error; a broken `time_sync` reply is FATAL naming the field |
| `test_decoders.py` | `H264Decoder` gives **bit-identical** RGB to decoding the same stream from a file with PyAV the way dataproc's `read_video_frames` does; the fake Pi's encoder gives Annex-B, a keyframe every interval with SPS and PPS first, and a Constrained Baseline SPS; frames before the first keyframe → `None`; **after `notify_gap()`, `None` until the next keyframe, then identical images again**; garbage data → `DecodeError`; a stream with B-frames → `DecodeError`; `JpegDecoder` matches `cv2.imdecode` + BGR→RGB; corrupt JPEG → `DecodeError` |
| `test_packet_pairer.py` | **the Pi's pairing rule**: normal 30/30 FPS (one ultrawide frame per packet, paired with it); phase offsets from −33 to +30 ms with jitter, against the rule written plainly, every ultrawide frame sent once and in order, pairs −5…38 ms apart; **0 or 2 ultrawide frames in a packet**; **a pair from an earlier packet**; **no pair (−1)**: none yet, too old, later than + 5 ms, and the exact + 5 ms and 50 ms edges; frames out of order refused |
| `test_packet_buffer.py` | packets older than `keep_s` leave; snapshot independence; images read-only; `image()` and `pair_gap_ns`; non-increasing time refused; one thread pushing while another snapshots: no errors, every snapshot increasing |
| `test_packet_selector.py` | **packet picking equals iPhUMI's `get_downsampled_camera_images`** (the copied reference, given the same timestamps through a fake camera client) on 300 seeded scenarios **with jitter (up to ±15 ms) and dropped packets (up to 40 %)**, per camera and for both cameras together; where the reference returns one packet twice, ours is a miss; the scenarios cover hits, misses and duplicates; the reference is the unchanged source function; the window is 4; 30 FPS picks; dropped packets → misses; a chosen packet without a usable image → a miss for that camera only; no packets; `targets()` |
| `test_camera_health.py` | fake clock: `stale` before the first frame, `down` after `down_s` without frames; **ok → stale → ok** and **ok → down**; latched `down` stays; fps over the window; dropped count; the report passes the contract (`CameraHealthReport`) |
| `test_receiver.py` | with `RecordingPacketSource` + `RawDecoder` + a synced `ClockMap`: packets land with the right times, images and pairs; **every ultrawide frame decoded exactly once and in order** (also those nobody pairs with, at six phase offsets with jitter, 0–2 per packet), each pair's image being that very frame; a pair from an earlier packet is found; no pair and a missing pair are counted; **a main or an ultrawide `frame_seq` gap, and a dropped packet: that camera serves nothing until its next keyframe, then resumes**, the other camera unaffected; **wrong sensor (rig the wrong way round), wrong size, wrong encoding → FATAL naming the camera**; a C0 contract error (wrong part count) → FATAL; a decoded size lie → FATAL; unsynced → decoded, not buffered, counted; **a restart (`packet_seq`, either `frame_seq` or either `t_pi_ns` going back) latches both cameras `down`** and later packets are ignored; out-of-order PC time dropped; with the real `H264Decoder` on a recorded episode: every ultrawide frame decoded once, in order, and every image bit-identical to decoding the stream in one go |
| `test_pi_recording.py` | the `pi_episode` fixture loads; roles come from the sensor model; bit rates from `meta.json`; `exposure_ns` from the column or 0; **the real Pi's `in_episode` column is accepted**; `max_seconds` respected; H.264 frames: first is a keyframe, then one every `keyframe_interval`; JPEG frames all keyframes; **`packets()` pass the C0 validator (multipart round trip) and follow the Pi's rules** (every ultrawide frame once, in order; `pair_seq` as the rule says; pairs −5…38 ms apart); `RecordingPacketSource` plays them; broken episodes refused (no `meta.json`, missing camera, unknown model, bad CSV header, row count mismatch, times not increasing, a video shorter than its CSV) |
| `test_fake_pi_sender.py` | over `127.0.0.1:*`: **C0 packets from `FakePiSender` pass the contract validator** (received with contracts' `PacketSubscriber`); `packet_seq` +1 per packet and equal to the main `frame_seq`; every ultrawide frame once, in order; data equal to the recording; pairs within −5…38 ms; `drop_packets(n)` makes a `packet_seq` gap of n and ultrawide gaps; `drop_frames("main", n)` a `packet_seq` gap with no ultrawide gap; `drop_frames("ultrawide", n)` an ultrawide gap; `pause("ultrawide")` gives packets without ultrawide frames, then unpaired; `pause("main")` stops the packets; the clip loops with counters and time still increasing; `time_sync` replies follow `pi_clock_ns`; after `stop()` nothing answers; a bind error is raised by `start()` |
| `test_frames_server.py` | with a hand-filled packet buffer and health (no sockets for the logic, one socket test for the binding): **C1 replies pass the contract validator** (encode, decode with `expect=FramesReply`); images equal `ImagePrep` of the picked packets' images; times are the picked packets' times, the ultrawide's too; **both or nothing**: one camera `stale` or `down` → no images at all; a failed pick (packets lost, or a chosen packet without a paired ultrawide) → no images and that camera `stale` in that reply; `get_health` |
| `test_pipeline.py` | the whole process inside the test (`FramesApp` with `--mock` settings on `127.0.0.1:*`, the real `H264Decoder`) and a C1 `RpcClient`: becomes ready; replies valid; times within 2 ms of the fake Pi's true capture times (the offset and drift removed); **a served image is bit-identical to the offline prep of the same recorded frame** (live path = training path); served pairs 0–33 ms apart (+5 ms); **at the v2 sizes (1600×1200, 1280×960): valid 224×224 replies, and every ultrawide frame sent is decoded exactly once, none lost**; **a dropped main frame, and a dropped packet: no image from after the drop is served until the next keyframe, then serving resumes**; **health ok → stale → ok** by `pause`/`resume` of the ultrawide (packets without a pair, no images meanwhile), and **ok → down** by `stop()` (the Pi lost); the same with `decoder: jpeg`; the stats line and `probe`; a rig file with the sensors the wrong way round → `run()` raises FATAL |
| `test_real_episode.py` | on the real recording `~/Downloads/episode_000045` (skipped when absent; read only): **the whole recording paired by the Pi's rule** (one packet per main frame, every ultrawide frame once and in order, pairs −5…38 ms apart, unpaired packets only where one camera runs alone, 0–2 ultrawide frames per packet; prints the packet count, the pair gaps' min/median/max and the frames-per-packet histogram); **`FakePiSender` → receiver → C1 with it at the v2 sizes**: valid, bit-exact 224×224 images with true times, every ultrawide frame decoded exactly once, none lost |
| `test_main.py` | `python -m process_frames --station <tmp> --mock` as a subprocess: writes the ready file; SIGINT → exit 0, no FATAL line; SIGTERM → exit 0, no FATAL line; a broken rig file → exit 1, last line `FATAL process_frames: rig file invalid: ...`; an unknown station key → exit 1 naming it; alone without `--session` it writes into `<session_root>/<ts>_process_frames_alone/process_frames/` |
| `test_launch.py` | (deploy/STRUCTURE.md §9) `program()`; `checks()` with the default v2 rig, with a broken station section and with a missing rig file (fail lines); another rig file's summary in the `rig file` row; `--mock` skips the Pi check and checks `mock_episode`; `Pi reachable` passes against a listening local port and fails against a closed one |

The contract tests PLAN §7 asks of each process are the C0 and C1 rows above (`test_fake_pi_sender.py`,
`test_pi_recording.py`, `test_receiver.py`, `test_frames_server.py`, `test_pipeline.py`); their examples are generated
during the tests, never stored (contracts/STRUCTURE.md §8).

## 14. What this design relies on from the Pi sender

The Pi's sending side doesn't exist yet (PLAN §10). What it must send is
[contracts/PI_PAYLOAD.md](../contracts/PI_PAYLOAD.md), which already includes the points below; they are listed here
because this design depends on each of them. `FakePiSender` does exactly this.

1. **Bind** a PUB socket for packets and a REP socket for `time_sync` at the addresses in `station.yaml`.
2. **One multipart packet per main frame**: the header, the main frame, then every ultrawide frame encoded since the
   previous packet, in capture order, each exactly once (H.264 needs them all, even those nobody pairs with).
3. `pair_seq` by the rule: the newest ultrawide frame captured not later than the main frame + 5 ms, at most 50 ms
   older; −1 if none.
4. H.264: each part is **one access unit in Annex-B form**; **SPS/PPS before every keyframe** (an IDR frame), so a
   decoder can start from any keyframe; a keyframe every 30 frames (1 s); Constrained Baseline (no B-frames); the same
   encoder settings as when recording (main 15 Mbit/s, ultrawide 10 Mbit/s).
5. `packet_seq` counts main frames captured, each camera's `frame_seq` its **captured** frames, so frames the Pi drops
   and packets lost on the way both show as gaps; all start at 0 and never reset during a run.
6. `t_pi_ns`: libcamera `SensorTimestamp`; `exposure_ns`: libcamera `ExposureTime` × 1000 (it is in µs), or 0.
7. `time_sync`: read `CLOCK_BOOTTIME` as late as possible before sending the reply.
8. JPEG backup: one JPEG per frame, `keyframe: true`; its quality is the Pi's choice (§5.5).

## 15. Dependencies (`pyproject.toml`)

Python `>=3.10,<3.11`. Light environment only (deploy/STRUCTURE.md §7.3). Pins from that table, identical to
dataproc's for everything image-related, so live image prep runs the same library versions as training:

| Package | Version | Why |
|---|---|---|
| `contracts` | local, editable (installed first) | messages, validators, transport |
| `numpy` | 2.2.6 | |
| `opencv-python-headless` | 4.12.0.88 | image prep (letterbox, remap), JPEG decode; also `cv2.aruco`, which the copied rig loader checks |
| `scipy` | 1.15.3 | the copied rig loader (`Rotation.from_euler`) |
| `av` (PyAV) | 17.1.0 | H.264 decode (and libx264 encode for the fake Pi; the PyPI wheel includes it) |
| `pyyaml` | 6.0.3 | `config.yaml`, `station.yaml`, rig files |
| `pytest` | 8.3.5 | `[project.optional-dependencies] test` |

`pyzmq` and `msgpack` come through `contracts`. `[tool.pytest.ini_options]`: `testpaths = ["tests"]`,
`addopts = "--import-mode=importlib"`. The package includes `process_frames` and `process_frames._dataproc*`;
`config.yaml` and `rigs/` stay in the folder, found through `FOLDER` (editable install).

**Performance risk** (unknown PC, PLAN §9): the receiver decodes and converts to RGB every frame of both cameras
(60 per second, 3.7–5.8 MB each at the v2 sizes) in one thread. PyAV releases Python's lock while decoding, so this should fit, but
it must be measured on the PC: the stats line shows it as rising latency, `unusable` frames or ZeroMQ drops. The
backups are the JPEG decoder (PLAN §9) and, later if needed, converting to RGB only the frames actually picked; both
fit behind the current interfaces.
