# process_frames/

The design of the frames process. Written from PLAN.md §6.1 (its plan), §4 (contracts C0 and C1), §5 (the failure
rule) and §9 (backups), following `contracts/STRUCTURE.md` and the conventions of `deploy/STRUCTURE.md` §7. PLAN.md is
the source of truth: where this file and PLAN.md disagree, PLAN.md wins and this file is fixed.

## 1. Role

A process that receives the live camera frames from the Pi 5 and turns them into the policy's 224×224 images.

It has two sides, running at the same time:

- **Receiving side:** always on, paced by the Pi. It listens to the Pi directly (contract C0); frames never pass
  through `message_comms/`.
- **Serving side:** answers `message_comms/` only when asked (contract C1).

### Owns

- **Receiving side:**
  - receiving every IMX519 (main) and IMX219 160° (ultrawide) frame from the Pi over Wi-Fi, 30 FPS each;
  - decoding the frames (H.264 by default, JPEG as the backup);
  - clock sync with the Pi, and converting each frame's Pi timestamp to PC mid-exposure time;
  - keeping the last ~1 s of frames of each camera;
  - tracking each camera's health (`ok` / `stale` / `down`).
- **Serving side:**
  - picking the frames the policy needs: 2 main frames 50 ms apart and 2 ultrawide frames 100 ms apart;
  - shrinking them to 224×224 with the exact image-prep code that made the training images.
- **The rig file** (camera sensors, resolutions, frame rates, lens models). Until the real ("v2") rig file exists, a
  clearly marked stand-in is used (`rigs/rig_pi_v1_standin.yaml`, `TODO: replace with the v2 rig file`).
- **The fake Pi** (`FakePiSender`), a stand-in for the Pi used only by the unit tests and `--mock`: a thread inside
  this process that sends real C0 messages over a local port. Nothing in this process replays recordings on the real
  arm; with the real Pi, every frame comes live from the Pi.

### Doesn't own

- Anything about the robot or the policy (poses, state, waypoints).
- Deciding whether the episode goes on: it **reports** health; `message_comms` decides "slow" vs "conclude" (§9).
- The Pi's sending side (PLAN §10 open item). What it must send is PLAN §4 C0; §14 sums up what this design relies on.

### Talks to

- The Pi 5 (C0): subscribes to its frame stream; sends `time_sync` requests about once a second.
- `message_comms/` (C1): on request, returns the selected 224×224 images, the PC time each one was taken, and health.
- `deploy/`: through `launch.py` (checks, ready file) and the common arguments.

### Terms used below

- **Ring buffer:** a list of the most recent frames of one camera; when a new frame comes in, frames older than
  `buffer.keep_s` are dropped from the front.
- **Keyframe:** an H.264 frame that can be decoded on its own (an IDR frame). The frames between keyframes only store
  differences to earlier frames, so after a lost frame they can't be decoded correctly until the next keyframe.
- **Mid-exposure time:** the middle of the time the sensor was collecting light for a frame. Training pairs every image
  with the pose at this time, so live frames must be stamped the same way.
- **Clock offset / drift:** the Pi's clock and the PC's clock count from different starting points (offset) and tick
  at very slightly different speeds (drift, in parts per million, ppm: 20 ppm = 20 µs per second).
- **Round trip (RTT):** the time from sending a `time_sync` request to getting its reply.
- **Anchor:** the time all four picked frames are measured back from (the newest main frame, §6.2).
- **Bit-exact:** the same bytes, not just similar images.

## 2. File tree

```
process_frames/
  STRUCTURE.md
  launch.py                    ProcessFramesProcess (section 11); not part of the package
  config.yaml                  every setting, with its default (section 10)
  pyproject.toml               package "process_frames"; pinned dependencies (section 15)
  rigs/
    rig_pi_v1_standin.yaml     COPIED from rig_pi_v1.yaml, 3 stated changes (section 3)
  process_frames/              the Python package
    __init__.py                FOLDER (the process_frames/ folder, for config.yaml and rigs/), NAME = "process_frames"
    __main__.py                `python -m process_frames`: common arguments, logging, FATAL line, exit code (section 12)
    app.py                     FramesApp: builds everything, runs the threads, ready file, stats, shutdown
    settings.py                Settings: config.yaml + the station's process_frames section + endpoints
    errors.py                  FatalError, DecodeError, SettingsError
    rig_file.py                load_live_rig(), is_standin(); `python -m process_frames.rig_file FILE --json`
    frame_source.py            FrameSource (interface)
    pi_stream_source.py        PiStreamSource: C0 subscriber (the real Pi, or the fake Pi)
    replay_source.py           ReplaySource: test stand-in for the Pi: C0 frames from a PiRecording, no network
    frame_decoder.py           FrameDecoder (interface), make_decoder()
    h264_decoder.py            H264Decoder (PyAV; the default)
    jpeg_decoder.py            JpegDecoder (OpenCV; the backup)
    clock_map.py               ClockMap, ClockSample, ClockStats   ADAPTED from main-sim clock_map.py
    clock_sync.py              ClockSync: the time_sync exchanges
    frame_buffer.py            FrameBuffer, BufferedFrame
    frame_selector.py          FrameSelector, Selection            ADAPTED from main-sim frame_select.py
    camera_health.py           CameraHealth
    receiver.py                Receiver: the receiving side
    frames_server.py           FramesServer: the serving side (C1)
    pi_recording.py            PiRecording, RecordedCamera: a recorded Pi episode as C0 frames (for the two stand-ins)
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
    data/
      rig_v1.yaml              COPIED unchanged (used by the copied test_imageprep.py)
      rig_small.yaml           new: tiny resolutions (main 160x90, ultrawide 128x96) for fast pipeline tests
    test_*.py                  section 13
```

One class per file, named after the class, except small value types that live with the class that uses them
(`BufferedFrame`, `Selection`, `ClockSample`, `ClockStats`, `RecordedCamera`), and the copied `_dataproc/` files, which
keep their own layout so they stay byte-identical to their source.

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
| `rigs/rig_pi_v1_standin.yaml` | `configs/rigs/rig_pi_v1.yaml` | `5330902` | first line `# TODO: replace with the v2 rig file`; `rig_id: rig_pi_v1_standin` (the loader requires `rig_id` = file name); `cameras.main.fps: 60` → `30`. Everything else (resolutions, lens models, TCP) stays nominal, as in the source. |
| `tests/data/rig_v1.yaml` | `configs/rigs/rig_v1.yaml` | `5330902` | none |
| `tests/test_imageprep.py` | `tests/test_imageprep.py` | `2f6d3b1` | imports `dataproc.*` → `process_frames._dataproc.*`; `RIG_PATH` → `tests/data/rig_v1.yaml`; the iPhUMI path is `parents[3]`; the vendored-file comparison skips 3 header lines instead of 1 |
| `tests/test_vendored_camera_model.py` | `tests/test_vendored_camera_model.py` | `3fb0ead` | imports `dataproc.geometry.*` → `process_frames._dataproc.geometry.*` |

**Adapted (new code following a reference; header says `# adapted from ...` and what differs):**

| File here | Reference | What is kept / what differs |
|---|---|---|
| `clock_map.py` | `rnoro/main-sim:controlSW/franka_deploy/franka_deploy/peripherals/clock_map.py` @ `935c0b3` | Only the idea of the `sender_stamp_ping` mode: use the shortest-round-trip exchanges of a window, with a max-drift fallback. Differs: a plain least-squares line instead of main-sim's Theil–Sen fit (simpler; the round-trip filter already removes the Wi-Fi outliers), integer ns, one Pi stamp per reply (C0's `time_sync` reply has one `t_pi_ns`), a minimum span before drift is used, a freshness limit, a lock (two threads use it), all numbers from config. The arrival-time modes are dropped. |
| `frame_selector.py` | `rnoro/main-sim:controlSW/franka_deploy/franka_deploy/tasks/frame_select.py` @ `4be9982` | Kept: step back from an anchor by a fixed spacing, take the nearest frame. Differs: the anchor is the newest **main** frame (§6.2), spacings and a fixed tolerance from config, ties go to the earlier frame (as dataproc's `nearest_frame_index`), a miss returns "no frames" instead of raising. |
| `h264_decoder.py`, `pi_recording.py` | `rnoro/data-work:data-processing/dataproc/raw_episode.py` @ `96ff703`, `read_video_frames` | The same PyAV decode and the same `frame.to_ndarray(format="rgb24")` conversion. `H264Decoder` decodes packet by packet instead of from a file. |
| `pi_recording.py` | `rnoro/data-work:data-processing/docs/PI_ADAPTER.md` and `dataproc/pi_format.py` @ `aac065c` | The Pi's file layout (`meta.json` camera list, `<stem>.mp4`, `<stem>_frames.csv` with `frame_index,t_ns[,exposure_ns]`, roles by sensor model: `imx519` = main, `imx219` = ultrawide). Not copied: `pi_format.py` writes the format and depends on `RawEpisode`; we only read it. |

`tests/test_copies.py` (§13) checks that each "copied" file equals its source at that commit, apart from the header and
the listed changes, so a drifted copy fails the tests.

## 4. Threads and how they share data

```
 thread "receiver"    PiStreamSource (SUB socket) -> decoders -> ClockMap.to_pc_ns -> FrameBuffer.push, CameraHealth
 thread "clock-sync"  RpcClient (REQ socket to the Pi) -> ClockMap.add_sample              about once a second
 main thread          FramesServer: RpcServer (REP socket, C1) -> FrameBuffer.snapshot -> FrameSelector -> ImagePrep
                      + the ready file, the stats line every 10 s, watching for a FATAL from the other threads
 --mock only:         FakePiSender: thread "fake-pi-frames" (PUB socket), thread "fake-pi-time" (REP socket)
```

**Rules that keep this safe and simple:**

1. **Each socket is created and used by exactly one thread** (contracts/STRUCTURE.md §6). Classes open their sockets
   inside the thread that uses them (in `run()` or on first use), never in the constructor. The one exception is the
   C1 `RpcServer`, which the main thread binds at start-up and also uses, so a busy port fails at once.
2. **Three shared objects, each with its own lock:** `FrameBuffer` (one per camera), `CameraHealth` (one per camera),
   `ClockMap` (one). A lock is held only for a few list or number operations, **never** during decoding, image prep or
   network I/O. No code ever holds two of these locks at once, so they can't deadlock.
3. **Frames are immutable once buffered.** A `BufferedFrame` is a frozen dataclass and its RGB array is marked
   read-only (`flags.writeable = False`) before it is pushed. `FrameBuffer.snapshot()` copies only the list of
   references (about 30 per camera) under the lock; the serving side then picks and preps outside the lock. The
   receiver can keep pushing meanwhile: the snapshot is unaffected, and a frame dropped from the buffer stays alive as
   long as the snapshot uses it.
4. **One stop flag** (`threading.Event`) for all threads. A thread that hits an unrecoverable problem stores the first
   `FatalError` in `FramesApp` and sets the flag; the main thread then shuts everything down and prints the FATAL
   line (§12). Every blocking call has a timeout (`receive.timeout_s`, `clock.timeout_s`, `server.poll_s`), so each
   thread sees the flag within a fraction of a second.

## 5. Classes

Signatures only. Every class that uses time takes `clock: Callable[[], int]` (returns PC monotonic ns, default
`time.monotonic_ns`) so tests can drive time. Contract message classes (`Frame`, `TimeSyncRequest`, `TimeSyncReply`,
`GetFramesRequest`, `FramesReply`, `GetHealthRequest`, `HealthReply`, `HealthReport`, `CameraHealthReport`) and the
transport helpers (`Subscriber`, `Publisher`, `RpcClient`, `RpcServer`) come from `contracts`.

### 5.1 Errors (`errors.py`)

```
class FatalError(RuntimeError):     # this process's own unrecoverable problem: exit 1 with a FATAL line
    what: str                       # e.g. "main stream sensor is imx219, rig file says imx519"
    check: str                      # e.g. "check which camera is on which Pi port, or process_frames.rig_file"
    # str(e) == f"{what}: {check}"; __main__ prints f"FATAL process_frames: {e}"

class DecodeError(Exception):       camera: str, seq: int, detail: str     # raised by a FrameDecoder
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
    mock_settings: MockSettings; check: CheckSettings      # one small frozen dataclass per config.yaml section

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
STANDIN_MARKER = "# TODO: replace with the v2 rig file"

def load_live_rig(path: Path) -> RigConfig      # load_rig(path), then: both cameras' sensor is one C0 allows
                                                #   (imx519 / imx219). Raises ConfigError naming the file and key.
def is_standin(path: Path) -> bool              # True if a comment line before `schema:` equals STANDIN_MARKER
def main(argv: Sequence[str] | None = None) -> int
    # python -m process_frames.rig_file FILE [--json]
    # prints the rig summary (or, with --json, one line:
    #   {"ok": true, "rig_id": ..., "standin": true,
    #    "cameras": {"main": {"sensor": "imx519", "width": 1600, "height": 900, "fps": 30.0}, "ultrawide": {...}}}
    #   or {"ok": false, "error": "<ConfigError text>"}); exit 0 if valid, else 1
```

`launch.py` runs this module in a subprocess (it may not import the package), so the rig is parsed by exactly one
loader everywhere.

### 5.4 `FrameSource` (interface, `frame_source.py`) and its two implementations

```
class FrameSource(ABC):
    @abstractmethod
    def next_frame(self, timeout_s: float) -> Frame | None   # the next C0 message, or None if none within timeout_s
                                                              # (ReplaySource: None at the end). ContractError if a
                                                              # message breaks C0.
    @abstractmethod
    def close(self) -> None

class PiStreamSource(FrameSource):                            # pi_stream_source.py
    def __init__(self, endpoint: str, recv_hwm: int) -> None  # opens nothing yet
    # next_frame: on first call, opens a contracts Subscriber(name="pi", endpoint, expect=Frame, recv_hwm)
    #             in the calling thread; then Subscriber.receive(timeout_s)

class ReplaySource(FrameSource):                              # replay_source.py; test stand-in for the Pi only
    def __init__(self, frames: Iterable[Frame]) -> None       # given order
    @classmethod
    def from_recording(cls, recording: PiRecording) -> "ReplaySource"   # both cameras merged by t_pi_ns
    exhausted: bool
```

`PiStreamSource` is the only source `FramesApp` ever uses: on the real Pi, and under `--mock` on `FakePiSender` (it
connects to the mock endpoints over `127.0.0.1` exactly as to the Pi). `ReplaySource` is a stand-in for the Pi in
unit tests only (the receiver's logic without sockets); it is never selectable from config or `station.yaml`. A change to C0 stays inside `FrameSource` and the contract (PLAN §6.1).

### 5.5 `FrameDecoder` (interface, `frame_decoder.py`) and its implementations

One decoder object **per camera** (H.264 keeps per-stream state).

```
class FrameDecoder(ABC):
    ENCODING: ClassVar[str]                                    # the C0 `encoding` it accepts: "h264" or "jpeg"
    @abstractmethod
    def decode(self, frame: Frame) -> np.ndarray | None        # (height, width, 3) uint8 RGB;
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
then decoding resumes. Each packet must give exactly one picture, converted with `frame.to_ndarray(format="rgb24")`,
the same call dataproc's `read_video_frames` uses on the training videos. Zero pictures (the decoder held one back)
or more than one, a B picture (`frame.pict_type`: the stream has B-frames, which C0 forbids), or any
`av.error.FFmpegError`, raise `DecodeError`.
`notify_gap()` returns to "waiting for a keyframe".

Why the images match training: the same FFmpeg build (PyAV 17.1.0, the version dataproc pins) decodes the same kind of
stream (the Pi's encoder at the same settings as when recording, PLAN §4) with the same RGB conversion, and H.264
decoding gives the same output with or without threads. Live and recorded streams are separate encodings of
different moments, so "match" means the same decode and prep path, proven bit-exact in tests (§13).

**`JpegDecoder`.** `cv2.imdecode(..., IMREAD_COLOR)` then `cv2.cvtColor(BGR2RGB)`. A `None` result raises
`DecodeError`. `notify_gap()` does nothing: every JPEG frame stands alone (C0 requires `keyframe: true` for JPEG).
JPEG quality is the Pi's choice, not part of C0; at quality 90 the two streams need roughly 100–190 Mbit/s, so the Pi
may have to lower it if this backup is ever used over Wi-Fi.

**`RawDecoder`** (PLAN's test decoder) lives in `tests/conftest.py`, not the package: `data` is raw RGB bytes of
`height × width × 3`. It reports `ENCODING = "h264"` and copies H.264's keyframe rule (unusable after
`notify_gap()` until a `keyframe: true` frame), so the receiver's drop handling can be tested without PyAV.

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

### 5.8 `FrameBuffer` and `BufferedFrame` (`frame_buffer.py`)

```
@dataclass(frozen=True)
class BufferedFrame:
    camera: str
    seq: int
    t_pc_ns: int             # PC mid-exposure time (what C1 reports)
    t_pi_mid_ns: int         # Pi mid-exposure time (kept for logs and tests)
    arrival_ns: int          # PC time the C0 message was taken from the socket
    rgb: np.ndarray          # (H, W, 3) uint8 at the rig resolution, read-only

class FrameBuffer:
    def __init__(self, camera: str, keep_s: float) -> None
    def push(self, f: BufferedFrame) -> None     # append; then drop frames with t_pc_ns < f.t_pc_ns - keep_s.
                                                 # ValueError if f.t_pc_ns <= the newest's (the receiver checks first)
    def snapshot(self) -> tuple[BufferedFrame, ...]   # oldest first; a copy of the list, taken under the lock
    def newest(self) -> BufferedFrame | None
    def __len__(self) -> int
```

Memory at the stand-in resolutions: 1 s × 30 FPS × (1600×900×3 + 1280×960×3) bytes ≈ 240 MB.

### 5.9 `FrameSelector` and `Selection` (`frame_selector.py`)

Pure: no lock, no clock. Rule in §6.2.

```
@dataclass(frozen=True)
class Selection:
    frames: tuple[BufferedFrame, ...] | None     # oldest first; None if the pick failed
    reason: str                                  # "" when ok, e.g. "nearest frame 31.2 ms from 1234567 (tolerance 25.0 ms)"

class FrameSelector:
    def __init__(self, tolerance_ns: int) -> None
    @staticmethod
    def targets(anchor_ns: int, spacing_ns: int, count: int) -> np.ndarray   # int64, oldest first:
                                                                              # anchor - (count-1-i)*spacing
    def pick(self, frames: Sequence[BufferedFrame], anchor_ns: int, spacing_ns: int, count: int) -> Selection
```

### 5.10 `CameraHealth` (`camera_health.py`)

One per camera. Written by the receiver, read by the server and the main loop; its own lock. Rules in §8.

```
class CameraHealth:
    def __init__(self, camera: str, stale_s: float, down_s: float, fps_window_s: float,
                 clock: Callable[[], int] = time.monotonic_ns) -> None    # start time = clock() now
    def record_frame(self, t_pc_ns: int) -> None     # a usable frame was buffered
    def record_dropped(self, n: int) -> None         # n frames lost (a seq gap)
    def mark_down(self, reason: str) -> None         # latched for the rest of the run (Pi stream restart, §6.1 step 4)
    def status(self, now_ns: int | None = None) -> str                       # "ok" | "stale" | "down"
    def report(self, now_ns: int | None = None) -> CameraHealthReport       # the C1 sub-message
    dropped: int
```

Every status change is logged once at INFO, e.g. `main: ok -> stale (newest frame 352 ms old)`.

### 5.11 `Receiver` (`receiver.py`), thread "receiver"

```
class Receiver:
    def __init__(self, source: FrameSource, decoders: Mapping[str, FrameDecoder], rig: RigConfig,
                 clock_map: ClockMap, buffers: Mapping[str, FrameBuffer], health: Mapping[str, CameraHealth],
                 default_exposure_ns: int, clock: Callable[[], int] = time.monotonic_ns) -> None
    def receive_one(self, timeout_s: float) -> bool    # handle at most one C0 message (§6.1); False if none came.
                                                       # FatalError for this process's own unrecoverable problems.
    def run(self, stop: threading.Event, timeout_s: float) -> None   # receive_one until stop
    counters: Mapping[str, ReceiverCounters]
        # per camera: received, buffered, dropped (seq gaps), unusable (waiting for a keyframe),
        #             unsynced (no fresh clock), out_of_order, latency_ms (arrival - t_pc, last stats period)
```

Tests call `receive_one` directly, with no thread.

### 5.12 `FramesServer` (`frames_server.py`), main thread

```
class FramesServer:
    def __init__(self, endpoint: str, buffers: Mapping[str, FrameBuffer], health: Mapping[str, CameraHealth],
                 selector: FrameSelector, prep: ImagePrep, main_spacing_ns: int, ultrawide_spacing_ns: int,
                 clock: Callable[[], int] = time.monotonic_ns) -> None
                 # binds a contracts RpcServer on endpoint with handlers for GetFramesRequest and GetHealthRequest
    endpoint: str                                          # the bound address
    def get_frames(self, req: GetFramesRequest) -> FramesReply      # §6.2
    def get_health(self, req: GetHealthRequest) -> HealthReply
    def serve_once(self, timeout_s: float) -> bool
    def close(self) -> None
    served: int; misses: int                               # replies with images / replies without (and why, in the log)
```

`prep` is the copied `ImagePrep`, built once at start-up (the ultrawide remap table is the expensive part).

### 5.13 `PiRecording` and `RecordedCamera` (`pi_recording.py`)

Reads a recorded Pi episode (the folder the Pi writes, PI_ADAPTER.md "Pi format") and turns it into C0 `Frame`
messages. Used only by the two stand-ins for the Pi, `ReplaySource` (unit tests) and `FakePiSender` (unit tests and
`--mock`); it never feeds the real arm.

```
@dataclass(frozen=True)
class RecordedCamera:
    camera: str; sensor: str; width: int; height: int; fps: float
    frames: tuple[Frame, ...]         # encoded, seq from 0, t_pi_ns and exposure_ns as recorded (0 if no column)

class PiRecordingError(ValueError): ...      # names the file and the problem

class PiRecording:
    @classmethod
    def load(cls, path: Path, encoding: str, max_seconds: float, keyframe_interval: int, bitrate: int,
             jpeg_quality: int) -> "PiRecording"
    path: Path
    cameras: Mapping[str, RecordedCamera]    # "main", "ultrawide"
    duration_ns: int                         # from the first frame to one period after the last
```

Loading: read `meta.json` (`cameras[]`: `file`, `model`, `size`, `fps`, `frames`); role by `model` (`imx519` = main,
`imx219` = ultrawide; the Pi's own `cam0`/`cam1` names are not trusted); read `<stem>_frames.csv` (header
`frame_index,t_ns` or `frame_index,t_ns,exposure_ns`); decode `<stem>.mp4` with PyAV as `read_video_frames` does;
keep the first `max_seconds`. Then encode each frame the way the live Pi sends it:

- `h264`: libx264 through PyAV, `tune=zerolatency` (no B-frames, no look-ahead), `preset=ultrafast`, `bitrate`,
  a keyframe every `keyframe_interval` frames and at frame 0, SPS/PPS repeated before every keyframe, Annex-B output
  (one access unit per message).
- `jpeg`: `cv2.imencode(".jpg", BGR, quality)`; every frame `keyframe: true`.

The recording is re-encoded rather than forwarding its mp4 packets, because a recording's keyframe spacing and B-frame
settings are unknown, and the fake must behave like the live encoder (PLAN §4 C0). Images under `--mock` are
therefore a second-generation encode of the recording, which is fine for a stand-in. Problems refused with
`PiRecordingError`: no `meta.json`, a camera missing, an unknown sensor model, a CSV with the wrong header or a row
count different from `frames`, times not increasing, a video that can't be decoded or is shorter than its CSV.

### 5.14 `FakePiSender` (`fake_pi_sender.py`), threads "fake-pi-frames" and "fake-pi-time"

The stand-in Pi, used **only** by the unit tests and `--mock` (PLAN §9 "Camera source" backup; there is no replay
anywhere on the real arm). It plays back a short recorded clip just to produce realistic C0 traffic, and speaks exactly C0: it **binds** a PUB socket for frames and a REP
socket for `time_sync` (the Pi side binds, contracts/STRUCTURE.md §6).

```
class FakePiSender:
    def __init__(self, recording: PiRecording, frames_endpoint: str, time_sync_endpoint: str, *,
                 pi_clock_offset_ns: int, drift_ppm: float, send_hwm: int,
                 clock: Callable[[], int] = time.monotonic_ns) -> None
    frames_endpoint: str; time_sync_endpoint: str     # the bound addresses, valid after start()
    def start(self) -> None          # starts both threads; each binds its own socket; returns once both are bound
                                     # (a bind error is raised here)
    def stop(self) -> None           # stops both threads and closes both sockets: the Pi is "lost"
    def pause(self, camera: str | None = None) -> None    # stop sending (None: both). Frames due meanwhile are lost
                                                          # on the way: their seq numbers are skipped, as over Wi-Fi
    def resume(self, camera: str | None = None) -> None
    def drop_next(self, camera: str, n: int = 1) -> None  # skip the next n frames of that camera (a seq gap)
    def pi_clock_ns(self, pc_ns: int) -> int              # pc_ns + offset + drift_ppm*1e-6*(pc_ns - start)
    sent: Mapping[str, int]
```

Sending: each camera's recorded frames are played at their recorded spacing, starting at `start()`. A frame is sent
when its time comes, with `t_pi_ns = pi_clock_ns(now)` and `seq` counting up by 1 per frame (skipped numbers for
drops and pauses). At the end of the clip it starts again from frame 0 (a keyframe), with `seq` and the clock still
counting up, so a long mock episode never sees time jump back. `time_sync` is answered with `pi_clock_ns(now)`. The
offset (1000 s by default) and drift (20 ppm) make the clock sync do real work under `--mock`.

`fake_pi.py`: `python -m process_frames.fake_pi --station FILE` runs the fake Pi alone, bound to the station's
`endpoints.mock_pi_*`, playing back `process_frames.mock_episode`, until Ctrl-C. With a station whose `endpoints.pi_*`
point at those addresses, a normal `python -m process_frames` (no `--mock`) can then be tested against it in two
terminals.

### 5.15 `FramesApp` (`app.py`), main thread

Builds everything from `Settings`, runs it, and shuts it down.

```
class FramesApp:
    def __init__(self, settings: Settings, ready_file: Path | None, out_dir: Path,
                 clock: Callable[[], int] = time.monotonic_ns) -> None
    def run(self) -> None            # returns on stop (SIGINT); raises FatalError for an unrecoverable problem
    def request_stop(self) -> None   # called by the SIGINT handler
```

`run()`, in order:

1. `load_live_rig(settings.rig_file)`; a `ConfigError` becomes `FatalError("rig file invalid: <error>", "check
   process_frames.rig_file")`. Log the rig summary, with `STAND-IN` if `is_standin`.
2. `ImagePrep(rig)` (builds the remap table).
3. `FramesServer(settings.frames, ...)`: binds C1. A bind error is FATAL ("is another process_frames running?").
4. With `--mock`: `PiRecording.load(mock_episode, encoding=settings.decoder, ...)`, then `FakePiSender(...).start()`
   on `settings.pi_frames` / `settings.pi_time_sync` (the mock endpoints).
5. Two `FrameBuffer`s, two `CameraHealth`s, one `ClockMap`, two decoders (`make_decoder(settings.decoder, ...)`).
6. Start thread "clock-sync" (`ClockSync.run`) and thread "receiver" (`Receiver.run` with a `PiStreamSource`). Each
   thread body is wrapped: a `FatalError` is stored (first one wins) and sets the stop flag; any other exception
   becomes `FatalError("internal error in thread <name>: <exception>", "see the traceback above")`, also logged with
   its traceback.
7. Main loop until the stop flag: `server.serve_once(server.poll_s)`; then
   - if no ready file has been written yet and both cameras are `ok`: create the ready file, log `ready`;
   - every `log.stats_every_s`: log the stats line and append it to `<out_dir>/stats.jsonl` (section 12);
   - every 5 s while not ready: log what it is waiting for ("no time_sync reply yet", "no main frames yet", ...).
8. Shutdown (always, also after a FATAL): set the stop flag, join the threads (`server.join_timeout_s` each), stop the
   fake Pi, close the server. Then return, or re-raise the stored `FatalError`.

## 6. Data flow, step by step

### 6.1 Receiving side (thread "receiver", one C0 message at a time)

1. **Take** the next C0 message: `source.next_frame(receive.timeout_s)`. Nothing within the timeout → return (the
   loop checks the stop flag). A message that breaks C0 (wrong schema or version, missing field, wrong type, a JPEG
   frame with `keyframe: false`) raises `ContractError` in the subscriber → **FATAL** "C0 frame breaks its contract:
   <field>: check the Pi sender version". `arrival_ns = clock()`.
2. **Ignore a latched camera.** If this camera was marked `down` by a stream restart (step 4), count and drop.
3. **Check against the rig file and config** (cheap, every frame). Any mismatch is **FATAL**, naming the camera:
   - `sensor` ≠ `rig.cameras[camera].sensor` → "main stream sensor is imx219, rig file says imx519: check which camera
     is on which Pi port, or process_frames.rig_file";
   - `(width, height)` ≠ `rig.cameras[camera].resolution` → "ultrawide frame size 1640x1232, rig file says 1280x960:
     check the Pi stream settings or process_frames.rig_file";
   - `encoding` ≠ the decoder's `ENCODING` → "main stream is jpeg, process_frames.decoder is h264: set
     process_frames.decoder, or fix the Pi sender".
4. **Check `seq`** (per camera; the first frame only sets the starting point):
   - `seq == last + 1`: normal;
   - `seq > last + 1`: `seq − last − 1` frames were lost on the way. `health.record_dropped(n)` and
     `decoder.notify_gap()`;
   - `seq <= last`, or `t_pi_ns <= last t_pi_ns`: the Pi's sender restarted (new numbering, maybe a new clock). The
     camera is latched `down` (`health.mark_down("Pi stream restarted: seq 12 after 9031")`) for the rest of the
     run, even if frames keep coming (PLAN §5). `message_comms` sees `down` and concludes (§9).
5. **Decode** (always, even if the clock isn't synced yet, so the H.264 reference chain stays intact):
   `rgb = decoder.decode(frame)`.
   - `None`: the frame comes after a gap and before the next keyframe. Count it as `unusable` and stop here. **It is
     never buffered, so it can never be served.**
   - `DecodeError` → **FATAL** "main H.264 decode failed at seq 1234 (<detail>): check the Pi encoder settings (no
     B-frames, SPS/PPS before every keyframe)".
   - `rgb.shape != (height, width, 3)` → **FATAL** (the stream lies about its size).
6. **Time.** If `clock_map.is_fresh(arrival_ns)` is false (no sync yet, or none for `clock.max_sync_age_s`), count
   as `unsynced` and stop here: a frame whose time can't be trusted is never served. Otherwise:
   - `exposure = frame.exposure_ns` (libcamera `ExposureTime` × 1000, PLAN §4 C0), or, if it is 0,
     `min(default_exposure_ns, period // 2)` with `period = round(1e9 / rig fps)`. The default lives in `config.yaml`
     (5 ms), not in the rig file, whose schema has no such key (the copied loader rejects unknown keys); value and cap
     are dataproc `pi_adapter`'s rule for training data without an exposure column;
   - `t_pi_mid = rig.camera_time_ns(camera, frame.t_pi_ns, exposure)`: the copied training formula
     `t = t_pi_ns − exposure_ns // 2 + time_offset_ns` (the sign comes from the rig's `timestamp_ref:
     end_of_exposure`);
   - `t_pc = clock_map.to_pc_ns(t_pi_mid, at_pc_ns=arrival_ns)`.
7. **Order.** If `t_pc <= buffer.newest().t_pc_ns`, count `out_of_order`, log a warning and stop here. This can only
   happen if a clock re-fit moved the offset by more than a frame period (§7), which in steady state it does not.
8. **Buffer.** Mark `rgb` read-only, `buffer.push(BufferedFrame(...))`, `health.record_frame(t_pc)`, and add
   `arrival_ns − t_pc` to the latency statistics.

Every frame of both cameras is decoded as it arrives, whether or not anyone asked for frames (PLAN §6.1).

### 6.2 Serving side (main thread, one `get_frames` request)

1. `now = clock()`. `report = {cam: health[cam].report(now)}`.
2. **Both or nothing.** If either camera's status is not `ok` → reply `FramesReply(main=None, main_t_ns=None,
   ultrawide=None, ultrawide_t_ns=None, health=report)` (contract C1: images only when both cameras are `ok`).
3. `main_frames = buffers["main"].snapshot()`, `uw_frames = buffers["ultrawide"].snapshot()`.
4. **Anchor** = `main_frames[-1].t_pc_ns`, the newest main frame. Both cameras are measured back from it.
5. **Pick** (`FrameSelector.pick`):
   - main: targets `[anchor − 50 ms, anchor]` (`select.main_spacing_ms`);
   - ultrawide: targets `[anchor − 100 ms, anchor]` (`select.ultrawide_spacing_ms`);
   - for each target, the frame with the smallest `|t_pc_ns − target|`; ties go to the earlier frame;
   - the pick fails if the buffer is empty, any chosen frame is more than `select.tolerance_ms` (25 ms) from its
     target, or the two chosen frames are the same frame.

   Why anchor on main: `frame_select.py` anchors every camera on the newest frame of either camera. With two
   free-running 30 FPS cameras, that time can be up to 33 ms after the newest main frame, so the main pick would miss
   the 25 ms tolerance about half the time. The newest main frame is also `message_comms`' pose anchor (PLAN §3,
   step 2), so all four images are measured from the same instant the poses are. At 30 FPS with no drops, the main
   frames are exact at the anchor and 16.7 ms from `anchor − 50 ms`; the ultrawide frames are at most 16.7 ms from
   either target.
6. **A pick fails** (a single drop, jitter, or a camera a little behind): reply with no images, and that camera's
   status set to `stale` **in this reply only** (its age, fps and dropped count unchanged). This keeps C1's rule
   "images if and only if both are `ok`" and gives the same reaction as `stale`: `message_comms` skips the step
   ("slow"). `misses` is counted; the reason is logged at most once per second. `message_comms` relies on exactly
   this: it never sees images without both cameras `ok`, and treats `stale` as "skip the step".
7. **Prep** (the copied code, unchanged): `np.stack([prep.prep_main(f.rgb) for f in main])` and
   `np.stack([prep.prep_ultrawide(f.rgb) for f in uw])`: (2, 224, 224, 3) uint8 each. Times:
   `np.array([f.t_pc_ns for f in ...], dtype=np.int64)`.
8. Reply `FramesReply(main, main_t_ns, ultrawide, ultrawide_t_ns, health=report)`. The contract class validates it as
   it is built (shapes, dtypes, times increasing, both or nothing).

`get_health` returns `HealthReply(health=report)` built as in step 1. Neither request ever waits for a new frame.
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

**Before the first sync,** and whenever the newest good sample is older than `clock.max_sync_age_s` (10 s), frames are
decoded but **not buffered** (`unsynced`): their PC time would be a guess. The cameras then show `stale` and, after
`down_s`, `down`, and the log says "no time_sync reply for 10 s". At start-up this only delays the ready file; during a
run, this is PLAN §5's "clock sync with the Pi lost for > 10 s": `message_comms` concludes on `down` (§9). The last
fit is deliberately not kept in use: a frame whose time can't be trusted is never served.

**Steps between re-fits.** Each new sample can move the line a little (sub-millisecond in steady state). Frames
already in the buffer keep the time they were given; the next frames use the new line. A jump bigger than a frame
period would make a frame look older than the one before it, and §6.1 step 7 drops it.

## 8. Health (`CameraHealth`)

Per camera, from the PC mid-exposure time of the newest **buffered** (usable, timed) frame:

`age = now − t_pc(newest)`, or `now − start` before the first frame, clamped at ≥ 0 (a frame can look a few ms "in
the future" if the clock estimate is slightly off).

| Status | When |
|---|---|
| `ok` | `age ≤ health.stale_s` (0.35 s) |
| `stale` | `health.stale_s < age ≤ health.down_s` (2 s) |
| `down` | `age > health.down_s`, or latched by `mark_down` (a Pi stream restart) |

- `last_frame_age_ms` = age in ms. `fps` = `(n − 1) / (t_newest − t_oldest)` over the buffered frames of the last
  `health.fps_window_s` (1 s), 0 with fewer than 2. `dropped` = frames lost to `seq` gaps since start.
- The status comes from age only. The rate is reported, not judged: missing frames already show up as failed picks
  (§6.2 step 6) and, if they last, as `stale`.
- `down` is not latched (except after a restart): if frames come back, the camera returns to `ok`. That is harmless,
  because `message_comms` concludes the first time it sees `down`.
- Frames discarded while waiting for a keyframe, or while unsynced, are not buffered, so they let the age grow: a
  dropped H.264 frame makes the camera `stale` until the next keyframe (≤ 1 s, PLAN §4 C0) if the wait is longer than
  0.35 s. The valid frames from just before the drop stay servable, with their true times, until they turn `stale`;
  no frame from after the drop is ever served before the next keyframe.

## 9. Failure handling: who decides what

PLAN §5: **process_frames reports; `message_comms` decides** "slow" (skip the step) or "conclude" (stop the episode and
shut down). process_frames exits by itself only for **its own** unrecoverable problems.

| Situation | What process_frames does | What `message_comms` does |
|---|---|---|
| A camera `stale` (no usable frame for > 0.35 s) | reports `stale`; replies with no images; keeps running | slow: skips steps until both are `ok` |
| A single dropped frame (`seq` gap) | counts it; H.264: buffers nothing until the next keyframe; older valid frames stay servable until they turn stale | slow: skips a step whenever no images come |
| No frame within the picking tolerance | no images; that camera `stale` in that reply (§6.2 step 6) | slow: skips that step |
| A camera `down` (> 2 s), or the Pi lost (both cameras stop, `time_sync` times out) | reports `down`; keeps running, keeps trying | **concludes**: "main camera down: check the Pi stream and cable" |
| Clock sync lost: no good `time_sync` reply for > 10 s (PLAN §5) | stops buffering frames (unsynced) → `stale` → `down`; logs why | concludes on `down` |
| A Pi stream restart: that camera's `seq` or `t_pi_ns` went back (PLAN §5) | latches that camera `down`, logs why | concludes on `down` |
| Rig file invalid, or the stream doesn't match it (sensor, size) | **FATAL**, exit 1 | `deploy run` sees the exit and shuts all down, quoting the FATAL line |
| Stream encoding ≠ `process_frames.decoder` | **FATAL** | as above |
| Decoder failure (H.264 or JPEG data that can't be decoded, wrong picture size, a held-back picture) | **FATAL** | as above |
| A C0 message (frame or `time_sync` reply) breaks its contract | **FATAL**, naming the message and field | as above |
| A C1 request that breaks its contract | `RpcServer` answers `ErrorReply` and keeps serving (contracts §6) | concludes on the `RemoteError` |
| Config or station section invalid; C1 port busy; an exception in any thread | **FATAL** | as above (or `deploy check` / start-up refuses) |
| SIGINT (from `deploy run`'s stop, or Ctrl-C when alone) | clean stop, exit 0, no FATAL line | — |

## 10. `config.yaml`

Every key, with its default. The station's `process_frames:` section may override any of them, and nothing else
(deploy/STRUCTURE.md §6.2–6.3). Durations in seconds unless the name says `_ms` or `_ns`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rig_file` | path | `rigs/rig_pi_v1_standin.yaml` | the rig file (relative to `process_frames/`); `deploy check` warns while it is the stand-in |
| `decoder` | `h264` \| `jpeg` | `h264` | PLAN §9 frame-encoding backup; also what `FakePiSender` sends under `--mock` |
| `mock_episode` | path \| null | `null` | a recorded Pi episode folder whose clip `FakePiSender` plays back as fake C0 traffic; used and required only with `--mock` |
| `default_exposure_ns` | int ≥ 0 | `5000000` | exposure used when C0's `exposure_ns` is 0, capped at half the rig frame period (dataproc `pi_adapter`'s rule and value) |
| `receive.timeout_s` | float > 0 | `0.1` | how long the receiver waits for one C0 message before checking the stop flag |
| `receive.recv_hwm` | int > 0 | `120` | C0 messages ZeroMQ may queue (2 s of both cameras); beyond that it drops, which shows as a `seq` gap |
| `buffer.keep_s` | float > 0 | `1.0` | frames kept per camera (PLAN: ~1 s) |
| `select.main_spacing_ms` | float > 0 | `50` | spacing of the 2 main frames (20 Hz) |
| `select.ultrawide_spacing_ms` | float > 0 | `100` | spacing of the 2 ultrawide frames (10 Hz) |
| `select.tolerance_ms` | float > 0 | `25` | a picked frame must be this close to its target, for both cameras (dataproc's `MAIN_MAX_DT_NS`, 25 ms; training had no limit for the ultrawide) |
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
| `mock.keyframe_interval` | int ≥ 1 | `30` | fake H.264: a keyframe every this many frames (1 s at 30 FPS, C0's minimum) |
| `mock.bitrate` | int > 0 | `10000000` | fake H.264 bit rate (the Pi's recording rate) |
| `mock.jpeg_quality` | int 1–100 | `90` | fake JPEG quality |
| `mock.pi_clock_offset_s` | float | `1000.0` | fake Pi clock = PC clock + this ... |
| `mock.drift_ppm` | float | `20.0` | ... running this much fast |
| `mock.send_hwm` | int > 0 | `120` | fake Pi PUB queue |
| `check.probe_s` | float > 0 | `3.0` | `deploy check`: how long to listen to the Pi stream |
| `check.time_sync_timeout_s` | float > 0 | `1.0` | `deploy check`: wait for the Pi's `time_sync` reply |
| `check.subprocess_timeout_s` | float > 0 | `10.0` | `deploy check`: the rig-file and import subprocesses |

The C1 shapes (2 frames per camera, 224×224) are fixed by the contract, not settings. The camera settings (sensor,
size, fps, lens models) come **only** from the rig file, so the v2 rig file replaces the stand-in with no code change.

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

**Ready** (written by `FramesApp`, §5.15 step 7): C1 is answering, and both cameras have been `ok` at the same time at
least once (so the clock is synced and both streams decode and match the rig file).

**`checks()`**, in this order (all quick; about 6 s in total, most of it the stream probe):

| Name | How | Result |
|---|---|---|
| `rig file` | `file_exists`, then a subprocess `python.light -m process_frames.rig_file <rig_file> --json` | pass: `rig_pi_v1_standin: main imx519 1600x900@30, ultrawide imx219 1280x960@30`; fail: the loader's error |
| `rig stand-in` | the JSON's `standin` | **warn** while true: "rig file is the stand-in copy of rig_pi_v1 (TODO: replace with the v2 rig file); resolutions, lens models and TCP are nominal" |
| `imports` | `python_imports(python.light, ["process_frames", "av", "cv2", "scipy", "zmq"])` | fail names the missing module (PyAV missing → "pip install av==17.1.0") |
| `decoder` | config | **info**: `h264 (default)`, or `jpeg (backup, PLAN §9)` |
| `C1 port` | `port_free(endpoints.frames)` | fail: busy |
| `Pi reachable` (not mock) | `host_reachable` on the host and port of `endpoints.pi_frames` | fail: "no TCP connection to 192.168.1.50:5600: is the Pi on, on this Wi-Fi, and its sender running?" |
| `Pi time_sync` (not mock) | `rpc_answers(endpoints.pi_time_sync, TimeSyncRequest(), TimeSyncReply, check.time_sync_timeout_s)` | pass with the RTT; fail: no reply / broken reply |
| `Pi stream` (not mock) | a contracts `Subscriber` on `endpoints.pi_frames` for up to `check.probe_s`, until both cameras have sent frames | pass: "main imx519 1600x900 h264 30.1 fps, ultrawide ..."; fail lists every problem: a camera with no frames, a sensor or size that differs from the rig JSON, an encoding that differs from `decoder`, a C0 contract error |
| `mock episode` (mock) | the folder exists and has `meta.json` | fail: "process_frames.mock_episode is not set" / not found |
| `mock ports` (mock) | `port_free` on `endpoints.mock_pi_frames` and `endpoints.mock_pi_time_sync` | fail: busy |
| `camera source` | — | **info**: `Pi at <host>` or `FakePiSender, stand-in Pi from <episode> (mock)` |

The stream check only reads message fields; it doesn't decode images (decoding is proven at start-up, before ready).

## 12. Running it: `__main__.py`, alone, and what it prints

```
python -m process_frames --station FILE [--session DIR] [--ready-file FILE] [--mock]
```

The common arguments of deploy/STRUCTURE.md §4.2, nothing else. `main(argv: Sequence[str] | None = None) -> int`:

1. Logging to stderr in the shared format (`%(asctime)s %(name)s %(levelname)s %(message)s`).
2. `Settings.load(station, mock)`; a `SettingsError` → FATAL "invalid settings: <every problem>: check
   station.yaml's process_frames section".
3. The output folder: `--session DIR` → `DIR/process_frames/`; alone → `<session_root>/<YYYY-MM-DD_HH-MM-SS>_process_frames_alone/`.
4. SIGINT and SIGTERM call `app.request_stop()`.
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
rig rig_pi_v1_standin (STAND-IN: TODO replace with the v2 rig file): main imx519 1600x900@30, ultrawide imx219 1280x960@30
decoder h264; C1 on tcp://127.0.0.1:5610; Pi frames tcp://192.168.1.50:5600, time_sync tcp://192.168.1.50:5601
clock synced: offset 412.337815 s, rtt min 2.1 ms (10 exchanges)
main: stale -> ok; ultrawide: stale -> ok
ready
stats main ok 30.0fps age 41ms drop 0 unusable 0 unsynced 0 lat 38/52ms | ultrawide ok 30.0fps ... | clock offset 412.337815s drift 19.8ppm rtt 2.1ms sync 0.4s ago | served 25 misses 0
```

The stats line is also appended as JSON to `<out_dir>/stats.jsonl`, the only file this process writes besides the
ready file, so Wi-Fi drops and latency can be looked at after a run.

## 13. Unit tests

Run from `deployment/` in the light environment:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest process_frames/tests`. No Pi, arm, GPU, or network beyond
`127.0.0.1`; sockets bind `tcp://127.0.0.1:*`; no sleeps (wait for a condition with a timeout, or an injected clock).
Socket tests use short limits (e.g. `stale_s` 0.2, `down_s` 0.5) so they run fast.

### 13.1 Fixtures and fakes (`conftest.py`)

- `RawDecoder` (§5.5) and a `raw_frame(...)` builder for valid C0 `Frame`s with raw RGB data.
- `fake_clock`: a settable ns clock.
- `standin_rig`, `small_rig`: loaded rig files (`rigs/rig_pi_v1_standin.yaml`, `tests/data/rig_small.yaml`).
- `pi_episode(tmp_path, ...)`: writes a tiny Pi-format episode (`meta.json`, `cam0.mp4` = imx219 and `cam1.mp4` =
  imx519 encoded with PyAV's libx264, `cam*_frames.csv` with or without `exposure_ns`) at the small rig's sizes, with
  a moving pattern so every frame differs.
- `wait_for(condition, timeout_s)`.

### 13.2 Test files

| File | Checks |
|---|---|
| `test_imageprep.py` | COPIED (§3): letterbox bit-identical to iPhUMI's `cv_util` on 9 sizes; the vendored `cv_util` equals the iPhUMI source; main prep; **the remap puts a known world point at the expected virtual and 224 pixel** (0° and 12° `R_phys_virt`); prep composition and input validation |
| `test_vendored_camera_model.py` | COPIED (§3): camera models against OpenCV, round trips, remap tables |
| `test_imageprep_live.py` | **224 images identical to `cv_util` at the stand-in rig's sizes** (main 1600×900; ultrawide: the remap output 640×480 letterboxed) and at other sizes; `ImagePrep(standin_rig)` builds; the remap puts a known point at the expected pixel for the stand-in rig's lens |
| `test_copies.py` | each COPIED file equals its source (`git show <branch>:<path>`) apart from the provenance header and the listed changes; the stand-in rig parses to the same values as `rig_pi_v1.yaml` except `rig_id` and the main fps. Skipped when git or the source branch is not available |
| `test_rig_file.py` | the stand-in loads with both cameras at 30 FPS and is detected as stand-in; `rig_v1.yaml` is not; **broken rig files are refused** with the key named: invalid YAML, missing key, unknown key, wrong `schema`, `rig_id` ≠ file name, non-positive size, principal point outside the image, ultrawide not a fisheye, a sensor C0 doesn't allow; `python -m process_frames.rig_file --json` prints the expected JSON and exit codes |
| `test_settings.py` | `config.yaml` loads and every key in §10 exists with its type; an override merges; **an unknown override key is refused with its path**; a wrong type or range is refused; relative paths resolve from the right folder; `--mock` swaps in the mock endpoints and requires `mock_episode`; every problem is listed at once |
| `test_frame_times.py` | **mid-exposure times match the formula**: `t_pi_ns − exposure_ns // 2 + time_offset_ns`; `exposure_ns = 0` uses `min(default, period // 2)`; a `start_of_exposure` rig gives `+`; the receiver's buffered `t_pc_ns` = formula − the clock offset |
| `test_clock_map.py` | pure, seeded, no sockets: **offset with uneven Wi-Fi round trips** (each direction 1 ms + exponential jitter, uneven between directions): after 60 s at 1 Hz the mapping error is below half the minimum RTT + 0.5 ms; **drift recovery**: a true drift of ±50 ppm is found within a few ppm once 10 s of best samples exist, and the mapping error stays within the same bound; drift is 0 before `drift_min_span_s`; **outlier rejection by round trip**: 5 % of exchanges (and a burst of several in a row) delayed 50–300 ms one way are never among the best samples and don't move offset or drift; a fitted drift above `max_drift_ppm` gives 0 and the mean offset; old samples leave the window; an impossible sample is ignored; `NotSynced` before the first sample; `is_fresh` turns false after `max_sync_age_s`; thread safety (one thread adds, one maps) |
| `test_clock_sync.py` | against a `FakePiSender` with an offset: `startup_exchanges` then `interval_s` rhythm (fake clock); the offset found within 1 ms; a server that doesn't answer gives timeouts counted, not an error; a broken `time_sync` reply is FATAL naming the field |
| `test_decoders.py` | `H264Decoder` gives **bit-identical** RGB to decoding the same stream from a file with PyAV the way dataproc's `read_video_frames` does; frames before the first keyframe → `None`; **after `notify_gap()`, `None` until the next keyframe, then identical images again**; garbage data → `DecodeError`; a stream with B-frames → `DecodeError`; `JpegDecoder` matches `cv2.imdecode` + BGR→RGB; corrupt JPEG → `DecodeError` |
| `test_frame_buffer.py` | frames older than `keep_s` leave; snapshot order and independence (later pushes don't change it); arrays read-only; non-increasing time refused; one thread pushing while another snapshots: no errors, every snapshot increasing |
| `test_frame_selector.py` | **frame picking at 30 and 60 FPS, with jitter (±3 ms) and dropped frames**: the nearest frames are picked, ties to the earlier one; a drop next to a target fails at 25 ms tolerance and passes when the neighbour is close enough; empty buffer, single frame, and "same frame twice" fail; `targets()` values |
| `test_camera_health.py` | fake clock: `stale` before the first frame, `down` after `down_s` without frames; **ok → stale → ok** and **ok → down**; latched `down` stays; fps over the window; dropped count; the report passes the contract (`CameraHealthReport`) |
| `test_receiver.py` | with `ReplaySource` + `RawDecoder` + a synced `ClockMap`: frames land in the right buffer with the right times; a `seq` gap counts as dropped and **nothing is buffered until the next keyframe**; **wrong sensor, wrong size, wrong encoding → FATAL naming the camera**; a C0 contract error → FATAL; unsynced → not buffered, counted; `seq` going back → camera latched `down`; out-of-order PC time dropped |
| `test_pi_recording.py` | the `pi_episode` fixture loads; roles come from the sensor model; `exposure_ns` from the column or 0; `max_seconds` respected; H.264 frames: first is a keyframe, then one every `keyframe_interval`; JPEG frames all keyframes; **every frame passes the C0 validator**; broken episodes refused (no `meta.json`, missing camera, unknown model, bad CSV header, row count mismatch) |
| `test_fake_pi_sender.py` | over `127.0.0.1:*`: **C0 messages from `FakePiSender` pass the contract validator** (received with contracts' `Subscriber`); `seq` +1 per frame; `drop_next` makes a gap of n; `pause`/`resume` per camera; the clip loops with `seq` and time still increasing; `time_sync` replies follow `pi_clock_ns`; after `stop()` nothing answers |
| `test_frames_server.py` | with hand-filled buffers and health (no sockets for the logic, one socket test for the binding): **C1 replies pass the contract validator** (encode, decode with `expect=FramesReply`); images equal `ImagePrep` of the picked frames; times are the picked frames' times; **both or nothing**: one camera `stale` or `down` → no images at all; a failed pick → no images and that camera `stale` in that reply; `get_health` |
| `test_pipeline.py` | the whole process inside the test (`FramesApp` with `--mock` settings on `127.0.0.1:*`, the real `H264Decoder`) and a C1 `RpcClient`: becomes ready; replies valid; times within 2 ms of the fake Pi's true capture times (the offset and drift removed); **a served image is bit-identical to the offline prep of the same recorded frame** (live path = training path); **a dropped H.264 frame: no frame from after the drop is served until the next keyframe, then serving resumes**; **health ok → stale → ok** by `pause`/`resume` of one camera (no images meanwhile), and **ok → down** by `stop()` (the Pi lost); the same with `decoder: jpeg`; a fake Pi with the wrong sensor in the rig → `run()` raises FATAL |
| `test_main.py` | `python -m process_frames --station <tmp> --mock` as a subprocess: writes the ready file; SIGINT → exit 0, no FATAL line; a broken rig file → exit 1, last line `FATAL process_frames: rig file invalid: ...`; an unknown station key → exit 1 naming it; alone without `--session` it makes its `_alone` folder |
| `test_launch.py` | (deploy/STRUCTURE.md §9) `program()`; `checks()` with a broken station section and with a missing rig file (fail lines); the stand-in warning; `--mock` skips the Pi checks and checks `mock_episode`; the `Pi stream` check against a `FakePiSender` passes, and fails for a wrong sensor and for a camera that sends nothing |

The contract tests PLAN §7 asks of each process are the C0 and C1 rows above (`test_fake_pi_sender.py`,
`test_pi_recording.py`, `test_receiver.py`, `test_frames_server.py`, `test_pipeline.py`); their examples are generated
during the tests, never stored (contracts/STRUCTURE.md §8).

## 14. What this design relies on from the Pi sender

The Pi's sending side doesn't exist yet (PLAN §10). What it must send is PLAN §4 C0 (and contracts/STRUCTURE.md
§4.2), which already includes the points below; they are listed here because this design depends on each of them.
`FakePiSender` does exactly this.

1. **Bind** a PUB socket for frames and a REP socket for `time_sync` at the addresses in `station.yaml`.
2. One C0 message per captured frame, both cameras on the same PUB socket.
3. H.264: `data` is **one access unit in Annex-B form**; **SPS/PPS before every keyframe** (an IDR frame), so a
   decoder can start from any keyframe; at least one keyframe per second; no B-frames; about 10 Mbit/s, the same
   encoder settings as when recording.
4. `seq`: per camera, +1 per **captured** frame, so frames lost in the camera and on the way both show as gaps.
5. `t_pi_ns`: libcamera `SensorTimestamp`; `exposure_ns`: libcamera `ExposureTime` × 1000 (it is in µs), or 0.
6. `time_sync`: read `CLOCK_BOOTTIME` as late as possible before sending the reply.
7. JPEG backup: one JPEG per frame, `keyframe: true`; its quality is the Pi's choice (§5.5).

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
(60 per second, about 4 MB each) in one thread. PyAV releases Python's lock while decoding, so this should fit, but
it must be measured on the PC: the stats line shows it as rising latency, `unusable` frames or ZeroMQ drops. The
backups are the JPEG decoder (PLAN §9) and, later if needed, converting to RGB only the frames actually picked; both
fit behind the current interfaces.
