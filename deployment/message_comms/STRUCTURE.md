# message_comms/

The design of the manager process. Written from PLAN.md sections 3, 4, 5, 6.4 and 9, contracts/STRUCTURE.md and
deploy/STRUCTURE.md. PLAN.md is the source of truth: where this file and PLAN.md disagree, PLAN.md wins and this
file is fixed.

## 1. Role

The manager at run time. It routes every request and reply between the other three processes, and converts poses
between the robot's absolute form and the policy's relative form while they pass through. Starting and stopping the
processes is done by the launcher in `deployment/deploy/` (PLAN section 6.5), not here.

### Owns

- **The operator's keyboard and the status line:** start an episode, success or fail, stop, home, quit.
- **The episode loop and its rhythm:** when to ask the policy, how many waypoints to send, and the heartbeat to
  `franka_control/`.
- **Deciding "slow" vs "conclude"** for every problem (PLAN section 5), in one place: `FailureMonitor`.
- **Routing.** The selected frames go from `process_frames/` to `policy/`. Waypoints go from the policy's answer to
  `franka_control/`.
- **Listening to the state.** It listens to `franka_control/`'s state broadcast (C2) and keeps the last ~1 s of it.
- **The conversions** (section 6):
  - **Input (absolute → relative):** the TCP pose at each main image time, relative to the newest one (the anchor),
    written as rot6d; the gripper width; and the orientation relative to the episode start pose (`wrt_start`).
  - **Output (relative → absolute):** each of the policy's steps multiplied onto the same anchor, and given its time.
    The result is the absolute, timed waypoints `franka_control/` accepts (C3).
- **Session logging:** one record per episode and per step, in the session folder.

### Doesn't own

- Image processing, camera health thresholds (that is `process_frames/`).
- Running the model (that is `policy/`).
- Safety checks, the watchdog and arm control (that is `franka_control/`).
- Starting, checking and stopping processes (that is `deploy/`; this folder only provides `launch.py`).

### Talks to

| Process | Contract | Pattern | What |
|---|---|---|---|
| `process_frames` | C1 | request/reply (client) | `get_frames` each policy step, `get_health` every 0.5 s |
| `policy` | C4 | request/reply (client) | `config` once, `reset` (+ `prompt` if `uses_prompt`) per episode, `infer` each step |
| `franka_control` | C3 | request/reply (client) | `heartbeat` every loop cycle, `set_waypoints` each step, `start`, `stop`, `home`, `get_state` |
| `franka_control` | C2 | publish/subscribe (subscriber) | every state message, ~100 Hz |

It binds no sockets: every connection goes out from here.

### Note

The conversions follow the policy's pose conventions: relative to the newest pose, rot6d, 16 steps at 50 ms. A model
with different conventions needs changes here as well as in `policy/`.

### Terms used below

- **Anchor:** the TCP pose at the time of the newest main image of one policy step. Everything the policy sees and
  returns in that step is relative to it.
- **Chunk:** the policy's answer to one `infer`: 16 relative steps, 50 ms apart.
- **Waypoint:** one absolute, timed target for the arm: `(t_ns, O_T_TCP, gripper_width_m)`.
- **`exec_h`:** how many of the 16 steps are sent to the arm as waypoints (default 8).
- **rot6d:** a rotation written as 6 numbers: the first two **rows** of its 3×3 rotation matrix (section 6.3).
- **slerp:** "spherical linear interpolation": the rotation a fraction `a` of the way between two rotations, turning
  at a constant rate about one axis.
- **Heartbeat:** a tiny C3 request meaning "the manager is alive". **Watchdog:** `franka_control`'s timer that stops
  the arm when heartbeats stop arriving.
- **REQ socket:** the client side of a ZeroMQ request/reply pair. It must receive the reply before it can send again,
  and it must only be used by one thread (contracts/STRUCTURE.md §6).
- **GIL:** Python's global interpreter lock: only one thread runs Python code at a time. pyzmq releases it while it
  waits for a reply, so other threads keep running during a long `infer`. (main-sim needed a separate stop-watcher
  process because its messaging library did not.)

## 2. File tree

One class per file, named after the class. Small value types (dataclasses) share the file of the class that makes
them.

```
message_comms/
  STRUCTURE.md
  launch.py                    MessageCommsProcess (section 9); not part of the package
  config.yaml                  every setting, with its default (section 8)
  pyproject.toml               package "message_comms"; deps: contracts (editable), numpy (via contracts), scipy==1.15.3,
                               pyyaml==6.0.3; test extra: pytest==8.3.5
  message_comms/
    __init__.py
    __main__.py                `python -m message_comms`: arguments, wiring, signals, FATAL line, exit code (section 10)
    errors.py                  SettingsError, Refused, RequestCancelled, ScriptError, ScriptTimeout      new
    settings.py                Settings, Timeouts, Endpoints, load_settings()                            new
    frames_client.py           FramesClient (C1)                                                         new
    policy_client.py           PolicyClient (C4)                                                         new
    franka_client.py           FrankaClient (C3)                                                         new
    state_history.py           StateHistory, PoseSample, LookupMiss                                      new; interpolation as in robot_utils interpolate_xyz_wxyz
    state_listener.py          StateListener (C2 thread)                                                 new
    pose_math.py               pure functions: relative, absolute, rot6d, interpolation (section 6)      adapted (below)
    pose_converter.py          PoseConverter, Anchor                                                     adapted (below)
    policy_step.py             PolicyStep, StepJob, StepDone, StepSkipped, NoNewFrame, StepTiming        new
    step_worker.py             StepWorker, WorkerResult                                                  new
    problems.py                Kind, ProblemCode, Problem, Verdict                                       new
    failure_monitor.py         FailureMonitor (the slow/conclude table, section 7)                       new
    episode_loop.py            EpisodeLoop, Phase, LoopProgress                                          new; main-sim franka_task.run_policy_control as reference only
    operator.py                Operator (ABC), Key                                                       new
    keyboard_operator.py       KeyboardOperator                                                          new; main-sim stop_watcher.py / _input_thread as reference only
    scripted_operator.py       ScriptedOperator, ScriptEntry                                             new
    status_line.py             StatusLine, StatusSnapshot                                                new
    session_log.py             SessionLog, Outcome, StepRecord, EpisodeRecord                            new; main-sim eval/ records as reference only
  tests/
    conftest.py                fixtures: settings for tests, FakeClock
    fakes.py                   FakeFramesClient, FakePolicyClient, FakeFrankaClient, ManualWorker
    fake_servers.py            fake process_frames / policy / franka_control on real sockets (flow test)
    reference_pose_utils.py    copied unchanged: the training-side pose functions (section 11)
    test_*.py                  section 11
```

**Adapted sources** (each file starts with the provenance header of deploy/STRUCTURE.md §7.6):

- `pose_math.py`, `pose_converter.py`:
  - `# adapted from robot-utils==0.1.4:robot_utils/pose_utils.py (get_relative_pose, get_absolute_pose,
    interpolate_xyz_wxyz)`;
  - and `behavior_prompting:behavior_prompting/common/pose_util.py @ <commit> (mat_to_rot6d, rot6d_to_mat)`.
  - `changes:` rewritten with 4×4 matrices instead of xyz+wxyz quaternions. Same math, checked against the copied
    originals in `tests/reference_pose_utils.py`.
- `real_env/agents/policy_agent.py` (`predict_actions`): the order of the steps (relative obs, infer, absolute from
  the last obs pose, timestamps from the last image time). Nothing is copied. Its `action_as_proprio`,
  `keep_one_step_dummy_proprio` and smoothing options are not carried over (section 6.7).

The fakes used only by tests live in `tests/` (deploy/STRUCTURE.md §7.6). Nothing in this process is swapped at run
time except the operator (keyboard or script). The other processes are swapped at the process level: a stand-in
speaks the same messages. So the three clients are plain classes, not abstract interfaces. Tests pass fakes with the
same methods.

## 3. Threads and how they share data

Four threads. Each ZeroMQ socket belongs to exactly one of them.

| Thread | Runs | Owns (sockets) | Blocks at most |
|---|---|---|---|
| **main** | `EpisodeLoop`: the rhythm, keys, heartbeats, waypoints, failure decisions, logging, status line, signal handling | `FrankaClient` (C3), a `FramesClient` for `get_health` (C1) | `timeouts.commands_s` + `timeouts.health_s` (0.2 s each) in one cycle; during `home` it waits in `loop_ms` slices (section 5.7) |
| **step worker** | `StepWorker`: one job at a time: `config`, `reset` (+ `prompt`), or a policy step (`PolicyStep.run`) | a `FramesClient` for `get_frames` (C1), `PolicyClient` (C4) | `timeouts.policy_hang_s` (5 s) |
| **state listener** | `StateListener`: receive every C2 message, add it to `StateHistory` | the C2 `Subscriber` | `loop_ms` per receive |
| **keyboard** | `KeyboardOperator`'s reader: one key at a time from the terminal, into a queue | none | forever (daemon thread) |

Why the policy call is not on the main thread: the main thread must send a heartbeat every cycle and react to the
stop key at once. If it waited on `infer` (up to 5 s), a stop key would wait too. So the main loop never waits on the
policy. It hands the step to the worker and keeps cycling every `loop_ms` (20 ms).

There are **two C1 clients** (two REQ sockets to the same `process_frames` server) because sockets are not shared
between threads. `process_frames`' REP socket answers them one at a time; each call takes a few ms.

How data is shared. Nothing else is shared:

| Shared thing | Writer → reader | How |
|---|---|---|
| `StateHistory` | listener → worker (lookups), main (latest state) | one `threading.Lock` inside `StateHistory`; every method copies what it returns |
| worker jobs and results | main → worker → main | `StepWorker` holds a job queue and a result queue (`queue.Queue`, size 1 each); only main submits, only main reads results |
| keys | keyboard → main | `queue.Queue` inside `KeyboardOperator` |
| interrupt and terminate flags | SIGINT / SIGTERM handler → main | two `threading.Event`s. Python runs signal handlers on the main thread, so these are only flags |
| listener error | listener → main | `StateListener.error()`: the first exception the thread hit, set once, read every cycle |

All decisions (slow / conclude, episode state, what to send) are made on the main thread. The worker and the
listener only report.

The clients are built in `__main__` and handed to their thread before it starts. After that, no other thread
touches them.

## 4. The classes

Times are `int` nanoseconds from `time.monotonic_ns()`. Every class that uses time takes a `clock: Callable[[], int]`
in its constructor (deploy/STRUCTURE.md §7.6), so tests drive time.

### 4.1 `Settings` (`settings.py`)

```
@dataclass(frozen=True)
class Endpoints:
    frames: str          # station endpoints.frames    (C1)
    state: str           # station endpoints.state     (C2)
    commands: str        # station endpoints.commands  (C3)
    policy: str          # station endpoints.policy    (C4)

@dataclass(frozen=True)
class Timeouts:          # every key of config.yaml `timeouts:` (section 8)
    commands_s: float; health_s: float; frames_s: float; policy_hang_s: float
    config_s: float; reset_s: float; prompt_s: float; home_s: float; startup_s: float

@dataclass(frozen=True)
class Settings:          # one field per config.yaml key, plus:
    endpoints: Endpoints
    session_dir: Path    # <session>/message_comms/
    mock: bool
    ...

def load_settings(station_file: Path, session: Path | None, mock: bool) -> Settings    # raises SettingsError
```

`load_settings` reads `message_comms/config.yaml` (next to the package), then merges `station.yaml`'s
`message_comms:` section over it. It uses the rule of deploy/STRUCTURE.md §6.3, restated here because this package
cannot import `deploy`: maps merge key by key; other values are replaced; an override key that is not in
`config.yaml` is an error. It also reads `endpoints.*` and `session_root` from the station file. It checks every value
(types, ranges, `exec_h` between 1 and 16, `pose_source` one of two words) and raises one `SettingsError` listing
every problem. With no `--session`, it makes `<session_root>/<YYYY-MM-DD_HH-MM-SS>_message_comms_alone/` as the
session folder and writes into its `message_comms/` subfolder, the same layout as in a run (deploy §4.2).

### 4.2 The three clients

Each client wraps one `contracts.RpcClient`. It always passes the expected reply class, so the contracts layer
rejects any reply that breaks its contract. It adds only the checks the contract cannot do (they involve the
request). Errors that pass through unchanged: `TransportTimeout`, `RemoteError`, `ContractError` (from `contracts`).
Added: `Refused(process, request, reason)` for an `Ack(ok=false)`, and `RequestCancelled` for a cancelled `home`.

```
class FramesClient:                                          # frames_client.py; process name "process_frames"
    def __init__(self, endpoint: str, timeout_s: float) -> None
    def get_frames(self) -> FramesReply                      # c1.get_frames/v1 -> c1.frames/v1
    def get_health(self) -> HealthReport                     # c1.get_health/v1 -> c1.health/v1
    def close(self) -> None

class PolicyClient:                                          # policy_client.py; process name "policy"
    def __init__(self, endpoint: str, timeouts: Timeouts) -> None
    def config(self) -> PolicyConfig                         # timeout config_s
    def reset(self) -> None                                  # timeout reset_s; Ack(ok=false) -> Refused
    def prompt(self, task_name: str, prompt_index: int) -> None   # timeout prompt_s; Ack(ok=false) -> Refused
    def infer(self, obs: Observation) -> ActionChunk         # timeout policy_hang_s (the hang limit)
    def close(self) -> None

class FrankaClient:                                          # franka_client.py; process name "franka_control"
    def __init__(self, endpoint: str, timeouts: Timeouts) -> None
    def heartbeat(self) -> None                              # timeout commands_s; Ack(ok=false) -> Refused
    def start(self) -> Ack                                   # timeout commands_s; a refusal is returned, not raised
    def stop(self) -> None                                   # timeout commands_s; Ack(ok=false) -> Refused
    def home(self, interrupted: Callable[[], bool]) -> None  # timeout home_s, waited in loop_ms slices;
                                                             # Ack(ok=false) -> Refused; interrupted() -> RequestCancelled
    def set_waypoints(self, msg: SetWaypoints) -> SetWaypointsReply   # timeout commands_s
    def get_state(self) -> ArmState                          # timeout commands_s
    def close(self) -> None
```

Extra checks (each failure is a `ContractError` naming the message and field, so it is a contract violation in
section 7):

- `set_waypoints`: the reply's `chunk_id` equals the request's. `accepted + skipped + len(rejected)` equals the
  number of waypoints sent: every waypoint is accounted for exactly once (`franka_control` guarantees this, so a
  mismatch means the two sides disagree on the contract).
- `home` is the one request that can take long (up to `home_s`). The client waits for its reply in `loop_ms`
  slices and calls `interrupted()` between them. If that returns true, it closes the C3 socket with linger 0
  (dropping the reply, exactly as the "lazy pirate" timeout does) and raises `RequestCancelled`. The slicing uses
  the `cancel` / `poll_s` arguments of `contracts.RpcClient.request` (contracts §6); every other request is one
  plain `request` call.
- `config` is not checked here. The values are compared in `EpisodeLoop.startup` (section 5.1), because what they
  must equal comes from `Settings`.

### 4.3 `StateHistory` (`state_history.py`) and `StateListener` (`state_listener.py`)

```
@dataclass(frozen=True)
class PoseSample:
    t_ns: int
    T: np.ndarray              # (4, 4) float64, base frame
    width_m: float

class LookupMiss(Enum):
    EMPTY = "no state yet"
    TOO_OLD = "image older than the state history"
    TOO_NEW = "image newer than the newest state"
    SEQ_GAP = "missed state message around the image time"

class StateHistory:
    def __init__(self, history_s: float) -> None
    def add(self, state: ArmState) -> None                         # listener thread
    def latest(self) -> ArmState | None
    def pose_at(self, t_ns: int, source: str) -> PoseSample | LookupMiss   # source: "target" | "measured"
    def gap_count(self) -> int                                     # C2 seq gaps seen so far (status line)
```

- Holds the states of the last `history_s` (1.0 s ≈ 100 states at 100 Hz), oldest first, trimmed on every `add`.
- `add` raises `ContractError` (`c2.state/v1`, field `seq` or `t_ns`, problem `value`) if `seq` does not strictly
  increase or `t_ns` decreases.
- A message with the **same** `t_ns` as the previous one is normal. `franka_control` repeats its last good reading
  during homing and after a fault. Such a message updates `latest()` (so its `mode` and `error` are seen), but adds
  no interpolation sample: two samples at one time would make the interpolation divide by zero.
- A `seq` that goes up by more than 1 is a gap. It is counted, and the next interpolation sample is marked "gap
  before" (`gap_before = True`). Gaps are judged on `seq` across **all** messages, repeats included, so skipping a
  repeat as a sample never looks like a gap.
- `pose_at(t, source)` reads `O_T_TCP_target` for `"target"` and `O_T_TCP` for `"measured"`. Section 6.1 gives the
  math. It finds the two neighbouring samples `s_i.t_ns ≤ t ≤ s_{i+1}.t_ns` by binary search. If `s_{i+1}` is
  marked "gap before", it returns `SEQ_GAP`. If `t` is before the oldest state, `TOO_OLD`; after the newest,
  `TOO_NEW`. It never extrapolates.

```
class StateListener:
    def __init__(self, endpoint: str, history: StateHistory, recv_hwm: int, poll_s: float) -> None
    def start(self) -> None                     # starts the thread
    def stop(self) -> None                      # sets its stop event, joins it (≤ poll_s), closes the socket
    def error(self) -> BaseException | None     # the first exception (e.g. ContractError) the thread hit; it then exits
    def received(self) -> int                   # messages received so far
```

The thread loops on `Subscriber.receive(timeout_s=poll_s)` and calls `history.add` for each message.
A broken message or a non-increasing `seq` stores the exception and ends the thread. The main thread sees it in its
next cycle and concludes (contract violation).

### 4.4 `pose_math.py` (functions) and `PoseConverter` (`pose_converter.py`)

`pose_math.py` holds pure numpy functions, float64, working on stacks of matrices (`(..., 4, 4)`). They are exactly
the math of section 6:

```
def invert(T: np.ndarray) -> np.ndarray                          # [Rᵀ, −Rᵀp]
def relative(T_anchor: np.ndarray, T: np.ndarray) -> np.ndarray  # inv(T_anchor) @ T
def absolute(T_anchor: np.ndarray, T_rel: np.ndarray) -> np.ndarray   # T_anchor @ T_rel
def rot6d_from_matrix(R: np.ndarray) -> np.ndarray               # (..., 3, 3) -> (..., 6): rows 0 and 1
def matrix_from_rot6d(d6: np.ndarray) -> np.ndarray              # (..., 6) -> (..., 3, 3): Gram-Schmidt on rows
def interpolate(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray   # linear position, slerp rotation
def position_distance_m(T0: np.ndarray, T1: np.ndarray) -> float
```

```
@dataclass(frozen=True)
class Anchor:
    t_ns: int                 # main_t_ns[-1]: the newest main image time
    T: np.ndarray             # (4, 4) float64: the pose the observation was made relative to
    source: str               # "target" or "measured" (after a fallback, section 6.7)

class PoseConverter:
    def __init__(self, exec_h: int, lookahead_ms: int) -> None    # the step time is contracts.STEP_MS
    def to_observation(self, poses: np.ndarray, widths: np.ndarray, start_pose: np.ndarray,
                       frames: FramesReply, anchor_t_ns: int, source: str) -> tuple[Observation, Anchor]
        # poses (2, 4, 4) at main_t_ns[0], main_t_ns[1]; widths (2,); start_pose (4, 4)
    def to_waypoints(self, chunk: ActionChunk, anchor: Anchor, chunk_id: int) -> SetWaypoints
```

**The same anchor both ways** is enforced by the types. `to_observation` returns the `Anchor` it used, and
`to_waypoints` only accepts an `Anchor`. `PolicyStep` passes that same object through, never a fresh state lookup.
The converter keeps no state between calls.

### 4.5 `PolicyStep` (`policy_step.py`)

One policy step, start to finish, synchronous (section 5.4 gives the order). It runs on the worker thread.

```
@dataclass(frozen=True)
class StepJob:
    episode_id: int
    chunk_id: int                 # unique in the session, increasing
    start_pose: np.ndarray        # (4, 4): the episode start pose, for wrt_start
    last_anchor_t_ns: int | None  # the previous step's anchor time in this episode

@dataclass(frozen=True)
class StepTiming:
    t_begin_ns: int; t_frames_ns: int; t_policy_ns: int; t_end_ns: int     # clock readings during the step
    image_age_ms: float                                                     # t_frames − anchor time

@dataclass(frozen=True)
class StepDone:
    job: StepJob; waypoints: SetWaypoints; anchor: Anchor; health: HealthReport
    timing: StepTiming; anchor_dev_m: float     # |p_target − p_measured| at the anchor time
    observation: Observation; chunk: ActionChunk  # what was sent and received (for the tap, section 4.10)

@dataclass(frozen=True)
class StepSkipped:
    job: StepJob; problem: Problem; health: HealthReport | None
    width_clipped: int            # the chunk's width_clipped (an odd chunk); 0 when no chunk arrived

@dataclass(frozen=True)
class NoNewFrame:
    job: StepJob; health: HealthReport

class PolicyStep:
    def __init__(self, frames: FramesClient, policy: PolicyClient, history: StateHistory,
                 converter: PoseConverter, settings: Settings, clock: Callable[[], int]) -> None
    def run(self, job: StepJob) -> StepDone | StepSkipped | NoNewFrame
```

`run` **detects** problems and reports them as a `Problem` inside `StepSkipped`. It does not decide what they mean;
`FailureMonitor` does (on the main thread). Exceptions from the clients (`TransportTimeout`, `RemoteError`,
`ContractError`, `Refused`) are not caught here. The worker catches them and passes them back.

### 4.6 `StepWorker` (`step_worker.py`)

```
@dataclass(frozen=True)
class WorkerResult:
    kind: str                     # "config" | "reset" | "step"
    value: object | None          # PolicyConfig | None | StepDone | StepSkipped | NoNewFrame
    error: BaseException | None   # set instead of value when the job raised
    process: str                  # whom the failing request went to: "policy" | "process_frames"
    request: str                  # the schema of the failing request, for the message
    started_ns: int; finished_ns: int

class StepWorker:
    def __init__(self, step: PolicyStep, policy: PolicyClient, clock: Callable[[], int]) -> None
    def start(self) -> None
    def stop(self) -> None                         # asks the thread to exit; does not wait for a hung infer (daemon thread)
    def busy(self) -> bool                         # a job is submitted and its result not yet taken
    def submit_config(self) -> None                # raises RuntimeError if busy (a bug, not an operator error)
    def submit_reset(self, prompt: tuple[str, int] | None) -> None    # reset, then prompt if given
    def submit_step(self, job: StepJob) -> None
    def wait(self, timeout_s: float) -> WorkerResult | None   # main thread: the result, or None after timeout_s
    def running_for_ms(self) -> float | None       # how long the current job has run (status line)
```

There is at most one job at a time. An unexpected exception in the worker thread (a bug) is returned as
`WorkerResult(error=...)`. If the thread itself dies, `busy()` stays true and `wait` raises `RuntimeError`. Both
conclude ("internal error").

### 4.7 `FailureMonitor` (`failure_monitor.py`) and `problems.py`

```
class Kind(Enum):
    SLOW = "slow"                 # keep running (PLAN §5)
    CONCLUDE = "conclude"         # unrecoverable: stop, abort, FATAL, exit 1
    STOP = "stop"                 # q / Ctrl-C: stop, record "stopped", exit 0; SIGTERM: stop, record "aborted", exit 0

class ProblemCode(Enum): ...      # one member per row of the table in section 7

@dataclass(frozen=True)
class Problem:
    code: ProblemCode
    process: str                  # where it came from: "process_frames", "policy", "franka_control", "message_comms", "operator"
    detail: str                   # one line, e.g. "main: stale, last frame 412 ms ago"

@dataclass(frozen=True)
class Verdict:
    kind: Kind
    problem: Problem
    what_failed: str              # e.g. "main camera down"
    what_to_check: str            # e.g. "check the Pi stream and cable"

class FailureMonitor:
    def __init__(self, settings: Settings) -> None
    def classify(self, problem: Problem) -> Verdict                 # counts it too
    def from_exception(self, exc: BaseException, process: str, request: str) -> Problem
    def check_state(self, latest: ArmState | None, now_ns: int, episode_running: bool,
                    stop_sent: bool, start_ack_ns: int | None) -> Problem | None
    def check_health(self, health: HealthReport) -> Problem | None  # only "down" is a problem here
    def counts(self) -> Mapping[ProblemCode, int]                   # for the status line and the episode record
    def reset_episode_counts(self) -> None
```

The table of section 7 is **one dict in this file**, `RULES: dict[ProblemCode, tuple[Kind, str, str]]` (kind, what
failed, what to check). "A new failure rule is one entry in `FailureMonitor`" (PLAN §6.4). `from_exception` maps:
- **`TransportTimeout`:**
  - on `c4.infer/v1` → `POLICY_HUNG`;
  - on `c3.home/v1` → `HOMING_FAILED`;
  - any other → `NO_ANSWER`.
- **`RemoteError`:**
  - with `ErrorReply.field` set (the server found our request broke its contract) → `CONTRACT_VIOLATION`;
  - otherwise, from `policy` → `POLICY_ERROR`;
  - otherwise, from another process → `INTERNAL_ERROR`, with the message naming that process (`<process>: internal
    error: <text>: see its log`).
- **`Refused`:**
  - from `policy` → `POLICY_ERROR`;
  - on `home` → `HOMING_FAILED`.
- **`ContractError`:**
  - on `c4.chunk/v1` → `INVALID_CHUNK`;
  - any other → `CONTRACT_VIOLATION`.
- **`ScriptError`** (in a run only its subclass `ScriptTimeout`, raised by `ScriptedOperator.poll_key`) →
  `OPERATOR_SCRIPT`.
- **Anything else** → `INTERNAL_ERROR`.

**Not a problem:** a cancelled `home` (`RequestCancelled`, raised by `FrankaClient.home` when `contracts`'
`TransportCancelled` comes back because `interrupted()` turned true). It happens only because this process asked for
it (Ctrl-C or SIGTERM), so it never reaches `from_exception` and never becomes `INTERNAL_ERROR`: the home handler
catches it and takes the Ctrl-C or the SIGTERM path (section 5.7).

`check_state` knows `franka_control`'s modes (C2): `idle`, `running`, `holding`, `stopped`, `homing`, `fault`.
- `fault` is a conclude in every phase.
- `stopped` is a conclude only in `RUNNING` when this process has not sent `stop`. After our own `stop` it is the
  normal mode, because `franka_control` goes to `stopped` on `stop`.
- **The `stop_sent` flag:** `EpisodeLoop` sets it when it sends `stop`. It clears it when `start` is acknowledged,
  and records that moment as `start_ack_ns`. `check_state` ignores C2 messages whose `t_ns` is before
  `start_ack_ns`. Those were read before the arm was running again, so a `stopped` still in the queue from the last
  episode is not taken for a stop on its own.
- `homing` is expected only in our `HOMING` phase and is otherwise ignored.
- No C2 message for `state_lost_s` is a conclude.

### 4.8 `EpisodeLoop` (`episode_loop.py`)

```
class Phase(Enum):
    STARTING = "starting"         # startup checks (section 5.1)
    IDLE = "idle"                 # no episode; waiting for a key
    RESETTING = "resetting"       # start key pressed: policy reset (+ prompt), then franka start
    RUNNING = "running"           # an episode: policy steps
    HOMING = "homing"             # blocking home request
    DONE = "done"                 # finished; run() returns

@dataclass(frozen=True)
class LoopProgress:               # what ScriptedOperator may wait on
    phase: Phase
    episode_id: int               # 0 before the first episode
    steps_sent: int               # waypoint sets sent in this episode
    now_ns: int
    worker_busy: bool = False     # a policy step (or reset) still on the worker: c and h are refused meanwhile

class EpisodeLoop:
    def __init__(self, settings: Settings, franka: FrankaClient, health: FramesClient, worker: StepWorker,
                 history: StateHistory, listener: StateListener, monitor: FailureMonitor,
                 operator: Operator, log: SessionLog, clock: Callable[[], int],
                 sleep: Callable[[float], None]) -> None
    def startup(self, ready_file: Path | None) -> None     # section 5.1; on failure sets the verdict, phase DONE
    def cycle(self) -> None                                # one loop cycle (tests call it directly)
    def run(self) -> int                                   # cycles until DONE; returns the exit code (0 or 1)
    def interrupt(self) -> None                            # SIGINT handler: sets the interrupt flag, nothing else
    def interrupt_set(self) -> bool                        # interrupt OR terminate flag; read by FrankaClient.home between slices
    def terminate(self) -> None                            # SIGTERM handler: sets the terminate flag, nothing else
    def progress(self) -> LoopProgress
    @property
    def verdict(self) -> Verdict | None                    # why it finished, for the FATAL line
```

`run()` repeats: `cycle()`, then `worker.wait(time until the next cycle boundary)`. When a result arrives before the
boundary, it is handled **at once** (section 5.4, steps 9–11), so a finished step never waits up to 20 ms for the
next cycle. What one `cycle()` does, in order:

1. Interrupt flag (Ctrl-C) → operator quit; terminate flag (SIGTERM from deploy) → deploy stop (section 5.8).
2. `monitor.check_state(history.latest(), ...)` → conclude if it returns a problem (fault, stopped on its own, state
   lost). Then `listener.error()` set → conclude. In that order, so an arm fault is reported as `ARM_FAULT` even when
   a broken message followed it.
3. `franka.heartbeat()`, unless the phase is `HOMING`.
4. A key from `operator.poll_key(progress)` → handle it (section 5.2).
5. Every `health_poll_ms`: `health.get_health()` → `monitor.check_health` (camera `down` → conclude).
6. If `RUNNING`, the worker is idle and no job was submitted in this cycle → submit the next step.
7. Every `status_ms`: `operator.show_status(StatusLine.format(snapshot))`.

Any exception raised by steps 1–7 is turned into a `Problem` by `monitor.from_exception` and classified. On the
main thread every error is a conclude (C3 or C1 not answering, a broken reply).

### 4.9 `Operator`, `KeyboardOperator`, `ScriptedOperator` (`operator.py`, ...) and `StatusLine`

```
class Key(Enum):
    START = "c"; SUCCESS = "y"; FAIL = "n"; STOP = "s"; HOME = "h"; QUIT = "q"

class Operator(ABC):
    def start(self) -> None                                # e.g. terminal into single-key mode
    def close(self) -> None                                # always restores the terminal
    @abstractmethod
    def poll_key(self, progress: LoopProgress) -> Key | None     # never blocks; ScriptedOperator may raise ScriptTimeout
    def key_refused(self, key: Key, hint: str) -> None     # the last key was ignored; default: nothing
    def show_status(self, line: str) -> None               # rewrites the one status line in place
    def show_message(self, text: str) -> None              # prints a full line above the status line (also logged)

class KeyboardOperator(Operator):
    def __init__(self, stdin: TextIO, stdout: TextIO) -> None

class ScriptedOperator(Operator):
    def __init__(self, entries: Sequence[ScriptEntry], stdout: TextIO, clock: Callable[[], int],
                 wait_limit_s: float, name: str = "script") -> None    # name: the file, in messages
    @classmethod
    def load(cls, path: Path, stdout: TextIO, clock: Callable[[], int],
             wait_limit_s: float) -> "ScriptedOperator"              # raises ScriptError
    def done(self) -> bool
```

**Keys** (one letter, no Enter):

| Key | When | What |
|---|---|---|
| `c` | `IDLE` | start an episode (section 5.3) |
| `y` | `RUNNING` | end the episode: success (section 5.6) |
| `n` | `RUNNING` | end the episode: fail |
| `s` | `RUNNING`, `RESETTING`, `IDLE` | stop: send `stop` (the arm holds), end the episode (if any) as `stopped`, back to `IDLE` with everything still running (PLAN §5 "operator presses `s`: episode ends (normal)") |
| `h` | `IDLE` | home the arm (section 5.7) |
| `q` | any time (not acted on during `HOMING`, section 5.7) | quit: send `stop`, record the episode (if any) as `stopped`, end the run (exit 0). Ctrl-C does the same |

A key that doesn't fit the phase is ignored with a one-line hint, e.g. `h: end the episode first (y / n / s)`, and
`EpisodeLoop` then calls `operator.key_refused(key, hint)`: the keyboard does nothing more (the operator reads the
hint), a script retries the key (below). `s` in `IDLE` is not a refusal: it sends `stop`, and its hint only says no
episode was running. Any other key shows the key list.

**An episode and a run** (PLAN rule 1). An episode runs from `c` until `y`, `n` or `s`. A run is one `deploy run`,
from start-up until `q`, Ctrl-C or an unrecoverable failure. So `y`, `n` and `s` end only the episode, and the run
goes on; `q` and Ctrl-C end the run.

**`EpisodeLoop._on_stop_key()`** holds everything `s` does. It uses the same path as `y` / `n` (section 5.6), only
with outcome `stopped`:

- `RUNNING`: `franka.stop()` (the arm holds; C2 mode `stopped`), `log.end_episode(STOPPED, "operator stop key",
  counts)`, phase `IDLE`. A step still on the worker finishes and its result is dropped (section 5.4, step 9).
- `RESETTING` (start pressed, episode not started yet): `franka.stop()`, phase `IDLE`. The pending reset result is
  dropped, no episode is recorded, and the next `c` resets again.
- `IDLE`: `franka.stop()` (harmless: the arm already holds), with the hint `s: no episode running (c start, h home,
  q quit)`.
- Afterwards, `h` homes the arm and `c` starts the next episode. Nothing is shut down.

- **`KeyboardOperator`:**
  - When stdin is a terminal, it switches it to single-key ("cbreak") mode with `termios`/`tty`. `close()` restores
    the terminal, and it is called in a `finally` and from `atexit`.
  - A daemon thread reads one character at a time into a queue.
  - When stdin is not a terminal (piped), it reads lines and uses the first letter of each.
  - Output goes to stdout: the status line is rewritten in place with `\r` and "clear to end of line". Under
    `deploy run`, stdout is the terminal and stderr is the log (deploy §5.5).
- **`ScriptedOperator`:** replaces the keyboard for tests and milestone 3 (`operator_script` in `config.yaml`). A
  script is a YAML list of entries, run in order:

  ```yaml
  - {wait_phase: idle}     # wait until the loop is in this phase
  - {key: c}
  - {wait_steps: 20}       # wait until this many waypoint sets were sent in the current episode
  - {key: y}
  - {wait_s: 1.0}          # wait this long (by the injected clock)
  - {key: q}
  ```

  The entry kinds are `key`, `wait_phase`, `wait_steps` and `wait_s`. `load` refuses an unknown entry, an unknown
  key, or a script whose last entry is not `{key: q}` (so a script always ends the run; `s` only ends an episode).

  **A script never hangs.** Two rules, both timed by the injected clock with the limit `operator_script_wait_s`
  (section 8, 30 s by default):
  - **Every wait has a time limit.** A `wait_phase` or `wait_steps` entry is timed from the first `poll_key` that
    reaches it. `wait_s` is a timed wait already and has no other limit.
  - **A refused key is retried, not consumed.** After `key_refused`, the script goes back to that key entry. It
    tries the key again as soon as anything a refusal depends on has changed: the phase, the episode, the steps sent
    or `worker_busy` (all in `LoopProgress`). So `c` in `IDLE` while the last policy step is still on the worker
    (`c: waiting for the last policy step to finish`) is sent again in the cycle the step's result is handled, and
    a key refused in a state that never changes is not retried at all (no hint every cycle). The limit counts from
    the key's first try.

  Past the limit, `poll_key` raises `ScriptTimeout` naming the entry and what it waited for, e.g.
  `keys.yaml:5: key h: still refused (h: end the episode first (y / n / s)) after 30.0 s
  (message_comms.operator_script_wait_s)`. The exception reaches `EpisodeLoop`'s guard like any other main-thread
  error, and `from_exception` maps it to `OPERATOR_SCRIPT`: a conclude (section 5.9), so `stop` is sent, an open
  episode is recorded `aborted`, and the run ends with exit 1 and the FATAL line
  `FATAL message_comms: operator script: <that text>: check the script (message_comms.operator_script) or raise
  operator_script_wait_s`. With a policy that uses a prompt, the limit must be above `timeouts.prompt_s`, since a
  `wait_steps` after `c` waits through the whole `RESETTING` phase.

```
@dataclass(frozen=True)
class StatusSnapshot:
    phase: Phase; episode_id: int; episode_s: float; steps_sent: int
    cycle_ms: float | None; policy_ms: float | None; image_age_ms: float | None
    margin_ms: float | None          # section 5.5: last waypoint time of the previous set − send time of the new one
    waiting_policy_ms: float | None  # worker running for (shown when > 2 × typical)
    skips: Mapping[str, int]; late: int; anchor_fallbacks: int; odd_chunks: int
    skipping_s: float | None; skipping_reason: str   # how long steps have been skipped without a break, and why
    shortened: int; pulled_back: int  # C2 gate.shortened / gate.pulled_back, counted since the episode started
    loop_late: int                    # C2 loop_late since the episode started (1 kHz cycles that ran late)
    width_clips: int                  # sum of C4 ActionChunk.width_clipped over the episode's chunks (section 7)
    cameras: HealthReport | None; arm_mode: str | None; anchor_dev_mm: float | None

class StatusLine:
    def format(self, snap: StatusSnapshot) -> str          # one line, at most 120 characters
```

The gate and `loop_late` numbers are cumulative in C2 since `franka_control` started. `EpisodeLoop` stores the values
at episode start and shows the difference. In `IDLE` it shows the session totals.

Examples:

```
[idle] arm stopped | cams ok 30/ok 30 fps | late loops 0 | keys: c start  h home  q quit
[ep 3 running 12.4 s] step 182 | cycle 118 ms (policy 96) | img 74 ms | margin +38 ms | short 4 | pulled 1 | clip 0 | odd 0 | late loops 0 | keys: y n s q
[ep 3 running 14.0 s] waiting for policy 1.9 s | arm holding | keys: y n s q
[ep 3 running 31.0 s] SKIPPING 12.4 s: main camera stale | arm holding | keys: y n s q
[homing] keys inactive; Ctrl-C quits; immediate stop: the robot's external stop button
```

A long run of skipped steps never concludes (PLAN §5: slow → keep running). The status line turns it into the
`SKIPPING <s>: <reason>` form once steps have been skipped for longer than `skip_warn_s` (1 s). The operator sees it
and can stop.

### 4.10 `SessionLog` (`session_log.py`)

```
class Outcome(Enum):
    SUCCESS = "success"; FAIL = "fail"; STOPPED = "stopped"; ABORTED = "aborted"

@dataclass(frozen=True)
class StepRecord:
    episode_id: int; chunk_id: int; t_anchor_ns: int; anchor_source: str; anchor_dev_mm: float
    image_age_ms: float; frames_ms: float; policy_ms: float; cycle_ms: float
    sent: bool; skip_reason: str            # "" when sent
    accepted: int; skipped: int; shortened: int; rejected: int
    margin_ms: float | None

@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int; outcome: Outcome; reason: str
    wall_start: str; wall_end: str; duration_s: float     # ISO 8601 local time; seconds
    steps_sent: int; skips: Mapping[str, int]; late: int; anchor_fallbacks: int; odd_chunks: int
    shortened: int; pulled_back: int; loop_late: int   # C2 deltas over the episode
    width_clips: int                                   # sum of width_clipped over the episode's chunks
    start_pose: list[list[float]]                          # 4×4

class SessionLog:
    def __init__(self, folder: Path, wall_clock: Callable[[], datetime], clock: Callable[[], int]) -> None
    def write_session(self, settings: Settings, policy_config: PolicyConfig) -> None
    def start_episode(self, episode_id: int, start_pose: np.ndarray) -> None
    def log_step(self, record: StepRecord) -> None         # only if settings.log_steps
    def end_episode(self, outcome: Outcome, reason: str, counts: Mapping[str, int]) -> EpisodeRecord
    def log_event(self, kind: str, detail: str) -> None    # startup, home, refused start, conclude, ...
    def save_tap(self, step: StepDone, reply: SetWaypointsReply) -> None   # only if settings.save_tap
    def close(self) -> None
```

Files, all in `<session>/message_comms/` (the process's own folder, deploy §3.2):

| File | Content |
|---|---|
| `session.yaml` | the merged settings and the policy's `config` reply, written once at startup |
| `episodes.jsonl` | one JSON line per finished episode (`EpisodeRecord`) |
| `steps.jsonl` | one JSON line per policy step, sent or skipped (`StepRecord`) |
| `events.jsonl` | one JSON line per event, with `t_ns` and wall time |
| `tap/ep_<NNNN>/chunk_<chunk_id>.npz` | only with `save_tap: true` (off by default): one file per sent step |

Each line is flushed as soon as it is written, so a crash loses nothing already logged. An episode still open when
the process exits is closed as `stopped` or `aborted` before exiting (section 5.9).

**The tap** is for the first real-arm runs: to check offline that the policy saw what we think it saw, and what it
answered. It is one switch, `save_tap`, and is off by default.

- **Each file (uncompressed `np.savez`) holds:**
  - the `Observation`'s low-dim fields (`eef_pos`, `eef_rot6d`, `eef_rot6d_wrt_start`, `gripper_width_m`);
  - the `main_t_ns` and `ultrawide_t_ns` image times;
  - the anchor (`T`, `t_ns`, `source`) and the start pose;
  - the whole `ActionChunk` (all 16 steps);
  - the sent waypoints (`t_ns`, `O_T_TCP`, widths);
  - the C3 reply counts.
- **Images:** the four 224×224 images are added only every `tap_images_every` steps (default 10), because they are
  ~600 KB per step.
- **When:** `EpisodeLoop` calls `save_tap` on the main thread right after it has submitted the next step (section
  5.4, step 11). Writing is then off the critical path: the arm already has its waypoints and the worker is busy. An
  uncompressed write takes about a millisecond.
- **What it needs from the step:** the `Observation` and `ActionChunk` it saves come in `StepDone` (section 4.5).

## 5. The episode lifecycle

### 5.1 Startup (phase `STARTING`)

Everything must be done within `timeouts.startup_s` (20 s, less than the launcher's 30 s ready timeout, so our
own message comes first). Otherwise: conclude with a FATAL line saying which part is missing.

1. `load_settings` (a problem → `FATAL message_comms: config: <problems>: check station.yaml message_comms: and
   message_comms/config.yaml`, exit 1). Open the `SessionLog`. Start the `StateListener` and the `StepWorker`.
2. **Arm state:** wait until `history.latest()` exists, its age is below `max_state_age_ms`, and its mode is not
   `fault`. Then `franka.get_state()` once, which proves C3 answers.
3. **Heartbeats start now** and are sent every cycle from here on (except while homing).
4. **Policy config:** `worker.submit_config()`, then wait (still cycling). Check the reply against what this
   process was built for: the `contracts` constants `OBS_STEPS` (2), `ACTION_STEPS` (16), `STEP_MS` (50) and
   `IMAGE_SIZE` (224). Also `exec_h ≤ ACTION_STEPS`. A mismatch is a contract violation (conclude),
   naming the field: e.g. `policy config: step_ms is 100, message_comms expects 50: check the checkpoint`. Remember
   `uses_prompt`. If it is true and `task_name` is unset, conclude (`check message_comms.task_name`).
5. **Cameras:** `health.get_health()` every `health_poll_ms` until both cameras are `ok`. `down` → conclude.
6. `log.write_session(...)`, `operator.start()`, show the key list on the status line, **write the ready file**
   (deploy §4.4: only now can it do its job). Phase `IDLE`.

### 5.2 Idle (phase `IDLE`)

The loop cycles every `loop_ms`: heartbeat, state checks, health poll, status line. The arm holds where it is.
`franka_control` accepts heartbeats in every mode, and is in `idle` before the first episode, `stopped` after one.
Keys: `c`, `h`, `q` (and `s`, which only sends `stop` here). A camera `down`, an arm `fault` or a
process not answering concludes here too: one rule, whatever the phase.

### 5.3 Start key `c` (phase `RESETTING`)

1. Accepted only in `IDLE` with the worker idle (a step left over from the previous episode may still be finishing;
   then the hint is `c: waiting for the last policy step to finish`).
2. `worker.submit_reset(prompt)`, where `prompt` is `(task_name, prompt_index)` if `uses_prompt` else `None`. A plain
   diffusion policy never receives `prompt`. The loop keeps cycling: `s` cancels the start (back to `IDLE`, no episode), and `q` and Ctrl-C end the run. An error or
   `Ack(ok=false)` → conclude (policy error).
3. `franka.start()`. `franka_control` allows `start` from `idle` or `stopped`, so it normally succeeds. If it is
   refused anyway, show the reason, log a `start_refused` event and go back to `IDLE`: no episode was started, so
   nothing is aborted. A refusal because of `fault` concludes anyway
   through the state check (step 2 of every cycle).
4. **Start pose:** `history.latest()` in the configured `pose_source` (section 6.7). The arm is holding still, so
   this equals the pose at the first image. Increase `episode_id`, `log.start_episode(...)`,
   `monitor.reset_episode_counts()`.
5. Phase `RUNNING`; submit the first step in the same cycle.

### 5.4 One policy step, in detail (phase `RUNNING`)

Steps 1–8 (with 7b) run on the worker (`PolicyStep.run`); steps 9–11 run on the main thread as soon as the result
arrives.

| # | What | Where | Kind if it fails (section 7) |
|---|---|---|---|
| 1 | `get_frames()` (C1). Read `health` from the reply. | worker | C1 timeout → conclude |
| 2 | Images missing (a camera not `ok`): a camera `down` → `StepSkipped(CAMERA_DOWN)`; otherwise `StepSkipped(CAMERA_STALE)`. | worker | down → conclude; stale → slow |
| 3 | `t_a = main_t_ns[1]` (the newest main image). If `t_a ≤ job.last_anchor_t_ns`: no new frame since the last step → `NoNewFrame`. | worker | (not a problem: try again next cycle) |
| 4 | Image age: `now − t_a > max_image_age_ms` → `StepSkipped(IMAGE_OLD)`. | worker | slow |
| 5 | Newest state age `> max_state_age_ms` → `StepSkipped(STATE_OLD)`. For each main time `main_t_ns[0]`, `main_t_ns[1]`: `history.pose_at(t, pose_source)` and the width. A `LookupMiss` → `StepSkipped(STATE_GAP or STATE_OLD)`. If `pose_source` is `target`: also look up the measured pose at `t_a` and apply the fallback rule (section 6.7). | worker | slow |
| 6 | `converter.to_observation(...)` → `Observation` + `Anchor` (section 6.2). | worker | a `ContractError` building our own message is a bug → internal error |
| 7 | `policy.infer(obs)` (C4) with timeout `policy_hang_s`. The reply is decoded and validated by contracts: shapes, types, finite values, unit and perpendicular rot6d. | worker | timeout → policy hung; `ErrorReply` → policy error; invalid → broken chunk: all conclude |
| 7b | **Odd-chunk check** (this process's, PLAN §4): step 0 must be within `odd_chunk_max_pos_m` (2 cm) and `odd_chunk_max_rot_deg` (5°) of the anchor, i.e. `|eef_pos[0]|` and the angle of `matrix_from_rot6d(eef_rot6d[0])`. Failing only this → `StepSkipped(ODD_CHUNK)`: nothing is sent, the arm holds, the chunk is counted. Its `width_clipped` is still added. | worker | slow |
| 8 | `converter.to_waypoints(chunk, anchor, chunk_id)` → `SetWaypoints` with the first `exec_h` steps (section 6.5). Return `StepDone`. | worker | |
| 9 | If the result belongs to an episode that has ended (`job.episode_id` differs from the current one) or the phase is not `RUNNING`: drop it, send nothing. | main | |
| 10 | `franka.set_waypoints(...)` (C3). They **replace** every waypoint not yet reached. Reply: `rejected` not empty → conclude (gate rejected; `franka_control` faults itself). `skipped == N` (every waypoint already in the past; `franka_control` answers `accepted 0, skipped N`, no fault) or a negative margin (section 5.5) → count as `late` (slow: the arm held). `shortened` → normal, counted. Pulled back to the box or cone → normal, seen in C2 `gate.pulled_back`. | main | |
| 11 | **Submit the next step right away** (continuous control), then `log.log_step(...)` and, if `save_tap`, `log.save_tap(...)`. Heartbeats continue every cycle meanwhile. | main | |

A `StepSkipped` that is slow, or a `NoNewFrame`, is logged (skips only). The next step is submitted in the next cycle,
so a skipped step is retried at most every `loop_ms` (20 ms), not in a tight loop. While steps are skipped, the arm
finishes the waypoints it has and holds the last one (PLAN §5 "slow"). However long the skipping lasts, it never
concludes. The status line shows how long it has lasted and why (section 4.9).

**When the previous waypoints haven't run out:** nothing special. The new set replaces the rest at
`franka_control`. That is the normal case in continuous control. **When they have run out:** the arm is already
holding the last one. The new set starts it moving again. Its past waypoints are `skipped` by `franka_control`, and
the arm moves toward the first future one (interpolation from the current target is `franka_control`'s job).

### 5.5 Timing budget

Per step, with typical numbers (to be measured on the real PC):

| Part | Symbol | Typical |
|---|---|---|
| Image latency (capture → PC): mid-exposure of the newest main frame → C1 reply (exposure/2 + Pi encode + Wi-Fi + decode + prep) | `L` | 50–100 ms |
| The time one step takes (frames + policy + conversion): C1 call + conversions + `infer` | `C` | C1 ~5–10 ms, conversions < 1 ms, `infer` the rest |
| Main-thread hand-off: result → `set_waypoints` reply | | < 2 ms |

Step n's waypoints are at `t_a + k × 50 ms` for `k = 0 … exec_h−1`. The last is at `t_a + (exec_h−1) × 50 ms`. They
are sent at about `t_a + L + C`. The next set is sent one step later, at `t_a + L + 2C`. **The arm never holds
if**

```
L + 2C ≤ (exec_h − 1) × STEP_MS          exec_h = 8: L + 2C ≤ 350 ms   (with L = 80 ms: C ≤ 135 ms)
```

This is PLAN §3's condition. `exec_h` counts from step 0, so about 4 of the 8 waypoints are still in the future when
they arrive. If a step is slower, the arm reaches the last waypoint and holds there until new ones arrive: that is
"slow", not a failure. The fix is PLAN §9's backup: a larger `exec_h`.

- The status line shows `margin = (last waypoint time of the previous set) − (send time of the new set)`. A
  negative margin is how long the arm held, and it is counted as `late`.
- Of the 8 waypoints, the first `⌊(L + C) / 50 ms⌋ + 1` are already in the past when they arrive (about 4 with the
  numbers above). `franka_control` counts them as `skipped`. That is expected and is not "late". Section 6.6
  explains why their times are not shifted.

### 5.6 Episode end: `y` / `n`

1. `franka.stop()`: the arm stops following waypoints and holds (C2 mode `stopped`; the next `start` or `home` is
   allowed from there).
2. `log.end_episode(SUCCESS or FAIL, "operator", monitor.counts())`.
3. Phase `IDLE`. A step still running on the worker finishes on its own. Its result is dropped (step 9) and its
   waypoints are never sent. The next `c` waits for it (section 5.3).

### 5.7 Home: `h` (phase `HOMING`)

1. Accepted only in `IDLE` with the worker idle. `franka_control` allows `home` from `idle` or `stopped`, which are
   the only modes `IDLE` can be in.
2. Show the homing status line (section 4.9), then pause heartbeats. Call `franka.home(interrupted=loop.interrupt_set)`;
   `interrupt_set()` is true when the interrupt flag (SIGINT, Ctrl-C) **or** the terminate flag (SIGTERM) is set.
   It waits for the reply for up to `timeouts.home_s` (90 s) in `loop_ms` slices, and checks the interrupt flag
   between slices. While it runs, C2 shows mode `homing`.
   - `franka_control` pauses its watchdog while homing and resumes it when homing ends.
   - Its own homing timeout is 30 s, well under `home_s`, so `franka_control` always answers first.
3. `Ack(ok=true)` → `franka_control` is now `idle`; heartbeats resume in the same cycle, phase `IDLE`, event logged.
   `Ack(ok=false)` or no reply → conclude (`homing failed: <reason>: check the arm's path to the home pose`). The
   arm's position is then unknown, so it is not retried.

**How homing can be interrupted.** Only two ways: the robot's external stop button (immediate), or Ctrl-C.

- **On Ctrl-C or SIGTERM:** `home()` sees the flag within one slice.
  - It drops the reply: it closes its C3 socket with linger 0, so nothing stays queued, and raises
    `RequestCancelled`.
  - The home handler catches `RequestCancelled` and takes the Ctrl-C path (exit 0; an open episode would be recorded
    `stopped`) or the SIGTERM path (exit 0; an open episode would be recorded `aborted` with the reason `stopped by
    deploy: another process exited`). Homing only runs in `IDLE`, so in practice there is no open episode.
  - message_comms then exits at once. It does not send
    `stop`: `franka_control` is busy with `home` and could not answer.
  - The launcher then stops `franka_control` (deploy §5.4), and that stops the arm.
- **Other keys** do nothing during homing.
- **The homing status line** says this: `keys inactive; Ctrl-C quits; immediate stop: the robot's external stop button`.

### 5.8 Stop `s`, quit `q`, Ctrl-C

- **`q` and Ctrl-C (SIGINT), in any phase except `HOMING`:**
  - `franka.stop()` first (harmless when idle, so the exit always sends `stop` first);
  - then record the open episode, if any, as `STOPPED` with the reason (`operator quit key` or `Ctrl-C`);
  - then exit 0 with no FATAL line (deploy §7.6).
  - The launcher then stops the others in reverse order (deploy §5.4–5.5).
  - During `HOMING`, see section 5.7.
- **SIGTERM** is what `deploy run` sends when another process has exited.
  - `franka.stop()` first;
  - then record the open episode, if any, as `ABORTED` with the reason `stopped by deploy: another process exited`;
  - then exit 0 with no FATAL line. The launcher reports the process that actually failed.
- `s` does **not** end the run. It sends `stop`, ends the episode as `STOPPED` and returns to `IDLE` with everything
  still running (section 4.9, `_on_stop_key()`). Then `h` homes and `c` starts the next episode.

### 5.9 Conclude (any phase)

When `FailureMonitor.classify` returns `CONCLUDE`:

1. `franka.stop()`, once, with `timeouts.commands_s`. If that fails too (e.g. `franka_control` is the process that
   died), it is logged and not retried. The watchdog stops the arm when the heartbeats stop.
2. If an episode is open: `log.end_episode(ABORTED, "<what failed>: <detail>", counts)`. `log.log_event("conclude",
   ...)`.
3. Stop heartbeats, stop the listener and the worker (a worker stuck in `infer` is abandoned; it is a daemon thread),
   `operator.close()` (restores the terminal), `log.close()`.
4. Write the FATAL line as the **last stderr line** and exit 1:
   `FATAL message_comms: <what failed>: <what to check>`, e.g.
   `FATAL message_comms: main camera down: check the Pi stream and cable`. `deploy run` quotes it and shuts the
   rest down (deploy §5.3).

All of it takes well under the launcher's 5 s stop timeout. There are no retries anywhere (PLAN §5).

## 6. The conversion math

All math is float64. `float32` only appears at the C4 boundary (the contract's dtypes). Poses are 4×4 rigid
transforms `T = [[R, p], [0 0 0 1]]` in the robot base frame (`O_T_TCP`), row-major (contracts §4).

### 6.1 Pose at an image time (`StateHistory.pose_at`)

For an image time `t` with neighbouring states `s_i` (time `t_i`) and `s_{i+1}` (time `t_{i+1}`), with `t_i ≤ t ≤
t_{i+1}` and consecutive `seq`:

```
a     = (t − t_i) / (t_{i+1} − t_i)                     (0 ≤ a ≤ 1; t exactly on a state: that state)
p(t)  = (1 − a) · p_i + a · p_{i+1}                     linear position
R(t)  = slerp(R_i, R_{i+1}, a)                          scipy.spatial.transform.Slerp, as in robot_utils interpolate_xyz_wxyz
w(t)  = (1 − a) · w_i + a · w_{i+1}                     gripper width, linear
```

`p, R` come from `O_T_TCP_target` or `O_T_TCP` according to `pose_source` (section 6.7). The width is always the
measured `gripper_width_m` (C2 has only that one).

### 6.2 Input: absolute → relative (`PoseConverter.to_observation`)

Two observation steps, `i = 0` (older main image) and `i = 1` (newest), with poses `T_i = pose_at(main_t_ns[i])`.
The anchor is `T_a = T_1`, at time `t_a = main_t_ns[1]`.

```
T_rel,i = inv(T_a) · T_i              inv(T) = [[Rᵀ, −Rᵀ·p], [0 0 0 1]]

eef_pos[i]             = p of T_rel,i = R_aᵀ · (p_i − p_a)        → row 1 is exactly 0
eef_rot6d[i]           = rot6d(R_aᵀ · R_i)                         → row 1 is [1,0,0, 0,1,0]
eef_rot6d_wrt_start[i] = rot6d(R_sᵀ · R_i)                         (section 6.4)
gripper_width_m[i]     = w(main_t_ns[i])                           absolute, not relative
main, ultrawide        = straight from the C1 reply
```

This is training's `convert_pose_mat_rep(pose_mat, base_pose_mat=pose_mat[-1], pose_rep="relative")`
(`behavior_prompting/train_network/dataset/umi_task_dataset.py`), which is `np.linalg.inv(base) @ pose`. It is also
robot_utils' `get_relative_pose(new=T_i, init=T_a)`. In the training sampler the low-dim poses are taken at the
same frame indices as the main images, which is why the poses are looked up at the main image times (not the
ultrawide ones).

### 6.3 rot6d

```
rot6d(R)  = [R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]       the first two ROWS

matrix_from_rot6d(d) :  b1 = d[0:3] / |d[0:3]|
                        b2 = d[3:6] − (b1 · d[3:6]) b1 ;  b2 = b2 / |b2|
                        b3 = b1 × b2
                        R  = rows (b1, b2, b3)
```

This was checked against the code. `behavior_prompting/common/pose_util.py` `mat_to_rot6d` is `mat[..., :2, :]` (rows),
and `rot6d_to_mat` stacks `b1, b2, b3` on `axis=-2` (as rows). robot_utils' `quat_wxyz_to_rot_6d` (`R[0,:], R[1,:]`)
and `mat_to_rot_6d` agree. This matches PLAN §4 ("the first two rows"), so there is **no mismatch**. Note that the
rot6d paper (Zhou et al.) uses the first two *columns*; this code base uses rows, and so do we. The test in section 11
uses a rotation where rows and columns give different numbers, so a mix-up fails.

### 6.4 `wrt_start`

`T_s` is the episode start pose, recorded when the episode starts (section 5.3), in the same `pose_source`.

```
eef_rot6d_wrt_start[i] = rot6d(R_sᵀ · R_i)          only the rotation; training's position part is commented out
```

This is training's `convert_pose_mat_rep(pose_mat, base_pose_mat=start_pose_mat, pose_rep="relative")`, rotation part
(`umi_task_dataset.py`, "generate relative pose with respect to episode or task start"). Training adds noise to the
start pose as augmentation. Deployment does not.

### 6.5 Output: relative → absolute (`PoseConverter.to_waypoints`)

For the chunk's steps `k = 0 … exec_h−1`, with the **same** `Anchor` (`T_a`, `t_a`) that made the observation:

```
R_rel,k   = matrix_from_rot6d(eef_rot6d[k])            (float32 → float64 first)
T_rel,k   = [[R_rel,k, eef_pos[k]], [0 0 0 1]]
T_abs,k   = T_a · T_rel,k                               = [[R_a·R_rel,k, p_a + R_a·eef_pos[k]], [0 0 0 1]]
t_ns[k]   = t_a + k · STEP_MS · 1 000 000 + lookahead_ms · 1 000 000      (integer ns)
width[k]  = gripper_width_m[k, 0]                       passed through unchanged (float32 → float64)
```

This is robot_utils' `get_absolute_pose(init=T_a, rel=T_rel,k)`. `SetWaypoints(chunk_id, t_ns, O_T_TCP, width)` then
validates itself: lengths, increasing times, rigid poses.

**Why the same anchor matters.** The policy predicted "move like this, starting from where you were at `t_a`". If the
output used a newer pose (the arm has moved since `t_a`), the whole plan would be shifted by that motion. real-env's
`policy_agent` with `action_as_proprio` used one anchor for the observation and another for the output; we don't.

### 6.6 Waypoint times and lookahead: `lookahead_ms: 0`

In training, action step `k` is the pose at the frame `k × 50 ms` after the newest observation frame (the sampler
slices `action[current_idx : … : down_sample_steps]` from the same index as the newest observation). So **the time of
step k is `t_a + k × 50 ms`**, measured from the newest image. No shift is needed to keep that meaning.

- With `lookahead_ms: 0` the steps that are already in the past when they arrive are skipped by `franka_control`, and
  the arm follows the rest at the times the policy meant. This is how UMI runs continuous control (it drops actions
  whose time has passed).
- A positive lookahead shifts every waypoint later. Fewer would be skipped, but the arm would then always run
  `lookahead_ms` behind the plan, and behind the world the images showed.
- The key stays in `config.yaml` (integer ms, ≥ 0) so it can be tried on the real arm without a code change.

### 6.7 Which pose: `pose_source: target`, with a fallback to measured

C2 has both `O_T_TCP` (measured) and `O_T_TCP_target` (what the 1 kHz loop is tracking).

**How training poses were made.** The human carried the gripper; the poses are its real trajectory (SLAM). The images
and the poses agree, and the human's hand had no "tracking error": where they meant to be and where they were are the
same.

**On the Franka the two differ.** Cartesian impedance control leaves a small, mostly steady error `e` between target
and measured (gravity or payload model error, friction, cables): typically a few mm.

- **Anchor on measured:** each new set is `T_meas · T_rel`. With a "hold still" chunk the new target is the
  measured pose, i.e. `e` short of the old target. The arm then settles `e` short of *that*. So the arm **creeps by
  about `e` every step**, several mm/s at ~7 steps/s.
- **Anchor on target:** `T_target · T_rel` = `(T_meas + e) · T_rel`. The steady error is carried along, so the
  *measured* pose follows the relative plan, which is what training taught. And because the target history is
  built from our own waypoints, this already gives what real-env's `action_as_proprio` was for. So **no separate
  action-as-proprio mode is needed** for a plain diffusion policy.
- **The risk of target:** in contact (pushing on the table or the block), the error is not steady but grows. Each new
  plan, anchored on a target that is already inside the obstacle, pushes further ("wind-up"). **Fallback:** when
  `|p_target − p_measured| > anchor_fallback_m` (2 cm, main-sim's value) at `t_a`, that step uses the measured
  poses for all of its rows and its anchor. It is counted (`anchor_fallbacks`) and shown on the status line. The
  gate's speed limit smooths the jump back toward the measured pose.
- The observation's history rows and the anchor always come from the **same** source in one step. Otherwise the last
  row would not be exactly identity.

This is also main-sim's choice (`anchor: target`, fallback to measured at 2 cm). It matters from the first real-arm run: the hold-still policy is the one used there, and with measured
anchoring it would creep. `pose_source: measured` stays available as one setting.

The **gripper width** in the observation is the measured width: C2 has no commanded width, and training widths
were measured too.

### 6.8 Worked example

Rotations about z only, so the numbers can be checked by hand. `Rz(θ)` is a rotation by θ about z.

**Interpolation.** States at `t = 1000 ms`: `p = (0.4950, 0, 0.30)`, `R = Rz(89.0°)`; and at `1010 ms`:
`p = (0.5000, 0, 0.30)`, `R = Rz(90.0°)`, with consecutive `seq`. An image at `1004 ms` gives `a = 0.4`,
`p = (0.4970, 0, 0.30)`, `R = Rz(89.4°)`.

**Input.** Anchor `T_a`: `p_a = (0.50, 0.00, 0.30)`, `R_a = Rz(90°)`. Older step `T_0`: `p_0 = (0.49, 0.00, 0.30)`,
`R_0 = Rz(85°)`. Start pose `T_s`: `R_s = Rz(80°)`.

| Field | Row 0 (older) | Row 1 (anchor) |
|---|---|---|
| `eef_pos` | `R_aᵀ(p_0 − p_a) = Rz(−90°)·(−0.01, 0, 0) = (0.00, 0.01, 0.00)` | `(0, 0, 0)` |
| `eef_rot6d` | `Rz(−5°)` rows: `[0.99619, 0.08716, 0, −0.08716, 0.99619, 0]` | `[1, 0, 0, 0, 1, 0]` |
| `eef_rot6d_wrt_start` | `Rz(5°)` rows: `[0.99619, −0.08716, 0, 0.08716, 0.99619, 0]` | `Rz(10°)` rows: `[0.98481, −0.17365, 0, 0.17365, 0.98481, 0]` |

(Columns instead of rows would give `[0.98481, 0.17365, 0, −0.17365, 0.98481, 0]` for `Rz(10°)`: the signs of the
off-diagonal terms swap. That is exactly what the rows-vs-columns test catches.)

**Output.** The chunk's step `k = 4` is `eef_pos = (0.02, 0, 0)` and a rotation of `Rz(10°)`:
`p = p_a + R_a·(0.02, 0, 0) = (0.50, 0.02, 0.30)` (the anchor's own x axis points along the base's y),
`R = Rz(90°)·Rz(10°) = Rz(100°)`, rows `[−0.17365, −0.98481, 0]`, `[0.98481, −0.17365, 0]`, `[0, 0, 1]`. With
`t_a = 10.000 s`, its time is `10.000 + 4 × 0.050 = 10.200 s`.

**Timing.** `exec_h = 8`: times 10.000, 10.050, …, 10.350 s. With `L = 80 ms` and `C = 110 ms` they are sent at
about 10.190 s. Waypoints 0–3 (10.000–10.150 s) are `skipped`, and 4–7 are followed. The next set is sent at about
10.300 s, before 10.350 s, so the margin is +50 ms and the arm never holds.

## 7. `FailureMonitor`: every condition

"Skip" means: no waypoints from this step. The arm finishes what it has and holds, heartbeats continue, and the next
step is tried in the next cycle. "Conclude" means section 5.9: `stop`, record `aborted` with the reason, FATAL line,
exit 1. Rows marked "PLAN" are PLAN §5's table; the others are this design's additions.

| Code | Condition | Detected by | Kind | Action / message |
|---|---|---|---|---|
| `CAMERA_STALE` | PLAN: a camera `stale` | C1 `get_frames` reply: no images, a camera `stale`, none `down` | slow | skip; resumes when both are `ok` |
| (same) | PLAN: a dropped frame, or none within the picking tolerance | `process_frames` reports that camera as `stale` for that reply, even though frames keep arriving (C1 has no other way to say it) | slow | skip |
| (same, long) | skipped steps that go on and on | the time since the last sent step while steps keep being skipped | slow | never concludes (PLAN §5); status line `SKIPPING <s>: <reason>` after `skip_warn_s` |
| `NO_NEW_FRAME` | the newest main image is the one already used | `main_t_ns[1] ≤` previous anchor time | none | wait for the next cycle (not counted as a skip) |
| `IMAGE_OLD` | the newest main image is older than `max_image_age_ms` | `now − main_t_ns[1]` | slow | skip (a sanity check; `process_frames` should already report `stale`) |
| `POLICY_LATE` | PLAN: policy reply slower than the waypoints it has | C3 reply `skipped == N`, or margin < 0 (section 5.5) | slow | nothing to do: the arm held; count `late` |
| `STATE_GAP` | PLAN: a missed state message (C2 `seq` gap) | `StateHistory.pose_at` → `SEQ_GAP` around an image time | slow | skip |
| `STATE_OLD` | PLAN: the newest state older than its limit | newest C2 `t_ns` older than `max_state_age_ms`, or `pose_at` → `TOO_NEW` / `TOO_OLD` / `EMPTY` | slow | skip |
| `SHORTENED` | PLAN normal operation: a waypoint shortened to the speed limit | C3 reply `shortened`; C2 `gate.shortened` | normal | count, status line `short N` |
| `PULLED_BACK` | PLAN normal operation: a waypoint just outside the box or cone (within the margin) pulled back to the boundary | C2 `gate.pulled_back` (episode delta) | normal | count, status line `pulled N` |
| `WIDTH_CLIPPED` | PLAN normal operation: a policy width outside [0, max opening] clipped by `policy` | C4 `ActionChunk.width_clipped` (uint64, always present: the number of widths `policy` clipped in that chunk) | normal | add it to the episode's sum (every received chunk, sent or not); status line `clip N`, episode record `width_clips` |
| `LOOP_LATE` | 1 kHz cycles that ran late (mainly with the ignore-real-time backup) | C2 `loop_late` (episode delta) | normal | status line `late loops N` |
| `ODD_CHUNK` | PLAN: an odd chunk: step 0 more than 2 cm / 5° from the anchor | the odd-chunk check (section 5.4, step 7b) | slow | skip that step (the arm holds), count it; status line `odd N`, episode record `odd_chunks` |
| `ANCHOR_FALLBACK` | target and measured more than `anchor_fallback_m` apart | section 6.7 | normal | use measured for that step; count |
| `CAMERA_DOWN` | PLAN: a camera `down`, or the Pi connection lost | `status == down` in any C1 reply (`get_frames` or `get_health`) | conclude | `<camera> camera down: check the Pi stream and cable` |
| (same) | PLAN: a Pi stream restart (that camera's `seq` or `t_pi_ns` goes backwards) | `process_frames` latches that camera `down`: seen here as `status == down` | conclude | as above |
| (same) | PLAN: clock sync with the Pi lost for > 10 s | `process_frames` stops using frames, so the cameras go `stale`, then `down`: seen here first as skips, then as `status == down` | conclude | as above |
| `POLICY_HUNG` | PLAN: no reply for > hang limit | `TransportTimeout` on `c4.infer/v1` after `policy_hang_s` (5 s) | conclude | `policy: no reply to infer within 5.0 s: check the policy log and the GPU` |
| `POLICY_ERROR` | PLAN: the policy returns an error | `RemoteError` on any C4 request, or `Refused` on `reset` / `prompt` | conclude | `policy: <error text>: check the policy log` |
| `INVALID_CHUNK` | PLAN: a broken chunk (fails the contract validator) | `ContractError` decoding `c4.chunk/v1` | conclude | `policy: invalid chunk: <field>: <detail>: check the checkpoint and policy.kind` |
| `ARM_FAULT` | PLAN: `franka_control` reports `fault` | C2 `mode == fault` (any phase) | conclude | `franka_control fault: <error>: check the arm (reflex, connection); a fault is never cleared automatically` |
| `GATE_REJECTED` | PLAN: a waypoint the gate rejects (outside the box or cone by more than the margin, width outside [0, max opening], non-finite) | C3 reply `rejected` not empty (`franka_control` is then in `fault`) | conclude | `franka_control: gate rejected waypoint <i>: <reason>: check the workspace box and the policy` |
| `ARM_STOPPED` | the arm went to `stopped` by itself during an episode | C2 `mode == stopped` in `RUNNING` when this process has not sent `stop` | conclude | `franka_control stopped on its own (watchdog?): check its log` |
| `NO_ANSWER` | PLAN: any process stops answering | `TransportTimeout` on any C1 or C3 request, or on C4 `config` / `reset` / `prompt` | conclude | `<process>: no reply to <request> within <t> s: check that <process> is running (see its log)` |
| `STATE_LOST` | PLAN: `franka_control` stops broadcasting | no C2 message for `state_lost_s` (1 s) | conclude | `franka_control: no state for 1.0 s: check that franka_control is running` |
| `CONTRACT_VIOLATION` | PLAN: a message fails its contract (schema version, shape, type) | `ContractError` on any C1/C2/C3 message, or a non-chunk C4 reply; client extra checks (4.2); `seq`/`t_ns` not increasing (4.3); policy config values (5.1) | conclude | `<message>: field '<f>': <detail>: check that all processes are from the same commit` |
| `HOMING_FAILED` | homing ends without reaching home | `Refused` on `home`, or `TransportTimeout` on `c3.home/v1` (no reply within `home_s`) | conclude | `franka_control: homing failed: <reason>: check the arm's path to the home pose` |
| `STARTUP` | PLAN: not ready at start-up | section 5.1 not complete within `startup_s` | conclude | e.g. `startup: no camera ok within 20 s (main stale): check process_frames` |
| `OPERATOR_SCRIPT` | a `ScriptedOperator` wait, or a key `EpisodeLoop` keeps refusing, not done within `operator_script_wait_s` | `ScriptTimeout` from `poll_key` (section 4.9) | conclude | `operator script: <file>:<line>: <what it waited for> after <t> s (message_comms.operator_script_wait_s): check the script (message_comms.operator_script) or raise operator_script_wait_s` |
| `INTERNAL_ERROR` | a bug in this process | an unexpected exception, the worker or listener thread died | conclude | `internal error: <exception>: see the message_comms log` |
| (episode end) | PLAN: operator presses `s` | key `s` | normal (episode ends) | `stop` (the arm holds), record `stopped`, back to `IDLE`; the run continues (section 4.9) |
| `OPERATOR_STOP` | PLAN: operator presses `q` or Ctrl-C | key `q` (any phase), SIGINT | stop | `stop`, record the open episode as `stopped`, exit 0, no FATAL line; the run ends (during homing: section 5.7) |
| `DEPLOY_STOP` | `deploy run` stops this process because another one exited | SIGTERM | stop | `stop`, record the open episode as `aborted` (`stopped by deploy: another process exited`), exit 0, no FATAL line |

Not failures: `franka_control` refusing `start` for a reason other than `fault` (section 5.3), and the expected
`skipped` count of past waypoints (section 5.5).

## 8. `config.yaml`

Every setting, with its default. `station.yaml`'s `message_comms:` section may override any of them (deploy §6.2:
`exec_h` and `operator_script` are the ones PLAN asks to be selectable). Times are in the unit in their name.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `exec_h` | int, 1–16 | `8` | waypoints sent per policy step (PLAN §9 backup: raise it if the policy is slower than the budget of section 5.5) |
| `lookahead_ms` | int, ≥ 0 | `0` | added to every waypoint time (section 6.6) |
| `pose_source` | `target` \| `measured` | `target` | which C2 pose is the proprio and the anchor (section 6.7) |
| `odd_chunk_max_pos_m` | float, > 0 | `0.02` | the odd-chunk check: step 0's largest distance from the anchor (PLAN §4) |
| `odd_chunk_max_rot_deg` | float, > 0 | `5.0` | the odd-chunk check: step 0's largest rotation from the anchor |
| `anchor_fallback_m` | float, > 0 | `0.02` | with `target`: use measured for a step when target and measured are farther apart than this |
| `loop_ms` | int, > 0 | `20` | main loop period = heartbeat period (50 Hz) |
| `status_ms` | int, > 0 | `250` | status line refresh |
| `health_poll_ms` | int, > 0 | `500` | `get_health` period, in every phase |
| `history_s` | float, > 0 | `1.0` | length of the C2 state history |
| `max_state_age_ms` | int, > 0 | `50` | newest state older than this → skip the step (5 messages at 100 Hz) |
| `state_lost_s` | float, > 0 | `1.0` | no C2 message for this long → conclude |
| `max_image_age_ms` | int, > 0 | `250` | newest main image older than this → skip (with `exec_h` 8, an older anchor leaves under 100 ms of future waypoints) |
| `state_recv_hwm` | int, > 0 | `1000` | C2 subscriber queue length (10 s at 100 Hz; every message is needed for the `seq` check) |
| `timeouts.commands_s` | float, > 0 | `0.2` | C3 `heartbeat`, `start`, `stop`, `set_waypoints`, `get_state` |
| `timeouts.health_s` | float, > 0 | `0.2` | C1 `get_health` (main thread) |
| `timeouts.frames_s` | float, > 0 | `1.0` | C1 `get_frames` (worker) |
| `timeouts.policy_hang_s` | float, > 0 | `5.0` | C4 `infer`: the hang limit (PLAN §5) |
| `timeouts.config_s` | float, > 0 | `5.0` | C4 `config` |
| `timeouts.reset_s` | float, > 0 | `1.0` | C4 `reset` (the policy must answer in < 0.5 s; the rest is margin) |
| `timeouts.prompt_s` | float, > 0 | `60.0` | C4 `prompt` (loading a prompt; never sent to a plain diffusion policy) |
| `timeouts.home_s` | float, > 0 | `90.0` | C3 `home`; longer than `franka_control`'s `homing.timeout_s` (30 s), so `franka_control` always answers first (checked by `launch.py`) |
| `timeouts.startup_s` | float, > 0 | `20.0` | section 5.1; below the launcher's 30 s ready timeout |
| `task_name` | str \| null | `null` | for `prompt`; required only if the policy says `uses_prompt` |
| `prompt_index` | int, ≥ 0 | `0` | for `prompt` |
| `operator_script` | path \| null | `null` | `null`: the keyboard. A path: a `ScriptedOperator` script (section 4.9). Relative paths as in deploy §6.2 |
| `operator_script_wait_s` | float, > 0 | `30.0` | with a script: the longest a `wait_phase` / `wait_steps` entry, or a key that keeps being refused, may take before the run concludes (`OPERATOR_SCRIPT`, section 4.9) |
| `skip_warn_s` | float, > 0 | `1.0` | steps skipped for longer than this without a break → the status line switches to `SKIPPING <s>: <reason>` (never a conclude) |
| `log_steps` | bool | `true` | write `steps.jsonl` |
| `save_tap` | bool | `false` | save each sent step's observation, chunk and waypoints to `tap/` (section 4.10); for the first real-arm runs |
| `tap_images_every` | int, ≥ 1 | `10` | with `save_tap`: include the four images every this many steps |

Two rules tie these to other processes. Both are written as comments in the file:

- **Heartbeat gap.** In one cycle the main thread may wait on a C3 request and a C1 `get_health`. So the longest it
  can go without a heartbeat (outside homing) is `loop_ms + commands_s + health_s` = 0.02 + 0.2 + 0.2 = 0.42 s. This
  must stay below `franka_control`'s `watchdog.limit_s` (0.5 s). `launch.py` checks it against the real value
  (section 9).
- **Homing.** `timeouts.home_s` must be above `franka_control`'s `homing.timeout_s`. `launch.py` checks this too.
- **Camera stale/down limits** are not here. They live only in `process_frames/config.yaml`, and this process trusts
  the `status` in C1, so the two can never disagree.

The sizes fixed by C1/C4 are not settings (deploy §7.6): `contracts.OBS_STEPS` (2), `ACTION_STEPS` (16), `STEP_MS`
(50 ms) and `IMAGE_SIZE` (224) are used wherever a size or the step time is needed.

## 9. `launch.py`

```
# message_comms/launch.py
from deploy import ManagedProcess, CheckResult, check_helpers

class MessageCommsProcess(ManagedProcess):
    NAME = "message_comms"
    READY_HINT = ("connected to franka_control, process_frames and policy; fresh arm state, both cameras ok, "
                  "policy config accepted; check the other three processes' logs and endpoints in station.yaml")
    USES_TERMINAL = True
    def program(self) -> list[str]        # [self.station.python("light"), "-m", "message_comms"]
    def checks(self) -> list[CheckResult]

PROCESS_CLASS = MessageCommsProcess
```

- **`USES_TERMINAL = True`:** its stdin and stdout are the launcher's terminal and only stderr goes to the log. It stays
  in the launcher's process group, so a Ctrl-C reaches it and it sends `stop` first (deploy §5.5).
- **Ready:** the default `is_ready` (the ready file). The file is written at the end of section 5.1: connected to all
  three, fresh state, both cameras `ok`, policy config accepted, key list shown.
- **`extra_env`:** none.
- **`checks()`** (quick, starts nothing; `--mock` changes nothing, since this is the real process under test, deploy
  §8):

| Check | Status on failure | How |
|---|---|---|
| its Python imports `message_comms`, `contracts`, `numpy`, `scipy`, `yaml` | fail | `check_helpers.python_imports(...)` with `python.light` |
| `operator_script` exists, when set | fail | `check_helpers.file_exists` |
| a terminal for the keyboard when `operator_script` is null | fail: `no terminal for the keyboard: run deploy run in a terminal, or set message_comms.operator_script` | `sys.stdin.isatty()` in the launcher |
| `exec_h` in 1–16, `pose_source` one of the two words, every timeout and odd-chunk limit > 0, `tap_images_every ≥ 1` | fail |
| heartbeat gap and homing against `franka_control`'s real settings: `loop_ms/1000 + commands_s + health_s < watchdog.limit_s` and `home_s > homing.timeout_s`, both read with `deploy.load_process_config(DEPLOYMENT_ROOT / "franka_control", self.station.section("franka_control"))` (section 8) | fail, naming both values | read from `self.config` and the loaded `franka_control` config |
| `exec_h`, `pose_source`, operator, tap | info | e.g. `exec_h = 8 (0.40 s per step; PLAN §9 backup)`, `pose_source = target`, `operator: keyboard`, `tap: off` |

It does **not** probe the other three processes' endpoints. At `deploy check` time nothing is started yet, and at
`deploy run` the launcher only starts `message_comms` after the other three are ready (deploy §5.1). It binds no ports,
so there is no port check.

## 10. `__main__` and running it alone

```
python -m message_comms --station FILE [--session DIR] [--ready-file FILE] [--mock]
```

The common arguments of deploy §4.2, and only those. `main(argv: Sequence[str] | None = None) -> int`:

1. Parse arguments. Set up `logging` to stderr in the shared format (deploy §7.6).
2. `load_settings(station, session, mock)`. `--mock` is accepted and logged and changes nothing (deploy §8).
3. Build in this order: `SessionLog`, `StateHistory`, `StateListener`, `FrankaClient`, the health `FramesClient`, and
   the worker's `FramesClient` + `PolicyClient` + `PoseConverter` + `PolicyStep` + `StepWorker`, `FailureMonitor`,
   the `Operator` (`ScriptedOperator.load(..., operator_script_wait_s)` when `operator_script` is set, else
   `KeyboardOperator(sys.stdin, sys.stdout)`), `EpisodeLoop`.
4. Install the signal handlers: SIGINT calls `loop.interrupt()` (operator quit), SIGTERM calls `loop.terminate()`
   (deploy stop, section 5.8). Both only set a flag.
5. `loop.startup(ready_file)`, then `code = loop.run()`.
6. In a `finally`: close the operator (terminal restored), stop the threads, close the sockets.
7. If `code == 1`: write `FATAL message_comms: <what failed>: <what to check>` as the last stderr line. Return `code`.

**Every failure exit has a FATAL line**, also when nothing was built yet:
- a settings problem: `FATAL message_comms: config: <problems>: check station.yaml message_comms: and
  message_comms/config.yaml`;
- a broken script: `FATAL message_comms: operator script: <file>:<line>: <problem>: check message_comms.operator_script`;
- a session folder `SessionLog` cannot write in: `FATAL message_comms: session folder <dir>: cannot write: <error>:
  check that the session folder (--session, or station.yaml session_root) is writable`;
- **any other exception** (building the clients, a bug): steps 2–6 run inside `_run(args)`, and `main` catches every
  `Exception` that leaves it, after `_run`'s `finally` has cleaned up. It logs the traceback (`log.exception`, so it
  is in `logs/message_comms.log`), then writes `FATAL message_comms: internal error: <type>: <message>: see the
  traceback in the message_comms log` and returns 1. So `deploy run` never reports "no FATAL line" for this process
  unless it was killed.

Exit codes: `0` for the operator's quit, a SIGINT or a SIGTERM; `1` for every failure (a conclude or a start-up
failure), always with a FATAL line. Like the other processes (deploy §7.6 asks only for a non-zero code), this
process never uses `2`: that is `deploy`'s own "not started" code.

**Running it alone** (PLAN rule 2) needs the other three already running on the station's endpoints, each started
alone in its own terminal (e.g. `python -m process_frames --station station.yaml --mock`, `python -m policy --station
station.yaml --mock`, and the `franka_control` binary with `--mock`). Then in another terminal:
`python -m message_comms --station deployment/station.yaml`. It makes its own session folder
`<session_root>/<timestamp>_message_comms_alone/`, writes into its `message_comms/` subfolder (the same layout as in
a run) and writes no ready file. Its log lines (stderr) and the status line
(stdout) share that terminal. If one of the three is missing, startup fails after `startup_s` naming it.

## 11. Unit tests

All in `message_comms/tests/`, run with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest message_comms/tests` (deploy §7.4).

- No Pi, arm or GPU.
- Sockets only on `tcp://127.0.0.1:*`.
- No sleeping to wait: `FakeClock`, `ManualWorker` (results are released by the test) and condition timeouts.

`tests/reference_pose_utils.py` holds copies, unchanged, of the training-side functions:

- `get_relative_pose`, `get_absolute_pose`, `qmult`, `qconjugate`, `interpolate_xyz_wxyz`, `to_xyzw`, `to_wxyz`,
  `xyz_wxyz_to_pose_9d`, `pose_9d_to_xyz_wxyz`, `mat_to_rot_6d`, `rot_6d_to_mat` from `robot-utils==0.1.4`
  `robot_utils/pose_utils.py`;
- `mat_to_rot6d`, `rot6d_to_mat` from `behavior_prompting/common/pose_util.py`.

It has a provenance header (`changes: only the functions listed kept`). It is test-only, so the light environment
does not need `robot-utils`.

| File | Checks |
|---|---|
| `test_pose_math.py` | `invert` is the inverse; `rot6d_from_matrix` takes **rows** (a rotation where rows ≠ columns, e.g. `Rz(10°)`, gives `[0.98481, −0.17365, 0, 0.17365, 0.98481, 0]`); `matrix_from_rot6d` round trip; a non-orthogonal 6-vector is made orthonormal (Gram-Schmidt); `interpolate` at a = 0, 0.4, 1 (section 6.8 values), slerp of 90° at a = 0.5 is 45°; equal to `reference_pose_utils` on 1000 random poses (relative, absolute, rot6d both ways, interpolation) within 1e-9 |
| `test_pose_converter.py` | **round trip:** absolute → relative → absolute gives the input back (1e-9); **known values:** the section 6.8 example field by field; **rotation, not just translation:** a pure-rotation and a mixed case; **same anchor both ways:** `to_waypoints` uses the `Anchor` from `to_observation`, not a newer pose (a test changes the history between the two calls; the output doesn't move); **`wrt_start`:** the section 6.8 values, and identity when the start equals the pose; last rows exactly 0 / `[1,0,0,0,1,0]` (passes the `Observation` validator); dtypes float32 in, float64 out; waypoint times `t_a + k·50 ms + lookahead` as int64; only the first `exec_h` steps; width passed through unchanged |
| `test_state_history.py` | interpolation between two states (linear position, slerp, linear width); an exact hit; `TOO_OLD`, `TOO_NEW`, `EMPTY`; **a `seq` gap** around the image time gives `SEQ_GAP`, a gap elsewhere doesn't; trimming to `history_s`; `target` vs `measured` read the right field; a `seq` that doesn't increase or a `t_ns` that goes back raises `ContractError`; **an equal `t_ns`** (a repeated reading) updates `latest()` (mode, error) but adds no sample, and is not taken for a gap; `latest()` and `gap_count()`; one thread adding while another reads (no exception, consistent results) |
| `test_state_listener.py` | a real `Publisher` on a free port: states arrive in the history in order; a broken message sets `error()` and ends the thread; `stop()` returns within `poll_s` |
| `test_clients.py` | each client against a `contracts.RpcServer` with fake handlers on a free port: good replies come back as their classes; **every client rejects a reply that breaks its contract** (wrong schema version, wrong shape, wrong reply type, C1 images while a camera is `down`, an `ErrorReply` → `RemoteError`); `Ack(ok=false)` → `Refused` (except `start`, which returns it); `set_waypoints` refuses a wrong `chunk_id` echo and counts that don't add up; a server that doesn't answer → `TransportTimeout` after the configured timeout, sent once |
| `test_policy_step.py` | with fake clients and a filled `StateHistory`: a normal step gives the expected `SetWaypoints`; camera `stale` → `StepSkipped(CAMERA_STALE)`; `down` → `StepSkipped(CAMERA_DOWN)`; same anchor time as last → `NoNewFrame`; old image → `IMAGE_OLD`; old state → `STATE_OLD`; seq gap → `STATE_GAP`; target/measured > 2 cm → measured used, `anchor_dev_m` reported; **odd chunk** (step 0 at 2.5 cm, or 6°) → `StepSkipped(ODD_CHUNK)`, while 1.9 cm / 4.9° passes; policy timeout / error / broken chunk propagate as exceptions |
| `test_step_worker.py` | one job at a time (a second `submit` while busy raises); results come back in order with timing; an exception in a job comes back as `error`; `reset` then `prompt` when a prompt is given, `reset` only otherwise |
| `test_episode_loop.py` | with fakes, `FakeClock`, `ManualWorker` and `ScriptedOperator`, driving `cycle()`. **The next chunk is asked while the current one runs:** the next step is submitted in the same cycle the waypoints are sent, before their last time. **Waypoints replace correctly:** chunk ids increase, each set is sent once, a result from an ended episode is dropped. **Heartbeats during slow steps:** a step held back for 2 s still gives one heartbeat per cycle. `s` or `q` while a step runs sends `stop` in that same cycle. `home` pauses heartbeats and they resume after. `prompt` is sent only when `uses_prompt`, after `reset`, before `start`. A refused `start` returns to idle with no episode. Ctrl-C → `stop`, then exit 0. SIGTERM → `stop`, episode `aborted` with `stopped by deploy: another process exited`, exit 0. Ctrl-C during homing (fake `home` that never answers) → exits within one slice, socket closed. A `stopped` C2 message older than the `start` acknowledgement does not conclude; one after it does. A fault together with a broken C2 message is reported as `ARM_FAULT`. **The `s` key:** in `RUNNING` it sends `stop`, records `stopped` and returns to `IDLE` with the run still going; then `h` homes and `c` starts episode 2 (the episode counter goes up, `reset` is sent again); in `RESETTING` it cancels the start without recording an episode; in `IDLE` it only sends `stop`. `q` and Ctrl-C end the run with exit 0. Long skipping never concludes. `home` shows the homing status line. Gate and `loop_late` deltas are taken from the episode start. With `save_tap` the tap is written after the next step is submitted, without it never. Startup: the ready file is written only after state, config and cameras are ok; a config mismatch concludes. Episode records are written with the right outcome. **`run()` with a script:** `c, wait_steps 5, s, h` with a step still in flight at `s`: `h` gets its hint, is retried and homes once, then episode 2 runs; a `wait_steps` that never ends concludes with `OPERATOR_SCRIPT` (exit 1, `stop` sent, the episode `aborted`) |
| `test_failure_monitor.py` | **every row of section 7 (and so of PLAN §5)** gives its stated kind and message (one parametrized case per row: the detection input → `Problem` → `Verdict`); `from_exception` maps each exception type and request as in 4.7; `check_state` for `fault`, self-`stopped` in `RUNNING` (and not after our own `stop`, nor from messages before the `start` acknowledgement), `homing` only in `HOMING`, state lost; the normal-operation rows (`SHORTENED`, `PULLED_BACK`, `WIDTH_CLIPPED`, `LOOP_LATE`) never stop anything; `width_clipped` is summed per episode and reset at the next start; counts and their per-episode reset |
| `test_operator.py` | `ScriptedOperator` runs whole episodes (c → wait_steps → y → q) against `LoopProgress` values; `wait_s` with `FakeClock`; a script with two episodes (c → y, c → s, h, c → n, q) runs to the end; a bad script (unknown entry or key, or a last entry other than `q`, e.g. ending on `s`) is refused with the line; **time limits:** a `wait_phase` / `wait_steps` that never ends raises `ScriptTimeout` naming the line and what it waited for exactly at the limit; `wait_s` has no limit; **refused keys:** retried once the state changes and not before, timed out when refused for too long (the limit counting from the first try), a `key_refused` for another key ignored; `KeyboardOperator` key mapping from a pipe (no terminal); a hint for a key that doesn't fit the phase |
| `test_status_line.py` | idle, running, waiting, `SKIPPING` and homing lines as in 4.9; `short`, `pulled`, `clip`, `odd` and `late loops` shown; never longer than 120 characters; negative margin shown |
| `test_session_log.py` | the four files appear in the session folder; each episode outcome and its fields; every line is flushed at once; an open episode is closed on exit; the tap: no `tap/` folder when `save_tap` is false; with it, one `.npz` per sent step holding the listed arrays, images only every `tap_images_every` steps |
| `test_settings.py` | `config.yaml` loads; a station override merges (maps key by key); an unknown key is refused with its path; every range check; no `--session` makes the `_alone` folder |
| `test_flow.py` | PLAN §7 "message flow": the real `message_comms` (`main()`) against `fake_servers.py` (a fake `process_frames`, a hold-still `policy` and a fake `franka_control` with C3 server + C2 publisher, all on free ports) with a script. One episode runs: waypoints arrive with increasing `chunk_id`, heartbeats keep coming, `stop` is sent first at exit, exit 0. It covers every row of section 7 (PLAN §7: the full normal step and every row of PLAN §5).
- **Slow cases** (keep running, exit 0 at the end): a camera `stale`, then resume (steps stop, then start again); a C2 state gap (that step skipped); an odd chunk (skipped and counted, the next chunk sent).
- **Operator keys:** `s`, then `h`, then `c` (episode 1 `stopped`, home, episode 2 runs); `q` (exit 0, `stop` sent first).
- **Concludes** (each giving exit 1 and the expected FATAL line, with `stop` sent first): camera `down`; policy hangs (with `policy_hang_s` set short in the test settings); a policy `ErrorReply`; a broken chunk; arm `fault`; gate rejection; a process that stops answering (the fake `process_frames` stops replying); a schema-version mismatch; a broken C2 message.
- **Not here:** the shutdown order of the other processes is `deploy/`'s `test_launcher.py`. |
| `test_main.py` | arguments: the four common ones, `--station` required, `--mock` accepted; the FATAL line is the last stderr line on exit 1 and absent on exit 0; a script key refused for longer than `operator_script_wait_s` gives exit 1 and the `operator script:` FATAL line, `stop` sent last; an unwritable session folder gives exit 1 and the `session folder` FATAL line; any other exception while building gives exit 1, the traceback in the log and the `internal error` FATAL line |
| `test_launch.py` | (deploy §9) `program()`; `USES_TERMINAL` is true; `checks()` fails for a missing `operator_script`, no terminal without a script, `exec_h` 0 or 17, a heartbeat gap ≥ `franka_control`'s `watchdog.limit_s` (from a test `franka_control/config.yaml`), `home_s ≤ homing.timeout_s`; mock checks need no other process |
