# deploy/

The design of the launcher, the `station.yaml` schema, the interface every `launch.py` implements, and the
conventions all folders follow. Written from PLAN.md sections 2, 3, 5, 6.5, 7 and 9. PLAN.md is the source of truth:
where this file and PLAN.md disagree, PLAN.md wins and this file is fixed.

## 1. Role

`deploy/` is **the launcher, not a process**. It gives the two commands of PLAN rule 1:

- `deploy check`: is everything ready? Runs every process's checks and prints one table.
- `deploy run`: checks, then starts the four processes in order, waits until each is ready, waits while they run, and
  stops them all in reverse order when any of them exits.

It knows nothing about frames, poses or the arm. Everything process-specific is in that process's `launch.py`; the
launcher only runs those descriptions. The one exception is the small **geometry tool** (`python -m deploy.geometry`,
section 6.6): it computes the rig file's `T_cam_tcp` from `gripper.yaml` and `franka_control`'s `F_T_TCP`, and
`deploy check` checks those three files still agree. Its job is small, and so is its code: eight small classes, five
for the launcher (`Station`, `Session`, `CheckResult`, `ManagedProcess`, `Launcher`) and three for the geometry tool
(`GripperFile`, `RigTcp`, `TcpTriangle`), plus a few pure functions.

Terms used below:

- **process group**: a set of OS processes that receive a signal together. Each child is started in its own group, so
  stopping it also stops anything it started (a grandchild), and so a Ctrl-C in the terminal reaches only the processes
  the launcher wants it to reach.
- **SIGINT**: the signal Ctrl-C sends. **SIGTERM**: the signal a program sends another to ask it to stop (what the
  launcher sends). Both mean "please stop cleanly". **SIGKILL**: the signal that ends a process without asking.
- **ready file**: an empty file a process creates when it is ready to work. The launcher waits for it.
- **ABC** (abstract base class): a class with methods marked `@abstractmethod`; a subclass must write them, or Python
  refuses to create it.

## 2. File tree

```
deploy/
  STRUCTURE.md
  pyproject.toml          package "deploy"; console script `deploy = deploy.cli:main`
  deploy/
    __init__.py           DEPLOYMENT_ROOT (the deployment/ folder), public names
    __main__.py           `python -m deploy` = the `deploy` command
    cli.py                argument parsing: check | run, --station, --mock
    errors.py             StationError, LaunchError, GeometryError
    station.py            Station: loads and checks station.yaml
    config_merge.py       load_process_config(), resolve_path(): config.yaml + the station's overrides (section 6.3)
    session.py            Session: the session folder and its paths
    check_result.py       CheckResult, and the table printer
    check_helpers.py      small reusable checks for launch.py files (file exists, port free, imports, probe)
    managed_process.py    ManagedProcess: the base class every launch.py subclasses
    loader.py             load_process_class(): imports <folder>/launch.py by path
    launcher.py           Launcher: check() and run()
    geometry/             the gripper geometry tool, `python -m deploy.geometry` (section 6.6)
      __init__.py
      __main__.py         the tool's command line
      gripper_file.py     GripperFile: loads and checks deployment/gripper.yaml
      frames.py           pure functions: camera_pose(), invert(), pose_error()
      rig_tcp.py          RigTcp: reads and writes tcp.T_cam_tcp in a rig file
      tcp_triangle.py     TcpTriangle: gathers the three inputs; compute() for the tool, check() for deploy check
  tests/
    data/                 example station files, tiny fake launch.py files, tiny scripts used as fake processes
    ...                   section 9
```

## 3. The classes

### 3.1 `Station` (`station.py`)

Loads `station.yaml` and checks its shared parts (section 6). Each process's own section is checked against that
process's `config.yaml` (section 6.3), so `Station` doesn't need to know any process's keys.

```
class Station:
    @classmethod
    def load(cls, path: Path | None, mock: bool) -> "Station"   # None: DEPLOYMENT_ROOT / "station.yaml"; mock: section 6.1
    path: Path                                              # the station file, absolute
    session_root: Path
    ready_timeout_s: Mapping[str, float]                    # per process name
    stop_timeout_s: float
    poll_s: float
    tcp_check_max_mm: float                                 # the TCP triangle check's limits (6.6)
    tcp_check_max_deg: float
    def python(self, env: str) -> str                       # "light" or "policy": the interpreter path
    def endpoint(self, name: str) -> str                    # e.g. "commands" -> "tcp://127.0.0.1:5630"
    def section(self, process: str) -> Mapping[str, Any]    # that process's overrides ({} if absent)
    def resolve(self, value: str) -> Path                   # "~" expanded; relative paths against the station file's folder

def parse_endpoint(endpoint: str) -> tuple[str, int]        # "tcp://host:port" -> (host, port); ValueError otherwise
```

`load` takes `mock` because a few keys are needed only without it (`python.policy`, `endpoints.pi_*`; section 6.1).
It collects **every** problem before raising one `StationError` that lists them all, each with its key path, e.g.
`station.yaml: endpoints.policy: expected "tcp://host:port", got "localhost:5640"`. Unknown top-level keys are errors
(they are almost always typos).

### 3.2 `Session` (`session.py`)

The folder for one `deploy run`.

```
class Session:
    @classmethod
    def create(cls, root: Path, mock: bool, station_file: Path) -> "Session"
    path: Path                         # <session_root>/<YYYY-MM-DD_HH-MM-SS>[_mock]/
    def log_path(self, name: str) -> Path      # path/logs/<name>.log
    def ready_file(self, name: str) -> Path    # path/ready/<name>.ready
    def process_dir(self, name: str) -> Path   # path/<name>/ : where that process writes its own outputs
```

`create` makes the folders and copies the station file in as `path/station.yaml`, so every session records the
settings it ran with. The launcher's own messages go to the terminal and to `logs/deploy.log`.

### 3.3 `CheckResult` (`check_result.py`)

```
@dataclass(frozen=True)
class CheckResult:
    process: str          # "policy", or "station" for the station file itself
    name: str             # short, e.g. "checkpoint"
    status: str           # "pass" | "warn" | "fail" | "info"
    detail: str           # one line: what was found, and on failure what to check

def print_table(results: Sequence[CheckResult], out: TextIO) -> None
def exit_code(results: Sequence[CheckResult]) -> int        # 2 if any fail, else 0
```

`warn` does not stop `deploy run` (e.g. "TCP triangle off by …" under `--mock`). `fail` does. `info` reports a fact, e.g. which
backups are active (PLAN §9: "`deploy check` reports which ones are active").

### 3.4 `ManagedProcess` (`managed_process.py`)

The base class for one process. It does everything generic: build the full command line, start it in its own process
group with its log captured, wait for ready, report its exit, stop it. A subclass (in each folder's `launch.py`) only
says **what** to run and **how to check it**: section 4 is that interface.

What the base class gives every subclass:

```
class ManagedProcess(ABC):
    def __init__(self, station: Station, mock: bool, session: Session | None) -> None
                                     # session is None for `deploy check` (nothing is started); the Launcher
                                     # sets `session` after the checks, before start()
    station: Station
    mock: bool
    session: Session | None
    folder: Path                     # DEPLOYMENT_ROOT / NAME
    config: Mapping[str, Any]        # NAME/config.yaml merged with station.section(NAME) (section 6.3)
    log_path: Path | None            # session.log_path(NAME) (a property: follows `session`)
    ready_file: Path | None          # session.ready_file(NAME) (a property)

    def config_path(self, key: str) -> Path      # a path setting, resolved by where its value came from (section 6.3)
    def command(self) -> list[str]   # program() + the common arguments (section 4.2); not overridden
    def base_checks(self) -> list[CheckResult]   # overrides valid, program found (section 4.3)
    def start(self) -> None
    def wait_ready(self, timeout_s: float, poll_s: float) -> None    # raises LaunchError
    def returncode(self) -> int | None             # None while running
    def stop(self, timeout_s: float) -> int | None # SIGTERM, wait timeout_s, then SIGKILL; returns the exit code
    def fatal_line(self) -> str | None             # the last "FATAL ..." line of its log (section 7.6)
    def failure_message(self, code: int, during_startup: bool = False) -> str   # the one-line report of 5.3
```

A broken station section for this process (an unknown override key) does not raise in `__init__`: it is reported by
`base_checks()` as a `fail` named `overrides`, and the launcher then skips that process's `checks()`, so `deploy
check` still lists every other problem.

- `start` opens `log_path`, writes the command as its first line, and starts the command with `subprocess.Popen`:
  stdout and stderr to the log, stdin closed, `cwd=folder`, its own process group, environment = the launcher's +
  `PYTHONUNBUFFERED=1` (so log lines appear at once) + `extra_env()`. The one exception is the terminal process
  (section 4.1, `USES_TERMINAL`). **`cwd` is never `deployment/`** (it is the process's own folder, also for the
  terminal process): `python -m <pkg>` puts its working folder first on the path, and from `deployment/` the folder
  `deployment/<pkg>/` (or `deployment/contracts/`) is imported as an empty namespace package before the editable
  install (section 7.2). For the same reason `check_helpers.python_imports` runs its subprocess with `cwd="/"`.
- `wait_ready` calls `is_ready()` every `poll_s` until it returns true. It raises `LaunchError` if the process exits
  first, or if `timeout_s` passes (section 5.3 gives the messages).
- `stop` sends **SIGTERM** to the process group, waits up to `timeout_s`, then sends SIGKILL to the group. For the
  `USES_TERMINAL` process (`message_comms`) it signals **the process's pid only, never a process group**: that
  process shares the launcher's group (5.5), and signalling the group would hit the launcher itself. Stopping a process
  that has already exited just returns its code.

### 3.5 `Launcher` (`launcher.py`)

```
class Launcher:
    ORDER = ("franka_control", "process_frames", "policy", "message_comms")   # PLAN §6.5; not a setting
    def __init__(self, station: Station, mock: bool,
                 processes: Sequence[ManagedProcess] | None = None) -> None   # processes: injected by tests
    def check(self) -> int          # prints the table; 0 if nothing failed, else 2
    def run(self) -> int            # section 5; returns the exit code
```

Without injected processes, `Launcher` builds them with `loader.load_process_class(DEPLOYMENT_ROOT / name)` for each
name in `ORDER`. `check()` also runs the TCP triangle check (section 6.6), reported under the name `geometry`.

### 3.6 `loader.py`

```
def load_process_class(folder: Path) -> type[ManagedProcess]
```

Imports `folder/launch.py` **by file path** (with `importlib.util.spec_from_file_location`, under the module name
`launch_<folder name>`), and returns its `PROCESS_CLASS`. It raises `LaunchError` if the file is missing, if
`PROCESS_CLASS` is missing or not a `ManagedProcess` subclass, or if `PROCESS_CLASS.NAME` differs from the folder name.

`launch.py` is loaded by path because it is not part of its folder's package: the package may need libraries the
launcher's environment doesn't have (torch for `policy`, PyAV for `process_frames`).

### 3.7 `cli.py`

```
deploy check [--station FILE] [--mock]
deploy run   [--station FILE] [--mock]
```

Only these two options (PLAN §6.5). `--station` defaults to `deployment/station.yaml`. `--mock` uses every process's
stand-in (section 8). `main(argv: Sequence[str] | None = None) -> int`. A `StationError` prints its list of problems and
exits with code 2.

## 4. The `launch.py` interface (what each process folder writes)

Each process folder has a `launch.py` beside its package. It defines **one** subclass of `ManagedProcess` and ends with
`PROCESS_CLASS = <that class>`.

**Imports allowed in `launch.py`:** the Python standard library, `yaml`, `deploy`, `contracts`. **Never its own
package** or any other process folder (a test checks this, section 9). It runs in the launcher's light environment.

### 4.1 What a subclass declares and writes

| Member | Kind | Required | Meaning |
|---|---|---|---|
| `NAME: ClassVar[str]` | class attribute | yes | the folder name and station section: `"process_frames"`, `"policy"`, `"franka_control"`, `"message_comms"` |
| `READY_HINT: ClassVar[str]` | class attribute | yes | what "ready" waits for and what to check if it never comes, e.g. `"checkpoint load + warm-up inference; check policy.checkpoint and the GPU"`. Shown on a ready timeout. |
| `USES_TERMINAL: ClassVar[bool]` | class attribute | no (default `False`) | `True` only for `message_comms` (section 5.5) |
| `program(self) -> list[str]` | abstract method | yes | the program part of the command: `[self.station.python("light"), "-m", "process_frames"]`, or `[str(binary)]` for `franka_control`. The base class appends the common arguments (4.2). Must not start anything. |
| `checks(self) -> list[CheckResult]` | abstract method | yes | this process's checks for `deploy check` (4.3). Quick: every check has its own timeout, the slowest being `policy`'s self-check (up to 20 s). Never starts the process, never moves the arm, and sends no contract messages. Uses `self.mock` to skip hardware checks. |
| `extra_env(self) -> dict[str, str]` | method | no (default `{}`) | extra environment variables, e.g. `CUDA_VISIBLE_DEVICES` |
| `is_ready(self) -> bool` | method | no (default: `self.ready_file.exists()`) | one quick probe, called every `poll_s`; none of the four overrides it. May raise `LaunchError` for a failure that won't fix itself, so the launcher doesn't wait for the timeout. |

The base class's `start`, `wait_ready`, `stop` and `command` are not overridden.

**There is no separate "health while running" method.** While running, the launcher only watches for exits. PLAN §5
already puts "stops answering" with the processes that talk to each other: `message_comms`' clients time out and it
exits with a FATAL line, and if `message_comms` itself dies, `franka_control`'s watchdog stops the arm. A second health
prober in the launcher would duplicate that and add a client on every socket.

### 4.2 The common arguments (every process's `__main__` must accept them)

`command()` = `program()` + these, in this order:

| Argument | Given by `deploy run` | Meaning |
|---|---|---|
| `--station FILE` | always (absolute path) | the station file: endpoints, its own overrides |
| `--session DIR` | always | the session folder; the process writes its outputs under `DIR/<name>/` |
| `--ready-file FILE` | always | create this file (empty) once ready to work (4.4) |
| `--mock` | only with `deploy run --mock` | use this process's stand-in (section 8) |

When a process is started alone for troubleshooting (`python -m policy --station station.yaml [--mock]`), only
`--station` is needed. With no `--session`, it uses `<session_root>/<YYYY-MM-DD_HH-MM-SS>_<name>_alone/` as its
session folder and writes into `<that folder>/<name>/`, the same layout as in a run. With no `--ready-file` it writes
none. The C++ binary takes the same four arguments.

### 4.3 Checks

`deploy check` runs the TCP triangle check (6.6), then, for each process in `ORDER`: `base_checks()` then
`checks()`. `deploy run` runs the same checks
first and refuses to start if any fails (PLAN §5, last row).

`base_checks()` (given):
- the station's overrides for this process are valid against its `config.yaml` (section 6.3);
- the first element of `program()` exists and is executable.

What each `checks()` should cover (each folder's STRUCTURE.md lists its own): required files (rig file, checkpoint,
binary, the fake Pi's recorded episode under `--mock`); that its Python can import its package and key libraries (in a subprocess, since it may be
another environment); that the ports it binds are free; that remote hosts are reachable when not `--mock` (a TCP connect to the Pi's
endpoints and to the arm's IP); and one `info` line per active backup (PLAN §9). No `launch.py` sends a contract
message: whether a peer *answers correctly* is checked by each process itself at start-up, and a failure there is
its FATAL line (5.3).

Helpers (`check_helpers.py`), all returning `CheckResult`:

```
def file_exists(process: str, label: str, path: Path) -> CheckResult
def executable(process: str, label: str, path: Path) -> CheckResult
def port_free(process: str, endpoint: str) -> CheckResult
def python_imports(process: str, python: str, modules: Sequence[str], timeout_s: float) -> CheckResult
def host_reachable(process: str, host: str, port: int, timeout_s: float) -> CheckResult     # TCP connect only
def info(process: str, name: str, detail: str) -> CheckResult
def warn(process: str, name: str, detail: str) -> CheckResult
```

### 4.4 Ready

**Rule for every process:** create the ready file only when it can do its job. Before that, it must not answer
requests as if it were ready.

| Process | Writes its ready file when | `is_ready` |
|---|---|---|
| `franka_control` | connected to the arm (or `FakeArm`), holding the current pose, C2 publishing and C3 answering | default |
| `process_frames` | C1 is answering and both cameras have been `ok` at least once | default |
| `policy` | checkpoint loaded and the warm-up inference done, C4 answering | default |
| `message_comms` | connected to all three, `policy` `config` received and accepted, the operator prompt shown | default |

All four use the ready file, not a probe of their sockets: it is one rule that works the same in Python and C++, and
only the process itself knows when its warm-up is done. A process that finds a lasting problem while starting (e.g.
the arm in `fault`) does not write the file: it writes its FATAL line and exits, and the launcher reports it at once.

### 4.5 A minimal `launch.py`, as signatures only

```
# process_frames/launch.py
from deploy import ManagedProcess, CheckResult, check_helpers

class ProcessFramesProcess(ManagedProcess):
    NAME = "process_frames"
    READY_HINT = "frames from both cameras; check the Pi stream, Wi-Fi and process_frames.rig_file"
    def program(self) -> list[str]: ...
    def checks(self) -> list[CheckResult]: ...

PROCESS_CLASS = ProcessFramesProcess
```

## 5. `deploy run`, step by step

### 5.1 Start

1. `Station.load` (errors → exit 2).
2. Build the four processes (`--mock` passed to each).
3. Run all checks and print the table. Any `fail` → `deploy: not started: N check(s) failed (see above)`, exit 2.
4. `Session.create`.
5. For each name in `ORDER`: `start()`, then `wait_ready(station.ready_timeout_s[name], station.poll_s)`.
   The next process is started only after the previous one is ready, because each one needs the earlier ones:
   `franka_control` first (the arm holds safely before anything else exists), then `process_frames`, then `policy`,
   then `message_comms` (which connects to all three).
6. `deploy: all 4 processes ready (session: <path>)`.

If step 5 fails for any process, the ones already started are stopped in reverse order (5.4), and `deploy run` exits
with code 1 and one message (5.3).

### 5.2 Wait

Every `poll_s`, check `returncode()` of each process. The first one that has exited decides how the run ends:

| What happened | Exit code | Message |
|---|---|---|
| `message_comms` exited with code 0 (operator quit: `q`, see 5.5) | 0 | `deploy: done (session: <path>)` |
| Ctrl-C or SIGTERM to the launcher | 130 | `deploy: stopped by Ctrl-C` |
| any process exited otherwise | 1 | section 5.3 |

### 5.3 Messages

One line, always naming the process, what failed, and where to look. The "what failed" part is the process's own
FATAL line when it wrote one (section 7.6), so the message tells what to check, e.g.:

```
deploy: FAILED message_comms (exit 1): FATAL message_comms: main camera down: check the Pi stream and cable  [log: <session>/logs/message_comms.log]
deploy: FAILED policy during start-up (exit 1): FATAL policy: checkpoint not found: /x.ckpt: check policy.checkpoint in station.yaml  [log: ...]
deploy: FAILED process_frames: not ready after 30 s (waiting for: frames from both cameras; check the Pi stream, Wi-Fi and process_frames.rig_file)  [log: ...]
deploy: FAILED franka_control (exit -9, no FATAL line): see the last lines of its log  [log: ...]
```

### 5.4 Stop

All started processes are stopped **in reverse order**: `message_comms`, `policy`, `process_frames`,
`franka_control`, each with `stop(station.stop_timeout_s)` (SIGTERM, then SIGKILL). After a Ctrl-C, `message_comms`
has already received the SIGINT itself (5.5), so the launcher first waits up to `stop_timeout_s` for it to exit on its
own, and sends SIGTERM only if it is still running. The order matters: `message_comms` sends `stop` to the arm
as it exits (PLAN §5), and `franka_control` goes last so the arm is the last thing to lose its controller. Each stop and
its exit code is written to `logs/deploy.log`.

A second Ctrl-C during the stop sends SIGKILL to everything still running and exits at once. The arm is still safe:
when `franka_control` dies, libfranka's connection drops and the robot's own controller stops the arm.

### 5.5 The terminal (keyboard and status line)

`message_comms` owns the keyboard and the status line (PLAN §6.4), so it alone gets the terminal:

- It is started with `USES_TERMINAL = True`: its stdin and stdout are the launcher's terminal; only its stderr goes to
  its log. The status line is therefore not in any log; the episode results are logged by `message_comms` itself
  (its `SessionLog`, under `<session>/message_comms/`). It stays in the launcher's process group, because a program in another group that reads the terminal is
  frozen by the OS.
- So a Ctrl-C reaches both the launcher and `message_comms`. `message_comms` treats it like `q`. The launcher waits
  for it to exit, then stops the others in reverse.
- **Operator keys** (read by `message_comms`, never by the launcher): `c` start an episode, `y` success, `n` fail,
  `s` stop, `h` home, `q` quit.
  - `s` stops the arm and ends **only the current episode** (recorded as stopped by the operator). All four processes
    keep running: `h` homes the arm and `c` starts the next episode. The launcher sees nothing, since no process
    exits.
  - `q` and Ctrl-C **end the run**: `message_comms` sends `stop`, records a running episode as **`stopped`**, and
    exits with code 0. The launcher then stops the others in reverse (5.4) and exits 0 (`q`) or 130 (Ctrl-C), as in
    5.2.
- **A SIGTERM from the launcher** (another process failed, or the launcher itself got SIGTERM) is different from the
  operator ending the run: `message_comms` sends `stop`, records a running episode as **aborted, with the reason**
  (`"shut down by deploy (SIGTERM)"`; the launcher's own line in `logs/deploy.log` names the process that caused it),
  and exits.
- The other three are in their own process groups, so a Ctrl-C does not hit them directly: they are stopped by the
  launcher, in order.
- `Launcher` refuses to run if more than one process sets `USES_TERMINAL`.

## 6. `station.yaml` and `gripper.yaml`

Two hand-written files sit at the top of `deployment/` (PLAN rule 4):

| File | Holds | Read by |
|---|---|---|
| `station.yaml` | every setting `deploy check` and `deploy run` need: hosts, ports, paths, which policy, the process overrides (6.1–6.4) | `Station`, every process |
| `gripper.yaml` | our hand-entered gripper constants: where the ultrawide camera sits, in the **hand** frame, and how it is tilted (6.5) | only the geometry tool and its `deploy check` (6.6) |

`station.yaml` is small: shared things at the top level, then one section per process holding only that process's
**overrides** of its own `config.yaml`.

### 6.1 Shared keys (checked by `Station`)

| Key | Type | Default | Read by |
|---|---|---|---|
| `station_version` | int | required, must be `1` | `Station` |
| `session_root` | path | **required** | `Session`; each process when started alone |
| `python.light` | path | the Python running `deploy` | `process_frames`, `message_comms` launch.py, `policy` launch.py under `--mock` (and `franka_control` checks) |
| `python.policy` | path | **required**, except with `--mock` | `policy` launch.py; used only for `kind: diffusion` (it must exist then) |
| `endpoints.pi_frames` | `tcp://host:port` | **required**, except with `--mock` | C0 frames: the Pi binds PUB; `process_frames` connects |
| `endpoints.pi_time_sync` | `tcp://host:port` | **required**, except with `--mock` | C0 clock sync: the Pi binds REP; `process_frames` connects |
| `endpoints.mock_pi_frames` | `tcp://host:port` | **required** | `--mock`: the fake Pi binds; `process_frames` connects |
| `endpoints.mock_pi_time_sync` | `tcp://host:port` | **required** | `--mock`: the fake Pi binds REP |
| `endpoints.frames` | `tcp://host:port` | **required** | C1: `process_frames` binds REP; `message_comms` connects |
| `endpoints.state` | `tcp://host:port` | **required** | C2: `franka_control` binds PUB; `message_comms` connects |
| `endpoints.commands` | `tcp://host:port` | **required** | C3: `franka_control` binds REP; `message_comms` connects |
| `endpoints.policy` | `tcp://host:port` | **required** | C4: `policy` binds REP; `message_comms` connects |
| `launcher.ready_timeout_s.<name>` | float, > 0 | `franka_control: 30`, `process_frames: 30`, `policy: 300`, `message_comms: 30` | `Launcher` |
| `launcher.stop_timeout_s` | float, > 0 | `5.0` | `Launcher` |
| `launcher.poll_s` | float, > 0 | `0.1` | `Launcher` |
| `launcher.tcp_check_max_mm` | float, > 0 | `2.0` | the TCP triangle check (6.6) |
| `launcher.tcp_check_max_deg` | float, > 0 | `0.5` | the TCP triangle check (6.6) |
| `franka_control`, `process_frames`, `policy`, `message_comms` | map | `{}` | that process (6.2, 6.3) |

**Every endpoint and `session_root` is required, with no default anywhere** (not in `Station`, not in any
`config.yaml`), so each port is written in exactly one place: this file. Under `--mock`, `endpoints.pi_*` are not
needed (the fake Pi binds `endpoints.mock_pi_*`, and nothing connects to the Pi), and `python.policy` is not needed
(`HoldStillPolicy` runs in the light environment); the `mock_pi_*` endpoints are required in both modes, so one
station file works for `deploy run` and `deploy run --mock`. The C1–C4 endpoints are on `127.0.0.1` (PLAN §4: local
ports on the PC). Paths are expanded (`~`) and relative
paths are taken from the station file's folder.

### 6.2 Per-process sections (checked against each `config.yaml`)

These are the keys PLAN asks to be selectable in `station.yaml` (the backups of PLAN §9, which policy, which rig
file). Their defaults live in each folder's `config.yaml`; the final names are set by each folder's STRUCTURE.md, but
the process planners should use these unless they have a reason not to.

| Key | Type | Default (in that `config.yaml`) | Meaning |
|---|---|---|---|
| `franka_control.binary` | path | `build/franka_control` (relative to the folder) | the built C++ program |
| `franka_control.robot_ip` | str | `172.16.0.2` (the Panda's usual FCI address) | the arm's FCI address; not used with `--mock` |
| `franka_control.realtime` | `enforce` \| `ignore` | `enforce` | PLAN §9 real-time backup |
| `process_frames.rig_file` | path | `rigs/rig_pi_v2.yaml` (in the folder) | the rig file (the only one: main 1600×1200 @ 30, ultrawide 1280×960 @ 30); its `tcp:` block is written by `python -m deploy.geometry` (§6.6) |
| `process_frames.decoder` | `h264` \| `jpeg` | `h264` | PLAN §9 frame-encoding backup |
| `process_frames.mock_episode` | path | required with `--mock` | the recorded Pi episode whose frames `FakePiSender` sends as C0 |
| `policy.kind` | `diffusion` \| `hold_still` | **required, no default** (`null` in `config.yaml`; `deploy check` fails while it is unset, except with `--mock`) | which policy runs (PLAN §6.2: one line) |
| `policy.checkpoint` | path | `null`; required when `kind: diffusion` | the trained checkpoint |
| `policy.device` | `cuda` \| `cpu` | `cuda` | PLAN §9 device backup |
| `policy.diffusion_steps` | int \| null | `null` (the checkpoint's own) | PLAN §9 fewer-steps backup |
| `message_comms.exec_h` | int | `8` | PLAN §9 waypoints per step |
| `message_comms.operator_script` | path \| null | `null` (keyboard) | scripted keys instead of the keyboard (tests, milestone 3) |

**Every station-selectable key, with its default, is in that process's `config.yaml`** (the process planners put the
keys of this table there). Any other key of a process's `config.yaml` (rates, limits, gains) may also be overridden
here, but nothing else needs to be. **Relative paths** resolve by where the value came from: a value set in the station file resolves against the
station file's folder; a default from `config.yaml` resolves against the process's own folder. `launch.py` tells them
apart with `station.section(NAME)`: `ManagedProcess.config_path(key)` returns `station.resolve(value)` when `key` is in
`station.section(NAME)`, else `folder / value`. Each process's own loader applies the same rule.

### 6.3 How overrides merge (`config_merge.py`)

```
def load_process_config(folder: Path, overrides: Mapping[str, Any]) -> dict      # raises StationError
def resolve_path(station: Station, process: str, config: Mapping[str, Any], key: str) -> Path
    # station-set value: station.resolve(value); config.yaml default: DEPLOYMENT_ROOT / process / value
```

`ManagedProcess.config_path(key)` is `resolve_path(self.station, NAME, self.config, key)`; the geometry tool uses the
same function for `process_frames`' rig file (6.6).

- Read `folder/config.yaml`, then merge the station section over it: maps merge key by key, all other values (lists,
  numbers, strings) are replaced.
- **Every override key must already exist in `config.yaml`.** An unknown key is an error with its path, e.g.
  `station.yaml: policy.chekpoint: not a key of policy/config.yaml`. So `config.yaml` is the full list of a process's
  settings, with their defaults.
- The launcher uses this for `ManagedProcess.config` and for `base_checks()`. **Each process applies the same rule
  when it starts** (its own small loader; it cannot import `deploy`), so a process started alone sees the same values.
  The rule is short enough to restate rather than share; a test in each process checks it with a bad override.

### 6.4 Example

```yaml
station_version: 1
session_root: ~/deploy_sessions
python:
  light: ~/envs/deploy/bin/python                               # your path
  policy: ~/miniconda3/envs/behavior_prompting/bin/python       # your path: the behavior_prompting conda env
endpoints:
  pi_frames: tcp://192.168.1.50:5600          # the Pi's address on the Wi-Fi
  pi_time_sync: tcp://192.168.1.50:5601
  mock_pi_frames: tcp://127.0.0.1:5600        # --mock: the fake Pi inside process_frames
  mock_pi_time_sync: tcp://127.0.0.1:5601
  frames: tcp://127.0.0.1:5610                # C1
  state: tcp://127.0.0.1:5620                 # C2
  commands: tcp://127.0.0.1:5630              # C3
  policy: tcp://127.0.0.1:5640                # C4
franka_control:
  realtime: ignore            # no PREEMPT_RT kernel yet (PLAN §9)
process_frames:
  mock_episode: ~/rig_episodes/ep_0003
policy:
  kind: hold_still            # until a rig-trained diffusion checkpoint exists (PLAN §6.2)
```

`franka_control.robot_ip` is left out: its default `172.16.0.2` is in `franka_control/config.yaml`.

### 6.5 `gripper.yaml`

`deployment/gripper.yaml` holds **only the constants we type in by hand** from our own measurements or design (PLAN
§6.6). There is no CAD import. It describes where the **ultrawide (IMX219) camera** sits on the gripper: the camera SLAM
tracks during training.

**It is written in the hand frame `H`, not the flange frame.** The Franka Hand is mounted rotated −45° on the flange,
so "the top of the gripper" is a diagonal in flange axes, while it is a plain axis in the hand's own axes. Frames
(PLAN §6.6):

- **flange `F`:** Franka's convention; origin at the centre of the plate, z out of the plate toward the fingers.
- **hand `H`:** the flange rotated −45° about its z axis. Same origin and z; y is the direction the fingers close; x is
  "up/down" on the gripper.
- **camera:** OpenCV optical (z out of the lens, x right in the image, y down in the image).

The real file, `deployment/gripper.yaml` (its comments carry TODOs to measure the position exactly and confirm the
side):

```yaml
ultrawide:
  position_mm: [55.0, 0.0, 10.0]   # lens centre in the hand frame, mm: 55 mm "up" from the centre line, 10 mm out from the plate
  up_side: +x                      # the hand-frame side the camera sits on, "up" in the image (the top of the gripper)
  tilt_deg: 15.0                   # looks 15° down toward the fingers, from the hand z axis
```

| Key | Type | Check |
|---|---|---|
| `ultrawide.position_mm` | list of 3 numbers, mm, **hand frame** | finite |
| `ultrawide.up_side` | string, **hand frame** | one of `+x`, `-x`, `+y`, `-y` (it must be perpendicular to the z axis) |
| `ultrawide.tilt_deg` | number, degrees | `0 ≤ tilt_deg < 90` |

All three are required; any other key is an error, with its path (`gripper.yaml: ultrawide.tilt: unknown key`).
`GripperFile.load(path: Path) -> GripperFile` (a frozen dataclass: `position_m: np.ndarray (3,)`, `up_axis:
np.ndarray (3,)`, `tilt_deg: float`, `path: Path`) collects every problem before raising one `GeometryError`.

**The finger length is not in this file.** It lives only in `franka_control`'s `tcp.F_T_TCP` (its `config.yaml`:
−45° about flange z, translation `[0, 0, 0.150]`: the fingertips are 150 mm from the plate, the Franka Hand's
0.1034 m plus our printed fingers), so no number is written twice.

### 6.6 The geometry tool (`python -m deploy.geometry`)

Run by hand during setup and after any hardware change (PLAN §6.6). It computes the rig file's `T_cam_tcp`, which is
**calculated, never typed in**, from the two hand-set constants:

```
python -m deploy.geometry [--station FILE] [--gripper FILE] [--dry-run]
```

`--station` defaults to `deployment/station.yaml`, `--gripper` to `deployment/gripper.yaml`; `--dry-run` prints the
result without writing.

1. **Reads** `gripper.yaml` (`GripperFile.load`) and `franka_control`'s `tcp.F_T_TCP`: `load_process_config(
   DEPLOYMENT_ROOT / "franka_control", station.section("franka_control"))["tcp"]["F_T_TCP"]`, a 4×4 row-major list,
   checked to be a rigid transform.
2. **Computes** `F_T_cam = F_T_H × camera_pose(position_m, up_axis, tilt_deg)`, where `camera_pose` works in the
   hand frame (`H_T_cam`) and **`F_T_H = Rz(−45°)`** (no translation) is a fixed constant in `frames.py`, named
   `F_T_H` and documented there: it is how the Franka Hand is mounted on the flange, a fact of the hardware, not a
   setting. `camera_pose`, all in `H`:
   - camera z (viewing direction) = hand z tilted by `tilt_deg` **away from `up_side`** (toward the fingers):
     `z = cos(tilt)·ẑ_H − sin(tilt)·up`;
   - image y (down in the image) points **away from `up_side`**: `y = −(up − (up·z)·z)`, normalised;
   - `x = y × z` (right-handed);
   - translation = `position_m`.

   Then `T_cam_tcp = invert(F_T_cam) × F_T_TCP`.
3. **Writes** `T_cam_tcp` into the rig file named by `process_frames`' `rig_file` (its `config.yaml`, or the station's
   `process_frames.rig_file` override; resolved with `resolve_path`). It replaces only the top-level `tcp:` block, as
   text, so every other line and comment of the rig file is kept:
   ```yaml
   # T_cam_tcp COMPUTED by `python -m deploy.geometry` on <date> from <absolute path of gripper.yaml>
   # and franka_control tcp.F_T_TCP. Do not edit by hand: change gripper.yaml and rerun the tool.
   tcp:
     T_cam_tcp:
       matrix: [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]   # 4x4, row-major
     provenance: nominal
   ```
   The rotation is written as a matrix (no Euler-angle ambiguity); `dataproc`'s rig loader accepts `matrix` there.
   `provenance` stays `nominal` because the inputs are typed-in values, not a calibration. After writing, the tool
   reads the file back and checks the matrix round-trips.
4. **Prints** `F_T_cam`, `T_cam_tcp`, and the camera → fingertip distance (the length of `T_cam_tcp`'s translation),
   as a sanity check. It refuses to write (and exits 1) if the fingertips are not in front of and below the camera
   (`T_cam_tcp` translation z > 0 and y > 0): that means a sign or `up_side` mistake in `gripper.yaml`.

Signatures:

```
# frames.py
F_T_H: np.ndarray        # Rz(-45 deg), zero translation: the Franka Hand's mounting on the flange
def camera_pose(position_m: np.ndarray, up_axis: np.ndarray, tilt_deg: float) -> np.ndarray   # H_T_cam, 4x4, hand frame
def invert(T: np.ndarray) -> np.ndarray                                                        # rigid inverse
def pose_error(A: np.ndarray, B: np.ndarray) -> tuple[float, float]                            # (mm, deg) between two poses
def as_rigid(value: object, label: str) -> np.ndarray     # a 4x4 list checked to be rigid; GeometryError naming label

# rig_tcp.py
class RigTcp:
    @staticmethod
    def read(rig_file: Path) -> np.ndarray          # tcp.T_cam_tcp, either `matrix` or rotation{euler_xyz_deg}+translation_m
    @staticmethod
    def write(rig_file: Path, T_cam_tcp: np.ndarray, gripper_file: Path) -> None

# tcp_triangle.py
class TcpTriangle:
    def __init__(self, station: Station, gripper_file: Path) -> None   # reads gripper.yaml, F_T_TCP, the rig file's path
    F_T_TCP: np.ndarray
    F_T_cam: np.ndarray                                                 # F_T_H @ camera_pose(...)
    def compute(self) -> np.ndarray                                     # T_cam_tcp = invert(F_T_cam) @ F_T_TCP
    def check(self, mock: bool) -> CheckResult                          # for deploy check (below)

def triangle_check(station: Station, gripper_file: Path, mock: bool) -> CheckResult
    # what deploy check runs: TcpTriangle(...).check(mock), with a file that can't be read reported as the same
    # fail (warn with --mock), naming it
```

`RigTcp.read` accepts the Euler form too (`rotation: {euler_xyz_deg}` + `translation_m`, as in older rig files); it
uses scipy `from_euler("xyz", …)`, as `dataproc` does. The rig file `process_frames/rigs/rig_pi_v2.yaml` holds the
matrix this tool computed from the real `gripper.yaml` and `F_T_TCP` (translation `[0, 0.016891, 0.149465]` m), in
place of the nominal iPhone-era value it was given with.

**`deploy check`** (`TcpTriangle.check`, process name `geometry`) recomputes the triangle from the three files,
`gripper.yaml`, `franka_control/config.yaml` (with the station's overrides) and the rig file: it compares
`F_T_cam × T_cam_tcp(rig file)` with `F_T_TCP` using `pose_error`, and **fails** if they differ by more than
`launcher.tcp_check_max_mm` (2 mm) or `launcher.tcp_check_max_deg` (0.5°), e.g. after someone edits one file but not the
others. The detail names all three files, e.g. `TCP triangle off by 12.3 mm / 4.0°: gripper.yaml, franka_control/
config.yaml (tcp.F_T_TCP) and process_frames/rigs/rig_pi_v2.yaml disagree; run python -m deploy.geometry`. A
missing or broken `gripper.yaml` also fails, naming it. With `--mock` the same result is a `warn` instead of a `fail`:
the fake arm and fake Pi don't depend on the TCP, so a mock run works before `gripper.yaml` has real numbers.

The `deploy` package needs numpy and scipy for this (7.3).

## 7. Conventions for all folders

The four process planners and executors follow these.

### 7.1 Folder shape

```
<folder>/                        process_frames, policy, message_comms (franka_control: section 7.5)
  STRUCTURE.md
  launch.py                      the ManagedProcess subclass (section 4); not part of the package
  config.yaml                    every setting of this process, with its default
  pyproject.toml                 this folder's package and its pinned dependencies
  <folder>/                      the Python package, named like the folder: process_frames/process_frames/
    __init__.py
    __main__.py                  `python -m <folder>`: parses the common arguments (4.2), starts the process
    ...                          one class per file where sensible, file named after the class in snake_case
  tests/
    conftest.py                  shared fixtures (optional)
    data/                        small test files (optional)
    test_<module>.py             no __init__.py in tests/
```

`contracts/` and `deploy/` have the same shape without `launch.py` and `config.yaml`.

### 7.2 Imports

- Absolute imports only: `from process_frames.frame_buffer import FrameBuffer`.
- **Every package may import `contracts`. No package imports another process's package** (PLAN rule 3), and no
  package imports `deploy`. Only `deploy` imports the `launch.py` files, by path (section 3.6).
- The `deployment/` folder itself is **never** put on `sys.path`. Each package is found because its folder is installed
  in editable mode (`pip install -e <folder>`), which puts `deployment/<folder>/` on the path. (If `deployment/` were on
  the path, `import policy` could find the outer `policy/` folder instead of the package inside it.)
- **Never run Python with `deployment/` as the working folder** for `python -m ...` or `python -c ...`: Python puts
  the working folder first on the path, so it is the same mistake. The outer folder (`deployment/contracts/`) is then
  imported as an empty namespace package, found before the editable install's finder, and imports fail
  (`cannot import name ... from 'contracts' (unknown location)`). So the launcher starts every process with
  `cwd` = its own folder (3.4); tests run from inside their folder (7.4). The `deploy` command (the console script)
  works from any folder, `deployment/` included, since a console script doesn't put the working folder on the path;
  `python -m deploy` and `python -m deploy.geometry` run from `deployment/` stop with a message saying so. (The
  alternative, `pip install -e <folder> --config-settings editable_mode=compat`, is a note for INSTRUCTIONS.md.)

### 7.3 Python, environments and pinned versions

- **Python 3.10** everywhere (the `behavior_prompting` environment is 3.10.20; the laptop's system Python is 3.10.12).
  `requires-python = ">=3.10,<3.11"` in every `pyproject.toml`.
- **One `pyproject.toml` per folder** (so each process lists only what it needs, and the policy environment doesn't
  get OpenCV pins it doesn't want). Dependencies pinned to exact versions (`==`) except where noted.
- **Two environments:**

| Environment | Python | Installed (all `pip install -e`) | Runs |
|---|---|---|---|
| light (`python.light`) | 3.10 venv | `contracts`, `deploy`, `process_frames`, `message_comms`, `policy` (without torch) | the launcher, every `launch.py`, `process_frames`, `message_comms`, and `policy` under `--mock` |
| policy (`python.policy`) | the `behavior_prompting` conda environment (its `environment.yml`: torch 2.7.1, numpy 1.26.4, pyzmq 27.1.0, pyyaml 6.0.3) | `contracts`, `policy` (+ `behavior_prompting` as it already is) | `policy` |

- Pins:

| Package | Version | Who |
|---|---|---|
| numpy | 2.2.6 (light); 1.26.4 (policy, from its environment) | `contracts` declares `numpy>=1.26,<2.3` so it installs in both; `deploy` (6.6) |
| opencv-python-headless | 4.12.0.88 | `process_frames` (image prep, PLAN §6.1) |
| scipy | 1.15.3 | `process_frames` (image prep); `deploy` (the geometry tool, 6.6); `message_comms` if it needs rotations |
| pyyaml | 6.0.3 | all |
| av (PyAV) | 17.1.0 | `process_frames` (same as `rnoro/data-work`'s `dataproc`) |
| pyzmq | 27.1.0 | `contracts` (same as the policy environment) |
| msgpack | 1.1.0 | `contracts` |
| pytest | 8.3.5 | tests, in both environments (`[project.optional-dependencies] test`) |

  Installing `contracts` into the policy environment pins **msgpack 1.1.0** (new there) and **pyzmq 27.1.0** (the
  version it already has) in that environment too.
  `policy/pyproject.toml` does not pin torch or `behavior_prompting`: they come from that environment's
  `environment.yml`, which is the source of truth for it. They are an optional extra (`policy[model]`), not a base
  dependency, so `policy` also installs in the light environment for `--mock`. **Requirement on `policy/`:** torch and
  `behavior_prompting` are imported only inside `DiffusionPolicy` (and its helpers), never at package import, so
  `HoldStillPolicy` runs without them. A test in `policy/tests` checks `import policy` works with
  torch unavailable. The image-prep pins matter because PLAN §6.1 requires live
  images identical to training; `process_frames`' tests prove it against `cv_util`.

### 7.4 Tests

- **pytest, one command per folder, run from inside that folder** (`cd <folder> && ... -m pytest tests`). Not from
  `deployment/`: there, the folder name (e.g. `deployment/contracts/`) is picked up as an empty namespace package
  before the installed (editable) package, and imports fail (`cannot import name ... from 'contracts' (unknown
  location)`). One folder per command also because pytest reads its options (below) from the folder's own
  `pyproject.toml` (or `pytest.ini`); two folders in one command would share one set of options.

  ```
  # light environment (python.light), from deployment/
  (cd contracts && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests)
  (cd deploy && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests)
  (cd process_frames && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests)
  (cd message_comms && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests)
  (cd franka_control && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests)   # franka_control's launch.py (Python)
  (cd policy && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests)           # without torch: torch tests skip
  # policy environment (python.policy)
  (cd contracts && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/miniconda3/envs/behavior_prompting/bin/python -m pytest tests)
  (cd policy && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/miniconda3/envs/behavior_prompting/bin/python -m pytest tests)
  ```

  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed on this machine: ROS Humble puts pytest plugins on the path that crash
  pytest when they load. `policy/tests` runs in both environments: in the light one it proves `--mock` works without
  torch (tests that need torch use `pytest.importorskip("torch")`); in the policy one everything runs. `contracts/tests`
  runs in both too, since both environments install it (numpy 2.2.6 and 1.26.4).
- Each folder's pytest options (`[tool.pytest.ini_options]` in its `pyproject.toml`; `franka_control/`, which has no
  Python package, has a `pytest.ini` with the same options):
  ```
  testpaths = ["tests"]
  pythonpath = ["tests"]                   # tests can import small helper modules kept in tests/
  addopts = "--import-mode=importlib"      # same-named test files in two folders (two test_config.py) don't clash
  ```
- Unit tests run on a laptop: no Pi, no arm, no GPU, no network beyond `127.0.0.1` (PLAN §7). Sockets in tests bind
  `tcp://127.0.0.1:*` (a free port), never the station's ports.
- Tests never sleep to wait for something: they use timeouts on a condition, or an injected clock.
- C++ (`franka_control`): CMake + CTest, GoogleTest, in three build folders (7.5). The tests are built with address
  and undefined-behaviour sanitizers (PLAN §6.3) in `franka_control/build-test/`, and with the thread sanitizer in
  `franka_control/build-tsan/` (it can't be combined with the address sanitizer), never in `franka_control/build/`,
  which holds the binary `deploy run` starts, built without sanitizers:
  ```
  cmake -S franka_control -B franka_control/build-test -DFRANKA_CONTROL_SANITIZE=ON
  cmake --build franka_control/build-test
  ctest --test-dir franka_control/build-test --output-on-failure
  ```
  The `build-tsan/` commands and their CMake option are in `franka_control/STRUCTURE.md`.

### 7.5 `franka_control/` (C++) shape

`STRUCTURE.md`, `launch.py`, `config.yaml`, `CMakeLists.txt`, `include/franka_control/`, `src/`, `tests/`, and
three build folders (not in git): `build/` (the binary `deploy run` uses, no sanitizers), `build-test/` (tests with
address and undefined-behaviour sanitizers) and `build-tsan/` (tests with the thread sanitizer), 7.4. It also has a
`pytest.ini` for the Python tests of its `launch.py` in `tests/`. C++17. One class per header/source pair, interfaces as abstract classes (pure virtual). Its
`launch.py` is Python like the others (it runs in the launcher). It reads `station.yaml` and its `config.yaml` with
yaml-cpp, following the same merge and path rules (6.3). It reads `station.yaml` itself rather than getting flags
from `launch.py`, so the binary started alone uses the same file (PLAN §6.5). Library versions: `franka_control/STRUCTURE.md` (contracts/STRUCTURE.md
§8 proposes msgpack-cxx 6.1.1, cppzmq 4.10.0, yaml-cpp 0.8.0).

### 7.6 Code style

- **Object-oriented, one job per class, one class per file where sensible** (PLAN rule 5). Small value types
  (dataclasses) may share a file with the class that uses them.
- **Swappable parts are abstract base classes** (`abc.ABC` + `@abstractmethod`) with a real and a fake
  implementation (PLAN rule 5). Runtime stand-ins used by `--mock` (`FakeArm`, `FakePiSender`, `HoldStillPolicy`) live
  in the package; fakes used only by tests live in `tests/`.
- **No numbers in code** that might change (rates, limits, ports, thresholds, timeouts): they come from `config.yaml`,
  `station.yaml`, `gripper.yaml` (hand-measured gripper geometry only), or `contracts/contracts/limits.yaml`, and are passed into constructors. Physical constants and the
  shapes fixed by a contract are not "settings".
- **Time:** `time.monotonic_ns()` (C++: `clock_gettime(CLOCK_MONOTONIC)`) for every `t_ns`. Classes that use time take
  a clock function in their constructor, so tests can drive time.
- `from __future__ import annotations`, type hints on public methods, PEP 8, lines up to 120 characters (as in the
  rest of the repo).
- **Logging** with the `logging` module to stderr: `"%(asctime)s %(name)s %(levelname)s %(message)s"`.
- **FATAL line:** a process that exits because of a failure writes, as its last stderr line,
  `FATAL <name>: <what failed>: <what to check>` and exits with a non-zero code. The launcher quotes it (5.3). Normal
  exit (operator quit, SIGINT, SIGTERM) is code 0 with no FATAL line.
- **SIGTERM and SIGINT** both mean "stop cleanly": every process handles both, cleans up within `stop_timeout_s`, and
  exits 0. The launcher sends SIGTERM (3.4); SIGINT comes from Ctrl-C, which only `message_comms` receives (5.5).
- Errors are specific exception classes; no bare `except:`; an exception is never swallowed silently.
- **Provenance header on copied files.** A file copied from another branch starts with
  ```
  # from <branch>:<path> @ <commit>
  # changes: none            (or one line saying what changed)
  ```
  e.g. `# from rnoro/data-work:data-processing/dataproc/imageprep.py @ aac065c`. The commit is the short hash the file
  was copied at (`git log -1 --format=%h <branch> -- <path>`). Copies meant to stay identical ("copied unchanged so
  the images match training") say `changes: none` and a test may compare them with the source.

## 8. `--mock`

`deploy run --mock` and `deploy check --mock` pass `--mock` to every process (common argument, 4.2); each process maps
it to its own stand-in. The launcher does nothing else differently.

| Process | Without `--mock` | With `--mock` | `checks()` with `--mock` |
|---|---|---|---|
| `franka_control` | `LibfrankaArm` + `FrankaHandGripper` at `robot_ip` (default `172.16.0.2`) | `FakeArm` + `FakeGripper` | skips the arm IP check and the RT-kernel check |
| `process_frames` | `PiStreamSource` on `endpoints.pi_*` | `FakePiSender` sending the frames of `mock_episode`, run as a thread inside `process_frames`, bound to `endpoints.mock_pi_*` and sending real C0 messages; `PiStreamSource` connects to it over that local port exactly as to the Pi | skips the Pi checks; checks `mock_episode` exists |
| `policy` | `policy.kind`, run with `python.policy` for `kind: diffusion` and with `python.light` for `kind: hold_still` (no torch needed) | always `HoldStillPolicy` ("stay at the anchor"), run with **`python.light`, without torch** | skips the checkpoint, GPU and policy-environment checks |
| `message_comms` | the real one | the real one (it is the thing under test); `operator_script` chooses scripted keys | none skipped |

PLAN §9's "Arm" and "Camera source" backups are exactly the `--mock` choices; the other backups are station settings
(6.2), so `--mock` and backups never mix.

## 9. Unit tests

All in `deploy/tests/`. Fake processes are tiny Python scripts in `tests/data/` (e.g. "write the ready file after 0.2 s
then wait for SIGTERM", "exit 1 with a FATAL line", "ignore SIGTERM"), wrapped by test-only `ManagedProcess` subclasses.

| File | Checks |
|---|---|
| `test_station.py` | the example station loads; launcher defaults are applied; every problem of a broken file is listed at once with its key path (missing required key, wrong type, bad endpoint string, unknown top-level key, wrong `station_version`); `session_root` and every endpoint are required; with `mock=True` a file without `python.policy` and `endpoints.pi_*` loads, without it the same file is refused; `~` and relative paths are resolved from the station file's folder |
| `test_config_merge.py` | maps merge key by key; lists and scalars are replaced; an unknown override key is refused with its path; an empty section changes nothing |
| `test_session.py` | folder name with and without `_mock`; `logs/`, `ready/` made; the station file copied in |
| `test_check_result.py` | table layout; statuses; exit code 2 when any `fail`, 0 with only `warn`/`info` |
| `test_check_helpers.py` | each helper with a passing and a failing input (missing file, busy port, missing module, an unreachable host) |
| `test_managed_process.py` | the command is `program()` + the common arguments in order, `--mock` only when mocked; the log starts with the command and captures stdout and stderr; ready via the ready file; a ready timeout raises naming the process and its `READY_HINT`; exiting during start-up raises with the FATAL line; `stop` sends SIGTERM, then SIGKILL to a process that ignores it; a grandchild is stopped too (process group); for a `USES_TERMINAL` process only its pid is signalled, never a group (the test process survives); `config_path` resolves a station-set path against the station file's folder and a default against the process folder; `is_ready` raising `LaunchError` fails at once |
| `test_loader.py` | loads a `launch.py` by path; missing file, missing `PROCESS_CLASS`, wrong base class, or `NAME` not matching the folder are refused |
| `test_launcher.py` | **start order** `franka_control → process_frames → policy → message_comms`, each started only after the previous is ready; **ready timeout** of the 2nd stops the 1st and never starts the 3rd and 4th, with the message; **one process exiting stops all the others in reverse order**; `message_comms` exiting 0 gives exit 0; SIGINT to the launcher gives reverse stop and exit 130, and the launcher waits for the `USES_TERMINAL` process to exit on its own before sending it SIGTERM; a failing check refuses to start (exit 2) and starts nothing; two `USES_TERMINAL` processes are refused |
| `test_check.py` | `deploy check` reports **every** failing check, grouped by process (the TCP triangle under `geometry`), and keeps going after a failure |
| `test_mock.py` | `--mock` reaches every process's command; with the real `launch.py` files of the four folders (skipped for a folder whose `launch.py` doesn't exist yet): each loads, its `NAME` matches, its mock command contains `--mock`, and its mock checks don't need the Pi or the arm |
| `test_launch_imports.py` | every `launch.py` in the four folders imports only the standard library, `yaml`, `deploy` and `contracts` (read with `ast`, not imported) |
| `test_gripper_file.py` | the example `gripper.yaml` loads; each missing key, unknown key, non-finite position, `up_side` other than `±x`/`±y`, and `tilt_deg` outside `[0, 90)` is refused with its path; all problems listed at once; a `finger_extra_m` key is refused (the finger length lives only in `tcp.F_T_TCP`) |
| `test_geometry.py` | **the worked check from planning** (`deploy/tests/tcp_check_reference.py`, copied by the executor into this file, with its example numbers as fixtures): the triangle closes, `F_T_cam × T_cam_tcp = F_T_TCP`; the rig-file form round-trips (`RigTcp.write` then `RigTcp.read`, matrix form; an Euler-form rig file reads the same as scipy `from_euler("xyz")`); the fingertips are in front of and below the camera (`T_cam_tcp` z > 0, y > 0); for 1,000 random arm poses `O_T_F`, the training side `(O_T_F × F_T_cam) × T_cam_tcp` and the robot side `O_T_F × F_T_TCP` give the identical point and axes; a policy motion `dT` applied to the TCP moves the fingertips identically on both sides. Plus: `camera_pose` is tested **in the hand frame `H`** (a right-handed orthonormal frame for each `up_side`; with
`tilt_deg = 0` it looks straight along hand z; image y points away from `up_side`); `F_T_cam = F_T_H × H_T_cam` with
`F_T_H = Rz(−45°)`; **the real numbers**: the real `deployment/gripper.yaml` (`[55, 0, 10]` mm, `+x`, 15°) with
`F_T_TCP` = (−45° about z, `[0, 0, 0.150]`) reproduces the v2 rig file's `T_cam_tcp` within 1e-6 (a 15° tilt
toward the fingers, translation `[0, 0.016891, 0.149465]` m); `RigTcp.write` keeps every other line of a rig file unchanged and adds the COMPUTED header naming `gripper.yaml`; the tool refuses to write when the fingertips would be behind the camera; `--dry-run` writes nothing |
| `test_tcp_triangle.py` | `check` passes on consistent files; fails beyond 2 mm, and beyond 0.5°, with a detail naming all three files; the v2 rig file passes with the real `gripper.yaml` and `F_T_TCP`; a rig file with the old iPhone-era value (15° about x, `[0, 0.06, 0.20]` m) fails; a missing `gripper.yaml` fails naming it; with `mock=True` each of these is a `warn`; a station override of `process_frames.rig_file` or `franka_control.tcp.F_T_TCP` is the one used |
| `test_cli.py` | `check` and `run` parse; only `--station` and `--mock` exist; `--station` defaults to `deployment/station.yaml`; a `StationError` exits 2 with its list |

Each process folder also has a `tests/test_launch.py` (its own planner lists it): its `program()`, its `checks()` with a
broken station section, and its mock choice.

The end-to-end test of PLAN §7 (milestone 3) is owned by `deploy/`: `tests/test_end_to_end_mock.py` runs
`python -m deploy run --mock` as a subprocess, with a temporary copy of the real station file (free local ports, a
temporary `session_root`, `message_comms.operator_script` set to a scripted-keys file: `c`, wait for steps, `y`, then
`q`, and every path the copy would resolve from its new folder made absolute: the `python` entries,
`process_frames.mock_episode` and `process_frames.rig_file`, which points at the real `process_frames/rigs/rig_pi_v2.yaml`),
and checks exit code 0, `deploy: done`, and the episode recorded as `success` in `<session>/message_comms/episodes.jsonl`.
It needs all four processes, so it is **skipped until the four `launch.py` files exist** (and while the
`franka_control` binary is not built or `process_frames.mock_episode` doesn't exist). It belongs to phase 3 (PLAN §8).
