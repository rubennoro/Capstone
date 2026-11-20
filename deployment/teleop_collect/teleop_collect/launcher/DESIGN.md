# launcher: design

`teleop_collect check | run [--mock] [--station FILE] [--no-browser]` (PLAN §2 rule 3, STRUCTURE §10.1). Also
`python -m teleop_collect`.

## What it does, and what it does not

- **check** prints one table: the station rows, then each process's rows. Exit 0 if nothing failed (warn and info do
  not count), else 2.
- **run** prints the same table and refuses to start if any row failed (exit 2). Otherwise it creates a session folder
  and starts `fake_pc` (only with `--mock` and `launcher.fake_pc: true`), then `frames`, `teleop` and `ui`. Each one
  starts only after the one before it is ready. Then it prints `teleop_collect: ready: open <url>`, opens the browser,
  and watches the processes. It stops them in reverse order.
- It only starts, watches and stops **the laptop's own processes**. It never starts, stops, ssh's into or reads files
  of the PC or the Pi (PLAN §2 rule 4). Their rows come from a TCP probe of the Pi's port and from `pc_probe`, which
  speaks only C2/C3.
- It never imports a process subpackage. Each process runs as `<sys.executable> -m teleop_collect.<name> --station S
  --session D --ready-file F [--mock]`, so OpenCV, PyAV and aiohttp never load in the launcher. The check subprocesses
  work the same way (`frames.rig_file`, `teleop.pc_probe`).

## Classes and data flow

| File | Class / function | Job |
|---|---|---|
| `cli.py` | `parse_args`, `main` | the command line. A station that does not load becomes failing `station.yaml` rows, then exit 2 |
| `launcher.py` | `LauncherSettings` | `config/launcher.yaml` + the station's `launcher:` section (checked) |
| | `Launcher` | `ORDER`, the station rows, `check()`, `run()`: start, ready waits, the page, supervision, stop |
| `managed_process.py` | `ManagedProcess` | one process: its command, its config, `base_checks` (overrides, program), `start` (own process group, log captured, stdin closed, cwd `deployment/teleop_collect/`), `wait_ready` (ready file), `stop` (SIGTERM to the group, then SIGKILL), the FATAL line from its log |
| `processes.py` | `FakePcProcess`, `FramesProcess`, `TeleopProcess`, `UiProcess` | each process's rows and ready hint |
| `check_helpers.py` | `port_free`, `host_reachable`, `address_is_local`, `json_check_lines`, `rig_file_check`, `writable_folder`, ... | small reusable checks, one `CheckResult` each |
| `check_result.py` | `CheckResult`, `print_table`, `exit_code` | one table row |
| `session.py` | `Session` | `<session_root>/<YYYY-MM-DD_HH-MM-SS>[_mock]/` with `logs/<name>.log`, `ready/<name>.ready`, `station.yaml` and `contracts/` (copies of the three contract files) |
| `errors.py` | `LaunchError` | a start or ready failure. Its message is the one-line `FAILED ...` |

**run, step by step:**

```
check table ─fail─► "not started: N check(s) failed" (2)
   │ pass
Session.create ─► SIGINT/SIGTERM/SIGHUP handlers ─► "starting fake_pc, frames, teleop, ui (session: ...)"
   │
for each process in ORDER: [teleop note] start() ─► wait_ready(ready_timeout_s[name])
   │   exit or timeout ─► "teleop_collect: FAILED <name> during start-up ...: <FATAL line>  [log: ...]" (1)
   ▼
"all N processes ready" ─► "teleop_collect: ready: open http://127.0.0.1:<http>" ─► webbrowser.open
   │
poll every poll_s: a process exited?
   teleop exit 0      ─► "teleop stopped (exit 0): done" (0)
   anything else      ─► "teleop_collect: FAILED <name> (exit N): <FATAL line>  [log: ...]" (1)
   Ctrl-C / SIGTERM / SIGHUP ─► "stopped by Ctrl-C" / "stopped by SIGTERM" / "stopped by SIGHUP" (0)
   ▼
stop in reverse (ui, teleop, frames, fake_pc): SIGTERM, stop_timeout_s, SIGKILL
   a second signal during the stop ─► SIGKILL everything, "killed by a second Ctrl-C during the stop" (130)
```

## Rows (STRUCTURE §10.1)

- **station:**
  - `station.yaml` loads.
  - `deploy_contracts` comes from this checkout's `deployment/contracts`. Otherwise it fails with the install hint
    `~/envs/teleop_collect/bin/pip install -e deployment/contracts`.
  - The three contract files load.
  - `laptop address`: one of `link.yaml laptop.addresses` is on this laptop (not under `--mock`).
  - `Wi-Fi`: on `wifi_ssid`, else WARN (not under `--mock`, as in pd:).
  - `commit`: INFO.
  - `disk`: free space under `data_root`, WARN below 2 GB.
  - `data_root`: writable. A temporary file is created and removed. A missing folder passes if its nearest existing
    parent is writable. Nothing is created.
- **every process:** `overrides` (its station section against its config file) and `program` (the interpreter).
- **fake_pc:** the `link.yaml` C2 and C3 ports are free.
- **frames:**
  - `rig file` (`python -m teleop_collect.frames.rig_file RIG --json`);
  - `R1 port` and `preview port` free;
  - `Pi reachable` (not `--mock`);
  - `mock episode` (`meta.json` present) and `fake Pi port` free (`--mock`).
- **teleop:**
  - `U1 port` free;
  - `PC margins` (INFO, also under `--mock`): `teleop assumes the PC's safety.box_margin_m 0.05 and cone_margin_deg 10
    (the teleop block of the PC's station.yaml); with smaller PC margins a press into the floor can fault the PC`, from
    `teleop.motion`. The PC never reports its margins, so this can only be shown, not checked; FAIL only when the two
    keys cannot be read. The text is a copy of `teleop.motion.planner.ASSUMED_MARGINS` (the launcher imports no
    process subpackage);
  - not under `--mock`: the rows `python -m teleop_collect.teleop.pc_probe --station S --check --json` prints, one per
    JSON line, or one failing `PC` row. The PC commit WARN is one of pc_probe's own rows.
  - under `--mock`: an INFO `PC` row (the fake PC, or the cross test's `franka_control --mock --local`).
- **ui:** `HTTP port` free.

## Decisions

- **No process uses the terminal.** The operator works in the page (PLAN §4.3), so pd:'s `USES_TERMINAL` is gone.
  Every process gets its own process group, so a Ctrl-C reaches only the launcher, which stops the rest in reverse.
- **Reverse order matters.** `teleop` gets its SIGTERM while `frames` and the (fake) PC still run. It can then abort
  an open episode and send its C3 `stop` (PLAN §7). The PC itself is never touched and keeps holding.
- **An operator stop exits 0** (PLAN §7: "Operator Ctrl-C: conclude (normal); exit 0"). That covers Ctrl-C,
  SIGTERM and SIGHUP to the launcher. pd: used 130.
- **Closing the terminal stops the run.** SIGHUP is handled like SIGTERM: stop in reverse, exit 0. The processes
  run in their own sessions, so they never get the terminal's hangup. Without this, the launcher would die while
  teleop and ui kept running: the page could still drive the arm, and the next run would find its ports busy. The
  last line is skipped if the terminal is already gone (`launcher.log` has it).
- **Any process exit while running concludes** (PLAN §7). There is one exception: teleop exiting 0 on its own (someone
  signalled it directly) is treated like pd:'s manager quit, exit 0. An exit 0 of `ui`, `frames` or `fake_pc` is a
  failure.
- **The browser** opens only on ready, and not with `--no-browser` or `ui.open_browser: false`. On Linux it also needs
  `DISPLAY`/`WAYLAND_DISPLAY`: without a display, `webbrowser` would start a console browser in the terminal. When it
  cannot open one, one line says to open the page by hand, and the run goes on.
- **A note before teleop starts** (PLAN §4.2 step 5): `teleop_collect: teleop: the start pose is the arm's pose at
  start-up: press h, then p on the PC first` (with `--mock`: `...: the fake PC starts at its start pose`). teleop's
  start-up never moves the arm (no start-up home since 2026-11-20, as final-deployment @ 927a5ba), so there is no
  "keep a hand on the user stop" note any more.
- **One environment:** `python()` is `sys.executable`. There is no station `python:` section.
- **Timeouts** come from `config/launcher.yaml`. teleop's 150 s is above its longest legal start-up (20 + 60 + 30 +
  5 = 115 s), so teleop's own FATAL line always comes first.

## Copied from fd: and changed

Every file is adapted from `pd:launcher/` @ 2ffe641, and each header lists its changes:

- `ORDER` and the process classes are this package's: `TeleopProcess` from `ManagerProcess` without the terminal and
  self-check rows, `UiProcess` new, no `PolicyProcess`.
- No `USES_TERMINAL`.
- Exit 0 for an operator stop.
- The `ready: open` line, the browser, the `starting` line and the teleop note.
- The disk row looks under `data_root`, plus the `data_root` row.
- The frames rows drop `imports` and the C1 port, and add the R1 and preview ports.
- The teleop rows add `PC margins` (2026-11-20).
- Messages say `teleop_collect`.

## Tests

`tests/launcher/` uses fake processes (`tests/launcher/fake_bin/fake_proc.py`, via `launcher_helpers.make_process`) and a
fake interpreter for the check subprocesses (`write_fake_python`, swapped in with `use_python`). They need none of the
real processes and use free ports only.

- `test_launcher.py`: order, ready waits, stop in reverse, the FAILED line, Ctrl-C, SIGTERM, SIGHUP and a second signal, the
  ready line and the browser.
- `test_managed_process.py`: command, log, cwd, process group, SIGTERM/SIGKILL, the FATAL line.
- `test_processes.py`: each process's rows.
- `test_check_rows.py`: every station row passing and failing, and the whole table.
- `test_check.py`, `test_check_helpers.py`, `test_check_result.py`, `test_cli.py`, `test_session.py`,
  `test_station.py` (LauncherSettings).
- `test_mock.py`: no remote contact under `--mock`.

## How to extend

- **A new check row:** a function in `check_helpers.py` that returns one `CheckResult`. Call it from the process's
  `checks()` (or `Launcher.station_checks()` for a shared file), and add a pass and a fail case to its test.
- **A new process:** a `ManagedProcess` subclass in `processes.py` (NAME, READY_HINT, checks), plus its entry in
  `PROCESS_CLASSES` and `Launcher.ORDER`, `config/launcher.yaml ready_timeout_s`, `process_args.PROCESS_NAMES` and
  `station.PROCESS_SECTIONS` (the last two are E1's).
