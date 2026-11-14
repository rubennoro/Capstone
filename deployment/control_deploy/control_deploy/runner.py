"""Runner: `control_deploy run` (control_deploy/STRUCTURE.md 6.4). Checks, a session folder, then franka_control on
this terminal: stdin inherited (the binary reads the keys itself), its output teed to the log and the terminal, a
status line from C2, SIGTERM/SIGHUP forwarded (SIGKILL 5 s later), Ctrl-C left to the binary."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from deploy_contracts import ArmState, LinkSettings, Subscriber

from control_deploy import netinfo, paths
from control_deploy.binary import FrankaControlBinary
from control_deploy.check_result import exit_code, print_table
from control_deploy.checks import PcChecks
from control_deploy.session import Session
from control_deploy.station import Station
from control_deploy.status_line import StatusLine

READY_TIMEOUT_S = 90.0        # FCI connect + finger homing + the hold
STOP_TIMEOUT_S = 5.0          # SIGTERM -> SIGKILL; the binary's own budget is 3 s + 1.5 s
STATUS_PERIOD_S = 0.25
PEERS_PERIOD_S = 1.0
READY_HINT = "check robot_ip, FCI enabled in Desk, brakes unlocked, user stop released, and the log"
KEYS = "Keys: h home arm, g home hand, s stop, q quit"


class Runner:
    def __init__(self, station_path: Path | str | None, mock: bool, local: bool, *,
                 checks: Callable[[Path, bool, bool], PcChecks] = PcChecks, out: TextIO | None = None,
                 stdin_fd: int = 0, ready_timeout_s: float = READY_TIMEOUT_S,
                 stop_timeout_s: float = STOP_TIMEOUT_S, status_line: bool | None = None) -> None:
        self.station_path = Path(station_path) if station_path else paths.default_station()
        self.mock = mock
        self.local = local
        self.checks = checks
        self.out = out or sys.stdout
        self.stdin_fd = stdin_fd
        self.ready_timeout_s = ready_timeout_s
        self.stop_timeout_s = stop_timeout_s
        is_tty = _isatty(stdin_fd) and _isatty(self.out)
        self.status_enabled = is_tty if status_line is None else status_line
        self._lock = threading.Lock()
        self._status_text = ""
        self._forward: int | None = None
        self._wrapper_log: TextIO | None = None
        self.session: Session | None = None
        self.proc: subprocess.Popen | None = None

    # ---- terminal output ----

    def _emit(self, line: str) -> None:
        with self._lock:
            if self.status_enabled and self._status_text:
                self.out.write("\r\033[K" + line + "\n" + self._status_text)
            else:
                self.out.write(line + "\n")
            self.out.flush()

    def _say(self, line: str) -> None:
        """A line of this wrapper: terminal and logs/control_deploy.log."""
        self._emit(line)
        if self._wrapper_log:
            self._wrapper_log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
            self._wrapper_log.flush()

    def _draw(self, text: str) -> None:
        with self._lock:
            self._status_text = text
            self.out.write("\r\033[K" + text)
            self.out.flush()

    # ---- run ----

    def run(self) -> int:
        checks = self.checks(self.station_path, self.mock, self.local)
        results = checks.run()
        print_table(results, self.out)
        if exit_code(results) != 0:
            self.out.write("control_deploy: a check failed: nothing started\n")
            self.out.flush()
            return 2
        station = Station.load(self.station_path)
        link = LinkSettings.load(paths.contracts_dir() / "link.yaml")
        self.session = Session.create(station.session_root, "control_deploy", self.mock)
        self.session.copy_inputs(station.path, paths.contracts_dir())
        binary = FrankaControlBinary(paths.binary(station.franka_control))
        cmd = binary.command(station.path, self.session.dir, self.session.ready_file, self.mock, self.local)
        state_ep, commands_ep = link.state_endpoint(self.local), link.commands_endpoint(self.local)
        version = checks.version
        self.session.write_yaml("run.yaml", {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "flags": {"mock": self.mock, "local": self.local, "station": str(station.path)},
            "command": cmd,
            "commit": version.commit if version else None,
            "dirty": version.dirty if version else None,
            "binds": {"state": state_ep, "commands": commands_ep},
            "checks": [{"process": r.process, "check": r.name, "status": r.status, "detail": r.detail}
                       for r in results],
        })
        self._wrapper_log = open(self.session.wrapper_log, "a", encoding="utf-8")
        saved_tty = _save_termios(self.stdin_fd)
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl-C reaches the binary through the process group
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGHUP, self._on_signal)
            return self._supervise(cmd, state_ep, commands_ep, link)
        finally:
            for s, h in old_handlers.items():
                signal.signal(s, h)
            _restore_termios(self.stdin_fd, saved_tty)         # in case the binary was killed in cbreak mode
            if self._wrapper_log:
                self._wrapper_log.close()
                self._wrapper_log = None

    def _on_signal(self, signum: int, _frame) -> None:
        self._forward = signum

    def _supervise(self, cmd: list[str], state_ep: str, commands_ep: str, link: LinkSettings) -> int:
        assert self.session is not None
        self._say(f"control_deploy: session {self.session.dir}")
        log = open(self.session.franka_control_log, "ab")
        self.proc = subprocess.Popen(cmd, stdin=None if self.stdin_fd == 0 else self.stdin_fd,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        reader = threading.Thread(target=self._tee, args=(self.proc, log), daemon=True)
        reader.start()
        stop = threading.Event()
        status_thread: threading.Thread | None = None
        t0 = time.monotonic()
        ready = False
        term_sent_at: float | None = None
        killed = False
        try:
            while self.proc.poll() is None:
                if self._forward is not None and term_sent_at is None:
                    name = signal.Signals(self._forward).name
                    self._say(f"control_deploy: received {name}: forwarding SIGTERM to franka_control")
                    self.proc.send_signal(signal.SIGTERM)
                    term_sent_at = time.monotonic()
                if term_sent_at is not None and not killed and time.monotonic() - term_sent_at > self.stop_timeout_s:
                    self._say(f"control_deploy: franka_control still running {self.stop_timeout_s:g} s after "
                              f"SIGTERM: SIGKILL")
                    self.proc.kill()
                    killed = True
                if not ready and self.session.ready_file.exists():
                    ready = True
                    self._say(f"control_deploy: ready: C2 {state_ep}, C3 {commands_ep}; start policy_deploy on the "
                              f"laptop. {KEYS}")
                    if self.status_enabled:
                        status_thread = threading.Thread(target=self._status, args=(state_ep, link, stop),
                                                         daemon=True)
                        status_thread.start()
                if not ready and term_sent_at is None and time.monotonic() - t0 > self.ready_timeout_s:
                    self._say(f"control_deploy: franka_control not ready after {self.ready_timeout_s:g} s: "
                              f"{READY_HINT}")
                    self.proc.send_signal(signal.SIGTERM)
                    term_sent_at = time.monotonic()
                time.sleep(0.05)
        finally:
            stop.set()
            if status_thread:
                status_thread.join(timeout=2.0)
            reader.join(timeout=5.0)
            log.close()
            if self.status_enabled:
                with self._lock:
                    self._status_text = ""
                    self.out.write("\r\033[K")
                    self.out.flush()
        rc = self.proc.returncode
        if rc == 0 and ready:
            self._say(f"control_deploy: franka_control stopped cleanly (session {self.session.dir})")
            return 0
        self._say(f"control_deploy: franka_control failed: see {self.session.franka_control_log}")
        return rc if rc and rc > 0 else 1

    def _tee(self, proc: subprocess.Popen, log) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            log.write(raw)
            log.flush()
            self._emit(raw.decode("utf-8", errors="replace").rstrip("\n"))

    def _status(self, endpoint: str, link: LinkSettings, stop: threading.Event) -> None:
        line = StatusLine()
        sub = Subscriber("control_deploy status line", endpoint, ArmState, recv_hwm=1000)
        peers: list[str] = []
        next_peers = 0.0
        next_draw = 0.0
        try:
            while not stop.is_set():
                msg = sub.receive(0.05)
                now = time.monotonic()
                if msg is not None:
                    line.feed(msg, now)
                if now >= next_peers:
                    peers = netinfo.c3_peers(link.commands_port)
                    next_peers = now + PEERS_PERIOD_S
                if now >= next_draw:
                    self._draw(line.render(now, peers))
                    next_draw = now + STATUS_PERIOD_S
        except Exception as exc:   # a status line problem never stops the run
            self._emit(f"control_deploy: status line stopped: {exc}")
        finally:
            sub.close()


def _isatty(f) -> bool:
    try:
        return os.isatty(f if isinstance(f, int) else f.fileno())
    except (OSError, ValueError, AttributeError):
        return False


def _save_termios(fd: int):
    try:
        return termios.tcgetattr(fd) if os.isatty(fd) else None
    except (termios.error, OSError):
        return None


def _restore_termios(fd: int, saved) -> None:
    if saved is None:
        return
    try:
        termios.tcsetattr(fd, termios.TCSANOW, saved)
    except (termios.error, OSError):
        pass
