"""``policy_deploy run --mock --station <frozen>`` in a pseudo-terminal (pexpect), with its output parsed.

The manager writes one status line rewritten in place (``\\r<line>\\x1b[K``, manager/status_line.py) and full message
lines above it (``\\r\\x1b[K<text>\\n``, manager/operator.py). The phase is the status line's head: ``[starting]``,
``[idle]``, ``[homing] ...``, ``[ep N starting]``, ``[ep N running 3.2 s] step 12 | ...``, ``[done]``. Keys are single
characters (cbreak mode): c start, y success, n fail, s stop, h home, q quit.
"""

from __future__ import annotations

import os
import re
import signal
import threading
import time
from collections import deque
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
READY = re.compile(r"policy_deploy: all \d+ processes ready \(session: (?P<path>[^)]+)\)")
NOT_STARTED = re.compile(r"policy_deploy: not started: (?P<why>.*)")
FAILED = re.compile(r"policy_deploy: (?P<why>FAILED .*)")            # E7: a process failed (e.g. the PC in fault)
EP_STARTED = re.compile(r"^episode (?P<id>\d+) started")
EP_ENDED = re.compile(r"^episode (?P<id>\d+): (?P<outcome>\w+) \((?P<reason>.*)\), (?P<steps>\d+) steps, "
                      r"(?P<dur>[\d.]+) s")
HEAD = re.compile(r"^\[(?P<head>[^\]]+)\]")


class PolicyDeploySession:
    def __init__(self, station: Path, log_path: Path, *, python_env: Path, cwd: Path,
                 dimensions=(50, 220)) -> None:
        self.station, self.log_path, self.cwd = Path(station), Path(log_path), Path(cwd)
        self.exe = Path(python_env).expanduser() / "bin" / "policy_deploy"
        self.dimensions = dimensions
        self.child = None
        self._lock = threading.Lock()
        self._buf = ""
        self.lines: deque = deque(maxlen=5000)   # (t, text) of message lines and status changes
        self.status = ""
        self.phase = ""
        self.session_dir: Path | None = None
        self.not_started: str | None = None
        self.eof = False
        self._thread: threading.Thread | None = None
        self._log = None

    # ------------------------------------------------------------------------------------------------ process
    def start(self) -> None:
        import pexpect

        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["TERM"] = env.get("TERM", "xterm") or "xterm"
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.exe} run --mock --station "
                        f"{self.station} (cwd {self.cwd})\n")
        self.child = pexpect.spawn(str(self.exe), ["run", "--mock", "--station", str(self.station)], cwd=str(self.cwd),
                                   env=env, encoding="utf-8", codec_errors="replace", dimensions=self.dimensions,
                                   timeout=None)
        self._thread = threading.Thread(target=self._reader, name="pty-reader", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        import pexpect

        while True:
            try:
                data = self.child.read_nonblocking(4096, timeout=0.2)
            except pexpect.TIMEOUT:
                continue
            except (pexpect.EOF, OSError):
                with self._lock:
                    self.eof = True
                return
            if self._log:
                self._log.write(data)
                self._log.flush()
            self._feed(data)

    def _feed(self, data: str) -> None:
        with self._lock:
            self._buf += data
            parts = re.split(r"(\r|\n)", self._buf)
            self._buf = parts.pop() if parts else ""           # the incomplete tail
            pending = ""
            for p in parts:
                if p in ("\n", "\r"):
                    # E7: a message can follow the status line's erase code on the same line
                    # ("\r[idle] ...\x1b[Kpolicy_deploy: all 3 processes ready (...)\r\n"): one piece per segment
                    for seg in pending.split("\x1b[K"):
                        txt = ANSI.sub("", seg).strip()
                        if txt:
                            (self._message if p == "\n" else self._status_line)(txt)
                    pending = ""
                else:
                    pending += p
            self._buf = pending + self._buf
            if self._buf.endswith("\x1b[K"):                  # a complete status line (show_status ends with it)
                tail = ANSI.sub("", self._buf).strip()
                if tail.startswith("["):
                    self._status_line(tail)

    def _status_line(self, txt: str) -> None:
        if txt.startswith("["):
            if txt != self.status:
                self.status = txt
                m = HEAD.match(txt)
                head = m.group("head") if m else ""
                phase = ("running" if " running " in f" {head} " else "starting_ep" if head.startswith("ep ")
                         else head.split()[0] if head else "")
                if phase != self.phase:
                    self.phase = phase
                    self.lines.append((time.monotonic(), f"<phase {phase}> {txt}"))
        else:
            self._message(txt)

    def _message(self, txt: str) -> None:
        if not txt:
            return
        if txt.startswith("[") and HEAD.match(txt):
            self._status_line(txt)
            return
        self.lines.append((time.monotonic(), txt))
        m = READY.search(txt)
        if m:
            self.session_dir = Path(m.group("path").strip())
        m = NOT_STARTED.search(txt) or FAILED.search(txt)
        if m and self.not_started is None:
            self.not_started = m.group("why")

    # ---------------------------------------------------------------------------------------------- queries
    def alive(self) -> bool:
        return self.child is not None and self.child.isalive() and not self.eof

    def send(self, key: str) -> None:
        if self.alive():
            self.child.send(key)
            self.lines.append((time.monotonic(), f"<key {key}>"))

    def messages_since(self, t0: float) -> list[str]:
        with self._lock:
            return [txt for t, txt in self.lines if t >= t0]

    def find_since(self, t0: float, pattern: re.Pattern):
        for txt in self.messages_since(t0):
            m = pattern.search(txt)
            if m:
                return m
        return None

    def wait(self, cond, timeout_s: float, poll_s: float = 0.05) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if cond():
                return True
            if not self.alive():
                return bool(cond())
            time.sleep(poll_s)
        return False

    def wait_phase(self, phase: str, timeout_s: float) -> bool:
        return self.wait(lambda: self.phase == phase, timeout_s)

    def quit(self, timeout_s: float = 30.0, sigint_wait_s: float = 20.0) -> int | None:
        """q; if the launcher is still running after timeout_s, SIGINT to its process group (a Ctrl-C in its
        terminal: the launcher stops frames, policy and the manager itself), then SIGKILL. Last, every process that
        was under the launcher when quit started and is still running (frames and policy run in sessions of their
        own: a launcher killed before its stop leaves them behind) gets SIGTERM, then SIGKILL (procs.py).
        Not pexpect's terminate(): its SIGHUP comes first and ends the launcher without that stop."""
        from sim_deploy.eval import procs

        if self.child is None:
            return None
        under = procs.tree(self.child.pid) if self.child.isalive() else []
        if self.alive():
            self.send("q")
            self.wait(lambda: not self.alive(), timeout_s)
        if self.child.isalive():
            try:
                os.killpg(self.child.pid, signal.SIGINT)     # pexpect's child leads its own session and group
            except (ProcessLookupError, PermissionError):
                pass
            end = time.monotonic() + sigint_wait_s
            while self.child.isalive() and time.monotonic() < end:
                time.sleep(0.1)
            if self.child.isalive():
                self.child.terminate(force=True)
        left = procs.stop(under[1:], grace_s=10.0)
        if left:
            self.lines.append((time.monotonic(), f"<stopped after quit: {'; '.join(left)}>"))
            if self._log:
                self._log.write(f"\n===== still running after the launcher ended, stopped: {'; '.join(left)}\n")
        self.child.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._log:
            self._log.close()
            self._log = None
        return self.child.exitstatus
