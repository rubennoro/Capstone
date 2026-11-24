# adapted from rnoro/deployment:deployment/deploy/deploy/managed_process.py @ 91b178b
# changes: the config is config/<NAME>.yaml merged with the station section (policy_deploy.contracts.load_config);
# program() defaults to `<python> -m policy_deploy.<NAME>` in the PYTHON_ENV interpreter; cwd is
# deployment/policy_deploy/ (paths.folder()) for every process; process classes live in processes.py (no launch.py
# loader).
"""ManagedProcess: one laptop process of a run (STRUCTURE.md 9.1). processes.py has one subclass per process."""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Mapping

from policy_deploy.contracts import paths
from policy_deploy.contracts.config_merge import load_config, resolve_path
from policy_deploy.contracts.station import Station, StationError
from policy_deploy.launcher.check_result import CheckResult
from policy_deploy.launcher.errors import LaunchError
from policy_deploy.launcher.session import Session

_FATAL_TAIL_BYTES = 65536          # the FATAL line is the last stderr line, so the end of the log is enough


class ManagedProcess(ABC):
    """One process of a run. A subclass says how to check it (checks) and, if needed, what to run (program); this
    class builds the command line, starts it in its own process group with its log captured, waits for its ready
    file, and stops it."""

    NAME: ClassVar[str]
    READY_HINT: ClassVar[str]
    USES_TERMINAL: ClassVar[bool] = False

    def __init__(self, station: Station, mock: bool, session: Session | None) -> None:
        self.station = station
        self.mock = mock
        self.session = session                       # None for `check`; the launcher sets it before start
        self.folder: Path = paths.folder()
        self._config_error: StationError | None = None
        try:
            self.config: Mapping[str, Any] = load_config(self.NAME, station)
        except StationError as exc:                  # reported by base_checks(), so `check` lists it
            self.config = {}
            self._config_error = exc
        self._popen: subprocess.Popen | None = None

    # --- what a subclass writes ---

    def python(self) -> str:
        """The interpreter this process runs in (light unless a subclass says otherwise)."""
        return self.station.python("light")

    def program(self) -> list[str]:
        """The program part of the command: [python, "-m", "policy_deploy.<NAME>"]. Must not start anything."""
        return [self.python(), "-m", f"policy_deploy.{self.NAME}"]

    @abstractmethod
    def checks(self) -> list[CheckResult]:
        """This process's rows of `policy_deploy check`. Quick; never starts the process."""

    def extra_env(self) -> dict[str, str]:
        return {}

    def ready_hint(self) -> str:
        return self.READY_HINT

    def is_ready(self) -> bool:
        return self.ready_file is not None and self.ready_file.exists()

    # --- given ---

    @property
    def log_path(self) -> Path | None:
        return None if self.session is None else self.session.log_path(self.NAME)

    @property
    def ready_file(self) -> Path | None:
        return None if self.session is None else self.session.ready_file(self.NAME)

    def config_path(self, key: str) -> Path | None:
        return resolve_path(self.station, self.NAME, self.config, key)

    def command(self) -> list[str]:
        """program() + --station FILE [--session DIR --ready-file FILE] [--mock] (STRUCTURE.md 2.6)."""
        cmd = [*self.program(), "--station", str(self.station.path)]
        if self.session is not None:
            cmd += ["--session", str(self.session.path), "--ready-file", str(self.ready_file)]
        if self.mock:
            cmd.append("--mock")
        return cmd

    def base_checks(self) -> list[CheckResult]:
        """The station's overrides are valid against the config file; the interpreter exists and is executable."""
        if self._config_error is not None:
            return [CheckResult(self.NAME, "overrides", "fail", "; ".join(self._config_error.problems))]
        n = len(self.station.section(self.NAME))
        label = paths.config_file(self.NAME).name
        results = [CheckResult(self.NAME, "overrides", "pass", f"{n} override(s) of config/{label}, all valid")]
        try:
            program = self.program()[0]
        except (StationError, LaunchError, KeyError, IndexError) as exc:
            results.append(CheckResult(self.NAME, "program", "fail", f"cannot build the command: {exc}"))
            return results
        found = shutil.which(program)
        if found is None:
            results.append(CheckResult(self.NAME, "program", "fail", f"not found or not executable: {program}"))
        else:
            results.append(CheckResult(self.NAME, "program", "pass", f"found: {found}"))
        return results

    def start(self) -> None:
        """Start the command: log captured, stdin closed, cwd = deployment/policy_deploy/, its own process group.
        The USES_TERMINAL process keeps the terminal as stdin/stdout and the launcher's process group."""
        if self.session is None:
            raise LaunchError(f"FAILED {self.NAME}: cannot start without a session")
        cmd = self.command()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **self.extra_env()}
        with open(self.log_path, "w") as log:
            log.write(shlex.join(cmd) + "\n")
            log.flush()
            try:
                if self.USES_TERMINAL:
                    self._popen = subprocess.Popen(cmd, stderr=log, cwd=self.folder, env=env)
                else:
                    self._popen = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                                   stderr=subprocess.STDOUT, cwd=self.folder, env=env,
                                                   start_new_session=True)
            except OSError as exc:
                raise LaunchError(f"FAILED {self.NAME}: cannot start {cmd[0]}: {exc}  [log: {self.log_path}]") \
                    from exc

    def wait_ready(self, timeout_s: float, poll_s: float) -> None:
        """Poll is_ready() every poll_s; LaunchError if the process exits first or timeout_s passes."""
        deadline = time.monotonic() + timeout_s
        while True:
            if self.is_ready():
                return
            code = self.returncode()
            if code is not None:
                raise LaunchError(self.failure_message(code, during_startup=True))
            if time.monotonic() >= deadline:
                raise LaunchError(f"FAILED {self.NAME} during start-up: not ready after {timeout_s:g} s "
                                  f"(waiting for: {self.ready_hint()})  [log: {self.log_path}]")
            time.sleep(poll_s)

    def returncode(self) -> int | None:
        """None while running (or never started)."""
        return None if self._popen is None else self._popen.poll()

    def stop(self, timeout_s: float) -> int | None:
        """SIGTERM, wait up to timeout_s, then SIGKILL. To the process group, except for the USES_TERMINAL process,
        which shares the launcher's group: only its pid is signalled. Returns the exit code (None if never
        started)."""
        if self._popen is None:
            return None
        if self._popen.poll() is not None:
            return self._popen.returncode
        self._signal(signal.SIGTERM)
        try:
            return self._popen.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self._signal(signal.SIGKILL)
            return self._popen.wait()

    def fatal_line(self) -> str | None:
        """The last line of the log that starts with "FATAL " (STRUCTURE.md 2.6), or None."""
        if self.log_path is None or not self.log_path.exists():
            return None
        with open(self.log_path, "rb") as log:
            log.seek(max(0, log.seek(0, os.SEEK_END) - _FATAL_TAIL_BYTES))
            lines = log.read().decode(errors="replace").splitlines()
        fatal = [line.strip() for line in lines if line.startswith("FATAL ")]
        return fatal[-1] if fatal else None

    def failure_message(self, code: int, during_startup: bool = False) -> str:
        """The one-line report of an exit (STRUCTURE.md 9.3), quoting the FATAL line when there is one."""
        when = " during start-up" if during_startup else ""
        fatal = self.fatal_line()
        if fatal is None:
            return (f"FAILED {self.NAME}{when} (exit {code}, no FATAL line): see the last lines of its log  "
                    f"[log: {self.log_path}]")
        return f"FAILED {self.NAME}{when} (exit {code}): {fatal}  [log: {self.log_path}]"

    def _signal(self, sig: signal.Signals) -> None:
        assert self._popen is not None
        try:
            if self.USES_TERMINAL:
                self._popen.send_signal(sig)
            else:
                os.killpg(self._popen.pid, sig)
        except ProcessLookupError:
            pass                                     # already gone: wait() collects its code
