"""Launcher: `deploy check` and `deploy run` (STRUCTURE.md sections 3.5 and 5)."""
from __future__ import annotations

import logging
import signal
import sys
import time
from collections.abc import Sequence

import deploy
from deploy.check_result import CheckResult, exit_code, print_table
from deploy.errors import LaunchError
from deploy.geometry.tcp_triangle import triangle_check
from deploy.loader import load_process_class
from deploy.managed_process import ManagedProcess
from deploy.session import Session
from deploy.station import Station

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


class _Interrupted(Exception):
    """Raised by the signal handler: Ctrl-C (SIGINT) or SIGTERM reached the launcher."""

    def __init__(self, signum: int) -> None:
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


def _raise_interrupted(signum: int, _frame: object) -> None:
    raise _Interrupted(signum)


class Launcher:
    ORDER = ("franka_control", "process_frames", "policy", "message_comms")   # PLAN §6.5; not a setting

    def __init__(self, station: Station, mock: bool, processes: Sequence[ManagedProcess] | None = None) -> None:
        self.station = station
        self.mock = mock
        self._load_failures: list[CheckResult] = []
        if processes is None:
            processes = self._build()
        unknown = [p.NAME for p in processes if p.NAME not in self.ORDER]
        if unknown:
            raise ValueError(f"unknown process name(s) {unknown}; known: {self.ORDER}")
        self.processes: list[ManagedProcess] = sorted(processes, key=lambda p: self.ORDER.index(p.NAME))
        self._log = logging.getLogger("deploy.launcher")

    def _build(self) -> list[ManagedProcess]:
        processes = []
        for name in self.ORDER:
            try:
                processes.append(load_process_class(deploy.DEPLOYMENT_ROOT / name)(self.station, self.mock, None))
            except LaunchError as exc:                # a check failure: deploy check keeps going
                self._load_failures.append(CheckResult(name, "launch.py", "fail", str(exc)))
        return processes

    # --- deploy check ---

    def check(self) -> int:
        """Print the table of every check; 0 if nothing failed, else 2."""
        results = self._results()
        print_table(results, sys.stdout)
        failed = sum(r.status == "fail" for r in results)
        warned = sum(r.status == "warn" for r in results)
        if failed:
            print(f"deploy check: {failed} check(s) failed")
        else:
            print(f"deploy check: ready ({warned} warning(s))")
        return exit_code(results)

    def _results(self) -> list[CheckResult]:
        """The TCP triangle, then for each process in ORDER: base_checks() then checks()."""
        results = [triangle_check(self.station, deploy.DEPLOYMENT_ROOT / "gripper.yaml", self.mock)]
        failures = {r.process: r for r in self._load_failures}
        by_name = {p.NAME: p for p in self.processes}
        for name in self.ORDER:
            if name in failures:
                results.append(failures[name])
            elif name in by_name:
                results += self._process_results(by_name[name])
        return results

    @staticmethod
    def _process_results(process: ManagedProcess) -> list[CheckResult]:
        try:
            results = process.base_checks()
        except Exception as exc:                      # someone else's code: report it, keep checking the others
            return [CheckResult(process.NAME, "base checks", "fail", f"crashed: {type(exc).__name__}: {exc}")]
        if any(r.name == "overrides" and r.status == "fail" for r in results):
            return results                            # checks() reads self.config, which is not valid
        try:
            return results + list(process.checks())
        except Exception as exc:
            return results + [CheckResult(process.NAME, "checks", "fail", f"crashed: {type(exc).__name__}: {exc}")]

    # --- deploy run ---

    def run(self) -> int:
        """Check, start in ORDER (each after the previous is ready), wait, stop in reverse (section 5)."""
        terminal = [p.NAME for p in self.processes if p.USES_TERMINAL]
        if len(terminal) > 1:
            print(f"deploy: not started: more than one process uses the terminal: {', '.join(terminal)}")
            return 2
        results = self._results()
        print_table(results, sys.stdout)
        failed = sum(r.status == "fail" for r in results)
        if failed:
            print(f"deploy: not started: {failed} check(s) failed (see above)")
            return 2
        try:
            session = Session.create(self.station.session_root, self.mock, self.station.path)
        except (LaunchError, OSError) as exc:
            print(f"deploy: not started: cannot create the session folder: {exc}")
            return 2
        handler = self._open_log(session)
        old = {sig: signal.signal(sig, _raise_interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        started: list[ManagedProcess] = []
        try:
            try:
                code, message = self._start_and_wait(session, started)
                sigint = False
            except _Interrupted as exc:
                sigint = exc.signum == signal.SIGINT
                code, message = 130, ("deploy: stopped by Ctrl-C" if sigint else "deploy: stopped by SIGTERM")
                self._log.info(message)
            try:
                self._stop_all(started, sigint)
            except _Interrupted:                      # a second Ctrl-C during the stop: SIGKILL, exit at once
                for sig in old:
                    signal.signal(sig, signal.SIG_IGN)
                self._log.info("second signal during the stop: killing everything still running")
                for process in reversed(started):
                    self._log.info("killed %s (exit %s)", process.NAME, process.stop(0.0))
                code, message = 130, "deploy: killed by a second Ctrl-C during the stop"
        finally:
            for sig, previous in old.items():
                signal.signal(sig, previous)
        self._log.info("exit %d", code)
        print(message)
        self._close_log(handler)
        return code

    def _start_and_wait(self, session: Session, started: list[ManagedProcess]) -> tuple[int, str]:
        for process in self.processes:
            process.session = session
        for process in self.processes:
            started.append(process)                   # before start(): a signal mid-start still stops it
            self._log.info("starting %s: %s", process.NAME, " ".join(process.command()))
            try:
                process.start()
                process.wait_ready(self.station.ready_timeout_s[process.NAME], self.station.poll_s)
            except LaunchError as exc:
                self._log.info("%s", exc)
                return 1, f"deploy: {exc}"
            self._log.info("%s ready", process.NAME)
        message = f"deploy: all {len(self.processes)} processes ready (session: {session.path})"
        print(message)
        self._log.info(message)
        while True:
            for process in started:
                code = process.returncode()
                if code is None:
                    continue
                if process.NAME == "message_comms" and code == 0:
                    self._log.info("message_comms exited 0 (operator quit)")
                    return 0, f"deploy: done (session: {session.path})"
                failure = process.failure_message(code)
                self._log.info("%s", failure)
                return 1, f"deploy: {failure}"
            time.sleep(self.station.poll_s)

    def _stop_all(self, started: list[ManagedProcess], sigint: bool) -> None:
        """Reverse order. After a Ctrl-C the USES_TERMINAL process got the SIGINT itself: wait for it first."""
        for process in reversed(started):
            if sigint and process.USES_TERMINAL:
                deadline = time.monotonic() + self.station.stop_timeout_s
                while process.returncode() is None and time.monotonic() < deadline:
                    time.sleep(self.station.poll_s)
            self._log.info("stopping %s", process.NAME)
            self._log.info("%s stopped (exit %s)", process.NAME, process.stop(self.station.stop_timeout_s))

    def _open_log(self, session: Session) -> logging.Handler:
        handler = logging.FileHandler(session.log_path("deploy"))
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger = logging.getLogger("deploy.launcher")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False                      # the terminal gets only the launcher's own printed lines
        return handler

    @staticmethod
    def _close_log(handler: logging.Handler) -> None:
        logging.getLogger("deploy.launcher").removeHandler(handler)
        handler.close()
