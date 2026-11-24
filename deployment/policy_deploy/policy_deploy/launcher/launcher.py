# adapted from rnoro/deployment:deployment/deploy/deploy/launcher.py @ 91b178b
# changes: ORDER = ("fake_pc", "frames", "policy", "manager"), fake_pc only with --mock and launcher.fake_pc true;
# processes come from processes.py (no launch.py loader); the station rows of STRUCTURE.md 9.2 (station.yaml,
# deploy_contracts, the three contract files, laptop address, Wi-Fi, commit, disk) instead of the TCP triangle;
# timeouts from config/launcher.yaml (+ station launcher:); messages start with `policy_deploy:`; the log is
# logs/launcher.log; the PC is never started or stopped.
# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/launcher/launcher.py @ f632576
# changes: LauncherSettings.fake_pi (launcher.fake_pi, checked like fake_pc), passed to FramesProcess (sim_deploy)
"""Launcher: `policy_deploy check` and `policy_deploy run` (STRUCTURE.md 9.2, 9.3)."""
from __future__ import annotations

import logging
import math
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from deploy_contracts import ContractLimits, LinkSettings, TcpFrame
from deploy_contracts import paths as contract_paths

from policy_deploy.contracts import paths
from policy_deploy.contracts.config_merge import load_config
from policy_deploy.contracts.station import Station, StationError
from policy_deploy.launcher import check_helpers
from policy_deploy.launcher.check_result import CheckResult, exit_code, print_table
from policy_deploy.launcher.errors import LaunchError
from policy_deploy.launcher.managed_process import ManagedProcess
from policy_deploy.launcher.processes import PROCESS_CLASSES, FramesProcess, ManagerProcess
from policy_deploy.launcher.session import Session

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
DISK_WARN_BYTES = 2_000_000_000
GIT_TIMEOUT_S = 5.0
NMCLI_TIMEOUT_S = 5.0
CONTRACTS_HINT = 'uv pip install -e deployment/contracts into ~/envs/policy_deploy (TOP 1.4)'


class _Interrupted(Exception):
    """Raised by the signal handler: Ctrl-C (SIGINT) or SIGTERM reached the launcher."""

    def __init__(self, signum: int) -> None:
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


def _raise_interrupted(signum: int, _frame: object) -> None:
    raise _Interrupted(signum)


@dataclass(frozen=True)
class LauncherSettings:
    """config/launcher.yaml merged with station.yaml's launcher: section."""

    stop_timeout_s: float
    poll_s: float
    fake_pc: bool
    fake_pi: bool
    ready_timeout_s: dict[str, float]

    @classmethod
    def load(cls, station: Station) -> LauncherSettings:
        cfg = load_config("launcher", station)
        problems = []
        for key in ("stop_timeout_s", "poll_s"):
            v = cfg.get(key)
            if not _positive(v):
                problems.append(f"launcher.{key}: expected a number > 0, got {v!r}")
        if not isinstance(cfg.get("fake_pc"), bool):
            problems.append(f"launcher.fake_pc: expected true or false, got {cfg.get('fake_pc')!r}")
        if not isinstance(cfg.get("fake_pi"), bool):
            problems.append(f"launcher.fake_pi: expected true or false, got {cfg.get('fake_pi')!r}")
        timeouts = cfg.get("ready_timeout_s") or {}
        for name in Launcher.ORDER:
            if not _positive(timeouts.get(name)):
                problems.append(f"launcher.ready_timeout_s.{name}: expected a number > 0, got {timeouts.get(name)!r}")
        if problems:
            raise StationError(problems)
        return cls(stop_timeout_s=float(cfg["stop_timeout_s"]), poll_s=float(cfg["poll_s"]), fake_pc=cfg["fake_pc"],
                   fake_pi=cfg["fake_pi"], ready_timeout_s={name: float(timeouts[name]) for name in Launcher.ORDER})


def _positive(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


class Launcher:
    ORDER = ("fake_pc", "frames", "policy", "manager")      # STRUCTURE.md 9.1; not a setting

    def __init__(self, station: Station, mock: bool, processes: Sequence[ManagedProcess] | None = None,
                 settings: LauncherSettings | None = None) -> None:
        self.station = station
        self.mock = mock
        self.settings = settings if settings is not None else LauncherSettings.load(station)
        if processes is None:
            processes = self._build()
        unknown = [p.NAME for p in processes if p.NAME not in self.ORDER]
        if unknown:
            raise ValueError(f"unknown process name(s) {unknown}; known: {self.ORDER}")
        self.processes: list[ManagedProcess] = sorted(processes, key=lambda p: self.ORDER.index(p.NAME))
        self._log = logging.getLogger("policy_deploy.launcher")

    def names(self) -> list[str]:
        """The processes this run starts: fake_pc only with --mock and launcher.fake_pc true."""
        return [n for n in self.ORDER if n != "fake_pc" or (self.mock and self.settings.fake_pc)]

    def _build(self) -> list[ManagedProcess]:
        out = []
        for name in self.names():
            cls = PROCESS_CLASSES[name]
            if cls is ManagerProcess:
                out.append(ManagerProcess(self.station, self.mock, None, fake_pc=self.settings.fake_pc))
            elif cls is FramesProcess:
                out.append(FramesProcess(self.station, self.mock, None, fake_pi=self.settings.fake_pi))
            else:
                out.append(cls(self.station, self.mock, None))
        return out

    # --- policy_deploy check ---

    def check(self) -> int:
        """Print the table of every check; 0 if nothing failed, else 2."""
        results = self._results()
        print_table(results, sys.stdout)
        failed = sum(r.status == "fail" for r in results)
        warned = sum(r.status == "warn" for r in results)
        if failed:
            print(f"policy_deploy check: {failed} check(s) failed")
        else:
            print(f"policy_deploy check: ready ({warned} warning(s))")
        return exit_code(results)

    def _results(self) -> list[CheckResult]:
        """The station rows, then for each process in ORDER: base_checks() then checks()."""
        results = self.station_checks()
        for process in self.processes:
            results += self._process_results(process)
        return results

    @staticmethod
    def _process_results(process: ManagedProcess) -> list[CheckResult]:
        try:
            results = process.base_checks()
        except Exception as exc:                      # report it, keep checking the others
            return [CheckResult(process.NAME, "base checks", "fail", f"crashed: {type(exc).__name__}: {exc}")]
        if any(r.name == "overrides" and r.status == "fail" for r in results):
            return results                            # checks() reads self.config, which is not valid
        try:
            return results + list(process.checks())
        except Exception as exc:
            return results + [CheckResult(process.NAME, "checks", "fail", f"crashed: {type(exc).__name__}: {exc}")]

    # --- the station rows (STRUCTURE.md 9.2) ---

    def station_checks(self) -> list[CheckResult]:
        out = [CheckResult("station", "station.yaml", "pass", f"{self.station.path} (all keys valid)"),
               self._contracts_package()]
        link = None
        for label, load in (("contracts/link.yaml", LinkSettings.load), ("contracts/frames.yaml", TcpFrame.load),
                            ("contracts/limits.yaml", ContractLimits.load)):
            try:
                value = load()
            except Exception as e:                    # FileNotFoundError (install hint), ValueError (file and key)
                out.append(CheckResult("station", label, "fail", f"{type(e).__name__}: {e}"))
                continue
            if isinstance(value, LinkSettings):
                link = value
            out.append(CheckResult("station", label, "pass", self._contract_detail(value)))
        if not self.mock:
            out.append(self._laptop_address(link))
            out.append(self._wifi())
        out.append(self._commit())
        out.append(self._disk())
        return out

    @staticmethod
    def _contract_detail(value: object) -> str:
        if isinstance(value, LinkSettings):
            return (f"PC {value.pc_address}:{value.state_port}/{value.commands_port}, watchdog "
                    f"{value.watchdog_limit_s:g} s, estimator {value.clock_estimator}")
        if isinstance(value, TcpFrame):
            return f"F_T_TCP loaded, settled: {str(value.settled).lower()}"
        if isinstance(value, ContractLimits):
            return (f"rotation_tol {value.rotation_tol:g}, gripper_width_max_m {value.gripper_width_max_m:g}, "
                    f"max_waypoints {value.max_waypoints}")
        return "loaded"

    @staticmethod
    def _contracts_package() -> CheckResult:
        want = (paths.deployment_dir() / "contracts").resolve()
        try:
            found = contract_paths.contracts_dir().resolve()
        except FileNotFoundError as e:
            return CheckResult("station", "deploy_contracts", "fail", f"{e}: {CONTRACTS_HINT}")
        if found != want:
            return CheckResult("station", "deploy_contracts", "fail",
                               f"deploy_contracts comes from {found}, not this checkout's {want}: {CONTRACTS_HINT}")
        return CheckResult("station", "deploy_contracts", "pass", f"from {found}")

    @staticmethod
    def _laptop_address(link: LinkSettings | None) -> CheckResult:
        if link is None:
            return CheckResult("station", "laptop address", "fail", "contracts/link.yaml could not be read")
        local = [a for a in link.laptop_addresses if check_helpers.address_is_local(a)]
        if local:
            return CheckResult("station", "laptop address", "pass", f"{', '.join(local)} is on this laptop")
        return CheckResult("station", "laptop address", "fail",
                           f"none of link.yaml laptop.addresses ({', '.join(link.laptop_addresses)}) is on this "
                           f"laptop: the PC's firewall would drop it; join ASUS_5g or fix laptop.addresses")

    def _wifi(self) -> CheckResult:
        want = self.station.wifi_ssid
        try:
            done = subprocess.run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"], capture_output=True,
                                  text=True, timeout=NMCLI_TIMEOUT_S, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as e:
            return CheckResult("station", "Wi-Fi", "warn", f"cannot ask nmcli ({e}); wired runs may be fine")
        active = [line.split(":", 1)[1] for line in done.stdout.splitlines() if line.startswith("yes:")]
        if want is None:
            return CheckResult("station", "Wi-Fi", "info", f"on {', '.join(active) or 'no Wi-Fi'} "
                                                           f"(station.yaml wifi_ssid not set)")
        if want in active:
            return CheckResult("station", "Wi-Fi", "pass", f"on {want}")
        return CheckResult("station", "Wi-Fi", "warn", f"not on {want} (on: {', '.join(active) or 'none'}); "
                                                       f"wired runs may be fine")

    @staticmethod
    def _commit() -> CheckResult:
        try:
            done = subprocess.run(["git", "-C", str(paths.deployment_dir()), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=GIT_TIMEOUT_S, stdin=subprocess.DEVNULL)
            commit = done.stdout.strip() if done.returncode == 0 else "unknown"
            if commit != "unknown":
                dirty = subprocess.run(["git", "-C", str(paths.deployment_dir()), "status", "--porcelain"],
                                       capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
                                       stdin=subprocess.DEVNULL)
                if dirty.returncode == 0 and dirty.stdout.strip():
                    commit += "+dirty"
        except (OSError, subprocess.TimeoutExpired):
            commit = "unknown"
        return CheckResult("station", "commit", "info", f"laptop commit {commit}")

    def _disk(self) -> CheckResult:
        root = Path(self.station.session_root)
        probe = root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            free = shutil.disk_usage(probe).free
        except OSError as e:
            return CheckResult("station", "disk", "warn", f"cannot read free space under {root}: {e}")
        text = f"{free / 1e9:.1f} GB free under {root}"
        if free < DISK_WARN_BYTES:
            return CheckResult("station", "disk", "warn", text + " (below 2 GB)")
        return CheckResult("station", "disk", "pass", text)

    # --- policy_deploy run ---

    def run(self) -> int:
        """Check, start in ORDER (each after the previous is ready), wait, stop in reverse (STRUCTURE.md 9.3)."""
        terminal = [p.NAME for p in self.processes if p.USES_TERMINAL]
        if len(terminal) > 1:
            print(f"policy_deploy: not started: more than one process uses the terminal: {', '.join(terminal)}")
            return 2
        results = self._results()
        print_table(results, sys.stdout)
        failed = sum(r.status == "fail" for r in results)
        if failed:
            print(f"policy_deploy: not started: {failed} check(s) failed (see above)")
            return 2
        try:
            session = Session.create(self.station.session_root, self.mock, self.station.path)
        except (LaunchError, OSError) as exc:
            print(f"policy_deploy: not started: cannot create the session folder: {exc}")
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
                code = 130
                message = "policy_deploy: stopped by Ctrl-C" if sigint else "policy_deploy: stopped by SIGTERM"
                self._log.info(message)
            try:
                self._stop_all(started, sigint)
            except _Interrupted:                      # a second Ctrl-C during the stop: SIGKILL, exit at once
                for sig in old:
                    signal.signal(sig, signal.SIG_IGN)
                self._log.info("second signal during the stop: killing everything still running")
                for process in reversed(started):
                    self._log.info("killed %s (exit %s)", process.NAME, process.stop(0.0))
                code, message = 130, "policy_deploy: killed by a second Ctrl-C during the stop"
        finally:
            for sig, previous in old.items():
                signal.signal(sig, previous)
        self._log.info("exit %d", code)
        print(message, flush=True)
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
                process.wait_ready(self.settings.ready_timeout_s[process.NAME], self.settings.poll_s)
            except LaunchError as exc:
                self._log.info("%s", exc)
                return 1, f"policy_deploy: {exc}"
            self._log.info("%s ready", process.NAME)
        message = f"policy_deploy: all {len(self.processes)} processes ready (session: {session.path})"
        print(message, flush=True)
        self._log.info(message)
        while True:
            for process in started:
                code = process.returncode()
                if code is None:
                    continue
                if process.NAME == "manager" and code == 0:
                    self._log.info("manager exited 0 (operator quit)")
                    return 0, "policy_deploy: done"
                failure = process.failure_message(code)
                self._log.info("%s", failure)
                return 1, f"policy_deploy: {failure}"
            time.sleep(self.settings.poll_s)

    def _stop_all(self, started: list[ManagedProcess], sigint: bool) -> None:
        """Reverse order. After a Ctrl-C the USES_TERMINAL process got the SIGINT itself: wait for it first."""
        for process in reversed(started):
            if sigint and process.USES_TERMINAL:
                deadline = time.monotonic() + self.settings.stop_timeout_s
                while process.returncode() is None and time.monotonic() < deadline:
                    time.sleep(self.settings.poll_s)
            self._log.info("stopping %s", process.NAME)
            self._log.info("%s stopped (exit %s)", process.NAME, process.stop(self.settings.stop_timeout_s))

    def _open_log(self, session: Session) -> logging.Handler:
        handler = logging.FileHandler(session.log_path("launcher"))
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger = logging.getLogger("policy_deploy.launcher")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False                      # the terminal gets only the launcher's own printed lines
        return handler

    @staticmethod
    def _close_log(handler: logging.Handler) -> None:
        logging.getLogger("policy_deploy.launcher").removeHandler(handler)
        handler.close()
