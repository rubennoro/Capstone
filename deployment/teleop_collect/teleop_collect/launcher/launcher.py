# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/launcher/launcher.py @ 2ffe641
# changes: ORDER = ("fake_pc", "frames", "teleop", "ui"); TeleopProcess gets the fake_pc switch (was ManagerProcess);
# messages start with `teleop_collect:`; the station rows of STRUCTURE.md 10.1: the install hint names
# ~/envs/teleop_collect, the disk row looks under data_root (was session_root), + the data_root writable row; on
# ready also `teleop_collect: ready: open <station.http_url()>` and webbrowser.open (unless --no-browser,
# ui.open_browser false, or no display); a `starting ...` line and a note before teleop starts (the start pose); no
# USES_TERMINAL process (no terminal check, no wait for it after a Ctrl-C); SIGHUP (the terminal closed) is handled
# like SIGTERM, since the processes run in their own sessions and would outlive the launcher; an operator stop
# (Ctrl-C, SIGTERM or SIGHUP to the launcher) exits 0 (was 130; PLAN 7), the last line is not printed if the
# terminal is gone; a second signal during the stop still kills everything and exits 130;
# teleop exiting 0 by itself is a stop (as the manager's operator quit), any other exit while running is the FAILED
# line (PLAN 7). + start pose (final-deployment's start-pose work, 927a5ba, 2026-11-20): the note before
# teleop says the start pose is the arm's pose at start-up (h, then p on the PC first); no start-up home note. The
# source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/deploy/launcher.py @ 91b178b
# changes: ORDER = ("fake_pc", "frames", "policy", "manager"), fake_pc only with --mock and launcher.fake_pc true;
# processes come from processes.py (no launch.py loader); the station rows of STRUCTURE.md 9.2 (station.yaml,
# deploy_contracts, the three contract files, laptop address, Wi-Fi, commit, disk) instead of the TCP triangle;
# timeouts from config/launcher.yaml (+ station launcher:); messages start with `policy_deploy:`; the log is
# logs/launcher.log; the PC is never started or stopped.
"""Launcher: `teleop_collect check` and `teleop_collect run` (STRUCTURE.md 10.1, PLAN.md 3, 4.2, 7).

The launcher only starts, watches and stops the laptop's own processes. It never starts, stops, reaches into or
reads files of the PC or the Pi: their rows come from network probes (the Pi's TCP port) and from pc_probe, which
speaks only C2/C3.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from deploy_contracts import ContractLimits, LinkSettings, TcpFrame
from deploy_contracts import paths as contract_paths

from teleop_collect.contracts import paths
from teleop_collect.contracts.config_merge import load_config
from teleop_collect.contracts.station import Station, StationError
from teleop_collect.launcher import check_helpers
from teleop_collect.launcher.check_result import CheckResult, exit_code, print_table
from teleop_collect.launcher.errors import LaunchError
from teleop_collect.launcher.managed_process import ManagedProcess
from teleop_collect.launcher.processes import PROCESS_CLASSES, TeleopProcess
from teleop_collect.launcher.session import Session

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
DISK_WARN_BYTES = 2_000_000_000
GIT_TIMEOUT_S = 5.0
NMCLI_TIMEOUT_S = 5.0
CONTRACTS_HINT = "~/envs/teleop_collect/bin/pip install -e deployment/contracts (PLAN 4.1)"
EXIT_KILLED = 130                                     # a second signal during the stop
START_POSE_NOTE = "teleop_collect: teleop: the start pose is the arm's pose at start-up"
#: The signals that stop a run. SIGHUP: closing the terminal must stop the processes too (they run in their own
#: sessions, so they never get the terminal's hangup themselves and would keep serving the page).
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class _Interrupted(Exception):
    """Raised by the signal handler: Ctrl-C (SIGINT), SIGTERM or SIGHUP (the terminal closed) reached the launcher."""

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
        timeouts = cfg.get("ready_timeout_s") or {}
        for name in Launcher.ORDER:
            if not _positive(timeouts.get(name)):
                problems.append(f"launcher.ready_timeout_s.{name}: expected a number > 0, got {timeouts.get(name)!r}")
        if problems:
            raise StationError(problems)
        return cls(stop_timeout_s=float(cfg["stop_timeout_s"]), poll_s=float(cfg["poll_s"]), fake_pc=cfg["fake_pc"],
                   ready_timeout_s={name: float(timeouts[name]) for name in Launcher.ORDER})


def _positive(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _ui_opens_browser(station: Station) -> bool:
    """ui.open_browser (config/ui.yaml + station ui:); a ui section that does not load is the ui's own FATAL."""
    try:
        return load_config("ui", station).get("open_browser") is not False
    except StationError:
        return True


def _has_display() -> bool:
    """A graphical session: without one, webbrowser would pick a console browser and take over the terminal."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


class Launcher:
    ORDER = ("fake_pc", "frames", "teleop", "ui")      # STRUCTURE.md 10.1; not a setting

    def __init__(self, station: Station, mock: bool, processes: Sequence[ManagedProcess] | None = None,
                 settings: LauncherSettings | None = None, open_browser: bool = True) -> None:
        self.station = station
        self.mock = mock
        self.settings = settings if settings is not None else LauncherSettings.load(station)
        self.open_browser = open_browser and _ui_opens_browser(station)
        if processes is None:
            processes = self._build()
        unknown = [p.NAME for p in processes if p.NAME not in self.ORDER]
        if unknown:
            raise ValueError(f"unknown process name(s) {unknown}; known: {self.ORDER}")
        self.processes: list[ManagedProcess] = sorted(processes, key=lambda p: self.ORDER.index(p.NAME))
        self._log = logging.getLogger("teleop_collect.launcher")

    def names(self) -> list[str]:
        """The processes this run starts: fake_pc only with --mock and launcher.fake_pc true."""
        return [n for n in self.ORDER if n != "fake_pc" or (self.mock and self.settings.fake_pc)]

    def _build(self) -> list[ManagedProcess]:
        out = []
        for name in self.names():
            cls = PROCESS_CLASSES[name]
            if cls is TeleopProcess:
                out.append(TeleopProcess(self.station, self.mock, None, fake_pc=self.settings.fake_pc))
            else:
                out.append(cls(self.station, self.mock, None))
        return out

    # --- teleop_collect check ---

    def check(self) -> int:
        """Print the table of every check; 0 if nothing failed, else 2."""
        results = self._results()
        print_table(results, sys.stdout)
        failed = sum(r.status == "fail" for r in results)
        warned = sum(r.status == "warn" for r in results)
        if failed:
            print(f"teleop_collect check: {failed} check(s) failed")
        else:
            print(f"teleop_collect check: ready ({warned} warning(s))")
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

    # --- the station rows (STRUCTURE.md 10.1) ---

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
        out.append(check_helpers.writable_folder("station", "data_root", self.station.data_root))
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
        root = Path(self.station.data_root)
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

    # --- teleop_collect run ---

    def run(self) -> int:
        """Check, start in ORDER (each after the previous is ready), open the page, watch, stop in reverse
        (STRUCTURE.md 10.1). Exit 0 for an operator stop, 1 for a failure, 2 if nothing was started."""
        results = self._results()
        print_table(results, sys.stdout)
        failed = sum(r.status == "fail" for r in results)
        if failed:
            print(f"teleop_collect: not started: {failed} check(s) failed (see above)")
            return 2
        try:
            session = Session.create(self.station.session_root, self.mock, self.station.path)
        except (LaunchError, OSError) as exc:
            print(f"teleop_collect: not started: cannot create the session folder: {exc}")
            return 2
        handler = self._open_log(session)
        old = {sig: signal.signal(sig, _raise_interrupted) for sig in STOP_SIGNALS}
        started: list[ManagedProcess] = []
        try:
            try:
                code, message = self._start_and_wait(session, started)
            except _Interrupted as exc:
                code = 0                              # an operator stop (PLAN 7)
                message = ("teleop_collect: stopped by Ctrl-C" if exc.signum == signal.SIGINT
                           else f"teleop_collect: stopped by {signal.Signals(exc.signum).name}")
                self._log.info(message)
            try:
                self._stop_all(started)
            except _Interrupted:                      # a second Ctrl-C during the stop: SIGKILL, exit at once
                for sig in old:
                    signal.signal(sig, signal.SIG_IGN)
                self._log.info("second signal during the stop: killing everything still running")
                for process in reversed(started):
                    self._log.info("killed %s (exit %s)", process.NAME, process.stop(0.0))
                code, message = EXIT_KILLED, "teleop_collect: killed by a second Ctrl-C during the stop"
        finally:
            for sig, previous in old.items():
                signal.signal(sig, previous)
        self._log.info("exit %d", code)
        try:
            print(message, flush=True)
        except OSError:                               # the terminal is gone (SIGHUP): the log has it
            pass
        self._close_log(handler)
        return code

    def _start_and_wait(self, session: Session, started: list[ManagedProcess]) -> tuple[int, str]:
        for process in self.processes:
            process.session = session
        print(f"teleop_collect: starting {', '.join(p.NAME for p in self.processes)} (session: {session.path})",
              flush=True)
        for process in self.processes:
            started.append(process)                   # before start(): a signal mid-start still stops it
            self._log.info("starting %s: %s", process.NAME, " ".join(process.command()))
            note = self._start_note(process)
            if note:
                print(note, flush=True)
            try:
                process.start()
                process.wait_ready(self.settings.ready_timeout_s[process.NAME], self.settings.poll_s)
            except LaunchError as exc:
                self._log.info("%s", exc)
                return 1, f"teleop_collect: {exc}"
            self._log.info("%s ready", process.NAME)
        message = f"teleop_collect: all {len(self.processes)} processes ready (session: {session.path})"
        print(message, flush=True)
        self._log.info(message)
        self._announce_page()
        while True:
            for process in started:
                code = process.returncode()
                if code is None:
                    continue
                if process.NAME == "teleop" and code == 0:
                    self._log.info("teleop exited 0 (stopped)")
                    return 0, "teleop_collect: teleop stopped (exit 0): done"
                failure = process.failure_message(code)
                self._log.info("%s", failure)
                return 1, f"teleop_collect: {failure}"
            time.sleep(self.settings.poll_s)

    def _start_note(self, process: ManagedProcess) -> str | None:
        """One line before teleop starts (PLAN 4.2 step 5): its start-up takes the arm's pose as the start pose."""
        if isinstance(process, TeleopProcess):
            if self.mock:
                return f"{START_POSE_NOTE}: the fake PC starts at its start pose"
            return f"{START_POSE_NOTE}: press h, then p on the PC first"
        return None

    def _announce_page(self) -> None:
        """`teleop_collect: ready: open <url>`, then the browser (PLAN 4.2)."""
        url = self.station.http_url()
        print(f"teleop_collect: ready: open {url}", flush=True)
        self._log.info("ready: open %s", url)
        if not self.open_browser:
            return
        if not _has_display():
            print("teleop_collect: no display: open the page by hand", flush=True)
            return
        try:
            opened = webbrowser.open(url)
        except Exception as exc:                      # webbrowser.Error, OSError: the run goes on without it
            opened = False
            self._log.info("webbrowser.open failed: %s", exc)
        if not opened:
            print("teleop_collect: could not open a browser: open the page by hand", flush=True)

    def _stop_all(self, started: list[ManagedProcess]) -> None:
        """Reverse order (ui, teleop, frames, fake_pc): teleop sends its C3 stop while the PC (or fake PC) is up."""
        for process in reversed(started):
            self._log.info("stopping %s", process.NAME)
            self._log.info("%s stopped (exit %s)", process.NAME, process.stop(self.settings.stop_timeout_s))

    def _open_log(self, session: Session) -> logging.Handler:
        handler = logging.FileHandler(session.log_path("launcher"))
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger = logging.getLogger("teleop_collect.launcher")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False                      # the terminal gets only the launcher's own printed lines
        return handler

    @staticmethod
    def _close_log(handler: logging.Handler) -> None:
        logging.getLogger("teleop_collect.launcher").removeHandler(handler)
        handler.close()
