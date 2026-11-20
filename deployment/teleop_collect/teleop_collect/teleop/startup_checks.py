# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/startup_checks.py @ 2ffe641
# changes: no settled rule and no policy (teleop_collect PLAN 8, 13.9); waypoint_lead_ms = K x Delta
# (teleop.motion.waypoints x spacing_ms); the hints name teleop.* keys and say "the same contract files" instead
# of "the same commit" (this branch's commit differs from the PC's by design, PLAN 8); laptop_commit() counts
# untracked files as dirty (no --untracked-files=no, as the launcher); + split_commit(); DIRTY; imports. + start pose
# (final-deployment's start-pose work, 927a5ba, 2026-11-20): no home timeout row (teleop sends no C3
# home, so timeouts.home_s is not compared with the PC's homing_timeout_s).
"""StartupChecks: the hello and budget checks of start-up step 5 (STRUCTURE.md 6.4), run by teleop, pc_probe --check
and so teleop_collect check.

| check           | rule                                                                                 | on failure  |
| equality        | deploy_contracts.compare_hello(reply, mock=<own --mock>)                             | refuse      |
| worst gap       | heartbeat + c3 timeout + health timeout + jitter allowance < reply.watchdog_limit_s  | refuse      |
| waypoint lead   | motion.waypoints x motion.spacing_ms < reply.max_lead_s                              | refuse      |
| clock agreement | |agreement_ms| of the shortest-RTT start-up hello <= clock.startup_agree_ms          | refuse      |
| commit          | reply.commit vs git rev-parse HEAD (a different commit is expected: shown)           | warning     |
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

from deploy_contracts import HelloReply, compare_hello

from teleop_collect.contracts import paths
from teleop_collect.teleop.pc_clock import HelloSample, PcClock
from teleop_collect.teleop.settings import Settings

DIRTY = "+dirty"


@dataclass(frozen=True)
class CheckFailure:
    name: str             # "equality", "worst_gap", "waypoint_lead", "clock_agreement"
    what: str             # the FATAL line's "what failed"
    check: str            # its "what to check"

    def __str__(self) -> str:
        return f"{self.what}: {self.check}"


@dataclass(frozen=True)
class CheckRow:
    """One row for pc_probe --check (JSON check lines)."""
    name: str
    status: str           # pass | warn | fail | info
    detail: str


def laptop_commit() -> str:
    """`git rev-parse HEAD` of this checkout (+dirty when there are uncommitted changes), or "unknown"."""
    folder = str(paths.deployment_dir())
    try:
        head = subprocess.run(["git", "-C", folder, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              stdin=subprocess.DEVNULL)
        if head.returncode != 0:
            return "unknown"
        commit = head.stdout.strip()
        dirty = subprocess.run(["git", "-C", folder, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
        return commit + (DIRTY if dirty.returncode == 0 and dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def split_commit(commit: str) -> tuple[str, bool]:
    """"<sha>+dirty" -> ("<sha>", True) (meta.yaml keeps them apart)."""
    if commit.endswith(DIRTY):
        return commit[:-len(DIRTY)], True
    return commit, False


def shortest_hello(hellos: list[HelloSample]) -> HelloSample | None:
    return min(hellos, key=lambda s: s.rtt_ns) if hellos else None


class StartupChecks:
    def __init__(self, settings: Settings, mock: bool, commit: str | None = None) -> None:
        self._s = settings
        self._mock = mock
        self._commit = commit

    @property
    def commit(self) -> str:
        if self._commit is None:
            self._commit = laptop_commit()
        return self._commit

    # ---- the rows

    def equality(self, reply: HelloReply) -> list[CheckFailure]:
        s = self._s
        out = []
        for m in compare_hello(reply, mock=self._mock, frame=s.frame, link=s.link):
            out.append(CheckFailure("equality", f"hello mismatch: {m.field}: PC {m.pc}, laptop {m.expected}",
                                    _EQUALITY_CHECK.get(m.field, "check that both machines run the same contract "
                                                                 "files")))
        return out

    def worst_gap(self, reply: HelloReply) -> CheckFailure | None:
        gap = self._s.worst_gap_s
        if gap < reply.watchdog_limit_s:
            return None
        return CheckFailure("worst_gap", f"hello mismatch: worst heartbeat gap {gap:.2f} s is not below the PC's "
                                         f"watchdog {reply.watchdog_limit_s:.2f} s",
                            "raise watchdog_limit_s in link.yaml or lower the laptop timeouts")

    def waypoint_lead_ms(self) -> float:
        return self._s.waypoint_lead_ms

    def waypoint_lead(self, reply: HelloReply) -> CheckFailure | None:
        lead_ms = self.waypoint_lead_ms()
        if lead_ms < reply.max_lead_s * 1000.0:
            return None
        m = self._s.motion
        return CheckFailure("waypoint_lead", f"hello mismatch: the longest waypoint lead {lead_ms:.0f} ms (waypoints "
                                             f"{m.waypoints} x spacing_ms {m.spacing_ms:g}) is not below the PC's "
                                             f"max_lead_s {reply.max_lead_s:g} s",
                            "lower teleop.motion.waypoints or spacing_ms")

    def clock_agreement(self, pc_clock: PcClock, hellos: list[HelloSample]) -> CheckFailure | None:
        best = shortest_hello(hellos)
        if best is None:
            return CheckFailure("clock_agreement", "clock mismatch at start-up: no hello sample", "run pc_probe")
        agree = pc_clock.agreement_ms(best)
        if abs(agree) <= self._s.clock.startup_agree_ms:
            return None
        return CheckFailure("clock_agreement", f"clock mismatch at start-up: map and hello differ by "
                                               f"{abs(agree):.1f} ms",
                            "run pc_probe; try clock.estimator round_trip")

    def commit_warning(self, reply: HelloReply) -> str | None:
        """None when the PC's commit equals the laptop's; else the warning text (expected: this branch differs from
        the PC's rnoro/final-deployment, PLAN 8)."""
        ours = self.commit
        if reply.commit != "unknown" and ours != "unknown" and ours.split("+")[0] == reply.commit:
            if ours.endswith(DIRTY):
                return f"laptop commit {ours} has uncommitted changes (PC {reply.commit})"
            return None
        return f"PC commit {reply.commit} differs from laptop commit {ours}"

    # ---- all of step 5

    def hello_failures(self, reply: HelloReply, pc_clock: PcClock, hellos: list[HelloSample]) -> list[CheckFailure]:
        """Every refusing row of start-up step 5, in table order."""
        out = list(self.equality(reply))
        for f in (self.worst_gap(reply), self.waypoint_lead(reply),
                  self.clock_agreement(pc_clock, hellos)):
            if f is not None:
                out.append(f)
        return out

    def rows(self, reply: HelloReply, pc_clock: PcClock, hellos: list[HelloSample]) -> list[CheckRow]:
        """The rows of the table above as check lines (pc_probe --check)."""
        rows: list[CheckRow] = []
        eq = self.equality(reply)
        rows.append(CheckRow("hello equality", "fail" if eq else "pass",
                             "; ".join(str(f) for f in eq) if eq else
                             "schemas, mock, F_T_TCP, limits and watchdog equal"))
        for name, f, ok in (("worst gap", self.worst_gap(reply),
                             f"{self._s.worst_gap_s:.2f} s < watchdog {reply.watchdog_limit_s:.2f} s"),
                            ("waypoint lead", self.waypoint_lead(reply),
                             f"{self.waypoint_lead_ms():.0f} ms < max_lead_s {reply.max_lead_s:g} s")):
            rows.append(CheckRow(name, "fail" if f else "pass", str(f) if f else ok))
        if pc_clock.has_fit():
            f = self.clock_agreement(pc_clock, hellos)
            best = shortest_hello(hellos)
            ok = (f"{pc_clock.NAME} map and the shortest-RTT hello agree within "
                  f"{abs(pc_clock.agreement_ms(best)):.2f} ms" if best is not None else "")
            rows.append(CheckRow("clock agreement", "fail" if f else "pass",
                                 f"{pc_clock.NAME}: {f}" if f else ok))
        else:
            rows.append(CheckRow("clock agreement", "fail",
                                 "no clock fit: no hello answered" if pc_clock.NAME == "round_trip" else
                                 f"no {pc_clock.NAME} clock fit yet: too few C2 states"))
        warn = self.commit_warning(reply)
        rows.append(CheckRow("PC commit", "warn" if warn else "pass", warn or f"PC and laptop on {reply.commit}"))
        rows.append(CheckRow("PC mock", "info", f"the PC runs {'--mock (fake arm)' if reply.mock else 'the real arm'}"))
        return rows


_EQUALITY_CHECK = {
    "schemas": "check that both machines run the same contract files (rebuild franka_control after every pull)",
    "mock": "run both sides with --mock, or neither",
    "F_T_TCP": "both machines need the same deployment/contracts/frames.yaml (restart franka_control)",
    "rotation_tol": "both machines need the same deployment/contracts/limits.yaml",
    "gripper_width_max_m": "both machines need the same deployment/contracts/limits.yaml",
    "max_waypoints": "both machines need the same deployment/contracts/limits.yaml",
    "watchdog_limit_s": "both machines need the same deployment/contracts/link.yaml (restart franka_control)",
}
