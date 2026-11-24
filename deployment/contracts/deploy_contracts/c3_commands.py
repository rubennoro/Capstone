# adapted from rnoro/deployment:deployment/contracts/contracts/c3_commands.py @ 91b178b
# changes: imports renamed; + Hello (c3.hello/v1) and HelloReply (c3.hello.reply/v1), STRUCTURE.md 3.8.
"""C3: laptop manager -> franka_control (PC): waypoints and commands (STRUCTURE.md 3.7, 3.8).

Replies: HelloReply for hello; Ack for heartbeat, start, stop, home; ArmState for get_state; SetWaypointsReply for
set_waypoints, or ErrorReply when set_waypoints is refused because of the mode (not a fault).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from deploy_contracts.c2_state import rigid_problem
from deploy_contracts.common import Ack
from deploy_contracts.fields import array, list_of, scalar, str_list
from deploy_contracts.limits import ContractLimits
from deploy_contracts.message import Message, Record
from deploy_contracts.registry import register
from deploy_contracts.schema import SchemaName


@register
@dataclass(frozen=True, eq=False)
class SetWaypoints(Message):
    """Three parallel arrays of length N: waypoint i is (t_ns[i], O_T_TCP[i], gripper_width_m[i])."""

    SCHEMA = "c3.set_waypoints/v1"
    chunk_id: int = scalar("uint64")                        # which policy step produced these
    t_ns: np.ndarray = array("int64", (None,))              # when each waypoint should be reached, PC monotonic
    O_T_TCP: np.ndarray = array("float64", (None, 4, 4))    # absolute TCP pose of each waypoint, base frame
    gripper_width_m: np.ndarray = array("float64", (None,))  # absolute opening, m, >= 0

    def check_values(self) -> None:
        limits = ContractLimits.load()
        n = len(self.t_ns)
        if not 1 <= n <= limits.max_waypoints:
            self.fail("t_ns", f"{n} waypoints, expected 1 to {limits.max_waypoints}")
        if self.O_T_TCP.shape[0] != n:
            self.fail("O_T_TCP", f"{self.O_T_TCP.shape[0]} poses for {n} times")
        if self.gripper_width_m.shape[0] != n:
            self.fail("gripper_width_m", f"{self.gripper_width_m.shape[0]} widths for {n} times")
        if np.any(np.diff(self.t_ns) <= 0):
            self.fail("t_ns", "times not strictly increasing")
        for name in ("O_T_TCP", "gripper_width_m"):
            if not np.all(np.isfinite(getattr(self, name))):
                self.fail(name, "not finite")
        for i in range(n):
            problem = rigid_problem(self.O_T_TCP[i], limits.rotation_tol)
            if problem is not None:
                self.fail("O_T_TCP", f"not a rigid transform at [{i}]: {problem}")
        widths = self.gripper_width_m.astype(np.float64)
        if np.any(widths < 0.0):
            self.fail("gripper_width_m", f"negative width: {widths.min()}")


@dataclass(frozen=True, eq=False)
class WaypointIssue(Record):
    index: int = scalar("uint64")      # position in the request
    reason: str = scalar("str")


@register
@dataclass(frozen=True, eq=False)
class SetWaypointsReply(Message):
    SCHEMA = "c3.set_waypoints.reply/v1"
    chunk_id: int = scalar("uint64")                  # echo of the request
    accepted: int = scalar("uint64")                  # accepted, shortened ones included
    shortened: int = scalar("uint64")                 # accepted ones shortened to the speed limit
    skipped: int = scalar("uint64")                   # skipped: their time had already passed
    rejected: tuple[WaypointIssue, ...] = list_of(WaypointIssue)   # refused by the gate, each with its reason

    def check_values(self) -> None:
        if self.shortened > self.accepted:
            self.fail("shortened", f"{self.shortened} > accepted {self.accepted}")
        seen: set[int] = set()
        for i, issue in enumerate(self.rejected):
            if issue.index in seen:
                self.fail(f"rejected[{i}].index", f"index {issue.index} listed twice")
            seen.add(issue.index)


@register
@dataclass(frozen=True, eq=False)
class Heartbeat(Message):
    SCHEMA = "c3.heartbeat/v1"


@register
@dataclass(frozen=True, eq=False)
class Start(Message):
    SCHEMA = "c3.start/v1"


@register
@dataclass(frozen=True, eq=False)
class Stop(Message):
    SCHEMA = "c3.stop/v1"


@register
@dataclass(frozen=True, eq=False)
class Home(Message):
    """Replied with Ack when the move ends: ok; or not ok on refusal ("refused: ..."), failure or timeout (any other
    text), or a stop (HOME_STOPPED_PREFIX, one space, the text; the PC is then `stopped`). STRUCTURE.md 3.7.2."""

    SCHEMA = "c3.home/v1"


HOME_STOPPED_PREFIX = "stopped:"   # exactly these 8 characters, then one space and the text (STRUCTURE.md 3.7.2)


def home_was_stopped(reply: Ack) -> bool:
    """True for the reply of a home aborted by a stop (the PC's s key): ok false, reason "stopped: <text>".

    No other home reply starts with the prefix; the laptop tells the cases apart by it alone.
    """
    return (not reply.ok) and reply.reason.startswith(HOME_STOPPED_PREFIX + " ")


@register
@dataclass(frozen=True, eq=False)
class GetState(Message):
    SCHEMA = "c3.get_state/v1"


@register
@dataclass(frozen=True, eq=False)
class Hello(Message):
    """Read-only: changes no state on the PC (it feeds the watchdog like every decoded C3 request)."""

    SCHEMA = "c3.hello/v1"


COMMIT_RE = re.compile(r"^([0-9a-f]{40}|unknown)$")
_POSITIVE_FLOATS = ("rotation_tol", "gripper_width_max_m", "watchdog_limit_s", "homing_timeout_s", "max_lead_s")


@register
@dataclass(frozen=True, eq=False)
class HelloReply(Message):
    """What the PC runs with, for the laptop's start-up check (STRUCTURE.md 3.8, 3.9)."""

    SCHEMA = "c3.hello.reply/v1"
    t_pc_ns: int = scalar("int64")                   # PC CLOCK_MONOTONIC, read immediately before encoding the reply
    commit: str = scalar("str")                      # git commit of the binary: 40 lowercase hex, or "unknown"
    mock: bool = scalar("bool")                      # franka_control --mock (fake arm and gripper)
    schemas: tuple[str, ...] = str_list()            # every schema the PC's codec knows, sorted by byte order
    F_T_TCP: np.ndarray = array("float64", (4, 4))   # frames.yaml as franka_control uses it, row-major, m
    rotation_tol: float = scalar("float64")          # limits.yaml
    gripper_width_max_m: float = scalar("float64")   # limits.yaml, m
    max_waypoints: int = scalar("uint64", min=1)     # limits.yaml
    watchdog_limit_s: float = scalar("float64")      # link.yaml timing.watchdog_limit_s
    homing_timeout_s: float = scalar("float64")      # franka_control/config.yaml homing.timeout_s
    max_lead_s: float = scalar("float64")            # franka_control/config.yaml safety.max_lead_s

    def check_values(self) -> None:
        if COMMIT_RE.fullmatch(self.commit) is None:  # fullmatch: no trailing newline, as in C++
            self.fail("commit", f"{self.commit!r} is not 40 lowercase hex characters or 'unknown'")
        for i, name in enumerate(self.schemas):
            try:
                SchemaName.parse(name)
            except ValueError as e:
                self.fail(f"schemas[{i}]", str(e))
        encoded = [name.encode("utf-8") for name in self.schemas]
        for i in range(1, len(encoded)):
            if encoded[i - 1] >= encoded[i]:
                self.fail("schemas", f"not strictly increasing by byte order at [{i}]: "
                                     f"{self.schemas[i - 1]!r} then {self.schemas[i]!r}")
        problem = rigid_problem(self.F_T_TCP, ContractLimits.load().rotation_tol)
        if problem is not None:
            self.fail("F_T_TCP", f"not a rigid transform: {problem}")
        for name in _POSITIVE_FLOATS:
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                self.fail(name, f"{value!r} must be finite and > 0")
