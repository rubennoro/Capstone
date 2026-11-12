"""C3: message_comms -> franka_control: waypoints and commands (STRUCTURE.md section 4.5).

Replies: Ack for heartbeat, start, stop, home; ArmState for get_state; SetWaypointsReply for set_waypoints, or
ErrorReply when set_waypoints is refused because of the state (not a fault).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.c2_state import rigid_problem
from contracts.fields import array, list_of, scalar
from contracts.limits import ContractLimits
from contracts.message import Message, Record
from contracts.registry import register


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
    SCHEMA = "c3.home/v1"


@register
@dataclass(frozen=True, eq=False)
class GetState(Message):
    SCHEMA = "c3.get_state/v1"
