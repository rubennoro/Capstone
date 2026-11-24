"""compare_hello: the equality part of the start-up check, one definition for both machines (STRUCTURE.md 3.9).

The rest of the start-up check (commit, heartbeat budget, home timeout, waypoint lead, clock agreement) is the
laptop's (policy_deploy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deploy_contracts.c3_commands import HelloReply
from deploy_contracts.frames import TcpFrame
from deploy_contracts.limits import ContractLimits
from deploy_contracts.link import LinkSettings
from deploy_contracts.shared import SHARED_SCHEMAS

F_T_TCP_ABS_TOL = 1e-9    # per element: franka_control re-normalises F_T_TCP's rotation
SCALAR_REL_TOL = 1e-9     # both sides parse the same decimal text, so scalars are normally bit-equal


@dataclass(frozen=True)
class HelloMismatch:
    field: str        # "schemas", "mock", "F_T_TCP", "rotation_tol", "gripper_width_max_m", "max_waypoints",
                      # "watchdog_limit_s"
    pc: str           # the PC's value, printable
    expected: str     # this side's value and its source, e.g. "1.0 (link.yaml timing.watchdog_limit_s)"

    def __str__(self) -> str:
        return f"{self.field}: PC {self.pc}, expected {self.expected}"


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= SCALAR_REL_TOL * max(abs(a), abs(b))


def _bool(v: bool) -> str:
    return "true" if v else "false"


def _entries(names: list[str]) -> str:
    return ", ".join(names) if names else "(none)"


def compare_hello(reply: HelloReply, *, mock: bool, limits: ContractLimits | None = None,
                  frame: TcpFrame | None = None, link: LinkSettings | None = None) -> list[HelloMismatch]:
    """Every difference between the PC's hello reply and this side's contract files and mock flag.

    limits, frame, link: None loads the file (cached). Returns the mismatches in the order schemas, mock, F_T_TCP,
    rotation_tol, gripper_width_max_m, max_waypoints, watchdog_limit_s; an empty list means all equal.
    """
    limits = ContractLimits.load() if limits is None else limits
    frame = TcpFrame.load() if frame is None else frame
    link = LinkSettings.load() if link is None else link
    out: list[HelloMismatch] = []

    if tuple(reply.schemas) != SHARED_SCHEMAS:
        pc_only = [s for s in reply.schemas if s not in SHARED_SCHEMAS]
        ours_only = [s for s in SHARED_SCHEMAS if s not in reply.schemas]
        out.append(HelloMismatch("schemas", _entries(pc_only),
                                 f"{_entries(ours_only)} (deploy_contracts SHARED_SCHEMAS)"))

    if bool(reply.mock) != bool(mock):
        out.append(HelloMismatch("mock", _bool(reply.mock), f"{_bool(mock)} (this side's mock flag)"))

    ours = np.asarray(frame.F_T_TCP, dtype=np.float64)
    theirs = np.asarray(reply.F_T_TCP, dtype=np.float64)
    if not np.all(np.abs(theirs - ours) <= F_T_TCP_ABS_TOL):
        worst = float(np.max(np.abs(theirs - ours)))
        out.append(HelloMismatch("F_T_TCP", f"{theirs.tolist()} (largest difference {worst:.3g})",
                                 f"{ours.tolist()} (frames.yaml F_T_TCP)"))

    for name in ("rotation_tol", "gripper_width_max_m"):
        pc_value, our_value = float(getattr(reply, name)), float(getattr(limits, name))
        if not _close(pc_value, our_value):
            out.append(HelloMismatch(name, repr(pc_value), f"{our_value!r} (limits.yaml {name})"))

    if int(reply.max_waypoints) != int(limits.max_waypoints):
        out.append(HelloMismatch("max_waypoints", str(int(reply.max_waypoints)),
                                 f"{int(limits.max_waypoints)} (limits.yaml max_waypoints)"))

    if not _close(float(reply.watchdog_limit_s), float(link.watchdog_limit_s)):
        out.append(HelloMismatch("watchdog_limit_s", repr(float(reply.watchdog_limit_s)),
                                 f"{float(link.watchdog_limit_s)!r} (link.yaml timing.watchdog_limit_s)"))
    return out
