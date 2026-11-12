"""C2: franka_control -> message_comms: ArmState, broadcast at ~100 Hz and the get_state reply (section 4.4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.fields import array, nested, scalar
from contracts.limits import ContractLimits
from contracts.message import Message, Record
from contracts.registry import register

MODES = ("idle", "running", "holding", "stopped", "homing", "fault")


def rigid_problem(T: np.ndarray, rotation_tol: float) -> str | None:
    """None if T (4, 4) is a rigid transform: finite, bottom row exactly [0, 0, 0, 1], rotation orthonormal within
    rotation_tol with determinant > 0. Else what is wrong. Computed in float64. A column-major matrix sent without
    transposing puts the translation in the bottom row and fails here."""
    m = np.asarray(T, dtype=np.float64)
    if not np.all(np.isfinite(m)):
        return "not finite"
    if not np.array_equal(m[3], np.array([0.0, 0.0, 0.0, 1.0])):
        return f"bottom row {m[3].tolist()}, expected [0, 0, 0, 1] (column-major matrix sent without transposing?)"
    R = m[:3, :3]
    err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if err > rotation_tol:
        return f"rotation not orthonormal: max |R^T R - I| = {err:.3g} > {rotation_tol}"
    if float(np.linalg.det(R)) <= 0.0:
        return "rotation has determinant <= 0 (a reflection)"
    return None


@dataclass(frozen=True, eq=False)
class GateCounters(Record):
    """Waypoints since franka_control started."""

    accepted: int = scalar("uint64")       # accepted (shortened and pulled-back ones included)
    shortened: int = scalar("uint64")      # accepted, shortened to the speed limit
    pulled_back: int = scalar("uint64")    # accepted, pulled back onto the box or cone (within the margin)
    rejected: int = scalar("uint64")       # refused: franka_control goes to fault


@register
@dataclass(frozen=True, eq=False)
class ArmState(Message):
    SCHEMA = "c2.state/v1"
    t_ns: int = scalar("int64")                        # PC time the state was read from the arm
    seq: int = scalar("uint64")                        # +1 per message
    O_T_TCP: np.ndarray = array("float64", (4, 4))     # measured TCP pose, base frame, row-major
    O_T_TCP_target: np.ndarray = array("float64", (4, 4))
    q: np.ndarray = array("float64", (7,))             # rad
    dq: np.ndarray = array("float64", (7,))            # rad/s
    gripper_width_m: float = scalar("float64", min=0.0)
    mode: str = scalar("str", choices=MODES)
    error: str = scalar("str")
    gate: GateCounters = nested(GateCounters)
    loop_late: int = scalar("uint64")                  # late 1 kHz cycles since start

    def check_values(self) -> None:
        if self.gate.shortened > self.gate.accepted:
            self.fail("gate.shortened", f"{self.gate.shortened} > accepted {self.gate.accepted}")
        if self.gate.pulled_back > self.gate.accepted:
            self.fail("gate.pulled_back", f"{self.gate.pulled_back} > accepted {self.gate.accepted}")
        for name in ("O_T_TCP", "O_T_TCP_target", "q", "dq"):
            if not np.all(np.isfinite(getattr(self, name))):
                self.fail(name, "not finite")
        if not np.isfinite(self.gripper_width_m):
            self.fail("gripper_width_m", "not finite")
        tol = ContractLimits.load().rotation_tol
        for name in ("O_T_TCP", "O_T_TCP_target"):
            problem = rigid_problem(getattr(self, name), tol)
            if problem is not None:
                self.fail(name, f"not a rigid transform: {problem}")
