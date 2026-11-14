"""FakeGate: the fake PC's waypoint gate (STRUCTURE.md 10.1). Only the time and width rules of the PC's gate:
a passed waypoint is skipped; one more than max_lead_s ahead, or a width outside [0, gripper_width_max_m], is
rejected (and the fake PC goes to fault). No box, cone or speed rules. With any rejection nothing is accepted and every
non-skipped waypoint is listed (TOP 3.7.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from deploy_contracts import SetWaypoints, WaypointIssue

NOT_CHECKED = "not checked: another waypoint of this request was rejected"


@dataclass(frozen=True)
class GateResult:
    accepted: tuple[int, ...]            # indices of accepted waypoints
    skipped: int
    rejected: tuple[WaypointIssue, ...]

    @property
    def fault(self) -> str | None:
        """The fault text when a waypoint was rejected."""
        for issue in self.rejected:
            if issue.reason != NOT_CHECKED:
                return f"gate rejected waypoint {issue.index}: {issue.reason}"
        return None


class FakeGate:
    def __init__(self, max_lead_s: float, gripper_width_max_m: float) -> None:
        self._max_lead_ns = int(round(max_lead_s * 1e9))
        self._width_max = float(gripper_width_max_m)

    def check(self, msg: SetWaypoints, now_pc_ns: int) -> GateResult:
        t = np.asarray(msg.t_ns, dtype=np.int64)
        widths = np.asarray(msg.gripper_width_m, dtype=np.float64)
        live = [i for i in range(len(t)) if t[i] > now_pc_ns]
        skipped = len(t) - len(live)
        problems: dict[int, str] = {}
        for i in live:
            lead_ns = int(t[i]) - now_pc_ns
            if lead_ns > self._max_lead_ns:
                problems[i] = (f"{lead_ns / 1e9:.3f} s ahead, more than max_lead_s "
                               f"{self._max_lead_ns / 1e9:g} s")
            elif not 0.0 <= widths[i] <= self._width_max:
                problems[i] = f"width {widths[i]!r} m outside [0, {self._width_max}]"
        if problems:
            rejected = tuple(WaypointIssue(index=i, reason=problems.get(i, NOT_CHECKED)) for i in live)
            return GateResult(accepted=(), skipped=skipped, rejected=rejected)
        return GateResult(accepted=tuple(live), skipped=skipped, rejected=())
