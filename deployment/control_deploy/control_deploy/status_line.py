"""StatusLine: the one line `control_deploy run` keeps at the bottom of the terminal, from C2 and the C3 peers
(control_deploy/STRUCTURE.md 6.4 step 6). Read-only: a C2 subscriber, like any other (decision 11)."""
from __future__ import annotations

from collections import deque

import numpy as np
from deploy_contracts import ArmState

WIDTH = 120
LATE_WINDOW_S = 10.0
RATE_WINDOW_S = 1.0
NO_C2_S = 1.0


class StatusLine:
    """Feed it every C2 message with its arrival time (seconds, any monotonic clock); render() gives the line."""

    def __init__(self, width: int = WIDTH) -> None:
        self.width = width
        self.state: ArmState | None = None
        self.last_arrival: float | None = None
        self._arrivals: deque[float] = deque()
        self._late: deque[tuple[float, int]] = deque()

    def feed(self, state: ArmState, arrival_s: float) -> None:
        self.state = state
        self.last_arrival = arrival_s
        self._arrivals.append(arrival_s)
        while self._arrivals and self._arrivals[0] < arrival_s - RATE_WINDOW_S:
            self._arrivals.popleft()
        self._late.append((arrival_s, int(state.loop_late)))
        while self._late and self._late[0][0] < arrival_s - LATE_WINDOW_S:
            self._late.popleft()

    def render(self, now_s: float, peers: list[str] | None = None) -> str:
        s = self.state
        if s is None or self.last_arrival is None or now_s - self.last_arrival > NO_C2_S:
            return self._fit("no C2: see the log")
        p = np.asarray(s.O_T_TCP)[:3, 3]
        late_now = int(s.loop_late)
        late_recent = late_now - self._late[0][1] if self._late else 0
        rate = len(self._arrivals) / RATE_WINDOW_S
        head = f"fault: {' '.join(s.error.split())}" if s.mode == "fault" else s.mode
        g = s.gate
        parts = [
            head,
            f"tcp {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} m",
            f"grip {s.gripper_width_m:.3f} m",
            f"gate a{g.accepted} s{g.shortened} p{g.pulled_back} r{g.rejected}",
            f"late {late_now} (+{late_recent} in {LATE_WINDOW_S:g} s)",
            f"C2 {rate:.0f} Hz",
            "C3 peer " + (", ".join(peers) if peers else "none"),
            "h p g s q",
        ]
        # The layout of STRUCTURE.md 6.4 is 134 columns for a typical state: compact it only as far as needed.
        sep = " | "
        line = sep.join(parts)
        if len(line) > self.width:
            parts[4] = f"late {late_now} (+{late_recent}/{LATE_WINDOW_S:g}s)"
            line = sep.join(parts)
        if len(line) > self.width:
            sep = "|"
            line = sep.join(parts)
        if len(line) > self.width:   # the key hint grew with p: drop the tcp unit and the C2 rate's space
            parts[1] = f"tcp {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"
            parts[5] = f"C2 {rate:.0f}Hz"
            line = sep.join(parts)
        if len(line) > self.width and s.mode == "fault":   # the error text matters most: keep it, the peers, the keys
            tail = " | " + " | ".join(parts[-2:])
            room = max(10, self.width - len(tail))
            head = parts[0] if len(parts[0]) <= room else parts[0][: room - 3] + "..."
            line = head + tail
        return self._fit(line)

    def _fit(self, line: str) -> str:
        return line if len(line) <= self.width else line[: self.width - 3] + "..."
