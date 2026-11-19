"""What, if anything, the teleop step sends this cycle (PLAN 5.1 step 3, 5.2, 5.3; STRUCTURE.md 7.4).

`MotionController.step` is called once per 20 ms cycle by E3's TeleopLoop (when no hello is due) with the mapped PC
time, the newest C2 as a `BaseState` and the keys in effect. It returns a `MotionCommand` (one set_waypoints) or None
(the loop sends a heartbeat). Rules, in order:

1. not begun, or stopped -> None;
2. now_pc or base missing, the base older than state_max_age_ms, or the PC not following (mode other than running /
   holding) -> None: nothing is sent, the arm ends its last plan within the lead and holds; whatever is pending (a
   release, a finger command) waits for a fresh state;
3. a non-zero twist -> `move` (a pending finger width rides on it);
4. the twist became zero after a move -> one `release`;
5. a pending finger command -> `fingers` (a hold at the base pose with the new width);
6. otherwise None.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from deploy_contracts import ArmState

from teleop_collect.contracts import GRIPPER_COMMANDS
from teleop_collect.teleop.motion.keys import HeldKeys
from teleop_collect.teleop.motion.planner import MotionSettings, Waypoint, WaypointPlanner
from teleop_collect.teleop.motion.twist import Twist, twist_from_keys

#: C2 modes in which the PC follows waypoints; set_waypoints is refused in the others.
FOLLOWING_MODES = ("running", "holding")
KINDS = ("move", "release", "fingers")


@dataclass(frozen=True)
class BaseState:
    """The base of a command, from the newest C2: the PC's own target and its PC time."""

    T_target: np.ndarray          # (4, 4) O_T_TCP_target
    t_pc_ns: int                  # C2 t_ns (PC clock)
    width_m: float                # C2 gripper_width_m
    mode: str

    @classmethod
    def from_arm_state(cls, state: ArmState) -> BaseState:
        return cls(state.O_T_TCP_target, state.t_ns, state.gripper_width_m, state.mode)


@dataclass(frozen=True)
class MotionCommand:
    """One set_waypoints to send: E3 builds the C3 SetWaypoints (with its chunk_id) from the waypoints."""

    kind: str                     # "move" | "release" | "fingers"
    waypoints: tuple[Waypoint, ...]
    twist: Twist                  # the twist the poses were advanced by (zero for fingers)
    slow: bool


class MotionController:
    """The episode's motion state: active or not, the commanded finger width, the last twist sent."""

    def __init__(self, s: MotionSettings, width_open_m: float) -> None:
        if not (math.isfinite(width_open_m) and width_open_m > 0.0):
            raise ValueError(f"width_open_m {width_open_m!r} must be finite and > 0")
        self._s = s
        self._planner = WaypointPlanner(s)
        self._width_open_m = float(width_open_m)
        self._active = False
        self._width_m = 0.0
        self._fingers_pending = False
        self._last_twist: Twist | None = None          # the twist of the last move sent, until its release
        self._plan_end_pc_ns: int | None = None        # the last move's last waypoint time
        self._moving = False
        self._width_clipped = False

    def begin(self, commanded_width_m: float) -> bool:
        """At Start: commands go out from now on, starting from the arm's measured width, clipped into
        [0, width_open_m] (an open Franka Hand reads a little above the PC gate's gripper_width_max_m, which the gate
        would reject: a fault). Never raises for the width. Returns True when it clipped (also `width_clipped`)."""
        measured = float(commanded_width_m)
        if not math.isfinite(measured):
            width = self._width_open_m           # cannot come from a valid C2; open is the hand's state at Start
        else:
            width = min(max(measured, 0.0), self._width_open_m)
        self._width_clipped = width != measured
        self._active = True
        self._width_m = width
        self._fingers_pending = False
        self._last_twist = None
        self._plan_end_pc_ns = None
        self._moving = False
        return self._width_clipped

    def stop(self) -> None:
        """At Stop / Abort: no more commands until the next begin."""
        self._active = False
        self._fingers_pending = False
        self._last_twist = None
        self._plan_end_pc_ns = None
        self._moving = False

    def gripper(self, command: str) -> None:
        """`close` -> width 0 (a grasp with force on the PC), `open` -> width_open_m. Ignored when not begun."""
        if command not in GRIPPER_COMMANDS:
            raise ValueError(f"unknown gripper command {command!r}; known: {', '.join(GRIPPER_COMMANDS)}")
        if not self._active:
            return
        self._width_m = 0.0 if command == "close" else self._width_open_m
        self._fingers_pending = True

    def step(self, now_pc_ns: int | None, base: BaseState | None, keys: HeldKeys) -> MotionCommand | None:
        if not self._active:
            return None
        twist = twist_from_keys(keys, self._s)
        self._moving = not twist.is_zero()
        if now_pc_ns is None or base is None:
            return None
        if int(now_pc_ns) - int(base.t_pc_ns) > self._s.state_max_age_ns:
            return None
        if base.mode not in FOLLOWING_MODES:
            return None
        if self._moving:
            wps = self._planner.move(base.T_target, base.t_pc_ns, twist, now_pc_ns, self._width_m)
            self._fingers_pending = False
            self._last_twist = twist
            self._plan_end_pc_ns = wps[-1].t_pc_ns
            return MotionCommand("move", tuple(wps), twist, keys.slow)
        if self._last_twist is not None:
            last = self._last_twist
            wps = self._planner.release(base.T_target, base.t_pc_ns, last, now_pc_ns, self._width_m,
                                        until_pc_ns=self._plan_end_pc_ns)
            self._fingers_pending = False
            self._last_twist = None
            self._plan_end_pc_ns = None
            return MotionCommand("release", tuple(wps), last, keys.slow)
        if self._fingers_pending:
            wps = self._planner.hold(base.T_target, now_pc_ns, self._width_m)
            self._fingers_pending = False
            return MotionCommand("fingers", tuple(wps), Twist.zero(), keys.slow)
        return None

    @property
    def commanded_width_m(self) -> float:
        """The width every waypoint carries (the last N / M, or the width at begin)."""
        return self._width_m

    @property
    def moving(self) -> bool:
        """A motion key was in effect at the last step (whether or not a set could be sent)."""
        return self._active and self._moving

    @property
    def width_clipped(self) -> bool:
        """The last begin() clipped the measured width into [0, width_open_m] (for E3's log)."""
        return self._width_clipped

    @property
    def active(self) -> bool:
        return self._active
