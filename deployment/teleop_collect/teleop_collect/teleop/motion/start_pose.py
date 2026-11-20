"""The start pose: Start episode only where the episodes start (PLAN 8; STRUCTURE.md 7.5).

The PC owns the start pose (its `p` key moves the arm there, and its C3 start refuses away from it once
`homing.q_start_rad` is set); the laptop never reads the PC's station. It learns the pose from the arm instead:

- `learn`: the end of teleop's start-up. The operator pressed h, then p on the PC first, so this pose is the
  candidate (not yet confirmed).
- `observe` (every C2 the main loop sees): while not confirmed, a `homing` -> `idle` transition in which the arm moved
  (a joint by more than tolerance_rad or the TCP target by more than tolerance_m, between the first `homing` C2 and the
  `idle` one) replaces the candidate with that idle pose, also when none is known: the PC's h, then p, end at the
  start pose. The PC's g (the hand alone) shows the same `homing` -> `idle` on C2 and moves neither: it never counts.
- `confirm`: the PC accepted a start at the candidate: the pose is the session's reference. After that, a moving
  `homing` -> `idle` re-anchors it to that idle pose only when every joint ends within reanchor_tol_rad (the PC's
  homing.at_pose_tol_rad, 0.05) of the confirmed q: the PC's p ends wherever its own looser check accepts, so the
  laptop follows it there. A move ending farther away (h, or a p elsewhere) changes nothing.
- `refused_here`: the PC refused a start with `refused: not at the start pose`: the candidate is dropped (while not
  confirmed) and this pose is kept, so the page can say the PC refused the start here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class StartPoseSettings:
    """The `start_pose:` keys of config/teleop.yaml (STRUCTURE.md 3.3)."""

    tolerance_rad: float          # every joint within this of the reference
    tolerance_m: float            # and the TCP target position within this
    reanchor_tol_rad: float       # once confirmed: a PC move ending with every joint within this re-anchors

    @classmethod
    def from_config(cls, m: Mapping[str, Any]) -> StartPoseSettings:
        names = ("tolerance_rad", "tolerance_m", "reanchor_tol_rad")
        missing = [n for n in names if n not in m]
        unknown = sorted(str(k) for k in m if k not in names)
        if missing:
            raise ValueError(f"start_pose: missing {', '.join(missing)}")
        if unknown:
            raise ValueError(f"start_pose: unknown {', '.join(unknown)}")
        return cls(**{n: m[n] for n in names})

    def __post_init__(self) -> None:
        for name in ("tolerance_rad", "tolerance_m", "reanchor_tol_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value)
                                                                                     and value > 0.0):
                raise ValueError(f"start_pose.{name}: {value!r} must be a finite number > 0")
            object.__setattr__(self, name, float(value))


def _q(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64).reshape(-1)
    if out.shape != (7,):
        raise ValueError(f"q must have 7 joints, got shape {out.shape}")
    return out


def _p(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_target must be (4, 4), got shape {T.shape}")
    return T[:3, 3].copy()


class StartPoseReference:
    """This session's start pose: a candidate (learned at start-up, or after the PC's h/p) until the PC accepts a start
    there, then confirmed (re-anchored only by a PC move ending within reanchor_tol_rad of it)."""

    def __init__(self, s: StartPoseSettings) -> None:
        self._s = s
        self._ref: tuple[np.ndarray, np.ndarray] | None = None        # q, p of the candidate or confirmed pose
        self._confirmed = False
        self._refused: tuple[np.ndarray, np.ndarray] | None = None    # q, p where the PC refused a start
        self._last_mode: str | None = None
        self._homing_start: tuple[np.ndarray, np.ndarray] | None = None   # q, p at the first `homing` C2

    def learn(self, q: np.ndarray, T_target: np.ndarray) -> None:
        """The end of start-up: this pose is the candidate (not confirmed). Ignored once confirmed."""
        if not self._confirmed:
            self._ref = (_q(q), _p(T_target))

    def observe(self, mode: str, q: np.ndarray, T_target: np.ndarray) -> None:
        """Every C2 the main loop sees. While not confirmed, a homing -> idle transition in which the arm moved
        replaces the candidate with that idle pose (also when none is known). Once confirmed, such a move re-anchors
        the reference to that idle pose only when every joint is within reanchor_tol_rad of the confirmed q. The PC's
        g (the hand alone) moves neither the joints nor the TCP target, so it never counts."""
        if mode == "homing":
            if self._last_mode != "homing" or self._homing_start is None:
                self._homing_start = (_q(q), _p(T_target))
        elif self._last_mode == "homing":
            start, self._homing_start = self._homing_start, None
            if mode == "idle" and start is not None and self._moved(start, q, T_target):
                if not self._confirmed or self._within_reanchor(q):
                    self._ref = (_q(q), _p(T_target))
        self._last_mode = mode

    def confirm(self) -> None:
        """The PC accepted a start at the candidate: the session's reference (re-anchored only by observe). Nothing to
        confirm when none is known."""
        if self._ref is not None:
            self._confirmed = True

    def refused_here(self, q: np.ndarray, T_target: np.ndarray) -> None:
        """The PC refused a start with `refused: not at the start pose`: the candidate is dropped (only while not
        confirmed), and this pose is kept as the "not the start pose" pose (the page hint)."""
        self._refused = (_q(q), _p(T_target))
        if not self._confirmed:
            self._ref = None

    @property
    def known(self) -> bool:
        """A candidate or a confirmed pose exists."""
        return self._ref is not None

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def at_start(self, q: np.ndarray, T_target: np.ndarray) -> bool:
        """Known, every |q - q_ref| <= tolerance_rad and |p - p_ref| <= tolerance_m. The mode rule is the caller's."""
        return self._ref is not None and self._near(self._ref, q, T_target)

    def at_refused_pose(self, q: np.ndarray, T_target: np.ndarray) -> bool:
        """Within the tolerances of the pose where the PC refused a start."""
        return self._refused is not None and self._near(self._refused, q, T_target)

    @property
    def q_ref(self) -> np.ndarray | None:
        return None if self._ref is None else self._ref[0].copy()

    @property
    def p_ref(self) -> np.ndarray | None:
        return None if self._ref is None else self._ref[1].copy()

    def _near(self, ref: tuple[np.ndarray, np.ndarray], q: np.ndarray, T_target: np.ndarray) -> bool:
        q_ref, p_ref = ref
        if float(np.max(np.abs(_q(q) - q_ref))) > self._s.tolerance_rad:
            return False
        return float(np.linalg.norm(_p(T_target) - p_ref)) <= self._s.tolerance_m

    def _within_reanchor(self, q: np.ndarray) -> bool:
        assert self._ref is not None
        return float(np.max(np.abs(_q(q) - self._ref[0]))) <= self._s.reanchor_tol_rad

    def _moved(self, start: tuple[np.ndarray, np.ndarray], q: np.ndarray, T_target: np.ndarray) -> bool:
        q0, p0 = start
        return (float(np.max(np.abs(_q(q) - q0))) > self._s.tolerance_rad
                or float(np.linalg.norm(_p(T_target) - p0)) > self._s.tolerance_m)
