"""The home reference: Start episode only at home (PLAN 8, 13.11; STRUCTURE.md 7.5).

The laptop never reads the PC's `homing.q_home_rad`; it learns the home pose from the arm. The reference is the joints
and TCP target of C2 right after a home that ended in `idle`: a C3 home teleop sent (E3 calls `set_from`), or the PC's
own `h` key, seen on C2 as `homing` -> `idle` (`observe`, fed every C2 the main loop looks at).

`observe` sets the reference only while none is known, and only when the arm moved during that homing (a joint by
more than tolerance_rad or the TCP target by more than tolerance_m): the PC's `g` (homing only the hand) shows the
same `homing` -> `idle` on C2 wherever the arm is, and moves neither. The PC's home pose is fixed, so the first moving
home is enough; `set_from` (after a C3 home teleop sent) always replaces it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class HomeSettings:
    """The `home:` keys of config/teleop.yaml (STRUCTURE.md 3.3)."""

    on_start: bool                # home the whole arm once at start-up
    tolerance_rad: float          # every joint within this of the reference
    tolerance_m: float            # and the TCP target position within this

    @classmethod
    def from_config(cls, home: Mapping[str, Any]) -> HomeSettings:
        names = ("on_start", "tolerance_rad", "tolerance_m")
        missing = [n for n in names if n not in home]
        unknown = sorted(str(k) for k in home if k not in names)
        if missing:
            raise ValueError(f"home: missing {', '.join(missing)}")
        if unknown:
            raise ValueError(f"home: unknown {', '.join(unknown)}")
        return cls(**{n: home[n] for n in names})

    def __post_init__(self) -> None:
        if not isinstance(self.on_start, bool):
            raise ValueError(f"home.on_start: {self.on_start!r} is not true or false")
        for name in ("tolerance_rad", "tolerance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value)
                                                                                     and value > 0.0):
                raise ValueError(f"home.{name}: {value!r} must be a finite number > 0")
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


class HomeReference:
    """The joints and TCP position the arm had right after the newest home that ended in idle."""

    def __init__(self, s: HomeSettings) -> None:
        self._s = s
        self._q_ref: np.ndarray | None = None
        self._p_ref: np.ndarray | None = None
        self._last_mode: str | None = None
        self._homing_start: tuple[np.ndarray, np.ndarray] | None = None   # q, p at the first `homing` C2

    def set_from(self, q: np.ndarray, T_target: np.ndarray) -> None:
        """After a C3 home teleop sent that ended in idle: this state is home (replaces any reference)."""
        self._q_ref, self._p_ref = _q(q), _p(T_target)

    def observe(self, mode: str, q: np.ndarray, T_target: np.ndarray) -> None:
        """Every C2 the main loop sees. A homing -> idle transition (the PC's h) sets the reference from that idle
        state, only while none is known, and only if the arm moved during that homing: a joint by more than
        tolerance_rad or the TCP target by more than tolerance_m, between the first `homing` C2 and the `idle` one.
        The PC's g (the hand alone) moves neither, and an h while already at home cannot be told from it, so neither
        counts (a C3 home, set_from, is then needed)."""
        if mode == "homing":
            if self._last_mode != "homing" or self._homing_start is None:
                self._homing_start = (_q(q), _p(T_target))
        elif self._last_mode == "homing":
            start, self._homing_start = self._homing_start, None
            if mode == "idle" and self._q_ref is None and start is not None and self._moved(start, q, T_target):
                self.set_from(q, T_target)
        self._last_mode = mode

    def _moved(self, start: tuple[np.ndarray, np.ndarray], q: np.ndarray, T_target: np.ndarray) -> bool:
        q0, p0 = start
        return (float(np.max(np.abs(_q(q) - q0))) > self._s.tolerance_rad
                or float(np.linalg.norm(_p(T_target) - p0)) > self._s.tolerance_m)

    @property
    def known(self) -> bool:
        return self._q_ref is not None

    def at_home(self, q: np.ndarray, T_target: np.ndarray) -> bool:
        """Known, all |q - q_ref| <= tolerance_rad and |p - p_ref| <= tolerance_m. The mode rule is the caller's."""
        if self._q_ref is None or self._p_ref is None:
            return False
        if float(np.max(np.abs(_q(q) - self._q_ref))) > self._s.tolerance_rad:
            return False
        return float(np.linalg.norm(_p(T_target) - self._p_ref)) <= self._s.tolerance_m

    @property
    def q_ref(self) -> np.ndarray | None:
        return None if self._q_ref is None else self._q_ref.copy()
