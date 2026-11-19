"""From held keys to a twist (PLAN 5.2, 5.3, 5.4, 13.2; STRUCTURE.md 7.2).

- Translation: +-v_max per axis in the robot base frame, one axis per key pair; opposite keys cancel.
- Twist (U/O): about the TCP's own z axis (the approach axis), whatever its tilt. `twist+` is positive (right hand).
- Tilt (I/K, J/L): about the base x and y axes through the TCP point, named by their effect on the TCP z axis, which
  points down at home: `tilt_fwd` moves it toward +x (a negative rotation about base y), `tilt_left` toward +y (a
  positive rotation about base x). With both tilt pairs held the 2-vector is normalised to at most w_max, so the
  cone bound of PLAN 5.3 holds for diagonal tilts too.
- Shift: everything times slow_factor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from teleop_collect.teleop.motion.keys import HeldKeys

if TYPE_CHECKING:                       # planner imports this module; only the type is needed here
    from teleop_collect.teleop.motion.planner import MotionSettings

#: The twist's six values in commands.csv order.
TWIST_COLUMNS = ("twist_vx_m_s", "twist_vy_m_s", "twist_vz_m_s", "twist_w_twist_rad_s", "twist_w_tilt_x_rad_s",
                 "twist_w_tilt_y_rad_s")


@dataclass(frozen=True)
class Twist:
    """A velocity of the TCP: translation in the base frame, twist about its own z, tilt about base x/y."""

    v_m_s: np.ndarray             # (3,) base frame
    w_twist_rad_s: float          # about the TCP's own z axis (the approach axis)
    w_tilt_rad_s: np.ndarray      # (2,) about the base x and y axes, through the TCP point

    def __post_init__(self) -> None:
        v = np.array(self.v_m_s, dtype=np.float64).reshape(3)
        tilt = np.array(self.w_tilt_rad_s, dtype=np.float64).reshape(2)
        w = float(self.w_twist_rad_s)
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(tilt)) and math.isfinite(w)):
            raise ValueError("twist not finite")
        v.flags.writeable = False
        tilt.flags.writeable = False
        object.__setattr__(self, "v_m_s", v)
        object.__setattr__(self, "w_tilt_rad_s", tilt)
        object.__setattr__(self, "w_twist_rad_s", w)

    @classmethod
    def zero(cls) -> Twist:
        return cls(np.zeros(3), 0.0, np.zeros(2))

    def is_zero(self) -> bool:
        return not (np.any(self.v_m_s != 0.0) or self.w_twist_rad_s != 0.0 or np.any(self.w_tilt_rad_s != 0.0))

    def values(self) -> list[float]:
        """The six values in TWIST_COLUMNS order."""
        return [*self.v_m_s.tolist(), self.w_twist_rad_s, *self.w_tilt_rad_s.tolist()]


def _axis(held: frozenset[str], plus: str, minus: str) -> float:
    return float((plus in held) - (minus in held))


def twist_from_keys(keys: HeldKeys, s: MotionSettings) -> Twist:
    """The twist the held keys ask for (PLAN 5.2)."""
    h = keys.held
    scale = s.slow_factor if keys.slow else 1.0
    v_max = s.v_max_m_s * scale
    w_max = math.radians(s.w_max_deg_s) * scale
    v = np.array([_axis(h, "x+", "x-"), _axis(h, "y+", "y-"), _axis(h, "z+", "z-")]) * v_max
    w_twist = _axis(h, "twist+", "twist-") * w_max
    # tilt_left: +rotation about base x (z axis toward +y); tilt_fwd: -rotation about base y (z axis toward +x)
    tilt = np.array([_axis(h, "tilt_left", "tilt_right"), -_axis(h, "tilt_fwd", "tilt_back")])
    norm = float(np.linalg.norm(tilt))
    if norm > 1.0:
        tilt = tilt / norm
    return Twist(v, w_twist, tilt * w_max)
