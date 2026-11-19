"""Waypoints from a twist, and the safety bound of PLAN 5.3 (PLAN 5.2-5.3; STRUCTURE.md 7.3).

Every command starts from the PC's own latest target (C2 `O_T_TCP_target` at its PC time `t_base`), never from a pose
integrated on the laptop. A move holds K waypoints at `now_pc + k * Delta`, each the base pose advanced by the twist
over `t_k - t_base`. So a waypoint sticks out past the base pose (which the PC's gate has already pulled back onto
its box and cone) by at most `speed * (K * Delta + state age)`; the controller sends nothing while the state is older
than `state_max_age_ms`, which makes that `worst_step`. `MotionSettings` refuses settings whose worst step exceeds
`margin_use` of the PC's pull-back margins, so the gate pulls back and never rejects (a fault).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from deploy_contracts.limits import ContractLimits

from teleop_collect.teleop.motion.twist import Twist

_FLOAT_FIELDS = ("v_max_m_s", "w_max_deg_s", "slow_factor", "spacing_ms", "key_stale_ms", "state_max_age_ms",
                 "assumed_pc_box_margin_m", "assumed_pc_cone_margin_deg", "margin_use")


@dataclass(frozen=True)
class MotionSettings:
    """The `motion:` keys of config/teleop.yaml (STRUCTURE.md 3.3). Refused (ValueError) when a value is bad or the
    worst step exceeds margin_use x the PC's margins."""

    v_max_m_s: float                    # per axis, each held translation key
    w_max_deg_s: float                  # the twist, and the tilt (both tilt axes together, normalised)
    slow_factor: float                  # Shift
    waypoints: int                      # K
    spacing_ms: float                   # Delta; the lead is K x Delta
    key_stale_ms: float                 # no key state for this long = all keys released
    state_max_age_ms: float             # no motion while the newest C2 is older (mapped)
    assumed_pc_box_margin_m: float      # the PC's safety.box_margin_m
    assumed_pc_cone_margin_deg: float   # the PC's safety.cone_margin_deg
    margin_use: float                   # share of a margin the worst step may use

    @classmethod
    def from_config(cls, motion: Mapping[str, Any]) -> MotionSettings:
        """From the merged `motion:` section: exactly the fields above, no more, no fewer."""
        names = [f.name for f in dataclasses.fields(cls)]
        missing = [n for n in names if n not in motion]
        unknown = sorted(str(k) for k in motion if k not in names)
        if missing:
            raise ValueError(f"motion: missing {', '.join(missing)}")
        if unknown:
            raise ValueError(f"motion: unknown {', '.join(unknown)}")
        return cls(**{n: motion[n] for n in names})

    def __post_init__(self) -> None:
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"motion.{name}: {value!r} is not a finite number")
            object.__setattr__(self, name, float(value))
        k = self.waypoints
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError(f"motion.waypoints: {k!r} is not an integer")
        max_k = ContractLimits.load().max_waypoints
        if not 1 <= k <= max_k:
            raise ValueError(f"motion.waypoints: {k} must be 1 to {max_k} (limits.yaml max_waypoints)")
        for name in ("v_max_m_s", "w_max_deg_s", "spacing_ms", "key_stale_ms", "assumed_pc_box_margin_m",
                     "assumed_pc_cone_margin_deg"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"motion.{name}: {getattr(self, name)!r} must be > 0")
        if not self.state_max_age_ms >= 0.0:
            raise ValueError(f"motion.state_max_age_ms: {self.state_max_age_ms!r} must be >= 0")
        for name in ("slow_factor", "margin_use"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"motion.{name}: {getattr(self, name)!r} must be in (0, 1]")
        step_m, step_deg = worst_step(self)
        if step_m > self.margin_use * self.assumed_pc_box_margin_m:
            raise ValueError(
                f"motion: worst step {step_m * 1e3:.2f} mm = v_max_m_s x (waypoints x spacing_ms + state_max_age_ms) "
                f"exceeds margin_use x assumed_pc_box_margin_m = {self.margin_use * self.assumed_pc_box_margin_m * 1e3:.2f}"
                " mm (PLAN 5.3): lower the speed or the lead")
        if step_deg > self.margin_use * self.assumed_pc_cone_margin_deg:
            raise ValueError(
                f"motion: worst tilt step {step_deg:.3f} deg = w_max_deg_s x (waypoints x spacing_ms + "
                f"state_max_age_ms) exceeds margin_use x assumed_pc_cone_margin_deg = "
                f"{self.margin_use * self.assumed_pc_cone_margin_deg:.3f} deg (PLAN 5.3): lower the speed or the lead")

    # --- derived, in the units the code uses -------------------------------------------------------------------
    @property
    def spacing_ns(self) -> int:
        return int(round(self.spacing_ms * 1e6))

    @property
    def lead_ns(self) -> int:
        """K x Delta: how far ahead the last waypoint lies."""
        return self.waypoints * self.spacing_ns

    @property
    def state_max_age_ns(self) -> int:
        return int(round(self.state_max_age_ms * 1e6))

    @property
    def w_max_rad_s(self) -> float:
        return math.radians(self.w_max_deg_s)

    def to_dict(self) -> dict[str, float | int]:
        """Plain values, for meta.yaml and task.yaml."""
        return dataclasses.asdict(self)


def worst_step(s: MotionSettings) -> tuple[float, float]:
    """(m, deg) the farthest a waypoint can lie past its base pose: per translation axis, and in tilt of the TCP z
    axis. Unscaled: slow mode only makes it smaller."""
    horizon_s = (s.waypoints * s.spacing_ms + s.state_max_age_ms) / 1e3
    return s.v_max_m_s * horizon_s, s.w_max_deg_s * horizon_s


@dataclass(frozen=True)
class Waypoint:
    """One waypoint of a set_waypoints: PC time, absolute TCP pose (base frame, row-major) and finger width."""

    t_pc_ns: int
    O_T_TCP: np.ndarray           # (4, 4) float64
    width_m: float

    def __post_init__(self) -> None:
        T = np.array(self.O_T_TCP, dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError(f"waypoint pose must be a finite (4, 4), got shape {T.shape}")
        width = float(self.width_m)
        if not (math.isfinite(width) and width >= 0.0):
            raise ValueError(f"waypoint width {width!r} must be finite and >= 0")
        T.flags.writeable = False
        object.__setattr__(self, "O_T_TCP", T)
        object.__setattr__(self, "width_m", width)
        object.__setattr__(self, "t_pc_ns", int(self.t_pc_ns))


def _orthonormal(R: np.ndarray) -> np.ndarray:
    """The nearest rotation (polar decomposition by SVD)."""
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    if np.linalg.det(out) < 0.0:            # cannot happen for a near-rotation; kept as a guard
        U[:, -1] = -U[:, -1]
        out = U @ Vt
    return out


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def advance(T_base: np.ndarray, twist: Twist, tau_s: float) -> np.ndarray:
    """T_base moved by the twist over tau_s: p = p_base + v tau; R = Rot_base([w_tilt_x, w_tilt_y, 0] tau) @ R_base @
    Rot_z(w_twist tau). The tilt turns about base axes through the TCP point and the twist about the TCP's own z, so
    a pure rotation keeps the TCP position. The result is rigid (re-orthonormalised)."""
    T_base = np.asarray(T_base, dtype=np.float64)
    tau = float(tau_s)
    R_tilt = Rotation.from_rotvec([twist.w_tilt_rad_s[0] * tau, twist.w_tilt_rad_s[1] * tau, 0.0]).as_matrix()
    R = _orthonormal(R_tilt @ T_base[:3, :3] @ _rot_z(twist.w_twist_rad_s * tau))
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = T_base[:3, 3] + twist.v_m_s * tau
    return out


class WaypointPlanner:
    """Builds the waypoint lists of a move, a release and a finger-only hold (PLAN 5.2)."""

    def __init__(self, s: MotionSettings) -> None:
        self._s = s

    def move(self, T_base: np.ndarray, t_base_pc_ns: int, twist: Twist, now_pc_ns: int,
             width_m: float) -> list[Waypoint]:
        """K waypoints at now_pc + k Delta (k = 1..K), each advance(T_base, twist, (t_k - t_base) / 1e9)."""
        out = []
        for k in range(1, self._s.waypoints + 1):
            t_k = int(now_pc_ns) + k * self._s.spacing_ns
            out.append(Waypoint(t_k, advance(T_base, twist, (t_k - int(t_base_pc_ns)) / 1e9), width_m))
        return out

    def release(self, T_base: np.ndarray, t_base_pc_ns: int, last_twist: Twist, now_pc_ns: int, width_m: float,
                until_pc_ns: int | None = None) -> list[Waypoint]:
        """One waypoint at now_pc + Delta: the base pose advanced by the old twist over the state's age only (the best
        guess of where the target is now). With until_pc_ns (the last sent plan's end), not past it: after its plan
        the PC's target stands still. The age is never negative."""
        until = int(now_pc_ns) if until_pc_ns is None else min(int(now_pc_ns), int(until_pc_ns))
        tau_s = max(0, until - int(t_base_pc_ns)) / 1e9
        return [Waypoint(int(now_pc_ns) + self._s.spacing_ns, advance(T_base, last_twist, tau_s), width_m)]

    def hold(self, T_base: np.ndarray, now_pc_ns: int, width_m: float) -> list[Waypoint]:
        """One waypoint at now_pc + Delta at T_base (a finger command with no motion), re-orthonormalised."""
        return [Waypoint(int(now_pc_ns) + self._s.spacing_ns, advance(T_base, Twist.zero(), 0.0), width_m)]
