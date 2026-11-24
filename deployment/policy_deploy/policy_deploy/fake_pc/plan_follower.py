"""PlanFollower: the fake PC's target (STRUCTURE.md 10.1). The target is the plan interpolated at "now" (linear
position, slerp rotation) from the target at the time the plan was accepted to each waypoint in turn, holding after
the last; the width moves toward the commanded width at width_speed_m_s."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray:
    """Linear position, slerp rotation, a fraction a in [0, 1] from T0 to T1."""
    a = min(max(float(a), 0.0), 1.0)
    out = np.eye(4)
    out[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
    rots = Rotation.from_matrix(np.stack((T0[:3, :3], T1[:3, :3])))
    R = Slerp([0.0, 1.0], rots)([a]).as_matrix()[0]
    u, _, vt = np.linalg.svd(R)                      # exactly orthonormal (the rigid check is 1e-6)
    out[:3, :3] = u @ vt
    return out


class PlanFollower:
    def __init__(self, target: np.ndarray, width_m: float, width_speed_m_s: float) -> None:
        self.target = np.array(target, dtype=np.float64)
        self.width_m = float(width_m)              # the measured opening
        self.commanded_width_m = float(width_m)
        self._speed = float(width_speed_m_s)
        self._plan: list[tuple[int, np.ndarray, float]] = []   # (t_pc_ns, T, width), the first is the start point

    def replace(self, now_pc_ns: int, waypoints: list[tuple[int, np.ndarray, float]]) -> None:
        """Accepted waypoints replace every waypoint not yet reached, starting from the current target now."""
        if not waypoints:
            return
        self._plan = [(int(now_pc_ns), self.target.copy(), self.commanded_width_m)] + [
            (int(t), np.array(T, dtype=np.float64), float(w)) for t, T, w in waypoints]

    def hold(self) -> None:
        """Forget the plan: hold the current target."""
        self._plan = []

    def set_target(self, T: np.ndarray) -> None:
        self.target = np.array(T, dtype=np.float64)
        self._plan = []

    def following(self) -> bool:
        return bool(self._plan)

    def step(self, now_pc_ns: int, dt_s: float) -> bool:
        """Move the target to `now` and the width by dt_s; True when the plan just ended (holding after the last)."""
        ended = False
        if self._plan:
            times = [p[0] for p in self._plan]
            if now_pc_ns >= times[-1]:
                self.target = self._plan[-1][1].copy()
                self.commanded_width_m = self._plan[-1][2]
                self._plan = []
                ended = True
            else:
                i = max(0, int(np.searchsorted(times, now_pc_ns, side="right")) - 1)
                t0, T0, w0 = self._plan[i]
                t1, T1, w1 = self._plan[i + 1]
                a = (now_pc_ns - t0) / max(1, t1 - t0)
                self.target = interpolate(T0, T1, a)
                self.commanded_width_m = (1.0 - a) * w0 + a * w1
        self.move_width(self.commanded_width_m, dt_s)
        return ended

    def move_width(self, toward_m: float, dt_s: float) -> None:
        step = self._speed * max(0.0, dt_s)
        diff = toward_m - self.width_m
        self.width_m = toward_m if abs(diff) <= step else self.width_m + step * np.sign(diff)
