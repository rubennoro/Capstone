"""Pure helpers of the bring-up console (control_deploy/STRUCTURE.md 6.5): the minimum-jerk profile, out-hold-back
paths, translating and rotating a pose, and the streaming schedule (every 0.2 s the next 8 waypoints, 50 ms apart, the
first at least 100 ms ahead, so none is more than about 0.55 s ahead; max_lead_s is 1.5)."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SPACING_NS = 50_000_000       # waypoints 50 ms apart
LEAD_NS = 100_000_000         # the first one at least 100 ms ahead
PER_REQUEST = 8               # waypoints per set_waypoints
REQUEST_PERIOD_NS = 200_000_000


def min_jerk(tau: float) -> float:
    """s = 10 tau^3 - 15 tau^4 + 6 tau^5 on [0, 1] (clamped)."""
    t = min(max(tau, 0.0), 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def min_jerk_peak_speed(distance: float, duration_s: float) -> float:
    return 1.875 * abs(distance) / duration_s


def out_hold_back(segment_s: float, hold_s: float) -> tuple[float, Callable[[float], float]]:
    """(total duration, s(t)): 0 -> 1 over segment_s (min-jerk), 1 for hold_s, 1 -> 0 over segment_s."""
    total = 2.0 * segment_s + hold_s

    def s(t: float) -> float:
        if t <= segment_s:
            return min_jerk(t / segment_s)
        if t <= segment_s + hold_s:
            return 1.0
        return 1.0 - min_jerk((t - segment_s - hold_s) / segment_s)

    return total, s


def translate(pose: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The pose moved by d in the base frame."""
    out = np.array(pose, dtype=np.float64, copy=True)
    out[:3, 3] += np.asarray(d, dtype=np.float64)
    return out


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_about_tcp_z(pose: np.ndarray, angle: float) -> np.ndarray:
    """The pose rotated about its own z axis (the TCP's): position and z axis unchanged."""
    out = np.array(pose, dtype=np.float64, copy=True)
    out[:3, :3] = pose[:3, :3] @ rot_z(angle)
    return out


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle (rad) of the rotation between two poses."""
    r = a[:3, :3].T @ b[:3, :3]
    return math.acos(max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0)))


@dataclass(frozen=True)
class Path:
    """A timed path: at t0_ns + t (t in [0, duration_s]) the TCP pose and the gripper width."""

    t0_ns: int
    duration_s: float
    at: Callable[[float], tuple[np.ndarray, float]]

    @property
    def end_ns(self) -> int:
        return self.t0_ns + grid_count(self.duration_s) * SPACING_NS

    def sample(self, t_ns: int) -> tuple[np.ndarray, float]:
        t = (t_ns - self.t0_ns) * 1e-9
        return self.at(min(max(t, 0.0), self.duration_s))


def grid_count(duration_s: float) -> int:
    """Grid steps of 50 ms covering the duration (the last grid point is at or after the end)."""
    return max(1, math.ceil(duration_s * 1e9 / SPACING_NS - 1e-9))


def time_grid(t0_ns: int, duration_s: float) -> list[int]:
    """t0, t0 + 50 ms, ..., up to the first point at or after the end: strictly increasing."""
    return [t0_ns + k * SPACING_NS for k in range(grid_count(duration_s) + 1)]


def next_request(path: Path, now_ns: int) -> list[int]:
    """The waypoint times of the request sent now: the next PER_REQUEST grid times at least LEAD_NS ahead (empty once
    the whole path has been sent)."""
    times = [t for t in time_grid(path.t0_ns, path.duration_s) if t >= now_ns + LEAD_NS]
    return times[:PER_REQUEST]


def chunks(items: list, size: int = PER_REQUEST) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# ---- the PC's plan, modelled (WaypointTrajectory: linear position and width, slerp rotation) ----

def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """(w, x, y, z), w >= 0."""
    m = np.asarray(r, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def slerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """The shorter way, as Eigen's slerp."""
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 1.0 - 1e-12:
        q = a + s * (b - a)
        return q / np.linalg.norm(q)
    theta = math.acos(min(1.0, d))
    q = (math.sin((1.0 - s) * theta) * a + math.sin(s * theta) * b) / math.sin(theta)
    return q / np.linalg.norm(q)


@dataclass(frozen=True)
class PlanPoint:
    t_ns: int
    pose: np.ndarray
    width_m: float


class PlanModel:
    """What franka_control's plan should be: after `start`, [the target, now]; after each accepted set_waypoints,
    [plan.at(t_handle), accepted...]. at(t) interpolates as WaypointTrajectory::at does. Each version is kept, so a C2
    sample is compared with the plan that was in effect at its t_ns."""

    def __init__(self, t_ns: int, pose: np.ndarray, width_m: float) -> None:
        self.versions: list[tuple[int, list[PlanPoint]]] = [(t_ns, [PlanPoint(t_ns, np.asarray(pose), width_m)])]

    @staticmethod
    def _at(points: list[PlanPoint], t_ns: int) -> tuple[np.ndarray, float]:
        if t_ns <= points[0].t_ns:
            return points[0].pose, points[0].width_m
        for a, b in zip(points, points[1:]):
            if t_ns < b.t_ns:
                s = (t_ns - a.t_ns) / (b.t_ns - a.t_ns)
                pose = np.eye(4)
                pose[:3, 3] = a.pose[:3, 3] + s * (b.pose[:3, 3] - a.pose[:3, 3])
                pose[:3, :3] = matrix_from_quat(slerp(quat_from_matrix(a.pose[:3, :3]),
                                                      quat_from_matrix(b.pose[:3, :3]), s))
                return pose, a.width_m + s * (b.width_m - a.width_m)
        return points[-1].pose, points[-1].width_m

    def apply(self, t_handle_ns: int, accepted: list[PlanPoint]) -> None:
        """An accepted request handled at t_handle_ns (the midpoint of its round trip on the PC clock)."""
        pose, width = self.at(t_handle_ns)
        self.versions.append((t_handle_ns, [PlanPoint(t_handle_ns, pose, width), *accepted]))

    def at(self, t_ns: int) -> tuple[np.ndarray, float]:
        points = self.versions[0][1]
        for start, pts in self.versions:
            if t_ns >= start:
                points = pts
        return self._at(points, t_ns)
