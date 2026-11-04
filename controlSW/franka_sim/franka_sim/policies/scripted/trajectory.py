"""Pure min-jerk chunks and the relative-pose conversion of the policy contract (``controlSW/docs/adding_a_policy.md``).

``chunk(T0, T1, ...)``: K = 16 TCP poses at 50 ms, sample k at t = t_offset + k * dt (t_offset 0: sample 0 = T0), position on a min-jerk
profile, rotation slerped on the same profile, the goal held past the end. Duration =
max(MINJERK_PEAK * dist / (speed_frac * v_max), MINJERK_PEAK * angle / (speed_frac * w_max), min_segment_s). The
min-jerk peak speed is 1.875 x the mean speed, so with this factor no step is faster than speed_frac * v_max / w_max
(dist / v alone would give a peak of 1.69 x v_max).

``step_chunk(T_ref, G, ...)``: one rest-to-rest move per call towards the goal (see its docstring).

``to_relative(T_ref, Ts)``: each pose in the frame of ``T_ref`` (inv(T_ref) @ T), rot6d = the first two ROWS of the
relative rotation (the training / robot_utils convention); ``to_absolute`` inverts it the way the task decodes a chunk.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

K_STEPS = 16
DT_S = 0.05
MINJERK_PEAK = 1.875


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return np.clip(10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5, 0.0, 1.0)     # the polynomial can round past 1


def rot_angle(Ra, Rb) -> float:
    c = (float(np.trace(np.asarray(Ra, float).T @ np.asarray(Rb, float))) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def duration_s(T0, T1, v_max: float, w_max: float, speed_frac: float, min_segment_s: float) -> float:
    T0, T1 = np.asarray(T0, float), np.asarray(T1, float)
    d = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    a = rot_angle(T0[:3, :3], T1[:3, :3])
    return max(MINJERK_PEAK * d / (speed_frac * v_max), MINJERK_PEAK * a / (speed_frac * w_max), float(min_segment_s))


def chunk(T0, T1, v_max: float, w_max: float, speed_frac: float = 0.9, min_segment_s: float = 0.2,
          k: int = K_STEPS, dt: float = DT_S, t_offset_s: float = 0.0) -> np.ndarray:
    """(k, 4, 4) absolute poses of the min-jerk segment T0 -> T1, sampled at t_offset_s + i * dt."""
    T0, T1 = np.asarray(T0, float), np.asarray(T1, float)
    D = duration_s(T0, T1, v_max, w_max, speed_frac, min_segment_s)
    s = min_jerk((float(t_offset_s) + np.arange(k) * dt) / D)
    sl = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))
    out = np.tile(np.eye(4), (k, 1, 1))
    out[:, :3, :3] = sl(np.clip(s, 0.0, 1.0)).as_matrix()
    out[:, :3, 3] = T0[:3, 3] + s[:, None] * (T1[:3, 3] - T0[:3, 3])
    out[s >= 1.0] = T1                                 # past the end: exactly the goal
    return out


def step_chunk(T_ref, G, v_max: float, w_max: float, speed_frac: float = 0.9, min_segment_s: float = 0.2,
               k: int = K_STEPS, dt: float = DT_S, exec_h: int = 8) -> tuple[np.ndarray, bool]:
    """(k, 4, 4) poses, reaches_goal: a rest-to-rest min-jerk move from ``T_ref`` that ends by the last EXECUTED
    action (index exec_h - 1, t = (exec_h - 1) * dt = 0.35 s), towards ``G`` but no further than such a move can go
    within the speed limits (MINJERK_PEAK x mean <= speed_frac x v_max / w_max): 4.2 cm / 9.6 deg per call at the
    default limits. Later samples hold that waypoint.

    Why rest-to-rest: the task runs with ``pause`` (deploy.yaml task.pause): it executes actions 0..exec_h-1, waits for
    the arm to settle, then asks again, so every chunk starts with the arm at rest. A chunk restarted on the full
    remaining distance would barely move (min-jerk starts slowly: ~1 % of a 0.4 m move per call), and a segment
    continued across calls at cruise speed starts each chunk with a velocity step (the impedance arm then lagged ~17 mm
    and overshot the commanded grasp height by 7.6 mm, into the table: cartesian reflex)."""
    T_ref, G = np.asarray(T_ref, float), np.asarray(G, float)
    Tw = (int(exec_h) - 1) * float(dt)
    d = float(np.linalg.norm(G[:3, 3] - T_ref[:3, 3]))
    a = rot_angle(T_ref[:3, :3], G[:3, :3])
    d_max = speed_frac * v_max * Tw / MINJERK_PEAK
    a_max = speed_frac * w_max * Tw / MINJERK_PEAK
    f = min(1.0, d_max / d if d > 0 else 1.0, a_max / a if a > 0 else 1.0)
    if f >= 1.0:
        return chunk(T_ref, G, v_max, w_max, speed_frac, min_segment_s, k, dt), True
    W = np.eye(4)
    W[:3, 3] = T_ref[:3, 3] + f * (G[:3, 3] - T_ref[:3, 3])
    W[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T_ref[:3, :3], G[:3, :3]])))(float(np.clip(f, 0.0, 1.0))).as_matrix()
    return chunk(T_ref, W, v_max, w_max, speed_frac, min_segment_s, k, dt), False


def to_relative(T_ref, Ts):
    """(pos (k,3), rot6d (k,6)) float64 of each pose of ``Ts`` in the frame of ``T_ref``."""
    rel = np.linalg.inv(np.asarray(T_ref, float)) @ np.asarray(Ts, float)
    return rel[:, :3, 3].copy(), rel[:, :2, :3].reshape(len(rel), 6).copy()


def rot6d_to_R(r6) -> np.ndarray:
    """Rows b1, b2 (Gram-Schmidt), b3 = b1 x b2 (as robot_utils.pose_utils.rot_6d_to_mat)."""
    r6 = np.asarray(r6, float)
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def to_absolute(T_ref, pos, rot6d) -> np.ndarray:
    rel = np.tile(np.eye(4), (len(pos), 1, 1))
    rel[:, :3, :3] = rot6d_to_R(rot6d)
    rel[:, :3, 3] = pos
    return np.asarray(T_ref, float) @ rel
