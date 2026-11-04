"""Pure pick-and-place planner: ``plan(state, params, mem) -> (phase, goal_T, goal_width_train)``.

Re-evaluated from the live sim state (I-SIMSTATE dict: ``tcp_T_base``, ``eraser_T_base``, ``hand_grasped``,
``hand_gap_m``) on every inference call. ``mem`` (a ``Memory``) holds the current phase, which only moves forward, plus
the values a phase locks when it starts. Phases, in order:

  ABOVE    TCP at the eraser xy, z = grasp z + approach_height_m, yawed to the eraser; hand open
  DESCEND  same xy / yaw, z = grasp z                                         (locks the grasp pose when done)
  CLOSE    hold the grasp pose, width closed; done when ``hand_grasped``
  LIFT     grasp pose + lift_height_m
  CARRY    TCP xy = place xy + (TCP xy - eraser xy at the CARRY start), lift height, same rotation
  LOWER    same xy, z = eraser start z + (TCP z - eraser z at the CARRY start) (= grasp z when the eraser did not slip)
  OPEN     hold, width open; done when ``hand_gap_m >= open_gap_min_m``
  RETREAT  LOWER pose + retreat_height_m
  DONE     hold

A move phase is done when the measured TCP is within ``pos_tol_m`` and ``rot_tol_deg`` of its goal.

Grasp z (absolute TCP z in the base frame) = ``scene.table.top_z_base_m`` + ``franka_sim.scene.grasp_tcp_height_m`` (the
sim's own convention: box fingers put the TCP at the pad centre, so 0.02 with 30 mm pads; kit fingers put it at the
fingertip, so 0.005); the server computes it and passes it in ``Params.grasp_z_m``.

Orientation: the TCP z axis stays as at the episode's first call (home, pointing down); only the yaw (heading of the TCP
x axis = the finger closing axis, ``tcp_x_along_hand``) changes, by a rotation about the base z axis. The goal yaw puts
the closing axis along the eraser's local y (its 20 mm side); of the two solutions (mod 180 deg) the one closest to the
current TCP yaw is used. Widths are training units: closed = width_map b (gap 0), open = a * open_gap_m + b.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

PHASES = ("ABOVE", "DESCEND", "CLOSE", "LIFT", "CARRY", "LOWER", "OPEN", "RETREAT", "DONE")
OPEN_PHASES = ("ABOVE", "DESCEND", "OPEN", "RETREAT", "DONE")


@dataclass
class Params:
    approach_height_m: float = 0.10
    lift_height_m: float = 0.10
    retreat_height_m: float = 0.10
    pos_tol_m: float = 0.003
    rot_tol_deg: float = 2.0
    open_gap_m: float = 0.075
    speed_frac_ratio: float = 0.9
    min_segment_s: float = 0.2
    # from the station (the server fills these in)
    grasp_z_m: float = 0.02              # absolute TCP z of the grasp (table top + grasp_tcp_height_m)
    place_xy_m: tuple = (0.45, 0.15)
    width_a: float = 1.0                 # width_map: train = a * gap + b
    width_b: float = 0.0431
    open_gap_min_m: float = 0.07         # OPEN is done at this hand gap (= criteria task.open_gap_min_m)
    v_max_m_s: float = 0.25
    w_max_rad_s: float = 1.0
    exec_h: int = 8                      # policy contract exec_h (each chunk moves until action exec_h - 1)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d or {}).items() if k in names})

    @property
    def width_closed(self) -> float:
        return float(self.width_b)

    @property
    def width_open(self) -> float:
        return float(self.width_a * self.open_gap_m + self.width_b)


@dataclass
class Memory:
    i: int = 0                                     # index into PHASES
    R_ref: np.ndarray | None = None                # TCP rotation at the first call (home)
    eraser_z0: float | None = None                 # eraser z at the first call
    locks: dict = field(default_factory=dict)      # values a phase fixed when it started

    @property
    def phase(self) -> str:
        return PHASES[self.i]

    def reset(self) -> None:
        self.i, self.R_ref, self.eraser_z0, self.locks = 0, None, None, {}


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(R) -> float:
    """Heading of the x axis in the base xy plane."""
    R = np.asarray(R, float)
    return math.atan2(R[1, 0], R[0, 0])


def Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def choose_yaw(eraser_yaw: float, current_yaw: float) -> float:
    """TCP yaw putting the TCP x axis (closing axis) along the eraser's local y axis: eraser_yaw +/- 90 deg, the one
    closest to ``current_yaw``."""
    cands = [wrap(eraser_yaw + math.pi / 2), wrap(eraser_yaw - math.pi / 2)]
    return min(cands, key=lambda c: abs(wrap(c - current_yaw)))


def goal_rotation(R_ref, yaw: float) -> np.ndarray:
    """R_ref turned about the base z axis so that its x heading is ``yaw`` (its z axis keeps its direction)."""
    R_ref = np.asarray(R_ref, float)
    return Rz(wrap(yaw - yaw_of(R_ref))) @ R_ref


def rot_err_rad(Ra, Rb) -> float:
    c = (float(np.trace(np.asarray(Ra, float).T @ np.asarray(Rb, float))) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def reached(T, G, p: Params) -> bool:
    T, G = np.asarray(T, float), np.asarray(G, float)
    return (float(np.linalg.norm(T[:3, 3] - G[:3, 3])) <= p.pos_tol_m
            and rot_err_rad(T[:3, :3], G[:3, :3]) <= math.radians(p.rot_tol_deg))


def make_T(R, pos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _lock(mem: Memory, key: str, fn):
    if key not in mem.locks:
        mem.locks[key] = fn()
    return mem.locks[key]


def _phase_goal(phase: str, s: dict, p: Params, mem: Memory):
    """(goal_T, done) of ``phase`` for the state ``s``."""
    T = np.asarray(s["tcp_T_base"], float)
    E = np.asarray(s["eraser_T_base"], float)
    if phase in ("ABOVE", "DESCEND"):
        yaw = choose_yaw(yaw_of(E[:3, :3]), yaw_of(T[:3, :3]))
        R = goal_rotation(mem.R_ref, yaw)
        z = p.grasp_z_m + (p.approach_height_m if phase == "ABOVE" else 0.0)
        G = make_T(R, [E[0, 3], E[1, 3], z])
        done = reached(T, G, p)
        if done and phase == "DESCEND":
            mem.locks["grasp"] = G
        return G, done
    grasp = _lock(mem, "grasp", lambda: _phase_goal("DESCEND", s, p, mem)[0])   # CLOSE entered without DESCEND lock
    R = grasp[:3, :3]
    if phase == "CLOSE":
        return grasp, bool(s.get("hand_grasped"))
    if phase == "LIFT":
        G = grasp.copy()
        G[2, 3] += p.lift_height_m
        return G, reached(T, G, p)
    # CARRY onward: the TCP - eraser offset when the carry starts (the grasp may be off the eraser centre)
    off = _lock(mem, "offset", lambda: (T[:3, 3] - E[:3, 3]).copy())
    xy = np.asarray(p.place_xy_m, float) + off[:2]
    z_lift = grasp[2, 3] + p.lift_height_m
    z_low = float(mem.eraser_z0) + float(off[2])
    if phase == "CARRY":
        G = make_T(R, [xy[0], xy[1], z_lift])
        return G, reached(T, G, p)
    G_low = make_T(R, [xy[0], xy[1], z_low])
    if phase == "LOWER":
        return G_low, reached(T, G_low, p)
    if phase == "OPEN":
        gap = s.get("hand_gap_m")
        return G_low, gap is not None and float(gap) >= p.open_gap_min_m
    G = G_low.copy()
    G[2, 3] += p.retreat_height_m
    if phase == "RETREAT":
        return G, reached(T, G, p)
    return G, False                                   # DONE: hold


def plan(state: dict, params: Params, mem: Memory):
    """(phase, goal_T (4x4 TCP pose in base), goal_width_train). Advances ``mem`` through every phase that is already
    complete (forward only)."""
    if mem.R_ref is None:
        mem.R_ref = np.asarray(state["tcp_T_base"], float)[:3, :3].copy()
    if mem.eraser_z0 is None:
        mem.eraser_z0 = float(np.asarray(state["eraser_T_base"], float)[2, 3])
    while True:
        phase = mem.phase
        G, done = _phase_goal(phase, state, params, mem)
        if done and phase != "DONE":
            mem.i += 1
            continue
        w = params.width_open if phase in OPEN_PHASES else params.width_closed
        return phase, G, w
