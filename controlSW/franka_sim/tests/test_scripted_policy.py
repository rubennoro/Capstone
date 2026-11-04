"""Scripted pick-and-place policy: pure tests (no sim, no sockets)."""

import math

import numpy as np

from franka_sim.policies.scripted import planner as pl
from franka_sim.policies.scripted import trajectory as tj

V, W = 0.25, 1.0
R_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])     # TCP z pointing down, x heading 0


def T_of(R, p):
    return pl.make_T(R, p)


def eraser_T(x, y, yaw, z=0.006):
    return T_of(pl.Rz(yaw), [x, y, z])


def test_relative_absolute_round_trip():
    T0 = T_of(pl.Rz(0.3) @ R_DOWN, [0.5, -0.02, 0.2])
    T1 = T_of(pl.Rz(-0.9) @ R_DOWN, [0.45, 0.15, 0.03])
    Ts = tj.chunk(T0, T1, V, W)
    pos, r6 = tj.to_relative(T0, Ts)
    assert pos.shape == (16, 3) and r6.shape == (16, 6)
    assert np.allclose(pos[0], 0, atol=1e-12) and np.allclose(r6[0], [1, 0, 0, 0, 1, 0], atol=1e-12)
    back = tj.to_absolute(T0, pos, r6)
    assert np.max(np.abs(back - Ts)) < 1e-9
    # rot6d = the first two ROWS of the relative rotation
    rel = np.linalg.inv(T0) @ Ts[7]
    assert np.allclose(r6[7], rel[:2, :3].reshape(6), atol=1e-12)


def test_chunk_respects_speed_limits():
    frac = 0.9
    for T1 in (T_of(R_DOWN, [0.8, 0.3, 0.4]), T_of(pl.Rz(2.5) @ R_DOWN, [0.5, 0.0, 0.2]),
               T_of(pl.Rz(-1.2) @ R_DOWN, [0.3, -0.2, 0.05]), T_of(R_DOWN, [0.5, 0.0, 0.2005])):
        T = T_of(R_DOWN, [0.5, 0.0, 0.2])
        for i in range(40):                           # follow the chunks as the task would (executes actions 0..7)
            Ts, at_goal = tj.step_chunk(T, T1, V, W, frac, 0.2)
            dp = np.linalg.norm(np.diff(Ts[:, :3, 3], axis=0), axis=1) / tj.DT_S
            da = np.array([tj.rot_angle(Ts[i, :3, :3], Ts[i + 1, :3, :3]) for i in range(15)]) / tj.DT_S
            assert dp.max() <= frac * V + 1e-9 and da.max() <= frac * W + 1e-9
            T = Ts[7]
        assert at_goal and np.allclose(Ts[7], T1) and np.allclose(Ts[-1], T1)    # converges and holds the goal


def test_yaw_choice_mod_180_closest_to_current():
    # closing axis (TCP x) along the eraser's local y: eraser yaw 100 deg -> 10 deg or -170 deg; current 0 -> 10 deg
    y = pl.choose_yaw(math.radians(100), 0.0)
    assert abs(math.degrees(y) - 10.0) < 1e-9
    y = pl.choose_yaw(math.radians(100), math.radians(-150))
    assert abs(math.degrees(y) + 170.0) < 1e-9
    for e in np.radians(np.arange(-180, 181, 7)):
        y = pl.choose_yaw(e, 0.0)
        assert abs(pl.wrap(y)) <= math.pi / 2 + 1e-12
        assert abs(math.cos(y - (e + math.pi / 2))) > 1 - 1e-9     # parallel to the eraser y axis
    R = pl.goal_rotation(R_DOWN, math.radians(10))
    assert np.allclose(R[:, 2], [0, 0, -1]) and abs(math.degrees(pl.yaw_of(R)) - 10) < 1e-9


def test_planner_walks_all_phases():
    p = pl.Params(grasp_z_m=0.02, place_xy_m=(0.45, 0.15), open_gap_min_m=0.07)
    mem = pl.Memory()
    home = T_of(R_DOWN, [0.5, 0.0, 0.25])
    E = eraser_T(0.52, -0.01, math.radians(15))
    s = {"tcp_T_base": home, "eraser_T_base": E, "hand_grasped": False, "hand_gap_m": 0.08}
    seen = []
    for _ in range(200):
        phase, G, w = pl.plan(s, p, mem)
        if not seen or seen[-1] != phase:
            seen.append(phase)
        if phase == "DONE":
            break
        s = dict(s, tcp_T_base=G)                     # the arm reaches each goal
        if phase == "CLOSE":
            assert abs(w - p.width_closed) < 1e-12
            s["hand_grasped"], s["hand_gap_m"] = True, 0.012
        elif phase in ("LIFT", "CARRY", "LOWER"):     # the eraser moves with the TCP
            off = mem.locks["grasp"][:3, 3] - E[:3, 3]
            s["eraser_T_base"] = T_of(E[:3, :3], G[:3, 3] - off)
        elif phase == "OPEN":
            assert abs(w - (p.width_a * 0.075 + p.width_b)) < 1e-12
            s["hand_grasped"], s["hand_gap_m"] = False, 0.075
    assert seen == list(pl.PHASES)
    g = mem.locks["grasp"]
    assert abs(g[2, 3] - 0.02) < 1e-12 and np.allclose(g[:2, 3], E[:2, 3])
    assert abs(math.degrees(pl.yaw_of(g[:3, :3])) - (15 - 90)) < 1e-9       # closest to the home yaw 0
    er = s["eraser_T_base"]
    assert np.allclose(er[:2, 3], p.place_xy_m) and abs(er[2, 3] - 0.006) < 1e-12
    assert abs(G[2, 3] - (0.02 + p.retreat_height_m)) < 1e-12
    # reset starts over at ABOVE
    mem.reset()
    assert mem.phase == "ABOVE" and not mem.locks
