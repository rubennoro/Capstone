"""A4 SimCore (deterministic, no threads) on both arms: hold, 2 cm steps, payload error, torque rate limit, reflexes
(joint limit, table push, self collision, injected), begin_joint_move, gravity model, determinism.

Run: cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_simcore.py -s
"""

from __future__ import annotations

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import model as M  # noqa: E402
from franka_sim.simcore import SimCore, collision_thresholds, make_core_cfg  # noqa: E402

ROBOTS = ("panda", "fr3")


def station(robot: str):
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = robot
    return load_station(spec)


@pytest.fixture(scope="module", params=ROBOTS)
def arm(request):
    st = station(request.param)
    return st, M.build_model(st)


def core_of(arm, **phys) -> SimCore:
    st, sm = arm
    cfg = make_core_cfg(st)
    cfg["physics"] = {**cfg["physics"], **phys}
    return SimCore(sm, cfg)


def pose(c):
    return c.state()["O_T_EE"]


def rot_deg(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def eff_mass(c) -> np.ndarray:
    """Translational operational-space inertia diagonal at the EE (kg)."""
    Mf = np.zeros((c.m.nv, c.m.nv))
    mujoco.mj_fullM(c.m, Mf, c.d.qM)
    Ma = Mf[np.ix_(c._dadr, c._dadr)]
    J = c._J[:3]
    return np.diag(np.linalg.inv(J @ np.linalg.solve(Ma, J.T)))


def test_gravity_matches_mj_rne(arm):
    c = core_of(arm)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(c.q_lo + 0.2, c.q_hi - 0.2)
        c.d.qpos[c._qadr] = q
        c.d.qvel[:] = 0
        mujoco.mj_forward(c.m, c.d)
        out = np.zeros(c.m.nv)
        mujoco.mj_rne(c.m, c.d, 0, out)
        gb, _ = c._gravity()
        assert np.abs(gb - out[c._dadr]).max() < 1e-9


def test_hold_drift(arm):
    c = core_of(arm)
    T0 = pose(c)
    c.start()
    c.step(5000)
    T = pose(c)
    assert np.linalg.norm(T[:3, 3] - T0[:3, 3]) < 0.5e-3
    assert rot_deg(T[:3, :3], T0[:3, :3]) < 0.1
    assert c.error is None and c.mode == "running"


def test_step_response_each_axis(arm):
    """2 cm target step per axis. The overshoot is that of the documented law (D = 2 zeta sqrt(K), not mass-normalised:
    effective zeta = zeta / sqrt(Lambda_ii)), bounded by the ideal 2nd-order overshoot of that zeta (+6 %)."""
    c = core_of(arm)
    out = {}
    for ax in range(3):
        c.recover()
        c.start()
        T = pose(c).copy()
        p0 = T[:3, 3].copy()
        lam = eff_mass(c)[ax]
        T[ax, 3] += 0.02
        c.set_target(T)
        xs = np.empty(3000)
        for k in range(3000):
            c.step(1)
            xs[k] = pose(c)[ax, 3] - p0[ax]
        ov = (xs.max() - 0.02) / 0.02
        out_of = np.flatnonzero(np.abs(xs - 0.02) > 1e-3)
        settle_ms = int(out_of[-1] + 1) if out_of.size else 0
        zeta = c.D[ax] / (2 * np.sqrt(c.K[ax] * lam))
        ov_ideal = float(np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2))) if zeta < 1 else 0.0
        out["xyz"[ax]] = {"overshoot_pct": round(100 * ov, 2), "ideal_pct": round(100 * ov_ideal, 2),
                          "lambda_kg": round(float(lam), 2), "steady_err_mm": round(abs(xs[-1] - 0.02) * 1e3, 4),
                          "settle_1mm_ms": settle_ms}
        assert abs(xs[-1] - 0.02) <= 1e-3
        assert ov <= ov_ideal + 0.06 and ov < 0.40       # Lambda is not diagonal: the 1-D theory is approximate
        assert settle_ms < 2000
        assert c.error is None
    print(f"\n[step] {c.sm.robot} K={c.K[0]:.0f}: {json.dumps(out)}")


def test_payload_error_steady_state(arm):
    c = core_of(arm, payload_error_kg=0.3)
    c.start()
    z0 = pose(c)[2, 3]
    c.step(3000)
    dz = pose(c)[2, 3] - z0
    expect = 0.3 * 9.81 / c.K[2]
    print(f"\n[payload] {c.sm.robot} dz={dz * 1e3:.3f} mm expected m g/K_z={expect * 1e3:.3f} mm")
    assert dz > 0                                         # Desk told a heavier payload: the arm sits higher
    assert abs(dz - expect) <= 0.3 * expect


def test_rate_limiter_never_exceeded(arm):
    c = core_of(arm)
    c.start()
    T = pose(c).copy()
    T[:3, 3] += [0.05, -0.05, -0.05]                     # a large jump: the raw torque step is >> 1 Nm
    c.set_target(T)
    prev = c.tau_prev.copy()
    worst = 0.0
    for _ in range(500):
        c.step(1)
        worst = max(worst, float(np.abs(c.tau_prev - prev).max()))
        prev = c.tau_prev.copy()
        assert np.all(np.abs(c.tau_prev) <= c.tau_max + 1e-12)
    assert worst <= 1.0 + 1e-12 and c.dtau_max_seen <= 1.0 + 1e-12
    assert worst > 0.99                                  # the limiter was active


def Rz(ang: float) -> np.ndarray:
    return np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])


BARE_TOOLING = ["tooling.cradle_boxes=[]", "tooling.housing.size_m=[0.002,0.002,0.002]",
                "tooling.housing.main_lens_in_body_m=[0,0,0]"]


def rotate_q7(c, deg: float, ramp_ms: int = 2500, n: int = 5000) -> int:
    """Rotate the EE target about its own z by up to ``deg`` (ramped, 200 Hz targets like the bridge); steps until a
    reflex or n ms. Returns the step count."""
    T0 = pose(c).copy()
    for k in range(n):
        if k % 5 == 0:
            T = T0.copy()
            T[:3, :3] = T0[:3, :3] @ Rz(np.radians(deg) * min(1.0, k / ramp_ms))
            c.set_target(T)
        c.step(1)
        if c.error:
            return k
    return n


def test_joint_limit_reflex_and_start_refused(arm):
    """A target beyond the q7 limit. The real phone/cradle hits link 5/6 before q7 reaches its limit on both arms (see
    test_self_collision_reflex), so this test shrinks the tooling to a 2 mm housing to isolate the limit check."""
    st0, _ = arm
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = st0.ids["robot"]
    st = load_station(spec, sets=BARE_TOOLING)
    c = core_of((st, M.build_model(st)))
    c.start()
    rotate_q7(c, 175.0)
    assert c.mode == "reflex" and c.error.startswith("joint_position_limits_violation: q7"), c.error
    with pytest.raises(RuntimeError):
        c.start()
    c.recover()
    assert c.error is None and c.mode == "idle"
    c.start()
    assert c.mode == "running"


def test_table_push_contact_reflex(arm):
    c = core_of(arm)
    m = c.m
    adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, c.sm.names["joints"]["object"])]
    c.d.qpos[adr:adr + 3] = [0.0, 0.6, 0.03]             # the eraser out of the way
    mujoco.mj_forward(m, c.d)
    c.recover()
    c.start()
    T0 = pose(c).copy()
    tcp_z = (T0 @ c.sm.EE_T_TCP)[2, 3]
    z_goal = T0[2, 3] - (tcp_z - 0.0) - 0.05              # TCP 5 cm below the table top (z = 0)
    hit = None
    for k in range(20000):
        if k % 5 == 0:
            T = T0.copy()
            T[2, 3] = max(z_goal, T0[2, 3] - 0.10 * k * 1e-3)   # 0.1 m/s down
            c.set_target(T)
        c.step(1)
        if c.error:
            hit = k
            break
    print(f"\n[table] {c.sm.robot} reflex after {hit} ms: {c.error}; |F_ext| {np.abs(c.O_F_ext).max():.1f}")
    assert c.mode == "reflex" and c.error.split(":")[0] in ("cartesian_reflex", "joint_reflex"), c.error


def test_self_collision_reflex(arm):
    """Rotating q7 with the real tooling: the phone/cradle reaches link 6 (+170 deg; a pair MuJoCo's parent filter
    never collides, caught by the distance check) or link 5 (-170 deg; a MuJoCo contact)."""
    out = {}
    for sign in (1, -1):
        c = core_of(arm)
        c.start()
        k = rotate_q7(c, sign * 170.0)
        out[sign] = (k, c.error)
        assert c.mode == "reflex" and c.error.startswith("self_collision_avoidance_violation"), c.error
    assert "distance check" in out[1][1] and ("link6" in out[1][1])
    assert "link5" in out[-1][1] and "distance check" not in out[-1][1]
    print(f"\n[self] {arm[1].robot}: {out}")


def test_begin_joint_move_accuracy(arm):
    c = core_of(arm)
    q0 = c.q
    goal = np.clip(q0 + np.array([0.2, -0.15, 0.1, 0.15, -0.2, 0.2, 0.3]), c.q_lo + 0.05, c.q_hi - 0.05)
    T = c.begin_joint_move(goal, 0.3)
    assert c.mode == "moving" and T > 0.05
    with pytest.raises(RuntimeError):
        c.start()                                         # refused during the move
    c.step(int(T * 1000) + 1500)
    assert c.mode == "idle" and c.error is None
    assert np.abs(c.q - goal).max() < 1e-3
    c.start()
    assert c.mode == "running"


def test_inject_reflex_and_targets_ignored(arm):
    c = core_of(arm)
    c.start()
    T0 = pose(c).copy()
    c.inject_reflex("cartesian_reflex (test)")
    assert c.mode == "reflex" and c.error == "cartesian_reflex (test)"
    T = T0.copy()
    T[0, 3] += 0.03
    c.set_target(T)
    c.step(500)
    assert np.linalg.norm(pose(c)[:3, 3] - T0[:3, 3]) < 1e-3
    s = c.state()
    assert set(s) >= {"O_T_EE", "q", "dq", "O_F_ext", "tau_ext", "mode", "error", "sim_t", "success_rate"}


def test_deterministic(arm):
    runs = []
    for _ in range(2):
        c = core_of(arm)
        c.start()
        T = pose(c).copy()
        T[:3, 3] += [0.01, 0.02, -0.01]
        c.set_target(T)
        c.step(400)
        runs.append(c.d.qpos.copy())
    assert np.array_equal(runs[0], runs[1])


def test_collision_threshold_formats():
    tau, f = collision_thresholds(None)
    assert tau.tolist() == [20, 20, 18, 18, 16, 14, 12] and f.tolist() == [20, 20, 20, 25, 25, 25]
    lo, hi = [10.0] * 7, [30.0] * 7
    tau, f = collision_thresholds([lo, hi, [5.0] * 6, [40.0] * 6])
    assert tau.tolist() == hi and f.tolist() == [40.0] * 6
    tau, f = collision_thresholds({"tau_nm": 15.0, "force_n": [22.0] * 6})
    assert tau.tolist() == [15.0] * 7 and f.tolist() == [22.0] * 6


def test_joint_limits_are_station_within_mjcf(arm):
    """The limit reflex uses the station datasheet limits intersected with the MJCF range (the same as Kin); on FR3
    the MJCF is wider, so a q inside the MJCF but outside the datasheet must trip it."""
    st, sm = arm
    c = core_of(arm)
    lo = np.maximum(st.get("robot.joints.q_min_rad"), sm.limits["mjcf_range_rad"][:, 0])
    hi = np.minimum(st.get("robot.joints.q_max_rad"), sm.limits["mjcf_range_rad"][:, 1])
    assert np.allclose(c.q_lo, lo) and np.allclose(c.q_hi, hi)
    if sm.robot == "fr3":
        assert np.any(c.q_lo > sm.limits["mjcf_range_rad"][:, 0] + 1e-3)     # FR3: datasheet tighter than MJCF
    i = 5                                                                    # q6
    c.d.qpos[c._qadr[i]] = c.q_hi[i] + 0.01
    mujoco.mj_forward(c.m, c.d)
    c.mode = "running"
    assert (c._check_reflex(False) or "").startswith("joint_position_limits_violation: q6")
