"""A2 kinematics.py on the A1 model (both arms): FK vs Franka modified DH, Jacobian, IK, collisions, check_traj.

Run: cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_kinematics.py -s
"""

from __future__ import annotations

import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import frames as F  # noqa: E402
from franka_sim import kinematics as K  # noqa: E402
from franka_sim import model as M  # noqa: E402

ROBOTS = ("panda", "fr3")

# Franka modified DH (Craig): A_i = RotX(alpha_{i-1}) TransX(a_{i-1}) RotZ(theta_i) TransZ(d_i); flange = TransZ(0.107)
DH_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
DH_D = [0.333, 0, 0.316, 0, 0.384, 0, 0]
DH_ALPHA = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
# FK-vs-DH offset between the MJCF `flange` site and the DH flange: none (measured ~1e-15); the test would catch one.
DH_OFFSET = np.eye(4)


def dh_flange(q) -> np.ndarray:
    T = np.eye(4)
    for i in range(7):
        ca, sa, ct, st = np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i]), np.cos(q[i]), np.sin(q[i])
        T = T @ np.array([[ct, -st, 0, DH_A[i]],
                          [st * ca, ct * ca, -sa, -sa * DH_D[i]],
                          [st * sa, ct * sa, ca, ca * DH_D[i]],
                          [0, 0, 0, 1]])
    return T @ F.make_T(None, [0, 0, 0.107]) @ DH_OFFSET


@pytest.fixture(scope="module", params=ROBOTS)
def sm(request):
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = request.param
    return M.build_model(load_station(spec))


@pytest.fixture(scope="module")
def kin(sm):
    return K.Kin(sm, "flange")


def rot_angle(Ra, Rb) -> float:
    return float(np.linalg.norm(K.rot_err(Ra, Rb)))


def find_singular(kin) -> np.ndarray:
    """A configuration with sigma_min(J) < 1e-6 (shoulder q2 = 0 family), found by a seeded local minimisation."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    q0 = np.array([0.465, 0.0, 1.541, -2.754, 2.515, 1.294, -2.286])
    f = lambda q: np.linalg.svd(kin.jac(q), compute_uv=False)[-1]  # noqa: E731
    r = scipy_opt.minimize(f, q0, bounds=list(zip(kin.q_lo, kin.q_hi)), method="L-BFGS-B")
    assert r.fun < 1e-6, r.fun
    return r.x


def q_into_table(sm, kin) -> np.ndarray:
    """A reachable configuration whose TCP is 2 cm below the table top, approach axis straight down."""
    k = K.Kin(sm, "tcp")
    top = float(sm.model.body("table").pos[2])
    T = F.make_T(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]]), [0.45, 0.0, top - 0.02])
    q, info = k.ik(T, kin.q_home)
    assert info["ok"]
    return q


def rand_q(kin, rng, n=None):
    return rng.uniform(kin.q_lo, kin.q_hi, size=(7,) if n is None else (n, 7))


# ------------------------------------------------------------------------------------------------------- FK


def test_fk_matches_modified_dh(sm, kin):
    rng = np.random.default_rng(1)
    dp = dr = 0.0
    for q in rand_q(kin, rng, 200):
        A, B = kin.fk(q), dh_flange(q)
        dp, dr = max(dp, np.abs(A[:3, 3] - B[:3, 3]).max()), max(dr, rot_angle(A[:3, :3], B[:3, :3]))
    print(f"\n[{sm.robot}] FK vs DH over 200 q: max position {dp:.2e} m, rotation {dr:.2e} rad")
    assert dp < 1e-6 and dr < 1e-6


def test_fk_frames_ee_and_tcp(sm, kin):
    q = rand_q(kin, np.random.default_rng(2))
    Tf = kin.fk(q)
    F_T_ee = np.array(M.export_frames(sm)["F_T_ee"])
    assert np.allclose(K.Kin(sm, "tcp").fk(q), Tf @ sm.F_T_TCP, atol=1e-9)
    assert np.allclose(K.Kin(sm, "ee").fk(q), Tf @ F_T_ee, atol=1e-9)
    assert np.allclose(sm.F_T_TCP, F_T_ee @ sm.EE_T_TCP, atol=1e-9)
    with pytest.raises(KeyError):
        K.Kin(sm, "nope")


def test_jacobian_matches_finite_difference(sm):
    rng = np.random.default_rng(3)
    for frame in ("flange", "tcp"):
        k = K.Kin(sm, frame)
        for q in rand_q(k, rng, 10):
            J, h = k.jac(q), 1e-6
            Jn = np.zeros((6, 7))
            for i in range(7):
                dq = np.zeros(7)
                dq[i] = h
                Tp, Tm = k.fk(q + dq), k.fk(q - dq)
                Jn[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
                Jn[3:, i] = K.rot_err(Tm[:3, :3], Tp[:3, :3]) / (2 * h)       # world-frame angular velocity
            assert np.abs(J - Jn).max() < 1e-5


def test_manipulability_and_margin(kin):
    assert kin.manipulability(kin.q_home) > 0.01
    qs = find_singular(kin)
    assert np.linalg.svd(kin.jac(qs), compute_uv=False)[-1] < 1e-6
    assert kin.manipulability(qs) < 1e-6
    mid = (kin.q_lo + kin.q_hi) / 2
    assert np.allclose(kin.joint_margin(mid), (kin.q_hi - kin.q_lo) / 2)
    assert np.allclose(kin.joint_margin(kin.q_lo), 0)


# ------------------------------------------------------------------------------------------------------- IK


@pytest.mark.parametrize("seed_kind", ["rest", "random"])
def test_ik_round_trip(sm, kin, seed_kind):
    rng = np.random.default_rng(10 if seed_kind == "rest" else 11)
    N, ok, t0 = 500, 0, time.perf_counter()
    tol_p, tol_r = 1e-4, 1e-3
    for q_true in rand_q(kin, rng, N):
        T = kin.fk(q_true)
        q0 = kin.q_home if seed_kind == "rest" else rand_q(kin, rng)
        q, info = kin.ik(T, q0, q_rest=kin.q_home)
        assert np.all(q >= kin.q_lo) and np.all(q <= kin.q_hi)
        Tq = kin.fk(q)
        assert np.isclose(info["err_p"], np.linalg.norm(Tq[:3, 3] - T[:3, 3]), atol=1e-9)
        if info["ok"]:
            ok += 1
            assert info["err_p"] <= tol_p and info["err_r"] <= tol_r
            assert np.linalg.norm(Tq[:3, 3] - T[:3, 3]) <= tol_p and rot_angle(Tq[:3, :3], T[:3, :3]) <= tol_r
            assert info["margin_min"] >= 0.02 - 1e-9
    dt = (time.perf_counter() - t0) / N * 1e3
    print(f"\n[{sm.robot}] IK round trip ({seed_kind} seeds): {ok}/{N} ok, {dt:.2f} ms per pose incl. restarts")
    assert ok / N >= 0.99


def test_ik_multistart_prefers_margin(sm, kin):
    rng = np.random.default_rng(12)
    T = kin.fk(rand_q(kin, rng))
    seeds = np.vstack([kin.q_home, rand_q(kin, rng, 6)])
    q, info = kin.ik_multistart(T, seeds)
    assert info["ok"] and info["n_ok"] >= 1 and 0 <= info["seed"] < len(seeds)
    singles = [kin.ik(T, s, restarts=0) for s in seeds]
    assert info["margin_min"] >= max(i["margin_min"] for _, i in singles if i["ok"]) - 1e-12


def test_ik_unreachable_and_singular_do_not_raise(sm, kin):
    far = kin.fk(kin.q_home)
    far[0, 3] += 2.0
    q, info = kin.ik(far, kin.q_home)
    assert not info["ok"] and info["err_p"] > 1.0 and np.all(np.isfinite(q))
    q, info = kin.ik_multistart(far, [kin.q_home, kin.q_lo, kin.q_hi])
    assert not info["ok"]
    # exactly singular start (q2 = 0) towards a reachable pose: no exception, finite, within limits
    qs = find_singular(kin)
    assert np.linalg.svd(kin.jac(qs), compute_uv=False)[-1] < 1e-6
    T = kin.fk(np.clip(qs + np.array([0.1, 0.1, -0.1, 0.1, 0.2, 0.2, 0.3]), kin.q_lo, kin.q_hi))
    q, info = kin.ik(T, qs, restarts=0)
    assert np.all(np.isfinite(q)) and np.all(q >= kin.q_lo) and np.all(q <= kin.q_hi)
    # stretched arm reaching further out along the singular (radial) direction: cannot be reached
    q_str = np.array([0, 0, 0, kin.q_hi[3], 0, 0, 0])
    T = kin.fk(q_str)
    T[:3, 3] += 0.3 * T[:3, 3] / np.linalg.norm(T[:3, 3])
    q, info = kin.ik(T, q_str)
    assert not info["ok"] and np.all(np.isfinite(q))
    nan = kin.fk(kin.q_home)
    nan[0, 3] = np.nan
    q, info = kin.ik(nan, kin.q_home)
    assert not info["ok"]


def test_ik_warm_start_timing(sm, kin):
    ts = np.linspace(0, 1, 1000)[:, None]
    amp = 0.4 * np.array([1, .5, 1, .5, 1, .5, 1])
    qa = kin.q_home + amp * np.sin(2 * np.pi * ts * np.array([1, 2, 1, 2, 1, 2, 1]))
    Ts = [kin.fk(q) for q in qa]
    q, ok, t0 = qa[0], 0, time.perf_counter()
    for T in Ts:
        q, info = kin.ik(T, q)
        ok += info["ok"]
    ms = (time.perf_counter() - t0) / len(Ts) * 1e3
    print(f"\n[{sm.robot}] warm-started IK: {ms:.3f} ms/frame, {ok}/1000 ok")
    assert ok == 1000 and ms <= 3.0


def test_nullspace_pulls_towards_rest(kin):
    rng = np.random.default_rng(13)
    q_true = kin.q_home + rng.uniform(-0.3, 0.3, 7)
    T = kin.fk(q_true)
    q_far = rand_q(kin, rng)
    q1, i1 = kin.ik(T, q_far, w_null=0.0, restarts=0)
    q2, i2 = kin.ik(T, q_far, w_null=0.3, q_rest=kin.q_home, restarts=0)
    if i1["ok"] and i2["ok"]:
        assert np.linalg.norm(q2 - kin.q_home) <= np.linalg.norm(q1 - kin.q_home) + 1e-9


# ------------------------------------------------------------------------------------------ collisions


def test_contacts_home_clear_and_fold_collides(sm, kin):
    assert kin.contacts(kin.q_home) == []
    assert kin.n_contacts(kin.q_home) == 0
    d = kin.min_distance(kin.q_home, [("tooling", "link5"), ("tooling", "link6"), ("tooling", "arm"), ("hand", "arm")])
    assert d > 0.0
    d_tab = kin.min_distance(kin.q_home, [("arm", "table"), ("tooling", "table"), ("hand", "table")])
    assert d_tab > 0.0
    # arm folded back on itself: the elbow and wrist come down on the base (self group)
    q_fold = np.array([0.0, 1.75, 0.0, -3.0, 0.0, 3.7, 0.0])
    q_fold = np.clip(q_fold, kin.q_lo, kin.q_hi)
    selfc = kin.contacts(q_fold, groups=("self",))
    assert len(selfc) > 0 and all(c["group"] == "self" and c["dist"] < 0 for c in selfc)
    # the same configuration is reported by n_contacts and sorted by distance
    allc = kin.contacts(q_fold)
    assert kin.n_contacts(q_fold) == len(allc) >= len(selfc)
    assert [c["dist"] for c in allc] == sorted(c["dist"] for c in allc)
    # margin widens the net
    assert len(kin.contacts(kin.q_home, margin=0.5)) > 0


def test_table_and_tooling_groups(sm, kin):
    q_down = q_into_table(sm, kin)
    c = kin.contacts(q_down, groups=("table",))
    assert c and all(x["group"] == "table" and "table" in (x["a"], x["b"]) for x in c)
    # group definitions
    g = kin.groups
    names = lambda ids: {kin.m.geom(i).name for i in ids}  # noqa: E731
    assert {"housing", "cradle_back_plate"} <= names(g["tooling"]) and names(g["pads"]) == {"pad_l", "pad_r"}
    eraser, table = set(g["eraser"]), set(g["table"])
    pad = set(g["pads"])
    for grp, pairs in kin._pairs.items():
        flat = {i for p in pairs for i in p}
        assert not (flat & eraser and flat & table and grp != "table")
        for a, b in pairs:
            assert {a, b} != {next(iter(eraser)), next(iter(table))}                     # eraser vs table excluded
            if grp == "object":
                assert not ({a, b} & pad)                                                 # pads vs eraser excluded
            assert not (grp != "object" and {a, b} & eraser)
            assert kin.m.geom_bodyid[a] != kin.m.geom_bodyid[b]
    links = [set(g[f"link{i}"]) for i in range(8)]
    assert not any(p[0] in links[0] or p[1] in links[0] for p in kin._pairs["table"])    # base stands on the table
    # tooling is checked against the moving links, not the rigidly attached link 7 / hand
    flat = {i for p in kin._pairs["tooling"] for i in p} - set(g["tooling"])
    assert flat and not (flat & links[7]) and not (flat & set(g["hand"]))
    with pytest.raises(KeyError):
        kin.min_distance(kin.q_home, [("tooling", "nonsense")])


def test_kin_has_private_data(sm, kin):
    k2 = K.Kin(sm, "tcp")
    assert k2.d is not kin.d
    before = kin.d.qpos.copy()
    k2.fk(k2.q_lo)
    assert np.array_equal(kin.d.qpos, before)


# ------------------------------------------------------------------------------------------- check_traj


def test_check_traj_reachable_line(sm, kin):
    k = K.Kin(sm, "ee")
    T0 = k.fk(kin.q_home)
    T = np.repeat(T0[None], 40, axis=0)
    T[:, 0, 3] += np.linspace(0, 0.1, 40)
    T[:, 2, 3] -= np.linspace(0, 0.05, 40)
    out = K.check_traj(sm, T, kin.q_home, frame="ee")
    assert out["reachable"].all() and out["q"].shape == (40, 7)
    assert out["joint_margin_rad"].shape == (40, 7) and out["collisions"].shape == (40,) and out["manip"].shape == (40,)
    assert (out["joint_margin_rad"].min(axis=1) >= 0.02 - 1e-9).all() and (out["manip"] > 0).all()
    assert out["err_p"].max() <= 1e-3 and out["err_r"].max() <= np.deg2rad(0.5)
    assert np.allclose(out["joint_margin_deg"], np.rad2deg(out["joint_margin_rad"]))
    assert (out["collisions"] == 0).all()
    out2 = K.check_traj(sm, T, kin.q_home, frame="ee", collision=False)
    assert (out2["collisions"] == 0).all() and np.allclose(out2["q"], out["q"])


def test_check_traj_flags_workspace_exit(sm, kin):
    k = K.Kin(sm, "ee")
    T0 = k.fk(kin.q_home)
    T = np.repeat(T0[None], 25, axis=0)
    T[:, 0, 3] += np.linspace(0, 0.05, 25)
    T[10:15, 0, 3] += 2.0                                         # 2 m away: outside the workspace
    out = K.check_traj(sm, T, kin.q_home, frame="ee")
    assert list(np.flatnonzero(~out["reachable"])) == [10, 11, 12, 13, 14]
    assert out["err_p"][10:15].min() > 0.5 and out["err_p"][np.r_[0:10, 15:25]].max() <= 1e-3


def test_check_traj_counts_collisions(sm, kin):
    k = K.Kin(sm, "tcp")
    q_down = q_into_table(sm, kin)
    T = np.stack([k.fk(kin.q_home), k.fk(q_down)])
    out = K.check_traj(sm, T, kin.q_home, frame="tcp")
    assert out["reachable"].all() and out["collisions"][0] == 0 and out["collisions"][1] > 0


@pytest.mark.parametrize("robot", ROBOTS)
def test_tooling_group_sees_link6_with_minus_y(robot):
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = robot
    sm = M.build_model(load_station(spec, sets=["tooling.tcp_x_along_hand=-y"]))
    k = K.Kin(sm, "flange")
    link6 = {k._gname(i) for i in k.groups["link6"]}
    c = k.contacts(k.q_home, groups=("tooling",))
    assert c and any(x["a"] in link6 or x["b"] in link6 for x in c)
    assert k.min_distance(k.q_home, [("tooling", "link6")]) < 0
    rep = M.clearance_report(sm)
    assert rep["per_link"]["link6"]["min_m"] < 0 or rep["at_q_base_m"] < 0
