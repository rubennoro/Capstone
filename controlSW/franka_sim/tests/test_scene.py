"""A6 scene runtime, in process (no bridge): both arms x both tooling sets (the station's and +10 cm / -y).

* MuJoCoBackend: the sim-state PUB received by judge.StateSubscriber at >= 100 Hz with every I-SIMSTATE key; world =
  robot base; the eraser at rest; sim_reset {} and a pose + hand gap (< 50 ms, arm untouched); refused while grasped;
  a pose overlapping the robot / the hand refused with the state restored; loop p99 with the PUB on.
* Scripted pick-and-lift on SimCore + HandSim (deterministic, no threads): 10 seeds per arm x tooling, sim lines at
  25 Hz -> ground_truth.evaluate -> success.

    cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_scene.py -s
Ports: 5570 + 1000 + FD_TEST_PORT_OFFSET + k (Track A).
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("zmq")
pytest.importorskip("robotmq")

from franka_deploy.gripper.hand_logic import WidthMap  # noqa: E402
from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import ground_truth as GT  # noqa: E402
from franka_sim import model as M  # noqa: E402
from franka_sim import scene as S  # noqa: E402
from franka_sim.hand_sim import HandSim  # noqa: E402
from franka_sim.simcore import SimCore, make_core_cfg  # noqa: E402

ROBOTS = ("panda", "fr3")
TOOLINGS = {"station": [], "ext10_my": ["tooling.finger_adapter.extension_m=0.1", "tooling.tcp_x_along_hand=-y"]}
CONFIGS = [(r, t) for r in ROBOTS for t in TOOLINGS]
OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
KEYS = {"t_mono", "qpos", "qvel", "eraser_T_base", "hand_gap_m", "grasped", "contacts", "reflex", "tau_ext"}
RESULTS: dict = {}


def station(robot, sets=()):
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = robot
    return load_station(spec, sets=list(sets))


def Rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


# --------------------------------------------------------------------------------------- MuJoCoBackend in process
@pytest.fixture(scope="module", params=CONFIGS, ids=lambda p: f"{p[0]}-{p[1]}")
def backend(request, tmp_path_factory):
    from franka_sim.backend import MuJoCoBackend

    robot, tool = request.param
    st = station(robot, TOOLINGS[tool])
    p = tmp_path_factory.mktemp("st") / "station_resolved.yaml"
    p.write_text(st.to_yaml())
    ep = f"tcp://127.0.0.1:{5570 + 1000 + OFFSET + 10 + CONFIGS.index(request.param)}"
    home = np.array([[1.0, 0, 0, 0.45], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.45], [0, 0, 0, 1]])
    b = MuJoCoBackend.from_config({"type": "mujoco", "mujoco": {"station": str(p), "state_endpoint": ep}}, home)
    yield request.param, st, b, ep
    b.close()


def test_pub_rate_keys_world_and_rest(backend):
    from franka_sim.eval.judge import StateSubscriber

    key, st, b, ep = backend
    sub = StateSubscriber(ep)
    try:
        time.sleep(0.3)
        b.loop_stats()
        ts, s = [], None
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            x = sub.recv(0.1)
            if x is not None:
                s = x
                ts.append(x["t_mono"])
        stats = b.loop_stats()
    finally:
        sub.close()
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    m = b.sm.model
    assert KEYS <= set(s) and {"hand_mode", "hand_grasped", "sim_t"} <= set(s)
    assert np.shape(s["qpos"]) == (m.nq,) and np.shape(s["qvel"]) == (m.nv,) and np.shape(s["tau_ext"]) == (7,)
    assert np.shape(s["eraser_T_base"]) == (4, 4) and isinstance(s["grasped"], bool) and s["reflex"] is None
    assert isinstance(s["hand_gap_m"], float) and all(set(c) == {"a", "b", "force_n"} for c in s["contacts"])
    assert rate >= 99.0 and len(ts) >= 297, (rate, len(ts))           # 100 Hz sim-time cadence, wall-clock jitter 1 %
    # world = robot base: the arm's root body sits at the world origin, unrotated
    d = b.core.d
    root = int(m.jnt_bodyid[m.joint(b.sm.names["joints"]["j1"]).id])
    while m.body_parentid[root] != 0:
        root = int(m.body_parentid[root])
    assert np.allclose(d.xpos[root], 0.0, atol=1e-9) and np.allclose(d.xmat[root], np.eye(3).ravel(), atol=1e-9)
    sc = b.scene
    T = s["eraser_T_base"]
    assert abs(T[2, 3] - (float(st.get("scene.table.top_z_base_m")) + sc.size[2] / 2)) < 0.001, T
    assert any({c["a"], c["b"]} == {"table", "eraser"} for c in s["contacts"]), s["contacts"]
    assert stats["p99_ms"] <= 2.0, stats
    RESULTS.setdefault(key, {}).update(pub_hz=round(rate, 2), loop_p99_ms=round(stats["p99_ms"], 3),
                                       rest_contacts=s["contacts"])


def _reset(b, data):
    q0 = b.get_state()["q"]
    t0 = time.perf_counter()
    rep = b.handle_extra("sim_reset", data)
    dt = time.perf_counter() - t0
    return rep, dt, q0


def test_sim_reset_moves_eraser_and_gap_never_the_arm(backend):
    key, st, b, ep = backend
    sc = b.scene
    lat = []
    for data, want, gap in (({}, (sc.home_xy[0], sc.home_xy[1], 0.0), None),
                            ({"eraser_xy_yaw": [0.5, 0.05, 0.3], "hand_gap_m": 0.06}, (0.5, 0.05, 0.3), 0.06)):
        rep, dt, q0 = _reset(b, data)
        lat.append(dt)
        assert rep["ok"] and dt < 0.05, (rep, dt)
        assert rep["hand_gap_m"] == gap and set(rep) == {"ok", "eraser_xy_yaw", "hand_gap_m", "t_mono"}
        s0 = b.sim_state()
        assert abs(s0["t_mono"] - rep["t_mono"]) < 1e-9                  # the snapshot is refreshed by the reset
        time.sleep(0.4)
        s = b.handle_extra("sim_state", None)
        T = s["eraser_T_base"]
        assert np.linalg.norm(T[:2, 3] - want[:2]) < 0.001, T
        assert abs(math.degrees(math.atan2(T[1, 0], T[0, 0]) - want[2])) < 0.5
        assert abs(T[2, 3] - sc.z_rest) < 0.001
        if gap is not None:
            assert abs(s["hand_gap_m"] - gap) < 0.001 and b.hand.last_result == "reset" and b.hand.mode == "idle"
        assert np.abs(b.get_state()["q"] - q0).max() < 1e-4
    for _ in range(48):
        lat.append(_reset(b, {"eraser_xy_yaw": [0.45, 0.0, 0.0]})[1])
    p95 = float(np.percentile(lat, 95))
    assert p95 < 0.05
    RESULTS.setdefault(key, {})["reset_ms_p95"] = round(p95 * 1e3, 3)


def test_sim_reset_refused_when_overlapping_and_state_restored(backend):
    key, st, b, ep = backend
    b.handle_extra("sim_reset", {})
    time.sleep(0.2)
    s0 = b.sim_state()
    q0 = b.get_state()["q"]
    with pytest.raises(RuntimeError, match="refused: the eraser would penetrate .*link0"):
        b.handle_extra("sim_reset", {"eraser_xy_yaw": [0.0, 0.0, 0.0]})            # inside the robot base
    with b._core_lock:
        d = b.core.d
        site = b.sm.model.site("tcp").id
        p = d.site_xpos[site].copy()
        p = p + d.site_xmat[site].reshape(3, 3)[:, 0] * 0.5 * (b.hand.gap(b.core) + b.sm.limits["pad_offset_m"])  # a pad
    with pytest.raises(RuntimeError, match="refused: the eraser would penetrate .*pad_"):
        b.handle_extra("sim_reset", {"object_pose": [*p, 1.0, 0.0, 0.0, 0.0]})
    s1 = b.sim_state()
    assert np.linalg.norm(s1["eraser_T_base"][:3, 3] - s0["eraser_T_base"][:3, 3]) < 0.001
    assert np.abs(b.get_state()["q"] - q0).max() < 1e-4
    for bad in ({"eraser_xy_yaw": [0.5, 0.0]}, {"hand_gap_m": 0.5}, {"eraser_xy_yaw": [float("nan"), 0, 0]}, [1, 2]):
        with pytest.raises(ValueError):
            b.handle_extra("sim_reset", bad)


def test_sim_reset_refused_while_grasped(backend):
    key, st, b, ep = backend
    hand = st.get("tooling.hand")
    wm = WidthMap.from_config(st.get("tooling.width_map"))
    b.handle_extra("hand_cmd", {"kind": "move", "width_m": 0.05, "speed_m_s": 0.1, "seq": 11})
    t_end = time.monotonic() + 3
    while time.monotonic() < t_end and b.handle_extra("hand_state", None)["seq_done"] != 11:
        time.sleep(0.02)
    with b._core_lock:                      # the eraser between the pads, its 20 mm side on the closing axis (TCP x)
        d = b.core.d
        site = b.sm.model.site("tcp").id
        R = d.site_xmat[site].reshape(3, 3) @ Rz(math.pi / 2)
        p = d.site_xpos[site].copy()
    rep = b.handle_extra("sim_reset", {"object_pose": [*p, *S.R_to_quat(R)]})
    assert rep["ok"]
    with b._core_lock:
        b.core.d.xfrc_applied[b.scene.body, 2] = b.sm.model.body_subtreemass[b.scene.body] * 9.81
    try:
        b.handle_extra("hand_cmd", {"kind": "grasp", "width_m": wm.gap(0.062), "speed_m_s": hand["speed_m_s"], "force_n": 30.0,
                                    "eps_inner_m": hand["grasp_epsilon_inner_m"], "eps_outer_m": hand["grasp_epsilon_outer_m"],
                                    "seq": 12})
        t_end = time.monotonic() + 3
        while time.monotonic() < t_end and b.handle_extra("hand_state", None)["mode"] != "holding":
            time.sleep(0.02)
        time.sleep(0.2)
        s = b.sim_state()
        assert b.hand.mode == "holding" and s["grasped"] and s["hand_grasped"], (b.hand.mode, s["contacts"])
        names = sorted({n for c in s["contacts"] for n in (c["a"], c["b"])})
        q0 = b.get_state()["q"]
        for data in ({}, {"eraser_xy_yaw": [0.5, 0.0, 0.0], "hand_gap_m": 0.08}):
            with pytest.raises(RuntimeError, match="sim_reset refused: grasping"):
                b.handle_extra("sim_reset", data)
        assert b.sim_state()["grasped"] and np.abs(b.get_state()["q"] - q0).max() < 1e-4
        RESULTS.setdefault(key, {})["grasp_contact_names"] = names
    finally:
        with b._core_lock:
            b.core.d.xfrc_applied[b.scene.body] = 0
        b.handle_extra("hand_cmd", {"kind": "move", "width_m": 0.06, "speed_m_s": 0.1, "seq": 13})
        time.sleep(0.6)
        b.handle_extra("sim_reset", {})


def test_state_pub_hz_zero_disables_and_bind_failure_names_endpoint(backend):
    from franka_sim.backend import MuJoCoBackend

    key, st, b, ep = backend
    if key != CONFIGS[0]:
        pytest.skip("once")
    b2 = MuJoCoBackend(b.sm, b.core.cfg, b.core.q, hand=True, state_endpoint=None, state_pub_hz=0)
    try:
        assert b2.scene is None and b2.sim_state() is None and b2.state_endpoint is None
        with pytest.raises(RuntimeError, match="no scene"):
            b2.handle_extra("sim_reset", {})
    finally:
        b2.close()
    with pytest.raises(RuntimeError, match=ep.replace(".", r"\.")):
        MuJoCoBackend(b.sm, b.core.cfg, b.core.q, state_endpoint=ep)          # already bound by the fixture


def test_sim_tick_writes_sim_lines_at_sim_log_hz(backend, tmp_path):
    key, st, b, ep = backend
    if key != CONFIGS[0]:
        pytest.skip("once")

    class Br:
        _jsonl = open(tmp_path / "bridge.jsonl", "w")

    t_end = time.monotonic() + 2.0
    while time.monotonic() < t_end:
        b.sim_tick(Br)
        time.sleep(0.005)
    Br._jsonl.close()
    lines = [json.loads(x) for x in open(tmp_path / "bridge.jsonl")]
    rate = (len(lines) - 1) / (lines[-1]["t"] - lines[0]["t"])
    assert 22 <= rate <= 28, rate
    ln = lines[-1]
    assert set(ln) == {"t", "kind", "obj", "gap", "grasped", "hand_mode", "contacts", "F_ext_max", "reflex"} and ln["kind"] == "sim"
    assert len(ln["obj"]) == 7 and isinstance(ln["gap"], float)
    b.sim_tick(type("NoLog", (), {"_jsonl": None}))           # no --log-jsonl: nothing to do


# --------------------------------------------------------------------------------------- scripted pick-and-lift
def quintic(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def pick_and_lift(st, sm, seed: int) -> dict:
    """Eraser at home xy +-1 cm, yaw +-20 deg; TCP 5 cm above it, closing axis (TCP x) across the 20 mm side; descend to
    the grasp height; grasp(gap(0.062), 30 N, station eps); settle 1.5 s; lift 6 cm in 1 s; hold 3 s."""
    from scipy.spatial.transform import Rotation, Slerp

    rng = np.random.default_rng(seed)
    c = SimCore(sm, make_core_cfg(st))
    h = HandSim(c, st.get("tooling"))
    c.hand_ctrl = h.ctrl
    sc = S.Scene(sm)
    probe = S.ContactProbe(sc)
    c.contact_probe = probe
    d, m = c.d, sm.model
    x, y = sc.home_xy[0] + rng.uniform(-0.01, 0.01), sc.home_xy[1] + rng.uniform(-0.01, 0.01)
    yaw = math.radians(rng.uniform(-20, 20))
    sc.set_eraser(d, x, y, yaw)
    mujoco.mj_forward(m, d)
    c._update_kin()
    c.start()
    site = m.site("tcp").id
    TCP_T_EE = np.linalg.inv(sm.EE_T_TCP)
    T0 = np.eye(4)
    T0[:3, :3], T0[:3, 3] = d.site_xmat[site].reshape(3, 3), d.site_xpos[site]
    ey = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    ang = math.atan2(np.cross(T0[:3, 0], ey)[2], T0[:3, 0] @ ey)
    ang = (ang + math.pi / 2) % math.pi - math.pi / 2                  # TCP x along +-eraser y: the smaller turn
    zg = sc.top_z + S.grasp_tcp_height_m(sm, st)
    lines, rel = [], []

    def step(n, T):
        for _ in range(n):
            c.set_target(T @ TCP_T_EE)
            due = (c.n_steps + 1) % 40 == 0                            # 25 Hz sim lines
            if due:
                probe.arm(1)
            c.step(1)
            if due:
                s = sc.sim_state(c, probe.contacts, float(d.time), h)
                lines.append(S.sim_line(s, float(np.max(np.abs(c.O_F_ext[:3])))))
                rel.append(float(d.xpos[sc.body, 2] - d.site_xpos[site, 2]))

    def move(Ta, Tb, dur):
        sl = Slerp([0, 1], Rotation.from_matrix([Ta[:3, :3], Tb[:3, :3]]))
        n = int(dur * 1000)
        for k in range(1, n + 1):
            a = quintic(k / n)
            T = np.eye(4)
            T[:3, 3], T[:3, :3] = Ta[:3, 3] + a * (Tb[:3, 3] - Ta[:3, 3]), sl(a).as_matrix()
            step(1, T)
        return Tb

    wm = WidthMap.from_config(st.get("tooling.width_map"))
    hand = st.get("tooling.hand")
    h.command({"kind": "move", "width_m": wm.gap(0.075), "speed_m_s": 0.1, "seq": 1})     # open past the 20 mm side
    Ta = np.eye(4)
    Ta[:3, :3], Ta[:3, 3] = Rz(ang) @ T0[:3, :3], [x, y, zg + 0.05]
    T = move(T0, Ta, 2.5)
    Tg = T.copy()
    Tg[2, 3] = zg
    T = move(T, Tg, 1.5)              # 1.0 s here trips the 20 N cartesian reflex (impedance overshoot, A4)
    step(300, T)
    h.command({"kind": "grasp", "width_m": wm.gap(0.062), "speed_m_s": hand["speed_m_s"], "force_n": 30.0,
               "eps_inner_m": hand["grasp_epsilon_inner_m"], "eps_outer_m": hand["grasp_epsilon_outer_m"], "seq": 2})
    step(1500, T)
    i_lift = len(rel)
    Tl = T.copy()
    Tl[2, 3] += 0.06
    T = move(T, Tl, 1.0)
    step(3000, T)
    r = GT.evaluate(lines, st.get("criteria"), 0.0)
    r["in_hand_slip_m"] = round(max(0.0, rel[i_lift - 1] - min(rel[i_lift:])), 5)   # eraser z rel. to the TCP: true slip
    r["reflex_core"] = c.error
    r["last_result"] = h.state(0.0)["last_result"]
    return r


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_scripted_pick_and_lift_9_of_10(cfg):
    robot, tool = cfg
    st = station(robot, TOOLINGS[tool])
    sm = M.build_model(st)
    rows = []
    for seed in range(10):
        r = pick_and_lift(st, sm, seed)
        rows.append({"seed": seed, "success": r["success"], "reason": r["reason"], "grasp_t_s": r["grasp_t_s"],
                     "t_success_s": r["t_success_s"], "max_lift_m": r["max_lift_m"], "slip_m": r["slip_m"],
                     "in_hand_slip_m": r["in_hand_slip_m"], "reflex": r["reflex_core"], "last_result": r["last_result"],
                     "arm_contacts": r["n_arm_table_contacts"], "pairs": r["contact_pairs"]})
    ok = sum(bool(x["success"]) for x in rows)
    RESULTS.setdefault(cfg, {})["pick"] = rows
    assert ok >= 9, json.dumps(rows, indent=1)
    assert all(x["reflex"] is None and x["arm_contacts"] == 0 for x in rows), rows


def test_zz_report():
    out = {}
    for k, v in RESULTS.items():
        v = dict(v)
        if "pick" in v:
            p = v.pop("pick")
            v["pick_ok"] = sum(bool(x["success"]) for x in p)
            v["pick_rows"] = [[x["seed"], x["success"], x["grasp_t_s"], x["max_lift_m"], x["slip_m"], x["in_hand_slip_m"]] for x in p]
            v["pick_pairs"] = sorted({tuple(pp) for x in p for pp in x["pairs"]})
        out[f"{k[0]}-{k[1]}"] = v
    print("\nA6 scene results " + json.dumps(out, default=str))
