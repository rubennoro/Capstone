"""A5 HandSim on a deterministic SimCore, both arms x both tooling sets (the station's tooling and the +10 cm / -y
candidate): move accuracy/speed, 45 mm box grasp (hold, pad force, drop, 5 cm lift with the arm running), grasp with
no object, blocked move, stop, the 20 mm eraser vs eps_inner, and the MuJoCoBackend hand topics.

    cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_hand_sim.py -s
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_deploy.gripper.hand_logic import HandPolicyConfig, WidthMap  # noqa: E402
from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import model as M  # noqa: E402
from franka_sim.hand_sim import HandSim  # noqa: E402
from franka_sim.simcore import SimCore, make_core_cfg  # noqa: E402

ROBOTS = ("panda", "fr3")
TOOLINGS = {"station": [], "ext10_my": ["tooling.finger_adapter.extension_m=0.1", "tooling.tcp_x_along_hand=-y"]}
BOX = ["scene.eraser.size_m=[0.045,0.045,0.03]"]
RESULTS: dict = {}


def station(robot, sets=()):
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = robot
    return load_station(spec, sets=list(sets))


@pytest.fixture(scope="module", params=[(r, t) for r in ROBOTS for t in TOOLINGS], ids=lambda p: f"{p[0]}-{p[1]}")
def box_arm(request):
    robot, tool = request.param
    st = station(robot, BOX + TOOLINGS[tool])
    return request.param, st, M.build_model(st)


class Rig:
    """SimCore + HandSim + helpers around the eraser body."""

    def __init__(self, st, sm):
        self.st, self.sm, self.m = st, sm, sm.model
        self.c = SimCore(sm, make_core_cfg(st))
        self.h = HandSim(self.c, st.get("tooling"))
        self.c.hand_ctrl = self.h.ctrl
        self.wm = WidthMap.from_config(st.get("tooling.width_map"))
        self.hand = st.get("tooling.hand")
        m = self.m
        nid = lambda kind, n: mujoco.mj_name2id(m, kind, n)  # noqa: E731
        self.eb = nid(mujoco.mjtObj.mjOBJ_BODY, "eraser")
        ej = nid(mujoco.mjtObj.mjOBJ_JOINT, "eraser_free")
        self.qa, self.va = m.jnt_qposadr[ej], m.jnt_dofadr[ej]
        self.site = nid(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.pads = (nid(mujoco.mjtObj.mjOBJ_GEOM, "pad_l"), nid(mujoco.mjtObj.mjOBJ_GEOM, "pad_r"))
        self.ge = nid(mujoco.mjtObj.mjOBJ_GEOM, "eraser")

    def box_at_tcp(self, yaw90=False):
        """Eraser at the tcp site, axes = TCP axes (yaw90: its y along TCP x), held up by xfrc_applied = m g."""
        d = self.c.d
        R = d.site_xmat[self.site].reshape(3, 3).copy()
        if yaw90:
            R = R @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        d.qpos[self.qa:self.qa + 3] = d.site_xpos[self.site]
        d.qpos[self.qa + 3:self.qa + 7] = q
        d.qvel[self.va:self.va + 6] = 0
        mujoco.mj_forward(self.m, d)
        self.c._update_kin()
        d.xfrc_applied[self.eb, 2] = self.m.body_subtreemass[self.eb] * -self.m.opt.gravity[2]

    def open_full(self):
        """Home gap is WidthMap.gap(home_width_m) (~0.037): open to the full stroke before placing a 45 mm box."""
        self.cmd("move", self.wm.gap_max_m, seq=99)
        assert self.run_until(lambda s: s["seq_done"] == 99 and done(s), 2000)[0] is not None

    def box_gap(self, size_m):
        """The finger gap at which the pads touch a box of this size (pad separation = gap + pad offset)."""
        return size_m - self.sm.limits["pad_offset_m"]

    def release_box(self):
        self.c.d.xfrc_applied[self.eb] = 0

    def pad_forces(self):
        d, f = self.c.d, np.zeros(6)
        out = [0.0, 0.0]
        for i in range(d.ncon):
            g = set(int(x) for x in d.contact[i].geom)
            for k, gp in enumerate(self.pads):
                if g == {gp, self.ge}:
                    mujoco.mj_contactForce(self.m, d, i, f)
                    out[k] += f[0]
        return out

    def cmd(self, kind, width=0.0, **kw):
        msg = {"kind": kind, "width_m": width, "speed_m_s": self.hand["speed_m_s"], "force_n": self.hand["force_n"],
               "eps_inner_m": self.hand["grasp_epsilon_inner_m"], "eps_outer_m": self.hand["grasp_epsilon_outer_m"],
               "seq": kw.pop("seq", 1)}
        msg.update(kw)
        assert self.h.command(msg)["accepted"]

    def run_until(self, pred, max_steps=3000):
        vmax = 0.0
        for k in range(max_steps):
            self.c.step(1)
            vmax = max(vmax, abs(self.h.gap_vel(self.c)))
            if pred(self.h.state(0.0)):
                return k + 1, vmax
        return None, vmax

    def gap(self):
        return self.h.gap(self.c)


def done(s):
    return s["last_result"] and not s["busy"]


def test_move_accuracy_and_speed(box_arm):
    key, st, sm = box_arm
    r = Rig(st, sm)
    v = r.hand["speed_m_s"]
    out = {}
    for seq, target in enumerate((0.03, 0.08), 1):
        r.cmd("move", target, seq=seq)
        n, vmax = r.run_until(lambda s, q=seq: s["seq_done"] == q and done(s))
        s = r.h.state(0.0)
        assert n is not None and s["last_result"] == "move_ok", s
        assert abs(r.gap() - target) < 0.001
        assert vmax <= 1.1 * v, vmax
        out[target] = {"err_mm": (r.gap() - target) * 1e3, "vmax_m_s": vmax, "t_s": n * 1e-3}
    RESULTS.setdefault(key, {})["move"] = out


def test_grasp_45mm_box_hold_force_drop_and_lift(box_arm):
    key, st, sm = box_arm
    r = Rig(st, sm)
    c, d = r.c, r.c.d
    c.start()                                                    # arm running: reflexes armed
    r.open_full()
    r.box_at_tcp()
    r.cmd("grasp", r.box_gap(0.045))
    n, _ = r.run_until(lambda s: s["mode"] == "holding", 2000)
    assert n is not None, r.h.state(0.0)
    c.step(200)
    s = r.h.state(0.0)
    assert s["mode"] == "holding" and s["grasped"] and s["last_result"] == "grasp_ok", s
    f = r.pad_forces()
    F = r.hand["force_n"]
    assert all(abs(x - F) <= 0.2 * F for x in f), f
    r.release_box()
    z0 = d.xpos[r.eb, 2]
    c.step(2000)
    drop = z0 - d.xpos[r.eb, 2]
    assert drop <= 0.002 and r.h.state(0.0)["grasped"], drop
    # 5 cm lift: quintic over 1 s (a step would trip the 20 N cartesian reflex at K 800), then settle 1.5 s
    T0 = c.state()["O_T_EE"].copy()
    p_tcp0, p_box0 = d.site_xpos[r.site].copy(), d.xpos[r.eb].copy()
    for k in range(1001):
        s_ = k / 1000
        T = T0.copy()
        T[2, 3] += 0.05 * (10 * s_ ** 3 - 15 * s_ ** 4 + 6 * s_ ** 5)
        c.set_target(T)
        c.step(1)
    c.step(1500)
    d_tcp, d_box = d.site_xpos[r.site] - p_tcp0, d.xpos[r.eb] - p_box0
    follow = float(np.linalg.norm(d_box - d_tcp))
    assert abs(d_tcp[2] - 0.05) < 0.003 and follow <= 0.003, (d_tcp, d_box)
    assert c.error is None and c.mode == "running", c.error
    assert r.h.state(0.0)["grasped"]
    RESULTS.setdefault(key, {})["grasp"] = {"t_contact_s": n * 1e-3, "gap_m": r.gap(), "pad_force_n": f,
                                            "drop_mm": drop * 1e3, "lift_follow_mm": follow * 1e3}


def test_grasp_no_object_fails(box_arm):
    key, st, sm = box_arm
    r = Rig(st, sm)
    r.cmd("grasp", 0.005, eps_inner_m=0.005)                      # target max(0, w - eps_i) = 0
    n, _ = r.run_until(lambda s: done(s), 3000)
    s = r.h.state(0.0)
    assert n is not None and s["last_result"] == "grasp_failed" and not s["grasped"] and s["mode"] == "idle", s
    assert r.gap() < 0.001


def test_blocked_move_fails_fast_and_stop_holds(box_arm):
    key, st, sm = box_arm
    r = Rig(st, sm)
    r.open_full()
    r.box_at_tcp()
    r.cmd("move", 0.02)
    n_contact, _ = r.run_until(lambda s: sum(r.pad_forces()) > 0.5, 2000)
    assert n_contact is not None
    n_fail, _ = r.run_until(lambda s: done(s), 1000)
    s = r.h.state(0.0)
    assert s["last_result"] == "move_failed" and n_fail is not None and n_fail * 1e-3 <= 0.3, (s, n_fail)
    g_hold = r.gap()
    r.c.step(500)
    assert abs(r.gap() - g_hold) < 0.001                         # holds where it is (no squeeze build-up)
    # stop mid-move holds the current gap
    r.release_box()
    r.cmd("move", 0.08, seq=2)
    r.c.step(150)
    r.cmd("stop", seq=3)
    r.c.step(2)
    g0 = r.gap()
    r.c.step(500)
    s = r.h.state(0.0)
    assert s["mode"] == "stopped" and s["last_result"] == "stopped" and not s["busy"], s
    assert abs(r.gap() - g0) < 0.0015
    RESULTS.setdefault(key, {})["blocked_move_fail_s"] = n_fail * 1e-3


@pytest.mark.parametrize("robot", ROBOTS)
def test_eraser_20mm_station_eps_inner(robot):
    """Open question 7 (after the A5 review width map b = 0.0431): the policy grasps at w 0.061-0.064 (gap
    0.018-0.021) and the 20 mm eraser side stops the pads at gap 0.012. The station eps_inner grasps it (held width
    train(gap) inside the training range); a too-small eps (0.005) stops the fingers before they touch it."""
    st = station(robot)
    sm = M.build_model(st)
    eps_st = float(st.get("tooling.hand.grasp_epsilon_inner_m"))
    res = {}
    for w in (0.061, 0.064):
        for eps in (eps_st, 0.005):
            r = Rig(st, sm)
            r.box_at_tcp(yaw90=True)                              # the 20 mm side along the closing axis
            r.cmd("grasp", r.wm.gap(w), eps_inner_m=eps)
            r.run_until(lambda s: done(s) or s["mode"] == "holding", 2500)
            r.c.step(100)
            res[(w, eps)] = r.h.state(0.0)
        s = res[(w, eps_st)]
        assert s["mode"] == "holding" and s["grasped"] and abs(s["gap_m"] - r.box_gap(0.020)) < 0.001, s
        held = r.wm.train(s["gap_m"])
        assert st.get("tooling.width_map.train_min_m") <= held <= st.get("tooling.width_map.train_max_m"), held
        s = res[(w, 0.005)]
        assert s["last_result"] == "grasp_failed" and not s["grasped"], s
    assert eps_st == HandPolicyConfig().eps_inner_m                 # hand_logic default in sync with the station
    RESULTS.setdefault((robot, "eraser20"), {})["held_train_w"] = held


def test_backend_hand_topics_and_state_keys():
    from franka_deploy.bridge.backends import FakeBackend
    from franka_sim.backend import MuJoCoBackend

    home = np.array([[1.0, 0, 0, 0.45], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.45], [0, 0, 0, 1]])
    b = MuJoCoBackend.from_config({"type": "mujoco", "mujoco": {"station": "sim-iphone"}}, home)
    try:
        assert set(b.extra_topics) >= {"hand_cmd", "hand_state"} and b.core.hand_ctrl == b.hand.ctrl
        fk = FakeBackend(home)
        assert set(b.get_state()) == set(fk.get_state()) | {"sim_t"}     # A4 key set unchanged
        fk.close()
        s0 = b.handle_extra("hand_state", None)
        assert set(s0) == {"gap_m", "gap_vel_m_s", "busy", "mode", "grasped", "last_result", "seq_done", "t"}
        st_ = station("panda")
        home_gap = WidthMap.from_config(st_.get("tooling.width_map")).gap(st_.get("tooling.hand.home_width_m"))
        assert abs(s0["gap_m"] - home_gap) < 0.002, (s0, home_gap)          # home key gap = gap(home_width_m)
        t0 = time.perf_counter()
        rep = b.handle_extra("hand_cmd", {"kind": "move", "width_m": 0.04, "speed_m_s": 0.1, "seq": 5})
        assert rep == {"accepted": True, "seq": 5} and time.perf_counter() - t0 < 0.005
        assert b.handle_extra("hand_cmd", {"kind": "squeeze", "seq": 6})["accepted"] is False
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            s = b.handle_extra("hand_state", None)
            if s["seq_done"] == 5 and not s["busy"]:
                break
            time.sleep(0.02)
        assert s["last_result"] == "move_ok" and abs(s["gap_m"] - 0.04) < 0.001, s
        with pytest.raises(ValueError, match="unknown request"):
            b.handle_extra("sim_nope", None)
    finally:
        b.close()


def test_zz_report():
    print("\nA5 HandSim results " + json.dumps({f"{k[0]}-{k[1]}": v for k, v in RESULTS.items()}, default=float))
