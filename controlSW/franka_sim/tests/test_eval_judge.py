"""Stage 3 judge (B9): the pure success rule, the contact classifier, the Judge thread on a StubSim, BridgeSim's payload.
cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_eval_judge.py tests/test_eval_sweeps.py tests/test_eval_criteria.py"""

import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from franka_sim.eval import judge as J
from franka_sim.eval import simlink as SL

CRIT = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "run" / "criteria" / "block_raise.yaml").read_text())
SUCCESS = CRIT["success"]


def _T(z, tilt_deg=0.0):
    T = np.eye(4)
    a = np.radians(tilt_deg)
    T[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    T[2, 3] = z
    return T


def _st(t, z, grasped=True, tilt=0.0, reflex=None):
    return {"t_mono": t, "eraser_T_base": _T(z, tilt), "grasped": grasped, "reflex": reflex, "contacts": []}


def _run(tracker, profile, dt=0.01, t_end=70.0):
    """profile(t) -> (z, grasped); returns (label, t_label)."""
    t = 0.0
    while t <= t_end:
        z, g = profile(t)
        lab = tracker.update(_st(t, z, g))
        if lab:
            return lab, t
        t += dt
    return None, None


def test_lift_held_gives_y_after_hold_s():
    tr = J.SuccessTracker(SUCCESS, 0.0)
    lab, t = _run(tr, lambda t: (0.02 + (0.05 if t >= 3.0 else 0.0), t >= 1.0))
    assert lab == "y"
    assert t == pytest.approx(3.0 + SUCCESS["hold_s"], abs=0.02)
    r = tr.result()
    assert r["success"] is True and r["t_success_s"] == pytest.approx(5.0, abs=0.02) and r["grasp_t_s"] == pytest.approx(1.0, abs=0.02)
    assert r["max_lift_m"] == pytest.approx(0.05) and r["reason"] == "lift_held"
    assert tr.update(_st(99.0, 0.0)) is None             # decided once


def test_short_lift_times_out_with_n():
    tr = J.SuccessTracker(SUCCESS, 0.0)
    lab, t = _run(tr, lambda t: (0.02 + (0.049 if t >= 3.0 else 0.0), True))
    assert lab == "n" and t == pytest.approx(SUCCESS["timeout_s"], abs=0.02) and tr.result()["reason"] == "timeout"


def test_threshold_is_a_config_value_not_a_constant():
    s = dict(SUCCESS, lift_min_m=0.03)                    # what --set criteria.success.lift_min_m=0.03 does
    tr = J.SuccessTracker(s, 0.0)
    lab, _ = _run(tr, lambda t: (0.02 + (0.049 if t >= 3.0 else 0.0), True))
    assert lab == "y"
    tr2 = J.SuccessTracker(dict(SUCCESS, timeout_s=5.0), 0.0)
    assert _run(tr2, lambda t: (0.02, False))[1] == pytest.approx(5.0, abs=0.02)


def test_lift_then_drop_is_no_success_and_slip_is_reported():
    tr = J.SuccessTracker(SUCCESS, 0.0)
    lab, t = _run(tr, lambda t: (0.02 + (0.06 if 2.0 <= t < 3.0 else 0.0), 1.0 <= t < 3.5))
    assert lab == "n" and tr.result()["success"] is False and tr.result()["reason"] == "timeout"
    assert tr.result()["slip_m"] == pytest.approx(0.06, abs=1e-6) and tr.result()["slip_exceeds_tol"] is True
    # a small slip under tolerance is reported, not an early stop
    tr = J.SuccessTracker(SUCCESS, 0.0)
    lab, _ = _run(tr, lambda t: (0.02 + (0.06 if t >= 1.0 else 0.0) - (0.001 if t > 3.0 else 0.0), True))
    assert lab == "y" and tr.result()["slip_m"] == pytest.approx(0.001, abs=1e-6) and tr.result()["slip_exceeds_tol"] is False


def test_knock_over_and_reflex_are_n_immediately():
    tr = J.SuccessTracker(SUCCESS, 0.0)
    assert tr.update(_st(1.0, 0.02, False, tilt=20)) is None
    assert tr.update(_st(1.1, 0.02, False, tilt=45)) == "n"
    assert tr.result()["knocked_over"] is True and tr.result()["reason"] == "knocked_over"
    tr = J.SuccessTracker(SUCCESS, 0.0)
    assert tr.update(_st(1.0, 0.02, False, reflex="cartesian_reflex")) == "n"
    assert tr.result()["reflex"] == "cartesian_reflex" and tr.result()["reason"] == "reflex"


def test_states_before_t_start_are_ignored_and_z0_comes_from_the_first_state():
    tr = J.SuccessTracker(SUCCESS, 10.0)
    assert tr.update(_st(9.0, 5.0)) is None and tr.z0 is None
    tr.update(_st(10.1, 0.02))
    assert tr.z0 == 0.02
    assert tr.check_timeout(10.0 + SUCCESS["timeout_s"]) == "n"        # a silent simulator still times out


def test_classify_contacts():
    pats, ign = ["pad_", "tag_", "finger"], [["eraser", "table"]]
    cs = [{"a": "pad_left", "b": "eraser_geom", "force_n": 10}, {"a": "eraser_geom", "b": "table_top", "force_n": 1},
          {"a": "finger_tip_r", "b": "table_top", "force_n": 3}, {"a": "link6_col", "b": "table_top", "force_n": 7},
          {"a": "hand_col", "b": "eraser_geom", "force_n": 2}]
    out = J.classify_contacts(cs, pats, ign)
    assert [(c["a"], c["b"]) for c in out] == [("link6_col", "table_top"), ("hand_col", "eraser_geom")]
    assert J.classify_contacts([], pats, ign) == [] and J.classify_contacts(None, pats, ign) == []
    assert J.classify_contacts([{"a": "table_top", "b": "eraser_x", "force_n": 1}], pats, ign) == []      # either order


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_judge_thread_on_stub_writes_the_label_once(tmp_path):
    ep = f"tcp://127.0.0.1:{_port()}"
    stub = SL.StubSim(ep, "lift_success")
    cmd = tmp_path / "commands.txt"
    try:
        stub.reset((0.45, 0.0, 0.0))
        t0 = time.monotonic()
        stub.on_episode_start(t0)
        j = J.Judge(ep, cmd, dict(SUCCESS, timeout_s=15.0), ["pad_", "tag_", "finger"], [["eraser", "table"]], t_start=t0)
        j.start()
        assert j.decided.wait(15)
        j.stop()
    finally:
        stub.close()
    assert cmd.read_text() == "y\n" and j.label == "y" and j.collisions == []
    assert j.tracker.result()["t_success_s"] == pytest.approx(5.0, abs=0.3)     # 3 s (6 cm reached) + hold 2 s
    assert j.grasped_at(t0 + 2.0) is True and j.grasped_at(t0 + 0.2) is False


@pytest.mark.parametrize("scenario,label,reason", [("no_lift", "n", "timeout"), ("lift_drop", "n", "timeout"), ("knock_over", "n", "knocked_over"),
                                                   ("reflex", "n", "reflex")])
def test_stub_scenarios_judged(tmp_path, scenario, label, reason):
    ep = f"tcp://127.0.0.1:{_port()}"
    stub = SL.StubSim(ep, scenario)
    cmd = tmp_path / "commands.txt"
    try:
        t0 = time.monotonic()
        stub.on_episode_start(t0)
        j = J.Judge(ep, cmd, dict(SUCCESS, timeout_s=6.0), ["pad_", "tag_", "finger"], [["eraser", "table"]], t_start=t0)
        j.start()
        assert j.decided.wait(12)
        j.stop()
    finally:
        stub.close()
    assert j.label == label and j.tracker.result()["reason"] == reason, j.tracker.result()


def test_stub_collision_scenario_counts_one_arm_table_contact(tmp_path):
    ep = f"tcp://127.0.0.1:{_port()}"
    stub = SL.StubSim(ep, "collision")
    try:
        t0 = time.monotonic()
        stub.on_episode_start(t0)
        j = J.Judge(ep, tmp_path / "c.txt", dict(SUCCESS, timeout_s=15.0), ["pad_", "tag_", "finger"], [["eraser", "table"]], t_start=t0)
        j.start()
        assert j.decided.wait(15)
        j.stop()
    finally:
        stub.close()
    assert j.label == "y" and [(c["a"], c["b"]) for c in j.collisions] == [("link6_col", "table_top")]   # pad/eraser, eraser/table ignored
    assert j.collisions[0]["t_s"] == pytest.approx(2.0, abs=0.2)


def test_stub_refuses_unknown_scenario():
    with pytest.raises(ValueError):
        SL.StubSim("tcp://127.0.0.1:1", "nope")


def test_bridge_sim_sends_the_a6_payload_and_maps_refusal(tmp_path):
    """BridgeSim vs a stand-in bridge REP socket: topic sim_reset, {"eraser_xy_yaw": [x, y, yaw], "hand_gap_m": g}; ok:false -> refused."""
    pytest.importorskip("zmq")
    from franka_deploy.bridge.link import BridgeServerSocket

    ep = f"tcp://127.0.0.1:{_port()}"
    srv = BridgeServerSocket(ep)
    got, stop = [], threading.Event()

    def serve():
        while not stop.is_set():
            topic, data = srv.poll(50)
            if topic is None:
                continue
            got.append((topic, data))
            if topic == "sim_reset" and data["eraser_xy_yaw"][0] > 1.0:
                srv.reply(False, error="grasping: refused")
            elif topic == "sim_reset":
                srv.reply(True, {"moved": True})
            else:
                srv.reply(True, {"t_mono": 1.0})

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    try:
        bs = SL.BridgeSim(ep, "tcp://127.0.0.1:5570", 2.0)
        assert bs.reset((0.4, 0.1, 0.5), 0.07) == {"moved": True}
        assert got[0] == ("sim_reset", {"eraser_xy_yaw": [0.4, 0.1, 0.5], "hand_gap_m": 0.07})
        bs.reset((0.4, 0.1, 0.5))
        assert got[1][1] == {"eraser_xy_yaw": [0.4, 0.1, 0.5]}
        with pytest.raises(SL.SimResetRefused, match="grasping"):
            bs.reset((2.0, 0.0, 0.0))
        assert bs.latest() == {"t_mono": 1.0} and got[-1][0] == "sim_state"
        bs.close()
    finally:
        stop.set()
        th.join(2)
        srv.close()


def test_pad_table_contacts_are_reported_apart_from_collisions(tmp_path):
    """A6 review finding 5: a finger geom on the table is not a collision but is reported on its own."""
    pats, ign, tbl = ["pad_", "tag_", "finger"], [["eraser", "table"]], ["table"]
    cs = [{"a": "pad_left", "b": "table", "force_n": 40.0}, {"a": "table", "b": "finger_r", "force_n": 38.0},
          {"a": "pad_left", "b": "eraser", "force_n": 10.0}, {"a": "eraser", "b": "table", "force_n": 1.0},
          {"a": "link6_col", "b": "table", "force_n": 7.0}]
    assert [(c["a"], c["b"]) for c in J.classify_pad_table(cs, pats, tbl)] == [("pad_left", "table"), ("table", "finger_r")]
    assert [(c["a"], c["b"]) for c in J.classify_contacts(cs, pats, ign)] == [("link6_col", "table")]
    j = J.Judge("tcp://127.0.0.1:1", tmp_path / "c.txt", SUCCESS, pats, ign, t_start=0.0, table_patterns=tbl)
    for i, c in enumerate([cs[:1], cs[:1], [], cs[:1]]):       # contact, still in contact, released, again: 2 onsets
        j._observe({"t_mono": 0.1 * i, "contacts": c, "grasped": False})
    assert len(j.pad_table) == 2 and j.pad_table_force_max_n == 40.0 and j.collisions == []
