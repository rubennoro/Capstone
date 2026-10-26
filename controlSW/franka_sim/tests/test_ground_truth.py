"""A6 ground_truth (pure; runs in RE and BP): synthetic 25 Hz sim-line streams classified like the B9 judge, the
criteria override, contact classification, evaluate_episode on a fake episode dir, and the cross-check against
``judge.SuccessTracker`` fed the same scripted streams at 100 Hz (|delta t_success| <= 0.1 s).

    cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_ground_truth.py
    cd controlSW/franka_sim && env -u PYTHONPATH $BP -m pytest -q tests/test_ground_truth.py
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import yaml

from franka_sim import ground_truth as GT

SUCCESS = {"lift_min_m": 0.05, "hold_s": 2.0, "timeout_s": 10.0, "slip_tol_m": 0.002}
CRIT = {"success": SUCCESS}
Z0 = 0.006
REST = [{"a": "table", "b": "eraser", "force_n": 0.15}]
PADS = [{"a": "pad_l", "b": "eraser", "force_n": 30.0}, {"a": "pad_r", "b": "eraser", "force_n": 30.0}]


def scenario(name: str, t: float) -> dict:
    """(z, tilt_deg, grasped, reflex, contacts) of a scripted episode at time t (episode start = 0)."""
    z, tilt, grasped, reflex, contacts = Z0, 0.0, False, None, list(REST)
    lift = {"lift5": 0.0505, "lift49": 0.049, "drop": 0.0505, "collision": 0.0505}.get(name, 0.0)
    if name in ("lift5", "lift49", "drop", "collision"):
        if t >= 1.0:
            grasped, contacts = True, list(PADS)
        z = Z0 + lift * min(1.0, max(0.0, (t - 2.0) / 0.5))
        if name == "drop" and t >= 3.5:                          # reached at 2.5 s, falls at 3.5 s (held only 1 s)
            z = max(Z0, Z0 + lift - 0.2 * (t - 3.5))
            grasped = t < 3.7
        if name == "collision" and 2.0 <= t < 2.5:
            contacts = contacts + [{"a": "link6_g3", "b": "table", "force_n": 8.0}]
    elif name == "tilt" and t >= 2.0:
        tilt = 45.0
    elif name == "reflex" and t >= 2.0:
        reflex = "cartesian_reflex (test)"
    return {"z": z, "tilt": tilt, "grasped": grasped, "reflex": reflex, "contacts": contacts}


def T_of(z, tilt_deg, x=0.45, y=0.0):
    a = math.radians(tilt_deg)
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
    T[:3, 3] = [x, y, z]
    return T


def line(t, s) -> dict:
    a = math.radians(s["tilt"])
    return {"t": t, "kind": "sim", "obj": [0.45, 0.0, s["z"], math.cos(a / 2), math.sin(a / 2), 0.0, 0.0], "gap": 0.012,
            "grasped": s["grasped"], "hand_mode": "holding" if s["grasped"] else "idle", "contacts": s["contacts"],
            "F_ext_max": 0.0, "reflex": s["reflex"]}


def stream(name, hz=25.0, t_end=12.0, t0=100.0):
    n = int(round(t_end * hz)) + 1
    return [line(t0 + k / hz, scenario(name, k / hz)) for k in range(n)]


def test_lift_held_is_success_at_t_reach_plus_hold():
    r = GT.evaluate(stream("lift5"), CRIT, 100.0)
    assert r["success"] is True and r["reason"] == "lift_held"
    assert abs(r["t_success_s"] - 4.5) <= 0.04, r                # reached at 2.5 s + hold 2.0 s
    assert r["grasp_t_s"] == pytest.approx(1.0, abs=0.04) and abs(r["max_lift_m"] - 0.0505) < 1e-4
    assert r["slip_m"] == 0.0 and not r["slip_exceeds_tol"] and r["n_arm_table_contacts"] == 0
    assert r["source"] == "bridge.jsonl" and abs(r["rate_hz"] - 25.0) < 0.1 and r["t_success_mono"] == pytest.approx(104.5, abs=0.04)
    keys = {"success", "t_success_s", "grasp_t_s", "max_lift_m", "slip_m", "slip_exceeds_tol", "knocked_over", "reflex",
            "reason", "lift_min_m", "hold_s", "timeout_s", "slip_tol_m", "arm_table_contact", "contact_pairs", "n_samples",
            "rate_hz", "source"}
    assert keys <= set(r), keys - set(r)


def test_short_lift_times_out_and_lift_min_override_flips_it():
    r = GT.evaluate(stream("lift49"), CRIT, 100.0)
    assert r["success"] is False and r["reason"] == "timeout" and r["t_success_s"] is None
    r2 = GT.evaluate(stream("lift49"), {"success": {**SUCCESS, "lift_min_m": 0.03}}, 100.0)
    assert r2["success"] is True and abs(r2["t_success_s"] - (2.0 + 0.5 * 0.03 / 0.049 + 2.0)) <= 0.05, r2


def test_lift_then_drop_fails_with_slip():
    r = GT.evaluate(stream("drop"), CRIT, 100.0)
    assert r["success"] is False and r["reason"] == "timeout"
    assert r["slip_m"] > 0.03 and r["slip_exceeds_tol"]


def test_tilt_is_knocked_over_and_reflex_fails():
    r = GT.evaluate(stream("tilt"), CRIT, 100.0)
    assert r["success"] is False and r["knocked_over"] and r["reason"] == "knocked_over"
    r = GT.evaluate(stream("reflex"), CRIT, 100.0)
    assert r["success"] is False and r["reason"] == "reflex" and "cartesian_reflex" in r["reflex"]


def test_arm_table_contact_counted_pad_and_rest_contacts_not():
    r = GT.evaluate(stream("collision"), CRIT, 100.0)
    assert r["success"] is True
    assert [(c["a"], c["b"]) for c in r["arm_table_contact"]] == [("link6_g3", "table")]
    assert r["arm_table_contact"][0]["t_s"] == pytest.approx(2.0, abs=0.04)
    pairs = {tuple(p) for p in r["contact_pairs"]}
    assert {("eraser", "pad_l"), ("eraser", "pad_r"), ("eraser", "table"), ("link6_g3", "table")} <= pairs
    assert GT.evaluate(stream("lift5"), CRIT, 100.0)["n_arm_table_contacts"] == 0


def test_samples_before_t_start_ignored_and_undecided_end():
    ev = stream("lift5", t_end=3.0)
    r = GT.evaluate(ev, CRIT, 100.0)
    assert r["success"] is False and r["reason"] == "ended_undecided"
    r = GT.evaluate(ev + [{"t": 0.0, "kind": "stats"}], CRIT, 101.2)       # z0 taken at 1.2 s
    assert r["n_samples"] == sum(1 for e in ev if e["t"] >= 101.2)


@pytest.mark.parametrize("name", ["lift5", "lift49", "drop", "tilt", "reflex", "collision", "rest"])
def test_cross_check_with_success_tracker(name):
    """The judge at 100 Hz (I-SIMSTATE dicts) vs the ground truth at 25 Hz (bridge.jsonl lines) of the same script."""
    from franka_sim.eval.judge import SuccessTracker, classify_contacts

    tr = SuccessTracker(SUCCESS, t_start=100.0)
    coll = []
    for k in range(int(12.0 * 100) + 1):
        t = k / 100
        s = scenario(name, t)
        st = {"t_mono": 100.0 + t, "eraser_T_base": T_of(s["z"], s["tilt"]), "grasped": s["grasped"], "reflex": s["reflex"],
              "contacts": s["contacts"]}
        coll += classify_contacts(s["contacts"], GT.FINGER_PATTERNS, GT.IGNORE_PAIRS)
        if tr.update(st):
            break
    j = tr.result()
    g = GT.evaluate(stream(name), CRIT, 100.0)
    assert g["success"] == j["success"] and g["reason"] == j["reason"], (g, j)
    if j["success"]:
        assert abs(g["t_success_s"] - j["t_success_s"]) <= 0.1, (g["t_success_s"], j["t_success_s"])
    assert g["knocked_over"] == j["knocked_over"] and bool(g["n_arm_table_contacts"]) == bool(coll)
    assert abs(g["max_lift_m"] - j["max_lift_m"]) < 1e-3


def test_evaluate_episode_reads_station_events_and_none_without_sim(tmp_path):
    ep = tmp_path / "episode_0001"
    (ep / "config").mkdir(parents=True)
    groups = {"criteria": {"success": {**SUCCESS, "timeout_s": 60}},
              "sim": {"stage3": {"finger_geom_patterns": ["pad_", "tag_", "finger"], "ignore_contact_pairs": [["eraser", "table"]]}}}
    (ep / "config" / "station_resolved.yaml").write_text(yaml.safe_dump({"ids": {}, "groups": groups}))
    with open(ep / "events.jsonl", "w") as f:
        f.write(json.dumps({"t": 100.0, "kind": "episode_start", "episode_dir": str(ep)}) + "\n")
    with open(ep / "bridge.jsonl", "w") as f:
        f.write(json.dumps({"t": 99.5, "kind": "stats", "state": "running"}) + "\n")
        for ln in stream("lift5"):
            f.write(json.dumps(ln) + "\n")
    r = GT.evaluate_episode(ep)
    assert r["success"] is True and r["timeout_s"] == 60 and abs(r["t_success_s"] - 4.5) <= 0.04
    r = GT.evaluate_episode(ep, CRIT, t_start=102.6)              # z0 after the lift: no success
    assert r["success"] is False
    (ep / "bridge.jsonl").write_text(json.dumps({"t": 99.5, "kind": "stats"}) + "\n")
    assert GT.evaluate_episode(ep) is None


def test_module_is_pure():
    src = open(GT.__file__).read()
    assert "import mujoco" not in src and "import zmq" not in src and "from franka_sim.eval" not in src


def test_in_hand_slip_from_tcp_z_separates_overshoot_from_slip():
    """A7 (A6 review finding 2): slip_m includes arm overshoot; in_hand_slip_m uses eraser z - tcp_z while grasped."""
    ev = stream("lift5")
    for e in ev:                                  # rigid grasp: TCP 4 mm above the eraser, the arm overshoots 5 mm
        e["tcp_z"] = e["obj"][2] + 0.004
    for e in ev[90:100]:
        e["obj"][2] -= 0.005
        e["tcp_z"] -= 0.005
    r = GT.evaluate(ev, CRIT, 100.0)
    assert r["slip_m"] == pytest.approx(0.005, abs=1e-5) and r["in_hand_slip_m"] == pytest.approx(0.0, abs=1e-6)
    for e in ev[100:]:                            # now the eraser slides 3 mm down in the fingers
        e["obj"][2] -= 0.003
    r = GT.evaluate(ev, CRIT, 100.0)
    assert r["in_hand_slip_m"] == pytest.approx(0.003, abs=1e-5)
    assert GT.evaluate(stream("lift5"), CRIT, 100.0)["in_hand_slip_m"] is None      # old logs: no tcp_z
