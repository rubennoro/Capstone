"""A4 MuJoCoBackend in real time (in process, both arms): 10 s at 1 kHz with 200 Hz set_target/get_state traffic like
the bridge's, loop period p99, resyncs, get_state latency; from_config (IK start at home_O_T_EE, station specs) and the
Backend API (start/stop/recover/inject_reflex/move_joints).

Run: cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_backend_rt.py -s
(the bridge + controller process stack test is franka_deploy/tests/test_sim_backend_stack.py)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim.backend import MuJoCoBackend, load_station_spec  # noqa: E402

HOME = np.array([[1.0, 0, 0, 0.45], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.45], [0, 0, 0, 1]])   # bridge.yaml default


def cfg(robot: str, **mj) -> dict:
    return {"type": "mujoco", "mujoco": {"station": "sim-iphone", "robot": robot, **mj}}


@pytest.mark.parametrize("robot", ["panda", "fr3"])
def test_realtime_10s(robot):
    b = MuJoCoBackend.from_config(cfg(robot), HOME)
    try:
        st = b.get_state()
        assert np.linalg.norm(st["O_T_EE"][:3, 3] - HOME[:3, 3]) < 1e-3       # q_init = IK of home_O_T_EE
        b.start()
        b.loop_stats()                                                     # reset the window
        T0 = b.get_state()["O_T_EE"]
        lat, t_end, k = [], time.monotonic() + 10.0, 0
        nxt = time.monotonic()
        while time.monotonic() < t_end:
            T = T0.copy()
            T[1, 3] += 0.02 * np.sin(2 * np.pi * 0.5 * k * 0.005)          # 0.5 Hz, 2 cm: the arm moves
            b.set_target(T)
            t0 = time.perf_counter()
            s = b.get_state()
            lat.append(time.perf_counter() - t0)
            k += 1
            nxt += 0.005
            time.sleep(max(0.0, nxt - time.monotonic()))
        ls = b.loop_stats()                                                # the whole 10 s window
        assert s["error"] is None and s["mode"] == "running"
    finally:
        b.close()
    lat = np.array(lat) * 1e3
    p99 = ls["p99_ms"]
    rep = {"robot": robot, "n": ls["n"], "period_p50_ms": round(ls["p50_ms"], 3), "period_p99_ms": round(p99, 3),
           "period_p999_ms": round(ls["p999_ms"], 3), "period_max_ms": round(ls["max_ms"], 3),
           "rt_resyncs": ls["rt_resyncs"], "sim_lag_ms_max": round(ls["sim_lag_ms_max"], 3),
           "steps_per_wake_max": ls["steps_per_wake_max"], "realtime_factor": round(ls["realtime_factor"], 4),
           "cycle_p50_ms": round(ls["cycle_p50_ms"], 3), "cycle_p999_ms": round(ls["cycle_p999_ms"], 3),
           "get_state_ms_p50": round(float(np.percentile(lat, 50)), 4),
           "get_state_ms_p99": round(float(np.percentile(lat, 99)), 4)}
    print(f"\n[rt] {json.dumps(rep)}")
    assert p99 <= 2.0
    if rep["rt_resyncs"]:                                    # spec: 0 on an idle laptop; reported, not gated
        print(f"[rt] NOTE {robot}: rt_resyncs={rep['rt_resyncs']} (load {os.getloadavg()})")
    assert rep["get_state_ms_p50"] <= 0.2
    assert rep["realtime_factor"] > 0.97


def test_backend_api_reflex_move_and_stats_keys():
    b = MuJoCoBackend.from_config(cfg("panda"), HOME)
    try:
        st = b.get_state()
        assert set(st) == {"O_T_EE", "q", "dq", "O_F_ext", "mode", "success_rate", "error", "t", "sim_t"}
        b.start()
        T = st["O_T_EE"].copy()
        T[0, 3] += 0.02
        b.set_target(T)
        time.sleep(1.5)
        assert abs(b.get_state()["O_T_EE"][0, 3] - T[0, 3]) < 1e-3
        b.inject_reflex("cartesian_reflex (test)")
        s = b.get_state()
        assert s["mode"] == "reflex" and s["error"] == "cartesian_reflex (test)"
        with pytest.raises(RuntimeError):
            b.start()
        b.recover()
        assert b.get_state()["error"] is None and b.get_state()["mode"] == "idle"
        q_goal = b.get_state()["q"] + np.array([0.1, -0.1, 0.05, 0.1, 0.0, 0.1, -0.2])
        t0 = time.monotonic()
        b.move_joints(q_goal, 0.5)                           # blocks until done, then restarts the controller
        s = b.get_state()
        assert time.monotonic() - t0 > 0.1 and np.abs(s["q"] - q_goal).max() < 1e-3
        assert s["mode"] == "running"
        b.stop()
        assert b.get_state()["mode"] == "stopped"
        ls = b.loop_stats()
        assert {"n", "p50_ms", "p99_ms", "cycle_hist_us", "sim_lag_ms_max", "steps_per_wake_max", "rt_resyncs",
                "realtime_factor"} <= set(ls)
        assert b.extra_topics == ()
        with pytest.raises(ValueError):
            b.handle_extra("nope", None)
    finally:
        b.close()


def test_station_specs(tmp_path):
    a = load_station_spec(None)
    assert a.ids["robot"] == "panda"
    assert load_station_spec("sim-rig", robot="fr3").ids["robot"] == "fr3"
    assert load_station_spec({"robot": "fr3", "tooling": "franka_hand_iphumi_iphone16pro",
                              "sensor": "iphone16pro_wifi", "scene": "block_raise_table"}).ids["robot"] == "fr3"
    st = load_station({k: dict(v) for k, v in PROFILES["sim-iphone"].items()})
    p = tmp_path / "station_resolved.yaml"
    p.write_text(st.to_yaml())
    st2 = load_station_spec(str(p))
    assert st2.fingerprint() == st.fingerprint() and st2.get("robot.home_q_rad") == st.get("robot.home_q_rad")
    with pytest.raises(ValueError):
        load_station_spec(str(p), robot="fr3")
    with pytest.raises(FileNotFoundError):
        load_station_spec("no-such-profile")


def test_unreachable_home_is_a_clear_error():
    far = HOME.copy()
    far[:3, 3] = [2.0, 0.0, 0.5]
    with pytest.raises(ValueError, match="not reachable"):
        MuJoCoBackend.from_config(cfg("panda"), far)
