"""W10A fix round (Track A): derived_tcp (I-TCP) + backend warning, F_T_C_main branch, loop stats, shm cleanup.

    cd controlSW/franka_sim && MUJOCO_GL=egl env -u PYTHONPATH $RE -m pytest -q tests/test_w10a_fix.py
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")

from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import backend as B  # noqa: E402
from franka_sim import campaign as C  # noqa: E402
from franka_sim import model as M  # noqa: E402

PORT = 5570 + 1000 + int(os.environ.get("FD_TEST_PORT_OFFSET", "0")) + 60


def _station(robot="panda", sets=()):
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = robot
    return load_station(spec, sets=list(sets))


def _rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@pytest.mark.parametrize("robot", ["panda", "fr3"])
@pytest.mark.parametrize("side,yaw", [("+y", 45.0), ("-y", -135.0)])
@pytest.mark.parametrize("ext", [0.0, 0.10])
def test_derived_tcp_matches_probe(robot, side, yaw, ext):
    st = _station(robot, [f"tooling.finger_adapter.extension_m={ext}", f"tooling.tcp_x_along_hand={side}"])
    T = M.derived_tcp(st)
    assert np.allclose(T[:3, 3], [0, 0, 0.0973 + ext], atol=2e-4), T[:3, 3]      # probe: 97.3 mm + 1000 ext
    assert np.allclose(T[:3, :3], _rz(yaw), atol=1e-6)
    sm = M.build_model(st, with_scene=False)
    assert np.allclose(T, sm.F_T_TCP, atol=1e-12)                                # model uses it (no station override)


def test_warning_on_mismatch_and_not_on_match(capsys):
    stale = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, -0.1034], [0, 0, 0, 1.0]]     # an explicit stale station TCP
    st = _station(sets=["tooling.F_T_TCP=" + str(stale)])
    sm = M.build_model(st, with_scene=False)
    B._warn_tcp_mismatch(st, sm)
    out = capsys.readouterr().out
    assert "WARNING" in out and sm.info["controller_F_T_TCP_deviation"]["mm"] > 100
    T = [[round(float(x), 12) for x in r] for r in M.derived_tcp(st)]
    st2 = _station(sets=["tooling.F_T_TCP=" + str(T)])
    sm2 = M.build_model(st2, with_scene=False)
    B._warn_tcp_mismatch(st2, sm2)
    assert "WARNING" not in capsys.readouterr().out
    assert sm2.info["controller_F_T_TCP_deviation"]["mm"] < 1e-3


def test_warning_when_station_tcp_is_null(capsys):
    """W10B: a null station F_T_TCP leaves the controller on franka_cartesian.yaml's placeholder: warn if it is off the
    model TCP, stay silent when they agree (what the launcher's derived fill produces)."""
    st = _station()
    assert st.get("tooling.F_T_TCP") is None                                       # shipped: null (hw)
    sm = M.build_model(st, with_scene=False)
    B._warn_tcp_mismatch(st, sm)                                                   # shipped controller file placeholder
    out = capsys.readouterr().out
    assert "WARNING" in out and "null" in out and sm.info["controller_F_T_TCP_deviation"]["mm"] > 100
    B._warn_tcp_mismatch(st, sm, controller_F_T_TCP=sm.F_T_TCP)                    # controller already on the model TCP
    assert "WARNING" not in capsys.readouterr().out
    assert sm.info["controller_F_T_TCP_deviation"]["mm"] < 1e-9


def _with_status(st, path, status):
    import copy
    from dataclasses import replace
    from franka_deploy.station import _thaw
    prov = _thaw(st.provenance)
    g, _, rest = path.partition(".")
    key = next(k for k in prov[g] if k == rest or rest.startswith(k + "."))
    prov[g][key] = dict(prov[g][key], status=status)
    return replace(st, provenance=copy.deepcopy(prov))


def test_F_T_C_main_hw_ignored_measured_used_and_warns(capsys):
    F_C = np.eye(4)
    F_C[:3, 3] = [0.0, 0.0, 0.0]
    base = M.build_model(_station(), with_scene=False)
    F_C = np.array(M.export_frames(base)["F_T_cam_main"], float)
    F_C[0, 3] += 0.020                                                            # 20 mm off
    st = _station(sets=["tooling.F_T_C_main=" + str(F_C.tolist())])
    assert st.prov("tooling.F_T_C_main").status == "hw"
    sm_hw = M.build_model(st, with_scene=False)
    assert sm_hw.info["F_T_C_main_source"].startswith("derived") and np.allclose(sm_hw.F_T_TCP, base.F_T_TCP)
    st_m = _with_status(st, "tooling.F_T_C_main", "measured")
    sm = M.build_model(st_m, with_scene=False)
    assert sm.info["F_T_C_main_source"] == "station" and np.linalg.norm(sm.F_T_TCP[:3, 3] - base.F_T_TCP[:3, 3]) > 0.019
    capsys.readouterr()
    T = [[round(float(x), 12) for x in r] for r in M.derived_tcp(st_m)]
    st_c = _with_status(_station(sets=["tooling.F_T_C_main=" + str(F_C.tolist()), "tooling.F_T_TCP=" + str(T)]),
                        "tooling.F_T_C_main", "measured")                         # controller still on derived_tcp
    B._warn_tcp_mismatch(st_c, M.build_model(st_c, with_scene=False))
    assert "WARNING" in capsys.readouterr().out


def test_F_T_C_main_non_placeholder_wins(capsys):
    st = _station()
    sm0 = M.build_model(st, with_scene=False)
    assert sm0.info["F_T_C_main_source"].startswith("derived")                   # null -> derived branch
    C_T_TCP = np.array(st.get("tooling.C_T_TCP"), float)
    F_C = np.eye(4)
    F_C[:3, :3] = _rz(10.0)
    F_C[:3, 3] = [0.05, -0.04, 0.02]
    st2 = _with_status(_station(sets=["tooling.F_T_C_main=" + str(F_C.tolist())]), "tooling.F_T_C_main", "measured")
    sm = M.build_model(st2, with_scene=False)
    assert sm.info["F_T_C_main_source"] == "station" and sm.info["F_T_TCP_source"] == "F_T_C_main"
    assert "F_T_C_main" in capsys.readouterr().out
    fr = M.export_frames(sm)
    assert np.allclose(np.array(fr["F_T_cam_main"]), F_C, atol=1e-9)
    assert np.allclose(sm.F_T_TCP, F_C @ M.orthonormalize(C_T_TCP), atol=1e-9)
    assert np.allclose(np.array(fr["F_T_TCP"]), sm.F_T_TCP, atol=1e-9)
    assert sm.info["F_T_TCP_vs_camera_deviation"]["mm"] > 1
    assert np.allclose(M.derived_tcp(st2), sm0.F_T_TCP, atol=1e-12)              # derived_tcp stays the pad chain


def test_lag_recorded_before_resync_and_resyncs_per_window():
    cfg = {"type": "mujoco", "mujoco": {"station": "sim-iphone", "robot": "panda",
                                        "state_endpoint": f"tcp://127.0.0.1:{PORT}", "with_scene": False}}
    b = B.MuJoCoBackend.from_config(cfg, np.array([[1, 0, 0, 0.3], [0, -1, 0, 0], [0, 0, -1, 0.5], [0, 0, 0, 1.0]]))
    try:
        b.start()
        time.sleep(0.3)
        b.loop_stats()
        with b._core_lock:
            time.sleep(0.4)                                                       # injected stall
        time.sleep(0.2)
        ls = b.loop_stats()
        assert ls["rt_resyncs_window"] >= 1 and ls["sim_lag_ms_max"] > 100, ls
        time.sleep(0.3)
        ls2 = b.loop_stats()
        assert ls2["rt_resyncs_window"] == 0 and ls2["rt_resyncs"] == ls["rt_resyncs"]   # cumulative key unchanged
    finally:
        b.close()


def test_kill_cleanup_shm_dead_pids_only():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    live = f"/dev/shm/rmq_tst_{p.pid}_w10a_live"
    p2 = subprocess.Popen([sys.executable, "-c", "pass"])
    p2.wait()
    dead = f"/dev/shm/rmq_tst_{p2.pid}_w10a_dead"
    other = "/dev/shm/rmq_tst_999999_w10a_other"
    embed = f"/dev/shm/rmq_tst_{p.pid}_srv_{p2.pid}_topic"                        # live owner, dead pid later in the name
    for f in (live, dead, other, embed):
        open(f, "w").close()
    try:
        tree = C.process_tree(os.getpid())
        assert p.pid in tree and p2.pid not in tree
        removed = C.kill_cleanup_shm([p.pid, p2.pid])
        assert removed == [os.path.basename(dead)]
        assert os.path.exists(live) and os.path.exists(other) and os.path.exists(embed) and not os.path.exists(dead)
        p.kill()
        p.wait()
        assert sorted(C.kill_cleanup_shm([p.pid])) == sorted([os.path.basename(live), os.path.basename(embed)])
    finally:
        p.kill()
        for f in (live, dead, other, embed):
            if os.path.exists(f):
                os.remove(f)
