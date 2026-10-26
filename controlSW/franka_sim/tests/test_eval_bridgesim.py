"""B9 Part 2 acceptance 1: ``simlink.BridgeSim`` against the real MuJoCo bridge process (A6 topics), Track B port offset.
sim_reset moves the eraser, is refused while the hand is grasping, the state PUB arrives at >= 100 Hz, and the shifted
bridge / state / hand endpoints are the ones in use (the hand is served by the same bridge in sim: hand endpoint = bridge).

    cd controlSW/franka_sim && flock $LOCK env -u PYTHONPATH FD_TEST_PORT_OFFSET=2000 MUJOCO_GL=egl $RE -m pytest -q -s tests/test_eval_bridgesim.py
"""

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("mujoco")
pytest.importorskip("robotmq")

SIM = Path(__file__).resolve().parents[1]
DEPLOY = SIM.parent / "franka_deploy"
OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
BRIDGE = f"tcp://127.0.0.1:{5565 + OFFSET + 40}"         # +40: not the e2e stage3 ports of the same offset
STATE = f"tcp://127.0.0.1:{5570 + OFFSET + 40}"


def _env():
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update({"PYTHONPATH": os.pathsep.join([str(DEPLOY), str(SIM)]), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return e


@pytest.fixture
def bridge(tmp_path):
    if str(DEPLOY) not in sys.path:
        sys.path.append(str(DEPLOY))
    from franka_deploy.bridge.link import BridgeClient

    cfg = yaml.safe_load(open(DEPLOY / "configs" / "bridge.yaml"))
    cfg["backend"]["type"] = "mujoco"
    cfg["backend"]["mujoco"] = {"station": "sim-iphone", "robot": "panda", "state_endpoint": STATE}
    p = tmp_path / "bridge.yaml"
    p.write_text(yaml.safe_dump(cfg))
    log = open(tmp_path / "bridge.log", "w")
    proc = subprocess.Popen([sys.executable, "-u", "-m", "franka_deploy.bridge.franka_bridge", "--config", str(p), "--endpoint", BRIDGE,
                             "--allow-sim-reset"], stdout=log, stderr=subprocess.STDOUT, env=_env(), cwd=str(tmp_path), start_new_session=True)
    b = BridgeClient(BRIDGE, 0.5)
    t0 = time.monotonic()
    while True:
        try:
            b.request("get_state")
            break
        except Exception:       # noqa: BLE001
            assert proc.poll() is None, (tmp_path / "bridge.log").read_text()[-2000:]
            assert time.monotonic() - t0 < 40, "bridge not ready"
            time.sleep(0.2)
    yield b
    b.close()
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def test_bridgesim_reset_refusal_pub_rate_and_offset_endpoints(bridge):
    from franka_sim.eval.judge import StateSubscriber
    from franka_sim.eval.simlink import BridgeSim, SimResetRefused

    sim = BridgeSim(BRIDGE.replace("127.0.0.1", "0.0.0.0"), STATE, reset_timeout_s=3.0)
    sub = StateSubscriber(STATE)
    try:
        rep = sim.reset((0.48, 0.03, math.radians(15)), hand_gap_m=0.06)
        assert rep["ok"], rep
        time.sleep(0.4)
        T = np.asarray(sim.latest()["eraser_T_base"])
        assert np.linalg.norm(T[:2, 3] - [0.48, 0.03]) < 0.001 and abs(math.atan2(T[1, 0], T[0, 0]) - math.radians(15)) < math.radians(0.5)
        ts = []
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            x = sub.recv(0.1)
            if x is not None:
                ts.append(x["t_mono"])
        rate = (len(ts) - 1) / (ts[-1] - ts[0])
        assert rate >= 99.0, rate
        # the hand on the same (shifted) bridge endpoint: a slow grasp on nothing keeps it in "grasping" for a while
        bridge.request("hand_cmd", {"kind": "grasp", "width_m": 0.0, "speed_m_s": 0.01, "force_n": 20.0, "eps_inner_m": 0.005,
                                    "eps_outer_m": 0.005, "seq": 1})
        time.sleep(0.2)
        st = bridge.request("hand_state")
        assert st["mode"] in ("grasping", "holding"), st
        with pytest.raises(SimResetRefused, match="grasping"):
            sim.reset((0.45, 0.0, 0.0))
        assert len(sim.reset_s) == 1 and sim.reset_s[0] < 0.5
        print(f"\n[bridgesim] reset {sim.reset_s[0] * 1e3:.1f} ms, PUB {rate:.1f} Hz, bridge {BRIDGE}, state {STATE}, hand mode {st['mode']}")
    finally:
        sub.close()
        sim.close()
