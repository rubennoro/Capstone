"""A6: the MuJoCo bridge process with --allow-sim-reset and the sim-state PUB, driven through B9's
``franka_sim.eval.simlink.BridgeSim`` and ``judge.StateSubscriber``: reset moves the eraser (``latest()`` = bridge
``sim_state``), the PUB rate, ``bridge.jsonl`` sim lines at 25 Hz, the reset log line, and the gate (no flag -> ok:false
-> ``SimResetRefused``; ``--help`` lists the flag for the launcher's probe).

    cd controlSW/franka_deploy && flock $LOCK env -u PYTHONPATH PYTHONPATH=.:../franka_sim FD_TEST_PORT_OFFSET=1000 \
        $RE -m pytest -q tests/test_sim_reset_stack.py
"""
import json
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

from conftest import PKG_ROOT
from test_sim_backend_stack import env, needs_sim, sim_bridge_cfg

OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
BRIDGE = f"tcp://127.0.0.1:{25671 + OFFSET}"
STATE = f"tcp://127.0.0.1:{5570 + OFFSET}"


class SimBridge:
    """The bridge process alone (MuJoCo backend, state PUB on STATE, --log-jsonl)."""

    def __init__(self, work: Path, robot: str = "panda", flags=("--allow-sim-reset",)):
        pytest.importorskip("robotmq")
        from franka_deploy.bridge.link import BridgeClient

        self.work = work
        p = sim_bridge_cfg(work, robot)
        cfg = yaml.safe_load(p.read_text())
        cfg["backend"]["mujoco"]["state_endpoint"] = STATE.replace("127.0.0.1", "localhost")    # normalised by the backend
        p.write_text(yaml.safe_dump(cfg))
        self.jsonl = work / "bridge.jsonl"
        self.log = open(work / "bridge.log", "w")
        self.p = subprocess.Popen([sys.executable, "-u", "-m", "franka_deploy.bridge.franka_bridge", "--config", str(p),
                                   "--endpoint", BRIDGE, "--log-jsonl", str(self.jsonl), *flags],
                                  stdout=self.log, stderr=subprocess.STDOUT, env=env(), cwd=str(work), start_new_session=True)
        self.b = BridgeClient(BRIDGE, timeout_s=0.5)
        t0 = time.monotonic()
        while True:
            try:
                self.b.request("get_state")
                break
            except Exception:
                assert self.p.poll() is None, (work / "bridge.log").read_text()[-2000:]
                assert time.monotonic() - t0 < 30, "bridge not ready"
                time.sleep(0.2)

    def lines(self, kind):
        return [json.loads(x) for x in open(self.jsonl) if f'"kind": "{kind}"' in x]

    def close(self):
        self.b.close()
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGINT)
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.log.close()


@needs_sim
@pytest.mark.parametrize("robot", ["panda", "fr3"])
def test_bridgesim_reset_pub_rate_and_sim_lines(tmp_path, robot):
    from franka_sim.eval.judge import StateSubscriber
    from franka_sim.eval.simlink import BridgeSim

    sb = SimBridge(tmp_path, robot)
    sim = BridgeSim(BRIDGE.replace("127.0.0.1", "0.0.0.0"), STATE)
    sub = StateSubscriber(STATE)
    try:
        q0 = sb.b.request("get_state")["q"]
        t0 = time.perf_counter()
        rep = sim.reset((0.5, 0.0, 0.3))
        dt = time.perf_counter() - t0
        assert rep["ok"] and dt < 0.05 and rep["hand_gap_m"] is None, (rep, dt)
        time.sleep(0.5)
        s = sim.latest()
        T = np.asarray(s["eraser_T_base"])
        assert np.linalg.norm(T[:2, 3] - [0.5, 0.0]) < 0.001 and abs(math.atan2(T[1, 0], T[0, 0]) - 0.3) < math.radians(0.5), T
        assert np.abs(np.asarray(sb.b.request("get_state")["q"]) - q0).max() < 1e-4
        rep2 = sim.reset((0.45, 0.02, -0.2), hand_gap_m=0.05)
        assert rep2["ok"] and rep2["hand_gap_m"] == 0.05
        ts = []
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            x = sub.recv(0.1)
            if x is not None:
                ts.append(x["t_mono"])
        rate = (len(ts) - 1) / (ts[-1] - ts[0])
        assert rate >= 99.0, rate
        assert abs(x["hand_gap_m"] - 0.05) < 0.001 and np.linalg.norm(np.asarray(x["eraser_T_base"])[:2, 3] - [0.45, 0.02]) < 0.001
        lines = sb.lines("sim")
        tl = [ln["t"] for ln in lines if ln["t"] >= ts[0]]
        lrate = (len(tl) - 1) / (tl[-1] - tl[0])
        assert 22.0 <= lrate <= 28.0, lrate
        assert len(sb.lines("sim_reset")) == 2
        print(json.dumps({"robot": robot, "reset_ms": dt * 1e3, "pub_hz": rate, "sim_log_hz": lrate,
                          "contacts": lines[-1]["contacts"]}))
    finally:
        sub.close()
        sim.close()
        sb.close()


@needs_sim
def test_sim_reset_gate_and_help_flag(tmp_path):
    from franka_sim.eval.simlink import BridgeSim, SimResetRefused

    r = subprocess.run([sys.executable, "-m", "franka_deploy.bridge.franka_bridge", "--help"], capture_output=True, text=True,
                       env=env(), cwd=str(PKG_ROOT), timeout=60)
    assert "--allow-sim-reset" in r.stdout                     # the launcher's allow_sim_reset_flag probe
    sb = SimBridge(tmp_path, "panda", flags=())
    sim = BridgeSim(BRIDGE, STATE)
    try:
        with pytest.raises(SimResetRefused, match=r"sim_reset disabled \(--allow-sim-reset\)"):
            sim.reset((0.5, 0.0, 0.0))
        assert sim.latest()["eraser_T_base"] is not None          # sim_state is read-only: always allowed
    finally:
        sim.close()
        sb.close()
