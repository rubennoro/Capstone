"""A5 stack: bridge (MuJoCo backend, serving hand_cmd/hand_state) + FrankaCartesian with gripper `franka_hand` whose
endpoint is the bridge, as separate processes (the A4 Stack with the Franka Hand). Widths 0.084 -> 0.045 -> 0.084
(training units) through the controller: the hand goes moving -> grasp_failed (no object) -> move_ok, and the 200 Hz
loop period error p99 stays within +0.2 ms of the same run with the fake gripper.

    cd controlSW/franka_deploy && flock $LOCK env -u PYTHONPATH PYTHONPATH=.:../franka_sim FD_TEST_PORT_OFFSET=1000 \
        $RE -m pytest -q tests/test_sim_hand_stack.py -s
"""
import json
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

SIM_ROOT = PKG_ROOT.parent / "franka_sim"
OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
BRIDGE = f"tcp://127.0.0.1:{25675 + OFFSET}"
CTRL = f"tcp://localhost:{26931 + OFFSET}"


def have_sim() -> bool:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return (SIM_ROOT / "franka_sim" / "hand_sim.py").is_file()


pytestmark = pytest.mark.skipif(not have_sim(), reason="mujoco / franka_sim not available")


def env():
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update({"PYTHONPATH": os.pathsep.join([str(PKG_ROOT), str(SIM_ROOT)]), "PYTHONNOUSERSITE": "1",
              "PYTHONDONTWRITEBYTECODE": "1"})
    return e


def station_gripper(robot="panda") -> dict:
    """The controller gripper dict the launcher would merge for sim-iphone (station fragment), endpoint = bridge."""
    from franka_deploy.station import PROFILES, load_station

    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = robot
    g = load_station(spec).controller_fragment()["gripper"]
    g["endpoint"] = BRIDGE
    return g


class HandStack:
    def __init__(self, work: Path, gripper: str, robot: str = "panda"):
        pytest.importorskip("real_env")
        self.robotmq = pytest.importorskip("robotmq")
        from franka_deploy.bridge.link import BridgeClient
        from franka_deploy.controllers.safe_client import SafeCartesianJointClient
        from franka_deploy.rmq_rpc import NoResendClient

        self.work = work
        bcfg = yaml.safe_load(open(PKG_ROOT / "configs" / "bridge.yaml"))
        bcfg["backend"]["type"] = "mujoco"
        bcfg["backend"]["mujoco"] = {"station": "sim-iphone", "robot": robot,
                                     "state_endpoint": f"tcp://127.0.0.1:{5570 + 1000 + OFFSET + 24}"}   # A6 PUB, own port
        (work / "bridge.yaml").write_text(yaml.safe_dump(bcfg))
        ccfg = yaml.safe_load(open(PKG_ROOT / "configs" / "franka_cartesian.yaml"))
        ccfg["bridge_endpoint"] = BRIDGE
        ccfg["controller"]["endpoint"] = CTRL
        if gripper == "franka_hand":
            ccfg["gripper"].update(station_gripper(robot))       # servo leftovers stay: the driver ignores them
        else:
            ccfg["gripper"]["speed"] = 3000.0
        (work / "fc.yaml").write_text(yaml.safe_dump(ccfg))
        self.logs = {}
        self.bridge = self._start("bridge", ["-m", "franka_deploy.bridge.franka_bridge", "--config",
                                             str(work / "bridge.yaml"), "--endpoint", BRIDGE])
        self.b = BridgeClient(BRIDGE, timeout_s=0.5)
        t0 = time.monotonic()
        while True:
            try:
                self.b.request("get_state")
                break
            except Exception:
                assert self.bridge.poll() is None, (work / "bridge.log").read_text()[-2000:]
                assert time.monotonic() - t0 < 20, "bridge not ready"
                time.sleep(0.2)
        self.ctrl = self._start("controller", ["-m", "franka_deploy.controllers.franka_cartesian", "--config",
                                               str(work / "fc.yaml")])
        self.client = SafeCartesianJointClient(name="test", server_endpoint=CTRL, timeout_s=0.1, connect_timeout_s=15)
        self.status_c = NoResendClient("st", CTRL, default_timeout_s=0.2)

    def _start(self, name, args):
        log = open(self.work / f"{name}.log", "w")
        self.logs[name] = log
        return subprocess.Popen([sys.executable, "-u", *args], stdout=log, stderr=subprocess.STDOUT, env=env(),
                                cwd=str(self.work), start_new_session=True)

    def status(self):
        raw, _ = self.status_c.peek_data(topic="status", n=-1)
        return self.robotmq.deserialize(raw[0])

    def width(self, w, dt=0.2):
        self.client.schedule_joint_traj(np.array([[w]]), np.array([time.monotonic() + dt]), use_relative_timestamps=False)

    def close(self):
        for p in (self.ctrl, self.bridge):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
        for log in self.logs.values():
            log.close()


def run_schedule(s: HandStack, hand: bool) -> dict:
    """0.084 -> 0.045 -> 0.084 (training), ~2.5 s each; records the hand_state (mode, last_result) sequence."""
    time.sleep(1.5)
    seq, t_end = [], None
    for w in (0.084, 0.045, 0.084):
        s.width(w)
        t_end = time.monotonic() + 2.5
        while time.monotonic() < t_end:
            if hand:
                st = s.b.request("hand_state")
                item = (st["mode"], st["last_result"])
                if not seq or seq[-1] != item:
                    seq.append(item)
            time.sleep(0.01)
    out = {"status": s.status(), "seq": seq, "width": float(np.asarray(s.client.get_joint_pos(time.monotonic())).reshape(-1)[0])}
    if hand:
        out["hand"] = s.b.request("hand_state")
    out["bridge"] = s.b.request("get_state")
    return out


def test_franka_hand_through_controller_and_loop_p99(tmp_path):
    from franka_deploy.gripper.hand_logic import WidthMap

    (tmp_path / "fake").mkdir()
    (tmp_path / "hand").mkdir()
    s = HandStack(tmp_path / "fake", "fake")
    try:
        fake = run_schedule(s, hand=False)
    finally:
        s.close()
    s = HandStack(tmp_path / "hand", "franka_hand")
    try:
        hand = run_schedule(s, hand=True)
    finally:
        s.close()
    wm = WidthMap.from_config(station_gripper()["width_map"])
    modes = [m for m, _ in hand["seq"]]
    results = [r for _, r in hand["seq"]]
    p_fake, p_hand = fake["status"]["period_err_ms_p99"], hand["status"]["period_err_ms_p99"]
    print("\nA5 stack " + json.dumps({"seq": hand["seq"], "final": hand["hand"], "p99_fake_ms": p_fake,
                                       "p99_hand_ms": p_hand, "hop_p99_fake": fake["status"]["hop_ms_p99"],
                                       "hop_p99_hand": hand["status"]["hop_ms_p99"]}, default=str))
    assert "grasping" in modes and "grasp_failed" in results, hand["seq"]
    i_fail = results.index("grasp_failed")
    assert "moving" in modes[i_fail:] and results[-1] == "move_ok", hand["seq"]
    assert abs(hand["hand"]["gap_m"] - wm.gap(0.084)) <= 0.002, hand["hand"]
    assert not hand["status"]["fault"] and hand["bridge"]["error"] is None and hand["bridge"]["mode"] == "running"
    assert abs(hand["width"] - wm.train(wm.gap(0.084))) < 0.002              # the controller's measured width
    assert p_hand <= p_fake + 0.2, (p_hand, p_fake)
