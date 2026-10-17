"""franka_bridge (FakeBackend) + FrankaCartesian as real processes, driven like the task drives them.
Runs in the real-env env. Ports are test-specific."""
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

pytest.importorskip("real_env")
robotmq = pytest.importorskip("robotmq")

from conftest import PKG_ROOT  # noqa: E402
from franka_deploy.bridge.link import BridgeClient  # noqa: E402
from franka_deploy.controllers.safe_client import SafeCartesianJointClient  # noqa: E402
from franka_deploy.rmq_rpc import NoResendClient  # noqa: E402

BRIDGE = "tcp://127.0.0.1:25655"
CTRL = "tcp://localhost:26911"


def env():
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update({"PYTHONPATH": str(PKG_ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return e


class Stack:
    def __init__(self, work: Path, object_width=None, bias=(0, 0, 0)):
        self.work = work
        bcfg = yaml.safe_load(open(PKG_ROOT / "configs" / "bridge.yaml"))
        bcfg["backend"]["bias_m"] = list(bias)
        (work / "bridge.yaml").write_text(yaml.safe_dump(bcfg))
        ccfg = yaml.safe_load(open(PKG_ROOT / "configs" / "franka_cartesian.yaml"))
        ccfg["bridge_endpoint"] = BRIDGE
        ccfg["controller"]["endpoint"] = CTRL
        ccfg["gripper"]["object_width_m"] = object_width
        ccfg["gripper"]["speed"] = 3000.0
        (work / "fc.yaml").write_text(yaml.safe_dump(ccfg))
        self.logs = {}
        self.bridge = self._start("bridge", ["-m", "franka_deploy.bridge.franka_bridge", "--config", str(work / "bridge.yaml"),
                                             "--endpoint", BRIDGE, "--allow-inject"])
        time.sleep(0.8)
        self.ctrl = self._start("controller", ["-m", "franka_deploy.controllers.franka_cartesian", "--config", str(work / "fc.yaml")])
        self.client = SafeCartesianJointClient(name="test", server_endpoint=CTRL, timeout_s=0.1, connect_timeout_s=15)
        self.status_c = NoResendClient("st", CTRL, default_timeout_s=0.2)
        self.b = BridgeClient(BRIDGE, timeout_s=0.2)

    def _start(self, name, args):
        log = open(self.work / f"{name}.log", "w")
        self.logs[name] = log
        return subprocess.Popen([sys.executable, "-u", *args], stdout=log, stderr=subprocess.STDOUT, env=env(),
                                cwd=str(self.work), start_new_session=True)

    def status(self):
        raw, _ = self.status_c.peek_data(topic="status", n=-1)
        return robotmq.deserialize(raw[0])

    def fk(self):
        return np.asarray(self.client.get_eef_xyz_wxyz(time.monotonic()), float)

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


@pytest.fixture
def stack(tmp_path):
    s = Stack(tmp_path)
    yield s
    s.close()


def test_state_hop_timing_and_overspeed_chunk_executed(stack):
    time.sleep(1.5)
    p0 = stack.fk()
    now = time.monotonic()
    ts = now + 0.15 + np.arange(3) * 0.05                       # 10 cm in 0.1 s = 1 m/s > bridge v_max 0.25
    traj = np.repeat(p0[None], 3, axis=0)
    traj[:, 0] = p0[0] + np.array([0.0, 0.05, 0.10])
    stack.client.schedule_eef_traj(traj, ts, use_relative_timestamps=False)
    time.sleep(0.30)
    mid = stack.fk()[0] - p0[0]
    time.sleep(1.5)
    end = stack.fk()[0] - p0[0]
    st = stack.status()
    print(json.dumps({"mid_dx": mid, "end_dx": end, "hop_ms_p99": st["hop_ms_p99"], "period_err_ms_p99": st["period_err_ms_p99"],
                      "rejected": st["counters"]["eef_rejected"]}))
    assert st["counters"]["eef_rejected"] == 0                  # interpolator did not freeze it
    assert mid < 0.075                                          # clamped by the bridge (commanded: done after 0.25 s)
    assert abs(end - 0.10) < 0.002                              # ... but executed to the end
    # smoke bound only (includes start-up hops on a loaded laptop); the acceptance number (p99 over a 5-min closed
    # loop with inference running) comes from the harness run's controller status
    assert st["hop_ms_p99"] < 5.0, st["hop_ms_p99"]


def test_hold_keeps_commanded_width_and_target(tmp_path):
    s = Stack(tmp_path, object_width=0.05)
    try:
        time.sleep(1.2)
        now = time.monotonic()
        s.client.schedule_joint_traj(np.array([[0.045]]), np.array([now + 0.2]), use_relative_timestamps=False)
        time.sleep(1.0)
        w_meas = float(s.client.get_joint_pos(time.monotonic())[0])
        assert abs(w_meas - 0.05) < 1e-3                         # fingers stalled on the object
        s.client.hold("test")
        time.sleep(0.3)
        st = s.status()
        assert st["held"]
        raw, _ = NoResendClient("jt", CTRL, 0.2).peek_data(topic="joint_target", n=-1)
        jt = float(np.asarray(robotmq.deserialize(raw[0])["joint_target"]).reshape(-1)[0])
        assert abs(jt - 0.045) < 1e-6, jt                        # last commanded width kept (grasp not released)
        s.client.release()
    finally:
        s.close()


def test_reflex_fault_no_auto_recovery(stack):
    time.sleep(1.2)
    stack.b.request("inject_reflex", "cartesian_reflex (test)")
    time.sleep(1.5)
    with pytest.raises(RuntimeError):
        stack.client.schedule_eef_traj(np.repeat(stack.fk()[None], 2, 0), time.monotonic() + np.array([0.2, 0.3]), False)
    st = stack.status()
    assert st["fault"] and "reflex" in st["fault"]
    time.sleep(2.0)
    assert stack.b.request("get_state")["bridge_state"] == "reflex"   # never recovered by itself
    stack.b.request("recover")
    assert stack.b.request("get_state")["bridge_state"] == "idle"


def test_kill_controller_bridge_hold_then_stop(stack):
    time.sleep(1.2)
    t_kill = time.monotonic()
    os.killpg(stack.ctrl.pid, signal.SIGKILL)
    t_hold = t_stop = None
    while time.monotonic() - t_kill < 3.0:
        bs = stack.b.request("get_state")["bridge_state"]
        if bs == "hold" and t_hold is None:
            t_hold = time.monotonic() - t_kill
        if bs == "stopped":
            t_stop = time.monotonic() - t_kill
            break
        time.sleep(0.01)
    print(json.dumps({"bridge_hold_after_s": t_hold, "bridge_stop_after_s": t_stop}))
    assert t_hold is not None and t_hold <= 0.2
    assert t_stop is not None and t_stop <= 2.0


def test_kill_bridge_state_goes_stale(stack):
    time.sleep(1.2)
    stack.fk()
    assert stack.client.state_age_s() < 0.05
    os.killpg(stack.bridge.pid, signal.SIGKILL)
    t = time.monotonic()
    time.sleep(0.06)
    stack.fk()
    age = stack.client.state_age_s()
    print(json.dumps({"state_age_after_bridge_kill_s": age}))
    assert age > 0.05  # the task's 50 ms guard trips on its next state read


def test_sigint_ordered_shutdown_and_port_reuse(stack):
    time.sleep(1.0)
    stack.bridge.send_signal(signal.SIGINT)
    assert stack.bridge.wait(5) == 0
    log = (stack.work / "bridge.log").read_text()
    assert "[bridge] shutdown" in log
    p = subprocess.Popen([sys.executable, "-u", "-m", "franka_deploy.bridge.franka_bridge", "--config",
                          str(stack.work / "bridge.yaml"), "--endpoint", BRIDGE], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, env=env(), start_new_session=True)
    try:
        time.sleep(1.0)
        assert p.poll() is None, p.stdout.read().decode()[-500:]
        assert BridgeClient(BRIDGE, timeout_s=0.5).request("get_state")["bridge_state"] == "idle"
    finally:
        p.send_signal(signal.SIGINT)
        p.wait(5)


def test_time_scaled_fast_chunk_never_clamped(stack):
    """D4 fix: the task time-scales the chunk (arm + gripper together) to <= 0.9 v_max -> 0 bridge clamps."""
    from franka_deploy.tasks.chunk_timing import chunk_scale, scale_timestamps

    time.sleep(1.5)
    p0 = stack.fk()
    c0 = stack.status()["counters"]
    poses = np.repeat(p0[None], 8, axis=0)
    poses[:, 0] = p0[0] + np.arange(8) * 0.05          # 5 cm per 50 ms step = 1 m/s
    s, vp, _ = chunk_scale(poses, p0, 0.05, v_max=0.25, w_max=1.0)
    ts = scale_timestamps(time.monotonic() + 0.1 + np.arange(8) * 0.05, s)
    stack.client.schedule_eef_traj(poses, ts + 0.1, use_relative_timestamps=False)
    stack.client.schedule_joint_traj(np.full((8, 1), 0.06), ts, use_relative_timestamps=False)
    time.sleep(ts[-1] - time.monotonic() + 0.6)
    c1 = stack.status()["counters"]
    end = stack.fk()[0] - p0[0]
    print(json.dumps({"scale": s, "peak_v": vp, "clamp_pos": c1["clamp_pos"] - c0["clamp_pos"],
                      "clamp_rot": c1["clamp_rot"] - c0["clamp_rot"], "end_dx": end}))
    assert abs(vp - 1.0) < 1e-9 and s > 4.4
    assert c1["clamp_pos"] - c0["clamp_pos"] == 0 and c1["clamp_rot"] - c0["clamp_rot"] == 0
    assert abs(end - 0.35) < 0.002


def test_bridge_loss_latches_fault_operator_clear(stack):
    """D3 fix: after > 1 s without bridge replies the controller latches a fault, stops hammering the bridge, and only
    an operator clear_fault (with the bridge back and running) resumes."""
    time.sleep(1.2)
    os.killpg(stack.bridge.pid, signal.SIGKILL)
    time.sleep(2.0)
    st1 = stack.status()
    assert st1["fault"] and "bridge lost" in st1["fault"]
    tr1 = st1["counters"]["transient"]
    time.sleep(2.0)
    st2 = stack.status()
    assert "bridge lost" in st2["fault"]
    rate = (st2["counters"]["transient"] - tr1) / 2.0
    print(json.dumps({"transient_per_s_after_latch": rate, "errors": st2["counters"]["errors"]}))
    assert rate == 0                                  # no retry hammering once latched
    # restart the bridge: still latched until the operator clears it
    stack.bridge = stack._start("bridge2", ["-m", "franka_deploy.bridge.franka_bridge", "--config",
                                            str(stack.work / "bridge.yaml"), "--endpoint", BRIDGE])
    time.sleep(1.5)
    BridgeClient(BRIDGE, timeout_s=0.5).request("start")
    time.sleep(1.2)
    assert "bridge lost" in stack.status()["fault"]
    stack.client.rmq_client.put_data(topic="hold", data=robotmq.serialize({"clear_fault": True}))
    time.sleep(0.5)
    assert not stack.status()["fault"]
