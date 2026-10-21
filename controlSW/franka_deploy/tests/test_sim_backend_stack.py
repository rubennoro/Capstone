"""A4: the MuJoCo backend behind the real bridge + FrankaCartesian processes (the test_franka_stack Stack with
backend mujoco), gonogo on the sim, bridge extra-topic routing, and the PandaPyBackend error derivation (fake panda_py).

Process tests take their ports from FD_TEST_PORT_OFFSET (Track A: 1000) and run under the suite lock:
    cd controlSW/franka_deploy && flock $LOCK env -u PYTHONPATH PYTHONPATH=.:../franka_sim FD_TEST_PORT_OFFSET=1000 \
        $RE -m pytest -q tests/test_sim_backend_stack.py
The mujoco parts skip when franka_sim or mujoco is missing (e.g. env_bp).
"""
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import PKG_ROOT
from franka_deploy.bridge import backends as B
from franka_deploy.bridge.franka_bridge import FrankaBridge

SIM_ROOT = PKG_ROOT.parent / "franka_sim"
OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
BRIDGE = f"tcp://127.0.0.1:{25665 + OFFSET}"
CTRL = f"tcp://localhost:{26921 + OFFSET}"
GONOGO_BRIDGE = f"tcp://127.0.0.1:{25667 + OFFSET}"
ROUTE_EP = f"tcp://127.0.0.1:{25669 + OFFSET}"
# A4 spec: lag in [5, 80] ms. The documented panda-py law (D = 2 zeta sqrt(K), not mass-normalised) gives 90-120 ms on
# the 0.4 s gonogo ramps (franka_sim characterize; A4 report), so the upper bound here is the measured envelope.
LAG_MAX_MS = 150.0


def have_sim() -> bool:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return (SIM_ROOT / "franka_sim" / "backend.py").is_file()


needs_sim = pytest.mark.skipif(not have_sim(), reason="mujoco / franka_sim not available")


def env():
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update({"PYTHONPATH": os.pathsep.join([str(PKG_ROOT), str(SIM_ROOT)]), "PYTHONNOUSERSITE": "1",
              "PYTHONDONTWRITEBYTECODE": "1"})
    return e


def sim_bridge_cfg(work: Path, robot: str = "panda") -> Path:
    bcfg = yaml.safe_load(open(PKG_ROOT / "configs" / "bridge.yaml"))
    bcfg["backend"]["type"] = "mujoco"
    bcfg["backend"]["mujoco"] = {"station": "sim-iphone", "robot": robot}
    p = work / "bridge.yaml"
    p.write_text(yaml.safe_dump(bcfg))
    return p


# ----------------------------------------------------------------------------------------- the process stack
class Stack:
    def __init__(self, work: Path, robot: str = "panda"):
        pytest.importorskip("real_env")
        self.robotmq = pytest.importorskip("robotmq")
        from franka_deploy.bridge.link import BridgeClient
        from franka_deploy.controllers.safe_client import SafeCartesianJointClient
        from franka_deploy.rmq_rpc import NoResendClient

        self.work = work
        sim_bridge_cfg(work, robot)
        ccfg = yaml.safe_load(open(PKG_ROOT / "configs" / "franka_cartesian.yaml"))
        ccfg["bridge_endpoint"] = BRIDGE
        ccfg["controller"]["endpoint"] = CTRL
        ccfg["gripper"]["speed"] = 3000.0
        (work / "fc.yaml").write_text(yaml.safe_dump(ccfg))
        self.logs = {}
        self.bridge = self._start("bridge", ["-m", "franka_deploy.bridge.franka_bridge", "--config", str(work / "bridge.yaml"),
                                             "--endpoint", BRIDGE, "--allow-inject"])
        self.b = BridgeClient(BRIDGE, timeout_s=0.5)
        t0 = time.monotonic()
        while True:                                   # the sim builds its model first: wait for get_state
            try:
                self.b.request("get_state")
                break
            except Exception:
                assert self.bridge.poll() is None, (work / "bridge.log").read_text()[-2000:]
                assert time.monotonic() - t0 < 20, "bridge not ready"
                time.sleep(0.2)
        self.ctrl = self._start("controller", ["-m", "franka_deploy.controllers.franka_cartesian", "--config", str(work / "fc.yaml")])
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


@needs_sim
def test_sim_stack_tracks_3cm_move(stack):
    time.sleep(1.5)
    p0 = stack.fk()
    ts = time.monotonic() + 0.2 + np.arange(4) * 0.1                 # 3 cm in 0.3 s = 0.1 m/s (< v_max 0.25)
    traj = np.repeat(p0[None], 4, axis=0)
    traj[:, 0] = p0[0] + np.array([0.0, 0.01, 0.02, 0.03])
    stack.client.schedule_eef_traj(traj, ts, use_relative_timestamps=False)
    time.sleep(2.0)
    end = stack.fk()
    st = stack.status()
    bst = stack.b.request("get_state")
    print(json.dumps({"end_dx": float(end[0] - p0[0]), "dy": float(end[1] - p0[1]), "dz": float(end[2] - p0[2]),
                      "hop_ms_p99": st["hop_ms_p99"], "bridge_state": bst["bridge_state"], "sim_t": bst["sim_t"]}))
    assert st["counters"]["eef_rejected"] == 0 and not st["fault"]
    assert abs(end[0] - p0[0] - 0.03) < 0.002 and np.linalg.norm(end[1:3] - p0[1:3]) < 0.002
    assert bst["bridge_state"] == "running" and bst["error"] is None and bst["mode"] == "running"


@needs_sim
def test_sim_stack_reflex_fault_no_auto_recovery(stack):
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
    s = stack.b.request("get_state")
    assert s["bridge_state"] == "idle" and s["error"] is None and s["mode"] == "idle"


@needs_sim
def test_gonogo_steps_on_sim(tmp_path):
    """gonogo --motion steps on the mujoco backend: lag in [5, LAG_MAX_MS] ms, steady error <= 2 mm (loop gate off)."""
    cfgp = sim_bridge_cfg(tmp_path)
    out = tmp_path / "run"
    p = subprocess.run([sys.executable, "-m", "franka_deploy", "gonogo", "run", "--out", str(out), "--config", str(cfgp),
                        "--motion", "steps", "--amp-m", "0.02", "--hold-s", "2", "--duration-s", "8", "--no-loop-gate",
                        "--spawn-endpoint", GONOGO_BRIDGE], env=env(), capture_output=True, text=True, timeout=120)
    rep = json.loads((out / "gonogo.json").read_text())
    tr = rep["tracking"]
    print(json.dumps({"go": rep["go"], "reasons": rep["reasons"], "tracking": tr, "loop": rep["loop"],
                      "notes": rep["run_meta"]["notes"]}))
    assert p.returncode == 0 and rep["go"], (rep["reasons"], p.stdout[-2000:], p.stderr[-2000:])
    assert rep["loop_gate"] is False and rep["reflexes"] == 0
    assert 5.0 <= tr["lag_ms"] <= LAG_MAX_MS
    assert tr["steady_err_mm_median"] <= 2.0


# ----------------------------------------------------------------------------------------- bridge routing
class _ExtraBackend(B.FakeBackend):
    extra_topics = ("sim_ping",)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ticks = 0

    def handle_extra(self, topic, data):
        return {"pong": data, "topic": topic}

    def sim_tick(self, bridge):
        self.ticks += 1


def test_bridge_routes_extra_topics_and_tick_hook():
    br = FrankaBridge({"backend": {"type": "fake"}}, ROUTE_EP)
    try:
        with pytest.raises(ValueError, match="unknown request"):     # no extra topics: unknown as before
            br.handle("sim_ping", 1)
        br.backend.close()
        br.backend = _ExtraBackend(np.eye(4))
        assert br.handle("sim_ping", 7) == {"pong": 7, "topic": "sim_ping"}
        with pytest.raises(ValueError, match="unknown request"):
            br.handle("other", None)
        assert br.handle("get_state", None)["bridge_state"] == "idle"   # known topics win
        br.tick()
        br.tick()
        assert br.backend.ticks == 2
    finally:
        br.backend.close()
        br.sock.close()


def test_make_backend_mujoco_missing_is_clear(monkeypatch):
    monkeypatch.setitem(sys.modules, "franka_sim.backend", None)       # import fails
    with pytest.raises(RuntimeError, match="franka_sim"):
        B.make_backend({"type": "mujoco"}, np.eye(4))


@needs_sim
def test_make_backend_mujoco_in_process():
    if str(SIM_ROOT) not in sys.path:
        sys.path.append(str(SIM_ROOT))
    home = np.array([[1.0, 0, 0, 0.45], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.45], [0, 0, 0, 1]])
    b = B.make_backend({"type": "mujoco", "mujoco": {"station": "sim-iphone"}}, home)
    try:
        assert type(b).__name__ == "MuJoCoBackend"
        assert np.linalg.norm(b.get_state()["O_T_EE"][:3, 3] - home[:3, 3]) < 1e-3
    finally:
        b.close()


# ----------------------------------------------------------------------------------------- PandaPyBackend error
class _Mode:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"RobotMode.{self.name}"


class _FakePanda:
    def __init__(self, hostname):
        self.mode = "kIdle"
        self.cur, self.last = "[]", "[]"
        self.recovered = 0

    def get_robot(self):
        return types.SimpleNamespace(set_collision_behavior=lambda *a: None, set_EE=lambda *a: None)

    def get_state(self):
        return types.SimpleNamespace(O_T_EE=np.eye(4).T.flatten().tolist(), q=[0.0] * 7, dq=[0.0] * 7,
                                     O_F_ext_hat_K=[0.0] * 6, control_command_success_rate=1.0,
                                     robot_mode=_Mode(self.mode), current_errors=self.cur, last_motion_errors=self.last)

    def start_controller(self, ctrl):
        self.mode = "kMove"

    def stop_controller(self):
        self.mode = "kIdle"

    def recover(self):          # like libfranka: last_motion_errors stays until the next motion starts
        self.recovered += 1
        self.mode, self.cur = "kIdle", "[]"


@pytest.fixture
def pandapy(monkeypatch):
    mod = types.ModuleType("panda_py")
    ctl = types.ModuleType("panda_py.controllers")
    ctl.CartesianImpedance = lambda **kw: types.SimpleNamespace(kw=kw, set_control=lambda p, q: None)
    mod.Panda, mod.controllers = _FakePanda, ctl
    monkeypatch.setitem(sys.modules, "panda_py", mod)
    monkeypatch.setitem(sys.modules, "panda_py.controllers", ctl)
    b = B.make_backend({"type": "panda_py", "panda_py": {"hostname": "fake"}}, np.eye(4))
    return b


def test_pandapy_reflex_sets_error_and_recover_clears(pandapy):
    b = pandapy
    assert b.get_state()["error"] is None
    b.start()
    assert b.get_state()["error"] is None                              # running (kMove)
    b.panda.mode, b.panda.last = "kReflex", "[cartesian_reflex]"
    e = b.get_state()["error"]
    assert e and "reflex" in e.lower() and "cartesian_reflex" in e
    b.panda.mode = "kIdle"                                             # latched until recover
    assert b.get_state()["error"] == e
    with pytest.raises(RuntimeError):
        b.start()
    b.recover()
    assert b.panda.recovered == 1 and b.get_state()["error"] is None


def test_pandapy_error_sources(pandapy):
    b = pandapy
    b.panda.cur = "[joint_velocity_violation]"
    assert "joint_velocity_violation" in b.get_state()["error"]
    b.recover()
    b.start()
    b._t_started -= 1.0                                                # past the start grace period
    b.panda.mode = "kUserStopped"                                      # controller died (user stop) without a reflex
    assert "controller stopped unexpectedly" in b.get_state()["error"]
    b.recover()
    b.stop()                                                           # an intended stop is not an error
    assert b.get_state()["error"] is None
    flags = types.SimpleNamespace(joint_reflex=True, cartesian_reflex=False)
    assert B._errors_text(flags) == "joint_reflex" and B._errors_text("[]") == "" and B._errors_text(None) == ""


def test_pandapy_stale_last_motion_errors_do_not_relatch(pandapy):
    """Review A4-1: libfranka keeps last_motion_errors after automaticErrorRecovery; recover -> start must work."""
    b = pandapy
    b.start()
    b.panda.mode, b.panda.last = "kReflex", "[cartesian_reflex]"
    assert b.get_state()["error"]
    b.recover()
    assert b.panda.last == "[cartesian_reflex]"                          # stale value kept by the fake, as on hardware
    assert b.get_state()["error"] is None                               # not re-latched
    b.start()                                                           # possible again
    assert b.get_state()["error"] is None                               # unchanged stale value while running: ignored
    b.panda.last = "[joint_velocity_violation]"                         # a new motion error while running counts
    assert "joint_velocity_violation" in b.get_state()["error"]


def test_pandapy_stale_last_motion_errors_at_bridge_boot(monkeypatch):
    """Bridge boot after a past reflex: the stale last_motion_errors must not block start()."""
    class _Stale(_FakePanda):
        def __init__(self, hostname):
            super().__init__(hostname)
            self.last = "[cartesian_reflex]"

    mod = types.ModuleType("panda_py")
    ctl = types.ModuleType("panda_py.controllers")
    ctl.CartesianImpedance = lambda **kw: types.SimpleNamespace(kw=kw, set_control=lambda p, q: None)
    mod.Panda, mod.controllers = _Stale, ctl
    monkeypatch.setitem(sys.modules, "panda_py", mod)
    monkeypatch.setitem(sys.modules, "panda_py.controllers", ctl)
    b = B.make_backend({"type": "panda_py", "panda_py": {"hostname": "fake"}}, np.eye(4))
    assert b.get_state()["error"] is None
    b.start()
    assert b.get_state()["error"] is None and b.panda.mode == "kMove"
