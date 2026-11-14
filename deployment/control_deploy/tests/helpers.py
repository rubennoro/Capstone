"""Test helpers: a fake franka_control script, a station file, a fake PC (RpcServer + Publisher from deploy_contracts)."""
from __future__ import annotations

import os
import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from deploy_contracts import (SHARED_SCHEMAS, Ack, ArmState, ContractLimits, GateCounters, GetState, Heartbeat, Hello,
                              HelloReply, Home, LinkSettings, Publisher, RpcServer, SetWaypoints, SetWaypointsReply,
                              Start, Stop, TcpFrame, WaypointIssue)

FAKE_BINARY = r'''#!{python}
# A fake franka_control for the wrapper tests: behaviour from FAKE_* environment variables.
import os, signal, sys, time, termios
args = sys.argv[1:]
if "--version" in args:
    print(os.environ.get("FAKE_VERSION", "franka_control commit={commit} dirty=no libfranka=0.9.2"))
    sys.exit(int(os.environ.get("FAKE_VERSION_RC", "0")))
if "--check-config" in args:
    if os.environ.get("FAKE_HANG"):
        time.sleep(60)
    sys.stdout.write(open(os.environ["FAKE_CHECK_CONFIG"]).read() if os.environ.get("FAKE_CHECK_CONFIG") else "{{}}\n")
    sys.exit(int(os.environ.get("FAKE_CHECK_RC", "0")))
if os.environ.get("FAKE_ARGS_FILE"):
    open(os.environ["FAKE_ARGS_FILE"], "w").write("\n".join(args))
if os.environ.get("FAKE_RAW_TTY"):
    t = termios.tcgetattr(0)
    t[3] &= ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(0, termios.TCSANOW, t)
code = int(os.environ.get("FAKE_EXIT_CODE", "0"))
signal.signal(signal.SIGINT, signal.SIG_IGN)
if os.environ.get("FAKE_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, lambda *a: (print("fake: SIGTERM", flush=True), sys.exit(code)))
print("fake: starting", flush=True)
ready = args[args.index("--ready-file") + 1]
if not os.environ.get("FAKE_NO_READY"):
    open(ready, "w").close()
    print("fake: ready", flush=True)
end = time.monotonic() + float(os.environ.get("FAKE_RUN_S", "30"))
while time.monotonic() < end:
    time.sleep(0.02)
print("fake: done", flush=True)
sys.exit(code)
'''


def write_fake_binary(folder: Path, commit: str = "0" * 40) -> Path:
    path = folder / "fake_franka_control"
    path.write_text(FAKE_BINARY.format(python=sys.executable, commit=commit))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def write_station(folder: Path, franka_control: dict | None = None, console: dict | None = None,
                  extra: dict | None = None) -> Path:
    data = {"station_version": 1, "session_root": str(folder / "sessions")}
    if franka_control is not None:
        data["franka_control"] = franka_control
    if console is not None:
        data["console"] = console
    data.update(extra or {})
    path = folder / "station.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def pose(x: float, y: float, z: float, yaw: float = 0.0) -> np.ndarray:
    """TCP z down (rotation pi about x), then yaw about the TCP z axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    m = np.eye(4)
    m[:3, :3] = down @ rz
    m[:3, 3] = [x, y, z]
    return m


def state(mode: str = "idle", t_ns: int = 0, seq: int = 1, p: np.ndarray | None = None,
          target: np.ndarray | None = None, width: float = 0.08, error: str = "", loop_late: int = 0,
          gate: GateCounters | None = None) -> ArmState:
    p = pose(0.45, 0.0, 0.45) if p is None else p
    return ArmState(t_ns=int(t_ns), seq=seq, O_T_TCP=p, O_T_TCP_target=p if target is None else target,
                    q=np.zeros(7), dq=np.zeros(7), gripper_width_m=float(width), mode=mode, error=error,
                    gate=gate or GateCounters(accepted=0, shortened=0, pulled_back=0, rejected=0),
                    loop_late=loop_late)


def hello_reply(**override) -> HelloReply:
    limits, frame, link = ContractLimits.load(), TcpFrame.load(), LinkSettings.load()
    values = dict(t_pc_ns=time.monotonic_ns(), commit="unknown", mock=True, schemas=tuple(SHARED_SCHEMAS),
                  F_T_TCP=np.array(frame.F_T_TCP), rotation_tol=limits.rotation_tol,
                  gripper_width_max_m=limits.gripper_width_max_m, max_waypoints=limits.max_waypoints,
                  watchdog_limit_s=link.watchdog_limit_s, homing_timeout_s=30.0, max_lead_s=1.5)
    values.update(override)
    return HelloReply(**values)


class FakePC:
    """C2 at 100 Hz and a C3 server on free local ports, with a scriptable mode. Accepts every waypoint unless
    `reject_box_z` is set (then a waypoint above it is rejected and the mode goes to fault)."""

    def __init__(self, mode: str = "idle", clock_offset_ns: int = 0, hello: dict | None = None,
                 reject_box_z: float | None = None) -> None:
        self.mode = mode
        self.error = ""
        self.clock_offset_ns = clock_offset_ns
        self.hello = hello or {}
        self.reject_box_z = reject_box_z
        self.received: list[str] = []
        self.target = pose(0.45, 0.0, 0.45)
        self.lock = threading.Lock()
        self.pub = Publisher("tcp://127.0.0.1:*", 1000)
        self.srv = RpcServer("tcp://127.0.0.1:*", {
            Hello: self._hello, Heartbeat: lambda m: self._ack(m), GetState: self._get_state, Start: self._start,
            Stop: self._stop, Home: lambda m: self._ack(m), SetWaypoints: self._waypoints}, schemas=SHARED_SCHEMAS)
        self.stop_event = threading.Event()
        self.seq = 0
        self.threads = [threading.Thread(target=self._publish, daemon=True),
                        threading.Thread(target=self.srv.serve_forever, args=(self.stop_event, 0.01), daemon=True)]
        for t in self.threads:
            t.start()

    @property
    def endpoints(self) -> tuple[str, str]:
        return self.pub.endpoint, self.srv.endpoint

    def now(self) -> int:
        return time.monotonic_ns() + self.clock_offset_ns

    def state(self) -> ArmState:
        with self.lock:
            return state(self.mode, self.now(), self.seq, self.target, self.target, error=self.error)

    def _publish(self) -> None:
        while not self.stop_event.is_set():
            self.seq += 1
            self.pub.publish(self.state())
            time.sleep(0.01)

    def _ack(self, m, ok: bool = True, reason: str = "") -> Ack:
        self.received.append(m.SCHEMA)
        return Ack(ok=ok, reason=reason)

    def _hello(self, m) -> HelloReply:
        self.received.append(m.SCHEMA)
        return hello_reply(**{"t_pc_ns": self.now(), **self.hello})

    def _get_state(self, m) -> ArmState:
        self.received.append(m.SCHEMA)
        return self.state()

    def _start(self, m) -> Ack:
        if self.mode in ("idle", "stopped"):
            self.mode = "holding"
            return self._ack(m)
        return self._ack(m, False, f"refused: mode is {self.mode}")

    def _stop(self, m) -> Ack:
        if self.mode in ("idle", "holding", "running"):
            self.mode = "stopped"
        return self._ack(m)

    def _waypoints(self, m: SetWaypoints) -> SetWaypointsReply:
        self.received.append(m.SCHEMA)
        n = len(m.t_ns)
        if self.reject_box_z is not None and float(m.O_T_TCP[0][2, 3]) > self.reject_box_z:
            self.mode, self.error = "fault", "safety gate rejected waypoint 0 of chunk 900: outside workspace box"
            return SetWaypointsReply(chunk_id=m.chunk_id, accepted=0, shortened=0, skipped=0,
                                     rejected=(WaypointIssue(index=0, reason="outside workspace box by 0.030 m on z"),))
        return SetWaypointsReply(chunk_id=m.chunk_id, accepted=n, shortened=0, skipped=0, rejected=())

    def close(self) -> None:
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2.0)
        self.srv.close()
        self.pub.close()


def free_port_pair_in_use(ports=(5620, 5630)) -> bool:
    import socket
    for port in ports:
        s = socket.socket()
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
        finally:
            s.close()
    return False


def env_without(*names: str) -> dict[str, str]:
    env = dict(os.environ)
    for n in names:
        env.pop(n, None)
    return env
