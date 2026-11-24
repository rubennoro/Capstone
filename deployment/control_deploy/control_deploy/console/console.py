"""Console: the bring-up C3 client on the PC (control_deploy/STRUCTURE.md 6.5, PLAN 8.1). Run in a second terminal
while `control_deploy run` shows idle or stopped, never while the laptop runs. Waypoint times use this machine's
CLOCK_MONOTONIC (time.monotonic_ns), which is the PC's clock: no clock map."""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml
from deploy_contracts import (Ack, ArmState, ContractLimits, ErrorReply, GetState, Heartbeat, Hello, HelloReply,
                              LinkSettings, RpcClient, SetWaypoints, SetWaypointsReply, Start, Stop, Subscriber,
                              compare_hello)
from deploy_contracts.errors import ContractError, RemoteError, TransportTimeout

from control_deploy import netinfo, paths
from control_deploy.console import waypoints as wpts
from control_deploy.console.steps import NEAR_FACE_M, ORDER, STEPS, Sample, Sent, StepResult
from control_deploy.session import Session
from control_deploy.station import Station

C2_FIRST_S = 2.0              # preflight 1: a C2 message within this
CLOCK_TOL_NS = 5_000_000      # preflight 4: |t_pc_ns - midpoint| <= 5 ms


class Refused(Exception):
    """A preflight refusal: one line, exit 2, nothing sent on C3 (or nothing more)."""


def workspace_box(station: Station, config: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """safety.workspace_min_m / max_m of config.yaml, with the station's franka_control.safety overrides."""
    cfg = config if config is not None else paths.load_config()
    safety = dict(cfg.get("safety", {}))
    safety.update((station.franka_control or {}).get("safety", {}) or {})
    return np.asarray(safety["workspace_min_m"], dtype=float), np.asarray(safety["workspace_max_m"], dtype=float)


class Console:
    def __init__(self, station_path: Path | str | None, mock: bool, local: bool, only: list[str] | None = None,
                 yes: bool = False, *, out: TextIO | None = None, input_fn: Callable[[str], str] = input,
                 peers: Callable[[int], list[str]] = netinfo.c3_peers, endpoints: tuple[str, str] | None = None,
                 link: LinkSettings | None = None, clock: Callable[[], int] = time.monotonic_ns,
                 config: dict[str, Any] | None = None, h2_wait_s: float | None = None,
                 homing_wait_s: float | None = None) -> None:
        self.station = Station.load(station_path or paths.default_station())
        self.settings = self.station.console
        self.mock = mock
        self.local = local
        self.only = only
        self.yes = yes
        self.out = out or sys.stdout
        self.input_fn = input_fn
        self.peers = peers
        self.link = link or LinkSettings.load(paths.contracts_dir() / "link.yaml")
        self.limits = ContractLimits.load(paths.contracts_dir() / "limits.yaml")
        self.state_ep, self.commands_ep = endpoints or (self.link.state_endpoint(local),
                                                        self.link.commands_endpoint(local))
        self.clock = clock
        self.box_min, self.box_max = workspace_box(self.station, config)
        self.h2_wait_s = 2 * self.link.watchdog_limit_s if h2_wait_s is None else h2_wait_s
        self.homing_wait_s = homing_wait_s
        self.samples: list[Sample] = []
        self.last_request_ns = 0
        self.rpc: RpcClient | None = None
        self.sub: Subscriber | None = None
        self.session: Session | None = None
        self._log: TextIO | None = None
        self._chunk = 0

    # ---- output ----

    def say(self, line: str) -> None:
        self.out.write(line + "\n")
        self.out.flush()
        if self._log:
            self._log.write(time.strftime("%H:%M:%S ") + line + "\n")
            self._log.flush()

    # ---- C2 and C3 ----

    def now(self) -> int:
        return self.clock()

    def drain(self) -> None:
        assert self.sub is not None
        while (msg := self.sub.receive(0.0)) is not None:
            self.samples.append(Sample(self.now(), msg))

    def latest(self) -> ArmState:
        self.drain()
        if not self.samples:
            raise Refused(f"no C2 on {self.state_ep}")
        return self.samples[-1].state

    def request(self, msg, expect, timeout_s: float | None = None):
        assert self.rpc is not None
        self.last_request_ns = self.now()
        return self.rpc.request(msg, expect, self.link.c3_timeout_s if timeout_s is None else timeout_s)

    def heartbeat(self) -> None:
        self.request(Heartbeat(), Ack)
        self.drain()

    def call_start(self) -> tuple[Any, int]:
        try:
            ack = self.request(Start(), Ack)
        except RemoteError as exc:
            ack = exc.reply
        t = self.now()
        self.drain()
        return ack, t

    def stop(self) -> None:
        try:
            self.request(Stop(), Ack)
        except (TransportTimeout, RemoteError, ContractError) as exc:
            self.say(f"console: stop: {exc}")

    def ensure_started(self) -> None:
        mode = self.latest().mode
        if mode not in ("holding", "running"):
            ack, _ = self.call_start()
            if not (isinstance(ack, Ack) and ack.ok):
                raise Refused(f"start refused: {getattr(ack, 'reason', ack)}")
            self.run_for(0.1)

    def current_target(self) -> tuple[np.ndarray, float]:
        s = self.latest()
        width = min(max(float(s.gripper_width_m), 0.0), self.limits.gripper_width_max_m)
        return np.array(s.O_T_TCP_target, dtype=np.float64), width

    def send_waypoints(self, points: list[wpts.PlanPoint], chunk_id: int | None = None) -> Any:
        self._chunk += 1
        msg = SetWaypoints(chunk_id=self._chunk if chunk_id is None else chunk_id,
                           t_ns=np.array([p.t_ns for p in points], dtype=np.int64),
                           O_T_TCP=np.array([p.pose for p in points], dtype=np.float64),
                           gripper_width_m=np.array([p.width_m for p in points], dtype=np.float64))
        try:
            return self.request(msg, SetWaypointsReply)
        except RemoteError as exc:
            return exc.reply

    def _sleep_cycle(self, cycle_start: int) -> None:
        left = self.link.heartbeat_period_s - (self.now() - cycle_start) * 1e-9
        if left > 0:
            time.sleep(left)

    def run_for(self, seconds: float) -> None:
        """One heartbeat per cycle (heartbeat_period_s); C2 drained every cycle."""
        end = self.now() + int(seconds * 1e9)
        while self.now() < end:
            t = self.now()
            self.heartbeat()
            self._sleep_cycle(t)

    def listen(self, seconds: float) -> None:
        """C2 only: nothing is sent (the watchdog step)."""
        end = self.now() + int(seconds * 1e9)
        while self.now() < end:
            time.sleep(0.005)
            self.drain()

    def stream(self, duration_s: float, at: Callable[[float], tuple[np.ndarray, float]], start: np.ndarray,
               width: float) -> tuple[list[Sent], wpts.PlanModel, wpts.Path]:
        """Streams a path as the manager does: one request per cycle, a set_waypoints every 0.2 s (the next 8
        waypoints, 50 ms apart, the first >= 100 ms ahead), else a heartbeat; then holds 0.5 s with heartbeats."""
        first = self.now()
        path = wpts.Path(first + wpts.LEAD_NS, duration_s, at)
        plan = wpts.PlanModel(first, start, width)
        sent: list[Sent] = []
        next_request = first
        while True:
            t = self.now()
            if t >= next_request:
                times = wpts.next_request(path, t)
                if not times:
                    break
                points = [wpts.PlanPoint(ti, *path.sample(ti)) for ti in times]
                send_ns = self.now()
                reply = self.send_waypoints(points)
                reply_ns = self.now()
                sent.append(Sent(send_ns, reply_ns, points, reply))
                if isinstance(reply, SetWaypointsReply) and reply.accepted and not reply.rejected:
                    plan.apply((send_ns + reply_ns) // 2, [p for p in points if p.t_ns > (send_ns + reply_ns) // 2])
                elif not isinstance(reply, SetWaypointsReply) or reply.rejected:
                    self.drain()
                    return sent, plan, path
                next_request += wpts.REQUEST_PERIOD_NS
                self.drain()
            else:
                self.heartbeat()
            self._sleep_cycle(t)
        self.run_for(max(0.0, (path.end_ns - self.now()) * 1e-9) + 0.5)
        return sent, plan, path

    # ---- preflight (6.5) ----

    def preflight(self) -> None:
        # 1. C2
        end = self.now() + int(C2_FIRST_S * 1e9)
        while self.now() < end and not self.samples:
            msg = self.sub.receive(0.05)
            if msg is not None:
                self.samples.append(Sample(self.now(), msg))
        if not self.samples:
            raise Refused(f"franka_control not running (no C2 on {self.state_ep}): start control_deploy run first")
        # 2. another C3 client
        peers = self.peers(self.link.commands_port)
        if peers:
            raise Refused(f"a C3 client from {', '.join(peers)} is connected (policy_deploy on the laptop?): stop it "
                          f"first")
        # the REQ socket connects at once (ss would list it as a peer): only now
        self.rpc = RpcClient("control_deploy console", self.commands_ep)
        # 3. rule H2 (deployment/STRUCTURE.md 3.7.3), homing, fault
        mode = self.latest().mode
        if mode in ("running", "holding"):
            self.say(f"console: C2 shows {mode}: waiting up to {self.h2_wait_s:g} s (rule H2)")
            mode = self._wait_mode_not(("running", "holding"), self.h2_wait_s)
            if mode in ("running", "holding"):
                raise Refused("the PC is driving an episode for another C3 client")
        if mode == "homing":
            wait = self.homing_wait_s if self.homing_wait_s is not None else 30.0
            self.say(f"console: C2 shows homing: waiting up to {wait:g} s")
            mode = self._wait_mode_not(("homing",), wait)
        if mode == "fault":
            raise Refused(f"franka_control is in fault: {self.latest().error}: Ctrl-C it and run control_deploy run "
                          f"again")
        if mode not in ("idle", "stopped"):
            raise Refused(f"C2 shows {mode}, expected idle or stopped")
        # 4. hello: the contract values, and the clock
        t_send = self.now()
        try:
            reply = self.request(Hello(), HelloReply)
        except (TransportTimeout, RemoteError, ContractError) as exc:
            raise Refused(f"hello failed: {exc}") from None
        t_recv = self.now()
        if self.homing_wait_s is None:
            self.homing_wait_s = reply.homing_timeout_s
        mismatches = compare_hello(reply, mock=self.mock)
        if mismatches:
            for m in mismatches:
                self.say(f"console: hello mismatch: {m}")
            raise Refused(f"{len(mismatches)} hello mismatch(es): the PC runs other contract values (same commit on "
                          f"both? rebuild franka_control after every pull)")
        offset = reply.t_pc_ns - (t_send + t_recv) // 2
        if abs(offset) > CLOCK_TOL_NS:
            raise Refused(f"hello t_pc_ns is {offset / 1e6:.1f} ms from this machine's clock: the console must run on "
                          f"the PC (its waypoint times use the local clock)")
        # 5. the box
        p = np.asarray(self.latest().O_T_TCP)[:3, 3]
        near = [f"{'xyz'[i]} {'min' if abs(p[i] - lo) < abs(p[i] - hi) else 'max'}"
                for i, (lo, hi) in enumerate(zip(self.box_min, self.box_max))
                if min(abs(p[i] - lo), abs(p[i] - hi)) < NEAR_FACE_M]
        if near:
            self.say(f"console: WARNING: the TCP is within {NEAR_FACE_M * 100:.0f} cm of the workspace box "
                     f"({', '.join(near)}): the moves may be pulled back")
        self.say(f"console: preflight ok: {mode}, commit {reply.commit}, mock {reply.mock}, clock offset "
                 f"{offset / 1e6:+.2f} ms")

    def _wait_mode_not(self, modes: tuple[str, ...], seconds: float) -> str:
        end = self.now() + int(seconds * 1e9)
        mode = self.latest().mode
        while mode in modes and self.now() < end:
            time.sleep(0.02)
            mode = self.latest().mode
        return mode

    # ---- the session ----

    def selected(self) -> list[str]:
        if not self.only:
            return list(ORDER)
        unknown = [s for s in self.only if s not in STEPS]
        if unknown:
            raise Refused(f"unknown step(s) {', '.join(unknown)}: choose from {', '.join(ORDER)}")
        return [s for s in ORDER if s in self.only]

    def run(self) -> int:
        if self.yes and not self.mock:
            self.say("console: --yes is refused without --mock: on the real arm each moving step waits for Enter")
            return 2
        try:
            names = self.selected()
        except Refused as exc:
            self.say(f"console: {exc}")
            return 2
        self.sub = Subscriber("control_deploy console", self.state_ep, ArmState, recv_hwm=10000)
        results: list[StepResult] = []
        try:
            try:
                self.preflight()
            except Refused as exc:
                self.say(f"console: refused: {exc}")
                return 2
            self.session = Session.create(self.station.session_root, "control_deploy_console", self.mock)
            self._log = open(self.session.dir / "console.log", "a", encoding="utf-8")
            self.say(f"console: session {self.session.dir}; steps {', '.join(names)}")
            for name in names:
                step = STEPS[name]()
                if step.moves and not self.yes:
                    self.input_fn(f"console: next: {name}: {step.describe(self)}. Hand on the user stop, then press "
                                  f"Enter (Ctrl-C to stop) ")
                else:
                    self.say(f"console: {name}: {step.describe(self)}")
                try:
                    result = step.run(self)
                except Refused as exc:
                    result = StepResult(name, "fail", str(exc))
                except (TransportTimeout, RemoteError, ContractError) as exc:
                    result = StepResult(name, "fail", f"C3: {exc}")
                results.append(result)
                self.say(f"console: {result.line()}")
                if result.status == "fail":
                    if name != "out_of_box":
                        self.stop()
                    break
            if "out_of_box" in [r.name for r in results]:
                self.say("console: franka_control is in fault by design (out_of_box): Ctrl-C control_deploy run and "
                         "start it again")
            elif results and results[-1].status != "fail":
                self.stop()
        except KeyboardInterrupt:
            self.say("console: Ctrl-C: sending stop")
            self.stop()
            self._write(results, interrupted=True)
            return 130
        finally:
            if self.rpc:
                self.rpc.close()
            self.sub.close()
        self._write(results)
        failed = any(r.status == "fail" for r in results)
        self.say("console: " + ("FAILED: see results.yaml" if failed else "all steps done") +
                 (f" ({self.session.dir / 'results.yaml'})" if self.session else ""))
        return 1 if failed else 0

    def _write(self, results: list[StepResult], interrupted: bool = False) -> None:
        if not self.session:
            return
        self.session.write_yaml("results.yaml", {
            "mock": self.mock, "local": self.local, "interrupted": interrupted,
            "settings": {k: getattr(self.settings, k) for k in self.settings.__dataclass_fields__},
            "steps": [{"name": r.name, "status": r.status, "detail": r.detail, "numbers": r.numbers}
                      for r in results],
        })
        if self._log:
            self._log.close()
            self._log = None
