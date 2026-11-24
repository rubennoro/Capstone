"""FakePc: the PC as the laptop sees it, for --mock and the tests (STRUCTURE.md 10.1).

Two threads: "fake-pc-state" publishes C2 at state_rate_hz on the drifting clock, and "fake-pc-commands" answers C3
through deploy_contracts.RpcServer with schemas=SHARED_SCHEMAS (a laptop-only schema is unknown_schema, as on the PC).
Replies per mode exactly as TOP 3.7.2. Every request that reaches a handler feeds the watchdog; broken ones never do.
A C3 home blocks the REP socket until the move ends (as the PC). The test hooks (inject_fault, restart, stall, ...)
and the PC keyboard (keyboard_home, keyboard_hand_home, keyboard_stop) are methods.

As the real franka_control (fix wave): a set whose waypoints are all skipped freezes the arm at the current target and
goes `holding`; C2 t_ns freezes only when the 1 kHz loop has ended (inject_fault, start_mode fault), not after a gate
rejection, where the real loop keeps running; refusals carry the real texts (StateMachine::check: "refused: mode is
<mode>[: <why>]", the fault detail, a stopped mode's cause; "set_waypoints refused: ...").

Not modelled (10.2): libfranka, dynamics, the workspace box and tilt cone, speed shortening (only counters via hooks),
joints, gripper physics beyond a width ramp, a real-time loop, the keyboard itself, bind addresses and firewall.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from deploy_contracts import (SHARED_SCHEMAS, Ack, ArmState, ErrorReply, GateCounters, GetState, Heartbeat, Hello,
                              HelloReply, Home, Message, Publisher, RpcServer, SetWaypoints, SetWaypointsReply, Start,
                              Stop)

from policy_deploy.contracts import paths
from policy_deploy.fake_pc.drifting_clock import DriftingClock
from policy_deploy.fake_pc.fake_gate import FakeGate
from policy_deploy.fake_pc.plan_follower import PlanFollower, interpolate
from policy_deploy.fake_pc.settings import FakePcSettings

log = logging.getLogger("fake_pc")
HOME_STOPPED_REASON = "stopped: homing stopped by the PC keyboard (fake PC)"


def laptop_commit() -> str:
    """The checkout's `git rev-parse HEAD` (40 hex), or "unknown"."""
    try:
        out = subprocess.run(["git", "-C", str(paths.deployment_dir()), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, stdin=subprocess.DEVNULL)
        commit = out.stdout.strip()
        if out.returncode == 0 and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
            return commit
    except (OSError, subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


@dataclass
class _Homing:
    kind: str                     # "c3" (a C3 home), "arm" (the PC's h), "hand" (the PC's g)
    start_ns: int                 # laptop ns
    end_ns: int
    T0: np.ndarray
    T1: np.ndarray
    width_back_m: float = 0.0     # hand: the width to return to
    outcome: str | None = None    # "done" | "stopped" (read by a pending C3 home)


class FakePc:
    def __init__(self, settings: FakePcSettings, session_dir: Path | None = None,
                 laptop_clock: Callable[[], int] = time.monotonic_ns, commit: str | None = None) -> None:
        self.s = settings
        self._laptop = laptop_clock
        self.clock = DriftingClock(settings.clock_offset_s, settings.clock_drift_ppm, laptop_clock)
        self._gate = FakeGate(settings.max_lead_s, settings.limits.gripper_width_max_m)
        self._follower = PlanFollower(settings.start_pose, settings.start_width_m, settings.width_speed_m_s)
        self._commit = laptop_commit() if commit is None else commit
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.ready = threading.Event()               # the first C2 is out
        self._mode = settings.start_mode
        self._error = "" if settings.start_mode != "fault" else "fake fault (start_mode)"
        self._loop_ended = settings.start_mode == "fault"   # C2 t_ns frozen (the real loop ended)
        self._stopped_detail = "stop"                # why the mode is stopped (the real PC's refusal names it)
        self._homing: _Homing | None = None
        if settings.start_mode == "homing":
            now = laptop_clock()
            self._homing = _Homing("arm", now, now + int(settings.home_duration_s * 1e9), self._follower.target.copy(),
                                   settings.home_pose.copy())
        self._seq = 0                                # the last broadcast's seq
        self._next_seq = 1
        self._last_t_pc: int | None = None
        self._frozen_until_ns = 0                    # repeat_t_ns: laptop time until which t_ns is frozen
        self._counters = {"accepted": 0, "shortened": 0, "pulled_back": 0, "rejected": 0}
        self._last_request_ns = laptop_clock()
        self._last_update_ns = laptop_clock()
        self._hold_forever = False
        self._hold_until_ns = laptop_clock() + int(settings.start_hold_s * 1e9)   # start_hold_s
        self._c2_stall_until = 0
        self._c3_stall_until = 0
        self._burst = False
        self._backlog: list[ArmState] = []
        self._drop = 0
        self._delay: Callable[[], float] | None = None
        self._delayed: deque[tuple[int, ArmState]] = deque()
        self.requests: list[tuple[int, str]] = []    # (laptop ns, schema) of every request that reached a handler
        self._log_file = None
        if session_dir is not None:
            Path(session_dir).mkdir(parents=True, exist_ok=True)
            self._log_file = open(Path(session_dir) / "commands.jsonl", "a", encoding="utf-8")
        self._pub: Publisher | None = None
        self._server: RpcServer | None = None
        self._threads: list[threading.Thread] = []

    # ---------------------------------------------------------------------------------------------- lifecycle

    def bind(self) -> None:
        """Bind C2 PUB and C3 REP (the endpoints of the settings)."""
        self._pub = Publisher(self.s.state_endpoint, self.s.state_send_hwm)
        handlers = {Hello: self._wrap(self._on_hello), GetState: self._wrap(self._on_get_state),
                    Heartbeat: self._wrap(self._on_heartbeat), Start: self._wrap(self._on_start),
                    Stop: self._wrap(self._on_stop), Home: self._wrap(self._on_home),
                    SetWaypoints: self._wrap(self._on_set_waypoints)}
        self._server = RpcServer(self.s.commands_endpoint, handlers, schemas=SHARED_SCHEMAS)

    @property
    def state_endpoint(self) -> str:
        return self._pub.endpoint if self._pub is not None else self.s.state_endpoint

    @property
    def commands_endpoint(self) -> str:
        return self._server.endpoint if self._server is not None else self.s.commands_endpoint

    def start(self) -> None:
        """Bind (if not yet) and start both threads."""
        if self._pub is None:
            self.bind()
        self._threads = [threading.Thread(target=self._state_loop, name="fake-pc-state", daemon=True),
                         threading.Thread(target=self._commands_loop, name="fake-pc-commands", daemon=True)]
        for t in self._threads:
            t.start()

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        for sock in (self._pub, self._server):
            if sock is not None:
                sock.close()
        self._pub = self._server = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> FakePc:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------------------------------------- state

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def target(self) -> np.ndarray:
        with self._lock:
            return self._follower.target.copy()

    @property
    def width_m(self) -> float:
        with self._lock:
            return self._follower.width_m

    def request_names(self) -> list[str]:
        with self._lock:
            return [name for _, name in self.requests]

    def _state_loop(self) -> None:
        period = 1.0 / self.s.state_rate_hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            self.tick()
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()

    def tick(self) -> None:
        """One C2 cycle: advance the modes and the plan, then publish one state (unless stalled or dropped)."""
        now = self._laptop()
        with self._lock:
            self._update(now)
            state = self._make_state(now, advance_seq=True)
            stalled = now < self._c2_stall_until
            to_send: list[ArmState] = []
            if state is not None:
                if stalled:
                    if self._burst:
                        self._backlog.append(state)
                elif self._drop > 0:
                    self._drop -= 1
                else:
                    if self._backlog:
                        to_send.extend(self._backlog)
                        self._backlog = []
                        self._burst = False
                    to_send.append(state)
            if self._delay is not None:
                for st in to_send:
                    release = now + int(max(0.0, self._delay()) * 1e9)
                    if self._delayed:
                        release = max(release, self._delayed[-1][0])
                    self._delayed.append((release, st))
                to_send = []
            while self._delayed and self._delayed[0][0] <= now:
                to_send.append(self._delayed.popleft()[1])
        if self._pub is not None:
            for st in to_send:
                self._pub.publish(st)
            if to_send:
                self.ready.set()

    def _now_pc(self, now_laptop: int) -> int:
        """The PC time for a C2 reading: frozen once the loop has ended and during repeat_t_ns."""
        t = self.clock.at(now_laptop)
        if self._loop_ended or now_laptop < self._frozen_until_ns:
            if self._last_t_pc is not None:
                return self._last_t_pc
        return t

    def _make_state(self, now_laptop: int, advance_seq: bool) -> ArmState | None:
        t_pc = self._now_pc(now_laptop)
        if self._last_t_pc is not None and t_pc < self._last_t_pc:
            t_pc = self._last_t_pc
        self._last_t_pc = t_pc
        if advance_seq:
            seq = self._next_seq
            self._next_seq += 1
            self._seq = seq
        else:
            seq = max(self._seq, 1)
        target = self._follower.target
        measured = target.copy()
        measured[0, 3] += self.s.measured_offset_m
        c = self._counters
        return ArmState(t_ns=int(t_pc), seq=int(seq), O_T_TCP=measured, O_T_TCP_target=target.copy(),
                        q=np.zeros(7), dq=np.zeros(7), gripper_width_m=float(max(0.0, self._follower.width_m)),
                        mode=self._mode, error=self._error,
                        gate=GateCounters(accepted=c["accepted"], shortened=c["shortened"],
                                          pulled_back=c["pulled_back"], rejected=c["rejected"]),
                        loop_late=0)

    def _update(self, now: int) -> None:
        """Advance homing ramps, the plan, the width and the watchdog to laptop time `now` (under the lock)."""
        dt = max(0.0, (now - self._last_update_ns) / 1e9)
        self._last_update_ns = now
        if self._hold_forever or now < self._hold_until_ns:
            self._last_request_ns = now
        if self._mode == "homing" and self._homing is not None:
            h = self._homing
            a = (now - h.start_ns) / max(1, h.end_ns - h.start_ns)
            if h.kind == "hand":
                half = 0.5
                goal = 0.0 if a < half else h.width_back_m
                self._follower.move_width(goal, dt)
            else:
                self._follower.target = interpolate(h.T0, h.T1, a)
            if now >= h.end_ns:
                self._finish_homing(now)
            return
        if self._mode in ("holding", "running"):
            ended = self._follower.step(self.clock.at(now), dt)
            if ended and self._mode == "running":
                self._mode = "holding"
            if now - self._last_request_ns > self.s.watchdog_limit_s * 1e9:
                log.warning("watchdog: no C3 request for %.2f s: stopped", (now - self._last_request_ns) / 1e9)
                self._follower.hold()
                self._mode = "stopped"
                self._stopped_detail = f"watchdog: no request for {(now - self._last_request_ns) / 1e9:.3f} s"
        else:
            self._follower.move_width(self._follower.commanded_width_m, dt)

    def _finish_homing(self, now: int) -> None:
        h = self._homing
        if h is None:
            return
        if h.kind == "hand":
            self._follower.width_m = h.width_back_m
        else:
            self._follower.set_target(h.T1)
        h.outcome = "done"
        self._homing = None
        self._mode = "idle"

    def _abort_homing(self, source: str) -> None:
        """A stop during homing: hold where the ramp is, mode stopped (the hand: the width where it is). source: "C3"
        or "PC keyboard"."""
        h = self._homing
        if h is None:
            return
        self._follower.set_target(self._follower.target)
        self._follower.commanded_width_m = self._follower.width_m
        h.outcome = "stopped"
        self._homing = None
        self._mode = "stopped"
        self._stopped_detail = f"stop during {'hand ' if h.kind == 'hand' else ''}homing ({source})"

    def _stop_now(self) -> None:
        """idle/holding/running -> stopped, holding the current target."""
        self._follower.hold()
        self._mode = "stopped"
        self._stopped_detail = "stop"

    def _refused(self, why: str) -> str:
        """The real StateMachine::check refusal: "refused: mode is <mode>", then ": <fault detail>" in fault, or
        " (<cause>)" when stopped by something other than a stop request, then ": <why>" (not in fault)."""
        r = f"refused: mode is {self._mode}"
        if self._mode == "fault":
            r += ": " + (self._error or "fault")
        elif self._mode == "stopped" and self._stopped_detail not in ("", "stop"):
            r += f" ({self._stopped_detail})"
        if why and self._mode != "fault":
            r += ": " + why
        return r

    def _home_refusal(self) -> str:
        return self._refused({"holding": "send stop first", "running": "send stop first",
                              "homing": "homing in progress"}.get(self._mode, ""))

    # ---------------------------------------------------------------------------------------------- C3 handlers

    def _wrap(self, handler: Callable[[Message], Message]) -> Callable[[Message], Message]:
        def run(msg: Message) -> Message:
            now = self._laptop()
            with self._lock:
                self._last_request_ns = now                  # every request that reaches a handler feeds it
                self.requests.append((now, msg.SCHEMA))
                # no _update here: the C2 thread advances the modes, so a stop that arrives after a homing ramp has
                # ended but before the C2 thread noticed still sees `homing` and the normal end wins (idle)
            reply = handler(msg)
            self._log_command(now, msg, reply)
            return reply
        return run

    def handle(self, msg: Message) -> Message:
        """Answer one request in-process, as the C3 thread would (tests)."""
        table = {Hello: self._on_hello, GetState: self._on_get_state, Heartbeat: self._on_heartbeat,
                 Start: self._on_start, Stop: self._on_stop, Home: self._on_home, SetWaypoints: self._on_set_waypoints}
        return self._wrap(table[type(msg)])(msg)

    def _commands_loop(self) -> None:
        while not self._stop.is_set():
            if self._laptop() < self._c3_stall_until:
                self._stop.wait(self.s.poll_s)
                continue
            try:
                self._server.serve_once(self.s.poll_s)
            except Exception as exc:                          # a closed socket at shutdown
                if not self._stop.is_set():
                    log.exception("C3: %s", exc)
                return

    def _on_hello(self, msg: Hello) -> HelloReply:
        lim = self.s.limits
        fields = dict(commit=self._commit, mock=True, schemas=SHARED_SCHEMAS,
                      F_T_TCP=np.array(self.s.frame.F_T_TCP, dtype=np.float64), rotation_tol=lim.rotation_tol,
                      gripper_width_max_m=lim.gripper_width_max_m, max_waypoints=lim.max_waypoints,
                      watchdog_limit_s=self.s.watchdog_limit_s, homing_timeout_s=self.s.homing_timeout_s,
                      max_lead_s=self.s.max_lead_s)
        return HelloReply(t_pc_ns=self.clock(), **fields)        # stamped immediately before the reply

    def _on_get_state(self, msg: GetState) -> ArmState:
        with self._lock:
            return self._make_state(self._laptop(), advance_seq=False)

    def _on_heartbeat(self, msg: Heartbeat) -> Ack:
        return Ack(ok=True, reason="")

    def _on_start(self, msg: Start) -> Ack:
        with self._lock:
            if self._mode in ("idle", "stopped"):
                self._follower.hold()
                self._mode = "holding"
                return Ack(ok=True, reason="")
            return Ack(ok=False, reason=self._refused({"holding": "already started", "running": "already started",
                                                       "homing": "homing in progress"}.get(self._mode, "")))

    def _on_stop(self, msg: Stop) -> Ack:
        with self._lock:
            now = self._laptop()
            if self._mode == "homing":
                h = self._homing
                if h is not None and now >= h.end_ns:
                    self._finish_homing(now)                  # the move finished first: the normal end wins
                else:
                    self._abort_homing("C3")
            elif self._mode in ("idle", "holding", "running"):
                self._stop_now()
            return Ack(ok=True, reason="")

    def _on_home(self, msg: Home) -> Ack:
        with self._lock:
            if self._mode not in ("idle", "stopped"):
                return Ack(ok=False, reason=self._home_refusal())
            now = self._laptop()
            h = _Homing("c3", now, now + int(self.s.home_duration_s * 1e9), self._follower.target.copy(),
                        self.s.home_pose.copy())
            self._homing = h
            self._follower.hold()
            self._mode = "homing"
        while not self._stop.is_set():                        # the REP socket answers nothing else meanwhile
            with self._lock:
                self._update(self._laptop())
                if h.outcome is not None:
                    break
            time.sleep(min(self.s.poll_s, 0.005))
        if h.outcome == "stopped":
            return Ack(ok=False, reason=HOME_STOPPED_REASON)
        if h.outcome == "done":
            return Ack(ok=True, reason="")
        return Ack(ok=False, reason="homing interrupted: the fake PC is shutting down")

    def _on_set_waypoints(self, msg: SetWaypoints) -> Message:
        with self._lock:
            if self._mode not in ("holding", "running"):
                why = {"idle": "send start first", "stopped": "send start first",
                       "homing": "homing in progress"}.get(self._mode, "")
                return ErrorReply(request=SetWaypoints.SCHEMA, error=f"set_waypoints {self._refused(why)}", field="")
            now_pc = self.clock()
            result = self._gate.check(msg, now_pc)
            if result.rejected:
                self._counters["rejected"] += sum(1 for i in result.rejected)
                self._mode = "fault"                          # the loop keeps holding: C2 t_ns keeps advancing
                self._error = result.fault or "gate rejected a waypoint"
                self._follower.hold()
            elif result.accepted:
                self._counters["accepted"] += len(result.accepted)
                self._follower.replace(now_pc, [(int(msg.t_ns[i]), msg.O_T_TCP[i], float(msg.gripper_width_m[i]))
                                                for i in result.accepted])
                self._mode = "running"
            else:                                             # all skipped: freeze at the current target, holding
                self._follower.step(now_pc, 0.0)              # the target at now (the real PC: plan.at(now))
                self._follower.hold()
                self._mode = "holding"
            return SetWaypointsReply(chunk_id=int(msg.chunk_id), accepted=len(result.accepted), shortened=0,
                                     skipped=int(result.skipped), rejected=result.rejected)

    def _log_command(self, now: int, msg: Message, reply: Message) -> None:
        if self._log_file is None:
            return
        row = {"t_ns": now, "t_pc_ns": self.clock.at(now), "request": msg.SCHEMA, "reply": reply.SCHEMA}
        if isinstance(msg, SetWaypoints):
            row.update(chunk_id=int(msg.chunk_id), n=len(msg.t_ns), t_first_pc_ns=int(msg.t_ns[0]),
                       t_last_pc_ns=int(msg.t_ns[-1]))
        if isinstance(reply, SetWaypointsReply):
            row.update(accepted=int(reply.accepted), skipped=int(reply.skipped), rejected=len(reply.rejected))
        elif isinstance(reply, Ack):
            row.update(ok=bool(reply.ok), reason=reply.reason)
        elif isinstance(reply, ErrorReply):
            row.update(error=reply.error)
        elif isinstance(reply, ArmState):
            row.update(mode=reply.mode, seq=int(reply.seq))
        with self._lock:
            if self._log_file is not None:
                self._log_file.write(json.dumps(row) + "\n")
                self._log_file.flush()

    # ---------------------------------------------------------------------------------------------- test hooks

    def inject_fault(self, text: str) -> None:
        """A fault that ends the 1 kHz loop (a reflex, a libfranka exception): C2 t_ns freezes from here on."""
        with self._lock:
            self._mode = "fault"
            self._error = text
            self._loop_ended = True
            self._follower.hold()

    def restart(self) -> None:
        """franka_control restarted: seq starts again at 1 (t_ns keeps going: CLOCK_MONOTONIC)."""
        with self._lock:
            self._next_seq = 1
            self._seq = 0

    def stall(self, seconds: float, c2: bool = True, c3: bool = True) -> None:
        """Neither C2 nor C3 (or only one of them) for that long."""
        until = self._laptop() + int(seconds * 1e9)
        with self._lock:
            if c2:
                self._c2_stall_until = until
            if c3:
                self._c3_stall_until = until

    def burst_after_stall(self) -> None:
        """The states of the current/next C2 stall are kept and sent all at once when it ends (old t_ns)."""
        with self._lock:
            self._burst = True

    def delay_states(self, dist: Callable[[], float] | None) -> None:
        """Each C2 message is held back dist() seconds (order kept, as on TCP); None turns it off."""
        with self._lock:
            self._delay = dist

    def drop_states(self, n: int) -> None:
        """The next n broadcasts use their seq but are not sent (a seq gap)."""
        with self._lock:
            self._drop += int(n)

    def repeat_t_ns(self, seconds: float) -> None:
        """C2 keeps coming but t_ns stays the same for that long (as after a stopped homing)."""
        with self._lock:
            self._frozen_until_ns = self._laptop() + int(seconds * 1e9)

    def set_mode(self, mode: str) -> None:
        """Force a mode (e.g. an orphaned holding for the H2 wait); the watchdog is re-armed from now."""
        with self._lock:
            self._mode = mode
            self._last_request_ns = self._laptop()
            self._stopped_detail = "stop"
            if mode != "fault":
                self._error = ""

    def hold_forever(self, on: bool = True) -> None:
        """Another C3 client keeps feeding the watchdog (an episode that stays holding)."""
        with self._lock:
            self._hold_forever = on

    def shift_clock(self, seconds: float) -> None:
        self.clock.shift(seconds)

    def report_shortened(self, n: int) -> None:
        with self._lock:
            self._counters["shortened"] += int(n)
            self._counters["accepted"] = max(self._counters["accepted"], self._counters["shortened"])

    def report_pulled_back(self, n: int) -> None:
        with self._lock:
            self._counters["pulled_back"] += int(n)
            self._counters["accepted"] = max(self._counters["accepted"], self._counters["pulled_back"])

    # ---------------------------------------------------------------------------------------------- the PC keyboard

    def keyboard_home(self) -> Ack:
        """The PC's h: refused unless idle/stopped; else homing (the REP keeps answering), then idle."""
        with self._lock:
            if self._mode not in ("idle", "stopped"):
                return Ack(ok=False, reason=self._home_refusal())
            now = self._laptop()
            self._homing = _Homing("arm", now, now + int(self.s.home_duration_s * 1e9),
                                   self._follower.target.copy(), self.s.home_pose.copy())
            self._follower.hold()
            self._mode = "homing"
            return Ack(ok=True, reason="")

    def keyboard_hand_home(self) -> Ack:
        """The PC's g: refused unless idle/stopped; else homing with the arm held, the width to 0 and back."""
        with self._lock:
            if self._mode not in ("idle", "stopped"):
                return Ack(ok=False, reason=self._home_refusal())
            now = self._laptop()
            back = self._follower.commanded_width_m
            duration = 2.0 * max(back, 1e-3) / self.s.width_speed_m_s + 0.1   # + a short rest at 0
            T = self._follower.target.copy()
            self._homing = _Homing("hand", now, now + int(duration * 1e9), T, T, width_back_m=back)
            self._follower.hold()
            self._mode = "homing"
            return Ack(ok=True, reason="")

    def keyboard_stop(self) -> Ack:
        """The PC's s: idle/holding/running -> stopped; no change in stopped and fault; aborts any homing."""
        with self._lock:
            if self._mode == "homing":
                self._abort_homing("PC keyboard")
            elif self._mode in ("idle", "holding", "running"):
                self._stop_now()
            return Ack(ok=True, reason="")
