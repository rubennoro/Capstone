# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/manager_fakes.py, manager_harness.py,
# manager_raw_server.py, manager_pc_harness.py and manager_pc_fakepc.py @ 2ffe641
# changes: teleop (STRUCTURE.md 12 row E3):
# - from manager_fakes: FakeClock, rz unchanged; pose() defaults to the TCP pointing down at (0.45, 0, 0.30) (was
#   identity at (0.5, 0, 0.3)); make_state defaults to mode idle and width 0.08 (were running, 0.05) and takes q;
#   + health / frames_health / summary for R1; hello_reply takes link; FakePcClient as before (hello on a fake PC
#   clock = laptop + 1000 s, home() returns its Ack: ok / stopped / refused_homing / refuse / hang, during_home) plus
#   counted interrupted() calls and stop() -> stopped only from running / holding; rejecting_reply names the last
#   waypoint; FakeListener + set_recorder / recorder_error and LinkStats; FakeFramesClient is R1 (start / stop
#   recording, FramesHealth); the policy, worker and operator fakes are gone;
# - from manager_harness: TeleopHarness (was LoopHarness) wires the real TeleopLoop, EpisodeFlow, E4's motion and
#   episode classes and a real TaskStore under tmp_path to those fakes on a FakeClock; + state_delay_ns, q, width, the
#   U1 helpers (u1, keys, hold, status, ready, start_episode); load_settings takes teleop: overrides;
# - from manager_pc_harness: FAST (the same values) and merge(); PcRig's role is SocketPc below plus test_sockets.Rig;
# - from manager_pc_fakepc: wait_for (unchanged); the in-process FakePc is replaced by SocketPc, a small fake PC on
#   real sockets (C2 PUB at 100 Hz, C3 REP) for the start-up, listener and pc_probe tests (the real fake PC is E6's);
# - from manager_raw_server: RawServer, unchanged.
# + start pose (final-deployment's start-pose work, 927a5ba, 2026-11-20): FakePcClient has no home()
#   any more (teleop sends no C3 home: no home_behaviour, during_home, interrupted_calls); + TeleopHarness.pc_move
#   (the PC's h or p as C2 shows it: homing while the arm moves, then idle).
# The sources' own headers: manager_fakes.py "adapted from rnoro/deployment:deployment/message_comms/tests/fakes.py
# @ 91b178b"; manager_harness.py "adapted from rnoro/deployment:deployment/message_comms/tests/harness.py @ 91b178b";
# manager_raw_server.py "copied from rnoro/deployment:deployment/message_comms/tests/raw_server.py @ 91b178b";
# manager_pc_harness.py and manager_pc_fakepc.py were new in rnoro/final-deployment.
"""Fakes and harnesses for the teleop tests. Same methods as the real classes; the harness has no sockets."""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import zmq
from deploy_contracts import (SHARED_SCHEMAS, Ack, ArmState, ContractLimits, GateCounters, HelloReply, LinkSettings,
                              Publisher, RpcServer, SetWaypointsReply, TcpFrame, WaypointIssue, encode)
from station_helpers import write_station

from teleop_collect.contracts import (CameraHealthReport, CameraSummary, FramesHealth, HealthReport,
                                      RecordingSummary)
from teleop_collect.teleop.episode.store import TaskStore
from teleop_collect.teleop.failure_monitor import FailureMonitor
from teleop_collect.teleop.heartbeat_sender import HeartbeatSender
from teleop_collect.teleop.link_stats import LinkStats
from teleop_collect.teleop.pc_clock import HelloSample, make_pc_clock
from teleop_collect.teleop.session_log import SessionLog
from teleop_collect.teleop.settings import Settings
from teleop_collect.teleop.startup_checks import StartupChecks
from teleop_collect.teleop.state_history import StateHistory
from teleop_collect.teleop.teleop_loop import Phase, TeleopLoop

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
PC_OFFSET_NS = 1_000 * NS_PER_S          # the fake PC clock runs 1000 s ahead of the laptop's
COMMIT = "0123456789abcdef0123456789abcdef01234567"
#: settings that make the start-up quick in tests (fake time: any value works; real time: fewer samples)
FAST = {"clock": {"min_samples_s": 0.2, "min_samples": 5}, "timeouts": {"startup_s": 5.0, "pc_homing_wait_s": 3.0},
        "health_poll_ms": 100}


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_settings(tmp_path: Path, mock: bool = True, **teleop) -> Settings:
    """Settings from a temporary station whose teleop: section is `teleop`."""
    return Settings.load(write_station(Path(tmp_path), sections={"teleop": teleop}), mock=mock)


def wait_for(condition: Callable[[], bool], timeout_s: float = 5.0, poll_s: float = 0.005) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(poll_s)
    return condition()


# ---- clock ---------------------------------------------------------------------------------------------------------

class FakeClock:
    """A monotonic ns clock driven by the test. sleep() advances it."""

    def __init__(self, start_ns: int = 1_000 * NS_PER_S) -> None:
        self.now = start_ns

    def __call__(self) -> int:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += int(round(seconds * NS_PER_S))

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


# ---- builders ------------------------------------------------------------------------------------------------------

DOWN = np.diag([1.0, -1.0, -1.0])        # the TCP z axis pointing down, as at home


def rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def pose(p=(0.45, 0.0, 0.30), R=None) -> np.ndarray:
    """A TCP pose; default rotation DOWN."""
    T = np.eye(4)
    T[:3, :3] = DOWN if R is None else R
    T[:3, 3] = p
    return T


def make_state(seq: int, t_ns: int, T=None, target=None, width: float = 0.08, mode: str = "idle", error: str = "",
               q=None, shortened: int = 0, pulled_back: int = 0, loop_late: int = 0) -> ArmState:
    T = pose() if T is None else T
    target = T if target is None else target
    accepted = max(shortened, pulled_back)
    return ArmState(t_ns=int(t_ns), seq=int(seq), O_T_TCP=np.array(T, dtype=np.float64),
                    O_T_TCP_target=np.array(target, dtype=np.float64),
                    q=np.zeros(7) if q is None else np.array(q, dtype=np.float64), dq=np.zeros(7),
                    gripper_width_m=float(width), mode=mode, error=error,
                    gate=GateCounters(accepted=accepted, shortened=shortened, pulled_back=pulled_back, rejected=0),
                    loop_late=int(loop_late))


def camera(status: str = "ok") -> CameraHealthReport:
    return CameraHealthReport(status=status, last_frame_age_ms=20.0 if status == "ok" else 500.0,
                              fps=30.0 if status == "ok" else 0.0, dropped=0)


def health(main: str = "ok", ultrawide: str = "ok") -> HealthReport:
    return HealthReport(main=camera(main), ultrawide=camera(ultrawide))


def frames_health(main: str = "ok", ultrawide: str = "ok", recording: bool = False, error: str = "",
                  frames_main: int = 0) -> FramesHealth:
    return FramesHealth(health=health(main, ultrawide), synced=True, recording=recording, recording_error=error,
                        frames_main=frames_main, frames_ultrawide=frames_main)


def summary(frames: int = 30, error: str = "") -> RecordingSummary:
    def cam(n):
        return CameraSummary(frames=n, frame_no_gaps=0, bytes=n * 1000, first_t_mid_ns=1 if n else None,
                             last_t_mid_ns=2 if n else None)
    return RecordingSummary(main=cam(frames), ultrawide=cam(frames), packets=frames, packet_no_gaps=0, uw_reused=0,
                            pair_refused=0, error=error, pi_clock_offset_ns=5, pi_clock_drift_ppm=1.0)


def hello_reply(t_pc_ns: int, mock: bool = True, commit: str = COMMIT, link: LinkSettings | None = None,
                **change) -> HelloReply:
    """A hello reply equal to this checkout's contract files."""
    lim, frame = ContractLimits.load(), TcpFrame.load()
    link = LinkSettings.load() if link is None else link
    d = dict(t_pc_ns=int(t_pc_ns), commit=commit, mock=mock, schemas=SHARED_SCHEMAS,
             F_T_TCP=np.array(frame.F_T_TCP, dtype=np.float64), rotation_tol=lim.rotation_tol,
             gripper_width_max_m=lim.gripper_width_max_m, max_waypoints=lim.max_waypoints,
             watchdog_limit_s=link.watchdog_limit_s, homing_timeout_s=30.0, max_lead_s=1.5)
    d.update(change)
    return HelloReply(**d)


# ---- clients -------------------------------------------------------------------------------------------------------

class FakePcClient:
    """PcClient stand-in: records every call with the laptop clock time; keeps the mode the PC would broadcast; its
    clock is the laptop fake clock + PC_OFFSET_NS (+ offset changes)."""

    def __init__(self, clock: FakeClock, poll_s: float = 0.02) -> None:
        self.clock = clock
        self.calls: list[str] = []
        self.timed: list[tuple[str, int]] = []
        self.mode = "idle"
        self.offset_ns = PC_OFFSET_NS
        self.rtt_ns = 1 * NS_PER_MS
        self.hello_change: dict = {}
        self.start_ack = Ack(ok=True, reason="")
        self.poll_s = poll_s
        self.reply: Callable | None = None           # set_waypoints(msg) -> SetWaypointsReply; default: by time
        self.sent: list = []
        self.error: dict[str, BaseException] = {}
        self.last_rtt_ns: int | None = None

    def pc_ns(self, laptop_ns: int | None = None) -> int:
        return int(self.clock() if laptop_ns is None else laptop_ns) + self.offset_ns

    def _record(self, name: str) -> None:
        self.calls.append(name)
        self.timed.append((name, self.clock()))
        if name in self.error:
            raise self.error[name]

    def hello(self) -> tuple[HelloReply, HelloSample]:
        t_send = self.clock()
        self._record("hello")
        self.clock.now += self.rtt_ns // 2
        t_pc = self.pc_ns()
        self.clock.now += self.rtt_ns - self.rtt_ns // 2
        t_recv = self.clock()
        self.last_rtt_ns = t_recv - t_send
        return hello_reply(t_pc, **self.hello_change), HelloSample(t_send, t_pc, t_recv)

    def heartbeat(self) -> None:
        self._record("heartbeat")

    def start(self) -> Ack:
        self._record("start")
        if self.start_ack.ok:
            self.mode = "holding"
        return self.start_ack

    def stop(self) -> None:
        self._record("stop")
        if self.mode in ("running", "holding"):
            self.mode = "stopped"

    def set_waypoints(self, msg) -> SetWaypointsReply:
        self._record("set_waypoints")
        self.sent.append((self.clock(), msg))
        self.mode = "running"
        if self.reply is not None:
            return self.reply(msg)
        n = len(msg.t_ns)
        past = int(np.sum(msg.t_ns < self.pc_ns()))
        return SetWaypointsReply(chunk_id=msg.chunk_id, accepted=n - past, shortened=0, skipped=past, rejected=())

    def get_state(self):
        self._record("get_state")
        return make_state(1, self.pc_ns(), mode=self.mode)

    def close(self) -> None:
        pass


def rejecting_reply(msg) -> SetWaypointsReply:
    n = len(msg.t_ns)
    issues = tuple(WaypointIssue(index=i, reason="outside the box by 4 cm" if i == n - 1 else
                                 "not checked: another waypoint of this request was rejected") for i in range(n))
    return SetWaypointsReply(chunk_id=msg.chunk_id, accepted=0, shortened=0, skipped=0, rejected=issues)


class FakeListener:
    """Pushes states into a real StateHistory, the PC clock map and the recorder, as the C2 thread would."""

    def __init__(self, history, clock: Callable[[], int], pc_clock=None, stats: LinkStats | None = None) -> None:
        self.history = history
        self.stats = stats
        self.clock = clock
        self.pc_clock = pc_clock
        self._error: BaseException | None = None
        self._recorder = None
        self._recorder_error: BaseException | None = None
        self._last_arrival: int | None = None
        self.started = False
        self.stopped = False
        self.recorder_sets: list = []

    def push(self, state: ArmState) -> None:
        try:
            self.history.add(state)
            if self.pc_clock is not None:
                self.pc_clock.add_state(int(state.t_ns), self.clock())
            if self.stats is not None:
                self.stats.c2(self.clock(), int(state.seq))
            self._last_arrival = self.clock()
        except Exception as exc:
            self._error = exc
            return
        if self._recorder is not None:
            try:
                t_ns = self.pc_clock.snapshot().to_laptop_ns(int(state.t_ns), self.clock()) \
                    if self.pc_clock is not None and self.pc_clock.has_fit() else None
                self._recorder.write(state, self.clock(), t_ns)
            except Exception as exc:
                self._recorder_error = exc
                self._recorder = None

    def set_recorder(self, recorder) -> None:
        self._recorder = recorder
        self._recorder_error = None
        self.recorder_sets.append(recorder)

    def recorder_error(self):
        return self._recorder_error

    def set_error(self, exc: BaseException) -> None:
        self._error = exc

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def error(self) -> BaseException | None:
        return self._error

    def last_arrival_ns(self) -> int | None:
        return self._last_arrival


class FakeFramesClient:
    """R1 stand-in: `cameras` sets both statuses; start refusals; the summary; `error` makes a call raise."""

    def __init__(self) -> None:
        self.main = "ok"
        self.ultrawide = "ok"
        self.recording = False
        self.recording_error = ""
        self.start_ack = Ack(ok=True, reason="")
        self.summary_error = ""
        self.frames = 30
        self.calls: list[str] = []
        self.dirs: list[str] = []
        self.stop_timeouts: list[float | None] = []
        self.error: dict[str, BaseException] = {}

    def _record(self, name: str) -> None:
        self.calls.append(name)
        if name in self.error:
            raise self.error[name]

    def start_recording(self, episode_dir: str) -> Ack:
        self._record("start_recording")
        self.dirs.append(episode_dir)
        if self.start_ack.ok:
            self.recording = True
            (Path(episode_dir) / "main.mkv").write_bytes(b"x")          # frames writes its files there
        return self.start_ack

    def stop_recording(self, timeout_s: float | None = None) -> RecordingSummary:
        self._record("stop_recording")
        self.stop_timeouts.append(timeout_s)
        if not self.recording:
            from deploy_contracts import ErrorReply, RemoteError
            raise RemoteError("frames", ErrorReply(request="r1.stop_recording/v1", error="not recording", field=""))
        self.recording = False
        return summary(self.frames, self.summary_error)

    def get_health(self) -> FramesHealth:
        self._record("get_health")
        return frames_health(self.main, self.ultrawide, self.recording, self.recording_error)

    def close(self) -> None:
        pass


# ---- the harness ---------------------------------------------------------------------------------------------------

class TeleopHarness:
    """A TeleopLoop wired to fakes, a FakeClock and the real StateHistory, PC clock map, HeartbeatSender,
    FailureMonitor, TaskStore (under data_root), E4's motion and episode classes. The PC broadcasts every 10 ms of fake
    time."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.clock = FakeClock()
        self.pc = FakePcClient(self.clock, settings.heartbeat_period_s)
        self.frames = FakeFramesClient()
        self.history = StateHistory(settings.history_s)
        self.stats = LinkStats(settings.link_stats_window_s, self.clock)
        self.pc_clock = make_pc_clock(settings.clock.estimator, settings.clock, self.clock)
        self.listener = FakeListener(self.history, self.clock, self.pc_clock, self.stats)
        self.heartbeats = HeartbeatSender(self.pc, self.pc_clock, self.stats, settings.clock.hello_period_s,
                                          settings.clock.startup_hello_period_s, self.clock)
        self.monitor = FailureMonitor(settings.state_lost_s)
        self.session = settings.session_root / "run" / "teleop"
        self.log = SessionLog(self.session, datetime.now, self.clock)
        self.checks = StartupChecks(settings, mock=True, commit=COMMIT)
        self.store = TaskStore(settings.data_root)
        self.loop = TeleopLoop(settings, self.pc, self.heartbeats, self.pc_clock, self.stats, self.frames,
                               self.history, self.listener, self.monitor, self.log, self.checks, self.store,
                               self.clock, self._sleep)
        self.loop.keep_requests()
        self.seq = 0
        self.T = pose()
        self.q = np.zeros(7)
        self.width = 0.08
        self.mode_override: str | None = None
        self.gate = dict(shortened=0, pulled_back=0, loop_late=0)
        self.publish = True
        self.state_delay_ns = 0             # the C2 t_ns is this much older than the PC's now
        self.replies: list = []

    def push_state(self, **kw) -> None:
        self.seq += 1
        values = dict(T=self.T, mode=self.mode_override or self.pc.mode, q=self.q, width=self.width, **self.gate)
        values.update(kw)
        self.listener.push(make_state(self.seq, self.pc.pc_ns() - self.state_delay_ns, **values))

    def _sleep(self, seconds: float) -> None:
        end = self.clock() + int(seconds * 1e9)
        while self.clock() < end:
            self.clock.advance(min(0.01, (end - self.clock()) / 1e9))
            if self.publish:
                self.push_state()

    def start(self, ready_file=None) -> None:
        """0.5 s of state history first (the PC was already broadcasting), then start-up."""
        self._sleep(0.5)
        self.loop.startup(ready_file)

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            if self.loop.phase is Phase.DONE:
                return
            self._sleep(self.settings.heartbeat_period_s)
            self.loop.cycle()

    def pc_move(self, T=None, q=None, cycles: int = 3) -> None:
        """The PC's h or p (or g, with neither T nor q) as C2 shows it: `homing` while the arm moves to T / q, then
        `idle`. Each phase lasts `cycles` loop cycles."""
        self.mode_override = "homing"
        self.tick(cycles)
        if T is not None:
            self.T = T
        if q is not None:
            self.q = np.asarray(q, dtype=np.float64)
        self.tick(cycles)
        self.mode_override = None
        self.pc.mode = "idle"
        self.tick(cycles)

    def run_until(self, condition, max_cycles: int = 500) -> None:
        for _ in range(max_cycles):
            if condition():
                return
            self.tick()
        raise AssertionError("condition not reached")

    def u1(self, msg):
        """One U1 request, served as the U1 server would in the cycle's step 6 (then the deferred exception)."""
        reply = self.loop.dispatch(msg)
        self.replies.append(reply)
        self.loop._guard(self.loop._raise_deferred)
        return reply

    def ready(self, task: str = "pick_block") -> None:
        """Start-up, a task, and the cameras polled: Start allowed."""
        from teleop_collect.contracts import CreateTask
        self.start()
        assert self.loop.phase is Phase.IDLE, self.verdict_text()
        assert self.u1(CreateTask(name=task, description="pick the block")).ok
        self.tick(10)

    def start_episode(self):
        from teleop_collect.contracts import StartEpisode
        reply = self.u1(StartEpisode())
        assert reply.ok, reply.reason
        assert self.loop.phase is Phase.RECORDING
        return reply

    def keys(self, *held: str, slow: bool = False, focus: bool = True):
        from teleop_collect.contracts import KeysUpdate
        return self.u1(KeysUpdate(held=tuple(held), slow=slow, focus=focus))

    def hold(self, keys, cycles: int, slow: bool = False) -> None:
        """Hold keys for n cycles, resending them every cycle as the page does."""
        for _ in range(cycles):
            self.keys(*keys, slow=slow)
            self.tick()

    def status(self, after: int = 0):
        from teleop_collect.contracts import GetStatus
        return self.u1(GetStatus(after_message=after))

    def verdict_text(self) -> str:
        v = self.loop.verdict
        return "" if v is None else f"{v.problem.code}: {v.what_failed}: {v.what_to_check}"

    @property
    def code(self):
        v = self.loop.verdict
        return None if v is None else v.problem.code

    def count(self, name: str) -> int:
        return self.pc.calls.count(name)

    def events(self) -> list[dict]:
        path = self.session / "events.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def task_dir(self, task: str = "pick_block") -> Path:
        return self.settings.data_root / task


# ---- a small fake PC on real sockets -------------------------------------------------------------------------------

class SocketPc:
    """C2 PUB at 100 Hz and C3 REP on tcp://127.0.0.1:* (a stand-in for franka_control, for the socket tests).

    Its clock is the laptop's CLOCK_MONOTONIC + 1000 s. start -> holding, stop -> stopped, home -> homing for home_s
    then idle (the REP blocked meanwhile), set_waypoints -> running and accepted by time. `start_mode` sets the first
    mode; `hold_forever()` keeps a holding mode as another C3 client would. Every request name is kept."""

    def __init__(self, start_mode: str = "idle", home_s: float = 0.3, link: LinkSettings | None = None,
                 holding_s: float = 0.5) -> None:
        self.mode = start_mode
        self.home_s = home_s
        self.link = link
        self.holding_s = holding_s           # an orphaned holding turns stopped after this (its watchdog)
        self._hold_forever = False
        self.requests: list[tuple[int, str]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._pub = Publisher("tcp://127.0.0.1:*", 1000)
        self.state_endpoint = self._pub.endpoint
        from deploy_contracts import GetState, Heartbeat, Hello, Home, SetWaypoints, Start, Stop
        self._rep = RpcServer("tcp://127.0.0.1:*", {
            Hello: self._hello, Heartbeat: self._ack, Start: self._start, Stop: self._stop, Home: self._home,
            GetState: self._get_state, SetWaypoints: self._set})
        self.commands_endpoint = self._rep.endpoint
        self._t_mode = time.monotonic()
        self._stop_ev = threading.Event()
        self._threads = [threading.Thread(target=self._publish, daemon=True),
                         threading.Thread(target=self._serve, daemon=True)]
        for t in self._threads:
            t.start()

    @staticmethod
    def pc_ns() -> int:
        return time.monotonic_ns() + PC_OFFSET_NS

    def hold_forever(self) -> None:
        self._hold_forever = True

    def request_names(self) -> list[str]:
        with self._lock:
            return [name for _, name in self.requests]

    def _note(self, msg) -> None:
        with self._lock:
            self.requests.append((time.monotonic_ns(), msg.SCHEMA))
        self._t_mode = time.monotonic()

    def _state(self) -> ArmState:
        self._seq += 1
        return make_state(self._seq, self.pc_ns(), mode=self.mode)

    def _publish(self) -> None:
        while not self._stop_ev.is_set():
            if (self.mode == "holding" and not self._hold_forever
                    and time.monotonic() - self._t_mode > self.holding_s):
                self.mode = "stopped"
            self._pub.publish(self._state())
            time.sleep(0.01)

    def _serve(self) -> None:
        while not self._stop_ev.is_set():
            self._rep.serve_once(0.01)

    def _hello(self, msg):
        self._note(msg)
        return hello_reply(self.pc_ns(), link=self.link)

    def _ack(self, msg):
        self._note(msg)
        return Ack(ok=True, reason="")

    def _start(self, msg):
        self._note(msg)
        self.mode = "holding"
        return Ack(ok=True, reason="")

    def _stop(self, msg):
        self._note(msg)
        if self.mode in ("running", "holding"):
            self.mode = "stopped"
        return Ack(ok=True, reason="")

    def _home(self, msg):
        self._note(msg)
        self.mode = "homing"
        time.sleep(self.home_s)
        self.mode = "idle"
        return Ack(ok=True, reason="")

    def _get_state(self, msg):
        self._note(msg)
        return self._state()

    def _set(self, msg):
        self._note(msg)
        self.mode = "running"
        n = len(msg.t_ns)
        past = int(np.sum(msg.t_ns < self.pc_ns()))
        return SetWaypointsReply(chunk_id=msg.chunk_id, accepted=n - past, shortened=0, skipped=past, rejected=())

    def close(self) -> None:
        self._stop_ev.set()
        for t in self._threads:
            t.join(1.0)
        self._pub.close()
        self._rep.close()


class RawServer:
    """A server on a free port that answers REQ clients with whatever bytes the test gives (or never answers)."""

    def __init__(self, answer) -> None:
        self._answer = answer
        self._sock = zmq.Context.instance().socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind("tcp://127.0.0.1:*")
        self.endpoint = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self.received: list[bytes] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._sock.poll(10, zmq.POLLIN):
                continue
            identity, empty, data = self._sock.recv_multipart()
            self.received.append(data)
            reply = self._answer(data)
            if reply is not None:
                self._sock.send_multipart([identity, empty, reply if isinstance(reply, bytes) else encode(reply)])

    def close(self) -> None:
        self._stop.set()
        self._thread.join(1.0)
        self._sock.close(linger=0)
