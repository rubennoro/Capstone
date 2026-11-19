# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/episode_loop.py @ 2ffe641
# changes: teleop_collect STRUCTURE.md 6.3-6.7. The policy step (StepWorker, PolicyStep, the anchor, odd chunks,
# POLICY_LATE, skips) is replaced by the teleop step: KeyState -> MotionController.step -> one set_waypoints, which is
# then the cycle's one C3 request (hello if due, else the set, else a heartbeat; rule H1); base and now_pc come from one
# PcClock snapshot; a C2 seq gap or an old state pauses motion (SLOW); a set skipped entirely is SET_SKIPPED (SLOW); the
# keyboard Operator and status line are replaced by the U1 server (served from the main loop, up to u1_max_per_cycle per
# cycle; dispatch() by request class; handler exceptions and conclude are re-raised on the main loop) and TeleopStatus;
# the episode lifecycle moved to EpisodeFlow (start / stop / discard / abort, R1 and E4's store and recorders);
# Kind.EPISODE problems abort the open episode and the loop goes on; home is a U1 request done in the next cycle as that
# cycle's C3 request, its wait serves U1 (status and keys only), and a home that ended in idle sets the HomeReference;
# R1 health (FramesHealth) every health_poll_ms; every C3 request while recording goes to commands.csv, every hello
# sample and the clock fit to clock_pc.csv, every key change to keys.csv; at_limit from C2 gate.pulled_back; Ctrl-C and
# SIGTERM abort (delete) an open episode; _finish catches a failing stop (logged, as _conclude does); the conclude
# reason carries the problem's detail into EpisodeFlow.abort_open; the SIGTERM text is "stopped by the launcher (Ctrl-C,
# or another process exited)"; up to 3 R1 get_health timeouts in a row are a warning, the next concludes; how long each
# camera has shown down (the start refusal's Pi hint); _wait_until_boundary only sleeps (no worker); process name
# teleop. Kept from the source: the rhythm (run, cycle boundary), _beat with the clock-jump log and the re-warm after a
# home, _check_state, the C3 reply checks and the clock-mismatch guard, _refusal_cause, _guard, _conclude / _finish /
# _shutdown, the periodic clock and link rows. The source's own header follows.
# adapted from rnoro/deployment:deployment/message_comms/message_comms/episode_loop.py @ 91b178b
# changes: start-up moved to StartupSequence; HeartbeatSender (a hello or a heartbeat per cycle) instead of a bare
# heartbeat; the cycle period is link.yaml heartbeat_period_s; margins in PC time; the CLOCK_MISMATCH counter (skipped
# sets the laptop timed ahead, or a C2 from the future); keys dispatched through KEY_HANDLERS; c and h refused locally
# while C2 shows homing; c needs the clock map ready; home returns its Ack: ok, HOME_STOPPED (the `stopped:` prefix),
# a refusal because the PC is homing (a hint), else HOMING_FAILED; after a home the clock map re-warms (a hello every
# cycle); the start ack time kept as start_ack_pc_ns (mapped); last_anchor_t_ns stays laptop time; STATE_LOST by the
# listener's arrival time; clock.jsonl / link.jsonl; conclude sends nothing on C3 before hellos started. Fix wave: the
# GATE_REJECTED text from gate_rejection_detail(); an entirely skipped set counts toward CLOCK_MISMATCH or POLICY_LATE,
# never both.
"""TeleopLoop: teleop's main thread. The 20 ms rhythm, every C3 request (rule H1), R1 calls, U1 serving, failure
decisions and logging (STRUCTURE.md 6.1, 6.3-6.7; PLAN 5, 7, 8)."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import numpy as np
from deploy_contracts import Ack, ArmState, Message, SetWaypoints, TransportTimeout

from teleop_collect.contracts import (CAMERAS, CreateTask, DeleteEpisode, FramesHealth, GetStatus, GripperCommand,
                                      HomeRequest, KeysUpdate, NotSynced, SelectTask, StartEpisode, StopArm,
                                      StopEpisode, TeleopStatus)
from teleop_collect.contracts import EpisodeInfo as EpisodeInfoRecord
from teleop_collect.teleop.episode.store import TaskStore
from teleop_collect.teleop.errors import RequestCancelled
from teleop_collect.teleop.failure_monitor import FailureMonitor
from teleop_collect.teleop.frames_client import FramesClient
from teleop_collect.teleop.heartbeat_sender import HeartbeatSender
from teleop_collect.teleop.link_stats import LinkStats
from teleop_collect.teleop.motion.controller import BaseState, MotionCommand, MotionController
from teleop_collect.teleop.motion.home import HomeReference
from teleop_collect.teleop.motion.keys import HeldKeys, KeyState
from teleop_collect.teleop.pc_client import PcClient
from teleop_collect.teleop.pc_clock import PcClock
from teleop_collect.teleop.problems import Kind, Problem, ProblemCode, Verdict
from teleop_collect.teleop.session_log import SessionLog
from teleop_collect.teleop.settings import Settings
from teleop_collect.teleop.startup_checks import StartupChecks
from teleop_collect.teleop.state_history import StateHistory
from teleop_collect.teleop.u1_server import MessageLog

log = logging.getLogger("teleop.teleop_loop")
C = ProblemCode
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
PC = "PC"
STOP_CALL = (PC, "c3.stop/v1")
LAUNCHER_STOP_REASON = "stopped by the launcher (Ctrl-C, or another process exited)"
NOT_CHECKED_PREFIX = "not checked"     # the other waypoints of a rejected request (TOP 3.7.1)
HOMING_REFUSAL = "homing: wait, or press s on the PC (the user stop is always there)"
STARTING_REFUSAL = "starting up: wait until teleop is ready"
SLOW_EVENT_EVERY_NS = NS_PER_S         # a SLOW problem goes to events.jsonl at most once a second per code
FIT_EVERY_NS = NS_PER_S                # clock_pc.csv fit rows while recording (PLAN 6.1: once per second)
HELLO = "c3.hello/v1"
HEARTBEAT = "c3.heartbeat/v1"
HEALTH_TIMEOUTS_TOLERATED = 3         # consecutive R1 get_health timeouts warned about; one more concludes


def gate_rejection_detail(reply) -> str:
    """'<index>: <reason>' of the first rejected waypoint that failed a rule itself (the PC lists every non-skipped
    waypoint of a rejected request; the others say 'not checked: another waypoint of this request was rejected')."""
    issue = next((i for i in reply.rejected if not i.reason.startswith(NOT_CHECKED_PREFIX)), reply.rejected[0])
    return f"{issue.index}: {issue.reason}"


def set_waypoints_message(chunk_id: int, cmd: MotionCommand) -> SetWaypoints:
    """E4's waypoints as the C3 request (exact dtypes: int64 times, float64 poses and widths)."""
    wps = cmd.waypoints
    return SetWaypoints(chunk_id=int(chunk_id),
                        t_ns=np.array([int(w.t_pc_ns) for w in wps], dtype=np.int64),
                        O_T_TCP=np.stack([np.asarray(w.O_T_TCP, dtype=np.float64) for w in wps]).astype(np.float64),
                        gripper_width_m=np.array([float(w.width_m) for w in wps], dtype=np.float64))


class Phase(Enum):
    STARTING = "starting"         # start-up (STRUCTURE.md 6.4)
    IDLE = "idle"                 # no episode
    RECORDING = "recording"       # an episode: keys live, everything recorded
    HOMING = "homing"             # a blocking C3 home
    STOPPING = "stopping"         # Stop pressed: stop, close the recording, save or discard
    DONE = "done"                 # finished; run() returns


class _Finished(Exception):
    """Control flow only: the loop has finished (phase DONE); the rest of this cycle is skipped."""


class TeleopLoop:
    def __init__(self, settings: Settings, pc: PcClient, heartbeats: HeartbeatSender, pc_clock: PcClock,
                 link_stats: LinkStats, frames: FramesClient, history: StateHistory, listener,
                 monitor: FailureMonitor, log: SessionLog, checks: StartupChecks, store: TaskStore,
                 clock: Callable[[], int], sleep: Callable[[float], None]) -> None:
        from teleop_collect.teleop.episode_flow import EpisodeFlow
        self._s = settings
        self._pc = pc
        self._heartbeats = heartbeats
        self._pc_clock = pc_clock
        self._link = link_stats
        self._frames = frames
        self._history = history
        self._listener = listener
        self._monitor = monitor
        self._log = log
        self._checks = checks
        self._clock = clock
        self._sleep = sleep
        self._keys = KeyState(settings.motion.key_stale_ms)
        self._motion = MotionController(settings.motion, settings.gripper_width_max_m)
        self._home_ref = HomeReference(settings.home)
        self._messages = MessageLog(settings.messages_kept)
        self.flow = EpisodeFlow(self, store)
        self._u1 = None                   # U1Server (attach_u1); tests call dispatch() directly
        self._loop_ns = int(round(settings.heartbeat_period_s * NS_PER_S))
        self._interrupt = threading.Event()
        self._terminate = threading.Event()

        self._phase = Phase.STARTING
        self._ready = False
        self._verdict: Verdict | None = None
        self._final_line = ""
        self._exit_code = 0
        self._shut = False
        self._call: tuple[str, str] = ("teleop", "")
        self._beating = False             # hellos/heartbeats started (start-up step 4); nothing on C3 before
        self._rewarm = False              # after a home: a hello every cycle until the map is ready
        self._hello = None                # the hello reply checked at start-up
        self._last_health: FramesHealth | None = None
        self._next_health_ns = 0
        self._next_clock_log_ns = 0
        self._next_link_log_ns = 0
        self._next_fit_ns = 0
        self._stop_sent = False
        self._start_ack_pc_ns: int | None = None
        self._next_chunk_id = 1
        self._mismatch_count = 0
        self._home_requested = False
        self._home_ref_pending = False    # a home ended ok: the next idle C2 is the reference
        self._deferred: BaseException | None = None   # raised by a U1 handler; re-raised on the loop
        self._gaps_seen = 0
        self._health_timeouts = 0         # consecutive R1 get_health timeouts
        self._down_since: dict[str, int] = {}   # camera -> laptop time of the first poll that showed it down
        self._last_pulled_back: int | None = None
        self._limit_until_ns = 0
        self._slow_logged: dict[ProblemCode, int] = {}
        self.requests: list[str] = []     # every C3 request sent, in order (tests; never grows past what is needed)
        self._keep_requests = False

    # ---- public interface ----------------------------------------------------------------------------------------

    @property
    def verdict(self) -> Verdict | None:
        return self._verdict

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def settings(self) -> Settings:
        return self._s

    @property
    def messages(self) -> MessageLog:
        return self._messages

    @property
    def home_reference(self) -> HomeReference:
        return self._home_ref

    @property
    def motion(self) -> MotionController:
        return self._motion

    def attach_u1(self, server) -> None:
        """The U1Server (its dispatch is self.dispatch)."""
        self._u1 = server

    def keep_requests(self, on: bool = True) -> None:
        """Tests: keep the name of every C3 request in self.requests."""
        self._keep_requests = on

    def interrupt(self) -> None:
        """SIGINT handler: sets the interrupt flag, nothing else."""
        self._interrupt.set()

    def terminate(self) -> None:
        """SIGTERM handler: sets the terminate flag, nothing else."""
        self._terminate.set()

    def interrupt_set(self) -> bool:
        return self._interrupt.is_set() or self._terminate.is_set()

    def startup(self, ready_file: Path | Callable[[], None] | None) -> None:
        """STRUCTURE.md 6.4. On failure sets the verdict and phase DONE."""
        from teleop_collect.teleop.startup_sequence import StartupSequence
        self._guard(StartupSequence(self).run, ready_file)

    def cycle(self) -> None:
        """One loop cycle (STRUCTURE.md 6.3, steps 1-7)."""
        if self._phase is Phase.DONE:
            return
        self._guard(self._cycle)

    def run(self) -> int:
        """Cycles until DONE; returns the exit code (0 or 1)."""
        boundary = self._clock()
        while self._phase is not Phase.DONE:
            self.cycle()
            boundary = max(boundary + self._loop_ns, self._clock())
            self._guard(self._wait_until_boundary, boundary)
        return self._exit_code

    # ---- one cycle -----------------------------------------------------------------------------------------------

    def _cycle(self) -> None:
        now = self._clock()
        self._check_flags()                                                         # 1
        self._check_state(now)                                                      # 2
        keys = self._keys.current(now)                                              # 3
        self._c3_request(now, keys)                                                 # 4
        now = self._clock()
        if now >= self._next_health_ns:                                             # 5
            self._poll_health(now)
        self._serve_u1()                                                            # 6
        now = self._clock()
        self._observe_state(now)                                                    # 7
        self._record_periodic(now)
        self._log_periodic(now)

    def _wait_until_boundary(self, boundary: int) -> None:
        remaining = boundary - self._clock()
        if remaining > 0 and self._phase is not Phase.DONE:
            self._sleep(remaining / NS_PER_S)

    def _c3_request(self, now: int, keys: HeldKeys) -> None:
        """Exactly one C3 request: the queued home, a hello when due, the motion set, else a heartbeat."""
        if not self._beating:
            return
        if self._home_requested:
            self._home_requested = False
            self._home()
            return
        if self._heartbeats.hello_due(now):
            self._beat(now)
            return
        command = None
        if self._phase is Phase.RECORDING:
            now_pc, base = self._motion_inputs(now, keys)
            command = self._motion.step(now_pc, base, keys)
        if command is not None:
            self._send_set(command)
        else:
            self._beat(now)

    def _motion_inputs(self, now: int, keys: HeldKeys) -> tuple[int | None, BaseState | None]:
        """now_pc and the base from ONE PcClock snapshot and the newest C2 (the state-age guard compares them). The
        slow problems that pause motion are counted here: no fresh clock sample, a C2 seq gap, an old state."""
        latest = self._history.latest()
        gaps = self._history.gap_count()
        new_gap = gaps > self._gaps_seen
        self._gaps_seen = gaps
        if latest is None:
            return None, None
        base = BaseState.from_arm_state(latest)
        try:
            snap = self._pc_clock.snapshot()
        except NotSynced:
            self._slow(Problem(C.CLOCK_STALE, PC, "no PC clock fit"), keys)
            return None, base
        if not self._pc_clock.ready(now):
            self._slow(Problem(C.CLOCK_STALE, PC, self._pc_clock.not_ready_reason(now)), keys)
            return None, base
        now_pc = snap.to_pc_ns(self._clock())
        if new_gap:
            self._slow(Problem(C.STATE_GAP, PC, "a C2 seq gap: no motion this cycle"), keys)
            return now_pc, None
        age_ms = (now_pc - int(latest.t_ns)) / NS_PER_MS
        if age_ms > self._s.motion.state_max_age_ms:
            self._slow(Problem(C.STATE_OLD, PC, f"newest C2 {age_ms:.0f} ms old: no motion this cycle"), keys)
        return now_pc, base

    def _slow(self, problem: Problem, keys: HeldKeys | None = None) -> None:
        """A SLOW problem: counted; logged at most once a second per code. Not counted when no key is held (nothing
        was going to move)."""
        if keys is not None and not keys.held:
            return
        verdict = self._monitor.classify(problem, episode_open=self._phase is Phase.RECORDING)
        now = self._clock()
        last = self._slow_logged.get(problem.code)
        if last is None or now - last >= SLOW_EVENT_EVERY_NS:
            self._slow_logged[problem.code] = now
            self._safe_event(problem.code.value, verdict.what_failed)
            self.flow.event(problem.code.value, verdict.what_failed)

    def _beat(self, now: int) -> None:
        """A hello or a heartbeat; hello samples feed the map, fit jumps are logged (guard 3)."""
        self._call = (PC, HEARTBEAT)
        t_send = self._clock()
        try:
            reply = self._heartbeats.beat(now)
        finally:
            self._call = (PC, self._heartbeats.last_request or HEARTBEAT)
        t_reply = self._clock()
        self._note_request(self._heartbeats.last_request)
        if reply is not None:
            sample = self._heartbeats.last_sample()
            self.flow.record_hello(sample, reply)
            for jump in self._pc_clock.take_jumps():
                self._safe_event("clock_jump", f"refit moved the PC time by {jump.jump_ms:+.2f} ms")
                self._log.log_clock({"kind": "jump", "estimator": self._pc_clock.NAME, "jump_ms": jump.jump_ms})
        else:
            self.flow.record_command(t_send, t_reply, HEARTBEAT, None, Ack(ok=True, reason=""), None, False)
        if self._rewarm and self._pc_clock.ready(self._clock()):
            self._rewarm = False
            self._heartbeats.startup_mode(False)
            self._safe_event("clock", "PC clock map ready again")

    def _note_request(self, name: str) -> None:
        if self._keep_requests:
            self.requests.append(name)

    def _send_set(self, cmd: MotionCommand) -> None:
        """STRUCTURE.md 6.3 step 4: set_waypoints, the reply checks and the clock guard."""
        s = self._s
        msg = set_waypoints_message(self._next_chunk_id, cmd)
        self._next_chunk_id += 1
        snap = self._pc_clock.snapshot()
        t_send = self._clock()
        send_pc = snap.to_pc_ns(t_send)
        latest = self._history.latest()
        reply = self._rpc(PC, "c3.set_waypoints/v1", self._pc.set_waypoints, msg)
        t_reply = self._clock()
        self._note_request(msg.SCHEMA)
        self.flow.record_command(t_send, t_reply, msg.SCHEMA, msg, reply, cmd.twist, cmd.slow)
        if reply.rejected:
            self._conclude(Problem(C.GATE_REJECTED, PC, gate_rejection_detail(reply)))
        n = len(msg.t_ns)
        lead_ms = (int(msg.t_ns[-1]) - send_pc) / NS_PER_MS
        future_ms = (int(latest.t_ns) - send_pc) / NS_PER_MS if latest is not None else 0.0
        skipped_ahead = reply.skipped == n and lead_ms >= s.clock.mismatch_min_lead_ms
        future_state = future_ms > s.clock.future_state_ms
        if skipped_ahead or future_state:
            self._mismatch_count += 1
            why = (f"the PC skipped a set the laptop timed {lead_ms:.0f} ms ahead" if skipped_ahead else
                   f"the newest C2 t_ns was {future_ms:.0f} ms ahead of the mapped now")
            self._safe_event("clock_mismatch", f"{self._mismatch_count}/{s.clock.mismatch_sets}: {why}")
            if self._mismatch_count >= s.clock.mismatch_sets:
                detail = (f"the PC skipped {self._mismatch_count} waypoint sets the laptop timed ahead"
                          if skipped_ahead else
                          f"{self._mismatch_count} waypoint sets sent while the newest C2 t_ns was more than "
                          f"{s.clock.future_state_ms:g} ms ahead of the mapped now")
                self._conclude(Problem(C.CLOCK_MISMATCH, PC, detail))
        elif reply.accepted > 0:
            self._mismatch_count = 0
        if reply.skipped == n and not (skipped_ahead or future_state):
            self._slow(Problem(C.SET_SKIPPED, PC, f"set {msg.chunk_id} skipped entirely by the PC (lead "
                                                  f"{lead_ms:.0f} ms)"))
        if reply.shortened:
            self._monitor.classify(Problem(C.SHORTENED, PC, f"{reply.shortened} waypoints shortened"),
                                   count=int(reply.shortened))

    def _check_flags(self) -> None:
        if self._interrupt.is_set():
            self._quit("operator quit (Ctrl-C)")
        if self._terminate.is_set():
            self._launcher_stop()

    def _check_state(self, now: int) -> None:
        """The state rules first (a fault is the cause when a broken message follows it), then the listener, then
        the recorders."""
        problem = self._monitor.check_state(self._history.latest(), now, self._listener.last_arrival_ns(),
                                            self._phase is Phase.RECORDING, self._stop_sent, self._start_ack_pc_ns)
        if problem is not None:
            self._problem(problem)
        error = self._listener.error()
        if error is not None:
            self._conclude(self._monitor.from_exception(error, PC, "c2.state/v1"))
        if self._phase is Phase.RECORDING:
            failure = self.flow.recorder_failure(self._listener.recorder_error())
            if failure:
                self._problem(Problem(C.RECORDING_FAILED, "teleop", failure))

    def _poll_health(self, now: int) -> FramesHealth | None:
        """R1 get_health. Up to HEALTH_TIMEOUTS_TOLERATED timeouts in a row are only a warning each (frames was busy
        for 50 ms); the next one concludes NO_ANSWER. None when this poll timed out."""
        self._next_health_ns = now + self._s.health_poll_ms * NS_PER_MS
        try:
            report = self._rpc("frames", "r1.get_health/v1", self._frames.get_health)
        except TransportTimeout as exc:
            self._health_timeouts += 1
            if self._health_timeouts > HEALTH_TIMEOUTS_TOLERATED:
                raise
            text = (f"frames: no reply to get_health within {exc.timeout_s:.2f} s "
                    f"({self._health_timeouts}/{HEALTH_TIMEOUTS_TOLERATED} in a row tolerated)")
            self._safe_event("health_timeout", text)
            self._messages.add("warn", text)
            return None
        self._health_timeouts = 0
        self._last_health = report
        for name in CAMERAS:
            if getattr(report.health, name).status == "down":
                self._down_since.setdefault(name, now)
            else:
                self._down_since.pop(name, None)
        problem = self._monitor.check_health(report, recording=self._phase is Phase.RECORDING)
        if problem is not None:
            self._problem(problem)
        return report

    def _problem(self, problem: Problem) -> None:
        """Act on a problem by its kind (PLAN 7): conclude, abort the open episode, or count (slow / normal)."""
        verdict = self._monitor.classify(problem, episode_open=self._phase is Phase.RECORDING)
        if verdict.kind is Kind.CONCLUDE:
            self._conclude(problem)
        elif verdict.kind is Kind.EPISODE:
            self.flow.abort(verdict)
        elif verdict.kind is Kind.SLOW:
            now = self._clock()
            last = self._slow_logged.get(problem.code)
            if last is None or now - last >= SLOW_EVENT_EVERY_NS:
                self._slow_logged[problem.code] = now
                self._safe_event(problem.code.value, verdict.what_failed)
                self.flow.event(problem.code.value, verdict.what_failed)

    def _observe_state(self, now: int) -> None:
        """Every cycle: the home reference (a home of ours that ended in idle; the PC's own h), at_limit."""
        latest = self._history.latest()
        if latest is None:
            return
        if self._home_ref_pending:
            if latest.mode == "idle":
                self._home_ref.set_from(latest.q, latest.O_T_TCP_target)
                self._home_ref_pending = False
                self._safe_event("home", "home reference set")
            elif latest.mode != "homing":
                self._home_ref_pending = False
        self._home_ref.observe(latest.mode, latest.q, latest.O_T_TCP_target)
        pulled = int(latest.gate.pulled_back)
        if self._last_pulled_back is not None and pulled > self._last_pulled_back:
            self._limit_until_ns = now + self._s.at_limit_hold_ms * NS_PER_MS
            self._monitor.classify(Problem(C.PULLED_BACK, PC, f"{pulled - self._last_pulled_back} waypoints pulled "
                                                              "back at the box or cone"),
                                   count=pulled - self._last_pulled_back)
        self._last_pulled_back = pulled

    def _record_periodic(self, now: int) -> None:
        """While recording: every key change to keys.csv; the PC clock fit to clock_pc.csv once a second."""
        changes = self._keys.changes()
        if self._phase is not Phase.RECORDING:
            return
        for t_ns, keys, focus in changes:
            self.flow.record_keys(t_ns, keys, focus)
        if now >= self._next_fit_ns:
            self._next_fit_ns = now + FIT_EVERY_NS
            self.flow.record_fit(now)

    # ---- U1 (STRUCTURE.md 6.6) -----------------------------------------------------------------------------------

    def _serve_u1(self) -> None:
        if self._u1 is None:
            return
        self._u1.serve(self._s.u1_max_per_cycle)
        self._raise_deferred()

    def _raise_deferred(self) -> None:
        exc, self._deferred = self._deferred, None
        if exc is not None:
            raise exc

    def dispatch(self, msg: Message) -> Message:
        """One U1 request -> its reply. Runs on the main thread (inside serve). An exception (a C3 timeout during
        start, a conclude) is kept and re-raised on the loop after the reply went out."""
        try:
            if self._phase is Phase.HOMING and not isinstance(msg, (GetStatus, KeysUpdate)):
                return Ack(ok=False, reason=HOMING_REFUSAL)
            if self._phase is Phase.STARTING and not isinstance(msg, (GetStatus, KeysUpdate)):
                return Ack(ok=False, reason=STARTING_REFUSAL)
            if self._phase is Phase.DONE and not isinstance(msg, GetStatus):
                return Ack(ok=False, reason=f"teleop has stopped: {self._final_line}")
            return U1_HANDLERS[type(msg)](self, msg)
        except _Finished as exc:
            self._deferred = exc
            return Ack(ok=False, reason=f"teleop has stopped: {self._final_line}")
        except Exception as exc:
            self._deferred = exc
            return Ack(ok=False, reason=f"teleop failed: {type(exc).__name__}: {exc}")

    def _on_status(self, msg: GetStatus) -> Message:
        return self.status(int(msg.after_message))

    def _on_keys(self, msg: KeysUpdate) -> Message:
        self._keys.update(msg.held, msg.slow, msg.focus, self._clock())
        return Ack(ok=True, reason="")

    def _on_gripper(self, msg: GripperCommand) -> Message:
        if self._phase is not Phase.RECORDING:
            return Ack(ok=False, reason="no episode recording: the fingers move only during an episode")
        self._motion.gripper(msg.command)
        return Ack(ok=True, reason="")

    def _on_select(self, msg: SelectTask) -> Message:
        return self.flow.select(msg.name)

    def _on_create(self, msg: CreateTask) -> Message:
        return self.flow.create(msg.name, msg.description)

    def _on_home(self, msg: HomeRequest) -> Message:
        if self._phase is not Phase.IDLE:
            return Ack(ok=False, reason=f"{self._phase.value}: stop the episode first")
        if self._home_requested:
            return Ack(ok=False, reason="homing: already asked")
        if self._pc_mode() == "homing":
            return Ack(ok=False, reason="the PC is homing (its h or g key): wait for idle")
        self._home_requested = True
        self._messages.add("info", "homing: the whole arm moves to the home pose (keep a hand on the user stop)")
        return Ack(ok=True, reason="")

    def _on_start(self, msg: StartEpisode) -> Message:
        return self.flow.start()

    def _on_stop(self, msg: StopEpisode) -> Message:
        return self.flow.stop(msg.outcome)

    def _on_delete(self, msg: DeleteEpisode) -> Message:
        return self.flow.delete(msg.episode)

    def _on_stop_arm(self, msg: StopArm) -> Message:
        if self._phase is Phase.RECORDING:
            self._problem(Problem(C.OPERATOR_ABORT, "operator", "Stop arm"))
            return Ack(ok=True, reason="")
        if not self._beating:
            return Ack(ok=False, reason="not connected to the PC yet")
        self._send_stop()
        self._messages.add("info", "arm stopped: it holds where it is")
        return Ack(ok=True, reason="")

    # ---- status --------------------------------------------------------------------------------------------------

    def status(self, after_message: int = 0) -> TeleopStatus:
        now = self._clock()
        latest = self._history.latest()
        refusal = self.flow.start_refusal(now)
        link = self._link.snapshot(now)
        hello = self._heartbeats.last_hello()
        recording = self._phase is Phase.RECORDING
        keys = self._keys.current(now)
        task = self.flow.task
        return TeleopStatus(
            phase=self._phase.value, ready=self._ready,
            task=task.name if task is not None else "",
            task_description=task.description if task is not None else "",
            tasks=tuple(self.flow.tasks()),
            episodes=tuple(EpisodeInfoRecord(name=e.name, outcome=e.outcome, duration_s=float(e.duration_s),
                                             frames_main=int(e.frames_main)) for e in self.flow.episodes()),
            recording_episode=self.flow.recording_name() if recording else "",
            episode_s=float(self.flow.episode_s(now)) if recording else 0.0,
            arm_mode=latest.mode if latest is not None else "",
            tcp_m=np.array(latest.O_T_TCP[:3, 3], dtype=np.float64) if latest is not None else None,
            gripper_width_m=float(latest.gripper_width_m) if latest is not None else None,
            commanded_width_m=float(self._motion.commanded_width_m),
            home_known=self._home_ref.known, at_home=self.at_home(latest),
            start_allowed=not refusal, start_refusal=refusal,
            at_limit=now < self._limit_until_ns,
            moving=bool(recording and self._motion.moving), slow=bool(recording and keys.slow),
            cameras=self._last_health.health if self._last_health is not None else None,
            c2_hz=float(link.c2_rate_hz) if link.c2_rate_hz is not None else 0.0,
            rtt_ms=float(link.rtt_p50_ms) if link.rtt_p50_ms is not None else None,
            clock_ok=bool(self._beating and self._pc_clock.ready(now)),
            pc_mock=bool(hello.mock) if hello is not None else False,
            messages=tuple(self._messages.after(after_message)), final_line=self._final_line)

    def at_home(self, latest: ArmState | None = None) -> bool:
        latest = self._history.latest() if latest is None else latest
        if latest is None or not self._home_ref.known:
            return False
        return self._home_ref.at_home(latest.q, latest.O_T_TCP_target)

    def camera_down_s(self, name: str, now: int) -> float:
        """How long the health polls have shown this camera down (0 when it is not)."""
        since = self._down_since.get(name)
        return 0.0 if since is None else (now - since) / NS_PER_S

    def clock_ready(self, now: int) -> bool:
        return self._beating and self._pc_clock.ready(now)

    # ---- home, stop, quit, launcher stop, conclude ---------------------------------------------------------------

    def _home(self) -> Ack:
        """STRUCTURE.md 6.7: heartbeats and hellos pause; the wait serves U1 (status, keys); Ctrl-C / SIGTERM cancel
        it (no stop is sent). Returns the PC's Ack once handled (a failure concludes)."""
        previous = self._phase
        self._phase = Phase.HOMING
        self._home_ref_pending = False
        self._log.log_event("home", "homing started")
        try:
            ack = self._rpc(PC, "c3.home/v1", self._pc.home, self._home_interrupted)
        except RequestCancelled:
            self._phase = previous
            self._log.log_event("home", "cancelled")
            if self._terminate.is_set():
                self._launcher_stop(send_stop=False)
            self._quit("operator quit (Ctrl-C) while homing", send_stop=False)
        self._note_request("c3.home/v1")
        self._phase = previous
        self._raise_deferred()
        if ack.ok:
            self._home_ref_pending = True
            self._log.log_event("home", "home pose reached")
            self._messages.add("info", "home pose reached")
        elif self._pc.home_stopped(ack):
            verdict = self._monitor.classify(Problem(C.HOME_STOPPED, PC, ack.reason))
            self._log.log_event("home_stopped", f"{verdict.what_failed}: {verdict.what_to_check}")
            self._messages.add("warn", f"{verdict.what_failed}: {verdict.what_to_check}")
        elif self._pc.home_refused_homing(ack):
            self._log.log_event("home_refused", ack.reason)
            self._messages.add("warn", f"home: the PC is already homing (its h or g key): {ack.reason}")
        else:
            self._conclude(Problem(C.HOMING_FAILED, PC, ack.reason or "refused"))
        self._rewarm = True                                  # the map had no samples during the home
        self._heartbeats.startup_mode(True)
        self._beat(self._clock())                            # heartbeats resume at once
        return ack

    def _home_interrupted(self) -> bool:
        """Called by PcClient.home between its slices: serves U1 (phase HOMING: status and keys only), then says
        whether Ctrl-C or SIGTERM came."""
        if self._u1 is not None:
            try:
                self._u1.serve(self._s.u1_max_per_cycle)
            except Exception as exc:                          # never breaks the home wait; handled after it
                self._deferred = self._deferred or exc
        return self.interrupt_set()

    def _send_stop(self) -> None:
        self._rpc(PC, "c3.stop/v1", self._pc.stop)
        self._note_request("c3.stop/v1")
        self._stop_sent = True

    def _quit(self, reason: str, send_stop: bool = True) -> None:
        """Ctrl-C: stop first, an open episode aborted (deleted, "operator quit"), exit 0."""
        self._finish(reason, Problem(C.OPERATOR_STOP, "operator", reason), send_stop)

    def _launcher_stop(self, send_stop: bool = True) -> None:
        """SIGTERM: stop first, an open episode aborted (deleted), exit 0."""
        self._finish(LAUNCHER_STOP_REASON, Problem(C.LAUNCHER_STOP, "launcher", LAUNCHER_STOP_REASON), send_stop)

    def _finish(self, reason: str, problem: Problem, send_stop: bool) -> None:
        if send_stop and self._beating:
            try:
                self._send_stop()
            except Exception as exc:              # logged, not retried: the watchdog stops the arm
                log.error("stop failed while quitting: %s", exc)
                self._safe_event("stop_failed", str(exc))
        self.flow.abort_open(reason)
        self._verdict = self._monitor.classify(problem)
        self._final_line = f"{self._verdict.what_failed}: {self._verdict.what_to_check}"
        self._safe_event("exit", reason)
        self._exit_code = 0
        self._shutdown()
        raise _Finished

    def _conclude(self, problem: Problem, skip_stop: bool = False) -> None:
        """Stop once (if C3 was started), abort (delete) an open episode, stop everything, exit 1 (the FATAL line is
        __main__'s). The PC keeps running, holding the arm."""
        verdict = self._monitor.classify(problem)
        if verdict.kind is not Kind.CONCLUDE:
            raise RuntimeError(f"conclude called for a {verdict.kind.value} problem: {problem}")
        self._verdict = verdict
        self._final_line = f"{verdict.what_failed}: {verdict.what_to_check}"
        log.error("conclude: %s", self._final_line)
        if not skip_stop and self._beating:
            try:
                self._pc.stop()
                self._note_request("c3.stop/v1")
                self._stop_sent = True
            except Exception as exc:              # logged, not retried: the watchdog stops the arm
                log.error("stop failed while concluding: %s", exc)
                self._safe_event("stop_failed", str(exc))
        reason = verdict.what_failed
        if problem.detail and problem.detail not in reason:
            reason = f"{reason}: {problem.detail}"
        self.flow.abort_open(reason, frames_ok=problem.process != "frames")
        self._safe_event("conclude", self._final_line)
        self._exit_code = 1
        self._shutdown()
        raise _Finished

    def _shutdown(self) -> None:
        self._phase = Phase.DONE
        if self._shut:
            return
        self._shut = True
        self._beating = False
        for step in (self._listener.stop, self._log.close):
            try:
                step()
            except Exception as exc:
                log.error("shutdown: %s", exc)

    def _safe_event(self, kind: str, detail: str) -> None:
        try:
            self._log.log_event(kind, detail)
        except Exception as exc:
            log.error("event log: %s", exc)

    # ---- helpers -------------------------------------------------------------------------------------------------

    def _pc_mode(self) -> str:
        latest = self._history.latest()
        return latest.mode if latest is not None else ""

    def _rpc(self, process: str, request: str, fn: Callable, *args):
        """A main-thread client call, remembered so a failure names whom it went to."""
        self._call = (process, request)
        return fn(*args)

    def _refusal_cause(self, problem: Problem) -> Problem:
        """The PC refused a request because of its mode, possibly before the C2 showing that mode arrived here: one
        get_state (a mode check only, never a C2 sample) names the cause, fault or an unrequested stop."""
        try:
            state = self._pc.get_state()
        except Exception as exc:                  # the refusal stays the reported problem
            log.error("get_state after a refusal: %s", exc)
            return problem
        if state.mode == "fault":
            return Problem(C.ARM_FAULT, PC, state.error or "no error text")
        if state.mode == "stopped" and self._phase is Phase.RECORDING and not self._stop_sent:
            return Problem(C.PC_STOPPED, PC, state.error or "mode stopped")
        return problem

    def _guard(self, fn: Callable, *args) -> None:
        """Any exception on the main thread is a conclude, except an EPISODE problem while recording (the PC stopped
        the arm and refused our next set): that aborts the episode and the loop goes on."""
        try:
            fn(*args)
        except _Finished:
            pass
        except Exception as exc:
            log.exception("main thread: %s", exc)
            problem = self._monitor.from_exception(exc, *self._call)
            latest = self._history.latest()
            if latest is not None and latest.mode == "fault" and problem.code is not C.PC_RESTARTED:
                # the arm faulted and the PC then refused a request: report the cause, as cycle step 2 does
                problem = Problem(C.ARM_FAULT, PC, latest.error or "no error text")
            elif problem.code is C.PC_REFUSED:
                problem = self._refusal_cause(problem)
            try:
                if problem.code is C.PC_STOPPED and self._phase is Phase.RECORDING:
                    self._problem(problem)
                else:
                    self._conclude(problem, skip_stop=self._call == STOP_CALL)
            except _Finished:
                pass
            except Exception as exc2:             # an abort that itself failed: conclude on that
                log.exception("while handling %s: %s", problem, exc2)
                try:
                    self._conclude(self._monitor.from_exception(exc2, *self._call))
                except _Finished:
                    pass

    def _log_periodic(self, now: int) -> None:
        """clock.jsonl every clock.log_every_s, link.jsonl every link_stats.window_s."""
        if now >= self._next_clock_log_ns:
            self._next_clock_log_ns = now + int(self._s.clock.log_every_s * NS_PER_S)
            self._log.log_clock(self.clock_row(now))
        if now >= self._next_link_log_ns:
            self._next_link_log_ns = now + int(self._s.link_stats_window_s * NS_PER_S)
            self._log.log_link(self.link_row(now))

    def clock_row(self, now: int) -> dict:
        st = self._pc_clock.stats(now)
        return {"kind": "fit", "estimator": st.estimator, "ready": st.ready, "offset_ns": st.offset_ns,
                "drift_ppm": st.drift_ppm, "samples": st.samples, "used": st.used, "rtt_min_ms": st.rtt_min_ms,
                "newest_age_s": st.newest_age_s, "span_s": st.span_s}

    def link_row(self, now: int) -> dict:
        snap = self._link.snapshot(now)
        newest_age_ms = None
        latest = self._history.latest()
        try:
            if latest is not None:
                newest_age_ms = (self._pc_clock.snapshot().to_pc_ns(now) - int(latest.t_ns)) / NS_PER_MS
        except NotSynced:
            pass
        return {"rtt_n": snap.rtt_n, "rtt_p50_ms": snap.rtt_p50_ms, "rtt_p95_ms": snap.rtt_p95_ms,
                "rtt_max_ms": snap.rtt_max_ms, "c2_rate_hz": snap.c2_rate_hz, "c2_received": snap.c2_received,
                "seq_gaps": snap.seq_gaps, "newest_c2_age_ms": newest_age_ms}


#: U1 request class -> handler (STRUCTURE.md 6.6). A new U1 request is one entry here plus its contract class.
U1_HANDLERS: dict[type[Message], Callable[[TeleopLoop, Message], Message]] = {
    GetStatus: TeleopLoop._on_status,
    KeysUpdate: TeleopLoop._on_keys,
    GripperCommand: TeleopLoop._on_gripper,
    SelectTask: TeleopLoop._on_select,
    CreateTask: TeleopLoop._on_create,
    HomeRequest: TeleopLoop._on_home,
    StartEpisode: TeleopLoop._on_start,
    StopEpisode: TeleopLoop._on_stop,
    DeleteEpisode: TeleopLoop._on_delete,
    StopArm: TeleopLoop._on_stop_arm,
}
