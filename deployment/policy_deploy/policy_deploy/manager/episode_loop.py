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
"""EpisodeLoop: the main thread. Rhythm, keys, every C3 request (rule H1), health polls, failure decisions, logging,
status line (STRUCTURE.md 6.1, 6.4, 6.5, 7.4, 7.5)."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from policy_deploy.contracts import NotSynced, PolicyConfig
from policy_deploy.manager.errors import RequestCancelled
from policy_deploy.manager.failure_monitor import SKIP_CODES, FailureMonitor
from policy_deploy.manager.frames_client import FramesClient
from policy_deploy.manager.heartbeat_sender import HeartbeatSender
from policy_deploy.manager.link_stats import LinkStats
from policy_deploy.manager.operator import Key, Operator
from policy_deploy.manager.pc_client import PcClient
from policy_deploy.manager.pc_clock import PcClock
from policy_deploy.manager.policy_step import NoNewFrame, StepDone, StepJob, StepSkipped
from policy_deploy.manager.problems import Kind, Problem, ProblemCode, Verdict
from policy_deploy.manager.session_log import SKIP_PREFIX, Outcome, SessionLog, StepRecord
from policy_deploy.manager.settings import Settings
from policy_deploy.manager.startup_checks import StartupChecks
from policy_deploy.manager.state_history import StateHistory
from policy_deploy.manager.status_line import StatusLine, StatusSnapshot
from policy_deploy.manager.step_worker import StepWorker, WorkerResult

log = logging.getLogger("manager.episode_loop")
C = ProblemCode
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
PC = "PC"
STOP_CALL = (PC, "c3.stop/v1")
LAUNCHER_STOP_REASON = "stopped by the launcher: another process exited"
NOT_CHECKED_PREFIX = "not checked"     # the other waypoints of a rejected request (TOP 3.7.1)


def gate_rejection_detail(reply) -> str:
    """'<index>: <reason>' of the first rejected waypoint that failed a rule itself (the PC lists every non-skipped
    waypoint of a rejected request; the others say 'not checked: another waypoint of this request was rejected')."""
    issue = next((i for i in reply.rejected if not i.reason.startswith(NOT_CHECKED_PREFIX)), reply.rejected[0])
    return f"{issue.index}: {issue.reason}"


class Phase(Enum):
    STARTING = "starting"         # start-up (STRUCTURE.md 6.3)
    IDLE = "idle"                 # no episode; waiting for a key
    RESETTING = "resetting"       # start key pressed: policy reset (+ prompt), then the PC's start
    RUNNING = "running"           # an episode: policy steps
    HOMING = "homing"             # blocking home request
    DONE = "done"                 # finished; run() returns


@dataclass(frozen=True)
class LoopProgress:               # what ScriptedOperator may wait on
    phase: Phase
    episode_id: int               # 0 before the first episode
    steps_sent: int               # waypoint sets sent in this episode
    now_ns: int
    worker_busy: bool = False     # a policy step (or reset) still on the worker: c and h are refused meanwhile
    pc_mode: str = ""             # the newest C2 mode (a refused c/h is retried when it changes)
    clock_ready: bool = True


class _Finished(Exception):
    """Control flow only: the loop has finished (phase DONE); the rest of this cycle is skipped."""


class EpisodeLoop:
    def __init__(self, settings: Settings, pc: PcClient, heartbeats: HeartbeatSender, pc_clock: PcClock,
                 link_stats: LinkStats, health: FramesClient, worker: StepWorker, history: StateHistory,
                 listener, monitor: FailureMonitor, operator: Operator, log: SessionLog, checks: StartupChecks,
                 clock: Callable[[], int], sleep: Callable[[float], None]) -> None:
        self._s = settings
        self._pc = pc
        self._heartbeats = heartbeats
        self._pc_clock = pc_clock
        self._link = link_stats
        self._health = health
        self._worker = worker
        self._history = history
        self._listener = listener
        self._monitor = monitor
        self._operator = operator
        self._log = log
        self._checks = checks
        self._clock = clock
        self._sleep = sleep
        self._status = StatusLine()
        self._loop_ns = int(round(settings.heartbeat_period_s * NS_PER_S))
        self._interrupt = threading.Event()
        self._terminate = threading.Event()

        self._phase = Phase.STARTING
        self._verdict: Verdict | None = None
        self._exit_code = 0
        self._shut = False
        self._call: tuple[str, str] = ("manager", "")
        self._beating = False             # hellos/heartbeats started (start-up step 4); nothing on C3 before
        self._rewarm = False              # after a home: a hello every cycle until the map is ready
        self._policy_config: PolicyConfig | None = None
        self._uses_prompt = False
        self._hello = None                # the newest hello reply at start-up (session.yaml, status line)
        self._last_health = None
        self._next_health_ns = 0
        self._next_status_ns = 0
        self._next_clock_log_ns = 0
        self._next_link_log_ns = 0
        self._submitted = False
        self._stop_sent = False
        self._start_ack_ns: int | None = None
        self._start_ack_pc_ns: int | None = None
        self._episode_id = 0
        self._next_chunk_id = 1
        self._mismatch_count = 0
        self._hint_text = ""
        self._new_episode_state()

    def _new_episode_state(self) -> None:
        self._steps_sent = 0
        self._start_pose: np.ndarray | None = None
        self._episode_t0 = 0
        self._gate0 = (0, 0, 0)
        self._last_anchor_t_ns: int | None = None     # laptop clock (NO_NEW_FRAME)
        self._prev_last_wp_pc_ns: int | None = None   # PC clock (the margin)
        self._prev_send_ns: int | None = None
        self._skipping_since: int | None = None
        self._skipping_reason = ""
        self._typical_step_ms: float | None = None
        self._mismatch_count = 0
        self._last = {"cycle_ms": None, "policy_ms": None, "image_age_ms": None, "margin_ms": None,
                      "anchor_dev_mm": None}

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

    def interrupt(self) -> None:
        """SIGINT handler: sets the interrupt flag, nothing else."""
        self._interrupt.set()

    def terminate(self) -> None:
        """SIGTERM handler: sets the terminate flag, nothing else."""
        self._terminate.set()

    def interrupt_set(self) -> bool:
        """Read by PcClient.home between slices: Ctrl-C or SIGTERM."""
        return self._interrupt.is_set() or self._terminate.is_set()

    def progress(self) -> LoopProgress:
        latest = self._history.latest()
        now = self._clock()
        return LoopProgress(self._phase, self._episode_id, self._steps_sent, now, self._worker.busy(),
                            latest.mode if latest is not None else "", self._pc_clock.ready(now))

    def startup(self, ready_file: Path | None) -> None:
        """STRUCTURE.md 6.3. On failure sets the verdict and phase DONE."""
        from policy_deploy.manager.startup_sequence import StartupSequence
        self._guard(StartupSequence(self).run, ready_file)

    def cycle(self) -> None:
        """One loop cycle (STRUCTURE.md 6.4, steps 1-7)."""
        if self._phase is Phase.DONE:
            return
        self._submitted = False
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
        if self._beating and self._phase is not Phase.HOMING:                       # 3
            self._beat(now)
        if self._worker.busy():                     # a result that is already there (run() handles most of them)
            result = self._worker.wait(0.0)
            if result is not None:
                self._handle_result(result)
        key = self._operator.poll_key(self.progress())                              # 4
        if key is not None:
            self._on_key(key)
        now = self._clock()
        if now >= self._next_health_ns:                                             # 5
            self._poll_health(now)
        if self._phase is Phase.RUNNING and not self._worker.busy() and not self._submitted:   # 6
            self._submit_step()
        if now >= self._next_status_ns:                                             # 7
            self._show_status(now)
        self._log_periodic(now)

    def _wait_until_boundary(self, boundary: int) -> None:
        while self._phase is not Phase.DONE:
            remaining = boundary - self._clock()
            if remaining <= 0:
                return
            if self._worker.busy():
                result = self._worker.wait(remaining / NS_PER_S)
                if result is not None:
                    self._handle_result(result)
            else:
                self._sleep(remaining / NS_PER_S)

    def _beat(self, now: int) -> None:
        """The cycle's one C3 keep-alive; hello samples feed the map, fit jumps are logged (guard 3)."""
        self._call = (PC, "c3.heartbeat/v1")
        try:
            reply = self._heartbeats.beat(now)
        finally:
            self._call = (PC, self._heartbeats.last_request or "c3.heartbeat/v1")
        if reply is not None:
            for jump in self._pc_clock.take_jumps():
                self._safe_event("clock_jump", f"refit moved the PC time by {jump.jump_ms:+.2f} ms")
                self._log.log_clock({"kind": "jump", "estimator": self._pc_clock.NAME, "jump_ms": jump.jump_ms})
        if self._rewarm and self._pc_clock.ready(self._clock()):
            self._rewarm = False
            self._heartbeats.startup_mode(False)
            self._safe_event("clock", "PC clock map ready again")

    def _check_flags(self) -> None:
        if self._interrupt.is_set():
            self._quit("Ctrl-C")
        if self._terminate.is_set():
            self._launcher_stop()

    def _check_state(self, now: int) -> None:
        """The state rules first (a fault is the cause when a broken message follows it), then the listener."""
        problem = self._monitor.check_state(self._history.latest(), now, self._listener.last_arrival_ns(),
                                            self._phase is Phase.RUNNING, self._stop_sent, self._start_ack_pc_ns)
        if problem is not None:
            self._conclude(problem)
        error = self._listener.error()
        if error is not None:
            self._conclude(self._monitor.from_exception(error, PC, "c2.state/v1"))

    def _poll_health(self, now: int):
        self._next_health_ns = now + self._s.health_poll_ms * NS_PER_MS
        report = self._rpc("frames", "c1.get_health/v1", self._health.get_health)
        self._last_health = report
        problem = self._monitor.check_health(report)
        if problem is not None:
            self._conclude(problem)
        return report

    def _pc_mode(self) -> str:
        latest = self._history.latest()
        return latest.mode if latest is not None else ""

    # ---- keys (STRUCTURE.md 6.5) ---------------------------------------------------------------------------------

    def _on_key(self, key: Key) -> None:
        KEY_HANDLERS[key.value](self, key)

    def _key_quit(self, key: Key) -> None:
        self._quit("operator quit key")

    def _key_start(self, key: Key) -> None:
        phase = self._phase
        if phase is Phase.IDLE:
            if self._worker.busy():
                self._refuse(key, "c: waiting for the last policy step to finish")
                return
            if self._pc_mode() == "homing":
                self._refuse(key, "c: the PC is homing (its h or g key): wait for idle")
                return
            if not self._pc_clock.ready(self._clock()):
                self._refuse(key, "c: waiting for the PC clock map")
                return
            prompt = (self._s.task_name, self._s.prompt_index) if self._uses_prompt else None
            self._worker.submit_reset(prompt)
            self._steps_sent = 0
            self._phase = Phase.RESETTING
            self._log.log_event("reset", "policy reset" + (" + prompt" if prompt else ""))
        elif phase is Phase.RUNNING:
            self._refuse(key, "c: an episode is running (y success, n fail, s stop)")
        else:
            self._refuse(key, "c: the episode is already starting (s cancels)")

    def _key_end(self, key: Key) -> None:
        phase = self._phase
        if phase is Phase.RUNNING:
            self._end_episode(Outcome.SUCCESS if key is Key.SUCCESS else Outcome.FAIL, "operator")
        elif phase is Phase.RESETTING:
            self._refuse(key, f"{key.value}: the episode is still starting (s cancels)")
        else:
            self._refuse(key, f"{key.value}: no episode running (c start, h home, q quit)")

    def _key_home(self, key: Key) -> None:
        if self._phase is not Phase.IDLE:
            self._refuse(key, "h: end the episode first (y / n / s)")
        elif self._worker.busy():
            self._refuse(key, "h: waiting for the last policy step to finish")
        elif self._pc_mode() == "homing":
            self._refuse(key, "h: the PC is already homing (its h or g key)")
        else:
            self._home()

    def _key_stop(self, key: Key) -> None:
        """Everything s does: stop (the arm holds), end the episode if any as stopped, back to IDLE."""
        phase = self._phase
        if phase is Phase.RUNNING:
            self._end_episode(Outcome.STOPPED, "operator stop key")
        elif phase is Phase.RESETTING:
            self._send_stop()
            self._phase = Phase.IDLE
            self._log.log_event("start_cancelled", "operator stop key")
            self._operator.show_message("start cancelled")
        else:
            self._send_stop()
            self._hint("s: no episode running (c start, h home, q quit)")

    def _hint(self, text: str) -> None:
        self._hint_text = text
        self._operator.show_message(text)

    def _refuse(self, key: Key, text: str) -> None:
        """The key doesn't fit now and is ignored: the hint, and the operator is told (a script retries it)."""
        self._hint(text)
        self._operator.key_refused(key, text)

    # ---- episodes ------------------------------------------------------------------------------------------------

    def _start_episode(self) -> None:
        """The reset (+ prompt) answered: the PC's start, the start pose, phase RUNNING, the first step."""
        ack = self._rpc(PC, "c3.start/v1", self._pc.start)
        if not ack.ok:
            self._phase = Phase.IDLE
            self._log.log_event("start_refused", ack.reason)
            self._hint(f"c: the PC refused start: {ack.reason}")
            return
        self._stop_sent = False
        self._start_ack_ns = self._clock()
        self._start_ack_pc_ns = self._pc_clock.snapshot().to_pc_ns(self._start_ack_ns)
        latest = self._history.latest()
        pose = latest.O_T_TCP_target if self._s.pose_source == "target" else latest.O_T_TCP
        self._new_episode_state()
        self._start_pose = np.array(pose, dtype=np.float64)
        self._episode_id += 1
        self._episode_t0 = self._clock()
        self._gate0 = (latest.gate.shortened, latest.gate.pulled_back, latest.loop_late)
        self._log.start_episode(self._episode_id, self._start_pose)
        self._monitor.reset_episode_counts()
        self._phase = Phase.RUNNING
        self._hint_text = ""
        self._log.log_event("episode_start", f"episode {self._episode_id}")
        self._operator.show_message(f"episode {self._episode_id} started")
        self._submit_step()

    def _end_episode(self, outcome: Outcome, reason: str) -> None:
        """y / n / s in RUNNING."""
        self._send_stop()
        self._close_episode(outcome, reason)
        self._phase = Phase.IDLE

    def _close_episode(self, outcome: Outcome, reason: str) -> None:
        record = self._log.end_episode(outcome, reason, self._episode_counts())
        self._operator.show_message(f"episode {record.episode_id}: {outcome.value} ({reason}), "
                                    f"{record.steps_sent} steps, {record.duration_s:.1f} s")

    def _episode_counts(self) -> dict[str, int]:
        counts = self._monitor.counts()
        shortened, pulled_back, loop_late = self._gate_deltas()
        out = {"steps_sent": self._steps_sent, "late": counts.get(C.POLICY_LATE, 0),
               "anchor_fallbacks": counts.get(C.ANCHOR_FALLBACK, 0), "odd_chunks": counts.get(C.ODD_CHUNK, 0),
               "shortened": shortened, "pulled_back": pulled_back, "loop_late": loop_late,
               "width_clips": counts.get(C.WIDTH_CLIPPED, 0)}
        for code in SKIP_CODES:
            if counts.get(code):
                out[SKIP_PREFIX + code.value] = counts[code]
        return out

    def _gate_deltas(self) -> tuple[int, int, int]:
        latest = self._history.latest()
        if latest is None:
            return 0, 0, 0
        now = (latest.gate.shortened, latest.gate.pulled_back, latest.loop_late)
        if self._phase is not Phase.RUNNING:
            return now                                  # session totals
        return tuple(a - b for a, b in zip(now, self._gate0))

    def _submit_step(self) -> None:
        job = StepJob(self._episode_id, self._next_chunk_id, self._start_pose, self._last_anchor_t_ns)
        self._next_chunk_id += 1
        self._worker.submit_step(job)
        self._submitted = True

    # ---- results -------------------------------------------------------------------------------------------------

    def _handle_result(self, result: WorkerResult) -> None:
        if result.error is not None:
            self._conclude(self._monitor.from_exception(result.error, result.process, result.request))
        if result.kind == "reset":
            if self._phase is Phase.RESETTING:
                self._start_episode()
            else:
                self._log.log_event("reset_dropped", "start was cancelled")
        elif result.kind == "step":
            self._handle_step(result)
        else:
            self._log.log_event("result_dropped", result.kind)

    def _handle_step(self, result: WorkerResult) -> None:
        value = result.value
        if value.job.episode_id != self._episode_id or self._phase is not Phase.RUNNING:
            self._log.log_event("step_dropped", f"chunk {value.job.chunk_id}: its episode has ended")
            return
        if isinstance(value, NoNewFrame):
            self._submitted = True                       # the next try is in the next cycle
            return
        if isinstance(value, StepSkipped):
            self._handle_skip(value, result)
            return
        self._send(value)

    def _handle_skip(self, skip: StepSkipped, result: WorkerResult) -> None:
        if skip.width_clipped:
            self._monitor.classify(Problem(C.WIDTH_CLIPPED, "policy", f"{skip.width_clipped} widths clipped"),
                                   count=skip.width_clipped)
        verdict = self._monitor.classify(skip.problem)
        if verdict.kind is Kind.CONCLUDE:
            self._conclude(skip.problem)
        now = self._clock()
        if self._skipping_since is None:
            self._skipping_since = now
        self._skipping_reason = verdict.what_failed
        self._submitted = True                           # retried in the next cycle, not in a tight loop
        self._log.log_step(StepRecord(
            episode_id=skip.job.episode_id, chunk_id=skip.job.chunk_id, t_anchor_ns=0, t_anchor_pc_ns=0,
            anchor_source="", anchor_dev_mm=0.0, image_age_ms=0.0, frames_ms=0.0, policy_ms=0.0,
            cycle_ms=(result.finished_ns - result.started_ns) / NS_PER_MS, sent=False,
            skip_reason=f"{skip.problem.code.value}: {skip.problem.detail}",
            accepted=0, skipped=0, shortened=0, rejected=0, margin_ms=None, c3_rtt_ms=None))

    def _send(self, done: StepDone) -> None:
        """STRUCTURE.md 7.4 step 4 and 7.5: set_waypoints, the reply checks, the margin and the clock guard."""
        s = self._s
        if done.anchor.source != s.pose_source:
            self._monitor.classify(Problem(C.ANCHOR_FALLBACK, PC,
                                           f"target and measured {done.anchor_dev_m * 1000:.0f} mm apart: "
                                           "measured used for this step"))
        if done.chunk.width_clipped:
            self._monitor.classify(Problem(C.WIDTH_CLIPPED, "policy", f"{done.chunk.width_clipped} widths clipped"),
                                   count=int(done.chunk.width_clipped))
        snap = self._pc_clock.snapshot()
        t_send = self._clock()
        send_pc = snap.to_pc_ns(t_send)
        latest = self._history.latest()
        reply = self._rpc(PC, "c3.set_waypoints/v1", self._pc.set_waypoints, done.waypoints)
        c3_rtt_ms = (self._clock() - t_send) / NS_PER_MS
        margin_ms = None if self._prev_last_wp_pc_ns is None else (self._prev_last_wp_pc_ns - send_pc) / NS_PER_MS
        if reply.rejected:
            self._conclude(Problem(C.GATE_REJECTED, PC, gate_rejection_detail(reply)))
        n = len(done.waypoints.t_ns)
        lead_ms = (int(done.waypoints.t_ns[-1]) - send_pc) / NS_PER_MS
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
        all_skipped = reply.skipped == n
        late_margin = margin_ms is not None and margin_ms < 0
        # one cause per set: an entirely skipped set that counted toward CLOCK_MISMATCH is not also POLICY_LATE
        if (all_skipped and not (skipped_ahead or future_state)) or (not all_skipped and late_margin):
            late = f"margin {margin_ms:+.0f} ms" if margin_ms is not None else "every waypoint already past"
            self._monitor.classify(Problem(C.POLICY_LATE, "policy", f"policy late: {late}"))
        timing = done.timing
        cycle_ms = (t_send - (self._prev_send_ns if self._prev_send_ns is not None else timing.t_begin_ns)) / NS_PER_MS
        self._steps_sent += 1
        self._last_anchor_t_ns = done.anchor.t_ns
        self._prev_last_wp_pc_ns = int(done.waypoints.t_ns[-1])
        self._prev_send_ns = t_send
        self._skipping_since = None
        self._typical_step_ms = (timing.t_end_ns - timing.t_begin_ns) / NS_PER_MS
        policy_ms = (timing.t_policy_ns - timing.t_frames_ns) / NS_PER_MS
        self._last = {"cycle_ms": cycle_ms, "policy_ms": policy_ms, "image_age_ms": timing.image_age_ms,
                      "margin_ms": margin_ms, "anchor_dev_mm": done.anchor_dev_m * 1000}
        self._submit_step()                                                         # continuous control
        self._log.log_step(StepRecord(
            episode_id=done.job.episode_id, chunk_id=done.job.chunk_id, t_anchor_ns=done.anchor.t_ns,
            t_anchor_pc_ns=done.anchor.t_pc_ns, anchor_source=done.anchor.source,
            anchor_dev_mm=done.anchor_dev_m * 1000, image_age_ms=timing.image_age_ms,
            frames_ms=(timing.t_frames_ns - timing.t_begin_ns) / NS_PER_MS, policy_ms=policy_ms, cycle_ms=cycle_ms,
            sent=True, skip_reason="", accepted=reply.accepted, skipped=reply.skipped, shortened=reply.shortened,
            rejected=len(reply.rejected), margin_ms=margin_ms, c3_rtt_ms=c3_rtt_ms))
        if s.save_tap:
            self._log.save_tap(done, reply)

    # ---- home, quit, launcher stop, conclude ---------------------------------------------------------------------

    def _home(self) -> None:
        """STRUCTURE.md 6.4: heartbeats and hellos pause; Ctrl-C / SIGTERM cancel the wait (no stop is sent)."""
        self._phase = Phase.HOMING
        self._show_status(self._clock())
        self._log.log_event("home", "homing started")
        try:
            ack = self._rpc(PC, "c3.home/v1", self._pc.home, self.interrupt_set)
        except RequestCancelled:
            self._log.log_event("home", "cancelled")
            if self._terminate.is_set():
                self._launcher_stop(send_stop=False)
            self._quit("Ctrl-C", send_stop=False)
        if ack.ok:
            self._phase = Phase.IDLE
            self._log.log_event("home", "home pose reached")
            self._operator.show_message("home pose reached")
        elif self._pc.home_stopped(ack):
            self._phase = Phase.IDLE
            verdict = self._monitor.classify(Problem(C.HOME_STOPPED, PC, ack.reason))
            self._log.log_event("home_stopped", f"{verdict.what_failed}: {verdict.what_to_check}")
            self._hint(f"{verdict.what_failed}: {verdict.what_to_check}")
        elif self._pc.home_refused_homing(ack):
            self._phase = Phase.IDLE
            self._log.log_event("home_refused", ack.reason)
            self._hint(f"h: the PC is already homing (its h or g key): {ack.reason}")
        else:
            self._conclude(Problem(C.HOMING_FAILED, PC, ack.reason or "refused"))
        self._rewarm = True                                  # the map had no samples during the home
        self._heartbeats.startup_mode(True)
        self._beat(self._clock())                            # heartbeats resume at once
        self._show_status(self._clock())

    def _send_stop(self) -> None:
        self._rpc(PC, "c3.stop/v1", self._pc.stop)
        self._stop_sent = True

    def _quit(self, reason: str, send_stop: bool = True) -> None:
        """q or Ctrl-C: stop first, the open episode recorded as stopped, exit 0."""
        self._finish(Outcome.STOPPED, reason, Problem(C.OPERATOR_STOP, "operator", reason), send_stop)

    def _launcher_stop(self, send_stop: bool = True) -> None:
        """SIGTERM: stop first, the open episode recorded as aborted, exit 0."""
        self._finish(Outcome.ABORTED, LAUNCHER_STOP_REASON,
                     Problem(C.LAUNCHER_STOP, "launcher", LAUNCHER_STOP_REASON), send_stop)

    def _finish(self, outcome: Outcome, reason: str, problem: Problem, send_stop: bool) -> None:
        if send_stop and self._beating:
            self._send_stop()
        if self._phase is Phase.RUNNING:
            self._close_episode(outcome, reason)
        self._verdict = self._monitor.classify(problem)
        self._log.log_event("exit", reason)
        self._exit_code = 0
        self._shutdown()
        raise _Finished

    def _conclude(self, problem: Problem, skip_stop: bool = False) -> None:
        """Stop once (if C3 was started), record aborted, stop everything, exit 1 (the FATAL line is __main__'s).
        The PC keeps running, holding the arm."""
        verdict = self._monitor.classify(problem)
        if verdict.kind is not Kind.CONCLUDE:
            raise RuntimeError(f"conclude called for a {verdict.kind.value} problem: {problem}")
        self._verdict = verdict
        log.error("conclude: %s: %s", verdict.what_failed, verdict.what_to_check)
        if not skip_stop and self._beating:
            try:
                self._pc.stop()
                self._stop_sent = True
            except Exception as exc:              # logged, not retried: the watchdog stops the arm
                log.error("stop failed while concluding: %s", exc)
                self._safe_event("stop_failed", str(exc))
        if self._phase is Phase.RUNNING:
            reason = verdict.what_failed
            if problem.detail and problem.detail not in reason:
                reason = f"{reason}: {problem.detail}"
            self._log.end_episode(Outcome.ABORTED, reason, self._episode_counts())
        self._safe_event("conclude", f"{verdict.what_failed}: {verdict.what_to_check}")
        self._exit_code = 1
        self._shutdown()
        raise _Finished

    def _shutdown(self) -> None:
        self._phase = Phase.DONE
        if self._shut:
            return
        self._shut = True
        self._beating = False
        for step in (self._listener.stop, self._worker.stop, self._operator.close, self._log.close):
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
        if state.mode == "stopped" and self._phase is Phase.RUNNING and not self._stop_sent:
            return Problem(C.PC_STOPPED, PC, state.error or "mode stopped")
        return problem

    def _guard(self, fn: Callable, *args) -> None:
        """Any exception on the main thread is a conclude."""
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
                self._conclude(problem, skip_stop=self._call == STOP_CALL)
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

    def _show_status(self, now: int) -> None:
        self._next_status_ns = now + self._s.status_ms * NS_PER_MS
        self._operator.show_status(self._status.format(self._snapshot(now)))

    def _snapshot(self, now: int) -> StatusSnapshot:
        counts = self._monitor.counts()
        running = self._phase is Phase.RUNNING
        waiting = None
        busy_ms = self._worker.running_for_ms()
        if running and busy_ms is not None and self._typical_step_ms and busy_ms > 2 * self._typical_step_ms:
            waiting = busy_ms
        skipping_s = None
        if running and self._skipping_since is not None:
            skipping = (now - self._skipping_since) / NS_PER_S
            if skipping > self._s.skip_warn_s:
                skipping_s = skipping
        shortened, pulled_back, loop_late = self._gate_deltas()
        latest = self._history.latest()
        link = self._link.snapshot(now)
        hello = self._heartbeats.last_hello()
        return StatusSnapshot(
            phase=self._phase, episode_id=self._episode_id,
            episode_s=(now - self._episode_t0) / NS_PER_S if running else 0.0,
            steps_sent=self._steps_sent, cycle_ms=self._last["cycle_ms"], policy_ms=self._last["policy_ms"],
            image_age_ms=self._last["image_age_ms"], margin_ms=self._last["margin_ms"], waiting_policy_ms=waiting,
            skips={code.value: counts[code] for code in SKIP_CODES if counts.get(code)},
            late=counts.get(C.POLICY_LATE, 0), anchor_fallbacks=counts.get(C.ANCHOR_FALLBACK, 0),
            odd_chunks=counts.get(C.ODD_CHUNK, 0), skipping_s=skipping_s, skipping_reason=self._skipping_reason,
            shortened=shortened, pulled_back=pulled_back, loop_late=loop_late,
            width_clips=counts.get(C.WIDTH_CLIPPED, 0), cameras=self._last_health,
            arm_mode=latest.mode if latest is not None else None, anchor_dev_mm=self._last["anchor_dev_mm"],
            rtt_ms=link.rtt_p50_ms, c2_hz=link.c2_rate_hz, clock_ok=self._pc_clock.ready(now) if self._beating else None,
            pc_mock=bool(hello.mock) if hello is not None else False,
            hint=self._hint_text if self._phase is Phase.IDLE else "")


#: Key value -> handler. A later jog key is one entry here plus one Key member (STRUCTURE.md 6.5).
KEY_HANDLERS: dict[str, Callable[[EpisodeLoop, Key], None]] = {
    Key.START.value: EpisodeLoop._key_start,
    Key.SUCCESS.value: EpisodeLoop._key_end,
    Key.FAIL.value: EpisodeLoop._key_end,
    Key.STOP.value: EpisodeLoop._key_stop,
    Key.HOME.value: EpisodeLoop._key_home,
    Key.QUIT.value: EpisodeLoop._key_quit,
}
