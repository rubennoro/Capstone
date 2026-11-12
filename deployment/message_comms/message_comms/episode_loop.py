"""EpisodeLoop: the main thread. Rhythm, keys, heartbeats, waypoints, failure decisions, logging, status line.

STRUCTURE.md 4.8 and section 5. main-sim franka_task.run_policy_control was the reference only.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from contracts import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS, PolicyConfig
from message_comms.errors import RequestCancelled
from message_comms.failure_monitor import SKIP_CODES, FailureMonitor
from message_comms.frames_client import FramesClient
from message_comms.franka_client import FrankaClient
from message_comms.operator import Key, Operator
from message_comms.policy_step import NoNewFrame, StepDone, StepJob, StepSkipped
from message_comms.problems import Kind, Problem, ProblemCode, Verdict
from message_comms.session_log import SKIP_PREFIX, Outcome, SessionLog, StepRecord
from message_comms.settings import Settings
from message_comms.state_history import StateHistory
from message_comms.state_listener import StateListener
from message_comms.status_line import StatusLine, StatusSnapshot
from message_comms.step_worker import StepWorker, WorkerResult

log = logging.getLogger("message_comms.episode_loop")
C = ProblemCode
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
STOP_CALL = ("franka_control", "c3.stop/v1")
DEPLOY_STOP_REASON = "stopped by deploy: another process exited"


class Phase(Enum):
    STARTING = "starting"         # startup checks (section 5.1)
    IDLE = "idle"                 # no episode; waiting for a key
    RESETTING = "resetting"       # start key pressed: policy reset (+ prompt), then franka start
    RUNNING = "running"           # an episode: policy steps
    HOMING = "homing"             # blocking home request
    DONE = "done"                 # finished; run() returns


@dataclass(frozen=True)
class LoopProgress:               # what ScriptedOperator may wait on
    phase: Phase
    episode_id: int               # 0 before the first episode
    steps_sent: int               # waypoint sets sent in this episode
    now_ns: int


class _Finished(Exception):
    """Control flow only: the loop has finished (phase DONE); the rest of this cycle is skipped."""


class EpisodeLoop:
    def __init__(self, settings: Settings, franka: FrankaClient, health: FramesClient, worker: StepWorker,
                 history: StateHistory, listener: StateListener, monitor: FailureMonitor,
                 operator: Operator, log: SessionLog, clock: Callable[[], int],
                 sleep: Callable[[float], None]) -> None:
        self._s = settings
        self._franka = franka
        self._health = health
        self._worker = worker
        self._history = history
        self._listener = listener
        self._monitor = monitor
        self._operator = operator
        self._log = log
        self._clock = clock
        self._sleep = sleep
        self._status = StatusLine()
        self._loop_ns = settings.loop_ms * NS_PER_MS
        self._interrupt = threading.Event()
        self._terminate = threading.Event()

        self._phase = Phase.STARTING
        self._verdict: Verdict | None = None
        self._exit_code = 0
        self._shut = False
        self._call: tuple[str, str] = ("message_comms", "")
        self._heartbeats = False
        self._policy_config: PolicyConfig | None = None
        self._uses_prompt = False
        self._deadline_ns = 0
        self._why = ""
        self._last_health = None
        self._next_health_ns = 0
        self._next_status_ns = 0
        self._rx_count = 0
        self._last_rx_ns: int | None = None
        self._submitted = False
        self._stop_sent = False
        self._start_ack_ns: int | None = None
        self._episode_id = 0
        self._next_chunk_id = 1
        self._new_episode_state()

    def _new_episode_state(self) -> None:
        self._steps_sent = 0
        self._start_pose: np.ndarray | None = None
        self._episode_t0 = 0
        self._gate0 = (0, 0, 0)
        self._last_anchor_t_ns: int | None = None
        self._prev_last_wp_ns: int | None = None
        self._prev_send_ns: int | None = None
        self._skipping_since: int | None = None
        self._skipping_reason = ""
        self._typical_step_ms: float | None = None
        self._last = {"cycle_ms": None, "policy_ms": None, "image_age_ms": None, "margin_ms": None,
                      "anchor_dev_mm": None}

    # ---- public interface ----------------------------------------------------------------------------------------

    @property
    def verdict(self) -> Verdict | None:
        return self._verdict

    @property
    def phase(self) -> Phase:
        return self._phase

    def interrupt(self) -> None:
        """SIGINT handler: sets the interrupt flag, nothing else."""
        self._interrupt.set()

    def terminate(self) -> None:
        """SIGTERM handler: sets the terminate flag, nothing else."""
        self._terminate.set()

    def interrupt_set(self) -> bool:
        """Read by FrankaClient.home between slices: Ctrl-C or SIGTERM."""
        return self._interrupt.is_set() or self._terminate.is_set()

    def progress(self) -> LoopProgress:
        return LoopProgress(self._phase, self._episode_id, self._steps_sent, self._clock())

    def startup(self, ready_file: Path | None) -> None:
        """Section 5.1. On failure sets the verdict and phase DONE."""
        self._deadline_ns = self._clock() + int(self._s.timeouts.startup_s * NS_PER_S)
        self._guard(self._startup, ready_file)

    def cycle(self) -> None:
        """One loop cycle (section 4.8, steps 1-7)."""
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

    # ---- startup -------------------------------------------------------------------------------------------------

    def _startup(self, ready_file: Path | None) -> None:
        s = self._s
        self._log.log_event("startup", "starting")
        self._listener.start()
        self._worker.start()
        self._wait_for(self._state_ready, "franka_control", "no fresh arm state")
        self._rpc("franka_control", "c3.get_state/v1", self._franka.get_state)
        self._heartbeats = True
        self._worker.submit_config()
        self._wait_for(self._config_ready, "policy", "no policy config")
        self._check_config(self._policy_config)
        self._wait_for(self._cameras_ready, "process_frames", "no camera ok")
        self._log.write_session(s, self._policy_config)
        self._operator.start()
        self._phase = Phase.IDLE
        self._show_status(self._clock())
        if ready_file is not None:
            Path(ready_file).write_text("")
        self._log.log_event("ready", "connected; arm state fresh, policy config accepted, both cameras ok")

    def _wait_for(self, ready: Callable[[int], bool], process: str, what: str) -> None:
        while True:
            now = self._clock()
            self._check_flags()
            self._check_state(now)
            if self._heartbeats:
                self._rpc("franka_control", "c3.heartbeat/v1", self._franka.heartbeat)
            if ready(now):
                return
            if now >= self._deadline_ns:
                detail = f"{what} within {self._s.timeouts.startup_s:g} s ({self._why})"
                self._conclude(Problem(C.STARTUP, process, detail))
            self._sleep(self._loop_ns / NS_PER_S)

    def _state_ready(self, now: int) -> bool:
        latest = self._history.latest()
        if latest is None:
            self._why = "no state yet"
            return False
        age_ms = (now - latest.t_ns) / NS_PER_MS
        if age_ms > self._s.max_state_age_ms:
            self._why = f"newest state {age_ms:.0f} ms old"
            return False
        return True

    def _config_ready(self, now: int) -> bool:
        self._why = "waiting for the reply"
        result = self._worker.wait(0.0)
        if result is None:
            return False
        if result.error is not None:
            self._conclude(self._monitor.from_exception(result.error, result.process, result.request))
        self._policy_config = result.value
        return True

    def _check_config(self, cfg: PolicyConfig) -> None:
        for name, expected in (("obs_steps", OBS_STEPS), ("action_steps", ACTION_STEPS), ("step_ms", STEP_MS),
                               ("image_size", IMAGE_SIZE)):
            value = getattr(cfg, name)
            if value != expected:
                self._conclude(Problem(C.CONTRACT_VIOLATION, "policy",
                                       f"policy config: {name} is {value}, message_comms expects {expected}",
                                       check="check the checkpoint"))
        if self._s.exec_h > cfg.action_steps:
            self._conclude(Problem(C.CONTRACT_VIOLATION, "policy",
                                   f"policy config: exec_h {self._s.exec_h} > action_steps {cfg.action_steps}",
                                   check="check message_comms.exec_h"))
        self._uses_prompt = bool(cfg.uses_prompt)
        if self._uses_prompt and self._s.task_name is None:
            self._conclude(Problem(C.STARTUP, "message_comms", "the policy uses a prompt but no task_name is set",
                                   check="check message_comms.task_name"))

    def _cameras_ready(self, now: int) -> bool:
        if now < self._next_health_ns:
            return False
        report = self._poll_health(now)
        bad = [f"{name} {getattr(report, name).status}" for name in ("main", "ultrawide")
               if getattr(report, name).status != "ok"]
        self._why = ", ".join(bad)
        return not bad

    # ---- one cycle -----------------------------------------------------------------------------------------------

    def _cycle(self) -> None:
        now = self._clock()
        self._check_flags()                                                         # 1
        self._check_state(now)                                                      # 2
        if self._heartbeats and self._phase is not Phase.HOMING:                    # 3
            self._rpc("franka_control", "c3.heartbeat/v1", self._franka.heartbeat)
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

    def _check_flags(self) -> None:
        if self._interrupt.is_set():
            self._quit("Ctrl-C")
        if self._terminate.is_set():
            self._deploy_stop()

    def _check_state(self, now: int) -> None:
        received = self._listener.received()
        if received != self._rx_count:
            self._rx_count = received
            self._last_rx_ns = now
        problem = self._monitor.check_state(self._history.latest(), now, self._phase is Phase.RUNNING,
                                            self._stop_sent, self._start_ack_ns, self._last_rx_ns)
        if problem is not None:
            self._conclude(problem)
        error = self._listener.error()
        if error is not None:
            self._conclude(self._monitor.from_exception(error, "franka_control", "c2.state/v1"))

    def _poll_health(self, now: int):
        self._next_health_ns = now + self._s.health_poll_ms * NS_PER_MS
        report = self._rpc("process_frames", "c1.get_health/v1", self._health.get_health)
        self._last_health = report
        problem = self._monitor.check_health(report)
        if problem is not None:
            self._conclude(problem)
        return report

    # ---- keys ----------------------------------------------------------------------------------------------------

    def _on_key(self, key: Key) -> None:
        phase = self._phase
        if key is Key.QUIT:
            self._quit("operator quit key")
        elif key is Key.STOP:
            self._on_stop_key()
        elif key is Key.START:
            if phase is Phase.IDLE:
                if self._worker.busy():
                    self._hint("c: waiting for the last policy step to finish")
                    return
                prompt = (self._s.task_name, self._s.prompt_index) if self._uses_prompt else None
                self._worker.submit_reset(prompt)
                self._steps_sent = 0
                self._phase = Phase.RESETTING
                self._log.log_event("reset", "policy reset" + (" + prompt" if prompt else ""))
            elif phase is Phase.RUNNING:
                self._hint("c: an episode is running (y success, n fail, s stop)")
            else:
                self._hint("c: the episode is already starting (s cancels)")
        elif key in (Key.SUCCESS, Key.FAIL):
            if phase is Phase.RUNNING:
                self._end_episode(Outcome.SUCCESS if key is Key.SUCCESS else Outcome.FAIL, "operator")
            elif phase is Phase.RESETTING:
                self._hint(f"{key.value}: the episode is still starting (s cancels)")
            else:
                self._hint(f"{key.value}: no episode running (c start, h home, q quit)")
        elif key is Key.HOME:
            if phase is not Phase.IDLE:
                self._hint("h: end the episode first (y / n / s)")
            elif self._worker.busy():
                self._hint("h: waiting for the last policy step to finish")
            else:
                self._home()

    def _on_stop_key(self) -> None:
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
        self._operator.show_message(text)

    # ---- episodes ------------------------------------------------------------------------------------------------

    def _start_episode(self) -> None:
        """The reset (+ prompt) answered: franka start, start pose, phase RUNNING, first step (section 5.3)."""
        ack = self._rpc("franka_control", "c3.start/v1", self._franka.start)
        if not ack.ok:
            self._phase = Phase.IDLE
            self._log.log_event("start_refused", ack.reason)
            self._operator.show_message(f"start refused by franka_control: {ack.reason}")
            return
        self._stop_sent = False
        self._start_ack_ns = self._clock()
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
        self._log.log_event("episode_start", f"episode {self._episode_id}")
        self._operator.show_message(f"episode {self._episode_id} started")
        self._submit_step()

    def _end_episode(self, outcome: Outcome, reason: str) -> None:
        """y / n / s in RUNNING (section 5.6)."""
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
        if value.job.episode_id != self._episode_id or self._phase is not Phase.RUNNING:      # 9
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
            episode_id=skip.job.episode_id, chunk_id=skip.job.chunk_id, t_anchor_ns=0, anchor_source="",
            anchor_dev_mm=0.0, image_age_ms=0.0, frames_ms=0.0, policy_ms=0.0,
            cycle_ms=(result.finished_ns - result.started_ns) / NS_PER_MS, sent=False,
            skip_reason=f"{skip.problem.code.value}: {skip.problem.detail}",
            accepted=0, skipped=0, shortened=0, rejected=0, margin_ms=None))

    def _send(self, done: StepDone) -> None:
        """Section 5.4 steps 10-11."""
        if done.anchor.source != self._s.pose_source:
            self._monitor.classify(Problem(C.ANCHOR_FALLBACK, "franka_control",
                                           f"target and measured {done.anchor_dev_m * 1000:.0f} mm apart: "
                                           "measured used for this step"))
        if done.chunk.width_clipped:
            self._monitor.classify(Problem(C.WIDTH_CLIPPED, "policy", f"{done.chunk.width_clipped} widths clipped"),
                                   count=int(done.chunk.width_clipped))
        t_send = self._clock()
        reply = self._rpc("franka_control", "c3.set_waypoints/v1", self._franka.set_waypoints, done.waypoints)
        margin_ms = None if self._prev_last_wp_ns is None else (self._prev_last_wp_ns - t_send) / NS_PER_MS
        if reply.rejected:
            issue = reply.rejected[0]
            self._conclude(Problem(C.GATE_REJECTED, "franka_control", f"{issue.index}: {issue.reason}"))
        n = len(done.waypoints.t_ns)
        if reply.skipped == n or (margin_ms is not None and margin_ms < 0):
            late = f"margin {margin_ms:+.0f} ms" if margin_ms is not None else "every waypoint already past"
            self._monitor.classify(Problem(C.POLICY_LATE, "policy", f"policy late: {late}"))
        timing = done.timing
        cycle_ms = (t_send - (self._prev_send_ns if self._prev_send_ns is not None else timing.t_begin_ns)) / NS_PER_MS
        self._steps_sent += 1
        self._last_anchor_t_ns = done.anchor.t_ns
        self._prev_last_wp_ns = int(done.waypoints.t_ns[-1])
        self._prev_send_ns = t_send
        self._skipping_since = None
        self._typical_step_ms = (timing.t_end_ns - timing.t_begin_ns) / NS_PER_MS
        policy_ms = (timing.t_policy_ns - timing.t_frames_ns) / NS_PER_MS
        self._last = {"cycle_ms": cycle_ms, "policy_ms": policy_ms, "image_age_ms": timing.image_age_ms,
                      "margin_ms": margin_ms, "anchor_dev_mm": done.anchor_dev_m * 1000}
        self._submit_step()                                                         # 11: continuous control
        self._log.log_step(StepRecord(
            episode_id=done.job.episode_id, chunk_id=done.job.chunk_id, t_anchor_ns=done.anchor.t_ns,
            anchor_source=done.anchor.source, anchor_dev_mm=done.anchor_dev_m * 1000,
            image_age_ms=timing.image_age_ms, frames_ms=(timing.t_frames_ns - timing.t_begin_ns) / NS_PER_MS,
            policy_ms=policy_ms, cycle_ms=cycle_ms, sent=True, skip_reason="", accepted=reply.accepted,
            skipped=reply.skipped, shortened=reply.shortened, rejected=len(reply.rejected), margin_ms=margin_ms))
        if self._s.save_tap:
            self._log.save_tap(done, reply)

    # ---- home, quit, deploy stop, conclude -----------------------------------------------------------------------

    def _home(self) -> None:
        """Section 5.7: heartbeats paused, Ctrl-C / SIGTERM cancel the wait."""
        self._phase = Phase.HOMING
        self._show_status(self._clock())
        self._log.log_event("home", "homing started")
        try:
            self._rpc("franka_control", "c3.home/v1", self._franka.home, self.interrupt_set)
        except RequestCancelled:
            self._log.log_event("home", "cancelled")
            if self._terminate.is_set():
                self._deploy_stop(send_stop=False)
            self._quit("Ctrl-C", send_stop=False)
        self._phase = Phase.IDLE
        self._log.log_event("home", "home pose reached")
        self._operator.show_message("home pose reached")
        self._rpc("franka_control", "c3.heartbeat/v1", self._franka.heartbeat)      # heartbeats resume at once
        self._show_status(self._clock())

    def _send_stop(self) -> None:
        self._rpc("franka_control", "c3.stop/v1", self._franka.stop)
        self._stop_sent = True

    def _quit(self, reason: str, send_stop: bool = True) -> None:
        """q or Ctrl-C: stop first, the open episode recorded as stopped, exit 0."""
        self._finish(Outcome.STOPPED, reason, Problem(C.OPERATOR_STOP, "operator", reason), send_stop)

    def _deploy_stop(self, send_stop: bool = True) -> None:
        """SIGTERM: stop first, the open episode recorded as aborted, exit 0."""
        self._finish(Outcome.ABORTED, DEPLOY_STOP_REASON, Problem(C.DEPLOY_STOP, "deploy", DEPLOY_STOP_REASON),
                     send_stop)

    def _finish(self, outcome: Outcome, reason: str, problem: Problem, send_stop: bool) -> None:
        if send_stop:
            self._send_stop()
        if self._phase is Phase.RUNNING:
            self._close_episode(outcome, reason)
        self._verdict = self._monitor.classify(problem)
        self._log.log_event("exit", reason)
        self._exit_code = 0
        self._shutdown()
        raise _Finished

    def _conclude(self, problem: Problem, skip_stop: bool = False) -> None:
        """Section 5.9: stop once, record aborted, stop everything, exit 1 (the FATAL line is __main__'s)."""
        verdict = self._monitor.classify(problem)
        if verdict.kind is not Kind.CONCLUDE:
            raise RuntimeError(f"conclude called for a {verdict.kind.value} problem: {problem}")
        self._verdict = verdict
        log.error("conclude: %s: %s", verdict.what_failed, verdict.what_to_check)
        if not skip_stop:
            try:
                self._franka.stop()
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
        self._heartbeats = False
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

    def _guard(self, fn: Callable, *args) -> None:
        """Any exception on the main thread is a conclude (section 4.8)."""
        try:
            fn(*args)
        except _Finished:
            pass
        except Exception as exc:
            log.exception("main thread: %s", exc)
            problem = self._monitor.from_exception(exc, *self._call)
            latest = self._history.latest()
            if latest is not None and latest.mode == "fault":
                # the arm faulted and franka_control then refused a request: report the cause, as cycle step 2 does
                problem = Problem(C.ARM_FAULT, "franka_control", latest.error or "no error text")
            try:
                self._conclude(problem, skip_stop=self._call == STOP_CALL)
            except _Finished:
                pass

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
            arm_mode=latest.mode if latest is not None else None, anchor_dev_mm=self._last["anchor_dev_mm"])
