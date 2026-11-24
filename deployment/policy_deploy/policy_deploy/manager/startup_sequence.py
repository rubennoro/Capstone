"""StartupSequence: the manager's start-up order, phase STARTING (STRUCTURE.md 6.3; PLAN 8.2; TOP 8.2). Replaces
rnoro/deployment:deployment/message_comms/message_comms/episode_loop.py:166-247.

| # | step                                                                       | done when                |
| 2 | C2 arriving                                                                | a valid C2 message       |
| 3 | H2 wait: nothing on C3 while C2 shows running/holding (2 x watchdog), or   | mode idle or stopped     |
|   | homing (timeouts.pc_homing_wait_s); fault refuses                          |                          |
| 4 | hellos start: a hello every cycle until the clock map is ready             | clock ready              |
| 5 | compare_hello + the laptop checks (StartupChecks)                          | no failure               |
| 6 | get_state: idle/stopped go on; homing waits (homing_timeout_s + 5 s);      | idle or stopped          |
|   | running/holding = another client; fault refuses                            |                          |
| 7 | heartbeats every cycle (already since step 4)                              |                          |
| 8 | policy config (C4 v2): sizes, exec_h, prompt, the settled rule             | reply checked            |
| 9 | cameras: get_health until both ok                                          | both ok                  |
| 10| session.yaml, operator start, key list, ready file, phase IDLE             |                          |

Steps 2-9 must finish within timeouts.startup_s, not counting the time spent waiting while C2 shows homing (steps 3
and 6). Every cycle checks the signal flags and the state rules (fault, seq going back, no C2 for state_lost_s).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from policy_deploy.contracts import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS
from policy_deploy.manager.problems import Problem, ProblemCode

if TYPE_CHECKING:
    from policy_deploy.manager.episode_loop import EpisodeLoop

C = ProblemCode
NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
PC = "PC"
HOMING_EXTRA_S = 5.0          # step 6: homing_timeout_s + this
GET_STATE_PERIOD_S = 0.5      # step 6: ask again this often while homing


class StartupSequence:
    def __init__(self, loop: EpisodeLoop) -> None:
        self.loop = loop
        self.s = loop._s
        self._deadline_ns = 0
        self._why = ""

    # ---- the steps

    def run(self, ready_file: Path | None) -> None:
        lp, s = self.loop, self.s
        lp._log.log_event("startup", "starting")
        lp._listener.start()                                          # 1
        lp._worker.start()
        self._deadline_ns = lp._clock() + int(s.timeouts.startup_s * NS_PER_S)

        first_c2 = self._wait_c2()                                    # 2
        self._h2_wait(first_c2)                                       # 3
        lp._beating = True                                            # 4
        lp._heartbeats.startup_mode(True)
        lp._log.log_event("startup", "C2 arriving, PC free: hellos start")
        self._wait(lambda now: lp._pc_clock.ready(now), PC, "PC clock map not ready",
                   lambda now: lp._pc_clock.not_ready_reason(now))
        lp._heartbeats.startup_mode(False)
        reply = lp._heartbeats.last_hello()
        lp._hello = reply
        failures = lp._checks.hello_failures(reply, lp._pc_clock, lp._pc_clock.hellos())   # 5
        for f in failures:
            lp._log.log_event("hello_check", str(f))
        if failures:
            f = failures[0]
            lp._conclude(Problem(C.HELLO_MISMATCH, PC, f.what, check=f.check))
        warning = lp._checks.commit_warning(reply)
        if warning:
            lp._log.log_event("commit", f"warning: {warning}")
            lp._operator.show_message(f"warning: {warning}")
        self._mode_check(reply.homing_timeout_s)                      # 6, 7
        lp._worker.submit_config()                                    # 8
        self._wait(self._config_ready, "policy", "no policy config", lambda now: "waiting for the reply")
        self._check_config(reply)
        self._wait(self._cameras_ready, "frames", "no camera ok", lambda now: self._why)   # 9
        now = lp._clock()                                             # 10
        lp._log.write_session(s, lp._policy_config, reply, lp._checks.commit, lp.clock_row(now))
        lp._operator.start()
        from policy_deploy.manager.episode_loop import Phase
        lp._phase = Phase.IDLE
        lp._show_status(lp._clock())
        if ready_file is not None:
            Path(ready_file).parent.mkdir(parents=True, exist_ok=True)
            Path(ready_file).write_text("ready\n")
        lp._log.log_event("ready", f"C2 arriving, hello checked (PC {reply.commit}, mock {reply.mock}), clock map "
                                   f"{lp._pc_clock.NAME} ready, policy config accepted, both cameras ok")

    # ---- step 2

    def _wait_c2(self) -> int:
        lp, s = self.loop, self.s
        while True:
            now = lp._clock()
            lp._check_flags()
            error = lp._listener.error()
            if error is not None:
                lp._conclude(lp._monitor.from_exception(error, PC, "c2.state/v1"))
            if lp._history.latest() is not None:
                return now
            if now >= self._deadline_ns:
                lp._conclude(Problem(C.PC_NOT_ANSWERING, PC,
                                     f"no C2 state from {s.state_endpoint} within {s.timeouts.startup_s:g} s",
                                     check="check that control_deploy run is running on the PC, ASUS_5g, and "
                                           "link.yaml pc.address"))
            lp._sleep(lp._loop_ns / NS_PER_S)

    # ---- step 3 (rule H2)

    def _h2_wait(self, first_c2_ns: int) -> None:
        lp, s = self.loop, self.s
        busy_limit_ns = first_c2_ns + int(2 * s.watchdog_limit_s * NS_PER_S)
        homing_since: int | None = None
        logged = ""
        while True:
            now = lp._clock()
            lp._check_flags()
            lp._check_state(now)                      # fault, seq going back, no C2 for state_lost_s
            mode = lp._history.latest().mode
            if mode != logged:
                lp._log.log_event("startup", f"C2 mode {mode}")
                logged = mode
            if mode in ("idle", "stopped"):
                if homing_since is not None:
                    self._deadline_ns += now - homing_since
                return
            if mode == "homing":
                if homing_since is None:
                    homing_since = now
                if now - homing_since > s.timeouts.pc_homing_wait_s * NS_PER_S:
                    lp._conclude(Problem(C.STARTUP, PC, f"the PC is still homing after "
                                                        f"{s.timeouts.pc_homing_wait_s:g} s",
                                         check="check the PC terminal"))
            else:
                if homing_since is not None:
                    self._deadline_ns += now - homing_since
                    homing_since = None
                if mode in ("running", "holding") and now > busy_limit_ns:
                    lp._conclude(Problem(C.PC_BUSY, PC, f"C2 still shows {mode} after "
                                                        f"{2 * s.watchdog_limit_s:g} s without C3"))
            lp._sleep(lp._loop_ns / NS_PER_S)

    # ---- step 6

    def _mode_check(self, homing_timeout_s: float) -> None:
        lp = self.loop
        homing_since: int | None = None
        next_ask = 0
        while True:
            now = lp._clock()
            lp._check_flags()
            lp._check_state(now)
            if now >= next_ask:
                next_ask = now + int(GET_STATE_PERIOD_S * NS_PER_S)
                state = lp._rpc(PC, "c3.get_state/v1", lp._pc.get_state)
                mode = state.mode
                if mode in ("idle", "stopped"):
                    if homing_since is not None:
                        self._deadline_ns += now - homing_since
                    return
                if mode in ("running", "holding"):
                    lp._conclude(Problem(C.PC_BUSY, PC, f"get_state shows {mode}"))
                if mode == "fault":
                    lp._conclude(Problem(C.ARM_FAULT, PC, state.error or "no error text",
                                         check="Ctrl-C and restart control_deploy run on the PC"))
                if mode == "homing":
                    if homing_since is None:
                        homing_since = now
                        lp._log.log_event("startup", "the PC is homing: waiting")
                    if now - homing_since > (homing_timeout_s + HOMING_EXTRA_S) * NS_PER_S:
                        lp._conclude(Problem(C.STARTUP, PC, f"the PC is still homing after "
                                                            f"{homing_timeout_s + HOMING_EXTRA_S:g} s",
                                             check="check the PC terminal"))
            lp._beat(now)
            lp._sleep(lp._loop_ns / NS_PER_S)

    # ---- the generic wait (steps 4, 8, 9)

    def _wait(self, ready: Callable[[int], bool], process: str, what: str, why: Callable[[int], str]) -> None:
        lp, s = self.loop, self.s
        while True:
            now = lp._clock()
            lp._check_flags()
            lp._check_state(now)
            if lp._beating:
                lp._beat(now)
            if ready(lp._clock()):
                return
            if now >= self._deadline_ns:
                lp._conclude(Problem(C.STARTUP, process, f"{what} within {s.timeouts.startup_s:g} s ({why(now)})"))
            lp._sleep(lp._loop_ns / NS_PER_S)

    def _config_ready(self, now: int) -> bool:
        lp = self.loop
        result = lp._worker.wait(0.0)
        if result is None:
            return False
        if result.error is not None:
            lp._conclude(lp._monitor.from_exception(result.error, result.process, result.request))
        lp._policy_config = result.value
        return True

    def _check_config(self, hello) -> None:
        lp, s = self.loop, self.s
        cfg = lp._policy_config
        for name, expected in (("obs_steps", OBS_STEPS), ("action_steps", ACTION_STEPS), ("step_ms", STEP_MS),
                               ("image_size", IMAGE_SIZE)):
            value = getattr(cfg, name)
            if value != expected:
                lp._conclude(Problem(C.CONTRACT_VIOLATION, "policy",
                                     f"policy config: {name} is {value}, manager expects {expected}",
                                     check="check the checkpoint"))
        if s.exec_h > cfg.action_steps:
            lp._conclude(Problem(C.CONTRACT_VIOLATION, "policy",
                                 f"policy config: exec_h {s.exec_h} > action_steps {cfg.action_steps}",
                                 check="check manager.exec_h"))
        lp._uses_prompt = bool(cfg.uses_prompt)
        if lp._uses_prompt and s.task_name is None:
            lp._conclude(Problem(C.STARTUP, "manager", "the policy uses a prompt but no task_name is set",
                                 check="check manager.task_name"))
        unsettled = lp._checks.settled(cfg.kind, bool(hello.mock))
        if unsettled is not None:
            lp._conclude(Problem(C.TCP_UNSETTLED, "policy", unsettled.what))

    def _cameras_ready(self, now: int) -> bool:
        lp = self.loop
        if now < lp._next_health_ns:
            return False
        report = lp._poll_health(now)
        bad = [f"{name} {getattr(report, name).status}" for name in ("main", "ultrawide")
               if getattr(report, name).status != "ok"]
        self._why = ", ".join(bad)
        return not bad
