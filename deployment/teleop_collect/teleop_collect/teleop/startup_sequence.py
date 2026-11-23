# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/startup_sequence.py @ 2ffe641
# changes: teleop_collect PLAN 8 / STRUCTURE.md 6.4: + step 1 TaskStore.clean_incomplete (now
# TaskStore.unsaved: kept, 2026-11-20); no policy config; cameras
# from R1 FramesHealth (step 8); + step 9 data_root writable; step 11 session.yaml, phase IDLE, the ready file; every
# wait also serves U1 (status and keys; the rest refused while starting) and follows the state; TeleopLoop instead of
# EpisodeLoop; no operator. + start pose (final-deployment's start-pose work, 927a5ba, 2026-11-20): no
# start-up home (home.on_start, its C3 home and the re-warm after it are gone); + step 10: the start pose is the arm's
# pose now (StartPoseReference.learn from the newest C2; h, then p on the PC come first). + wave 3 (no software
# stops, 2026-11-20): step 11 logs and shows the PC margins teleop assumes (motion.assumed_pc_*) and the risk.
"""StartupSequence: teleop's start-up order, phase STARTING (STRUCTURE.md 6.4; PLAN 8).

| #  | step                                                                       | done when                |
| 1  | TaskStore.unsaved: kept, each named on the page (never deleted)            |                          |
| 2  | C2 arriving                                                                | a valid C2 message       |
| 3  | H2 wait: nothing on C3 while C2 shows running/holding (2 x watchdog), or   | mode idle or stopped     |
|    | homing (timeouts.pc_homing_wait_s); fault refuses                          |                          |
| 4  | hellos start: a hello every cycle until the clock map is ready             | clock ready              |
| 5  | compare_hello + the laptop checks (StartupChecks)                          | no failure               |
| 6  | get_state: idle/stopped go on; homing waits (homing_timeout_s + 5 s);      | idle or stopped          |
|    | running/holding = another client; fault refuses                            |                          |
| 7  | heartbeats every cycle (already since step 4)                              |                          |
| 8  | cameras: R1 get_health until both ok                                       | both ok                  |
| 9  | data_root writable                                                         |                          |
| 10 | the start pose: StartPoseReference.learn(q, O_T_TCP_target) of the newest  |                          |
|    | C2 when it shows idle or stopped (else the PC's next h/p gives it)         |                          |
| 11 | session.yaml, the ready file, phase IDLE; the assumed PC margins shown      |                          |

Steps 2-9 must finish within timeouts.startup_s, not counting the waits while C2 shows homing (steps 3 and 6). No step
moves the arm: homing (h) and the start pose (p) are the PC's keyboard keys, pressed before `teleop_collect run`.
Every cycle checks the signal flags and the state rules (fault, seq going back, no C2 for state_lost_s).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from teleop_collect.contracts import CAMERAS
from teleop_collect.contracts.station import STORAGE_FILE, storage_file
from teleop_collect.teleop.episode.store import probe_writable
from teleop_collect.teleop.motion.planner import assumed_margins_text
from teleop_collect.teleop.problems import Problem, ProblemCode

if TYPE_CHECKING:
    from teleop_collect.teleop.teleop_loop import TeleopLoop

C = ProblemCode
NS_PER_S = 1_000_000_000
PC = "PC"
HOMING_EXTRA_S = 5.0          # step 6: homing_timeout_s + this
GET_STATE_PERIOD_S = 0.5      # step 6: ask again this often while homing


class StartupSequence:
    def __init__(self, loop: TeleopLoop) -> None:
        self.loop = loop
        self.s = loop._s
        self._deadline_ns = 0
        self._why = ""

    # ---- the steps

    def run(self, ready_file: Path | Callable[[], None] | None) -> None:
        """ready_file: ProcessArgs.write_ready (the process), or a path (tests), or None."""
        from teleop_collect.teleop.teleop_loop import Phase
        lp, s = self.loop, self.s
        lp._log.log_event("startup", "starting")
        lp.flow.report_unsaved()                                      # 1
        lp._listener.start()
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
            lp._messages.add("warn", f"warning: {warning}")
        self._mode_check(reply.homing_timeout_s)                      # 6, 7
        self._wait(self._cameras_ready, "frames", "no camera ok", lambda now: self._why)   # 8
        self._check_data_root()                                       # 9
        self._learn_start_pose()                                      # 10
        now = lp._clock()                                             # 11
        lp._log.write_session(s, reply, lp._checks.commit, lp.clock_row(now))
        lp._phase = Phase.IDLE
        lp._ready = True
        if callable(ready_file):
            ready_file()                                              # ProcessArgs.write_ready
        elif ready_file is not None:
            Path(ready_file).parent.mkdir(parents=True, exist_ok=True)
            Path(ready_file).write_text("ready\n", encoding="utf-8")
        lp._log.log_event("ready", f"C2 arriving, hello checked (PC {reply.commit}, mock {reply.mock}), clock map "
                                   f"{lp._pc_clock.NAME} ready, both cameras ok, data_root writable")
        margins = assumed_margins_text(s.motion)
        lp._log.log_event("margins", margins)
        lp._messages.add("info", margins)
        lp._messages.add("info", "teleop ready")

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
            lp._serve_u1()
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
            lp._observe_state(now)
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
            lp._serve_u1()
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
            lp._observe_state(now)
            if now >= next_ask:
                next_ask = now + int(GET_STATE_PERIOD_S * NS_PER_S)
                state = lp._rpc(PC, "c3.get_state/v1", lp._pc.get_state)
                lp._note_request("c3.get_state/v1")
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
            lp._serve_u1()
            lp._sleep(lp._loop_ns / NS_PER_S)

    # ---- the generic wait (steps 4, 8)

    def _wait(self, ready: Callable[[int], bool], process: str, what: str, why: Callable[[int], str]) -> None:
        lp, s = self.loop, self.s
        while True:
            now = lp._clock()
            lp._check_flags()
            lp._check_state(now)
            lp._observe_state(now)
            if lp._beating:
                lp._beat(now)
            if ready(lp._clock()):
                return
            if now >= self._deadline_ns:
                lp._conclude(Problem(C.STARTUP, process, f"{what} within {s.timeouts.startup_s:g} s ({why(now)})"))
            lp._serve_u1()
            lp._sleep(lp._loop_ns / NS_PER_S)

    def _cameras_ready(self, now: int) -> bool:
        lp = self.loop
        if now < lp._next_health_ns:
            return False
        report = lp._poll_health(now)
        if report is None:
            self._why = "frames did not answer get_health"
            return False
        bad = [f"{name} {getattr(report.health, name).status}" for name in CAMERAS
               if getattr(report.health, name).status != "ok"]
        self._why = ", ".join(bad)
        return not bad

    # ---- step 9

    def _check_data_root(self) -> None:
        lp = self.loop
        root = Path(self.s.data_root)
        try:
            probe_writable(root)
        except OSError as exc:
            if self.s.data_root_source == STORAGE_FILE:
                where = (f"{storage_file(self.s.station_file)} (the folder chosen in the page or with --data-root: "
                         "run with --data-root PATH, or delete that file)")
            else:
                where = "station.yaml data_root (or teleop_collect run --data-root PATH)"
            lp._conclude(Problem(C.STARTUP, "teleop", f"data_root {root} is not writable: {exc}",
                                 check=f"check {where} and the disk"))

    # ---- step 10

    def _learn_start_pose(self) -> None:
        """The start pose is the arm's pose now: the operator pressed h, then p on the PC before `teleop_collect run`.
        The PC checks it at the first Start (StartPoseReference.confirm / refused_here). While the PC is moving the arm
        (homing: its h, g or p) nothing is learned: the pose after that move is the start pose; this C2 is passed to
        StartPoseReference.observe so the loop's observe sees the end of that move (homing -> idle)."""
        lp = self.loop
        latest = lp._history.latest()
        if latest is None or latest.mode not in ("idle", "stopped"):
            if latest is not None:
                lp.start_pose.observe(latest.mode, latest.q, latest.O_T_TCP_target)
            mode = latest.mode if latest is not None else "no C2"
            text = f"start pose not learned: the PC is {mode}; the pose after its move is the start pose"
            lp._log.log_event("start_pose", text)
            lp._messages.add("warn", text)
            return
        lp.start_pose.learn(latest.q, latest.O_T_TCP_target)
        xyz = ", ".join(f"{float(v):.3f}" for v in latest.O_T_TCP_target[:3, 3])
        text = f"start pose: the arm's pose now ({xyz} m); every episode starts here (h, then p on the PC)"
        lp._log.log_event("start_pose", text)
        lp._messages.add("info", text)
