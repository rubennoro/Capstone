# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/failure_monitor.py @ 2ffe641
# changes: the codes and kinds of teleop_collect STRUCTURE.md 6.5 (CAMERA_DOWN, PC_STOPPED, OPERATOR_ABORT and
# RECORDING_FAILED are EPISODE; SET_SKIPPED and CAMERA_STALE are SLOW; the policy codes are gone); classify() turns an
# EPISODE problem outside an episode into NORMAL (nothing to abort); check_health reads R1's FramesHealth and the
# recording flag (recording_error, a camera down, a camera stale); from_exception: no policy branches, ScriptError
# gone, the restart hint names teleop_collect run; process name teleop; SKIP_CODES -> SLOW_CODES (every SLOW code,
# SET_SKIPPED included); hint texts: GATE_REJECTED checks teleop.motion, HELLO_MISMATCH and CONTRACT_VIOLATION say
# "the same contract files", INTERNAL_ERROR "the teleop log", LAUNCHER_STOP "(Ctrl-C, or another process exited)"
# (the launcher sends SIGTERM for both). + start pose (final-deployment's start-pose work, 927a5ba,
# 2026-11-20): no laptop home: HOME_STOPPED, HOMING_FAILED and the c3.home branches of from_exception dropped; homing
# (the PC's h, g or p) is followed, Start refused. + wave 3 (no software stops, 2026-11-20): CAMERA_DOWN,
# CAMERA_STALE and RECORDING_FAILED are WARN (shown, in meta.yaml warnings; no stop, no abort, no pause; WARN outside
# an episode is NORMAL); STATE_GAP is NORMAL (counted, logged; motion goes on); + WARN_CODES, COUNTED_CODES.
# + keep episodes (2026-11-20): an EPISODE problem keeps the episode unsaved (EpisodeFlow.abort). The source's own
# header follows.
# adapted from rnoro/deployment:deployment/message_comms/message_comms/failure_monitor.py @ 91b178b
# changes: the codes and texts of STRUCTURE.md 8.5; check_state takes the PC "now" (mapped) and the newest C2 arrival
# time: STATE_LOST by laptop arrival (liveness), PC_STOPPED compares C2 t_ns with the start ack mapped to PC time;
# from_exception: a C3 TransportTimeout (not home) is PC_NOT_ANSWERING, a RemoteError from the PC with a field is
# CONTRACT_VIOLATION and without one PC_REFUSED, PcRestarted is PC_RESTARTED.
"""FailureMonitor: every problem -> normal, slow, warn, episode or conclude, in one place (STRUCTURE.md 6.5; PLAN 7).

A new failure rule is one entry in RULES.
"""
from __future__ import annotations

from collections.abc import Mapping

from deploy_contracts import ArmState, ContractError, RemoteError, TransportTimeout

from teleop_collect.contracts import CAMERAS, FramesHealth
from teleop_collect.teleop.errors import PcRestarted, Refused
from teleop_collect.teleop.problems import Kind, Problem, ProblemCode, Verdict

C = ProblemCode
SELF = "teleop"
PC = "PC"
NORMAL_CHECK = "nothing to do (normal operation)"
PC_LINK_CHECK = "check franka_control on the PC and the link"
STATE_CHECK = "check franka_control's state broadcast and the link"
CAMERA_CHECK = "check the Pi and ASUS_5g"
RESTARTED = "franka_control restarted on the PC"
REBOOTED = "the PC rebooted (its clock went back)"

# code -> (kind, what failed, what to check). Templates use {process}, {detail} and {origin} ("" for this process,
# "<process>: " for another).
RULES: dict[ProblemCode, tuple[Kind, str, str]] = {
    C.CAMERA_STALE: (Kind.WARN, "{detail}", "check the Pi stream and ASUS_5g"),
    C.CLOCK_STALE: (Kind.SLOW, "{detail}", "the PC clock map has no fresh sample: check the link (pc_probe)"),
    C.STATE_GAP: (Kind.NORMAL, "{detail}", STATE_CHECK),
    C.STATE_OLD: (Kind.SLOW, "{detail}", STATE_CHECK),
    C.SET_SKIPPED: (Kind.SLOW, "{detail}", "a Wi-Fi spike: the next cycle sends a new set"),
    C.SHORTENED: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.PULLED_BACK: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.LOOP_LATE: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.CAMERA_DOWN: (Kind.WARN, "{detail}", CAMERA_CHECK),
    C.PC_STOPPED: (Kind.EPISODE, "the PC stopped the arm (watchdog or the PC's `s`)", "check the link and the PC"),
    C.OPERATOR_ABORT: (Kind.EPISODE, "operator stop", "nothing to check"),
    C.RECORDING_FAILED: (Kind.WARN, "recording could not keep up / disk error: {detail}",
                         "check the free disk space under data_root and the frames log"),
    C.NO_ANSWER: (Kind.CONCLUDE, "{process}: {detail}", "check that {process} is running (see its log)"),
    C.PC_NOT_ANSWERING: (Kind.CONCLUDE, "PC not answering: {detail}", PC_LINK_CHECK),
    C.STATE_LOST: (Kind.CONCLUDE, "PC not answering: {detail}", PC_LINK_CHECK),
    C.PC_RESTARTED: (Kind.CONCLUDE, "{detail}", "restart teleop_collect run"),
    C.ARM_FAULT: (Kind.CONCLUDE, "the PC reports fault: {detail}",
                  "the PC holds; Ctrl-C and restart control_deploy run on the PC (release the user stop if it was a "
                  "reflex)"),
    C.GATE_REJECTED: (Kind.CONCLUDE, "the PC's gate rejected waypoint {detail}",
                      "check the workspace box and teleop.motion"),
    C.PC_REFUSED: (Kind.CONCLUDE, "the PC refused {detail}", "check the PC terminal"),
    C.CLOCK_MISMATCH: (Kind.CONCLUDE, "clock mismatch: {detail}", "run pc_probe; check clock.estimator"),
    C.HELLO_MISMATCH: (Kind.CONCLUDE, "{detail}", "check that both machines run the same contract files"),
    C.PC_BUSY: (Kind.CONCLUDE, "the PC is driving an episode for another C3 client",
                "stop the other laptop session, pc_probe or console"),
    C.CONTRACT_VIOLATION: (Kind.CONCLUDE, "{detail}", "check that both machines run the same contract files"),
    C.STARTUP: (Kind.CONCLUDE, "startup: {detail}", "check {process}"),
    C.INTERNAL_ERROR: (Kind.CONCLUDE, "{origin}internal error: {detail}", "see the teleop log"),
    C.OPERATOR_STOP: (Kind.STOP, "{detail}", "nothing to check"),
    C.LAUNCHER_STOP: (Kind.STOP, "stopped by the launcher (Ctrl-C, or another process exited)",
                      "see the launcher message"),
}

# the slow problems that pause motion for a cycle (counted, shown on the page)
SLOW_CODES = tuple(code for code, (kind, _, _) in RULES.items() if kind is Kind.SLOW)
# the data problems: never stop or pause the arm; each shown once per episode and listed in meta.yaml `warnings`
WARN_CODES = tuple(code for code, (kind, _, _) in RULES.items() if kind is Kind.WARN)
# the problem counts an episode's meta.yaml `counts` keeps (by code value)
COUNTED_CODES = SLOW_CODES + (C.STATE_GAP,) + WARN_CODES


class FailureMonitor:
    def __init__(self, state_lost_s: float) -> None:
        self._state_lost_s = state_lost_s
        self._counts: dict[ProblemCode, int] = {}

    def classify(self, problem: Problem, count: int = 1, episode_open: bool = True) -> Verdict:
        """The verdict, from RULES; counts it too. An EPISODE or WARN problem with no episode open is NORMAL
        (nothing to abort or note: a camera down between episodes only disables Start)."""
        kind, failed, check = RULES[problem.code]
        if kind in (Kind.EPISODE, Kind.WARN) and not episode_open:
            kind = Kind.NORMAL
        values = {"process": problem.process, "detail": problem.detail,
                  "origin": "" if problem.process == SELF else f"{problem.process}: "}
        self._counts[problem.code] = self._counts.get(problem.code, 0) + count
        return Verdict(kind, problem, failed.format(**values), problem.check or check.format(**values))

    def from_exception(self, exc: BaseException, process: str, request: str) -> Problem:
        if isinstance(exc, PcRestarted):
            if exc.rebooted:
                return Problem(C.PC_RESTARTED, PC, REBOOTED,
                               check="start control_deploy run on the PC, then teleop_collect run")
            return Problem(C.PC_RESTARTED, PC, RESTARTED)
        if isinstance(exc, TransportTimeout):
            request = exc.request or request
            within = f"within {exc.timeout_s:.1f} s"
            if request.startswith("c3."):
                return Problem(C.PC_NOT_ANSWERING, PC, f"no reply to {request} {within}")
            return Problem(C.NO_ANSWER, exc.process or process, f"no reply to {request} {within}")
        if isinstance(exc, RemoteError):
            reply = exc.reply
            req = reply.request or request
            if reply.field:
                return Problem(C.CONTRACT_VIOLATION, exc.process, f"{exc.process} refused {req}: {reply.error}")
            if exc.process == PC:
                return Problem(C.PC_REFUSED, PC, f"{req}: {reply.error}")
            return Problem(C.INTERNAL_ERROR, exc.process, reply.error, check="see its log")
        if isinstance(exc, Refused):
            if exc.process == PC:
                return Problem(C.PC_REFUSED, PC, f"{exc.request}: {exc.reason}")
            return Problem(C.INTERNAL_ERROR, exc.process, f"{exc.request} refused: {exc.reason}",
                           check="see its log")
        if isinstance(exc, ContractError):
            return Problem(C.CONTRACT_VIOLATION, process, str(exc))
        return Problem(C.INTERNAL_ERROR, SELF, f"{type(exc).__name__}: {exc}")

    def check_state(self, latest: ArmState | None, now_ns: int, last_arrival_ns: int | None, episode_running: bool,
                    stop_sent: bool, start_ack_pc_ns: int | None) -> Problem | None:
        """fault (any phase); stopped while recording without our stop; no C2 for state_lost_s.

        now_ns / last_arrival_ns: laptop clock (liveness by arrival: t_ns can repeat, TOP 3.6 obligation 1).
        start_ack_pc_ns: the start ack's laptop time mapped to the PC clock; C2 older than it is ignored.
        """
        if last_arrival_ns is not None and now_ns - last_arrival_ns > self._state_lost_s * 1e9:
            return Problem(C.STATE_LOST, PC, f"no C2 state for {self._state_lost_s:.1f} s")
        if latest is None:
            return None
        if latest.mode == "fault":
            return Problem(C.ARM_FAULT, PC, latest.error or "no error text")
        if (latest.mode == "stopped" and episode_running and not stop_sent
                and (start_ack_pc_ns is None or latest.t_ns >= start_ack_pc_ns)):
            return Problem(C.PC_STOPPED, PC, latest.error or "mode stopped")
        return None                              # homing (the PC's h, g or p): followed, Start refused

    def check_health(self, health: FramesHealth, recording: bool) -> Problem | None:
        """R1 health. While recording: the writer's error, a camera down, a camera stale (all WARN: the episode goes
        on). Between episodes a camera down is NORMAL (Start is refused with the reason); stale is not a problem."""
        if recording:
            if health.recording_error:
                return Problem(C.RECORDING_FAILED, "frames", health.recording_error)
            if not health.recording:
                return Problem(C.RECORDING_FAILED, "frames", "frames is not recording")
        for name in CAMERAS:
            if getattr(health.health, name).status == "down":
                return Problem(C.CAMERA_DOWN, "frames", f"{name} camera down")
        if recording:
            for name in CAMERAS:
                cam = getattr(health.health, name)
                if cam.status == "stale":
                    return Problem(C.CAMERA_STALE, "frames",
                                   f"{name} camera stale, last frame {cam.last_frame_age_ms:.0f} ms ago")
        return None

    def counts(self) -> Mapping[ProblemCode, int]:
        return dict(self._counts)

    def reset_episode_counts(self) -> None:
        self._counts.clear()
