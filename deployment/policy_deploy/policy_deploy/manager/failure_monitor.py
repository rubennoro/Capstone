# adapted from rnoro/deployment:deployment/message_comms/message_comms/failure_monitor.py @ 91b178b
# changes: the codes and texts of STRUCTURE.md 8.5; check_state takes the PC "now" (mapped) and the newest C2 arrival
# time: STATE_LOST by laptop arrival (liveness), PC_STOPPED compares C2 t_ns with the start ack mapped to PC time;
# from_exception: a C3 TransportTimeout (not home) is PC_NOT_ANSWERING, a RemoteError from the PC with a field is
# CONTRACT_VIOLATION and without one PC_REFUSED, PcRestarted is PC_RESTARTED.
"""FailureMonitor: every problem -> normal, slow or conclude, in one place (STRUCTURE.md 8.5; PLAN 7).

A new failure rule is one entry in RULES.
"""
from __future__ import annotations

from collections.abc import Mapping

from deploy_contracts import ArmState, ContractError, RemoteError, TransportTimeout

from policy_deploy.contracts import HealthReport
from policy_deploy.manager.errors import PcRestarted, Refused, ScriptError
from policy_deploy.manager.problems import Kind, Problem, ProblemCode, Verdict

C = ProblemCode
SELF = "manager"
PC = "PC"
NORMAL_CHECK = "nothing to do (normal operation)"
PC_LINK_CHECK = "check franka_control on the PC and the link"
STATE_CHECK = "check franka_control's state broadcast and the link"
RESTARTED = "franka_control restarted on the PC"
REBOOTED = "the PC rebooted (its clock went back)"

# code -> (kind, what failed, what to check). Templates use {process}, {detail} and {origin} ("" for this process,
# "<process>: " for another).
RULES: dict[ProblemCode, tuple[Kind, str, str]] = {
    C.CAMERA_STALE: (Kind.SLOW, "{detail}", "check the Pi stream and ASUS_5g"),
    C.NO_NEW_FRAME: (Kind.NORMAL, "no new main frame since the last step", "nothing to do: the next cycle tries again"),
    C.IMAGE_OLD: (Kind.SLOW, "{detail}", "check frames (the Pi stream and its clock map)"),
    C.CLOCK_STALE: (Kind.SLOW, "{detail}", "the PC clock map has no fresh sample: check the link (pc_probe)"),
    C.STATE_GAP: (Kind.SLOW, "{detail}", STATE_CHECK),
    C.STATE_OLD: (Kind.SLOW, "{detail}", STATE_CHECK),
    C.ODD_CHUNK: (Kind.SLOW, "{detail}", "check the policy: step 0 should start at the anchor"),
    C.POLICY_LATE: (Kind.SLOW, "{detail}", "the arm held; raise manager.exec_h if it happens often (PLAN 13)"),
    C.SHORTENED: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.PULLED_BACK: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.WIDTH_CLIPPED: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.LOOP_LATE: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.ANCHOR_FALLBACK: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.CAMERA_DOWN: (Kind.CONCLUDE, "{detail}", "check the Pi and ASUS_5g"),
    C.POLICY_HUNG: (Kind.CONCLUDE, "policy: {detail}", "check the policy log and the GPU"),
    C.POLICY_ERROR: (Kind.CONCLUDE, "policy: {detail}", "check the policy log"),
    C.INVALID_CHUNK: (Kind.CONCLUDE, "policy: invalid chunk: {detail}", "check the checkpoint and policy.kind"),
    C.NO_ANSWER: (Kind.CONCLUDE, "{process}: {detail}", "check that {process} is running (see its log)"),
    C.PC_NOT_ANSWERING: (Kind.CONCLUDE, "PC not answering: {detail}", PC_LINK_CHECK),
    C.STATE_LOST: (Kind.CONCLUDE, "PC not answering: {detail}", PC_LINK_CHECK),
    C.PC_RESTARTED: (Kind.CONCLUDE, "{detail}", "restart policy_deploy run"),
    C.ARM_FAULT: (Kind.CONCLUDE, "the PC reports fault: {detail}",
                  "the PC holds; Ctrl-C and restart control_deploy run on the PC (release the user stop if it was a "
                  "reflex)"),
    C.GATE_REJECTED: (Kind.CONCLUDE, "the PC's gate rejected waypoint {detail}",
                      "check the workspace box and the policy"),
    C.PC_STOPPED: (Kind.CONCLUDE, "the PC stopped the arm (watchdog or the PC's `s`)", "check the link and the PC"),
    C.PC_REFUSED: (Kind.CONCLUDE, "the PC refused {detail}", "check the PC terminal"),
    C.CLOCK_MISMATCH: (Kind.CONCLUDE, "clock mismatch: {detail}", "run pc_probe; check clock.estimator"),
    C.HELLO_MISMATCH: (Kind.CONCLUDE, "{detail}", "check that both machines run the same commit and contract files"),
    C.PC_BUSY: (Kind.CONCLUDE, "the PC is driving an episode for another C3 client",
                "stop the other laptop session, pc_probe or console"),
    C.TCP_UNSETTLED: (Kind.CONCLUDE,
                      "trained policy refused on the real arm: contracts/frames.yaml settled is false (PLAN §9)",
                      "use policy.kind hold_still, or settle the TCP"),
    C.CONTRACT_VIOLATION: (Kind.CONCLUDE, "{detail}", "check that both machines run the same commit"),
    C.HOMING_FAILED: (Kind.CONCLUDE, "the PC's homing failed: {detail}", "check the arm's path to the home pose"),
    C.HOME_STOPPED: (Kind.NORMAL, "home stopped on the PC: {detail}", "the arm holds where it stopped"),
    C.STARTUP: (Kind.CONCLUDE, "startup: {detail}", "check {process}"),
    C.OPERATOR_SCRIPT: (Kind.CONCLUDE, "operator script: {detail}",
                        "check the script (manager.operator_script) or raise operator_script_wait_s"),
    C.INTERNAL_ERROR: (Kind.CONCLUDE, "{origin}internal error: {detail}", "see the manager log"),
    C.EPISODE_STOP: (Kind.NORMAL, "{detail}", "nothing to do: next episode: the PC's h, then p, then c here"),
    C.OPERATOR_STOP: (Kind.STOP, "{detail}", "nothing to check"),
    C.LAUNCHER_STOP: (Kind.STOP, "stopped by the launcher: another process exited", "see the launcher message"),
}

# the slow problems a skipped step can report (counted as "skips")
SKIP_CODES = (C.CAMERA_STALE, C.IMAGE_OLD, C.CLOCK_STALE, C.STATE_GAP, C.STATE_OLD, C.ODD_CHUNK)


class FailureMonitor:
    def __init__(self, state_lost_s: float) -> None:
        self._state_lost_s = state_lost_s
        self._counts: dict[ProblemCode, int] = {}

    def classify(self, problem: Problem, count: int = 1) -> Verdict:
        """The verdict, from RULES; counts it too (count > 1 for sums, e.g. a chunk's width_clipped)."""
        kind, failed, check = RULES[problem.code]
        values = {"process": problem.process, "detail": problem.detail,
                  "origin": "" if problem.process == SELF else f"{problem.process}: "}
        self._counts[problem.code] = self._counts.get(problem.code, 0) + count
        return Verdict(kind, problem, failed.format(**values), problem.check or check.format(**values))

    def from_exception(self, exc: BaseException, process: str, request: str) -> Problem:
        if isinstance(exc, PcRestarted):
            if exc.rebooted:
                return Problem(C.PC_RESTARTED, PC, REBOOTED,
                               check="start control_deploy run on the PC, then policy_deploy run")
            return Problem(C.PC_RESTARTED, PC, RESTARTED)
        if isinstance(exc, TransportTimeout):
            request = exc.request or request
            within = f"within {exc.timeout_s:.1f} s"
            if request == "c4.infer/v1":
                return Problem(C.POLICY_HUNG, "policy", f"no reply to infer {within}")
            if request == "c3.home/v1":
                return Problem(C.HOMING_FAILED, PC, f"no reply {within}")
            if request.startswith("c3."):
                return Problem(C.PC_NOT_ANSWERING, PC, f"no reply to {request} {within}")
            return Problem(C.NO_ANSWER, exc.process or process, f"no reply to {request} {within}")
        if isinstance(exc, RemoteError):
            reply = exc.reply
            req = reply.request or request
            if exc.process == PC and req == "c3.home/v1" and not reply.field:
                return Problem(C.HOMING_FAILED, PC, reply.error)
            if reply.field:
                return Problem(C.CONTRACT_VIOLATION, exc.process, f"{exc.process} refused {req}: {reply.error}")
            if exc.process == PC:
                return Problem(C.PC_REFUSED, PC, f"{req}: {reply.error}")
            if exc.process == "policy":
                return Problem(C.POLICY_ERROR, "policy", reply.error)
            return Problem(C.INTERNAL_ERROR, exc.process, reply.error, check="see its log")
        if isinstance(exc, Refused):
            if exc.process == "policy":
                return Problem(C.POLICY_ERROR, "policy", f"{exc.request} refused: {exc.reason}")
            if exc.request == "c3.home/v1":
                return Problem(C.HOMING_FAILED, exc.process, exc.reason)
            if exc.process == PC:
                return Problem(C.PC_REFUSED, PC, f"{exc.request}: {exc.reason}")
            return Problem(C.INTERNAL_ERROR, exc.process, f"{exc.request} refused: {exc.reason}",
                           check="see its log")
        if isinstance(exc, ContractError):
            if exc.schema == "c4.chunk/v1":
                return Problem(C.INVALID_CHUNK, "policy", f"{exc.field}: {exc.detail}" if exc.field else exc.detail)
            return Problem(C.CONTRACT_VIOLATION, process, str(exc))
        if isinstance(exc, ScriptError):                     # ScriptTimeout, from ScriptedOperator.poll_key
            return Problem(C.OPERATOR_SCRIPT, "operator", str(exc))
        return Problem(C.INTERNAL_ERROR, SELF, f"{type(exc).__name__}: {exc}")

    def check_state(self, latest: ArmState | None, now_ns: int, last_arrival_ns: int | None, episode_running: bool,
                    stop_sent: bool, start_ack_pc_ns: int | None) -> Problem | None:
        """fault (any phase); stopped while running without our stop; no C2 for state_lost_s.

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
        return None                              # homing (the PC's h, g or p): followed, c refused

    def check_health(self, health: HealthReport) -> Problem | None:
        """Only "down" is a problem here (stale is seen by the steps)."""
        for name in ("main", "ultrawide"):
            if getattr(health, name).status == "down":
                return Problem(C.CAMERA_DOWN, "frames", f"{name} camera down")
        return None

    def counts(self) -> Mapping[ProblemCode, int]:
        return dict(self._counts)

    def reset_episode_counts(self) -> None:
        self._counts.clear()
