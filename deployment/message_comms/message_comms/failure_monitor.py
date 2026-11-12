"""FailureMonitor: every problem -> slow or conclude, in one place (STRUCTURE.md 4.7, section 7; PLAN section 5).

A new failure rule is one entry in RULES.
"""
from __future__ import annotations

from collections.abc import Mapping

from contracts import ArmState, ContractError, HealthReport, RemoteError, TransportTimeout
from message_comms.errors import Refused
from message_comms.problems import Kind, Problem, ProblemCode, Verdict
from message_comms.settings import Settings

C = ProblemCode
SELF = "message_comms"
NORMAL_CHECK = "nothing to do (normal operation)"

# code -> (kind, what failed, what to check). Templates use {process}, {detail} and {origin} ("" for this process,
# "<process>: " for another).
RULES: dict[ProblemCode, tuple[Kind, str, str]] = {
    C.CAMERA_STALE: (Kind.SLOW, "{detail}", "check the Pi stream and Wi-Fi"),
    C.NO_NEW_FRAME: (Kind.NORMAL, "no new main frame since the last step", "nothing to do: the next cycle tries again"),
    C.IMAGE_OLD: (Kind.SLOW, "{detail}", "check process_frames (its clock sync with the Pi)"),
    C.POLICY_LATE: (Kind.SLOW, "{detail}",
                    "the arm held; raise message_comms.exec_h if it happens often (PLAN section 9)"),
    C.STATE_GAP: (Kind.SLOW, "{detail}", "check franka_control's state broadcast"),
    C.STATE_OLD: (Kind.SLOW, "{detail}", "check franka_control's state broadcast"),
    C.SHORTENED: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.PULLED_BACK: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.WIDTH_CLIPPED: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.LOOP_LATE: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.ODD_CHUNK: (Kind.SLOW, "{detail}", "check the policy: step 0 should start at the anchor"),
    C.ANCHOR_FALLBACK: (Kind.NORMAL, "{detail}", NORMAL_CHECK),
    C.CAMERA_DOWN: (Kind.CONCLUDE, "{detail}", "check the Pi stream and cable"),
    C.POLICY_HUNG: (Kind.CONCLUDE, "policy: {detail}", "check the policy log and the GPU"),
    C.POLICY_ERROR: (Kind.CONCLUDE, "policy: {detail}", "check the policy log"),
    C.INVALID_CHUNK: (Kind.CONCLUDE, "policy: invalid chunk: {detail}", "check the checkpoint and policy.kind"),
    C.ARM_FAULT: (Kind.CONCLUDE, "franka_control fault: {detail}",
                  "check the arm (reflex, connection); a fault is never cleared automatically"),
    C.GATE_REJECTED: (Kind.CONCLUDE, "franka_control: gate rejected waypoint {detail}",
                      "check the workspace box and the policy"),
    C.ARM_STOPPED: (Kind.CONCLUDE, "franka_control stopped on its own (watchdog?)", "check its log"),
    C.NO_ANSWER: (Kind.CONCLUDE, "{process}: {detail}", "check that {process} is running (see its log)"),
    C.STATE_LOST: (Kind.CONCLUDE, "franka_control: {detail}", "check that franka_control is running"),
    C.CONTRACT_VIOLATION: (Kind.CONCLUDE, "{detail}", "check that all processes are from the same commit"),
    C.HOMING_FAILED: (Kind.CONCLUDE, "franka_control: homing failed: {detail}",
                      "check the arm's path to the home pose"),
    C.STARTUP: (Kind.CONCLUDE, "startup: {detail}", "check {process}"),
    C.INTERNAL_ERROR: (Kind.CONCLUDE, "{origin}internal error: {detail}", "see the message_comms log"),
    C.EPISODE_STOP: (Kind.NORMAL, "{detail}", "nothing to do: h homes the arm, c starts the next episode"),
    C.OPERATOR_STOP: (Kind.STOP, "{detail}", "nothing to check"),
    C.DEPLOY_STOP: (Kind.STOP, "stopped by deploy: another process exited", "see the deploy message"),
}

# the slow problems a skipped step can report (counted as "skips")
SKIP_CODES = (C.CAMERA_STALE, C.IMAGE_OLD, C.STATE_GAP, C.STATE_OLD, C.ODD_CHUNK)


class FailureMonitor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._counts: dict[ProblemCode, int] = {}

    def classify(self, problem: Problem, count: int = 1) -> Verdict:
        """The verdict, from RULES; counts it too (count > 1 for sums, e.g. a chunk's width_clipped)."""
        kind, failed, check = RULES[problem.code]
        values = {"process": problem.process, "detail": problem.detail,
                  "origin": "" if problem.process == SELF else f"{problem.process}: "}
        self._counts[problem.code] = self._counts.get(problem.code, 0) + count
        return Verdict(kind, problem, failed.format(**values), problem.check or check.format(**values))

    def from_exception(self, exc: BaseException, process: str, request: str) -> Problem:
        if isinstance(exc, TransportTimeout):
            request = exc.request or request
            within = f"within {exc.timeout_s:.1f} s"
            if request == "c4.infer/v1":
                return Problem(C.POLICY_HUNG, "policy", f"no reply to infer {within}")
            if request == "c3.home/v1":
                return Problem(C.HOMING_FAILED, "franka_control", f"no reply {within}")
            return Problem(C.NO_ANSWER, exc.process or process, f"no reply to {request} {within}")
        if isinstance(exc, RemoteError):
            reply = exc.reply
            if reply.field:
                return Problem(C.CONTRACT_VIOLATION, exc.process,
                               f"{exc.process} refused {reply.request or request}: {reply.error}")
            if exc.process == "policy":
                return Problem(C.POLICY_ERROR, "policy", reply.error)
            return Problem(C.INTERNAL_ERROR, exc.process, reply.error, check="see its log")
        if isinstance(exc, Refused):
            if exc.process == "policy":
                return Problem(C.POLICY_ERROR, "policy", f"{exc.request} refused: {exc.reason}")
            if exc.request == "c3.home/v1":
                return Problem(C.HOMING_FAILED, exc.process, exc.reason)
            return Problem(C.INTERNAL_ERROR, exc.process, f"{exc.request} refused: {exc.reason}",
                           check="see its log")
        if isinstance(exc, ContractError):
            if exc.schema == "c4.chunk/v1":
                return Problem(C.INVALID_CHUNK, "policy", f"{exc.field}: {exc.detail}" if exc.field else exc.detail)
            return Problem(C.CONTRACT_VIOLATION, process, str(exc))
        return Problem(C.INTERNAL_ERROR, SELF, f"{type(exc).__name__}: {exc}")

    def check_state(self, latest: ArmState | None, now_ns: int, episode_running: bool, stop_sent: bool,
                    start_ack_ns: int | None, last_rx_ns: int | None) -> Problem | None:
        """fault (any phase), stopped on its own while running, no C2 message for state_lost_s.

        last_rx_ns: when the newest C2 message arrived (by this process's clock). franka_control repeats its last
        reading with the same t_ns during homing and after a fault, so the message t_ns can't tell "lost".
        """
        if last_rx_ns is not None and now_ns - last_rx_ns > self._settings.state_lost_s * 1e9:
            return Problem(C.STATE_LOST, "franka_control", f"no state for {self._settings.state_lost_s:.1f} s")
        if latest is None:
            return None
        if latest.mode == "fault":
            return Problem(C.ARM_FAULT, "franka_control", latest.error or "no error text")
        if (latest.mode == "stopped" and episode_running and not stop_sent
                and (start_ack_ns is None or latest.t_ns >= start_ack_ns)):
            return Problem(C.ARM_STOPPED, "franka_control", latest.error or "mode stopped")
        return None                              # homing: expected in HOMING, otherwise ignored

    def check_health(self, health: HealthReport) -> Problem | None:
        """Only "down" is a problem here (stale is seen by the steps)."""
        for name in ("main", "ultrawide"):
            if getattr(health, name).status == "down":
                return Problem(C.CAMERA_DOWN, "process_frames", f"{name} camera down")
        return None

    def counts(self) -> Mapping[ProblemCode, int]:
        return dict(self._counts)

    def reset_episode_counts(self) -> None:
        self._counts.clear()
