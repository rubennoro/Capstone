# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/problems.py @ 2ffe641
# changes: the kinds and codes of teleop_collect STRUCTURE.md 6.5: + Kind.EPISODE (abort the open episode, stay up);
# + SET_SKIPPED, OPERATOR_ABORT, RECORDING_FAILED; the policy codes (NO_NEW_FRAME, IMAGE_OLD, ODD_CHUNK, POLICY_LATE,
# WIDTH_CLIPPED, ANCHOR_FALLBACK, POLICY_HUNG, POLICY_ERROR, INVALID_CHUNK), TCP_UNSETTLED, OPERATOR_SCRIPT and
# EPISODE_STOP dropped; process names frames, PC, teleop, ui, operator, launcher. The source's own header follows.
# adapted from rnoro/deployment:deployment/message_comms/message_comms/problems.py @ 91b178b
# changes: the codes of STRUCTURE.md 8.5 (new: CLOCK_STALE, PC_NOT_ANSWERING, PC_RESTARTED, PC_STOPPED (was
# ARM_STOPPED), PC_REFUSED, CLOCK_MISMATCH, HELLO_MISMATCH, PC_BUSY, TCP_UNSETTLED, HOME_STOPPED, LAUNCHER_STOP (was
# DEPLOY_STOP)); process names frames, policy, PC, manager, operator, launcher.
"""Kind, ProblemCode, Problem, Verdict: the vocabulary of FailureMonitor (STRUCTURE.md 6.5; PLAN 7)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(Enum):
    NORMAL = "normal"             # normal operation: counted, never stops anything
    SLOW = "slow"                 # keep going: no motion this cycle, the recording continues (PLAN 7)
    EPISODE = "episode"           # abort the open episode (C3 stop, its folder deleted); the app stays up
    CONCLUDE = "conclude"         # unrecoverable: stop, abort the open episode, FATAL, exit 1
    STOP = "stop"                 # Ctrl-C / SIGTERM: stop, abort the open episode, exit 0


class ProblemCode(Enum):
    """One member per row of STRUCTURE.md 6.5."""
    CAMERA_STALE = "camera_stale"
    CLOCK_STALE = "clock_stale"
    STATE_GAP = "state_gap"
    STATE_OLD = "state_old"
    SET_SKIPPED = "set_skipped"
    SHORTENED = "shortened"
    PULLED_BACK = "pulled_back"
    LOOP_LATE = "loop_late"
    HOME_STOPPED = "home_stopped"
    CAMERA_DOWN = "camera_down"
    PC_STOPPED = "pc_stopped"
    OPERATOR_ABORT = "operator_abort"
    RECORDING_FAILED = "recording_failed"
    NO_ANSWER = "no_answer"
    PC_NOT_ANSWERING = "pc_not_answering"
    STATE_LOST = "state_lost"
    PC_RESTARTED = "pc_restarted"
    ARM_FAULT = "arm_fault"
    GATE_REJECTED = "gate_rejected"
    PC_REFUSED = "pc_refused"
    CLOCK_MISMATCH = "clock_mismatch"
    HELLO_MISMATCH = "hello_mismatch"
    PC_BUSY = "pc_busy"
    CONTRACT_VIOLATION = "contract_violation"
    HOMING_FAILED = "homing_failed"
    STARTUP = "startup"
    INTERNAL_ERROR = "internal_error"
    OPERATOR_STOP = "operator_stop"       # Ctrl-C: the run ends
    LAUNCHER_STOP = "launcher_stop"       # SIGTERM from the launcher


@dataclass(frozen=True)
class Problem:
    code: ProblemCode
    process: str                  # "frames", "PC", "teleop", "ui", "operator", "launcher"
    detail: str                   # one line, e.g. "main camera down"
    check: str = ""               # overrides the rule's "what to check" when set


@dataclass(frozen=True)
class Verdict:
    kind: Kind
    problem: Problem
    what_failed: str              # e.g. "main camera down"
    what_to_check: str            # e.g. "check the Pi and ASUS_5g"
