# adapted from rnoro/deployment:deployment/message_comms/message_comms/problems.py @ 91b178b
# changes: the codes of STRUCTURE.md 8.5 (new: CLOCK_STALE, PC_NOT_ANSWERING, PC_RESTARTED, PC_STOPPED (was
# ARM_STOPPED), PC_REFUSED, CLOCK_MISMATCH, HELLO_MISMATCH, PC_BUSY, TCP_UNSETTLED, HOME_STOPPED, LAUNCHER_STOP (was
# DEPLOY_STOP)); process names frames, policy, PC, manager, operator, launcher.
"""Kind, ProblemCode, Problem, Verdict: the vocabulary of FailureMonitor (STRUCTURE.md 8.5)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(Enum):
    NORMAL = "normal"             # normal operation: counted, never stops anything
    SLOW = "slow"                 # keep running (PLAN 7)
    CONCLUDE = "conclude"         # unrecoverable: stop, abort, FATAL, exit 1
    STOP = "stop"                 # q / Ctrl-C: stop, record "stopped", exit 0; SIGTERM: stop, record "aborted", exit 0


class ProblemCode(Enum):
    """One member per row of STRUCTURE.md 8.5."""
    CAMERA_STALE = "camera_stale"
    NO_NEW_FRAME = "no_new_frame"
    IMAGE_OLD = "image_old"
    CLOCK_STALE = "clock_stale"
    STATE_GAP = "state_gap"
    STATE_OLD = "state_old"
    ODD_CHUNK = "odd_chunk"
    POLICY_LATE = "policy_late"
    SHORTENED = "shortened"
    PULLED_BACK = "pulled_back"
    WIDTH_CLIPPED = "width_clipped"
    LOOP_LATE = "loop_late"
    ANCHOR_FALLBACK = "anchor_fallback"
    CAMERA_DOWN = "camera_down"
    POLICY_HUNG = "policy_hung"
    POLICY_ERROR = "policy_error"
    INVALID_CHUNK = "invalid_chunk"
    NO_ANSWER = "no_answer"
    PC_NOT_ANSWERING = "pc_not_answering"
    STATE_LOST = "state_lost"
    PC_RESTARTED = "pc_restarted"
    ARM_FAULT = "arm_fault"
    GATE_REJECTED = "gate_rejected"
    PC_STOPPED = "pc_stopped"
    PC_REFUSED = "pc_refused"
    CLOCK_MISMATCH = "clock_mismatch"
    HELLO_MISMATCH = "hello_mismatch"
    PC_BUSY = "pc_busy"
    TCP_UNSETTLED = "tcp_unsettled"
    CONTRACT_VIOLATION = "contract_violation"
    HOMING_FAILED = "homing_failed"
    HOME_STOPPED = "home_stopped"
    STARTUP = "startup"
    OPERATOR_SCRIPT = "operator_script"   # a ScriptedOperator wait or refused key past operator_script_wait_s
    INTERNAL_ERROR = "internal_error"
    EPISODE_STOP = "episode_stop"         # the operator's s: the episode ends, the run goes on
    OPERATOR_STOP = "operator_stop"       # q or Ctrl-C: the run ends
    LAUNCHER_STOP = "launcher_stop"       # SIGTERM from the launcher


@dataclass(frozen=True)
class Problem:
    code: ProblemCode
    process: str                  # "frames", "policy", "PC", "manager", "operator", "launcher"
    detail: str                   # one line, e.g. "main camera stale, last frame 412 ms ago"
    check: str = ""               # overrides the rule's "what to check" when set


@dataclass(frozen=True)
class Verdict:
    kind: Kind
    problem: Problem
    what_failed: str              # e.g. "main camera down"
    what_to_check: str            # e.g. "check the Pi and ASUS_5g"
