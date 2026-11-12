"""Kind, ProblemCode, Problem, Verdict: the vocabulary of FailureMonitor (STRUCTURE.md 4.7, 7)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Kind(Enum):
    NORMAL = "normal"             # normal operation: counted, never stops anything
    SLOW = "slow"                 # keep running (PLAN section 5)
    CONCLUDE = "conclude"         # unrecoverable: stop, abort, FATAL, exit 1
    STOP = "stop"                 # q / Ctrl-C: stop, record "stopped", exit 0; SIGTERM: stop, record "aborted", exit 0


class ProblemCode(Enum):
    """One member per row of STRUCTURE.md section 7."""
    CAMERA_STALE = "camera_stale"
    NO_NEW_FRAME = "no_new_frame"
    IMAGE_OLD = "image_old"
    POLICY_LATE = "policy_late"
    STATE_GAP = "state_gap"
    STATE_OLD = "state_old"
    SHORTENED = "shortened"
    PULLED_BACK = "pulled_back"
    WIDTH_CLIPPED = "width_clipped"
    LOOP_LATE = "loop_late"
    ODD_CHUNK = "odd_chunk"
    ANCHOR_FALLBACK = "anchor_fallback"
    CAMERA_DOWN = "camera_down"
    POLICY_HUNG = "policy_hung"
    POLICY_ERROR = "policy_error"
    INVALID_CHUNK = "invalid_chunk"
    ARM_FAULT = "arm_fault"
    GATE_REJECTED = "gate_rejected"
    ARM_STOPPED = "arm_stopped"
    NO_ANSWER = "no_answer"
    STATE_LOST = "state_lost"
    CONTRACT_VIOLATION = "contract_violation"
    HOMING_FAILED = "homing_failed"
    STARTUP = "startup"
    INTERNAL_ERROR = "internal_error"
    EPISODE_STOP = "episode_stop"         # the operator's s: the episode ends, the run goes on
    OPERATOR_STOP = "operator_stop"       # q or Ctrl-C: the run ends
    DEPLOY_STOP = "deploy_stop"           # SIGTERM from deploy run


@dataclass(frozen=True)
class Problem:
    code: ProblemCode
    process: str                  # "process_frames", "policy", "franka_control", "message_comms", "operator"
    detail: str                   # one line, e.g. "main camera stale, last frame 412 ms ago"
    check: str = ""               # overrides the rule's "what to check" when set (e.g. a policy config mismatch)


@dataclass(frozen=True)
class Verdict:
    kind: Kind
    problem: Problem
    what_failed: str              # e.g. "main camera down"
    what_to_check: str            # e.g. "check the Pi stream and cable"
