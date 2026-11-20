"""FailureMonitor (STRUCTURE.md 6.5; PLAN 7): the kinds of every code, EPISODE and WARN outside an episode, the state
rules, R1 health and the exceptions. Wave 3 (no software stops): the data problems are WARN, a C2 seq gap is NORMAL."""
from __future__ import annotations

import pytest
from deploy_contracts import ContractError, ErrorReply, RemoteError, TransportTimeout
from teleop_helpers import frames_health, make_state

from teleop_collect.teleop.errors import PcRestarted, Refused
from teleop_collect.teleop.failure_monitor import COUNTED_CODES, RULES, SLOW_CODES, WARN_CODES, FailureMonitor
from teleop_collect.teleop.problems import Kind, Problem, ProblemCode as C

S = 1_000_000_000

KINDS = {
    Kind.SLOW: {C.STATE_OLD, C.CLOCK_STALE, C.SET_SKIPPED},
    Kind.NORMAL: {C.STATE_GAP, C.PULLED_BACK, C.SHORTENED, C.LOOP_LATE},
    Kind.WARN: {C.CAMERA_DOWN, C.CAMERA_STALE, C.RECORDING_FAILED},
    Kind.EPISODE: {C.PC_STOPPED, C.OPERATOR_ABORT},
    Kind.CONCLUDE: {C.PC_NOT_ANSWERING, C.STATE_LOST, C.PC_RESTARTED, C.ARM_FAULT, C.GATE_REJECTED, C.CLOCK_MISMATCH,
                    C.HELLO_MISMATCH, C.PC_BUSY, C.CONTRACT_VIOLATION, C.STARTUP, C.NO_ANSWER,
                    C.INTERNAL_ERROR, C.PC_REFUSED},
    Kind.STOP: {C.OPERATOR_STOP, C.LAUNCHER_STOP},
}


def test_every_code_has_its_kind():
    assert set(RULES) == set(C)
    for kind, codes in KINDS.items():
        for code in codes:
            assert RULES[code][0] is kind, code
    assert set(SLOW_CODES) == KINDS[Kind.SLOW] and set(WARN_CODES) == KINDS[Kind.WARN]
    assert set(COUNTED_CODES) == KINDS[Kind.SLOW] | KINDS[Kind.WARN] | {C.STATE_GAP}     # meta.yaml counts


def test_episode_and_warn_problems_outside_an_episode_are_normal():
    m = FailureMonitor(2.0)
    p = Problem(C.CAMERA_DOWN, "frames", "main camera down")
    assert m.classify(p).kind is Kind.WARN
    v = m.classify(p, episode_open=False)
    assert v.kind is Kind.NORMAL and f"{v.what_failed}: {v.what_to_check}" == \
        "main camera down: check the Pi and ASUS_5g"
    assert m.counts()[C.CAMERA_DOWN] == 2
    stopped = Problem(C.PC_STOPPED, "PC", "mode stopped")
    assert m.classify(stopped).kind is Kind.EPISODE
    assert m.classify(stopped, episode_open=False).kind is Kind.NORMAL


def test_texts_of_plan_7():
    m = FailureMonitor(2.0)
    assert m.classify(Problem(C.PC_STOPPED, "PC", "x")).what_failed.startswith("the PC stopped the arm")
    assert m.classify(Problem(C.OPERATOR_ABORT, "operator", "Esc")).what_failed == "operator stop"
    v = m.classify(Problem(C.RECORDING_FAILED, "frames", "queue full"))
    assert v.what_failed == "recording could not keep up / disk error: queue full"
    assert m.classify(Problem(C.STATE_LOST, "PC", "no C2 state for 2.0 s")).what_failed.startswith(
        "PC not answering")


def test_check_state():
    m = FailureMonitor(2.0)
    now = 100 * S
    assert m.check_state(None, now, None, False, False, None) is None
    assert m.check_state(make_state(1, 5), now, now - 3 * S, False, False, None).code is C.STATE_LOST
    assert m.check_state(make_state(1, 5, mode="fault", error="reflex"), now, now, False, False, None).code \
        is C.ARM_FAULT
    stopped = make_state(1, 50 * S, mode="stopped")
    assert m.check_state(stopped, now, now, True, False, 40 * S).code is C.PC_STOPPED
    assert m.check_state(stopped, now, now, True, True, 40 * S) is None          # our own stop
    assert m.check_state(stopped, now, now, True, False, 60 * S) is None         # older than the start ack
    assert m.check_state(stopped, now, now, False, False, None) is None          # no episode


def test_check_health():
    m = FailureMonitor(2.0)
    assert m.check_health(frames_health(), recording=False) is None
    assert m.check_health(frames_health("stale"), recording=False) is None
    assert m.check_health(frames_health("stale", recording=True), recording=True).code is C.CAMERA_STALE
    assert m.check_health(frames_health(ultrawide="down"), recording=False).detail == "ultrawide camera down"
    assert m.check_health(frames_health(recording=True, error="queue full"), recording=True).code \
        is C.RECORDING_FAILED
    assert m.check_health(frames_health(recording=False), recording=True).detail == "frames is not recording"


@pytest.mark.parametrize("exc, code", [
    (PcRestarted("seq back"), C.PC_RESTARTED),
    (TransportTimeout("PC", "tcp://x", "c3.heartbeat/v1", 0.3), C.PC_NOT_ANSWERING),
    (TransportTimeout("frames", "tcp://x", "r1.get_health/v1", 0.05), C.NO_ANSWER),
    (RemoteError("PC", ErrorReply(request="c3.set_waypoints/v1", error="refused: mode is stopped", field="")),
     C.PC_REFUSED),
    (RemoteError("PC", ErrorReply(request="c3.set_waypoints/v1", error="bad", field="t_ns")), C.CONTRACT_VIOLATION),
    (Refused("frames", "r1.start_recording/v1", "x"), C.INTERNAL_ERROR),
    (ContractError("r1.health/v1", "recording", "type", "not a bool"), C.CONTRACT_VIOLATION),
    (ValueError("oops"), C.INTERNAL_ERROR),
])
def test_from_exception(exc, code):
    assert FailureMonitor(2.0).from_exception(exc, "PC", "c3.heartbeat/v1").code is code
    v = FailureMonitor(2.0).classify(FailureMonitor(2.0).from_exception(PcRestarted("x", rebooted=True), "PC", ""))
    assert v.what_to_check == "start control_deploy run on the PC, then teleop_collect run"
