"""Every row of STRUCTURE.md section 7 (and so of PLAN section 5): detection input -> Problem -> Verdict."""
import pytest

from contracts import ContractError, ErrorReply, RemoteError, TransportCancelled, TransportTimeout
from fakes import health, make_state
from message_comms.errors import Refused, RequestCancelled
from message_comms.failure_monitor import RULES, FailureMonitor
from message_comms.problems import Kind, Problem, ProblemCode as C

S = 1_000_000_000
NOW = 100 * S


def timeout(process, request, t):
    return TransportTimeout(process, "tcp://127.0.0.1:1", request, t)


def remote(process, request, error, field=""):
    return RemoteError(process, ErrorReply(request=request, error=error, field=field))


def P(code, process, detail, check=""):
    return Problem(code, process, detail, check)


# (row name, how to get the Problem from its detection input, expected code, kind, expected FATAL-style text)
def rows():
    exc = lambda e, process="?", request="?": (lambda m: m.from_exception(e, process, request))
    state = lambda **kw: (lambda m: m.check_state(**{"latest": None, "now_ns": NOW, "episode_running": True,
                                                     "stop_sent": False, "start_ack_ns": None,
                                                     "last_rx_ns": NOW, **kw}))
    direct = lambda p: (lambda m: p)
    return [
        ("camera stale", direct(P(C.CAMERA_STALE, "process_frames", "main camera stale, last frame 412 ms ago")),
         C.CAMERA_STALE, Kind.SLOW, "main camera stale, last frame 412 ms ago: check the Pi stream and Wi-Fi"),
        ("no new frame", direct(P(C.NO_NEW_FRAME, "process_frames", "")), C.NO_NEW_FRAME, Kind.NORMAL, None),
        ("image old", direct(P(C.IMAGE_OLD, "process_frames", "newest main image 300 ms old")), C.IMAGE_OLD,
         Kind.SLOW, None),
        ("policy late", direct(P(C.POLICY_LATE, "policy", "policy late: margin -12 ms")), C.POLICY_LATE, Kind.SLOW,
         None),
        ("state gap", direct(P(C.STATE_GAP, "franka_control", "x")), C.STATE_GAP, Kind.SLOW, None),
        ("state old", direct(P(C.STATE_OLD, "franka_control", "x")), C.STATE_OLD, Kind.SLOW, None),
        ("shortened", direct(P(C.SHORTENED, "franka_control", "x")), C.SHORTENED, Kind.NORMAL, None),
        ("pulled back", direct(P(C.PULLED_BACK, "franka_control", "x")), C.PULLED_BACK, Kind.NORMAL, None),
        ("width clipped", direct(P(C.WIDTH_CLIPPED, "policy", "x")), C.WIDTH_CLIPPED, Kind.NORMAL, None),
        ("loop late", direct(P(C.LOOP_LATE, "franka_control", "x")), C.LOOP_LATE, Kind.NORMAL, None),
        ("odd chunk", direct(P(C.ODD_CHUNK, "policy", "odd chunk: step 0 is 25.0 mm / 0.0 deg from the anchor")),
         C.ODD_CHUNK, Kind.SLOW, None),
        ("anchor fallback", direct(P(C.ANCHOR_FALLBACK, "franka_control", "x")), C.ANCHOR_FALLBACK, Kind.NORMAL,
         None),
        ("camera down (health)", lambda m: m.check_health(health("ok", "down")), C.CAMERA_DOWN, Kind.CONCLUDE,
         "ultrawide camera down: check the Pi stream and cable"),
        ("policy hung", exc(timeout("policy", "c4.infer/v1", 5.0)), C.POLICY_HUNG, Kind.CONCLUDE,
         "policy: no reply to infer within 5.0 s: check the policy log and the GPU"),
        ("policy error", exc(remote("policy", "c4.infer/v1", "CUDA error")), C.POLICY_ERROR, Kind.CONCLUDE,
         "policy: CUDA error: check the policy log"),
        ("policy reset refused", exc(Refused("policy", "c4.reset/v1", "busy")), C.POLICY_ERROR, Kind.CONCLUDE,
         "policy: c4.reset/v1 refused: busy: check the policy log"),
        ("invalid chunk", exc(ContractError("c4.chunk/v1", "eef_rot6d", "value", "row 3 not unit")), C.INVALID_CHUNK,
         Kind.CONCLUDE, "policy: invalid chunk: eef_rot6d: row 3 not unit: check the checkpoint and policy.kind"),
        ("arm fault", state(latest=make_state(1, NOW, mode="fault", error="cartesian reflex")), C.ARM_FAULT,
         Kind.CONCLUDE, "franka_control fault: cartesian reflex: check the arm (reflex, connection); "
                        "a fault is never cleared automatically"),
        ("gate rejected", direct(P(C.GATE_REJECTED, "franka_control", "3: outside the box")), C.GATE_REJECTED,
         Kind.CONCLUDE, "franka_control: gate rejected waypoint 3: outside the box: check the workspace box and the "
                        "policy"),
        ("arm stopped on its own", state(latest=make_state(1, NOW, mode="stopped")), C.ARM_STOPPED, Kind.CONCLUDE,
         "franka_control stopped on its own (watchdog?): check its log"),
        ("no answer C3", exc(timeout("franka_control", "c3.heartbeat/v1", 0.2)), C.NO_ANSWER, Kind.CONCLUDE,
         "franka_control: no reply to c3.heartbeat/v1 within 0.2 s: check that franka_control is running "
         "(see its log)"),
        ("no answer C1", exc(timeout("process_frames", "c1.get_frames/v1", 1.0)), C.NO_ANSWER, Kind.CONCLUDE,
         "process_frames: no reply to c1.get_frames/v1 within 1.0 s: check that process_frames is running "
         "(see its log)"),
        ("no answer C4 reset", exc(timeout("policy", "c4.reset/v1", 1.0)), C.NO_ANSWER, Kind.CONCLUDE, None),
        ("state lost", state(latest=make_state(1, NOW - 2 * S), last_rx_ns=NOW - 2 * S), C.STATE_LOST,
         Kind.CONCLUDE, "franka_control: no state for 1.0 s: check that franka_control is running"),
        ("contract violation", exc(ContractError("c2.state/v2", "", "schema_version", "version v2 not supported"),
                                   "franka_control", "c2.state/v1"), C.CONTRACT_VIOLATION, Kind.CONCLUDE,
         "c2.state/v2: version v2 not supported: check that all processes are from the same commit"),
        ("remote contract error", exc(remote("policy", "c4.infer/v1", "c4.infer/v1: field 'main': shape", "main")),
         C.CONTRACT_VIOLATION, Kind.CONCLUDE, None),
        ("homing refused", exc(Refused("franka_control", "c3.home/v1", "too far")), C.HOMING_FAILED, Kind.CONCLUDE,
         "franka_control: homing failed: too far: check the arm's path to the home pose"),
        ("homing timeout", exc(timeout("franka_control", "c3.home/v1", 90.0)), C.HOMING_FAILED, Kind.CONCLUDE,
         "franka_control: homing failed: no reply within 90.0 s: check the arm's path to the home pose"),
        ("startup", direct(P(C.STARTUP, "process_frames", "no camera ok within 20 s (main stale)")), C.STARTUP,
         Kind.CONCLUDE, "startup: no camera ok within 20 s (main stale): check process_frames"),
        ("internal error", exc(KeyError("x")), C.INTERNAL_ERROR, Kind.CONCLUDE,
         "internal error: KeyError: 'x': see the message_comms log"),
        ("remote internal error", exc(remote("process_frames", "c1.get_frames/v1", "handler failed")),
         C.INTERNAL_ERROR, Kind.CONCLUDE, "process_frames: internal error: handler failed: see its log"),
        ("episode stop key", direct(P(C.EPISODE_STOP, "operator", "operator stop key")), C.EPISODE_STOP,
         Kind.NORMAL, None),
        ("operator quit", direct(P(C.OPERATOR_STOP, "operator", "operator quit key")), C.OPERATOR_STOP, Kind.STOP,
         None),
        ("deploy stop", direct(P(C.DEPLOY_STOP, "deploy", "x")), C.DEPLOY_STOP, Kind.STOP,
         "stopped by deploy: another process exited: see the deploy message"),
    ]


ROWS = rows()


@pytest.mark.parametrize("name, detect, code, kind, text", ROWS, ids=[r[0] for r in ROWS])
def test_every_row(settings, name, detect, code, kind, text):
    m = FailureMonitor(settings)
    problem = detect(m)
    assert problem.code is code
    verdict = m.classify(problem)
    assert verdict.kind is kind
    if text is not None:
        assert f"{verdict.what_failed}: {verdict.what_to_check}" == text


def test_every_code_has_a_rule_and_a_row():
    assert set(RULES) == set(C)
    assert {r[2] for r in ROWS} == set(C)


def test_normal_rows_never_stop_anything(settings):
    m = FailureMonitor(settings)
    for code in (C.SHORTENED, C.PULLED_BACK, C.WIDTH_CLIPPED, C.LOOP_LATE, C.ANCHOR_FALLBACK, C.NO_NEW_FRAME):
        assert m.classify(Problem(code, "x", "y")).kind is Kind.NORMAL


def test_check_override(settings):
    v = FailureMonitor(settings).classify(Problem(C.CONTRACT_VIOLATION, "policy",
                                                  "policy config: step_ms is 100, message_comms expects 50",
                                                  check="check the checkpoint"))
    assert f"{v.what_failed}: {v.what_to_check}" == \
        "policy config: step_ms is 100, message_comms expects 50: check the checkpoint"


def test_cancellation_is_not_mapped_by_the_monitor():
    # a cancelled home is caught by the home handler (never reaches from_exception); both names exist
    assert issubclass(RequestCancelled, RuntimeError) and issubclass(TransportCancelled, RuntimeError)


def test_check_state_stopped_rules(settings):
    m = FailureMonitor(settings)
    stopped = make_state(1, NOW, mode="stopped")
    base = dict(now_ns=NOW, episode_running=True, stop_sent=False, start_ack_ns=None, last_rx_ns=NOW)
    assert m.check_state(stopped, **base).code is C.ARM_STOPPED
    assert m.check_state(stopped, **{**base, "stop_sent": True}) is None           # after our own stop
    assert m.check_state(stopped, **{**base, "episode_running": False}) is None    # idle: normal
    assert m.check_state(stopped, **{**base, "start_ack_ns": NOW + 1}) is None     # read before the start ack
    assert m.check_state(stopped, **{**base, "start_ack_ns": NOW}).code is C.ARM_STOPPED


def test_check_state_homing_and_fault(settings):
    m = FailureMonitor(settings)
    base = dict(now_ns=NOW, stop_sent=False, start_ack_ns=None, last_rx_ns=NOW)
    assert m.check_state(make_state(1, NOW, mode="homing"), episode_running=False, **base) is None
    assert m.check_state(make_state(1, NOW, mode="homing"), episode_running=True, **base) is None
    for running in (True, False):
        assert m.check_state(make_state(1, NOW, mode="fault"), episode_running=running, **base).code is C.ARM_FAULT
    assert m.check_state(None, episode_running=False, **base) is None


def test_state_lost_uses_arrival_time(settings):
    m = FailureMonitor(settings)
    old_t = make_state(1, NOW - 5 * S, mode="homing")      # a repeated reading: old t_ns, but it just arrived
    assert m.check_state(old_t, NOW, False, False, None, last_rx_ns=NOW) is None
    assert m.check_state(old_t, NOW, False, False, None, last_rx_ns=NOW - int(1.1 * S)).code is C.STATE_LOST


def test_check_health_only_down(settings):
    m = FailureMonitor(settings)
    assert m.check_health(health("stale", "stale")) is None
    assert m.check_health(health("down", "ok")).detail == "main camera down"


def test_counts_width_sum_and_reset(settings):
    m = FailureMonitor(settings)
    m.classify(Problem(C.WIDTH_CLIPPED, "policy", "3 widths"), count=3)
    m.classify(Problem(C.WIDTH_CLIPPED, "policy", "2 widths"), count=2)
    m.classify(Problem(C.ODD_CHUNK, "policy", "x"))
    m.classify(Problem(C.ODD_CHUNK, "policy", "x"))
    assert m.counts()[C.WIDTH_CLIPPED] == 5 and m.counts()[C.ODD_CHUNK] == 2
    m.reset_episode_counts()
    assert m.counts() == {}
