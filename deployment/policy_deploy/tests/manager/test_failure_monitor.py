# adapted from rnoro/deployment:deployment/message_comms/tests/test_failure_monitor.py @ 91b178b
# changes: the rows and texts of STRUCTURE.md 8.5 (new codes: CLOCK_STALE, PC_NOT_ANSWERING, PC_RESTARTED, PC_STOPPED,
# PC_REFUSED, CLOCK_MISMATCH, HELLO_MISMATCH, PC_BUSY, TCP_UNSETTLED, HOME_STOPPED, LAUNCHER_STOP); check_state takes
# the newest arrival and the start ack mapped to PC time; from_exception mapping of C3 errors; imports.
"""Every row of STRUCTURE.md 8.5 (and so of PLAN 7): detection input -> Problem -> Verdict."""
import pytest
from deploy_contracts import ContractError, ErrorReply, RemoteError, TransportCancelled, TransportTimeout
from manager_fakes import health, make_state

from policy_deploy.manager.errors import PcRestarted, Refused, RequestCancelled, ScriptTimeout
from policy_deploy.manager.failure_monitor import RULES, FailureMonitor
from policy_deploy.manager.problems import Kind, Problem
from policy_deploy.manager.problems import ProblemCode as C

S = 1_000_000_000
NOW = 100 * S
PC_LINK = "check franka_control on the PC and the link"


def timeout(process, request, t):
    return TransportTimeout(process, "tcp://127.0.0.1:1", request, t)


def remote(process, request, error, field=""):
    return RemoteError(process, ErrorReply(request=request, error=error, field=field))


def P(code, process, detail, check=""):
    return Problem(code, process, detail, check)


def rows():
    exc = lambda e, process="?", request="?": (lambda m: m.from_exception(e, process, request))
    state = lambda **kw: (lambda m: m.check_state(**{"latest": None, "now_ns": NOW, "last_arrival_ns": NOW,
                                                     "episode_running": True, "stop_sent": False,
                                                     "start_ack_pc_ns": None, **kw}))
    direct = lambda p: (lambda m: p)
    return [
        ("camera stale", direct(P(C.CAMERA_STALE, "frames", "main camera stale, last frame 412 ms ago")),
         C.CAMERA_STALE, Kind.SLOW, "main camera stale, last frame 412 ms ago: check the Pi stream and ASUS_5g"),
        ("no new frame", direct(P(C.NO_NEW_FRAME, "frames", "")), C.NO_NEW_FRAME, Kind.NORMAL, None),
        ("image old", direct(P(C.IMAGE_OLD, "frames", "newest main image 300 ms old")), C.IMAGE_OLD, Kind.SLOW,
         None),
        ("clock stale", direct(P(C.CLOCK_STALE, "PC", "PC clock map not ready: newest clock sample 6.0 s old")),
         C.CLOCK_STALE, Kind.SLOW, None),
        ("policy late", direct(P(C.POLICY_LATE, "policy", "policy late: margin -12 ms")), C.POLICY_LATE, Kind.SLOW,
         None),
        ("state gap", direct(P(C.STATE_GAP, "PC", "x")), C.STATE_GAP, Kind.SLOW, None),
        ("state old", direct(P(C.STATE_OLD, "PC", "x")), C.STATE_OLD, Kind.SLOW, None),
        ("shortened", direct(P(C.SHORTENED, "PC", "x")), C.SHORTENED, Kind.NORMAL, None),
        ("pulled back", direct(P(C.PULLED_BACK, "PC", "x")), C.PULLED_BACK, Kind.NORMAL, None),
        ("width clipped", direct(P(C.WIDTH_CLIPPED, "policy", "x")), C.WIDTH_CLIPPED, Kind.NORMAL, None),
        ("loop late", direct(P(C.LOOP_LATE, "PC", "x")), C.LOOP_LATE, Kind.NORMAL, None),
        ("odd chunk", direct(P(C.ODD_CHUNK, "policy", "odd chunk: step 0 is 25.0 mm / 0.0 deg from the anchor")),
         C.ODD_CHUNK, Kind.SLOW, None),
        ("anchor fallback", direct(P(C.ANCHOR_FALLBACK, "PC", "x")), C.ANCHOR_FALLBACK, Kind.NORMAL, None),
        ("camera down (health)", lambda m: m.check_health(health("ok", "down")), C.CAMERA_DOWN, Kind.CONCLUDE,
         "ultrawide camera down: check the Pi and ASUS_5g"),
        ("policy hung", exc(timeout("policy", "c4.infer/v1", 5.0)), C.POLICY_HUNG, Kind.CONCLUDE,
         "policy: no reply to infer within 5.0 s: check the policy log and the GPU"),
        ("policy error", exc(remote("policy", "c4.infer/v1", "CUDA error")), C.POLICY_ERROR, Kind.CONCLUDE,
         "policy: CUDA error: check the policy log"),
        ("policy reset refused", exc(Refused("policy", "c4.reset/v1", "busy")), C.POLICY_ERROR, Kind.CONCLUDE,
         "policy: c4.reset/v1 refused: busy: check the policy log"),
        ("invalid chunk", exc(ContractError("c4.chunk/v1", "eef_rot6d", "value", "row 3 not unit")), C.INVALID_CHUNK,
         Kind.CONCLUDE, "policy: invalid chunk: eef_rot6d: row 3 not unit: check the checkpoint and policy.kind"),
        ("no answer C1", exc(timeout("frames", "c1.get_frames/v1", 1.0)), C.NO_ANSWER, Kind.CONCLUDE,
         "frames: no reply to c1.get_frames/v1 within 1.0 s: check that frames is running (see its log)"),
        ("no answer C4 reset", exc(timeout("policy", "c4.reset/v1", 1.0)), C.NO_ANSWER, Kind.CONCLUDE, None),
        ("PC not answering", exc(timeout("PC", "c3.heartbeat/v1", 0.3)), C.PC_NOT_ANSWERING, Kind.CONCLUDE,
         f"PC not answering: no reply to c3.heartbeat/v1 within 0.3 s: {PC_LINK}"),
        ("PC not answering hello", exc(timeout("PC", "c3.hello/v1", 0.3)), C.PC_NOT_ANSWERING, Kind.CONCLUDE,
         f"PC not answering: no reply to c3.hello/v1 within 0.3 s: {PC_LINK}"),
        ("state lost", state(latest=make_state(1, NOW - 3 * S), last_arrival_ns=NOW - int(2.1 * S)), C.STATE_LOST,
         Kind.CONCLUDE, f"PC not answering: no C2 state for 2.0 s: {PC_LINK}"),
        ("PC restarted", exc(PcRestarted("C2 seq went back from 9 to 1")), C.PC_RESTARTED, Kind.CONCLUDE,
         "franka_control restarted on the PC: restart policy_deploy run"),
        ("PC rebooted", exc(PcRestarted("hello t_pc_ns went back", rebooted=True)), C.PC_RESTARTED, Kind.CONCLUDE,
         "the PC rebooted (its clock went back): start control_deploy run on the PC, then policy_deploy run"),
        ("arm fault", state(latest=make_state(1, NOW, mode="fault", error="cartesian reflex")), C.ARM_FAULT,
         Kind.CONCLUDE, "the PC reports fault: cartesian reflex: the PC holds; Ctrl-C and restart control_deploy run "
                        "on the PC (release the user stop if it was a reflex)"),
        ("gate rejected", direct(P(C.GATE_REJECTED, "PC", "3: outside the box")), C.GATE_REJECTED, Kind.CONCLUDE,
         "the PC's gate rejected waypoint 3: outside the box: check the workspace box and the policy"),
        ("PC stopped", state(latest=make_state(1, NOW, mode="stopped")), C.PC_STOPPED, Kind.CONCLUDE,
         "the PC stopped the arm (watchdog or the PC's `s`): check the link and the PC"),
        ("PC refused (error/v1 without field)",
         exc(remote("PC", "c3.set_waypoints/v1", "c3.set_waypoints/v1: refused: mode is stopped")), C.PC_REFUSED,
         Kind.CONCLUDE, "the PC refused c3.set_waypoints/v1: c3.set_waypoints/v1: refused: mode is stopped: check the "
                        "PC terminal"),
        ("PC refused stop", exc(Refused("PC", "c3.heartbeat/v1", "?")), C.PC_REFUSED, Kind.CONCLUDE, None),
        ("clock mismatch", direct(P(C.CLOCK_MISMATCH, "PC", "the PC skipped 3 waypoint sets the laptop timed ahead")),
         C.CLOCK_MISMATCH, Kind.CONCLUDE, "clock mismatch: the PC skipped 3 waypoint sets the laptop timed ahead: run "
                                          "pc_probe; check clock.estimator"),
        ("hello mismatch", direct(P(C.HELLO_MISMATCH, "PC", "hello mismatch: mock: PC false, laptop true (x)",
                                    check="run both sides with --mock, or neither")), C.HELLO_MISMATCH, Kind.CONCLUDE,
         "hello mismatch: mock: PC false, laptop true (x): run both sides with --mock, or neither"),
        ("PC busy", direct(P(C.PC_BUSY, "PC", "C2 still shows holding")), C.PC_BUSY, Kind.CONCLUDE,
         "the PC is driving an episode for another C3 client: stop the other laptop session, pc_probe or console"),
        ("TCP unsettled", direct(P(C.TCP_UNSETTLED, "policy", "x")), C.TCP_UNSETTLED, Kind.CONCLUDE,
         "trained policy refused on the real arm: contracts/frames.yaml settled is false (PLAN §9): use policy.kind "
         "hold_still, or settle the TCP"),
        ("contract violation", exc(ContractError("c2.state/v2", "", "schema_version", "version v2 not supported"),
                                   "PC", "c2.state/v1"), C.CONTRACT_VIOLATION, Kind.CONCLUDE,
         "c2.state/v2: version v2 not supported: check that both machines run the same commit"),
        ("remote contract error (PC, with field)",
         exc(remote("PC", "c3.set_waypoints/v1", "c3.set_waypoints/v1: field 'O_T_TCP': not rigid", "O_T_TCP")),
         C.CONTRACT_VIOLATION, Kind.CONCLUDE, None),
        ("remote contract error (policy)", exc(remote("policy", "c4.infer/v1", "c4.infer/v1: field 'main'", "main")),
         C.CONTRACT_VIOLATION, Kind.CONCLUDE, None),
        ("homing failed", direct(P(C.HOMING_FAILED, "PC", "homing failed: too far")), C.HOMING_FAILED, Kind.CONCLUDE,
         "the PC's homing failed: homing failed: too far: check the arm's path to the home pose"),
        ("homing timeout", exc(timeout("PC", "c3.home/v1", 90.0)), C.HOMING_FAILED, Kind.CONCLUDE,
         "the PC's homing failed: no reply within 90.0 s: check the arm's path to the home pose"),
        ("homing error/v1", exc(remote("PC", "c3.home/v1", "handler failed")), C.HOMING_FAILED, Kind.CONCLUDE, None),
        ("home stopped", direct(P(C.HOME_STOPPED, "PC", "stopped: homing stopped by the PC keyboard")),
         C.HOME_STOPPED, Kind.NORMAL, "home stopped on the PC: stopped: homing stopped by the PC keyboard: the arm "
                                      "holds where it stopped"),
        ("startup", direct(P(C.STARTUP, "frames", "no camera ok within 20 s (main stale)")), C.STARTUP,
         Kind.CONCLUDE, "startup: no camera ok within 20 s (main stale): check frames"),
        ("operator script", exc(ScriptTimeout("s.yaml:3: wait_phase idle: not reached (phase running) after 30.0 s")),
         C.OPERATOR_SCRIPT, Kind.CONCLUDE, "operator script: s.yaml:3: wait_phase idle: not reached (phase running) "
         "after 30.0 s: check the script (manager.operator_script) or raise operator_script_wait_s"),
        ("internal error", exc(KeyError("x")), C.INTERNAL_ERROR, Kind.CONCLUDE,
         "internal error: KeyError: 'x': see the manager log"),
        ("remote internal error", exc(remote("frames", "c1.get_frames/v1", "handler failed")),
         C.INTERNAL_ERROR, Kind.CONCLUDE, "frames: internal error: handler failed: see its log"),
        ("episode stop key", direct(P(C.EPISODE_STOP, "operator", "operator stop key")), C.EPISODE_STOP,
         Kind.NORMAL, None),
        ("operator quit", direct(P(C.OPERATOR_STOP, "operator", "operator quit key")), C.OPERATOR_STOP, Kind.STOP,
         None),
        ("launcher stop", direct(P(C.LAUNCHER_STOP, "launcher", "x")), C.LAUNCHER_STOP, Kind.STOP,
         "stopped by the launcher: another process exited: see the launcher message"),
    ]


ROWS = rows()


@pytest.mark.parametrize("name, detect, code, kind, text", ROWS, ids=[r[0] for r in ROWS])
def test_every_row(name, detect, code, kind, text):
    m = FailureMonitor(2.0)
    problem = detect(m)
    assert problem.code is code
    verdict = m.classify(problem)
    assert verdict.kind is kind
    if text is not None:
        assert f"{verdict.what_failed}: {verdict.what_to_check}" == text


def test_every_code_has_a_rule_and_a_row():
    assert set(RULES) == set(C)
    assert {r[2] for r in ROWS} == set(C)


def test_normal_rows_never_stop_anything():
    m = FailureMonitor(2.0)
    for code in (C.SHORTENED, C.PULLED_BACK, C.WIDTH_CLIPPED, C.LOOP_LATE, C.ANCHOR_FALLBACK, C.NO_NEW_FRAME,
                 C.HOME_STOPPED):
        assert m.classify(Problem(code, "x", "y")).kind is Kind.NORMAL


def test_check_override():
    v = FailureMonitor(2.0).classify(Problem(C.CONTRACT_VIOLATION, "policy",
                                             "policy config: step_ms is 100, manager expects 50",
                                             check="check the checkpoint"))
    assert f"{v.what_failed}: {v.what_to_check}" == \
        "policy config: step_ms is 100, manager expects 50: check the checkpoint"


def test_cancellation_is_not_mapped_by_the_monitor():
    # a cancelled home is caught by the home handler (never reaches from_exception); both names exist
    assert issubclass(RequestCancelled, RuntimeError) and issubclass(TransportCancelled, RuntimeError)


def test_check_state_stopped_rules_on_the_pc_clock():
    m = FailureMonitor(2.0)
    stopped = make_state(1, NOW, mode="stopped")
    base = dict(now_ns=5, last_arrival_ns=5, episode_running=True, stop_sent=False, start_ack_pc_ns=None)
    assert m.check_state(stopped, **base).code is C.PC_STOPPED
    assert m.check_state(stopped, **{**base, "stop_sent": True}) is None           # after our own stop
    assert m.check_state(stopped, **{**base, "episode_running": False}) is None    # idle: normal
    assert m.check_state(stopped, **{**base, "start_ack_pc_ns": NOW + 1}) is None  # read before the start ack
    assert m.check_state(stopped, **{**base, "start_ack_pc_ns": NOW}).code is C.PC_STOPPED


def test_check_state_homing_and_fault():
    m = FailureMonitor(2.0)
    base = dict(now_ns=NOW, last_arrival_ns=NOW, stop_sent=False, start_ack_pc_ns=None)
    assert m.check_state(make_state(1, NOW, mode="homing"), episode_running=False, **base) is None
    assert m.check_state(make_state(1, NOW, mode="homing"), episode_running=True, **base) is None
    for running in (True, False):
        assert m.check_state(make_state(1, NOW, mode="fault"), episode_running=running, **base).code is C.ARM_FAULT
    assert m.check_state(None, episode_running=False, **base) is None


def test_state_lost_uses_arrival_time():
    m = FailureMonitor(2.0)
    old_t = make_state(1, NOW - 5 * S, mode="homing")      # a repeated reading: old t_ns, but it just arrived
    assert m.check_state(old_t, NOW, NOW, False, False, None) is None
    assert m.check_state(old_t, NOW, NOW - int(2.1 * S), False, False, None).code is C.STATE_LOST
    assert m.check_state(None, NOW, None, False, False, None) is None             # nothing arrived yet


def test_check_health_only_down():
    m = FailureMonitor(2.0)
    assert m.check_health(health("stale", "stale")) is None
    p = m.check_health(health("down", "ok"))
    assert p.detail == "main camera down" and p.process == "frames"


def test_counts_width_sum_and_reset():
    m = FailureMonitor(2.0)
    m.classify(Problem(C.WIDTH_CLIPPED, "policy", "3 widths"), count=3)
    m.classify(Problem(C.WIDTH_CLIPPED, "policy", "2 widths"), count=2)
    m.classify(Problem(C.ODD_CHUNK, "policy", "x"))
    m.classify(Problem(C.ODD_CHUNK, "policy", "x"))
    assert m.counts()[C.WIDTH_CLIPPED] == 5 and m.counts()[C.ODD_CHUNK] == 2
    m.reset_episode_counts()
    assert m.counts() == {}
