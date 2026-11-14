"""The console against a fake PC built from deploy_contracts (RpcServer + Publisher), and its judging on synthetic C2
sequences (control_deploy/STRUCTURE.md 6.5, 9.2)."""
from __future__ import annotations

import io
import math
import threading
import time

import numpy as np
import pytest
from deploy_contracts import Ack, ErrorReply, GateCounters, SetWaypointsReply, WaypointIssue
from helpers import FakePC, pose, state, write_station

from control_deploy.console import steps as st
from control_deploy.console import waypoints as wpts
from control_deploy.console.console import Console, Refused, workspace_box
from control_deploy.console.steps import Sample, Sent


@pytest.fixture
def station(tmp_path):
    return write_station(tmp_path)


def make(station, pc: FakePC | None, out=None, mock=True, peers=lambda port: [], **kw) -> Console:
    endpoints = pc.endpoints if pc else ("tcp://127.0.0.1:1", "tcp://127.0.0.1:2")
    return Console(station, mock, True, out=out or io.StringIO(), peers=peers, endpoints=endpoints, **kw)


def run_preflight(c: Console) -> str:
    from deploy_contracts import ArmState, Subscriber
    c.sub = Subscriber("t", c.state_ep, ArmState, 1000)
    try:
        c.preflight()
        return "ok"
    except Refused as exc:
        return str(exc)
    finally:
        c.sub.close()
        if c.rpc:
            c.rpc.close()


# ---- preflight ----

def test_no_c2_is_refused(station):
    assert run_preflight(make(station, None)).startswith("franka_control not running (no C2 on tcp://127.0.0.1:1)")


def test_a_c3_peer_is_refused(station):
    pc = FakePC()
    try:
        text = run_preflight(make(station, pc, peers=lambda port: ["192.168.50.20"]))
        assert text == "a C3 client from 192.168.50.20 is connected (policy_deploy on the laptop?): stop it first"
        assert pc.received == []   # nothing sent on C3
    finally:
        pc.close()


def test_rule_h2_waits_then_refuses(station):
    pc = FakePC(mode="holding")
    try:
        t0 = time.monotonic()
        assert run_preflight(make(station, pc, h2_wait_s=0.3)) == "the PC is driving an episode for another C3 client"
        assert time.monotonic() - t0 >= 0.3
        assert pc.received == []
    finally:
        pc.close()


def test_rule_h2_waits_and_goes_on_when_the_episode_ends(station):
    pc = FakePC(mode="holding")
    threading.Timer(0.3, lambda: setattr(pc, "mode", "stopped")).start()
    try:
        assert run_preflight(make(station, pc, h2_wait_s=2.0)) == "ok"
    finally:
        pc.close()


def test_homing_is_waited(station):
    pc = FakePC(mode="homing")
    threading.Timer(0.3, lambda: setattr(pc, "mode", "idle")).start()
    try:
        assert run_preflight(make(station, pc, homing_wait_s=3.0)) == "ok"
    finally:
        pc.close()


def test_fault_is_refused(station):
    pc = FakePC(mode="fault")
    pc.error = "libfranka: reflex"
    try:
        assert run_preflight(make(station, pc)) == ("franka_control is in fault: libfranka: reflex: Ctrl-C it and run "
                                                    "control_deploy run again")
    finally:
        pc.close()


def test_hello_mismatches_are_listed(station):
    pc = FakePC(hello={"max_waypoints": 32, "watchdog_limit_s": 0.5})
    out = io.StringIO()
    try:
        c = make(station, pc, out=out)
        text = run_preflight(c)
        assert "2 hello mismatch(es)" in text
        log = c.out.getvalue()
        assert "max_waypoints" in log and "watchdog_limit_s" in log
    finally:
        pc.close()


def test_hello_mock_must_match(station):
    pc = FakePC()   # mock true
    try:
        assert "hello mismatch" in run_preflight(make(station, pc, mock=False))
    finally:
        pc.close()


def test_the_clock_check(station):
    pc = FakePC(clock_offset_ns=1_000_000_000)
    try:
        assert "the console must run on the PC" in run_preflight(make(station, pc))
    finally:
        pc.close()


def test_yes_without_mock_is_refused(station):
    out = io.StringIO()
    assert make(station, None, out=out, mock=False, yes=True).run() == 2
    assert "--yes is refused without --mock" in out.getvalue()


def test_unknown_only_step_is_refused(station):
    out = io.StringIO()
    assert make(station, None, out=out, only=["jump"]).run() == 2
    assert "unknown step(s) jump" in out.getvalue()


def test_the_box_comes_from_config_and_station(tmp_path):
    s = write_station(tmp_path, franka_control={"safety": {"workspace_max_m": [0.7, 0.3, 0.6]}})
    from control_deploy.station import Station
    lo, hi = workspace_box(Station.load(s))
    assert list(lo) == [0.25, -0.40, 0.05] and list(hi) == [0.7, 0.3, 0.6]


# ---- the out-of-box step against a fake PC that wrongly accepts everything ----

def test_an_accepted_out_of_box_waypoint_sends_stop_and_fails_in_capitals(station):
    pc = FakePC()
    out = io.StringIO()
    try:
        assert make(station, pc, out=out, only=["out_of_box"], yes=True).run() == 1
        text = out.getvalue()
        assert "FAIL: THE GATE ACCEPTED 1 WAYPOINT(S) OUTSIDE THE WORKSPACE BOX" in text
        i = pc.received.index("c3.set_waypoints/v1")
        assert "c3.stop/v1" in pc.received[i + 1:]
    finally:
        pc.close()


def test_a_whole_session_against_a_fake_pc_that_rejects_out_of_box(station):
    """hold and out_of_box with the fake PC: the session folder, results.yaml, the closing hint. A failed step ends the
    session (the fake PC has no watchdog: that step fails, and nothing after it runs)."""
    pc = FakePC(reject_box_z=0.70)
    out = io.StringIO()
    try:
        c = make(station, pc, out=out, only=["hold", "out_of_box"], yes=True)
        c.settings = c.settings.__class__(hold_s=0.3)
        assert c.run() == 0, out.getvalue()
        text = out.getvalue()
        assert "hold: PASS" in text and "out_of_box: PASS" in text
        assert "in fault by design (out_of_box): Ctrl-C control_deploy run and start it again" in text
        import yaml
        results = yaml.safe_load((c.session.dir / "results.yaml").read_text())
        assert [s["name"] for s in results["steps"]] == ["hold", "out_of_box"]
        assert (c.session.dir / "console.log").is_file()
    finally:
        pc.close()
    pc = FakePC()
    out = io.StringIO()
    try:
        c = make(station, pc, out=out, only=["watchdog", "out_of_box"], yes=True)
        assert c.run() == 1
        assert "watchdog: FAIL" in out.getvalue() and "out_of_box" not in out.getvalue().split("watchdog: FAIL")[1]
        assert pc.received[-1] == "c3.stop/v1"   # a failed step ends with stop
    finally:
        pc.close()


# ---- judging on synthetic C2 sequences ----

MS = 1_000_000


def seq(modes_and_poses, t0=0, dt=10 * MS, **kw):
    return [Sample(t0 + i * dt, state(m, t_ns=t0 + i * dt, p=p, target=kw.get("target", p), loop_late=kw.get("late", 0)))
            for i, (m, p) in enumerate(modes_and_poses)]


def test_hold_pass_warn_fail():
    p = pose(0.45, 0, 0.45)
    samples = seq([("holding", p)] * 50)
    assert st.evaluate_hold(Ack(ok=True, reason=""), 0, samples, samples).status == "pass"
    late = [Sample(s.arrival_ns, state("holding", p=p, loop_late=i)) for i, s in enumerate(samples)]
    assert st.evaluate_hold(Ack(ok=True, reason=""), 0, late, late).status == "warn"
    moved = samples + seq([("holding", pose(0.45, 0, 0.453))], t0=600 * MS)
    r = st.evaluate_hold(Ack(ok=True, reason=""), 0, moved, moved)
    assert r.status == "fail" and r.numbers["drift_mm"] == pytest.approx(3.0, abs=1e-6)
    slow = seq([("idle", p)] * 20 + [("holding", p)] * 10)
    assert st.evaluate_hold(Ack(ok=True, reason=""), 0, slow, slow).status == "fail"   # holding after 200 ms
    assert st.evaluate_hold(Ack(ok=False, reason="refused: mode is fault"), 0, [], []).status == "fail"


def follow_case(track_offset_m=0.0, plan_offset_m=0.0, rot_track=0.0, rotation=False, rejected=False, end="holding"):
    start = pose(0.45, 0, 0.45)
    plan = wpts.PlanModel(0, start, 0.08)
    goal = wpts.translate(start, [0, 0, 0.015]) if not rotation else wpts.rotate_about_tcp_z(start, math.radians(5))
    pts = [wpts.PlanPoint(1000 * MS, goal, 0.08)]
    reply = SetWaypointsReply(chunk_id=1, accepted=0 if rejected else 1, shortened=0, skipped=0,
                              rejected=(WaypointIssue(index=0, reason="outside workspace box"),) if rejected else ())
    plan.apply(0, pts)
    sent = [Sent(0, 0, pts, reply)]
    samples = []
    for i in range(0, 1300, 10):
        t = i * MS
        want, _ = plan.at(t)
        target = wpts.translate(want, [plan_offset_m, 0, 0])
        measured = wpts.rotate_about_tcp_z(wpts.translate(target, [track_offset_m, 0, 0]), rot_track)
        mode = "running" if i < 1000 else end
        samples.append(Sample(t, state(mode, t_ns=t, p=measured, target=target)))
    return st.evaluate_follow("rotate" if rotation else "move", rotation, sent, plan, samples, 0, 1000 * MS)


def test_follow_pass_warn_fail():
    assert follow_case().status == "pass"
    assert follow_case(track_offset_m=0.005).status == "warn"
    assert follow_case(track_offset_m=0.011).status == "fail"
    assert follow_case(plan_offset_m=0.001).status == "fail"     # the target is not the sent path
    assert follow_case(rejected=True).status == "fail"
    assert follow_case(end="running").status == "fail"           # never back to holding
    assert follow_case(rotation=True).status == "pass"
    assert follow_case(rotation=True, rot_track=math.radians(2)).status == "warn"
    assert follow_case(rotation=True, rot_track=math.radians(4)).status == "fail"


def test_follow_counts_every_waypoint_in_the_reply():
    start = pose(0.45, 0, 0.45)
    pts = [wpts.PlanPoint(100 * MS, start, 0.08), wpts.PlanPoint(150 * MS, start, 0.08)]
    short = SetWaypointsReply(chunk_id=1, accepted=1, shortened=0, skipped=0, rejected=())
    assert st.check_replies([Sent(0, 0, pts, short)])[0][0] == "fail"
    shortened = SetWaypointsReply(chunk_id=1, accepted=2, shortened=1, skipped=0, rejected=())
    assert "shortened" in st.check_replies([Sent(0, 0, pts, shortened)])[0][1]
    err = ErrorReply(request="c3.set_waypoints/v1", error="refused", field="")
    assert st.check_replies([Sent(0, 0, pts, err)])[0][0] == "fail"


def test_gripper_pass_and_warn_never_fail_on_the_offset():
    samples = [Sample(0, state("running", width=0.062))]
    assert st.evaluate_gripper([], samples, 0.06, 10).status == "pass"
    samples = [Sample(0, state("running", width=0.07))]
    r = st.evaluate_gripper([], samples, 0.06, 10)
    assert r.status == "warn" and "width_offset_m" in r.detail


def test_watchdog_pass_and_fail():
    p = pose(0.45, 0, 0.45)
    ok = [Sample(t * MS, state("holding" if t < 1010 else "stopped", p=p)) for t in range(0, 1300, 10)]
    restart = [Sample(1400 * MS, state("holding", p=p))]
    assert st.evaluate_watchdog(ok, 0, 1.0, Ack(ok=True, reason=""), restart, 1400 * MS).status == "pass"
    late = [Sample(t * MS, state("holding" if t < 1100 else "stopped", p=p)) for t in range(0, 1300, 10)]
    assert st.evaluate_watchdog(late, 0, 1.0, Ack(ok=True, reason=""), restart, 1400 * MS).status == "fail"
    early = [Sample(t * MS, state("holding" if t < 900 else "stopped", p=p)) for t in range(0, 1300, 10)]
    assert st.evaluate_watchdog(early, 0, 1.0, Ack(ok=True, reason=""), restart, 1400 * MS).status == "fail"
    never = [Sample(t * MS, state("holding", p=p)) for t in range(0, 1300, 10)]
    assert st.evaluate_watchdog(never, 0, 1.0, Ack(ok=True, reason=""), restart, 1400 * MS).status == "fail"


def test_out_of_box_pass_and_fail():
    p = pose(0.45, 0, 0.45)
    rejected = SetWaypointsReply(chunk_id=900, accepted=0, shortened=0, skipped=0,
                                 rejected=(WaypointIssue(index=0, reason="outside workspace box by 0.030 m on z"),))
    fault = [Sample(0, state("fault", p=p, error="safety gate rejected waypoint 0 of chunk 900: outside"))]
    assert st.evaluate_out_of_box(rejected, fault).status == "pass"
    assert st.evaluate_out_of_box(rejected, [Sample(0, state("holding", p=p))]).status == "fail"
    accepted = SetWaypointsReply(chunk_id=900, accepted=1, shortened=0, skipped=0, rejected=())
    r = st.evaluate_out_of_box(accepted, fault)
    assert r.status == "fail" and r.detail.startswith("FAIL: THE GATE ACCEPTED")
