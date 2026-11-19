# copied from rnoro/final-deployment:deployment/policy_deploy/tests/fake_pc/test_fake_pc.py @ 2ffe641
# changes: imports only (the helpers are in launcher_helpers.py)
"""FakePc (STRUCTURE.md 10.1, 12.2): modes and replies exactly as TOP 3.7.2, the PC keyboard hooks, hello on the
drifting clock, the watchdog, C2 seq/t_ns, the follower and the gate. Fix wave: the real franka_control's refusal
texts, an all-skipped set freezes and goes holding, t_ns freezes only when the loop ends."""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from deploy_contracts import (SHARED_SCHEMAS, Ack, ArmState, ErrorReply, GetState, Heartbeat, Hello, HelloReply,
                              Home, LinkSettings, RpcClient, SetWaypoints, SetWaypointsReply, Start, Stop, TcpFrame,
                              TransportTimeout, compare_hello, decode, encode)
from launcher_helpers import FakeLaptopClock, make_fake_pc, wait_for

NS = 1_000_000_000
MS = 1_000_000


class Recorder:
    """Stands in for the C2 Publisher: keeps every published state."""

    def __init__(self) -> None:
        self.states: list[ArmState] = []

    def publish(self, msg) -> None:
        self.states.append(msg)

    def close(self) -> None:
        pass


@pytest.fixture
def clock():
    return FakeLaptopClock()


def manual(tmp_path, clock, **overrides):
    """A FakePc without threads or sockets, on a fake laptop clock; C2 goes to a Recorder."""
    fake = make_fake_pc(tmp_path, laptop_clock=clock, **overrides)
    fake._pub = Recorder()
    return fake


def run(fake, clock, seconds, step=0.01):
    """Advance the fake laptop clock, ticking the C2 thread every `step`."""
    for _ in range(int(round(seconds / step))):
        clock.advance(step)
        fake.tick()


def waypoints(fake, offsets_ms, dx=0.0, width=0.04, chunk_id=1):
    now = fake.clock()
    T = fake.target
    poses = []
    for i, _ in enumerate(offsets_ms):
        P = T.copy()
        P[0, 3] += dx * (i + 1)
        poses.append(P)
    return SetWaypoints(chunk_id=chunk_id, t_ns=np.array([now + int(o * MS) for o in offsets_ms], dtype=np.int64),
                        O_T_TCP=np.array(poses), gripper_width_m=np.full(len(offsets_ms), width))


def roundtrip(reply):
    """The reply passes the shared validators: it re-decodes to the same schema."""
    back = decode(encode(reply))
    assert back.SCHEMA == reply.SCHEMA
    return back


# ---------------------------------------------------------------------------------------------- the mode table

def enter(fake, clock, mode):
    if mode == "homing":
        assert fake.keyboard_home().ok
    elif mode == "fault":
        fake.inject_fault("reflex")
    elif mode != "idle":
        fake.set_mode(mode)
    assert fake.mode == mode


REQUESTS = {
    "hello": (Hello(), {m: ("reply", m) for m in ("idle", "holding", "running", "stopped", "homing", "fault")}),
    "get_state": (GetState(), {m: ("reply", m) for m in ("idle", "holding", "running", "stopped", "homing", "fault")}),
    "heartbeat": (Heartbeat(), {m: ("ok", m) for m in ("idle", "holding", "running", "stopped", "homing", "fault")}),
    "start": (Start(), {"idle": ("ok", "holding"), "holding": ("refused", "holding"),
                        "running": ("refused", "running"), "stopped": ("ok", "holding"),
                        "homing": ("refused", "homing"), "fault": ("refused", "fault")}),
    "stop": (Stop(), {"idle": ("ok", "stopped"), "holding": ("ok", "stopped"), "running": ("ok", "stopped"),
                      "stopped": ("ok", "stopped"), "homing": ("ok", "stopped"), "fault": ("ok", "fault")}),
    "home": (Home(), {"idle": ("ok", "idle"), "holding": ("refused", "holding"), "running": ("refused", "running"),
                      "stopped": ("ok", "idle"), "homing": ("refused", "homing"), "fault": ("refused", "fault")}),
}
CELLS = [(req, mode) for req, (_, table) in REQUESTS.items() for mode in table]
# the real franka_control's refusal texts (StateMachine::check); the fault detail is enter()'s "reflex"
REFUSALS = {("start", "holding"): "refused: mode is holding: already started",
            ("start", "running"): "refused: mode is running: already started",
            ("start", "homing"): "refused: mode is homing: homing in progress",
            ("start", "fault"): "refused: mode is fault: reflex",
            ("home", "holding"): "refused: mode is holding: send stop first",
            ("home", "running"): "refused: mode is running: send stop first",
            ("home", "homing"): "refused: mode is homing: homing in progress",
            ("home", "fault"): "refused: mode is fault: reflex"}


@pytest.mark.parametrize("request_name,mode", CELLS, ids=[f"{r}-{m}" for r, m in CELLS])
def test_replies_per_mode(tmp_path, request_name, mode):
    clock = FakeLaptopClock()
    fake = manual(tmp_path, clock, home_duration_s=0.05)
    enter(fake, clock, mode)
    msg, table = REQUESTS[request_name]
    want, after = table[mode]
    if request_name == "home" and want == "ok":
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("r", fake.handle(msg)))
        t.start()
        assert wait_for(lambda: fake.mode == "homing", 2.0)
        clock.advance(0.06)                                  # the ramp ends; the pending home answers
        t.join(timeout=5.0)
        reply = result["r"]
    else:
        reply = fake.handle(msg)
    roundtrip(reply)
    if want == "reply":
        assert isinstance(reply, HelloReply if request_name == "hello" else ArmState)
        if request_name == "get_state":
            assert reply.mode == mode
    elif want == "ok":
        assert isinstance(reply, Ack) and reply.ok and reply.reason == ""
    else:
        assert isinstance(reply, Ack) and not reply.ok and reply.reason == REFUSALS[(request_name, mode)]
    assert fake.mode == after


@pytest.mark.parametrize("mode", ["idle", "holding", "running", "stopped", "homing", "fault"])
def test_set_waypoints_per_mode(tmp_path, clock, mode):
    fake = manual(tmp_path, clock)
    enter(fake, clock, mode)
    reply = roundtrip(fake.handle(waypoints(fake, [100, 200])))
    if mode in ("holding", "running"):
        assert isinstance(reply, SetWaypointsReply) and reply.accepted == 2 and fake.mode == "running"
    else:
        assert isinstance(reply, ErrorReply) and reply.field == ""
        assert reply.error == {"idle": "set_waypoints refused: mode is idle: send start first",
                               "stopped": "set_waypoints refused: mode is stopped: send start first",
                               "homing": "set_waypoints refused: mode is homing: homing in progress",
                               "fault": "set_waypoints refused: mode is fault: reflex"}[mode]
        assert fake.mode == mode


def test_a_stopped_mode_names_its_cause_in_refusals(tmp_path, clock):
    fake = manual(tmp_path, clock, home_duration_s=1.0)
    assert fake.keyboard_home().ok
    fake.keyboard_stop()                                     # the PC's s during homing
    reply = fake.handle(waypoints(fake, [100]))
    assert reply.error == "set_waypoints refused: mode is stopped (stop during homing (PC keyboard)): send start first"
    assert fake.handle(Start()).ok
    fake.handle(Stop())
    assert fake.handle(waypoints(fake, [100])).error == "set_waypoints refused: mode is stopped: send start first"


# ---------------------------------------------------------------------------------------------- hello

def test_hello_fields_on_the_drifting_clock(tmp_path, clock):
    fake = manual(tmp_path, clock)
    reply = fake.handle(Hello())
    expected = int(round(clock.now * (1 + 30e-6))) + 1000 * NS
    assert reply.t_pc_ns == expected == fake.clock.at(clock.now)
    assert reply.mock is True and tuple(reply.schemas) == SHARED_SCHEMAS
    assert np.array_equal(reply.F_T_TCP, TcpFrame.load().F_T_TCP)
    link = LinkSettings.load()
    assert reply.watchdog_limit_s == link.watchdog_limit_s
    assert (reply.homing_timeout_s, reply.max_lead_s) == (30.0, 1.5)
    assert compare_hello(reply, mock=True) == []
    clock.advance(1.0)
    assert fake.handle(Hello()).t_pc_ns - reply.t_pc_ns == int(round(NS * (1 + 30e-6)))


def test_hello_commit_from_git_or_unknown(tmp_path):
    from teleop_collect.fake_pc.fake_pc import laptop_commit
    c = laptop_commit()
    assert c == "unknown" or (len(c) == 40 and all(x in "0123456789abcdef" for x in c))


def test_shift_clock_moves_hello_and_c2(tmp_path, clock):
    fake = manual(tmp_path, clock)
    a = fake.handle(Hello()).t_pc_ns
    fake.shift_clock(2.0)
    assert fake.handle(Hello()).t_pc_ns - a == 2 * NS


# ---------------------------------------------------------------------------------------------- the PC keyboard

def test_c3_stop_during_keyboard_home_stops_where_the_ramp_is(tmp_path, clock):
    fake = manual(tmp_path, clock, home_duration_s=1.0)
    start = fake.target
    assert fake.keyboard_home().ok and fake.mode == "homing"
    run(fake, clock, 0.5)
    mid = fake.target
    assert not np.allclose(mid, start)
    assert fake.handle(Stop()).ok
    assert fake.mode == "stopped"
    run(fake, clock, 1.0)
    assert np.allclose(fake.target, mid)


def test_c3_stop_after_a_finished_ramp_ends_idle(tmp_path, clock):
    fake = manual(tmp_path, clock, home_duration_s=1.0)
    assert fake.keyboard_home().ok
    clock.advance(1.5)                                       # the ramp ended; the C2 thread has not noticed yet
    assert fake.mode == "homing"
    assert fake.handle(Stop()).ok
    assert fake.mode == "idle"                               # the normal end wins
    assert np.allclose(fake.target, fake.s.home_pose)


def test_keyboard_home_ramps_to_home_and_rep_keeps_answering(tmp_path, clock):
    fake = manual(tmp_path, clock, home_duration_s=1.0)
    fake.set_mode("holding")
    assert fake.keyboard_home().reason == "refused: mode is holding: send stop first"
    fake.set_mode("idle")
    assert fake.keyboard_home().ok
    assert fake.handle(Start()).reason == "refused: mode is homing: homing in progress"
    assert fake.handle(Home()).reason == "refused: mode is homing: homing in progress"
    assert fake.handle(Heartbeat()).ok
    run(fake, clock, 1.1)
    assert fake.mode == "idle" and np.allclose(fake.target, fake.s.home_pose)


def test_keyboard_hand_home_width_to_zero_and_back(tmp_path, clock):
    fake = manual(tmp_path, clock, start_width_m=0.04, width_speed_m_s=0.05)
    fake.set_mode("stopped")
    T = fake.target
    assert fake.keyboard_hand_home().ok and fake.mode == "homing"
    assert fake.handle(GetState()).mode == "homing"           # the REP keeps answering
    widths = []
    for _ in range(180):
        clock.advance(0.01)
        fake.tick()
        widths.append(fake.width_m)
    assert min(widths) == pytest.approx(0.0, abs=1e-9)
    assert fake.width_m == pytest.approx(0.04) and fake.mode == "idle"
    assert np.allclose(fake.target, T)                       # the arm target held


def test_keyboard_stop(tmp_path, clock):
    fake = manual(tmp_path, clock)
    for mode, after in (("idle", "stopped"), ("holding", "stopped"), ("running", "stopped"), ("stopped", "stopped")):
        fake.set_mode(mode)
        assert fake.keyboard_stop().ok and fake.mode == after
    fake.inject_fault("x")
    fake.keyboard_stop()
    assert fake.mode == "fault"
    assert fake.keyboard_home().reason == "refused: mode is fault: x"


def test_keyboard_stop_during_a_pending_c3_home_replies_stopped(tmp_path):
    fake = make_fake_pc(tmp_path, home_duration_s=5.0)
    fake.start()
    client = RpcClient("PC", fake.commands_endpoint)
    try:
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("r", client.request(Home(), Ack, 10.0)))
        t.start()
        assert wait_for(lambda: fake.mode == "homing", 3.0)
        assert fake.keyboard_stop().ok
        t.join(timeout=5.0)
        reply = result["r"]
        assert not reply.ok and reply.reason.startswith("stopped: ")
        assert fake.mode == "stopped"
    finally:
        client.close()
        fake.close()


def test_c3_home_blocks_other_requests(tmp_path):
    fake = make_fake_pc(tmp_path, home_duration_s=0.6)
    fake.start()
    a, b = RpcClient("PC", fake.commands_endpoint), RpcClient("PC", fake.commands_endpoint)
    try:
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("r", a.request(Home(), Ack, 5.0)))
        t.start()
        assert wait_for(lambda: fake.mode == "homing", 3.0)
        with pytest.raises(TransportTimeout):
            b.request(Heartbeat(), Ack, 0.15)
        t.join(timeout=5.0)
        assert result["r"].ok and fake.mode == "idle"
        assert b.request(Heartbeat(), Ack, 1.0).ok
    finally:
        a.close()
        b.close()
        fake.close()


# ---------------------------------------------------------------------------------------------- the watchdog

def test_watchdog_arms_only_in_holding_and_running(tmp_path, clock):
    fake = manual(tmp_path, clock)
    limit = fake.s.watchdog_limit_s
    run(fake, clock, limit + 0.2)
    assert fake.mode == "idle"                              # not armed in idle
    fake.handle(Start())
    run(fake, clock, limit - 0.2)
    fake.handle(Heartbeat())                                 # fed
    run(fake, clock, limit - 0.2)
    assert fake.mode == "holding"
    run(fake, clock, 0.4)
    assert fake.mode == "stopped"                            # tripped: stopped, not fault


def test_watchdog_paused_while_homing(tmp_path, clock):
    fake = manual(tmp_path, clock, home_duration_s=3.0)
    assert fake.keyboard_home().ok
    run(fake, clock, 2.0)
    assert fake.mode == "homing"
    run(fake, clock, 1.1)
    assert fake.mode == "idle"


def test_every_handled_request_feeds_the_watchdog(tmp_path, clock):
    fake = manual(tmp_path, clock)
    fake.handle(Start())
    for _ in range(5):
        run(fake, clock, 0.6)
        fake.handle(Hello())
    assert fake.mode == "holding"


# ---------------------------------------------------------------------------------------------- C2

def test_seq_from_1_plus_1_and_a_gap_on_drop_states(tmp_path, clock):
    fake = manual(tmp_path, clock)
    run(fake, clock, 0.03)
    assert [s.seq for s in fake._pub.states] == [1, 2, 3]
    fake.drop_states(2)
    run(fake, clock, 0.03)
    assert [s.seq for s in fake._pub.states] == [1, 2, 3, 6]
    assert fake.handle(GetState()).seq == 6                  # the latest broadcast's seq, not +1
    fake.restart()
    run(fake, clock, 0.01)
    assert fake._pub.states[-1].seq == 1


def test_t_ns_on_the_drifting_clock_frozen_in_fault(tmp_path, clock):
    fake = manual(tmp_path, clock)
    run(fake, clock, 0.02)
    s = fake._pub.states
    assert s[-1].t_ns == fake.clock.at(clock.now) and s[-1].t_ns > s[-2].t_ns
    fake.inject_fault("reflex: joint 4")
    run(fake, clock, 0.05)
    assert len({x.t_ns for x in s[-5:]}) == 1
    assert s[-1].mode == "fault" and s[-1].error == "reflex: joint 4"


def test_t_ns_keeps_advancing_after_a_gate_rejection(tmp_path, clock):
    """The real loop keeps holding after a gate fault: only a loop end (inject_fault) freezes t_ns."""
    fake = manual(tmp_path, clock)
    fake.handle(Start())
    fake.handle(waypoints(fake, [100, 1600]))                # beyond max_lead_s: rejected, fault
    run(fake, clock, 0.05)
    s = fake._pub.states
    assert s[-1].mode == "fault" and s[-1].t_ns > s[-2].t_ns


def test_repeat_t_ns_keeps_c2_coming(tmp_path, clock):
    fake = manual(tmp_path, clock)
    run(fake, clock, 0.02)
    fake.repeat_t_ns(0.1)
    n = len(fake._pub.states)
    run(fake, clock, 0.05)
    assert len(fake._pub.states) == n + 5 and len({x.t_ns for x in fake._pub.states[-5:]}) == 1
    run(fake, clock, 0.1)
    assert fake._pub.states[-1].t_ns > fake._pub.states[-6].t_ns


def test_stall_and_burst(tmp_path, clock):
    fake = manual(tmp_path, clock)
    run(fake, clock, 0.02)
    fake.burst_after_stall()
    fake.stall(0.1, c3=False)
    n = len(fake._pub.states)
    run(fake, clock, 0.09)
    assert len(fake._pub.states) == n                         # nothing during the stall
    run(fake, clock, 0.03)
    assert len(fake._pub.states) >= n + 10                    # the backlog, all at once
    seqs = [s.seq for s in fake._pub.states]
    assert seqs == sorted(seqs)


# ---------------------------------------------------------------------------------------------- follower and gate

def test_follower_reaches_waypoints_at_their_times_and_holds(tmp_path, clock):
    fake = manual(tmp_path, clock)
    fake.handle(Start())
    msg = waypoints(fake, [100, 200, 300], dx=0.01)
    reply = fake.handle(msg)
    assert (reply.accepted, reply.skipped, reply.rejected) == (3, 0, ())
    assert fake.mode == "running"
    run(fake, clock, 0.2)
    assert fake.target[0, 3] == pytest.approx(msg.O_T_TCP[1][0, 3], abs=2e-4)
    run(fake, clock, 0.15)
    assert np.allclose(fake.target, msg.O_T_TCP[2]) and fake.mode == "holding"
    run(fake, clock, 0.3)
    assert np.allclose(fake.target, msg.O_T_TCP[2])
    states = fake._pub.states
    assert np.allclose(states[-1].O_T_TCP_target, msg.O_T_TCP[2])


def test_past_waypoints_skipped(tmp_path, clock):
    fake = manual(tmp_path, clock)
    fake.handle(Start())
    reply = fake.handle(waypoints(fake, [-100, -50, 100]))
    assert (reply.accepted, reply.skipped) == (1, 2)


def test_an_all_skipped_set_freezes_at_the_current_target_and_holds(tmp_path, clock):
    """As the real PC: the plan becomes [the target now], mode holding (not the rest of the previous plan)."""
    fake = manual(tmp_path, clock)
    fake.handle(Start())
    assert fake.handle(waypoints(fake, [500], dx=0.1)).accepted == 1 and fake.mode == "running"
    run(fake, clock, 0.2)                                    # partway along the 0.5 s move
    reply = fake.handle(waypoints(fake, [-50, -10]))
    assert (reply.accepted, reply.skipped, len(reply.rejected)) == (0, 2, 0)
    assert fake.mode == "holding"
    frozen = fake.target
    assert 0.01 < frozen[0, 3] - fake.s.start_pose[0, 3] < 0.09   # where the target was at that moment
    run(fake, clock, 0.5)
    assert np.allclose(fake.target, frozen) and fake.mode == "holding"


def test_beyond_max_lead_rejected_with_fault(tmp_path, clock):
    fake = manual(tmp_path, clock)
    fake.handle(Start())
    reply = roundtrip(fake.handle(waypoints(fake, [-10, 100, 1600])))
    assert reply.accepted == 0 and reply.skipped == 1
    reasons = {i.index: i.reason for i in reply.rejected}
    assert set(reasons) == {1, 2} and "max_lead_s" in reasons[2] and reasons[1].startswith("not checked")
    assert reply.accepted + reply.skipped + len(reply.rejected) == 3
    assert fake.mode == "fault" and "max_lead_s" in fake.handle(GetState()).error


def test_width_outside_range_rejected(tmp_path, clock):
    fake = manual(tmp_path, clock)
    fake.handle(Start())
    reply = fake.handle(waypoints(fake, [100], width=0.2))
    assert reply.accepted == 0 and "width" in reply.rejected[0].reason and fake.mode == "fault"


def test_measured_offset_and_counters(tmp_path, clock):
    fake = manual(tmp_path, clock, measured_offset_m=0.003)
    fake.report_shortened(2)
    fake.report_pulled_back(1)
    st = fake.handle(GetState())
    assert st.O_T_TCP[0, 3] - st.O_T_TCP_target[0, 3] == pytest.approx(0.003)
    assert (st.gate.shortened, st.gate.pulled_back) == (2, 1) and st.gate.accepted >= 2


def test_commands_log(tmp_path, clock):
    fake = make_fake_pc(tmp_path, laptop_clock=clock, session_dir=tmp_path / "fake_pc")
    fake._pub = Recorder()
    fake.handle(Hello())
    fake.handle(Start())
    fake.handle(waypoints(fake, [100]))
    fake.close()
    rows = [json.loads(x) for x in (tmp_path / "fake_pc" / "commands.jsonl").read_text().splitlines()]
    assert [r["request"] for r in rows] == ["c3.hello/v1", "c3.start/v1", "c3.set_waypoints/v1"]
    assert rows[2]["accepted"] == 1 and rows[1]["ok"] is True


def test_start_mode_and_sockets(tmp_path):
    fake = make_fake_pc(tmp_path, start_mode="holding")
    with fake:
        assert fake.ready.wait(3.0) and fake.mode == "holding"
        client = RpcClient("PC", fake.commands_endpoint)
        try:
            assert client.request(Hello(), HelloReply, 1.0).mock is True
            assert client.request(GetState(), ArmState, 1.0).mode == "holding"
        finally:
            client.close()
        assert fake.request_names()[:2] == ["c3.hello/v1", "c3.get_state/v1"]
