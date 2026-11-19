# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/manager/test_heartbeat_sender.py @ 2ffe641
# changes: imports (teleop_helpers); the reboot hint names teleop_collect run
"""HeartbeatSender (STRUCTURE.md 8.2, 12.2): one C3 request per beat; a hello every hello_period_s in place of the
heartbeat, every cycle in start-up mode; samples to the clock map; round trips to LinkStats; a hello t_pc_ns going
back means the PC rebooted."""
from __future__ import annotations

import time

import pytest
from deploy_contracts import Ack, decode
from teleop_helpers import FakeClock, FakePcClient, RawServer, hello_reply

from teleop_collect.teleop.errors import PcRestarted
from teleop_collect.teleop.failure_monitor import FailureMonitor
from teleop_collect.teleop.heartbeat_sender import HeartbeatSender
from teleop_collect.teleop.link_stats import LinkStats
from teleop_collect.teleop.pc_client import PcClient
from teleop_collect.teleop.pc_clock import RoundTripPcClock
from teleop_collect.teleop.problems import ProblemCode

MS = 1_000_000
CYCLE_S = 0.02


@pytest.fixture
def rig(settings):
    clock = FakeClock()
    pc = FakePcClient(clock)
    pc_clock = RoundTripPcClock(settings.clock, clock)
    stats = LinkStats(10.0, clock)
    hb = HeartbeatSender(pc, pc_clock, stats, settings.clock.hello_period_s, settings.clock.startup_hello_period_s,
                         clock)
    return clock, pc, pc_clock, stats, hb


def run(clock, hb, cycles):
    for _ in range(cycles):
        hb.beat(clock())
        clock.advance(CYCLE_S)


def test_one_request_per_beat(rig):
    clock, pc, _, _, hb = rig
    run(clock, hb, 100)
    assert len(pc.calls) == 100 and set(pc.calls) <= {"hello", "heartbeat"}


def test_a_hello_every_hello_period_in_place_of_the_heartbeat(rig, settings):
    clock, pc, _, _, hb = rig
    run(clock, hb, 100)                                               # 2 s at 50 Hz
    assert pc.calls.count("hello") == 4 and pc.calls[0] == "hello"   # every 0.5 s
    gaps = [i for i, c in enumerate(pc.calls) if c == "hello"]
    assert all(b - a == round(settings.clock.hello_period_s / CYCLE_S) for a, b in zip(gaps, gaps[1:]))


def test_every_cycle_in_startup_mode(rig):
    clock, pc, _, _, hb = rig
    hb.startup_mode(True)
    assert hb.in_startup_mode()
    run(clock, hb, 50)
    assert pc.calls == ["hello"] * 50
    hb.startup_mode(False)
    run(clock, hb, 50)
    assert pc.calls[50:].count("hello") <= 2


def test_samples_reach_the_clock_and_the_reply_is_kept(rig):
    clock, pc, pc_clock, _, hb = rig
    hb.startup_mode(True)
    reply = hb.beat(clock())
    assert reply is not None and hb.last_hello() is reply and hb.last_request == "c3.hello/v1"
    assert len(pc_clock.hellos()) == 1 and pc_clock.hellos()[0] == hb.last_sample()
    clock.advance(CYCLE_S)
    hb.startup_mode(False)
    assert hb.beat(clock()) is None and hb.last_request == "c3.heartbeat/v1"
    run(clock, hb, 120)
    assert pc_clock.ready(clock()) is False                           # 2.4 s at 2 Hz: only 6 samples
    hb.startup_mode(True)
    run(clock, hb, 101)
    assert pc_clock.ready(clock())


def test_a_hello_t_pc_ns_going_back_is_a_pc_reboot(rig):
    clock, pc, _, _, hb = rig
    hb.startup_mode(True)
    run(clock, hb, 3)
    pc.offset_ns -= 10 * 1_000_000_000                                # the PC's CLOCK_MONOTONIC restarted
    with pytest.raises(PcRestarted) as e:
        hb.beat(clock())
    assert e.value.rebooted
    v = FailureMonitor(2.0).classify(FailureMonitor(2.0).from_exception(e.value, "PC", "c3.hello/v1"))
    assert v.problem.code is ProblemCode.PC_RESTARTED
    assert f"{v.what_failed}: {v.what_to_check}" == ("the PC rebooted (its clock went back): start control_deploy "
                                                     "run on the PC, then teleop_collect run")


def test_round_trips_reach_link_stats_and_hello_uses_no_poll(settings, monkeypatch):
    def answer(data):
        req = decode(data)
        if req.SCHEMA == "c3.hello/v1":
            return hello_reply(time.monotonic_ns() + 10 ** 12)
        return Ack(ok=True, reason="")
    srv = RawServer(answer)
    try:
        stats = LinkStats(10.0)
        pc = PcClient(srv.endpoint, 0.3, 90.0, 0.02, stats)
        seen = []
        original = pc._rpc.request

        def spy(msg, expect, timeout_s, poll_s=None, cancel=None):
            seen.append((msg.SCHEMA, poll_s, cancel))
            return original(msg, expect, timeout_s, poll_s, cancel)
        monkeypatch.setattr(pc._rpc, "request", spy)
        hb = HeartbeatSender(pc, RoundTripPcClock(settings.clock), stats, 0.5, 0.02)
        for _ in range(3):
            hb.beat(time.monotonic_ns())
        snap = stats.snapshot()
        assert snap.rtt_n == 3 and snap.rtt_max_ms is not None and snap.rtt_max_ms < 300
        assert seen[0] == ("c3.hello/v1", None, None)                 # no sliced poll on the hello
        assert [s[0] for s in seen] == ["c3.hello/v1", "c3.heartbeat/v1", "c3.heartbeat/v1"]
        pc.close()
    finally:
        srv.close()
