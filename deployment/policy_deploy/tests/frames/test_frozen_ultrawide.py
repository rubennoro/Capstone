"""The frozen-ultrawide fix (STRUCTURE.md 4.3 fixes 1-4, 12.2; PLAN 7).

A pair is usable only within [main - 60 ms, main + 25 ms] (mid-exposures) and when the same ultrawide frame was the
pair of at most 2 consecutive packets (by packet_seq); ultrawide health counts a frame once, when it is first served as
an accepted pair, at its own capture time (so a pair always refused makes the ultrawide stale, then down); the store
evicts by time on every packet; the fake Pi can repeat its last ultrawide frame.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from frames_helpers import FakeClock, RawDecoder, raw_frame, raw_packet, wait_for

from policy_deploy.contracts import ClockMap, ClockSample, GetFramesRequest, GetHealthRequest, Packet
from policy_deploy.frames._dataproc.imageprep import ImagePrep
from policy_deploy.frames.camera_health import CameraHealth
from policy_deploy.frames.frames_server import FramesServer
from policy_deploy.frames.packet_buffer import PacketBuffer
from policy_deploy.frames.packet_selector import PacketSelector
from policy_deploy.frames.packet_source import PacketSource
from policy_deploy.frames.pair_rule import OK, REUSED, TOO_NEW, TOO_OLD, PairRule
from policy_deploy.frames.receiver import Receiver

MS = 1_000_000
P = 33_333_333
OFFSET = 1_000_000_000_000           # Pi minus laptop
EXP = 2_000_000                      # both cameras: mid = sensor stamp - 1 ms, so gaps are sensor-stamp gaps
LIVE_LEAD_NS = 16_292_500            # PI_PAYLOAD.md 0.6: the ultrawide mid-exposure 16.3 ms after the main one


class ListSource(PacketSource):
    """Packets appended by the test, taken one by one."""

    def __init__(self) -> None:
        self.queue: list[Packet] = []

    def next_packet(self, timeout_s):
        return self.queue.pop(0) if self.queue else None

    def close(self):
        pass


class Rig:
    """A receiver (and a C1 server over the same buffer) fed packet by packet on a fake clock. Captures happen at
    the fake clock's now: t_pi = now + OFFSET."""

    def __init__(self, rig, rule: PairRule | None = None, stale_s=0.35, down_s=2.0):
        self.clock = FakeClock()
        self.map = ClockMap(60.0, 0.3, 10.0, 500.0, 60.0)
        self.map.add_sample(ClockSample(t_send_ns=self.clock(), t_remote_ns=self.clock() + OFFSET,
                                        t_recv_ns=self.clock()))
        self.buffer = PacketBuffer(1.0)
        self.health = {c: CameraHealth(c, stale_s, down_s, 1.0, self.clock) for c in ("main", "ultrawide")}
        self.source = ListSource()
        self.rx = Receiver(self.source, {c: RawDecoder() for c in self.health}, rig, self.map, self.buffer,
                           self.health, 5_000_000, 1.0, self.clock, rule or PairRule(60.0, 25.0, 2))
        self.rig = rig
        self.n = 0
        self.uw_seq = -1
        self.uw_t = None

    def packet(self, uw_offset_ns: int | None, pair_seq: int | None = None, lost: bool = False) -> None:
        """One main frame captured now; with uw_offset_ns a new ultrawide frame captured at now + offset (its pair);
        without, the last ultrawide frame repeats as the pair. lost: the packet never arrives (a packet_seq gap). Then
        the clock advances one period."""
        t_main = self.clock() + OFFSET
        uw = []
        if uw_offset_ns is not None:
            self.uw_seq += 1
            self.uw_t = t_main + uw_offset_ns
            uw = [raw_frame("ultrawide", self.uw_seq, self.uw_t, exposure_ns=EXP)]
        pair = self.uw_seq if pair_seq is None else pair_seq
        if not lost:
            self.source.queue.append(raw_packet(raw_frame("main", self.n, t_main, exposure_ns=EXP), uw, pair))
            assert self.rx.receive_one(0.0)
        self.n += 1
        self.clock.advance(P)

    def newest(self):
        return self.buffer.newest()

    def served_uw(self) -> bool:
        return self.newest().ultrawide_rgb is not None


@pytest.fixture
def rig(small_rig):
    return Rig(small_rig)


# ---------------------------------------------------------------------------------------------- the reuse rule

@pytest.mark.parametrize("phase_ns, gaps_ms", [
    (+LIVE_LEAD_NS, (-16.2925, 17.0408, 50.3742)),        # the live phase: the ultrawide 16.3 ms newer
    (-17 * MS, (17.0, 50.3333, 83.6667)),                  # the opposite phase: the ultrawide 17 ms older
])
def test_one_repeat_served_the_second_refused_at_both_phases(rig, phase_ns, gaps_ms):
    rig.packet(phase_ns)                                   # a new frame: its first packet
    assert rig.served_uw() and rig.newest().pair_gap_ns / 1e6 == pytest.approx(gaps_ms[0], abs=1e-3)
    rig.packet(None)                                       # one repeat: the 2nd consecutive packet with it
    assert rig.served_uw() and rig.newest().pair_gap_ns / 1e6 == pytest.approx(gaps_ms[1], abs=1e-3)
    rig.packet(None)                                       # a second repeat: the 3rd consecutive packet
    assert not rig.served_uw() and rig.newest().main_rgb is not None
    assert rig.rx.packets.pair_reused == 1 and rig.rx.packets.uw_reused == 2
    assert rig.rx.packets.pair_too_old == 0                # refused by the reuse rule, whatever the timing
    rig.packet(None)                                       # and every further one until the pair changes
    assert not rig.served_uw() and rig.rx.packets.pair_reused == 2
    rig.packet(phase_ns)                                   # a new pair_seq: the run resets
    assert rig.served_uw() and rig.rx.pair_rule.run == 1


def test_at_the_live_phase_the_third_packets_gap_is_inside_the_window():
    """So only the reuse rule refuses it (STRUCTURE.md 4.3 step 10)."""
    rule = PairRule(60.0, 25.0, 1000)
    main0 = 1_000 * MS
    assert rule.check(7, 0, main0, main0 + LIVE_LEAD_NS) == OK
    assert rule.check(7, 2, main0 + 2 * P, main0 + LIVE_LEAD_NS) == OK         # +50.4 ms: inside [-25, +60]
    strict = PairRule(60.0, 25.0, 2)
    assert [strict.check(7, k, main0 + k * P, main0 + LIVE_LEAD_NS) for k in range(4)] == [OK, OK, REUSED, REUSED]


def test_run_count_resets_on_a_new_pair_and_on_a_restart():
    rule = PairRule(60.0, 25.0, 2)
    assert [rule.note(5, k) for k in range(3)] == [1, 2, 3]
    assert rule.note(6, 3) == 1 and rule.note(6, 4) == 2
    rule.reset()
    assert rule.run == 0 and rule.note(6, 0) == 1


def test_the_run_counts_packet_seq_so_a_lost_packet_lets_no_second_repeat_through():
    rule = PairRule(60.0, 25.0, 2)
    assert rule.note(5, 10) == 1
    assert rule.note(5, 12) == 3                      # packet 11 lost: it was the first repeat
    assert rule.note(6, 14) == 1                      # a new pair after another loss: a new run, not inflated


def test_after_a_lost_packet_the_second_repeat_is_still_refused(rig):
    rig.packet(-5 * MS)
    rig.packet(None, lost=True)                       # the first repeat is lost on the way
    rig.packet(None)                                  # the second repeat: the 3rd packet_seq with this pair
    assert not rig.served_uw() and rig.rx.packets.pair_reused == 1 and rig.rx.packets.lost == 1


def test_a_pi_restart_resets_the_pair_rule(rig):
    rig.packet(-5 * MS)
    rig.packet(None)
    assert rig.rx.pair_rule.run == 2
    rig.source.queue.append(raw_packet(raw_frame("main", 0, rig.clock() + OFFSET, exposure_ns=EXP),
                                       [raw_frame("ultrawide", 0, rig.clock() + OFFSET, exposure_ns=EXP)], 0,
                                       packet_seq=0))
    assert rig.rx.receive_one(0.0)
    assert rig.rx.pair_rule.run == 0 and rig.health["ultrawide"].status() == "down"


# ---------------------------------------------------------------------------------------------- the window

def test_window_edges_accepted_and_one_ns_beyond_refused():
    rule = PairRule(60.0, 25.0, 1000)
    main = 10_000 * MS
    assert rule.check(1, 1, main, main - 60 * MS) == OK             # +60 ms exactly (the pair older)
    assert rule.check(2, 2, main, main + 25 * MS) == OK             # -25 ms exactly (the pair newer)
    assert rule.check(3, 3, main, main - 60 * MS - 1) == TOO_OLD
    assert rule.check(4, 4, main, main + 25 * MS + 1) == TOO_NEW


@pytest.mark.parametrize("offset_ms, counter, served", [
    (+30, "pair_too_new", False),                           # newer than pair_max_lead_ms (25)
    (-70, "pair_too_old", False),                           # older than pair_max_age_ms (60)
    (-60, None, True),                                      # exactly 60 ms older: accepted
    (+25, None, True),                                      # exactly 25 ms newer: accepted
])
def test_too_new_and_too_old_pairs_are_refused_and_counted(small_rig, offset_ms, counter, served):
    r = Rig(small_rig)
    r.packet(offset_ms * MS)
    assert r.served_uw() is served
    if counter:
        assert getattr(r.rx.packets, counter) == 1
    else:
        assert r.newest().pair_gap_ns == -offset_ms * MS and r.rx.counters["ultrawide"].usable == 1


def test_the_window_settings_are_used(small_rig):
    r = Rig(small_rig, PairRule(10.0, 5.0, 2))
    r.packet(-12 * MS)
    assert r.rx.packets.pair_too_old == 1
    r.packet(+6 * MS)
    assert r.rx.packets.pair_too_new == 1
    r.packet(-9 * MS)
    assert r.served_uw()


# ---------------------------------------------------------------------------------------------- health (fix 2)

def test_ultrawide_health_ages_from_the_frames_own_capture_time(rig):
    rig.packet(-30 * MS)                                    # the ultrawide captured 30 ms before the main frame
    now = rig.clock()
    main_age = rig.health["main"].report(now).last_frame_age_ms
    uw_age = rig.health["ultrawide"].report(now).last_frame_age_ms
    assert uw_age - main_age == pytest.approx(30.0, abs=1e-3)   # not the packet time
    assert rig.newest().t_ns == rig.clock() - P - EXP // 2       # the packet time: the main frame's mid-exposure


def test_a_pair_always_refused_for_timing_makes_the_ultrawide_stale_then_down(rig):
    """PLAN 7 "slow, then conclude": new ultrawide frames keep coming, but from some point each pairs 30 ms ahead
    (pair_too_new). Their arrival must not keep the ultrawide ok."""
    server = _server(rig)
    try:
        for _ in range(6):
            rig.packet(-5 * MS)
        assert rig.health["ultrawide"].status() == "ok"
        t0 = rig.clock()
        while rig.clock() - t0 < 400 * MS:
            rig.packet(+30 * MS)
        assert rig.rx.packets.pair_too_new > 0
        assert rig.health["ultrawide"].status() == "stale" and rig.health["main"].status() == "ok"
        assert server.get_health(GetHealthRequest()).health.ultrawide.status == "stale"
        while rig.clock() - t0 < 2_100 * MS:
            rig.packet(+30 * MS)
        assert rig.health["ultrawide"].status() == "down" and rig.health["main"].status() == "ok"
        rig.packet(0)                                     # in the window again: served and counted
        assert rig.served_uw() and rig.health["ultrawide"].status() == "ok"
    finally:
        server.close()


def test_a_repeat_records_no_ultrawide_frame(rig):
    rig.packet(-5 * MS)
    t_uw = rig.health["ultrawide"].report(rig.clock()).last_frame_age_ms
    rig.packet(None)
    assert rig.health["ultrawide"].report(rig.clock()).last_frame_age_ms == pytest.approx(t_uw + P / 1e6, abs=1e-3)
    assert rig.health["main"].report(rig.clock()).last_frame_age_ms < t_uw + P / 1e6


def _server(r: Rig) -> FramesServer:
    sel = PacketSelector({c: r.rig.cameras[c].fps for c in ("main", "ultrawide")},
                         {"main": 50 * MS, "ultrawide": 100 * MS}, 2, 2.0)
    return FramesServer("tcp://127.0.0.1:*", r.buffer, r.health, sel, ImagePrep(r.rig), r.clock)


def test_a_repeat_of_0_4_s_makes_the_ultrawide_stale_and_steps_get_no_images(rig):
    server = _server(rig)
    try:
        for _ in range(6):
            rig.packet(-5 * MS)
        assert server.get_frames(GetFramesRequest()).main is not None
        while rig.clock() - (rig.uw_t - OFFSET) < 400 * MS:  # 0.4 s without a new ultrawide frame
            rig.packet(None)
        assert rig.health["ultrawide"].status() == "stale" and rig.health["main"].status() == "ok"
        reply = server.get_frames(GetFramesRequest())
        assert reply.main is None and reply.health.ultrawide.status == "stale"
        while rig.clock() - (rig.uw_t - OFFSET) < 2_100 * MS:   # 2.1 s
            rig.packet(None)
        assert rig.health["ultrawide"].status() == "down" and rig.health["main"].status() == "ok"
        rig.packet(-5 * MS)                                      # the IMX219 comes back
        for _ in range(4):
            rig.packet(-5 * MS)
        assert rig.health["ultrawide"].status() == "ok"
        assert server.get_frames(GetFramesRequest()).main is not None
    finally:
        server.close()


# ---------------------------------------------------------------------------------------------- eviction (fix 3)

def test_the_store_evicts_by_time_on_every_packet(small_rig):
    r = Rig(small_rig, PairRule(10_000.0, 25.0, 10_000))   # neither the window nor the reuse rule in the way
    r.packet(-5 * MS)
    for _ in range(int(0.9e9 / P)):                         # repeats for 0.9 s: still found
        r.packet(None)
    assert r.served_uw() and r.rx.counters["ultrawide"].pair_missing == 0
    for _ in range(4):                                      # beyond buffer.keep_s (1 s): evicted, not found
        r.packet(None)
    assert not r.served_uw() and r.rx.counters["ultrawide"].pair_missing >= 1
    assert len(r.rx._store) == 0


# ---------------------------------------------------------------------------------------------- the fake Pi (fix 4)

def test_fake_pi_repeat_mode_through_the_whole_process(running):
    run = running(health={"stale_s": 0.35, "down_s": 2.0})
    run.wait_ready()
    run.wait_frames()
    rx, h = run.app.receiver, run.app.health["ultrawide"]
    run.app.fake_pi.repeat_ultrawide(0.4)                  # 0.4 s without a new ultrawide frame
    seen = set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not ("stale" in seen and seen >= {"stale"} and h.status() == "ok"):
        seen.add(h.status())
        time.sleep(0.002)
    assert "stale" in seen and "down" not in seen
    assert rx.packets.pair_reused >= 1 and rx.packets.uw_reused >= 3 and run.app.fake_pi.sent["reused"] >= 3
    run.wait_frames()                                       # serving resumes by itself
    run.app.fake_pi.repeat_ultrawide(2.1)
    wait_for(lambda: h.status() == "down", 5, "ultrawide down")
    assert run.app.health["main"].status() == "ok"
    assert run.frames().main is None
    wait_for(lambda: h.status() == "ok", 5, "ultrawide ok again")
    run.wait_frames()
