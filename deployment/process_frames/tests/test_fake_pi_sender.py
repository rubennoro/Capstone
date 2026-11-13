"""FakePiSender over 127.0.0.1:*: its C0 packets pass the contract validator and follow the Pi's rules
(PI_PAYLOAD.md §5, STRUCTURE.md §5.16)."""

from __future__ import annotations

import time

import pytest

from contracts import PacketSubscriber, RpcClient, TimeSyncReply, TimeSyncRequest, TransportTimeout

from conftest import wait_for
from process_frames.fake_pi_sender import FakePiSender
from process_frames.packet_pairer import PAIR_AHEAD_NS
from process_frames.pi_recording import PiRecording

OFFSET = 1000 * 1_000_000_000


@pytest.fixture(scope="module")
def recording(pi_episode_no_exposure):
    return PiRecording.load(pi_episode_no_exposure, encoding="h264", max_seconds=10.0, keyframe_interval=4,
                            jpeg_quality=90)


@pytest.fixture
def fake(recording):
    f = FakePiSender(recording, "tcp://127.0.0.1:*", "tcp://127.0.0.1:*", pi_clock_offset_ns=OFFSET, drift_ppm=20.0,
                     send_hwm=30)
    f.start()
    yield f
    f.stop()


class Listener:
    """A contracts PacketSubscriber collecting C0 packets (each validated by Packet.from_parts)."""

    def __init__(self, endpoint):
        self.sub = PacketSubscriber(name="pi", endpoint=endpoint, recv_hwm=1000)
        self.packets = []

    def pump(self, until, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while not until(self):
            assert time.monotonic() < deadline, "timed out"
            p = self.sub.receive(0.05)
            if p is not None:
                self.packets.append(p)

    def seqs(self):
        return [p.header.packet_seq for p in self.packets]

    def uw_seqs(self):
        return [f.info.frame_seq for p in self.packets for f in p.ultrawide]

    def close(self):
        self.sub.close()


@pytest.fixture
def listener(fake):
    lst = Listener(fake.frames_endpoint)
    yield lst
    lst.close()


def gaps(seqs):
    return [b - a - 1 for a, b in zip(seqs, seqs[1:]) if b - a > 1]


def test_endpoints_are_bound(fake):
    assert fake.frames_endpoint.startswith("tcp://127.0.0.1:") and not fake.frames_endpoint.endswith("*")
    assert fake.time_sync_endpoint.startswith("tcp://127.0.0.1:") and not fake.time_sync_endpoint.endswith("*")


def test_packets_pass_the_contract_and_follow_the_pi_rules(fake, listener, recording):
    listener.pump(lambda l: len(l.packets) >= 10)
    pk = listener.packets
    assert all(b == a + 1 for a, b in zip(listener.seqs(), listener.seqs()[1:]))
    uw = listener.uw_seqs()
    assert all(b == a + 1 for a, b in zip(uw, uw[1:]))                     # every ultrawide frame once, in order
    times = {f.info.frame_seq: f.info.t_pi_ns for p in pk for f in p.ultrawide}
    rm, ru = recording.cameras["main"], recording.cameras["ultrawide"]
    for p in pk:
        m = p.header.main
        assert m.frame_seq == p.header.packet_seq                         # both count main frames captured
        assert (m.sensor, m.width, m.height, m.encoding) == (rm.sensor, rm.width, rm.height, "h264")
        assert p.main_data == rm.frames[m.frame_seq % len(rm.frames)].data
        assert m.keyframe == rm.frames[m.frame_seq % len(rm.frames)].info.keyframe
        for f in p.ultrawide:
            assert f.data == ru.frames[f.info.frame_seq % len(ru.frames)].data
        if p.header.pair_seq >= 0 and p.header.pair_seq in times:
            assert -PAIR_AHEAD_NS <= m.t_pi_ns - times[p.header.pair_seq] <= 33_333_333 + PAIR_AHEAD_NS
    assert sum(p.header.pair_seq >= 0 for p in pk) >= len(pk) - 1


def test_drop_packets_makes_gaps_in_every_counter(fake, listener):
    listener.pump(lambda l: len(l.packets) >= 2)
    fake.drop_packets(3)
    listener.pump(lambda l: bool(gaps(l.seqs())) and len(l.packets) >= 8)
    assert gaps(listener.seqs()) == [3]
    assert gaps(listener.uw_seqs()) and sum(gaps(listener.uw_seqs())) >= 2      # their ultrawide frames are gone


def test_drop_frames_main_skips_packets_but_no_ultrawide(fake, listener):
    listener.pump(lambda l: len(l.packets) >= 2)
    fake.drop_frames("main", 2)
    listener.pump(lambda l: bool(gaps(l.seqs())) and len(l.packets) >= 8)
    assert gaps(listener.seqs()) == [2]
    assert gaps(listener.uw_seqs()) == []                    # pending ultrawide frames go in the next packet
    assert 2 in {len(p.ultrawide) for p in listener.packets} or 3 in {len(p.ultrawide) for p in listener.packets}


def test_drop_frames_ultrawide_makes_an_ultrawide_gap(fake, listener):
    listener.pump(lambda l: len(l.packets) >= 2)
    fake.drop_frames("ultrawide", 2)
    listener.pump(lambda l: bool(gaps(l.uw_seqs())) and len(l.packets) >= 8)
    assert gaps(listener.uw_seqs()) == [2] and gaps(listener.seqs()) == []


def test_pause_ultrawide_gives_unpaired_packets_then_resume(fake, listener):
    listener.pump(lambda l: len(l.packets) >= 2)
    fake.pause("ultrawide")
    n = len(listener.packets)
    listener.pump(lambda l: len(l.packets) >= n + 6)
    later = listener.packets[n + 2:]
    assert all(not p.ultrawide for p in later) and all(p.header.pair_seq == -1 for p in later[2:])
    fake.resume("ultrawide")
    listener.pump(lambda l: any(p.ultrawide for p in l.packets[n + 6:]))
    assert gaps(listener.uw_seqs())                         # frames due while paused were never encoded


def test_pause_main_stops_the_packets(fake, listener):
    listener.pump(lambda l: len(l.packets) >= 2)
    fake.pause("main")
    n = len(listener.packets) + 1                           # at most one already on its way
    t_end = time.monotonic() + 0.3                          # 9 frame periods
    listener.pump(lambda l: time.monotonic() > t_end)
    assert len(listener.packets) <= n
    fake.resume("main")
    listener.pump(lambda l: len(l.packets) >= n + 2)


def test_the_clip_loops_with_counters_and_time_still_increasing(fake, listener, recording):
    n = len(recording.cameras["main"].frames)
    listener.pump(lambda l: len(l.packets) >= n + 4, timeout_s=15)
    pk = listener.packets
    assert all(b.header.packet_seq == a.header.packet_seq + 1 and b.header.main.t_pi_ns > a.header.main.t_pi_ns
               for a, b in zip(pk, pk[1:]))
    uw = [f.info for p in pk for f in p.ultrawide]
    assert all(b.frame_seq == a.frame_seq + 1 and b.t_pi_ns > a.t_pi_ns for a, b in zip(uw, uw[1:]))
    looped = [p for p in pk if p.header.packet_seq % n == 0 and p.header.packet_seq > 0]
    assert looped and looped[0].header.main.keyframe


def test_time_sync_follows_pi_clock(fake):
    client = RpcClient(name="pi", endpoint=fake.time_sync_endpoint)
    try:
        before = time.monotonic_ns()
        reply = client.request(TimeSyncRequest(), TimeSyncReply, 2.0)
        after = time.monotonic_ns()
    finally:
        client.close()
    assert fake.pi_clock_ns(before) <= reply.t_pi_ns <= fake.pi_clock_ns(after)
    assert abs(reply.t_pi_ns - before - OFFSET) < 50_000_000


def test_pi_clock_has_offset_and_drift(recording):
    f = FakePiSender(recording, "tcp://127.0.0.1:*", "tcp://127.0.0.1:*", pi_clock_offset_ns=OFFSET, drift_ppm=20.0,
                     send_hwm=10, clock=lambda: 5_000_000_000)
    f._start_ns = 5_000_000_000
    assert f.pi_clock_ns(5_000_000_000) == 5_000_000_000 + OFFSET
    assert f.pi_clock_ns(6_000_000_000) == 6_000_000_000 + OFFSET + 20_000     # 20 ppm = 20 us per second


def test_after_stop_nothing_answers(recording):
    f = FakePiSender(recording, "tcp://127.0.0.1:*", "tcp://127.0.0.1:*", pi_clock_offset_ns=OFFSET, drift_ppm=0.0,
                     send_hwm=10)
    f.start()
    endpoint = f.time_sync_endpoint
    f.stop()
    client = RpcClient(name="pi", endpoint=endpoint)
    try:
        with pytest.raises(TransportTimeout):
            client.request(TimeSyncRequest(), TimeSyncReply, 0.3)
    finally:
        client.close()


def test_a_bind_error_is_raised_by_start(recording, fake):
    other = FakePiSender(recording, fake.frames_endpoint, "tcp://127.0.0.1:*", pi_clock_offset_ns=0, drift_ppm=0.0,
                         send_hwm=10)
    with pytest.raises(Exception):
        other.start()


def test_sent_counts(fake):
    wait_for(lambda: fake.sent["packets"] >= 3, 10, "packets sent")
    assert fake.sent["ultrawide"] >= 1
