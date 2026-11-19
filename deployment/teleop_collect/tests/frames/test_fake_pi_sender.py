# copied from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_fake_pi_sender.py @ 2ffe641
# changes: imports only
# adapted from rnoro/deployment:deployment/process_frames/tests/test_fake_pi_sender.py @ 91b178b
# changes: pi_v1 only (3-part messages read with a raw SUB and PiStreamPacket; no c0 packets, no time_sync); the pair is
# the nearest frame; uw_reused and skew_ns follow the pair; repeat_ultrawide (fix 4); pause("ultrawide") repeats the
# last frame. Fix wave: skew_ns with the live Pi's sign and the live 45.8 ms send delay (PI_PAYLOAD.md 0.6).
"""FakePiSender over 127.0.0.1: its pi_stream/v1 messages pass the contract and pair the nearest frame."""

from __future__ import annotations

import time

import pytest
import zmq
from frames_helpers import wait_for

from teleop_collect.contracts import PiStreamPacket
from teleop_collect.frames.fake_pi_sender import SEND_DELAY_NS, FakePiSender
from teleop_collect.frames.pi_recording import PiRecording

OFFSET = 1000 * 1_000_000_000
P = 33_333_333


@pytest.fixture(scope="module")
def recording(pi_episode):
    return PiRecording.load(pi_episode, max_seconds=10.0, jpeg_quality=90)


@pytest.fixture
def fake(recording):
    f = FakePiSender(recording, "tcp://127.0.0.1:*", pi_clock_offset_ns=OFFSET, drift_ppm=20.0, send_hwm=30)
    f.start()
    yield f
    f.stop()


class Listener:
    """A raw SUB collecting pi_stream/v1 messages (each checked by PiStreamPacket.from_parts)."""

    def __init__(self, endpoint):
        self.sub = zmq.Context.instance().socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.connect(endpoint)
        self.msgs: list[PiStreamPacket] = []

    def pump(self, until, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while not until(self):
            assert time.monotonic() < deadline, "timed out"
            if self.sub.poll(50):
                self.msgs.append(PiStreamPacket.from_parts(self.sub.recv_multipart()))

    def nos(self):
        return [m.header.packet_no for m in self.msgs]

    def pairs(self):
        return [m.header.ultrawide.frame_no for m in self.msgs]

    def close(self):
        self.sub.close()


@pytest.fixture
def listener(fake):
    lst = Listener(fake.frames_endpoint)
    yield lst
    lst.close()


def gaps(seqs):
    return [b - a - 1 for a, b in zip(seqs, seqs[1:]) if b - a > 1]


def test_endpoint_is_bound(fake):
    assert fake.frames_endpoint.startswith("tcp://127.0.0.1:") and not fake.frames_endpoint.endswith("*")


def test_messages_pass_the_contract_and_pair_the_nearest_frame(fake, listener, recording):
    listener.pump(lambda l: len(l.msgs) >= 10)
    rm, ru = recording.cameras["main"], recording.cameras["ultrawide"]
    assert all(b == a + 1 for a, b in zip(listener.nos(), listener.nos()[1:]))
    for m in listener.msgs:
        h = m.header
        assert h.main.frame_no == h.packet_no                           # both count main frames captured
        assert m.main_jpeg == rm.frames[h.main.frame_no % len(rm.frames)].data
        assert m.ultrawide_jpeg == ru.frames[h.ultrawide.frame_no % len(ru.frames)].data
        # the episode's ultrawide is captured 11 ms after the main frame: that one is the nearest
        assert h.ultrawide.t_sensor_ns - h.main.t_sensor_ns == pytest.approx(11_000_000, abs=1_000)
        assert h.skew_ns == h.ultrawide.t_mid_ns - h.main.t_mid_ns         # the live Pi's sign (PI_PAYLOAD 0.6)
        assert h.main.t_mid_ns == h.main.t_sensor_ns - h.main.exposure_us * 500
        assert h.uw_reused is False
        assert h.t_send_ns >= h.main.t_sensor_ns + SEND_DELAY_NS - 1_000_000   # sent ~45.8 ms after, as the live Pi
        assert h.t_send_ns < h.main.t_sensor_ns + SEND_DELAY_NS + 30_000_000


def test_drop_packets_makes_gaps(fake, listener):
    listener.pump(lambda l: len(l.msgs) >= 2)
    fake.drop_packets(3)
    listener.pump(lambda l: bool(gaps(l.nos())) and len(l.msgs) >= 8)
    assert gaps(listener.nos()) == [3]


def test_drop_frames_main_skips_packets(fake, listener):
    listener.pump(lambda l: len(l.msgs) >= 2)
    fake.drop_frames("main", 2)
    listener.pump(lambda l: bool(gaps(l.nos())) and len(l.msgs) >= 8)
    assert gaps(listener.nos()) == [2]


def test_drop_frames_ultrawide_repeats_the_nearest_remaining_frame(fake, listener):
    listener.pump(lambda l: len(l.msgs) >= 2)
    fake.drop_frames("ultrawide", 1)
    listener.pump(lambda l: any(m.header.uw_reused for m in l.msgs) and len(l.msgs) >= 8)
    reused = [m for m in listener.msgs if m.header.uw_reused]
    assert len(reused) == 1
    i = listener.msgs.index(reused[0])
    assert reused[0].header.ultrawide.frame_no == listener.msgs[i - 1].header.ultrawide.frame_no
    assert gaps(sorted(set(listener.pairs()))) == [1]


def test_repeat_ultrawide_repeats_then_resumes(fake, listener):
    listener.pump(lambda l: len(l.msgs) >= 2)
    fake.repeat_ultrawide(0.3)
    n = len(listener.msgs)
    listener.pump(lambda l: sum(m.header.uw_reused for m in l.msgs[n:]) >= 5)
    run = [m for m in listener.msgs[n:] if m.header.uw_reused]
    assert len({m.header.ultrawide.frame_no for m in run}) == 1           # always the same frame
    assert all(m.ultrawide_jpeg == run[0].ultrawide_jpeg for m in run)
    assert all(m.header.skew_ns == m.header.ultrawide.t_mid_ns - m.header.main.t_mid_ns for m in run)
    listener.pump(lambda l: not l.msgs[-1].header.uw_reused and len(l.msgs) > n + len(run) + 1)
    assert listener.msgs[-1].header.ultrawide.frame_no > run[0].header.ultrawide.frame_no
    assert fake.sent["reused"] >= 5


def test_pause_ultrawide_repeats_the_last_frame_then_resume(fake, listener):
    listener.pump(lambda l: len(l.msgs) >= 2)
    fake.pause("ultrawide")
    n = len(listener.msgs)
    listener.pump(lambda l: len(l.msgs) >= n + 6)
    later = listener.msgs[n + 2:]
    assert all(m.header.uw_reused for m in later) and len({m.header.ultrawide.frame_no for m in later}) == 1
    fake.resume("ultrawide")
    listener.pump(lambda l: not l.msgs[-1].header.uw_reused and len(l.msgs) >= n + 8)


def test_pause_main_stops_the_packets(fake, listener):
    listener.pump(lambda l: len(l.msgs) >= 2)
    fake.pause("main")
    n = len(listener.msgs) + 1                              # at most one already on its way
    t_end = time.monotonic() + 0.3                          # 9 frame periods
    listener.pump(lambda l: time.monotonic() > t_end)
    assert len(listener.msgs) <= n
    fake.resume("main")
    listener.pump(lambda l: len(l.msgs) >= n + 2)


def test_the_clip_loops_with_counters_and_time_still_increasing(fake, listener, recording):
    n = len(recording.cameras["main"].frames)
    listener.pump(lambda l: len(l.msgs) >= n + 4, timeout_s=15)
    ms = listener.msgs
    assert all(b.header.packet_no == a.header.packet_no + 1 and b.header.main.t_sensor_ns > a.header.main.t_sensor_ns
               and b.header.ultrawide.frame_no > a.header.ultrawide.frame_no
               and b.header.ultrawide.t_sensor_ns > a.header.ultrawide.t_sensor_ns for a, b in zip(ms, ms[1:]))


def test_pi_clock_has_offset_and_drift(recording):
    f = FakePiSender(recording, "tcp://127.0.0.1:*", pi_clock_offset_ns=OFFSET, drift_ppm=20.0, send_hwm=10,
                     clock=lambda: 5_000_000_000)
    f._start_ns = 5_000_000_000
    assert f.pi_clock_ns(5_000_000_000) == 5_000_000_000 + OFFSET
    assert f.pi_clock_ns(6_000_000_000) == 6_000_000_000 + OFFSET + 20_000     # 20 ppm = 20 us per second


def test_t_send_follows_the_pi_clock(fake, listener):
    before = time.monotonic_ns()
    listener.pump(lambda l: len(l.msgs) >= 1)
    after = time.monotonic_ns()
    t = listener.msgs[0].header.t_send_ns
    assert fake.pi_clock_ns(before) - 200_000_000 <= t <= fake.pi_clock_ns(after)


def test_a_bind_error_is_raised_by_start(recording, fake):
    other = FakePiSender(recording, fake.frames_endpoint, pi_clock_offset_ns=0, drift_ppm=0.0, send_hwm=10)
    with pytest.raises(Exception):
        other.start()


def test_sent_counts(fake):
    wait_for(lambda: fake.sent["packets"] >= 3, 10, "packets sent")
    assert fake.sent["ultrawide"] >= 1
