"""FakePiSender over 127.0.0.1:*: its C0 messages pass the contract validator (STRUCTURE.md §5.14)."""

from __future__ import annotations

import time

import pytest

from contracts import Frame, RpcClient, Subscriber, TimeSyncReply, TimeSyncRequest, TransportTimeout

from conftest import wait_for
from process_frames.fake_pi_sender import FakePiSender
from process_frames.pi_recording import PiRecording

OFFSET = 1000 * 1_000_000_000


@pytest.fixture(scope="module")
def recording(pi_episode_no_exposure):
    return PiRecording.load(pi_episode_no_exposure, encoding="h264", max_seconds=10.0, keyframe_interval=4,
                            bitrate=2_000_000, jpeg_quality=90)


@pytest.fixture
def fake(recording):
    f = FakePiSender(recording, "tcp://127.0.0.1:*", "tcp://127.0.0.1:*", pi_clock_offset_ns=OFFSET, drift_ppm=20.0,
                     send_hwm=120)
    f.start()
    yield f
    f.stop()


class Listener:
    """A contracts Subscriber collecting C0 frames per camera."""

    def __init__(self, endpoint):
        self.sub = Subscriber(name="pi", endpoint=endpoint, expect=Frame, recv_hwm=1000)
        self.frames = {"main": [], "ultrawide": []}

    def pump(self, until, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while not until(self):
            assert time.monotonic() < deadline, "timed out"
            f = self.sub.receive(0.05)          # validated against C0 by the Subscriber
            if f is not None:
                self.frames[f.camera].append(f)

    def seqs(self, cam):
        return [f.seq for f in self.frames[cam]]

    def close(self):
        self.sub.close()


@pytest.fixture
def listener(fake):
    lst = Listener(fake.frames_endpoint)
    yield lst
    lst.close()


def test_endpoints_are_bound(fake):
    assert fake.frames_endpoint.startswith("tcp://127.0.0.1:") and not fake.frames_endpoint.endswith("*")
    assert fake.time_sync_endpoint.startswith("tcp://127.0.0.1:") and not fake.time_sync_endpoint.endswith("*")


def test_frames_pass_the_contract_and_seq_counts_up_by_one(fake, listener, recording):
    listener.pump(lambda l: len(l.frames["main"]) >= 8 and len(l.frames["ultrawide"]) >= 8)
    for cam in ("main", "ultrawide"):
        s = listener.seqs(cam)
        assert all(b == a + 1 for a, b in zip(s, s[1:])), s
        f = listener.frames[cam][0]
        rc = recording.cameras[cam]
        assert (f.sensor, f.width, f.height, f.encoding) == (rc.sensor, rc.width, rc.height, "h264")
        assert f.data == rc.frames[f.seq % len(rc.frames)].data
        assert f.keyframe == rc.frames[f.seq % len(rc.frames)].keyframe


def test_drop_next_makes_a_gap_of_n(fake, listener):
    listener.pump(lambda l: len(l.frames["main"]) >= 2)
    fake.drop_next("main", 3)
    listener.pump(lambda l: any(b - a > 1 for a, b in zip(l.seqs("main"), l.seqs("main")[1:])))
    s = listener.seqs("main")
    gaps = [b - a - 1 for a, b in zip(s, s[1:]) if b - a > 1]
    assert gaps == [3]


def test_pause_and_resume_per_camera(fake, listener):
    listener.pump(lambda l: len(l.frames["ultrawide"]) >= 2)
    fake.pause("ultrawide")
    n_uw = len(listener.frames["ultrawide"])
    listener.pump(lambda l: len(l.frames["main"]) >= len(l.frames["ultrawide"]) + 8)
    assert len(listener.frames["ultrawide"]) <= n_uw + 1          # at most one already on its way
    n_uw = len(listener.frames["ultrawide"])
    fake.resume("ultrawide")
    listener.pump(lambda l: len(l.frames["ultrawide"]) >= n_uw + 2)
    s = listener.seqs("ultrawide")
    assert any(b - a > 1 for a, b in zip(s, s[1:]))               # frames due while paused were lost


def test_the_clip_loops_with_seq_and_time_still_increasing(fake, listener, recording):
    n = len(recording.cameras["main"].frames)
    listener.pump(lambda l: len(l.frames["main"]) >= n + 4, timeout_s=15)
    fr = listener.frames["main"]
    assert all(b.seq == a.seq + 1 and b.t_pi_ns > a.t_pi_ns for a, b in zip(fr, fr[1:]))
    looped = [f for f in fr if f.seq % n == 0 and f.seq > 0]
    assert looped and looped[0].keyframe


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
