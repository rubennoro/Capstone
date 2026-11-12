"""Receiver with ReplaySource + RawDecoder + a synced ClockMap (STRUCTURE.md §6.1)."""

from __future__ import annotations

import pytest

from contracts import ContractError, Frame

from conftest import RawDecoder, raw_frame
from process_frames.camera_health import CameraHealth
from process_frames.clock_map import ClockMap, ClockSample
from process_frames.errors import FatalError
from process_frames.frame_buffer import FrameBuffer
from process_frames.frame_source import FrameSource
from process_frames.receiver import Receiver
from process_frames.replay_source import ReplaySource

OFFSET = 1_000_000_000_000
P = 33_333_333


class Rig:
    """Builds a receiver around a list of frames."""

    def __init__(self, rig, clock, frames, synced=True, source=None):
        self.clock = clock
        self.map = ClockMap(60.0, 0.3, 10.0, 500.0, 10.0)
        if synced:
            self.map.add_sample(ClockSample(t_send_ns=clock(), t_pi_ns=clock() + OFFSET, t_recv_ns=clock()))
        self.buffers = {c: FrameBuffer(c, 1.0) for c in ("main", "ultrawide")}
        self.health = {c: CameraHealth(c, 0.35, 2.0, 1.0, clock) for c in ("main", "ultrawide")}
        self.decoders = {c: RawDecoder() for c in ("main", "ultrawide")}
        self.source = source or ReplaySource(frames)
        self.rx = Receiver(self.source, self.decoders, rig, self.map, self.buffers, self.health, 5_000_000, clock)

    def drain(self):
        while self.rx.receive_one(0.0):
            self.clock.advance(1_000_000)


T_START = 1_000_000_000_000          # the fake clock's start


def t_pi(clock, i):
    return T_START + OFFSET + i * P


def test_frames_land_in_the_right_buffer_with_the_right_times(small_rig, fake_clock):
    frames = []
    for i in range(5):
        frames += [raw_frame("main", i, t_pi(fake_clock, i), exposure_ns=2_000_000),
                   raw_frame("ultrawide", i, t_pi(fake_clock, i) + 7_000_000, exposure_ns=2_000_000)]
    r = Rig(small_rig, fake_clock, frames)
    r.drain()
    assert [f.seq for f in r.buffers["main"].snapshot()] == [0, 1, 2, 3, 4]
    assert [f.seq for f in r.buffers["ultrawide"].snapshot()] == [0, 1, 2, 3, 4]
    m = r.buffers["main"].snapshot()[2]
    assert m.t_pc_ns == t_pi(fake_clock, 2) - 1_000_000 - OFFSET
    assert m.rgb.shape == (90, 160, 3) and not m.rgb.flags.writeable and m.rgb[0, 0, 0] == (2 * 7) % 256
    assert r.rx.counters["main"].buffered == 5 and r.rx.counters["ultrawide"].received == 5
    assert r.health["main"].status(m.t_pc_ns) == "ok"
    assert r.source.exhausted


def test_seq_gap_is_dropped_and_nothing_buffered_until_the_next_keyframe(small_rig, fake_clock):
    seqs = [0, 1, 2, 5, 6, 7, 8, 9]                       # 3 and 4 lost; keyframe again at 8
    frames = [raw_frame("main", s, t_pi(fake_clock, s), keyframe=(s in (0, 8))) for s in seqs]
    r = Rig(small_rig, fake_clock, frames)
    r.drain()
    assert [f.seq for f in r.buffers["main"].snapshot()] == [0, 1, 2, 8, 9]
    c = r.rx.counters["main"]
    assert c.dropped == 2 and c.unusable == 3 and r.health["main"].dropped == 2


@pytest.mark.parametrize("kw,needle", [
    ({"sensor": "imx219"}, "main stream sensor is imx219, rig file says imx519"),
    ({"size": (1600, 900)}, "main frame size 1600x900, rig file says 160x90"),
    ({"encoding": "jpeg"}, "main stream is jpeg, process_frames.decoder is h264"),
])
def test_wrong_sensor_size_or_encoding_is_fatal_naming_the_camera(small_rig, fake_clock, kw, needle):
    r = Rig(small_rig, fake_clock, [raw_frame("main", 0, t_pi(fake_clock, 0), **kw)])
    with pytest.raises(FatalError) as e:
        r.rx.receive_one(0.0)
    assert needle in str(e.value)


def test_wrong_ultrawide_sensor_is_fatal(small_rig, fake_clock):
    r = Rig(small_rig, fake_clock, [raw_frame("ultrawide", 0, t_pi(fake_clock, 0), sensor="imx519")])
    with pytest.raises(FatalError, match="ultrawide stream sensor is imx519, rig file says imx219"):
        r.rx.receive_one(0.0)


class BrokenSource(FrameSource):
    def next_frame(self, timeout_s):
        raise ContractError("c0.frame/v1", "keyframe", "value", "a jpeg frame must be a keyframe")

    def close(self):
        pass


def test_contract_error_is_fatal(small_rig, fake_clock):
    r = Rig(small_rig, fake_clock, [], source=BrokenSource())
    with pytest.raises(FatalError) as e:
        r.rx.receive_one(0.0)
    assert "C0 frame breaks its contract" in str(e.value) and "keyframe" in str(e.value)
    assert e.value.check == "check the Pi sender version"


def test_decoded_size_lie_is_fatal(small_rig, fake_clock):
    data = bytes(90 * 160 * 3)
    bad = Frame(camera="main", sensor="imx519", seq=0, t_pi_ns=t_pi(fake_clock, 0), exposure_ns=0, width=160,
                height=90, encoding="h264", keyframe=True, data=data)

    class Lying(RawDecoder):
        def decode(self, frame):
            return super().decode(frame)[:, :80]

    r = Rig(small_rig, fake_clock, [bad])
    r.rx._decoders["main"] = Lying()
    with pytest.raises(FatalError, match="decoded picture is 80x90"):
        r.rx.receive_one(0.0)


def test_unsynced_frames_are_not_buffered(small_rig, fake_clock):
    r = Rig(small_rig, fake_clock, [raw_frame("main", i, t_pi(fake_clock, i)) for i in range(3)], synced=False)
    r.drain()
    assert len(r.buffers["main"]) == 0 and r.rx.counters["main"].unsynced == 3
    # a sample that becomes too old also stops buffering
    r = Rig(small_rig, fake_clock, [raw_frame("main", 0, t_pi(fake_clock, 0))])
    fake_clock.advance_s(10.5)
    r.drain()
    assert len(r.buffers["main"]) == 0 and r.rx.counters["main"].unsynced == 1


@pytest.mark.parametrize("second", [dict(seq=1, back_t=True), dict(seq=0, back_t=False)])
def test_seq_or_time_going_back_latches_the_camera_down(small_rig, fake_clock, second):
    t1 = t_pi(fake_clock, 5)
    t2 = t1 - P if second["back_t"] else t1 + P
    frames = [raw_frame("main", 3, t1), raw_frame("main", second["seq"] if second["back_t"] else 2, t2),
              raw_frame("main", 4, t1 + 2 * P)]
    if second["back_t"]:
        frames[1] = raw_frame("main", 4, t2)              # seq goes up, time goes back
    r = Rig(small_rig, fake_clock, frames)
    r.drain()
    assert r.health["main"].status() == "down"
    assert [f.seq for f in r.buffers["main"].snapshot()] == [3]
    assert r.rx.counters["main"].ignored == 2             # the restart frame, and everything after it


def test_out_of_order_pc_time_is_dropped(small_rig, fake_clock):
    frames = [raw_frame("main", 0, t_pi(fake_clock, 0)), raw_frame("main", 1, t_pi(fake_clock, 1))]
    r = Rig(small_rig, fake_clock, frames)
    assert r.rx.receive_one(0.0)
    # a clock re-fit moving the offset by more than a frame period
    for _ in range(10):
        r.map.add_sample(ClockSample(t_send_ns=fake_clock.now, t_pi_ns=fake_clock.now + OFFSET + 3 * P,
                                     t_recv_ns=fake_clock.now))
    assert r.rx.receive_one(0.0)
    assert r.rx.counters["main"].out_of_order == 1 and len(r.buffers["main"]) == 1
