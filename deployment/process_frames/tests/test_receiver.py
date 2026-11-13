"""Receiver with RecordingPacketSource + RawDecoder (or the real H264Decoder) + a synced ClockMap (STRUCTURE.md
§6.1)."""

from __future__ import annotations

import av
import numpy as np
import pytest

from contracts import ContractError, Packet

from conftest import RawDecoder, raw_frame, raw_packet, raw_packets, write_rig
from process_frames._dataproc.config import load_rig
from process_frames.camera_health import CameraHealth
from process_frames.clock_map import ClockMap, ClockSample
from process_frames.errors import FatalError
from process_frames.h264_decoder import H264Decoder
from process_frames.packet_buffer import PacketBuffer
from process_frames.packet_source import PacketSource
from process_frames.pi_recording import PiRecording
from process_frames.receiver import Receiver
from process_frames.recording_packet_source import RecordingPacketSource

OFFSET = 1_000_000_000_000           # Pi minus PC
P = 33_333_333
MS = 1_000_000
T_START = 1_000_000_000_000          # the fake clock's start


def t_pi(i, shift_ns=0):
    return T_START + OFFSET + i * P + shift_ns


class Setup:
    """A receiver around a list of packets."""

    def __init__(self, rig, clock, packets, synced=True, source=None, decoders=None):
        self.clock = clock
        self.map = ClockMap(60.0, 0.3, 10.0, 500.0, 10.0)
        if synced:
            self.map.add_sample(ClockSample(t_send_ns=clock(), t_pi_ns=clock() + OFFSET, t_recv_ns=clock()))
        self.buffer = PacketBuffer(1.0)
        self.health = {c: CameraHealth(c, 0.35, 2.0, 1.0, clock) for c in ("main", "ultrawide")}
        self.decoders = decoders or {c: RawDecoder() for c in ("main", "ultrawide")}
        self.source = source or RecordingPacketSource(packets)
        self.rx = Receiver(self.source, self.decoders, rig, self.map, self.buffer, self.health, 5_000_000, 1.0, clock)

    def drain(self):
        while self.rx.receive_one(0.0):
            self.clock.advance(1_000_000)
        return self.buffer.snapshot()


def mains(packets):
    return [p.main_seq if p.main_rgb is not None else None for p in packets]


def uws(packets):
    return [p.pair_seq if p.ultrawide_rgb is not None else None for p in packets]


def test_packets_land_with_the_right_times_images_and_pairs(small_rig, fake_clock):
    pk = raw_packets([t_pi(i) for i in range(6)], [t_pi(i, -10 * MS) for i in range(6)], exposure_ns=2_000_000)
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert [p.packet_seq for p in got] == list(range(6))
    assert mains(got) == list(range(6)) and uws(got) == list(range(6))
    p = got[2]
    assert p.t_pc_ns == t_pi(2) - 1_000_000 - OFFSET                  # main mid-exposure, mapped to PC
    assert p.t_pi_mid_ns == t_pi(2) - 1_000_000 and p.pair_gap_ns == 10 * MS
    assert p.main_rgb.shape == (90, 160, 3) and not p.main_rgb.flags.writeable and p.main_rgb[0, 0, 0] == 14
    assert p.ultrawide_rgb.shape == (96, 128, 3) and p.ultrawide_rgb[0, 0, 0] == 114
    assert r.rx.packets.buffered == 6 and r.rx.counters["ultrawide"].usable == 6
    assert r.health["main"].status(p.t_pc_ns) == "ok" and r.health["ultrawide"].status(p.t_pc_ns) == "ok"
    assert r.source.exhausted


@pytest.mark.parametrize("uw_shift_ms", [-30, -12, 0, 4, 9, 20])
def test_every_ultrawide_frame_decoded_exactly_once_and_in_order(small_rig, fake_clock, uw_shift_ms):
    rng = np.random.default_rng(uw_shift_ms + 40)
    main_t = [t_pi(i) + int(rng.integers(-3 * MS, 3 * MS)) for i in range(40)]
    uw_t = [t_pi(i, uw_shift_ms * MS) + int(rng.integers(-3 * MS, 3 * MS)) for i in range(40)]
    pk = raw_packets(main_t, uw_t)
    counts = {len(p.ultrawide_data) for p in pk}
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    sent = [f.info.frame_seq for p in pk for f in p.ultrawide]
    assert r.decoders["ultrawide"].decoded == sent == sorted(set(sent))        # once each, in capture order
    assert r.decoders["main"].decoded == list(range(40))
    for p in got:                                    # each pair's image is that very frame
        if p.ultrawide_rgb is not None:
            assert p.ultrawide_rgb[0, 0, 0] == (p.pair_seq * 7 + 100) % 256
    assert counts <= {0, 1, 2}


def test_a_pair_from_an_earlier_packet_is_found(small_rig, fake_clock):
    uw0 = raw_frame("ultrawide", 0, t_pi(0, 4 * MS))
    pk = [raw_packet(raw_frame("main", 0, t_pi(0)), [uw0], 0),
          raw_packet(raw_frame("main", 1, t_pi(1)), [], 0)]                   # pair_seq 0, sent in packet 0
    got = Setup(small_rig, fake_clock, pk).drain()
    assert uws(got) == [0, 0] and got[1].ultrawide_rgb is got[0].ultrawide_rgb


def test_no_pair_and_a_missing_pair(small_rig, fake_clock):
    pk = [raw_packet(raw_frame("main", 0, t_pi(0)), [raw_frame("ultrawide", 0, t_pi(0, -5 * MS))], -1),
          raw_packet(raw_frame("main", 1, t_pi(1)), [raw_frame("ultrawide", 2, t_pi(1, -5 * MS))], 1)]  # 1 lost
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert uws(got) == [None, None] and mains(got) == [0, 1]
    c = r.rx.counters["ultrawide"]
    assert c.unpaired == 1 and c.pair_missing == 1 and c.dropped == 1


def test_a_main_gap_serves_no_main_until_its_keyframe(small_rig, fake_clock):
    n = 12
    pk = raw_packets([t_pi(i) for i in range(n)], [t_pi(i, -8 * MS) for i in range(n)], keyframes=(0, 8),
                     skip_main=(3, 4))                       # the Pi dropped main frames 3 and 4
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert [p.packet_seq for p in got] == [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    assert mains(got) == [0, 1, 2, None, None, None, 8, 9, 10, 11]
    assert uws(got) == [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]           # the ultrawide is unaffected
    c = r.rx.counters["main"]
    assert c.dropped == 2 and c.unusable == 3 and r.health["main"].dropped == 2 and r.rx.packets.lost == 2


def test_an_ultrawide_gap_serves_no_ultrawide_until_its_keyframe(small_rig, fake_clock):
    n = 12
    pk = raw_packets([t_pi(i) for i in range(n)], [t_pi(i, -8 * MS) for i in range(n)], keyframes=(0, 8),
                     skip_uw=(4,))
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert mains(got) == list(range(n))
    # packet 4 pairs with frame 3 (41 ms old, from packet 3); frames 5-7 decode while waiting for keyframe 8
    assert uws(got) == [0, 1, 2, 3, 3, None, None, None, 8, 9, 10, 11]
    c = r.rx.counters["ultrawide"]
    assert c.dropped == 1 and c.unusable == 3                      # 5, 6, 7 decoded while waiting


def test_a_dropped_packet_serves_nothing_until_the_keyframes(small_rig, fake_clock):
    n = 12
    pk = raw_packets([t_pi(i) for i in range(n)], [t_pi(i, -8 * MS) for i in range(n)], keyframes=(0, 8))
    pk = pk[:5] + pk[6:]                                           # packet 5 lost on the way
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert mains(got) == [0, 1, 2, 3, 4, None, None, 8, 9, 10, 11]
    assert uws(got) == [0, 1, 2, 3, 4, None, None, 8, 9, 10, 11]
    assert r.rx.packets.lost == 1
    assert r.rx.counters["main"].dropped == 1 and r.rx.counters["ultrawide"].dropped == 1


def swapped_rig(tmp_path):
    def swap(d):
        d["cameras"]["main"]["sensor"], d["cameras"]["ultrawide"]["sensor"] = "imx219", "imx519"
    return load_rig(write_rig(tmp_path, "rig_swapped", swap))


@pytest.mark.parametrize("case,needle", [
    ("sensor", "main stream sensor is imx519, rig file says imx219"),
    ("size", "ultrawide frame size 1280x960, rig file says 128x96"),
    ("encoding", "main stream is jpeg, process_frames.decoder is h264"),
])
def test_wrong_sensor_size_or_encoding_is_fatal_naming_the_camera(small_rig, fake_clock, tmp_path, case, needle):
    rig = swapped_rig(tmp_path) if case == "sensor" else small_rig
    main = raw_frame("main", 0, t_pi(0), **({"encoding": "jpeg"} if case == "encoding" else {}))
    uw = raw_frame("ultrawide", 0, t_pi(0, -5 * MS), **({"size": (1280, 960)} if case == "size" else {}))
    r = Setup(rig, fake_clock, [raw_packet(main, [uw], 0)])
    with pytest.raises(FatalError) as e:
        r.rx.receive_one(0.0)
    assert needle in str(e.value)


class BrokenSource(PacketSource):
    def next_packet(self, timeout_s):
        return Packet.from_parts([raw_packet(raw_frame("main", 0, 1)).to_parts()[0], b"m", b"extra"])

    def close(self):
        pass


def test_a_c0_contract_error_is_fatal(small_rig, fake_clock):
    r = Setup(small_rig, fake_clock, [], source=BrokenSource())
    with pytest.raises(FatalError) as e:
        r.rx.receive_one(0.0)
    assert "C0 packet breaks its contract" in str(e.value) and "parts" in str(e.value)
    assert e.value.check == "check the Pi sender version"
    assert isinstance(e.value.__cause__, ContractError)


def test_a_decoded_size_lie_is_fatal(small_rig, fake_clock):
    class Lying(RawDecoder):
        def decode(self, frame):
            return super().decode(frame)[:, :80]

    r = Setup(small_rig, fake_clock, [raw_packet(raw_frame("main", 0, t_pi(0)))],
              decoders={"main": Lying(), "ultrawide": RawDecoder()})
    with pytest.raises(FatalError, match="decoded picture is 80x90"):
        r.rx.receive_one(0.0)


def test_unsynced_packets_are_decoded_but_not_buffered(small_rig, fake_clock):
    pk = raw_packets([t_pi(i) for i in range(3)], [t_pi(i, -5 * MS) for i in range(3)])
    r = Setup(small_rig, fake_clock, pk, synced=False)
    assert r.drain() == () and r.rx.packets.unsynced == 3
    assert r.decoders["main"].decoded == [0, 1, 2] and r.decoders["ultrawide"].decoded == [0, 1, 2]
    r = Setup(small_rig, fake_clock, pk[:1])                         # a sample that became too old
    fake_clock.advance_s(10.5)
    assert r.drain() == () and r.rx.packets.unsynced == 1


def restart_cases():
    m = lambda seq, i, **kw: raw_frame("main", seq, t_pi(i), **kw)          # noqa: E731
    u = lambda seq, i: raw_frame("ultrawide", seq, t_pi(i, -5 * MS))        # noqa: E731
    first = raw_packet(m(10, 10), [u(20, 10)], 20)
    return {
        "packet_seq back": [first, raw_packet(m(11, 11), [u(21, 11)], 21, packet_seq=3)],
        "main frame_seq back": [first, raw_packet(m(2, 11), [u(21, 11)], 21, packet_seq=11)],
        "ultrawide frame_seq back": [first, raw_packet(m(11, 11), [u(4, 11)], 4)],
        "main t_pi_ns back": [first, raw_packet(m(11, 9), [u(21, 11)], 21)],
        "ultrawide t_pi_ns back": [first, raw_packet(m(11, 11), [u(21, 9)], 21)],
    }


@pytest.mark.parametrize("case", list(restart_cases()))
def test_a_pi_restart_latches_both_cameras_down(small_rig, fake_clock, case):
    pk = restart_cases()[case]
    pk.append(raw_packet(raw_frame("main", 0, t_pi(30)), [raw_frame("ultrawide", 0, t_pi(30, -5 * MS))], 0,
                         packet_seq=50))
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert [p.packet_seq for p in got] == [10]
    assert r.health["main"].status() == "down" and r.health["ultrawide"].status() == "down"
    assert r.rx.packets.ignored == 2                     # the restart packet, and everything after it


def test_an_out_of_order_pc_time_is_dropped(small_rig, fake_clock):
    pk = raw_packets([t_pi(i) for i in range(2)], [t_pi(i, -5 * MS) for i in range(2)])
    r = Setup(small_rig, fake_clock, pk)
    assert r.rx.receive_one(0.0)
    for _ in range(10):                                  # a clock re-fit moving the offset by more than a period
        r.map.add_sample(ClockSample(t_send_ns=fake_clock.now, t_pi_ns=fake_clock.now + OFFSET + 3 * P,
                                     t_recv_ns=fake_clock.now))
    assert r.rx.receive_one(0.0)
    assert r.rx.packets.out_of_order == 1 and len(r.buffer) == 1


class CountingH264(H264Decoder):
    def __init__(self, camera):
        super().__init__(camera, "SLICE")
        self.decoded = []

    def decode(self, frame):
        self.decoded.append(frame.info.frame_seq)
        return super().decode(frame)


def offline(frames):
    """Decode a whole recorded stream with PyAV in one go, as a file would be."""
    ctx = av.CodecContext.create("h264", "r")
    out = []
    for f in frames:
        out += [p.to_ndarray(format="rgb24") for p in ctx.decode(av.Packet(f.data))]
    out += [p.to_ndarray(format="rgb24") for p in ctx.decode(None)]
    return out


def test_real_h264_every_ultrawide_frame_once_in_order_and_identical(small_rig, fake_clock, pi_episode):
    rec = PiRecording.load(pi_episode, encoding="h264", max_seconds=10.0, keyframe_interval=10, jpeg_quality=90)
    t0 = rec.start_ns
    packets = [Packet.build(p.header.packet_seq, *_shift(p, T_START + OFFSET - t0)) for p in rec.packets()]
    decoders = {"main": CountingH264("main"), "ultrawide": CountingH264("ultrawide")}
    r = Setup(small_rig, fake_clock, packets, decoders=decoders)
    got = r.drain()
    sent = [f.info.frame_seq for p in packets for f in p.ultrawide]
    assert decoders["ultrawide"].decoded == sent == list(range(len(sent)))
    ref = {cam: offline(rec.cameras[cam].frames) for cam in ("main", "ultrawide")}
    for p in got:
        assert np.array_equal(p.main_rgb, ref["main"][p.main_seq])
        if p.ultrawide_rgb is not None:
            assert np.array_equal(p.ultrawide_rgb, ref["ultrawide"][p.pair_seq])
    assert len(got) == len(rec.cameras["main"].frames)


def _shift(p, by):
    """The packet's frames with every t_pi_ns moved by ``by`` (the fake clock's Pi time)."""
    from process_frames.pi_recording import with_seq_and_time
    main = with_seq_and_time(p.main, p.main.info.frame_seq, p.main.info.t_pi_ns + by)
    uw = [with_seq_and_time(f, f.info.frame_seq, f.info.t_pi_ns + by) for f in p.ultrawide]
    return main, uw, p.header.pair_seq
