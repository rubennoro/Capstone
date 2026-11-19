# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_receiver.py @ 2ffe641
# changes: nothing is decoded: no RawDecoder or PacketBuffer; the accepted packets are the RecordItems a list sink
# collects (bytes, times, pairs instead of images); "every frame decoded once" became "every new ultrawide frame
# recorded once, the very bytes sent"; the decoded-size lie became a JPEG-header lie (and a non-JPEG); the encoding
# message names JPEG; + the raw one-way sample of the source in the item; imports.
"""Receiver with RecordingPacketSource + a synced ClockMap + a list sink (STRUCTURE.md 5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from deploy_contracts import ContractError
from frames_helpers import jpeg_bytes, raw_frame, raw_packet, raw_packets, write_rig

from teleop_collect.contracts import Packet
from teleop_collect.contracts.clock_fit import ClockMap, ClockSample
from teleop_collect.frames._dataproc.config import load_rig
from teleop_collect.frames.camera_health import CameraHealth
from teleop_collect.frames.errors import FatalError
from teleop_collect.frames.packet_source import PacketSource
from teleop_collect.frames.receiver import Receiver
from teleop_collect.frames.recording_packet_source import RecordingPacketSource

OFFSET = 1_000_000_000_000           # Pi minus laptop
P = 33_333_333
MS = 1_000_000
T_START = 1_000_000_000_000          # the fake clock's start


def t_pi(i, shift_ns=0):
    return T_START + OFFSET + i * P + shift_ns


class Setup:
    """A receiver around a list of packets; the accepted packets land in self.items."""

    def __init__(self, rig, clock, packets, synced=True, source=None):
        self.clock = clock
        self.map = ClockMap(60.0, 0.3, 10.0, 500.0, 10.0)
        if synced:
            self.map.add_sample(ClockSample(t_send_ns=clock(), t_remote_ns=clock() + OFFSET, t_recv_ns=clock()))
        self.health = {c: CameraHealth(c, 0.35, 2.0, 1.0, clock) for c in ("main", "ultrawide")}
        self.source = source or RecordingPacketSource(packets)
        self.items = []
        self.rx = Receiver(self.source, rig, self.map, self.health, 5_000_000, 1.0, clock,
                           sinks=[self.items.append])

    def drain(self):
        while self.rx.receive_one(0.0):
            self.clock.advance(1_000_000)
        return self.items


def mains(items):
    return [it.main.info.frame_seq for it in items]


def uws(items):
    """The pair each packet used (None: refused or none)."""
    return [it.ultrawide_frame_no if it.pair_used else None for it in items]


def test_packets_land_with_the_right_times_bytes_and_pairs(small_rig, fake_clock):
    pk = raw_packets([t_pi(i) for i in range(6)], [t_pi(i, -10 * MS) for i in range(6)], exposure_ns=2_000_000)
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert [it.packet_no for it in got] == list(range(6))
    assert mains(got) == list(range(6)) and uws(got) == list(range(6))
    it = got[2]
    assert it.main_t_mid_ns == t_pi(2) - 1_000_000 - OFFSET                  # main mid-exposure, on the laptop
    assert it.main_t_mid_pi_ns == t_pi(2) - 1_000_000 and it.pair_gap_ms == pytest.approx(10.0)
    assert it.main.data == pk[2].main_data                                  # the bytes exactly as received
    assert it.ultrawide.data == pk[2].ultrawide_data[0] and it.ultrawide.info.frame_seq == 2
    assert it.ultrawide_t_mid_pi_ns == t_pi(2, -10 * MS) - 1_000_000
    assert it.ultrawide_t_mid_ns == it.ultrawide_t_mid_pi_ns - OFFSET
    assert not it.uw_reused and it.pair_used
    assert it.t_send_pi_ns == t_pi(2)                    # RecordingPacketSource has no send stamp: the main t_pi_ns
    assert r.rx.packets.accepted == 6 and r.rx.counters["ultrawide"].usable == 6
    assert r.health["main"].status(it.main_t_mid_ns) == "ok"
    assert r.health["ultrawide"].status(it.main_t_mid_ns) == "ok"
    assert r.source.exhausted


@pytest.mark.parametrize("uw_shift_ms", [-30, -12, 0, 4, 9, 20])
def test_every_new_ultrawide_frame_recorded_once_in_order_with_its_bytes(small_rig, fake_clock, uw_shift_ms):
    """The live stream carries at most one new ultrawide frame per packet; in a hand-made C0 packet with two, the
    newest is the one recorded. Every packet's main frame is recorded."""
    rng = np.random.default_rng(uw_shift_ms + 40)
    main_t = [t_pi(i) + int(rng.integers(-3 * MS, 3 * MS)) for i in range(40)]
    uw_t = [t_pi(i, uw_shift_ms * MS) + int(rng.integers(-3 * MS, 3 * MS)) for i in range(40)]
    pk = raw_packets(main_t, uw_t)
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert mains(got) == list(range(40))
    recorded = [it.ultrawide.info.frame_seq for it in got if it.ultrawide is not None]
    expected = [p.header.ultrawide[-1].frame_seq for p in pk if p.header.ultrawide]
    assert recorded == expected == sorted(set(recorded))
    for p, it in zip(pk, got):
        if it.ultrawide is not None:
            assert it.ultrawide.data == p.ultrawide_data[-1]
        assert it.uw_reused == (not p.header.ultrawide and p.header.pair_seq >= 0)
    assert r.rx.counters["ultrawide"].received == sum(len(p.ultrawide) for p in pk)


def test_a_pair_from_an_earlier_packet_is_found(small_rig, fake_clock):
    uw0 = raw_frame("ultrawide", 0, t_pi(0, 4 * MS))
    pk = [raw_packet(raw_frame("main", 0, t_pi(0)), [uw0], 0),
          raw_packet(raw_frame("main", 1, t_pi(1)), [], 0)]                   # pair_seq 0, sent in packet 0
    got = Setup(small_rig, fake_clock, pk).drain()
    assert uws(got) == [0, 0]
    assert got[0].ultrawide is not None and got[1].ultrawide is None and got[1].uw_reused


def test_no_pair_and_a_missing_pair(small_rig, fake_clock):
    pk = [raw_packet(raw_frame("main", 0, t_pi(0)), [raw_frame("ultrawide", 0, t_pi(0, -5 * MS))], -1),
          raw_packet(raw_frame("main", 1, t_pi(1)), [raw_frame("ultrawide", 2, t_pi(1, -5 * MS))], 1)]  # 1 lost
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert uws(got) == [None, None] and mains(got) == [0, 1]
    assert [it.pair_gap_ms for it in got] == [None, None]
    assert [it.ultrawide.info.frame_seq for it in got] == [0, 2]      # recorded even when not the pair
    c = r.rx.counters["ultrawide"]
    assert c.unpaired == 1 and c.pair_missing == 1 and c.dropped == 1


def test_a_main_gap_costs_only_the_lost_frames(small_rig, fake_clock):
    n = 12
    pk = raw_packets([t_pi(i) for i in range(n)], [t_pi(i, -8 * MS) for i in range(n)],
                     skip_main=(3, 4))                       # the Pi dropped main frames 3 and 4
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert [it.packet_no for it in got] == [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    assert mains(got) == [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    assert uws(got) == [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]            # the ultrawide is unaffected
    c = r.rx.counters["main"]
    assert c.dropped == 2 and r.health["main"].dropped == 2 and r.rx.packets.lost == 2


def test_an_ultrawide_gap_pairs_the_nearest_frame_sent(small_rig, fake_clock):
    n = 12
    pk = raw_packets([t_pi(i) for i in range(n)], [t_pi(i, -8 * MS) for i in range(n)], skip_uw=(4,))
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert mains(got) == list(range(n))
    # packet 4 is sent before frame 5 exists, so it pairs frame 3 again (41 ms older, its 2nd packet: accepted)
    assert uws(got) == [0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11]
    assert got[4].ultrawide is None and got[4].uw_reused                   # the reused frame is not recorded twice
    c = r.rx.counters["ultrawide"]
    assert c.dropped == 1 and r.rx.packets.uw_reused == 1


def test_a_dropped_packet_costs_only_that_packet(small_rig, fake_clock):
    n = 12
    pk = raw_packets([t_pi(i) for i in range(n)], [t_pi(i, -8 * MS) for i in range(n)])
    pk = pk[:5] + pk[6:]                                           # packet 5 lost on the way
    r = Setup(small_rig, fake_clock, pk)
    got = r.drain()
    assert mains(got) == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    assert uws(got) == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    assert r.rx.packets.lost == 1
    assert r.rx.counters["main"].dropped == 1 and r.rx.counters["ultrawide"].dropped == 1


def swapped_rig(tmp_path):
    def swap(d):
        d["cameras"]["main"]["sensor"], d["cameras"]["ultrawide"]["sensor"] = "imx219", "imx519"
    return load_rig(write_rig(tmp_path, "rig_swapped", swap))


@pytest.mark.parametrize("case,needle", [
    ("sensor", "main stream sensor is imx519, rig file says imx219"),
    ("size", "ultrawide frame size 1280x960, rig file says 128x96"),
    ("encoding", "main stream is h264, frames records jpeg"),
])
def test_wrong_sensor_size_or_encoding_is_fatal_naming_the_camera(small_rig, fake_clock, tmp_path, case, needle):
    rig = swapped_rig(tmp_path) if case == "sensor" else small_rig
    main = raw_frame("main", 0, t_pi(0), **({"encoding": "h264"} if case == "encoding" else {}))
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


def test_a_jpeg_size_lie_is_fatal(small_rig, fake_clock):
    lying = raw_frame("main", 0, t_pi(0), data=jpeg_bytes(80, 90, 1))      # the map says 160x90
    r = Setup(small_rig, fake_clock, [raw_packet(lying)])
    with pytest.raises(FatalError, match="main JPEG is 80x90 at frame_seq 0, the C0 header says 160x90"):
        r.rx.receive_one(0.0)


def test_a_frame_that_is_not_a_jpeg_is_fatal(small_rig, fake_clock):
    r = Setup(small_rig, fake_clock, [raw_packet(raw_frame("main", 0, t_pi(0), data=b"\x00\x01raw rgb"))])
    with pytest.raises(FatalError, match="main frame_seq 0 is not a readable JPEG") as e:
        r.rx.receive_one(0.0)
    assert e.value.check == "check the Pi encoder settings"


def test_unsynced_packets_are_checked_but_not_accepted(small_rig, fake_clock):
    pk = raw_packets([t_pi(i) for i in range(3)], [t_pi(i, -5 * MS) for i in range(3)])
    r = Setup(small_rig, fake_clock, pk, synced=False)
    assert r.drain() == [] and r.rx.packets.unsynced == 3
    assert r.rx.counters["main"].received == 3 and r.rx.counters["ultrawide"].received == 3
    r = Setup(small_rig, fake_clock, pk[:1])                         # a sample that became too old
    fake_clock.advance_s(10.5)
    assert r.drain() == [] and r.rx.packets.unsynced == 1


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
    assert [it.packet_no for it in got] == [10]                      # nothing after the restart is recorded
    assert r.health["main"].status() == "down" and r.health["ultrawide"].status() == "down"
    assert r.rx.packets.ignored == 2                     # the restart packet, and everything after it


def test_an_out_of_order_laptop_time_is_dropped(small_rig, fake_clock):
    pk = raw_packets([t_pi(i) for i in range(2)], [t_pi(i, -5 * MS) for i in range(2)])
    r = Setup(small_rig, fake_clock, pk)
    assert r.rx.receive_one(0.0)
    for _ in range(10):                                  # a clock re-fit moving the offset by more than a period
        r.map.add_sample(ClockSample(t_send_ns=fake_clock.now, t_remote_ns=fake_clock.now + OFFSET + 3 * P,
                                     t_recv_ns=fake_clock.now))
    assert r.rx.receive_one(0.0)
    assert r.rx.packets.out_of_order == 1 and len(r.items) == 1


class StampedSource(RecordingPacketSource):
    """A source with its own one-way sample, as PiStreamPacketSource has."""

    def next_packet(self, timeout_s):
        p = super().next_packet(timeout_s)
        if p is not None:
            self.stamps = (p.header.main.t_pi_ns + 45_800_000, T_START + 123)
        return p

    @property
    def last_t_send_ns(self):
        return self.stamps[0]

    @property
    def last_arrival_ns(self):
        return self.stamps[1]


def test_the_item_carries_the_sources_raw_one_way_sample(small_rig, fake_clock):
    pk = raw_packets([t_pi(0)], [t_pi(0, -5 * MS)])
    r = Setup(small_rig, fake_clock, pk, source=StampedSource(pk))
    (it,) = r.drain()
    assert (it.t_send_pi_ns, it.arrival_ns) == (t_pi(0) + 45_800_000, T_START + 123)
