import dataclasses

import pytest

from contracts import (ContractError, EncodedFrame, FrameInfo, Packet, PacketHeader, TimeSyncReply, TimeSyncRequest,
                       decode, encode, pack_obj)
from helpers import contract_error


def info(sensor="imx519", seq=0, **change):
    size = (1600, 1200) if sensor == "imx519" else (1280, 960)
    d = dict(sensor=sensor, frame_seq=seq, t_pi_ns=1_000 + seq, exposure_ns=0, width=size[0], height=size[1],
             encoding="h264", keyframe=False)
    d.update(change)
    return FrameInfo(**d)


def header(packet_seq=0, main=None, ultrawide=(), pair_seq=-1):
    return PacketHeader(packet_seq=packet_seq, main=main or info(), ultrawide=tuple(ultrawide), pair_seq=pair_seq)


def packet(n_ultrawide=1, pair_seq=None):
    uw = [info("imx219", 10 + i) for i in range(n_ultrawide)]
    pair = (uw[-1].frame_seq if uw else -1) if pair_seq is None else pair_seq
    data = tuple(b"uw%d" % i for i in range(len(uw)))
    return Packet(header(5, info(seq=5), uw, pair), b"\x00\x00\x01\x41main", data)


def test_field_order_is_the_wire_order():
    assert [f.name for f in dataclasses.fields(PacketHeader)] == ["packet_seq", "main", "ultrawide", "pair_seq"]
    assert [f.name for f in dataclasses.fields(FrameInfo)] == ["sensor", "frame_seq", "t_pi_ns", "exposure_ns",
                                                                "width", "height", "encoding", "keyframe"]
    assert list(header().to_dict()) == ["schema", "packet_seq", "main", "ultrawide", "pair_seq"]
    assert header().to_dict()["schema"] == "c0.packet/v1"


def test_good_headers_round_trip():
    for h in (header(), header(3, info(seq=3), [info("imx219", 1), info("imx219", 2)], 2),
              header(4, info(seq=4), [], 2), header(5, info(seq=5, encoding="jpeg", keyframe=True), [], -1)):
        assert decode(encode(h)) == h


@pytest.mark.parametrize("field,value", [("sensor", "imx477"), ("encoding", "png"), ("exposure_ns", -1),
                                         ("width", -1), ("height", 2**32), ("frame_seq", -1)])
def test_frame_map_enums_and_ranges(field, value):
    contract_error(lambda: info(**{field: value}), field, "value")


def test_uint32_size_and_uint64_counters():
    assert info(width=2**32 - 1).width == 2**32 - 1
    contract_error(lambda: header(packet_seq=-1), "packet_seq", "value")
    assert header(packet_seq=2**64 - 1).packet_seq == 2**64 - 1


def test_jpeg_must_be_a_keyframe():
    contract_error(lambda: info(encoding="jpeg", keyframe=False), "keyframe", "value")


def test_sensors_per_role():
    contract_error(lambda: header(main=info("imx219")), "main.sensor", "value")
    contract_error(lambda: header(ultrawide=[info("imx219", 1), info("imx519", 2)]), "ultrawide[1].sensor", "value")


def test_ultrawide_frames_in_capture_order_and_one_encoding():
    contract_error(lambda: header(ultrawide=[info("imx219", 2), info("imx219", 2)]), "ultrawide[1].frame_seq", "value")
    assert header(ultrawide=[info("imx219", 2), info("imx219", 5)], pair_seq=5).pair_seq == 5   # a Pi-side drop
    contract_error(lambda: header(ultrawide=[info("imx219", 1), info("imx219", 2, encoding="jpeg", keyframe=True)]),
                   "ultrawide[1].encoding", "value")


def test_pair_seq():
    assert header(pair_seq=-1).pair_seq == -1
    assert header(ultrawide=[], pair_seq=40).pair_seq == 40                    # from an earlier packet
    assert header(ultrawide=[info("imx219", 7), info("imx219", 8)], pair_seq=3).pair_seq == 3
    contract_error(lambda: header(pair_seq=-2), "pair_seq", "value")
    contract_error(lambda: header(ultrawide=[info("imx219", 7)], pair_seq=8), "pair_seq", "value")


def test_packet_parts_round_trip():
    for n in (0, 1, 2):
        p = packet(n)
        parts = p.to_parts()
        assert len(parts) == 2 + n
        assert parts[0] == encode(p.header) and parts[1] == p.main_data
        back = Packet.from_parts(parts)
        assert back == p
        assert back.main.info == p.header.main and back.main.data == p.main_data
        assert [f.data for f in back.ultrawide] == list(p.ultrawide_data)


def test_build_from_encoded_frames():
    main = EncodedFrame(info(seq=9), b"m")
    uw = [EncodedFrame(info("imx219", 4), b"a"), EncodedFrame(info("imx219", 5), b"b")]
    p = Packet.build(9, main, uw, 4)
    assert p.header.packet_seq == 9 and p.header.pair_seq == 4 and p.ultrawide_data == (b"a", b"b")
    assert p.ultrawide[1].info.frame_seq == 5


def part_error(parts, field, problem):
    with pytest.raises(ContractError) as e:
        Packet.from_parts(parts)
    assert (e.value.schema, e.value.field, e.value.problem) == ("c0.packet/v1", field, problem), str(e.value)
    return e.value


def test_wrong_part_count_refused():
    parts = packet(1).to_parts()
    e = part_error(parts[:2], "parts", "value")
    assert "expected 3" in str(e)
    part_error(parts + [b"extra"], "parts", "value")
    part_error(parts[:1], "parts", "value")
    part_error([], "parts", "value")
    part_error(packet(0).to_parts() + [b"x"], "parts", "value")


def test_empty_part_refused():
    parts = packet(2).to_parts()
    part_error([parts[0], b"", *parts[2:]], "parts[1]", "value")
    part_error([*parts[:3], b""], "parts[3]", "value")


def test_broken_header_refused_with_its_field():
    d = packet(1).header.to_dict()
    d["main"]["width"] = 1600.0
    part_error([pack_obj(d), b"m", b"u"], "main.width", "type")
    contract_error(lambda: Packet.from_parts([b"\xc1", b"m"]), "", "not_msgpack")
    contract_error(lambda: Packet.from_parts([encode(TimeSyncRequest()), b"m"]), "", "unexpected_message")


def test_time_sync():
    assert decode(encode(TimeSyncRequest())) == TimeSyncRequest()
    assert TimeSyncRequest().to_dict() == {"schema": "c0.time_sync_request/v1"}
    assert TimeSyncReply(t_pi_ns=5).to_dict() == {"schema": "c0.time_sync_reply/v1", "t_pi_ns": 5}
    assert decode(encode(TimeSyncReply(t_pi_ns=-5))).t_pi_ns == -5
    contract_error(lambda: TimeSyncReply(t_pi_ns=1.5), "t_pi_ns", "type")
