from __future__ import annotations

import time

import numpy as np
import pytest

from contracts import (Ack, ArmState, ContractError, ErrorReply, FrameInfo, GateCounters, Packet, PacketHeader,
                       PacketPublisher, PacketSubscriber, Publisher, Subscriber, encode, pack_obj)

ENDPOINT = "tcp://127.0.0.1:*"


def state(seq):
    return ArmState(t_ns=seq, seq=seq, O_T_TCP=np.eye(4), O_T_TCP_target=np.eye(4), q=np.zeros(7), dq=np.zeros(7),
                    gripper_width_m=0.0, mode="idle", error="",
                    gate=GateCounters(accepted=0, shortened=0, pulled_back=0, rejected=0), loop_late=0)


@pytest.fixture
def pair():
    pub = Publisher(ENDPOINT, send_hwm=1000)
    sub = Subscriber("franka_control", pub.endpoint, ArmState, recv_hwm=1000)
    # wait until the subscription has reached the publisher (no sleep: probe with a timeout on a condition)
    deadline = time.monotonic() + 5.0
    while True:
        pub.publish(state(0))
        if sub.receive(0.02) is not None:
            break
        assert time.monotonic() < deadline, "subscriber never connected"
    while sub.receive(0.05) is not None:  # drain the remaining probes
        pass
    yield pub, sub
    sub.close()
    pub.close()


def test_messages_arrive_in_order(pair):
    pub, sub = pair
    for seq in range(1, 21):
        pub.publish(state(seq))
    got = [sub.receive(2.0) for _ in range(20)]
    assert [m.seq for m in got] == list(range(1, 21))


def test_receive_returns_none_after_the_timeout(pair):
    _, sub = pair
    t0 = time.monotonic()
    assert sub.receive(0.1) is None
    assert 0.09 <= time.monotonic() - t0 < 1.0


def test_broken_message_raises(pair):
    pub, sub = pair
    d = state(5).to_dict()
    d["mode"] = "dancing"
    pub._socket.send(pack_obj(d))  # bypass the class: it would refuse to build this
    with pytest.raises(ContractError) as info:
        sub.receive(2.0)
    assert (info.value.field, info.value.problem) == ("mode", "value")
    pub.publish(state(6))
    assert sub.receive(2.0).seq == 6  # the subscriber still works


def test_message_of_another_type_raises(pair):
    pub, sub = pair
    pub.publish(Ack(ok=True, reason=""))
    with pytest.raises(ContractError) as info:
        sub.receive(2.0)
    assert info.value.problem == "unexpected_message"
    pub._socket.send(encode(ErrorReply(request="", error="x", field="")))  # never expected by a subscriber
    with pytest.raises(ContractError):
        sub.receive(2.0)


def test_publish_never_blocks_without_subscribers():
    pub = Publisher(ENDPOINT, send_hwm=2)
    try:
        t0 = time.monotonic()
        for seq in range(200):
            pub.publish(state(seq))
        assert time.monotonic() - t0 < 2.0
    finally:
        pub.close()


# ------------------------------------------------------------------------------------------------ C0 packets

def packet(seq, n_ultrawide=1):
    main = FrameInfo(sensor="imx519", frame_seq=seq, t_pi_ns=1_000 + seq, exposure_ns=0, width=1600, height=1200,
                     encoding="h264", keyframe=seq == 0)
    uw = tuple(FrameInfo(sensor="imx219", frame_seq=seq * 2 + i, t_pi_ns=990 + seq + i, exposure_ns=0, width=1280,
                         height=960, encoding="h264", keyframe=False) for i in range(n_ultrawide))
    header = PacketHeader(packet_seq=seq, main=main, ultrawide=uw, pair_seq=uw[-1].frame_seq if uw else -1)
    return Packet(header, b"main%d" % seq, tuple(b"uw%d" % f.frame_seq for f in uw))


@pytest.fixture
def packet_pair():
    pub = PacketPublisher(ENDPOINT, send_hwm=1000)
    sub = PacketSubscriber("pi", pub.endpoint, recv_hwm=1000)
    deadline = time.monotonic() + 5.0
    while True:
        pub.publish(packet(0))
        if sub.receive(0.02) is not None:
            break
        assert time.monotonic() < deadline, "subscriber never connected"
    while sub.receive(0.05) is not None:
        pass
    yield pub, sub
    sub.close()
    pub.close()


def test_packets_arrive_whole_and_in_order(packet_pair):
    pub, sub = packet_pair
    sent = [packet(seq, n_ultrawide=seq % 3) for seq in range(1, 16)]       # 1, 2, 0, 1, 2, ... ultrawide frames
    for p in sent:
        pub.publish(p)
    got = [sub.receive(2.0) for _ in sent]
    assert got == sent
    assert [len(p.ultrawide_data) for p in got[:3]] == [1, 2, 0]


def test_packet_receive_returns_none_after_the_timeout(packet_pair):
    _, sub = packet_pair
    assert sub.receive(0.1) is None


def test_broken_packet_raises_and_the_subscriber_keeps_working(packet_pair):
    pub, sub = packet_pair
    parts = packet(3, n_ultrawide=2).to_parts()
    pub._socket.send_multipart(parts[:-1])                  # one part short
    with pytest.raises(ContractError) as info:
        sub.receive(2.0)
    assert (info.value.field, info.value.problem) == ("parts", "value")
    pub.publish(packet(4))
    assert sub.receive(2.0).header.packet_seq == 4


def test_packet_publish_never_blocks_without_subscribers():
    pub = PacketPublisher(ENDPOINT, send_hwm=2)
    try:
        t0 = time.monotonic()
        for seq in range(200):
            pub.publish(packet(seq))
        assert time.monotonic() - t0 < 2.0
    finally:
        pub.close()
