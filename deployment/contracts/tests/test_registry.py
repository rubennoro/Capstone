from __future__ import annotations

from dataclasses import dataclass

import pytest

from contracts import (Ack, ArmState, ContractError, ErrorReply, Heartbeat, Message, SetWaypointsReply, decode,
                       encode, register)
from contracts.fields import scalar
from contracts.registry import lookup, registered
from helpers import contract_error

ALL_SCHEMAS = {
    "ack/v1", "error/v1",
    "c0.frame/v1", "c0.time_sync/v1", "c0.time_sync.reply/v1",
    "c1.get_frames/v1", "c1.frames/v1", "c1.get_health/v1", "c1.health/v1",
    "c2.state/v1",
    "c3.set_waypoints/v1", "c3.set_waypoints.reply/v1", "c3.heartbeat/v1", "c3.start/v1", "c3.stop/v1",
    "c3.home/v1", "c3.get_state/v1",
    "c4.config/v1", "c4.config.reply/v1", "c4.reset/v1", "c4.infer/v1", "c4.chunk/v1", "c4.prompt/v1",
}


def test_every_schema_maps_to_exactly_one_class():
    classes = registered()
    schemas = [c.SCHEMA for c in classes]
    assert len(schemas) == len(set(schemas))
    assert set(schemas) == ALL_SCHEMAS
    for cls in classes:
        assert lookup(cls.SCHEMA) is cls


def test_duplicate_registration_refused():
    @dataclass(frozen=True, eq=False)
    class OtherAck(Message):
        SCHEMA = "ack/v1"
        ok: bool = scalar("bool")

    with pytest.raises(ValueError):
        register(OtherAck)
    assert lookup("ack/v1") is Ack

    @dataclass(frozen=True, eq=False)
    class OtherAckV2(Message):  # another version of a known name is a duplicate too: one version at a time
        SCHEMA = "ack/v2"

    with pytest.raises(ValueError):
        register(OtherAckV2)


def test_register_needs_field_helpers():
    @dataclass(frozen=True, eq=False)
    class Plain(Message):
        SCHEMA = "test.plain/v1"
        x: int = 0

    with pytest.raises(TypeError):
        register(Plain)
    assert "test.plain/v1" not in {c.SCHEMA for c in registered()}


def test_decode_expect_refuses_another_valid_type():
    data = encode(Ack(ok=True, reason=""))
    contract_error(lambda: decode(data, ArmState), "", "unexpected_message")
    contract_error(lambda: decode(data, (ArmState, SetWaypointsReply)), "", "unexpected_message")
    assert isinstance(decode(data, (ArmState, Ack)), Ack)


def test_decode_expect_lets_error_reply_through():
    err = ErrorReply(request="c3.heartbeat/v1", error="boom", field="")
    assert decode(encode(err), Heartbeat) == err


def test_unexpected_message_names_both():
    with pytest.raises(ContractError) as info:
        decode(encode(Heartbeat()), Ack)
    assert "expected ack/v1" in str(info.value) and "c3.heartbeat/v1" in str(info.value)
