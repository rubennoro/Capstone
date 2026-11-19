# adapted from rnoro/deployment:deployment/contracts/tests/test_registry.py @ 91b178b
# changes: imports renamed; importing deploy_contracts alone registers exactly SHARED_SCHEMAS (checked in a fresh
# interpreter)
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import pytest

from deploy_contracts import (SHARED_SCHEMAS, Ack, ArmState, ContractError, ErrorReply, Heartbeat, Message,
                              SetWaypointsReply, decode, encode, register)
from deploy_contracts.fields import scalar
from deploy_contracts.registry import lookup, registered
from helpers import contract_error


def test_every_shared_schema_maps_to_exactly_one_class():
    mine = [c for c in registered() if c.__module__.startswith("deploy_contracts.")]
    schemas = [c.SCHEMA for c in mine]
    assert len(schemas) == len(set(schemas))
    assert sorted(schemas) == list(SHARED_SCHEMAS)
    for cls in mine:
        assert lookup(cls.SCHEMA) is cls


def test_importing_deploy_contracts_alone_registers_exactly_the_shared_schemas():
    # a fresh interpreter: other tests in this process may register test classes into a patched registry
    code = ("import deploy_contracts, sys; from deploy_contracts.registry import registered; "
            "print('\\n'.join(sorted(c.SCHEMA for c in registered())))")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.split()
    assert out == list(SHARED_SCHEMAS)


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
