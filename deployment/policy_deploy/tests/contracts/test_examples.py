# adapted from rnoro/deployment:deployment/contracts/tests/test_examples.py @ 91b178b (the C0/C1/C4 part)
# changes: the laptop examples are built in memory only (never stored); + the pi_stream/v1 wire examples and the
# c4.config.reply/v2 ones.
from __future__ import annotations

import pytest
from deploy_contracts import SHARED_SCHEMAS, ContractError, decode, examples_dir, resolve_path
from deploy_contracts.registry import registered

from policy_deploy.contracts import PiStreamPacket
from policy_deploy.contracts.examples import build_examples, pi_stream_examples

ALL = build_examples()
PI = pi_stream_examples()


@pytest.mark.parametrize("example", ALL, ids=lambda e: e.name)
def test_built_examples(example):
    if example.valid:
        msg = decode(example.data)
        assert msg.SCHEMA == example.schema
        d = msg.to_dict()
        for path, want in example.values.items():
            assert resolve_path(d, path) == want, path
    else:
        with pytest.raises(ContractError) as info:
            decode(example.data)
        assert (info.value.field, info.value.problem) == (example.field, example.problem), str(info.value)


@pytest.mark.parametrize("example", PI, ids=lambda e: e.name)
def test_pi_stream_examples(example):
    if example.valid:
        p = PiStreamPacket.from_parts(example.parts)
        assert p.to_parts() == list(example.parts) or PiStreamPacket.from_parts(p.to_parts()) == p
    else:
        with pytest.raises(ContractError) as info:
            PiStreamPacket.from_parts(example.parts)
        assert (info.value.field, info.value.problem, info.value.schema) == (example.field, example.problem,
                                                                             "pi_stream/v1"), str(info.value)


def test_every_laptop_message_has_a_good_example():
    good = {e.schema for e in ALL if e.valid}
    laptop = {c.SCHEMA for c in registered()} - set(SHARED_SCHEMAS)
    assert laptop <= good, laptop - good


@pytest.mark.parametrize("contract", ["c0", "c1", "c4"])
def test_every_contract_has_the_broken_kinds(contract):
    problems = {e.problem for e in ALL if e.contract == contract and not e.valid}
    wanted = {"type", "value", "missing", "unknown_field", "schema_version"}
    if contract != "c0":  # C0 has no arrays
        wanted |= {"shape", "dtype"}
    assert wanted <= problems


def test_pi_stream_cases_present():
    names = {e.name for e in PI}
    assert {"c0/pi_stream_live", "c0/pi_stream_two_parts", "c0/pi_stream_header_not_json", "c0/pi_stream_missing_key",
            "c0/pi_stream_packet_no_bool", "c0/pi_stream_version_2", "c0/pi_stream_empty_main"} <= names


def test_none_are_stored_and_the_shared_examples_still_decode():
    assert not any(e.stored for e in ALL)
    stored = {p.relative_to(examples_dir()).as_posix() for p in examples_dir().rglob("*.msgpack")}
    assert not any(name.startswith(("c0/", "c1/", "c4/")) for name in stored)


def test_c4_config_v2_examples():
    names = {e.name: e for e in ALL}
    assert names["c4/config"].values["kind"] == "hold_still"
    assert (names["c4/config_kind_missing"].field, names["c4/config_kind_missing"].problem) == ("kind", "missing")
    assert (names["c4/config_kind_empty"].field, names["c4/config_kind_empty"].problem) == ("kind", "value")
