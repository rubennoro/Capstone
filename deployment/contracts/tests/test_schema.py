import pytest

from contracts import ContractError, SchemaName, decode, pack_obj
from contracts.registry import lookup


def test_parse_name_and_version():
    s = SchemaName.parse("c3.set_waypoints.reply/v1")
    assert (s.name, s.version) == ("c3.set_waypoints.reply", 1)
    assert str(s) == "c3.set_waypoints.reply/v1"
    assert SchemaName.parse("ack/v12").version == 12


@pytest.mark.parametrize("bad", ["", "c2.state", "c2.state/1", "c2.state/v0", "c2.state/v01", "C2.state/v1",
                                 "c2..state/v1", "/v1", "c2.state/v1 ", "c2.state/vx", 3, None, b"c2.state/v1"])
def test_malformed_refused(bad):
    with pytest.raises(ValueError):
        SchemaName.parse(bad)


def test_compare_names():
    a, b = SchemaName.parse("c2.state/v1"), SchemaName.parse("c2.state/v2")
    assert a.same_name(b) and a != b
    assert not a.same_name(SchemaName.parse("c3.stop/v1"))


def test_unknown_name_vs_version_mismatch():
    with pytest.raises(ContractError) as unknown:
        lookup("c9.teleport/v1")
    with pytest.raises(ContractError) as version:
        lookup("c2.state/v2")
    assert unknown.value.problem == "unknown_schema"
    assert version.value.problem == "schema_version"
    assert str(unknown.value) == "c9.teleport/v1: unknown message name 'c9.teleport'"
    assert str(version.value) == "c2.state/v2: version v2 not supported (this side knows v1)"


def test_malformed_schema_in_a_message_is_unknown_schema():
    with pytest.raises(ContractError) as info:
        decode(pack_obj({"schema": "garbage"}))
    assert info.value.problem == "unknown_schema"
    with pytest.raises(ContractError) as info:
        decode(pack_obj({"schema": 7}))
    assert info.value.problem == "unknown_schema" and info.value.schema == "?"
