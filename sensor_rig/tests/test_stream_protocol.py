"""B4: wire protocol helpers (stdlib-only module; runs in the iphumi and real-env interpreters)."""
import base64

import pytest

from sensor_rig.stream import protocol as P


def good(**kw):
    d = dict(seq=1, main_jpeg=b"\xff\xd8a", uw_jpeg=b"\xff\xd8b", t_main_ns=10, t_uw_ns=11, clock="sender_monotonic", w=960, h=720)
    d.update(kw)
    return P.make_packet(**d)


def test_v1_roundtrip_is_str_without_newlines():
    raw = bytes(range(256)) * 50
    s = P.encode_v1(raw)
    assert isinstance(s, str) and "\n" not in s
    assert s == base64.b64encode(raw).decode()
    assert P.decode_v1(s) == raw


def test_v1_decode_rejects_bytes_and_garbage():
    with pytest.raises(TypeError):
        P.decode_v1(b"abcd")
    with pytest.raises(Exception):
        P.decode_v1("not base64 !!")


def test_packet_valid():
    assert P.check_packet(good()) == []


def test_packet_missing_and_extra():
    d = good()
    del d["seq"]
    assert any("missing field: seq" in e for e in P.check_packet(d))
    d = good()
    d["foo"] = 1
    assert any("unexpected field: foo" in e for e in P.check_packet(d))


@pytest.mark.parametrize("field,val", [("seq", "1"), ("seq", True), ("main_jpeg", "str"), ("t_main_ns", 1.5), ("w", None), ("clock", 3)])
def test_packet_wrong_types(field, val):
    d = good()
    d[field] = val
    assert P.check_packet(d)


def test_packet_bad_values():
    assert P.check_packet(good(clock="wall"))
    assert P.check_packet(good(w=0))
    assert P.check_packet(good(seq=-1))
    d = good()
    d["v"] = 1
    assert P.check_packet(d)
    assert P.check_packet("nope")


def test_hello():
    h = P.make_hello("rig", "v2", {"main": {"w": 960}}, 30.0, {"main": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    assert P.check_hello(h) == []
    del h["K"]
    assert P.check_hello(h)


def test_constants():
    assert P.EVENT_RGB == "rgb" and P.EVENT_VALIDATE == "validate" and P.V1_PORTS == {"main": 5555, "ultrawide": 5556}
