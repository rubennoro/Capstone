# adapted from rnoro/deployment:deployment/contracts/tests/test_codec.py @ 91b178b
# changes: imports renamed; canonical-encoding check over the stored (C2/C3) examples only; + hello reply wire
# layout
from __future__ import annotations

import msgpack
import numpy as np
import pytest

from deploy_contracts import ContractError, decode, encode, load_index, pack_array, pack_obj, unpack_array, unpack_obj
from helpers import contract_error

DTYPES = ["uint8", "int64", "uint64", "float32", "float64"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (5,), (2, 3, 4, 2)])
def test_array_round_trip(dtype, shape):
    a = (np.arange(int(np.prod(shape)), dtype=np.int64) % 100).astype(dtype).reshape(shape)
    d = pack_array(a)
    assert list(d) == ["__nd__", "dtype", "shape", "data"]
    assert d["__nd__"] is True and d["dtype"] == dtype and d["shape"] == list(shape)
    assert len(d["data"]) == a.size * a.itemsize
    b = unpack_array(d)
    assert b.dtype == np.dtype(dtype) and b.shape == shape and np.array_equal(a, b)


def test_non_contiguous_and_big_endian_are_normalised():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    t = a.T  # not C-contiguous
    assert pack_array(t)["data"] == np.ascontiguousarray(t).tobytes()
    big = a.astype(">f8")
    assert pack_array(big)["data"] == a.astype("<f8").tobytes()
    assert pack_array(big)["dtype"] == "float64"
    assert np.array_equal(unpack_array(pack_array(t)), t)


def test_little_endian_bytes():
    assert pack_array(np.array([1], dtype=np.uint64))["data"] == b"\x01" + b"\x00" * 7


def test_byte_count_mismatch_refused():
    d = pack_array(np.zeros((4, 4)))
    d["data"] = d["data"][:-1]
    with pytest.raises(ContractError) as info:
        unpack_array(d)
    assert info.value.problem == "shape"


@pytest.mark.parametrize("change,problem", [
    (lambda d: d.update(__nd__=False), "type"), (lambda d: d.pop("data"), "type"),
    (lambda d: d.update(dtype="int32"), "dtype"), (lambda d: d.update(shape=[-1]), "shape"),
    (lambda d: d.update(shape="4"), "shape"), (lambda d: d.update(data="text"), "type"),
])
def test_malformed_array_maps(change, problem):
    d = pack_array(np.zeros(4))
    change(d)
    with pytest.raises(ContractError) as info:
        unpack_array(d)
    assert info.value.problem == problem


def test_unsupported_dtype_cannot_be_packed():
    with pytest.raises(ValueError):
        pack_array(np.zeros(2, dtype=np.int32))


def test_decoded_arrays_are_read_only_and_share_the_bytes():
    data = pack_obj({"a": np.arange(6, dtype=np.float32)})
    a = unpack_obj(data)["a"]
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0] = 1.0
    b = a.copy()
    b[0] = 1.0  # a copy is writeable
    base = a
    while isinstance(base, np.ndarray):
        base = base.base
    assert isinstance(base, bytes)  # a view of the received bytes, not a copy


def test_float64_fields_encoded_as_float64():
    data = pack_obj({"x": 0.5})
    assert data[-9] == 0xCB  # msgpack float64 marker, never float32 (0xCA)


def _good_examples():
    return [e for e in load_index() if e.valid]


@pytest.mark.parametrize("example", _good_examples(), ids=lambda e: e.name)
def test_encode_decode_is_canonical(example):
    assert encode(decode(example.data)) == example.data


def test_not_msgpack_and_not_a_map():
    contract_error(lambda: decode(b"\xc1"), "", "not_msgpack")
    contract_error(lambda: decode(b""), "", "not_msgpack")
    contract_error(lambda: decode(msgpack.packb({"schema": "ack/v1", "ok": True, "reason": ""}) + b"\x00"), "",
                   "not_msgpack")
    contract_error(lambda: decode("text"), "", "not_msgpack")  # type: ignore[arg-type]
    contract_error(lambda: decode(msgpack.packb([1, 2])), "", "not_a_map")
    contract_error(lambda: decode(msgpack.packb(3)), "", "not_a_map")
    contract_error(lambda: decode(msgpack.packb({1: 2})), "", "not_msgpack")  # strict_map_key: str keys only
    contract_error(lambda: decode(msgpack.packb({"ok": True})), "", "no_schema")


def test_hello_reply_wire_layout():
    from deploy_contracts import SHARED_SCHEMAS
    raw = unpack_obj(next(e for e in load_index() if e.name == "c3/hello_reply").data)
    assert list(raw) == ["schema", "t_pc_ns", "commit", "mock", "schemas", "F_T_TCP", "rotation_tol",
                         "gripper_width_max_m", "max_waypoints", "watchdog_limit_s", "homing_timeout_s", "max_lead_s"]
    assert raw["schemas"] == list(SHARED_SCHEMAS)
    assert raw["F_T_TCP"].dtype == np.float64 and raw["F_T_TCP"].shape == (4, 4)
