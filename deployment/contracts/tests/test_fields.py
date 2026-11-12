from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from contracts import ContractError, Record
from contracts.fields import array, list_of, nested, optional, scalar, str_list
from helpers import contract_error


@dataclass(frozen=True, eq=False)
class Leaf(Record):
    n: int = scalar("uint64")


@dataclass(frozen=True, eq=False)
class Everything(Record):
    i64: int = scalar("int64")
    u64: int = scalar("uint64")
    i: int = scalar("int", min=1, max=10)
    f: float = scalar("float64", min=0.0)
    b: bool = scalar("bool")
    s: str = scalar("str", choices=("a", "b"))
    raw: bytes = scalar("bytes")
    arr: np.ndarray = array("float32", (None, 3))
    leaf: Leaf = nested(Leaf)
    leaves: tuple = list_of(Leaf)
    names: tuple = str_list(choices=("x", "y"))
    maybe: np.ndarray | None = optional(array("uint8", (2,)))


GOOD = dict(i64=-5, u64=2**64 - 1, i=3, f=0.5, b=True, s="a", raw=b"\x00", arr=np.zeros((4, 3), np.float32),
            leaf=Leaf(n=1), leaves=(Leaf(n=2),), names=("x",), maybe=None)


def build(**change):
    return Everything(**{**GOOD, **change})


def test_good_passes():
    e = build()
    assert e.u64 == 2**64 - 1
    assert build(maybe=np.zeros(2, np.uint8)).maybe is not None
    assert build(arr=np.zeros((0, 3), np.float32)).arr.shape == (0, 3)
    assert build(i64=np.int64(3), u64=np.uint64(4)).to_dict()["i64"] == 3  # numpy integers are integers


@pytest.mark.parametrize("field,value,problem", [
    ("i64", "1", "type"), ("i64", 1.0, "type"), ("i64", 2**63, "value"), ("i64", -(2**63) - 1, "value"),
    ("u64", -1, "value"), ("u64", 2**64, "value"),
    ("i", 0, "value"), ("i", 11, "value"),
    ("f", 1, "type"), ("f", np.float32(0.5), "type"), ("f", -0.1, "value"), ("f", float("nan"), "value"),
    ("b", 1, "type"), ("b", "true", "type"),
    ("s", "c", "value"), ("s", b"a", "type"),
    ("raw", "x", "type"), ("raw", bytearray(b"x"), "type"),
    ("arr", [[0, 0, 0]], "type"), ("arr", np.zeros((4, 3), np.float64), "dtype"),
    ("arr", np.zeros((4, 2), np.float32), "shape"), ("arr", np.zeros(3, np.float32), "shape"),
    ("leaf", {"n": 1}, "type"), ("leaf", Everything, "type"),
    ("names", "x", "type"), ("maybe", np.zeros(3, np.uint8), "shape"),
])
def test_each_check_fails_with_its_problem(field, value, problem):
    contract_error(lambda: build(**{field: value}), field, problem)


def test_bool_is_not_an_int():
    contract_error(lambda: build(i64=True), "i64", "type")
    contract_error(lambda: build(u64=np.bool_(False)), "u64", "type")


def test_list_items_are_checked_with_their_index():
    contract_error(lambda: build(leaves=(Leaf(n=1), "x")), "leaves[1]", "type")
    contract_error(lambda: build(names=("x", 3)), "names[1]", "type")
    contract_error(lambda: build(names=("x", "z")), "names[1]", "value")


def test_lists_are_stored_as_tuples():
    e = build(leaves=[Leaf(n=1)], names=["y"])
    assert e.leaves == (Leaf(n=1),) and e.names == ("y",)


def test_optional_accepts_none_but_not_a_missing_key():
    d = build().to_dict()
    assert d["maybe"] is None
    assert Everything.from_dict(d).maybe is None
    del d["maybe"]
    contract_error(lambda: Everything.from_dict(d), "maybe", "missing")


def test_nested_errors_carry_the_path():
    d = build().to_dict()
    d["leaf"]["n"] = -1
    contract_error(lambda: Everything.from_dict(d), "leaf.n", "value")
    d = build().to_dict()
    d["leaves"][0]["n"] = "1"
    contract_error(lambda: Everything.from_dict(d), "leaves[0].n", "type")
    d = build().to_dict()
    d["leaves"][0]["extra"] = 1
    contract_error(lambda: Everything.from_dict(d), "leaves[0].extra", "unknown_field")


def test_record_errors_name_the_record():
    with pytest.raises(ContractError) as info:
        Leaf(n=-3)
    assert info.value.schema == "Leaf"
