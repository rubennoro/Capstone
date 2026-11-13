"""Field declarations for message classes: scalar(), array(), nested(), list_of(), str_list(), optional().

Each helper returns a dataclass field carrying a FieldSpec. The validator reads those specs, so the field table and
the checks can never drift apart (STRUCTURE.md section 3).

A spec raises ContractError with schema "?" and a field path relative to itself ("" for the field itself, "[2]" for
a list item, "main.status" inside a sub-message). The message class re-raises it with its own schema and the field
name in front.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from contracts.errors import ContractError

SPEC_KEY = "contract_spec"

_INT_RANGES = {
    "int64": (-(2**63), 2**63 - 1),
    "int": (-(2**63), 2**63 - 1),
    "uint64": (0, 2**64 - 1),
    "uint32": (0, 2**32 - 1),
}
SCALAR_KINDS = ("int64", "uint64", "uint32", "int", "float64", "bool", "str", "bytes")
ARRAY_DTYPES = ("uint8", "int64", "uint64", "float32", "float64")


def _fail(problem: str, detail: str, field: str = "") -> ContractError:
    return ContractError("?", field, problem, detail)


def _shape_text(shape: tuple[int | None, ...]) -> str:
    parts = ["N" if d is None else str(d) for d in shape]
    return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"


class FieldSpec:
    """How one field is checked, read from the wire and turned into plain data."""

    def check(self, value: Any) -> None:
        """Raise ContractError if value breaks the spec: type first, then dtype, shape, allowed values."""
        raise NotImplementedError

    def from_wire(self, raw: Any) -> Any:
        """Turn the decoded msgpack value into the Python value (arrays, sub-messages, tuples)."""
        return raw

    def normalize(self, value: Any) -> Any:
        """What the constructor stores (lists become tuples; nothing else is converted)."""
        return value

    def to_plain(self, value: Any) -> Any:
        """The value as to_dict() gives it: plain Python scalars, arrays stay arrays, sub-messages as dicts."""
        return value

    def describe(self) -> str:
        raise NotImplementedError


class ScalarSpec(FieldSpec):
    def __init__(self, kind: str, choices: tuple | None = None, min: float | None = None,  # noqa: A002
                 max: float | None = None) -> None:  # noqa: A002
        if kind not in SCALAR_KINDS:
            raise ValueError(f"unknown scalar kind {kind!r}")
        self.kind = kind
        self.choices = tuple(choices) if choices is not None else None
        self.min = min
        self.max = max

    def describe(self) -> str:
        return self.kind

    def _type_ok(self, v: Any) -> bool:
        if self.kind in _INT_RANGES:
            return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        if self.kind == "float64":
            return isinstance(v, float)  # includes np.float64; not int, not np.float32
        if self.kind == "bool":
            return isinstance(v, (bool, np.bool_))
        if self.kind == "str":
            return isinstance(v, str)
        return isinstance(v, bytes)

    def check(self, value: Any) -> None:
        if not self._type_ok(value):
            raise _fail("type", f"expected {self.kind}, got {type(value).__name__}")
        if self.kind in _INT_RANGES:
            lo, hi = _INT_RANGES[self.kind]
            if not lo <= int(value) <= hi:
                raise _fail("value", f"{int(value)} is out of the {self.kind} range")
        if self.kind == "float64" and math.isnan(value):
            raise _fail("value", "nan is not a number")
        if self.choices is not None and value not in self.choices:
            raise _fail("value", f"{value!r} is not one of {', '.join(map(str, self.choices))}")
        if self.min is not None and value < self.min:
            raise _fail("value", f"{value!r} is below the minimum {self.min}")
        if self.max is not None and value > self.max:
            raise _fail("value", f"{value!r} is above the maximum {self.max}")

    def to_plain(self, value: Any) -> Any:
        if self.kind in _INT_RANGES:
            return int(value)
        if self.kind == "float64":
            return float(value)
        if self.kind == "bool":
            return bool(value)
        return value


class ArraySpec(FieldSpec):
    def __init__(self, dtype: str, shape: tuple[int | None, ...]) -> None:
        if dtype not in ARRAY_DTYPES:
            raise ValueError(f"unknown array dtype {dtype!r}")
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)

    def describe(self) -> str:
        return f"{self.dtype.name} {_shape_text(self.shape)}"

    def check(self, value: Any) -> None:
        if not isinstance(value, np.ndarray):
            raise _fail("type", f"expected a numpy array, got {type(value).__name__}")
        if value.dtype != self.dtype:
            raise _fail("dtype", f"dtype {value.dtype}, expected {self.dtype.name}")
        if len(value.shape) != len(self.shape) or any(
                want is not None and got != want for got, want in zip(value.shape, self.shape)):
            raise _fail("shape", f"shape {tuple(value.shape)}, expected {_shape_text(self.shape)}")

    def from_wire(self, raw: Any) -> Any:
        if isinstance(raw, np.ndarray):  # from_dict(to_dict()): arrays stay arrays
            return raw
        from contracts.codec import unpack_array  # local import: codec sits above fields in the import order
        return unpack_array(raw)


class NestedSpec(FieldSpec):
    def __init__(self, cls: type) -> None:
        self.cls = cls

    def describe(self) -> str:
        return self.cls.__name__

    def check(self, value: Any) -> None:
        if not isinstance(value, self.cls):
            raise _fail("type", f"expected {self.cls.__name__}, got {type(value).__name__}")

    def from_wire(self, raw: Any) -> Any:
        if not isinstance(raw, dict):
            raise _fail("type", f"expected a map ({self.cls.__name__}), got {type(raw).__name__}")
        return self.cls.from_dict(raw)

    def to_plain(self, value: Any) -> Any:
        return value.to_dict()


class ListOfSpec(FieldSpec):
    def __init__(self, cls: type) -> None:
        self.cls = cls

    def describe(self) -> str:
        return f"list of {self.cls.__name__}"

    def check(self, value: Any) -> None:
        if not isinstance(value, (tuple, list)):
            raise _fail("type", f"expected a list of {self.cls.__name__}, got {type(value).__name__}")
        for i, item in enumerate(value):
            if not isinstance(item, self.cls):
                raise _fail("type", f"expected {self.cls.__name__}, got {type(item).__name__}", f"[{i}]")

    def from_wire(self, raw: Any) -> Any:
        if not isinstance(raw, list):
            raise _fail("type", f"expected a list of {self.cls.__name__}, got {type(raw).__name__}")
        items = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise _fail("type", f"expected a map ({self.cls.__name__}), got {type(item).__name__}", f"[{i}]")
            try:
                items.append(self.cls.from_dict(item))
            except ContractError as e:
                raise e.within("?", f"[{i}]") from None
        return tuple(items)

    def normalize(self, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    def to_plain(self, value: Any) -> Any:
        return [item.to_dict() for item in value]


class StrListSpec(FieldSpec):
    def __init__(self, choices: tuple[str, ...] | None = None) -> None:
        self.choices = tuple(choices) if choices is not None else None

    def describe(self) -> str:
        return "list of str"

    def check(self, value: Any) -> None:
        if not isinstance(value, (tuple, list)):
            raise _fail("type", f"expected a list of str, got {type(value).__name__}")
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise _fail("type", f"expected str, got {type(item).__name__}", f"[{i}]")
            if self.choices is not None and item not in self.choices:
                raise _fail("value", f"{item!r} is not one of {', '.join(self.choices)}", f"[{i}]")

    def from_wire(self, raw: Any) -> Any:
        if not isinstance(raw, list):
            raise _fail("type", f"expected a list of str, got {type(raw).__name__}")
        return tuple(raw)

    def normalize(self, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    def to_plain(self, value: Any) -> Any:
        return list(value)


class OptionalSpec(FieldSpec):
    def __init__(self, inner: FieldSpec) -> None:
        self.inner = inner

    def describe(self) -> str:
        return f"optional {self.inner.describe()}"

    def check(self, value: Any) -> None:
        if value is not None:
            self.inner.check(value)

    def from_wire(self, raw: Any) -> Any:
        return None if raw is None else self.inner.from_wire(raw)

    def normalize(self, value: Any) -> Any:
        return None if value is None else self.inner.normalize(value)

    def to_plain(self, value: Any) -> Any:
        return None if value is None else self.inner.to_plain(value)


def _field(spec: FieldSpec) -> Any:
    return dataclasses.field(metadata={SPEC_KEY: spec})


def _spec_of(x: Any) -> FieldSpec:
    """optional() takes either a spec or the field another helper returned."""
    if isinstance(x, FieldSpec):
        return x
    if isinstance(x, dataclasses.Field) and SPEC_KEY in x.metadata:
        return x.metadata[SPEC_KEY]
    raise TypeError(f"optional() needs a field helper's result, got {type(x).__name__}")


def scalar(kind: str, choices: tuple | None = None, min: float | None = None,  # noqa: A002
           max: float | None = None) -> Any:  # noqa: A002
    """One value: kind is int64, uint64, uint32, int, float64, bool, str or bytes."""
    return _field(ScalarSpec(kind, choices, min, max))


def array(dtype: str, shape: tuple[int | None, ...]) -> Any:
    """A numpy array of exactly this dtype; None in shape means any length."""
    return _field(ArraySpec(dtype, shape))


def nested(cls: type) -> Any:
    """A sub-message (a Record): a map on the wire with no schema of its own."""
    return _field(NestedSpec(cls))


def list_of(cls: type) -> Any:
    """A list of sub-messages; stored as a tuple."""
    return _field(ListOfSpec(cls))


def str_list(choices: tuple[str, ...] | None = None) -> Any:
    """A list of strings, each one of choices when given; stored as a tuple."""
    return _field(StrListSpec(choices))


def optional(spec: Any) -> Any:
    """The field may be None (msgpack nil). It is never left out."""
    return _field(OptionalSpec(_spec_of(spec)))
