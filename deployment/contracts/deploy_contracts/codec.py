# adapted from rnoro/deployment:deployment/contracts/contracts/codec.py @ 91b178b
# changes: imports renamed; decode() takes an optional keyword `schemas` (the schema filter of STRUCTURE.md 3.11).
"""The wire format: msgpack maps, arrays as {"__nd__", "dtype", "shape", "data"} (deployment/STRUCTURE.md 3.3)."""
from __future__ import annotations

import math
from typing import Any, Collection, TypeVar

import msgpack
import numpy as np

from deploy_contracts.errors import ContractError
from deploy_contracts.message import Message
from deploy_contracts.registry import lookup
from deploy_contracts.schema import SchemaName

M = TypeVar("M", bound=Message)

WIRE_DTYPES = ("uint8", "int64", "uint64", "float32", "float64")
_ARRAY_KEYS = ("__nd__", "dtype", "shape", "data")


def _fail(problem: str, detail: str) -> ContractError:
    return ContractError("?", "", problem, detail)


def pack_array(a: np.ndarray) -> dict:
    """One array -> its map form: C order, little-endian bytes (copied only if needed)."""
    if not isinstance(a, np.ndarray):
        raise TypeError(f"pack_array needs a numpy array, got {type(a).__name__}")
    name = a.dtype.name
    if name not in WIRE_DTYPES:
        raise ValueError(f"dtype {name} cannot be sent; allowed: {', '.join(WIRE_DTYPES)}")
    le = np.asarray(a, dtype=np.dtype(name).newbyteorder("<"), order="C")
    return {"__nd__": True, "dtype": name, "shape": [int(d) for d in a.shape], "data": le.tobytes()}


def unpack_array(d: Any) -> np.ndarray:
    """Map form -> a read-only array sharing the received bytes. ContractError (type, dtype, shape) if malformed."""
    if not isinstance(d, dict):
        raise _fail("type", f"expected an array map, got {type(d).__name__}")
    if set(d) != set(_ARRAY_KEYS) or len(d) != len(_ARRAY_KEYS):
        raise _fail("type", f"an array map has exactly the keys {', '.join(_ARRAY_KEYS)}; got {', '.join(map(str, d))}")
    if d["__nd__"] is not True:
        raise _fail("type", "an array map needs __nd__: true")
    name = d["dtype"]
    if not isinstance(name, str) or name not in WIRE_DTYPES:
        raise _fail("dtype", f"array dtype {name!r} is not one of {', '.join(WIRE_DTYPES)}")
    shape = d["shape"]
    if not isinstance(shape, list) or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0
                                              for n in shape):
        raise _fail("shape", f"array shape must be a list of non-negative integers, got {shape!r}")
    data = d["data"]
    if not isinstance(data, bytes):
        raise _fail("type", f"array data must be bin, got {type(data).__name__}")
    dtype = np.dtype(name).newbyteorder("<")
    need = math.prod(shape) * dtype.itemsize
    if len(data) != need:
        raise _fail("shape", f"array data has {len(data)} bytes; shape {tuple(shape)} {name} needs {need}")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return pack_array(obj)
    raise TypeError(f"cannot encode {type(obj).__name__}")


def pack_obj(obj: Any) -> bytes:
    """Low-level packer: dicts, lists, plain scalars and numpy arrays (as array maps)."""
    return msgpack.packb(obj, use_bin_type=True, default=_default)


def _unpack_raw(data: Any) -> Any:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise _fail("not_msgpack", f"expected bytes, got {type(data).__name__}")
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=True)
    except (ValueError, TypeError, msgpack.UnpackException) as e:
        reason = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        raise _fail("not_msgpack", f"not msgpack ({reason})") from None


def _arrays_in(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return unpack_array(obj)
        return {k: _arrays_in(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_arrays_in(v) for v in obj]
    return obj


def unpack_obj(data: bytes) -> Any:
    """Low-level unpacker: msgpack -> plain Python, with every array map turned into an array. No contract checks."""
    return _arrays_in(_unpack_raw(data))


def encode(msg: Message) -> bytes:
    """A message -> its canonical msgpack bytes."""
    if not isinstance(msg, Message):
        raise TypeError(f"encode needs a Message, got {type(msg).__name__}")
    return pack_obj(msg.to_dict())


def _expected(expect: type[Message] | tuple[type[Message], ...]) -> tuple[type[Message], ...]:
    return expect if isinstance(expect, tuple) else (expect,)


def schema_names(schemas: Collection[str]) -> frozenset[str]:
    """The message names of a schema list ("c3.hello/v1" -> "c3.hello"). ValueError for a malformed entry."""
    if isinstance(schemas, str):
        raise TypeError("schemas must be a collection of schema strings, not one string")
    return frozenset(SchemaName.parse(s).name for s in schemas)


def check_schema_allowed(schema: Any, names: frozenset[str]) -> None:
    """The schema filter (STRUCTURE.md 3.11): unknown_schema when the message name is not one of `names`.

    It compares names, not versions, exactly as franka_control's codec does: a known name with another version passes
    here and is refused by lookup() as schema_version. A malformed schema passes here too; lookup() refuses it as
    unknown_schema. So the filter changes no problem code except for a name outside `names`.
    """
    try:
        parsed = SchemaName.parse(schema)
    except ValueError:
        return
    if parsed.name not in names:
        raise ContractError(schema, "", "unknown_schema", f"unknown message name '{parsed.name}'")


def decode(data: bytes, expect: type[M] | tuple[type[Message], ...] | None = None, *,
           schemas: Collection[str] | None = None) -> Message:
    """Bytes -> a validated message. Checks run in the fixed order of STRUCTURE.md section 3.3.

    With `expect`, a valid message of another type raises ContractError(problem="unexpected_message"), except
    ErrorReply, which always passes so callers can raise RemoteError.

    With `schemas`, a message whose name is not among those schemas' names is unknown_schema even when it is
    registered in this process (step 2, after no_schema and before the registry lookup; STRUCTURE.md 3.11).
    """
    from deploy_contracts.common import ErrorReply  # local import: common registers its classes through this package

    names = None if schemas is None else schema_names(schemas)
    raw = _unpack_raw(data)
    if not isinstance(raw, dict):
        raise _fail("not_a_map", f"expected a map, got {type(raw).__name__}")
    if "schema" not in raw:
        raise _fail("no_schema", "no 'schema' key")
    if names is not None:
        check_schema_allowed(raw["schema"], names)
    cls = lookup(raw["schema"])
    msg = cls.from_dict(raw)
    if expect is not None:
        wanted = _expected(expect)
        if not isinstance(msg, wanted) and not isinstance(msg, ErrorReply):
            names_text = ", ".join(c.SCHEMA for c in wanted)
            raise ContractError(msg.SCHEMA, "", "unexpected_message", f"expected {names_text}, got {msg.SCHEMA}")
    return msg
