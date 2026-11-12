"""The wire format: msgpack maps, arrays as {"__nd__", "dtype", "shape", "data"} (STRUCTURE.md section 5)."""
from __future__ import annotations

import math
from typing import Any, TypeVar

import msgpack
import numpy as np

from contracts.errors import ContractError
from contracts.message import Message
from contracts.registry import lookup

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


def decode(data: bytes, expect: type[M] | tuple[type[Message], ...] | None = None) -> Message:
    """Bytes -> a validated message. Checks run in the fixed order of STRUCTURE.md section 5.

    With `expect`, a valid message of another type raises ContractError(problem="unexpected_message"), except
    ErrorReply, which always passes so callers can raise RemoteError.
    """
    from contracts.common import ErrorReply  # local import: common registers its classes through this package

    raw = _unpack_raw(data)
    if not isinstance(raw, dict):
        raise _fail("not_a_map", f"expected a map, got {type(raw).__name__}")
    if "schema" not in raw:
        raise _fail("no_schema", "no 'schema' key")
    cls = lookup(raw["schema"])
    msg = cls.from_dict(raw)
    if expect is not None:
        wanted = _expected(expect)
        if not isinstance(msg, wanted) and not isinstance(msg, ErrorReply):
            names = ", ".join(c.SCHEMA for c in wanted)
            raise ContractError(msg.SCHEMA, "", "unexpected_message", f"expected {names}, got {msg.SCHEMA}")
    return msg
