"""Schema string -> message class, filled by the @register decorator (STRUCTURE.md sections 5 and 7)."""
from __future__ import annotations

from typing import TypeVar

from contracts.errors import ContractError
from contracts.message import Message
from contracts.schema import SchemaName

M = TypeVar("M", bound=type)

_BY_NAME: dict[str, type[Message]] = {}


def register(cls: M) -> M:
    """Class decorator: register a Message class under its SCHEMA. One class per schema name."""
    if not (isinstance(cls, type) and issubclass(cls, Message)):
        raise TypeError("@register is for Message subclasses")
    name = SchemaName.parse(cls.SCHEMA)
    other = _BY_NAME.get(name.name)
    if other is not None:
        raise ValueError(f"schema {name.name} is already registered by {other.__name__}; "
                         f"cannot register {cls.__name__}")
    cls.field_specs()  # every field declared with a helper, or TypeError now
    _BY_NAME[name.name] = cls
    return cls


def lookup(schema: object) -> type[Message]:
    """The class for a schema string. ContractError unknown_schema (malformed or unknown name) or schema_version."""
    label = schema if isinstance(schema, str) else "?"
    try:
        parsed = SchemaName.parse(schema)
    except ValueError as e:
        raise ContractError(label, "", "unknown_schema", str(e)) from None
    cls = _BY_NAME.get(parsed.name)
    if cls is None:
        raise ContractError(label, "", "unknown_schema", f"unknown message name '{parsed.name}'")
    known = SchemaName.parse(cls.SCHEMA)
    if parsed.version != known.version:
        raise ContractError(label, "", "schema_version",
                            f"version v{parsed.version} not supported (this side knows v{known.version})")
    return cls


def registered() -> tuple[type[Message], ...]:
    """Every registered message class, in registration order."""
    return tuple(_BY_NAME.values())
