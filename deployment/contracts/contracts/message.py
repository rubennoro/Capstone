"""Message and Record: the base classes of every contract message (STRUCTURE.md section 3).

A Message is a frozen dataclass whose fields are declared with the helpers in fields.py. Building one validates it,
so a Message object that exists is valid. A Record is a sub-message: the same field helpers and validation, no
`schema` key.
"""
from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, NoReturn, TypeVar

from contracts.errors import ContractError
from contracts.fields import SPEC_KEY, FieldSpec
from contracts.schema import SchemaName

T = TypeVar("T", bound="_Fields")


class _Fields:
    """What Message and Record share: the field table, validation, to_dict and from_dict."""

    _specs_cache: ClassVar[dict[type, tuple[tuple[str, FieldSpec], ...]]] = {}
    _WITH_SCHEMA: ClassVar[bool] = False

    @classmethod
    def field_specs(cls) -> tuple[tuple[str, FieldSpec], ...]:
        """(name, spec) for every field, in declaration (= wire) order."""
        specs = _Fields._specs_cache.get(cls)
        if specs is None:
            if not dataclasses.is_dataclass(cls):
                raise TypeError(f"{cls.__name__} must be a dataclass")
            out = []
            for f in dataclasses.fields(cls):
                spec = f.metadata.get(SPEC_KEY)
                if spec is None:
                    raise TypeError(f"{cls.__name__}.{f.name} is not declared with a contracts.fields helper")
                if f.name == "schema":
                    raise TypeError(f"{cls.__name__}: 'schema' is reserved")
                out.append((f.name, spec))
            specs = tuple(out)
            _Fields._specs_cache[cls] = specs
        return specs

    @classmethod
    def _label(cls) -> str:
        return cls.__name__

    def __post_init__(self) -> None:
        for name, spec in self.field_specs():
            object.__setattr__(self, name, spec.normalize(getattr(self, name)))
        self.validate()

    def validate(self) -> None:
        """Check every field against its spec in declaration order, then check_values(). Raises ContractError."""
        for name, spec in self.field_specs():
            try:
                spec.check(getattr(self, name))
            except ContractError as e:
                raise e.within(self._label(), name) from None
        try:
            self.check_values()
        except ContractError as e:
            raise e.within(self._label()) from None

    def check_values(self) -> None:
        """Checks that involve several fields or numbers. Default: nothing."""

    @staticmethod
    def fail(field: str, detail: str, problem: str = "value") -> NoReturn:
        """Raise a ContractError from check_values(); the schema is filled in by validate()."""
        raise ContractError("?", field, problem, detail)

    def _plain_fields(self) -> dict[str, Any]:
        return {name: spec.to_plain(getattr(self, name)) for name, spec in self.field_specs()}

    @classmethod
    def _build(cls: type[T], d: dict) -> T:
        names = [name for name, _ in cls.field_specs()]
        for name in names:
            if name not in d:
                raise ContractError(cls._label(), name, "missing", "field is missing")
        allowed = set(names) | ({"schema"} if cls._WITH_SCHEMA else set())
        for key in d:
            if key not in allowed:
                raise ContractError(cls._label(), str(key), "unknown_field", "unknown field")
        values = {}
        for name, spec in cls.field_specs():
            try:
                value = spec.normalize(spec.from_wire(d[name]))
                spec.check(value)
            except ContractError as e:
                raise e.within(cls._label(), name) from None
            values[name] = value
        return cls(**values)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        from contracts.codec import pack_obj  # local import: codec sits above this module in the import order
        return pack_obj(self.to_dict()) == pack_obj(other.to_dict())  # type: ignore[attr-defined]

    __hash__ = None  # type: ignore[assignment]


class Record(_Fields):
    """A sub-message: a map on the wire with no schema of its own."""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in field order; sub-messages become dicts, arrays stay arrays."""
        return self._plain_fields()

    @classmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """Check missing and unknown keys, then build (and so validate)."""
        if not isinstance(d, dict):
            raise ContractError(cls._label(), "", "not_a_map", f"expected a map, got {type(d).__name__}")
        return cls._build(d)


class Message(_Fields):
    """A top-level message: carries `schema` first."""

    SCHEMA: ClassVar[str] = ""
    _WITH_SCHEMA: ClassVar[bool] = True

    @classmethod
    def _label(cls) -> str:
        return cls.SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in field order with `schema` first; sub-messages become dicts, arrays stay arrays."""
        out: dict[str, Any] = {"schema": self.SCHEMA}
        out.update(self._plain_fields())
        return out

    @classmethod
    def check_schema(cls, d: dict) -> None:
        """Raise ContractError unless d["schema"] is exactly this class's schema."""
        if "schema" not in d:
            raise ContractError("?", "", "no_schema", "no 'schema' key")
        raw = d["schema"]
        label = raw if isinstance(raw, str) else "?"
        try:
            got = SchemaName.parse(raw)
        except ValueError as e:
            raise ContractError(label, "", "unknown_schema", str(e)) from None
        mine = SchemaName.parse(cls.SCHEMA)
        if not got.same_name(mine):
            raise ContractError(label, "", "unexpected_message", f"expected {cls.SCHEMA}, got {label}")
        if got.version != mine.version:
            raise ContractError(label, "", "schema_version",
                                f"version v{got.version} not supported (this side knows v{mine.version})")

    @classmethod
    def from_dict(cls: type[T], d: dict) -> T:
        """Check `schema`, missing and unknown keys, then build (and so validate)."""
        if not isinstance(d, dict):
            raise ContractError("?", "", "not_a_map", f"expected a map, got {type(d).__name__}")
        cls.check_schema(d)  # type: ignore[attr-defined]
        return cls._build(d)
