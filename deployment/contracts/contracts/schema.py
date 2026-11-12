"""Schema names: "<name>/v<N>", e.g. "c2.state/v1" (STRUCTURE.md section 7)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_SCHEMA_RE = re.compile(r"^([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)/v([1-9][0-9]*)$")


@dataclass(frozen=True)
class SchemaName:
    name: str
    version: int

    @classmethod
    def parse(cls, text: object) -> SchemaName:
        """Parse "c2.state/v1". Raises ValueError for anything that is not a well-formed schema string."""
        if not isinstance(text, str):
            raise ValueError(f"schema must be a string like 'c2.state/v1', got {type(text).__name__}")
        m = _SCHEMA_RE.match(text)
        if m is None:
            raise ValueError(f"malformed schema {text!r}: expected '<name>/v<N>', e.g. 'c2.state/v1'")
        return cls(m.group(1), int(m.group(2)))

    def same_name(self, other: SchemaName) -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return f"{self.name}/v{self.version}"
