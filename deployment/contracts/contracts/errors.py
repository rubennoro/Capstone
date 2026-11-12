"""The errors every contract check and transport helper raises (STRUCTURE.md section 6)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts.common import ErrorReply

PROBLEMS = (
    "not_msgpack",
    "not_a_map",
    "no_schema",
    "unknown_schema",
    "schema_version",
    "missing",
    "unknown_field",
    "type",
    "dtype",
    "shape",
    "value",
    "unexpected_message",
)


def join_field(prefix: str, field: str) -> str:
    """Dotted field path: join_field("health", "main.status") == "health.main.status"; "[2]" attaches directly."""
    if not prefix:
        return field
    if not field:
        return prefix
    if field.startswith("["):
        return prefix + field
    return prefix + "." + field


class ContractError(ValueError):
    """A message broke its contract. str(e) == "c2.state/v1: field 'O_T_TCP': shape (3, 4), expected (4, 4)"."""

    def __init__(self, schema: str, field: str, problem: str, detail: str) -> None:
        if problem not in PROBLEMS:
            raise ValueError(f"unknown contract problem code {problem!r}")
        self.schema = schema
        self.field = field
        self.problem = problem
        self.detail = detail
        super().__init__(self._text())

    def _text(self) -> str:
        if self.field:
            return f"{self.schema}: field '{self.field}': {self.detail}"
        return f"{self.schema}: {self.detail}"

    def __str__(self) -> str:
        return self._text()

    def __reduce__(self):
        return (type(self), (self.schema, self.field, self.problem, self.detail))

    def within(self, schema: str, prefix: str = "") -> ContractError:
        """The same problem, reported for another schema and/or under a field prefix (for sub-messages)."""
        return ContractError(schema, join_field(prefix, self.field), self.problem, self.detail)


class TransportTimeout(TimeoutError):
    """No reply within the caller's timeout. The request was sent once and is never resent."""

    def __init__(self, process: str, endpoint: str, request: str, timeout_s: float) -> None:
        self.process = process
        self.endpoint = endpoint
        self.request = request
        self.timeout_s = timeout_s
        super().__init__(f"{process} ({endpoint}): {request}: no reply within {timeout_s:.3f} s "
                         "(sent once, not resent)")

    def __reduce__(self):
        return (type(self), (self.process, self.endpoint, self.request, self.timeout_s))


class TransportCancelled(RuntimeError):
    """The caller's cancel() turned true while waiting for a reply; the reply (if any comes) is dropped."""

    def __init__(self, process: str, endpoint: str, request: str, waited_s: float) -> None:
        self.process = process
        self.endpoint = endpoint
        self.request = request
        self.waited_s = waited_s
        super().__init__(f"{process} ({endpoint}): {request}: cancelled after {waited_s:.3f} s (reply dropped)")

    def __reduce__(self):
        return (type(self), (self.process, self.endpoint, self.request, self.waited_s))


class RemoteError(RuntimeError):
    """The server answered with an ErrorReply instead of its normal reply."""

    def __init__(self, process: str, reply: ErrorReply) -> None:
        self.process = process
        self.reply = reply
        request = reply.request or "?"
        super().__init__(f"{process}: error reply to {request}: {reply.error}")

    def __reduce__(self):
        return (type(self), (self.process, self.reply))
