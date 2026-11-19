# adapted from rnoro/deployment:deployment/contracts/contracts/common.py @ 91b178b
# changes: imports renamed contracts -> deploy_contracts; section references point to deployment/STRUCTURE.md
"""Common replies: Ack and ErrorReply (deployment/STRUCTURE.md section 3.5)."""
from __future__ import annotations

from dataclasses import dataclass

from deploy_contracts.fields import scalar
from deploy_contracts.message import Message
from deploy_contracts.registry import register


@register
@dataclass(frozen=True, eq=False)
class Ack(Message):
    """The reply to requests whose answer is "ok" or "refused"."""

    SCHEMA = "ack/v1"
    ok: bool = scalar("bool")          # true: done. false: refused
    reason: str = scalar("str")        # empty when ok; otherwise one line, e.g. "refused: mode is fault"


@register
@dataclass(frozen=True, eq=False)
class ErrorReply(Message):
    """Sent instead of the normal reply when the server can't answer (broken request, unknown type, handler failed)."""

    SCHEMA = "error/v1"
    request: str = scalar("str")       # the schema of the request, or "" if it couldn't be read
    error: str = scalar("str")         # one line
    field: str = scalar("str")         # the field at fault when it is a contract error, else ""
