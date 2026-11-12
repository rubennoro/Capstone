"""The errors this package adds to the ones of `contracts` (STRUCTURE.md 4.2)."""
from __future__ import annotations

from collections.abc import Sequence


class SettingsError(ValueError):
    """config.yaml / station.yaml problems, all of them at once."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class Refused(RuntimeError):
    """A server answered Ack(ok=false)."""

    def __init__(self, process: str, request: str, reason: str) -> None:
        self.process = process
        self.request = request
        self.reason = reason
        super().__init__(f"{process}: {request} refused: {reason}")


class RequestCancelled(RuntimeError):
    """A long request (home) was cancelled because the operator or deploy asked to stop; its reply is dropped."""

    def __init__(self, process: str, request: str) -> None:
        self.process = process
        self.request = request
        super().__init__(f"{process}: {request} cancelled (reply dropped)")


class ScriptError(ValueError):
    """A broken operator script."""
