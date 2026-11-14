# copied from rnoro/deployment:deployment/policy/policy/errors.py @ 91b178b
# changes: docstring references only
"""The policy process's exception classes."""

from __future__ import annotations


class SettingsError(ValueError):
    """config/policy.yaml or the station file is invalid. Lists every problem at once."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class PolicyLoadError(RuntimeError):
    """The policy could not be built (checkpoint missing, fails to load or fails its checks, no CUDA, ...)."""

    def __init__(self, what: str, check: str) -> None:
        self.what = what
        self.check = check
        super().__init__(f"{what}: {check}")


class BadRequest(ValueError):
    """The request is the problem. The message is the one-line reason sent in the ErrorReply; serving continues."""


class PolicyFatal(RuntimeError):
    """The policy's own unrecoverable failure while starting or serving: the process writes its FATAL line."""

    def __init__(self, what: str, check: str) -> None:
        self.what = what
        self.check = check
        super().__init__(f"{what}: {check}")
