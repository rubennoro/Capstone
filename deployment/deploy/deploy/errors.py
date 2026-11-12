"""The three error types of the deploy package (STRUCTURE.md section 3)."""
from __future__ import annotations

from collections.abc import Sequence


class _ProblemsError(Exception):
    """An error that carries every problem found, one line each, so they can all be reported at once."""

    def __init__(self, problems: str | Sequence[str]) -> None:
        self.problems: list[str] = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("\n".join(self.problems))


class StationError(_ProblemsError):
    """station.yaml (or a process's config.yaml merged with its station section) is wrong."""


class GeometryError(_ProblemsError):
    """gripper.yaml, franka_control's tcp.F_T_TCP or the rig file's tcp block is wrong."""


class LaunchError(Exception):
    """A process could not be loaded, started, or made ready. The message is one line, starting with FAILED for a
    process failure (section 5.3)."""
