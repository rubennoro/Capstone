# adapted from rnoro/deployment:deployment/deploy/deploy/errors.py @ 91b178b
# changes: StationError comes from policy_deploy.contracts.station (shared with the processes), re-exported here;
# GeometryError dropped (no geometry tool on the laptop).
"""The launcher's errors (STRUCTURE.md 9)."""
from __future__ import annotations

from policy_deploy.contracts.station import StationError


class LaunchError(Exception):
    """A process could not be started or made ready. The message is one line, starting with FAILED for a process
    failure (STRUCTURE.md 9.3)."""


__all__ = ["LaunchError", "StationError"]
