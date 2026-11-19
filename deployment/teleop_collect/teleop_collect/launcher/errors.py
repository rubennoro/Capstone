# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/launcher/errors.py @ 2ffe641
# changes: imports (teleop_collect.contracts.station); the section numbers. The source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/deploy/errors.py @ 91b178b
# changes: StationError comes from teleop_collect.contracts.station (shared with the processes), re-exported here;
# GeometryError dropped (no geometry tool on the laptop).
"""The launcher's errors (STRUCTURE.md 10.1)."""
from __future__ import annotations

from teleop_collect.contracts.station import StationError


class LaunchError(Exception):
    """A process could not be started or made ready. The message is one line, starting with FAILED for a process
    failure (STRUCTURE.md 10.1)."""


__all__ = ["LaunchError", "StationError"]
