# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/contracts/c1_frames.py @ 2ffe641
# changes: only the two health Records (CameraHealthReport, HealthReport); the C1 messages (GetFramesRequest,
# FramesReply, GetHealthRequest, HealthReply) and the sizes import dropped; + CAMERAS ("main", "ultrawide"); R1 and
# U2 carry these Records (STRUCTURE.md 4). The source's own header follows.
# copied from rnoro/deployment:deployment/contracts/contracts/c1_frames.py @ 91b178b
# changes: imports (deploy_contracts, policy_deploy.contracts.sizes); every time is now laptop CLOCK_MONOTONIC
# (comments only, the fields are unchanged).
"""The cameras' health, as the frames process reports it in R1 and U2 (STRUCTURE.md 4.2, 4.3). Laptop time."""
from __future__ import annotations

from dataclasses import dataclass

from deploy_contracts import Record
from deploy_contracts.fields import nested, scalar

CAMERAS = ("main", "ultrawide")


@dataclass(frozen=True, eq=False)
class CameraHealthReport(Record):
    """One camera's health (named so it doesn't clash with the frames process's own CameraHealth class)."""

    status: str = scalar("str", choices=("ok", "stale", "down"))
    last_frame_age_ms: float = scalar("float64", min=0.0)    # age of the newest frame, laptop time, ms
    fps: float = scalar("float64", min=0.0)                  # measured frame rate, Hz
    dropped: int = scalar("uint64")                          # seq gaps since the frames process started


@dataclass(frozen=True, eq=False)
class HealthReport(Record):
    main: CameraHealthReport = nested(CameraHealthReport)
    ultrawide: CameraHealthReport = nested(CameraHealthReport)
