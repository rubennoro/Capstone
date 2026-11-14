# copied from rnoro/deployment:deployment/contracts/contracts/c1_frames.py @ 91b178b
# changes: imports (deploy_contracts, policy_deploy.contracts.sizes); every time is now laptop CLOCK_MONOTONIC
# (comments only, the fields are unchanged).
"""C1: frames <-> manager, inside the laptop (STRUCTURE.md 3.2). Every time is laptop CLOCK_MONOTONIC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deploy_contracts import Message, Record, register
from deploy_contracts.fields import array, nested, optional, scalar

from policy_deploy.contracts.sizes import IMAGE_SIZE, OBS_STEPS


@register
@dataclass(frozen=True, eq=False)
class GetFramesRequest(Message):
    SCHEMA = "c1.get_frames/v1"


@register
@dataclass(frozen=True, eq=False)
class GetHealthRequest(Message):
    SCHEMA = "c1.get_health/v1"


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


_IMAGES = (OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3)


@register
@dataclass(frozen=True, eq=False)
class FramesReply(Message):
    """2 main + 2 ultrawide frames, oldest first, prepared as in training; or no images at all, and only health.

    main_t_ns / ultrawide_t_ns: the packet times (the main frame's mid-exposure mapped to the laptop clock).
    """

    SCHEMA = "c1.frames/v1"
    main: np.ndarray | None = optional(array("uint8", _IMAGES))
    main_t_ns: np.ndarray | None = optional(array("int64", (OBS_STEPS,)))
    ultrawide: np.ndarray | None = optional(array("uint8", _IMAGES))
    ultrawide_t_ns: np.ndarray | None = optional(array("int64", (OBS_STEPS,)))
    health: HealthReport = nested(HealthReport)

    def check_values(self) -> None:
        not_ok = [(cam, getattr(self.health, cam).status) for cam in ("main", "ultrawide")
                  if getattr(self.health, cam).status != "ok"]
        for name in ("main", "main_t_ns", "ultrawide", "ultrawide_t_ns"):
            present = getattr(self, name) is not None
            if not not_ok and not present:
                self.fail(name, "missing while both cameras are ok (images and times come together)")
            if not_ok and present:
                cam, status = not_ok[0]
                self.fail(name, f"present while camera '{cam}' is {status} (both cameras or nothing)")
        for name in ("main_t_ns", "ultrawide_t_ns"):
            t = getattr(self, name)
            if t is not None and np.any(np.diff(t) <= 0):
                self.fail(name, f"times not strictly increasing: {t.tolist()}")


@register
@dataclass(frozen=True, eq=False)
class HealthReply(Message):
    SCHEMA = "c1.health/v1"
    health: HealthReport = nested(HealthReport)
