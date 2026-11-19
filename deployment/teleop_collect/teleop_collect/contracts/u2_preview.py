"""U2: ui -> frames, the live previews (STRUCTURE.md 4.3). REP on station.preview_endpoint().

| request             | reply           |
|---------------------|-----------------|
| u2.get_preview/v1   | u2.preview/v1   |

The ui says which frame_no of each camera it already has; frames sends a camera's newest JPEG, exactly as the Pi sent
it, only when it is newer. Health always comes along.
"""
from __future__ import annotations

from dataclasses import dataclass

from deploy_contracts import Message, register
from deploy_contracts.fields import nested, optional, scalar

from teleop_collect.contracts.health import CAMERAS, HealthReport

NO_FRAME = -1


@register
@dataclass(frozen=True, eq=False)
class GetPreview(Message):
    """main_after / ultrawide_after: the newest frame_no the ui already has (-1: none)."""

    SCHEMA = "u2.get_preview/v1"
    main_after: int = scalar("int64", min=NO_FRAME)
    ultrawide_after: int = scalar("int64", min=NO_FRAME)


@register
@dataclass(frozen=True, eq=False)
class Preview(Message):
    """A camera's JPEG is present only when its newest frame_no is above the request's *_after."""

    SCHEMA = "u2.preview/v1"
    main_jpeg: bytes | None = optional(scalar("bytes"))
    ultrawide_jpeg: bytes | None = optional(scalar("bytes"))
    main_frame_no: int = scalar("int64", min=NO_FRAME)          # the newest, -1 if none yet
    ultrawide_frame_no: int = scalar("int64", min=NO_FRAME)
    health: HealthReport = nested(HealthReport)

    def check_values(self) -> None:
        for cam in CAMERAS:
            jpeg = getattr(self, f"{cam}_jpeg")
            if jpeg is None:
                continue
            if not jpeg:
                self.fail(f"{cam}_jpeg", "empty JPEG")
            if getattr(self, f"{cam}_frame_no") == NO_FRAME:
                self.fail(f"{cam}_jpeg", f"a JPEG while {cam}_frame_no is -1 (no frame yet)")
