"""R1: teleop -> frames, the recording and the health of the cameras (STRUCTURE.md 4.2). REP on station.r1_endpoint().

| request                     | reply                                            |
|-----------------------------|--------------------------------------------------|
| r1.start_recording/v1       | ack/v1 (ok, or refused with the reason)          |
| r1.stop_recording/v1        | r1.summary/v1 (error/v1 when not recording)      |
| r1.get_health/v1            | r1.health/v1                                     |

Every time is laptop CLOCK_MONOTONIC (`_ns`). Whether the folder exists and is empty is frames' check (an ack refusal);
the contract only asks for an absolute path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from deploy_contracts import Message, Record, register
from deploy_contracts.fields import nested, optional, scalar

from teleop_collect.contracts.health import HealthReport


@register
@dataclass(frozen=True, eq=False)
class StartRecording(Message):
    """Start writing every frame into episode_dir (absolute, existing, without recording files)."""

    SCHEMA = "r1.start_recording/v1"
    episode_dir: str = scalar("str")

    def check_values(self) -> None:
        if not self.episode_dir:
            self.fail("episode_dir", "empty path")
        if not PurePosixPath(self.episode_dir).is_absolute():
            self.fail("episode_dir", f"{self.episode_dir!r} is not an absolute path")


@register
@dataclass(frozen=True, eq=False)
class StopRecording(Message):
    """Drain the writer, close every file and answer with the summary."""

    SCHEMA = "r1.stop_recording/v1"


@register
@dataclass(frozen=True, eq=False)
class GetFramesHealth(Message):
    """Always answered, recording or not."""

    SCHEMA = "r1.get_health/v1"


@dataclass(frozen=True, eq=False)
class CameraSummary(Record):
    """One camera's part of a recording. first/last are set exactly when frames > 0."""

    frames: int = scalar("uint64")                       # frames written (video frames of its .mkv)
    frame_no_gaps: int = scalar("uint64")                # missing frame_no while recording
    bytes: int = scalar("uint64")                        # JPEG bytes written
    first_t_mid_ns: int | None = optional(scalar("int64"))   # laptop, the first frame's mid-exposure
    last_t_mid_ns: int | None = optional(scalar("int64"))    # laptop, the last frame's mid-exposure

    def check_values(self) -> None:
        for name in ("first_t_mid_ns", "last_t_mid_ns"):
            present = getattr(self, name) is not None
            if present != (self.frames > 0):
                self.fail(name, f"set exactly when frames > 0 (frames = {self.frames})")
        if self.first_t_mid_ns is not None and self.last_t_mid_ns is not None \
                and self.last_t_mid_ns < self.first_t_mid_ns:
            self.fail("last_t_mid_ns", f"{self.last_t_mid_ns} is before first_t_mid_ns {self.first_t_mid_ns}")


@register
@dataclass(frozen=True, eq=False)
class RecordingSummary(Message):
    """The reply to stop_recording: what was written."""

    SCHEMA = "r1.summary/v1"
    main: CameraSummary = nested(CameraSummary)
    ultrawide: CameraSummary = nested(CameraSummary)
    packets: int = scalar("uint64")                      # C0 packets recorded (rows of packets.csv)
    packet_no_gaps: int = scalar("uint64")
    uw_reused: int = scalar("uint64")                    # packets whose ultrawide frame was reused (not written again)
    pair_refused: int = scalar("uint64")                 # packets the pair rule refused
    error: str = scalar("str")                           # "" unless the writer failed (queue full, disk)
    pi_clock_offset_ns: int | None = optional(scalar("int64"))     # the final Pi -> laptop fit, Pi - laptop
    pi_clock_drift_ppm: float | None = optional(scalar("float64"))


@register
@dataclass(frozen=True, eq=False)
class FramesHealth(Message):
    """The reply to get_health."""

    SCHEMA = "r1.health/v1"
    health: HealthReport = nested(HealthReport)
    synced: bool = scalar("bool")                        # the Pi clock map is fresh
    recording: bool = scalar("bool")
    recording_error: str = scalar("str")                 # "" or why the writer failed (queue full, disk)
    frames_main: int = scalar("uint64")                  # written in this recording
    frames_ultrawide: int = scalar("uint64")
