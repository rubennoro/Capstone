"""C0: Pi 5 -> process_frames: Frame (PUB/SUB) and the time_sync request/reply (STRUCTURE.md section 4.2).

All of C0 is a stand-in until the Pi's sending side exists (PLAN section 10).
"""
from __future__ import annotations

from dataclasses import dataclass

from contracts.fields import scalar
from contracts.message import Message
from contracts.registry import register


@register
@dataclass(frozen=True, eq=False)
class Frame(Message):
    """One encoded frame of one camera. H.264: exactly one access unit, Annex-B framing."""

    SCHEMA = "c0.frame/v1"
    camera: str = scalar("str", choices=("main", "ultrawide"))
    sensor: str = scalar("str", choices=("imx519", "imx219"))
    seq: int = scalar("uint64")                       # per camera, counts captured frames
    t_pi_ns: int = scalar("int64")                    # libcamera SensorTimestamp, Pi CLOCK_BOOTTIME
    exposure_ns: int = scalar("int64", min=0)         # ExposureTime (us) x 1000; 0 if unknown
    width: int = scalar("int", min=1)
    height: int = scalar("int", min=1)
    encoding: str = scalar("str", choices=("h264", "jpeg"))
    keyframe: bool = scalar("bool")                   # JPEG: must be true
    data: bytes = scalar("bytes")                     # non-empty

    def check_values(self) -> None:
        if self.encoding == "jpeg" and not self.keyframe:
            self.fail("keyframe", "a jpeg frame must be a keyframe")
        if len(self.data) == 0:
            self.fail("data", "empty frame data")


@register
@dataclass(frozen=True, eq=False)
class TimeSyncRequest(Message):
    """process_frames -> the Pi, about once a second."""

    SCHEMA = "c0.time_sync/v1"


@register
@dataclass(frozen=True, eq=False)
class TimeSyncReply(Message):
    SCHEMA = "c0.time_sync.reply/v1"
    t_pi_ns: int = scalar("int64")                    # the Pi's current CLOCK_BOOTTIME, ns
