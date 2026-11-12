"""FrameDecoder: one encoded C0 frame -> RGB (STRUCTURE.md §5.5). One decoder object per camera."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from contracts import Frame

if TYPE_CHECKING:
    from process_frames.settings import H264Settings


class FrameDecoder(ABC):
    ENCODING: ClassVar[str]        # the C0 `encoding` it accepts: "h264" or "jpeg"

    @abstractmethod
    def decode(self, frame: Frame) -> np.ndarray | None:
        """(height, width, 3) uint8 RGB; None = unusable (after a gap, before the next keyframe);
        DecodeError if the data can't be decoded."""

    def notify_gap(self) -> None:
        """A frame was lost before the next one. Default: nothing."""


def make_decoder(kind: str, camera: str, h264: "H264Settings") -> FrameDecoder:
    if kind == "h264":
        from process_frames.h264_decoder import H264Decoder
        return H264Decoder(camera, h264.thread_type)
    if kind == "jpeg":
        from process_frames.jpeg_decoder import JpegDecoder
        return JpegDecoder(camera)
    raise ValueError(f"unknown decoder {kind!r} (expected h264 or jpeg)")
