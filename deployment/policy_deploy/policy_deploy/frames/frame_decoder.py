# copied from rnoro/deployment:deployment/process_frames/process_frames/frame_decoder.py @ 91b178b
# changes: make_decoder knows only "jpeg" (H264Decoder dropped: the Pi contract is final); the interface stays for a
# swap; imports.
"""FrameDecoder: one encoded C0 frame (a packet part and its frame map) -> RGB. One decoder object per camera."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from policy_deploy.contracts import EncodedFrame


class FrameDecoder(ABC):
    ENCODING: ClassVar[str]        # the C0 `encoding` it accepts ("jpeg")

    @abstractmethod
    def decode(self, frame: EncodedFrame) -> np.ndarray | None:
        """(height, width, 3) uint8 RGB; None = unusable; DecodeError if the data can't be decoded."""

    def notify_gap(self) -> None:
        """A frame was lost before the next one. Default: nothing."""


def make_decoder(kind: str, camera: str) -> FrameDecoder:
    if kind == "jpeg":
        from policy_deploy.frames.jpeg_decoder import JpegDecoder
        return JpegDecoder(camera)
    raise ValueError(f"unknown decoder {kind!r} (expected jpeg)")
