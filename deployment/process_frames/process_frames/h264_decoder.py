# adapted from rnoro/data-work:data-processing/dataproc/raw_episode.py @ 96ff703 (read_video_frames)
# differs: decodes one Annex-B access unit per call instead of a file; same PyAV decode and to_ndarray("rgb24")
"""H264Decoder: the default decoder, PyAV (STRUCTURE.md §5.5)."""

from __future__ import annotations

import av
import numpy as np
from av.codec.context import Flags
from av.video.frame import PictureType

from contracts import Frame

from process_frames.errors import DecodeError
from process_frames.frame_decoder import FrameDecoder

THREAD_TYPES = ("SLICE", "NONE")


class H264Decoder(FrameDecoder):
    ENCODING = "h264"

    def __init__(self, camera: str, thread_type: str) -> None:
        if thread_type not in THREAD_TYPES:
            raise ValueError(f"thread_type must be one of {THREAD_TYPES} (never FRAME: it delays output), "
                             f"got {thread_type!r}")
        self.camera = camera
        self.thread_type = thread_type
        self._ctx = None
        self._waiting = True           # waiting for a keyframe

    @property
    def waiting_for_keyframe(self) -> bool:
        return self._waiting

    def _fresh_context(self):
        ctx = av.CodecContext.create("h264", "r")
        ctx.flags |= Flags.low_delay
        ctx.thread_type = self.thread_type
        return ctx

    def decode(self, frame: Frame) -> np.ndarray | None:
        if self._waiting:
            if not frame.keyframe:
                return None
            self._ctx = self._fresh_context()          # nothing from before the gap can leak in
            self._waiting = False
        try:
            pictures = self._ctx.decode(av.Packet(frame.data))
        except av.error.FFmpegError as e:
            raise DecodeError(self.camera, frame.seq, f"FFmpeg: {e}") from e
        if len(pictures) != 1:
            raise DecodeError(self.camera, frame.seq,
                              f"{len(pictures)} pictures from one access unit, expected exactly 1")
        pic = pictures[0]
        if pic.pict_type == PictureType.B:
            raise DecodeError(self.camera, frame.seq, "B picture: the stream has B-frames, which C0 forbids")
        return pic.to_ndarray(format="rgb24")

    def notify_gap(self) -> None:
        self._waiting = True
