"""MkvWriter: one camera's MJPEG video in Matroska, the received JPEG bytes muxed as they are (STRUCTURE.md 5.2).

New. Nothing is decoded or re-encoded: each JPEG becomes one av.Packet of an `mjpeg` stream (`yuvj420p`, the rig
size, time_base 1/1000), so demuxing the file gives back exactly the bytes that were written (PLAN 6.1). Frame i of
the video is row i of `<camera>_frames.csv`; the video's own timestamps (`pts = i x 1000 / fps` ms) are only for
playback.

The Matroska muxer buffers a cluster before writing it: `cluster_time_limit` (500 ms of playback time) and
`flush_packets` keep what was written on disk within about half a second. A file closed without any frame holds only
the header (no player opens it; the episode's CSV then has no rows either).

Used by the FrameRecorder's writer thread only.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av

TIME_BASE = Fraction(1, 1000)
#: libavformat options: write each finished cluster at once, and close a cluster every 500 ms of playback time.
MUX_OPTIONS = {"flush_packets": "1", "cluster_time_limit": "500"}


class MkvWriter:
    """One camera's .mkv. write() returns the frame's video_index (0, 1, ...)."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        if not (isinstance(fps, int) and fps > 0):
            raise ValueError(f"fps must be a positive integer, got {fps!r}")
        self.path = Path(path)
        self.fps = fps
        self.count = 0
        self._container = av.open(str(self.path), "w", format="matroska", options=dict(MUX_OPTIONS))
        try:
            stream = self._container.add_stream("mjpeg", rate=fps)
            stream.width, stream.height, stream.pix_fmt = int(width), int(height), "yuvj420p"
            stream.time_base = TIME_BASE
            self._container.start_encoding()          # the header now: a bad path or a full disk fails here
        except BaseException:
            self._container.close()
            raise
        self._stream = stream

    def write(self, jpeg: bytes) -> int:
        """Mux one JPEG as received; returns its video_index. av.error.FFmpegError / OSError on a write failure."""
        if self._container is None:
            raise RuntimeError(f"{self.path}: already closed")
        if not jpeg:
            raise ValueError("empty JPEG")
        index = self.count
        packet = av.Packet(bytes(jpeg))
        packet.stream = self._stream
        packet.pts = packet.dts = index * 1000 // self.fps
        packet.time_base = TIME_BASE
        packet.is_keyframe = True
        self._container.mux(packet)
        self.count += 1
        return index

    def close(self) -> None:
        """Write the trailer and close the file; idempotent."""
        container, self._container = self._container, None
        if container is not None:
            container.close()

    @property
    def closed(self) -> bool:
        return self._container is None
