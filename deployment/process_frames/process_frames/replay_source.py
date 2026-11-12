"""ReplaySource: a test stand-in for the Pi, C0 frames from memory with no network (STRUCTURE.md §5.4).

Never selectable from config or station.yaml: unit tests only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from contracts import Frame

from process_frames.frame_source import FrameSource

if TYPE_CHECKING:
    from process_frames.pi_recording import PiRecording


class ReplaySource(FrameSource):
    def __init__(self, frames: Iterable[Frame]) -> None:
        self._it = iter(frames)
        self.exhausted = False

    @classmethod
    def from_recording(cls, recording: "PiRecording") -> "ReplaySource":
        """Both cameras' frames merged by ``t_pi_ns``."""
        frames = [f for cam in recording.cameras.values() for f in cam.frames]
        return cls(sorted(frames, key=lambda f: f.t_pi_ns))

    def next_frame(self, timeout_s: float) -> Frame | None:
        try:
            return next(self._it)
        except StopIteration:
            self.exhausted = True
            return None

    def close(self) -> None:
        self._it = iter(())
