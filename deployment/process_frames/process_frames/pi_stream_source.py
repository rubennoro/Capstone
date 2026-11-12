"""PiStreamSource: the C0 subscriber, connected to the real Pi or to the fake Pi (STRUCTURE.md §5.4)."""

from __future__ import annotations

from contracts import Frame, Subscriber

from process_frames.frame_source import FrameSource


class PiStreamSource(FrameSource):
    def __init__(self, endpoint: str, recv_hwm: int) -> None:
        self.endpoint = endpoint
        self._recv_hwm = recv_hwm
        self._sub: Subscriber | None = None       # opened on first use, in the thread that uses it

    def next_frame(self, timeout_s: float) -> Frame | None:
        if self._sub is None:
            self._sub = Subscriber(name="pi", endpoint=self.endpoint, expect=Frame, recv_hwm=self._recv_hwm)
        return self._sub.receive(timeout_s)

    def close(self) -> None:
        if self._sub is not None:
            self._sub.close()
            self._sub = None
