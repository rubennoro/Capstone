"""RecordingPacketSource: a test stand-in for the Pi, C0 packets from memory with no network (STRUCTURE.md §5.4).

Never selectable from config or station.yaml: unit tests only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from contracts import Packet

from process_frames.packet_source import PacketSource

if TYPE_CHECKING:
    from process_frames.pi_recording import PiRecording


class RecordingPacketSource(PacketSource):
    def __init__(self, packets: Iterable[Packet]) -> None:
        self._it = iter(packets)
        self.exhausted = False

    @classmethod
    def from_recording(cls, recording: "PiRecording") -> "RecordingPacketSource":
        """One pass over the recording, paired by PacketPairer, with the recorded times (PiRecording.packets)."""
        return cls(recording.packets())

    def next_packet(self, timeout_s: float) -> Packet | None:
        try:
            return next(self._it)
        except StopIteration:
            self.exhausted = True
            return None

    def close(self) -> None:
        self._it = iter(())
