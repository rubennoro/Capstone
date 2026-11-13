"""PiPacketSource: the C0 packet subscriber, connected to the real Pi or to the fake Pi (STRUCTURE.md §5.4)."""

from __future__ import annotations

from contracts import Packet, PacketSubscriber

from process_frames.packet_source import PacketSource


class PiPacketSource(PacketSource):
    def __init__(self, endpoint: str, recv_hwm: int) -> None:
        self.endpoint = endpoint
        self._recv_hwm = recv_hwm
        self._sub: PacketSubscriber | None = None       # opened on first use, in the thread that uses it

    def next_packet(self, timeout_s: float) -> Packet | None:
        if self._sub is None:
            self._sub = PacketSubscriber(name="pi", endpoint=self.endpoint, recv_hwm=self._recv_hwm)
        return self._sub.receive(timeout_s)

    def close(self) -> None:
        if self._sub is not None:
            self._sub.close()
            self._sub = None
