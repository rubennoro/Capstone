"""PacketSource: where the receiver takes its C0 packets from (STRUCTURE.md §5.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from contracts import Packet


class PacketSource(ABC):
    @abstractmethod
    def next_packet(self, timeout_s: float) -> Packet | None:
        """The next C0 packet, or None if none came within ``timeout_s``. ContractError if a packet breaks C0."""

    @abstractmethod
    def close(self) -> None:
        """Release the source (its socket, if any)."""
