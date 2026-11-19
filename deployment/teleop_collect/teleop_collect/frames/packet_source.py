# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/packet_source.py @ 2ffe641
# changes: imports; + last_t_send_ns and last_arrival_ns (the raw one-way clock sample of the packet last returned,
# recorded in packets.csv; None for a source without them). The source's own header follows.
# copied from rnoro/deployment:deployment/process_frames/process_frames/packet_source.py @ 91b178b
# changes: imports (policy_deploy.contracts)
"""PacketSource: where the receiver takes its C0 packets from (internal packet form)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from teleop_collect.contracts import Packet


class PacketSource(ABC):
    @abstractmethod
    def next_packet(self, timeout_s: float) -> Packet | None:
        """The next C0 packet, or None if none came within ``timeout_s``. ContractError if a packet breaks C0."""

    @abstractmethod
    def close(self) -> None:
        """Release the source (its socket, if any)."""

    @property
    def last_t_send_ns(self) -> int | None:
        """The Pi's send stamp (``t_send_ns``, Pi clock) of the packet last returned; None if the source has none."""
        return None

    @property
    def last_arrival_ns(self) -> int | None:
        """The laptop arrival time the clock map got for the packet last returned; None if the source has none."""
        return None
