"""FrameSource: where the receiver takes its C0 frame messages from (STRUCTURE.md §5.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from contracts import Frame


class FrameSource(ABC):
    @abstractmethod
    def next_frame(self, timeout_s: float) -> Frame | None:
        """The next C0 message, or None if none came within ``timeout_s``. ContractError if a message breaks C0."""

    @abstractmethod
    def close(self) -> None:
        """Release the source (its socket, if any)."""
