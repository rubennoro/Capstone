"""PacketPairer: the Pi sender's pairing rule and ultrawide bookkeeping (contracts/PI_PAYLOAD.md §5, STRUCTURE.md
§5.15).

Used by FakePiSender and RecordingPacketSource, and by the tests, so they all follow exactly the rule the Pi must:

- rule 2: every encoded ultrawide frame is sent exactly once, in the first packet after it was encoded, in capture
  order (``add_ultrawide`` queues it, the next ``take`` hands it out);
- rule 3: ``pair_seq`` is the ultrawide frame with the latest capture time not later than the main frame's + 5 ms,
  at most 50 ms older than the main frame; -1 if none qualifies (comparing ``t_pi_ns``).
"""

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

#: PI_PAYLOAD.md §5 rule 3: a pair may be captured up to this much after the main frame (overlapping exposures) ...
PAIR_AHEAD_NS = 5_000_000
#: ... and at most this much before it.
PAIR_MAX_AGE_NS = 50_000_000
NO_PAIR = -1

T = TypeVar("T")


class PacketPairer(Generic[T]):
    def __init__(self, ahead_ns: int = PAIR_AHEAD_NS, max_age_ns: int = PAIR_MAX_AGE_NS) -> None:
        self.ahead_ns = int(ahead_ns)
        self.max_age_ns = int(max_age_ns)
        self._pending: list[T] = []                       # encoded, not sent yet, in capture order
        self._recent: deque[tuple[int, int]] = deque()    # (frame_seq, t_pi_ns) of recent frames: pair candidates
        self._last: tuple[int, int] | None = None

    def add_ultrawide(self, frame_seq: int, t_pi_ns: int, item: T) -> None:
        """An encoded ultrawide frame, in capture order. It goes out with the next packet."""
        if self._last is not None and (frame_seq <= self._last[0] or t_pi_ns <= self._last[1]):
            raise ValueError(f"ultrawide frame {frame_seq} at {t_pi_ns} is not after frame {self._last[0]} at "
                             f"{self._last[1]}: frames must be added in capture order")
        self._last = (int(frame_seq), int(t_pi_ns))
        self._pending.append(item)
        self._recent.append(self._last)

    def take(self, main_t_pi_ns: int) -> tuple[list[T], int]:
        """The next packet's ultrawide frames (every pending one, in order, each exactly once) and its pair_seq."""
        oldest = main_t_pi_ns - self.max_age_ns
        while self._recent and self._recent[0][1] < oldest:          # too old for this and every later main frame
            self._recent.popleft()
        pair = NO_PAIR
        for seq, t in reversed(self._recent):
            if t <= main_t_pi_ns + self.ahead_ns:
                pair = seq
                break
        items, self._pending = self._pending, []
        return items, pair

    @property
    def pending(self) -> int:
        """Encoded ultrawide frames waiting for the next packet."""
        return len(self._pending)
