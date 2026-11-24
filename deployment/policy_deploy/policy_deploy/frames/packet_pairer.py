# adapted from rnoro/deployment:deployment/process_frames/process_frames/packet_pairer.py @ 91b178b
# changes: nearest-frame pairing as the live Pi does it (STRUCTURE.md 4.2): pair_seq is the added ultrawide frame whose
# capture t_pi_ns is nearest the main frame's, a tie to the older frame; no age or lead cut (the receiver's PairRule
# judges); NO_PAIR only before the first ultrawide frame. The old "latest frame <= main + 5 ms" rule (PAIR_AHEAD_NS)
# and its max_age_ns cut are removed. Rule 2 (every encoded frame sent once, in the next packet) unchanged.
"""PacketPairer: the fake Pi's pairing, as the live Pi does it, and its ultrawide bookkeeping.

Used by FakePiSender and RecordingPacketSource (through PiRecording), and by the tests:

- rule 2: every encoded ultrawide frame is sent exactly once, in the first packet after it was encoded, in capture
  order (``add_ultrawide`` queues it, the next ``take`` hands it out);
- pairing: ``pair_seq`` is the added ultrawide frame whose capture time is nearest the main frame's (ties: the older
  one); a stalled stream keeps pairing its last frame; -1 only before the first ultrawide frame.

The caller must add every ultrawide frame that could be the nearest before taking (the fake Pi adds the frames
captured up to half an ultrawide frame period after the main frame).
"""

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

from policy_deploy.contracts import NO_PAIR

#: Candidates kept (the nearest frame is always among the newest few; ~2 s at 30 FPS).
KEEP = 64

T = TypeVar("T")


class PacketPairer(Generic[T]):
    def __init__(self) -> None:
        self._pending: list[T] = []                       # encoded, not sent yet, in capture order
        self._recent: deque[tuple[int, int]] = deque(maxlen=KEEP)   # (frame_seq, t_pi_ns): pair candidates
        self._last: tuple[int, int] | None = None

    def add_ultrawide(self, frame_seq: int, t_pi_ns: int, item: T) -> None:
        """An encoded ultrawide frame, in capture order. It goes out with the next packet."""
        if self._last is not None and (frame_seq <= self._last[0] or t_pi_ns <= self._last[1]):
            raise ValueError(f"ultrawide frame {frame_seq} at {t_pi_ns} is not after frame {self._last[0]} at "
                             f"{self._last[1]}: frames must be added in capture order")
        self._last = (int(frame_seq), int(t_pi_ns))
        self._pending.append(item)
        self._recent.append(self._last)

    def nearest(self, main_t_pi_ns: int) -> int:
        """The frame_seq of the added frame captured nearest main_t_pi_ns (ties: the older), or NO_PAIR."""
        best, best_d = NO_PAIR, None
        for seq, t in self._recent:                       # oldest first: a tie keeps the older one
            d = abs(t - int(main_t_pi_ns))
            if best_d is None or d < best_d:
                best, best_d = seq, d
        return best

    def take(self, main_t_pi_ns: int) -> tuple[list[T], int]:
        """The next packet's ultrawide frames (every pending one, in order, each exactly once) and its pair_seq."""
        pair = self.nearest(main_t_pi_ns)
        items, self._pending = self._pending, []
        return items, pair

    @property
    def pending(self) -> int:
        """Encoded ultrawide frames waiting for the next packet."""
        return len(self._pending)
