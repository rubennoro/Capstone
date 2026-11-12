# adapted from rnoro/main-sim:controlSW/franka_deploy/franka_deploy/tasks/frame_select.py @ 4be9982
# kept: step back from an anchor by a fixed spacing, take the nearest frame; differs: the anchor is given (the newest
# main frame), fixed tolerance from config, ties to the earlier frame, a miss returns a reason instead of raising
"""FrameSelector and Selection: pick the frames nearest the wanted times (STRUCTURE.md §5.9, §6.2). Pure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from process_frames.frame_buffer import BufferedFrame


@dataclass(frozen=True)
class Selection:
    frames: tuple[BufferedFrame, ...] | None     # oldest first; None if the pick failed
    reason: str                                  # "" when ok

    @property
    def ok(self) -> bool:
        return self.frames is not None


class FrameSelector:
    def __init__(self, tolerance_ns: int) -> None:
        self.tolerance_ns = int(tolerance_ns)

    @staticmethod
    def targets(anchor_ns: int, spacing_ns: int, count: int) -> np.ndarray:
        """int64, oldest first: ``anchor - (count-1-i) * spacing``."""
        i = np.arange(count, dtype=np.int64)
        return np.int64(anchor_ns) - (count - 1 - i) * np.int64(spacing_ns)

    def pick(self, frames: Sequence[BufferedFrame], anchor_ns: int, spacing_ns: int, count: int) -> Selection:
        if not frames:
            return Selection(None, "no frames")
        times = np.array([f.t_pc_ns for f in frames], dtype=np.int64)
        chosen: list[int] = []
        for target in self.targets(anchor_ns, spacing_ns, count):
            dist = np.abs(times - target)
            idx = int(np.argmin(dist))                # first minimum: ties go to the earlier frame
            if int(dist[idx]) > self.tolerance_ns:
                return Selection(None, f"nearest frame {dist[idx] / 1e6:.1f} ms from {int(target)} "
                                       f"(tolerance {self.tolerance_ns / 1e6:.1f} ms)")
            chosen.append(idx)
        if len(set(chosen)) != len(chosen):
            return Selection(None, f"the same frame picked for two targets (seq {frames[chosen[0]].seq})")
        return Selection(tuple(frames[i] for i in chosen), "")
