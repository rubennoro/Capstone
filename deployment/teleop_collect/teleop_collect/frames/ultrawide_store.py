# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/ultrawide_store.py @ 2ffe641
# changes: holds the EncodedFrame as received (no DecodedUltrawide, nothing is decoded: frame_seq, t_pi_ns and
# exposure_ns come from its FrameInfo); evict_before and the time-based eviction unchanged. The source's own header
# follows.
# adapted from rnoro/deployment:deployment/process_frames/process_frames/ultrawide_store.py @ 91b178b
# changes: evict_before(t_pi_ns), called on every packet (STRUCTURE.md 4.3 fix 3: a frame the Pi keeps repeating does
# not stay findable forever); DecodedUltrawide carries exposure_ns.
"""UltrawideStore: the ultrawide frames of the last ~1 s as received, by frame_seq.

Lets the receiver find a packet's pair when it came in an earlier packet. Used by the receiver thread only.
"""

from __future__ import annotations

from collections import OrderedDict

from teleop_collect.contracts import EncodedFrame


class UltrawideStore:
    def __init__(self, keep_s: float) -> None:
        self._keep_ns = int(keep_s * 1e9)
        self._frames: OrderedDict[int, EncodedFrame] = OrderedDict()

    def add(self, f: EncodedFrame) -> None:
        """Keep it; then forget frames captured more than keep_s before it. Frames come in capture order."""
        self._frames[int(f.info.frame_seq)] = f
        self.evict_before(int(f.info.t_pi_ns) - self._keep_ns)

    def evict_before(self, t_pi_ns: int) -> None:
        """Forget every frame captured before t_pi_ns (Pi clock)."""
        while self._frames and next(iter(self._frames.values())).info.t_pi_ns < t_pi_ns:
            self._frames.popitem(last=False)

    def get(self, frame_seq: int) -> EncodedFrame | None:
        """That frame, or None if it was never received (lost in a gap) or is older than keep_s."""
        return self._frames.get(int(frame_seq))

    def __len__(self) -> int:
        return len(self._frames)
