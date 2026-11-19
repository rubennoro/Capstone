# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/frames/pair_rule.py @ 2ffe641
# changes: none
"""PairRule: is a packet's ultrawide pair usable? (STRUCTURE.md 4.3 fix 1, PLAN 7). New. Used by the receiver thread.

A pair is usable only if both:

- window: gap = main_mid - pair_mid (Pi clock, mid-exposures) with -pair_max_lead_ms <= gap <= pair_max_age_ms
  (defaults: lead 25 ms, age 60 ms); the edges are accepted;
- reuse: the same ultrawide frame has been the pair of at most pair_max_reuse (2) consecutive packets, this one
  included, counted by packet_seq (a lost packet in between counts as one of the run: it neither resets it nor lets a
  second repeat through). A third consecutive packet with the same pair is unusable whatever its timing, and so is
  every further one until the pair changes. The run count resets to 1 on a new pair_seq, and on reset() (a Pi
  restart).

The live Pi pairs the nearest frame (gap about -16 ms), so one repeat (about +17 ms) is served and a second repeat
(about +50 ms, inside the window) is refused by the reuse rule; the window is a sanity bound that does not depend on
the phase being right.
"""

from __future__ import annotations

OK = "ok"
TOO_NEW = "too_new"          # the pair is more than pair_max_lead_ms newer than its main frame
TOO_OLD = "too_old"          # the pair is more than pair_max_age_ms older than its main frame
REUSED = "reused"            # the same pair for more than pair_max_reuse consecutive packets
VERDICTS = (OK, TOO_NEW, TOO_OLD, REUSED)


class PairRule:
    def __init__(self, max_age_ms: float, max_lead_ms: float, max_reuse: int) -> None:
        self.max_age_ns = int(round(max_age_ms * 1e6))
        self.max_lead_ns = int(round(max_lead_ms * 1e6))
        self.max_reuse = int(max_reuse)
        self._last_seq: int | None = None
        self._first_packet_seq = 0       # the packet_seq of the run's first packet
        self.run = 0                     # consecutive packet_seqs (this one included) whose pair is _last_seq

    def note(self, pair_seq: int, packet_seq: int) -> int:
        """Count this packet into the run (call once per packet that names a pair); returns the run."""
        if pair_seq == self._last_seq:
            self.run = int(packet_seq) - self._first_packet_seq + 1
        else:
            self._last_seq = int(pair_seq)
            self._first_packet_seq = int(packet_seq)
            self.run = 1
        return self.run

    def check(self, pair_seq: int, packet_seq: int, main_mid_pi_ns: int, pair_mid_pi_ns: int | None) -> str:
        """Note pair_seq, then judge it: OK, REUSED (checked first), TOO_NEW or TOO_OLD. pair_mid_pi_ns None (the
        pair was not found): only the run is counted, and the verdict is REUSED or OK by the run alone."""
        run = self.note(pair_seq, packet_seq)
        if run > self.max_reuse:
            return REUSED
        if pair_mid_pi_ns is None:
            return OK
        gap = int(main_mid_pi_ns) - int(pair_mid_pi_ns)
        if gap < -self.max_lead_ns:
            return TOO_NEW
        if gap > self.max_age_ns:
            return TOO_OLD
        return OK

    def reset(self) -> None:
        """Forget the run (a Pi restart)."""
        self._last_seq = None
        self.run = 0
