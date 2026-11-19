"""The key state from the ui, with the deadman rule (PLAN 5.1 step 2, 5.4; STRUCTURE.md 7.1).

The ui sends its key state on every change and at least every 50 ms (U1 `keys`). `KeyState` keeps the newest one.
What teleop acts on (`current`) is **every key released** when the page has lost focus or when no key state arrived
for `stale_ms`: the deadman. Every change of the raw state is also kept for keys.csv (`changes`).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from teleop_collect.contracts import MOTION_KEYS

_KNOWN = frozenset(MOTION_KEYS)
#: Changes kept when nobody drains them (between episodes); the oldest are dropped first.
CHANGES_KEPT = 10_000


@dataclass(frozen=True)
class HeldKeys:
    """The motion keys in effect and Shift (slow mode)."""

    held: frozenset[str]          # subset of MOTION_KEYS
    slow: bool

    def __post_init__(self) -> None:
        unknown = set(self.held) - _KNOWN
        if unknown:
            raise ValueError(f"unknown motion keys {sorted(unknown)}; known: {', '.join(MOTION_KEYS)}")
        object.__setattr__(self, "held", frozenset(self.held))
        object.__setattr__(self, "slow", bool(self.slow))

    def names(self) -> list[str]:
        """The held keys in MOTION_KEYS order (stable for keys.csv)."""
        return [k for k in MOTION_KEYS if k in self.held]


NO_KEYS = HeldKeys(frozenset(), False)


class KeyState:
    """The newest key state from U1 and the deadman rule. Main thread only (no lock)."""

    def __init__(self, stale_ms: float) -> None:
        if not stale_ms > 0:
            raise ValueError(f"stale_ms must be > 0, got {stale_ms!r}")
        self._stale_ns = int(round(float(stale_ms) * 1e6))
        self._keys = NO_KEYS
        self._focus = False
        self._t_ns: int | None = None
        self._last_change: tuple[HeldKeys, bool] | None = None
        self._changes: deque[tuple[int, HeldKeys, bool]] = deque(maxlen=CHANGES_KEPT)

    def update(self, held: Sequence[str], slow: bool, focus: bool, now_ns: int) -> None:
        """A U1 `keys` request arrived at now_ns. ValueError for an unknown key name."""
        keys = HeldKeys(frozenset(held), bool(slow))
        focus = bool(focus)
        self._keys, self._focus, self._t_ns = keys, focus, int(now_ns)
        if self._last_change != (keys, focus):
            self._last_change = (keys, focus)
            self._changes.append((int(now_ns), keys, focus))

    def current(self, now_ns: int) -> HeldKeys:
        """The keys in effect: none when the page has no focus or the newest state is older than stale_ms."""
        if self._t_ns is None or not self._focus or int(now_ns) - self._t_ns > self._stale_ns:
            return NO_KEYS
        return self._keys

    def changes(self) -> list[tuple[int, HeldKeys, bool]]:
        """(t_ns, keys, focus) of every change since the last call, oldest first (for keys.csv). Drains."""
        out = list(self._changes)
        self._changes.clear()
        return out
