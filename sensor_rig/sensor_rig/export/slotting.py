"""Ultrawide decimation and slotting onto the main grid (brief D6, P1 §4.4, R1 invariants 2 and 4).

The iPhone attaches at most one ultrawide image to a main frame and writes the *main* frame's time string for it
(``""`` for slots without one); the UW video PTS are the slot times relative to the first occupied slot.
"""

from __future__ import annotations

import numpy as np


def select_uw(t_uw_ns, rate_hz: float | None) -> np.ndarray:
    """Greedy decimation to ~``rate_hz`` (None: keep all). Keeps frame j if it is >= 1/rate - dt/2 after the last kept."""
    t = np.asarray(t_uw_ns, dtype=np.int64)
    if rate_hz is None or len(t) < 2:
        return np.arange(len(t), dtype=np.int64)
    dt = float(np.median(np.diff(t)))
    min_spacing = 1e9 / rate_hz - 0.5 * dt
    keep = [0]
    last = int(t[0])
    for j in range(1, len(t)):
        if t[j] - last >= min_spacing:
            keep.append(j)
            last = int(t[j])
    return np.asarray(keep, dtype=np.int64)


def slot_uw(t_grid_ns, t_uw_ns, sel_idx, tol_ns: int = 1_000_000) -> np.ndarray:
    """Assign selected UW frames to their nearest grid slot.

    A frame is rejected if it is farther than half the larger neighbouring grid spacing (+ ``tol_ns``) from its
    nearest slot. On a collision the frame closer to the slot time wins. Returns ``slot_of`` (G,) with the UW frame
    index per slot, -1 for empty slots.
    """
    g = np.asarray(t_grid_ns, dtype=np.int64)
    t = np.asarray(t_uw_ns, dtype=np.int64)
    G = len(g)
    slot_of = np.full(G, -1, dtype=np.int64)
    if G == 0:
        return slot_of
    best_dist = np.full(G, np.iinfo(np.int64).max, dtype=np.int64)
    dg = np.diff(g)
    for j in np.asarray(sel_idx, dtype=np.int64):
        tj = int(t[j])
        i = int(np.searchsorted(g, tj))
        cands = [c for c in (i - 1, i) if 0 <= c < G]
        i = min(cands, key=lambda c: abs(int(g[c]) - tj))
        dist = abs(int(g[i]) - tj)
        spacing = max(int(dg[i - 1]) if i > 0 else 0, int(dg[i]) if i < G - 1 else 0)
        if G == 1:
            spacing = 2 * tol_ns
        if dist > spacing // 2 + tol_ns:
            continue
        if dist < best_dist[i]:
            best_dist[i] = dist
            slot_of[i] = j
    return slot_of


def uw_pts(t_grid_ns, slot_of, den: int = 600) -> np.ndarray:
    """PTS (ticks of 1/den s) of each occupied slot, relative to the first occupied slot (R1 invariant 4)."""
    g = np.asarray(t_grid_ns, dtype=np.int64)
    occ = np.flatnonzero(np.asarray(slot_of) >= 0)
    if len(occ) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.round((g[occ] - g[occ[0]]) * den / 1e9).astype(np.int64)
