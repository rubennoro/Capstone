"""The 60 Hz record grid that drives every exported list (brief D6, P1 §4.2, RECONCILIATION R5).

The iPhone pipeline hardcodes a 1/60 s main-frame cadence (detect.py:69, align fps 60), so:
- native mode (main camera >= ``native_min_fps``): grid = the actual main frame times (VFR, gaps stay gaps, like the
  iPhone's dropped frames);
- fill mode (slower main camera, e.g. 30 fps): a uniform 60 Hz grid from the first to the last main frame; each slot
  shows the NEAREST main frame (lead's decision on review finding F4, superseding the brief's "repeat the latest
  frame"). Poses are interpolated at the slot time, and on the iPhone image time == pose time, so "latest" would add a
  systematic ~8 ms image-behind-pose lag; nearest keeps the error zero-mean (|error| <= half a main frame period).
  Exact ties (a slot exactly midway between two frames, as with an ideal 30 fps camera) alternate between the earlier
  and the later frame so the mean error stays zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    t_ns: np.ndarray  # (G,) int64 corrected rig-clock times of the slots
    src_idx: np.ndarray  # (G,) int64 main frame shown in each slot
    mode: str  # "native" | "fill"
    fps_main: float  # measured median main fps

    def __len__(self) -> int:
        return len(self.t_ns)


def build_grid(t_main_ns, grid_hz: float = 60.0, native_min_fps: float = 50.0) -> Grid:
    t = np.asarray(t_main_ns, dtype=np.int64)
    if len(t) < 2:
        raise ValueError("need at least 2 main frames to build a grid")
    if np.any(np.diff(t) <= 0):
        raise ValueError("main frame times must be strictly increasing")
    fps = 1e9 / float(np.median(np.diff(t)))
    if fps >= native_min_fps:
        return Grid(t.copy(), np.arange(len(t), dtype=np.int64), "native", fps)
    step = 1e9 / grid_hz
    n = int(np.floor((t[-1] - t[0]) / step + 1e-9)) + 1
    k = np.arange(n, dtype=np.float64)
    g = t[0] + np.round(k * step).astype(np.int64)
    g = g[g <= t[-1]]
    return Grid(g, nearest_index(t, g), "fill", fps)


def nearest_index(t: np.ndarray, g: np.ndarray, tie_ns: int = 1_000) -> np.ndarray:
    """Index of the frame in ``t`` nearest to each ``g``. Ties (within ``tie_ns``) alternate earlier/later."""
    hi = np.clip(np.searchsorted(t, g, side="left"), 1, len(t) - 1)
    lo = hi - 1
    d_lo = g - t[lo]
    d_hi = t[hi] - g
    out = np.where(d_hi < d_lo, hi, lo).astype(np.int64)
    tie = np.flatnonzero(np.abs(d_hi - d_lo) <= tie_ns)
    out[tie[1::2]] = hi[tie[1::2]]
    out[tie[0::2]] = lo[tie[0::2]]
    return out


def fps_abort(t_main_ns, min_frames_per_s: float, warmup_s: float = 1.0) -> str | None:
    """App rule (VC:1821): after the first second, every 1-s window must hold >= ``min_frames_per_s`` main frames.

    Windows are [t0 + warmup + k, t0 + warmup + k + 1) for every whole window inside the episode. Returns a reason
    string, or None if the episode passes.
    """
    t = np.asarray(t_main_ns, dtype=np.int64)
    start = int(t[0]) + int(warmup_s * 1e9)
    k = 0
    while start + (k + 1) * 1_000_000_000 <= int(t[-1]):
        a = start + k * 1_000_000_000
        n = int(np.searchsorted(t, a + 1_000_000_000, side="left") - np.searchsorted(t, a, side="left"))
        if n < min_frames_per_s:
            return f"main fps {n} < {min_frames_per_s:g} in the 1-s window starting {(a - t[0]) / 1e9:.1f} s"
        k += 1
    return None
