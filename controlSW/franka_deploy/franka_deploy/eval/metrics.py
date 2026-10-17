"""Evaluation metrics (pure numpy, unit-tested).

- wilson(k, n): 95 % Wilson score interval.
- relative_positions(p, rotvec_or_quat0): TCP positions expressed in the episode-start TCP frame (frame-invariant
  comparison between robot rollouts and iPhone demos).
- dtw(a, b): dynamic time warping, mean Euclidean step cost along the optimal path (same units as a, b).
- phases(width, height, t): approach / grasp / lift / hold segmentation from the gripper width + height along the
  gravity "up" axis.
- chunk_consistency(...): overlap disagreement between consecutive chunks.
- ood_fraction(x, lo, hi): share of values outside the normalizer's training range.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def relative_positions(pos: np.ndarray, rot0: Rotation) -> np.ndarray:
    """p_rel = R0^T (p - p0)."""
    pos = np.asarray(pos, float)
    return rot0.inv().apply(pos - pos[0])


def resample(t: np.ndarray, x: np.ndarray, hz: float) -> np.ndarray:
    t = np.asarray(t, float)
    tt = np.arange(t[0], t[-1] + 1e-9, 1.0 / hz)
    return np.stack([np.interp(tt, t, x[:, j]) for j in range(x.shape[1])], 1)


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1))
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(choices))
            prev = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
            D[i, j] = cost[i - 1, j - 1] + choices[k]
            L[i, j] = L[prev] + 1
    return float(D[n, m] / L[n, m])


def phases(width: np.ndarray, height: np.ndarray, t: np.ndarray, min_close_m: float = 0.01,
           min_lift_m: float = 0.01) -> dict | None:
    """Segments from the width: open level = median of the first 10 %; close start = first sample below
    open - 0.2 * depth; grasp end = first sample within 2 mm of the closed plateau (min over the closed part); lift =
    from grasp end until the height peak (while closed); hold = after the peak until the width re-opens past the
    midpoint (or the end). Returns None if no grasp (closing depth < min_close_m) is found."""
    w, h, t = np.asarray(width, float), np.asarray(height, float), np.asarray(t, float)
    n = len(w)
    w_open = float(np.median(w[: max(3, n // 10)]))
    w_min = float(w.min())
    depth = w_open - w_min
    if depth < min_close_m:
        return None
    i_close = int(np.argmax(w < w_open - 0.2 * depth))
    mid = w_open - 0.5 * depth
    reopen = np.where((np.arange(n) > i_close) & (w > mid))[0]
    reopen = [i for i in reopen if i > int(np.argmin(w))]
    i_open = int(reopen[0]) if reopen else n - 1
    closed = w[i_close:i_open + 1]
    plateau = float(np.min(closed))
    i_grasp = i_close + int(np.argmax(closed <= plateau + 0.002))
    seg = h[i_grasp:i_open + 1]
    i_peak = i_grasp + int(np.argmax(seg))
    lift = float(h[i_peak] - h[i_grasp])
    return {"t_close_start": float(t[i_close] - t[0]), "t_grasp": float(t[i_grasp] - t[0]),
            "t_lift_peak": float(t[i_peak] - t[0]), "t_release": float(t[i_open] - t[0]),
            "approach_s": float(t[i_close] - t[0]), "grasp_s": float(t[i_grasp] - t[i_close]),
            "lift_s": float(t[i_peak] - t[i_grasp]), "hold_s": float(t[i_open] - t[i_peak]),
            "grasp_width_m": plateau, "open_width_m": w_open, "lift_height_m": lift, "lifted": lift >= min_lift_m}


def speed_profile(t: np.ndarray, pos: np.ndarray, scale: np.ndarray | None = None) -> dict:
    """Speed along the path; with `scale` (chunk time-scale per sample) also the speed the policy intended."""
    t, pos = np.asarray(t, float), np.asarray(pos, float)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    out = {"v_p50": float(np.percentile(v, 50)), "v_p95": float(np.percentile(v, 95)), "v_max": float(v.max())}
    if scale is not None:
        s = np.asarray(scale, float)[1:]
        vi = v * np.where(np.isfinite(s), s, 1.0)
        out.update(intended_v_p95=float(np.percentile(vi, 95)), intended_v_max=float(vi.max()))
    return out


def chunk_consistency(prev_abs_pos: np.ndarray, prev_ts: np.ndarray, next_abs_pos: np.ndarray, next_ts: np.ndarray) -> float | None:
    """Mean distance between the previous chunk's prediction and the next chunk over their common time span."""
    lo, hi = max(prev_ts[0], next_ts[0]), min(prev_ts[-1], next_ts[-1])
    if hi <= lo:
        return None
    tt = np.linspace(lo, hi, 10)
    a = np.stack([np.interp(tt, prev_ts, prev_abs_pos[:, j]) for j in range(3)], 1)
    b = np.stack([np.interp(tt, next_ts, next_abs_pos[:, j]) for j in range(3)], 1)
    return float(np.linalg.norm(a - b, axis=1).mean())


def ood_fraction(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: float = 0.0) -> tuple[float, float]:
    """(share of values outside [lo - margin*range, hi + margin*range], max excess in units of the range)."""
    x = np.asarray(x, float).reshape(-1, len(lo))
    rng = np.maximum(np.asarray(hi) - np.asarray(lo), 1e-9)
    below = (np.asarray(lo) - margin * rng - x) / rng
    above = (x - np.asarray(hi) - margin * rng) / rng
    exc = np.maximum(below, above)
    return float((exc > 0).mean()), float(max(0.0, exc.max()))
