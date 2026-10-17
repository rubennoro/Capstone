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


# --------------------------------------------------------------------------------------------------------------------
# Fill mode, align-exact (T3 / F2)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class FillSlots:
    """Fill-mode slots that iPhUMI's align stage maps one-to-one (``rgb_frame_idx == k``).

    align.py builds ``sample_times = arange(t(string_0), t(string_last), 1/60)`` (float seconds) and picks, for every
    sample, the most recent rgb/pose string that is ``<=`` it. The ms strings are therefore written as
    ``string_0 = S0`` (a ms boundary) and ``string_k = floor_ms(S0 + k/60 - margin)`` for k >= 1, so
    ``string_k <= sample_k < string_k+1`` with >= 0.5 ms of slack on both sides (float64 error at UNIX-time magnitude
    is ~0.2 us). Poses are sampled at the string times (align interpolates them to the sample times, <= 1.5 ms away).

    Images: each slot shows one of the two main frames around its *sample* time (what align pairs it with). At 30 fps
    every other sample falls between two frames; picking "nearest" there is biased whenever the frames are not
    exactly midway (phase drift of a real sensor clock, ms-quantised stamps), so the choice between the two is made
    by error diffusion: the candidate that keeps the running mean image-pose lag closest to zero, as long as its
    |lag| <= half the local frame interval + ``max_extra_ns``. ``S0`` is the ms boundary (among the first 17) that
    puts the frames on the sample grid on average over the whole episode. Result: mean lag ~ 0, |lag| <= half a
    frame interval + the residual phase (~1 ms).
    align uses ``n - 1`` samples (the last string is the end of its range); the last slot is still written.
    """

    sample_ns: np.ndarray      # (n,) rig-clock times of align's samples
    string_ns: np.ndarray      # (n,) rig-clock times of the written ms strings (poses are sampled here)
    string_unix_ns: np.ndarray  # (n,) UNIX ns of the strings (exact ms)
    src_idx: np.ndarray        # (n,) main frame shown in each slot

    def __len__(self) -> int:
        return len(self.sample_ns)


MS = 1_000_000


def _phase(t_ns: np.ndarray, s0: int, step: float) -> float:
    """Mean offset (ns) of the frames from the nearest sample-grid point (circular mean, step-periodic)."""
    if not len(t_ns):
        return 0.0
    ang = 2 * np.pi * np.mod((t_ns - s0).astype(np.float64), step) / step
    return float(np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi) * step)


def balanced_index(t: np.ndarray, g: np.ndarray, max_extra_ns: int = 2_000_000) -> np.ndarray:
    """Frame shown at each slot time ``g``: the earlier or later neighbour, chosen by error diffusion (see FillSlots)."""
    t = np.asarray(t, dtype=np.int64)
    hi = np.clip(np.searchsorted(t, g, side="left"), 1, len(t) - 1)
    lo = hi - 1
    lag_lo = (t[lo] - g).astype(np.float64)
    lag_hi = (t[hi] - g).astype(np.float64)
    half = (t[hi] - t[lo]) / 2.0 + max_extra_ns
    out = np.empty(len(g), dtype=np.int64)
    cum = 0.0
    for k in range(len(g)):
        ok_lo, ok_hi = abs(lag_lo[k]) <= half[k], abs(lag_hi[k]) <= half[k]
        if ok_lo and ok_hi:
            use_hi = abs(cum + lag_hi[k]) < abs(cum + lag_lo[k])
        elif ok_lo or ok_hi:
            use_hi = ok_hi
        else:                                   # outside the frame range: nearest
            use_hi = abs(lag_hi[k]) < abs(lag_lo[k])
        out[k] = hi[k] if use_hi else lo[k]
        cum += lag_hi[k] if use_hi else lag_lo[k]
    return out


def fill_slots(t_main_ns, t_first_ns: int, t_last_ns: int, wall_offset_ns: int, grid_hz: float = 60.0,
               margin_ns: int = 500_000, max_extra_ns: int = 2_000_000) -> FillSlots:
    """Align-exact fill slots covering ``[t_first, t_last]`` (rig clock). See ``FillSlots``."""
    t = np.asarray(t_main_ns, dtype=np.int64)
    step = 1e9 / grid_hz
    wall = int(wall_offset_ns)
    s0_min = -((-(int(t_first_ns) + wall)) // MS) * MS            # ceil to ms (UNIX)
    win = t[(t >= t_first_ns) & (t <= t_last_ns)] + wall
    cands = [s0_min + m * MS for m in range(int(np.ceil(step / MS)) + 1)]
    s0 = min(cands, key=lambda s: (round(abs(_phase(win, s, step)) / 1e3), s))   # 1-us resolution, earliest on ties
    last_unix = int(t_last_ns) + wall
    if last_unix < s0:
        raise ValueError("episode shorter than one fill slot")
    # align's sample times, reproduced bit for bit: datetime(string_0).timestamp(), then np.arange(start, stop, 1/60)
    # (numpy fills start + k * ((start + step) - start), which drifts by up to ~1 ULP of the UNIX time (0.24 us) per
    # sample, i.e. up to ~1 ms per minute; the margin must hold against these values, not the ideal grid)
    import datetime as _dt

    s0f = (_dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(milliseconds=s0 // MS)).timestamp()
    n_max = int(np.floor((last_unix - s0) / step + 1e-9)) + 2
    samples_f = np.arange(s0f, s0f + (n_max + 1) / grid_hz, 1.0 / grid_hz, dtype=np.float64)[:n_max]
    sample_unix = np.round(samples_f * 1e9).astype(np.int64)
    n = int(np.searchsorted(sample_unix, last_unix, side="right"))
    sample_unix = sample_unix[:n]
    string_unix = ((sample_unix - int(margin_ns)) // MS) * MS
    string_unix[0] = s0
    if np.any(np.diff(string_unix) <= 0):
        raise ValueError("fill slot strings not strictly increasing")
    sample_ns = sample_unix - wall
    return FillSlots(sample_ns, string_unix - wall, string_unix, balanced_index(t, sample_ns, max_extra_ns))


def lag_stats(t_main_ns, slots: FillSlots, obs_step: int = 3) -> dict:
    """Image-vs-pose lag over the slots align uses (all but the last), and how often the k / k-``obs_step``
    observation pair would show the same main frame."""
    t = np.asarray(t_main_ns, dtype=np.int64)
    n = max(len(slots) - 1, 1)
    lag = (t[slots.src_idx[:n]] - slots.sample_ns[:n]) / 1e6
    a, b = slots.src_idx[:max(n - obs_step, 0)], slots.src_idx[obs_step:n]
    same_mask = a == b
    # a same-frame pair is "explained" when the camera lost a frame next to it (gap > 1.5 x the median interval)
    dt = np.diff(t)
    gap = np.zeros(len(t), bool)
    if len(dt):
        big = dt > 1.5 * np.median(dt)
        gap[:-1] |= big
        gap[1:] |= big
    unexplained = int(np.sum(same_mask & ~gap[a]))
    return {"lag_ms_mean": float(np.mean(lag)), "lag_ms_max_abs": float(np.max(np.abs(lag))),
            "lag_ms_p99_abs": float(np.percentile(np.abs(lag), 99)), "obs_pairs_same_frame": int(same_mask.sum()),
            "obs_pairs_same_frame_no_drop": unexplained,
            "max_main_gap_ms": float(dt.max() / 1e6) if len(dt) else None}
