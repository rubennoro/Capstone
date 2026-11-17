"""Frame selection on the 60 Hz output grid (pure functions, int64 nanoseconds).

Inputs are the grid times ``t_grid`` (N,) and per-camera **mid-exposure** times (``RigConfig.camera_time_ns``),
all on one clock. Camera times must be strictly increasing (row i = decoded frame i, CONTRACT).

* Main camera: :func:`nearest_frame_index`, the nearest captured frame per grid step (fills in when the main
  camera runs below 60 fps), with a validity flag ``|dt| <= max_dt_ns``.
* Ultrawide: :func:`ultrawide_substream` picks a ~10 Hz sub-stream (the frame nearest to each point of a
  ``rate_hz`` schedule starting at ``t_grid[0]``, duplicates removed) and, per grid step, the local index of the
  latest selected frame at or before that time, clamped at 0. That is iPhUMI's ``_compute_frame_indices`` rule
  (``align.py``: ``clip(searchsorted(ts, t, 'right') - 1, 0, len - 1)``), so ``upsample_local`` is exactly iPhUMI's
  per-episode ``pose_idx_to_ultrawide_idx`` and the zarr maps follow from it as in ``gen_replay_buffer_zarr.py``:

      n_uw            = upsample_local[-1] + 1        (selected frames after the last used one are dropped)
      downsample_local = searchsorted(upsample_local, arange(n_uw), 'left')    (:func:`downsample_local`)
      meta/upsample_index_*   += previous episodes' ultrawide total
      meta/downsample_index_* += previous episodes' main total (prev_main_end)
      meta/episode_ends_*      = cumulative n_uw
"""

from __future__ import annotations

import numpy as np

__all__ = ["nearest_frame_index", "latest_at_or_before", "ultrawide_substream", "downsample_local"]


def _times(t, name: str) -> np.ndarray:
    t = np.asarray(t)
    if t.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D array, got shape {t.shape}")
    if t.dtype.kind not in "iu":
        raise TypeError(f"{name}: expected integer nanoseconds, got {t.dtype}")
    return t.astype(np.int64, copy=False)


def _increasing(t: np.ndarray, name: str) -> None:
    if len(t) == 0:
        raise ValueError(f"{name}: empty")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: must be strictly increasing")


def nearest_frame_index(t_grid, t_cam, max_dt_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """For each grid time, the index of the nearest camera frame (ties -> the earlier frame).

    Returns ``(idx (N,) int64, valid (N,) bool)`` with ``valid = |t_cam[idx] - t_grid| <= max_dt_ns``.
    """
    t_grid = _times(t_grid, "t_grid")
    t_cam = _times(t_cam, "t_cam")
    _increasing(t_cam, "t_cam")
    hi = np.clip(np.searchsorted(t_cam, t_grid, side="left"), 0, len(t_cam) - 1)
    lo = np.clip(hi - 1, 0, len(t_cam) - 1)
    d_hi = np.abs(t_cam[hi] - t_grid)
    d_lo = np.abs(t_cam[lo] - t_grid)
    idx = np.where(d_lo <= d_hi, lo, hi).astype(np.int64)
    valid = np.abs(t_cam[idx] - t_grid) <= int(max_dt_ns)
    return idx, valid


def latest_at_or_before(t_query, t_frames) -> np.ndarray:
    """Index of the latest frame with ``t_frames <= t_query``, clamped to ``[0, len - 1]`` (iPhUMI rule)."""
    t_query = _times(t_query, "t_query")
    t_frames = _times(t_frames, "t_frames")
    _increasing(t_frames, "t_frames")
    return np.clip(np.searchsorted(t_frames, t_query, side="right") - 1, 0, len(t_frames) - 1).astype(np.int64)


def ultrawide_substream(t_grid, t_uw, rate_hz: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Select a ~``rate_hz`` ultrawide sub-stream over the grid span.

    Returns ``(sel, upsample_local)``:

    * ``sel`` (M,) int64: indices into the ultrawide frames, strictly increasing, the frame nearest to each
      schedule time ``t_grid[0] + k / rate_hz`` (``k = 0, 1, ...`` while ``<= t_grid[-1]``), duplicates removed, and
      trimmed after the last frame any grid step uses (so ``M == upsample_local[-1] + 1``);
    * ``upsample_local`` (N,) int64: per grid step, the index into ``sel`` of the latest selected frame at or before
      that step's time, clamped at 0. Non-decreasing, starts at 0.
    """
    t_grid = _times(t_grid, "t_grid")
    t_uw = _times(t_uw, "t_uw")
    _increasing(t_grid, "t_grid")
    _increasing(t_uw, "t_uw")
    if not rate_hz > 0:
        raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
    period = 1e9 / float(rate_hz)
    n_sched = int(np.floor((t_grid[-1] - t_grid[0]) / period)) + 1
    sched = t_grid[0] + np.round(np.arange(n_sched) * period).astype(np.int64)
    near, _ = nearest_frame_index(sched, t_uw, max_dt_ns=np.iinfo(np.int64).max)
    sel = np.unique(near)                               # nearest is monotone in time: sorted == schedule order
    up = latest_at_or_before(t_grid, t_uw[sel])
    sel = sel[: int(up[-1]) + 1]
    return sel.astype(np.int64), up


def downsample_local(upsample_local) -> np.ndarray:
    """Per selected ultrawide frame, the first grid step that uses it (iPhUMI ``downsample_mapping``)."""
    up = np.asarray(upsample_local, dtype=np.int64)
    n_uw = int(up[-1]) + 1
    return np.searchsorted(up, np.arange(n_uw, dtype=np.int64), side="left").astype(np.int64)
