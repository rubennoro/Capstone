"""W2 step 4: metrics of a policy's first chunk vs the demo, shared by ``first_chunk.py`` (torch env) and
``attribution_table.py`` (any env with numpy + scipy). numpy only; no torch, no zarr package.

Conventions (training's, ``behavior_prompting`` ``SequenceSampler`` + ``UmiTaskDataset`` with ``action_pose_repr:
relative``): chunk step k (0..15) is the pose at 60 Hz data row T + 3k, relative to the pose at row T (the current TCP
frame), so step 0 is about 0 and step 15 is 0.75 s ahead; past the episode end training repeats the last row.

* ``demo_displacement(dataset, index, row)``: the demo's own TCP displacement over the same horizon, in the TCP frame
  at T: d_k = R_T^T (p_{T+3k} - p_T), (16, 3) m (package low-dim via ``sim_deploy/scene/zarr_lowdim.py``).
* ``chunk_summary(eef_pos, demo, exec_h)``: per sample the last step's TCP z (toward the fingertips; P3's metric)
  and at step exec_h-1 (the last waypoint the manager sends), the end point projected on the unit direction of the
  demo's displacement at step 15 (``approach``), all in cm; their means, sample stds and standard errors.
* ``chunk_distance_cm(a, b)``: distance between two MEAN chunks (N, 16, 3) -> RMS over the 16 steps of the Euclidean
  distance between the sample-mean positions, cm (``d_real_cm`` when b is the real-image chunk).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

STEP_ROWS = 3        # obs/action down_sample_steps: 60 Hz rows -> 50 ms steps
H = 16               # action horizon (C4 action_steps)


@lru_cache(maxsize=64)
def _episode_poses(dataset: str, index: int) -> np.ndarray:
    from sim_deploy.scene.zarr_lowdim import load_episode
    return load_episode(Path(dataset), int(index))["poses"]


def demo_displacement(dataset: str | Path, index: int, row: int, horizon: int = H) -> np.ndarray:
    """(horizon, 3) m: the demo's TCP positions at rows row + 3k, in the TCP frame at `row` (last row repeated)."""
    P = _episode_poses(str(dataset), int(index))
    n = len(P)
    rows = np.minimum(row + STEP_ROWS * np.arange(horizon), n - 1)
    R, p = P[row, :3, :3], P[row, :3, 3]
    return ((P[rows, :3, 3] - p) @ R).astype(np.float64)      # R^T (p_k - p) for each k


def chunk_summary(eef_pos: np.ndarray, demo: np.ndarray | None, exec_h: int) -> dict:
    """eef_pos (N, 16, 3) m in the current TCP frame; demo (16, 3) m or None. Returns cm numbers."""
    e = np.asarray(eef_pos, np.float64) * 100.0
    n = e.shape[0]
    out = {"n": int(n)}

    def stat(name, v):
        out[name] = float(v.mean())
        out[name + "_std"] = float(v.std(ddof=1)) if n > 1 else 0.0
        out[name + "_se"] = out[name + "_std"] / np.sqrt(n) if n > 1 else 0.0
        out[name + "_samples"] = [round(float(x), 3) for x in v]

    stat("z_cm", e[:, -1, 2])
    stat("z_exec_cm", e[:, exec_h - 1, 2])
    out["end_mean_cm"] = [round(float(x), 3) for x in e[:, -1].mean(0)]
    if demo is not None:
        d = np.asarray(demo, np.float64) * 100.0
        dn = float(np.linalg.norm(d[-1]))
        u = d[-1] / dn if dn > 1e-9 else np.zeros(3)
        stat("approach_cm", e[:, -1] @ u)
        stat("approach_exec_cm", e[:, exec_h - 1] @ u)
        out["demo_along_cm"] = dn
        out["demo_along_exec_cm"] = float(d[exec_h - 1] @ u)
        out["demo_z_cm"] = float(d[-1, 2])
        out["demo_end_cm"] = [round(float(x), 3) for x in d[-1]]
        out["demo_dir"] = [round(float(x), 4) for x in u]
    return out


def chunk_distance_cm(a: np.ndarray, b: np.ndarray) -> float:
    """RMS over steps of |mean(a)_k - mean(b)_k|, cm; a, b (N, 16, 3) m."""
    ma, mb = np.asarray(a, np.float64).mean(0), np.asarray(b, np.float64).mean(0)
    return float(np.sqrt(np.mean(np.sum((ma - mb) ** 2, axis=-1))) * 100.0)


def paired_distance_cm(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over samples (same seed index) of the RMS-over-steps distance between the two chunks, cm."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.mean(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1), axis=-1))) * 100.0)
