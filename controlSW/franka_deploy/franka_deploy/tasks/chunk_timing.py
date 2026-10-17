"""Task-side chunk time-scaling and the pause-mode settle gate (D4).

Time-scaling: the bridge's per-update speed clamp only slows the ARM, while the gripper keeps its own schedule
(+0.1 s lead), which breaks arm-gripper coordination. Instead each chunk is stretched in time, arm AND gripper
together (one shared timestamp array), so its peak segment speed is <= margin * v_max (and <= margin * w_max for
rotation). The bridge clamp then becomes a backstop that should never fire. Segments: anchor -> action 0 -> ... at
1/f each (timestamps are uniform before scaling).

Settle gate (pause mode): before the next observation, wait until the arm has caught up with its plan: clamp
backlog < backlog_max (2 mm) AND (|safe target - FK| < track_max (3 mm) OR the arm is stationary, i.e. the
deviation changed < still_tol over the last window: a steady impedance bias is not "unsettled"); give up after
timeout (1 s) -> SettleTimeoutError -> hold.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation


class SettleTimeoutError(RuntimeError):
    pass


def chunk_scale(poses_xyz_wxyz: np.ndarray, anchor_xyz_wxyz: np.ndarray | None, dt: float, v_max: float,
                w_max: float, margin: float = 0.9) -> tuple[float, float, float]:
    """Return (scale >= 1, peak linear speed, peak angular speed) of the unscaled chunk."""
    P = np.asarray(poses_xyz_wxyz, float)
    if anchor_xyz_wxyz is not None:
        P = np.vstack([np.asarray(anchor_xyz_wxyz, float)[None], P])
    if len(P) < 2:
        return 1.0, 0.0, 0.0
    v = np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1) / dt
    r = Rotation.from_quat(P[:, [4, 5, 6, 3]])
    w = (r[:-1].inv() * r[1:]).magnitude() / dt
    vp, wp = float(v.max()), float(w.max())
    s = max(1.0, vp / (margin * v_max), wp / (margin * w_max))
    return float(s), vp, wp


def scale_timestamps(ts: np.ndarray, s: float) -> np.ndarray:
    ts = np.asarray(ts, float)
    return ts[0] + (ts - ts[0]) * s


def settle_gate(read, timeout_s: float = 1.0, backlog_max: float = 0.002, track_max: float = 0.003,
                still_tol: float = 0.0005, window_s: float = 0.05, poll_s: float = 0.01, sleep=time.sleep,
                clock=time.monotonic) -> dict:
    """`read()` -> (backlog_m or None, track_dev_m). Returns stats; raises SettleTimeoutError."""
    t0 = clock()
    hist: list[tuple[float, float]] = []
    while True:
        backlog, dev = read()
        now = clock()
        hist.append((now, dev))
        while len(hist) > 1 and now - hist[0][0] > window_s:
            hist.pop(0)
        still = len(hist) > 1 and now - hist[0][0] >= window_s * 0.8 and \
            max(d for _, d in hist) - min(d for _, d in hist) < still_tol
        if (backlog is None or backlog < backlog_max) and (dev < track_max or still):
            return {"settle_s": now - t0, "backlog_m": backlog, "track_dev_m": dev, "stationary": bool(still and dev >= track_max)}
        if now - t0 > timeout_s:
            raise SettleTimeoutError(f"arm not settled after {timeout_s:.1f} s (backlog {backlog}, track_dev {dev:.4f} m)")
        sleep(poll_s)
