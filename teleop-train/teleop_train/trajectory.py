"""The 60 Hz grid, pose and width interpolation, motion detection and trimming (PLAN.md §5.2 steps 1-5). Owner: E2.

Pure functions on numpy arrays (integer ns times, float64 values); no file IO. The constants and signatures below
are the contract with the other modules.

Precision: times stay int64 ns (around 1e12 on the laptop clock). Interpolation weights are computed from int64
differences (relative to the neighbouring state, or to the first state time for Slerp), converted to float64 only
after the subtraction, so no precision is lost to the large absolute times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

GRID_HZ = 60
TRANS_SPEED_MPS = 0.005          # moving: TCP speed above 5 mm/s ...
ROT_SPEED_DEG_S = 1.0            # ... or rotation speed above 1 deg/s ...
WIDTH_RATE_MPS = 0.005           # ... or |gripper width rate| above 5 mm/s
SMOOTH_S = 0.25                  # each speed is averaged over this window (centred) first
MARGIN_S = 0.5                   # kept before the first and after the last moving step
MAX_STATE_GAP_NS = 50_000_000    # §5.3 gap limits inside the kept span
MAX_MAIN_GAP_NS = 100_000_000
MAX_UW_GAP_NS = 200_000_000
MIN_KEPT_S = 2.0


class TrajectoryError(ValueError):
    """The episode cannot be used; the message is the one-line reason stored in status.json."""


@dataclass(eq=False)
class GridPoses:
    t_grid: np.ndarray           # (N,) int64
    pos: np.ndarray              # (N, 3) float64, metres
    rotvec: np.ndarray           # (N, 3) float64, rotation vector (scipy as_rotvec)
    quat: np.ndarray             # (N, 4) float64, scipy xyzw, for speed computations
    width: np.ndarray            # (N,) float64, metres


def common_window(t_state_ns: np.ndarray, main_t_ns: np.ndarray, uw_t_ns: np.ndarray) -> tuple[int, int]:
    """(lo, hi): the latest first sample and the earliest last sample of the three streams.
    TrajectoryError if hi <= lo."""
    streams = (("arm states", t_state_ns), ("main frames", main_t_ns), ("ultrawide frames", uw_t_ns))
    for name, t in streams:
        if np.size(t) == 0:
            raise TrajectoryError(f"no {name} with a time; check the episode's {name} file")
    lo = int(max(int(np.asarray(t)[0]) for _, t in streams))
    hi = int(min(int(np.asarray(t)[-1]) for _, t in streams))
    if hi <= lo:
        raise TrajectoryError(f"the arm states, main frames and ultrawide frames do not overlap in time "
                              f"(common window {(hi - lo) / 1e6:.1f} ms); check the recording's clocks")
    return lo, hi


def _round_div(num: np.ndarray, den: int) -> np.ndarray:
    """round(num / den) for non-negative int64 num and positive int den, halves rounded up, exact integers."""
    return (2 * num + den) // (2 * den)


def make_grid(lo_ns: int, hi_ns: int, hz: int = GRID_HZ) -> np.ndarray:
    """t_k = lo + round(k * 1e9 / hz) for every k with t_k <= hi (int64; exact integer arithmetic).

    Exact halves (possible only when hz does not divide 2e9 evenly into thirds etc.) round up. Empty if hi < lo."""
    lo, hi, hz = int(lo_ns), int(hi_ns), int(hz)
    if hz <= 0:
        raise ValueError(f"grid rate must be positive, got {hz}")
    if hi < lo:
        return np.zeros(0, np.int64)
    k_max = (hi - lo) * hz // 1_000_000_000 + 1          # one past any possible k
    k = np.arange(k_max + 1, dtype=np.int64)
    t = lo + _round_div(k * 1_000_000_000, hz)
    return t[t <= hi].astype(np.int64)


def interpolate(t_state_ns: np.ndarray, T_state: np.ndarray, width_m: np.ndarray, t_grid: np.ndarray) -> GridPoses:
    """Position: linear; rotation: scipy Slerp between the two neighbouring states; width: linear. A grid time equal
    to a state time gives that state exactly. ValueError if any grid time is outside [t_state[0], t_state[-1]].

    Also ValueError for mismatched shapes or state times that are not strictly increasing. The rotation part of
    T_state is read with Rotation.from_matrix (the rigid-transform check is the reader's / convert's job)."""
    t_s = np.asarray(t_state_ns, dtype=np.int64)
    T = np.asarray(T_state, dtype=np.float64)
    w = np.asarray(width_m, dtype=np.float64)
    t_g = np.asarray(t_grid, dtype=np.int64)
    n = t_s.shape[0]
    if t_s.ndim != 1 or T.shape != (n, 4, 4) or w.shape != (n,) or t_g.ndim != 1:
        raise ValueError(f"interpolate: shapes t_state {t_s.shape}, T_state {T.shape}, width {w.shape}, "
                         f"t_grid {t_g.shape} do not match")
    if n == 0:
        raise ValueError("interpolate: no states")
    if n > 1 and np.any(np.diff(t_s) <= 0):
        raise ValueError("interpolate: state times are not strictly increasing")
    m = t_g.shape[0]
    if m and (t_g.min() < t_s[0] or t_g.max() > t_s[-1]):
        raise ValueError("interpolate: grid times outside the state times")

    p_s = T[:, :3, 3]
    R_s = Rotation.from_matrix(T[:, :3, :3])
    quat_s = R_s.as_quat()
    rotvec_s = R_s.as_rotvec()

    # Index of the last state at or before each grid time; hits are grid times equal to a state time.
    idx = np.clip(np.searchsorted(t_s, t_g, side="right") - 1, 0, n - 1)
    hit = t_s[idx] == t_g
    pos = p_s[idx].copy()
    width = w[idx].copy()
    quat = quat_s[idx].copy()
    rotvec = rotvec_s[idx].copy()

    miss = ~hit
    if np.any(miss):                                      # here n >= 2 and idx <= n - 2
        i = idx[miss]
        frac = ((t_g[miss] - t_s[i]).astype(np.float64) / (t_s[i + 1] - t_s[i]).astype(np.float64))
        pos[miss] = p_s[i] + frac[:, None] * (p_s[i + 1] - p_s[i])
        width[miss] = w[i] + frac * (w[i + 1] - w[i])
        rel_s = (t_s - t_s[0]).astype(np.float64)        # exact below 2**53 ns (~104 days)
        R_g = Slerp(rel_s, R_s)((t_g[miss] - t_s[0]).astype(np.float64))
        quat[miss] = R_g.as_quat()
        rotvec[miss] = R_g.as_rotvec()
    return GridPoses(t_grid=t_g.copy(), pos=pos, rotvec=rotvec, quat=quat, width=width)


def _interval_to_steps(r: np.ndarray, n: int) -> np.ndarray:
    """Maps N-1 per-interval rates (interval j = steps j..j+1) onto N steps: step k gets the mean of the intervals
    that touch it (j = k-1 and j = k), i.e. one interval at each end and two inside."""
    if n < 2:
        return np.zeros(n, np.float64)
    out = np.empty(n, np.float64)
    out[0], out[-1] = r[0], r[-1]
    out[1:-1] = 0.5 * (r[:-1] + r[1:])
    return out


def _centred_mean(x: np.ndarray, half: int) -> np.ndarray:
    """Mean of x[k-half .. k+half] for every k, over the samples that exist (shorter windows at the edges)."""
    n = x.shape[0]
    if n == 0 or half <= 0:
        return x.astype(np.float64, copy=True)
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    k = np.arange(n)
    a = np.clip(k - half, 0, n)
    b = np.clip(k + half + 1, 0, n)
    return (c[b] - c[a]) / (b - a)


def step_rates(poses: GridPoses, hz: int = GRID_HZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(trans m/s, rot deg/s, |width rate| m/s) per grid step, smoothed as moving_mask uses them (each (N,)).

    Per interval between consecutive steps: |Δpos| * hz, the angle of the relative rotation * hz (deg), |Δwidth| * hz.
    Mapped onto steps by _interval_to_steps, then a centred mean over 2 * (round(SMOOTH_S * hz) // 2) + 1 steps
    (15 at 60 Hz; an even round(SMOOTH_S * hz) gets one more step so the window stays centred)."""
    n = poses.t_grid.shape[0]
    if n < 2:
        z = np.zeros(n, np.float64)
        return z, z.copy(), z.copy()
    d_pos = np.linalg.norm(np.diff(poses.pos, axis=0), axis=1) * hz
    R = Rotation.from_quat(poses.quat)
    d_rot = np.degrees((R[:-1].inv() * R[1:]).magnitude()) * hz
    d_w = np.abs(np.diff(poses.width)) * hz
    half = max(int(round(SMOOTH_S * hz)), 1) // 2
    return tuple(_centred_mean(_interval_to_steps(r, n), half) for r in (d_pos, d_rot, d_w))  # type: ignore[return-value]


def moving_mask(poses: GridPoses, hz: int = GRID_HZ) -> np.ndarray:
    """(N,) bool: TCP speed > TRANS_SPEED_MPS or rotation speed > ROT_SPEED_DEG_S or |width rate| >
    WIDTH_RATE_MPS, each rate computed between consecutive grid steps and averaged over SMOOTH_S (centred).

    The N-1 interval rates go onto the N steps as in _interval_to_steps; the smoothing is step_rates'."""
    v, w, g = step_rates(poses, hz)
    return (v > TRANS_SPEED_MPS) | (w > ROT_SPEED_DEG_S) | (g > WIDTH_RATE_MPS)


def trim_span(t_grid: np.ndarray, moving: np.ndarray, margin_s: float = MARGIN_S) -> tuple[int, int]:
    """(i0, i1), the kept grid steps t_grid[i0:i1]: from margin_s before the first moving step to margin_s after
    the last one, clamped to the grid. Pauses in between are kept. TrajectoryError if nothing moves or the kept span
    is shorter than MIN_KEPT_S.

    i0 is the first step with t >= t_first_moving - margin, i1 - 1 the last with t <= t_last_moving + margin. The
    kept span is t_grid[i1 - 1] - t_grid[i0]."""
    t = np.asarray(t_grid, dtype=np.int64)
    mv = np.asarray(moving, dtype=bool)
    if t.shape != mv.shape:
        raise ValueError(f"trim_span: t_grid {t.shape} and moving {mv.shape} differ")
    where = np.flatnonzero(mv)
    if where.size == 0:
        raise TrajectoryError(f"no motion found (no step above {TRANS_SPEED_MPS * 1e3:g} mm/s, "
                              f"{ROT_SPEED_DEG_S:g} deg/s or {WIDTH_RATE_MPS * 1e3:g} mm/s gripper rate)")
    margin_ns = int(round(margin_s * 1e9))
    i0 = int(np.searchsorted(t, t[where[0]] - margin_ns, side="left"))
    i1 = int(np.searchsorted(t, t[where[-1]] + margin_ns, side="right"))
    kept_ns = int(t[i1 - 1] - t[i0])
    if kept_ns < round(MIN_KEPT_S * 1e9):
        raise TrajectoryError(f"kept span {kept_ns / 1e9:.2f} s is shorter than {MIN_KEPT_S:g} s")
    return i0, i1


def max_gap_ns(t_samples: np.ndarray, lo_ns: int, hi_ns: int) -> int:
    """Largest gap between consecutive samples that matter for [lo, hi]: the samples inside it plus the last one at
    or before lo and the first one at or after hi (the ones interpolation or frame selection may use).

    Where no sample lies at or before lo (or at or after hi) the first (last) sample is used instead: coverage of
    the window is common_window's job. 0 when fewer than two samples are involved."""
    t = np.asarray(t_samples, dtype=np.int64)
    n = t.shape[0]
    if n < 2:
        return 0
    a = max(int(np.searchsorted(t, int(lo_ns), side="right")) - 1, 0)
    b = min(int(np.searchsorted(t, int(hi_ns), side="left")), n - 1)
    if b <= a:
        return 0
    return int(np.max(np.diff(t[a:b + 1])))


def check_gaps(t_state_ns: np.ndarray, main_t_ns: np.ndarray, uw_t_ns: np.ndarray, lo_ns: int, hi_ns: int) -> None:
    """TrajectoryError naming the stream and the gap (ms) if a max_gap_ns exceeds its MAX_*_GAP_NS.

    Streams are checked in the order arm states, main frames, ultrawide frames; the first failure is reported."""
    for name, t, limit in (("arm states", t_state_ns, MAX_STATE_GAP_NS),
                           ("main frames", main_t_ns, MAX_MAIN_GAP_NS),
                           ("ultrawide frames", uw_t_ns, MAX_UW_GAP_NS)):
        gap = max_gap_ns(t, lo_ns, hi_ns)
        if gap > limit:
            raise TrajectoryError(f"gap of {gap / 1e6:.1f} ms between consecutive {name} inside the kept span "
                                  f"(limit {limit / 1e6:g} ms)")
