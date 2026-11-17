# Vendored/adapted from sensor_rig/sensor_rig/pose/trajectory.py at commit 4be9982.
# Changes: no sensor_rig.pose.frames dependency (frame names checked locally); the IMU rotation in frame S comes
# from dataproc RigConfig (cameras.<role>.T_imu_cam) instead of sensor_rig's rig.T(); times are on the dataproc
# IMU time base (RigConfig.camera_time_ns / imu_time_ns).
"""Trajectories (TUM / ORB-SLAM3 EuRoC), interpolation at grid times, and sanity checks against the IMU.

Times are int64 ns on the IMU time base of the rig yaml (``RigConfig.camera_time_ns`` for camera frames,
``RigConfig.imu_time_ns`` for IMU samples). ``Trajectory.frame`` names the sensor frame S whose pose ``T_W_S`` the
rows hold (``imu``, ``ultrawide`` or ``main``); ``world`` names the world (``z_up_gravity`` once gravity-aligned). Trajectory files are parsed with exact decimal arithmetic so a ``t`` written with 9 decimals round-trips to the
same integer ns (float64 cannot hold ~1e15 ns exactly).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


__all__ = [
    "GRAVITY", "FRAME_ALIASES", "rig_frame", "Trajectory", "detect_time_unit", "load_tum", "load_orbslam3_euroc",
    "save_tum", "gravity_check", "gravity_align", "time_offset_check", "MOTION_LIMITS", "motion_consistency_check",
    "SCALE_TOLERANCE", "SCALE_MIN_EXCITATION", "SCALE_MAX_STD_ERR", "SCALE_MIN_ATTENUATION", "position_jitter",
    "scale_check",
]

GRAVITY = 9.80665

# Trajectory frame names. Trajectory files may say "imu0" (EuRoC name); map it.
FRAME_ALIASES = {"imu0": "imu", "imu": "imu", "main": "main", "ultrawide": "ultrawide"}


def rig_frame(name: str) -> str:
    if name not in FRAME_ALIASES:
        raise ValueError(f"unknown trajectory frame {name!r}; expected one of {sorted(FRAME_ALIASES)}")
    return FRAME_ALIASES[name]


@dataclass
class Trajectory:
    t_ns: np.ndarray  # (N,) int64, strictly increasing
    p: np.ndarray  # (N,3) metres
    q_xyzw: np.ndarray  # (N,4) unit, sign-continuous
    frame: str = "imu"  # sensor frame S of the poses: imu | main | ultrawide
    world: str = "z_up_gravity"

    def __post_init__(self):
        self.t_ns = np.asarray(self.t_ns, dtype=np.int64)
        self.p = np.asarray(self.p, dtype=np.float64).reshape(-1, 3)
        q = np.asarray(self.q_xyzw, dtype=np.float64).reshape(-1, 4)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        for i in range(1, len(q)):  # sign continuity
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        self.q_xyzw = q
        self.frame = rig_frame(self.frame)
        if not (len(self.t_ns) == len(self.p) == len(self.q_xyzw)):
            raise ValueError("trajectory arrays differ in length")
        if len(self.t_ns) > 1 and np.any(np.diff(self.t_ns) <= 0):
            raise ValueError("trajectory times must be strictly increasing")

    def __len__(self) -> int:
        return len(self.t_ns)

    @classmethod
    def from_T(cls, t_ns, T: np.ndarray, frame: str = "imu", world: str = "z_up_gravity") -> "Trajectory":
        T = np.asarray(T, dtype=np.float64)
        q = Rotation.from_matrix(T[:, :3, :3]).as_quat()
        return cls(np.asarray(t_ns, dtype=np.int64), T[:, :3, 3], q, frame, world)

    def T(self) -> np.ndarray:
        out = np.tile(np.eye(4), (len(self), 1, 1))
        out[:, :3, :3] = Rotation.from_quat(self.q_xyzw).as_matrix()
        out[:, :3, 3] = self.p
        return out

    def sample(self, t_ns, max_gap_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate poses at ``t_ns``: linear position, piecewise SLERP rotation.

        Returns (T (M,4,4), valid (M,)). ``valid[i]`` iff ``t`` lies inside the trajectory span and the bracketing key
        interval is <= ``max_gap_ns`` (exact key hits are always valid). Invalid rows hold the nearest key pose.
        """
        t = np.asarray(t_ns, dtype=np.int64)
        M = len(t)
        out = np.tile(np.eye(4), (M, 1, 1))
        valid = np.zeros(M, dtype=bool)
        if len(self) == 0 or M == 0:
            return out, valid
        t0 = int(self.t_ns[0])
        tk = (self.t_ns - t0).astype(np.float64) * 1e-9
        tq = (t - t0).astype(np.float64) * 1e-9
        inside = (t >= self.t_ns[0]) & (t <= self.t_ns[-1])
        hi = np.clip(np.searchsorted(self.t_ns, t, side="left"), 0, len(self) - 1)
        exact = self.t_ns[hi] == t
        lo = np.clip(hi - 1, 0, len(self) - 1)
        gap = self.t_ns[hi] - self.t_ns[lo]
        valid = inside & (exact | (gap <= max_gap_ns))
        tq_c = np.clip(tq, tk[0], tk[-1])
        for ax in range(3):
            out[:, ax, 3] = np.interp(tq_c, tk, self.p[:, ax])
        rots = Rotation.from_quat(self.q_xyzw)
        if len(self) == 1:
            out[:, :3, :3] = rots.as_matrix()[0]
        else:
            out[:, :3, :3] = Slerp(tk, rots)(tq_c).as_matrix()
        # exact key hits: copy the key (no interpolation round-off)
        if np.any(exact):
            out[exact, :3, :3] = rots[hi[exact]].as_matrix()
            out[exact, :3, 3] = self.p[hi[exact]]
        return out, valid


# ---------------------------------------------------------------- I/O

def _time_token_to_ns(tok: str, unit: str) -> int:
    d = Decimal(tok)
    if unit == "s":
        d = d * 1_000_000_000
    return int(d.to_integral_value())


def detect_time_unit(t: np.ndarray) -> str:
    """'ns' if the median magnitude is > 1e10 (rig uptimes are < 317 years), else 's'."""
    t = np.asarray(t, dtype=np.float64)
    return "ns" if np.median(np.abs(t)) > 1e10 else "s"


def _read_rows(path: Path) -> tuple[list[str], np.ndarray]:
    toks, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 8:
                raise ValueError(f"{path}: expected 't x y z qx qy qz qw', got {line!r}")
            toks.append(parts[0])
            vals.append([float(v) for v in parts[1:8]])
    return toks, np.asarray(vals, dtype=np.float64).reshape(-1, 7)


def load_tum(path, time_unit: str = "auto", frame: str = "imu", world: str = "z_up_gravity") -> Trajectory:
    """TUM ``t x y z qx qy qz qw`` ('#' comments, spaces or commas). ``time_unit``: auto | s | ns."""
    toks, vals = _read_rows(Path(path))
    if time_unit == "auto":
        time_unit = detect_time_unit(np.array([float(x) for x in toks])) if toks else "s"
    if time_unit not in ("s", "ns"):
        raise ValueError(f"time_unit must be auto|s|ns, got {time_unit!r}")
    t = np.array([_time_token_to_ns(x, time_unit) for x in toks], dtype=np.int64)
    # VIO outputs occasionally repeat a timestamp (keyframe + frame); keep the last pose for each time.
    if len(t) > 1:
        order = np.argsort(t, kind="stable")
        t, vals = t[order], vals[order]
        keep = np.append(np.diff(t) != 0, True)
        t, vals = t[keep], vals[keep]
    return Trajectory(t, vals[:, :3], vals[:, 3:7], frame, world)


def load_orbslam3_euroc(path, frame: str = "imu") -> Trajectory:
    """ORB-SLAM3 ``SaveTrajectoryEuRoC`` output: same column order, t in ns (printed as fixed-point), Twb = IMU pose."""
    return load_tum(path, time_unit="ns", frame=frame)


def save_tum(traj: Trajectory, path) -> None:
    """Seconds with 9 decimals (exact ns), positions/quaternions with 9 significant decimals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# t[s] x y z qx qy qz qw  frame={traj.frame} world={traj.world}"]
    for t, p, q in zip(traj.t_ns.tolist(), traj.p, traj.q_xyzw):
        s, ns = divmod(int(t), 1_000_000_000)
        lines.append(
            f"{s}.{ns:09d} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f}"
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- checks against the IMU

def _imu_rotation_in_S(rig, frame: str) -> np.ndarray:
    """R_S_imu: the IMU axes in sensor frame S (identity for S = imu; R_imu_cam^T for a camera)."""
    frame = rig_frame(frame)
    if frame == "imu":
        return np.eye(3)
    return np.asarray(rig.cameras[frame].T_imu_cam, dtype=np.float64)[:3, :3].T


def gravity_check(traj: Trajectory, imu_t, imu_acc, rig, max_angle_deg: float = 3.0, max_norm_err: float = 0.5) -> dict:
    """Mean accelerometer (specific force) rotated into the VIO world must point along +z with |g| ~ 9.81.

    ``imu_t`` must be on the IMU time base (``rig.imu_time_ns(raw)``). Over an episode the mean acceleration
    is ~0, so the mean specific force is +g along the world's up axis if the world really is z-up.
    """
    imu_t = np.asarray(imu_t, dtype=np.int64)
    sel = (imu_t >= traj.t_ns[0]) & (imu_t <= traj.t_ns[-1])
    if sel.sum() < 10:
        return {"ok": False, "reason": "too few IMU samples inside the trajectory span", "n": int(sel.sum())}
    T, valid = traj.sample(imu_t[sel], max_gap_ns=10**18)
    R_S_imu = _imu_rotation_in_S(rig, traj.frame)
    f_imu = np.asarray(imu_acc, dtype=np.float64)[sel]
    f_W = np.einsum("nij,jk,nk->ni", T[:, :3, :3], R_S_imu, f_imu)
    m = f_W.mean(axis=0)
    norm = float(np.linalg.norm(m))
    angle = float(np.degrees(np.arccos(np.clip(m[2] / max(norm, 1e-12), -1, 1))))
    ok = angle < max_angle_deg and abs(norm - GRAVITY) < max_norm_err
    return {"ok": bool(ok), "angle_to_up_deg": angle, "g_norm": norm, "mean_world": m.tolist(), "n": int(sel.sum())}


def _rotation_to_up(m: np.ndarray) -> np.ndarray:
    """Minimal rotation taking direction ``m`` to +z (no rotation about z, i.e. yaw untouched)."""
    a = np.asarray(m, dtype=np.float64) / np.linalg.norm(m)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(a, z)
    s, c = np.linalg.norm(axis), float(a @ z)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    return Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()


def gravity_align(traj: Trajectory, imu_t, imu_acc, rig) -> tuple[Trajectory, dict]:
    """Rotate the trajectory's world so that gravity is along -z (world up = +z).

    The up direction is the mean specific force over the trajectory span, ``mean_t R_W_imu(t) @ acc_imu(t)``
    (``imu_t`` on the IMU time base). Every pose (rotation and position) is pre-multiplied by the minimal rotation
    taking that direction to +z. Valid for any VIO whose world is merely rigid (e.g. ORB-SLAM3's first-keyframe
    frame). Returns the aligned trajectory and {angle_deg, axis, mean_world_before}.
    """
    g = gravity_check(traj, imu_t, imu_acc, rig, max_angle_deg=180.0, max_norm_err=np.inf)
    if "mean_world" not in g:
        raise ValueError(f"cannot gravity-align: {g.get('reason')}")
    R = _rotation_to_up(np.asarray(g["mean_world"]))
    T = traj.T()
    T[:, :3, :] = np.einsum("ij,njk->nik", R, T[:, :3, :])
    rv = Rotation.from_matrix(R).as_rotvec()
    info = {"applied": True, "angle_deg": float(np.degrees(np.linalg.norm(rv))), "rotvec": rv.tolist(),
            "mean_world_before": g["mean_world"]}
    return Trajectory.from_T(traj.t_ns, T, frame=traj.frame, world=traj.world), info


def _box_smooth(t_ns: np.ndarray, x: np.ndarray, width_ns: int) -> np.ndarray:
    """Centred moving average over ``width_ns`` (cumulative-sum implementation)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.searchsorted(t_ns, t_ns - width_ns // 2, side="left")
    hi = np.searchsorted(t_ns, t_ns + width_ns // 2, side="right")
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def time_offset_check(traj: Trajectory, imu_t, imu_gyro, rig=None, rate_hz: float = 100.0,
                      max_lag_ms: float = 100.0, step_ms: float = 0.25) -> dict:
    """Estimate the delay of the trajectory clock relative to the IMU from |omega| cross-correlation.

    ``traj_delay_ms > 0`` means trajectory stamps are later than the true motion seen by the gyro. |omega| is
    frame-independent, so the extrinsic rotation is not needed (``rig`` is accepted for API symmetry).
    Warns if |delay| > 5 ms or the peak correlation < 0.8.
    """
    imu_t = np.asarray(imu_t, dtype=np.int64)
    dt_ns = int(round(1e9 / rate_hz))
    margin = int(max_lag_ms * 1e6) + dt_ns
    tg = np.arange(traj.t_ns[0] + margin, traj.t_ns[-1] - margin, dt_ns, dtype=np.int64)
    if len(tg) < 20 or len(imu_t) < 20:
        return {"ok": False, "reason": "trajectory too short", "traj_delay_ms": None, "corr": None}
    T, _ = traj.sample(tg, max_gap_ns=10**18)
    R = Rotation.from_matrix(T[:, :3, :3])
    w_traj = (R[:-1].inv() * R[1:]).magnitude() / (dt_ns * 1e-9)
    t_mid = (tg[:-1] + tg[1:]) // 2
    g_mag = np.linalg.norm(np.asarray(imu_gyro, dtype=np.float64), axis=1)
    g_s = _box_smooth(imu_t, g_mag, dt_ns)
    rel_imu = (imu_t - imu_t[0]).astype(np.float64)
    lags = np.arange(-max_lag_ms, max_lag_ms + 1e-9, step_ms)
    a = w_traj - w_traj.mean()
    corr = np.empty(len(lags))
    for k, lag in enumerate(lags):
        g = np.interp((t_mid - imu_t[0]).astype(np.float64) + lag * 1e6, rel_imu, g_s)
        b = g - g.mean()
        corr[k] = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    k = int(np.argmax(corr))
    lag = lags[k]
    if 0 < k < len(lags) - 1:  # parabolic refinement
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lag += 0.5 * (y0 - y2) / den * step_ms
    delay = float(-lag)
    peak = float(corr[k])
    ok = abs(delay) <= 5.0 and peak >= 0.8
    return {"ok": bool(ok), "traj_delay_ms": delay, "corr": peak}


# ---------------------------------------------------------------- F5: VIO consistency (jumps / resets)

MOTION_LIMITS = {"gyro_resid_rms": 0.15, "gyro_resid_max": 1.5, "step_speed_max": 5.0}


def motion_consistency_check(traj: Trajectory, imu_t, imu_gyro, rig, max_interval_s: float = 0.2,
                             limits: dict | None = None) -> dict:
    """Gyro vs VIO angular rate, and per-step translation jumps (VIO resets / relocalisation jumps).

    For every pair of consecutive poses (gap <= ``max_interval_s``): the VIO body rate
    ``rotvec(R_k^T R_k+1) / dt``, rotated into the IMU frame, vs the mean gyro over the same interval. A constant
    gyro bias (median residual) is removed first. Reports the residual RMS / max (rad/s) and the largest per-step
    translation and speed. ``ok`` is False above ``limits`` (default ``MOTION_LIMITS``).
    """
    lim = {**MOTION_LIMITS, **(limits or {})}
    imu_t = np.asarray(imu_t, dtype=np.int64)
    g = np.asarray(imu_gyro, dtype=np.float64)
    t = traj.t_ns
    T = traj.T()
    dt = np.diff(t) / 1e9
    use = (dt > 0) & (dt <= max_interval_s)
    out: dict = {"limits": lim}
    step = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
    if use.any():
        speed = step[use] / dt[use]
        k = int(np.argmax(speed))
        out.update(step_max_m=float(step[use].max()), step_speed_max=float(speed[k]),
                   step_speed_at_s=float((t[np.flatnonzero(use)[k]] - t[0]) / 1e9))
    else:
        out.update(step_max_m=None, step_speed_max=None)
    resid_rms = resid_max = None
    if len(imu_t) >= 2 and use.any():
        R = Rotation.from_matrix(T[:, :3, :3])
        w_S = (R[:-1].inv() * R[1:]).as_rotvec() / np.maximum(dt, 1e-9)[:, None]
        R_S_imu = _imu_rotation_in_S(rig, traj.frame)
        w_imu = w_S @ R_S_imu                     # = (R_S_imu^T w_S^T)^T
        c = np.vstack([np.zeros((1, 3)), np.cumsum(g, axis=0)])
        lo = np.searchsorted(imu_t, t[:-1], side="left")
        hi = np.searchsorted(imu_t, t[1:], side="right")
        cnt = hi - lo
        ok_iv = use & (cnt >= 1)
        if ok_iv.any():
            g_mean = (c[hi[ok_iv]] - c[lo[ok_iv]]) / cnt[ok_iv][:, None]
            r = g_mean - w_imu[ok_iv]
            bias = np.median(r, axis=0)
            rn = np.linalg.norm(r - bias, axis=1)
            resid_rms, resid_max = float(np.sqrt(np.mean(rn ** 2))), float(rn.max())
            out.update(gyro_bias_est=bias.tolist(), gyro_resid_at_s=float(
                (t[:-1][ok_iv][int(np.argmax(rn))] - t[0]) / 1e9), n_intervals=int(ok_iv.sum()))
    out.update(gyro_resid_rms=resid_rms, gyro_resid_max=resid_max)
    problems = []
    if resid_rms is not None and resid_rms > lim["gyro_resid_rms"]:
        problems.append(f"gyro vs VIO rate residual RMS {resid_rms:.3f} rad/s > {lim['gyro_resid_rms']}")
    if resid_max is not None and resid_max > lim["gyro_resid_max"]:
        problems.append(f"gyro vs VIO rate residual max {resid_max:.2f} rad/s > {lim['gyro_resid_max']} at "
                        f"{out['gyro_resid_at_s']:.2f} s (VIO jump/reset?)")
    if out["step_speed_max"] is not None and out["step_speed_max"] > lim["step_speed_max"]:
        problems.append(f"translation jump: {out['step_max_m']:.3f} m step, {out['step_speed_max']:.1f} m/s > "
                        f"{lim['step_speed_max']} m/s at {out['step_speed_at_s']:.2f} s")
    if resid_rms is None:
        out["note"] = "no IMU samples inside pose intervals: gyro residual not computed"
    out["ok"] = not problems
    out["problems"] = problems
    return out


# ---------------------------------------------------------------- dataproc: metric scale vs the accelerometer

#: ``scale_mismatch`` when ``|s - 1|`` exceeds this.
SCALE_TOLERANCE = 0.1
#: ``scale_unobservable`` below this RMS (m/s^2) of the smoothed trajectory acceleration (mean removed).
SCALE_MIN_EXCITATION = 0.05
#: ``scale_unobservable`` when the (correlation-corrected) relative standard error ``std_err / s`` exceeds this.
SCALE_MAX_STD_ERR = 0.05
#: ``scale_unobservable`` when position jitter would shrink the uncorrected estimate below this factor (the
#: errors-in-variables correction is then larger than the signal it corrects; see :func:`scale_check`).
SCALE_MIN_ATTENUATION = 0.5


def position_jitter(traj: Trajectory, rel_tol: float = 0.1) -> dict:
    """Per-axis std (m) of the white, frame-to-frame position noise of ``traj`` ("VIO jitter").

    Estimated from 4th differences of the positions at the pose times: for white noise of std ``sigma``,
    ``var(d4) = 70 sigma^2``, while a smooth hand motion contributes (4th derivative) * dt^4, ~1e-4 mm at 30 Hz.
    Only windows of 5 consecutive poses spaced within ``rel_tol`` of the median interval count (gaps and dropped
    frames break the uniform-grid assumption). Robust and efficient: per axis, the RMS of the 4th differences within
    4 sigma of the MAD estimate (``1.4826 * MAD / sqrt(70)``), so isolated jumps do not inflate it; ~5 % std on
    450 poses (the MAD alone: ~7 %). Returns ``{sigma_m: [sx, sy, sz] | None, dt_s, n}``.
    """
    t = np.asarray(traj.t_ns, dtype=np.int64)
    if len(t) < 6:
        return {"sigma_m": None, "dt_s": None, "n": 0}
    dt = np.diff(t).astype(np.float64)
    med = float(np.median(dt))
    ok_dt = np.abs(dt - med) <= rel_tol * med
    win = np.convolve(ok_dt.astype(np.int64), np.ones(4, dtype=np.int64), mode="valid") == 4  # (N-4,)
    d4 = np.diff(traj.p, n=4, axis=0)[win]
    if len(d4) < 10:
        return {"sigma_m": None, "dt_s": med * 1e-9, "n": int(len(d4))}
    mad = np.median(np.abs(d4 - np.median(d4, axis=0)), axis=0)
    cut = 4.0 * 1.4826 * mad
    sigma = np.array([np.sqrt(np.mean(np.square(d[np.abs(d) <= c]))) if c > 0 else 0.0
                      for d, c in zip(d4.T, cut)]) / np.sqrt(70.0)
    return {"sigma_m": sigma.tolist(), "dt_s": med * 1e-9, "n": int(len(d4))}


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index ranges of consecutive True values."""
    v = np.concatenate([[False], np.asarray(valid, bool), [False]])
    d = np.diff(v.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def scale_check(traj: Trajectory, imu_t, imu_acc, rig, rate_hz: float = 100.0, smooth_s: float = 0.15,
                max_gap_ns: int = 200_000_000, tolerance: float = SCALE_TOLERANCE,
                min_excitation: float = SCALE_MIN_EXCITATION, max_std_err: float = SCALE_MAX_STD_ERR,
                correct_jitter: bool = True, min_attenuation: float = SCALE_MIN_ATTENUATION) -> dict:
    """Metric scale of the trajectory against the accelerometer (dataproc addition, not in sensor_rig).

    Fits ``R_W_imu(t) @ f_imu(t) + g_W = s * p''(t) + R_W_imu(t) @ b_imu + c_W`` by least squares: ``b_imu`` is a
    constant accelerometer bias (body frame), ``c_W`` a constant world offset (residual gravity tilt). ``s`` is the
    factor the trajectory positions would need (``s = 2``: the trajectory is half the metric size). Both sides are resampled at ``rate_hz`` on the
    tracked span, split at trajectory gaps > ``max_gap_ns``, and filtered with the same Gaussian (std ``smooth_s``):
    ``p''`` as the Gaussian second derivative of the positions, the IMU side as the Gaussian smooth, so the
    comparison is band-matched and position noise is suppressed. ``imu_t`` on the IMU time base; the trajectory
    world must be z-up (run after gravity alignment).

    **Position jitter (errors in variables).** Noise on the positions is noise on the regressor ``p''``, which
    biases ordinary least squares towards 0 by the attenuation factor ``var(p''_true) / var(p''_measured)`` (5 mm
    white jitter at 30 Hz on a 15 s demo: s ~ 0.90). With ``correct_jitter`` (default) the jitter std per axis is
    estimated by :func:`position_jitter` and its known contribution to ``p''`` is removed from the normal equations
    (corrected least squares): for white noise of std ``sigma`` at pose interval ``dt``, the Gaussian
    second-derivative filter of std ``smooth_s`` adds ``sigma^2 dt * 3 / (8 sqrt(pi) smooth_s^5)`` per axis to the
    variance of ``p''`` (exact to 2 % at 30 Hz after the 100 Hz linear resampling). The correction is invariant to
    a scale error of the trajectory (jitter and signal scale together), so a wrong scale is still detected.
    Position error that is not white (drift, noise correlated over the ~0.5-3 Hz band the filter passes) is not
    corrected: the high-frequency estimate then under-corrects. Also returned: ``scale_uncorrected`` (plain least
    squares), ``attenuation`` (the estimated shrink factor of that fit) and ``jitter``; below ``min_attenuation``
    the scale is ``scale_unobservable`` (jitter dominates the in-band motion).

    Returns ``{ok, status, scale, std_err, excitation_rms, resid_rms, r2, n, problems}`` with ``status`` one of
    ``ok`` | ``scale_mismatch`` (|s - 1| > tolerance) | ``scale_unobservable`` (too short, too little motion or too
    uncertain), plus ``accel_bias_imu`` (the fitted body-frame bias) once fitted.
    """
    from scipy.ndimage import gaussian_filter1d

    out: dict = {"scale": None, "std_err": None, "excitation_rms": None, "resid_rms": None, "r2": None, "n": 0,
                 "tolerance": tolerance, "rate_hz": rate_hz, "smooth_s": smooth_s, "scale_uncorrected": None,
                 "attenuation": None, "jitter": None}
    imu_t = np.asarray(imu_t, dtype=np.int64)
    acc = np.asarray(imu_acc, dtype=np.float64)

    def unobservable(reason):
        out.update(ok=False, status="scale_unobservable", problems=[f"scale_unobservable: {reason}"])
        return out

    if len(traj) < 2 or len(imu_t) < 2:
        return unobservable("too few poses or IMU samples")
    dt_ns = int(round(1e9 / rate_hz))
    lo, hi = max(int(traj.t_ns[0]), int(imu_t[0])), min(int(traj.t_ns[-1]), int(imu_t[-1]))
    if hi - lo < 2 * dt_ns:
        return unobservable("trajectory and IMU barely overlap")
    tg = np.arange(lo, hi + 1, dt_ns, dtype=np.int64)
    T, valid = traj.sample(tg, max_gap_ns=max_gap_ns)
    sig = smooth_s * rate_hz                       # Gaussian std in samples
    edge = int(np.ceil(3 * sig))
    R_S_imu = _imu_rotation_in_S(rig, traj.frame)
    rel = (imu_t - imu_t[0]).astype(np.float64)
    a_vio, a_imu, r_imu = [], [], []
    for a, b in _segments(valid):
        if b - a < 2 * edge + 10:
            continue
        seg = slice(a, b)
        p = T[seg, :3, 3]
        acc_p = gaussian_filter1d(p, sig, axis=0, order=2, mode="nearest") * (rate_hz ** 2)
        tq = (tg[seg] - imu_t[0]).astype(np.float64)
        f = np.stack([np.interp(tq, rel, acc[:, k]) for k in range(3)], axis=1)
        R_W_imu = T[seg, :3, :3] @ R_S_imu
        f_W = np.einsum("nij,nj->ni", R_W_imu, f) - [0.0, 0.0, GRAVITY]
        f_W = gaussian_filter1d(f_W, sig, axis=0, order=0, mode="nearest")
        R_s = gaussian_filter1d(R_W_imu.reshape(-1, 9), sig, axis=0, order=0, mode="nearest").reshape(-1, 3, 3)
        a_vio.append(acc_p[edge:-edge])
        a_imu.append(f_W[edge:-edge])
        r_imu.append(R_s[edge:-edge])
    if not a_vio:
        return unobservable(f"no tracked stretch longer than {(2 * edge + 10) / rate_hz:.2f} s")
    x = np.concatenate(a_vio)                      # (n,3) trajectory acceleration
    y = np.concatenate(a_imu)                      # (n,3) accelerometer acceleration in the world
    Rs = np.concatenate(r_imu)                     # (n,3,3) smoothed R_W_imu (multiplies the body-frame bias)
    n = len(x)
    xc = x - x.mean(axis=0)
    excitation = float(np.sqrt(np.mean(np.sum(xc ** 2, axis=1))))
    out.update(n=int(n), excitation_rms=excitation)
    if excitation < min_excitation:
        return unobservable(f"excitation {excitation:.3f} m/s^2 rms < {min_excitation}")
    # y = s * x + R_W_imu @ b_imu + c_W : unknowns [s, b_imu (3), c_W (3)]
    A = np.zeros((n, 3, 7))
    A[:, :, 0] = x
    A[:, :, 1:4] = Rs
    A[:, :, 4:7] = np.eye(3)
    A = A.reshape(-1, 7)
    yv = y.reshape(-1)
    AtA, Aty = A.T @ A, A.T @ yv
    try:
        s_ols = float(np.linalg.solve(AtA, Aty)[0])
    except np.linalg.LinAlgError:
        return unobservable("degenerate fit")
    out["scale_uncorrected"] = s_ols
    if correct_jitter:
        jit = position_jitter(traj)
        out["jitter"] = jit
        if jit["sigma_m"] is not None:
            # variance that white jitter adds to each axis of p'' (Gaussian 2nd-derivative filter, see docstring)
            var_ex = np.square(jit["sigma_m"]) * jit["dt_s"] * 3.0 / (8.0 * np.sqrt(np.pi) * smooth_s ** 5)
            noise00 = float(n * var_ex.sum())
            attenuation = 1.0 - noise00 / max(float(np.sum(xc ** 2)), 1e-300)
            out["attenuation"] = attenuation
            if attenuation < min_attenuation:
                return unobservable(f"position jitter {1e3 * float(np.linalg.norm(jit['sigma_m'])):.1f} mm dominates "
                                    f"the in-band motion (attenuation {attenuation:.2f} < {min_attenuation})")
            AtA = AtA.copy()
            AtA[0, 0] -= noise00
    try:
        sol = np.linalg.solve(AtA, Aty)
        cov00 = float(np.linalg.inv(AtA)[0, 0])
    except np.linalg.LinAlgError:
        return unobservable("degenerate fit")
    s = float(sol[0])
    resid = yv - A @ sol
    resid_rms = float(np.sqrt(np.mean(np.sum(resid.reshape(-1, 3) ** 2, axis=1))))
    corr_len = max(1.0, 2.5 * sig)                # smoothing correlates ~2.5 sigma of neighbouring samples
    sigma2 = float(resid @ resid) / max(1.0, len(yv) - 7)
    std_err = float(np.sqrt(sigma2 * cov00 * corr_len))
    yc = yv - np.tile(y.mean(axis=0), n)
    r2 = float(1.0 - (resid @ resid) / max(float(yc @ yc), 1e-12))
    out["accel_bias_imu"] = sol[1:4].tolist()
    out.update(scale=s, std_err=std_err, resid_rms=resid_rms, r2=r2)
    if std_err > max_std_err * abs(s):
        return unobservable(f"scale {s:.3f} too uncertain (relative std err {std_err / max(abs(s), 1e-12):.3f} > "
                            f"{max_std_err})")
    if abs(s - 1.0) > tolerance:
        out.update(ok=False, status="scale_mismatch",
                   problems=[f"scale_mismatch: trajectory scale {s:.3f} vs the accelerometer (|s-1| > {tolerance})"])
        return out
    out.update(ok=True, status="ok", problems=[])
    return out
