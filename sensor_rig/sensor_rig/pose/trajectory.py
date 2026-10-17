"""Trajectories (TUM / ORB-SLAM3 EuRoC), interpolation at grid times, and sanity checks against the IMU.

Times are int64 ns on the *corrected* rig clock (raw CLOCK_MONOTONIC stamp + rig.yaml offset), the same clock export
uses. Trajectory files are parsed with exact decimal arithmetic so a ``t`` written with 9 decimals round-trips to the
same integer ns (float64 cannot hold ~1e15 ns exactly).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sensor_rig.pose.frames import rig_frame

GRAVITY = 9.80665


@dataclass
class Trajectory:
    t_ns: np.ndarray  # (N,) int64, strictly increasing
    p: np.ndarray  # (N,3) metres
    q_xyzw: np.ndarray  # (N,4) unit, sign-continuous
    frame: str = "imu"  # sensor frame S of the poses (rig.T name: imu | main | ultrawide)
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
    return np.asarray(rig.T(rig_frame(frame), "imu"), dtype=np.float64)[:3, :3]


def gravity_check(traj: Trajectory, imu_t, imu_acc, rig, max_angle_deg: float = 3.0, max_norm_err: float = 0.5) -> dict:
    """Mean accelerometer (specific force) rotated into the VIO world must point along +z with |g| ~ 9.81.

    ``imu_t`` must be on the corrected clock (raw + rig.imu.time_offset_ns). Over an episode the mean acceleration
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
    (``imu_t`` on the corrected clock). Every pose (rotation and position) is pre-multiplied by the minimal rotation
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
