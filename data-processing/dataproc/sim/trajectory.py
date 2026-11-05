"""Smooth, seeded, hand-like trajectories of the IMU body in the gravity-aligned z-up world (docs/SIM.md).

Every scalar channel (x, y, z, three angles, gripper width) is a sum of terms with **analytic** first and second
derivatives:

* :class:`MinJerk` - piecewise quintic min-jerk between waypoints (C2: velocity and acceleration are zero at the
  knots, like a hand that pauses between reach, grasp and lift);
* :class:`WindowedSines` - a sum of sines multiplied by a C2 window (quintic smoothstep ramps), used for the pre-roll
  excitation and for a slow, small hand wobble during the whole episode.

Orientation: ``R_W_imu(t) = Rz(a(t)) @ Ry(b(t)) @ Rx(c(t)) @ R0`` (rotations about the **world** axes applied to a
constant base orientation ``R0``). The body angular rate is exact:
``omega_W = a' e_z + Rz(a) b' e_y + Rz(a) Ry(b) c' e_x`` and ``omega_imu = R_W_imu^T omega_W``.

The IMU specific force is ``f_imu = R_W_imu^T (a_W - g_W)`` with ``g_W = (0, 0, -9.80665)`` (as in
``dataproc.sim.imu_synth``), so a static IMU reads ``+9.81`` on its world-up axis.

Time ``t`` is in seconds since the start of the recording (the first IMU sample).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dataproc.sim.imu_synth import G_WORLD

__all__ = ["MinJerk", "WindowedSines", "Channel", "Trajectory", "make_trajectory", "camera_base_rotation",
           "WIDTH_CLOSED_M", "WIDTH_OPEN_M", "CALIB_WIDTH_RANGE_M", "POSTROLL_S", "STATIC_START_S"]

#: Nominal tag-centre separations (virtual ultrawide x, metres): demonstration closed / open (randomised by
#: +-2 mm / +-5 mm per episode) and the full sweep of a calibration episode.
WIDTH_CLOSED_M = 0.034
WIDTH_OPEN_M = 0.078
CALIB_WIDTH_RANGE_M = (0.030, 0.090)
#: Recording after the stop press, and the motionless start (both seconds).
POSTROLL_S = 0.5
STATIC_START_S = 0.3
#: Target peak speed of the demonstration segments (m/s).
V_PEAK = 0.35


def _smoothstep(x):
    """Quintic smoothstep S(x) on [0, 1] (clipped), with S' and S'' w.r.t. x."""
    x = np.clip(x, 0.0, 1.0)
    s = x**3 * (10 - 15 * x + 6 * x * x)
    ds = 30 * x * x * (1 - x) ** 2
    dds = 60 * x * (1 - x) * (1 - 2 * x)
    return s, ds, dds


@dataclass(frozen=True)
class MinJerk:
    """Min-jerk interpolation through ``values`` at strictly increasing ``knots`` (constant outside)."""

    knots: tuple[float, ...]
    values: tuple[float, ...]

    def __call__(self, t):
        k = np.asarray(self.knots, float)
        v = np.asarray(self.values, float)
        t = np.asarray(t, float)
        if len(k) == 1:
            return np.full_like(t, v[0]), np.zeros_like(t), np.zeros_like(t)
        i = np.clip(np.searchsorted(k, t, side="right") - 1, 0, len(k) - 2)
        T = k[i + 1] - k[i]
        s, ds, dds = _smoothstep((t - k[i]) / T)
        dv = v[i + 1] - v[i]
        return v[i] + dv * s, dv * ds / T, dv * dds / T**2


@dataclass(frozen=True)
class WindowedSines:
    """``w(t) * sum_i A_i sin(2 pi f_i t + phi_i)``; ``w`` ramps 0->1 over ``[t0, t1]`` and 1->0 over ``[t2, t3]``
    (``t2 = inf``: no ramp down)."""

    amps: tuple[float, ...]
    freqs_hz: tuple[float, ...]
    phases: tuple[float, ...]
    t0: float
    t1: float
    t2: float = float("inf")
    t3: float = float("inf")

    def __call__(self, t):
        t = np.asarray(t, float)
        A = np.asarray(self.amps, float)[:, None]
        w = 2 * np.pi * np.asarray(self.freqs_hz, float)[:, None]
        ph = np.asarray(self.phases, float)[:, None]
        arg = w * t[None] + ph
        S = (A * np.sin(arg)).sum(0)
        dS = (A * w * np.cos(arg)).sum(0)
        ddS = (-A * w * w * np.sin(arg)).sum(0)
        u, du, ddu = _smoothstep((t - self.t0) / (self.t1 - self.t0))
        du, ddu = du / (self.t1 - self.t0), ddu / (self.t1 - self.t0) ** 2
        if np.isfinite(self.t2):
            d, dd_, ddd = _smoothstep((t - self.t2) / (self.t3 - self.t2))
            L = self.t3 - self.t2
            v, dv, ddv = 1 - d, -dd_ / L, -ddd / L**2
        else:
            v, dv, ddv = np.ones_like(t), np.zeros_like(t), np.zeros_like(t)
        W, dW, ddW = u * v, du * v + u * dv, ddu * v + 2 * du * dv + u * ddv
        return W * S, dW * S + W * dS, ddW * S + 2 * dW * dS + W * ddS


@dataclass(frozen=True)
class Channel:
    """Sum of terms (each callable ``t -> (f, f', f'')``)."""

    terms: tuple = ()

    def __call__(self, t):
        t = np.asarray(t, float)
        f, df, ddf = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
        for term in self.terms:
            a, b, c = term(t)
            f, df, ddf = f + a, df + b, ddf + c
        return f, df, ddf


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _Ry(b):
    c, s = np.cos(b), np.sin(b)
    o, z = np.ones_like(b), np.zeros_like(b)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _Rx(c_):
    c, s = np.cos(c_), np.sin(c_)
    o, z = np.ones_like(c_), np.zeros_like(c_)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


@dataclass(frozen=True, eq=False)
class Trajectory:
    """IMU body trajectory ``T_W_imu(t)`` plus the gripper (tag-separation) width schedule ``width_m(t)``."""

    pos: tuple[Channel, Channel, Channel]
    ang: tuple[Channel, Channel, Channel]       # a (about world z), b (world y), c (world x)
    R0: np.ndarray                             # base R_W_imu
    width: Channel
    duration_s: float
    button_start_s: float
    button_stop_s: float
    kind: str
    phases: tuple = field(default=())          # (name, t_start_s, t_end_s) of the demonstration segments

    def kinematics(self, t) -> dict:
        """``p`` (N,3), ``v``, ``a`` (world, m, m/s, m/s^2), ``R`` (N,3,3) = R_W_imu, ``omega`` (N,3) body rate
        in the IMU frame (rad/s)."""
        t = np.atleast_1d(np.asarray(t, float))
        P = [c(t) for c in self.pos]
        p = np.stack([x[0] for x in P], 1)
        v = np.stack([x[1] for x in P], 1)
        a = np.stack([x[2] for x in P], 1)
        (ya, dya, _), (pb, dpb, _), (rc, drc, _) = (c(t) for c in self.ang)
        Rz, Ry, Rx = _Rz(ya), _Ry(pb), _Rx(rc)
        Q = Rz @ Ry @ Rx
        R = Q @ self.R0
        n = len(t)
        ez = np.tile([0.0, 0.0, 1.0], (n, 1))
        ey = np.tile([0.0, 1.0, 0.0], (n, 1))
        ex = np.tile([1.0, 0.0, 0.0], (n, 1))
        omega_W = (dya[:, None] * ez + np.einsum("nij,nj->ni", Rz, dpb[:, None] * ey)
                   + np.einsum("nij,nj->ni", Rz @ Ry, drc[:, None] * ex))
        omega = np.einsum("nji,nj->ni", R, omega_W)
        return {"p": p, "v": v, "a": a, "R": R, "omega": omega}

    def T_W_imu(self, t) -> np.ndarray:
        k = self.kinematics(t)
        T = np.tile(np.eye(4), (len(k["p"]), 1, 1))
        T[:, :3, :3] = k["R"]
        T[:, :3, 3] = k["p"]
        return T

    def imu(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Exact noise-free ``(gyro, accel)`` (N,3) in the IMU frame at times ``t`` (s)."""
        k = self.kinematics(t)
        return k["omega"], np.einsum("nji,nj->ni", k["R"], k["a"] - G_WORLD)

    def width_m(self, t) -> np.ndarray:
        return self.width(np.atleast_1d(np.asarray(t, float)))[0]


def camera_base_rotation(pitch_down_rad: float) -> np.ndarray:
    """R_W_cam of an optical camera looking along world +x, pitched down by ``pitch_down_rad`` (x_cam = -y_W)."""
    c, s = np.cos(pitch_down_rad), np.sin(pitch_down_rad)
    z = np.array([c, 0.0, -s])
    x = np.array([0.0, -1.0, 0.0])
    return np.stack([x, np.cross(z, x), z], 1)


def _sines(rng, n, amp, fmin, fmax, t0, t1, t2=float("inf"), t3=float("inf")) -> WindowedSines:
    return WindowedSines(tuple(rng.uniform(0.6, 1.0, n) * amp / np.sqrt(n)), tuple(rng.uniform(fmin, fmax, n)),
                         tuple(rng.uniform(0, 2 * np.pi, n)), t0, t1, t2, t3)


def make_trajectory(rig, kind: str = "demonstration", seed: int = 0, duration_s: float = 15.0,
                    preroll_s: float = 2.5, postroll_s: float = POSTROLL_S) -> Trajectory:
    """A seeded, deterministic episode trajectory (``kind``: ``demonstration`` | ``calibration``).

    Timeline (seconds from the first sample): motionless for :data:`STATIC_START_S`; pre-roll excitation (+-3 to 5 cm,
    +-5 to 8 deg, 0.5-1.1 Hz, all six axes) until ``preroll_s`` (= the start press); the demonstration (or the
    calibration open/close sweeps) until ``duration_s - postroll_s`` (the stop press); then a short post-roll. A slow
    +-1 cm / +-1.5 deg wobble runs from the end of the static start to the end.
    """
    if kind not in ("demonstration", "calibration"):
        raise ValueError(f"kind must be 'demonstration' or 'calibration', got {kind!r}")
    if preroll_s < 1.0:
        raise ValueError(f"preroll_s must be >= 1.0 s, got {preroll_s}")
    if duration_s < preroll_s + postroll_s + 1.0:
        raise ValueError(f"duration_s must be >= preroll_s + postroll_s + 1 s, got {duration_s}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5153]))
    b0, b1 = float(preroll_s), float(duration_s - postroll_s)

    # base pose: ultrawide looking at the table along +x, pitched down ~30 deg
    pitch0 = np.deg2rad(30.0 + rng.uniform(-4, 4))
    R_W_uw0 = camera_base_rotation(pitch0)
    R_imu_uw = rig.cameras["ultrawide"].T_imu_cam[:3, :3]
    R0 = R_W_uw0 @ R_imu_uw.T
    p0 = np.array([-0.20, 0.0, 1.15]) + rng.uniform(-0.04, 0.04, 3)

    # pre-roll excitation and wobble (per channel: translation x y z, angle a b c)
    s0 = STATIC_START_S
    e_end = b0 - 0.05
    ramp = min(0.4, (e_end - s0) / 3)
    exc_amp = (0.05, 0.05, 0.04, 0.14, 0.12, 0.10)
    wob_amp = (0.010, 0.010, 0.008, 0.025, 0.020, 0.015)
    terms = [[_sines(rng, 2, exc_amp[i], 0.5, 1.1, s0, s0 + ramp, e_end - ramp, e_end),
              _sines(rng, 3, wob_amp[i], 0.15, 0.5, s0, s0 + 0.6)] for i in range(6)]

    # demonstration / calibration waypoints: (name, duration_s, [dx, dy, dz, da, db, dc], width) cumulative deltas
    closed = WIDTH_CLOSED_M + rng.uniform(-0.002, 0.002)
    opened = WIDTH_OPEN_M + rng.uniform(-0.005, 0.005)
    phases = []
    start = b0 + 0.1
    avail = b1 - 0.1 - start
    if kind == "demonstration":
        reach = np.array([0.50 + rng.uniform(-0.05, 0.05), rng.uniform(-0.08, 0.08), -0.17 + rng.uniform(-0.02, 0.02),
                          rng.uniform(-0.15, 0.15), np.deg2rad(22.0), rng.uniform(-0.1, 0.1)])
        lift = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.16 + rng.uniform(-0.02, 0.03),
                         0.0, np.deg2rad(-8.0), 0.0])
        move = np.array([-0.10 + rng.uniform(-0.04, 0.04), rng.choice([-1, 1]) * (0.18 + rng.uniform(0, 0.06)), 0.03,
                         rng.uniform(-0.2, 0.2), np.deg2rad(-6.0), rng.uniform(-0.1, 0.1)])
        segs = [("reach", reach), ("grasp", None), ("lift", lift), ("hold", None), ("move", move), ("settle", None)]
        hold_s = {"grasp": 0.7, "hold": 0.5, "settle": 0.4}
        dur = [max(1.0, 1.875 * np.linalg.norm(d[:3]) / V_PEAK) if d is not None else hold_s[n] for n, d in segs]
        factor = avail / sum(dur)
        scale_disp = min(1.0, factor)                      # short episodes: smaller motions at the same speed
        dur = [d * min(factor, 1.4) for d in dur]
        knots, vals = [0.0, start], [np.zeros(6), np.zeros(6)]
        wk, wv = [0.0, start], [closed, closed]
        t = start
        for (name, d), T in zip(segs, dur):
            t_end = t + T
            vals.append(vals[-1] + (0 if d is None else d * scale_disp))
            knots.append(t_end)
            phases.append((name, t, t_end))
            if name == "reach":                            # open during the first 60 % of the reach
                wk.append(t + 0.6 * T)
                wv.append(opened)
            elif name == "grasp":                          # close while pausing at the block
                wk += [t + 0.1 * T, t + 0.9 * T]
                wv += [opened, closed]
            t = t_end
        wk.append(duration_s + 1.0)
        wv.append(wv[-1])
    else:
        lo, hi = CALIB_WIDTH_RANGE_M
        knots, vals = [0.0, start, start + avail], [np.zeros(6)] * 3
        n_half = max(2, int(avail // 1.0) // 2 * 2)        # even number of half-cycles: ends closed
        hc = avail / n_half
        wk, wv = [0.0, start], [lo, lo]
        for k in range(n_half):
            wk.append(start + (k + 1) * hc)
            wv.append(hi if k % 2 == 0 else lo)
        wk.append(duration_s + 1.0)
        wv.append(lo)
        phases.append(("sweep", start, start + avail))
    vals = np.asarray(vals)
    for i in range(6):
        terms[i].append(MinJerk(tuple(knots), tuple(vals[:, i] + (p0[i] if i < 3 else 0.0))))
    chans = [Channel(tuple(ts)) for ts in terms]
    width = Channel((MinJerk(tuple(wk), tuple(wv)),))
    return Trajectory(pos=tuple(chans[:3]), ang=tuple(chans[3:]), R0=R0, width=width, duration_s=float(duration_s),
                      button_start_s=b0, button_stop_s=b1, kind=kind, phases=tuple(phases))
