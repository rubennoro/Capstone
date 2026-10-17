"""Target safety filter of the Franka bridge (pure, unit-tested). Applied to EVERY set_target, in this order:

1. reject non-finite targets (the caller keeps holding the previous safe target);
2. clamp the position to the workspace box (base frame);
3. clamp the orientation to a cone: the angle between the EE z-axis and `cone_axis` is at most `cone_max_deg`
   (rotated back onto the cone boundary about the shortest axis);
4. clamp the step relative to the previous safe target with a LEAKY BUCKET (independent of the request rate):
   the allowed displacement accrues as v_max * elapsed, capped at v_max * slack_s (10 ms of slack); each step spends
   from it (same for rotation with w_max). Over any window T the motion is <= v_max * (T + slack_s), whether targets
   arrive at 200 Hz, 1 kHz or in bursts, while reply jitter of a few ms is absorbed by the slack. Clamp, never
   reject, so an over-speed chunk is executed slower instead of freezing the arm.
Poses are 4x4 homogeneous matrices of the frame the backend controls (O_T_EE).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class SafetyConfig:
    workspace_min: tuple = (0.25, -0.40, 0.05)
    workspace_max: tuple = (0.80, 0.40, 0.70)
    cone_axis: tuple = (0.0, 0.0, -1.0)     # EE z pointing down (approach axis) at home
    cone_max_deg: float = 60.0
    v_max: float = 0.25                      # m/s
    w_max: float = 1.0                       # rad/s
    slack_s: float = 0.010
    dt_min: float = 0.0          # unused (kept so old configs load)
    dt_max: float = 0.05         # unused

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyConfig":
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SafetyFilter:
    cfg: SafetyConfig
    last: np.ndarray | None = None
    last_t: float | None = None
    counts: dict = field(default_factory=lambda: {"nan": 0, "box": 0, "cone": 0, "step_pos": 0, "step_rot": 0, "ok": 0})
    bucket_p: float | None = None
    bucket_r: float | None = None

    def reset(self, T: np.ndarray | None, t: float | None = None):
        self.last = None if T is None else np.array(T, float)
        self.last_t = t
        self.bucket_p = self.cfg.v_max * self.cfg.slack_s
        self.bucket_r = self.cfg.w_max * self.cfg.slack_s

    def apply(self, T_target: np.ndarray, now: float) -> tuple[np.ndarray | None, dict]:
        flags = {}
        T = np.array(T_target, float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            self.counts["nan"] += 1
            return None, {"nan": True}
        R = T[:3, :3]
        u, _, vt = np.linalg.svd(R)  # re-orthonormalise
        R = u @ vt
        p = T[:3, 3]
        lo, hi = np.asarray(self.cfg.workspace_min), np.asarray(self.cfg.workspace_max)
        pc = np.clip(p, lo, hi)
        if not np.allclose(pc, p):
            flags["box"] = True
            self.counts["box"] += 1
        p = pc
        axis = np.asarray(self.cfg.cone_axis, float)
        axis = axis / np.linalg.norm(axis)
        z = R[:, 2]
        ang = float(np.arccos(np.clip(np.dot(z, axis), -1, 1)))
        amax = np.radians(self.cfg.cone_max_deg)
        if ang > amax:
            k = np.cross(z, axis)
            n = np.linalg.norm(k)
            k = k / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
            R = Rotation.from_rotvec(k * (ang - amax)).as_matrix() @ R
            flags["cone"] = True
            self.counts["cone"] += 1
        if self.last is not None:
            if self.bucket_p is None:
                self.bucket_p = self.cfg.v_max * self.cfg.slack_s
                self.bucket_r = self.cfg.w_max * self.cfg.slack_s
            el = max(0.0, now - self.last_t)
            self.bucket_p = min(self.bucket_p + self.cfg.v_max * el, self.cfg.v_max * self.cfg.slack_s)
            self.bucket_r = min(self.bucket_r + self.cfg.w_max * el, self.cfg.w_max * self.cfg.slack_s)
            dp = p - self.last[:3, 3]
            dn = float(np.linalg.norm(dp))
            if dn > self.bucket_p:
                p = self.last[:3, 3] + dp * (self.bucket_p / dn)
                flags["step_pos"] = True
                self.counts["step_pos"] += 1
                dn = self.bucket_p
            self.bucket_p -= dn
            r_last = Rotation.from_matrix(self.last[:3, :3])
            r_new = Rotation.from_matrix(R)
            dth = float((r_last.inv() * r_new).magnitude())
            if dth > self.bucket_r:
                R = Slerp([0, 1], Rotation.concatenate([r_last, r_new]))(self.bucket_r / dth).as_matrix()
                flags["step_rot"] = True
                self.counts["step_rot"] += 1
                dth = self.bucket_r
            self.bucket_r -= dth
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = R, p
        self.last, self.last_t = out, now
        if not flags:
            self.counts["ok"] += 1
        return out, flags
