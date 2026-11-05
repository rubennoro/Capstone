"""``oracle`` backend: the synthetic ground truth (``truth.npz`` in the episode dir) plus configurable SLAM errors.

``truth.npz`` (written by ``dataproc sim``):

* ``t_ns`` (N,) int64: dense, strictly increasing true times on the episode's IMU time base (with the rig's time
  offsets 0, as in rig_v1, that is the raw Pi clock);
* ``T_W_imu`` (N,4,4) float64: true IMU pose in the gravity-aligned z-up world;
* ``width_m`` (N,): true gripper width; ``rig_id``, ``seed``, ``version`` (scalars).

What the oracle does, in order (every random draw from ``seed``, default the truth's seed):

1. samples ``T_W_imu`` at the ultrawide frame times (linear position, SLERP rotation); frames outside the truth
   span (or across a truth gap > ``max_truth_gap_ns``) are untracked;
2. per-frame white noise: position ``noise_pos_m`` (per axis, world), rotation ``noise_rot_deg`` (per axis, body);
3. drift: a random walk in position (``drift_pos_m_per_sqrt_s``) and rotation (``drift_rot_deg_per_sqrt_s``),
   applied on the world side;
4. an arbitrary rigid change of world ``T_slam_W = [Rz(world_yaw_deg) @ R_tilt | world_translation_m]``, the tilt
   being ``world_tilt_deg`` about a horizontal axis at azimuth ``world_tilt_azimuth_deg``. This is what a VIO
   world looks like (ORB-SLAM3: the first keyframe's IMU frame);
5. the shared :func:`dataproc.slam.base.finalize_trajectory` (gravity alignment from the episode's IMU, checks).
   With ``gravity_align=False`` step 5 skips the alignment (only sensible with ``world_tilt_deg=0``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dataproc.geometry.trajectory import Trajectory
from dataproc.slam.base import SlamBackend, SlamError, SlamResult, finalize_trajectory, ultrawide_times_ns

__all__ = ["TRUTH_FILE", "load_truth", "OracleBackend"]

TRUTH_FILE = "truth.npz"
_TRUTH_KEYS = ("t_ns", "T_W_imu", "width_m", "rig_id", "seed", "version")


def load_truth(path) -> dict:
    """Load and check ``truth.npz`` (keys above). Returns a dict of numpy arrays / Python scalars."""
    path = Path(path)
    if not path.is_file():
        raise SlamError(f"oracle: no ground truth {path}")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in _TRUTH_KEYS if k not in z.files]
        if missing:
            raise SlamError(f"oracle: {path} lacks {missing}")
        d = {k: z[k] for k in z.files}
    t = np.asarray(d["t_ns"], dtype=np.int64)
    T = np.asarray(d["T_W_imu"], dtype=np.float64)
    if t.ndim != 1 or T.shape != (len(t), 4, 4) or np.asarray(d["width_m"]).shape[0] != len(t):
        raise SlamError(f"oracle: {path}: inconsistent shapes t_ns {t.shape}, T_W_imu {T.shape}, "
                        f"width_m {np.asarray(d['width_m']).shape}")
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise SlamError(f"oracle: {path}: t_ns must be strictly increasing with >= 2 samples")
    d["t_ns"], d["T_W_imu"] = t, T
    d["width_m"] = np.asarray(d["width_m"], dtype=np.float64).reshape(len(t))
    for k in ("rig_id", "version"):
        d[k] = str(np.asarray(d[k]).item())
    d["seed"] = int(np.asarray(d["seed"]).item())
    return d


def _tilt(deg: float, azimuth_deg: float) -> np.ndarray:
    a = np.radians(azimuth_deg)
    return Rotation.from_rotvec(np.radians(deg) * np.array([np.cos(a), np.sin(a), 0.0])).as_matrix()


class OracleBackend(SlamBackend):
    name = "oracle"

    def __init__(self, *, noise_pos_m: float = 0.0, noise_rot_deg: float = 0.0,
                 drift_pos_m_per_sqrt_s: float = 0.0, drift_rot_deg_per_sqrt_s: float = 0.0,
                 world_yaw_deg: float = 0.0, world_tilt_deg: float = 0.0, world_tilt_azimuth_deg: float = 0.0,
                 world_translation_m=(0.0, 0.0, 0.0), gravity_align: bool = True, align_min_deg: float = 0.0,
                 seed: int | None = None, max_truth_gap_ns: int | None = None, truth_file: str = TRUTH_FILE):
        for k, v in dict(noise_pos_m=noise_pos_m, noise_rot_deg=noise_rot_deg,
                         drift_pos_m_per_sqrt_s=drift_pos_m_per_sqrt_s,
                         drift_rot_deg_per_sqrt_s=drift_rot_deg_per_sqrt_s).items():
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"oracle: {k} must be >= 0, got {v}")
        self.noise_pos_m, self.noise_rot_deg = float(noise_pos_m), float(noise_rot_deg)
        self.drift_pos, self.drift_rot_deg = float(drift_pos_m_per_sqrt_s), float(drift_rot_deg_per_sqrt_s)
        self.T_slam_W = np.eye(4)
        self.T_slam_W[:3, :3] = Rotation.from_euler("z", world_yaw_deg, degrees=True).as_matrix() @ _tilt(
            world_tilt_deg, world_tilt_azimuth_deg)
        self.T_slam_W[:3, 3] = np.asarray(world_translation_m, dtype=np.float64).reshape(3)
        self.gravity_align, self.align_min_deg = bool(gravity_align), float(align_min_deg)
        self.seed, self.max_truth_gap_ns, self.truth_file = seed, max_truth_gap_ns, truth_file

    def options(self) -> dict:
        return {"noise_pos_m": self.noise_pos_m, "noise_rot_deg": self.noise_rot_deg,
                "drift_pos_m_per_sqrt_s": self.drift_pos, "drift_rot_deg_per_sqrt_s": self.drift_rot_deg,
                "T_slam_W": self.T_slam_W.tolist(), "gravity_align": self.gravity_align, "seed": self.seed}

    def run(self, episode, rig, work_dir=None) -> SlamResult:
        truth = load_truth(Path(episode.path) / self.truth_file)
        if truth["rig_id"] != rig.rig_id:
            raise SlamError(f"oracle: truth rig_id {truth['rig_id']!r} != rig {rig.rig_id!r}")
        t_uw = ultrawide_times_ns(episode, rig)
        tt = truth["t_ns"]
        gap = self.max_truth_gap_ns or int(5 * np.median(np.diff(tt)))
        T_true, ok = Trajectory.from_T(tt, truth["T_W_imu"]).sample(t_uw, max_gap_ns=gap)
        if ok.sum() < 2:
            raise SlamError(f"oracle: only {int(ok.sum())} ultrawide frames inside the ground-truth span")
        t, T = t_uw[ok], T_true[ok]
        n = len(t)
        rng = np.random.default_rng(truth["seed"] if self.seed is None else self.seed)

        # white noise: position in the world, rotation in the body
        T = T.copy()
        T[:, :3, 3] += rng.normal(0.0, self.noise_pos_m, (n, 3)) if self.noise_pos_m else 0.0
        if self.noise_rot_deg:
            dR = Rotation.from_rotvec(np.radians(rng.normal(0.0, self.noise_rot_deg, (n, 3)))).as_matrix()
            T[:, :3, :3] = T[:, :3, :3] @ dR
        # drift: random walks applied on the world side, zero at the first tracked frame
        if self.drift_pos or self.drift_rot_deg:
            dt = np.diff(t, prepend=t[0]) * 1e-9
            sq = np.sqrt(dt)[:, None]
            dp = np.cumsum(rng.normal(0.0, 1.0, (n, 3)) * sq * self.drift_pos, axis=0)
            dth = np.cumsum(rng.normal(0.0, 1.0, (n, 3)) * sq * np.radians(self.drift_rot_deg), axis=0)
            Rd = Rotation.from_rotvec(dth).as_matrix()
            T[:, :3, :3] = Rd @ T[:, :3, :3]
            T[:, :3, 3] = np.einsum("nij,nj->ni", Rd, T[:, :3, 3]) + dp
        T = self.T_slam_W @ T
        traj = Trajectory.from_T(t, T, frame="imu", world="oracle_slam")
        return finalize_trajectory(traj, episode, rig, t_uw, align=self.gravity_align,
                                   align_min_deg=self.align_min_deg,
                                   diagnostics={"backend": self.name, "options": self.options(),
                                                "truth_version": truth["version"]})
