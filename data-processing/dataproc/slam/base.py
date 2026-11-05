"""The SLAM backend interface and the post-processing every backend shares.

A backend turns one raw episode into the IMU (body) trajectory ``T_W_imu`` at the ultrawide frame times::

    result = get_backend("oracle", noise_pos_m=0.001).run(episode, rig, work_dir)
    result.trajectory          # Trajectory(frame="imu", world="z_up_gravity"), t_ns = tracked ultrawide frame times
    result.fraction_tracked    # tracked ultrawide frames / all ultrawide frames

Conventions (normative for every backend):

* **Times**: ``trajectory.t_ns`` is a subset of :func:`ultrawide_times_ns` (the mid-exposure times of the
  ultrawide frames on the IMU time base, ``rig.camera_time_ns("ultrawide", t_ns, exposure_ns)``), strictly
  increasing. Untracked frames are simply absent; ``SlamResult.tracked`` marks them per ultrawide frame.
* **Frame**: rows are ``T_W_imu``, the pose of the IMU (ORB-SLAM3's "body") in the world. The ultrawide camera is
  ``T_W_imu @ rig.cameras["ultrawide"].T_imu_cam``.
* **World**: gravity-aligned **z-up** (gravity along -z), metres. Yaw and origin are arbitrary (whatever the
  backend's world was, tilted by the minimal rotation of :func:`dataproc.geometry.trajectory.gravity_align`).
  The training ``world_offset`` is **not** applied here (the pose stage applies it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dataproc.geometry.trajectory import (
    Trajectory, gravity_align, gravity_check, motion_consistency_check, scale_check, time_offset_check,
)

__all__ = ["SlamError", "SlamResult", "SlamBackend", "ultrawide_camera", "ultrawide_times_ns", "imu_samples",
           "finalize_trajectory"]


class SlamError(RuntimeError):
    """The backend could not produce a trajectory (container failed, nothing tracked, bad inputs)."""


@dataclass(eq=False)
class SlamResult:
    """Output of :meth:`SlamBackend.run` (conventions: module docstring).

    ``t_frames_ns``/``tracked`` cover every ultrawide frame of the episode; ``trajectory.t_ns`` equals
    ``t_frames_ns[tracked]``. ``problems`` are human-readable failed sanity checks (gravity, time offset, jumps);
    the backend does not reject an episode for them, the validate stage judges.
    """

    trajectory: Trajectory
    t_frames_ns: np.ndarray
    tracked: np.ndarray
    n_maps: int = 1
    n_resets: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def fraction_tracked(self) -> float:
        return float(np.mean(self.tracked)) if len(self.tracked) else 0.0

    def summary(self) -> dict:
        """JSON-serialisable summary (no per-frame arrays)."""
        return {"n_frames": int(len(self.t_frames_ns)), "n_tracked": int(np.sum(self.tracked)),
                "fraction_tracked": self.fraction_tracked, "n_maps": int(self.n_maps), "n_resets": int(self.n_resets),
                "problems": list(self.problems), "diagnostics": self.diagnostics}


class SlamBackend:
    """Interface. Subclasses set ``name`` and implement :meth:`run`."""

    name = "base"

    def run(self, episode, rig, work_dir) -> SlamResult:   # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------------------------- shared helpers

def ultrawide_camera(episode) -> str:
    """The camera id (``cam1`` in the contract) whose role is ``ultrawide``."""
    try:
        return episode.meta.camera("ultrawide")
    except Exception:       # meta unreadable: fall back to the contract's fixed assignment
        return "cam1"


def ultrawide_times_ns(episode, rig) -> np.ndarray:
    """Mid-exposure times (IMU time base) of every ultrawide frame, int64, one per decoded frame."""
    fr = episode.frames(ultrawide_camera(episode))
    return np.asarray(rig.camera_time_ns("ultrawide", fr.t_ns, fr.exposure_ns), dtype=np.int64)


def imu_samples(episode, rig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t_ns on the IMU time base, gyro (N,3) rad/s, accel (N,3) m/s^2)`` in the IMU frame."""
    imu = episode.imu
    return np.asarray(rig.imu_time_ns(imu.t_ns), dtype=np.int64), np.asarray(imu.gyro), np.asarray(imu.accel)


def finalize_trajectory(traj: Trajectory, episode, rig, t_frames_ns: np.ndarray, *, align: bool = True,
                        align_min_deg: float = 0.0, motion_limits: dict | None = None,
                        diagnostics: dict | None = None, n_maps: int = 1, n_resets: int = 0) -> SlamResult:
    """Gravity-align an IMU trajectory (world arbitrary, frame imu) and run the IMU sanity checks.

    ``align_min_deg``: the estimated tilt is only applied above this angle (0 = always). The estimate is the mean
    specific force: on a moving episode it is good to ~0.5 deg (the mean acceleration is only ~0), and an
    accelerometer bias tilts it by ``|b_perp| / g`` (20 mg ~ 1.2 deg). That is harmless for training, which uses
    relative poses. ``gravity_check`` after the alignment is then ~0 deg by construction: it only informs through
    ``|g|``. Checks: gravity, time offset, motion (jumps / gyro residual) and metric scale (:func:`scale_check`,
    problems ``scale_mismatch`` / ``scale_unobservable``).
    """
    if traj.frame != "imu":
        raise ValueError(f"finalize_trajectory expects an imu-frame trajectory, got {traj.frame!r}")
    diag = dict(diagnostics or {})
    problems: list[str] = []
    if len(traj) < 2:
        raise SlamError(f"backend produced {len(traj)} poses (need >= 2)")
    imu_t, gyro, acc = imu_samples(episode, rig)
    align_info: dict = {"applied": False}
    if align:
        try:
            aligned, align_info = gravity_align(traj, imu_t, acc, rig)
        except ValueError as e:
            align_info = {"applied": False, "reason": str(e)}
            problems.append(f"gravity alignment impossible: {e}")
        else:
            if align_info["angle_deg"] >= align_min_deg:
                traj = aligned
            else:
                align_info = {**align_info, "applied": False,
                              "reason": f"estimated tilt below {align_min_deg} deg; world kept"}
    traj = Trajectory(traj.t_ns, traj.p, traj.q_xyzw, frame="imu", world="z_up_gravity")
    grav = gravity_check(traj, imu_t, acc, rig)
    toff = time_offset_check(traj, imu_t, gyro, rig)
    motion = motion_consistency_check(traj, imu_t, gyro, rig, limits=motion_limits)
    if not grav.get("ok"):
        problems.append(f"gravity check failed: {_short(grav)}")
    if toff.get("traj_delay_ms") is not None and not toff.get("ok"):
        problems.append(f"time offset check failed: {_short(toff)}")
    problems += [f"motion check: {p}" for p in motion["problems"]]
    scale = scale_check(traj, imu_t, acc, rig)
    problems += scale["problems"]
    diag.update(gravity_alignment=align_info, gravity_check=grav, time_offset_check=toff, motion_check=motion,
                scale_check=scale)

    t_frames_ns = np.asarray(t_frames_ns, dtype=np.int64)
    tracked = np.isin(t_frames_ns, traj.t_ns)
    if int(tracked.sum()) != len(traj):
        raise SlamError("trajectory times are not a subset of the ultrawide frame times")
    return SlamResult(trajectory=traj, t_frames_ns=t_frames_ns, tracked=tracked, n_maps=n_maps, n_resets=n_resets,
                      diagnostics=_jsonable(diag), problems=problems)


def _short(d: dict) -> str:
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()
                     if not isinstance(v, (list, dict)))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x

