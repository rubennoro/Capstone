"""Stage ``pose``: SLAM IMU trajectory -> TCP poses on the output grid.

::

    T_W'_tcp(t) = rig.T_world_offset @ T_W_imu(t) @ rig.cameras["ultrawide"].T_imu_cam @ rig.T_cam_tcp

``T_W_imu`` is the backend's gravity-aligned z-up trajectory (:mod:`dataproc.slam.base`), interpolated at the grid
times (linear position, SLERP rotation); ``T_cam_tcp`` is in the physical ultrawide optical frame. The result is
stored as position (metres) and rotation vector (scipy ``as_rotvec``, the axis-angle iPhUMI's ``pose_6d_to_4x4``
reads back).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dataproc.geometry.trajectory import Trajectory

__all__ = ["MIN_ROTVEC_NORM", "T_imu_tcp", "tcp_poses"]

#: iPhUMI's ``pose_6d_to_4x4`` divides by the rotation angle; below this norm a row is flagged invalid.
MIN_ROTVEC_NORM = 1e-6


def T_imu_tcp(rig) -> np.ndarray:
    """The TCP in the IMU frame: ``T_imu_cam(ultrawide) @ T_cam_tcp``."""
    return np.asarray(rig.cameras["ultrawide"].T_imu_cam, np.float64) @ np.asarray(rig.T_cam_tcp, np.float64)


def tcp_poses(traj: Trajectory, t_grid_ns, rig, max_gap_ns: int, return_reasons: bool = False):
    """TCP poses at ``t_grid_ns`` (int64 on the IMU time base).

    Returns ``(eef_pos (N,3) float32, eef_rot_axis_angle (N,3) float32, valid (N,) bool)``, plus ``reasons``
    ((N,) str: ``""`` if valid, else ``outside_trajectory`` | ``gap`` | ``rotvec_near_zero``) with
    ``return_reasons=True``. A grid time is valid if it lies inside the trajectory span with the bracketing poses at
    most ``max_gap_ns`` apart (see :meth:`Trajectory.sample`) and ``|rotvec| > MIN_ROTVEC_NORM``. Invalid rows hold
    the nearest-pose value (finite), never NaN.
    """
    if traj.frame != "imu":
        raise ValueError(f"tcp_poses expects an imu-frame trajectory, got {traj.frame!r}")
    t = np.asarray(t_grid_ns, dtype=np.int64).reshape(-1)
    T, valid = traj.sample(t, max_gap_ns=int(max_gap_ns))
    T_W_tcp = np.asarray(rig.T_world_offset, np.float64) @ T @ T_imu_tcp(rig)
    pos = T_W_tcp[:, :3, 3]
    rotvec = Rotation.from_matrix(T_W_tcp[:, :3, :3]).as_rotvec() if len(t) else np.zeros((0, 3))
    reasons = np.full(len(t), "", dtype=object)
    if len(traj) and len(t):
        inside = (t >= traj.t_ns[0]) & (t <= traj.t_ns[-1])
        reasons[~inside] = "outside_trajectory"
        reasons[inside & ~valid] = "gap"
    elif len(t):
        reasons[:] = "outside_trajectory"
    near_zero = np.linalg.norm(rotvec, axis=1) <= MIN_ROTVEC_NORM
    reasons[valid & near_zero] = "rotvec_near_zero"
    valid = valid & ~near_zero
    out = (pos.astype(np.float32), rotvec.astype(np.float32), valid)
    return (*out, reasons.astype(str)) if return_reasons else out
