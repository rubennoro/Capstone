"""Pure functions on 4x4 rigid transforms. T_a_b maps coordinates in frame b to frame a (metres).

Frames (PLAN §6.6): flange F (Franka's; z out of the plate toward the fingers), hand H (the Franka Hand's axes),
camera (OpenCV optical: z out of the lens, x right, y down in the image).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from deploy.errors import GeometryError

# The Franka Hand is mounted on the flange rotated -45 deg about the flange z axis, with no offset: hand axes in
# flange coordinates. A fact of the hardware, not a setting (gripper.yaml is written in the hand frame).
F_T_H: np.ndarray = np.eye(4)
F_T_H[:3, :3] = Rotation.from_euler("z", -45.0, degrees=True).as_matrix()
F_T_H.setflags(write=False)

_RIGID_TOL = 1e-6        # numerical tolerance for "orthonormal" and the last row; typed-in values have 9 decimals


def camera_pose(position_m: np.ndarray, up_axis: np.ndarray, tilt_deg: float) -> np.ndarray:
    """H_T_cam: a camera at position_m (hand frame) on the up_axis side, looking along hand z tilted by tilt_deg
    away from up_axis (toward the fingers), with image y pointing away from up_axis."""
    up = np.asarray(up_axis, dtype=float)
    up = up / np.linalg.norm(up)
    a = math.radians(tilt_deg)
    z = math.cos(a) * np.array([0.0, 0.0, 1.0]) - math.sin(a) * up
    y = -(up - (up @ z) * z)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = np.asarray(position_m, dtype=float)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """The inverse of a rigid transform."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def pose_error(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(translation difference in mm, rotation angle between them in degrees)."""
    mm = float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000.0)
    cos = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1.0) / 2.0
    deg = math.degrees(math.acos(min(1.0, max(-1.0, float(cos)))))
    return mm, deg


def as_rigid(value: object, label: str) -> np.ndarray:
    """A 4x4 row-major list (or array) checked to be a rigid transform; GeometryError naming `label` otherwise."""
    try:
        T = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise GeometryError(f"{label}: expected a 4x4 matrix of numbers, got {value!r}") from None
    if T.shape != (4, 4):
        raise GeometryError(f"{label}: expected a 4x4 matrix, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise GeometryError(f"{label}: not every entry is finite")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=_RIGID_TOL):
        raise GeometryError(f"{label}: last row must be [0, 0, 0, 1], got {T[3].tolist()}")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=_RIGID_TOL) or np.linalg.det(R) < 0:
        raise GeometryError(f"{label}: the rotation part is not a rotation (orthonormal, determinant +1)")
    return T
