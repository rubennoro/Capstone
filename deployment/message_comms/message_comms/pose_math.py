# adapted from robot-utils==0.1.4:robot_utils/pose_utils.py (get_relative_pose, get_absolute_pose, interpolate_xyz_wxyz)
# and behavior_prompting:behavior_prompting/common/pose_util.py @ edfe8bd (mat_to_rot6d, rot6d_to_mat)
# changes: rewritten with 4x4 matrices instead of xyz+wxyz quaternions. Same math, checked against the copied
#          originals in tests/reference_pose_utils.py.
"""Pure pose functions, float64, on stacks of matrices (STRUCTURE.md section 6)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_EPS = 1e-12      # the reference normalize()'s eps: only guards a division by zero


def invert(T: np.ndarray) -> np.ndarray:
    """Rigid inverse [R^T, -R^T p] of (..., 4, 4)."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    out = np.zeros(T.shape, dtype=np.float64)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, p)
    out[..., 3, 3] = 1.0
    return out


def relative(T_anchor: np.ndarray, T: np.ndarray) -> np.ndarray:
    """inv(T_anchor) @ T: T expressed in the anchor's frame."""
    return invert(T_anchor) @ np.asarray(T, dtype=np.float64)


def absolute(T_anchor: np.ndarray, T_rel: np.ndarray) -> np.ndarray:
    """T_anchor @ T_rel: a relative pose back in the base frame."""
    return np.asarray(T_anchor, dtype=np.float64) @ np.asarray(T_rel, dtype=np.float64)


def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two ROWS."""
    R = np.asarray(R, dtype=np.float64)
    return R[..., :2, :].reshape(R.shape[:-2] + (6,)).copy()


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def matrix_from_rot6d(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3): Gram-Schmidt on the two rows; the result's rows are (b1, b2, b1 x b2)."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def interpolate(T0: np.ndarray, T1: np.ndarray, a: float) -> np.ndarray:
    """Linear position, slerp rotation, a fraction a in [0, 1] of the way from T0 to T1 (one pair of (4, 4))."""
    T0 = np.asarray(T0, dtype=np.float64)
    T1 = np.asarray(T1, dtype=np.float64)
    out = np.eye(4)
    out[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
    rots = Rotation.from_matrix(np.stack((T0[:3, :3], T1[:3, :3])))
    out[:3, :3] = Slerp([0.0, 1.0], rots)([a]).as_matrix()[0]
    return out


def position_distance_m(T0: np.ndarray, T1: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(T0)[:3, 3] - np.asarray(T1)[:3, 3]))


def rotation_angle_deg(R: np.ndarray) -> float:
    """The angle of one rotation matrix (its distance from identity), degrees."""
    R = np.asarray(R, dtype=np.float64)
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
