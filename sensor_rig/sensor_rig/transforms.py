"""Rigid-transform helpers and frame conventions.

Conventions (normative for the whole package):

* ``T_a_b`` is a 4x4 float64 matrix mapping points in frame ``b`` to frame ``a`` (``p_a = T_a_b @ p_b``),
  i.e. the pose of ``b`` expressed in ``a``. Metres.
* Camera frames are OpenCV optical (x right, y down, z forward) unless a name ends in ``_arkit``. ARKit camera
  axes are x right, y up, z backward: ``T_x_cam_arkit = T_x_cam_cv @ D_CV_ARKIT``.
* Inertial VIO worlds are gravity-aligned z-up; the exported ARKit world is gravity-aligned y-up:
  ``p_yup = T_YUP_ZUP @ p_zup``.
* Quaternions follow TUM / scipy order: scalar last ``(qx, qy, qz, qw)``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "make_T", "inv_T", "compose", "is_rigid", "orthonormalize",
    "rot_x", "rot_y", "rot_z", "R_from_euler_xyz_deg",
    "D_CV_ARKIT", "cv_to_arkit", "arkit_to_cv", "T_YUP_ZUP",
    "T_from_tum", "tum_from_T", "rotation_angle_deg",
]

#: ``T_arkit = T_cv @ D_CV_ARKIT`` (flip camera y and z). Its own inverse.
D_CV_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])
D_CV_ARKIT.flags.writeable = False

#: Maps points from a z-up world to a y-up world: x -> x, z -> y, y -> -z.
T_YUP_ZUP = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
T_YUP_ZUP.flags.writeable = False


def make_T(R=None, t=None) -> np.ndarray:
    """4x4 from a 3x3 rotation (default identity) and a 3-vector translation (default zero)."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (or a stack ``(..., 4, 4)`` of them), using R^T."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Ti = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    Ti[..., 3, 3] = 1.0
    return Ti


def compose(*Ts: np.ndarray) -> np.ndarray:
    """``compose(A, B, C) = A @ B @ C``; identity when called without arguments."""
    out = np.eye(4)
    for T in Ts:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def is_rigid(T: np.ndarray, tol: float = 1e-6) -> bool:
    """True if ``T`` is a 4x4 with an orthonormal, det=+1 rotation and last row [0,0,0,1] (all to ``tol``)."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return bool(
        np.allclose(R @ R.T, np.eye(3), atol=tol)
        and abs(np.linalg.det(R) - 1.0) <= tol
        and np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=tol)
    )


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (Frobenius norm) via SVD, with det=+1 enforced."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ S @ Vt


def _rot(axis: int, deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


def rot_x(deg: float) -> np.ndarray:
    """Right-handed rotation about +x by ``deg`` degrees."""
    return _rot(0, deg)


def rot_y(deg: float) -> np.ndarray:
    """Right-handed rotation about +y by ``deg`` degrees."""
    return _rot(1, deg)


def rot_z(deg: float) -> np.ndarray:
    """Right-handed rotation about +z by ``deg`` degrees."""
    return _rot(2, deg)


def R_from_euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic x-y-z Euler angles in degrees (scipy ``"xyz"``): ``Rz(rz) @ Ry(ry) @ Rx(rx)``.

    Entries within 1e-15 of zero are snapped to exactly 0.0 so that e.g. ``[0, 0, 0]`` gives an exact identity.
    """
    R = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    R[np.abs(R) < 1e-15] = 0.0
    return R


def cv_to_arkit(T_x_cam_cv: np.ndarray) -> np.ndarray:
    """Pose of an OpenCV-axes camera -> pose of the same camera with ARKit axes (``T @ D``)."""
    return np.asarray(T_x_cam_cv, dtype=np.float64) @ D_CV_ARKIT


def arkit_to_cv(T_x_cam_arkit: np.ndarray) -> np.ndarray:
    """Inverse of :func:`cv_to_arkit` (``D`` is its own inverse)."""
    return np.asarray(T_x_cam_arkit, dtype=np.float64) @ D_CV_ARKIT


def T_from_tum(t, x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """Pose from one TUM row ``t x y z qx qy qz qw`` (``t`` is ignored). Arrays give a ``(N, 4, 4)`` stack."""
    q = np.stack(np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (qx, qy, qz, qw))), axis=-1)
    p = np.stack(np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (x, y, z))), axis=-1)
    single = q.ndim == 1
    q2 = q.reshape(-1, 4)
    norm = np.linalg.norm(q2, axis=1)
    if np.any(np.abs(norm - 1.0) > 1e-3):
        raise ValueError("TUM quaternion is not unit length (|q| deviates by more than 1e-3)")
    T = np.tile(np.eye(4), (len(q2), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(q2).as_matrix()
    T[:, :3, 3] = p.reshape(-1, 3)
    return T[0] if single else T


def tum_from_T(T: np.ndarray) -> np.ndarray:
    """``(x, y, z, qx, qy, qz, qw)`` with ``qw >= 0``; a ``(N, 4, 4)`` stack gives ``(N, 7)``."""
    T = np.asarray(T, dtype=np.float64)
    single = T.ndim == 2
    T2 = T.reshape(-1, 4, 4)
    q = Rotation.from_matrix(T2[:, :3, :3]).as_quat()
    q[q[:, 3] < 0] *= -1.0
    out = np.concatenate([T2[:, :3, 3], q], axis=1)
    return out[0] if single else out


def rotation_angle_deg(R: np.ndarray) -> float:
    """Rotation angle of a 3x3 rotation (or of the rotation part of a 4x4) in degrees, in [0, 180]."""
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    return float(np.rad2deg(Rotation.from_matrix(R).magnitude()))
