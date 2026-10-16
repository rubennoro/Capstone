"""Frame conventions for turning a VIO trajectory into ARKit ``camera.transform`` poses (brief D4, RECONCILIATION R2).

Chain (P2 §8)::

    T_out(t) = T_world_offset @ T_YUP_ZUP @ T_Wz_S(t) @ T_S_vmain_arkit

- ``T_Wz_S``: VIO pose of sensor frame S in a gravity-aligned **z-up** world (OpenCV axes for camera frames).
- ``T_S_vmain_arkit = rig.T(S, "virtual_main") @ CV_TO_ARKIT`` (``rig.T_S_virtual_main_arkit(S)``): the virtual main
  camera in S, with ARKit camera axes (+x right, +y up, +z backward).
- ``T_YUP_ZUP`` rotates the z-up world into ARKit's +y-up world (x->x, z->y, y->-z).
- ``T_world_offset`` (``rig.export.T_world_offset``): fixed yaw about +y + translation so no pose is ever exactly the
  identity (R2: identity rotation -> NaN in iphumi ``pose_6d_to_4x4``).
"""

from __future__ import annotations

import numpy as np

from sensor_rig.transforms import D_CV_ARKIT, T_YUP_ZUP, inv_T  # noqa: F401 (re-exported)

A_YUP_FROM_ZUP = np.array(T_YUP_ZUP[:3, :3])  # p_yup = A @ p_zup (x->x, z->y, y->-z), det +1
CV_TO_ARKIT = np.array(D_CV_ARKIT)  # T_arkit = T_cv @ CV_TO_ARKIT (an involution)

# rig.T() frame names (P2). Trajectory files may say "imu0" (EuRoC name); map it.
FRAME_ALIASES = {"imu0": "imu", "imu": "imu", "main": "main", "ultrawide": "ultrawide"}


def rig_frame(name: str) -> str:
    if name not in FRAME_ALIASES:
        raise ValueError(f"unknown trajectory frame {name!r}; expected one of {sorted(FRAME_ALIASES)}")
    return FRAME_ALIASES[name]


def hom(R: np.ndarray | None = None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project (...,3,3) matrices onto SO(3) (SVD; batched, unlike transforms.orthonormalize)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    d = np.sign(np.linalg.det(U @ Vt))
    U = U.copy()
    U[..., :, 2] *= d[..., None]
    return U @ Vt


def vio_to_arkit_main(T_Wz_S: np.ndarray, T_S_vmain_arkit: np.ndarray, T_world_offset: np.ndarray) -> np.ndarray:
    """(N,4,4) VIO poses of S (z-up world) -> (N,4,4) ARKit camera.transform of the virtual main camera."""
    return T_world_offset @ T_YUP_ZUP @ np.asarray(T_Wz_S, dtype=np.float64) @ T_S_vmain_arkit


def arkit_main_to_vio(T_arkit: np.ndarray, T_S_vmain_arkit: np.ndarray, T_Wz_change: np.ndarray) -> np.ndarray:
    """Inverse chain (simulator): ARKit main poses -> poses of S in a z-up world changed by ``T_Wz_change``.

    ``vio_to_arkit_main(arkit_main_to_vio(T, X, C), X, inv(T_YUP_ZUP @ C @ inv(T_YUP_ZUP))) == T``.
    """
    return T_Wz_change @ inv_T(T_YUP_ZUP) @ np.asarray(T_arkit, dtype=np.float64) @ inv_T(T_S_vmain_arkit)


def rig_vio_to_arkit_main(rig, T_Wz_S: np.ndarray, frame: str) -> np.ndarray:
    """The export chain with every constant taken from the rig config."""
    X = rig.T_S_virtual_main_arkit(rig_frame(frame))
    return vio_to_arkit_main(T_Wz_S, X, rig.export.T_world_offset)
