"""Metrics shared by the D10 acceptance tests and the end-to-end comparison script (poses, images)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from sensor_rig.transforms import inv_T


def rel_pose_errors(T_a: np.ndarray, T_b: np.ndarray, k: int | str) -> tuple[float, float]:
    """Max (rotation rad, translation m) difference between relative motions of two pose sequences.

    ``k`` int: ``inv(T[i]) T[i+k]`` for all i; ``"all"``: ``inv(T[0]) T[i]`` for all i.
    """
    T_a = np.asarray(T_a, dtype=np.float64)
    T_b = np.asarray(T_b, dtype=np.float64)
    if k == "all":
        Ra = inv_T(T_a[0])[None] @ T_a
        Rb = inv_T(T_b[0])[None] @ T_b
    else:
        k = int(k)
        if len(T_a) <= k:
            return 0.0, 0.0
        Ra = inv_T(T_a[:-k]) @ T_a[k:]
        Rb = inv_T(T_b[:-k]) @ T_b[k:]
    dR = np.swapaxes(Ra[:, :3, :3], 1, 2) @ Rb[:, :3, :3]
    rot = Rotation.from_matrix(dR).magnitude()
    trans = np.linalg.norm(Ra[:, :3, 3] - Rb[:, :3, 3], axis=1)
    return float(rot.max()), float(trans.max())


def world_transform_residual(T_a: np.ndarray, T_b: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Best constant D with ``T_a[i] ~ D T_b[i]``: D from the first pair; returns (D, max rot err rad, max trans err m)."""
    D = T_a[0] @ inv_T(T_b[0])
    P = D[None] @ T_b
    dR = np.swapaxes(P[:, :3, :3], 1, 2) @ T_a[:, :3, :3]
    return D, float(Rotation.from_matrix(dR).magnitude().max()), float(
        np.linalg.norm(P[:, :3, 3] - T_a[:, :3, 3], axis=1).max())


def pose6_to_T(pos: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    T[:, :3, 3] = pos
    return T


def pad_resize(img: np.ndarray, out: int = 224, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """iphumi ``get_image_transform_with_border``: pad to a centred black square, then resize (INTER_AREA)."""
    h, w = img.shape[:2]
    s = max(w, h)
    top, left = (s - h) // 2, (s - w) // 2
    sq = cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(sq, (out, out), interpolation=interpolation)


def mask_224(valid: np.ndarray, out: int = 224, erode_px: int = 0) -> np.ndarray:
    """A full-resolution validity mask carried through :func:`pad_resize`; a 224 pixel is valid only if every source
    pixel it averages is valid."""
    m = valid.astype(np.uint8)
    if erode_px:
        m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    m = pad_resize(m[..., None].repeat(3, axis=2) * 255, out)[..., 0]
    return m == 255


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if d.ndim == 3:
        d = d.mean(axis=2)
    if mask is not None:
        d = d[mask]
    return float(d.mean()) if d.size else float("nan")


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if d.ndim == 3:
        d = d.mean(axis=2)
    if mask is not None:
        d = d[mask]
    mse = float(d.mean()) if d.size else float("nan")
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def warp_to_K(img: np.ndarray, K_src: np.ndarray, K_dst: np.ndarray) -> np.ndarray:
    """Re-render a distortion-free pinhole image with another K (same size, same optical centre): H = K_dst K_src^-1."""
    K_src = np.asarray(K_src, dtype=np.float64)
    K_dst = np.asarray(K_dst, dtype=np.float64)
    if np.allclose(K_src, K_dst, atol=1e-4):
        return img
    H = K_dst @ np.linalg.inv(K_src)
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
