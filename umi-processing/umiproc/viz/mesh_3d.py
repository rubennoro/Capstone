"""EXPERIMENTAL TRIAL (may be removed if it does not work out): a textured mesh of a map's mapping clip.

What it does: the per-frame stereo depth maps of :mod:`umiproc.viz.dense_3d` (its steps 1-3, unchanged) are fused
into a truncated signed distance field (TSDF) on a voxel grid around the table, the zero surface is extracted with
marching cubes, and every triangle is textured from the one real frame that sees it best (most head-on, close,
sharp, not occluded, not on the gripper), after a per-frame gain and a lens-shading (vignetting) correction fitted
from the frames themselves. Writes ``analysis/mesh_3d.{html,glb,json}`` (a three.js viewer, the mesh with its
texture for MeshLab / Blender, the parameters and numbers).

``python -m umiproc.viz.mesh_3d FOLDER [--root DIR]`` (not a ``umiproc`` command). ``FOLDER``: a task folder's name
(:mod:`umiproc.viz.target`); the same inputs as :mod:`umiproc.viz.dense_3d`; nothing but ``analysis/mesh_3d.*`` is
written (and ``--depth-cache FILE``, if given, to skip the 1-minute stereo step on a re-run).

The steps (each one function; the defaults are in :class:`MeshParams`):

1. **Depth** (:func:`umiproc.viz.dense_3d.compute_depths`, :func:`consistent_depths`): dense_3d's depth maps, each
   pixel kept only when at least ``min_agree`` of the nearest frames' depth maps agree (dense_3d's fusion test;
   the unfiltered maps give ~4x the surface, mostly noise).
2. **Box** (:func:`crop_box`): the scene frame (gravity +z, the table tag at the origin, as in the viewers); the box
   is the 0.5-99.5 % range of the kept depth points no farther than ``depth_max_m`` from their camera, padded by
   ``box_pad_m``, the floor below ``z_min_m`` cut away.
3. **TSDF** (:class:`Tsdf`, :func:`filter_tsdf`): ``voxel_m`` voxels, truncation ``trunc_vox`` voxels. Per frame
   the voxels near each depth pixel's ray (sampled within the band) are found, then each voxel's centre is
   projected into the depth map (projective signed distance ``depth - z``, clipped to +-1 truncation, running
   weighted mean, weight ``(depth_weight_ref_m / depth)^2``). Then a normalised Gaussian smoothing
   (``smooth_vox``) that also fills cracks of unseen voxels (``fill_holes``).
4. **Surface** (:func:`extract_mesh`, :func:`clean_mesh`, :func:`plane_fill`): marching cubes where all 8 corners
   were seen at least ``w_min`` times; connected pieces under ``min_piece_frac`` of the triangles dropped. The
   untextured light wood gives no stereo depth in places, so holes in the fitted table plane (enclosed by the
   table after a ``close_m`` closing, not under an object, at most ``plane_fill_max_m2``) get a flat patch; a
   patch face that no frame sees unoccluded is dropped again in step 6 (``faces_table_fill`` in the json).
5. **Photometry** (:func:`fit_photometry`, :func:`tag_white_balance`): the same surface points seen by many frames
   give, by alternating least squares on log colours, a gain per frame and channel and a lens-shading field per
   channel (a 9 x 7 bilinear grid over the fisheye image; the IMX219's pink corners are mostly this: green drops
   to ~0.7 of the centre at the edge); the tag's white paper margin then sets the white balance.
6. **Views** (:func:`view_scores`, :func:`choose_views`): per frame a z-buffer of the mesh (vertex splats in the
   fisheye at 1/``zbuf_div`` size); per triangle a score ``cos(incidence) / distance x angle weight x sharpness``
   in every frame that sees it unoccluded and off the gripper mask (grown ``mask_grow_px``); the best ``n_cand``
   are kept and an ICM pass (``smooth_lambda``) makes neighbours prefer the same frame (fewer seams). A triangle no
   frame sees unoccluded takes the best frame that faces it (occlusion and mask ignored); one that faces away from
   every frame is dropped (noise, or the back of something never seen).
7. **Texture** (:func:`seam_levels`, :func:`bake_atlas`): one ``cell_px`` square per triangle in a texture atlas
   (the triangle in its upper-left half, the rest of the square filled from the triangle's plane so that filtering
   never bleeds), sampled from the full-size fisheye frame with the photometric correction and a per-vertex seam
   levelling (neighbouring faces from different frames meet at the same colour).

All of it on the CPU (numpy, scipy, OpenCV, scikit-image's marching cubes).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from umiproc.viz import dense_3d as D

__all__ = ["MeshParams", "View", "consistent_depths", "crop_box", "Tsdf", "filter_tsdf", "extract_mesh", "clean_mesh",
           "face_normals", "vertex_normals", "table_plane", "plane_fill", "fisheye_project", "zbuffer", "fit_photometry", "tag_white_balance", "view_scores", "choose_views",
           "face_neighbours", "pmap", "atlas_layout", "seam_levels", "bake_atlas", "vertex_colours", "write_glb", "build", "main"]

OUT_STEM = "mesh_3d"


@dataclass
class MeshParams:
    voxel_m: float = 0.004           # TSDF voxel
    trunc_vox: float = 5.0           # truncation band, in voxels
    depth_weight_ref_m: float = 0.5  # observation weight (ref / depth)^2 (0: weight 1)
    smooth_vox: float = 1.5          # Gaussian smoothing of the TSDF, in voxels (0: none)
    fill_holes: bool = True          # ... and fill cracks of unseen voxels with it
    plane_fill_max_m2: float = 0.3   # table holes up to this area get a flat patch (0: none)
    depth_max_m: float = 1.3         # depth pixels farther than this are not fused
    box_pad_m: float = 0.03
    z_min_m: float = -0.06           # scene z (tag plane = 0) below this is cut (the floor)
    w_min: float = 3.0               # a voxel is used when this many frames saw it
    min_piece_frac: float = 0.002    # connected pieces with fewer triangles than this fraction are dropped
    min_piece_tris: int = 200
    zbuf_div: int = 4                # z-buffer at 1/zbuf_div of the fisheye size
    mask_grow_px: int = 30           # the gripper mask grown by this for texturing (fingers move, shadows)
    max_theta_deg: float = 80.0      # fisheye angle limit for texturing
    min_cos: float = 0.15            # incidence: cos(angle between normal and view ray)
    n_cand: int = 6                  # candidate frames per triangle
    smooth_lambda: float = 0.6       # ICM: weight of "my neighbours use this frame" against the score ratio
    smooth_iters: int = 10
    seam_leveling: bool = True       # pull neighbouring faces' colours together at shared vertices
    cell_px: int = 8                 # atlas square per triangle (0: largest that fits max_atlas_px)
    max_atlas_px: int = 8192
    jpeg_quality: int = 88
    photo_points: int = 20000        # surface points used to fit the photometry
    photo_iters: int = 12
    tag_white_balance: bool = True   # make the tag's white paper margin neutral


@dataclass
class View:
    """A frame used for texturing: pose in the scene frame, its fisheye image, sharpness."""
    fed: int
    T_S_cam: np.ndarray              # 4x4 camera -> scene
    raw: np.ndarray                  # full-size fisheye RGB
    sharp_w: float = 1.0
    keyframe: bool = False


# ---------------------------------------------------------------------------------------------------- 1. depth


def consistent_depths(frames: list, dp: D.Params) -> list[np.ndarray | None]:
    """Each frame's depth map keeping only the pixels that pass dense_3d's multi-view agreement test."""
    out = []
    for i, f in enumerate(frames):
        if f.depth is None:
            out.append(None)
            continue
        _, _, pix, votes, _ = D.frame_votes(frames, i, dp)
        z = np.full_like(f.depth, np.nan)
        ok = votes >= dp.min_agree
        u, v = pix[ok, 0], pix[ok, 1]
        z[v, u] = f.depth[v, u]
        out.append(z)
    return out


def _cache_save(path: Path, frames: list, filtered: list) -> None:
    nan = np.full(frames[0].rgb.shape[:2], np.nan, np.float32)
    np.savez_compressed(path, fed=np.array([f.fed for f in frames]),
                        depth=np.array([f.depth if f.depth is not None else nan for f in frames]),
                        filtered=np.array([z if z is not None else nan for z in filtered]))


def load_depths(inp, dp: D.Params, cache: Path | None, clock) -> tuple[list, list, dict]:
    """``(frames, filtered depth maps, counts)``: from ``cache`` when it matches the frames, else computed (and
    cached)."""
    if cache is not None and Path(cache).is_file():
        frames = D.undistort_frames(inp, D.select_frames(inp, dp.step), dp)
        z = np.load(cache)
        if list(z["fed"]) == [f.fed for f in frames]:
            filtered = []
            for f, d, fz in zip(frames, z["depth"], z["filtered"]):
                has = bool(np.isfinite(d).any())
                f.depth = d if has else None
                filtered.append(fz if has else None)
            D.median_depths(frames, inp.export.points, dp.fov_deg / 2)
            clock.lap("decode + cached depth")
            return frames, filtered, {"after_blur_check": len(frames), "from_cache": str(cache),
                                      "with_depth": sum(f.depth is not None for f in frames)}
        print(f"  {cache}: other frames, recomputing", flush=True)
    frames, _, counts = D.compute_depths(inp, dp, clock)
    filtered = consistent_depths(frames, dp)
    clock.lap("multi-view check")
    if cache is not None:
        _cache_save(Path(cache), frames, filtered)
    return frames, filtered, counts


# ---------------------------------------------------------------------------------------------------- 2. box


def _apply(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ T[:3, :3].T + T[:3, 3]


def depth_points(z: np.ndarray, K: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(points in the camera (N, 3), u, v)`` of the finite pixels of a pinhole depth map."""
    zz = z[::stride, ::stride]
    vv, uu = np.mgrid[0:z.shape[0]:stride, 0:z.shape[1]:stride]
    ok = np.isfinite(zz)
    u, v, d = uu[ok], vv[ok], zz[ok]
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    return np.stack([x, y, d], 1), u, v


def crop_box(T_S_cams: list[np.ndarray], depths: list, K: np.ndarray, p: MeshParams) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` of the fusion box in the scene frame (module docstring, step 2)."""
    pts = []
    for T, z in zip(T_S_cams, depths):
        if z is None:
            continue
        X, _, _ = depth_points(z, K, 4)
        X = X[X[:, 2] <= p.depth_max_m]
        pts.append(_apply(T, X))
    X = np.concatenate(pts)
    X = X[X[:, 2] >= p.z_min_m]
    lo = np.percentile(X, 0.5, axis=0) - p.box_pad_m
    hi = np.percentile(X, 99.5, axis=0) + p.box_pad_m
    lo[2] = max(lo[2], p.z_min_m)
    return lo, hi


# ---------------------------------------------------------------------------------------------------- 3. TSDF


class Tsdf:
    """A dense TSDF grid over ``[lo, hi]`` (module docstring, step 3). ``tsdf`` in units of the truncation."""

    def __init__(self, lo, hi, voxel: float, trunc: float):
        self.lo = np.asarray(lo, np.float64)
        self.voxel, self.trunc = float(voxel), float(trunc)
        self.shape = tuple(int(n) for n in np.ceil((np.asarray(hi) - self.lo) / voxel).astype(int) + 1)
        self.tsdf = np.ones(self.shape, np.float32)
        self.weight = np.zeros(self.shape, np.float32)
        self.updates = 0

    def integrate(self, z: np.ndarray, K: np.ndarray, T_S_cam: np.ndarray, depth_max: float = np.inf,
                  w_ref: float = 0.0) -> int:
        """Fuse one pinhole depth map (z, NaN = none) taken from pose ``T_S_cam``; returns the voxels updated.
        Weight 1 per observation, or ``(w_ref / depth)^2`` (clipped to 0.05..4) when ``w_ref`` > 0 (stereo depth
        gets worse with the square of the distance)."""
        z = np.where(z <= depth_max, z, np.nan)
        Xc, _, _ = depth_points(z, K)
        if not len(Xc):
            return 0
        ray = Xc / Xc[:, 2:3]                                        # z = 1 rays
        step = 0.7 * self.voxel                                      # in z (sparser along oblique rays)
        offs = np.arange(-self.trunc, self.trunc + step / 2, step)
        R, t = T_S_cam[:3, :3], T_S_cam[:3, 3]
        n = np.asarray(self.shape)
        keys = []
        for s in offs:                                               # voxels near the rays, in the band
            P = (ray * (Xc[:, 2:3] + s)) @ R.T + t
            ijk = np.floor((P - self.lo) / self.voxel + 0.5).astype(np.int64)
            ok = np.all((ijk >= 0) & (ijk < n), axis=1)
            ijk = ijk[ok]
            keys.append((ijk[:, 0] * n[1] + ijk[:, 1]) * n[2] + ijk[:, 2])
        key = np.unique(np.concatenate(keys))
        ijk = np.stack(np.unravel_index(key, self.shape), 1)
        C = (self.lo + ijk * self.voxel - t) @ R                      # voxel centres in the camera
        zc = C[:, 2]
        with np.errstate(all="ignore"):
            u = np.rint(K[0, 0] * C[:, 0] / zc + K[0, 2])
            v = np.rint(K[1, 1] * C[:, 1] / zc + K[1, 2])
        h, wd = z.shape
        inside = (zc > 0.01) & (u >= 0) & (v >= 0) & (u < wd) & (v < h)
        key, zc = key[inside], zc[inside]
        dz = z[v[inside].astype(int), u[inside].astype(int)]
        sdf = dz - zc
        ok = np.isfinite(sdf) & (sdf >= -self.trunc)
        key, val = key[ok], np.minimum(1.0, sdf[ok] / self.trunc).astype(np.float32)
        w = np.clip((w_ref / dz[ok]) ** 2, 0.05, 4.0).astype(np.float32) if w_ref > 0 else np.float32(1.0)
        T, W = self.tsdf.reshape(-1), self.weight.reshape(-1)
        W0 = W[key]
        T[key] = (T[key] * W0 + val * w) / (W0 + w)
        W[key] = W0 + w
        self.updates += len(key)
        return int(len(key))


def filter_tsdf(tsdf: Tsdf, sigma_vox: float, w_min: float, smooth: bool = True, fill: bool = True,
                min_support: float = 0.35) -> int:
    """Normalised Gaussian convolution of the TSDF (weights capped at 4 x ``w_min`` so that well-seen voxels do not
    dominate). ``smooth``: seen voxels take the smoothed value (less bumpy surfaces from noisy depth). ``fill``:
    voxels seen fewer than ``w_min`` times whose neighbourhood holds ``min_support`` x ``w_min`` weight take it too
    and count as seen (closes cracks a few voxels wide; open rims grow by about as much). Returns the voxels
    filled."""
    from scipy.ndimage import gaussian_filter

    if sigma_vox <= 0 or not (smooth or fill):
        return 0
    W = np.minimum(tsdf.weight, 4 * w_min)
    den = gaussian_filter(W, sigma_vox, mode="constant", truncate=2.5)
    num = gaussian_filter(tsdf.tsdf * W, sigma_vox, mode="constant", truncate=2.5)
    seen = tsdf.weight >= w_min
    ok = den > 1e-6
    hole = ~seen & (den >= min_support * w_min) if fill else np.zeros_like(seen)
    upd = (seen & ok if smooth else np.zeros_like(seen)) | hole
    tsdf.tsdf[upd] = num[upd] / den[upd]
    tsdf.weight[hole] = w_min
    return int(hole.sum())


# ---------------------------------------------------------------------------------------------------- 4. surface


def extract_mesh(tsdf: Tsdf, w_min: float) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (V, 3) scene, faces (F, 3))``: marching cubes on cubes whose 8 corners were seen ``w_min``
    times; faces wound so that their normals point to the free (positive) side."""
    from skimage.measure import marching_cubes

    seen = tsdf.weight >= w_min
    cube = seen[:-1, :-1, :-1].copy()
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                cube &= seen[dx:dx + seen.shape[0] - 1, dy:dy + seen.shape[1] - 1, dz:dz + seen.shape[2] - 1]
    mask = np.zeros_like(seen)
    mask[1:, 1:, 1:] = cube                # skimage: mask[i] enables the cube with corners i-1 .. i
    if not cube.any():
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    V, F, _, _ = marching_cubes(tsdf.tsdf, 0.0, spacing=(tsdf.voxel,) * 3, mask=mask, allow_degenerate=False)
    V = V.astype(np.float64) + tsdf.lo
    F = F.astype(np.int64)
    # orientation: the TSDF grows towards free space, so the normal should follow its gradient
    n = face_normals(V, F)
    c = V[F].mean(axis=1)
    g = np.stack(np.gradient(tsdf.tsdf), -1)
    ijk = np.clip(np.rint((c - tsdf.lo) / tsdf.voxel).astype(int), 0, np.asarray(tsdf.shape) - 1)
    if np.mean(np.sum(n * g[ijk[:, 0], ijk[:, 1], ijk[:, 2]], axis=1)) < 0:
        F = F[:, ::-1].copy()
    return V, F


def face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def clean_mesh(V: np.ndarray, F: np.ndarray, min_frac: float, min_tris: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop connected pieces smaller than ``max(min_tris, min_frac x faces)`` triangles and unused vertices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not len(F):
        return V, F, {"pieces": 0, "pieces_kept": 0}
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    n_comp, lab = connected_components(g, directed=False)
    flab = lab[F[:, 0]]
    size = np.bincount(flab, minlength=n_comp)
    keep_c = size >= max(min_tris, min_frac * len(F))
    F = F[keep_c[flab]]
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    remap = np.cumsum(used) - 1
    return V[used], remap[F], {"pieces": int(n_comp), "pieces_kept": int(keep_c.sum())}


def table_plane(V: np.ndarray, Vn: np.ndarray, z_tol: float = 0.04) -> np.ndarray | None:
    """``(a, b, c)`` of the table plane z = a x + b y + c, robustly fitted to the upward vertices within
    ``z_tol`` of the tag plane (z = 0); ``None`` when there are too few."""
    X = V[(np.abs(V[:, 2]) < z_tol) & (Vn[:, 2] > 0.9)]
    coef = None
    for _ in range(5):
        if len(X) < 200:
            return coef
        A = np.c_[X[:, 0], X[:, 1], np.ones(len(X))]
        coef = np.linalg.lstsq(A, X[:, 2], rcond=None)[0]
        r = np.abs(X[:, 2] - A @ coef)
        X = X[r < max(3 * 1.4826 * np.median(r), 0.002)]
    return coef


def plane_fill(V: np.ndarray, F: np.ndarray, Vn: np.ndarray, cell: float, max_hole_m2: float,
               close_m: float = 0.15, near_m: float = 0.006, above_m: float = 0.012):
    """``(V_fill, F_fill, plane)``: flat patches closing the holes in the table (module docstring, step 4): on a
    ``cell`` grid over the fitted plane, cells with table vertices or face centres (within ``near_m``) are the
    table, cells with something above them (``above_m``) are objects; the table mask is closed over ``close_m``
    (gaps and bays that narrow), and its holes that are not under an object and at most ``max_hole_m2`` get two
    triangles per cell. (Whether a frame saw the patch is left to the view choice.)"""
    from scipy import ndimage

    empty = (np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    plane = table_plane(V, Vn)
    if plane is None:
        return (*empty, None)
    P = np.concatenate([V, V[F].mean(axis=1)])
    lo = P[:, :2].min(axis=0) - 2 * cell
    ij = np.floor((P[:, :2] - lo) / cell).astype(int)
    n = ij.max(axis=0) + 3
    h = P[:, 2] - (plane[0] * P[:, 0] + plane[1] * P[:, 1] + plane[2])
    table = np.zeros(n, bool)
    above = np.zeros(n, bool)
    occupied = np.zeros(n, bool)                                     # any surface near the plane: no patch
    table[tuple(ij[np.abs(h) < near_m].T)] = True
    above[tuple(ij[h > above_m].T)] = True
    occupied[tuple(ij[np.abs(h) < 5 * near_m].T)] = True
    it = max(1, int(round(close_m / cell / 2)))
    closed = ndimage.binary_fill_holes(ndimage.binary_closing(table, iterations=it))
    holes = closed & ~ndimage.binary_closing(occupied, iterations=1) & ~ndimage.binary_dilation(above, iterations=1)
    lab, k = ndimage.label(holes)
    if k == 0:
        return (*empty, plane)
    size = np.bincount(lab.ravel(), minlength=k + 1) * cell * cell
    ok = size <= max_hole_m2
    ok[0] = False
    fill = ndimage.binary_dilation(ok[lab], iterations=1) & ~above   # one cell of overlap with the mesh
    ci, cj = np.nonzero(fill)
    corners = np.stack([np.stack([ci, cj], 1), np.stack([ci + 1, cj], 1), np.stack([ci + 1, cj + 1], 1),
                        np.stack([ci, cj + 1], 1)], 1)                  # (cells, 4, 2), counter-clockwise
    key = corners[..., 0] * (n[1] + 2) + corners[..., 1]
    uk, inv = np.unique(key.ravel(), return_inverse=True)
    inv = inv.reshape(-1, 4)
    xy = lo + np.stack([uk // (n[1] + 2), uk % (n[1] + 2)], 1) * cell
    Vf = np.c_[xy, plane[0] * xy[:, 0] + plane[1] * xy[:, 1] + plane[2]]
    Ff = np.concatenate([inv[:, [0, 1, 2]], inv[:, [0, 2, 3]]])
    return Vf, Ff, plane


# ---------------------------------------------------------------------------------------------------- 5. photometry


def fisheye_project(cam, T_S_cam: np.ndarray, X: np.ndarray):
    """``(u, v, theta, dist, Xc)`` of scene points in a Kannala-Brandt fisheye at pose ``T_S_cam``."""
    Xc = (X - T_S_cam[:3, 3]) @ T_S_cam[:3, :3]
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    t2 = th * th
    k1, k2, k3, k4 = cam.k
    thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    with np.errstate(all="ignore"):
        s = np.where(r > 1e-12, thd / r, 0.0)
    return cam.fx * s * x + cam.cx, cam.fy * s * y + cam.cy, th, np.linalg.norm(Xc, axis=1), Xc


def zbuffer(cam, T_S_cam: np.ndarray, pts: np.ndarray, div: int) -> np.ndarray:
    """Nearest distance per pixel of a 1/div fisheye image from point splats, closed with a 3x3 minimum."""
    import cv2

    h, w = int(np.ceil(cam.height / div)), int(np.ceil(cam.width / div))
    u, v, th, d, _ = fisheye_project(cam, T_S_cam, pts)
    ok = (th < np.radians(100)) & (u >= 0) & (v >= 0) & (u < cam.width) & (v < cam.height)
    iu, iv = (u[ok] / div).astype(int), (v[ok] / div).astype(int)
    zb = np.full(h * w, np.inf, np.float32)
    dd = d[ok].astype(np.float32)
    order = np.argsort(-dd)                                          # far to near: the nearest is written last
    zb[(iv * w + iu)[order]] = dd[order]
    return cv2.erode(zb.reshape(h, w), np.ones((3, 3), np.uint8))


def visible(cam, view: View, zb: np.ndarray | None, X: np.ndarray, normals: np.ndarray, mask_full, p: MeshParams):
    """``(ok, u, v, theta, dist, cos)`` of points with normals in a view: in the image, inside the angle limit,
    facing it, not occluded (z-buffer; ``None``: not checked), not on the gripper mask (``None``: none)."""
    u, v, th, d, _ = fisheye_project(cam, view.T_S_cam, X)
    ray = view.T_S_cam[:3, 3] - X
    cos = np.sum(ray * normals, axis=1) / np.maximum(d, 1e-9)
    ok = (th < np.radians(p.max_theta_deg)) & (cos > p.min_cos) & (u >= 1) & (v >= 1) & \
         (u < cam.width - 2) & (v < cam.height - 2)
    iu = np.where(ok, u, 0).astype(int)
    iv = np.where(ok, v, 0).astype(int)
    div = p.zbuf_div
    if zb is not None:
        with np.errstate(over="ignore"):
            ok &= d <= zb[iv // div, iu // div].astype(np.float64) * 1.03 + 0.012
    if mask_full is not None:
        ok &= ~mask_full[iv, iu]
    return ok, u, v, th, d, cos


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    import cv2

    n = len(u)
    if n == 0:
        return np.zeros((0, img.shape[2]), np.float32)
    cols = 1024
    rows = int(np.ceil(n / cols))
    mu = np.zeros(rows * cols, np.float32)
    mv = np.zeros(rows * cols, np.float32)
    mu[:n], mv[:n] = u, v
    out = cv2.remap(img, mu.reshape(rows, cols), mv.reshape(rows, cols), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * cols, -1)[:n].astype(np.float32)


SHADING_GRID = (9, 7)                # lens-shading nodes across x and y of the fisheye image


@dataclass
class Photometry:
    log_gain: np.ndarray             # (n_views, 3)
    size: tuple[int, int]            # fisheye image (width, height)
    log_shading: np.ndarray          # (ny, nx, 3) on a regular grid over the image, bilinear in between
    stats: dict

    def _weights(self, u: np.ndarray, v: np.ndarray):
        ny, nx = self.log_shading.shape[:2]
        x = np.clip(np.asarray(u, float) / (self.size[0] - 1), 0, 1) * (nx - 1)
        y = np.clip(np.asarray(v, float) / (self.size[1] - 1), 0, 1) * (ny - 1)
        i, j = np.minimum(x.astype(int), nx - 2), np.minimum(y.astype(int), ny - 2)
        a, b = x - i, y - j
        idx = np.stack([j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1], 1)
        w = np.stack([(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b], 1)
        return idx, w

    def shading(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        idx, w = self._weights(u, v)
        return np.einsum("nk,nkc->nc", w, self.log_shading.reshape(-1, 3)[idx])

    def factor(self, view_i: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Multiplicative factor (N, 3) the camera applied at fisheye pixels (u, v): observed = factor x true."""
        return np.exp(self.log_gain[view_i] + self.shading(u, v))


def _fit_shading(ph: Photometry, u: np.ndarray, v: np.ndarray, r: np.ndarray, smooth: float = 0.05) -> np.ndarray:
    """Least squares bilinear grid through r (N, 3) at pixels (u, v), with a second-difference penalty (``smooth``
    x the mean diagonal); returns (ny, nx, 3)."""
    from scipy.sparse import csr_matrix

    ny, nx = ph.log_shading.shape[:2]
    nn = nx * ny
    idx, w = ph._weights(u, v)
    B = csr_matrix((w.ravel(), (np.repeat(np.arange(len(u)), 4), idx.ravel())), shape=(len(u), nn))
    A = (B.T @ B).toarray()
    rhs = B.T @ r
    D = []
    for j in range(ny):                                               # second differences along x and y
        for i in range(nx - 2):
            d = np.zeros(nn); d[[j * nx + i, j * nx + i + 1, j * nx + i + 2]] = (1, -2, 1); D.append(d)
    for i in range(nx):
        for j in range(ny - 2):
            d = np.zeros(nn); d[[j * nx + i, (j + 1) * nx + i, (j + 2) * nx + i]] = (1, -2, 1); D.append(d)
    D = np.array(D)
    A = A + smooth * np.mean(np.diag(A)) * (D.T @ D) + 1e-6 * np.mean(np.diag(A)) * np.eye(nn)
    return np.linalg.solve(A, rhs).reshape(ny, nx, 3)


def fit_photometry(cam, views: list[View], zbs: list, V: np.ndarray, Vn: np.ndarray, mask_full,
                   p: MeshParams, seed: int = 0) -> Photometry:
    """Per-view log gain and the lens-shading field per channel (module docstring, step 5): log colour =
    surface colour + gain(view) + shading(pixel), by alternating least squares, gross outliers dropped."""
    import cv2

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), min(p.photo_points, len(V)), replace=False)
    X, N = V[idx], Vn[idx]

    def one(i: int):
        ok, u, v, _, _, _ = visible(cam, views[i], zbs[i], X, N, mask_full, p)
        blur = cv2.GaussianBlur(views[i].raw, (0, 0), 2.0)
        c = _bilinear(blur, u[ok], v[ok])
        good = np.all((c > 6) & (c < 250), axis=1)
        k = np.flatnonzero(ok)[good]
        return np.full(len(k), i), k, u[k], v[k], c[good]

    vi, pi, uu, vv, L = [np.concatenate(a) for a in zip(*pmap(one, range(len(views))))]
    nv, npnt = len(views), len(X)
    L = np.log(L)
    keep = np.bincount(pi, minlength=npnt)[pi] >= 3                  # points seen at least 3 times
    n_obs = len(L)
    vi, pi, uu, vv, L = vi[keep], pi[keep], uu[keep], vv[keep], L[keep]
    g = np.zeros((nv, 3))
    ph = Photometry(g, (cam.width, cam.height), np.zeros((SHADING_GRID[1], SHADING_GRID[0], 3)), {})
    for it in range(p.photo_iters):
        sh = ph.shading(uu, vv)
        cp = np.maximum(np.bincount(pi, minlength=npnt), 1)[:, None]
        srf = np.stack([np.bincount(pi, L[:, c] - g[vi, c] - sh[:, c], npnt) for c in range(3)], 1) / cp
        R = L - srf[pi]
        cv = np.maximum(np.bincount(vi, minlength=nv), 1)[:, None]
        g = np.stack([np.bincount(vi, R[:, c] - sh[:, c], nv) for c in range(3)], 1) / cv
        g -= g.mean(axis=0)
        ph = Photometry(g, ph.size, _fit_shading(ph, uu, vv, R - g[vi]), {})
        if 2 <= it < p.photo_iters - 1:                              # drop gross outliers (misregistration)
            res = np.abs(R - g[vi] - ph.shading(uu, vv)).max(axis=1)
            ok = res < max(3 * np.median(res), 0.05)
            vi, pi, uu, vv, L = vi[ok], pi[ok], uu[ok], vv[ok], L[ok]
    res = L - srf[pi] - g[vi] - ph.shading(uu, vv)
    # the level is free (it trades with the surface colours): the image centre (within ~30 degrees) keeps its
    # colours
    ny, nx = ph.log_shading.shape[:2]
    gy, gx = np.mgrid[0:ny, 0:nx]
    px, py = gx / (nx - 1) * (cam.width - 1), gy / (ny - 1) * (cam.height - 1)
    centre = np.hypot(px - cam.cx, py - cam.cy) <= 0.55 * cam.fx
    ph.log_shading -= ph.log_shading[centre].mean(axis=0)
    named = {"centre": (cam.cx, cam.cy), "left": (0.1 * cam.width, cam.cy), "right": (0.9 * cam.width, cam.cy),
             "top": (cam.cx, 0.1 * cam.height), "bottom": (cam.cx, 0.9 * cam.height),
             "top_left": (0.15 * cam.width, 0.15 * cam.height), "top_right": (0.85 * cam.width, 0.15 * cam.height)}
    ph.stats = {"observations": int(n_obs), "used": int(len(L)), "points": int(npnt),
                "residual_log_mad": round(float(np.median(np.abs(res))), 4),
                "gain_spread_log_p5_p95": np.round(np.percentile(g, [5, 95], axis=0), 3).tolist(),
                "shading_rgb": {k: np.round(np.exp(ph.shading(np.array([x]), np.array([y]))[0]), 3).tolist()
                                for k, (x, y) in named.items()}}
    return ph


def tag_white_balance(cam, views: list[View], zbs: list, photo: Photometry, T_S_tag: np.ndarray, size_m: float,
                      mask_full, p: MeshParams) -> dict:
    """Turn the corrected colours so that the tag's white paper margin (a band ``size_m / 2 + 5..15 mm`` from
    its centre) is neutral grey: folds a per-channel gain into every view's ``log_gain``. Returns the numbers."""
    h = size_m / 2
    g = np.linspace(-h - 0.015, h + 0.015, 31)
    X = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    m = np.abs(X).max(axis=1)
    X = X[(m > h + 0.005) & (m < h + 0.015)]
    P = np.c_[X, np.zeros(len(X))] @ T_S_tag[:3, :3].T + T_S_tag[:3, 3]
    N = np.tile(T_S_tag[:3, 2], (len(P), 1))
    if N[0, 2] < 0:
        N = -N
    cols = []
    for i, view in enumerate(views):
        ok, u, v, _, _, _ = visible(cam, view, zbs[i], P, N, mask_full, p)
        if ok.any():
            cols.append(_bilinear(view.raw, u[ok], v[ok]) / photo.factor(i, u[ok], v[ok]))
    if not cols:
        return {}
    c = np.concatenate(cols)
    c = c[np.all(c < 250, axis=1)]
    med = np.median(c, axis=0)
    wb = med.mean() / med
    photo.log_gain -= np.log(wb)
    return {"tag_margin_samples": int(len(c)), "tag_margin_rgb_before": np.round(med, 1).tolist(),
            "gain_rgb": np.round(wb, 3).tolist()}


# ---------------------------------------------------------------------------------------------------- 6. views


def pmap(fn, items, workers: int = 8):
    """``map(fn, items)`` in threads (numpy releases the GIL), in order, a batch at a time."""
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    with ThreadPoolExecutor(workers) as ex:
        for k in range(0, len(items), 4 * workers):
            yield from ex.map(fn, items[k:k + 4 * workers])


def view_scores(cam, views: list[View], zbs: list | None, V: np.ndarray, F: np.ndarray, mask_full, p: MeshParams):
    """``(cand (F, n_cand) view index or -1, score (F, n_cand))``, best first (module docstring, step 6).
    ``zbs=None``: occlusion not checked."""
    c = V[F].mean(axis=1)
    n = face_normals(V, F)
    k = p.n_cand
    best_s = np.zeros((len(F), k), np.float32)
    best_v = np.full((len(F), k), -1, np.int32)
    tmax = np.radians(p.max_theta_deg)
    p_corner = replace(p, min_cos=-2.0)                             # corners: in the image, unmasked, unoccluded
    zero = np.zeros_like(V)

    def one(i: int) -> np.ndarray:
        zb = zbs[i] if zbs is not None else None
        ok, _, _, th, d, cos = visible(cam, views[i], zb, c, n, mask_full, p)
        okv, *_ = visible(cam, views[i], zb, V, zero, mask_full, p_corner)
        ok &= okv[F].all(axis=1)
        s = np.where(ok, cos / np.maximum(d, 0.05) * np.clip(1 - (th / tmax) ** 4, 0, 1) * views[i].sharp_w, 0)
        return s.astype(np.float32)

    for i, s in enumerate(pmap(one, range(len(views)))):
        # insert into the running top-k (replace the worst where better)
        worst = np.argmin(best_s, axis=1)
        better = s > best_s[np.arange(len(F)), worst]
        r = np.flatnonzero(better)
        best_s[r, worst[r]] = s[r]
        best_v[r, worst[r]] = i
    order = np.argsort(-best_s, axis=1)
    best_s = np.take_along_axis(best_s, order, 1)
    best_v = np.take_along_axis(best_v, order, 1)
    best_v[best_s <= 0] = -1
    return best_v, best_s


def face_neighbours(F: np.ndarray) -> np.ndarray:
    """(F, 3) the faces across each edge (-1: none)."""
    nf = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    fid = np.tile(np.arange(nf), 3)
    slot = np.repeat(np.arange(3), nf)
    key = e[:, 0] * (F.max() + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    out = np.full((nf, 3), -1, np.int64)
    same = np.flatnonzero(ks[1:] == ks[:-1])
    a, b = order[same], order[same + 1]
    out[fid[a], slot[a]] = fid[b]
    out[fid[b], slot[b]] = fid[a]
    return out


def choose_views(cand: np.ndarray, score: np.ndarray, nbr: np.ndarray, lam: float, iters: int,
                 seed: int = 0) -> np.ndarray:
    """(F,) the chosen view per face (-1: none): ICM over the candidates, score ratio + lam x neighbour agreement."""
    rng = np.random.default_rng(seed)
    nf, k = cand.shape
    lab = np.zeros(nf, np.int64)                                     # index into the candidates
    rel = score / np.maximum(score[:, :1], 1e-12)
    valid = cand >= 0
    for _ in range(iters):
        cur = cand[np.arange(nf), lab]
        nv = np.where(nbr >= 0, cur[np.maximum(nbr, 0)], -2)         # neighbours' views
        agree = (nv[:, None, :] == cand[:, :, None]).sum(axis=2) / 3.0
        energy = np.where(valid, rel + lam * agree, -np.inf)
        new = np.argmax(energy, axis=1)
        upd = rng.random(nf) < 0.5
        lab = np.where(upd, new, lab)
    out = cand[np.arange(nf), lab]
    return out


# ---------------------------------------------------------------------------------------------------- 7. texture


def atlas_layout(n_faces: int, cell_px: int, max_px: int) -> tuple[int, int, int, int]:
    """``(cell, cols, width, height)`` of the atlas: one cell per face, row-major."""
    cols = int(np.ceil(np.sqrt(max(n_faces, 1))))
    cell = cell_px or max(4, max_px // cols)
    cell = min(cell, max(4, max_px // cols))
    rows = int(np.ceil(n_faces / cols))
    return cell, cols, cols * cell, rows * cell


def face_uvs(n_faces: int, cell: int, cols: int, W: int, H: int) -> np.ndarray:
    """(F, 3, 2) the corners' texture coordinates (0..1, v down as in glTF): texel centres (0.5, 0.5),
    (cell - 0.5, 0.5), (0.5, cell - 0.5) of the face's cell."""
    f = np.arange(n_faces)
    x0, y0 = (f % cols) * cell, (f // cols) * cell
    corner = np.array([[0.5, 0.5], [cell - 0.5, 0.5], [0.5, cell - 0.5]])
    uv = np.stack([x0, y0], 1)[:, None, :] + corner[None]
    return uv / np.array([W, H])


def seam_levels(cam, views: list[View], photo: Photometry | None, V: np.ndarray, F: np.ndarray,
                label: np.ndarray, blur_px: float = 1.5, clip: float = 1.5) -> np.ndarray:
    """(F, 3 corners, 3) colour factors that make neighbouring faces textured from different frames meet at the
    same colour: each corner's (blurred, corrected) colour in its face's frame is pulled to the mean over all the
    faces at that vertex (a simple per-vertex version of the local colour adjustment of Waechter et al. 2014)."""
    import cv2

    c = np.full((len(F), 3, 3), np.nan)
    for vi in np.unique(label[label >= 0]):
        fs = np.flatnonzero(label == vi)
        P = V[F[fs]].reshape(-1, 3)
        u, v, _, _, _ = fisheye_project(cam, views[vi].T_S_cam, P)
        col = _bilinear(cv2.GaussianBlur(views[vi].raw, (0, 0), blur_px), u, v)
        if photo is not None:
            col = col / photo.factor(vi, u, v)
        c[fs] = col.reshape(-1, 3, 3)
    ok = np.isfinite(c[..., 0]).ravel()
    vid = F.ravel()
    cc = np.maximum(c.reshape(-1, 3), 1.0)
    cnt = np.maximum(np.bincount(vid[ok], minlength=len(V)), 1)
    t = np.stack([np.bincount(vid[ok], np.log(cc[ok, k]), len(V)) for k in range(3)], 1) / cnt[:, None]
    r = np.exp(t[F] - np.log(np.maximum(c, 1.0)))                     # geometric mean / own
    r[~np.isfinite(r)] = 1.0
    return np.clip(r, 1 / clip, clip)


def bake_atlas(cam, views: list[View], photo: Photometry | None, V: np.ndarray, F: np.ndarray,
               label: np.ndarray, cell: int, cols: int, W: int, H: int, fill=(128, 128, 128),
               levels: np.ndarray | None = None) -> np.ndarray:
    """The texture atlas (H, W, 3) uint8 (module docstring, step 7); ``levels``: :func:`seam_levels`."""
    atlas = np.zeros((H, W, 3), np.uint8)
    atlas[:] = fill
    j, i = np.mgrid[0:cell, 0:cell]
    b1 = ((i + 0.5) - 0.5) / (cell - 1.0)                            # barycentrics of each texel (extrapolated)
    b2 = ((j + 0.5) - 0.5) / (cell - 1.0)
    b0 = 1 - b1 - b2
    bary = np.stack([b0.ravel(), b1.ravel(), b2.ravel()], 1)         # (cell^2, 3)
    bin_ = np.clip(bary, 0, None)
    bin_ /= bin_.sum(axis=1, keepdims=True)                          # (inside the triangle, for the levels)
    for vi in np.unique(label[label >= 0]):
        fs = np.flatnonzero(label == vi)
        for chunk in np.array_split(fs, max(1, len(fs) // 20000)):
            P = np.einsum("tk,fkd->ftd", bary, V[F[chunk]]).reshape(-1, 3)
            u, v, _, _, _ = fisheye_project(cam, views[vi].T_S_cam, P)
            c = _bilinear(views[vi].raw, u, v)
            if photo is not None:
                c = c / photo.factor(vi, u, v)
            if levels is not None:
                c = c * np.einsum("tk,fkc->ftc", bin_, levels[chunk]).reshape(-1, 3)
            c = np.clip(np.rint(c), 0, 255).astype(np.uint8).reshape(len(chunk), cell, cell, 3)
            x0, y0 = (chunk % cols) * cell, (chunk // cols) * cell
            yy = y0[:, None, None] + j[None]
            xx = x0[:, None, None] + i[None]
            atlas[yy, xx] = c
    return atlas


def vertex_colours(cam, views: list[View], photo: Photometry | None, V: np.ndarray, F: np.ndarray,
                   label: np.ndarray) -> np.ndarray:
    """(V, 3) uint8: each vertex's colour in the view of one of its faces (grey when none)."""
    vl = np.full(len(V), -1, np.int64)
    for j in range(3):
        vl[F[:, j]] = np.maximum(vl[F[:, j]], label)
    out = np.full((len(V), 3), 128, np.uint8)
    for vi in np.unique(vl[vl >= 0]):
        k = np.flatnonzero(vl == vi)
        u, v, _, _, _ = fisheye_project(cam, views[vi].T_S_cam, V[k])
        c = _bilinear(views[vi].raw, u, v)
        if photo is not None:
            c = c / photo.factor(vi, u, v)
        out[k] = np.clip(np.rint(c), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------------------------------- output


def write_glb(path: Path, V: np.ndarray, F: np.ndarray, uv: np.ndarray, jpeg: bytes) -> None:
    """A binary glTF 2.0: one unlit textured mesh, corners unshared (each face its own UVs). The scene frame
    (z up) is turned into glTF's y up: (x, y, z) -> (x, z, -y)."""
    P = V[F].reshape(-1, 3)
    P = np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1).astype("<f4")
    T = uv.reshape(-1, 2).astype("<f4")
    parts, views, off = [], [], 0
    for blob, target in ((P.tobytes(), 34962), (T.tobytes(), 34962), (jpeg, None)):
        pad = (-len(blob)) % 4
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(blob), **({"target": target} if target else {})})
        parts.append(blob + b"\0" * pad)
        off += len(blob) + pad
    gltf = {
        "asset": {"version": "2.0", "generator": "umiproc mesh_3d (experimental)"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "mapping_clip_mesh"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                                                "roughnessFactor": 1.0},
                       "extensions": {"KHR_materials_unlit": {}}, "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 2, "mimeType": "image/jpeg"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(P), "type": "VEC3",
             "min": P.min(axis=0).tolist(), "max": P.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(T), "type": "VEC2"}],
        "bufferViews": views, "buffers": [{"byteLength": off}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binb = b"".join(parts)
    with open(path, "wb") as fh:
        fh.write(b"glTF" + np.array([2, 12 + 8 + len(js) + 8 + len(binb)], "<u4").tobytes())
        fh.write(np.array([len(js), 0x4E4F534A], "<u4").tobytes() + js)
        fh.write(np.array([len(binb), 0x004E4942], "<u4").tobytes() + binb)


def _jpeg(rgb: np.ndarray, quality: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted mean of the faces' normals."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    Vn = np.zeros_like(V)
    for j in range(3):
        np.add.at(Vn, F[:, j], fn)
    return Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-12)


def _keep_faces(V: np.ndarray, F: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = F[keep]
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    return V[used], (np.cumsum(used) - 1)[F]


def build(folder: str, root=None, p: MeshParams | None = None, dp: D.Params | None = None,
          out_dir: Path | None = None, depth_cache: Path | None = None) -> dict:
    """The whole trial for a task folder; returns the summary written to mesh_3d.json."""
    from umiproc.viz.dense_3d_viewer import scene_transform
    from umiproc.viz.mesh_3d_viewer import write_page

    p, dp = p or MeshParams(), dp or D.Params()
    clock = D._Clock()
    inp = D.load_inputs(folder, root)
    clock.lap("load")
    frames, filtered, fcount = load_depths(inp, dp, depth_cache, clock)
    T_SW = scene_transform(inp.export.meta.get("up_W"), inp.tag)
    T_S = [T_SW @ f.T_W_cam for f in frames]
    K = dp.K
    lo, hi = crop_box(T_S, filtered, K, p)
    tsdf = Tsdf(lo, hi, p.voxel_m, p.trunc_vox * p.voxel_m)
    for T, z in zip(T_S, filtered):
        if z is not None:
            tsdf.integrate(z, K, T, p.depth_max_m, p.depth_weight_ref_m)
    n_filled = filter_tsdf(tsdf, p.smooth_vox, p.w_min, smooth=True, fill=p.fill_holes)
    clock.lap("TSDF")
    V, F = extract_mesh(tsdf, p.w_min)
    n_raw = len(F)
    V, F, pieces = clean_mesh(V, F, p.min_piece_frac, p.min_piece_tris)
    clock.lap("marching cubes + clean")
    if not len(F):
        raise ValueError("no surface: the TSDF is empty")
    Vn = vertex_normals(V, F)
    is_fill = np.zeros(len(F), bool)
    plane = None
    if p.plane_fill_max_m2 > 0:
        Vf, Ff, plane = plane_fill(V, F, Vn, 2 * p.voxel_m, p.plane_fill_max_m2)
        F = np.concatenate([F, Ff + len(V)])
        V = np.concatenate([V, Vf])
        is_fill = np.r_[is_fill, np.ones(len(Ff), bool)]
        Vn = vertex_normals(V, F)
        clock.lap("table hole fill")
    med = np.median([f.sharpness for f in frames])
    views = [View(fed=f.fed, T_S_cam=T, raw=f.raw, keyframe=f.keyframe,
                  sharp_w=float(np.clip(np.sqrt(f.sharpness / med), 0.5, 1.3))) for f, T in zip(frames, T_S)]
    mask_full = D.grown_mask(inp.mask_full, (inp.cam.height, inp.cam.width), p.mask_grow_px)
    splats = np.concatenate([V, V[F].mean(axis=1)])
    zbs = list(pmap(lambda v: zbuffer(inp.cam, v.T_S_cam, splats, p.zbuf_div), views))
    clock.lap("z-buffers")
    photo = fit_photometry(inp.cam, views, zbs, V, Vn, mask_full, p)
    if inp.tag and p.tag_white_balance:
        photo.stats["white_balance"] = tag_white_balance(
            inp.cam, views, zbs, photo, T_SW @ np.asarray(inp.tag["tx_slam_tag"], float),
            float(inp.tag.get("tag", {}).get("size_m", 0.16)), mask_full, p)
    clock.lap("photometry")
    cand, score = view_scores(inp.cam, views, zbs, V, F, mask_full, p)
    label = choose_views(cand, score, face_neighbours(F), p.smooth_lambda, p.smooth_iters)
    # faces no frame sees unoccluded: the best frame that faces them anyway (occlusion, mask ignored); faces that
    # face away from every frame are noise (or the back of something never seen) and are dropped
    # (a table fill face that no frame sees unoccluded is dropped: it would be made up)
    none = np.flatnonzero((label < 0) & ~is_fill)
    rc, _ = view_scores(inp.cam, views, None, V, F[none], None, p)
    label[none] = rc[:, 0]
    n_relaxed = int((rc[:, 0] >= 0).sum())
    keep = label >= 0
    n_dropped = int((~keep & ~is_fill).sum())
    n_fill = int((keep & is_fill).sum())
    V, F, label, cand = (*_keep_faces(V, F, keep), label[keep], cand[keep])
    clock.lap("view choice")
    cell, cols, W, H = atlas_layout(len(F), p.cell_px, p.max_atlas_px)
    levels = seam_levels(inp.cam, views, photo, V, F, label) if p.seam_leveling else None
    atlas = bake_atlas(inp.cam, views, photo, V, F, label, cell, cols, W, H, levels=levels)
    vcol = vertex_colours(inp.cam, views, photo, V, F, label)
    clock.lap("texture")
    out_dir = Path(out_dir) if out_dir is not None else inp.analysis
    out_dir.mkdir(parents=True, exist_ok=True)
    glb, html, js = (out_dir / f"{OUT_STEM}{ext}" for ext in (".glb", ".html", ".json"))
    jpeg = _jpeg(atlas, p.jpeg_quality)
    uv = face_uvs(len(F), cell, cols, W, H)
    write_glb(glb, V, F, uv, jpeg)
    labs = label[label >= 0]
    summary = {
        "format": "umiproc.mesh_3d.v0 (experimental)", "folder": inp.folder, "map": inp.map_dir.name,
        "world": "the scene frame: W turned gravity-up (+z), the tag's centre at 0 (T_scene_W); the GLB is the same "
                 "turned to glTF's y up",
        "params": asdict(p), "dense_params": asdict(dp), "frames": fcount,
        "box": {"lo": np.round(lo, 3).tolist(), "hi": np.round(hi, 3).tolist(), "voxels": list(tsdf.shape),
                "voxels_seen": int((tsdf.weight >= p.w_min).sum())},
        "mesh": {"faces_marching_cubes": int(n_raw), "faces": int(len(F)), "vertices": int(len(V)), **pieces,
                 "voxels_filled": n_filled, "faces_back_facing_dropped": n_dropped, "faces_table_fill": n_fill,
                 "table_plane_z_axb_c": np.round(plane, 5).tolist() if plane is not None else None,
                 "area_m2": round(float(np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]),
                                                       axis=1).sum() / 2), 3)},
        "texture": {"atlas_px": [W, H], "cell_px": cell, "jpeg_mb": round(len(jpeg) / 1e6, 2),
                    "faces_textured_unoccluded": int(len(labs)) - n_relaxed, "faces_textured_relaxed": n_relaxed,
                    "views_used": int(len(np.unique(labs))),
                    "faces_on_best_candidate": round(float(np.mean(label[label >= 0] == cand[label >= 0, 0])), 3)},
        "photometry": photo.stats,
        "T_scene_W": np.round(T_SW, 6).tolist(),
        "timings_s": clock.steps,
    }
    write_page(html, inp, views, {f.fed: f.rgb for f in frames}, T_SW, V, F, vcol, jpeg, cell, cols, W, H,
               summary, dp)
    clock.lap("write")
    summary["timings_s"] = clock.steps
    js.write_text(json.dumps(summary, indent=1) + "\n")
    summary["outputs"] = [str(glb), str(html), str(js)]
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m umiproc.viz.mesh_3d", description=__doc__.splitlines()[0])
    ap.add_argument("folder", metavar="FOLDER", help="a task folder's name (under <root>/tasks/)")
    ap.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    ap.add_argument("--out-dir", default=None, metavar="DIR", help="write here instead of <folder>/analysis/")
    ap.add_argument("--voxel", type=float, default=MeshParams.voxel_m, help="TSDF voxel size in metres")
    ap.add_argument("--cell", type=int, default=MeshParams.cell_px, help="atlas pixels per triangle side")
    ap.add_argument("--depth-cache", default=None, metavar="FILE",
                    help="an .npz to read the depth maps from (or write them to): skips the stereo on a re-run")
    args = ap.parse_args(argv)
    from umiproc.viz.slam_3d import Slam3dError
    from umiproc.viz.slam_run import PlotRunError

    try:
        s = build(args.folder, args.root, MeshParams(voxel_m=args.voxel, cell_px=args.cell), None, args.out_dir,
                  args.depth_cache)
    except (PlotRunError, Slam3dError, OSError, ValueError) as e:
        print(f"mesh_3d: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: s[k] for k in ("box", "mesh", "texture", "photometry")}, indent=1))
    for o in s["outputs"]:
        print(f"wrote {o} ({Path(o).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
