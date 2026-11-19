"""EXPERIMENTAL TRIAL (may be removed if it does not work out): a dense coloured point cloud of a map's mapping clip.

What it does: every keyframe and every ``--step``-th tracked frame of the mapping clip is paired with 1-2 nearby
frames, each pair is stereo-matched (OpenCV StereoSGBM) on images rectified from the fisheye using the SLAM poses,
and a pixel's 3-D point is kept only when other frames' depth maps agree with it; the result is voxel-thinned,
cleaned and written as ``analysis/dense_3d.{html,ply,json}`` (a three.js viewer, the cloud, the parameters).

``python -m umiproc.viz.dense_3d FOLDER [--root DIR]`` (not a ``umiproc`` command yet). ``FOLDER``: a task folder's
name (:mod:`umiproc.viz.target`); its map's ``mapping`` run, its raw recording (the ultrawide), the lens
(``orbslam3_settings.yaml``), the gripper mask, the map export (``map_points*.csv``, for depth ranges and ``up_W``)
and ``tx_slam_tag.json`` (the table tag, for the viewer) are read; nothing but ``analysis/dense_3d.*`` is written.

The steps (each one function; the defaults are in :class:`Params`):

1. **Frames** (:func:`select_frames`, :func:`undistort_frames`): keyframes (their pose from the map export) plus
   every ``step``-th tracked frame (the trajectory's pose). Each is undistorted to a pinhole view (``width`` x
   ``height``, ``fov_deg`` horizontal); blurry frames (variance of the Laplacian below ``blur_ratio`` x the median)
   are dropped. The gripper mask is grown by ``mask_grow_px`` and undistorted the same way.
2. **Pairs** (:func:`choose_pairs`): per frame the partners whose baseline is ``baseline_min..baseline_max`` of the
   frame's median sparse-point depth, whose optical axis is within ``max_axis_deg``, and whose baseline is mostly
   sideways (at least ``min_side_deg`` from the optical axis: no forward-motion pairs); the best is the closest to
   ``baseline_target``, the second one a different baseline direction (at least 30 degrees off).
3. **Depth** (:func:`pair_depth`): both frames are rectified straight from the fisheye (a common orientation whose x
   axis is the baseline), StereoSGBM with its own left-right check (``disp12MaxDiff``), then masked (gripper in
   either image, flat texture, disparity below ``min_disp_px``) and resampled into the frame's pinhole view as a
   depth map. A frame's pairs are merged (:func:`merge_depths`: both agree within ``agree_rel`` or only one).
4. **Fusion** (:func:`fuse`): every ``pixel_stride``-th pixel's point is projected into the ``n_check`` nearest
   frames' depth maps; it is kept when at least ``min_agree`` of them agree (within ``agree_rel``); its colour is the
   mean over the agreeing views. Then a ``voxel_m`` grid (mean position and colour per voxel, voxels with fewer than
   ``min_voxel_samples`` samples dropped) and a statistical outlier filter (:func:`drop_outliers`).

All in W (the SLAM world; metres when the map's check M4 passed). The PLY stays in W; the viewer is turned so that
gravity is up, with the tag at the origin. Runs on the CPU (OpenCV, numpy, scipy).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Params", "Frame", "Inputs", "load_inputs", "select_frames", "undistort_frames", "choose_pairs",
           "pair_depth", "merge_depths", "fuse", "voxel_thin", "drop_outliers", "write_ply", "build", "main"]

OUT_STEM = "dense_3d"


@dataclass
class Params:
    step: int = 5                    # every step-th tracked frame (plus all keyframes)
    width: int = 640                 # pinhole / rectified view size
    height: int = 480
    fov_deg: float = 110.0           # horizontal field of view of the pinhole views
    blur_ratio: float = 0.5          # drop frames whose Laplacian variance < this x the median
    mask_grow_px: int = 12           # grow the gripper mask (full-size pixels)
    baseline_min: float = 0.04       # x the frame's median sparse depth
    baseline_max: float = 0.18
    baseline_target: float = 0.10
    max_axis_deg: float = 25.0       # optical axes at most this far apart
    min_side_deg: float = 50.0       # the baseline at least this far from the optical axis
    pairs_per_frame: int = 2
    block: int = 5                   # SGBM block size
    uniqueness: int = 10
    min_disp_px: float = 6.0         # smaller disparities are too far (depth error > ~4 %)
    min_texture: float = 2.0         # local std of the grey image (0-255) below this: no depth
    depth_min_m: float = 0.1
    depth_max_m: float = 3.0
    agree_rel: float = 0.02          # depths agree within this fraction
    pixel_stride: int = 1
    n_check: int = 8
    min_agree: int = 2
    voxel_m: float = 0.003
    min_voxel_samples: int = 2
    outlier_k: int = 12
    outlier_std: float = 2.0

    @property
    def K(self) -> np.ndarray:
        f = (self.width / 2) / np.tan(np.radians(self.fov_deg) / 2)
        return np.array([[f, 0, (self.width - 1) / 2], [0, f, (self.height - 1) / 2], [0, 0, 1]])


@dataclass
class Frame:
    fed: int                         # fed frame index (trajectory row)
    T_W_cam: np.ndarray              # 4x4
    keyframe: bool
    raw: np.ndarray | None = None    # full-size fisheye RGB
    rgb: np.ndarray | None = None    # pinhole view RGB
    valid: np.ndarray | None = None  # pinhole view: not masked
    sharpness: float = 0.0
    median_depth: float = 0.0
    depth: np.ndarray | None = None  # pinhole depth map (z, NaN = none)


@dataclass
class Inputs:
    folder: str
    analysis: Path
    map_dir: Path
    traj: object
    i0: int
    video: Path
    cam: object
    export: object
    mask_full: np.ndarray            # (H, W) bool, True = gripper
    tag: dict | None
    timings: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------- inputs


def load_inputs(folder: str, root=None) -> Inputs:
    """The task folder's map: trajectory (frame times from the recording), lens, mask, map export, tag."""
    import cv2

    from umiproc.slam.trajectory_csv import read_fork_csv
    from umiproc.viz.slam_3d import read_map_export
    from umiproc.viz.slam_3d_images import read_camera
    from umiproc.viz.slam_run import count_rows, read_t0_ns, times_from_frames_csv
    from umiproc.viz.target import find_recording, resolve, ultrawide_video

    t = resolve(folder, "mapping", root=root)
    csv_path = t.run_dir / "trajectory.csv"
    video, frames_csv = ultrawide_video(find_recording(t))
    times, i0 = times_from_frames_csv(frames_csv, read_t0_ns(csv_path), count_rows(csv_path))
    traj = read_fork_csv(csv_path, times)
    mask = cv2.imread(str(t.map_dir / "slam_mask.png"), cv2.IMREAD_GRAYSCALE)
    tag_path = t.map_dir / "tx_slam_tag.json"
    return Inputs(folder=t.folder or folder, analysis=t.analysis, map_dir=t.map_dir, traj=traj, i0=i0, video=video,
                  cam=read_camera(t.map_dir / "orbslam3_settings.yaml"),
                  export=read_map_export(t.run_dir / "map_points.csv"),
                  mask_full=(mask > 0) if mask is not None else None,
                  tag=json.loads(tag_path.read_text()) if tag_path.is_file() else None)


# ---------------------------------------------------------------------------------------------------- 1. frames


def select_frames(inp: Inputs, step: int) -> list[Frame]:
    """Keyframes (map export pose) and every ``step``-th tracked frame (trajectory pose), by fed index."""
    traj, ex = inp.traj, inp.export
    out: dict[int, Frame] = {}
    for j, k in enumerate(ex.keyframe_frames):
        if 0 <= k < traj.n_frames:
            out[int(k)] = Frame(fed=int(k), T_W_cam=ex.T_W_kf[j], keyframe=True)
    for k in np.flatnonzero(traj.tracked)[::step]:
        out.setdefault(int(k), Frame(fed=int(k), T_W_cam=traj.T_W_cam[k], keyframe=False))
    return [out[k] for k in sorted(out)]


def fisheye_maps(cam, K: np.ndarray, size: tuple[int, int], R: np.ndarray = np.eye(3)):
    """cv2.remap maps from the fisheye image to a pinhole view (K, size) turned by R (camera -> view)."""
    import cv2

    Kf = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
    return cv2.fisheye.initUndistortRectifyMap(Kf, np.array(cam.k, np.float64), R, K, size, cv2.CV_32FC1)


def grown_mask(mask_full: np.ndarray | None, shape: tuple[int, int], grow_px: int) -> np.ndarray:
    """The gripper mask (full size, True = masked) grown by ``grow_px``; all False when there is none."""
    import cv2

    if mask_full is None:
        return np.zeros(shape, bool)
    m = mask_full.astype(np.uint8)
    if grow_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)))
    return m > 0


def undistort_frames(inp: Inputs, frames: list[Frame], p: Params) -> list[Frame]:
    """Decode the frames, make their pinhole views, drop the blurry ones."""
    import cv2

    from umiproc.viz.slam_3d_images import read_keyframe_images

    imgs = read_keyframe_images(inp.video, {f.fed: inp.i0 + f.fed for f in frames}, 0)
    mx, my = fisheye_maps(inp.cam, p.K, (p.width, p.height))
    mask = grown_mask(inp.mask_full, (inp.cam.height, inp.cam.width), p.mask_grow_px)
    valid = cv2.remap((~mask).astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
    kept = []
    for f in frames:
        if f.fed not in imgs:
            continue
        f.raw = imgs[f.fed]
        f.rgb = cv2.remap(f.raw, mx, my, cv2.INTER_LINEAR)
        f.valid = valid
        grey = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY)
        f.sharpness = float(cv2.Laplacian(grey, cv2.CV_64F)[valid].var())
        kept.append(f)
    med = np.median([f.sharpness for f in kept])
    return [f for f in kept if f.sharpness >= p.blur_ratio * med]


# ---------------------------------------------------------------------------------------------------- 2. pairs


def median_depths(frames: list[Frame], pts_W: np.ndarray, half_fov_deg: float) -> None:
    """Each frame's median depth of the sparse points inside its view."""
    for f in frames:
        R, t = f.T_W_cam[:3, :3], f.T_W_cam[:3, 3]
        pc = (pts_W - t) @ R
        ang = np.degrees(np.arctan2(np.hypot(pc[:, 0], pc[:, 1]), pc[:, 2]))
        vis = (pc[:, 2] > 0.05) & (ang < half_fov_deg)
        f.median_depth = float(np.median(pc[vis, 2])) if vis.sum() >= 20 else 0.5


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def choose_pairs(frames: list[Frame], p: Params) -> list[tuple[int, int]]:
    """``(i, j)`` index pairs: frame i's partners j (module docstring, step 2)."""
    C = np.array([f.T_W_cam[:3, 3] for f in frames])
    Z = np.array([f.T_W_cam[:3, 2] for f in frames])          # optical axes in W
    pairs = []
    for i, f in enumerate(frames):
        d = C - C[i]
        b = np.linalg.norm(d, axis=1)
        ratio = b / f.median_depth
        axis_deg = np.degrees(np.arccos(np.clip(Z @ Z[i], -1, 1)))
        with np.errstate(invalid="ignore", divide="ignore"):
            side_deg = np.degrees(np.arccos(np.clip(np.abs(d @ Z[i]) / b, 0, 1)))
        ok = (ratio >= p.baseline_min) & (ratio <= p.baseline_max) & (axis_deg < p.max_axis_deg) & \
             (side_deg >= p.min_side_deg)
        ok[i] = False
        cand = np.flatnonzero(ok)
        cand = cand[np.argsort(np.abs(ratio[cand] - p.baseline_target) + 0.002 * axis_deg[cand])]
        chosen: list[int] = []
        for j in cand:
            if len(chosen) >= p.pairs_per_frame:
                break
            if all(abs(float(_unit(d[j]) @ _unit(d[c]))) < np.cos(np.radians(30)) for c in chosen):
                chosen.append(int(j))
        pairs += [(i, j) for j in chosen]
    return pairs


# ---------------------------------------------------------------------------------------------------- 3. depth


def rect_rotation(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """W -> rectified rotation (rows = axes): x along the baseline a -> b, z the mean optical axis made orthogonal."""
    x = _unit(Tb[:3, 3] - Ta[:3, 3])
    z = Ta[:3, 2] + Tb[:3, 2]
    z = _unit(z - (z @ x) * x)
    return np.stack([x, np.cross(z, x), z])


def sgbm(p: Params, num_disp: int, min_disp: int):
    import cv2

    b = p.block
    return cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=b, P1=8 * b * b,
                                 P2=32 * b * b, disp12MaxDiff=1, uniquenessRatio=p.uniqueness,
                                 speckleWindowSize=100, speckleRange=2, preFilterCap=31,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def local_std(grey: np.ndarray, k: int = 7) -> np.ndarray:
    import cv2

    g = grey.astype(np.float32)
    m = cv2.blur(g, (k, k))
    return np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - m * m, 0))


def rectify(fr: Frame, R: np.ndarray, inp: Inputs, p: Params, mask_full: np.ndarray):
    """``(grey, masked)`` of frame fr's rectified view (camera -> rectified rotation R), straight from the fisheye;
    outside the fisheye counts as masked."""
    import cv2

    mx, my = fisheye_maps(inp.cam, p.K, (p.width, p.height), R)
    g = cv2.cvtColor(cv2.remap(fr.raw, mx, my, cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY)
    m = cv2.remap(mask_full.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=1) > 0
    return g, m


def pair_depth(a: Frame, b: Frame, inp: Inputs, p: Params, mask_full: np.ndarray) -> np.ndarray:
    """Frame a's depth map (pinhole view, z in its camera, NaN = none) from the pair (a, b)."""
    import cv2

    K, size = p.K, (p.width, p.height)
    f = K[0, 0]
    R_rW = rect_rotation(a.T_W_cam, b.T_W_cam)
    Ra, Rb = R_rW @ a.T_W_cam[:3, :3], R_rW @ b.T_W_cam[:3, :3]        # camera -> rectified
    B = float(np.linalg.norm(b.T_W_cam[:3, 3] - a.T_W_cam[:3, 3]))
    (ga, ma), (gb, mb) = rectify(a, Ra, inp, p, mask_full), rectify(b, Rb, inp, p, mask_full)
    d_max = f * B / max(p.depth_min_m, 0.4 * a.median_depth)
    d_min = max(p.min_disp_px, f * B / p.depth_max_m)
    min_disp = int(np.floor(d_min))
    num_disp = int(min(256, max(16, np.ceil((d_max - min_disp) / 16) * 16)))
    disp = sgbm(p, num_disp, min_disp).compute(ga, gb).astype(np.float32) / 16.0
    bad = (disp < d_min) | ma | (local_std(ga) < p.min_texture)
    h, w = disp.shape
    vv, uu = np.mgrid[0:h, 0:w]
    ur = np.clip(np.rint(uu - disp), 0, w - 1).astype(int)
    bad |= mb[vv, ur]                                                   # the match lands on the gripper
    disp[bad] = np.nan
    Zr = f * B / disp                                                   # depth in the rectified frame
    # into a's pinhole view: pixel x (z = 1 ray d) sits at rectified pixel K Ra d; its z_cam = Z_r / (Ra d).z
    H = K @ Ra @ np.linalg.inv(K)
    Zw = cv2.warpPerspective(Zr, H, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    rays = np.linalg.inv(K) @ np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)])
    rz = (Ra @ rays)[2].reshape(h, w)
    z = Zw / rz
    z[(z < p.depth_min_m) | (z > p.depth_max_m) | ~a.valid] = np.nan
    return z.astype(np.float32)


def merge_depths(depths: list[np.ndarray], agree_rel: float) -> np.ndarray:
    """One depth map from a frame's pairs: the mean where they agree, the one value where only one has depth,
    nothing where they disagree."""
    if len(depths) == 1:
        return depths[0]
    D = np.stack(depths)
    n = np.isfinite(D).sum(axis=0)
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN pixels: no depth in any pair
        lo, hi, mean = np.nanmin(D, axis=0), np.nanmax(D, axis=0), np.nanmean(D, axis=0)
    out = np.where((n == 1) | ((n >= 2) & (hi - lo <= agree_rel * mean)), mean, np.nan)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------------- 4. fusion


def frame_points(f: Frame, K: np.ndarray, stride: int):
    """``(X_W (N, 3), rgb (N, 3), pixel (N, 2))`` of frame f's depth pixels on a ``stride`` grid."""
    z = f.depth[::stride, ::stride]
    vv, uu = np.mgrid[0:f.depth.shape[0]:stride, 0:f.depth.shape[1]:stride]
    ok = np.isfinite(z)
    u, v, z = uu[ok], vv[ok], z[ok]
    Ki = np.linalg.inv(K)
    pc = (Ki @ np.stack([u, v, np.ones_like(u)])) * z
    X = pc.T @ f.T_W_cam[:3, :3].T + f.T_W_cam[:3, 3]
    return X, f.rgb[v, u].astype(np.float32), np.stack([u, v], 1)


def nearest_frames(frames: list[Frame], i: int, n: int, max_axis_deg: float = 35.0) -> list[int]:
    """The n frames nearest to frame i (centre distance plus a view-angle term) that look the same way."""
    C = np.array([f.T_W_cam[:3, 3] for f in frames])
    Z = np.array([f.T_W_cam[:3, 2] for f in frames])
    ang = np.degrees(np.arccos(np.clip(Z @ Z[i], -1, 1)))
    cost = np.linalg.norm(C - C[i], axis=1) / frames[i].median_depth + ang / 60.0
    cost[(ang > max_axis_deg) | (np.arange(len(frames)) == i)] = np.inf
    cost[[f.depth is None for f in frames]] = np.inf
    order = np.argsort(cost)
    return [int(j) for j in order[:n] if np.isfinite(cost[j])]


def check_in(f: Frame, X: np.ndarray, K: np.ndarray, agree_rel: float):
    """``(agree (N,), rgb (N, 3))``: X's depth in frame f matches f's depth map; f's colour there."""
    R, t = f.T_W_cam[:3, :3], f.T_W_cam[:3, 3]
    pc = (X - t) @ R
    z = pc[:, 2]
    h, w = f.depth.shape
    with np.errstate(all="ignore"):
        u = np.rint(K[0, 0] * pc[:, 0] / z + K[0, 2])
        v = np.rint(K[1, 1] * pc[:, 1] / z + K[1, 2])
    inside = (z > 0.05) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
    ui, vi = np.where(inside, u, 0).astype(int), np.where(inside, v, 0).astype(int)
    dz = f.depth[vi, ui]
    agree = inside & np.isfinite(dz) & (np.abs(dz - z) <= agree_rel * z)
    return agree, f.rgb[vi, ui].astype(np.float32)


def fuse(frames: list[Frame], p: Params) -> tuple[np.ndarray, np.ndarray, dict]:
    """Points kept by the multi-view agreement test, with their mean colour over the agreeing views."""
    K = p.K
    xs, cs = [], []
    n_raw = 0
    for i, f in enumerate(frames):
        if f.depth is None:
            continue
        X, rgb, _ = frame_points(f, K, p.pixel_stride)
        n_raw += len(X)
        votes = np.zeros(len(X), int)
        csum = rgb.copy()
        for j in nearest_frames(frames, i, p.n_check):
            agree, c = check_in(frames[j], X, K, p.agree_rel)
            votes += agree
            csum[agree] += c[agree]
        keep = votes >= p.min_agree
        xs.append(X[keep])
        cs.append(csum[keep] / (votes[keep, None] + 1))
    X = np.concatenate(xs) if xs else np.zeros((0, 3))
    C = np.concatenate(cs) if cs else np.zeros((0, 3))
    return X, C, {"pixels_with_depth": int(n_raw), "agreed": int(len(X))}


def voxel_thin(X: np.ndarray, C: np.ndarray, voxel: float, min_samples: int):
    """Mean position and colour per voxel; voxels with fewer than ``min_samples`` points dropped."""
    key = np.floor(X / voxel).astype(np.int64)
    key -= key.min(axis=0)
    span = key.max(axis=0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    n = len(cnt)
    Xm = np.stack([np.bincount(inv, X[:, k], n) for k in range(3)], 1) / cnt[:, None]
    Cm = np.stack([np.bincount(inv, C[:, k], n) for k in range(3)], 1) / cnt[:, None]
    keep = cnt >= min_samples
    return Xm[keep], np.clip(np.rint(Cm[keep]), 0, 255).astype(np.uint8), cnt[keep]


def drop_outliers(X: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """Keep mask: the mean distance to the k nearest neighbours is within mean + std_ratio x std."""
    from scipy.spatial import cKDTree

    if len(X) <= k:
        return np.ones(len(X), bool)
    d, _ = cKDTree(X).query(X, k + 1, workers=-1)
    m = d[:, 1:].mean(axis=1)
    return m <= m.mean() + std_ratio * m.std()


def sparse_check(X: np.ndarray, sparse: np.ndarray, obs: np.ndarray, frames: list[Frame]) -> dict:
    """How far the sparse ORB points (seen by >= 3 keyframes, within 1.5 m of a camera) are from the dense cloud."""
    from scipy.spatial import cKDTree

    C = np.array([f.T_W_cam[:3, 3] for f in frames])
    near = cKDTree(C).query(sparse)[0] < 1.5
    s = sparse[(obs >= 3) & near]
    if not len(s) or not len(X):
        return {}
    d = cKDTree(X).query(s, workers=-1)[0]
    return {"sparse_points": int(len(s)), "median_mm": round(float(np.median(d)) * 1e3, 2),
            "p90_mm": round(float(np.percentile(d, 90)) * 1e3, 2),
            "within_1cm": round(float((d < 0.01).mean()), 3)}


# ---------------------------------------------------------------------------------------------------- output


def write_ply(path: Path, X: np.ndarray, C: np.ndarray) -> None:
    """Binary little-endian PLY: x, y, z (float32), red, green, blue (uint8)."""
    rec = np.zeros(len(X), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = X[:, 0], X[:, 1], X[:, 2]
    rec["r"], rec["g"], rec["b"] = C[:, 0], C[:, 1], C[:, 2]
    head = ("ply\nformat binary_little_endian 1.0\ncomment umiproc dense_3d (experimental), world W of the SLAM map\n"
            f"element vertex {len(X)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(rec.tobytes())


class _Clock:
    def __init__(self):
        self.t, self.steps = time.perf_counter(), {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.steps[name] = round(now - self.t, 2)
        self.t = now
        print(f"  {name}: {self.steps[name]:.1f} s", flush=True)


def build(folder: str, root=None, p: Params | None = None, out_dir: Path | None = None) -> dict:
    """The whole trial for a task folder; returns the summary written to dense_3d.json."""
    from umiproc.viz.dense_3d_viewer import write_page

    p = p or Params()
    clock = _Clock()
    inp = load_inputs(folder, root)
    clock.lap("load")
    frames = select_frames(inp, p.step)
    n_selected = len(frames)
    frames = undistort_frames(inp, frames, p)
    clock.lap("decode + undistort")
    median_depths(frames, inp.export.points, p.fov_deg / 2)
    pairs = choose_pairs(frames, p)
    clock.lap("pairs")
    mask_full = grown_mask(inp.mask_full, (inp.cam.height, inp.cam.width), p.mask_grow_px)
    by_frame: dict[int, list[np.ndarray]] = {}
    for i, j in pairs:
        by_frame.setdefault(i, []).append(pair_depth(frames[i], frames[j], inp, p, mask_full))
    for i, ds in by_frame.items():
        frames[i].depth = merge_depths(ds, p.agree_rel)
    clock.lap("stereo depth")
    X, C, fstats = fuse(frames, p)
    clock.lap("fusion")
    Xv, Cv, cnt = voxel_thin(X, C, p.voxel_m, p.min_voxel_samples)
    keep = drop_outliers(Xv, p.outlier_k, p.outlier_std)
    Xf, Cf = Xv[keep], Cv[keep]
    clock.lap("voxel + outliers")
    out_dir = Path(out_dir) if out_dir is not None else inp.analysis
    out_dir.mkdir(parents=True, exist_ok=True)
    ply, html, js = (out_dir / f"{OUT_STEM}{ext}" for ext in (".ply", ".html", ".json"))
    write_ply(ply, Xf, Cf)
    summary = {
        "format": "umiproc.dense_3d.v0 (experimental)", "folder": inp.folder, "map": inp.map_dir.name,
        "world": "W, the SLAM map's world (metres as far as check M4 holds); the viewer turns it gravity-up, tag at 0",
        "params": asdict(p), "K_pinhole": np.round(p.K, 3).tolist(),
        "frames": {"selected": n_selected, "after_blur_check": len(frames),
                   "keyframes": sum(f.keyframe for f in frames), "with_depth": len(by_frame)},
        "pairs": len(pairs),
        "pair_baseline_ratio_median": round(float(np.median(
            [np.linalg.norm(frames[i].T_W_cam[:3, 3] - frames[j].T_W_cam[:3, 3]) / frames[i].median_depth
             for i, j in pairs])), 3) if pairs else None,
        "points": {**fstats, "voxels": int(len(Xv)), "after_outliers": int(len(Xf))},
        "sparse_to_dense": sparse_check(Xf, inp.export.points, inp.export.observations, frames),
        "timings_s": clock.steps,
    }
    write_page(html, inp, frames, Xf, Cf, summary)
    clock.lap("write")
    summary["timings_s"] = clock.steps
    js.write_text(json.dumps(summary, indent=1) + "\n")
    summary["outputs"] = [str(ply), str(html), str(js)]
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m umiproc.viz.dense_3d", description=__doc__.splitlines()[0])
    ap.add_argument("folder", metavar="FOLDER", help="a task folder's name (under <root>/tasks/)")
    ap.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    ap.add_argument("--out-dir", default=None, metavar="DIR", help="write here instead of <folder>/analysis/")
    ap.add_argument("--step", type=int, default=Params.step, help="every N-th tracked frame (plus all keyframes)")
    ap.add_argument("--voxel", type=float, default=Params.voxel_m, help="voxel size in metres")
    args = ap.parse_args(argv)
    from umiproc.viz.slam_3d import Slam3dError
    from umiproc.viz.slam_run import PlotRunError

    try:
        s = build(args.folder, args.root, Params(step=args.step, voxel_m=args.voxel), args.out_dir)
    except (PlotRunError, Slam3dError, OSError, ValueError) as e:
        print(f"dense_3d: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: s[k] for k in ("frames", "pairs", "points", "sparse_to_dense")}, indent=1))
    for o in s["outputs"]:
        print(f"wrote {o} ({Path(o).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
