"""The keyframe images for ``umiproc slam-3d``: each map point's real colour and a thumbnail per keyframe.

**Camera.** The ultrawide's Kannala-Brandt model (``KannalaBrandt8`` = ``cv2.fisheye``) from the map's
``orbslam3_settings.yaml`` (``Camera1.fx fy cx cy k1..k4``, ``Camera.width/height``): a point ``(x, y, z)`` in the
camera frame (x right, y down, z forward) goes to ``theta = atan2(r, z)``, ``theta_d = theta (1 + k1 theta^2 + ... +
k4 theta^8)``, ``u = fx theta_d x / r + cx``, ``v = fy theta_d y / r + cy`` (:func:`project`).

**Colours** (:func:`sample_colours`, :func:`point_colours`): every point is projected into every keyframe
(``T_W_kf`` from the map export); a sample counts when the point is in front of the camera, inside the image (a
``EDGE_PX`` margin) and not on the gripper mask (``slam_mask.png``, 255 = masked). The colour is the per-channel
median of the samples from the ``n_views`` nearest keyframes that see it (nearest: the least occlusion and the most
detail). No occlusion test: a point hidden behind an object in one view is outvoted by the others. Points no keyframe
sees get no colour (grey in the page).

**Frames.** A keyframe's ``frame_id`` is the fed frame index (the trajectory CSV's ``frame_idx``); fed frame ``k`` is
video frame ``i0 + k`` of the recording's ultrawide MP4 (:func:`umiproc.viz.slam_run.times_from_frames_csv`).
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Camera", "read_camera", "project", "sample_colours", "point_colours", "read_keyframe_images",
           "thumbnail_uri", "EDGE_PX", "N_VIEWS"]

EDGE_PX = 3
N_VIEWS = 5


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    k: tuple[float, float, float, float]
    width: int
    height: int


def read_camera(settings: str | Path) -> Camera:
    """The ultrawide's model from an ORB-SLAM3 settings file (module docstring). ValueError when it is not there."""
    text = Path(settings).read_text()

    def num(key: str) -> float:
        m = re.search(rf"^\s*{re.escape(key)}\s*:\s*([-+0-9.eE]+)", text, re.M)
        if m is None:
            raise ValueError(f"{settings}: no {key}")
        return float(m.group(1))

    typ = re.search(r'^\s*Camera\.type\s*:\s*"?(\w+)', text, re.M)
    if typ is None or typ.group(1) != "KannalaBrandt8":
        raise ValueError(f"{settings}: Camera.type is not KannalaBrandt8")
    return Camera(num("Camera1.fx"), num("Camera1.fy"), num("Camera1.cx"), num("Camera1.cy"),
                  tuple(num(f"Camera1.k{i}") for i in range(1, 5)), int(num("Camera.width")),
                  int(num("Camera.height")))


def project(cam: Camera, T_W_cam: np.ndarray, pts_W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(uv (N, 2), in_front (N,), dist (N,))`` of world points in a camera at pose ``T_W_cam`` (4x4)."""
    R, t = T_W_cam[:3, :3], T_W_cam[:3, 3]
    p = (np.asarray(pts_W, np.float64) - t) @ R                    # R^T (p - t): camera coordinates
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    th2 = th * th
    k1, k2, k3, k4 = cam.k
    thd = th * (1 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(r > 1e-12, thd / r, 0.0)
    uv = np.stack([cam.fx * s * x + cam.cx, cam.fy * s * y + cam.cy], axis=1)
    return uv, z > 1e-6, np.linalg.norm(p, axis=1)


def sample_colours(cam: Camera, T_W_kf: np.ndarray, pts_W: np.ndarray, images: dict[int, np.ndarray],
                   mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(rgb (P, K, 3) float with NaN where not seen, dist (P, K) with inf where not seen)``. ``images``: keyframe
    index (row of ``T_W_kf``) -> RGB image at any size (scaled from the camera's); keyframes without one are not
    used. ``mask``: (H, W) at the camera's size, nonzero = masked."""
    P, K = len(pts_W), len(T_W_kf)
    rgb = np.full((P, K, 3), np.nan)
    dist = np.full((P, K), np.inf)
    for j, img in images.items():
        uv, front, d = project(cam, T_W_kf[j], pts_W)
        h, w = img.shape[:2]
        sx, sy = w / cam.width, h / cam.height
        u, v = uv[:, 0], uv[:, 1]
        ok = front & (u >= EDGE_PX) & (v >= EDGE_PX) & (u < cam.width - EDGE_PX) & (v < cam.height - EDGE_PX)
        ui, vi = np.zeros(P, int), np.zeros(P, int)
        ui[ok], vi[ok] = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int)
        if mask is not None:
            ok[ok] &= mask[vi[ok], ui[ok]] == 0
        if not ok.any():
            continue
        iu = np.clip((u[ok] * sx).astype(int), 0, w - 1)
        iv = np.clip((v[ok] * sy).astype(int), 0, h - 1)
        rgb[ok, j] = img[iv, iu, :3]
        dist[ok, j] = d[ok]
    return rgb, dist


def point_colours(rgb: np.ndarray, dist: np.ndarray, n_views: int = N_VIEWS) -> tuple[np.ndarray, np.ndarray]:
    """``(colour (P, 3) uint8, views (P,))``: the median of the ``n_views`` nearest samples per point (module
    docstring); ``views`` = the samples used, 0 = not seen (colour 0)."""
    P = len(rgb)
    out = np.zeros((P, 3), np.uint8)
    views = np.zeros(P, np.int64)
    if P == 0 or rgb.shape[1] == 0:
        return out, views
    order = np.argsort(dist, axis=1)[:, :n_views]                   # inf (not seen) sorts last
    near = np.take_along_axis(rgb, order[:, :, None], axis=1)
    seen = np.isfinite(np.take_along_axis(dist, order, axis=1))
    views = seen.sum(axis=1)
    has = views > 0
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(seen[:, :, None], near, np.nan)[has], axis=1)
    out[has] = np.clip(np.rint(med), 0, 255).astype(np.uint8)
    return out, views


def read_keyframe_images(video: str | Path, video_index: dict[int, int], width: int) -> dict[int, np.ndarray]:
    """``{key: rgb}`` for ``video_index`` (key -> video frame), scaled to ``width`` px (0: full size)."""
    import av
    import cv2

    from umiproc.viz.slam_run import PlotRunError

    by_frame: dict[int, list[int]] = {}
    for key, i in video_index.items():
        by_frame.setdefault(int(i), []).append(key)
    out: dict[int, np.ndarray] = {}
    if not by_frame:
        return out
    last = max(by_frame)
    try:
        with av.open(str(video)) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(c.decode(stream)):
                if i in by_frame:
                    img = frame.to_ndarray(format="rgb24")
                    if width and img.shape[1] != width:
                        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])),
                                         interpolation=cv2.INTER_AREA)
                    for key in by_frame[i]:
                        out[key] = img
                if i >= last:
                    break
    except (av.error.FFmpegError, OSError) as e:
        raise PlotRunError(f"cannot decode {video}: {e}") from e
    return out


def thumbnail_uri(rgb: np.ndarray, width: int, quality: int = 75, mask: np.ndarray | None = None) -> str:
    """A JPEG data URI of ``rgb`` scaled to ``width`` px; ``mask`` (any size, nonzero = masked) darkened."""
    import cv2

    h = round(rgb.shape[0] * width / rgb.shape[1])
    img = cv2.resize(rgb, (width, h), interpolation=cv2.INTER_AREA)
    if mask is not None:
        m = cv2.resize(mask, (width, h), interpolation=cv2.INTER_NEAREST) > 0
        img[m] = (img[m] * 0.45).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
