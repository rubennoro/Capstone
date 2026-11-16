"""The gripper mask for SLAM: ``cameras.ultrawide.slam_mask`` drawn as an image (format: :mod:`dataproc.config`).

Upstream ORB-SLAM3 (``mono_inertial_euroc``) takes no mask, so :func:`dataproc.slam.orbslam3.euroc_image` blacks
the masked pixels out in the frames written for it. Only those frame copies are masked: training images and tag
detection read the videos unmasked.

Convention: uint8, **255 = keep** the pixel, **0 = masked**, as ``sensor_rig/sensor_rig/pose/mask.py`` (UMI's
ORB-SLAM3 fork convention).

Coordinates: the polygons are pixel **edges** in the camera's native resolution ``W x H`` (``0 <= x <= W``,
``0 <= y <= H``), so ``[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]`` reads like the numpy slice ``[y0:y1, x0:x1]``. To
draw at ``w x h``, each point is scaled per axis (``x * w / W``, ``y * h / H``), shifted by -0.5 into OpenCV's
pixel-centre convention and filled with ``cv2.fillPoly(..., shift=8)`` (8 fractional bits). ``fillPoly`` also draws
the outline, so the mask can reach up to 1 px past the slice (on the lower and right edges of a rectangle): it can
come out slightly larger than the polygons but never leaves a gap. Each polygon is filled in its own call. One call
with several polygons leaves the areas where they overlap *unfilled*, because OpenCV's scanline fill toggles between
inside and outside at every edge.
"""

from __future__ import annotations

import functools

import cv2
import numpy as np

__all__ = ["keep_mask", "slam_keep_mask"]


def keep_mask(polygons, native_size, size) -> np.ndarray:
    """The mask of ``polygons`` (pixel-edge ``(x, y)`` points at ``native_size = (W, H)``) drawn at
    ``size = (w, h)``: uint8 ``(h, w)``, 255 = keep, 0 = masked (conventions: module docstring).

    Cached: repeated calls with equal arguments return the same **read-only** array, so do not modify it.
    """
    polys = tuple(tuple((float(x), float(y)) for x, y in poly) for poly in polygons)
    return _draw(polys, (int(native_size[0]), int(native_size[1])), (int(size[0]), int(size[1])))


@functools.lru_cache(maxsize=16)
def _draw(polygons: tuple, native_size: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    (W, H), (w, h) = native_size, size
    if min(W, H, w, h) <= 0:
        raise ValueError(f"sizes must be positive, got native {native_size} and {size}")
    m = np.full((h, w), 255, np.uint8)
    scale = np.array([w / W, h / H])
    for poly in polygons:                       # one polygon per call: see the module docstring
        pts = np.round((np.asarray(poly, np.float64) * scale - 0.5) * 256).astype(np.int32)
        cv2.fillPoly(m, [pts], 0, lineType=cv2.LINE_8, shift=8)
    m.setflags(write=False)
    return m


def slam_keep_mask(rig, size=None) -> np.ndarray | None:
    """:func:`keep_mask` of the rig's ``cameras.ultrawide.slam_mask`` at ``size = (w, h)`` (default: the
    ultrawide's resolution), or None when the rig has no mask."""
    cam = rig.cameras["ultrawide"]
    if cam.slam_mask is None:
        return None
    native = (int(cam.resolution[0]), int(cam.resolution[1]))
    return keep_mask(cam.slam_mask.polygons, native, native if size is None else size)
