"""Board corners in every frame of the ultrawide video, the gripper mask, and the motion filter.

* **Detection**: OpenCV's ``CharucoDetector`` with the board of :mod:`umiproc.lens.board` (``DICT_5X5_100``; the
  gripper's ``DICT_4X4_50`` tags are never read as board markers). No camera model is given to the detector, so
  each corner is interpolated from its neighbouring markers by a local homography, then refined to sub-pixel by
  OpenCV. One detected neighbouring marker is enough (``minMarkers = 1``, OpenCV's default is 2): near the image's
  edges and corners, where the lens model is least constrained, the second marker of a corner is often cut off
  by the image border. On renders through a known fisheye model such corners are as accurate as two-marker ones
  (RMS ~0.05 px) once they are >= 12 px from the border, and much worse closer (the refinement window is cut), so
  :data:`EDGE_PX` is 12.
* **Pixel convention**: OpenCV 4.12's ``detectBoard`` returns ChArUco corners shifted by +0.5 px in x and y from
  the pixel-centre convention (pixel ``(i, j)`` centred at ``(u=j, v=i)``) that ``cornerSubPix``, the marker
  corners, ``cv2.fisheye``, ORB-SLAM3 and :mod:`umiproc.dp.geometry.camera_model` all use. Left in, the shift would
  move the fitted principal point by 0.5 px. :func:`detector_offset` measures it once per run on an exact
  rendering of the board (where every corner's true position is known) and :func:`detect_frame` subtracts it; the
  measured value is reported, and anything that is neither 0 nor 0.5 px stops the calibration.
* **Mask**: corners inside the gripper mask (``configs/slam/<rig>.yaml``, drawn by
  :func:`umiproc.slam.config.mask_keep`) are dropped: the fingers and the held object move there, and the mask was
  traced with a >= 12 px margin around them. So are corners within :data:`EDGE_PX` of the image border (their
  refinement window would leave the image). A frame is usable with >= :data:`MIN_CORNERS` corners left.
* **Motion filter** (:func:`still_flags`): a frame is kept only if its corners moved at most :data:`STILL_PX`
  (median over the corner ids it shares with that frame) relative to **both** neighbouring frames, each sharing at
  least :data:`MIN_COMMON` ids with it. A frame whose neighbour has too few corners in common cannot be shown to be
  still and is dropped too. At 30 fps, 1 px per frame is 30 px/s: slow enough that motion blur (exposure of a few
  ms) and the rolling shutter's skew stay well below a pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["MIN_CORNERS", "STILL_PX", "MIN_COMMON", "EDGE_PX", "FrameCorners", "VideoCorners", "make_detector",
           "detector_offset", "detect_frame", "detect_video", "still_flags"]

MIN_CORNERS = 12          # corners outside the mask for a frame to be usable
STILL_PX = 1.0            # max median corner motion to each neighbouring frame (px)
MIN_COMMON = 8            # corner ids a neighbouring frame must share to compare motion
EDGE_PX = 12.0            # corners closer than this to the image border (pixel edge) are dropped


@dataclass(frozen=True)
class FrameCorners:
    """The board corners of one frame that survived the mask: ``ids`` (N,) int, ``uv`` (N, 2) float64 pixels
    (pixel-centre convention), sorted by id."""

    frame: int
    ids: np.ndarray
    uv: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class VideoCorners:
    """Detections of a whole video: ``frames[i]`` is a :class:`FrameCorners` (any number of corners, possibly 0) or
    None when no board corner was found at all. ``backdrop`` is a copy of the frame with the most corners (for the
    coverage image)."""

    frames: list
    size: tuple[int, int]
    n_decoded: int
    n_masked_corners: int
    backdrop: np.ndarray | None
    offset_px: float


def make_detector():
    import cv2

    from umiproc.lens.board import make_board

    params = cv2.aruco.CharucoParameters()
    params.minMarkers = 1                 # one neighbouring marker is enough (module docstring)
    params.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(make_board(), params, cv2.aruco.DetectorParameters())


def detector_offset(detector=None) -> float:
    """``detectBoard``'s corner shift from the pixel-centre convention, in px (same in x and y), measured on an
    exact rendering of the board: 0.5 with OpenCV 4.12, 0 if a later OpenCV fixes it. Raises RuntimeError for
    anything else (the detector would then need a closer look before it can be trusted)."""
    import cv2

    from umiproc.lens import board as lens_board

    detector = detector or make_detector()
    scale, margin = 2, 14.0
    p = lens_board.EXACT_PX_PER_MM * scale
    img = cv2.GaussianBlur(lens_board.board_image(scale, margin), (0, 0), 0.8)
    cc, ci, _, _ = detector.detectBoard(img)
    n = lens_board.board_info()["inner_corners"]
    if ci is None or len(ci) != n:
        raise RuntimeError(f"the ChArUco detector found {0 if ci is None else len(ci)} of the {n} corners on an "
                           "ideal image of the board")
    corners_mm = np.asarray(lens_board.make_board().getChessboardCorners(), np.float64)[:, :2] * 1000.0
    truth = (corners_mm[ci.ravel()] + margin) * p - 0.5
    d = np.median(cc.reshape(-1, 2).astype(np.float64) - truth, axis=0)
    for expected in (0.0, 0.5):
        if np.all(np.abs(d - expected) < 0.05):
            return expected
    raise RuntimeError(f"unexpected ChArUco corner convention: median shift ({d[0]:.3f}, {d[1]:.3f}) px on an ideal "
                       "image (expected 0 or 0.5)")


def detect_frame(detector, gray: np.ndarray, keep: np.ndarray, offset_px: float, frame: int = 0):
    """``(FrameCorners or None, number of corners dropped by the mask)`` for one grey frame."""
    cc, ci, _, _ = detector.detectBoard(gray)
    if ci is None or len(ci) == 0:
        return None, 0
    ids = ci.ravel().astype(np.int64)
    uv = cc.reshape(-1, 2).astype(np.float64) - offset_px
    H, W = gray.shape[:2]
    inside = ((uv[:, 0] + 0.5 >= EDGE_PX) & (uv[:, 0] + 0.5 <= W - EDGE_PX)
              & (uv[:, 1] + 0.5 >= EDGE_PX) & (uv[:, 1] + 0.5 <= H - EDGE_PX))
    px = np.clip(np.rint(uv).astype(np.int64), 0, [W - 1, H - 1])
    unmasked = keep[px[:, 1], px[:, 0]] != 0
    n_masked = int((inside & ~unmasked).sum())
    sel = inside & unmasked
    order = np.argsort(ids[sel], kind="stable")
    return FrameCorners(frame, ids[sel][order], uv[sel][order]), n_masked


def detect_video(path: Path, keep: np.ndarray, progress=None, every: int = 300) -> VideoCorners:
    """Decode ``path`` (grey) and detect the board in every frame. ``keep`` is the gripper mask at the video's size
    (uint8, 0 = masked). ``progress(n_frames)`` is called every ``every`` frames. Raises ValueError when a frame
    does not have the mask's size."""
    import av

    detector = make_detector()
    offset = detector_offset(detector)
    frames, n_masked, best, backdrop = [], 0, -1, None
    H, W = keep.shape
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(stream)):
            gray = frame.to_ndarray(format="gray")
            if gray.shape != (H, W):
                raise ValueError(f"{path.name}: frame {i} is {gray.shape[1]}x{gray.shape[0]}, expected {W}x{H}")
            fc, nm = detect_frame(detector, gray, keep, offset, i)
            n_masked += nm
            frames.append(fc)
            if fc is not None and len(fc) > best:
                best, backdrop = len(fc), gray.copy()
            if backdrop is None and i == 0:
                backdrop = gray.copy()
            if progress is not None and (i + 1) % every == 0:
                progress(i + 1)
    return VideoCorners(frames, (W, H), len(frames), n_masked, backdrop, offset)


def _median_motion(a: FrameCorners | None, b: FrameCorners | None) -> tuple[float, int]:
    """(median displacement over shared ids, number shared); (inf, n) when fewer than MIN_COMMON are shared."""
    if a is None or b is None:
        return float("inf"), 0
    common, ia, ib = np.intersect1d(a.ids, b.ids, assume_unique=True, return_indices=True)
    if len(common) < MIN_COMMON:
        return float("inf"), len(common)
    return float(np.median(np.linalg.norm(a.uv[ia] - b.uv[ib], axis=1))), len(common)


def still_flags(frames: list) -> tuple[np.ndarray, np.ndarray]:
    """``(still, motion)`` per frame: ``motion`` is the larger of the two neighbour motions (inf when not
    measurable); ``still`` is ``motion <= STILL_PX`` for frames with >= MIN_CORNERS corners (module docstring)."""
    n = len(frames)
    motion = np.full(n, np.inf)
    for i, fc in enumerate(frames):
        if fc is None or len(fc) < MIN_CORNERS or i == 0 or i == n - 1:
            continue
        motion[i] = max(_median_motion(fc, frames[i - 1])[0], _median_motion(fc, frames[i + 1])[0])
    return motion <= STILL_PX, motion
