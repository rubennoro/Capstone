# Tag detection vendored from iPhUMI python_package/iphumi/demonstration_processing/scripts/detect_aruco_iphone.py
# (detect_localize_aruco_tags_iphone); the gripper-range fit extracted from .../process_stages/calibrate.py
# (calibrate_gripper_range_iphone, inline logic as a pure function); per-frame width from .../utils/gripper_util.py
# (get_gripper_width_offset); interpolators from .../utils/interpolation_util.py. All at commit 4be9982.
"""Stages ``gripper_range`` and ``width``: finger ArUco tags on the virtual ultrawide -> gripper width.

* :func:`detect_tags`: ``DICT_4X4_50`` (``rig.aruco``) on the virtual pinhole image, ``CORNER_REFINE_SUBPIX``,
  ``solvePnP`` with the virtual K and zero distortion, every tag ``marker_size_m``. ``tvec`` is in the **virtual**
  ultrawide optical frame (x right, y down, z forward), metres.
* :func:`fit_gripper_range` (calibration episode): frames with both tags at ``|tvec_z - nominal_z| < 0.08``;
  ``width = right.x - left.x``; raw min / max; robust linear maps ``right.x = a + b * left.x`` and back (polyfit,
  one 3-MAD outlier pass); written as iPhUMI's ``<side>_gripper_range.json``.
* :func:`episode_widths`: per frame, both tags -> ``right.x - left.x``; one tag -> the other predicted with the
  linear map; clipped to [min, max]; then the calibration interpolator subtracts ``min_width`` (closed reads 0).
* :func:`width_on_grid`: linear interpolation onto the output grid with a validity mask.

Images come from the caller (``to_virtual`` callable / iterable of virtual frames), so this module does not depend
on the image-prep stage.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

import cv2
import numpy as np

__all__ = [
    "Z_TOLERANCE_M", "DEFAULT_TAG_DET_THRESHOLD", "TagDetection", "GripperRange", "GripperRangeError",
    "aruco_dictionary", "detect_tags", "detect_tags_in_frames", "detect_episode_tags", "fit_gripper_range",
    "gripper_width_offset", "calibration_interpolator", "episode_widths", "width_on_grid",
]

Z_TOLERANCE_M = 0.08                 # iPhUMI z_tolerance
DEFAULT_TAG_DET_THRESHOLD = 0.2      # iPhUMI process_demos_iphone.yaml tag_det_threshold


class TagDetection(NamedTuple):
    """One tag: ``rvec`` (3,), ``tvec`` (3,) in the virtual ultrawide frame (m), ``corners`` (4,2) pixels."""

    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray


class GripperRangeError(ValueError):
    """The calibration episode cannot give a gripper range (tags not seen, too few both-tag frames)."""


def aruco_dictionary(rig) -> "cv2.aruco.Dictionary":
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, rig.aruco.dict))


# --------------------------------------------------------------------------------------------- detection

def _detect(img: np.ndarray, aruco_dict, marker_size_m: float, K: np.ndarray, refine_subpix: bool = True):
    """iPhUMI ``detect_localize_aruco_tags_iphone`` with one marker size for every id."""
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=param)
    corners, ids, _ = detector.detectMarkers(image=img)
    if len(corners) == 0:
        return {}
    h = marker_size_m / 2
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left (marker plane, y up)
    obj_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
    dist = np.zeros((5, 1), dtype=np.float32)
    out = {}
    for this_id, this_corners in zip(ids, corners):
        img_points = this_corners.reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
        if not ok:
            continue
        out[int(this_id[0])] = TagDetection(rvec.squeeze().astype(np.float64), tvec.squeeze().astype(np.float64),
                                            this_corners.squeeze().astype(np.float64))
    return out


def detect_tags(virtual_rgb: np.ndarray, rig) -> dict[int, TagDetection]:
    """Tags in one virtual ultrawide image (uint8 RGB or gray): ``{id: (rvec, tvec, corners)}``."""
    K = np.asarray(rig.virtual_ultrawide.K, dtype=np.float64)
    return _detect(np.ascontiguousarray(virtual_rgb), aruco_dictionary(rig), rig.aruco.marker_size_m, K)


def detect_tags_in_frames(virtual_frames: Iterable[np.ndarray], rig) -> list[dict[int, TagDetection]]:
    """:func:`detect_tags` over an iterable of virtual images (one dict per image, empty if nothing found)."""
    K = np.asarray(rig.virtual_ultrawide.K, dtype=np.float64)
    d = aruco_dictionary(rig)
    return [_detect(np.ascontiguousarray(img), d, rig.aruco.marker_size_m, K) for img in virtual_frames]


def detect_episode_tags(episode, rig, to_virtual: Callable[[np.ndarray], np.ndarray],
                        indices=None) -> tuple[np.ndarray, list[dict[int, TagDetection]]]:
    """Decode the ultrawide MP4 (all frames, or ``indices``), remap each with ``to_virtual`` (e.g.
    ``UltrawidePrep(rig).to_virtual``) and detect. Returns ``(t_frames_ns, detections)`` with the mid-exposure
    times on the IMU time base."""
    from dataproc.slam.base import ultrawide_camera, ultrawide_times_ns

    t_all = ultrawide_times_ns(episode, rig)
    cam = ultrawide_camera(episode)
    idx = []

    def gen():
        for i, rgb in episode.read_video_frames(cam, indices):
            idx.append(i)
            yield to_virtual(np.ascontiguousarray(rgb))

    dets = detect_tags_in_frames(gen(), rig)
    return t_all[np.asarray(idx, dtype=np.int64)], dets


# --------------------------------------------------------------------------------------------- range fit

def _tvec(det) -> np.ndarray:
    if isinstance(det, dict):
        return np.asarray(det["tvec"], dtype=np.float64).reshape(-1)
    if hasattr(det, "tvec"):
        return np.asarray(det.tvec, dtype=np.float64).reshape(-1)
    return np.asarray(det[1], dtype=np.float64).reshape(-1)


def _norm_frame(frame) -> dict:
    """Accept {id: TagDetection | (rvec, tvec, corners) | {'tvec': ...}} with int or str keys."""
    return {int(k): v for k, v in frame.items()}


@dataclass
class GripperRange:
    """iPhUMI ``<side>_gripper_range.json`` (same field names; ``x_basis`` always ``tvec0``)."""

    gripper_id: int
    left_finger_tag_id: int
    right_finger_tag_id: int
    max_width: float
    min_width: float
    left_to_right: dict[str, float]
    right_to_left: dict[str, float]
    fit_stats: dict[str, Any] = field(default_factory=dict)
    x_basis: str = "tvec0"

    def to_json_dict(self) -> dict:
        return {"gripper_id": int(self.gripper_id), "left_finger_tag_id": int(self.left_finger_tag_id),
                "right_finger_tag_id": int(self.right_finger_tag_id), "max_width": float(self.max_width),
                "min_width": float(self.min_width), "x_basis": self.x_basis,
                "left_to_right": {k: float(v) for k, v in self.left_to_right.items()},
                "right_to_left": {k: float(v) for k, v in self.right_to_left.items()},
                "fit_stats": dict(self.fit_stats)}

    @classmethod
    def from_json_dict(cls, d: dict) -> "GripperRange":
        x_basis = d.get("x_basis", "tvec0")
        if x_basis != "tvec0":
            raise ValueError(f"unsupported gripper_range x_basis={x_basis!r}; expected 'tvec0'")
        for k in ("left_finger_tag_id", "right_finger_tag_id", "max_width", "min_width", "left_to_right",
                  "right_to_left"):
            if k not in d:
                raise ValueError(f"gripper_range: missing {k!r}")
        return cls(gripper_id=int(d.get("gripper_id", int(d["left_finger_tag_id"]) // 2)),
                   left_finger_tag_id=int(d["left_finger_tag_id"]), right_finger_tag_id=int(d["right_finger_tag_id"]),
                   max_width=float(d["max_width"]), min_width=float(d["min_width"]),
                   left_to_right={"a": float(d["left_to_right"]["a"]), "b": float(d["left_to_right"]["b"])},
                   right_to_left={"a": float(d["right_to_left"]["a"]), "b": float(d["right_to_left"]["b"])},
                   fit_stats=dict(d.get("fit_stats", {})), x_basis=x_basis)

    def save(self, path) -> Path:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path) -> "GripperRange":
        return cls.from_json_dict(json.loads(Path(path).read_text()))


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """iPhUMI fit_line: y = a + b*x with one 3-MAD outlier rejection pass; returns (a, b, mae)."""
    b, a = np.polyfit(x, y, 1)
    resid = y - (b * x + a)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    if mad > 0:
        mask = np.abs(resid - med) <= 3.0 * mad
        if np.sum(mask) >= 3:
            b, a = np.polyfit(x[mask], y[mask], 1)
            resid = y[mask] - (b * x[mask] + a)
    return float(a), float(b), float(np.mean(np.abs(resid)))


def fit_gripper_range(detections_per_frame, rig, tag_det_threshold: float = DEFAULT_TAG_DET_THRESHOLD,
                      z_tolerance: float = Z_TOLERANCE_M) -> GripperRange:
    """iPhUMI ``calibrate_gripper_range_iphone`` for one calibration episode, as a pure function.

    Differences: the tag ids come from ``rig.aruco`` (iPhUMI picks the most-seen gripper id pair; here the pair is
    fixed per rig), ``nominal_z`` is ``rig.aruco.nominal_z_m``, and failures raise :class:`GripperRangeError`.
    """
    frames = [_norm_frame(f) for f in detections_per_frame]
    n = len(frames)
    if n == 0:
        raise GripperRangeError("no frames")
    left_id, right_id = rig.aruco.left_id, rig.aruco.right_id
    counts = collections.Counter(k for f in frames for k in f)
    det_rate = min(counts[left_id], counts[right_id]) / n
    if det_rate <= 0:
        raise GripperRangeError(f"finger tags {left_id}/{right_id} never both detected")
    if det_rate < tag_det_threshold:
        raise GripperRangeError(f"detection rate {det_rate:.3f} < {tag_det_threshold} threshold")
    nominal_z = rig.aruco.nominal_z_m
    zmin, zmax = nominal_z - z_tolerance, nominal_z + z_tolerance
    lx, rx = [], []
    for f in frames:
        l = _tvec(f[left_id]) if left_id in f else None
        r = _tvec(f[right_id]) if right_id in f else None
        if l is not None and r is not None and zmin < l[-1] < zmax and zmin < r[-1] < zmax:
            lx.append(l[0])
            rx.append(r[0])
    lx, rx = np.asarray(lx, np.float64), np.asarray(rx, np.float64)
    widths = rx - lx
    if widths.size < 3:
        raise GripperRangeError(f"not enough valid both-tag frames: n_both_frames={int(widths.size)}")
    a_lr, b_lr, mae_lr = _fit_line(lx, rx)
    a_rl, b_rl, mae_rl = _fit_line(rx, lx)
    return GripperRange(gripper_id=left_id // 2, left_finger_tag_id=left_id, right_finger_tag_id=right_id,
                        max_width=float(np.nanmax(widths)), min_width=float(np.nanmin(widths)),
                        left_to_right={"a": a_lr, "b": b_lr}, right_to_left={"a": a_rl, "b": b_rl},
                        fit_stats={"n_both_frames": int(widths.size), "mae_left_to_right": mae_lr,
                                   "mae_right_to_left": mae_rl, "n_frames": n, "detection_rate": det_rate})


# --------------------------------------------------------------------------------------------- widths

def gripper_width_offset(tag_dict, left_id: int, right_id: int, nominal_z: float, *, left_to_right: dict,
                         right_to_left: dict, z_tolerance: float = Z_TOLERANCE_M, min_width: float | None = None,
                         max_width: float | None = None):
    """iPhUMI ``get_gripper_width_offset``: ``(width | None, left_present, right_present)``."""
    zmin, zmax = nominal_z - z_tolerance, nominal_z + z_tolerance
    left_x = right_x = None
    if left_id in tag_dict:
        t = _tvec(tag_dict[left_id])
        if zmin < t[-1] < zmax:
            left_x = float(t[0])
    if right_id in tag_dict:
        t = _tvec(tag_dict[right_id])
        if zmin < t[-1] < zmax:
            right_x = float(t[0])
    width = None
    if left_x is not None and right_x is not None:
        width = right_x - left_x
    elif left_x is not None:
        width = (float(left_to_right["a"]) + float(left_to_right["b"]) * left_x) - left_x
    elif right_x is not None:
        width = right_x - (float(right_to_left["a"]) + float(right_to_left["b"]) * right_x)
    if width is not None and min_width is not None and max_width is not None:
        width = float(np.clip(width, min_width, max_width))
    return width, left_x is not None, right_x is not None


def calibration_interpolator(gripper_range: GripperRange) -> Callable:
    """iPhUMI ``get_gripper_calibration_interpolator([min, max], [min, max])``: measured -> width - min, clamped."""
    lo, hi = gripper_range.min_width, gripper_range.max_width
    xs, ys = np.array([lo, hi]), np.array([lo, hi]) - lo
    if hi <= lo:
        return lambda w: np.zeros_like(np.asarray(w, dtype=np.float64))
    return lambda w: np.interp(np.asarray(w, dtype=np.float64), xs, ys)


def episode_widths(detections_per_frame, t_frames_ns, gripper_range: GripperRange, rig):
    """Width at every frame with a usable tag. Returns ``(t_ns int64, width float64 (closed = 0), left_present,
    right_present)`` for those frames only (as iPhUMI's detected lists)."""
    t_frames = np.asarray(t_frames_ns, dtype=np.int64).reshape(-1)
    frames = list(detections_per_frame)
    if len(frames) != len(t_frames):
        raise ValueError(f"{len(frames)} detection frames vs {len(t_frames)} times")
    gr = gripper_range
    interp = calibration_interpolator(gr)
    t_out, w_out, lp, rp = [], [], [], []
    for t, f in zip(t_frames.tolist(), frames):
        w, l, r = gripper_width_offset(_norm_frame(f), gr.left_finger_tag_id, gr.right_finger_tag_id,
                                       rig.aruco.nominal_z_m, left_to_right=gr.left_to_right,
                                       right_to_left=gr.right_to_left, min_width=gr.min_width,
                                       max_width=gr.max_width)
        if w is not None:
            t_out.append(t)
            w_out.append(float(interp(w)))
            lp.append(l)
            rp.append(r)
    return (np.asarray(t_out, np.int64), np.asarray(w_out, np.float64), np.asarray(lp, bool), np.asarray(rp, bool))


def width_on_grid(t_ns, width, t_grid_ns, max_gap_ns: int):
    """Linear interpolation of the per-frame widths onto the grid.

    Returns ``(width (N,1) float32, valid (N,) bool)``. Inside the detection span a grid time is valid if its
    bracketing detections are at most ``max_gap_ns`` apart (an exact hit is always valid); outside, the end value
    is held (iPhUMI's ``fill_value``) and valid only within ``max_gap_ns`` of the end detection. No detection at all:
    zeros, all invalid.
    """
    t = np.asarray(t_ns, dtype=np.int64).reshape(-1)
    w = np.asarray(width, dtype=np.float64).reshape(-1)
    tg = np.asarray(t_grid_ns, dtype=np.int64).reshape(-1)
    if len(t) != len(w):
        raise ValueError("t_ns and width differ in length")
    if len(t) == 0:
        return np.zeros((len(tg), 1), np.float32), np.zeros(len(tg), bool)
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_ns must be strictly increasing")
    t0 = int(t[0])
    out = np.interp((tg - t0).astype(np.float64), (t - t0).astype(np.float64), w)
    hi = np.clip(np.searchsorted(t, tg, side="left"), 0, len(t) - 1)
    lo = np.clip(hi - 1, 0, len(t) - 1)
    exact = t[hi] == tg
    inside = (tg >= t[0]) & (tg <= t[-1])
    valid = inside & (exact | (t[hi] - t[lo] <= max_gap_ns))
    valid |= (tg < t[0]) & (t[0] - tg <= max_gap_ns)
    valid |= (tg > t[-1]) & (tg - t[-1] <= max_gap_ns)
    return out.astype(np.float32).reshape(-1, 1), valid
