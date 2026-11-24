"""The two sensor chains, from the copy's deployed rig file (every lens constant comes from there).

Like the real Pi, each camera is produced at its native sensor size and scaled down to the stream size:

* ``main`` (IMX519): one undistorted pinhole render at the native size (``1600x1200``) with the rig's focal length,
  widened just enough to hold every ray of the distorted image; the rig's radtan distortion applied by remap
  (``PinholeRadtan.unproject`` of the copy, per native pixel); then ``INTER_AREA`` down to the rig's 640x480.
* ``ultrawide`` (IMX219): a cube map around the lens (only the face rectangles the sensor sees, each an off-centre
  pinhole at about the fisheye's own angular density) warped into the equidistant fisheye at the native size
  (``1280x960``, the rig's measured lens scaled up with the half-pixel convention); then ``INTER_AREA`` to 640x480.
  The measured lens reaches about 97 deg off axis in the sensor corners, past the copy model's 89 deg validity cap
  (``FisheyeEquidistant.theta_valid``); the same polynomial is used there while it is still monotonic (it is up to
  about 105 deg), so the whole sensor is covered, as in the real frames (no black image circle). A pixel whose ray
  the polynomial cannot reach would be black.

Native sizes: training's recorded sizes (``rig_pi_v3.yaml``: main 1600x1200, ultrawide 1280x960); the rig file is
their 640x480 scaled version (``deploy_rig.scale_to``), so ``model.scaled(native)`` inverts it exactly.

The 640x480 output is what the Pi streams before JPEG; nothing the real Pi does not do is applied (no unwarp, no crop,
no letterbox).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from sim_deploy.sim_pi._deploy import FisheyeEquidistant, PinholeRadtan, RigConfig, physical_model
from sim_deploy.sim_pi.render import GLRenderer, View

#: Training's recorded sensor sizes (umi-processing configs/rigs/rig_pi_v3.yaml), per role.
NATIVE_SIZE = {"main": (1600, 1200), "ultrawide": (1280, 960)}
PAD_PX = 3                      # extra pixels around each rendered rectangle (bilinear footprint + rounding)
_INVALID = -16.0                # remap coordinate for "no source sample" (black), as camera_model.py


def _grid(w: int, h: int) -> np.ndarray:
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.stack([u.ravel(), v.ravel()], axis=1)


def fisheye_rays(model: FisheyeEquidistant, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Unit rays (N, 3) of fisheye pixels, the equidistant polynomial inverted over its whole monotonic range (not
    only up to theta_max_deg 89). Returns (rays, ok, theta_mono_rad)."""
    k1, k2, k3, k4 = model.k1, model.k2, model.k3, model.k4

    def td(t):
        t2 = t * t
        return t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def dtd(t):
        t2 = t * t
        return 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))

    ts = np.linspace(0, math.pi, 200001)
    bad = np.flatnonzero(dtd(ts) <= 0)
    theta_mono = float(ts[bad[0] - 1]) if len(bad) else math.pi
    xd = (uv[:, 0] - model.cx) / model.fx
    yd = (uv[:, 1] - model.cy) / model.fy
    r = np.hypot(xd, yd)
    ok = r <= td(theta_mono)
    lo, hi = np.zeros_like(r), np.full_like(r, theta_mono)
    for _ in range(64):                                  # bisection: td is increasing on [0, theta_mono]
        mid = 0.5 * (lo + hi)
        below = td(mid) < r
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    theta = 0.5 * (lo + hi)
    s = np.where(r > 1e-15, np.sin(theta) / np.maximum(r, 1e-300), 0.0)
    rays = np.stack([xd * s, yd * s, np.cos(theta)], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rays[~ok] = np.nan
    return rays, ok, theta_mono


def _face_rotation(n: np.ndarray) -> np.ndarray:
    """Rotation whose z column is the face normal n (a camera axis), right-handed, y as close to camera +y as can be."""
    n = np.asarray(n, float)
    y = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.5 else np.array([0.0, 0.0, -np.sign(n[1])])
    y = y - n * (y @ n)
    y /= np.linalg.norm(y)
    x = np.cross(y, n)
    R = np.column_stack([x, y, n])
    assert abs(np.linalg.det(R) - 1) < 1e-9
    return R


def _view_for(name: str, tx: np.ndarray, ty: np.ndarray, f: float, R: np.ndarray, hide: tuple[int, ...]) -> View:
    """The smallest pinhole (focal f) whose pixel grid holds every tangent (tx, ty) = (x/z, y/z), plus PAD_PX."""
    x0, x1 = float(tx.min()), float(tx.max())
    y0, y1 = float(ty.min()), float(ty.max())
    cx = -x0 * f + PAD_PX
    cy = -y0 * f + PAD_PX
    w = int(math.ceil((x1 - x0) * f + 2 * PAD_PX)) + 1
    h = int(math.ceil((y1 - y0) * f + 2 * PAD_PX)) + 1
    return View(name, w, h, f, cx, cy, R_cam_view=R, hide=hide)


@dataclass
class ChainInfo:
    role: str
    native: tuple[int, int]
    out: tuple[int, int]
    views: list[dict]
    rendered_px: int


class CameraChain:
    """One camera: views to render, the remap from them to the native sensor image, the scale to the stream size."""

    def __init__(self, role: str, rig: RigConfig, *, hide: tuple[int, ...] = (), density: float = 1.0):
        self.role = role
        cam = rig.cameras[role]
        self.out_size = (int(cam.resolution[0]), int(cam.resolution[1]))
        self.native_size = NATIVE_SIZE[role]
        self.model_out = physical_model(rig, role)                       # the rig file's model, 640x480
        self.native = self.model_out.scaled(*self.native_size)            # same lens at the sensor's size
        W, H = self.native_size
        uv = _grid(W, H)
        if isinstance(self.native, PinholeRadtan):
            rays, ok = self.native.unproject(uv)
            self.theta_mono = None
            normals = [np.array([0.0, 0.0, 1.0])]
        elif isinstance(self.native, FisheyeEquidistant):
            rays, ok, self.theta_mono = fisheye_rays(self.native, uv)
            normals = [np.array(v, float) for v in ((0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                                                    (0, 0, -1))]
        else:
            raise TypeError(f"{role}: unsupported camera model {type(self.native).__name__}")
        f = float(density) * 0.5 * (self.native.fx + self.native.fy)
        self.views: list[View] = []
        face_of = np.full(len(uv), -1, int)
        if len(normals) > 1:
            dots = rays @ np.stack(normals, axis=1)
            face_of[ok] = np.argmax(dots[ok], axis=1)
        else:
            face_of[ok] = 0
        map_x = np.full(len(uv), _INVALID, np.float32)
        map_y = np.full(len(uv), _INVALID, np.float32)
        offsets, y_off = [], 0
        for i, n in enumerate(normals):
            sel = np.flatnonzero(face_of == i)
            if len(sel) == 0:
                continue
            R = _face_rotation(n)
            rv = rays[sel] @ R                                            # ray in view axes (R^T ray)
            tx, ty = rv[:, 0] / rv[:, 2], rv[:, 1] / rv[:, 2]
            v = _view_for(f"{role}_{'front' if i == 0 else i}", tx, ty, f, R, hide)
            self.views.append(v)
            map_x[sel] = (v.f * tx + v.cx).astype(np.float32)
            # views are stored as read back (bottom row first): row v of the view is atlas row y_off + h - 1 - v
            map_y[sel] = (y_off + (v.h - 1) - (v.f * ty + v.cy)).astype(np.float32)
            offsets.append(y_off)
            y_off += v.h
        self._offsets = offsets
        self._atlas_shape = (y_off, max(v.w for v in self.views), 3)
        self.map1, self.map2 = cv2.convertMaps(map_x.reshape(H, W), map_y.reshape(H, W), cv2.CV_16SC2)
        self.covered = float(ok.mean())

    def set_hide(self, geom_ids) -> None:
        """Geoms not drawn in this camera's views (its own board and lens barrel)."""
        ids = tuple(int(g) for g in geom_ids if g is not None and g >= 0)
        for v in self.views:
            v.hide = ids

    @property
    def max_view(self) -> tuple[int, int]:
        return max(v.w for v in self.views), max(v.h for v in self.views)

    def info(self) -> ChainInfo:
        return ChainInfo(self.role, self.native_size, self.out_size,
                         [{"name": v.name, "w": v.w, "h": v.h, "f": round(v.f, 3)} for v in self.views],
                         int(sum(v.w * v.h for v in self.views)))

    def new_buffer(self) -> np.ndarray:
        """An atlas for ``render_into`` (one per frame in flight)."""
        return np.zeros(self._atlas_shape, np.uint8)

    def render_into(self, r: GLRenderer, R_world_cam: np.ndarray, p_world_cam: np.ndarray, atlas: np.ndarray) -> None:
        """GL stage: render every view (``r.update`` done for this camera's state) into ``atlas``, as read back."""
        for v, y0 in zip(self.views, self._offsets):
            atlas[y0:y0 + v.h, :v.w] = r.render_raw(v, R_world_cam, p_world_cam)

    def native_from(self, atlas: np.ndarray) -> np.ndarray:
        """CPU stage: the views warped into the native sensor image (RGB, top row first)."""
        return cv2.remap(atlas, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))

    def stream_from(self, atlas: np.ndarray) -> np.ndarray:
        """CPU stage: RGB uint8 (480, 640, 3), what the Pi hands to its JPEG encoder: native image, then INTER_AREA
        to the stream size (as the Pi's 640x480 stream relates to its recordings)."""
        return cv2.resize(self.native_from(atlas), self.out_size, interpolation=cv2.INTER_AREA)

    def frame(self, r: GLRenderer, R_world_cam: np.ndarray, p_world_cam: np.ndarray) -> np.ndarray:
        """Both stages at once (tools and checks)."""
        atlas = self.new_buffer()
        self.render_into(r, R_world_cam, p_world_cam, atlas)
        return self.stream_from(atlas)
