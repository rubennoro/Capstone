"""Vectorised ray-cast renderer of a textured box room, through the rig's exact camera models (docs/SIM.md).

* Scene (:func:`build_scene`): a 3 x 3 x 2.5 m room (``x, y in [-1.5, 1.5]``, ``z in [0, 2.5]``, gravity-aligned
  z-up), a table, a block on it and a few boxes, all axis-aligned. Every face has its own seeded texture
  (multi-octave band-limited noise + hundreds of random anti-aliased shapes), packed into one RGB atlas at
  ``texels_per_m`` (default 400, 2.5 mm per texel).
* Planar quads (ArUco tags with a white quiet zone): the two finger tags, rigid in the ultrawide frame, and optional
  world-fixed markers (tests). Double-sided, square, textured from the atlas.
* Rendering (:class:`CameraRenderer`): rays are the model's ``unproject`` of a ``ss x ss`` supersampled pixel grid
  (computed once per camera), so the image geometry is exactly the camera model. A numba kernel intersects every ray
  with the scene (nearest hit wins) and returns atlas coordinates; ``cv2.remap`` (bilinear) samples the atlas, and
  ``cv2.resize(INTER_AREA)`` averages the ``ss x ss`` samples into one pixel. Rays outside the model's valid range
  or hitting nothing are black.

Supersample ``k`` of output pixel ``(u, v)`` sits at ``u + (k + 0.5) / ss - 0.5``: the box average is centred on the
pixel centre (OpenCV convention, pixel ``(i, j)`` centred at ``(u=j, v=i)``).
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
from numba import njit, prange
from numba.core.errors import NumbaWarning

from dataproc.geometry.camera_model import model_from_dict

__all__ = ["Scene", "Quad", "CameraRenderer", "build_scene", "camera_model", "marker_image", "ROOM_LO", "ROOM_HI",
           "TABLE", "BLOCK", "QUIET_ZONE_CELLS"]

# numba falls back from an old system TBB to OpenMP / workqueue by itself; the warning is noise.
warnings.filterwarnings("ignore", message="The TBB threading layer", category=NumbaWarning)

ROOM_LO = (-1.5, -1.5, 0.0)
ROOM_HI = (1.5, 1.5, 2.5)
TABLE = ((0.30, -0.70, 0.0), (1.10, 0.70, 0.75))
BLOCK = ((0.56, -0.06, 0.75), (0.68, 0.06, 0.79))       # the "eraser"
#: Width of the white quiet zone around an ArUco tag, in marker cells (a 4x4 tag is 6 cells incl. its black border).
QUIET_ZONE_CELLS = 1
_CELL_PX = 24                                            # atlas pixels per marker cell
_PAD = 4                                                 # atlas padding (edge replicate) around each texture
_MISS = -1.0e5                                           # map value for "no hit" (remap paints it black)


def camera_model(cam):
    """Vendored camera model (``PinholeRadtan`` / ``FisheyeEquidistant``) of a ``config.CameraConfig``."""
    return model_from_dict({"type": cam.model.type, "intrinsics": list(cam.model.intrinsics),
                            "distortion": list(cam.model.distortion)}, cam.resolution)


# --------------------------------------------------------------------------------------------- textures

def _octave_noise(rng, h, w, finest=2.0, coarsest=256.0) -> np.ndarray:
    """Band-limited multi-octave value noise in [0, 1] (pink-ish: coarser octaves weigh more)."""
    out = np.zeros((h, w), np.float32)
    s, tot = finest, 0.0
    while s <= coarsest:
        gh, gw = max(2, int(np.ceil(h / s)) + 2), max(2, int(np.ceil(w / s)) + 2)
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (int(gw * s), int(gh * s)), interpolation=cv2.INTER_CUBIC)[:h, :w]
        wgt = s**0.35
        out += wgt * up
        tot += wgt
        s *= 2
    out /= tot
    lo, hi = np.percentile(out, (1, 99))
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def _face_texture(rng, h, w, ppm) -> np.ndarray:
    """Unique, natural-ish RGB texture of h x w texels (``ppm`` texels per metre)."""
    base = rng.uniform(70, 190, 3)
    tint = rng.uniform(-50, 50, 3)
    n = _octave_noise(rng, h, w)
    img = (base[None, None] + tint[None, None] * (2 * n[..., None] - 1)
           + 25 * (2 * _octave_noise(rng, h, w, 2.0, 16.0)[..., None] - 1)).astype(np.float32)
    img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
    area_m2 = (h / ppm) * (w / ppm)
    n_shapes = int(np.clip(90 * area_m2, 20, 1200))
    for _ in range(n_shapes):
        col = tuple(int(c) for c in rng.integers(0, 256, 3))
        size = float(np.exp(rng.uniform(np.log(0.012), np.log(0.25)))) * ppm      # 1.2 cm .. 25 cm
        cx, cy = rng.uniform(-0.1 * w, 1.1 * w), rng.uniform(-0.1 * h, 1.1 * h)
        kind = rng.integers(0, 5)
        S = 4                                                                      # cv2 sub-pixel shift bits
        P = lambda x, y: (int(round(x * 2**S)), int(round(y * 2**S)))              # noqa: E731
        if kind == 0:
            cv2.circle(img, P(cx, cy), int(size / 2 * 2**S), col, -1, cv2.LINE_AA, S)
        elif kind == 1:
            ang = rng.uniform(0, 180)
            cv2.ellipse(img, P(cx, cy), (int(size / 2 * 2**S), int(size * rng.uniform(0.15, 0.5) * 2**S)), ang, 0,
                        360, col, -1, cv2.LINE_AA, S)
        elif kind == 2:
            k = int(rng.integers(3, 7))
            a = np.sort(rng.uniform(0, 2 * np.pi, k))
            r = size / 2 * rng.uniform(0.4, 1.0, k)
            pts = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1) * 2**S
            cv2.fillPoly(img, [pts.round().astype(np.int32)], col, cv2.LINE_AA, S)
        elif kind == 3:
            ang = rng.uniform(0, np.pi)
            dx, dy = np.cos(ang) * size, np.sin(ang) * size
            cv2.line(img, P(cx - dx, cy - dy), P(cx + dx, cy + dy), col, max(1, int(size * rng.uniform(0.02, 0.12))),
                     cv2.LINE_AA, S)
        else:
            ww, hh = size * rng.uniform(0.3, 1.0), size * rng.uniform(0.3, 1.0)
            box = cv2.boxPoints(((cx, cy), (ww, hh), rng.uniform(0, 90))) * 2**S
            cv2.fillPoly(img, [box.round().astype(np.int32)], col, cv2.LINE_AA, S)
    # a light fine-grain layer on top so shapes are not perfectly flat
    grain = (18 * (2 * _octave_noise(rng, h, w, 2.0, 8.0) - 1)).astype(np.float32)
    return np.clip(img.astype(np.float32) + grain[..., None], 0, 255).astype(np.uint8)


def marker_image(aruco_dict: str, marker_id: int) -> np.ndarray:
    """RGB image of a marker with its white quiet zone, ``_CELL_PX`` pixels per cell."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
    cells = d.markerSize + 2
    m = cv2.aruco.generateImageMarker(d, int(marker_id), cells * _CELL_PX)
    q = QUIET_ZONE_CELLS * _CELL_PX
    m = cv2.copyMakeBorder(m, q, q, q, q, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)


def quad_half_extent(aruco_dict: str, marker_size_m: float) -> float:
    """Half side (m) of the tag quad incl. the quiet zone, for a black square of ``marker_size_m``."""
    cells = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict)).markerSize + 2
    return 0.5 * marker_size_m * (cells + 2 * QUIET_ZONE_CELLS) / cells


# --------------------------------------------------------------------------------------------- scene

@dataclass(frozen=True)
class Quad:
    """A square textured quad: pose ``T_W_quad`` (quad x right, y up, z out of the front face, centred), half side
    ``half_m``, atlas rect (``u0, v0``, ``size_px`` square)."""

    T_W_quad: np.ndarray
    half_m: float
    rect: tuple[float, float, float]

    def row(self) -> np.ndarray:
        T = self.T_W_quad
        return np.concatenate([T[:3, 3], T[:3, 0], T[:3, 1], [self.half_m, *self.rect]])


@dataclass(frozen=True, eq=False)
class Scene:
    atlas: np.ndarray            # (H, W, 3) uint8
    boxes: np.ndarray            # (B, 6) lo xyz, hi xyz (box 0 is the room, seen from inside)
    inside: np.ndarray           # (B,) int64: 1 = rays start inside the box (the room)
    face_tex: np.ndarray         # (B, 6, 3): u0, v0, texels_per_m for face (axis * 2 + side), side 0 = lo
    marker_rects: dict           # marker id -> (u0, v0, size_px) in the atlas
    aruco_dict: str
    texels_per_m: float
    seed: int
    meta: dict = field(default_factory=dict)


def _face_dims(lo, hi, axis):
    a, b = [(1, 2), (0, 2), (0, 1)][axis]
    return hi[a] - lo[a], hi[b] - lo[b]


@functools.lru_cache(maxsize=4)
def build_scene(seed: int = 0, texels_per_m: float = 400.0, aruco_dict: str = "DICT_4X4_50",
                marker_ids: tuple[int, ...] = tuple(range(8))) -> Scene:
    """The (cached) seeded room. ``marker_ids`` get atlas slots for tags (finger tags and test markers)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xB0C5]))
    boxes = [(ROOM_LO, ROOM_HI), TABLE, BLOCK,
             ((0.55, -1.45, 0.0), (1.30, -0.95, 1.10)),                 # cabinet against the -y wall
             ((-1.45, 0.60, 0.0), (-0.95, 1.40, 0.60)),                 # crate behind the start pose
             ((0.80, 0.25, 0.75), (1.02, 0.55, 1.05)),                  # box at the back of the table
             ((0.40, 0.95, 0.0), (0.85, 1.45, 0.45))]                   # box on the floor, +y side
    ppm = float(texels_per_m)
    textures = []                                                       # (box, face, img)
    for bi, (lo, hi) in enumerate(boxes):
        for axis in range(3):
            for side in range(2):
                wa, hb = _face_dims(lo, hi, axis)
                h, w = max(8, int(np.ceil(hb * ppm))), max(8, int(np.ceil(wa * ppm)))
                textures.append((bi, axis * 2 + side, _face_texture(rng, h, w, ppm)))
    marker_imgs = [(mid, marker_image(aruco_dict, mid)) for mid in marker_ids]

    # shelf packing into the atlas
    items = [(img.shape[0], img.shape[1], ("face", bi, f), img) for bi, f, img in textures]
    items += [(img.shape[0], img.shape[1], ("marker", mid), img) for mid, img in marker_imgs]
    order = sorted(range(len(items)), key=lambda i: -items[i][0])
    W = 4096
    x = y = row_h = 0
    pos = {}
    for i in order:
        h, w = items[i][0] + 2 * _PAD, items[i][1] + 2 * _PAD
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        pos[i] = (x, y)
        x += w
        row_h = max(row_h, h)
    H = y + row_h
    atlas = np.zeros((H, W, 3), np.uint8)
    face_tex = np.zeros((len(boxes), 6, 3))
    marker_rects = {}
    for i, (h, w, key, img) in enumerate(items):
        x0, y0 = pos[i]
        atlas[y0:y0 + h + 2 * _PAD, x0:x0 + w + 2 * _PAD] = cv2.copyMakeBorder(img, _PAD, _PAD, _PAD, _PAD,
                                                                               cv2.BORDER_REPLICATE)
        if key[0] == "face":
            face_tex[key[1], key[2]] = (x0 + _PAD, y0 + _PAD, ppm)
        else:
            marker_rects[key[1]] = (float(x0 + _PAD), float(y0 + _PAD), float(w))
    atlas.flags.writeable = False
    return Scene(atlas=atlas, boxes=np.array([[*lo, *hi] for lo, hi in boxes], float),
                 inside=np.array([1] + [0] * (len(boxes) - 1), np.int64), face_tex=face_tex,
                 marker_rects=marker_rects, aruco_dict=aruco_dict, texels_per_m=ppm, seed=int(seed),
                 meta={"boxes": boxes})


# --------------------------------------------------------------------------------------------- kernel

@njit(cache=True, inline="always")
def _face_uv(px, py, pz, boxes, b, axis, tex):
    """Atlas (u, v) of world point P on face ``axis`` of box ``b``: in-plane axes (a, b) = the other two axes in
    order, u along +a from lo[a], v along -b from hi[b] (so walls are upright in the atlas)."""
    if axis == 0:
        pa, pb, ia, ib = py, pz, 1, 2
    elif axis == 1:
        pa, pb, ia, ib = px, pz, 0, 2
    else:
        pa, pb, ia, ib = px, py, 0, 1
    ppm = tex[2]
    return tex[0] + (pa - boxes[b, ia]) * ppm - 0.5, tex[1] + (boxes[b, 3 + ib] - pb) * ppm - 0.5


@njit(parallel=True, cache=True, fastmath=False)
def _trace(rays, R, o, boxes, inside, face_tex, quads, out_u, out_v):
    n = rays.shape[0]
    nb = boxes.shape[0]
    nq = quads.shape[0]
    o0, o1, o2 = o[0], o[1], o[2]
    for i in prange(n):
        rx, ry, rz = np.float64(rays[i, 0]), np.float64(rays[i, 1]), np.float64(rays[i, 2])
        best = np.inf
        bu = _MISS
        bv = _MISS
        if rx == rx:                                     # not NaN (valid model ray)
            d0 = R[0, 0] * rx + R[0, 1] * ry + R[0, 2] * rz
            d1 = R[1, 0] * rx + R[1, 1] * ry + R[1, 2] * rz
            d2 = R[2, 0] * rx + R[2, 1] * ry + R[2, 2] * rz
            for b in range(nb):
                tmin = -np.inf
                tmax = np.inf
                amin = -1
                smin = 0
                amax = -1
                smax = 0
                ok = True
                for k in range(3):
                    if k == 0:
                        dk, ok_ = d0, o0
                    elif k == 1:
                        dk, ok_ = d1, o1
                    else:
                        dk, ok_ = d2, o2
                    lo, hi = boxes[b, k], boxes[b, 3 + k]
                    if abs(dk) < 1e-12:
                        if ok_ < lo or ok_ > hi:
                            ok = False
                    else:
                        t1 = (lo - ok_) / dk                    # crossing of the lo face
                        t2 = (hi - ok_) / dk
                        s1, s2 = 0, 1
                        if t1 > t2:
                            t1, t2 = t2, t1
                            s1, s2 = 1, 0
                        if t1 > tmin:
                            tmin, amin, smin = t1, k, s1
                        if t2 < tmax:
                            tmax, amax, smax = t2, k, s2
                if not ok:
                    continue
                if inside[b] == 1:                              # seen from inside: the exit face
                    t, ax, sd = tmax, amax, smax
                else:                                           # solid box: the entry face
                    t, ax, sd = tmin, amin, smin
                if ax >= 0 and tmin <= tmax and 1e-9 < t < best:
                    best = t
                    bu, bv = _face_uv(o0 + t * d0, o1 + t * d1, o2 + t * d2, boxes, b, ax,
                                      face_tex[b, ax * 2 + sd])
            for q in range(nq):
                c0, c1, c2 = quads[q, 0], quads[q, 1], quads[q, 2]
                ex0, ex1, ex2 = quads[q, 3], quads[q, 4], quads[q, 5]
                ey0, ey1, ey2 = quads[q, 6], quads[q, 7], quads[q, 8]
                h = quads[q, 9]
                n0 = ex1 * ey2 - ex2 * ey1
                n1 = ex2 * ey0 - ex0 * ey2
                n2 = ex0 * ey1 - ex1 * ey0
                den = d0 * n0 + d1 * n1 + d2 * n2
                if abs(den) > 1e-12:
                    t = ((c0 - o0) * n0 + (c1 - o1) * n1 + (c2 - o2) * n2) / den
                    if 1e-9 < t < best:
                        p0 = o0 + t * d0 - c0
                        p1 = o1 + t * d1 - c1
                        p2 = o2 + t * d2 - c2
                        a = p0 * ex0 + p1 * ex1 + p2 * ex2
                        bb = p0 * ey0 + p1 * ey1 + p2 * ey2
                        if abs(a) <= h and abs(bb) <= h:
                            best = t
                            sz = quads[q, 12]
                            bu = quads[q, 10] + (a + h) / (2 * h) * sz - 0.5
                            bv = quads[q, 11] + (h - bb) / (2 * h) * sz - 0.5
        out_u[i] = bu
        out_v[i] = bv


# --------------------------------------------------------------------------------------------- camera

class CameraRenderer:
    """Renders one camera model (``PinholeRadtan`` / ``FisheyeEquidistant`` / ``VirtualPinhole``)."""

    def __init__(self, model, ss: int = 2):
        if ss < 1:
            raise ValueError("ss must be >= 1")
        self.model, self.ss = model, int(ss)
        W, H = model.width, model.height
        Ws, Hs = W * self.ss, H * self.ss
        off = (np.arange(self.ss) + 0.5) / self.ss - 0.5
        us = (np.arange(W)[:, None] + off[None]).ravel()
        vs = (np.arange(H)[:, None] + off[None]).ravel()
        uu, vv = np.meshgrid(us, vs)
        rays, ok = model.unproject(np.stack([uu.ravel(), vv.ravel()], 1))
        rays[~ok] = np.nan
        self.rays = np.ascontiguousarray(rays.astype(np.float32))
        self.shape_ss = (Hs, Ws)
        self._u = np.empty(Hs * Ws, np.float32)
        self._v = np.empty(Hs * Ws, np.float32)

    def maps(self, scene: Scene, T_W_cam: np.ndarray, quads=()) -> tuple[np.ndarray, np.ndarray]:
        """Atlas coordinate maps (Hs, Ws) float32 for camera pose ``T_W_cam``."""
        Q = np.array([q.row() for q in quads], float).reshape(-1, 13)
        T = np.asarray(T_W_cam, float)
        with warnings.catch_warnings():                  # (pytest resets the module-level filter)
            warnings.filterwarnings("ignore", message="The TBB threading layer", category=NumbaWarning)
            _trace(self.rays, np.ascontiguousarray(T[:3, :3]), np.ascontiguousarray(T[:3, 3]), scene.boxes,
                   scene.inside, scene.face_tex, Q, self._u, self._v)
        return self._u.reshape(self.shape_ss), self._v.reshape(self.shape_ss)

    def render(self, scene: Scene, T_W_cam: np.ndarray, quads=()) -> np.ndarray:
        """(H, W, 3) uint8 RGB image."""
        mu, mv = self.maps(scene, T_W_cam, quads)
        img = cv2.remap(scene.atlas, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if self.ss > 1:
            img = cv2.resize(img, (self.model.width, self.model.height), interpolation=cv2.INTER_AREA)
        return img
