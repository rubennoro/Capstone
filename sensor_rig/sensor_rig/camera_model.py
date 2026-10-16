"""Camera models and physical <-> virtual remap tables.

Models (all frozen dataclasses, float64 math, vectorised over ``(N, 3)`` points / ``(N, 2)`` pixels):

* :class:`PinholeRadtan` - Kalibr ``pinhole``+``radtan`` == OpenCV ``distCoeffs = (k1, k2, p1, p2)``.
* :class:`FisheyeEquidistant` - Kalibr ``pinhole``+``equidistant`` == ``cv2.fisheye`` == ORB-SLAM3 KannalaBrandt8.
* :class:`VirtualPinhole` - distortion-free target camera (the iPhone cameras the rig reproduces).

Pixel convention: OpenCV/Kalibr, pixel ``(i, j)`` has its centre at ``(u=j, v=i)``. Uniform rescaling of a camera
uses the half-pixel convention of ``cv2.resize``: ``f' = s*f``, ``c' = s*(c + 0.5) - 0.5``.

``project`` returns ``(uv, valid)``. ``uv`` is NaN where the projection is undefined (behind the camera, or beyond
the model's monotonic range ``r_valid`` / ``theta_valid``, where distorted rays would fold back into the image).
Only :class:`VirtualPinhole` additionally requires the pixel to lie inside the image for ``valid``.

Remaps: :func:`build_remap` (physical image -> virtual image, used by export, deployment and calib-check) and
:func:`build_inverse_remap` (virtual image -> physical image, used by the simulator) share one implementation.
Destination pixels without a source sample get map value -16, so ``BORDER_CONSTANT`` paints them ``border_value``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Union

import cv2
import numpy as np

__all__ = [
    "PinholeRadtan", "FisheyeEquidistant", "VirtualPinhole", "CameraModel", "PhysicalModel", "RemapTable",
    "build_remap", "build_inverse_remap", "coverage", "coverage_mask", "virtual_to_physical",
    "virtual_border_in_physical", "model_from_dict", "model_to_dict", "model_from_kalibr", "scale_intrinsics",
]

_INVALID_MAP = -16.0                    # map coordinate for "no source sample" (outside even the bilinear footprint)
_RADTAN_R_CAP = math.tan(math.radians(89.0))


# --------------------------------------------------------------------------------------------- helpers

def scale_intrinsics(fx, fy, cx, cy, s: float):
    """Scale pinhole intrinsics by a uniform factor ``s`` (half-pixel convention, like ``cv2.resize``)."""
    return s * fx, s * fy, s * (cx + 0.5) - 0.5, s * (cy + 0.5) - 0.5


def _uniform_scale(w0: int, h0: int, w1: int, h1: int) -> float:
    if int(w1) != w1 or int(h1) != h1 or w1 <= 0 or h1 <= 0:
        raise ValueError(f"target size must be positive integers, got {(w1, h1)}")
    if w1 * h0 != h1 * w0:
        raise ValueError(f"non-uniform scaling {w0}x{h0} -> {w1}x{h1} (aspect ratio change) is not allowed")
    return w1 / w0


def _as_points(P) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P[None]
    if P.shape[-1] != 3:
        raise ValueError(f"expected (N,3) points, got shape {P.shape}")
    return P.reshape(-1, 3)


def _as_pixels(uv) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim == 1:
        uv = uv[None]
    if uv.shape[-1] != 2:
        raise ValueError(f"expected (N,2) pixels, got shape {uv.shape}")
    return uv.reshape(-1, 2)


def _first_nonpositive(g, hi: float, n: int = 20001) -> float:
    """Smallest x in (0, hi] with g(x) <= 0 (dense scan + bisection); ``hi`` if there is none."""
    x = np.linspace(0.0, hi, n)
    bad = np.nonzero(g(x) <= 0.0)[0]
    if len(bad) == 0:
        return float(hi)
    lo, up = x[bad[0] - 1], x[bad[0]]
    for _ in range(100):
        mid = 0.5 * (lo + up)
        if g(mid) > 0.0:
            lo = mid
        else:
            up = mid
    return float(lo)


def _monotone_inverse(f, df, target: np.ndarray, hi: float, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``f(x) = target`` for x in [0, hi], f increasing with derivative ``df`` (safeguarded Newton).

    Returns ``(x, solvable)``; ``solvable`` is False where ``target > f(hi)``.
    """
    solvable = target <= f(np.float64(hi)) * (1.0 + 1e-12)
    lo = np.zeros_like(target)
    up = np.full_like(target, hi)
    x = np.clip(target, 0.0, hi)
    for _ in range(iters):
        e = f(x) - target
        if np.max(np.abs(e[solvable]), initial=0.0) <= 4e-16 * max(1.0, float(hi)):
            break
        lo = np.where(e < 0, x, lo)
        up = np.where(e < 0, up, x)
        d = df(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = x - e / d
        bad = ~np.isfinite(xn) | (xn < lo) | (xn > up)
        x = np.where(bad, 0.5 * (lo + up), xn)
    return x, solvable


def _check_size_and_intrinsics(m) -> None:
    for name in ("width", "height"):
        v = getattr(m, name)
        if isinstance(v, bool) or int(v) != v or v <= 0:
            raise ValueError(f"{name}: must be a positive integer, got {v!r}")
        object.__setattr__(m, name, int(v))
    for name in ("fx", "fy", "cx", "cy"):
        v = float(getattr(m, name))
        if not math.isfinite(v):
            raise ValueError(f"{name}: must be finite, got {v!r}")
        object.__setattr__(m, name, v)
    if m.fx <= 0 or m.fy <= 0:
        raise ValueError(f"intrinsics: focal lengths must be > 0, got fx={m.fx}, fy={m.fy}")


class _Pinholeish:
    """Shared K / size helpers (mixin; subclasses are dataclasses with width, height, fx, fy, cx, cy)."""

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        return (self.width, self.height)

    def in_image(self, uv: np.ndarray) -> np.ndarray:
        """True where ``uv`` lies within the pixel-centre range [0, W-1] x [0, H-1] (bilinear-safe)."""
        uv = _as_pixels(uv)
        with np.errstate(invalid="ignore"):
            return (uv[:, 0] >= 0) & (uv[:, 0] <= self.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= self.height - 1)

    def _scaled_kwargs(self, width: int, height: int) -> dict:
        s = _uniform_scale(self.width, self.height, width, height)
        fx, fy, cx, cy = scale_intrinsics(self.fx, self.fy, self.cx, self.cy, s)
        return dict(width=int(width), height=int(height), fx=fx, fy=fy, cx=cx, cy=cy)


# --------------------------------------------------------------------------------------------- models

@dataclass(frozen=True, eq=False)
class PinholeRadtan(_Pinholeish):
    """Pinhole with radial-tangential distortion (Kalibr ``radtan``; OpenCV ``k1, k2, p1, p2``, k3 = 0)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    r_valid: float = field(init=False, repr=False)
    kind: ClassVar[str] = "pinhole-radtan"

    def __post_init__(self):
        _check_size_and_intrinsics(self)
        for name in ("k1", "k2", "p1", "p2"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"distortion: {name} must be finite")
            object.__setattr__(self, name, v)
        k1, k2 = self.k1, self.k2
        # Largest undistorted radius x/z on which r -> r*(1 + k1 r^2 + k2 r^4) is still increasing.
        r_valid = _first_nonpositive(lambda r: 1.0 + 3.0 * k1 * r**2 + 5.0 * k2 * r**4, _RADTAN_R_CAP)
        object.__setattr__(self, "r_valid", r_valid)

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2])

    def _distort(self, x, y):
        r2 = x * x + y * y
        rho = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        xd = x * rho + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * rho + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return xd, yd

    def project(self, P):
        P = _as_points(P)
        Z = P[:, 2]
        ok = Z > 1e-9
        Zs = np.where(ok, Z, 1.0)
        x, y = P[:, 0] / Zs, P[:, 1] / Zs
        ok &= x * x + y * y <= self.r_valid**2
        xd, yd = self._distort(x, y)
        uv = np.stack([self.fx * xd + self.cx, self.fy * yd + self.cy], axis=1)
        uv[~ok] = np.nan
        return uv, ok

    def unproject(self, uv):
        uv = _as_pixels(uv)
        xd = (uv[:, 0] - self.cx) / self.fx
        yd = (uv[:, 1] - self.cy) / self.fy
        k1, k2, p1, p2 = self.k1, self.k2, self.p1, self.p2
        # 1) radial-only initial guess by bisection on the monotonic branch, 2) Newton for the tangential terms.
        rd = np.hypot(xd, yd)
        r, ok = _monotone_inverse(lambda r: r * (1.0 + k1 * r * r + k2 * r**4),
                                  lambda r: 1.0 + 3.0 * k1 * r * r + 5.0 * k2 * r**4, rd, self.r_valid)
        s = np.where(rd > 1e-15, r / np.maximum(rd, 1e-300), 1.0)
        x, y = xd * s, yd * s
        if p1 != 0.0 or p2 != 0.0:
            for _ in range(20):
                fx_, fy_ = self._distort(x, y)
                ex, ey = fx_ - xd, fy_ - yd
                if np.max(np.abs(ex) + np.abs(ey), initial=0.0) < 1e-15:
                    break
                r2 = x * x + y * y
                rho = 1.0 + k1 * r2 + k2 * r2 * r2
                a = 2.0 * k1 + 4.0 * k2 * r2
                j11 = rho + a * x * x + 2.0 * p1 * y + 6.0 * p2 * x
                j12 = a * x * y + 2.0 * p1 * x + 2.0 * p2 * y
                j22 = rho + a * y * y + 6.0 * p1 * y + 2.0 * p2 * x
                det = j11 * j22 - j12 * j12
                det = np.where(np.abs(det) > 1e-300, det, 1e-300)
                x = x - (j22 * ex - j12 * ey) / det
                y = y - (j11 * ey - j12 * ex) / det
                rr = np.hypot(x, y)
                over = rr > self.r_valid
                x = np.where(over, x * self.r_valid / np.maximum(rr, 1e-300), x)
                y = np.where(over, y * self.r_valid / np.maximum(rr, 1e-300), y)
        fx_, fy_ = self._distort(x, y)
        ok &= (np.hypot(fx_ - xd, fy_ - yd) < 1e-10) & (x * x + y * y <= self.r_valid**2 * (1 + 1e-12))
        ok &= np.isfinite(x) & np.isfinite(y)
        rays = np.stack([x, y, np.ones_like(x)], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        rays[~ok] = np.nan
        return rays, ok

    def scaled(self, width: int, height: int) -> "PinholeRadtan":
        return PinholeRadtan(**self._scaled_kwargs(width, height), k1=self.k1, k2=self.k2, p1=self.p1, p2=self.p2)

    def to_kalibr(self) -> dict:
        return {"camera_model": "pinhole", "intrinsics": [self.fx, self.fy, self.cx, self.cy],
                "distortion_model": "radtan", "distortion_coeffs": [self.k1, self.k2, self.p1, self.p2],
                "resolution": [self.width, self.height]}

    @classmethod
    def from_kalibr(cls, d: dict) -> "PinholeRadtan":
        _check_kalibr(d, "radtan")
        (fx, fy, cx, cy), (w, h) = d["intrinsics"], d["resolution"]
        return cls(w, h, fx, fy, cx, cy, *_coeffs(d["distortion_coeffs"], 4, "distortion_coeffs"))


@dataclass(frozen=True, eq=False)
class FisheyeEquidistant(_Pinholeish):
    """Equidistant fisheye (Kalibr ``equidistant``; ``cv2.fisheye`` D = k1..k4; KannalaBrandt8)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    theta_max_deg: float = 89.0
    theta_valid: float = field(init=False, repr=False)
    kind: ClassVar[str] = "fisheye-equidistant"

    def __post_init__(self):
        _check_size_and_intrinsics(self)
        for name in ("k1", "k2", "k3", "k4", "theta_max_deg"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"distortion: {name} must be finite")
            object.__setattr__(self, name, v)
        if not 0.0 < self.theta_max_deg < 90.0:
            raise ValueError(f"theta_max_deg: must be in (0, 90), got {self.theta_max_deg}")
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4
        g = lambda t: 1 + 3 * k1 * t**2 + 5 * k2 * t**4 + 7 * k3 * t**6 + 9 * k4 * t**8  # noqa: E731
        object.__setattr__(self, "theta_valid", _first_nonpositive(g, math.radians(self.theta_max_deg)))

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3, self.k4])

    def _theta_d(self, t):
        t2 = t * t
        return t * (1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))

    def _dtheta_d(self, t):
        t2 = t * t
        return 1.0 + t2 * (3 * self.k1 + t2 * (5 * self.k2 + t2 * (7 * self.k3 + t2 * 9 * self.k4)))

    def project(self, P):
        P = _as_points(P)
        X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
        r = np.hypot(X, Y)
        theta = np.arctan2(r, Z)
        ok = (Z > 0) & (theta <= self.theta_valid)
        td = self._theta_d(theta)
        s = np.where(r > 1e-12, td / np.maximum(r, 1e-300), 0.0)
        uv = np.stack([self.fx * X * s + self.cx, self.fy * Y * s + self.cy], axis=1)
        uv[~ok] = np.nan
        return uv, ok

    def unproject(self, uv):
        uv = _as_pixels(uv)
        xd = (uv[:, 0] - self.cx) / self.fx
        yd = (uv[:, 1] - self.cy) / self.fy
        td = np.hypot(xd, yd)
        theta, ok = _monotone_inverse(self._theta_d, self._dtheta_d, td, self.theta_valid)
        ok &= np.abs(self._theta_d(theta) - td) < 1e-12
        s = np.where(td > 1e-15, np.sin(theta) / np.maximum(td, 1e-300), 1.0)
        rays = np.stack([xd * s, yd * s, np.cos(theta)], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        rays[~ok] = np.nan
        return rays, ok

    def scaled(self, width: int, height: int) -> "FisheyeEquidistant":
        return FisheyeEquidistant(**self._scaled_kwargs(width, height), k1=self.k1, k2=self.k2, k3=self.k3,
                                  k4=self.k4, theta_max_deg=self.theta_max_deg)

    def to_kalibr(self) -> dict:
        return {"camera_model": "pinhole", "intrinsics": [self.fx, self.fy, self.cx, self.cy],
                "distortion_model": "equidistant", "distortion_coeffs": [self.k1, self.k2, self.k3, self.k4],
                "resolution": [self.width, self.height]}

    @classmethod
    def from_kalibr(cls, d: dict) -> "FisheyeEquidistant":
        _check_kalibr(d, "equidistant")
        (fx, fy, cx, cy), (w, h) = d["intrinsics"], d["resolution"]
        return cls(w, h, fx, fy, cx, cy, *_coeffs(d["distortion_coeffs"], 4, "distortion_coeffs"))


@dataclass(frozen=True, eq=False)
class VirtualPinhole(_Pinholeish):
    """Distortion-free pinhole camera (virtual iPhone camera). ``valid`` from ``project`` includes image bounds."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    kind: ClassVar[str] = "pinhole"

    def __post_init__(self):
        _check_size_and_intrinsics(self)

    @classmethod
    def from_K(cls, K, size: tuple[int, int]) -> "VirtualPinhole":
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        if abs(K[0, 1]) > 1e-9 or np.any(np.abs(K[2] - [0, 0, 1]) > 1e-9) or abs(K[1, 0]) > 1e-9:
            raise ValueError(f"K must be [[fx,0,cx],[0,fy,cy],[0,0,1]], got {K.tolist()}")
        return cls(int(size[0]), int(size[1]), K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    def project(self, P):
        P = _as_points(P)
        Z = P[:, 2]
        front = Z > 1e-9
        Zs = np.where(front, Z, 1.0)
        uv = np.stack([self.fx * P[:, 0] / Zs + self.cx, self.fy * P[:, 1] / Zs + self.cy], axis=1)
        uv[~front] = np.nan
        return uv, front & self.in_image(uv)

    def unproject(self, uv):
        uv = _as_pixels(uv)
        rays = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy,
                         np.ones(len(uv))], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays, np.isfinite(rays).all(axis=1)

    def scaled(self, width: int, height: int) -> "VirtualPinhole":
        return VirtualPinhole(**self._scaled_kwargs(width, height))

    def K_float32_rows(self) -> list[list[float]]:
        """K as the iPhone app writes it into the export JSON: float32 values, printed shortest (``0``/``1`` as ints).

        ``json.dumps`` of the result reproduces Swift's ``JSONEncoder`` text, e.g. ``1406.3428`` and ``0``.
        """
        def f32(v: float):
            x = float(str(np.float32(v)))
            return int(x) if x.is_integer() else x
        return [[f32(self.fx), 0, f32(self.cx)], [0, f32(self.fy), f32(self.cy)], [0, 0, 1]]


PhysicalModel = Union[PinholeRadtan, FisheyeEquidistant]
CameraModel = Union[PinholeRadtan, FisheyeEquidistant, VirtualPinhole]
_MODEL_TYPES = {"pinhole-radtan": PinholeRadtan, "fisheye-equidistant": FisheyeEquidistant, "pinhole": VirtualPinhole}


# --------------------------------------------------------------------------------------------- dict / kalibr I/O

def _coeffs(vals, n: int, name: str) -> list[float]:
    vals = list(vals) if vals is not None else []
    if len(vals) != n:
        raise ValueError(f"{name}: expected {n} coefficients, got {len(vals)}")
    return [float(v) for v in vals]


def _check_kalibr(d: dict, distortion_model: str) -> None:
    if d.get("camera_model") != "pinhole" or d.get("distortion_model") != distortion_model:
        raise ValueError(f"expected Kalibr camera_model 'pinhole' + distortion_model '{distortion_model}', got "
                         f"{d.get('camera_model')!r} + {d.get('distortion_model')!r}")


def model_from_kalibr(d: dict) -> PhysicalModel:
    """Physical model from one Kalibr camchain camera entry (``pinhole-radtan`` or ``pinhole-equi`` only)."""
    cm, dm = d.get("camera_model"), d.get("distortion_model")
    if cm != "pinhole":
        raise NotImplementedError(f"Kalibr camera_model {cm!r} is not supported; recalibrate with "
                                  "pinhole-radtan (main) or pinhole-equi (ultrawide)")
    if dm == "radtan":
        return PinholeRadtan.from_kalibr(d)
    if dm == "equidistant":
        return FisheyeEquidistant.from_kalibr(d)
    raise NotImplementedError(f"Kalibr distortion_model {dm!r} is not supported; recalibrate with "
                              "pinhole-radtan (main) or pinhole-equi (ultrawide)")


def model_from_dict(d: dict, size: tuple[int, int]) -> CameraModel:
    """Model from a rig.yaml ``model:`` block (``type``, ``intrinsics``, ``distortion``, opt. ``theta_max_deg``).

    Errors are ``ValueError`` whose message starts with the offending key (callers prefix the yaml path).
    """
    if not isinstance(d, dict):
        raise ValueError(f": must be a mapping, got {type(d).__name__}")
    typ = d.get("type")
    if typ not in _MODEL_TYPES:
        raise ValueError(f"type: must be one of {sorted(_MODEL_TYPES)}, got {typ!r}")
    allowed = {"pinhole": {"type", "intrinsics"}, "pinhole-radtan": {"type", "intrinsics", "distortion"},
               "fisheye-equidistant": {"type", "intrinsics", "distortion", "theta_max_deg"}}[typ]
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"{sorted(unknown)[0]}: unknown key")
    intr = d.get("intrinsics")
    if not isinstance(intr, (list, tuple)) or len(intr) != 4:
        raise ValueError(f"intrinsics: expected [fu, fv, pu, pv], got {intr!r}")
    try:
        fx, fy, cx, cy = (float(v) for v in intr)
    except (TypeError, ValueError):
        raise ValueError(f"intrinsics: must be numbers, got {intr!r}") from None
    w, h = size
    if typ == "pinhole":
        return VirtualPinhole(w, h, fx, fy, cx, cy)
    dist = d.get("distortion")
    if not isinstance(dist, (list, tuple)):
        raise ValueError(f"distortion: expected a list of 4 numbers, got {dist!r}")
    try:
        k = _coeffs(dist, 4, "distortion")
    except (TypeError, ValueError) as e:
        raise ValueError(str(e) if str(e).startswith("distortion") else f"distortion: {e}") from None
    if typ == "pinhole-radtan":
        return PinholeRadtan(w, h, fx, fy, cx, cy, *k)
    return FisheyeEquidistant(w, h, fx, fy, cx, cy, *k, theta_max_deg=float(d.get("theta_max_deg", 89.0)))


def model_to_dict(m: CameraModel) -> dict:
    """Inverse of :func:`model_from_dict` (the resolution lives next to the block in rig.yaml)."""
    d = {"type": m.kind, "intrinsics": [m.fx, m.fy, m.cx, m.cy]}
    if isinstance(m, (PinholeRadtan, FisheyeEquidistant)):
        d["distortion"] = [float(v) for v in m.dist]
    if isinstance(m, FisheyeEquidistant):
        d["theta_max_deg"] = m.theta_max_deg
    return d


# --------------------------------------------------------------------------------------------- remaps

@dataclass(frozen=True, eq=False)
class RemapTable:
    """Precomputed ``cv2.remap`` maps from a source image of ``src_size`` to a destination of ``dst_size``.

    Use only :meth:`apply`, ``valid``, ``coverage``, ``src_size`` and ``dst_size``; the map layout is an
    implementation detail (``CV_16SC2`` + ``uint16`` by default, float32 x/y maps with ``fixed_point=False``).
    """

    src_size: tuple[int, int]
    dst_size: tuple[int, int]
    map1: np.ndarray
    map2: np.ndarray
    valid: np.ndarray
    coverage: float

    def apply(self, src: np.ndarray, out: np.ndarray | None = None, interpolation: int = cv2.INTER_LINEAR,
              border_value: int = 0) -> np.ndarray:
        """Remap ``src`` (uint8, C-contiguous, HxW or HxWxC); uncovered pixels get ``border_value``."""
        sw, sh = self.src_size
        if src.shape[:2] != (sh, sw):
            raise ValueError(f"remap expects a {sw}x{sh} source image, got shape {src.shape}")
        if src.dtype != np.uint8:
            raise TypeError(f"remap expects uint8 images, got {src.dtype}")
        if not src.flags.c_contiguous:
            raise ValueError("remap expects a C-contiguous source image")
        bv = tuple(border_value) if isinstance(border_value, (tuple, list)) else (border_value,) * 4
        return cv2.remap(src, self.map1, self.map2, interpolation, dst=out,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=bv)


def _rot_or_none(R) -> np.ndarray | None:
    if R is None:
        return None
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (4, 4):
        R = R[:3, :3]
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {R.shape}")
    return None if np.array_equal(R, np.eye(3)) else R


def _pixel_grid(w: int, h: int) -> np.ndarray:
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.stack([u.ravel(), v.ravel()], axis=1)


def _map_rays(src, dst, R_src_dst, uv_dst):
    """For pixels of ``dst``: their position in ``src`` (rays rotated by ``R_src_dst``); (uv_src, model_ok)."""
    rays, ok = dst.unproject(uv_dst)
    if R_src_dst is not None:
        rays = rays @ R_src_dst.T
    uv_src, ok_src = src.project(rays)
    return uv_src, ok & ok_src


def _table(src, dst, R_src_dst, fixed_point: bool) -> RemapTable:
    W, H = dst.width, dst.height
    uv_src, ok = _map_rays(src, dst, _rot_or_none(R_src_dst), _pixel_grid(W, H))
    valid = ok & src.in_image(uv_src)
    mx = np.where(valid, uv_src[:, 0], _INVALID_MAP).astype(np.float32).reshape(H, W)
    my = np.where(valid, uv_src[:, 1], _INVALID_MAP).astype(np.float32).reshape(H, W)
    if fixed_point:
        mx, my = cv2.convertMaps(mx, my, cv2.CV_16SC2)
    valid = valid.reshape(H, W)
    for a in (mx, my, valid):
        a.flags.writeable = False
    return RemapTable(src_size=src.size, dst_size=dst.size, map1=mx, map2=my, valid=valid,
                      coverage=float(valid.mean()))


def build_remap(phys: CameraModel, virt: VirtualPinhole, R_phys_virt: np.ndarray | None = None, *,
                fixed_point: bool = True) -> RemapTable:
    """FORWARD remap: physical image -> virtual image (export, deployment, calib-check).

    ``R_phys_virt`` rotates virtual ray directions into the physical camera frame (identity by default).
    """
    return _table(phys, virt, R_phys_virt, fixed_point)


def build_inverse_remap(virt: VirtualPinhole, phys: CameraModel, R_phys_virt: np.ndarray | None = None, *,
                        fixed_point: bool = True) -> RemapTable:
    """INVERSE remap: virtual (iPhone) image -> physical rig image (simulator). ``valid`` = physical pixels seen
    by the virtual image."""
    R = _rot_or_none(R_phys_virt)
    return _table(virt, phys, None if R is None else R.T, fixed_point)


def virtual_to_physical(phys: CameraModel, virt: VirtualPinhole, uv_virt, R_phys_virt=None):
    """Exact float64 positions in the physical image of virtual pixels ``uv_virt`` (N,2).

    Returns ``(uv_phys, ok)``; ``ok`` is model validity only (it does not check the physical image bounds).
    """
    return _map_rays(phys, virt, _rot_or_none(R_phys_virt), _as_pixels(uv_virt))


def coverage_mask(phys: CameraModel, virt: VirtualPinhole, R_phys_virt=None) -> np.ndarray:
    """(virt.height, virt.width) bool: virtual pixels the physical camera actually sees."""
    uv, ok = virtual_to_physical(phys, virt, _pixel_grid(virt.width, virt.height), R_phys_virt)
    return (ok & phys.in_image(uv)).reshape(virt.height, virt.width)


def coverage(phys: CameraModel, virt: VirtualPinhole, R_phys_virt=None) -> float:
    """Fraction of virtual pixels with a physical sample (== ``build_remap(...).coverage``)."""
    return float(coverage_mask(phys, virt, R_phys_virt).mean())


def virtual_border_in_physical(phys: CameraModel, virt: VirtualPinhole, R_phys_virt=None,
                               n_per_side: int = 64) -> np.ndarray:
    """Closed polyline (M,2) of the virtual image border in physical pixels, for overlays.

    Points may lie outside the physical image; points the model cannot project are NaN (split the polyline there).
    """
    W, H = virt.width - 1, virt.height - 1
    t = np.linspace(0.0, 1.0, n_per_side, endpoint=False)
    border = np.concatenate([
        np.stack([t * W, np.zeros_like(t)], 1), np.stack([np.full_like(t, W), t * H], 1),
        np.stack([W - t * W, np.full_like(t, H)], 1), np.stack([np.zeros_like(t), H - t * H], 1),
        [[0.0, 0.0]],
    ])
    uv, ok = virtual_to_physical(phys, virt, border, R_phys_virt)
    uv[~ok] = np.nan
    return uv
