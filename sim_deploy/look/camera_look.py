"""The Pi cameras' image formation as an opt-in post-process on the sim Pi's 640x480 RGB frames (W2 step 2).

The sim renders a clean, noise-free, sharp image of the scene. A real camera adds its own colour response, tone
curve, lens colour shading / vignetting, sensor noise and a little blur. ``CameraLook.apply(role, rgb640, rng)`` adds
those, per camera (``main`` = IMX519, ``ultrawide`` = IMX219), to the frame the sim Pi hands to its JPEG encoder
(after the native render and INTER_AREA to 640x480, before JPEG q90; neither is repeated here). Parameters: a YAML
file (``camera_look_real.yaml``, fitted by ``fit_camera_look.py``), named by a station's optional
``sim_pi.camera_look`` key; without that key nothing here is imported and the sim Pi's frames are unchanged.

Per camera, in this order (any block may be absent = identity; a role missing from ``cameras:`` = pass-through)::

    x   = GaussianBlur(rgb, blur_sigma_px)                       lens / demosaic / codec softness, at 640
                                                                 (kernel truncated at +-2 sigma, for speed)
    lin = (x / 255) ** decode_gamma                              the render's RGB taken as display-referred
    lin = matrix @ lin + offset                                  colour response / white balance / black level
    lin = lin * shading(u, v)                                    colour shading + vignetting (per channel)
    s   = sqrt(clip(lin, 0, 1)) + (noise.s0[c] + noise.s1[c] * sqrt(clip(lin, 0, 1))) * n
                                                                 signal-dependent sensor noise, added in the
                                                                 sqrt(linear) domain (about the display domain)
    out = 255 * clip(s, 0, 1) ** (2 / tone_gamma[c])             tone curve per channel (LUT), uint8

``shading``: ``exp(sum_k radial[c][k] * rho^(2k+2) + tilt[c][0] * X + tilt[c][1] * Y)`` with
``X, Y = (u - cx, v - cy) / norm_px`` and ``rho^2 = X^2 + Y^2`` (pixel centres at integers, ``centre_px`` = (cx, cy));
with ``rho_max`` (optional) X, Y are pulled in to rho_max along each ray (the map is constant beyond it: the fitted
polynomial is not extrapolated past the data).
``n``: unit-variance Gaussian noise, ``noise.spatial_sigma_px`` Gaussian-correlated in space (renormalised) and
``noise.channel_corr`` correlated between channels; drawn as a random window of a per-camera noise bank made once
from ``seed`` (fast: no per-frame normal draws). ``rng`` (numpy Generator) picks the window, so a frame is
deterministic given ``rng``'s state. The noise sd in output DN at a level follows from s0, s1 and the tone curve
(``CameraModel.noise_dn``); fit_camera_look.py fits s0, s1 to the real frames' noise-vs-level.

``s`` is rounded and saturated to 8 bits (OpenCV) and mapped through a per-channel LUT (the tone curve): with the
noise as dither this costs < 0.3 DN. About 3-4 ms per 640x480 frame on one core with every block on.

Use::

    from sim_deploy.look.camera_look import load_camera_look
    look = load_camera_look("sim_deploy/look/camera_look_real.yaml")
    out = look.apply("ultrawide", rgb640, np.random.default_rng(0))

``python -m sim_deploy.look.camera_look FILE.yaml [--bench]`` prints the parameters (and times ``apply``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

SCHEMA = "sim_deploy.camera_look/v1"
ROLES = ("main", "ultrawide")
SIZE = (640, 480)                 # (w, h) of the stream frames
BANK_PAD = 160                    # noise bank margin: windows at (BANK_PAD + 1)^2 offsets


def shading_map(cfg: dict | None, w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray | None:
    """(h, w, 3) float32 gain map of a ``shading`` block, or None (identity)."""
    if not cfg:
        return None
    cx, cy = (float(v) for v in cfg.get("centre_px", ((w - 1) / 2, (h - 1) / 2)))
    nrm = float(cfg.get("norm_px", math.hypot(w / 2, h / 2)))
    X = (np.arange(w, dtype=np.float64) - cx) / nrm
    Y = (np.arange(h, dtype=np.float64) - cy) / nrm
    X, Y = np.meshgrid(X, Y)
    rho_max = cfg.get("rho_max")
    if rho_max:                                   # constant beyond rho_max along each ray (no extrapolation)
        f = np.minimum(1.0, float(rho_max) / np.maximum(np.hypot(X, Y), 1e-12))
        X, Y = X * f, Y * f
    r2 = X * X + Y * Y
    radial = cfg.get("radial") or [[0.0]] * 3
    tilt = cfg.get("tilt") or [[0.0, 0.0]] * 3
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        e = np.zeros_like(r2)
        p = np.ones_like(r2)
        for a in radial[c]:
            p = p * r2
            e += float(a) * p
        tx, ty = (float(v) for v in (list(tilt[c]) + [0.0, 0.0])[:2])
        e += tx * X + ty * Y
        out[..., c] = np.exp(e)
    return out


def noise_bank(seed: int, sigma_px: float, rho: float, w: int = SIZE[0], h: int = SIZE[1],
               pad: int = BANK_PAD) -> np.ndarray:
    """(h + pad, w + pad, 3) float32 unit-variance noise, spatially Gaussian-correlated, channel-correlated."""
    g = np.random.default_rng(int(seed))
    H, W = h + pad, w + pad
    e = g.standard_normal((H, W, 3), dtype=np.float32)
    rho = float(np.clip(rho, 0.0, 0.999))
    if rho > 0:
        common = g.standard_normal((H, W, 1), dtype=np.float32)
        e = math.sqrt(1.0 - rho) * e + math.sqrt(rho) * common
    if sigma_px > 0:
        e = cv2.GaussianBlur(e, (0, 0), float(sigma_px), borderType=cv2.BORDER_REFLECT_101)
    e /= e.reshape(-1, 3).std(axis=0)
    return np.ascontiguousarray(e, dtype=np.float32)


@dataclass
class CameraModel:
    """One camera's post-process (see the module docstring); built from its YAML block."""
    role: str
    cfg: dict
    seed: int

    def __post_init__(self):
        c = self.cfg
        self.blur = float(c.get("blur_sigma_px", 0.0) or 0.0)
        self.ksize = 2 * int(math.ceil(2.0 * self.blur)) + 1         # kernel to +-2 sigma (renormalised)
        dg = float(c.get("decode_gamma", 2.2))
        v = np.arange(256, dtype=np.float64) / 255.0
        self.dec = np.repeat((v ** dg).astype(np.float32)[:, None, None], 3, axis=2)          # (256, 1, 3)
        M = np.asarray(c.get("matrix", np.eye(3)), np.float64).reshape(3, 3)
        off = np.asarray(c.get("offset", [0.0, 0.0, 0.0]), np.float64).reshape(3)
        self.M34 = np.c_[M, off].astype(np.float32)
        self.identity_colour = bool(np.allclose(M, np.eye(3)) and np.allclose(off, 0))
        self.G = shading_map(c.get("shading"))
        tg = np.asarray(c.get("tone_gamma", [dg, dg, dg]), np.float64).reshape(3)
        s = np.arange(256, dtype=np.float64) / 255.0                                         # sqrt(lin) levels
        level = np.stack([255.0 * s ** (2.0 / tg[k]) for k in range(3)], axis=1)            # (256, 3)
        self.tg = tg
        self.level = level
        self.enc_u8 = np.clip(np.round(level), 0, 255).astype(np.uint8)[:, None, :]
        nz = c.get("noise") or {}
        self.s0 = np.asarray(nz.get("s0", [0.0] * 3), np.float64).reshape(3)
        self.s1 = np.asarray(nz.get("s1", [0.0] * 3), np.float64).reshape(3)
        self.noisy = bool(np.any(self.s0 != 0) or np.any(self.s1 != 0))
        if self.noisy:
            n = noise_bank(self.seed, float(nz.get("spatial_sigma_px", 0.0) or 0.0),
                           float(nz.get("channel_corr", 0.0) or 0.0))
            self.bank_mul = np.ascontiguousarray(255.0 * (1.0 + self.s1.astype(np.float32) * n), dtype=np.float32)
            self.bank_add = np.ascontiguousarray(255.0 * self.s0.astype(np.float32) * n, dtype=np.float32)

    def noise_dn(self, level_dn) -> np.ndarray:
        """Output noise sd (DN) per channel at output levels ``level_dn`` (first-order, from s0, s1 and the tone)."""
        y = np.clip(np.atleast_1d(np.asarray(level_dn, float)), 0.5, 254.5)
        out = []
        for k in range(3):
            e = 2.0 / self.tg[k]
            s = (y / 255.0) ** (1.0 / e)
            dyds = 255.0 * e * s ** (e - 1.0)
            out.append(dyds * np.abs(self.s0[k] + self.s1[k] * s))
        return np.stack(out, axis=-1)

    def sqrt_lin(self, rgb: np.ndarray) -> np.ndarray:
        """Blur, decode, colour matrix + offset, shading, clip at 0, sqrt: float32 (h, w, 3) in [0, ~1]."""
        x = rgb
        if self.blur > 0:
            x = cv2.GaussianBlur(x, (self.ksize, self.ksize), self.blur, borderType=cv2.BORDER_REFLECT_101)
        lin = cv2.LUT(x, self.dec)
        if not self.identity_colour:
            lin = cv2.transform(lin, self.M34)
        if self.G is not None:
            lin = cv2.multiply(lin, self.G)
        return cv2.sqrt(cv2.threshold(lin, 0.0, 0.0, cv2.THRESH_TOZERO)[1])

    def apply(self, rgb: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
        s = self.sqrt_lin(rgb)
        if self.noisy:
            rng = rng if rng is not None else np.random.default_rng()
            oy, ox = (int(v) for v in rng.integers(0, BANK_PAD + 1, size=2))
            h, w = s.shape[:2]
            q = cv2.add(cv2.multiply(s, self.bank_mul[oy:oy + h, ox:ox + w]), self.bank_add[oy:oy + h, ox:ox + w],
                        dtype=cv2.CV_8U)                          # 255 * (s + (s0 + s1 s) n), rounded, saturated
        else:
            q = cv2.convertScaleAbs(s, alpha=255.0)
        return cv2.LUT(q, self.enc_u8)


class CameraLook:
    """``apply(role, rgb640_uint8, rng) -> uint8 RGB (480, 640, 3)``; roles without a block pass through."""

    def __init__(self, doc: dict, path: Path | None = None):
        if doc.get("schema") not in (None, SCHEMA):
            raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA}")
        self.path = Path(path) if path else None
        self.doc = doc
        self.seed = int(doc.get("seed", 0))
        cams = doc.get("cameras") or {}
        unknown = sorted(set(cams) - set(ROLES))
        if unknown:
            raise ValueError(f"{path}: cameras.{unknown[0]}: unknown role (known: {', '.join(ROLES)})")
        self.models = {r: CameraModel(r, dict(cams[r]), self.seed * 2 + i) for i, r in enumerate(ROLES)
                       if cams.get(r)}

    def apply(self, role: str, rgb: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        m = self.models.get(role)
        if m is None:
            return rgb
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"camera_look: expected uint8 (h, w, 3), got {rgb.dtype} {rgb.shape}")
        if rgb.shape[:2] != (SIZE[1], SIZE[0]):
            raise ValueError(f"camera_look: expected {SIZE[0]}x{SIZE[1]}, got {rgb.shape[1]}x{rgb.shape[0]}")
        return m.apply(np.ascontiguousarray(rgb), rng)

    def describe(self) -> str:
        return f"camera_look {self.path} (cameras: {', '.join(self.models) or 'none'})"


def load_camera_look(path: str | Path) -> CameraLook:
    p = Path(path).expanduser()
    doc: Any = yaml.safe_load(p.read_text()) or {}
    return CameraLook(doc, p)


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.look.camera_look", description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--bench", action="store_true", help="time apply() on a synthetic frame (1 thread)")
    a = ap.parse_args(argv)
    look = load_camera_look(a.file)
    print(look.describe())
    if a.bench:
        cv2.setNumThreads(1)
        g = np.random.default_rng(0)
        img = cv2.GaussianBlur(g.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8), (0, 0), 2.0)
        for r in ROLES:
            rng = np.random.default_rng(1)
            look.apply(r, img, rng)
            t0 = time.perf_counter()
            n = 200
            for _ in range(n):
                look.apply(r, img, rng)
            print(f"  {r}: {(time.perf_counter() - t0) / n * 1e3:.2f} ms per frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
