"""W2 look: projective backdrops. Real camera images painted onto proxy planes (floor, walls) of the sim scene.

Library (no CLI): ``make_robot.py`` and ``make_assets.py backdrop`` use it.

A ``Quad`` is a textured rectangle in O (corner ``p0``, edge vectors ``u`` (texture columns) and ``v`` (texture rows
from the top), ``mm`` per texel). ``bake(quads, views)`` fills every texel with the median, over the views that see
it, of the real pixel its O point projects to (a ``View`` = a projection O -> image pixel, the image, a validity mask,
an optional colour correction per pixel, an occluder test); texels no view sees get the quad's median colour
(``cv2.inpaint`` near the seen ones). From each view's own pose the painted proxy reproduces that view's image; from
nearby poses it is a flat-backdrop approximation (parallax of things off the proxy plane). ``write_obj`` packs the
quads into one atlas PNG + an OBJ in O (for the builder's ``appearance.background_mesh``).

The colour correction for the IMX219 (``radial_gain``): its frames carry the lens colour shading (pink corners) and
its own response; the textures of the sim are kept in IMX519 colours (the camera model maps them to each camera), so
IMX219 pixels are divided by a per-channel gain that depends on the fisheye radius, fitted on table pixels whose
IMX519 colour is known (``fit_radial_gain``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass
class Quad:
    name: str
    p0: np.ndarray            # O position of texel (0, 0)'s corner (top-left of the image)
    u: np.ndarray             # O vector along the image columns (full width)
    v: np.ndarray             # O vector along the image rows, downward (full height)
    mm: float = 5.0

    @property
    def shape(self) -> tuple[int, int]:
        return (max(1, int(round(np.linalg.norm(self.v) * 1000 / self.mm))),
                max(1, int(round(np.linalg.norm(self.u) * 1000 / self.mm))))

    def points(self) -> np.ndarray:
        h, w = self.shape
        a = (np.arange(w) + 0.5) / w
        b = (np.arange(h) + 0.5) / h
        A, B = np.meshgrid(a, b)
        return (np.asarray(self.p0, float)[None, None] + A[..., None] * np.asarray(self.u, float)
                + B[..., None] * np.asarray(self.v, float))


@dataclass
class View:
    name: str
    project: Callable          # (N, 3) O points -> (uv (N, 2), ok (N,))
    image: np.ndarray          # RGB uint8
    mask: np.ndarray           # bool, pixels usable
    correct: Callable | None = None      # (rgb float (N, 3), uv (N, 2)) -> rgb float
    occluded: Callable | None = None     # (N, 3) O points -> bool (N,): something real sits in front
    weight: float = 1.0


def sample(view: View, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rgb float (N, 3), ok (N,)) of the O points ``P`` (N, 3) in ``view``."""
    uv, ok = view.project(P)
    h, w = view.image.shape[:2]
    ok = np.asarray(ok, bool) & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    n = len(P)
    k = 1024                                     # remap maps must be < 32767 on each side: (N / k, k)
    pad = (-n) % k
    mx = np.r_[np.where(ok, uv[:, 0], -10), np.full(pad, -10.0)].astype(np.float32).reshape(-1, k)
    my = np.r_[np.where(ok, uv[:, 1], -10), np.full(pad, -10.0)].astype(np.float32).reshape(-1, k)
    rgb = cv2.remap(view.image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(-1, 3)[:n]
    mk = cv2.remap(view.mask.astype(np.uint8) * 255, mx, my, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT).reshape(-1)[:n] > 0
    ok &= mk
    if view.occluded is not None:
        ok &= ~np.asarray(view.occluded(P), bool)
    rgb = rgb.astype(np.float64)
    if view.correct is not None:
        rgb = view.correct(rgb, uv)
    return rgb, ok


def bake(quads: list[Quad], views: list[View], fill_rgb=None) -> dict:
    """{quad name: (texture uint8 RGB (h, w, 3), seen fraction)}."""
    out = {}
    for q in quads:
        P = q.points().reshape(-1, 3)
        cols, oks = [], []
        for v in views:
            c, ok = sample(v, P)
            cols.append(np.where(ok[:, None], c, np.nan))
            oks.append(ok)
        C = np.stack(cols)
        n = np.stack(oks).sum(0)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(C, axis=0)
        h, w = q.shape
        med = med.reshape(h, w, 3)
        seen = (n > 0).reshape(h, w)
        if seen.any():
            base = np.nanmedian(med[seen], axis=0) if fill_rgb is None else np.asarray(fill_rgb, float)
        else:
            base = np.asarray(fill_rgb if fill_rgb is not None else (128, 128, 128), float)
        img = np.where(seen[..., None], np.nan_to_num(med), base[None, None]).clip(0, 255).astype(np.uint8)
        if seen.any() and not seen.all():
            near = cv2.dilate(seen.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~seen
            img = cv2.inpaint(img, near.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        out[q.name] = (img, float(seen.mean()))
    return out


def write_obj(path_obj: Path, path_png: Path, quads: list[Quad], tex: dict, header: str = "") -> dict:
    """One atlas (quads stacked vertically, 2 px gutters replicated) + an OBJ of the quads (two triangles each, both
    sides) in O with texcoords into the atlas."""
    imgs = [tex[q.name][0] for q in quads]
    W = max(i.shape[1] for i in imgs) + 4
    rows, y = [], 0
    place = []
    for im in imgs:
        pad = cv2.copyMakeBorder(im, 2, 2, 2, W - 2 - im.shape[1], cv2.BORDER_REPLICATE)
        rows.append(pad)
        place.append((y + 2, im.shape[0], im.shape[1]))
        y += pad.shape[0]
    atlas = np.concatenate(rows, 0)
    Ht = atlas.shape[0]
    cv2.imwrite(str(path_png), atlas[:, :, ::-1])
    lines = [f"# {header}" if header else "# sim_deploy W2 look: projective backdrop quads (O frame)"]
    vt = []
    k = 0
    faces = []
    for q, (y0, h, w) in zip(quads, place):
        p0, u, v = (np.asarray(x, float) for x in (q.p0, q.u, q.v))
        corners = [p0, p0 + u, p0 + u + v, p0 + v]
        # texcoords (OBJ v up): image row r -> 1 - r / Ht
        uvs = [(2 / W, 1 - y0 / Ht), ((2 + w) / W, 1 - y0 / Ht), ((2 + w) / W, 1 - (y0 + h) / Ht),
               (2 / W, 1 - (y0 + h) / Ht)]
        for c in corners:
            lines.append("v %.5f %.5f %.5f" % tuple(c))
        for t in uvs:
            vt.append("vt %.6f %.6f" % t)
        a, b, c, d = (k + i + 1 for i in range(4))
        faces += [f"f {a}/{a} {b}/{b} {c}/{c}", f"f {a}/{a} {c}/{c} {d}/{d}",
                  f"f {a}/{a} {c}/{c} {b}/{b}", f"f {a}/{a} {d}/{d} {c}/{c}"]
        k += 4
    path_obj.write_text("\n".join(lines + vt + faces) + "\n")
    return {"atlas_px": [int(atlas.shape[1]), int(atlas.shape[0])], "quads": {q.name: {"px": list(q.shape[::-1]),
            "seen_frac": round(tex[q.name][1], 3), "p0": np.round(q.p0, 4).tolist(), "u": np.round(q.u, 4).tolist(),
            "v": np.round(q.v, 4).tolist(), "mm_per_texel": q.mm} for q in quads}}


# ---------------------------------------------------------------------------------------------- colour shading


def fit_radial_gain(r: np.ndarray, measured: np.ndarray, expected: np.ndarray, r_max: float, deg: int = 3,
                    bins: int = 24) -> dict:
    """Per-channel gain g_c(r) = measured / expected as a polynomial in (r / r_max)^2 (degree ``deg``), fitted on
    the per-bin medians of the ratio (robust to texture), weighted by the bin counts."""
    x = (np.asarray(r, float) / r_max) ** 2
    ratio = np.asarray(measured, float) / np.maximum(np.asarray(expected, float), 1.0)
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    xc, med, wts = [], [], []
    for i in range(bins):
        s = (x >= edges[i]) & (x <= edges[i + 1])
        if s.sum() < 50:
            continue
        xc.append(float(np.median(x[s])))
        med.append(np.median(ratio[s], axis=0))
        wts.append(np.sqrt(s.sum()))
    xc, med, wts = np.array(xc), np.array(med), np.array(wts)
    coef = [np.polyfit(xc, med[:, c], deg, w=wts).tolist() for c in range(3)]
    return {"coef_rgb": coef, "r_max": float(r_max), "x_range": [float(xc.min()), float(xc.max())],
            "bins_x": np.round(xc, 4).tolist(), "bins_gain_rgb": np.round(med, 4).tolist()}


def radial_gain(fit: dict, r: np.ndarray) -> np.ndarray:
    """(N, 3) gains at radii ``r``; beyond the fitted range the gain is held at its end value."""
    x = np.clip((np.asarray(r, float) / fit["r_max"]) ** 2, fit["x_range"][0], fit["x_range"][1])
    return np.stack([np.polyval(fit["coef_rgb"][c], x) for c in range(3)], 1).clip(0.2, 5.0)
