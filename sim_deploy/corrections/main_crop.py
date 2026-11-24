"""IMX519 (main) crop/zoom correction for the Franka mount (camera study B, sim only): a similarity transform of the
decoded 640x480 main frame, applied BEFORE the copy's ``ImagePrep.prep_main`` (letterbox -> 224).

    from sim_deploy.corrections import main_crop
    crop = main_crop.load("sim_deploy/corrections/main_crop_corr.yaml")
    rgb640_corr = crop.apply(rgb640)                 # (480, 640, 3) uint8 -> same shape/dtype
    img224 = prep.prep_main(rgb640_corr)             # ImagePrep of the copy, unchanged

Pure functions, no state beyond the parameters; depends on numpy, cv2, yaml only (no sim_deploy imports), so it can be
copied as is into another process. Parameters (fitted by ``fit_main.py``, stored in ``main_crop_corr.yaml``):

    p_out = scale * R(rotation_deg) @ (p_in - centre_px) + centre_px + shift_px

in OpenCV pixel coordinates of the 640x480 frame (pixel (i, j) has its centre at (u=j, v=i); v points down, so a
positive ``rotation_deg`` turns +u toward +v, i.e. clockwise on screen). ``scale`` > 1 zooms in (a crop: no black
border while the shifted, rotated window stays inside the frame); ``scale`` < 1 zooms out and leaves a black border
(``BORDER_CONSTANT`` 0, as the letterbox's own padding).

Interpolation: ``cv2.warpAffine`` with ``INTER_LINEAR`` (default; ``cubic`` selectable in the yaml). OpenCV's warpAffine
does not implement ``INTER_AREA`` (it falls back to bilinear), and the scale here is within a few % of 1, so there is no
downsampling to anti-alias; the 640 -> 224 ``INTER_AREA`` of prep_main then does the real reduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

SCHEMA = "sim_deploy.main_crop/v1"
_INTERP = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}


@dataclass(frozen=True)
class MainCrop:
    scale: float
    rotation_deg: float
    centre_px: tuple[float, float] = (319.5, 239.5)
    shift_px: tuple[float, float] = (0.0, 0.0)
    in_res: tuple[int, int] = (640, 480)                 # (width, height)
    interpolation: str = "linear"
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if not (math.isfinite(self.scale) and 0.5 < self.scale < 2.0):
            raise ValueError(f"main_crop: scale must be in (0.5, 2), got {self.scale!r}")
        if not math.isfinite(self.rotation_deg) or abs(self.rotation_deg) > 45:
            raise ValueError(f"main_crop: rotation_deg must be within +-45, got {self.rotation_deg!r}")
        if self.interpolation not in _INTERP:
            raise ValueError(f"main_crop: interpolation must be one of {sorted(_INTERP)}, got {self.interpolation!r}")

    @property
    def A(self) -> np.ndarray:
        t = math.radians(self.rotation_deg)
        return self.scale * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])

    @property
    def matrix(self) -> np.ndarray:
        """2x3 forward map (input pixel -> output pixel), as cv2.warpAffine takes it."""
        c, s = np.asarray(self.centre_px, float), np.asarray(self.shift_px, float)
        A = self.A
        return np.hstack([A, (c + s - A @ c)[:, None]])

    def map_points(self, uv) -> np.ndarray:
        """Where input pixel positions (n, 2) land in the corrected frame."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        M = self.matrix
        return uv @ M[:, :2].T + M[:, 2]

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 at in_res -> the corrected (H, W, 3) uint8 frame."""
        w, h = self.in_res
        if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"main_crop: expected a ({h}, {w}, 3) uint8 frame, got {rgb.shape} {rgb.dtype}")
        return cv2.warpAffine(np.ascontiguousarray(rgb), self.matrix, (w, h), flags=_INTERP[self.interpolation],
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "in_res": list(self.in_res), "scale": float(self.scale),
                "rotation_deg": float(self.rotation_deg), "centre_px": [float(x) for x in self.centre_px],
                "shift_px": [float(x) for x in self.shift_px], "interpolation": self.interpolation,
                "matrix_2x3": np.round(self.matrix, 9).tolist()}


def from_dict(d: dict) -> MainCrop:
    if d.get("schema") != SCHEMA:
        raise ValueError(f"main_crop: schema must be {SCHEMA!r}, got {d.get('schema')!r}")
    crop = MainCrop(scale=float(d["scale"]), rotation_deg=float(d["rotation_deg"]),
                    centre_px=tuple(float(x) for x in d.get("centre_px", (319.5, 239.5))),
                    shift_px=tuple(float(x) for x in d.get("shift_px", (0.0, 0.0))),
                    in_res=tuple(int(x) for x in d.get("in_res", (640, 480))),
                    interpolation=str(d.get("interpolation", "linear")),
                    meta={k: v for k, v in d.items() if k not in ("schema", "scale", "rotation_deg", "centre_px",
                                                                   "shift_px", "in_res", "interpolation",
                                                                   "matrix_2x3")})
    if "matrix_2x3" in d and not np.allclose(np.asarray(d["matrix_2x3"], float), crop.matrix, atol=1e-6):
        raise ValueError("main_crop: matrix_2x3 does not match scale/rotation_deg/centre_px/shift_px")
    return crop


def load(path) -> MainCrop:
    return from_dict(yaml.safe_load(Path(path).expanduser().read_text()))


def dump(crop: MainCrop, path, header: str = "", extra: dict | None = None) -> None:
    d = crop.to_dict()
    if extra:
        d.update(extra)
    text = "".join(f"# {ln}\n" if ln else "#\n" for ln in header.splitlines())
    Path(path).write_text(text + yaml.safe_dump(d, sort_keys=False, width=120))
