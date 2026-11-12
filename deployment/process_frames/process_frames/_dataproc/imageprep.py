# from rnoro/data-work:data-processing/dataproc/imageprep.py @ ce9ba42
# changes: none
"""Image preparation: raw camera frames -> the 224x224 RGB images stored in the training zarr.

PURE functions and precomputed tables only: no file IO, no video decoding, no global state. Live deployment must
import these exact classes (``MainPrep``, ``UltrawidePrep``, or ``ImagePrep``) so that the images the policy sees at
run time are produced by the same code, with the same tables and interpolation, as the images it was trained on.
Never re-implement the letterbox or the remap elsewhere.

Pipeline per camera (input: decoded RGB ``uint8`` frames at the rig resolution, ``rigs/<rig>.yaml``):

* **main** (``prep_main``): the native view, no undistortion. Letterbox to a square with black borders (centred,
  ``(size - h) // 2`` rows on top) then ``cv2.INTER_AREA`` resize to 224x224. Bit-identical to iPhUMI's
  ``get_image_transform_with_border(in_res, (224, 224), mode="rgb")`` (``dataproc/_vendor/cv_util.py``).
* **ultrawide** (``prep_ultrawide``): ``to_virtual`` remaps the physical fisheye to the distortion-free virtual
  pinhole ``rig.virtual_ultrawide`` (640x480, the iPhone ultrawide K) with a precomputed ``build_remap`` table
  (``R_phys_virt`` applied, bilinear, uncovered pixels black), then the same letterbox -> 224x224. The width stage
  detects the finger tags on ``to_virtual`` output (its K is ``rig.virtual_ultrawide.K``).

Pixel convention (OpenCV): pixel ``(i, j)`` has its centre at ``(u=j, v=i)``. A virtual pixel ``(u, v)`` lands at
``((u + left + 0.5) * s - 0.5, (v + top + 0.5) * s - 0.5)`` in the 224 image, ``s = 224 / max(w, h)``
(:func:`letterbox_point`).
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import RigConfig
from .geometry.camera_model import RemapTable, VirtualPinhole, build_remap, model_from_dict

__all__ = ["OUT_RES", "letterbox_params", "letterbox", "letterbox_point", "physical_model", "virtual_model",
           "MainPrep", "UltrawidePrep", "ImagePrep"]

OUT_RES = 224                            # training image side (iPhUMI out_res_rgb)


def letterbox_params(in_res: tuple[int, int]) -> tuple[int, int, int, int, int]:
    """``(size, top, bottom, left, right)`` of the black padding for a ``(width, height)`` image (iPhUMI math)."""
    iw, ih = int(in_res[0]), int(in_res[1])
    size = max(iw, ih)
    top = (size - ih) // 2
    left = (size - iw) // 2
    return size, top, size - ih - top, left, size - iw - left


def letterbox(rgb: np.ndarray, out_res: int = OUT_RES) -> np.ndarray:
    """Pad ``rgb`` (H, W, 3) uint8 to a centred black square, then ``INTER_AREA`` resize to ``out_res``."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {rgb.dtype}")
    ih, iw = rgb.shape[:2]
    _, top, bottom, left, right = letterbox_params((iw, ih))
    sq = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(sq, (out_res, out_res), interpolation=cv2.INTER_AREA)


def letterbox_point(uv, in_res: tuple[int, int], out_res: int = OUT_RES) -> np.ndarray:
    """Where pixel positions ``uv`` (N, 2) of a ``(width, height)`` image land after :func:`letterbox`."""
    size, top, _, left, _ = letterbox_params(in_res)
    s = out_res / size
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    return np.stack([(uv[:, 0] + left + 0.5) * s - 0.5, (uv[:, 1] + top + 0.5) * s - 0.5], axis=1)


def physical_model(rig: RigConfig, role: str):
    """The geometry camera model (``PinholeRadtan`` / ``FisheyeEquidistant``) of ``rig.cameras[role]``."""
    cam = rig.cameras[role]
    d = {"type": cam.model.type, "intrinsics": list(cam.model.intrinsics), "distortion": list(cam.model.distortion)}
    return model_from_dict(d, tuple(cam.resolution))


def virtual_model(rig: RigConfig) -> VirtualPinhole:
    """The virtual ultrawide pinhole (``rig.virtual_ultrawide``)."""
    v = rig.virtual_ultrawide
    w, h = v.resolution
    fx, fy, cx, cy = v.intrinsics
    return VirtualPinhole(int(w), int(h), fx, fy, cx, cy)


def _check_frame(rgb: np.ndarray, res: tuple[int, int], what: str) -> None:
    w, h = res
    if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"{what}: expected a ({h}, {w}, 3) uint8 frame, got {rgb.shape} {rgb.dtype}")


class MainPrep:
    """Main camera: native view -> letterbox -> 224x224."""

    def __init__(self, rig: RigConfig, out_res: int = OUT_RES):
        self.in_res = tuple(int(x) for x in rig.cameras["main"].resolution)   # (width, height)
        self.out_res = int(out_res)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.prep_main(rgb)

    def prep_main(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 at the rig main resolution -> (224, 224, 3) uint8."""
        _check_frame(rgb, self.in_res, "main")
        return letterbox(rgb, self.out_res)


class UltrawidePrep:
    """Ultrawide: physical fisheye -> virtual pinhole (``rig.virtual_ultrawide``) -> letterbox -> 224x224."""

    def __init__(self, rig: RigConfig, out_res: int = OUT_RES):
        self.in_res = tuple(int(x) for x in rig.cameras["ultrawide"].resolution)
        self.phys = physical_model(rig, "ultrawide")
        self.virt = virtual_model(rig)
        self.R_phys_virt = np.asarray(rig.virtual_ultrawide.R_phys_virt, dtype=np.float64)
        self.remap: RemapTable = build_remap(self.phys, self.virt, self.R_phys_virt)
        self.virtual_res = (self.virt.width, self.virt.height)
        self.out_res = int(out_res)

    @property
    def virtual_K(self) -> np.ndarray:
        return self.virt.K

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.prep_ultrawide(rgb)

    def to_virtual(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 physical fisheye frame -> (480, 640, 3) uint8 virtual pinhole image (black = unseen)."""
        _check_frame(rgb, self.in_res, "ultrawide")
        return self.remap.apply(np.ascontiguousarray(rgb))

    def prep_ultrawide(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 physical fisheye frame -> (224, 224, 3) uint8."""
        return letterbox(self.to_virtual(rgb), self.out_res)

    def virtual_to_out(self, uv_virt) -> np.ndarray:
        """Virtual-image pixel positions (N, 2) -> positions in the 224 image."""
        return letterbox_point(uv_virt, self.virtual_res, self.out_res)


class ImagePrep:
    """Both cameras for one rig (build once; the remap table is the expensive part)."""

    def __init__(self, rig: RigConfig, out_res: int = OUT_RES):
        self.main = MainPrep(rig, out_res)
        self.ultrawide = UltrawidePrep(rig, out_res)

    def prep_main(self, rgb: np.ndarray) -> np.ndarray:
        return self.main.prep_main(rgb)

    def prep_ultrawide(self, rgb: np.ndarray) -> np.ndarray:
        return self.ultrawide.prep_ultrawide(rgb)

    def to_virtual(self, rgb: np.ndarray) -> np.ndarray:
        return self.ultrawide.to_virtual(rgb)
