"""Finger / gripper mask for the VIO camera (F3).

The ultrawide sees the gripper fingers and their tags, which move rigidly with the camera; ORB features there break
the static-scene assumption of the VIO. ``finger_mask`` builds a mask in the VIRTUAL iPhone ultrawide view (where
the finger geometry is the same for every rig, because calib-align makes the virtual view match the iPhone) and
renders it into the PHYSICAL camera image with ``rig.inverse_remap`` (the same tables the simulator uses).

Template: ``rig.vio.mask_template`` is not part of the schema yet, so the built-in template is used: the two
finger regions at the lower left / lower right and a strip along the bottom edge of the 640x480 virtual ultrawide
(iPhone UMI gripper layout). Tune it on a real frame with ``sensor-rig calib-check`` overlays before relying on it.
Convention: 255 = use the pixel, 0 = masked (UMI's ORB-SLAM3 fork convention). Pixels the virtual view does not
cover stay 255.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# polygons in normalised virtual-ultrawide coordinates (x right, y down)
FINGER_TEMPLATE = [
    [(0.00, 0.55), (0.30, 0.62), (0.36, 1.00), (0.00, 1.00)],     # left finger + tag
    [(1.00, 0.55), (0.70, 0.62), (0.64, 1.00), (1.00, 1.00)],     # right finger + tag
    [(0.00, 0.88), (1.00, 0.88), (1.00, 1.00), (0.00, 1.00)],     # gripper base strip
]


def virtual_template(width: int, height: int, polygons=FINGER_TEMPLATE) -> np.ndarray:
    m = np.full((height, width), 255, np.uint8)
    for poly in polygons:
        pts = np.round(np.array(poly, np.float64) * [width - 1, height - 1]).astype(np.int32)
        cv2.fillPoly(m, [pts], 0)
    return m


def finger_mask(rig, camera: str = "ultrawide", size: tuple[int, int] | None = None) -> np.ndarray:
    """uint8 mask at the physical resolution of ``camera`` (or resized to ``size`` = (w, h))."""
    virt_name = next((n for n, v in rig.virtual.items() if v.source == camera), None)
    if virt_name is None:
        raise ValueError(f"no virtual camera renders from {camera!r}")
    inv = rig.inverse_remap(virt_name)
    vw, vh = inv.src_size
    tmpl = virtual_template(vw, vh)
    phys = inv.apply(255 - tmpl, interpolation=cv2.INTER_NEAREST, border_value=0)   # 255 where masked
    mask = np.where(phys > 127, 0, 255).astype(np.uint8)
    if size is not None and (mask.shape[1], mask.shape[0]) != tuple(size):
        mask = cv2.resize(mask, tuple(size), interpolation=cv2.INTER_NEAREST)
    return mask


def write_finger_mask(rig, camera: str, path, size: tuple[int, int] | None = None) -> dict:
    m = finger_mask(rig, camera, size)
    path = Path(path)
    cv2.imwrite(str(path), m)
    return {"path": str(path), "masked_fraction": float((m == 0).mean()), "size": [m.shape[1], m.shape[0]]}
