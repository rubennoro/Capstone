"""JPEG encoders for the recorder (P1 §2.3). ``cv2.imencode`` releases the GIL, so K worker threads scale."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class JpegEncoder(Protocol):
    quality: int

    def encode(self, img_bgr: np.ndarray) -> bytes: ...


class OpenCvJpeg:
    name = "opencv"

    def __init__(self, quality: int = 90):
        import cv2

        self._cv2 = cv2
        self.quality = int(quality)
        self._params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]

    def encode(self, img_bgr: np.ndarray) -> bytes:
        ok, buf = self._cv2.imencode(".jpg", img_bgr, self._params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()


class TurboJpeg:
    """PyTurboJPEG (libjpeg-turbo, 4:2:0). Only available if ``turbojpeg`` is installed."""

    name = "turbojpeg"

    def __init__(self, quality: int = 90):
        from turbojpeg import TJPF_BGR, TJSAMP_420, TurboJPEG

        self._tj = TurboJPEG()
        self._pf, self._samp = TJPF_BGR, TJSAMP_420
        self.quality = int(quality)

    def encode(self, img_bgr: np.ndarray) -> bytes:
        return self._tj.encode(img_bgr, quality=self.quality, pixel_format=self._pf, jpeg_subsample=self._samp)


ENCODERS = {"opencv": OpenCvJpeg, "turbojpeg": TurboJpeg}


def make_encoder(name: str = "opencv", quality: int = 90) -> JpegEncoder:
    if name not in ENCODERS:
        raise ValueError(f"unknown JPEG encoder {name!r} (valid: {sorted(ENCODERS)})")
    return ENCODERS[name](quality)
