"""JpegDecoder: the backup decoder, OpenCV (STRUCTURE.md §5.5, PLAN §9). Every JPEG frame stands alone."""

from __future__ import annotations

import cv2
import numpy as np

from contracts import EncodedFrame

from process_frames.errors import DecodeError
from process_frames.frame_decoder import FrameDecoder


class JpegDecoder(FrameDecoder):
    ENCODING = "jpeg"

    def __init__(self, camera: str) -> None:
        self.camera = camera

    def decode(self, frame: EncodedFrame) -> np.ndarray | None:
        bgr = cv2.imdecode(np.frombuffer(frame.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise DecodeError(self.camera, frame.info.frame_seq, "not a decodable JPEG")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
