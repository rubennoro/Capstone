# copied from rnoro/deployment:deployment/process_frames/process_frames/jpeg_decoder.py @ 91b178b
# changes: imports
"""JpegDecoder: OpenCV. Every JPEG frame stands alone (the Pi sends JPEG, PI_PAYLOAD.md 0.2)."""

from __future__ import annotations

import cv2
import numpy as np

from policy_deploy.contracts import EncodedFrame

from policy_deploy.frames.errors import DecodeError
from policy_deploy.frames.frame_decoder import FrameDecoder


class JpegDecoder(FrameDecoder):
    ENCODING = "jpeg"

    def __init__(self, camera: str) -> None:
        self.camera = camera

    def decode(self, frame: EncodedFrame) -> np.ndarray | None:
        bgr = cv2.imdecode(np.frombuffer(frame.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise DecodeError(self.camera, frame.info.frame_seq, "not a decodable JPEG")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
