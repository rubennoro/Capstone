# adapted from rnoro/deployment:deployment/process_frames/tests/test_decoders.py @ 91b178b (the JPEG part)
# changes: JPEG only (H264Decoder dropped); make_decoder knows only jpeg; imports.
"""JpegDecoder: identical to cv2.imdecode + BGR->RGB, every frame stands alone, corrupt data is a DecodeError."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from frames_helpers import MAIN_SIZE, pattern

from policy_deploy.contracts import EncodedFrame, FrameInfo
from policy_deploy.frames.errors import DecodeError
from policy_deploy.frames.frame_decoder import make_decoder
from policy_deploy.frames.jpeg_decoder import JpegDecoder
from policy_deploy.frames.pi_recording import encode_jpeg


def info(seq, size=MAIN_SIZE, sensor="imx519"):
    w, h = size
    return FrameInfo(sensor=sensor, frame_seq=seq, t_pi_ns=1000 + seq, exposure_ns=0, width=w, height=h,
                     encoding="jpeg", keyframe=True)


def to_frames(encoded, size=MAIN_SIZE):
    return [EncodedFrame(info(i, size), data) for i, (data, _key) in enumerate(encoded)]


def test_make_decoder():
    assert isinstance(make_decoder("jpeg", "main"), JpegDecoder)
    for kind in ("h264", "vp9"):
        with pytest.raises(ValueError):
            make_decoder(kind, "main")


def test_jpeg_matches_imdecode_and_bgr_to_rgb():
    images = [pattern(i, MAIN_SIZE, 3) for i in range(3)]
    frames = to_frames(encode_jpeg(images, 90))
    dec = JpegDecoder("main")
    for f in frames:
        ref = cv2.cvtColor(cv2.imdecode(np.frombuffer(f.data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert np.array_equal(dec.decode(f), ref)
    dec.notify_gap()                                       # does nothing: every JPEG stands alone
    assert dec.decode(frames[1]) is not None


def test_jpeg_is_close_to_the_original_image():
    img = pattern(4, MAIN_SIZE, 0)
    (data, key), = encode_jpeg([img], 95)
    assert key is True
    out = JpegDecoder("main").decode(EncodedFrame(info(0), data))
    assert out.shape == img.shape and np.abs(out.astype(int) - img.astype(int)).mean() < 6


def test_corrupt_jpeg_is_a_decode_error():
    bad = EncodedFrame(info(3, (128, 96), "imx219"), b"\xff\xd8not a jpeg")
    with pytest.raises(DecodeError) as e:
        JpegDecoder("ultrawide").decode(bad)
    assert e.value.seq == 3
