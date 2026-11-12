"""H264Decoder and JpegDecoder (STRUCTURE.md §5.5): bit-identical to the offline decode, keyframe rule, errors."""

from __future__ import annotations

from fractions import Fraction

import av
import cv2
import numpy as np
import pytest

from contracts import Frame

from conftest import MAIN_SIZE, pattern
from process_frames.errors import DecodeError
from process_frames.frame_decoder import make_decoder
from process_frames.h264_decoder import H264Decoder
from process_frames.jpeg_decoder import JpegDecoder
from process_frames.pi_recording import encode_h264, encode_jpeg
from process_frames.settings import H264Settings

N = 25
KEYINT = 10


def to_frames(encoded, size=MAIN_SIZE, encoding="h264"):
    w, h = size
    return [Frame(camera="main", sensor="imx519", seq=i, t_pi_ns=1000 + i, exposure_ns=0, width=w, height=h,
                  encoding=encoding, keyframe=key, data=data) for i, (data, key) in enumerate(encoded)]


@pytest.fixture(scope="module")
def stream():
    images = [pattern(i, MAIN_SIZE, 0) for i in range(N)]
    return images, to_frames(encode_h264(images, 30.0, KEYINT, 2_000_000))


def offline_decode(path):
    """As dataproc's read_video_frames: av.open, thread_type AUTO, frame.to_ndarray(format="rgb24")."""
    out = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for frame in c.decode(s):
            out.append(frame.to_ndarray(format="rgb24"))
    return out


@pytest.fixture(scope="module")
def reference(stream, tmp_path_factory):
    path = tmp_path_factory.mktemp("h264") / "stream.h264"
    path.write_bytes(b"".join(f.data for f in stream[1]))
    ref = offline_decode(path)
    assert len(ref) == N
    return ref


@pytest.mark.parametrize("thread_type", ["SLICE", "NONE"])
def test_h264_bit_identical_to_offline_decode(stream, reference, thread_type):
    dec = H264Decoder("main", thread_type)
    out = [dec.decode(f) for f in stream[1]]
    assert all(o is not None for o in out)
    for o, r in zip(out, reference):
        assert o.dtype == np.uint8 and o.shape == (90, 160, 3)
        assert np.array_equal(o, r)
    # and close to the source images (it is a lossy encode of them)
    assert np.abs(out[3].astype(int) - stream[0][3].astype(int)).mean() < 20


def test_keyframes_every_interval(stream):
    assert [f.keyframe for f in stream[1]] == [i % KEYINT == 0 for i in range(N)]
    for f in stream[1]:
        if f.keyframe:
            assert f.data.startswith(b"\x00\x00\x00\x01") and (f.data[4] & 0x1F) == 7      # SPS first


def test_frames_before_the_first_keyframe_are_none(stream, reference):
    dec = H264Decoder("main", "SLICE")
    got = [dec.decode(f) for f in stream[1][3:]]
    assert all(g is None for g in got[:KEYINT - 3])
    assert np.array_equal(got[KEYINT - 3], reference[KEYINT])


def test_after_a_gap_none_until_the_next_keyframe_then_identical(stream, reference):
    dec = H264Decoder("main", "SLICE")
    frames = stream[1]
    for i in range(4):
        assert np.array_equal(dec.decode(frames[i]), reference[i])
    dec.notify_gap()                                       # frame 4 lost
    for i in range(5, 2 * KEYINT):
        out = dec.decode(frames[i])
        if i < KEYINT:
            assert out is None
        else:
            assert np.array_equal(out, reference[i])


def test_garbage_is_a_decode_error():
    dec = H264Decoder("main", "SLICE")
    junk = Frame(camera="main", sensor="imx519", seq=7, t_pi_ns=1, exposure_ns=0, width=160, height=90,
                 encoding="h264", keyframe=True, data=b"\x00\x00\x00\x01\x65" + bytes(range(256)) * 4)
    with pytest.raises(DecodeError) as e:
        dec.decode(junk)
    assert e.value.camera == "main" and e.value.seq == 7


def test_a_stream_with_b_frames_is_a_decode_error():
    images = [pattern(i, MAIN_SIZE, 0) for i in range(20)]
    ctx = av.CodecContext.create("libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = 160, 90, "yuv420p"
    ctx.time_base = Fraction(1, 30)
    ctx.framerate = Fraction(30, 1)
    ctx.options = {"preset": "medium", "bframes": "3", "x264-params": "keyint=50:scenecut=0:b-adapt=0"}
    packets = []
    for i, img in enumerate(images):
        vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
        vf.pts = i
        packets += ctx.encode(vf)
    packets += ctx.encode(None)
    frames = to_frames([(bytes(p), bool(p.is_keyframe)) for p in packets])
    dec = H264Decoder("main", "SLICE")
    with pytest.raises(DecodeError):
        for f in frames:
            dec.decode(f)


def test_thread_type_frame_is_refused():
    with pytest.raises(ValueError):
        H264Decoder("main", "FRAME")


def test_make_decoder():
    assert isinstance(make_decoder("h264", "main", H264Settings("NONE")), H264Decoder)
    assert isinstance(make_decoder("jpeg", "main", H264Settings("SLICE")), JpegDecoder)
    with pytest.raises(ValueError):
        make_decoder("vp9", "main", H264Settings("SLICE"))


def test_jpeg_matches_imdecode_and_bgr_to_rgb():
    images = [pattern(i, MAIN_SIZE, 3) for i in range(3)]
    frames = to_frames(encode_jpeg(images, 90), encoding="jpeg")
    dec = JpegDecoder("main")
    for f in frames:
        ref = cv2.cvtColor(cv2.imdecode(np.frombuffer(f.data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        assert np.array_equal(dec.decode(f), ref)
    dec.notify_gap()                                       # does nothing: every JPEG stands alone
    assert dec.decode(frames[1]) is not None


def test_corrupt_jpeg_is_a_decode_error():
    bad = Frame(camera="ultrawide", sensor="imx219", seq=3, t_pi_ns=1, exposure_ns=0, width=128, height=96,
                encoding="jpeg", keyframe=True, data=b"\xff\xd8not a jpeg")
    with pytest.raises(DecodeError) as e:
        JpegDecoder("ultrawide").decode(bad)
    assert e.value.seq == 3
