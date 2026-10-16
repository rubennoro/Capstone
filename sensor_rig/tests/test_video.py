"""T6: Mp4Writer output is readable exactly the way the iPhUMI pipeline reads videos."""

from fractions import Fraction

import av
import cv2
import numpy as np
import pytest

from sensor_rig.export.video import Mp4Writer, probe_video


def _frames(n, w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    bars = np.zeros((h, w, 3), np.uint8)
    colours = [(255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255),
               (255, 0, 0), (40, 40, 40)]
    for i, c in enumerate(colours):
        bars[:, i * w // 8:(i + 1) * w // 8] = c
    out = []
    for k in range(n):
        img = np.roll(bars, 4 * k, axis=1).astype(np.int16) + rng.integers(-6, 7, (h, w, 3))
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "x_rgb.mp4"
    imgs = _frames(40)
    # VFR: 60 Hz with jitter (9/10/11 ticks) and one 50 ms gap (30 ticks)
    steps = [10, 10, 9, 11, 10, 30] + [10] * 33
    pts = np.concatenate([[0], np.cumsum(steps)]).astype(int)
    with Mp4Writer(path, 320, 240, crf=18) as w:
        for img, p in zip(imgs, pts):
            w.write(img, int(p))
    return path, imgs, pts


def test_header_count_and_pts(video):
    path, imgs, pts = video
    info = probe_video(path)
    assert info["frames_header"] == len(imgs)
    assert info["time_base"] == Fraction(1, 600)
    assert info["pts_list"] == pts.tolist()
    assert (info["width"], info["height"]) == (320, 240)


def test_decode_like_gen_replay_buffer_zarr(video):
    """gen_replay_buffer_zarr.py:584-603: in_stream.frames bounds the indices; frames decoded with to_ndarray(rgb24)."""
    path, imgs, pts = video
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        assert st.frames == len(imgs)
        decoded = [(f.pts, f.to_ndarray(format="rgb24")) for f in c.decode(st)]
    assert [p for p, _ in decoded] == pts.tolist()
    mae = np.mean([np.abs(rgb[..., ::-1].astype(int) - ref.astype(int)).mean() for (_, rgb), ref in zip(decoded, imgs)])
    assert mae < 3.0, mae
    # detect_aruco_iphone.py: frame.pts * time_base is the frame time in seconds
    assert float(decoded[6][0] * Fraction(1, 600)) == pytest.approx(80 / 600)  # after the 50 ms gap


def test_decode_cv2(video):
    path, imgs, _ = video
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(imgs)
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out.append(img)
    assert len(out) == len(imgs)
    mae = np.mean([np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(out, imgs)])
    assert mae < 3.0, mae


def test_quicktime_container(video):
    path, _, _ = video
    head = path.read_bytes()[:32]
    assert head[4:8] == b"ftyp" and head[8:12] == b"qt  "


def test_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        Mp4Writer(tmp_path / "odd.mp4", 321, 240)
    w = Mp4Writer(tmp_path / "a.mp4", 64, 48)
    w.write(np.zeros((48, 64, 3), np.uint8), 0)
    with pytest.raises(ValueError):
        w.write(np.zeros((48, 64, 3), np.uint8), 0)  # non-increasing PTS
    with pytest.raises(ValueError):
        w.write(np.zeros((40, 64, 3), np.uint8), 10)
    assert w.close() == 1


def test_colour_round_trip_unbiased(tmp_path):
    """Decoding with to_ndarray('rgb24') returns the input colours without a systematic shift (lead review: PyAV's
    reformat path darkened frames by ~2 levels, which would separate training images from deployment images)."""
    path = tmp_path / "c_rgb.mp4"
    flat = np.full((240, 320, 3), (30, 128, 220), np.uint8)
    imgs = _frames(10) + [flat]
    with Mp4Writer(path, 320, 240, crf=18) as w:
        for k, img in enumerate(imgs):
            w.write(img, 10 * k)
    with av.open(str(path)) as c:
        dec = [f.to_ndarray(format="rgb24")[..., ::-1].astype(float) for f in c.decode(video=0)]
    assert np.abs(dec[-1][100:140, 140:180] - flat[100:140, 140:180]).max() <= 1.5
    # interior of each colour bar (chroma edges excluded): mean signed error per channel within 1 level
    interior = np.stack([d[:, 8:-8] - i[:, 8:-8] for d, i in zip(dec[:-1], imgs[:-1])])
    assert np.all(np.abs(interior.mean(axis=(0, 1, 2))) < 1.0), interior.mean(axis=(0, 1, 2))
