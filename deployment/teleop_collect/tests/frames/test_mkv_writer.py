"""MkvWriter: the JPEG bytes muxed as they are, read back byte for byte (STRUCTURE.md 5.2, PLAN 6.1). New."""

from __future__ import annotations

import time

import av
import cv2
import numpy as np
import pytest
from frames_helpers import read_mkv

from teleop_collect.frames.mkv_writer import MkvWriter


def jpegs(n, w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 95))])
        assert ok
        out.append(buf.tobytes())
    return out


def test_bytes_round_trip_and_video_index(tmp_path):
    data = jpegs(25)
    w = MkvWriter(tmp_path / "main.mkv", 64, 48, 30)
    assert [w.write(j) for j in data] == list(range(25))
    w.close()
    w.close()                                              # idempotent
    assert w.closed and read_mkv(tmp_path / "main.mkv") == data


def test_the_stream_is_mjpeg_at_the_size_and_playback_times(tmp_path):
    w = MkvWriter(tmp_path / "v.mkv", 640, 480, 30)
    for j in jpegs(4, 640, 480):
        w.write(j)
    w.close()
    with av.open(str(tmp_path / "v.mkv")) as c:
        s = c.streams.video[0]
        assert s.codec_context.name == "mjpeg" and (s.width, s.height) == (640, 480)
        pts_ms = [float(p.pts * p.time_base) * 1000 for p in c.demux(video=0) if p.size]
    assert pts_ms == pytest.approx([0, 33, 66, 100], abs=1.0)
    with av.open(str(tmp_path / "v.mkv")) as c:                 # and a player can decode it
        frame = next(c.decode(video=0))
        assert frame.to_ndarray(format="rgb24").shape == (480, 640, 3)


def test_written_frames_reach_the_disk_while_still_open(tmp_path):
    """Clusters are written about every 0.5 s of playback, so a crash loses at most that much (PLAN 6.1)."""
    path = tmp_path / "v.mkv"
    w = MkvWriter(path, 64, 48, 30)
    data = jpegs(60)
    for j in data:
        w.write(j)
    size_open = path.stat().st_size
    w.close()
    assert size_open >= sum(map(len, data[:40]))
    assert read_mkv(path) == data


def test_bad_arguments_and_a_bad_path(tmp_path):
    with pytest.raises(ValueError):
        MkvWriter(tmp_path / "a.mkv", 64, 48, 0)
    w = MkvWriter(tmp_path / "b.mkv", 64, 48, 30)
    with pytest.raises(ValueError):
        w.write(b"")
    w.close()
    with pytest.raises(RuntimeError):
        w.write(jpegs(1)[0])
    with pytest.raises((OSError, av.error.FFmpegError)):
        MkvWriter(tmp_path / "missing" / "c.mkv", 64, 48, 30)


def test_writing_is_cheap(tmp_path):
    """60 JPEGs of 640x480 per second must be far below the writer's budget (no re-encoding)."""
    data = jpegs(60, 640, 480)
    w = MkvWriter(tmp_path / "v.mkv", 640, 480, 30)
    t0 = time.monotonic()
    for j in data:
        w.write(j)
    w.close()
    assert time.monotonic() - t0 < 1.0
