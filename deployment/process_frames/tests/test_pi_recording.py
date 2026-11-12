"""PiRecording: a recorded Pi episode as C0 frames (STRUCTURE.md §5.13)."""

from __future__ import annotations

import json
import shutil

import pytest

from contracts import Frame, decode, encode

from conftest import write_pi_episode
from process_frames.pi_recording import PiRecording, PiRecordingError
from process_frames.replay_source import ReplaySource


def load(path, encoding="h264", max_seconds=10.0, keyframe_interval=10):
    return PiRecording.load(path, encoding=encoding, max_seconds=max_seconds, keyframe_interval=keyframe_interval,
                            bitrate=2_000_000, jpeg_quality=90)


def test_episode_loads_with_roles_from_the_sensor_model(pi_episode):
    rec = load(pi_episode)
    assert set(rec.cameras) == {"main", "ultrawide"}
    m, u = rec.cameras["main"], rec.cameras["ultrawide"]
    assert (m.sensor, m.width, m.height, m.fps) == ("imx519", 160, 90, 30.0)    # the Pi's cam1
    assert (u.sensor, u.width, u.height) == ("imx219", 128, 96)                  # the Pi's cam0
    assert len(m.frames) == 30 and [f.seq for f in m.frames] == list(range(30))
    assert rec.duration_ns == (5_000_000_000 + 11_000_000 + 29 * 33_333_333 + 33_333_333) - 5_000_000_000


def test_exposure_from_the_column_or_zero(pi_episode, pi_episode_no_exposure):
    rec = load(pi_episode)
    assert {f.exposure_ns for f in rec.cameras["main"].frames} == {8_000_000}
    assert {f.exposure_ns for f in rec.cameras["ultrawide"].frames} == {4_000_000}
    rec = load(pi_episode_no_exposure)
    assert {f.exposure_ns for f in rec.cameras["main"].frames} == {0}


def test_max_seconds_is_respected(pi_episode):
    rec = load(pi_episode, max_seconds=0.5)
    assert len(rec.cameras["main"].frames) == 16             # t < t0 + 0.5 s: frames 0..15 (15 x 33.3 ms < 0.5 s)


def test_h264_keyframes_and_every_frame_passes_the_c0_validator(pi_episode):
    rec = load(pi_episode, keyframe_interval=7)
    for cam in rec.cameras.values():
        assert [f.keyframe for f in cam.frames] == [i % 7 == 0 for i in range(len(cam.frames))]
        for f in cam.frames:
            back = decode(encode(f), expect=Frame)
            assert back.encoding == "h264" and back.width == cam.width and back.data == f.data


def test_jpeg_frames_are_all_keyframes(pi_episode):
    rec = load(pi_episode, encoding="jpeg")
    for cam in rec.cameras.values():
        assert all(f.keyframe and f.encoding == "jpeg" for f in cam.frames)
        for f in cam.frames:
            decode(encode(f), expect=Frame)


def test_replay_source_merges_by_pi_time(pi_episode):
    src = ReplaySource.from_recording(load(pi_episode, max_seconds=0.2))
    got = []
    while (f := src.next_frame(0.0)) is not None:
        got.append(f)
    assert src.exhausted and len(got) == 14                  # 7 per camera
    assert [f.t_pi_ns for f in got] == sorted(f.t_pi_ns for f in got)
    assert {f.camera for f in got} == {"main", "ultrawide"}


def _edit_meta(path, change):
    meta = json.loads((path / "meta.json").read_text())
    change(meta)
    (path / "meta.json").write_text(json.dumps(meta))


def test_broken_episodes_are_refused(pi_episode, tmp_path):
    def copy(name):
        dst = tmp_path / name
        shutil.copytree(pi_episode, dst)
        return dst

    ep = copy("no_meta")
    (ep / "meta.json").unlink()
    with pytest.raises(PiRecordingError, match="meta.json"):
        load(ep)
    ep = copy("missing_cam")
    _edit_meta(ep, lambda m: m["cameras"].pop(0))
    with pytest.raises(PiRecordingError, match="no ultrawide camera"):
        load(ep)
    ep = copy("unknown_model")
    _edit_meta(ep, lambda m: m["cameras"][0].update(model="imx477"))
    with pytest.raises(PiRecordingError, match="unknown sensor model"):
        load(ep)
    ep = copy("bad_header")
    p = ep / "cam1_frames.csv"
    p.write_text(p.read_text().replace("frame_index,t_ns", "idx,time", 1))
    with pytest.raises(PiRecordingError, match="header"):
        load(ep)
    ep = copy("row_count")
    _edit_meta(ep, lambda m: m["cameras"][1].update(frames=31))
    with pytest.raises(PiRecordingError, match="rows"):
        load(ep)
    ep = copy("times")
    p = ep / "cam0_frames.csv"
    lines = p.read_text().splitlines()
    lines[3], lines[4] = lines[4].replace("3,", "2,", 1), lines[3].replace("2,", "3,", 1)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(PiRecordingError):
        load(ep)


def test_video_shorter_than_its_csv_is_refused(tmp_path):
    ep = write_pi_episode(tmp_path / "short", n_frames=10)
    longer = write_pi_episode(tmp_path / "long", n_frames=12)
    shutil.copyfile(longer / "cam1_frames.csv", ep / "cam1_frames.csv")
    _edit_meta(ep, lambda m: m["cameras"][1].update(frames=12))
    with pytest.raises(PiRecordingError, match="shorter than its CSV"):
        load(ep)
