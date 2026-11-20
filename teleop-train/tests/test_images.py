"""images.py: frame selection, decoding and the image prep (PLAN.md §5.2 steps 6-8, §11 images bullet).

The synthetic videos encode each frame's own video_index (``synthetic_episode.decode_index``), so a decoded full-size
frame says which packet it came from. The grids are built here (60 Hz from the window start), independently of
trajectory.py.
"""
from __future__ import annotations

import copy
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from synthetic_episode import decode_index, load_rig, make_episode, make_task, write_mkv, jpeg_bytes, frame_image
from teleop_train import images
from teleop_train._copied._dataproc.config import parse_rig
from teleop_train._copied._dataproc.imageprep import ImagePrep
from teleop_train._copied.zarr_writer import EpisodeArrays
from teleop_train.images import (MAIN_MAX_DT_NS, UW_RATE_HZ, EpisodeImages, ImageError, decode_frames, decode_jpeg,
                                 prepare_images)

RES = (640, 480)


# --------------------------------------------------------------------------------------------- helpers

def grid_60hz(*streams: np.ndarray) -> np.ndarray:
    """t_k = t_0 + round(k * 1e9 / 60) over the common window of the streams."""
    t0 = max(int(s[0]) for s in streams)
    t1 = min(int(s[-1]) for s in streams)
    n = int((t1 - t0) * 60 // 1_000_000_000) + 1
    t = t0 + np.round(np.arange(n) * 1e9 / 60).astype(np.int64)
    return t[t <= t1]


def packets(video: Path) -> list[bytes]:
    """Every non-empty packet's bytes, in order (independent of images.py)."""
    with av.open(str(video)) as c:
        return [bytes(p) for p in c.demux(video=0) if p.size > 0]


def nearest_brute(t_grid: np.ndarray, t_cam: np.ndarray) -> np.ndarray:
    """Per grid time, the index of the nearest camera time (ties -> the earlier one), by brute force."""
    return np.array([int(np.argmin(np.abs(t_cam - t))) for t in t_grid], dtype=np.int64)


def cv2_rgb(data: bytes) -> np.ndarray:
    """policy_deploy's live JpegDecoder, written out here."""
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def run(ep, grid=None, *, main_t=None, main_vi=None, uw_t=None, uw_vi=None, rig="meta") -> EpisodeImages:
    main_t = ep.main_t_ns if main_t is None else main_t
    uw_t = ep.uw_t_ns if uw_t is None else uw_t
    main_vi = np.arange(len(main_t), dtype=np.int64) if main_vi is None else main_vi
    uw_vi = np.arange(len(uw_t), dtype=np.int64) if uw_vi is None else uw_vi
    grid = grid_60hz(ep.t_state_ns, main_t, uw_t) if grid is None else grid
    return prepare_images(grid, main_t, main_vi, uw_t, uw_vi, ep.episode_dir / "main.mkv",
                          ep.episode_dir / "ultrawide.mkv", ep.meta["rig"] if rig == "meta" else rig)


def assert_one_line(e: pytest.ExceptionInfo) -> None:
    assert "\n" not in str(e.value) and str(e.value)


@pytest.fixture(scope="module")
def default(tmp_path_factory):
    """The default synthetic episode, its grid, and prepare_images' result on it."""
    task = make_task(tmp_path_factory.mktemp("raw"))
    ep = make_episode(task, 0)
    grid = grid_60hz(ep.t_state_ns, ep.main_t_ns, ep.uw_t_ns)
    return ep, grid, run(ep, grid)


@pytest.fixture
def task_dir(tmp_path) -> Path:
    return make_task(tmp_path / "raw")


@pytest.fixture(scope="module")
def prep() -> ImagePrep:
    return ImagePrep(parse_rig(load_rig()))


def short(task_dir: Path, number: int = 0, **kw):
    return make_episode(task_dir, number, still_start_s=0.5, still_end_s=0.5, **kw)


# --------------------------------------------------------------------------------------------- constants

def test_constants():
    assert MAIN_MAX_DT_NS == 50_000_000
    assert UW_RATE_HZ == 10.0
    assert issubclass(ImageError, ValueError)


# --------------------------------------------------------------------------------------------- decode_frames

def test_decode_frames_index_is_packet_number(default):
    ep, _, _ = default
    asked = [0, 7, 7, 3, len(ep.main_t_ns) - 1]
    out = decode_frames(ep.episode_dir / "main.mkv", asked, RES)
    assert sorted(out) == sorted(set(asked))
    for i, rgb in out.items():
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        assert decode_index(rgb) == i


def test_decode_frames_matches_live_decoder(default):
    ep, _, _ = default
    raw = packets(ep.episode_dir / "ultrawide.mkv")
    out = decode_frames(ep.episode_dir / "ultrawide.mkv", [0, 11, 42], RES)
    for i in (0, 11, 42):
        np.testing.assert_array_equal(out[i], cv2_rgb(raw[i]))


def test_decode_jpeg_matches_live_decoder():
    data = jpeg_bytes(frame_image(5))
    np.testing.assert_array_equal(decode_jpeg(data), cv2_rgb(data))
    with pytest.raises(ImageError, match="not a decodable JPEG") as e:
        decode_jpeg(b"not a jpeg", "main.mkv frame 3")
    assert_one_line(e)


def test_decode_frames_empty_request(default):
    ep, _, _ = default
    assert decode_frames(ep.episode_dir / "main.mkv", [], RES) == {}


def test_decode_frames_extra_packets_are_frames(task_dir):
    ep = short(task_dir, extra_main_packets=2)
    f = len(ep.main_t_ns)
    out = decode_frames(ep.episode_dir / "main.mkv", [f, f + 1], RES)
    assert [decode_index(out[f]), decode_index(out[f + 1])] == [f, f + 1]


def test_decode_frames_index_past_end(default):
    ep, _, _ = default
    n = len(packets(ep.episode_dir / "main.mkv"))
    with pytest.raises(ImageError, match=f"frame index {n} is past the end \\({n} frames") as e:
        decode_frames(ep.episode_dir / "main.mkv", [0, n], RES)
    assert_one_line(e)


def test_decode_frames_negative_index(default):
    ep, _, _ = default
    with pytest.raises(ImageError, match="negative frame index"):
        decode_frames(ep.episode_dir / "main.mkv", [-1, 2], RES)


def test_decode_frames_resolution_mismatch(default):
    ep, _, _ = default
    with pytest.raises(ImageError, match="640x480, the rig expects 320x240") as e:
        decode_frames(ep.episode_dir / "main.mkv", [0], (320, 240))
    assert_one_line(e)


def test_decode_frames_undecodable(tmp_path):
    video = tmp_path / "main.mkv"
    write_mkv(video, [jpeg_bytes(frame_image(0)), b"\xff\xd8 garbage, not a JPEG"], *RES)
    assert decode_index(decode_frames(video, [0], RES)[0]) == 0
    with pytest.raises(ImageError, match="main.mkv frame 1: not a decodable JPEG") as e:
        decode_frames(video, [1], RES)
    assert_one_line(e)


def test_decode_frames_missing_file(tmp_path):
    with pytest.raises(ImageError, match="cannot demux") as e:
        decode_frames(tmp_path / "nope.mkv", [0], RES)
    assert_one_line(e)


# --------------------------------------------------------------------------------------------- prepare_images

def test_shapes_and_types(default):
    ep, grid, out = default
    n, m = len(grid), len(out.uw_rgb)
    assert out.main_rgb.shape == (n, 224, 224, 3) and out.main_rgb.dtype == np.uint8
    assert out.uw_rgb.shape == (m, 224, 224, 3) and out.uw_rgb.dtype == np.uint8
    assert out.uw_upsample_local.shape == (n,) and out.uw_upsample_local.dtype == np.int64
    assert out.main_video_index.shape == (n,) and out.main_video_index.dtype == np.int64
    assert out.uw_video_index.shape == (m,) and out.uw_video_index.dtype == np.int64


def test_main_frame_per_step(default):
    ep, grid, out = default
    np.testing.assert_array_equal(out.main_video_index, nearest_brute(grid, ep.main_t_ns))
    assert np.all(np.abs(ep.main_t_ns[out.main_video_index] - grid) <= 1e9 / 60)   # 30 fps: within half a period
    # read back: the full-size frames used carry their own index
    used = np.unique(out.main_video_index)
    dec = decode_frames(ep.episode_dir / "main.mkv", used, RES)
    assert all(decode_index(dec[int(i)]) == i for i in used)


def test_main_images_bit_identical_to_live_path(default, prep):
    ep, grid, out = default
    raw = packets(ep.episode_dir / "main.mkv")
    cache: dict[int, np.ndarray] = {}
    for k, i in enumerate(out.main_video_index):
        i = int(i)
        if i not in cache:
            cache[i] = prep.prep_main(cv2_rgb(raw[i]))
        np.testing.assert_array_equal(out.main_rgb[k], cache[i])
    # the steps really differ (the encoded index changes the image)
    assert len(cache) > 100 and not np.array_equal(out.main_rgb[0], out.main_rgb[-1])


def test_ultrawide_substream(default):
    ep, grid, out = default
    t_uw, sel, up = ep.uw_t_ns, out.uw_video_index, out.uw_upsample_local
    m = len(sel)
    # ~10 Hz: the schedule t0 + k * 100 ms, nearest frame each (brute force), duplicates removed, trimmed
    sched = grid[0] + np.round(np.arange(int((grid[-1] - grid[0]) // 100_000_000) + 1) * 1e8).astype(np.int64)
    expect = np.unique(nearest_brute(sched, t_uw))
    np.testing.assert_array_equal(sel, expect[:m])
    assert m >= len(expect) - 1
    assert np.all(np.diff(sel) > 0)
    gaps_ms = np.diff(t_uw[sel]) / 1e6
    assert np.all((gaps_ms > 60) & (gaps_ms < 140))
    # per step: the latest selected frame at or before the step (clamped at 0)
    assert up[0] == 0 and int(up[-1]) == m - 1 and np.all(np.diff(up) >= 0)
    for k, t in enumerate(grid):
        j = int(up[k])
        assert t_uw[sel[j]] <= t or j == 0
        assert j + 1 >= m or t_uw[sel[j + 1]] > t


def test_ultrawide_images_bit_identical_to_live_path(default, prep):
    ep, _, out = default
    raw = packets(ep.episode_dir / "ultrawide.mkv")
    for j, i in enumerate(out.uw_video_index):
        np.testing.assert_array_equal(out.uw_rgb[j], prep.prep_ultrawide(cv2_rgb(raw[int(i)])))
    dec = decode_frames(ep.episode_dir / "ultrawide.mkv", out.uw_video_index, RES)
    assert [decode_index(dec[int(i)]) for i in out.uw_video_index] == out.uw_video_index.tolist()


def test_maps_valid_for_episode_arrays(default):
    _, grid, out = default
    n = len(grid)
    rotvec = np.tile([np.pi - 0.1, 0.0, 0.0], (n, 1))
    ea = EpisodeArrays(name="pick_test_episode_000", task_label="pick_test", main_rgb=out.main_rgb, uw_rgb=out.uw_rgb,
                       uw_upsample_local=out.uw_upsample_local, eef_pos=np.zeros((n, 3)), eef_rot_axis_angle=rotvec,
                       gripper_width=np.zeros(n))
    assert ea.n_steps == n and ea.n_uw == len(out.uw_rgb)


def test_each_used_frame_decoded_once(task_dir, monkeypatch):
    ep = short(task_dir)
    calls: list[str] = []
    real = images.decode_jpeg

    def counting(data, what="frame"):
        calls.append(what)
        return real(data, what)

    monkeypatch.setattr(images, "decode_jpeg", counting)
    out = run(ep)
    assert len(calls) == len(set(calls))                                   # nothing decoded twice
    n_main = len(np.unique(out.main_video_index))
    assert n_main < len(out.main_video_index)                              # 30 fps on a 60 Hz grid: reuse happens
    assert sum(c.startswith("main.mkv") for c in calls) == n_main
    assert sum(c.startswith("ultrawide.mkv") for c in calls) == len(out.uw_video_index)


def test_video_index_maps_dropped_rows(task_dir, prep):
    """Rows with an empty time are dropped upstream; video_index keeps each time's frame in the .mkv."""
    ep = short(task_dir, blank_main_t_ns=5)
    main_t, main_vi = ep.main_t_ns[5:], np.arange(5, len(ep.main_t_ns), dtype=np.int64)
    grid = grid_60hz(ep.t_state_ns, main_t, ep.uw_t_ns)
    out = run(ep, grid, main_t=main_t, main_vi=main_vi)
    np.testing.assert_array_equal(out.main_video_index, main_vi[nearest_brute(grid, main_t)])
    assert out.main_video_index.min() >= 5
    raw = packets(ep.episode_dir / "main.mkv")
    for k in (0, len(grid) // 2, len(grid) - 1):
        i = int(out.main_video_index[k])
        np.testing.assert_array_equal(out.main_rgb[k], prep.prep_main(cv2_rgb(raw[i])))
        assert decode_index(decode_frames(ep.episode_dir / "main.mkv", [i], RES)[i]) == i


def test_deterministic(task_dir):
    ep = short(task_dir)
    a, b = run(ep), run(ep)
    for k in ("main_rgb", "uw_rgb", "uw_upsample_local", "main_video_index", "uw_video_index"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))


# --------------------------------------------------------------------------------------------- refusals

def test_step_without_main_frame(task_dir):
    ep = short(task_dir, main_gap=(1.0, 0.2))                     # 200 ms without main frames mid-episode
    with pytest.raises(ImageError, match="grid step\\(s\\) have no main frame within 50 ms") as e:
        run(ep)
    assert_one_line(e)


def test_one_dropped_main_frame_passes(task_dir):
    ep = short(task_dir, main_gap=(1.0, 0.04))                    # 40 ms window: one or two frames missing at 30 fps
    assert np.diff(ep.main_t_ns).max() > 60_000_000
    imgs = run(ep)
    assert len(imgs.main_rgb) > 0


def test_index_past_end(task_dir):
    ep = short(task_dir)
    shifted = np.arange(len(ep.main_t_ns), dtype=np.int64) + 3
    with pytest.raises(ImageError, match="past the end") as e:
        run(ep, main_vi=shifted)
    assert_one_line(e)


def test_resolution_mismatch(task_dir):
    ep = short(task_dir, frame_size=(320, 240))
    with pytest.raises(ImageError, match="320x240, the rig expects 640x480") as e:
        run(ep)
    assert_one_line(e)


@pytest.mark.parametrize("edit", ["schema", "none", "no_ultrawide", "bad_resolution"])
def test_refused_rig(task_dir, edit):
    ep = short(task_dir)
    rig = copy.deepcopy(ep.meta["rig"])
    if edit == "schema":
        rig["schema"] = "dataproc.rig.v0"
    elif edit == "none":
        rig = None
    elif edit == "no_ultrawide":
        del rig["cameras"]["ultrawide"]
    else:
        rig["cameras"]["main"]["resolution"] = [640]
    with pytest.raises(ImageError, match="^rig refused: ") as e:
        run(ep, rig=rig)
    assert_one_line(e)


def test_other_rig_resolution_is_checked(task_dir):
    """The resolution comes from the rig: a rig saying 1280x960 refuses 640x480 frames."""
    ep = short(task_dir)
    rig = copy.deepcopy(ep.meta["rig"])
    rig["cameras"]["main"]["resolution"] = [1280, 960]
    with pytest.raises(ImageError, match="main.mkv frame \\d+: 640x480, the rig expects 1280x960"):
        run(ep, rig=rig)


def test_mismatched_inputs(task_dir):
    ep = short(task_dir)
    with pytest.raises(ImageError, match="main: times .* must be equal-length"):
        run(ep, main_vi=np.arange(len(ep.main_t_ns) - 1, dtype=np.int64))
    with pytest.raises(ImageError, match="ultrawide frame selection: .*strictly increasing"):
        run(ep, uw_t=ep.uw_t_ns[::-1].copy(), grid=grid_60hz(ep.t_state_ns, ep.main_t_ns, ep.uw_t_ns))
    with pytest.raises(ImageError, match="grid: expected a non-empty 1-D integer array"):
        run(ep, grid=np.zeros(0, np.int64))
