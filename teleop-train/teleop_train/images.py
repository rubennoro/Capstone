"""Frame selection, decoding and the 224x224 image prep (PLAN.md §5.2 steps 6-8). Owner: E3.

Frame selection is the copied ``_copied/frames.py`` (nearest main frame, the ~10 Hz ultrawide sub-stream). The image
prep is the copied ``_copied/_dataproc/imageprep.py`` ``ImagePrep``, the exact code policy_deploy runs live, built
from the rig content recorded in meta.yaml (``_copied/_dataproc/config.py`` ``parse_rig``). Decoding: ``av`` demuxes
the .mkv (no decoding) to get the JPEG bytes of the frames used, each once; each is decoded exactly as policy_deploy's
live ``JpegDecoder`` does (``cv2.imdecode(buf, cv2.IMREAD_COLOR)`` then ``cv2.COLOR_BGR2RGB``), so the pixels match
the deployment. The constants and signatures below are the contract with the other modules; E3 implements the bodies.

Video frame ``i`` is the ``i``-th non-empty packet of the .mkv's first video stream. Frames are decoded while the
file is demuxed, in index order, and only the ones asked for are kept (as JPEG bytes for at most one frame at a
time); ``prepare_images`` prepares each decoded frame straight away, so it holds only the 224x224 results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import av
import cv2
import numpy as np

from ._copied._dataproc.config import parse_rig
from ._copied._dataproc.imageprep import ImagePrep
from ._copied.frames import nearest_frame_index, ultrawide_substream

__all__ = ["MAIN_MAX_DT_NS", "UW_RATE_HZ", "ImageError", "EpisodeImages", "decode_jpeg", "decode_frames",
           "prepare_images"]

MAIN_MAX_DT_NS = 50_000_000      # each grid step needs a main frame within 50 ms: half of trajectory.MAX_MAIN_GAP_NS,
                                 # so a dropped frame at 30 fps (a step up to 33 ms away) passes and the gap rule decides
UW_RATE_HZ = 10.0                # the ultrawide sub-stream rate (umiproc UW_RATE_HZ)


class ImageError(ValueError):
    """The episode cannot be used; the message is the one-line reason stored in status.json."""


@dataclass(eq=False)
class EpisodeImages:
    main_rgb: np.ndarray             # (N, 224, 224, 3) uint8
    uw_rgb: np.ndarray               # (M, 224, 224, 3) uint8
    uw_upsample_local: np.ndarray    # (N,) int64: per grid step, index into uw_rgb (EpisodeArrays.uw_upsample_local)
    main_video_index: np.ndarray     # (N,) int64: the .mkv frame used per grid step
    uw_video_index: np.ndarray       # (M,) int64: the .mkv frame of each uw_rgb image


def _one_line(e: BaseException) -> str:
    return " ".join(str(e).split()) or type(e).__name__


def decode_jpeg(data: bytes, what: str = "frame") -> np.ndarray:
    """RGB uint8 (H, W, 3) from one JPEG, exactly as policy_deploy's live ``JpegDecoder``. ImageError if undecodable
    (``what`` names the frame in the message)."""
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageError(f"{what}: not a decodable JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _distinct_indices(indices: Sequence[int] | np.ndarray, video: Path) -> np.ndarray:
    a = np.asarray(indices)
    if a.size == 0:
        return np.zeros(0, np.int64)
    if a.dtype.kind not in "iu" or a.ndim != 1:
        raise ImageError(f"{video.name}: frame indices must be a 1-D integer sequence, got {a.dtype} {a.shape}")
    a = np.unique(a.astype(np.int64))
    if a[0] < 0:
        raise ImageError(f"{video.name}: negative frame index {int(a[0])}")
    return a


def _iter_jpegs(video: Path, wanted: np.ndarray) -> Iterator[tuple[int, bytes]]:
    """``(index, JPEG bytes)`` for each index in ``wanted`` (sorted, distinct, >= 0), in order, demuxing ``video``
    once and stopping after the last one. ImageError if the file cannot be read or ends before an index."""
    if len(wanted) == 0:
        return
    pos = 0
    count = 0
    try:
        with av.open(str(video), "r") as container:
            if not container.streams.video:
                raise ImageError(f"{video}: no video stream")
            for packet in container.demux(video=0):
                if packet.size == 0:            # demux ends with an empty flush packet
                    continue
                if count == wanted[pos]:
                    yield count, bytes(packet)
                    pos += 1
                    if pos == len(wanted):
                        return
                count += 1
    except (av.FFmpegError, OSError) as e:
        raise ImageError(f"{video}: cannot demux ({_one_line(e)})") from None
    raise ImageError(f"{video.name}: frame index {int(wanted[pos])} is past the end ({count} frames in the file); "
                     f"check the frames CSV against the video")


def _iter_decoded(video: Path, wanted: np.ndarray, resolution: tuple[int, int]) -> Iterator[tuple[int, np.ndarray]]:
    """``(index, RGB)`` per wanted index, decoded once each and checked against ``resolution`` (width, height)."""
    w, h = int(resolution[0]), int(resolution[1])
    for i, data in _iter_jpegs(video, wanted):
        rgb = decode_jpeg(data, f"{video.name} frame {i}")
        if rgb.shape[:2] != (h, w):
            raise ImageError(f"{video.name} frame {i}: {rgb.shape[1]}x{rgb.shape[0]}, the rig expects {w}x{h}; "
                             f"check the rig in meta.yaml against the camera stream")
        yield i, rgb


def decode_frames(video: Path, indices: Sequence[int], resolution: tuple[int, int]) -> dict[int, np.ndarray]:
    """{video index: RGB uint8 (H, W, 3)} for the distinct indices asked. ImageError for an index past the end, an
    undecodable frame, or a frame whose (width, height) != resolution."""
    video = Path(video)
    return dict(_iter_decoded(video, _distinct_indices(indices, video), resolution))


def _decode_prepared(video: Path, video_index: np.ndarray, resolution: tuple[int, int],
                     prep: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """(len(video_index), 224, 224, 3): ``prep`` of the frame at each entry; each distinct frame decoded once."""
    uniq, inv = np.unique(video_index, return_inverse=True)
    if len(uniq) and uniq[0] < 0:
        raise ImageError(f"{video.name}: negative frame index {int(uniq[0])}")
    out: np.ndarray | None = None
    k = 0
    for k, (_, rgb) in enumerate(_iter_decoded(video, uniq, resolution)):
        img = prep(rgb)
        if out is None:
            out = np.empty((len(uniq),) + img.shape, np.uint8)
        out[k] = img
    assert out is not None and k == len(uniq) - 1
    return out[inv.reshape(-1)]


def _times_and_map(t_ns: np.ndarray, video_index: np.ndarray, what: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_ns)
    vi = np.asarray(video_index)
    if t.ndim != 1 or vi.ndim != 1 or len(t) != len(vi) or len(t) == 0:
        raise ImageError(f"{what}: times {t.shape} and video_index {vi.shape} must be equal-length non-empty 1-D")
    if t.dtype.kind not in "iu" or vi.dtype.kind not in "iu":
        raise ImageError(f"{what}: times and video_index must be integers, got {t.dtype} and {vi.dtype}")
    return t.astype(np.int64), vi.astype(np.int64)


def prepare_images(t_grid: np.ndarray, main_t_ns: np.ndarray, main_video_index: np.ndarray, uw_t_ns: np.ndarray,
                   uw_video_index: np.ndarray, main_video: Path, uw_video: Path, rig: dict[str, Any]) -> EpisodeImages:
    """Main: ``nearest_frame_index(t_grid, main_t_ns, MAIN_MAX_DT_NS)``; ImageError if any step has no main frame
    within the limit. Ultrawide: ``ultrawide_substream(t_grid, uw_t_ns, UW_RATE_HZ)``. Times index the given arrays;
    ``*_video_index`` maps them to frames in the .mkv. Resolutions come from the rig (cameras.main / .ultrawide).
    ImageError also for a rig that parse_rig refuses."""
    main_video, uw_video = Path(main_video), Path(uw_video)
    try:
        rig_cfg = parse_rig(rig, "meta.yaml rig")
    except (ValueError, TypeError) as e:
        raise ImageError(f"rig refused: {_one_line(e)}") from None
    main_res = tuple(int(x) for x in rig_cfg.cameras["main"].resolution)
    uw_res = tuple(int(x) for x in rig_cfg.cameras["ultrawide"].resolution)

    t_grid = np.asarray(t_grid)
    if t_grid.ndim != 1 or len(t_grid) == 0 or t_grid.dtype.kind not in "iu":
        raise ImageError(f"grid: expected a non-empty 1-D integer array, got {t_grid.dtype} {t_grid.shape}")
    t_grid = t_grid.astype(np.int64)
    t_main, vi_main = _times_and_map(main_t_ns, main_video_index, "main")
    t_uw, vi_uw = _times_and_map(uw_t_ns, uw_video_index, "ultrawide")

    try:
        idx, valid = nearest_frame_index(t_grid, t_main, MAIN_MAX_DT_NS)
    except (ValueError, TypeError) as e:
        raise ImageError(f"main frame selection: {_one_line(e)}") from None
    if not np.all(valid):
        bad = np.flatnonzero(~valid)
        k = int(bad[0])
        dt_ms = abs(int(t_main[idx[k]]) - int(t_grid[k])) / 1e6
        raise ImageError(f"main: {len(bad)} grid step(s) have no main frame within {MAIN_MAX_DT_NS / 1e6:g} ms "
                         f"(first: step {k}, t={int(t_grid[k])} ns, nearest {dt_ms:.1f} ms away); "
                         f"check main_frames.csv for dropped frames")
    try:
        sel, up = ultrawide_substream(t_grid, t_uw, UW_RATE_HZ)
    except (ValueError, TypeError) as e:
        raise ImageError(f"ultrawide frame selection: {_one_line(e)}") from None

    main_vid = vi_main[idx]
    uw_vid = vi_uw[sel]
    try:
        prep = ImagePrep(rig_cfg)
    except (ValueError, TypeError, KeyError) as e:
        raise ImageError(f"rig refused by the image prep: {_one_line(e)}") from None
    main_rgb = _decode_prepared(main_video, main_vid, main_res, prep.prep_main)
    uw_rgb = _decode_prepared(uw_video, uw_vid, uw_res, prep.prep_ultrawide)
    return EpisodeImages(main_rgb=main_rgb, uw_rgb=uw_rgb, uw_upsample_local=up.astype(np.int64),
                         main_video_index=main_vid, uw_video_index=uw_vid)
