"""Stage ``images`` helper: decode a raw episode's videos into the 224x224 training images on the 60 Hz grid.

Glue only (decode + :mod:`dataproc.export.frames` + :mod:`dataproc.imageprep`); the orchestrator owns the grid,
the stage bookkeeping and what to do with invalid steps. Times: ``t_grid_ns`` is on the IMU time base, and the
camera times are ``rig.camera_time_ns(role, t_ns, exposure_ns)`` (mid-exposure).

Memory: the result holds N + M frames of 150 KB (a 60 s episode at 60 Hz is ~0.6 GB); each source frame is
decoded and prepared once, even when the main camera runs below 60 fps and a frame fills several steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..imageprep import ImagePrep
from .frames import nearest_frame_index, ultrawide_substream

__all__ = ["EpisodeImages", "camera_times", "prepare_episode_images"]


@dataclass(eq=False)
class EpisodeImages:
    main_rgb: np.ndarray            # (N, 224, 224, 3) uint8
    uw_rgb: np.ndarray              # (M, 224, 224, 3) uint8
    uw_upsample_local: np.ndarray   # (N,) int64
    main_frame_idx: np.ndarray      # (N,) int64 decoded main frame per grid step
    main_valid: np.ndarray          # (N,) bool |t_main - t_grid| <= main_max_dt_ns
    main_dt_ns: np.ndarray          # (N,) int64 t_main[idx] - t_grid
    uw_frame_idx: np.ndarray        # (M,) int64 decoded ultrawide frame per selected sub-stream entry
    uw_t_ns: np.ndarray             # (M,) int64 mid-exposure time of each selected ultrawide frame


def camera_times(raw, rig, role: str) -> np.ndarray:
    """Mid-exposure times (int64 ns) of every decoded frame of the camera with ``role``."""
    ft = raw.frames(raw.meta.camera(role))
    return np.asarray(rig.camera_time_ns(role, ft.t_ns, ft.exposure_ns), dtype=np.int64)


def _decode_prepped(raw, cam: str, indices: np.ndarray, fn) -> np.ndarray:
    """``fn`` applied to the decoded frames ``indices`` (strictly increasing) -> (len, 224, 224, 3)."""
    out = None
    for k, (_, rgb) in enumerate(raw.read_video_frames(cam, indices.tolist())):
        img = fn(rgb)
        if out is None:
            out = np.empty((len(indices),) + img.shape, np.uint8)
        out[k] = img
    return out


def prepare_episode_images(raw, rig, t_grid_ns, *, main_max_dt_ns: int, uw_rate_hz: float = 10.0,
                           prep: ImagePrep | None = None) -> EpisodeImages:
    """Main: nearest frame per grid step; ultrawide: the ~``uw_rate_hz`` sub-stream; both prepared to 224x224.

    ``raw`` is a :class:`dataproc.raw_episode.RawEpisode`; ``prep`` defaults to ``ImagePrep(rig)`` (pass one in to
    reuse its remap table across episodes).
    """
    prep = prep or ImagePrep(rig)
    t_grid = np.asarray(t_grid_ns, dtype=np.int64)
    t_main = camera_times(raw, rig, "main")
    t_uw = camera_times(raw, rig, "ultrawide")

    idx, valid = nearest_frame_index(t_grid, t_main, main_max_dt_ns)
    uniq, inv = np.unique(idx, return_inverse=True)
    main_u = _decode_prepped(raw, raw.meta.camera("main"), uniq, prep.prep_main)
    sel, up = ultrawide_substream(t_grid, t_uw, rate_hz=uw_rate_hz)
    uw = _decode_prepped(raw, raw.meta.camera("ultrawide"), sel, prep.prep_ultrawide)
    return EpisodeImages(main_rgb=main_u[inv.reshape(-1)], uw_rgb=uw, uw_upsample_local=up, main_frame_idx=idx,
                         main_valid=valid, main_dt_ns=t_main[idx] - t_grid, uw_frame_idx=sel, uw_t_ns=t_uw[sel])
