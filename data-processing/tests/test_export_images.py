"""dataproc.export.images: decode a (tiny) raw episode into grid-aligned 224 images."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from dataproc.config import load_rig
from dataproc.export.frames import nearest_frame_index, ultrawide_substream
from dataproc.export.images import camera_times, prepare_episode_images
from dataproc.export.zarr_writer import EpisodeArrays
from dataproc.imageprep import ImagePrep
from dataproc.raw_episode import RawEpisode
from fixtures import SIZES, T0_NS, build_episode

RIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"


def small_rig():
    """rig_v1 with the cameras scaled to the fixture video sizes (64x36 main, 64x48 ultrawide)."""
    rig = load_rig(RIG_PATH)
    cams = {}
    for role, cam in (("main", "cam0"), ("ultrawide", "cam1")):
        c = rig.cameras[role]
        w, h = SIZES[cam]
        s = w / c.resolution[0]
        fx, fy, cx, cy = c.model.intrinsics
        model = dataclasses.replace(c.model, intrinsics=(fx * s, fy * s, (cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5))
        cams[role] = dataclasses.replace(c, resolution=(w, h), model=model)
    return dataclasses.replace(rig, cameras=cams)


def test_prepare_episode_images(tmp_path):
    ep = build_episode(tmp_path / "ep", n_cam0=60, n_cam1=30)
    raw = RawEpisode.open(ep)
    rig = small_rig()
    prep = ImagePrep(rig)
    t_grid = T0_NS + 20_000_000 + np.round(np.arange(50) * 1e9 / 60).astype(np.int64)
    out = prepare_episode_images(raw, rig, t_grid, main_max_dt_ns=9_000_000, prep=prep)

    t_main = camera_times(raw, rig, "main")
    assert np.array_equal(t_main, raw.frames("cam0").t_ns - 2_500_000)       # end_of_exposure, 5 ms exposure
    idx, valid = nearest_frame_index(t_grid, t_main, 9_000_000)
    sel, up = ultrawide_substream(t_grid, camera_times(raw, rig, "ultrawide"))
    assert np.array_equal(out.main_frame_idx, idx) and np.array_equal(out.main_valid, valid)
    assert np.array_equal(out.uw_frame_idx, sel) and np.array_equal(out.uw_upsample_local, up)
    main_all = {i: prep.prep_main(f) for i, f in raw.read_video_frames("cam0")}
    uw_all = {i: prep.prep_ultrawide(f) for i, f in raw.read_video_frames("cam1")}
    assert np.array_equal(out.main_rgb, np.stack([main_all[i] for i in idx]))
    assert np.array_equal(out.uw_rgb, np.stack([uw_all[i] for i in sel]))
    assert out.main_rgb.shape == (50, 224, 224, 3) and out.uw_rgb.shape == (len(sel), 224, 224, 3)
    # feeds straight into the per-episode arrays
    n = len(t_grid)
    EpisodeArrays(name=raw.name, task_label="block_raise", main_rgb=out.main_rgb, uw_rgb=out.uw_rgb,
                  uw_upsample_local=out.uw_upsample_local, eef_pos=np.zeros((n, 3)),
                  eef_rot_axis_angle=np.ones((n, 3)), gripper_width=np.zeros(n))
