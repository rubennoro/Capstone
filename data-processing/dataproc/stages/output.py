"""The output stages: **images** (decode + 224x224 prep of the kept span) and **export** (``episode.zarr``).

The prepared images are kept in memory only (``ep.products["images"]``), never written as a separate file, so
``images`` re-runs whenever ``export`` re-runs (``EpisodeProcessor`` enforces this through ``feeds``).
"""

from __future__ import annotations

import numpy as np

from dataproc.stages.base import Stage, StageFailed
from dataproc.stages.validate import MAIN_MAX_DT_NS

__all__ = ["UW_RATE_HZ", "ImagesStage", "ExportStage"]

UW_RATE_HZ = 10.0                          # the ultrawide sub-stream rate in the zarr


class ImagesStage(Stage):
    """**images**. Reads the kept grid rows and decodes the raw main + ultrawide MP4s; ``prepare_episode_images``
    with the run's shared ``ImagePrep`` (main: nearest frame within 25 ms; ultrawide: the ~10 Hz sub-stream).
    Writes ``processed/images.npz`` (frame indices only; the images stay in memory for export). Fails if a kept
    step has no main frame. Skipped when the camera sections, ``virtual_ultrawide`` and the validate hash are
    unchanged, validate did not re-run, and export does not need to run."""

    name = "images"
    deps = ("validate",)
    feeds = "export"           # its product lives in memory: it re-runs whenever export runs

    def inputs(self, ctx, ep) -> dict:
        s = ctx.sections
        return {"cameras.main": s["cameras.main"], "cameras.ultrawide": s["cameras.ultrawide"],
                "virtual_ultrawide": s["virtual_ultrawide"], "uw_rate_hz": UW_RATE_HZ,
                "main_max_dt_ns": MAIN_MAX_DT_NS}

    def run(self, ctx, ep) -> dict:
        from dataproc.export.images import prepare_episode_images

        k = ep.kept()
        imgs = prepare_episode_images(ep.raw, ctx.rig, ep.grid()[k["i0"]:k["i1"]], main_max_dt_ns=MAIN_MAX_DT_NS,
                                      uw_rate_hz=UW_RATE_HZ, prep=ctx.prep)
        if not imgs.main_valid.all():      # validate kept only rows with a main frame within the same limit
            raise StageFailed("failed", ["main_frame_missing"], f"{int((~imgs.main_valid).sum())} kept steps "
                              "without a main frame")
        np.savez(ep.processed / "images.npz", main_frame_idx=imgs.main_frame_idx, main_dt_ns=imgs.main_dt_ns,
                 uw_frame_idx=imgs.uw_frame_idx, uw_t_ns=imgs.uw_t_ns, uw_upsample_local=imgs.uw_upsample_local)
        ep.products["images"] = imgs
        return {"file": "images.npz", "n_steps": int(len(imgs.main_rgb)), "n_uw": int(len(imgs.uw_rgb)),
                "max_main_dt_ms": float(np.abs(imgs.main_dt_ns).max() / 1e6)}


class ExportStage(Stage):
    """**export**. Reads the in-memory images plus the kept rows of pose and width; builds ``EpisodeArrays``
    (``name`` = the ledger ``pi_name``, ``task_label`` = task.yaml ``label``). Writes ``processed/episode.zarr``
    (replaced near-atomically). Skipped when the label, the name and the images / pose / width / validate hashes
    are unchanged and none of them re-ran."""

    name = "export"
    deps = ("images", "pose", "width", "validate")

    def inputs(self, ctx, ep) -> dict:
        return {"task_label": ctx.task.label, "name": ep.pi_name}

    def run(self, ctx, ep) -> dict:
        from dataproc.export.zarr_writer import EpisodeArrays, write_episode_zarr

        imgs = ep.products["images"]
        k = ep.kept()
        sl = slice(k["i0"], k["i1"])
        pose, width = ep.pose(), ep.width()
        arrays = EpisodeArrays(name=ep.pi_name, task_label=ctx.task.label, main_rgb=imgs.main_rgb,
                               uw_rgb=imgs.uw_rgb, uw_upsample_local=imgs.uw_upsample_local,
                               eef_pos=pose["pos"][sl], eef_rot_axis_angle=pose["rotvec"][sl],
                               gripper_width=width["width"][sl])
        write_episode_zarr(arrays, ep.processed / "episode.zarr")
        return {"dir": "episode.zarr", "n_steps": arrays.n_steps, "n_uw": arrays.n_uw, "name": ep.pi_name,
                "task_label": ctx.task.label}
