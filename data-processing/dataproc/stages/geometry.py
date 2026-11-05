"""The geometry stages: **slam**, **grid**, **pose**, **width** (docs/PROCESS.md, docs/GEOMETRY.md).

This module holds the stage *classes* (inputs for the hash, file handling). The math lives in the pure-function
modules they call: :mod:`dataproc.slam` (backends), :mod:`dataproc.stages.pose` (``tcp_poses``) and
:mod:`dataproc.stages.width` (tag detection, gripper range, widths); those two names are the pure functions, not
stages. All times are on the IMU time base.
"""

from __future__ import annotations

import shutil

import numpy as np

from dataproc.files import write_json
from dataproc.stages.base import Stage, StageFailed

__all__ = ["GRID_HZ", "GRID_DT_NS", "POSE_MAX_GAP_NS", "WIDTH_MAX_GAP_NS", "SlamStage", "GridStage", "PoseStage",
           "WidthStage"]

GRID_HZ = 60
GRID_DT_NS = round(1e9 / GRID_HZ)          # 16_666_667
POSE_MAX_GAP_NS = 100_000_000              # interpolate the pose across at most 100 ms between SLAM poses
WIDTH_MAX_GAP_NS = 200_000_000             # interpolate the width across at most 200 ms between tag detections


class SlamStage(Stage):
    """**slam**. Reads ``raw/`` (ultrawide frames + IMU; ``truth.npz`` for ``--slam oracle``) and runs the SLAM
    backend in ``processed/slam/work/``. Writes ``processed/slam/{trajectory.tum, slam.npz, result.json}``
    (``slam.npz``: exact ns times, positions, quaternions, per-frame ``tracked``). Fails on ``SlamError`` or, for the
    oracle, a missing ``truth.npz``; scale problems are only recorded here (validate rejects). Skipped when the
    manifest, the backend name, ``cameras.ultrawide`` and ``imu`` are unchanged."""

    name = "slam"
    # Only the backend *name* is hashed, not its options: bump this version when the ORB-SLAM3 image tag
    # (orbslam3.DEFAULT_IMAGE) or the backend's settings defaults change, so every episode's slam re-runs.
    version = 1
    deps = ("check",)

    def inputs(self, ctx, ep) -> dict:
        s = ctx.sections
        return {"backend": ctx.backend_name, "cameras.ultrawide": s["cameras.ultrawide"], "imu": s["imu"]}

    def run(self, ctx, ep) -> dict:
        from dataproc.geometry.trajectory import save_tum
        from dataproc.slam import SlamError
        from dataproc.slam.oracle import TRUTH_FILE

        if ctx.backend_name == "oracle" and not (ep.raw_dir / TRUTH_FILE).is_file():
            raise StageFailed("failed", ["oracle_no_truth"], f"--slam oracle needs raw/{TRUTH_FILE} (synthetic "
                              "episodes only); use --slam orbslam3 for real episodes")
        out = ep.processed / "slam"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        try:
            res = ctx.backend.run(ep.raw, ctx.rig, out / "work")
        except SlamError as e:
            raise StageFailed("failed", ["slam_error"], str(e)) from None
        tr = res.trajectory
        save_tum(tr, out / "trajectory.tum")
        np.savez(out / "slam.npz", t_ns=tr.t_ns, p=tr.p, q_xyzw=tr.q_xyzw, t_frames_ns=res.t_frames_ns,
                 tracked=res.tracked)
        write_json(out / "result.json", {"backend": ctx.backend_name, **res.summary()})
        ep.products["slam"] = {"trajectory": tr, "t_frames_ns": res.t_frames_ns, "tracked": res.tracked,
                               "problems": list(res.problems)}
        return {"dir": "slam", "fraction_tracked": res.fraction_tracked, "n_poses": len(tr),
                "problems": list(res.problems)}


class GridStage(Stage):
    """**grid**. Reads ``meta.json`` button times. Writes ``processed/grid.npy``: the 60 Hz grid
    ``arange(T0, T1, 16_666_667)`` with ``T0/T1 = rig.imu_time_ns(button_start/stop_ns)``. Rejects a button span
    shorter than 2 grid steps. Skipped when the manifest and ``imu.time_offset_ns`` are unchanged."""

    name = "grid"
    deps = ("check",)

    def inputs(self, ctx, ep) -> dict:
        return {"imu.time_offset_ns": ctx.sections["imu.time_offset_ns"], "grid_dt_ns": GRID_DT_NS}

    def run(self, ctx, ep) -> dict:
        meta = ep.raw.meta
        t0, t1 = ctx.rig.imu_time_ns(meta.button_start_ns), ctx.rig.imu_time_ns(meta.button_stop_ns)
        grid = np.arange(t0, t1, GRID_DT_NS, dtype=np.int64)
        if len(grid) < 2:
            raise StageFailed("rejected", ["too_short"], f"button span {(t1 - t0) / 1e9:.3f} s gives {len(grid)} "
                              "grid steps")
        np.save(ep.processed / "grid.npy", grid)
        ep.products["grid"] = grid
        return {"file": "grid.npy", "n": int(len(grid)), "t0_ns": int(grid[0]), "dt_ns": GRID_DT_NS}


class PoseStage(Stage):
    """**pose**. Reads the slam trajectory and the grid; ``tcp_poses`` (``T_world_offset @ T_W_imu @ T_imu_cam @
    T_cam_tcp``, at most 100 ms between poses). Writes ``processed/pose.npz`` (``pos``, ``rotvec``, ``valid``,
    ``reasons``). Never rejects (invalid rows are trimmed by validate). Skipped when ``tcp``, ``world_offset``,
    the ultrawide ``T_imu_cam`` and the slam / grid hashes are unchanged and neither re-ran."""

    name = "pose"
    deps = ("slam", "grid")

    def inputs(self, ctx, ep) -> dict:
        s = ctx.sections
        return {"tcp": s["tcp"], "world_offset": s["world_offset"],
                "T_imu_cam_ultrawide": s["cameras.ultrawide"]["T_imu_cam"], "max_gap_ns": POSE_MAX_GAP_NS}

    def run(self, ctx, ep) -> dict:
        from dataproc.stages.pose import tcp_poses

        pos, rotvec, valid, reasons = tcp_poses(ep.slam()["trajectory"], ep.grid(), ctx.rig,
                                                max_gap_ns=POSE_MAX_GAP_NS, return_reasons=True)
        np.savez(ep.processed / "pose.npz", pos=pos, rotvec=rotvec, valid=valid, reasons=reasons.astype(str))
        ep.products["pose"] = {"pos": pos, "rotvec": rotvec, "valid": valid, "reasons": reasons}
        counts = {r: int(np.sum(reasons == r)) for r in sorted(set(reasons.tolist())) if r}
        return {"file": "pose.npz", "n_valid": int(valid.sum()), "n": int(len(valid)), "invalid_reasons": counts}


class WidthStage(Stage):
    """**width**. Reads every ultrawide frame (remapped with the run's ``ImagePrep.to_virtual``), the task's
    gripper range and the grid; tag detection, ``episode_widths``, ``width_on_grid`` (at most 200 ms between
    detections). Writes ``processed/width.npz`` (grid ``width``/``valid``, per-frame ``frame_has_width``, the
    detected widths) and ``processed/width_detections.json``. Fails with ``no_calibration`` /
    ``calibration_failed`` when the calibration step has no gripper range. Skipped when ``aruco``,
    ``virtual_ultrawide``, ``cameras.ultrawide``, the gripper_range.json bytes and the grid hash are unchanged."""

    name = "width"
    deps = ("check", "grid")

    def inputs(self, ctx, ep) -> dict:
        s = ctx.sections
        return {"aruco": s["aruco"], "virtual_ultrawide": s["virtual_ultrawide"],
                "cameras.ultrawide": s["cameras.ultrawide"], "gripper_range": ctx.calibration.token,
                "max_gap_ns": WIDTH_MAX_GAP_NS}

    def run(self, ctx, ep) -> dict:
        from dataproc.stages.width import detect_episode_tags, episode_widths, width_on_grid

        cal = ctx.calibration
        if cal.state != "done":
            raise StageFailed("failed", [cal.state], cal.reason or cal.state)
        t_frames, dets = detect_episode_tags(ep.raw, ctx.rig, ctx.prep.to_virtual)
        t_w, w, lp, rp = episode_widths(dets, t_frames, cal.gripper_range, ctx.rig)
        width, valid = width_on_grid(t_w, w, ep.grid(), max_gap_ns=WIDTH_MAX_GAP_NS)
        has_width = np.isin(t_frames, t_w)
        write_json(ep.processed / "width_detections.json",
                   [{"t_ns": int(t), "tags": {str(k): {"tvec": v.tvec, "rvec": v.rvec} for k, v in sorted(d.items())}}
                    for t, d in zip(t_frames.tolist(), dets)])
        arrays = {"width": width, "valid": valid, "frame_t_ns": t_frames, "frame_has_width": has_width,
                  "det_t_ns": t_w, "det_width": w, "det_left": lp, "det_right": rp}
        np.savez(ep.processed / "width.npz", **arrays)
        ep.products["width"] = arrays
        return {"file": "width.npz", "detections": "width_detections.json", "n_valid": int(valid.sum()),
                "n": int(len(valid)), "n_frames": int(len(t_frames)), "n_frames_with_width": int(has_width.sum()),
                "calibration": cal.name}
