"""Stage ``validate``: trim each episode to its longest fully valid stretch, then reject bad episodes.

Pure function of arrays (:func:`validate_episode`); the orchestrator (``dataproc/process.py``) gathers them. All
thresholds are module constants, listed in docs/PROCESS.md, and part of the stage's ``inputs_hash``.

1. A grid row is usable when its pose, width and main image are all valid (main: the nearest main frame within
   :data:`MAIN_MAX_DT_NS`). The **kept span** is the longest contiguous run of usable rows.
2. ``too_short``: the kept span lasts < :data:`MIN_KEPT_S`; ``kept_fraction``: it covers
   < :data:`MIN_KEPT_FRACTION` of the grid (the button span).
3. Inside the kept span ``[t_lo, t_hi]`` (every reason is recorded):

   * ``speed``: TCP speed (per-step position delta x 60 Hz, iPhUMI ``align_validate``) > :data:`MAX_SPEED_MPS`;
     also ``speed_extreme`` when > :data:`EXTREME_SPEED_MPS` (a SLAM jump rather than a fast hand);
   * ``ultrawide_gap``: a gap between consecutive ultrawide frames (or between the span edge and the nearest frame)
     > :data:`MAX_ULTRAWIDE_GAP_NS`;
   * ``imu_gap``: the same for IMU samples, > :data:`MAX_IMU_GAP_NS`;
   * ``slam_tracking``: fraction of the ultrawide frames in the span with a SLAM pose < :data:`MIN_TRACKED_FRACTION`;
   * ``scale_mismatch`` / ``scale_unobservable``: the SLAM metric-scale check (``SlamResult`` problems);
   * ``tag_coverage``: fraction of the ultrawide frames in the span with at least one usable finger tag (a frame
     the width stage measured) < :data:`MIN_TAG_FRACTION`.
"""

from __future__ import annotations

import numpy as np

from dataproc.files import read_json, write_json
from dataproc.stages.base import Stage, StageFailed

__all__ = [
    "MAIN_MAX_DT_NS", "MIN_KEPT_S", "MIN_KEPT_FRACTION", "MAX_SPEED_MPS", "EXTREME_SPEED_MPS",
    "MAX_ULTRAWIDE_GAP_NS", "MAX_IMU_GAP_NS", "MIN_TRACKED_FRACTION", "MIN_TAG_FRACTION", "SCALE_PROBLEMS",
    "THRESHOLDS", "longest_run", "max_gap_in_span", "validate_episode", "ValidateStage",
]

MAIN_MAX_DT_NS = 25_000_000          # main frame used for a grid step must be within 25 ms of it
MIN_KEPT_S = 2.0                     # kept span at least 2 s ...
MIN_KEPT_FRACTION = 0.8              # ... and at least 80 % of the button span
MAX_SPEED_MPS = 3.0                  # iPhUMI align_validate max_pose_delta_m_per_s
EXTREME_SPEED_MPS = 7.0              # reported separately
MAX_ULTRAWIDE_GAP_NS = 100_000_000   # 100 ms between ultrawide frames
MAX_IMU_GAP_NS = 20_000_000          # 20 ms between IMU samples
MIN_TRACKED_FRACTION = 0.95          # SLAM-tracked ultrawide frames in the span
MIN_TAG_FRACTION = 0.8               # ultrawide frames in the span with >= 1 usable finger tag
SCALE_PROBLEMS = ("scale_mismatch", "scale_unobservable")

THRESHOLDS = {
    "main_max_dt_ns": MAIN_MAX_DT_NS, "min_kept_s": MIN_KEPT_S, "min_kept_fraction": MIN_KEPT_FRACTION,
    "max_speed_mps": MAX_SPEED_MPS, "extreme_speed_mps": EXTREME_SPEED_MPS,
    "max_ultrawide_gap_ns": MAX_ULTRAWIDE_GAP_NS, "max_imu_gap_ns": MAX_IMU_GAP_NS,
    "min_tracked_fraction": MIN_TRACKED_FRACTION, "min_tag_fraction": MIN_TAG_FRACTION,
}


def longest_run(mask) -> tuple[int, int]:
    """``(i0, i1)`` of the longest run of True in ``mask`` (``mask[i0:i1]`` all True; the earliest on ties);
    ``(0, 0)`` if there is none."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if not m.any():
        return 0, 0
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def max_gap_in_span(t_ns, lo: int, hi: int) -> int:
    """Largest time without a sample over ``[lo, hi]``: gaps between consecutive samples that overlap the span,
    plus the edges (``t[0] - lo`` if the stream starts late, ``hi - t[-1]`` if it ends early). No sample: hi - lo."""
    t = np.asarray(t_ns, dtype=np.int64).reshape(-1)
    if len(t) == 0:
        return int(hi - lo)
    a = max(int(np.searchsorted(t, lo, side="right")) - 1, 0)
    b = min(int(np.searchsorted(t, hi, side="left")), len(t) - 1)
    gaps = [int(np.diff(t[a:b + 1]).max())] if b > a else []
    gaps.append(max(int(t[0]) - lo, 0))
    gaps.append(max(hi - int(t[-1]), 0))
    return max(gaps)


def _in_span(t, lo, hi) -> np.ndarray:
    t = np.asarray(t, dtype=np.int64)
    return (t >= lo) & (t <= hi)


def validate_episode(*, t_grid_ns, pose_valid, width_valid, main_valid, eef_pos, uw_t_ns, imu_t_ns,
                     slam_t_frames_ns, slam_tracked, slam_problems, uw_frame_has_width) -> dict:
    """Run the rules of the module docstring.

    ``uw_t_ns``: every ultrawide frame time; ``slam_t_frames_ns``/``slam_tracked``: the SLAM result's per-frame
    arrays; ``uw_frame_has_width`` (per ``uw_t_ns``): the width stage found at least one usable tag. Returns
    ``{"status": "done" | "rejected", "reasons", "kept": {i0, i1, n, seconds, fraction, t_lo_ns, t_hi_ns},
    "metrics", "thresholds"}``; the kept rows are ``t_grid_ns[i0:i1]``.
    """
    tg = np.asarray(t_grid_ns, dtype=np.int64).reshape(-1)
    n = len(tg)
    usable = (np.asarray(pose_valid, bool) & np.asarray(width_valid, bool) & np.asarray(main_valid, bool))
    i0, i1 = longest_run(usable)
    dt_s = (float(tg[1] - tg[0]) * 1e-9) if n > 1 else 1.0 / 60.0
    k = i1 - i0
    kept = {"i0": i0, "i1": i1, "n": k, "seconds": k * dt_s, "fraction": (k / n) if n else 0.0,
            "t_lo_ns": int(tg[i0]) if k else None, "t_hi_ns": int(tg[i1 - 1]) if k else None}
    metrics = {"n_grid": n, "n_usable": int(usable.sum()), "n_pose_valid": int(np.sum(pose_valid)),
               "n_width_valid": int(np.sum(width_valid)), "n_main_valid": int(np.sum(main_valid))}
    reasons: list[str] = []
    out = {"status": "rejected", "reasons": reasons, "kept": kept, "metrics": metrics, "thresholds": dict(THRESHOLDS)}
    if k < 2:
        reasons.append("no_valid_span")
        return out
    if kept["seconds"] < MIN_KEPT_S:
        reasons.append("too_short")
    if kept["fraction"] < MIN_KEPT_FRACTION:
        reasons.append("kept_fraction")
    lo, hi = kept["t_lo_ns"], kept["t_hi_ns"]

    pos = np.asarray(eef_pos, dtype=np.float64)[i0:i1]
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / (np.diff(tg[i0:i1]) * 1e-9)
    vmax = float(speed.max())
    metrics["max_speed_mps"] = vmax
    if vmax > MAX_SPEED_MPS:
        reasons.append("speed")
    if vmax > EXTREME_SPEED_MPS:
        reasons.append("speed_extreme")

    uw_gap = max_gap_in_span(uw_t_ns, lo, hi)
    imu_gap = max_gap_in_span(imu_t_ns, lo, hi)
    metrics["max_ultrawide_gap_ms"] = uw_gap / 1e6
    metrics["max_imu_gap_ms"] = imu_gap / 1e6
    if uw_gap > MAX_ULTRAWIDE_GAP_NS:
        reasons.append("ultrawide_gap")
    if imu_gap > MAX_IMU_GAP_NS:
        reasons.append("imu_gap")

    sel = _in_span(slam_t_frames_ns, lo, hi)
    tracked = float(np.mean(np.asarray(slam_tracked, bool)[sel])) if sel.any() else 0.0
    metrics["slam_tracked_fraction"] = tracked
    if tracked < MIN_TRACKED_FRACTION:
        reasons.append("slam_tracking")

    scale = [c for c in SCALE_PROBLEMS if any(str(p).startswith(c) for p in slam_problems)]
    metrics["slam_scale_problems"] = scale
    reasons += scale

    sel = _in_span(uw_t_ns, lo, hi)
    tags = float(np.mean(np.asarray(uw_frame_has_width, bool)[sel])) if sel.any() else 0.0
    metrics["tag_fraction"] = tags
    if tags < MIN_TAG_FRACTION:
        reasons.append("tag_coverage")

    out["status"] = "rejected" if reasons else "done"
    return out


# --------------------------------------------------------------------------------------------- the stage


class ValidateStage(Stage):
    """**validate**. Reads the grid, pose, width and slam products plus the raw main / ultrawide frame times and
    IMU times; runs :func:`validate_episode`. Writes ``processed/validate.json`` (status, reasons, kept span,
    metrics, thresholds). Rejects with every reason found. Skipped when the thresholds, the camera / IMU time
    models and the slam, grid, pose and width hashes are unchanged and none of them re-ran."""

    name = "validate"
    deps = ("slam", "grid", "pose", "width")

    def inputs(self, ctx, ep) -> dict:
        s = ctx.sections
        return {"cameras.main": s["cameras.main"], "cameras.ultrawide": s["cameras.ultrawide"],
                "imu.time_offset_ns": s["imu.time_offset_ns"], "thresholds": THRESHOLDS}

    def run(self, ctx, ep) -> dict:
        from dataproc.export.frames import nearest_frame_index
        from dataproc.export.images import camera_times
        from dataproc.slam.base import ultrawide_times_ns

        rig, raw, grid = ctx.rig, ep.raw, ep.grid()
        _, main_valid = nearest_frame_index(grid, camera_times(raw, rig, "main"), MAIN_MAX_DT_NS)
        slam, pose, width = ep.slam(), ep.pose(), ep.width()
        uw_t = ultrawide_times_ns(raw, rig)
        has_width = np.isin(uw_t, np.asarray(width["frame_t_ns"])[np.asarray(width["frame_has_width"], bool)])
        res = validate_episode(
            t_grid_ns=grid, pose_valid=pose["valid"], width_valid=width["valid"], main_valid=main_valid,
            eef_pos=pose["pos"], uw_t_ns=uw_t, imu_t_ns=rig.imu_time_ns(raw.imu.t_ns),
            slam_t_frames_ns=slam["t_frames_ns"], slam_tracked=slam["tracked"], slam_problems=slam["problems"],
            uw_frame_has_width=has_width)
        write_json(ep.processed / "validate.json", res)
        ep.products["kept"] = read_json(ep.processed / "validate.json")["kept"]
        if res["status"] != "done":
            raise StageFailed("rejected", res["reasons"], details={"file": "validate.json", "kept": res["kept"]})
        return {"file": "validate.json", "kept": res["kept"], "metrics": res["metrics"]}
