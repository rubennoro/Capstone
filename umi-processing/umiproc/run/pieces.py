"""What happens to each piece of a localized demo (docs/DESIGN.md "Checks -> run", "Episodes"): its poses in the
**table frame**, the copied pose checks, and, once the folder has a gripper calibration, its widths, the copied
validate checks, the training images and its ``episode.zarr``. Pure functions of arrays and the raw episode;
:mod:`umiproc.run.recording` does the bookkeeping.

**Table frame** (UMI ``06_generate_dataset_plan.py:110-119, 617-626``: ``tx_tag_slam = inv(tx_slam_tag)``,
``tx_tag_cam = tx_tag_slam @ tx_slam_cam``): every pose of the localized demo is in the map's SLAM world (the
camera frame of the map's first keyframe), and the map's ``tx_slam_tag.json`` is the table tag's pose in that same
world (map check M5). So ``T_tag_cam = inv(tx_slam_tag) @ T_W_cam`` and ``T_tag_imu = T_tag_cam @ inv(T_imu_cam)``.
The tag lies flat on the table (z up, M5: its normal within 5 deg of gravity), so every demo of the session comes
out in the same table frame. The TCP is then the copied pose stage's ``T_world_offset @ T_tag_imu @ T_imu_cam @
T_cam_tcp`` (``umiproc.dp.stages.pose.tcp_poses``; ``T_world_offset`` is the rig's training-world offset, as the
copy always applied it), on the copied 60 Hz grid (the recording's button span).

**Pose checks per piece** (:func:`pose_checks`): the copied ``finalize_trajectory`` on the piece's visually tracked
IMU poses in the SLAM world (gravity alignment first, as map check M4 does, so the checks do not depend on the
tag's small tilt): its ``gravity_check`` (|g| within 0.5 m/s^2 after the alignment) and ``scale_check`` (|s - 1| <=
0.1, observable); the copied validate's TCP speed (per 60 Hz step) <= 3 m/s, ``speed_extreme`` > 7 m/s. The
recording's **orientation** (:func:`orientation_check`): map check M4's ``orientation_consistency`` over every
tracked frame of the demo (the SLAM orientation agrees with gravity, every 2.5 s window within 3 deg of the demo's
mean), judged when the demo has >= 4 windows (10 s tracked); shorter demos are reported as not judged.

**Finishing a piece** (:func:`finish_piece`, needs the gripper range): the copied width functions on the effective
rig (``episode_widths``, ``width_on_grid``: at most 200 ms between detections), the copied ``validate_episode`` on
the piece's grid rows (the kept span = its longest run with pose, width and a main frame within 25 ms; kept
fraction >= 0.8 of the piece; speed; ultrawide and IMU gaps; finger-tag coverage >= 0.8 of the frames), except its
``too_short`` (2 s), which UMI's piece rule replaces: the kept span must still cover >= 12 of the piece's frames.
Then the copied images (``prepare_episode_images``) and export (``write_episode_zarr``, schema unchanged).
"""

from __future__ import annotations

import numpy as np

from umiproc.run.rules import MIN_PIECE_FRAMES

__all__ = ["PIECE_REASONS", "table_imu_trajectory", "slam_imu_trajectory", "grid_rows", "pose_checks",
           "orientation_check", "finish_piece", "piece_zarr_name", "frames_covered"]

PIECE_REASONS = {
    "no_grid_rows": "the piece lies outside the recording's button span (no 60 Hz rows)",
    "gravity": "the copied gravity check failed (|mean specific force| not within 0.5 m/s^2 of g)",
    "scale_mismatch": "the copied metric-scale check: the trajectory's scale is off by more than 10 %",
    "scale_unobservable": "the copied metric-scale check could not measure the scale (too short or too little "
                          "motion; it needs at least about 1 s)",
    "speed": "the TCP moved faster than 3 m/s (the copied validate)",
    "speed_extreme": "the TCP moved faster than 7 m/s (a SLAM jump)",
    "orientation": "the SLAM orientation disagrees with gravity by more than 3 deg (map check M4's sub-check)",
    "piece_too_short": f"after trimming to its usable span the piece covers fewer than {MIN_PIECE_FRAMES} frames",
}


def _T_cam_imu(T_imu_cam) -> np.ndarray:
    return np.linalg.inv(np.asarray(T_imu_cam, dtype=np.float64))


def slam_imu_trajectory(traj, sel, T_imu_cam):
    """The rows ``sel`` (bool mask or index array; visually tracked rows only) of a ForkTrajectory as a copied
    ``Trajectory`` of the IMU in the SLAM world (``T_W_imu = T_W_cam @ inv(T_imu_cam)``)."""
    from umiproc.dp.geometry.trajectory import Trajectory

    idx = np.flatnonzero(sel) if np.asarray(sel).dtype == bool else np.asarray(sel, dtype=np.int64)
    if not np.all(np.asarray(traj.tracked, bool)[idx]):
        raise ValueError("slam_imu_trajectory: only visually tracked rows have a pose to use")
    T = np.einsum("nij,jk->nik", traj.T_W_cam[idx], _T_cam_imu(T_imu_cam))
    return Trajectory.from_T(traj.t_ns[idx], T, frame="imu", world="slam_map")


def table_imu_trajectory(traj, sel, tx_slam_tag, T_imu_cam):
    """The rows ``sel`` as IMU poses in the **table frame**: ``T_tag_imu = inv(tx_slam_tag) @ T_W_cam @
    inv(T_imu_cam)`` (UMI 06:119, 626)."""
    from umiproc.dp.geometry.trajectory import Trajectory

    s = slam_imu_trajectory(traj, sel, T_imu_cam)
    T = np.einsum("ij,njk->nik", np.linalg.inv(np.asarray(tx_slam_tag, dtype=np.float64)), s.T())
    return Trajectory.from_T(s.t_ns, T, frame="imu", world="table_tag")


def grid_rows(grid_ns, piece) -> np.ndarray:
    """The 60 Hz grid times inside the piece's first..last frame time (inclusive)."""
    g = np.asarray(grid_ns, dtype=np.int64)
    return g[(g >= piece.t_first_ns) & (g <= piece.t_last_ns)]


def _speed(pos: np.ndarray, rows: np.ndarray) -> float | None:
    """The copied validate's TCP speed: per-step position change over the step's time; the largest, m/s."""
    if len(rows) < 2:
        return None
    v = np.linalg.norm(np.diff(np.asarray(pos, np.float64), axis=0), axis=1) / (np.diff(rows) * 1e-9)
    return float(v.max())


def pose_checks(piece, traj, episode, rig, tx_slam_tag, grid_ns) -> dict:
    """The pose checks of one piece (module docstring). Returns ``{ok, reasons, values, problems}``; ``values``
    holds the figures (gravity, scale, speed, the copied diagnostics) and ``problems`` the copied
    ``finalize_trajectory`` problem strings."""
    from umiproc.dp.slam.base import SlamError, finalize_trajectory
    from umiproc.dp.stages.geometry import POSE_MAX_GAP_NS
    from umiproc.dp.stages.pose import tcp_poses
    from umiproc.dp.stages.validate import EXTREME_SPEED_MPS, MAX_SPEED_MPS

    T_imu_cam = rig.cameras["ultrawide"].T_imu_cam
    sel = np.arange(piece.start, piece.stop)
    reasons: list[str] = []
    values: dict = {"frames": [piece.start, piece.stop], "n_frames": piece.n_frames}
    problems: list[str] = []
    try:
        fin = finalize_trajectory(slam_imu_trajectory(traj, sel, T_imu_cam), episode, rig, traj.t_ns[sel],
                                  align=True, align_min_deg=0.0)
    except (SlamError, ValueError) as e:
        fin = None
        values["finalize_error"] = str(e)
        reasons += ["gravity", "scale_unobservable"]
    if fin is not None:
        d = fin.diagnostics
        g, s = d.get("gravity_check", {}), d.get("scale_check", {})
        problems = list(fin.problems)
        values.update(gravity_ok=bool(g.get("ok")), g_norm=g.get("g_norm"), scale_status=s.get("status"),
                      scale=s.get("scale"), scale_std_err=s.get("std_err"), scale_problems=s.get("problems", []),
                      gravity_alignment_deg=(d.get("gravity_alignment") or {}).get("angle_deg"),
                      time_offset_check=d.get("time_offset_check"), motion_check=d.get("motion_check"))
        if not g.get("ok"):
            reasons.append("gravity")
        if s.get("status") != "ok":
            reasons.append(s.get("status") or "scale_unobservable")
    rows = grid_rows(grid_ns, piece)
    values["grid_rows"] = int(len(rows))
    if len(rows) < 2:
        reasons.append("no_grid_rows")
    else:
        table = table_imu_trajectory(traj, sel, tx_slam_tag, T_imu_cam)
        pos, _rot, valid = tcp_poses(table, rows, rig, max_gap_ns=POSE_MAX_GAP_NS)
        vmax = _speed(pos, rows)
        values.update(max_speed_mps=vmax, pose_valid_rows=int(valid.sum()))
        if vmax is not None and vmax > MAX_SPEED_MPS:
            reasons.append("speed")
        if vmax is not None and vmax > EXTREME_SPEED_MPS:
            reasons.append("speed_extreme")
    return {"ok": not reasons, "reasons": list(dict.fromkeys(reasons)), "values": values, "problems": problems}


def orientation_check(traj, episode, rig) -> dict:
    """Map check M4's orientation sub-check on every visually tracked frame of the demo (module docstring).
    Returns ``{ok, judged, ...the orientation_consistency result}``; ``ok`` is True when it is not judged
    (fewer than 4 windows)."""
    from umiproc.dp.slam.base import imu_samples
    from umiproc.map.checks import M4_MIN_WINDOWS, orientation_consistency

    tr = np.asarray(traj.tracked, bool)
    if tr.sum() < 2:
        return {"ok": True, "judged": False, "reason": "fewer than 2 tracked frames", "windows": 0}
    imu_t, _g, acc = imu_samples(episode, rig)
    o = orientation_consistency(slam_imu_trajectory(traj, tr, rig.cameras["ultrawide"].T_imu_cam), imu_t, acc,
                                t_ref_ns=int(traj.t_ns[0]))
    judged = (o.get("windows") or 0) >= M4_MIN_WINDOWS
    return {**o, "judged": judged, "ok": bool(o["ok"]) if judged else True}


def frames_covered(times_ns, piece, t_lo_ns: int, t_hi_ns: int) -> int:
    """The piece's frames within one grid step of the kept span ``[t_lo, t_hi]`` (the frames its rows were made
    from)."""
    from umiproc.dp.stages.geometry import GRID_DT_NS

    t = np.asarray(times_ns, dtype=np.int64)[piece.start:piece.stop]
    return int(np.sum((t >= t_lo_ns - GRID_DT_NS) & (t <= t_hi_ns + GRID_DT_NS)))


def piece_zarr_name(pi_name: str, index: int, n_pieces: int) -> str:
    """``episode.zarr``'s name: the recording's ``pi_name`` when it is one piece, else ``<pi_name>_piece<k>`` (the
    copied packaging refuses two episodes with one name)."""
    return pi_name if n_pieces == 1 else f"{pi_name}_piece{index}"


def finish_piece(piece, traj, raw, rig, prep, tx_slam_tag, grid_ns, widths, *, uw_t_ns, imu_t_ns, label: str,
                 name: str, out_zarr) -> dict:
    """Widths, the copied validate, images and ``episode.zarr`` (written to ``out_zarr``) of one piece that passed
    its pose checks. ``widths``: ``(t_w, w, frame_t_ns, frame_has_width)`` of the recording (the copied
    ``episode_widths`` on every ultrawide frame). Returns ``{ok, stage, reasons, message, values}``: ``stage`` is
    where it stopped (``validate``, ``images``) or ``export`` when it was written."""
    from umiproc.dp.export.frames import nearest_frame_index
    from umiproc.dp.export.images import camera_times, prepare_episode_images
    from umiproc.dp.export.zarr_writer import EpisodeArrays, write_episode_zarr
    from umiproc.dp.stages.geometry import POSE_MAX_GAP_NS, WIDTH_MAX_GAP_NS
    from umiproc.dp.stages.output import UW_RATE_HZ
    from umiproc.dp.stages.pose import tcp_poses
    from umiproc.dp.stages.validate import MAIN_MAX_DT_NS, validate_episode
    from umiproc.dp.stages.width import width_on_grid

    t_w, w, frame_t, frame_has_width = widths
    rows = grid_rows(grid_ns, piece)
    sel = np.arange(piece.start, piece.stop)
    table = table_imu_trajectory(traj, sel, tx_slam_tag, rig.cameras["ultrawide"].T_imu_cam)
    pos, rotvec, pose_valid = tcp_poses(table, rows, rig, max_gap_ns=POSE_MAX_GAP_NS)
    width, width_valid = width_on_grid(t_w, w, rows, max_gap_ns=WIDTH_MAX_GAP_NS)
    _, main_valid = nearest_frame_index(rows, camera_times(raw, rig, "main"), MAIN_MAX_DT_NS)
    has_width = np.isin(uw_t_ns, np.asarray(frame_t)[np.asarray(frame_has_width, bool)])
    v = validate_episode(t_grid_ns=rows, pose_valid=pose_valid, width_valid=width_valid, main_valid=main_valid,
                         eef_pos=pos, uw_t_ns=uw_t_ns, imu_t_ns=imu_t_ns, slam_t_frames_ns=traj.t_ns,
                         slam_tracked=traj.tracked, slam_problems=[], uw_frame_has_width=has_width)
    reasons = [r for r in v["reasons"] if r != "too_short"]       # replaced by UMI's piece length below
    k = v["kept"]
    values = {"validate": v, "grid_rows": int(len(rows))}
    if k["n"] >= 2:
        values["frames_covered"] = n_cov = frames_covered(traj.t_ns, piece, k["t_lo_ns"], k["t_hi_ns"])
        if n_cov < MIN_PIECE_FRAMES:
            reasons.append("piece_too_short")
    if reasons:
        return {"ok": False, "stage": "validate", "reasons": reasons,
                "message": "; ".join(reasons), "values": values}
    kept = rows[k["i0"]:k["i1"]]
    imgs = prepare_episode_images(raw, rig, kept, main_max_dt_ns=MAIN_MAX_DT_NS, uw_rate_hz=UW_RATE_HZ, prep=prep)
    if not imgs.main_valid.all():
        return {"ok": False, "stage": "images", "reasons": ["main_frame_missing"],
                "message": f"{int((~imgs.main_valid).sum())} kept steps without a main frame", "values": values}
    sl = slice(k["i0"], k["i1"])
    arrays = EpisodeArrays(name=name, task_label=label, main_rgb=imgs.main_rgb, uw_rgb=imgs.uw_rgb,
                           uw_upsample_local=imgs.uw_upsample_local, eef_pos=pos[sl], eef_rot_axis_angle=rotvec[sl],
                           gripper_width=width[sl])
    write_episode_zarr(arrays, out_zarr)
    values.update(n_steps=arrays.n_steps, n_uw=arrays.n_uw, name=name, task_label=label,
                  kept_seconds=float(k["seconds"]), t_lo_ns=k["t_lo_ns"], t_hi_ns=k["t_hi_ns"])
    return {"ok": True, "stage": "export", "reasons": [], "message": "", "values": values}
