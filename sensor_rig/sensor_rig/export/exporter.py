"""Raw rig episode + trajectory + rig.yaml -> iPhone-format episode folder (brief D2.3, D4-D7; P1 §4.2-4.7).

Output: ``<out_root>/<NAME[:10]>/<NAME>/<NAME>{.json,_rgb.mp4,_ultrawidergb.mp4}`` (calibrations: no ``_rgb.mp4``).
The folder is built in ``<out_root>/.partial-<NAME>/`` (hidden, so group.py's ``**/*.json`` glob never sees it),
validated, and moved into place. Each attempt appends one line to ``<out_root>/.sensor_rig/export_log.jsonl``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sensor_rig.export.grid import build_grid, fill_slots, fps_abort, lag_stats
from sensor_rig.export.iphone_json import build_json, dumps_swift
from sensor_rig.export.slotting import select_uw, slot_uw, uw_pts
from sensor_rig.export.video import Mp4Writer, probe_video
from sensor_rig.pose.frames import orthonormalize, rig_vio_to_arkit_main
from sensor_rig.pose.trajectory import load_tum
from sensor_rig.raw_episode import RawEpisode, find_raw_episodes
from sensor_rig.timefmt import ISO_RE, date_dir, iso_ms_array

LOG_DIR = ".sensor_rig"
LOG_FILE = "export_log.jsonl"


@dataclass
class ExportOptions:
    grid_hz: float = 60.0
    native_min_fps: float = 50.0
    ultrawide_rate_hz: float | None = 10.0  # None = keep every UW frame (max one per slot)
    max_pose_gap_s: float = 0.10
    max_speed_m_s: float = 7.0
    warn_speed_m_s: float = 3.0
    min_fps: float = 31.0  # native-mode live-fps rule (VC:1821)
    min_fps_ratio: float = 0.52  # fill-mode rule: frames/s >= ratio * nominal main fps
    min_duration_s: float = 1.0
    max_main_gap_s: float | None = 0.10  # F6: abort on a longer gap between main frames (None = off)
    max_uw_gap_s: float | None = 0.25  # F6: abort on a longer gap between ultrawide frames (None = off)
    max_rotvec_deg: float = 170.0  # F9: warn above this rotation angle (align's rotvec interpolation breaks near pi)
    fix_world_yaw: bool = False  # F9: also pick a per-episode world yaw that keeps the angles below max_rotvec_deg
    fill_max_extra_ns: int = 2_000_000  # fill mode: a frame may be up to half an interval + this from its slot
    min_coverage_warn: float = 0.98
    container: str = "mov"
    time_base_den: int = 600
    main_crf: int = 18
    uw_crf: int = 18
    preset: str = "faster"
    gop: int = 30
    bframes: int = 2
    label_task: str | None = None  # if set: labelType Predefined, one task spanning rgbTimes[0..-1]
    calibration_run: str | None = None  # override meta.gripper_calibration_run_name
    gripper_id: str | None = None
    sides_present: list[str] | None = None
    rig_override: Path | None = None  # re-export with an improved calibration
    trajectory: Path | None = None  # default <ep>/trajectory/trajectory.tum
    trajectory_frame: str | None = None  # default: trajectory.json "frame", else rig.vio.trajectory_frame
    force: bool = False  # export even if the episode meta status is not "complete"
    overwrite: bool = False
    validate: bool = True


@dataclass
class ExportResult:
    name: str
    status: str  # "exported" | "aborted" | "skipped"
    reason: str = ""
    out_dir: Path | None = None
    frame_count: int = 0
    uw_count: int = 0
    trim_head_s: float = 0.0
    trim_tail_s: float = 0.0
    coverage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    grid_mode: str = ""
    lag: dict = field(default_factory=dict)  # fill mode: image-vs-pose lag stats over align's samples
    world_yaw_extra_deg: float = 0.0  # F9: extra world yaw applied to keep rotation angles < max_rotvec_deg

    def to_json(self) -> dict:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir) if self.out_dir else None
        return d


class ExportAbort(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _check_roots(raw_dir: Path, out_root: Path) -> None:
    """The export root must neither contain the raw episode nor live inside the raw root."""
    r, o = raw_dir.resolve(), out_root.resolve()
    if o == r or o in r.parents:
        raise ExportAbort("bad_out_root", f"export root {o} contains the raw episode {r}")
    raw_root = r.parent.parent if re.fullmatch(r"\d{4}-\d\d-\d\d", r.parent.name) else r.parent
    if raw_root in [o, *o.parents]:
        raise ExportAbort("bad_out_root", f"export root {o} is inside the raw root {raw_root}")


def _log(out_root: Path, res: ExportResult) -> None:
    d = out_root / LOG_DIR
    d.mkdir(parents=True, exist_ok=True)
    rec = res.to_json()
    rec["time"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with open(d / LOG_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _load_trajectory(ep: RawEpisode, rig, opts: ExportOptions):
    path = Path(opts.trajectory) if opts.trajectory else ep.trajectory_path()
    if path is None or not path.is_file():
        raise ExportAbort("no_trajectory", f"no trajectory for {ep.name} (run pose-import first)")
    frame = opts.trajectory_frame
    if frame is None:
        info = ep.trajectory_info() if not opts.trajectory else None
        frame = (info or {}).get("frame") or rig.vio.trajectory_frame
    return load_tum(path, frame=frame)


def _trim_by_pose(valid: np.ndarray) -> tuple[int, int]:
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        raise ExportAbort("no_pose", "no grid slot has a valid pose")
    i0, i1 = int(idx[0]), int(idx[-1])
    if not valid[i0:i1 + 1].all():
        bad = i0 + int(np.flatnonzero(~valid[i0:i1 + 1])[0])
        raise ExportAbort("pose_gap_mid_episode", f"first slot without pose inside the episode: {bad}")
    return i0, i1


def _check_poses(T: np.ndarray, t_ns: np.ndarray, opts: ExportOptions, warnings: list[str]) -> None:
    if not np.all(np.isfinite(T)):
        raise ExportAbort("nan_pose")
    if len(T) > 1:
        speed = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) / (np.diff(t_ns) * 1e-9)
        vmax = float(speed.max())
        if vmax > opts.max_speed_m_s:
            raise ExportAbort("speed", f"{vmax:.2f} m/s > {opts.max_speed_m_s} m/s")
        if vmax > opts.warn_speed_m_s:
            warnings.append(f"max speed {vmax:.2f} m/s > {opts.warn_speed_m_s} m/s (align_validate may reject)")
    angles = Rotation.from_matrix(T[:, :3, :3]).magnitude()
    if np.any(angles < 1e-6):
        raise ExportAbort("identity_pose", "an exported rotation is (numerically) the identity")


def _yaw_y(deg: float) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("y", deg, degrees=True).as_matrix()
    return M


def choose_world_yaw(T: np.ndarray, max_deg: float = 170.0, step_deg: float = 5.0) -> float:
    """F9: iPhUMI align interpolates rotation vectors component-wise, which breaks near |theta| = pi. The ARKit world
    yaw (about +y, gravity) is arbitrary for the relative policy representation, so if any pose's rotation angle
    exceeds ``max_deg``, return the extra yaw (deg) that minimises the largest angle (0 if no change is needed)."""
    ang = np.degrees(Rotation.from_matrix(T[:, :3, :3]).magnitude())
    if ang.max() <= max_deg:
        return 0.0
    best, best_val = 0.0, float(ang.max())
    R = Rotation.from_matrix(T[:, :3, :3])
    for psi in np.arange(step_deg, 360.0, step_deg):
        a = np.degrees((Rotation.from_euler("y", psi, degrees=True) * R).magnitude())
        if a.min() > 0.1 and a.max() < best_val:
            best, best_val = float(psi), float(a.max())
    return best


def _gap_abort(t_ns: np.ndarray, t0: int, max_gap_s: float | None, what: str) -> None:
    """F6: abort if two consecutive frames are more than ``max_gap_s`` apart (reason names the gap and its time)."""
    if max_gap_s is None or len(t_ns) < 2:
        return
    d = np.diff(t_ns)
    i = int(np.argmax(d))
    if d[i] > max_gap_s * 1e9:
        raise ExportAbort(f"{what}_gap", f"{d[i] / 1e9:.3f} s gap between {what} frames at "
                          f"{(t_ns[i] - t0) / 1e9:.2f} s into the episode (> {max_gap_s} s)")


def export_episode(raw_dir, out_root, opts: ExportOptions | None = None) -> ExportResult:
    opts = opts or ExportOptions()
    raw_dir, out_root = Path(raw_dir), Path(out_root)
    ep = RawEpisode(raw_dir)
    res = ExportResult(name=ep.name, status="aborted")
    out_root.mkdir(parents=True, exist_ok=True)
    final = out_root / date_dir(ep.name) / ep.name
    partial = out_root / f".partial-{ep.name}"
    try:
        _check_roots(raw_dir, out_root)
        if final.exists() and not opts.overwrite:
            res.status, res.reason, res.out_dir = "skipped", "exists", final
            return res
        if ep.meta.get("status") != "complete" and not opts.force:
            raise ExportAbort("not_complete", f"raw episode status is {ep.meta.get('status')!r}")
        if partial.exists():  # leftover from a crashed export
            shutil.rmtree(partial)
        _export_into(ep, partial, opts, res)
        if final.exists():
            shutil.rmtree(final)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, final)
        res.status, res.out_dir = "exported", final
    except ExportAbort as e:
        res.status, res.reason, res.out_dir = "aborted", e.reason, None
        if e.detail:
            res.warnings.append(e.detail)
    finally:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        _log(out_root, res)
    return res


def _export_into(ep: RawEpisode, out: Path, opts: ExportOptions, res: ExportResult) -> None:
    rig = ep.rig(opts.rig_override)
    meta = ep.meta
    demo = not ep.is_calibration
    warnings = res.warnings

    gripper_id = opts.gripper_id or meta.get("gripper_id") or rig.export.gripper_id
    calib_run = opts.calibration_run if opts.calibration_run is not None else meta.get(
        "gripper_calibration_run_name") or ""
    if demo and meta.get("side") != "head" and not calib_run:
        raise ExportAbort("no_calibration_run", "demonstration has no gripperCalibrationRunName; pass "
                          "--calibration-run <NAME of the gripper calibration episode, incl. side>")

    # ---- grid (main camera drives the 60 Hz record)
    t_main = np.asarray(rig.capture_time_ns("main", ep.camera_times("main")), dtype=np.int64)
    if len(t_main) < 2:
        raise ExportAbort("no_main_frames")
    grid = build_grid(t_main, opts.grid_hz, opts.native_min_fps)
    res.grid_mode = grid.mode
    min_frames = opts.min_fps if grid.mode == "native" else opts.min_fps_ratio * float(rig.cameras["main"].fps)
    why = fps_abort(t_main, min_frames)
    if why:
        raise ExportAbort("fps", why)
    G = len(grid)

    # ---- poses and trimming (demonstrations only)
    if demo:
        traj = _load_trajectory(ep, rig, opts)
        T_S, valid = traj.sample(grid.t_ns, max_gap_ns=int(opts.max_pose_gap_s * 1e9))
        i0, i1 = _trim_by_pose(valid)
    else:
        i0, i1 = 0, G - 1
    wall = ep.wall_offset_ns()
    if grid.mode == "fill":
        # F2: align-exact strings; images at align's sample times, poses at the string times
        try:
            fs = fill_slots(t_main, int(grid.t_ns[i0]), int(grid.t_ns[i1]), wall, opts.grid_hz,
                            max_extra_ns=opts.fill_max_extra_ns)
        except ValueError as e:
            raise ExportAbort("too_short", str(e))
        t_grid, src_idx = fs.string_ns, fs.src_idx
        if demo:
            T_S, valid = traj.sample(t_grid, max_gap_ns=int(opts.max_pose_gap_s * 1e9))
            j0, j1 = _trim_by_pose(valid)
            if j0 != 0:
                raise ExportAbort("pose_gap_mid_episode", "fill grid start lost its pose")
            t_grid, src_idx, T_S = t_grid[:j1 + 1], src_idx[:j1 + 1], T_S[:j1 + 1]
            fs = type(fs)(fs.sample_ns[:j1 + 1], fs.string_ns[:j1 + 1], fs.string_unix_ns[:j1 + 1], src_idx)
        res.lag = lag_stats(t_main, fs)
        p0, p1 = 0, len(t_grid) - 1
    else:
        t_grid = grid.t_ns[i0:i1 + 1]
        src_idx = grid.src_idx[i0:i1 + 1]
        p0, p1 = i0, i1
    n = len(t_grid)
    if n < 2 or (t_grid[-1] - t_grid[0]) * 1e-9 < opts.min_duration_s:
        raise ExportAbort("too_short", f"{(t_grid[-1] - t_grid[0]) * 1e-9:.2f} s")
    res.trim_head_s = float((t_grid[0] - grid.t_ns[0]) * 1e-9)
    res.trim_tail_s = float((grid.t_ns[-1] - t_grid[-1]) * 1e-9)
    in_main = t_main[(t_main >= t_grid[0]) & (t_main <= t_grid[-1])]
    _gap_abort(in_main, int(t_grid[0]), opts.max_main_gap_s, "main")
    if demo:
        T = rig_vio_to_arkit_main(rig, T_S if grid.mode == "fill" else T_S[p0:p1 + 1], traj.frame)
        T[:, :3, :3] = orthonormalize(T[:, :3, :3])
        T[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
        yaw = choose_world_yaw(T, opts.max_rotvec_deg)
        if yaw and opts.fix_world_yaw:
            T = _yaw_y(yaw)[None] @ T
            res.world_yaw_extra_deg = yaw
            warnings.append(f"rotation angle > {opts.max_rotvec_deg} deg: world yaw +{yaw:g} deg applied (F9)")
        elif yaw:
            amax = float(np.degrees(Rotation.from_matrix(T[:, :3, :3]).magnitude()).max())
            warnings.append(f"rotation angle up to {amax:.1f} deg > {opts.max_rotvec_deg} deg: iPhUMI align "
                            f"interpolates rotation vectors component-wise and is wrong near 180 deg; re-export with "
                            f"--fix-world-yaw (would add {yaw:g} deg of world yaw) (F9)")
        _check_poses(T, t_grid, opts, warnings)

    # ---- time strings (one UNIX offset per episode, taken at record start)
    times = iso_ms_array(t_grid + wall)
    if any(a >= b for a, b in zip(times, times[1:])) or not all(ISO_RE.match(s) for s in times):
        raise ExportAbort("time_strings", "slot time strings are not strictly increasing ISO-ms strings")

    # ---- ultrawide: decimate, slot, PTS
    t_uw = np.asarray(rig.capture_time_ns("ultrawide", ep.camera_times("ultrawide")), dtype=np.int64)
    h = int(np.median(np.diff(t_grid)) // 2) if n > 1 else 8_333_333
    in_range = np.flatnonzero((t_uw >= t_grid[0] - h) & (t_uw <= t_grid[-1] + h))
    _gap_abort(t_uw[in_range], int(t_grid[0]), opts.max_uw_gap_s, "ultrawide")
    sel = in_range[select_uw(t_uw[in_range], opts.ultrawide_rate_hz)] if len(in_range) else in_range
    slot_of = slot_uw(t_grid, t_uw, sel)
    occ = np.flatnonzero(slot_of >= 0)
    if len(occ) == 0:
        raise ExportAbort("no_ultrawide", "no ultrawide frame falls on the exported grid")
    pts_uw = uw_pts(t_grid, slot_of, opts.time_base_den)
    if np.any(np.diff(pts_uw) <= 0):
        raise ExportAbort("pts_collision", "ultrawide PTS not strictly increasing at this time base")
    uw_times = [times[i] if slot_of[i] >= 0 else "" for i in range(n)]
    res.frame_count, res.uw_count = n, int(len(occ))

    # ---- images
    vmain = rig.virtual_camera("main", "export")
    vuw = rig.virtual_camera("ultrawide", "export")
    rm_uw = rig.remap("ultrawide", "export")
    res.coverage["ultrawide"] = float(rm_uw.coverage)
    out.mkdir(parents=True)
    name = ep.name
    vopts = dict(time_base_den=opts.time_base_den, container=opts.container, preset=opts.preset, gop=opts.gop,
                 bframes=opts.bframes)
    expected = {}
    if demo:
        rm_main = rig.remap("main", "export")
        res.coverage["main"] = float(rm_main.coverage)
        pts_main = np.round((t_grid - t_grid[0]) * opts.time_base_den / 1e9).astype(np.int64)
        if np.any(np.diff(pts_main) <= 0):
            raise ExportAbort("pts_collision", "main PTS not strictly increasing at this time base")
        path = out / f"{name}_rgb.mp4"
        with Mp4Writer(path, vmain.width, vmain.height, crf=opts.main_crf, **vopts) as w:
            last_src, img = None, None
            for k in range(n):
                if src_idx[k] != last_src:
                    img = rm_main.apply(ep.read_image("main", int(src_idx[k])))
                    last_src = src_idx[k]
                w.write(img, int(pts_main[k]))
        expected[path] = (pts_main, vmain.width, vmain.height)
    path = out / f"{name}_ultrawidergb.mp4"
    with Mp4Writer(path, vuw.width, vuw.height, crf=opts.uw_crf, **vopts) as w:
        for k, i in enumerate(occ):
            w.write(rm_uw.apply(ep.read_image("ultrawide", int(slot_of[i]))), int(pts_uw[k]))
    expected[path] = (pts_uw, vuw.width, vuw.height)
    for cam, cov in res.coverage.items():
        if cov < opts.min_coverage_warn:
            warnings.append(f"{cam} coverage {cov:.3f} < {opts.min_coverage_warn}: black borders enter the images")

    # ---- video postconditions (demux only)
    for path, (pts, w_, h_) in expected.items():
        p = probe_video(path)
        tb = p["time_base"]
        got = [int(round(x * tb * opts.time_base_den)) for x in p["pts_list"]]
        if p["frames_header"] != len(pts) or got != pts.tolist() or (p["width"], p["height"]) != (w_, h_):
            raise ExportAbort("video_postcondition",
                              f"{path.name}: header frames {p['frames_header']} (want {len(pts)}), "
                              f"size {p['width']}x{p['height']}, pts match {got == pts.tolist()}")

    # ---- JSON (written last)
    obj = build_json(
        kind=meta["kind"], frame_count=n, pose_times=times if demo else [],
        pose_transforms=T if demo else [], uw_times=uw_times,
        recording_start_time=meta.get("recording_start_time") or times[0], side=meta["side"],
        session=meta["session"], gripper_id=gripper_id, calibration_run=calib_run,
        sides_present=opts.sides_present or meta.get("sides_present") or [meta["side"]],
        device_name=rig.export.device_name, device_identifier=rig.export.device_identifier,
        K_main=vmain.K_float32_rows(), K_uw=vuw.K_float32_rows(), task_label=opts.label_task,
    )
    (out / f"{name}.json").write_text(dumps_swift(obj))
    if opts.validate:
        from sensor_rig.export.validate import validate_iphone_episode

        errors = validate_iphone_episode(out, name=name)
        if errors:
            raise ExportAbort("validation", "; ".join(errors[:5]))


def export_many(raw_dirs, out_root, opts: ExportOptions | None = None, workers: int = 1) -> list[ExportResult]:
    """Export calibrations first, then demonstrations; warn if a demo's calibration is not in the output."""
    opts = opts or ExportOptions()
    eps = []
    for d in raw_dirs:
        eps.extend(find_raw_episodes(d))
    eps.sort(key=lambda e: (not e.is_calibration, e.name))
    dirs = [e.path for e in eps]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        calib = [d for d, e in zip(dirs, eps) if e.is_calibration]
        demos = [d for d, e in zip(dirs, eps) if not e.is_calibration]
        results = []
        with ProcessPoolExecutor(workers) as ex:
            for group in (calib, demos):
                results += list(ex.map(export_episode, group, [out_root] * len(group), [opts] * len(group)))
    else:
        results = [export_episode(d, out_root, opts) for d in dirs]
    out_root = Path(out_root)
    for e, r in zip(eps, results):
        if e.is_calibration or r.status != "exported":
            continue
        run = opts.calibration_run or e.meta.get("gripper_calibration_run_name") or ""
        if run and not (out_root / date_dir(run) / run / f"{run}.json").is_file():
            r.warnings.append(f"calibration run {run} is not in {out_root}")
    return results
