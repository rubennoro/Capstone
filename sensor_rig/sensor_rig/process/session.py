"""``sensor-rig process-session``: raw rig session -> validated iPhone-format exports, in one command (T4).

    sensor-rig process-session --raw RAW --session S --out EXPORT_ROOT --vio {import-dir,cmd,orbslam3-docker}
        [--traj-dir D] [--vio-cmd TEMPLATE] [--docker-image IMG] [--rig calibrated.yaml] [--task-label T]
        [--workers N] [--overwrite] [--force-episode NAME ...] [--allow-partial]

Per session (every finalized episode of session S under RAW; hidden aborted / discarded / rejected folders are
never read):

1. Checks: the session name is rig-only (``rig*``) and is not used by any iPhone episode in ``data/raw``; every demo
   names a calibration run that is a kept calibration episode of this session.
2. Demos: pose-prep (EuRoC + ORB-SLAM3 settings + finger mask, under ``EXPORT_ROOT/.sensor_rig/vio/S/``; not for
   ``import-dir``) -> VIO hook -> pose-import (gravity, overlap, motion checks; the ``collect --mock-replay`` time
   offset is applied automatically). Calibration episodes skip the pose stage: export never reads a trajectory for
   them (``exporter._export_into`` loads one only ``if demo``; calibration JSONs carry no poses).
3. Export (calibrations first, ``--workers`` processes), then ``validate`` on the export root (incl. the demo ->
   calibration links and, if found, the repo's gripper_calibration.yaml device entry).
4. ``EXPORT_ROOT/.sensor_rig/process_<S>.json``: per-episode status + reason, VIO residuals, lag stats, coverage,
   trim. Prints the ``scripts/run_pipeline.sh all ...`` line (not run).

One failing episode never stops the others. Re-running resumes: an episode whose trajectory was already imported
from the same file, or that is already exported, is skipped (``--overwrite`` redoes everything).
Exit 0 if the calibration and >= 1 demo exported and no demo failed (``--allow-partial``: failed demos allowed);
1 otherwise; 2 on usage errors (bad / colliding session name, nothing found).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from sensor_rig.naming import DEFAULT_IPHONE_RAW, REPO, session_name_problems  # noqa: F401 (shared check)
GRIPPER_CAL_YAML = REPO / "iPhUMI" / "python_package" / "iphumi" / "demonstration_processing" / "calibration" / \
    "gripper_calibration.yaml"


class UsageError(ValueError):
    pass


@dataclass
class EpisodeReport:
    name: str
    kind: str
    status: str = "pending"          # exported | failed
    step: str | None = None          # step that failed
    reason: str | None = None
    calibration_run: str | None = None
    vio: dict = field(default_factory=dict)
    pose_import: dict = field(default_factory=dict)
    export: dict = field(default_factory=dict)

    def fail(self, step: str, reason: str) -> None:
        self.status, self.step, self.reason = "failed", step, reason


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _export_one(args):
    raw_dir, out_root, opts = args
    from sensor_rig.export.exporter import export_episode

    try:
        return export_episode(raw_dir, out_root, opts).to_json()
    except Exception as e:  # noqa: BLE001 - one episode must not stop the others
        return {"name": Path(raw_dir).name, "status": "aborted", "reason": f"exception: {type(e).__name__}: {e}",
                "warnings": [traceback.format_exc(limit=3)]}


def pipeline_line(session: str, out_root: Path, task: str) -> tuple[str, str | None]:
    """(command, note): the run_pipeline.sh call for this export root (repo-relative --raw-dir)."""
    out = out_root.resolve()
    dataset = f"rig-{session}-single"
    note = None
    try:
        rel = out.relative_to(REPO.resolve())
    except ValueError:
        rel = Path("data") / "rig_export" / session
        note = (f"the export root {out} is outside the repo: copy it first, e.g. "
                f"`mkdir -p {REPO / rel} && rsync -a {out}/ {REPO / rel}/` (hidden .sensor_rig/ may be skipped)")
    cmd = (f"scripts/run_pipeline.sh all --task {task} --dataset {dataset} --sessions {session} "
           f"--raw-dir {rel} --smoke-first --note \"rig session {session} (sensor_rig)\"")
    return cmd, note


def process_session(raw, session: str, out_root, vio: str, traj_dir=None, vio_cmd: str | None = None,
                    docker_image: str | None = None, rig_path=None, task_label: str = "block_raise",
                    workers: int = 1, overwrite: bool = False, force_episodes=(), allow_partial: bool = False,
                    iphone_raw=DEFAULT_IPHONE_RAW, log=print) -> tuple[int, dict]:
    from sensor_rig.export.exporter import ExportOptions
    from sensor_rig.export.validate import validate_export_root
    from sensor_rig.pose import vio_hooks
    from sensor_rig.pose.stage import TrajectoryImportError, import_trajectory, prepare
    from sensor_rig.raw_episode import find_raw_episodes

    raw, out_root = Path(raw), Path(out_root)
    probs = session_name_problems(session, Path(iphone_raw))
    if probs:
        raise UsageError("; ".join(probs))
    if vio == "import-dir" and not traj_dir:
        raise UsageError("--vio import-dir needs --traj-dir")
    if vio == "cmd" and not vio_cmd:
        raise UsageError("--vio cmd needs --vio-cmd")
    eps = [e for e in find_raw_episodes(raw) if e.meta.get("session") == session]
    if not eps:
        raise UsageError(f"no finalized episodes of session {session!r} under {raw}")
    eps.sort(key=lambda e: (not e.is_calibration, e.name))
    force = set(force_episodes or ())
    out_root.mkdir(parents=True, exist_ok=True)
    reports = {e.name: EpisodeReport(e.name, e.kind) for e in eps}
    cals = {e.name for e in eps if e.is_calibration}
    rig_override = Path(rig_path) if rig_path else None
    prep_root = out_root / ".sensor_rig" / "vio" / session
    log(f"process-session {session}: {len(cals)} calibration(s), {len(eps) - len(cals)} demo(s) under {raw}")

    # ---- pose stage (demos only)
    for ep in eps:
        r = reports[ep.name]
        if ep.is_calibration:
            r.vio = {"skipped": "calibration episodes need no trajectory"}
            continue
        cal = ep.meta.get("gripper_calibration_run_name") or ""
        r.calibration_run = cal
        if not cal:
            r.fail("check", "no gripper calibration run in meta.json")
            continue
        if cal not in cals:
            r.fail("check", f"calibration run {cal} is not a kept calibration episode of session {session} "
                            f"(missing, aborted, rejected or discarded)")
            continue
        rig = ep.rig(rig_override) if rig_override else ep.rig()
        try:
            if vio == "import-dir":
                traj, fmt, frame = vio_hooks.find_trajectory(ep, traj_dir)
                r.vio = {"hook": vio, "trajectory": str(traj), "format": fmt}
            else:
                prep = prep_root / ep.name
                if overwrite or not (prep / "pose_prep.json").is_file():
                    info = prepare(ep, prep_root, target="orbslam3", rig=rig)
                    r.vio["prep"] = {k: info[k] for k in ("n_images", "n_imu") if k in info}
                    r.vio["mask"] = info.get("mask")
                done = prep / f"f_{ep.name}.txt"
                if done.is_file() and not overwrite:
                    traj, fmt, frame = done, "orbslam3-euroc", "imu"
                    r.vio.update(hook=vio, trajectory=str(done), resumed=True)
                elif vio == "cmd":
                    traj, fmt, frame = vio_hooks.run_cmd(vio_cmd, prep, ep.name)
                    r.vio.update(hook=vio, trajectory=str(traj), format=fmt)
                else:
                    traj, fmt, frame, extra = vio_hooks.run_orbslam3_docker(
                        prep, ep.name, image=docker_image or vio_hooks.DEFAULT_IMAGE)
                    r.vio.update(hook=vio, trajectory=str(traj), format=fmt, mask_used=extra["mask_used"],
                                 image=extra["image"])
        except (vio_hooks.VioError, OSError, ValueError) as e:
            r.fail("vio", str(e))
            continue
        tj = ep.trajectory_dir() / "trajectory.json"
        if tj.is_file() and not overwrite and json.loads(tj.read_text()).get("source_sha256") == _sha(traj):
            info = json.loads(tj.read_text())
            r.pose_import = {"resumed": True}
        else:
            try:
                info = import_trajectory(ep, traj, fmt=fmt, frame=frame, force=ep.name in force, rig=rig)
            except (TrajectoryImportError, ValueError, OSError) as e:
                r.fail("pose_import", str(e))
                continue
        m = info.get("motion_check") or {}
        r.pose_import.update({"gravity_angle_deg": (info.get("gravity_check") or {}).get("angle_to_up_deg"),
                              "traj_delay_ms": (info.get("time_offset_check") or {}).get("traj_delay_ms"),
                              "gyro_resid_rms": m.get("gyro_resid_rms"), "gyro_resid_max": m.get("gyro_resid_max"),
                              "step_speed_max": m.get("step_speed_max"),
                              "time_offset_applied_ns": info.get("time_offset_applied_ns"),
                              "overlap": info.get("main_overlap_fraction"), "forced": info.get("forced")})

    # ---- export (calibrations first, then demos)
    todo = [e for e in eps if reports[e.name].status != "failed"]
    groups = [[e for e in todo if e.is_calibration], [e for e in todo if not e.is_calibration]]
    for group in groups:
        jobs = [(e.path, out_root, ExportOptions(rig_override=rig_override, overwrite=overwrite,
                                                 force=e.name in force)) for e in group]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
                results = list(ex.map(_export_one, jobs))
        else:
            results = [_export_one(j) for j in jobs]
        for e, res in zip(group, results):
            r = reports[e.name]
            r.export = {k: res.get(k) for k in ("status", "reason", "frame_count", "uw_count", "grid_mode", "lag",
                                                 "coverage", "trim_head_s", "trim_tail_s", "warnings",
                                                 "world_yaw_extra_deg")}
            if res["status"] in ("exported", "skipped"):
                r.status = "exported"
            else:
                r.fail("export", f"{res.get('reason')}: {'; '.join(res.get('warnings') or [])}"[:500])
    for e in eps:                       # a demo is only usable if its calibration exported too
        r = reports[e.name]
        if not e.is_calibration and r.status == "exported" and reports[r.calibration_run].status != "exported":
            r.fail("export", f"its calibration {r.calibration_run} did not export")

    # ---- validate the export root
    yaml_path = GRIPPER_CAL_YAML if GRIPPER_CAL_YAML.is_file() else None
    val = validate_export_root(out_root, yaml_path)
    for name, errs in val.items():
        if name in reports and errs and reports[name].status == "exported":
            reports[name].fail("validate", "; ".join(errs[:5]))

    cal_ok = any(reports[c].status == "exported" for c in cals)
    demos = [r for r in reports.values() if r.kind == "demonstration"]
    n_ok = sum(r.status == "exported" for r in demos)
    n_fail = len(demos) - n_ok
    rc = 0 if (cal_ok and n_ok >= 1 and (n_fail == 0 or allow_partial)) else 1
    cmd, note = pipeline_line(session, out_root, task_label)
    doc = {"format": "sensor_rig.process_session.v1", "session": session, "raw": str(raw.resolve()),
           "export_root": str(out_root.resolve()), "vio": vio, "rig_override": str(rig_override) if rig_override
           else None, "time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "exit_code": rc,
           "counts": {"calibrations_exported": sum(reports[c].status == "exported" for c in cals),
                      "demos_exported": n_ok, "demos_failed": n_fail},
           "pipeline_command": cmd, "pipeline_note": note,
           "episodes": [reports[e.name].__dict__ for e in eps]}
    rep_path = out_root / ".sensor_rig" / f"process_{session}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = rep_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, rep_path)

    for e in eps:
        r = reports[e.name]
        extra = ""
        if r.export.get("lag"):
            extra = f"  lag {r.export['lag']['lag_ms_mean']:+.2f} ms (p99 {r.export['lag']['lag_ms_p99_abs']:.1f})"
        log(f"{'OK  ' if r.status == 'exported' else 'FAIL'} {e.name}" + extra
            + ("" if r.status == "exported" else f"  [{r.step}] {r.reason}"))
    log(f"calibration exported: {cal_ok}; demos exported {n_ok}, failed {n_fail}; report: {rep_path}")
    if rc == 0:
        if note:
            log(f"next: {note}")
        log(f"then (not run here): {cmd}")
    return rc, doc


def register(subparsers) -> None:
    p = subparsers.add_parser("process-session", help="raw rig session -> pose -> export -> validate (one command)",
                              description=__doc__.splitlines()[0])
    p.add_argument("--raw", required=True, help="raw root written by collect")
    p.add_argument("--session", required=True)
    p.add_argument("--out", required=True, help="export root (iPhone-format episodes)")
    p.add_argument("--vio", required=True, choices=["import-dir", "cmd", "orbslam3-docker"])
    p.add_argument("--traj-dir", default=None, help="import-dir: folder with the trajectories")
    p.add_argument("--vio-cmd", default=None, help="cmd: shell template ({prep} {settings} {times} {name} {mask} {out})")
    p.add_argument("--docker-image", default=None, help="orbslam3-docker: image (default sensor-rig/orbslam3:...)")
    p.add_argument("--rig", default=None, help="calibrated rig.yaml to use instead of each episode's snapshot")
    p.add_argument("--task-label", default="block_raise", help="task name for the printed pipeline command")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="redo every step (default: resume)")
    p.add_argument("--force-episode", action="append", default=[], metavar="NAME",
                   help="import/export this episode even if its checks fail (repeatable)")
    p.add_argument("--allow-partial", action="store_true", help="exit 0 even if some demos failed")
    p.add_argument("--iphone-raw", default=str(DEFAULT_IPHONE_RAW), help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_process_session)


def cmd_process_session(args) -> int:
    if args.rig:
        from sensor_rig.rig_config import load_rig_arg

        load_rig_arg(args.rig)
    try:
        rc, _ = process_session(args.raw, args.session, args.out, args.vio, traj_dir=args.traj_dir,
                                vio_cmd=args.vio_cmd, docker_image=args.docker_image, rig_path=args.rig,
                                task_label=args.task_label, workers=args.workers, overwrite=args.overwrite,
                                force_episodes=args.force_episode, allow_partial=args.allow_partial,
                                iphone_raw=Path(args.iphone_raw))
    except UsageError as e:
        print(f"process-session: {e}", file=sys.stderr)
        return 2
    return rc
