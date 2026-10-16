"""CLI: ``export`` and ``validate``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers) -> None:
    p = subparsers.add_parser("export", help="raw rig episode(s) + trajectory -> iPhone-format episode folders")
    p.add_argument("--raw", required=True, nargs="+", help="raw episode dir(s), date dir(s) or raw root(s)")
    p.add_argument("--out", required=True, help="export root (episodes go to <out>/<YYYY-MM-DD>/<NAME>/)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--uw-rate", default="10", help="ultrawide export rate in Hz, or 'all' (default 10, like the iPhone)")
    p.add_argument("--label-task", default=None, help="label each demo as one task with this name (labelType Predefined)")
    p.add_argument("--calibration-run", default=None, help="override gripperCalibrationRunName (full NAME incl. side)")
    p.add_argument("--gripper-id", default=None, help="override gripperID")
    p.add_argument("--rig", default=None, help="re-export with this rig.yaml instead of the episode snapshot")
    p.add_argument("--trajectory", default=None, help="trajectory file (default <episode>/trajectory/trajectory.tum)")
    p.add_argument("--trajectory-frame", default=None, choices=["imu", "imu0", "main", "ultrawide"],
                   help="sensor frame of --trajectory (default: trajectory.json, else rig.vio.trajectory_frame)")
    p.add_argument("--max-pose-gap", type=float, default=0.10, help="max gap between trajectory poses (s)")
    p.add_argument("--crf", type=int, default=18, help="x264 CRF for both videos")
    p.add_argument("--overwrite", action="store_true", help="replace existing exported episodes")
    p.add_argument("--force", action="store_true", help="export raw episodes whose status is not 'complete'")
    p.set_defaults(func=_cmd_export)

    v = subparsers.add_parser("validate", help="check iPhone-format episodes against the pipeline's invariants")
    v.add_argument("path", help="export root or one episode folder")
    v.add_argument("--gripper-calibration", default=None, help="gripper_calibration.yaml to check device ids against")
    v.add_argument("--decode", action="store_true", help="also decode every video and count frames")
    v.set_defaults(func=_cmd_validate)


def _cmd_export(args: argparse.Namespace) -> int:
    from sensor_rig.export.exporter import ExportOptions, export_many

    if args.trajectory and len(args.raw) != 1:
        print("--trajectory needs exactly one --raw episode", file=sys.stderr)
        return 2
    opts = ExportOptions(
        ultrawide_rate_hz=None if args.uw_rate == "all" else float(args.uw_rate),
        label_task=args.label_task, calibration_run=args.calibration_run, gripper_id=args.gripper_id,
        rig_override=Path(args.rig) if args.rig else None,
        trajectory=Path(args.trajectory) if args.trajectory else None, trajectory_frame=args.trajectory_frame,
        max_pose_gap_s=args.max_pose_gap, main_crf=args.crf, uw_crf=args.crf,
        overwrite=args.overwrite, force=args.force,
    )
    results = export_many([Path(r) for r in args.raw], Path(args.out), opts, workers=args.workers)
    if not results:
        print(f"no raw episodes found under {args.raw}", file=sys.stderr)
        return 1
    failed = 0
    for r in results:
        line = f"{r.status:9s} {r.name}"
        if r.status == "exported":
            line += f"  frames={r.frame_count} uw={r.uw_count} grid={r.grid_mode}"
            line += "".join(f" cov_{k}={v:.3f}" for k, v in r.coverage.items())
            if r.trim_head_s or r.trim_tail_s:
                line += f" trim={r.trim_head_s:.2f}/{r.trim_tail_s:.2f}s"
        else:
            line += f"  {r.reason}"
            failed += r.status == "aborted"
        print(line)
        for w in r.warnings:
            print(f"          warning: {w}")
    return 1 if failed else 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from sensor_rig.export.validate import validate_export_root

    res = validate_export_root(args.path, args.gripper_calibration, args.decode)
    if not res:
        print(f"no episodes found under {args.path}", file=sys.stderr)
        return 1
    bad = 0
    for name, errors in sorted(res.items()):
        print(f"{'OK  ' if not errors else 'FAIL'} {name}")
        for e in errors:
            print(f"     {e}")
        bad += bool(errors)
    print(f"{len(res) - bad}/{len(res)} episodes valid")
    return 1 if bad else 0
