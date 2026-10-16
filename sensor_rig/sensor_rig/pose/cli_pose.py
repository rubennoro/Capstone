"""CLI: ``pose-prep`` and ``pose-import``."""

from __future__ import annotations

import argparse
import json
import sys


def register(subparsers) -> None:
    p = subparsers.add_parser("pose-prep", help="raw episode(s) -> EuRoC folder + VIO config (ORB-SLAM3/OpenVINS)")
    p.add_argument("--raw", required=True, help="raw episode dir, date dir or raw root")
    p.add_argument("--out", required=True, help="output dir (one subfolder per episode)")
    p.add_argument("--target", choices=["orbslam3", "openvins", "euroc"], default="orbslam3")
    p.add_argument("--camera", default=None, help="camera for the VIO (default: rig.vio.camera)")
    p.add_argument("--scale", type=float, default=1.0, help="image downscale for the VIO (intrinsics scaled)")
    p.add_argument("--imu-rate", type=float, default=None, help="block-average the IMU down to this rate (Hz)")
    p.add_argument("--rig", default=None, help="use this rig.yaml instead of the episode snapshot")
    p.set_defaults(func=_cmd_prep)

    q = subparsers.add_parser("pose-import", help="import a VIO trajectory into a raw episode (with sanity checks)")
    q.add_argument("--raw", required=True, help="raw episode dir")
    q.add_argument("--traj", required=True, help="trajectory file")
    q.add_argument("--format", choices=["tum", "orbslam3-euroc"], default="tum")
    q.add_argument("--time-unit", choices=["auto", "s", "ns"], default="auto")
    q.add_argument("--frame", choices=["imu", "imu0", "ultrawide", "main"], default=None,
                   help="sensor frame of the poses (default: rig.vio.trajectory_frame)")
    q.add_argument("--rig", default=None, help="use this rig.yaml instead of the episode snapshot")
    q.add_argument("--no-gravity-align", action="store_true",
                   help="keep the trajectory's world as is (default: rotate it so IMU gravity is along -z)")
    q.add_argument("--gravity-align-min-deg", type=float, default=2.0,
                   help="only re-align when the estimated world tilt exceeds this (default 2 deg)")
    q.add_argument("--force", action="store_true", help="import even if the overlap/gravity checks fail")
    q.set_defaults(func=_cmd_import)


def _rig(args, ep):
    return ep.rig(args.rig) if args.rig else ep.rig()


def _cmd_prep(args: argparse.Namespace) -> int:
    from sensor_rig.pose.stage import prepare
    from sensor_rig.raw_episode import find_raw_episodes

    eps = find_raw_episodes(args.raw)
    if not eps:
        print(f"no raw episodes under {args.raw}", file=sys.stderr)
        return 1
    for ep in eps:
        info = prepare(ep, args.out, target=args.target, camera=args.camera, scale=args.scale,
                       imu_rate_hz=args.imu_rate, rig=_rig(args, ep))
        print(f"{ep.name}: {info['n_images']} images, {info['n_imu']} IMU samples -> {info['out_dir']}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from sensor_rig.pose.stage import TrajectoryImportError, import_trajectory
    from sensor_rig.raw_episode import RawEpisode

    ep = RawEpisode(args.raw)
    try:
        info = import_trajectory(ep, args.traj, fmt=args.format, time_unit=args.time_unit, frame=args.frame,
                                 force=args.force, rig=_rig(args, ep),
                                 gravity_align=not args.no_gravity_align,
                                 gravity_align_min_deg=args.gravity_align_min_deg)
    except TrajectoryImportError as e:
        print(f"{ep.name}: refused: {e}", file=sys.stderr)
        return 1
    keys = ("n_poses", "frame", "main_overlap_fraction", "gravity_alignment", "gravity_check", "time_offset_check")
    print(json.dumps({k: info[k] for k in keys}, indent=1))
    return 0
