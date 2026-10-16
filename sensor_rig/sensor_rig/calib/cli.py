"""Calibration commands: calib-import, calib-check, calib-align, calib-tcp, rig-validate, bench-remap.

Registered into ``python -m sensor_rig`` via :func:`register`; also runnable on its own:
``python -m sensor_rig.calib.cli <command> ...``.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

__all__ = ["register", "main"]


def _load(path):
    from sensor_rig.rig_config import load_rig
    return load_rig(path)


def _print_coverage(rig) -> None:
    for name, cov in rig.coverage_report().items():
        flag = "" if cov >= rig.virtual[name].min_coverage else f"  < min_coverage {rig.virtual[name].min_coverage}"
        print(f"  coverage {name:9s} = {cov:.4f}{flag}")


def cmd_rig_validate(args) -> int:
    from sensor_rig.rig_config import RigConfigError, validate
    try:
        rig = _load(args.rig)
    except RigConfigError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"{rig.source_path}: OK (rig_name={rig.rig_name}, fingerprint={rig.fingerprint()[:16]})")
    _print_coverage(rig)
    for w in validate(rig):
        print(f"  warning: {w}")
    return 0


def cmd_calib_import(args) -> int:
    from sensor_rig.calib.kalibr_import import import_kalibr, parse_cam_map
    rig = _load(args.rig)
    new, log = import_kalibr(rig, args.camchain, args.imu, parse_cam_map(args.map))
    for line in log:
        print(line)
    new.dump(args.out)
    print(f"wrote {args.out} (fingerprint {new.fingerprint()[:16]})")
    _print_coverage(_load(args.out))
    return 1 if (args.strict and any(line.startswith("WARNING") for line in log)) else 0


def cmd_calib_check(args) -> int:
    from sensor_rig.calib.check import run_calib_check
    rig = _load(args.rig)
    if args.live:
        source, kw = "live", {"backend": args.backend}
    elif args.episode:
        source, kw = "episode", {"episode": args.episode, "frame": args.frame}
    else:
        source, kw = "images", {"images": args.images}
    while True:
        rep = run_calib_check(rig, source, args.out, purpose=args.purpose, reference_iphone=args.reference_iphone,
                              **kw)
        print(f"wrote {args.out}: coverage {rep['coverage']}, tag_z_m {rep['tag_z_m']}, "
              f"tag_sep_m {rep['tag_sep_m']}, rms_vs_reference_px {rep['rms_vs_reference_px']}")
        if not (args.live and args.watch):
            return 0
        time.sleep(args.watch)


def cmd_calib_align(args) -> int:
    from sensor_rig.calib.align import (
        align_main_from_chain, align_ultrawide, estimate_iphone_main_uw_rotation, with_virtual_rotation,
    )
    from sensor_rig.transforms import rotation_angle_deg
    rig = _load(args.rig)
    res = align_ultrawide(rig, args.iphone_calib, args.rig_calib)
    print(res.summary())
    rig = with_virtual_rotation(rig, "ultrawide", res.R_phys_virt)
    R_iu_im = None
    if args.iphone_main_uw:
        R_iu_im, ratio = estimate_iphone_main_uw_rotation(args.iphone_main_uw)
        print(f"iPhone UW<-main rotation {rotation_angle_deg(R_iu_im):.3f} deg (inlier ratio {ratio:.2f})")
    R_main = align_main_from_chain(rig, R_iu_im)
    rig = with_virtual_rotation(rig, "main", R_main)
    np.set_printoptions(precision=8, suppress=True)
    print(f"virtual_cameras.ultrawide.rotation.matrix:\n{rig.virtual['ultrawide'].R_phys_virt}")
    print(f"virtual_cameras.main.rotation.matrix:\n{rig.virtual['main'].R_phys_virt}")
    if args.write:
        rig.dump(args.write)
        print(f"wrote {args.write}")
        _print_coverage(_load(args.write))
    return 0


def cmd_calib_tcp(args) -> int:
    from sensor_rig.calib.align import tcp_pose_in_arkit_frame, tcp_yaml_snippet
    from sensor_rig.transforms import R_from_euler_xyz_deg, make_T
    rig = _load(args.rig)
    T = make_T(R_from_euler_xyz_deg(*args.tcp_rot_euler_xyz_deg), args.tcp_in_main_cv)
    print(tcp_yaml_snippet(tcp_pose_in_arkit_frame(rig, T), rig.export.device_identifier, args.artag_z), end="")
    return 0


def cmd_bench_remap(args) -> int:
    import cv2
    rig = _load(args.rig)
    default_threads = cv2.getNumThreads()
    rng = np.random.default_rng(0)
    try:
        for threads, label in ((1, "1 thread"), (default_threads, f"default ({default_threads} threads)")):
            cv2.setNumThreads(threads)
            for name, v in rig.virtual.items():
                table = rig.remap(name, args.purpose)
                w, h = rig.cameras[v.source].resolution
                src = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
                out = np.empty((table.dst_size[1], table.dst_size[0], 3), np.uint8)
                for _ in range(10):
                    table.apply(src, out=out)
                ts = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    table.apply(src, out=out)
                    ts.append((time.perf_counter() - t0) * 1e3)
                p50, p99 = np.percentile(ts, [50, 99])
                print(f"{label:24s} {name:9s} {w}x{h} -> {table.dst_size[0]}x{table.dst_size[1]}: "
                      f"p50 {p50:.2f} ms, p99 {p99:.2f} ms")
    finally:
        cv2.setNumThreads(default_threads)
    return 0


def register(subparsers) -> None:
    """Add the calibration commands to an argparse ``subparsers`` object (sets ``func`` on each)."""
    p = subparsers.add_parser("rig-validate", help="load + validate a rig.yaml, print coverage and warnings")
    p.add_argument("rig", help="rig yaml (a bare name is looked up in sensor_rig/configs/)")
    p.set_defaults(func=cmd_rig_validate)

    p = subparsers.add_parser("calib-import", help="import Kalibr camchain-imucam.yaml (+ imu.yaml) into a rig.yaml")
    p.add_argument("--rig", required=True, help="input rig yaml")
    p.add_argument("--camchain", required=True, help="Kalibr camchain-imucam.yaml (or camchain.yaml)")
    p.add_argument("--imu", help="Kalibr / allan_variance_ros imu.yaml")
    p.add_argument("--map", help="camera mapping, e.g. cam0=ultrawide,cam1=main (default: match rostopic)")
    p.add_argument("--out", required=True, help="output rig yaml")
    p.add_argument("--strict", action="store_true", help="exit 1 if the import produced warnings")
    p.set_defaults(func=cmd_calib_import)

    p = subparsers.add_parser("calib-check", help="render virtual views + coverage (PNG files + report.json)")
    p.add_argument("--rig", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--live", action="store_true", help="grab frames from the cameras")
    src.add_argument("--episode", help="raw rig episode directory")
    src.add_argument("--images", nargs=2, metavar=("MAIN", "ULTRAWIDE"), help="physical frames as image files")
    p.add_argument("--backend", choices=["argus", "mock"], help="camera backend for --live")
    p.add_argument("--frame", type=int, help="frame index for --episode (default: middle)")
    p.add_argument("--watch", type=float, default=0.0, help="--live: refresh every N seconds")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--reference-iphone", help="iPhUMI export folder of the same scene (blend/edge overlays)")
    p.add_argument("--purpose", default="export", help="virtual output: export | deploy")
    p.set_defaults(func=cmd_calib_check)

    p = subparsers.add_parser("calib-align", help="estimate virtual camera rotations from finger tags")
    p.add_argument("--rig", required=True)
    p.add_argument("--iphone-calib", required=True, help="iPhUMI gripper-calibration export folder")
    p.add_argument("--rig-calib", required=True, help="raw rig gripper-calibration episode directory")
    p.add_argument("--iphone-main-uw", help="iPhUMI demonstration export of a distant (>= 3 m) scene")
    p.add_argument("--write", help="write the rig with updated virtual rotations to this yaml")
    p.set_defaults(func=cmd_calib_align)

    p = subparsers.add_parser("calib-tcp", help="print the gripper_calibration.yaml entry for a CAD TCP pose")
    p.add_argument("--rig", required=True)
    p.add_argument("--tcp-in-main-cv", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                   help="TCP origin in the PHYSICAL main camera optical frame (m)")
    p.add_argument("--tcp-rot-euler-xyz-deg", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("A", "B", "C"), help="TCP axes in the physical main camera frame (extrinsic xyz)")
    p.add_argument("--artag-z", type=float, default=0.098079, help="artag_z_distance_from_ultrawide (m)")
    p.set_defaults(func=cmd_calib_tcp)

    p = subparsers.add_parser("bench-remap", help="time the remaps (p50/p99 per camera)")
    p.add_argument("--rig", required=True)
    p.add_argument("--purpose", default="deploy")
    p.add_argument("--iters", type=int, default=600)
    p.set_defaults(func=cmd_bench_remap)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sensor_rig.calib.cli", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    register(sub)
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
