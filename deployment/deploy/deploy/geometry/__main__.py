"""python -m deploy.geometry [--station FILE] [--gripper FILE] [--dry-run]

Computes the rig file's T_cam_tcp from gripper.yaml and franka_control's tcp.F_T_TCP and writes it into the rig file
named by process_frames' rig_file (STRUCTURE.md section 6.6).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import deploy
from deploy.cli import shadowed_message
from deploy.errors import GeometryError, StationError
from deploy.geometry.rig_tcp import RigTcp
from deploy.geometry.tcp_triangle import TcpTriangle


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m deploy.geometry",
                                     description="Compute the rig file's tcp.T_cam_tcp from gripper.yaml and "
                                                 "franka_control's tcp.F_T_TCP.")
    parser.add_argument("--station", type=Path, default=None, help="default: deployment/station.yaml")
    parser.add_argument("--gripper", type=Path, default=None, help="default: deployment/gripper.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the result without writing the rig file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    message = shadowed_message("deploy.geometry")
    if message is not None:
        print(message, file=sys.stderr)
        return 1
    gripper = args.gripper if args.gripper is not None else deploy.DEPLOYMENT_ROOT / "gripper.yaml"
    try:
        # mock=True: the tool needs only the process sections, not the Pi endpoints or the policy Python.
        station = deploy.Station.load(args.station, mock=True)
        triangle = TcpTriangle(station, gripper)
    except (StationError, GeometryError) as exc:
        print("deploy.geometry: cannot compute T_cam_tcp:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    T = triangle.compute()
    t = T[:3, 3]
    with np.printoptions(precision=6, suppress=True):
        print(f"F_T_cam (the ultrawide camera on the flange, from {triangle.gripper.path}):\n{triangle.F_T_cam}")
        print(f"T_cam_tcp (camera -> fingertips, for the rig file):\n{T}")
    print(f"camera -> fingertips: {np.linalg.norm(t) * 1000:.1f} mm "
          f"({t[2] * 1000:.1f} mm in front, {t[1] * 1000:.1f} mm below, {t[0] * 1000:.1f} mm right)")
    if not (t[2] > 0 and t[1] > 0):
        print("deploy.geometry: refusing to write: the fingertips are not in front of and below the camera "
              "(T_cam_tcp z and y must be > 0); check ultrawide.up_side, tilt_deg and position_mm in "
              f"{triangle.gripper.path}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"dry run: {triangle.rig_file} not written")
        return 0
    try:
        RigTcp.write(triangle.rig_file, T, triangle.gripper.path)
    except (GeometryError, OSError) as exc:
        print(f"deploy.geometry: cannot write {triangle.rig_file}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote tcp.T_cam_tcp to {triangle.rig_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
