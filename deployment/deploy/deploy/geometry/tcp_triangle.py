"""TcpTriangle: gathers gripper.yaml, franka_control's tcp.F_T_TCP and the rig file's path; compute() for the tool,
check() for deploy check (STRUCTURE.md section 6.6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import deploy
from deploy.check_result import CheckResult
from deploy.config_merge import load_process_config, resolve_path
from deploy.errors import GeometryError, StationError
from deploy.geometry.frames import F_T_H, as_rigid, camera_pose, invert, pose_error
from deploy.geometry.gripper_file import GripperFile
from deploy.geometry.rig_tcp import RigTcp
from deploy.station import Station

PROCESS = "geometry"
CHECK_NAME = "tcp triangle"
F_T_TCP_LABEL = "franka_control/config.yaml (tcp.F_T_TCP)"


class TcpTriangle:
    def __init__(self, station: Station, gripper_file: Path) -> None:
        """Raises GeometryError (gripper.yaml, F_T_TCP) or StationError (a config.yaml or its overrides)."""
        self.station = station
        self.gripper = GripperFile.load(gripper_file)
        root = deploy.DEPLOYMENT_ROOT
        franka = load_process_config(root / "franka_control", station.section("franka_control"))
        try:
            raw = franka["tcp"]["F_T_TCP"]
        except (KeyError, TypeError):
            raise GeometryError(f"{F_T_TCP_LABEL}: missing") from None
        self.F_T_TCP: np.ndarray = as_rigid(raw, F_T_TCP_LABEL)
        frames = load_process_config(root / "process_frames", station.section("process_frames"))
        self.rig_file: Path = resolve_path(station, "process_frames", frames, "rig_file")
        self.F_T_cam: np.ndarray = F_T_H @ camera_pose(self.gripper.position_m, self.gripper.up_axis,
                                                       self.gripper.tilt_deg)

    def compute(self) -> np.ndarray:
        """T_cam_tcp = invert(F_T_cam) @ F_T_TCP."""
        return invert(self.F_T_cam) @ self.F_T_TCP

    def check(self, mock: bool) -> CheckResult:
        """Does F_T_cam @ T_cam_tcp(rig file) still equal F_T_TCP? A failure is a warn with --mock."""
        bad = "warn" if mock else "fail"
        rig = _show(self.rig_file)
        try:
            T_cam_tcp = RigTcp.read(self.rig_file)
        except GeometryError as exc:
            return CheckResult(PROCESS, CHECK_NAME, bad, f"{exc}; run python -m deploy.geometry")
        mm, deg = pose_error(self.F_T_cam @ T_cam_tcp, self.F_T_TCP)
        files = f"{self.gripper.path.name}, {F_T_TCP_LABEL} and {rig}"
        if mm > self.station.tcp_check_max_mm or deg > self.station.tcp_check_max_deg:
            return CheckResult(PROCESS, CHECK_NAME, bad,
                               f"TCP triangle off by {mm:.1f} mm / {deg:.1f}°: {files} disagree; "
                               f"run python -m deploy.geometry")
        return CheckResult(PROCESS, CHECK_NAME, "pass", f"closes within {mm:.3f} mm / {deg:.3f}°: {files}")


def triangle_check(station: Station, gripper_file: Path, mock: bool) -> CheckResult:
    """The check deploy check runs: TcpTriangle(...).check(mock), with a file that cannot be read reported as the
    same fail (warn with --mock), naming it."""
    try:
        return TcpTriangle(station, gripper_file).check(mock)
    except (GeometryError, StationError) as exc:
        return CheckResult(PROCESS, CHECK_NAME, "warn" if mock else "fail",
                           "; ".join(exc.problems) + "; fix it, then run python -m deploy.geometry")


def _show(path: Path) -> str:
    """The path relative to deployment/ when it is inside it, e.g. process_frames/rigs/rig_pi_v2.yaml."""
    try:
        return str(Path(path).relative_to(deploy.DEPLOYMENT_ROOT))
    except ValueError:
        return str(path)
