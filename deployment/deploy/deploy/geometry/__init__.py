"""The gripper geometry tool, `python -m deploy.geometry` (STRUCTURE.md section 6.6): computes the rig file's
T_cam_tcp from gripper.yaml and franka_control's tcp.F_T_TCP, and checks the three still agree."""
from __future__ import annotations

from deploy.geometry.frames import F_T_H, as_rigid, camera_pose, invert, pose_error
from deploy.geometry.gripper_file import GripperFile
from deploy.geometry.rig_tcp import RigTcp
from deploy.geometry.tcp_triangle import TcpTriangle, triangle_check

__all__ = ["F_T_H", "GripperFile", "RigTcp", "TcpTriangle", "as_rigid", "camera_pose", "invert", "pose_error",
           "triangle_check"]
