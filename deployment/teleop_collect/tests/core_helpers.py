"""Helpers for E4's tests (tests/motion, tests/episode): settings, poses, C2 states, C3 replies, a hello and a frames
summary, all valid contract objects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from deploy_contracts import SHARED_SCHEMAS, ArmState, GateCounters, HelloReply, TcpFrame

from teleop_collect.contracts import CameraSummary, RecordingSummary, paths
from teleop_collect.teleop.motion.home import HomeSettings
from teleop_collect.teleop.motion.planner import MotionSettings

T0_PC = 123_456_789_000_000        # a PC monotonic time, ns
MS = 1_000_000
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def teleop_config() -> dict:
    return yaml.safe_load(Path(paths.config_file("teleop")).read_text(encoding="utf-8"))


def motion_config(**changes) -> dict:
    cfg = dict(teleop_config()["motion"])
    cfg.update(changes)
    return cfg


def motion_settings(**changes) -> MotionSettings:
    return MotionSettings.from_config(motion_config(**changes))


def home_settings(**changes) -> HomeSettings:
    cfg = dict(teleop_config()["home"])
    cfg.update(changes)
    return HomeSettings.from_config(cfg)


def home_pose(x: float = 0.45, y: float = 0.0, z: float = 0.30) -> np.ndarray:
    """TCP pointing down (z axis = (0, 0, -1)): rotation by pi about base x, exact."""
    return np.array([[1.0, 0.0, 0.0, x],
                     [0.0, -1.0, 0.0, y],
                     [0.0, 0.0, -1.0, z],
                     [0.0, 0.0, 0.0, 1.0]])


def tilted_pose() -> np.ndarray:
    """A pose away from home: tilted and twisted (built from a rotation vector, then exact rigid)."""
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.2, -0.3, 0.9]).as_matrix() @ home_pose()[:3, :3]
    T[:3, 3] = [0.5, -0.1, 0.25]
    return T


def arm_state(seq: int = 1, *, t_ns: int = T0_PC, mode: str = "holding", T: np.ndarray | None = None,
              q: np.ndarray | None = None, width: float = 0.08, pulled_back: int = 0, shortened: int = 0,
              loop_late: int = 0, error: str = "") -> ArmState:
    T = home_pose() if T is None else T
    return ArmState(t_ns=t_ns, seq=seq, O_T_TCP=T.copy(), O_T_TCP_target=T.copy(),
                    q=(Q_HOME if q is None else q).astype(np.float64).copy(), dq=np.zeros(7), gripper_width_m=width,
                    mode=mode, error=error,
                    gate=GateCounters(accepted=100 + pulled_back + shortened, shortened=shortened,
                                      pulled_back=pulled_back, rejected=0),
                    loop_late=loop_late)


def hello_reply() -> HelloReply:
    return HelloReply(t_pc_ns=T0_PC, commit="unknown", mock=True, schemas=SHARED_SCHEMAS,
                      F_T_TCP=TcpFrame.load().F_T_TCP.copy(), rotation_tol=1.0e-6, gripper_width_max_m=0.08,
                      max_waypoints=64, watchdog_limit_s=1.0, homing_timeout_s=30.0, max_lead_s=1.5)


def recording_summary(main: int = 300, ultrawide: int = 290) -> RecordingSummary:
    def cam(n: int) -> CameraSummary:
        return CameraSummary(frames=n, frame_no_gaps=1, bytes=n * 48_000,
                             first_t_mid_ns=1_000 if n else None, last_t_mid_ns=10_000_000_000 if n else None)
    return RecordingSummary(main=cam(main), ultrawide=cam(ultrawide), packets=main, packet_no_gaps=0,
                            uw_reused=10, pair_refused=0, error="", pi_clock_offset_ns=1_000_000_000_000,
                            pi_clock_drift_ppm=12.5)


def z_tilt_deg(R0: np.ndarray, R1: np.ndarray) -> float:
    """Angle between the z axes of two rotations, degrees."""
    c = float(np.clip(np.dot(R0[:, 2], R1[:, 2]), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def rigid_error(T: np.ndarray) -> float:
    R = T[:3, :3]
    return float(np.max(np.abs(R.T @ R - np.eye(3))))
