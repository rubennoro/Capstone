"""MockFrankaController: a laptop stand-in for the Franka arm + gripper (CARTESIAN_JOINT, SafeCartesianJointController).

Modes
- integrate: the "measured" TCP follows the commanded target through a first-order lag (tau, default 50 ms;
  position lerp, rotation slerp, width lerp). `offset_m` adds a constant base-frame position error to the reported FK
  (emulates Cartesian-impedance steady-state error; used by the drift test).
- replay: the reported FK and width are the demo's TCP trajectory, time-aligned with the camera replay through a
  shared clock origin (`start_now_ns` = the live CLOCK_MONOTONIC time that the episode origin maps to, the same value
  passed to the RigCamera mock as backend_opts.start_now_ns; `span_ns` = the camera's loop period). Commands are only
  logged ("teacher forcing").

    python -m franka_deploy.controllers.mock_franka --endpoint tcp://localhost:26010 [--mode integrate|replay]
        [--tau 0.05] [--offset 0.005 0 0] [--replay-npz demo_tcp.npz --start-now-ns N --span-ns S] [--log out.npz]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from franka_deploy.controllers.safe_controller import SafeCartesianJointController

DEFAULT_HOME = [0.45, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0]  # xyz + wxyz: TCP z pointing down (rotated 180 deg about x)


def _wxyz_to_rot(q):
    q = np.asarray(q, float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _rot_to_wxyz(r):
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


class MockFrankaController(SafeCartesianJointController):
    def __init__(self, mode: str = "integrate", tau_s: float = 0.05, offset_m=(0.0, 0.0, 0.0),
                 home_xyz_wxyz=DEFAULT_HOME, home_width_m: float = 0.08, replay_npz: str | None = None,
                 start_now_ns: int | None = None, span_ns: int | None = None, log_path: str | None = None,
                 log_hz: float = 50.0, **kwargs):
        super().__init__(**kwargs)
        assert mode in ("integrate", "replay"), mode
        self.mode = mode
        self.tau = float(tau_s)
        self.offset = np.asarray(offset_m, float)
        self.home = (np.asarray(home_xyz_wxyz, float), np.array([home_width_m], float))
        self.x_pos = self.home[0][:3].copy()
        self.x_rot = _wxyz_to_rot(self.home[0][3:])
        self.x_width = self.home[1].copy()
        self._t_last = time.monotonic()
        if mode == "replay":
            z = np.load(replay_npz)
            self.r_t = z["t_ns"].astype(np.int64)        # episode clock (same origin as the camera replay)
            self.r_pose = z["xyz_wxyz"].astype(float)
            self.r_width = z["width"].astype(float).reshape(-1, 1)
            self.r_origin = int(z["origin_ns"])
            self.start_now_ns = int(start_now_ns)
            self.span_ns = int(span_ns) if span_ns else None
            self._r_rot = Rotation.from_quat(self.r_pose[:, [4, 5, 6, 3]])
        self.log_path = log_path
        self.log_period = 1.0 / log_hz
        self._next_log = 0.0
        self._log: dict[str, list] = {k: [] for k in ("t", "fk", "target", "width", "target_width", "held")}
        self.is_connected = True

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def home_target(self):
        return self.home

    def _replay_at(self, t_mono: float):
        tau = int(t_mono * 1e9) - self.start_now_ns
        if self.span_ns:
            tau %= self.span_ns
        tau += self.r_origin
        tau = int(np.clip(tau, self.r_t[0], self.r_t[-1]))
        i = int(np.clip(np.searchsorted(self.r_t, tau), 1, len(self.r_t) - 1))
        a = (tau - self.r_t[i - 1]) / max(1, self.r_t[i] - self.r_t[i - 1])
        pos = (1 - a) * self.r_pose[i - 1, :3] + a * self.r_pose[i, :3]
        rot = Slerp([0, 1], Rotation.concatenate([self._r_rot[i - 1], self._r_rot[i]]))(a)
        w = (1 - a) * self.r_width[i - 1] + a * self.r_width[i]
        return np.concatenate([pos, _rot_to_wxyz(rot)]), w

    def read_robot_state(self):
        now = time.monotonic()
        if self.mode == "replay":
            pose, width = self._replay_at(now)
            return pose, width, now
        dt = now - self._t_last
        self._t_last = now
        if self.target_pose is not None:
            alpha = 1.0 - np.exp(-dt / self.tau) if self.tau > 0 else 1.0
            self.x_pos = self.x_pos + alpha * (self.target_pose[:3] - self.x_pos)
            tr = _wxyz_to_rot(self.target_pose[3:])
            self.x_rot = Slerp([0, 1], Rotation.concatenate([self.x_rot, tr]))(alpha)
            self.x_width = self.x_width + alpha * (np.asarray(self.target_width, float).reshape(-1) - self.x_width)
        pose = np.concatenate([self.x_pos + self.offset, _rot_to_wxyz(self.x_rot)])
        return pose, self.x_width.copy(), now

    def command_robot(self, target_xyz_wxyz, target_width):
        now = time.monotonic()
        if self.log_path and now >= self._next_log:
            self._next_log = now + self.log_period
            self._log["t"].append(now)
            self._log["fk"].append(self.last_pose.copy())
            self._log["target"].append(np.asarray(target_xyz_wxyz, float).copy())
            self._log["width"].append(self.last_width.copy())
            self._log["target_width"].append(np.asarray(target_width, float).reshape(-1).copy())
            self._log["held"].append(self.held)

    def on_shutdown(self):
        print(f"[{self.robot_name}] shutdown counters {json.dumps(self.counters)}", flush=True)
        if self.log_path:
            np.savez_compressed(self.log_path, **{k: np.asarray(v) for k, v in self._log.items()},
                                counters=json.dumps(self.counters))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="franka")
    ap.add_argument("--endpoint", default="tcp://localhost:26010")
    ap.add_argument("--mode", default="integrate", choices=["integrate", "replay"])
    ap.add_argument("--ctrl-freq", type=float, default=200.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--lookahead", type=float, default=0.1)
    ap.add_argument("--trim", type=float, default=0.09)
    ap.add_argument("--max-pos-speed", type=float, default=1.0)
    ap.add_argument("--max-rot-speed", type=float, default=1.5)
    ap.add_argument("--max-gripper-speed", type=float, default=0.3)
    ap.add_argument("--replay-npz", default=None)
    ap.add_argument("--start-now-ns", type=int, default=None)
    ap.add_argument("--span-ns", type=int, default=None)
    ap.add_argument("--home", type=float, nargs=7, default=DEFAULT_HOME)
    ap.add_argument("--home-width", type=float, default=0.08)
    ap.add_argument("--log", default=None)
    a = ap.parse_args(argv)
    cfg = vars(a).copy()
    c = MockFrankaController(
        robot_name=a.name, server_endpoint=a.endpoint, ctrl_freq=a.ctrl_freq, config_str=json.dumps(cfg),
        lookahead_time_s=a.lookahead, trim_new_trajectory_time_s=a.trim, max_pos_speed_m_per_s=a.max_pos_speed,
        max_rot_speed_rad_per_s=a.max_rot_speed, max_gripper_speed_m_per_s=a.max_gripper_speed,
        mode=a.mode, tau_s=a.tau, offset_m=a.offset, home_xyz_wxyz=a.home, home_width_m=a.home_width,
        replay_npz=a.replay_npz, start_now_ns=a.start_now_ns, span_ns=a.span_ns, log_path=a.log)
    c.run()


if __name__ == "__main__":
    main()
