"""SafeCartesianJointController: BaseController (CARTESIAN_JOINT) with a run loop that cannot die silently.

Differences from upstream BaseController.run (which is not used):
- every exception (command handling, state read, target computation, robot command) is caught and published on the
  "error" topic (upstream publishes on "errors", which no client reads and robotmq drops: N1). The client's next
  schedule_* raises, so the task holds;
- published state timestamps are strictly increasing (bumped by 1 us if the source repeats a time);
- a HOLD latch on topic "hold": {"hold": True} freezes the arm target at the current target pose if it is within
  `hold_target_max_dev_m` (2 cm) of the measured pose, else at the measured pose, and keeps the LAST COMMANDED gripper
  width (re-seeding at the measured width would release a current-limited grasp); scheduled trajectories are
  dropped until {"hold": False}; the release re-seeds with the same rule so nothing jumps. {"reseed": True}
  re-seeds the arm interpolator at the measured pose without holding (task anchor fallback, no wind-up). Hold
  latency (request -> applied) is published on "status";
- `fault()` hook: a persistent fault (e.g. a robot reflex) is published on "error" once per second while active;
- identical errors are published at most once per second (no 200 Hz flood);
- PoseTrajInterpolator/JointTrajInterpolator (upstream, unchanged) with counters: chunks scheduled, points trimmed
  (older than now + trim), whole chunks out of date, chunks rejected by the speed check, chunks dropped while held;
- `stop()` / SIGTERM end the loop cleanly and call `on_shutdown()`;
- episode trace (D5): {"trace_start": path} on "hold" records one row per control tick (t, FK, commanded target,
  measured/commanded width, held, fault, annotations e.g. the task's chunk time-scale, plus `trace_extra()` of the
  subclass) until {"trace_stop": True}. Rows are flushed every second, append-only, to `<path stem>_parts/part_NNNNN.npz`
  (survives a killed controller); trace_stop merges the parts into `path` and removes them. {"annotate": {...}} sets annotation columns;
  {"clear_fault": True} calls `clear_fault()` (operator recovery of a latched fault).

Subclasses implement: `read_robot_state() -> (xyz_wxyz (7,), width (1,), t_monotonic)` and
`command_robot(target_xyz_wxyz, target_width)`; optionally `connect`, `disconnect`, `home_target()`.
"""

from __future__ import annotations

import signal
import threading
import time
import traceback
from typing import Any

import numpy as np
from robotmq import deserialize, serialize

from real_env.common.data_classes import ControllerCommand, Trajectory
from real_env.common.interpolator import JointTrajInterpolator, PoseTrajInterpolator
from real_env.controllers.base_controller import BaseController


def merge_trace_parts(parts_dir: str, out_path: str) -> bool:
    """Concatenate part_*.npz (written every second) into out_path and remove the parts. Used at trace_stop and
    by the post-mortem salvage of a killed controller."""
    import glob
    import os
    import shutil

    parts = sorted(glob.glob(os.path.join(parts_dir, "part_*.npz")))
    if not parts:
        return False
    loaded = [dict(np.load(p, allow_pickle=False)) for p in parts]
    keys = sorted({k for d in loaded for k in d})
    out = {}
    for k in keys:
        arrs = [d[k] for d in loaded if k in d]
        out[k] = np.concatenate(arrs) if len(arrs) == len(loaded) else np.concatenate(arrs)
    n = len(out.get("t", []))
    out = {k: v for k, v in out.items() if len(v) == n}
    np.savez_compressed(out_path, **out)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return True


class TransientControllerError(RuntimeError):
    """A recoverable hiccup (e.g. one late bridge reply): the tick publishes nothing, and an error is published only
    after `max_transient` consecutive occurrences."""


class SafeCartesianJointController(BaseController):
    def __init__(self, robot_name: str, server_endpoint: str, ctrl_freq: float = 200.0, config_str: str = "",
                 lookahead_time_s: float = 0.1, trim_new_trajectory_time_s: float = 0.09,
                 traj_smooth_time_s: float = 0.0, max_pos_speed_m_per_s: float = 1.0,
                 max_rot_speed_rad_per_s: float = 1.5, max_gripper_speed_m_per_s: float = 0.3,
                 status_freq_hz: float = 20.0, hold_target_max_dev_m: float = 0.02, max_transient: int = 3):
        super().__init__(robot_name=robot_name, server_endpoint=server_endpoint, control_mode="CARTESIAN_JOINT",
                         ctrl_freq=ctrl_freq, config_str=config_str)
        self.rmq_server.add_topic(topic="hold", message_remaining_time_s=10.0)
        self.rmq_server.add_topic(topic="status", message_remaining_time_s=2.0)
        self.lookahead_time_s = lookahead_time_s
        self.trim_s = trim_new_trajectory_time_s
        self.smooth_s = traj_smooth_time_s
        self.limits = (max_pos_speed_m_per_s, max_rot_speed_rad_per_s, max_gripper_speed_m_per_s)
        self.status_period = 1.0 / status_freq_hz
        self.hold_target_max_dev_m = float(hold_target_max_dev_m)
        self._last_err = (None, 0.0)
        self.max_transient = int(max_transient)
        self._transient = 0
        self._last_fault_pub = 0.0
        self._stop = threading.Event()
        self.held = False
        self.hold_reason = ""
        self.hold_request_t = None
        self.hold_applied_t = None
        self.counters = {k: 0 for k in ("eef_chunks", "joint_chunks", "eef_points_trimmed", "joint_points_trimmed",
                                        "eef_out_of_date", "joint_out_of_date", "eef_rejected", "joint_rejected",
                                        "dropped_while_held", "errors", "holds", "ticks", "overruns", "transient")}
        self._last_state_t = -np.inf
        self.pose_interp: PoseTrajInterpolator | None = None
        self.joint_interp: JointTrajInterpolator | None = None
        self.target_pose = None
        self.target_width = None
        self.target_t = None
        self.last_pose = None
        self.last_width = None
        self._trace = None
        self._trace_path = None
        self._trace_part = 0
        self.homing_until = 0.0
        self._trace_flush_t = 0.0
        self.last_chunk_t = -1e9
        self.annotations: dict = {}

    # ------------------------------------------------------------------ subclass API
    def read_robot_state(self):
        raise NotImplementedError

    def command_robot(self, target_xyz_wxyz: np.ndarray, target_width: np.ndarray):
        raise NotImplementedError

    def home_target(self):
        return None

    def fault(self) -> str | None:
        return None

    def clear_fault(self) -> None:
        pass

    def trace_extra(self) -> dict:
        return {}

    def _trace_row(self):
        if self._trace is None or self.last_pose is None or self.target_pose is None:
            return
        row = {"t": time.monotonic(), "fk": self.last_pose, "commanded": self.target_pose,
               "width_measured": float(np.asarray(self.last_width).reshape(-1)[0]),
               "width_commanded": float(np.asarray(self.target_width).reshape(-1)[0]),
               "held": self.held, "fault": bool(self.fault())}
        for k, v in self.annotations.items():
            row[f"ann_{k}"] = v
        row.update(self.trace_extra())
        self._trace.append(row)

    @staticmethod
    def _rows_to_arrays(rows):
        keys = sorted({k for r in rows for k in r})
        out = {}
        for k in keys:
            vals = [r.get(k, np.nan) for r in rows]
            try:
                out[k] = np.asarray(vals, dtype=float)
            except (TypeError, ValueError):
                out[k] = np.asarray([str(v) for v in vals])
        return out

    def _parts_dir(self, path):
        return str(path)[:-4] + "_parts" if str(path).endswith(".npz") else str(path) + "_parts"

    def _flush_trace(self):
        if not self._trace or not self._trace_path:
            return
        import os

        d = self._parts_dir(self._trace_path)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f"part_{self._trace_part:05d}.npz"), **self._rows_to_arrays(self._trace))
        self._trace_part += 1
        self._trace = []

    def _write_trace(self):
        if self._trace_path is None:
            return
        self._flush_trace()
        path = self._trace_path
        self._trace, self._trace_path = None, None
        merge_trace_parts(self._parts_dir(path), path)

    def on_shutdown(self):
        pass

    # ------------------------------------------------------------------ helpers
    def _seed_interpolators(self, pose, width, t):
        self.pose_interp = PoseTrajInterpolator(Trajectory(np.asarray(pose, float)[None], np.array([t])),
                                                traj_smooth_time_s=self.smooth_s, max_pos_speed_m_per_s=self.limits[0],
                                                max_rot_speed_rad_per_s=self.limits[1],
                                                trim_new_trajectory_time_s=self.trim_s)
        self.joint_interp = JointTrajInterpolator(Trajectory(np.asarray(width, float).reshape(1, -1), np.array([t])),
                                                  traj_smooth_time_s=self.smooth_s,
                                                  max_joint_speed_per_s=self.limits[2],
                                                  trim_new_trajectory_time_s=self.trim_s)

    def _update(self, interp, kind, data, timestamps, now):
        n_trim = int(np.sum(timestamps <= now + self.trim_s))
        self.counters[f"{kind}_points_trimmed"] += n_trim
        if n_trim == len(timestamps):
            self.counters[f"{kind}_out_of_date"] += 1
            return
        interp.update(Trajectory(np.asarray(data, float), np.asarray(timestamps, float)), now)
        if not np.isclose(interp.timestamps[-1], timestamps[-1]):
            self.counters[f"{kind}_rejected"] += 1

    def _publish_error(self, where: str, e: BaseException):
        self.counters["errors"] += 1
        key, now = (where, type(e).__name__, str(e)[:80]), time.monotonic()
        if self._last_err[0] == key and now - self._last_err[1] < 1.0:
            return
        self._last_err = (key, now)
        msg = f"Error in controller {self.robot_name} ({where}): {e!r}\n{traceback.format_exc()}"
        print(msg, flush=True)
        try:
            self.rmq_server.put_data(topic="error", data=serialize({"error": msg}))
        except Exception:
            pass

    def schedule_eef_traj(self, eef_traj_xyz_wxyz, timestamps):
        self.counters["eef_chunks"] += 1
        self.last_chunk_t = time.monotonic()
        self._update(self.pose_interp, "eef", eef_traj_xyz_wxyz, timestamps, time.monotonic())

    def schedule_joint_traj(self, joint_traj, timestamps):
        self.counters["joint_chunks"] += 1
        self._update(self.joint_interp, "joint", joint_traj, timestamps, time.monotonic())

    def get_eef_pose_and_joint_pos(self, wait_until_update: bool = False):
        pose, width, t = self.read_robot_state()
        if t <= self._last_state_t:
            t = self._last_state_t + 1e-6
        self._last_state_t = t
        self.last_pose, self.last_width = np.asarray(pose, float), np.asarray(width, float).reshape(-1)
        return self.last_pose, self.last_width, t

    def get_eef_target(self):
        return self.target_pose, self.target_t

    def get_joint_target(self):
        return self.target_width, self.target_t

    def send_robot_target(self):
        now = time.monotonic()
        if self.held:
            pose, width = self.pose_interp.interpolate_xyz_wxyz(now), self.joint_interp.interpolate(now)
        else:
            pose = self.pose_interp.interpolate_xyz_wxyz(now + self.lookahead_time_s)
            width = self.joint_interp.interpolate(now + self.lookahead_time_s)
        self.target_pose, self.target_width, self.target_t = np.asarray(pose, float), np.asarray(width, float), now
        self.command_robot(self.target_pose, self.target_width)

    def record_data(self):
        pass

    def reset(self):
        home = self.home_target()
        if home is None:
            return
        pose, width = home
        now = time.monotonic()
        dist = float(np.linalg.norm(np.asarray(pose[:3]) - np.asarray(self.last_pose[:3])))
        dur = max(2.0, dist / 0.1)  # <= 0.1 m/s homing
        self._seed_interpolators(self.last_pose, self.last_width, now)
        self.pose_interp.update(Trajectory(np.asarray(pose, float)[None], np.array([now + dur])), now)
        self.joint_interp.update(Trajectory(np.asarray(width, float).reshape(1, -1), np.array([now + dur])), now)
        self.homing_until = now + dur + self.lookahead_time_s + 0.3  # scheduled trajectories are dropped meanwhile

    def _hold_pose_width(self):
        """Arm: the current target if within hold_target_max_dev_m of the measured pose, else the measured pose.
        Gripper: the last commanded width (never the measured one)."""
        pose = self.last_pose
        if self.target_pose is not None and \
                np.linalg.norm(self.target_pose[:3] - self.last_pose[:3]) < self.hold_target_max_dev_m:
            pose = self.target_pose
        width = self.target_width if self.target_width is not None else self.last_width
        return pose, width

    def _handle_hold(self, msg: dict):
        now = time.monotonic()
        if "trace_start" in msg:
            self._trace, self._trace_path, self._trace_part = [], str(msg["trace_start"]), 0
            self._trace_flush_t = now + 1.0
            return
        if msg.get("trace_stop"):
            self._write_trace()
            return
        if "annotate" in msg:
            self.annotations.update(dict(msg["annotate"]))
            return
        if msg.get("clear_fault"):
            self.clear_fault()
            return
        if msg.get("reseed"):
            self.counters["reseeds"] = self.counters.get("reseeds", 0) + 1
            width = self.target_width if self.target_width is not None else self.last_width
            self._seed_interpolators(self.last_pose, width, now)
            return
        if msg.get("hold"):
            if not self.held:
                self.counters["holds"] += 1
            self.held = True
            self.hold_reason = str(msg.get("reason", ""))
            self.hold_request_t = float(msg.get("t", now))
            self._seed_interpolators(*self._hold_pose_width(), now)
            self.hold_applied_t = now
            print(f"[{self.robot_name}] HOLD ({self.hold_reason}) applied {1000 * (now - self.hold_request_t):.1f} ms "
                  f"after request", flush=True)
        else:
            if self.held:
                self._seed_interpolators(*self._hold_pose_width(), now)
                print(f"[{self.robot_name}] hold released", flush=True)
            self.held = False

    def _handle_command(self, data_dict: dict):
        command = data_dict.pop("command")
        if command == ControllerCommand.RESET:
            self.reset()
        elif command == ControllerCommand.CONNECT:
            self.connect()
        elif command == ControllerCommand.DISCONNECT:
            self.disconnect()
        elif command in (ControllerCommand.SCHEDULE_EEF_TRAJ, ControllerCommand.SCHEDULE_JOINT_TRAJ):
            if self.held:
                self.counters["dropped_while_held"] += 1
                return
            if time.monotonic() < self.homing_until:
                self.counters["dropped_while_homing"] = self.counters.get("dropped_while_homing", 0) + 1
                return
            rel = data_dict.pop("use_relative_timestamps")
            traj = Trajectory(**data_dict)
            if rel:
                traj.timestamps = traj.timestamps + time.monotonic()
            if command == ControllerCommand.SCHEDULE_EEF_TRAJ:
                self.schedule_eef_traj(traj.data, traj.timestamps)
            else:
                self.schedule_joint_traj(traj.data, traj.timestamps)
        else:
            raise ValueError(f"unsupported command {command}")

    def status(self) -> dict[str, Any]:
        return {"held": self.held, "homing": time.monotonic() < self.homing_until, "hold_reason": self.hold_reason, "hold_request_t": self.hold_request_t,
                "hold_applied_t": self.hold_applied_t, "counters": dict(self.counters), "t": time.monotonic()}

    def stop(self, *_):
        self._stop.set()

    # ------------------------------------------------------------------ main loop
    def run(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        if not self.is_connected:
            self.connect()
        pose, width, t = self.get_eef_pose_and_joint_pos()
        self._seed_interpolators(pose, width, time.monotonic())
        period = 1.0 / self.ctrl_freq
        next_t = time.monotonic()
        next_status = next_t
        try:
            while not self._stop.is_set():
                try:
                    for data in self.rmq_server.pop_data(topic="hold", n=0)[0]:
                        self._handle_hold(deserialize(data))
                    for data in self.rmq_server.pop_data(topic="command", n=0)[0]:
                        self._handle_command(deserialize(data))
                except Exception as e:
                    self._publish_error("command", e)
                state_ok = False
                try:
                    pose, width, t = self.get_eef_pose_and_joint_pos()
                    self.rmq_server.put_data(topic="joint_state", data=serialize({"joint_pos": width, "timestamp": t}))
                    self.rmq_server.put_data(topic="eef_pose", data=serialize({"eef_xyz_wxyz": pose, "timestamp": t}))
                    state_ok = True
                    self._transient = 0
                except TransientControllerError as e:
                    if not getattr(e, "latched", False):  # latched faults are reported by fault() at 1 Hz
                        self.counters["transient"] += 1
                        self._transient += 1
                        if self._transient >= self.max_transient:
                            self._publish_error("state", e)
                except Exception as e:
                    self._publish_error("state", e)
                next_t += period
                now = time.monotonic()
                if next_t > now:
                    time.sleep(next_t - now)
                else:
                    self.counters["overruns"] += 1
                    if now - next_t > 10 * period:
                        next_t = now
                try:
                    if not state_ok and self.last_pose is None:
                        raise TransientControllerError("no state yet")
                    self.send_robot_target()
                    self.rmq_server.put_data(topic="eef_target", data=serialize(
                        {"eef_xyz_wxyz": self.target_pose, "timestamp": self.target_t}))
                    self.rmq_server.put_data(topic="joint_target", data=serialize(
                        {"joint_target": self.target_width, "timestamp": self.target_t}))
                except TransientControllerError as e:
                    if not getattr(e, "latched", False):
                        self.counters["transient"] += 1
                except Exception as e:
                    self._publish_error("target", e)
                f = None
                try:
                    f = self.fault()
                except Exception as e:
                    self._publish_error("fault", e)
                if f and time.monotonic() - self._last_fault_pub >= 1.0:
                    self._last_fault_pub = time.monotonic()
                    self.counters["errors"] += 1
                    print(f"[{self.robot_name}] FAULT {f}", flush=True)
                    self.rmq_server.put_data(topic="error", data=serialize({"error": f"fault in {self.robot_name}: {f}"}))
                self.counters["ticks"] += 1
                try:
                    self._trace_row()
                    if self._trace is not None and time.monotonic() >= self._trace_flush_t:
                        self._trace_flush_t = time.monotonic() + 1.0
                        self._flush_trace()
                except Exception as e:
                    self._publish_error("trace", e)
                if time.monotonic() >= next_status:
                    next_status += self.status_period
                    self.rmq_server.put_data(topic="status", data=serialize(self.status()))
        finally:
            try:
                self._write_trace()
            finally:
                self.on_shutdown()
