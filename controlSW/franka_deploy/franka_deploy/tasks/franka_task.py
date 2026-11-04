"""FrankaPromptTask: single-arm iPhUMI behaviour-prompted policy on a CARTESIAN_JOINT arm (Franka or the mock).

Subclass of real-env iPhumiARX5Task (observations without crop/mask, upstream UmiARX5Task.run_policy_control
scheduling and action history). Additions, all in this file (no upstream edits):
- contract asserts: agent_update_freq_hz == 20, crop_image None, gripper mask off, robot_num 1, camera fps % 20 == 0;
- `self.task_name` set (upstream iPhumiARX5Task reads it but nobody sets it);
- prompting: at every episode start (c), lazily on C, and again after R or a stop; always policy_reset -> prompt
  (the server refuses a second prompt without a reset);
- anchor (decision a): target by default; the deviation judged is |target - FK| (mock) or the controller's
  track_dev_m = |bridge safe target - FK| (Franka; the speed-clamp backlog is benign lag); > anchor_max_dev_m (2 cm)
  -> measured anchor + controller interpolator re-seeded from FK; > anchor_hold_m (5 cm) -> hold + stop;
- safe controller and camera clients (no resend, bounded timeouts; race-free frame/timestamp snapshot) + a
  separate hold channel; stale-state guard
  (state_max_age_s); measured or target action anchor (anchor="target" falls back to measured when
  |target - FK| > anchor_max_dev_m);
- headless control loop (replaces BaseTask.run, no pynput): commands from stdin (if a tty) and/or a command file
  (one key per line, appended). Stops: a stop file or a s/y/n line in the command file is turned into a HOLD by a
  separate stop-watcher process (tasks/stop_watcher.py) within ~10 ms, independent of the inference cycle (robotmq
  keeps the GIL during blocking calls, so no thread of this process could do it); the main loop then ends the
  episode, releases the hold and returns to idle (spacemouse). A stop key typed on this process's stdin holds only
  after the current RPC returns (<= inference timeout);
- any exception in policy control -> HOLD at the current FK, stop the episode, idle, log; in idle, errors are logged
  (rate-limited) and retried, never hang;
- per-call tap (npz, small): obs_to_send low-dim + shapes/dtypes, optional 224 px images every N calls, reply,
  absolute actions, timing, camera timestamps (main + ultrawide), state age, anchor, controller status.
"""

from __future__ import annotations

import json
import os
import queue
import select
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np
import numpy.typing as npt

import real_env.tasks.base_task as base_task_mod
from real_env.tasks.base_task import EpisodeStatus, TaskControlMode
from real_env.tasks.iphumi_arx5_task import iPhumiARX5Task
from franka_deploy.tasks.chunk_timing import chunk_scale, scale_timestamps, settle_gate
from franka_deploy.tasks.frame_select import FrameSelectionError, get_downsampled_camera_images
from robot_utils.pose_utils import get_relative_poses
from robotmq import deserialize

from franka_deploy.controllers.safe_client import SafeCartesianJointClient
from franka_deploy.peripherals.safe_camera_client import SafeCameraClient
from franka_deploy.paths import check_writable_dir
from franka_deploy.rmq_rpc import NoResendClient

STOP_KEYS = ("s", "y", "n")


class StaleStateError(RuntimeError):
    pass


class AnchorDeviationError(RuntimeError):
    pass


class FrankaPromptTask(iPhumiARX5Task):
    max_image_age_s: float | None = None   # image-age guard (B6); class default for instances built without __init__
    def __init__(self, task_name: str, prompt_index: int = 0, anchor: str = "target",
                 anchor_max_dev_m: float = 0.02, anchor_hold_m: float = 0.05, state_max_age_s: float = 0.05,
                 controller_timeout_s: float = 0.05, command_file: str | None = None, stop_file: str | None = None,
                 read_stdin: bool = True, tap_dir: str | None = None, tap_images_every: int = 0,
                 record_episodes: bool = True, width_proprio: str = "commanded", history_source: str = "commanded",
                 replay_npz: str | None = None, replay_start_now_ns: int | None = None,
                 spacemouse_endpoint: str | None = None, v_max: float = 0.25, w_max: float = 1.0,
                 scale_margin: float = 0.9, settle: bool = True, settle_timeout_s: float = 1.0,
                 session_dir: str | None = None, bridge_log: str | None = None, camera_log: str | None = None,
                 home_xyz=None, home_max_dist_m: float = 0.05, max_image_age_s: float | None = None, **kwargs):
        base_task_mod.continue_running = False  # upstream's termios stdin thread exits at once (headless-safe)
        assert anchor in ("measured", "target"), anchor
        assert width_proprio in ("commanded", "measured"), width_proprio
        assert history_source in ("commanded", "replay_gt"), history_source
        self.anchor_hold_m = float(anchor_hold_m)
        self.v_max, self.w_max, self.scale_margin = float(v_max), float(w_max), float(scale_margin)
        self.settle, self.settle_timeout_s = bool(settle), float(settle_timeout_s)
        self.session_dir, self.bridge_log, self.camera_log = session_dir, bridge_log, camera_log
        self.episode_dir = None
        self._ep = None
        self._ev_file = None
        self.home_xyz = None if home_xyz is None else np.asarray(home_xyz, float)
        self.home_max_dist_m = float(home_max_dist_m)
        self.max_image_age_s = None if max_image_age_s is None else float(max_image_age_s)
        self.width_proprio = width_proprio
        self.history_source = history_source
        self._replay = None
        self._replay_start = None
        if history_source == "replay_gt":  # TEST ONLY: feed the demo's pose+width history (replay harness)
            assert replay_npz and replay_start_now_ns is not None, "replay_gt needs replay_npz + replay_start_now_ns"
            z = np.load(replay_npz)
            self._replay = {k: z[k] for k in ("t_ns", "xyz_wxyz", "width")}
            self._replay.update(origin=int(z["origin_ns"]), span=int(z["span_ns"]), start=int(replay_start_now_ns))
        kwargs["root_dir"] = str(check_writable_dir(kwargs["root_dir"], "task root_dir"))
        self.task_name = task_name
        self.prompt_index = int(prompt_index)
        self.anchor = anchor
        self.anchor_max_dev_m = anchor_max_dev_m
        self.state_max_age_s = state_max_age_s
        self.record_episodes = record_episodes
        super().__init__(task_name=task_name, **kwargs)
        self.main_logger.success_config = "none"  # never block on a stdin success prompt (headless)

        pa = self.policy_agent
        assert float(pa.agent_update_freq_hz) == 20.0, f"agent_update_freq_hz must be 20 (trained 50 ms), got {pa.agent_update_freq_hz}"
        assert self.crop_image is None, f"crop_image must be null (training pads, never crops), got {self.crop_image}"
        assert not self.draw_gripper_mask, "draw_gripper_mask must be false"
        assert pa.robot_num == 1, f"single arm only, robot_num={pa.robot_num}"
        for cam in ("main", "ultrawide"):
            fps = self.wrist_camera_client.config["camera_configs"][cam]["fps"]
            assert fps % 20 == 0, f"camera {cam} fps {fps} must be a multiple of 20"

        old_cam = self.wrist_camera_client
        self.wrist_camera_client = SafeCameraClient(name="wrist_camera", server_endpoint=kwargs["wrist_camera_endpoint"])
        self.clients[self.clients.index(old_cam)] = self.wrist_camera_client
        ep = kwargs["arx5_controller_endpoint"]
        old = self.arx5_client
        self.arx5_client = SafeCartesianJointClient(name="franka", server_endpoint=ep, timeout_s=controller_timeout_s)
        self.clients[self.clients.index(old)] = self.arx5_client
        self._hold_client = NoResendClient("franka_hold", ep, default_timeout_s=0.2)
        self._status_client = NoResendClient("franka_status", ep, default_timeout_s=0.2)
        self._hold_lock = threading.Lock()

        self._prompted = False
        self.prompt_count = 0
        self.episode_active = False
        self.counters = {k: 0 for k in ("policy_calls", "policy_errors", "idle_errors", "holds", "stale_state",
                                        "anchor_fallbacks", "episodes", "obs_retries", "stale_image")}
        self.obs_retries = 3
        self.ping_timeout_s = 0.2
        pa.policy_rmq_client.timeouts["policy_reset"] = min(pa.policy_rmq_client.timeouts.get("policy_reset", 5.0), 0.5)
        if spacemouse_endpoint:  # fix 3: no robotmq resend for the spacemouse either
            self.spacemouse_agent.spacemouse_client.rmq_client = NoResendClient("spacemouse_safe", spacemouse_endpoint,
                                                                                default_timeout_s=0.1)
        self.events: list[dict] = []
        self._keys: queue.Queue = queue.Queue()
        self._stop_requested: str | None = None
        self.command_file, self.stop_file, self.read_stdin = command_file, stop_file, read_stdin
        self.tap_dir = str(check_writable_dir(tap_dir, "tap dir")) if tap_dir else None
        if self.tap_dir:
            os.makedirs(self.tap_dir, exist_ok=True)
        self.tap_images_every = int(tap_images_every)
        self._last_request: dict | None = None
        pa.set_tap(self._on_request) if hasattr(pa, "set_tap") else None
        self._obs_info: dict = {}
        self._quit = False
        self.abort_reason = None
        self._watcher = None
        if command_file:
            import subprocess

            open(command_file, "a").close()
            cmd = [sys.executable, "-u", "-m", "franka_deploy.tasks.stop_watcher", "--controller-endpoint", ep,
                   "--command-file", command_file] + (["--stop-file", stop_file] if stop_file else [])
            self._watcher = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)

    # ------------------------------------------------------------------ events / hold
    def log_event(self, kind: str, **kw):
        ev = {"t": time.monotonic(), "kind": kind, **kw}
        self.events.append(ev)
        print(f"[franka_task] {kind} {json.dumps(kw, default=str)}", flush=True)
        f = getattr(self, "_ev_file", None)
        if f is not None:  # append-only, flushed: survives an abnormal end of the episode
            try:
                f.write(json.dumps(ev, default=str) + "\n")
                f.flush()
            except Exception:
                pass

    def hold(self, reason: str):
        self.counters["holds"] += 1
        try:
            with self._hold_lock:
                from robotmq import serialize

                self._hold_client.put_data(topic="hold", data=serialize({"hold": True, "reason": reason, "t": time.monotonic()}))
            self.log_event("hold", reason=reason)
        except Exception as e:  # controller unreachable: nothing more we can do from here; bridge watchdog stops the arm
            self.log_event("hold_failed", reason=reason, error=repr(e))

    def _reseed_controller(self):
        from robotmq import serialize

        with self._hold_lock:
            self._hold_client.put_data(topic="hold", data=serialize({"reseed": True, "t": time.monotonic()}))
        self.log_event("anchor_fallback_reseed")

    def release(self):
        from robotmq import serialize

        with self._hold_lock:
            self._hold_client.put_data(topic="hold", data=serialize({"hold": False, "t": time.monotonic()}))
        self.log_event("release")

    def controller_status(self) -> dict | None:
        try:
            raw, _ = self._status_client.peek_data(topic="status", n=-1)
            return deserialize(raw[0]) if raw else None
        except Exception:
            return None

    # ------------------------------------------------------------------ prompting / episodes
    def _ensure_prompted(self):
        if self._prompted:
            return
        self.policy_agent.reset()  # the server refuses a second prompt without a reset
        self.policy_agent.send_prompt_request(self.task_name, self.prompt_index)
        self._prompted = True
        self.prompt_count += 1
        self.log_event("prompted", task=self.task_name, index=self.prompt_index, count=self.prompt_count)

    def robot_ready(self) -> str | None:
        """None if a policy episode may start; else the reason (fault, or the bridge not running)."""
        st = self.controller_status()
        if st is None:
            return "no controller status"
        if st.get("fault"):
            return f"controller fault: {st['fault']}"
        if "bridge_state" in st and st["bridge_state"] not in ("running", "hold"):
            return f"bridge state {st['bridge_state']}"
        return None

    def home_check(self) -> str | None:
        """Every episode start: FK must be within home_max_dist_m of the home TCP position (press `r` to home)."""
        if self.home_xyz is None:
            return None
        fk = np.asarray(self.arx5_client.get_eef_xyz_wxyz(timestamp=time.monotonic()), float)
        d = float(np.linalg.norm(fk[:3] - self.home_xyz))
        return None if d <= self.home_max_dist_m else f"FK {1000 * d:.0f} mm from home (max {1000 * self.home_max_dist_m:.0f} mm): press r"

    def _ctrl_msg(self, msg: dict):
        from robotmq import serialize

        with self._hold_lock:
            self._hold_client.put_data(topic="hold", data=serialize(msg))

    def start_episode(self):
        self._prompted = False
        self.episode_dir = None
        if self.record_episodes:
            idx = super().start_episode()
            self.episode_dir = os.path.join(self.main_logger.run_dir, f"episode_{idx:06d}")
            os.makedirs(os.path.join(self.episode_dir, "tap"), exist_ok=True)
            self._ep = {"idx": idx, "t_start": time.monotonic(), "wall_start": time.time(), "event0": len(self.events),
                        "counters0": dict(self.counters), "ctrl0": (self.controller_status() or {}).get("counters", {}),
                        "scales": [], "backlogs": [], "label": None, "notes": []}
            self._ev_file = open(os.path.join(self.episode_dir, "events.jsonl"), "a")
            with open(os.path.join(self.episode_dir, "label.json"), "w") as f:  # until finalized
                json.dump({"label": "aborted:unfinalized", "is_successful": None, "notes": []}, f)
            try:
                self._ctrl_msg({"trace_start": os.path.join(self.episode_dir, "controller_trace.npz")})
            except Exception as e:
                self.log_event("trace_start_failed", error=repr(e))
            if self.session_dir and os.path.isdir(self.session_dir):
                import shutil

                shutil.copytree(self.session_dir, os.path.join(self.episode_dir, "config"), dirs_exist_ok=True)
        self._ensure_prompted()
        self.episode_active = True
        self.counters["episodes"] += 1
        self.log_event("episode_start", prompt_index=self.prompt_index, episode_dir=self.episode_dir)

    def stop_episode(self, is_successful: bool | None = None, label: str | None = None):
        was_active = self.episode_active
        if self.episode_active and self.record_episodes:
            try:
                super().stop_episode(is_successful=is_successful)
            except Exception as e:
                self.log_event("stop_episode_error", error=repr(e))
        self.episode_active = False
        self._prompted = False
        self.log_event("episode_stop", is_successful=is_successful, label=label)
        if was_active and self.episode_dir and self._ep is not None:
            try:
                self._finalize_episode(is_successful, label)
            except Exception as e:
                self.log_event("finalize_failed", error=repr(e), tb=traceback.format_exc(limit=3))

    def _finalize_episode(self, is_successful, label):
        ep, d = self._ep, self.episode_dir
        t_end = time.monotonic()
        try:
            self._ctrl_msg({"trace_stop": True})
        except Exception as e:
            self.log_event("trace_stop_failed", error=repr(e))
        if self._ev_file is not None:
            self._ev_file.close()
            self._ev_file = None
        if self.bridge_log and os.path.exists(self.bridge_log):
            with open(self.bridge_log) as fi, open(os.path.join(d, "bridge.jsonl"), "w") as fo:
                for ln in fi:
                    try:
                        t = json.loads(ln).get("t", 0)
                    except Exception:
                        continue
                    if ep["t_start"] - 1.0 <= t <= t_end + 1.0:
                        fo.write(ln)
        if self.camera_log and os.path.exists(self.camera_log):
            with open(self.camera_log, errors="replace") as f:
                lines = [ln for ln in f.read().replace("\r", "\n").splitlines() if "rig:" in ln]
            with open(os.path.join(d, "camera_stats.txt"), "w") as f:
                f.write("\n".join(lines[-3:]) + "\n")
        ctrl1 = (self.controller_status() or {}).get("counters", {})
        dctrl = {k: ctrl1.get(k, 0) - ep["ctrl0"].get(k, 0) for k in ctrl1 if isinstance(ctrl1.get(k), (int, float))}
        dtask = {k: self.counters.get(k, 0) - ep["counters0"].get(k, 0) for k in self.counters}
        dur = t_end - ep["t_start"]
        label = label or {True: "y", False: "n", None: "s"}[is_successful]
        contacts = sum(1 for e in self.events[ep["event0"]:] if e["kind"] == "workspace_contact")
        with open(os.path.join(d, "label.json"), "w") as f:
            json.dump({"label": label, "is_successful": is_successful, "notes": ep["notes"]}, f)
        sc = np.array(ep["scales"]) if ep["scales"] else np.ones(1)
        bl = np.array([b for b in ep["backlogs"] if b == b]) if ep["backlogs"] else np.zeros(0)
        summary = {"episode_idx": ep["idx"], "duration_s": dur, "wall_start": ep["wall_start"], "label": label,
                   "policy_calls": dtask.get("policy_calls", 0), "task_counters": dtask, "controller_counters": dctrl,
                   "chunk_scale": {"mean": float(sc.mean()), "max": float(sc.max()), "n_scaled": int((sc > 1.0).sum())},
                   "clamp_count": int(dctrl.get("clamp_pos", 0) + dctrl.get("clamp_rot", 0)),
                   "clamp_backlog_m": {"max": float(bl.max()) if bl.size else None,
                                       "median": float(np.median(bl)) if bl.size else None},
                   "prompt_index": self.prompt_index, "task_name": self.task_name,
                   "workspace_contacts": contacts, "flags": {"workspace_contact": contacts > 0}}
        size = sum(os.path.getsize(os.path.join(r, fn)) for r, _, fs in os.walk(d) for fn in fs)
        summary["size_mb"] = size / 1e6
        summary["log_rate_mb_per_min"] = size / 1e6 / max(dur / 60.0, 1e-6)
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)
        self.log_event("episode_finalized", episode_dir=d, size_mb=round(summary["size_mb"], 2),
                       log_rate_mb_per_min=round(summary["log_rate_mb_per_min"], 2))
        self._ep = None

    def reset_policy_agent(self):
        """Clear local policy state; reset the server only if a short health ping answers (<= 0.2 s), and never
        raise here (the next episode start always does reset -> prompt anyway). Keeps the idle loop responsive
        when the server is down (no idle call blocks > 0.5 s)."""
        self._prompted = False
        self.policy_history_actions = {}
        self.policy_start_eef_xyz_wxyz = None
        self._replay_start = None
        now = time.monotonic()
        if now < getattr(self, "_next_reset_try", 0.0):
            return
        try:
            self._ping_server()
            self.policy_agent.reset()
            self._reset_pending = False
        except Exception as e:
            self._reset_pending = True
            self._next_reset_try = now + 2.0
            self.log_event("policy_reset_skipped", error=repr(e))

    def _ping_server(self):
        from robotmq import serialize

        c = self.policy_agent.policy_rmq_client
        old = c.timeouts.get("policy_config")
        c.timeouts["policy_config"] = self.ping_timeout_s
        try:
            c.request_with_data(topic="policy_config", data=serialize(True))
        finally:
            c.timeouts["policy_config"] = old

    # ------------------------------------------------------------------ observations
    def get_observations(self):
        images, cam_ts = get_downsampled_camera_images(
            self.wrist_camera_client, self.policy_agent.image_history_len, self.policy_agent.agent_update_freq_hz,
            cam_obs_freq_hz=self.policy_agent.cam_obs_freq_hz)
        main_ts = cam_ts["main"]
        if self.max_image_age_s is not None:   # robotmq keeps packets ~1 s: a stalled camera still "has" frames
            img_age = time.monotonic() - float(main_ts[-1])
            if img_age > self.max_image_age_s:
                self.counters["stale_image"] = self.counters.get("stale_image", 0) + 1
                raise FrameSelectionError(f"image {1000 * img_age:.0f} ms old (max {1000 * self.max_image_age_s:.0f} ms)")
        f = self.policy_agent.agent_update_freq_hz
        last_ts = main_ts[-1] if self.match_camera_latency else time.monotonic()
        state_ts = [last_ts - i / f for i in range(self.policy_agent.proprio_history_len - 1, -1, -1)]
        poses = np.array([self.arx5_client.get_eef_xyz_wxyz(timestamp=t) for t in state_ts])
        widths = np.array([self.arx5_client.get_joint_pos(timestamp=t) for t in state_ts])
        age = self.arx5_client.state_age_s()
        if age > self.state_max_age_s:
            self.counters["stale_state"] += 1
            raise StaleStateError(f"robot state is {1000 * age:.0f} ms old (max {1000 * self.state_max_age_s:.0f} ms)")
        anchor_used = "measured"
        fk = poses.copy()
        targets = np.array([self.arx5_client.get_eef_target_xyz_wxyz(timestamp=t) for t in state_ts])
        dev = float(np.linalg.norm(targets[-1, :3] - poses[-1, :3]))
        st = self.controller_status()
        flags = (st or {}).get("safety_flags") or {}
        if self.episode_active and (flags.get("box") or flags.get("cone")):
            self.log_event("workspace_contact", box=bool(flags.get("box")), cone=bool(flags.get("cone")))
        track_dev = float(st["track_dev_m"]) if st and "track_dev_m" in st else None
        backlog = float(st["clamp_backlog_m"]) if st and "clamp_backlog_m" in st else None
        if track_dev is not None:  # Franka: judge blockage against the bridge's safe target, not the clamp backlog
            dev = track_dev
        if self.anchor == "target":
            if dev > self.anchor_hold_m:
                raise AnchorDeviationError(f"|target - FK| = {1000 * dev:.1f} mm > {1000 * self.anchor_hold_m:.0f} mm")
            if dev <= self.anchor_max_dev_m:
                poses, anchor_used = targets, "target"
            else:  # fall back to measured AND re-seed the controller interpolator from FK (no wind-up)
                self.counters["anchor_fallbacks"] += 1
                self._reseed_controller()
        obs = {
            "robot0_eef_xyz_wxyz": poses, "robot0_gripper_width": widths,
            "robot0_main_camera": images["main"], "robot0_ultrawide_camera": images["ultrawide"],
            "timestamps": main_ts, "task_name": self.task_name,
        }
        if self.policy_start_eef_xyz_wxyz is None and self.control_mode in (TaskControlMode.POLICY,
                                                                             TaskControlMode.POLICY_SINGLE_STEP):
            self.policy_start_eef_xyz_wxyz = poses[0]
        start = self.policy_start_eef_xyz_wxyz if self.policy_start_eef_xyz_wxyz is not None else poses[0]
        obs["robot0_eef_xyz_wxyz_wrt_start"] = get_relative_poses(poses, start)
        self._obs_info = {"t_obs": time.monotonic(), "main_ts": np.asarray(main_ts), "uw_ts": np.asarray(cam_ts["ultrawide"]),
                          "state_age_s": age, "anchor_used": anchor_used, "state_ts": np.asarray(state_ts),
                          "fk_pose": fk, "target_pose": targets, "anchor_pose": poses[-1].copy(),
                          "anchor_dev_m": dev, "target_fk_dev_m": float(np.linalg.norm(targets[-1, :3] - fk[-1, :3])),
                          "clamp_backlog_m": backlog if backlog is not None else np.nan,
                          "measured_width": widths.copy()}
        return obs

    def _replay_history(self, ts):
        """TEST ONLY (replay harness): the demo's TCP pose and width at the observation (image) times."""
        from scipy.spatial.transform import Rotation, Slerp

        r = self._replay
        poses, widths = [], []
        for t in np.asarray(ts, float):
            tau = (int(t * 1e9) - r["start"]) % r["span"] + r["origin"]
            tau = int(np.clip(tau, r["t_ns"][0], r["t_ns"][-1]))
            i = int(np.clip(np.searchsorted(r["t_ns"], tau), 1, len(r["t_ns"]) - 1))
            a = (tau - r["t_ns"][i - 1]) / max(1, r["t_ns"][i] - r["t_ns"][i - 1])
            P = r["xyz_wxyz"]
            rot = Slerp([0, 1], Rotation.from_quat(P[[i - 1, i]][:, [4, 5, 6, 3]]))(a).as_quat()
            poses.append(np.r_[(1 - a) * P[i - 1, :3] + a * P[i, :3], rot[3], rot[:3]])
            widths.append([(1 - a) * float(r["width"][i - 1]) + a * float(r["width"][i])])
        return {"action0_eef_xyz_wxyz": np.array(poses), "action0_gripper_width": np.array(widths),
                "timestamps": np.asarray(ts, float)}

    # ------------------------------------------------------------------ tap
    def _on_request(self, topic, req, rep, seconds):
        if topic == "policy_inference":
            self._last_request = {"req": req, "rep": rep, "seconds": seconds, "t_reply": time.monotonic()}

    def _write_tap(self, abs_actions: dict, history: dict):
        tap_dir = os.path.join(self.episode_dir, "tap") if self.episode_dir and self.episode_active else self.tap_dir
        if not tap_dir or self._last_request is None:
            return
        n = self.counters["policy_calls"]
        obs = deserialize(self._last_request["req"])
        rep = deserialize(self._last_request["rep"])
        out: dict[str, Any] = {}
        for k, v in obs.items():
            a = np.asarray(v)
            out[f"meta_shape_{k}"] = np.array(a.shape)
            out[f"meta_dtype_{k}"] = np.array(str(a.dtype))
            if "camera" not in k:
                out[f"obs_{k}"] = a
        if self.tap_images_every and n % self.tap_images_every == 0:
            import cv2

            for k in ("robot0_main_camera", "robot0_ultrawide_camera"):
                a = np.asarray(obs[k])
                h, w = a.shape[1:3]
                s = max(h, w)
                pad = np.zeros((a.shape[0], s, s, 3), np.uint8)
                pad[:, (s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
                out[f"img224_{k}"] = np.stack([cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA) for x in pad])
        for k, v in rep.items():
            out[f"reply_{k}"] = np.asarray(v)
        for k, v in abs_actions.items():
            out[f"abs_{k}"] = np.asarray(v)
        for k, v in history.items():
            out[f"hist_{k}"] = np.asarray(v)
        for k, v in self._obs_info.items():
            if v is not None:
                out[f"info_{k}"] = np.asarray(v)
        out["info_inference_s"] = np.array(self._last_request["seconds"])
        out["info_t_reply"] = np.array(self._last_request["t_reply"])
        out["info_mode"] = np.array(int(self.control_mode))
        out["info_pause"] = np.array(bool(self.pause_when_policy_inference))
        st = self.controller_status()
        if st:
            out["info_controller_counters"] = np.array(json.dumps(st.get("counters", {})))
        np.savez_compressed(os.path.join(tap_dir, f"call_{n:06d}.npz"), **out)

    # ------------------------------------------------------------------ policy step (upstream logic + tap/prompt)
    def run_policy_control(self):
        self._ensure_prompted()
        hist = self.get_history_actions(length=self.policy_agent.action_history_len, action_type="policy")
        for attempt in range(self.obs_retries + 1):  # a transient camera stall: wait for fresh packets, then retry
            try:
                obs = self.get_observations()
                break
            except FrameSelectionError as e:
                self.counters["obs_retries"] += 1
                if attempt == self.obs_retries:
                    raise
                self.log_event("obs_retry", error=str(e))
                time.sleep(0.02)
        if self._stop_requested:
            return EpisodeStatus.RUNNING
        if self.history_source == "replay_gt":
            hist = self._replay_history(self._obs_info["main_ts"])
            # B8: teacher forcing also covers the model's *_wrt_start input. The robot does not follow the demo, so
            # its own wrt_start would be meaningless: use the demo pose history relative to the demo pose at the
            # oldest observation of the first call (what get_observations uses as the start pose, poses[0]).
            if self._replay_start is None:
                self._replay_start = np.asarray(hist["action0_eef_xyz_wxyz"][0], float).copy()
            obs["robot0_eef_xyz_wxyz_wrt_start"] = get_relative_poses(
                np.asarray(hist["action0_eef_xyz_wxyz"], float), self._replay_start)
        elif self.width_proprio == "measured":
            hist = dict(hist)
            hist["action0_gripper_width"] = np.asarray(obs["robot0_gripper_width"], float).copy()
        abs_actions = self.policy_agent.predict_actions(obs, hist)
        self.counters["policy_calls"] += 1
        pa = self.policy_agent
        rebase = (self.pause_when_policy_inference or self.last_control_mode in (TaskControlMode.POLICY_SINGLE_STEP,
                                                                                 TaskControlMode.SPACEMOUSE))
        if rebase:
            abs_actions["timestamps"] = time.monotonic() + np.arange(pa.action_prediction_horizon) / pa.agent_update_freq_hz
        if self.control_mode == TaskControlMode.POLICY_SINGLE_STEP or self.pause_when_policy_inference:
            abs_actions = {k: v[: pa.action_execution_horizon] for k, v in abs_actions.items()}
        # time-scale the chunk (arm AND gripper share the timestamps) so its peak speed <= margin * v_max / w_max
        anchor = self._obs_info.get("anchor_pose")
        rebased_anchor = rebase and anchor is not None
        scale_poses = abs_actions["action0_eef_xyz_wxyz"].copy()
        if rebased_anchor:  # action 0 will be executed as the anchor itself: scale anchor -> action 1 -> ...
            scale_poses[0] = np.asarray(anchor, float)
        s_fac, vp, wp = chunk_scale(scale_poses, None if rebased_anchor else anchor,
                                    1.0 / pa.agent_update_freq_hz, self.v_max, self.w_max, self.scale_margin)
        if s_fac > 1.0:
            abs_actions["timestamps"] = scale_timestamps(abs_actions["timestamps"], s_fac)
        self._obs_info.update(chunk_scale=s_fac, chunk_peak_v=vp, chunk_peak_w=wp)
        if self._ep is not None:
            self._ep["scales"].append(s_fac)
            self._ep["backlogs"].append(float(self._obs_info.get("clamp_backlog_m", np.nan)))
        try:
            self._ctrl_msg({"annotate": {"chunk_scale": s_fac}})
        except Exception:
            pass
        if self._stop_requested:  # a stop arrived during inference: do not schedule
            self._write_tap(abs_actions, hist)
            return EpisodeStatus.RUNNING
        arm_poses = abs_actions["action0_eef_xyz_wxyz"].copy()
        arm_widths = abs_actions["action0_gripper_width"].copy()
        if rebase:
            # re-base right before scheduling (a delay here would trim action 0 and step the arm forward)
            ts = abs_actions["timestamps"]
            abs_actions["timestamps"] = ts - ts[0] + time.monotonic()
            if rebased_anchor:
                # training: the action slice starts at the observation index and is relative to the pose there, so
                # action 0 IS the current pose (identity). Execute it as the anchor exactly (its predicted ~1 mm
                # error, outliers > 1 cm, would otherwise be a one-tick jump at every chunk joint); actions k >= 1
                # keep their times t0 + k*step; width at action 0 = the last commanded width.
                arm_poses[0] = np.asarray(anchor, float)
                arm_widths[0] = np.asarray(hist["action0_gripper_width"][-1], float).reshape(-1)
        arm_ts = abs_actions["timestamps"]
        try:
            self.arx5_client.schedule_eef_traj(eef_traj_xyz_wxyz=arm_poses, timestamps=arm_ts + self.arx5_lookahead_time_s,
                                               use_relative_timestamps=False)
            self.arx5_client.schedule_joint_traj(joint_traj_pos=arm_widths, timestamps=arm_ts, use_relative_timestamps=False)
        finally:
            self._write_tap(abs_actions, hist)  # after scheduling (compressing images takes tens of ms); always
        self.policy_history_actions = {k: v[: min(pa.action_execution_horizon + 1, v.shape[0])]
                                       for k, v in abs_actions.items()}
        if self.pause_when_policy_inference or self.last_control_mode == TaskControlMode.POLICY_SINGLE_STEP:
            target = abs_actions["timestamps"][pa.action_execution_horizon - 1] + self.camera_latency_s
        else:
            target = abs_actions["timestamps"][pa.action_execution_horizon - 1] - 0.02
        while time.monotonic() < target and not self._stop_requested:
            time.sleep(min(0.005, max(0.0, target - time.monotonic())))
        if self.settle and self.pause_when_policy_inference and not self._stop_requested:
            st = settle_gate(self._settle_read, timeout_s=self.settle_timeout_s)
            self.counters["settle_waits"] = self.counters.get("settle_waits", 0) + (st["settle_s"] > 0.011)
        self.last_control_mode = self.control_mode
        return EpisodeStatus.RUNNING

    def _settle_read(self):
        """(clamp backlog or None, tracking deviation): Franka = controller status (safe target vs FK);
        mock = |commanded target - FK| from the state stream."""
        st = self.controller_status()
        if st and "track_dev_m" in st:
            flags = st.get("safety_flags") or {}
            if flags.get("box") or flags.get("cone"):
                # target outside the workspace box / orientation cone: the backlog is permanent by design (the
                # bridge holds the boundary); do not wait for it
                self.counters["workspace_clamps"] = self.counters.get("workspace_clamps", 0) + 1
                return 0.0, float(st["track_dev_m"])
            return float(st.get("clamp_backlog_m", 0.0)), float(st["track_dev_m"])
        now = time.monotonic()
        fk = self.arx5_client.get_eef_xyz_wxyz(timestamp=now)
        tg = self.arx5_client.get_eef_target_xyz_wxyz(timestamp=now)
        return None, float(np.linalg.norm(np.asarray(tg[:3]) - np.asarray(fk[:3])))

    # ------------------------------------------------------------------ input
    def _request_stop(self, key: str, source: str):
        if self._stop_requested is None and self.control_mode in (TaskControlMode.POLICY,
                                                                  TaskControlMode.POLICY_SINGLE_STEP):
            self._stop_requested = key
            self.hold(f"stop key {key!r} ({source})")
        self._keys.put(key)

    def _input_thread(self):
        fpos = 0
        tty = self.read_stdin and sys.stdin is not None and sys.stdin.isatty()
        old = None
        if tty:
            import termios
            import tty as ttym

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            ttym.setcbreak(fd)
        try:
            while not self._quit:
                keys = []
                if tty:
                    r, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if r:
                        keys.append(("stdin", sys.stdin.read(1)))
                else:
                    time.sleep(0.01)
                if self.command_file and os.path.exists(self.command_file):
                    with open(self.command_file) as fh:
                        fh.seek(fpos)
                        data = fh.read()
                        fpos = fh.tell()
                    keys += [("file", ln.strip()) for ln in data.splitlines() if ln.strip()]
                for src, k in keys:
                    if k in STOP_KEYS:
                        self._request_stop(k, src)
                    else:
                        self._keys.put(k)
        finally:
            if old is not None:
                import termios

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)

    # ------------------------------------------------------------------ main loop (replaces BaseTask.run)
    def _finish_stop(self, key: str):
        is_successful = {"y": True, "n": False}.get(key)
        self.control_mode = TaskControlMode.SPACEMOUSE
        self.stop_episode(is_successful=is_successful, label=key)
        self._stop_requested = None
        try:
            self.release()
        except Exception as e:
            self.log_event("release_failed", error=repr(e))

    def _fail_policy(self, e: BaseException):
        self.counters["policy_errors"] += 1
        self.hold(f"exception: {type(e).__name__}")
        self.log_event("policy_exception", error=repr(e), tb=traceback.format_exc(limit=4),
                       failed_call_start=getattr(e, "t_call_start", None), failed_call_waited_s=getattr(e, "waited_s", None))
        self.control_mode = TaskControlMode.SPACEMOUSE
        label = f"aborted:{self.abort_reason}" if self.abort_reason else f"exception:{type(e).__name__}"
        self.stop_episode(is_successful=None, label=label)
        self._stop_requested = None
        try:
            self.release()
        except Exception as e2:
            self.log_event("release_failed", error=repr(e2))

    def handle_key(self, k: str):
        if k in STOP_KEYS:
            if self._stop_requested:
                self._finish_stop(self._stop_requested)
            else:  # not in policy mode: the stop watcher may still have latched a hold
                try:
                    self.release()
                except Exception as e:
                    self.log_event("release_failed", error=repr(e))
            return
        if k.startswith("note:"):
            note = {"t": time.monotonic(), "wall": time.time(), "text": k[5:].strip()}
            if self._ep is not None:
                self._ep["notes"].append(note)
            self.log_event("note", text=note["text"])
            return
        if k in ("c", "C"):
            why = self.robot_ready() or self.home_check()
            if why:
                self.log_event("refused_start", key=k, reason=why)
                return
        if k == "c":
            self.start_episode()
            self.control_mode = TaskControlMode.POLICY
        elif k == "C":
            self.control_mode = TaskControlMode.POLICY_SINGLE_STEP
        elif k == "r":
            self.reset()
        elif k == "R":
            self.reset_policy_agent()
        elif k == "d":
            self.display_robot_state()
        elif k == "q":
            self._quit = True
        else:
            self.log_event("unknown_key", key=k)

    def run(self, max_duration_s: float | None = None):
        th = threading.Thread(target=self._input_thread, daemon=True)
        th.start()
        t_end = time.monotonic() + max_duration_s if max_duration_s else None
        last_err_print = 0.0
        try:
            while not self._quit and (t_end is None or time.monotonic() < t_end):
                while not self._keys.empty():
                    k = self._keys.get()
                    try:
                        self.handle_key(k)
                    except Exception as e:
                        self._fail_policy(e)
                if self._stop_requested and self.control_mode == TaskControlMode.SPACEMOUSE:
                    self._stop_requested = None
                if self.control_mode in (TaskControlMode.POLICY, TaskControlMode.POLICY_SINGLE_STEP):
                    single = self.control_mode == TaskControlMode.POLICY_SINGLE_STEP
                    try:
                        self.run_policy_control()
                    except Exception as e:
                        self._fail_policy(e)
                        continue
                    if single and not self._stop_requested:
                        self.control_mode = TaskControlMode.SPACEMOUSE
                else:
                    try:
                        self.run_spacemouse_control()
                    except Exception as e:
                        self.counters["idle_errors"] += 1
                        if time.monotonic() - last_err_print > 1.0:
                            last_err_print = time.monotonic()
                            self.log_event("idle_exception", error=repr(e))
                        time.sleep(0.05)
        finally:
            self._quit = True
            if self.episode_active:  # shutdown with an episode running: finalize it as aborted
                try:
                    self.hold(f"shutdown: {self.abort_reason or 'quit'}")
                    self.stop_episode(is_successful=None, label=f"aborted:{self.abort_reason or 'shutdown'}")
                except Exception as e:
                    self.log_event("abort_finalize_failed", error=repr(e))
                self._watcher.terminate()
                try:
                    self._watcher.wait(5)
                except Exception:
                    self._watcher.kill()
            self.log_event("task_exit", counters=self.counters, prompts=self.prompt_count)

    def summary(self) -> dict:
        return {"counters": self.counters, "prompt_count": self.prompt_count, "events": self.events}
