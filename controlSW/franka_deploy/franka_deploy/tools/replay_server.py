"""Single-arm action-replay server (upstream detached_inference_action_replay is bimanual-only, E7).

Drop-in for the inference server (same robotmq topics and reply format, robot0 = the single arm): every
policy_inference returns the demo's own future instead of a model prediction, in the TRAINING convention:
action[k] = pose(t + stride*k) expressed relative to pose(t) (action 0 = identity), width absolute,
stride = 3 frames (60 Hz -> 20 Hz actions) at 1x; `--speed 0.5` takes 1.5 frames per action step (half speed).
After each call the demo pointer advances by (exec_h - 1) action steps in pause mode (the arm stops at action
exec_h - 1, which is where the next chunk's action 0 must be), or exec_h steps with --continuous.
The reply also carries "replay_frame" (pointer) so taps can be compared with the demo.

    python -m franka_deploy replay-server --zarr Z --episode NAME|INDEX [--start-frame 0] [--speed 1.0]
        [--exec-h 8] [--endpoint tcp://0.0.0.0:18765] [--ready-file F]
    python -m franka_deploy run --config deploy.yaml --replay-episode E --replay-speed 0.5   (launcher, hardware day)
"""

from __future__ import annotations

import argparse
import json
import signal
import sys

import numpy as np
import robotmq
from scipy.spatial.transform import Rotation

CONFIG = {"workspace": {"proprio_length": 1, "img_obs_horizon": 2, "obs_down_sample_steps": 3,
                        "ultrawide_down_sample_steps": 1},
          "use_prompting": True, "policy_name": "action_replay", "run_name": "action_replay", "epoch": 0,
          "date_str": "replay", "time_str": "replay"}


def load_episode(zarr_path, episode):
    import zarr

    root = zarr.open(zarr.ZipStore(zarr_path, mode="r"), mode="r")
    names = [str(x) for x in root["meta/episode_names"][:]]
    i = names.index(episode) if isinstance(episode, str) and episode in names else int(episode)
    ends = root["meta/episode_ends"][:]
    s, e = (0 if i == 0 else int(ends[i - 1])), int(ends[i])
    pos = root["data/gripper_right_eef_pos"][s:e].astype(float)
    rv = root["data/gripper_right_eef_rot_axis_angle"][s:e].astype(float)
    w = root["data/gripper_right_gripper_width"][s:e, 0].astype(float)
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(rv).as_matrix()
    T[:, :3, 3] = pos
    return names[i], T, w


def interp_T(T, f):
    f = float(np.clip(f, 0, len(T) - 1))
    i = int(np.floor(f))
    j = min(i + 1, len(T) - 1)
    a = f - i
    out = np.eye(4)
    out[:3, 3] = (1 - a) * T[i, :3, 3] + a * T[j, :3, 3]
    from scipy.spatial.transform import Slerp

    out[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([T[i, :3, :3], T[j, :3, :3]])))(a).as_matrix()
    return out


class ReplayServer:
    def __init__(self, T, w, start=0, speed=1.0, exec_h=8, horizon=16, stride=3, continuous=False):
        self.T, self.w = T, w
        self.start, self.t = float(start), float(start)
        self.step = stride * speed
        self.exec_h, self.horizon = exec_h, horizon
        self.advance = (exec_h if continuous else exec_h - 1) * self.step

    def chunk(self) -> dict:
        P0 = interp_T(self.T, self.t)
        inv0 = np.linalg.inv(P0)
        pos, rot6, wid = [], [], []
        for k in range(self.horizon):
            f = self.t + k * self.step
            rel = inv0 @ interp_T(self.T, f)
            pos.append(rel[:3, 3])
            rot6.append(rel[:3, :3][:2].reshape(-1))   # rot6d = first two rows (training / robot_utils convention)
            wid.append([np.interp(f, np.arange(len(self.w)), self.w)])
        out = {"action0_eef_pos": np.array(pos, np.float32), "action0_eef_rot_axis_angle": np.array(rot6, np.float32),
               "action0_gripper_width": np.array(wid, np.float32), "replay_frame": np.array(self.t)}
        self.t = min(self.t + self.advance, len(self.T) - 1)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--start-frame", type=float, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--exec-h", type=int, default=8)
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--endpoint", default="tcp://0.0.0.0:18765")
    ap.add_argument("--ready-file", default=None, help="written once serving (the launcher waits for it)")
    a = ap.parse_args(argv)
    name, T, w = load_episode(a.zarr, a.episode)
    rs = ReplayServer(T, w, a.start_frame, a.speed, a.exec_h, continuous=a.continuous)
    srv = robotmq.RMQServer("policy_server", a.endpoint)
    for t in ("policy_config", "policy_reset", "policy_prompt", "policy_inference", "export_recorded_data"):
        srv.add_topic(t, message_remaining_time_s=60)
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    print(f"[replay] {name}: {len(T)} frames, speed {a.speed}, serving on {a.endpoint}", flush=True)
    if a.ready_file:
        with open(a.ready_file, "w") as f:
            json.dump({"replay": name, "frames": len(T), "speed": a.speed, "start_frame": a.start_frame}, f)
    while not stop["v"]:
        data, topic = srv.wait_for_request(timeout_s=0.05)
        if not topic:
            continue
        if topic == "policy_config":
            rep = CONFIG
        elif topic == "policy_reset":
            rs.t = rs.start
            rep = "OK"
        elif topic == "policy_prompt":
            rep = "OK"
        elif topic == "policy_inference":
            rep = rs.chunk()
        else:
            rep = ""
        srv.reply_request(topic=topic, data=robotmq.serialize(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
