"""Stage 2 of obs_equivalence (runs in the real-env env): the unchanged real-env PolicyAgent.predict_actions on raw
deployment inputs, with a fake RMQ client that records the exact `obs_to_send` and replies with the dataset's GT
relative action chunk, so the relative->absolute action path is checked against the demo's true future poses.

    python -m franka_deploy.tools.agent_capture --in-dir D [--agent-hz 20]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

RE_ROOT = Path(__file__).resolve().parents[4] / "behavior_prompting" / "deps" / "real-env" / "real_env"


class FakePolicyClient:
    """Stands in for robotmq.RMQClient inside real_env.agents.policy_agent."""

    reply_actions: dict | None = None
    captured: dict | None = None

    def __init__(self, client_name, server_endpoint, *a, **k):
        self.name = client_name

    def request_with_data(self, topic, data, timeout_s=1.0, automatic_resend=True):
        from robotmq import deserialize, serialize

        if topic == "policy_config":
            return serialize({"workspace": {"proprio_length": 1, "img_obs_horizon": 2, "obs_down_sample_steps": 3,
                                            "ultrawide_down_sample_steps": 1},
                              "use_prompting": True, "policy_name": "fake", "run_name": "fake", "epoch": 0,
                              "date_str": "x", "time_str": "y"})
        if topic == "policy_inference":
            FakePolicyClient.captured = deserialize(data)
            return serialize(FakePolicyClient.reply_actions)
        return serialize("OK")


def make_agent(agent_hz: float):
    from omegaconf import OmegaConf
    import real_env.agents.policy_agent as pa

    pa.RMQClient = FakePolicyClient
    cfg = OmegaConf.to_container(OmegaConf.load(RE_ROOT / "configs" / "agents" / "policy_agent.yaml"))
    cfg.pop("_target_")
    cfg["agent_update_freq_hz"] = agent_hz
    return pa.PolicyAgent(**cfg)


def rot_angle_deg(q1_wxyz, q2_wxyz):
    d = np.abs(np.sum(q1_wxyz * q2_wxyz, axis=-1)).clip(0, 1)
    return np.degrees(2 * np.arccos(d))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--agent-hz", type=float, default=20.0)
    a = ap.parse_args(argv)
    from robot_utils.pose_utils import get_relative_pose

    agent = make_agent(a.agent_hz)
    report = []
    for p in sorted(glob.glob(os.path.join(a.in_dir, "deploy_in_*.npz"))):
        tag = os.path.basename(p)[len("deploy_in_"):-4]
        d = dict(np.load(p))
        poses, widths = d["eef_xyz_wxyz"], d["gripper_width"]
        ts = np.array([10.0 - 1 / a.agent_hz, 10.0])
        obs = {"robot0_eef_xyz_wxyz": poses.copy(), "robot0_gripper_width": widths.copy(),
               "robot0_main_camera": d["main"], "robot0_ultrawide_camera": d["ultrawide"], "timestamps": ts,
               "task_name": "block_raise",
               "robot0_eef_xyz_wxyz_wrt_start": np.array([get_relative_pose(x, d["start_xyz_wxyz"]) for x in poses])}
        hist = {"action0_eef_xyz_wxyz": poses.copy(), "action0_gripper_width": widths.copy(), "timestamps": ts}
        g = d["gt_action"]
        FakePolicyClient.reply_actions = {"action0_eef_pos": g[:, :3], "action0_eef_rot_axis_angle": g[:, 3:9],
                                          "action0_gripper_width": g[:, 9:]}
        out = {}
        for variant, o in (("captured", obs), ("captured_nowrt", {k: v for k, v in obs.items() if "wrt_start" not in k})):
            acts = agent.predict_actions(o, hist)
            cap = {k: np.asarray(v) for k, v in FakePolicyClient.captured.items() if k != "task_name"}
            abs_pose = acts["action0_eef_xyz_wxyz"]
            fut = d["gt_future_xyz_wxyz"]
            cap["abs_err_pos_m_max"] = np.array(np.linalg.norm(abs_pose[:, :3] - fut[:, :3], axis=1).max())
            cap["abs_err_rot_deg_max"] = np.array(rot_angle_deg(abs_pose[:, 3:], fut[:, 3:]).max())
            cap["abs_err_width_m_max"] = np.array(np.abs(acts["action0_gripper_width"][:, 0] - d["gt_future_width"][:, 0]).max())
            cap["abs_timestamps_dt"] = np.diff(acts["timestamps"])
            np.savez(os.path.join(a.in_dir, f"{variant}_{tag}.npz"), **cap)
            out[variant] = {k: float(cap[k]) for k in ("abs_err_pos_m_max", "abs_err_rot_deg_max", "abs_err_width_m_max")}
            out[variant]["keys"] = sorted(k for k in cap if not k.startswith("abs_"))
            out[variant]["action_dt"] = float(cap["abs_timestamps_dt"].mean())
        report.append({"tag": tag, **out})
        print(json.dumps(report[-1]), flush=True)
    with open(os.path.join(a.in_dir, "capture_summary.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
