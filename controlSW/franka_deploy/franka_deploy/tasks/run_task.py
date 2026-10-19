"""Start FrankaPromptTask (headless) with the trained-contract settings.

    python -m franka_deploy.tasks.run_task --work-dir W [--task-name block_raise --prompt-index 0]
        [--pause | --continuous] [--exec-h 8] [--anchor measured|target] [--duration 600]
        [--command-file W/commands.txt --stop-file W/STOP] [--tap-dir W/tap --tap-images-every 20]

Endpoints default to: inference tcp://localhost:18765, camera tcp://localhost:16702 (rig_wrist_*), controller
tcp://localhost:26010, spacemouse tcp://localhost:15557. Agent configs are real-env's policy_agent.yaml /
spacemouse_agent.yaml with the overrides the trained policy needs (agent_update_freq_hz 20). Writes
<work>/task_summary.json at exit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from franka_deploy.paths import check_writable_dir

RE_CFG = Path(__file__).resolve().parents[4] / "behavior_prompting" / "deps" / "real-env" / "real_env" / "configs" / "agents"


def build_task(a):
    from real_env.agents.spacemouse_agent import SpacemouseAgent
    from franka_deploy.agents.safe_policy_agent import SafePolicyAgent
    from franka_deploy.tasks.franka_task import FrankaPromptTask

    pa_cfg = OmegaConf.to_container(OmegaConf.load(RE_CFG / "policy_agent.yaml"))
    pa_cfg.pop("_target_")
    pa_cfg.update({"agent_update_freq_hz": 20, "action_execution_horizon": a.exec_h,
                   "policy_server_endpoint": a.policy_endpoint,
                   "rpc_timeouts_s": {"policy_inference": a.inference_timeout, "policy_prompt": 60.0}})
    sm_cfg = OmegaConf.to_container(OmegaConf.load(RE_CFG / "spacemouse_agent.yaml"))
    sm_cfg.pop("_target_")
    sm_cfg.pop("defaults", None)
    sm_cfg.update({"pose_orientation_types": [0], "rmq_server_address": a.spacemouse_endpoint})
    policy_agent = SafePolicyAgent(**pa_cfg)
    spacemouse_agent = SpacemouseAgent(**sm_cfg)
    cfg = {k: v for k, v in vars(a).items()}
    return FrankaPromptTask(
        task_name=a.task_name, prompt_index=a.prompt_index, anchor=a.anchor, state_max_age_s=a.state_max_age,
        anchor_max_dev_m=a.anchor_fallback, anchor_hold_m=a.anchor_hold, width_proprio=a.width_proprio,
        history_source=a.history_source, replay_npz=a.replay_npz, replay_start_now_ns=a.replay_start_now_ns,
        spacemouse_endpoint=a.spacemouse_endpoint, v_max=a.v_max, w_max=a.w_max, settle=not a.no_settle,
        session_dir=a.session_dir, bridge_log=a.bridge_log, camera_log=a.camera_log,
        home_xyz=a.home_xyz, home_max_dist_m=a.home_max_dist, max_image_age_s=a.max_image_age_s,
        controller_timeout_s=a.controller_timeout, command_file=a.command_file, stop_file=a.stop_file,
        read_stdin=not a.no_stdin, tap_dir=a.tap_dir, tap_images_every=a.tap_images_every,
        record_episodes=not a.no_record,
        root_dir=os.path.join(a.work_dir, "episodes"), project_name="franka_deploy", run_name=a.run_name,
        logger_endpoints=({"right_wrist_camera_0": a.video_logger_endpoint} if a.video_logger_endpoint else {}),
        spacemouse_agent=spacemouse_agent, policy_agent=policy_agent,
        config_str=json.dumps(cfg), arx5_controller_endpoint=a.controller_endpoint, arx5_lookahead_time_s=a.lookahead_s,
        pause_when_policy_inference=a.pause, wrist_camera_endpoint=a.camera_endpoint,
        third_person_camera_endpoint="", arx5_home_gripper_pos_m=a.home_gripper_m, draw_gripper_mask=False,
        draw_mirrors_mask=False, camera_latency_s=a.camera_latency_s, policy_latency_s=a.policy_latency_s, crop_image=None,
        match_camera_latency=a.match_camera_latency)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--task-name", default="block_raise")
    ap.add_argument("--prompt-index", type=int, default=0)
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--pause", dest="pause", action="store_true", default=True)
    m.add_argument("--continuous", dest="pause", action="store_false")
    ap.add_argument("--exec-h", type=int, default=8)
    ap.add_argument("--anchor", default="target", choices=["measured", "target"])
    ap.add_argument("--anchor-fallback", type=float, default=0.02)
    ap.add_argument("--anchor-hold", type=float, default=0.05)
    ap.add_argument("--width-proprio", default="commanded", choices=["commanded", "measured"])
    ap.add_argument("--history-source", default="commanded", choices=["commanded", "replay_gt"],
                    help="replay_gt = TEST ONLY (replay harness)")
    ap.add_argument("--replay-npz", default=None)
    ap.add_argument("--replay-start-now-ns", type=int, default=None)
    ap.add_argument("--match-camera-latency", action="store_true")
    ap.add_argument("--state-max-age", type=float, default=0.05)
    ap.add_argument("--controller-timeout", type=float, default=0.05)
    ap.add_argument("--inference-timeout", type=float, default=2.0)
    ap.add_argument("--policy-endpoint", default="tcp://localhost:18765")
    ap.add_argument("--camera-endpoint", default="tcp://localhost:16702")
    ap.add_argument("--controller-endpoint", default="tcp://localhost:26010")
    ap.add_argument("--spacemouse-endpoint", default="tcp://localhost:15557")
    ap.add_argument("--command-file", default=None)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--no-stdin", action="store_true")
    ap.add_argument("--tap-dir", default=None)
    ap.add_argument("--tap-images-every", type=int, default=0)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--v-max", type=float, default=0.25, help="chunk time-scaling limit (= bridge safety v_max)")
    ap.add_argument("--w-max", type=float, default=1.0)
    ap.add_argument("--no-settle", action="store_true")
    ap.add_argument("--session-dir", default=None, help="config snapshot copied into every episode dir")
    ap.add_argument("--bridge-log", default=None)
    ap.add_argument("--camera-log", default=None)
    ap.add_argument("--video-logger-endpoint", default=None, help="camera VideoLogger endpoint -> libx264 video per episode")
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--home-xyz", type=float, nargs=3, default=None, help="refuse c/C unless FK is near this TCP position")
    ap.add_argument("--home-max-dist", type=float, default=0.05)
    ap.add_argument("--lookahead-s", type=float, default=0.1, help="controller lookahead (station policy.task.lookahead_s)")
    ap.add_argument("--home-gripper-m", type=float, default=0.08, help="gripper width at home (station tooling.hand.home_width_m)")
    ap.add_argument("--camera-latency-s", type=float, default=0.08, help="camera latency (station sensor.task.camera_latency_s)")
    ap.add_argument("--policy-latency-s", type=float, default=0.1, help="policy latency (station policy.task.policy_latency_s)")
    ap.add_argument("--max-image-age-s", type=float, default=None,
                    help="refuse (retryable FrameSelectionError) an observation whose newest main image is older "
                         "(station sensor.task.max_image_age_s); default: no check")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    a.work_dir = str(check_writable_dir(a.work_dir, "work dir"))
    os.makedirs(a.work_dir, exist_ok=True)
    task = build_task(a)
    import signal

    def _sig(signum, _frame):
        task.abort_reason = {signal.SIGINT: "sigint", signal.SIGTERM: "sigterm"}.get(signum, str(signum))
        task._quit = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    if a.ready_file:
        with open(a.ready_file, "w") as f:
            json.dump({"pid": os.getpid()}, f)
    try:
        task.run(max_duration_s=a.duration)
    finally:
        with open(os.path.join(a.work_dir, "task_summary.json"), "w") as f:
            json.dump(task.summary(), f, indent=1, default=str)


if __name__ == "__main__":
    main()
