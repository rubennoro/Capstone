"""Shared test helpers: the real files, a valid station, small config.yaml contents."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

import deploy

REAL_ROOT = deploy.DEPLOYMENT_ROOT                     # the real deployment/ folder, captured before any patch
REAL_GRIPPER = REAL_ROOT / "gripper.yaml"
REAL_RIG = REAL_ROOT / "process_frames" / "rigs" / "rig_pi_v1_standin.yaml"

# F_T_TCP as franka_control/config.yaml holds it (PLAN §6.6): -45 deg about flange z, fingertips 150 mm out.
_C = 0.7071067811865476
F_T_TCP_LIST = [[_C, _C, 0.0, 0.0], [-_C, _C, 0.0, 0.0], [0.0, 0.0, 1.0, 0.150], [0.0, 0.0, 0.0, 1.0]]

STATION = {
    "station_version": 1,
    "session_root": "sessions",
    "python": {"light": "/usr/bin/python3", "policy": "/opt/policy/bin/python"},
    "endpoints": {
        "pi_frames": "tcp://192.168.1.50:5600",
        "pi_time_sync": "tcp://192.168.1.50:5601",
        "mock_pi_frames": "tcp://127.0.0.1:5600",
        "mock_pi_time_sync": "tcp://127.0.0.1:5601",
        "frames": "tcp://127.0.0.1:5610",
        "state": "tcp://127.0.0.1:5620",
        "commands": "tcp://127.0.0.1:5630",
        "policy": "tcp://127.0.0.1:5640",
    },
}

CONFIGS = {
    "franka_control": {"binary": "build/franka_control", "robot_ip": "172.16.0.2", "realtime": "enforce",
                       "tcp": {"F_T_TCP": F_T_TCP_LIST}, "watchdog": {"limit_s": 0.2, "enabled": True}},
    "process_frames": {"rig_file": "rigs/rig_pi_v1_standin.yaml", "decoder": "h264", "mock_episode": None},
    "policy": {"kind": None, "checkpoint": None, "device": "cuda", "diffusion_steps": None},
    "message_comms": {"exec_h": 8, "operator_script": None},
}


def station_data(**sections) -> dict:
    """A copy of the valid station, with top-level keys replaced (None removes the key)."""
    data = copy.deepcopy(STATION)
    for key, value in sections.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def write_yaml(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
