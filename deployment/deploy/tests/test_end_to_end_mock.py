"""Milestone 3 (PLAN §7): `deploy run --mock` with scripted keys runs an episode and shuts down cleanly.

Needs all four processes, so it is skipped until the four launch.py files exist, the franka_control binary is built,
and the real station's process_frames.mock_episode (a recorded Pi episode) exists.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from deploy import Station, StationError, load_process_config, resolve_path
from helpers import REAL_ROOT, write_yaml

FOLDERS = ["franka_control", "process_frames", "policy", "message_comms"]
LOCAL_ENDPOINTS = ["mock_pi_frames", "mock_pi_time_sync", "frames", "state", "commands", "policy"]
RUN_TIMEOUT_S = 600.0          # the whole run, start-up included (the policy's ready timeout alone is 300 s)
SCRIPT = [{"wait_phase": "idle"}, {"key": "c"}, {"wait_steps": 3}, {"key": "y"}, {"key": "q"}]


def free_ports(n: int) -> list[int]:
    sockets = [socket.socket() for _ in range(n)]
    try:
        for s in sockets:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


def prerequisites() -> tuple[Station, Path, Path]:
    """The real station (mock), the absolute mock episode and rig file; skips the test when something is missing."""
    missing = [name for name in FOLDERS if not (REAL_ROOT / name / "launch.py").exists()]
    if missing:
        pytest.skip(f"launch.py does not exist yet in: {', '.join(missing)}")
    station = Station.load(REAL_ROOT / "station.yaml", mock=True)
    try:
        franka = load_process_config(REAL_ROOT / "franka_control", station.section("franka_control"))
        binary = resolve_path(station, "franka_control", franka, "binary")
        frames = load_process_config(REAL_ROOT / "process_frames", station.section("process_frames"))
        episode = resolve_path(station, "process_frames", frames, "mock_episode")
        rig_file = resolve_path(station, "process_frames", frames, "rig_file")
    except StationError as exc:
        pytest.skip(f"not set up for an end-to-end run: {exc}")
    if not binary.exists():
        pytest.skip(f"franka_control is not built: {binary}")
    if not episode.exists():
        pytest.skip(f"process_frames.mock_episode does not exist: {episode}")
    return station, episode, rig_file


def test_deploy_run_mock_with_scripted_keys(tmp_path):
    station, episode, rig_file = prerequisites()
    data = yaml.safe_load(station.path.read_text())
    data["session_root"] = str(tmp_path / "sessions")
    # The copy lives elsewhere, so its relative paths are made absolute (the real station uses ~ paths otherwise).
    data["python"] = {key: str(station.resolve(value)) for key, value in (data.get("python") or {}).items()}
    data["endpoints"].update({name: f"tcp://127.0.0.1:{port}"
                              for name, port in zip(LOCAL_ENDPOINTS, free_ports(len(LOCAL_ENDPOINTS)))})
    data.setdefault("process_frames", {})["mock_episode"] = str(episode)
    data["process_frames"]["rig_file"] = str(rig_file)       # the real process_frames/rigs/rig_pi_v2.yaml
    data.setdefault("message_comms", {})["operator_script"] = str(write_yaml(tmp_path / "keys.yaml", SCRIPT))
    station_file = write_yaml(tmp_path / "station.yaml", data)

    done = subprocess.run([sys.executable, "-m", "deploy", "run", "--mock", "--station", str(station_file)],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
                          cwd=tmp_path)                   # not deployment/: see deploy/STRUCTURE.md 7.2
    assert done.returncode == 0, f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    assert "deploy: all 4 processes ready" in done.stdout
    assert done.stdout.strip().splitlines()[-1].startswith("deploy: done (session: ")

    [session] = list((tmp_path / "sessions").iterdir())
    assert session.name.endswith("_mock")
    episodes = [json.loads(line) for line in (session / "message_comms" / "episodes.jsonl").read_text().splitlines()]
    assert [e["outcome"] for e in episodes] == ["success"]
    log = (session / "logs" / "deploy.log").read_text()
    for name in FOLDERS:
        assert f"{name} stopped (exit " in log
