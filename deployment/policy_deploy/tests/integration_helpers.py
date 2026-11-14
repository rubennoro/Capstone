"""Helpers for the end-to-end tests (tests/integration): a temporary laptop station, running `policy_deploy run`
as a subprocess, reading the session folder."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from deploy_contracts import LinkSettings
from station_helpers import free_port

from policy_deploy.contracts import paths

EPISODE = Path("~/Downloads/episode_000045").expanduser()


def need_episode() -> None:
    if not (EPISODE / "meta.json").is_file():
        pytest.skip(f"no recorded Pi episode at {EPISODE}")


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def need_pc_ports_free() -> None:
    link = LinkSettings.load()
    for port in (link.state_port, link.commands_port):
        if port_busy(port):
            pytest.skip(f"127.0.0.1:{port} (link.yaml) is busy")


def write_run_station(tmp_path: Path, script: list[dict], fake_pc: dict | None = None,
                      launcher: dict | None = None) -> Path:
    (tmp_path / "script.yaml").write_text(yaml.safe_dump(script, sort_keys=False))
    ports = set()
    while len(ports) < 3:
        ports.add(free_port())
    frames, policy, mock_pi = sorted(ports)
    station = {
        "station_version": 1,
        "session_root": str(tmp_path / "sessions"),
        "python": {"light": sys.executable},
        "pi": {"port": 5555},
        "ports": {"frames": frames, "policy": policy, "mock_pi": mock_pi},
        "frames": {"mock_episode": str(EPISODE)},
        "policy": {"kind": "hold_still"},
        "manager": {"operator_script": "script.yaml", "operator_script_wait_s": 30.0},
        "fake_pc": fake_pc or {},
        "launcher": launcher or {},
    }
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(station, sort_keys=False))
    return path


def run_policy_deploy(station: Path, timeout_s: float = 150.0) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, "-m", "policy_deploy", "run", "--mock", "--station", str(station)],
                          cwd=paths.folder(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=timeout_s, env=env)


def session_dir(tmp_path: Path) -> Path:
    found = sorted((tmp_path / "sessions").glob("*_mock"))
    assert len(found) == 1, found
    return found[0]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
