"""`python -m policy --mock` as a subprocess (light environment, hold_still)."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from contracts import PolicyConfig, PolicyConfigRequest, RpcClient

START_TIMEOUT_S = 30.0


def _free_endpoint() -> str:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{s.getsockname()[1]}"


def _start(args: list[str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "policy", *args], cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def _wait_for(condition, proc: subprocess.Popen, timeout_s: float) -> None:
    """Wait (with a deadline) until condition() is true; fail early if the process exits."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if proc.poll() is not None:
            raise AssertionError(f"policy exited early ({proc.returncode}): {proc.stderr.read()}")
        if time.monotonic() > deadline:
            proc.kill()
            raise AssertionError("timed out")
        time.sleep(0.02)


def _stop(proc: subprocess.Popen) -> tuple[int, str]:
    proc.send_signal(signal.SIGINT)
    try:
        _, err = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    return proc.returncode, err


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_mock_run_ready_startup_config_and_sigint(station_file, run_dir):
    endpoint = _free_endpoint()
    station = station_file(None, endpoint=endpoint)
    session = run_dir / "session"
    ready = session / "ready" / "policy.ready"
    proc = _start(["--station", str(station), "--session", str(session), "--ready-file", str(ready), "--mock"],
                  run_dir)
    try:
        _wait_for(ready.exists, proc, START_TIMEOUT_S)
        client = RpcClient("policy", endpoint)                       # the port answers once the ready file exists
        try:
            config = client.request(PolicyConfigRequest(), PolicyConfig, timeout_s=5.0)
        finally:
            client.close()
        assert list(config.inputs) == ["gripper_width_m"]
        assert ready.read_text() == "" and not ready.with_name("policy.ready.tmp").exists()
        startup = json.loads((session / "policy" / "startup.json").read_text())
        assert startup["kind"] == "hold_still" and startup["mock"] is True
        assert startup["endpoint"] == endpoint and startup["pid"] == proc.pid
        assert len(startup["warmup_s"]) == 2 and startup["policy"] == {"kind": "hold_still"}
    finally:
        code, err = _stop(proc)
    assert code == 0, err
    assert "FATAL" not in err


def test_broken_station_section_exits_1_with_one_fatal_line(station_file, run_dir):
    station = station_file({"kind": "hold_still", "chekpoint": "x"})
    proc = subprocess.run([sys.executable, "-m", "policy", "--station", str(station), "--session",
                           str(run_dir / "s")], cwd=run_dir, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    lines = proc.stderr.strip().splitlines()
    assert lines[-1] == "FATAL policy: station.yaml: policy.chekpoint: not a key of policy/config.yaml"
    assert sum(line.startswith("FATAL") for line in lines) == 1


def test_kind_unset_fatal_line(station_file, run_dir):
    proc = subprocess.run([sys.executable, "-m", "policy", "--station", str(station_file({}))], cwd=run_dir,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    assert proc.stderr.strip().splitlines()[-1] == (
        "FATAL policy: policy.kind is not set: choose diffusion or hold_still in station.yaml (policy.kind)")


def test_without_session_writes_the_policy_alone_folder(station_file, run_dir):
    endpoint = _free_endpoint()
    station = station_file(None, endpoint=endpoint, session_root="sessions")
    ready = run_dir / "alone.ready"
    proc = _start(["--station", str(station), "--ready-file", str(ready), "--mock"], run_dir)
    try:
        _wait_for(ready.exists, proc, START_TIMEOUT_S)
    finally:
        code, err = _stop(proc)
    assert code == 0, err
    folders = list((run_dir / "sessions").iterdir())
    assert len(folders) == 1 and folders[0].name.endswith("_policy_alone")
    assert (folders[0] / "policy" / "startup.json").exists()
