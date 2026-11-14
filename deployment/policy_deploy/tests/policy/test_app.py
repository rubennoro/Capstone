# adapted from rnoro/deployment:deployment/policy/tests/test_app.py @ 91b178b
# changes: python -m policy_deploy.policy; the new station (endpoint from ports.policy); the FATAL lines of the new
# settings path; startup.json adds precision, memory and gpu; the config reply carries kind.
"""`python -m policy_deploy.policy --mock` as a subprocess (light environment, hold_still)."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from deploy_contracts import RpcClient
from policy_fakes import endpoint_of

from policy_deploy.contracts import PolicyConfig, PolicyConfigRequest

START_TIMEOUT_S = 30.0


def _start(args: list[str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "policy_deploy.policy", *args], cwd=cwd, stdout=subprocess.PIPE,
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


def test_mock_run_ready_startup_config_and_sigint(policy_station, tmp_path):
    station = policy_station(None)
    endpoint = endpoint_of(station)
    session = tmp_path / "session"
    ready = session / "ready" / "policy.ready"
    proc = _start(["--station", str(station), "--session", str(session), "--ready-file", str(ready), "--mock"],
                  tmp_path)
    try:
        _wait_for(ready.exists, proc, START_TIMEOUT_S)
        client = RpcClient("policy", endpoint)                       # the port answers once the ready file exists
        try:
            config = client.request(PolicyConfigRequest(), PolicyConfig, timeout_s=5.0)
        finally:
            client.close()
        assert list(config.inputs) == ["gripper_width_m"] and config.kind == "hold_still"
        assert ready.read_text() == "" and not ready.with_name("policy.ready.tmp").exists()
        startup = json.loads((session / "policy" / "startup.json").read_text())
        assert startup["kind"] == "hold_still" and startup["mock"] is True and startup["precision"] == "fp32"
        assert startup["endpoint"] == endpoint and startup["pid"] == proc.pid
        assert len(startup["warmup_s"]) == 2 and startup["policy"] == {"kind": "hold_still"}
        assert startup["memory"] == {} and startup["gpu"] is None             # hold_still: no GPU numbers
    finally:
        code, err = _stop(proc)
    assert code == 0, err
    assert "FATAL" not in err


def test_broken_station_section_exits_1_with_one_fatal_line(policy_station, tmp_path):
    station = policy_station({"kind": "hold_still", "chekpoint": "x"})
    proc = subprocess.run([sys.executable, "-m", "policy_deploy.policy", "--station", str(station), "--session",
                           str(tmp_path / "s")], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    lines = proc.stderr.strip().splitlines()
    assert lines[-1] == ("FATAL policy: invalid settings: station.yaml: policy.chekpoint: not a key of "
                         "config/policy.yaml: check station.yaml policy: and config/policy.yaml")
    assert sum(line.startswith("FATAL") for line in lines) == 1


def test_kind_unset_fatal_line(policy_station, tmp_path):
    proc = subprocess.run([sys.executable, "-m", "policy_deploy.policy", "--station", str(policy_station({}))],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    assert proc.stderr.strip().splitlines()[-1] == (
        "FATAL policy: invalid settings: policy.kind is not set: choose diffusion or hold_still in station.yaml "
        "(policy.kind): check station.yaml policy: and config/policy.yaml")


def test_diffusion_without_torch_fatal_line(policy_station, tmp_path):
    ckpt = tmp_path / "b.ckpt"
    ckpt.write_bytes(b"x")
    station = policy_station({"kind": "diffusion", "checkpoint": str(ckpt)},
                             python={"light": sys.executable, "torch": sys.executable})
    script = ("import sys; sys.modules['torch'] = None\n"
              "from policy_deploy.policy.__main__ import main\nsys.exit(main(sys.argv[1:]))\n")
    proc = subprocess.run([sys.executable, "-c", script, "--station", str(station), "--session", str(tmp_path / "s")],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1
    assert proc.stderr.strip().splitlines()[-1] == (
        "FATAL policy: torch is not importable (kind diffusion): run the policy with python.torch, the torch "
        "environment (station.yaml)")


def test_without_session_writes_the_policy_alone_folder(policy_station, tmp_path):
    station = policy_station(None)
    ready = tmp_path / "alone.ready"
    proc = _start(["--station", str(station), "--ready-file", str(ready), "--mock"], tmp_path)
    try:
        _wait_for(ready.exists, proc, START_TIMEOUT_S)
    finally:
        code, err = _stop(proc)
    assert code == 0, err
    folders = list((tmp_path / "sessions").iterdir())
    assert len(folders) == 1 and folders[0].name.endswith("_policy_alone")
    assert (folders[0] / "policy" / "startup.json").exists()
