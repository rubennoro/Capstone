"""policy/launch.py (PolicyProcess), as the launcher uses it. Needs the `deploy` package."""

from __future__ import annotations

import importlib.util
import json
import socket
import stat
import sys
from pathlib import Path

import pytest
import yaml

deploy = pytest.importorskip("deploy", reason="launch.py tests need the deploy package (pip install -e deploy)")
from deploy import Station  # noqa: E402

LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch.py"


def _process_class():
    spec = importlib.util.spec_from_file_location("launch_policy_test", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROCESS_CLASS


def _station(tmp_path: Path, policy: dict | None = None, light: str = sys.executable,
             policy_python: str | None = "/nonexistent/policy/python", policy_port: int | None = None) -> Path:
    port = policy_port or _free_port()
    data = {
        "station_version": 1,
        "session_root": str(tmp_path / "sessions"),
        "python": {"light": light},
        "endpoints": {
            "pi_frames": "tcp://127.0.0.1:15600", "pi_time_sync": "tcp://127.0.0.1:15601",
            "mock_pi_frames": "tcp://127.0.0.1:15602", "mock_pi_time_sync": "tcp://127.0.0.1:15603",
            "frames": "tcp://127.0.0.1:15610", "state": "tcp://127.0.0.1:15620",
            "commands": "tcp://127.0.0.1:15630", "policy": f"tcp://127.0.0.1:{port}",
        },
        "policy": policy or {},
    }
    if policy_python is not None:
        data["python"]["policy"] = policy_python
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fake_python(tmp_path: Path, body: str) -> str:
    """An executable that stands in for a Python running policy.self_check."""
    path = tmp_path / "fake_python"
    path.write_text(f"#!{sys.executable}\nimport sys, time\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _make(tmp_path: Path, mock: bool, **station_kwargs):
    station_path = _station(tmp_path, **station_kwargs)
    return _process_class()(Station.load(station_path, mock), mock, None)


def test_program_uses_light_with_mock_and_policy_otherwise(tmp_path):
    p = _make(tmp_path, mock=True, policy={"kind": "hold_still"})
    assert p.program() == [sys.executable, "-m", "policy"]
    p = _make(tmp_path, mock=False, policy={"kind": "hold_still"})
    assert p.program() == ["/nonexistent/policy/python", "-m", "policy"]
    assert p.NAME == "policy" and "checkpoint" in p.READY_HINT


def test_unset_kind_fails_and_skips_the_self_check(tmp_path):
    python = _fake_python(tmp_path, "open(sys.argv[0] + '.ran', 'w').close()")
    results = _make(tmp_path, mock=False, policy_python=python).checks()
    kind = [r for r in results if r.name == "kind"]
    assert len(kind) == 1 and kind[0].status == "fail" and "policy.kind is not set" in kind[0].detail
    assert not Path(python + ".ran").exists()
    assert all(r.name != "self_check" for r in results)


def test_busy_port_fails(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        results = _make(tmp_path, mock=True, policy_port=s.getsockname()[1]).checks()
    assert [r.status for r in results if r.status == "fail"] == ["fail"]          # only the port check


def test_self_check_lines_become_check_results(tmp_path):
    lines = [{"name": "settings", "status": "pass", "detail": "ok"},
             {"name": "device", "status": "info", "detail": "device: cpu (PLAN §9 backup)"}]
    body = "\n".join(f"print({json.dumps(json.dumps(item))})" for item in lines)
    python = _fake_python(tmp_path, body)
    results = _make(tmp_path, mock=False, policy={"kind": "diffusion"}, policy_python=python).checks()
    got = [(r.process, r.name, r.status, r.detail) for r in results if r.name in ("settings", "device")]
    assert got == [("policy", "settings", "pass", "ok"), ("policy", "device", "info", "device: cpu (PLAN §9 backup)")]


def test_self_check_crash_is_one_fail_with_its_stderr(tmp_path):
    python = _fake_python(tmp_path, "sys.stderr.write('No module named policy\\n'); sys.exit(1)")
    results = _make(tmp_path, mock=False, policy={"kind": "hold_still"}, policy_python=python).checks()
    fails = [r for r in results if r.name == "self_check"]
    assert len(fails) == 1 and fails[0].status == "fail" and "No module named policy" in fails[0].detail


def test_self_check_garbage_line_is_one_fail(tmp_path):
    python = _fake_python(tmp_path, "print('hello')")
    results = _make(tmp_path, mock=False, policy={"kind": "hold_still"}, policy_python=python).checks()
    fails = [r for r in results if r.name == "self_check"]
    assert len(fails) == 1 and "not a check" in fails[0].detail


def test_self_check_timeout_is_one_fail(tmp_path):
    python = _fake_python(tmp_path, "sys.stderr.write('importing torch\\n'); sys.stderr.flush(); time.sleep(30)")
    results = _make(tmp_path, mock=False, policy={"kind": "hold_still", "self_check_timeout_s": 0.5},
                    policy_python=python).checks()
    fails = [r for r in results if r.name == "self_check"]
    assert len(fails) == 1 and "timed out" in fails[0].detail


def test_mock_checks_need_no_checkpoint_gpu_or_policy_python(tmp_path):
    results = _make(tmp_path, mock=True, policy={"kind": "diffusion"}, policy_python=None).checks()
    assert [r.status for r in results if r.status == "fail"] == []
    assert {r.name for r in results} >= {"settings", "kind"}
