from __future__ import annotations

import socket
import sys

from deploy import check_helpers


def test_file_exists(tmp_path):
    (tmp_path / "f").write_text("x")
    ok = check_helpers.file_exists("policy", "checkpoint", tmp_path / "f")
    assert (ok.process, ok.name, ok.status) == ("policy", "checkpoint", "pass")
    bad = check_helpers.file_exists("policy", "checkpoint", tmp_path / "missing")
    assert bad.status == "fail" and str(tmp_path / "missing") in bad.detail


def test_executable(tmp_path):
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\n")
    assert check_helpers.executable("franka_control", "binary", script).status == "fail"     # not executable
    script.chmod(0o755)
    assert check_helpers.executable("franka_control", "binary", script).status == "pass"
    assert check_helpers.executable("franka_control", "binary", tmp_path / "nope").detail.startswith("not found")
    assert check_helpers.executable("franka_control", "binary", tmp_path).status == "fail"      # a folder


def test_port_free_and_busy():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = check_helpers.port_free("policy", f"tcp://127.0.0.1:{port}")
        assert result.status == "fail" and str(port) in result.name
    assert check_helpers.port_free("policy", f"tcp://127.0.0.1:{port}").status == "pass"
    assert check_helpers.port_free("policy", "localhost:1").status == "fail"


def test_python_imports():
    ok = check_helpers.python_imports("policy", sys.executable, ["json", "yaml"], 20.0)
    assert ok.status == "pass", ok.detail
    bad = check_helpers.python_imports("policy", sys.executable, ["json", "no_such_module_xyz"], 20.0)
    assert bad.status == "fail" and "no_such_module_xyz" in bad.detail and "json (" not in bad.detail
    missing = check_helpers.python_imports("policy", "/no/such/python", ["json"], 20.0)
    assert missing.status == "fail" and "/no/such/python" in missing.detail


def test_python_imports_ignores_the_working_folder(tmp_path, monkeypatch):
    """From deployment/, a folder named like a package must not count as that package (namespace shadowing)."""
    (tmp_path / "only_a_folder_xyz").mkdir()
    monkeypatch.chdir(tmp_path)
    result = check_helpers.python_imports("policy", sys.executable, ["only_a_folder_xyz"], 20.0)
    assert result.status == "fail" and "only_a_folder_xyz" in result.detail


def test_host_reachable():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert check_helpers.host_reachable("franka_control", "127.0.0.1", port, 2.0).status == "pass"
    unreachable = check_helpers.host_reachable("franka_control", "127.0.0.1", port, 2.0)    # nothing listens now
    assert unreachable.status == "fail" and f"127.0.0.1:{port}" in unreachable.detail


def test_info_and_warn():
    assert check_helpers.info("policy", "device", "cpu (backup)").status == "info"
    assert check_helpers.warn("process_frames", "rig stand-in", "stand-in").status == "warn"
