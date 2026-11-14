# adapted from rnoro/deployment:deployment/deploy/tests/test_check_helpers.py @ 91b178b
# changes: imports, laptop process names; + parse_endpoint, port label, address_is_local, json_check_lines,
# rig_file_check.
from __future__ import annotations

import socket
import sys

import pytest

from launcher_fakes import check_line, write_fake_python

from policy_deploy.launcher import check_helpers


def test_file_exists(tmp_path):
    (tmp_path / "f").write_text("x")
    ok = check_helpers.file_exists("policy", "checkpoint", tmp_path / "f")
    assert (ok.process, ok.name, ok.status) == ("policy", "checkpoint", "pass")
    bad = check_helpers.file_exists("policy", "checkpoint", tmp_path / "missing")
    assert bad.status == "fail" and str(tmp_path / "missing") in bad.detail


def test_executable(tmp_path):
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\n")
    assert check_helpers.executable("frames", "binary", script).status == "fail"     # not executable
    script.chmod(0o755)
    assert check_helpers.executable("frames", "binary", script).status == "pass"
    assert check_helpers.executable("frames", "binary", tmp_path / "nope").detail.startswith("not found")
    assert check_helpers.executable("frames", "binary", tmp_path).status == "fail"      # a folder


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
        assert check_helpers.host_reachable("frames", "127.0.0.1", port, 2.0).status == "pass"
    unreachable = check_helpers.host_reachable("frames", "127.0.0.1", port, 2.0)    # nothing listens now
    assert unreachable.status == "fail" and f"127.0.0.1:{port}" in unreachable.detail


def test_info_and_warn():
    assert check_helpers.info("policy", "device", "cpu (backup)").status == "info"
    assert check_helpers.warn("frames", "decoder", "jpeg (backup)").status == "warn"


def test_parse_endpoint():
    assert check_helpers.parse_endpoint("tcp://127.0.0.1:5610") == ("127.0.0.1", 5610)
    for bad in ("127.0.0.1:5610", "tcp://h:0", "tcp://h:70000", None):
        with pytest.raises(ValueError):
            check_helpers.parse_endpoint(bad)


def test_port_free_label():
    assert check_helpers.port_free("frames", "tcp://127.0.0.1:1", "C1 port").name == "C1 port"


def test_address_is_local():
    assert check_helpers.address_is_local("127.0.0.1")
    assert not check_helpers.address_is_local("192.0.2.123")          # TEST-NET-1: never on a real interface


def test_json_check_lines_rows(tmp_path):
    out = check_line("settings", "pass", "ok") + "\n\n" + check_line("gpu", "warn", "busy") + "\n"
    py = write_fake_python(tmp_path / "py", {"some.check": (out, 0)})
    rows = check_helpers.json_check_lines("policy", [str(py), "-m", "some.check"], 20.0)
    assert [(r.process, r.name, r.status, r.detail) for r in rows] == [("policy", "settings", "pass", "ok"),
                                                                       ("policy", "gpu", "warn", "busy")]


@pytest.mark.parametrize("text,code,why", [
    ("not json\n", 0, "printed a line that is not a check"),
    ('{"name": "a", "status": "ok", "detail": "x"}\n', 0, "printed a line that is not a check"),
    ("", 0, "printed no checks"),
    ("boom\n", 3, "exited with code 3"),
])
def test_json_check_lines_failures(tmp_path, text, code, why):
    py = write_fake_python(tmp_path / "py", {"some.check": (text, code)})
    [row] = check_helpers.json_check_lines("manager", [str(py), "-m", "some.check"], 20.0, label="PC", hint="h")
    assert (row.process, row.name, row.status) == ("manager", "PC", "fail")
    assert why in row.detail and row.detail.endswith("(h)")


def test_json_check_lines_missing_python():
    [row] = check_helpers.json_check_lines("policy", ["/no/such/python", "-m", "x"], 5.0)
    assert row.status == "fail" and "could not run /no/such/python" in row.detail


def test_json_check_lines_timeout(tmp_path):
    py = tmp_path / "slow"
    py.write_text(f"#!{sys.executable}\nimport time; time.sleep(5)\n")
    py.chmod(0o755)
    [row] = check_helpers.json_check_lines("policy", [str(py), "-m", "x"], 0.3)
    assert row.status == "fail" and "timed out after 0.3 s" in row.detail


def test_rig_file_check(tmp_path):
    rig = tmp_path / "rig.yaml"
    rig.write_text("x")
    summary = '{"ok": true, "rig_id": "rig_pi_v2_640", "cameras": {"main": {"sensor": "imx519", "width": 640, ' \
              '"height": 480, "fps": 30.0}, "ultrawide": {"sensor": "imx219", "width": 640, "height": 480, ' \
              '"fps": 30.0}}}\n'
    py = write_fake_python(tmp_path / "py", {"policy_deploy.frames.rig_file": (summary, 0)})
    row = check_helpers.rig_file_check("frames", str(py), rig, 20.0)
    assert row.status == "pass"
    assert row.detail == "rig_pi_v2_640: main imx519 640x480@30, ultrawide imx219 640x480@30"
    bad = write_fake_python(tmp_path / "bad", {"policy_deploy.frames.rig_file":
                                               ('{"ok": false, "error": "cameras.main.sensor: bad"}\n', 1)})
    assert check_helpers.rig_file_check("frames", str(bad), rig, 20.0).detail == "cameras.main.sensor: bad"
    crash = write_fake_python(tmp_path / "crash", {"policy_deploy.frames.rig_file": ("", 1)})
    assert "failed" in check_helpers.rig_file_check("frames", str(crash), rig, 20.0).detail
    missing = check_helpers.rig_file_check("frames", str(py), tmp_path / "nope.yaml", 20.0)
    assert missing.status == "fail" and "not found" in missing.detail
