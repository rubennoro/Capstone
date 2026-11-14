"""FrankaControlBinary with a fake binary script (control_deploy/STRUCTURE.md 9.2)."""
from __future__ import annotations

import pytest
import yaml
from helpers import write_fake_binary

from control_deploy.binary import BinaryError, FrankaControlBinary


def test_version_is_parsed(tmp_path, monkeypatch):
    b = FrankaControlBinary(write_fake_binary(tmp_path, commit="a" * 40))
    v = b.version()
    assert (v.commit, v.dirty, v.libfranka) == ("a" * 40, False, "0.9.2")
    monkeypatch.setenv("FAKE_VERSION", "franka_control commit=unknown dirty=yes libfranka=none")
    v = b.version()
    assert (v.commit, v.dirty, v.libfranka) == ("unknown", True, "none")


def test_an_unexpected_version_line_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VERSION", "franka_control abc123 libfranka 0.9.2")   # the 91b178b format
    with pytest.raises(BinaryError, match="unexpected --version output"):
        FrankaControlBinary(write_fake_binary(tmp_path)).version()


def test_check_config_yaml_is_parsed(tmp_path, monkeypatch):
    out = tmp_path / "cfg.yaml"
    out.write_text(yaml.safe_dump({
        "contracts": {"contracts_dir": "/x/contracts", "limits_file": "/x/contracts/limits.yaml",
                      "local": True, "watchdog_limit_s": 1.0},
        "franka_control": {"robot_ip": "172.16.0.2", "realtime": "ignore", "load": {"mass_kg": 0.4}}}))
    monkeypatch.setenv("FAKE_CHECK_CONFIG", str(out))
    r = FrankaControlBinary(write_fake_binary(tmp_path)).check_config(tmp_path / "s.yaml", True, True)
    assert r.ok
    assert r.contracts()["contracts_dir"] == "/x/contracts"
    assert r.effective("load", "mass_kg") == 0.4
    assert r.effective("realtime") == "ignore"
    assert r.effective("nope", default=3) == 3


def test_check_config_problems(tmp_path, monkeypatch):
    out = tmp_path / "p.txt"
    out.write_text("link.yaml: pc.address: expected an IPv4 address, got \"pc\"\nstation.yaml: x: bad\n")
    monkeypatch.setenv("FAKE_CHECK_CONFIG", str(out))
    monkeypatch.setenv("FAKE_CHECK_RC", "2")
    r = FrankaControlBinary(write_fake_binary(tmp_path)).check_config(tmp_path / "s.yaml", False, False)
    assert not r.ok and len(r.problems) == 2 and r.problems[0].startswith("link.yaml: pc.address")


def test_a_hang_is_caught_by_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_HANG", "1")
    with pytest.raises(BinaryError, match="did not answer within 0.5 s"):
        FrankaControlBinary(write_fake_binary(tmp_path), timeout_s=0.5).check_config(tmp_path / "s", True, True)


def test_a_missing_binary_is_an_error(tmp_path):
    with pytest.raises(BinaryError, match="cannot run"):
        FrankaControlBinary(tmp_path / "nope").version()


def test_the_run_command():
    b = FrankaControlBinary("/b/franka_control")
    assert b.command("/s.yaml", "/sess", "/sess/ready", mock=True, local=True) == [
        "/b/franka_control", "--station", "/s.yaml", "--session", "/sess", "--ready-file", "/sess/ready", "--keyboard",
        "--mock", "--local"]
    assert "--mock" not in b.command("/s", "/d", "/r", mock=False, local=False)
