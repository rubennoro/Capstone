# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/test_mock.py @ 2ffe641
# changes: the teleop_collect ORDER; the interpreter swapped with launcher_helpers.use_python (no station python:
# section); the fake interpreter answers teleop_collect.frames.rig_file; imports. The source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/tests/test_mock.py @ 91b178b
# changes: the real process classes of processes.py instead of launch.py files; their mock checks never contact
# another machine (the check subprocesses answered by a fake interpreter).
"""--mock reaches every process; under --mock the real process classes contact nothing but this machine."""
from __future__ import annotations

import ipaddress
import socket

import pytest
from launcher_helpers import make_process, use_python, write_fake_python

from teleop_collect.contracts import Station
from teleop_collect.launcher import check_helpers
from teleop_collect.launcher.launcher import Launcher

ORDER = ["fake_pc", "frames", "teleop", "ui"]


@pytest.mark.parametrize("mock", [False, True])
def test_mock_reaches_every_command(station, mock):
    launcher = Launcher(station, mock, [make_process(name, "serve")(station, mock, None) for name in ORDER])
    for process in launcher.processes:
        assert process.mock is mock
        assert ("--mock" in process.command()) is mock


@pytest.fixture
def no_remote(monkeypatch):
    """Records every TCP connection or reachability probe to a host that is not this machine."""
    remote: list[str] = []

    def is_remote(host) -> bool:
        try:
            return not ipaddress.ip_address(socket.gethostbyname(host)).is_loopback
        except OSError:
            return True

    real_connect = socket.create_connection

    def create_connection(address, *args, **kwargs):
        if is_remote(address[0]):
            remote.append(f"{address[0]}:{address[1]}")
            raise OSError("test: no remote connections under --mock")
        return real_connect(address, *args, **kwargs)

    real_reachable = check_helpers.host_reachable

    def host_reachable(process, host, port, timeout_s):
        if is_remote(host):
            remote.append(f"{host}:{port}")
        return real_reachable(process, host, port, timeout_s)

    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(check_helpers, "host_reachable", host_reachable)
    return remote


def test_real_process_classes_under_mock(make_station, tmp_path, no_remote, link_ports, monkeypatch):
    py = write_fake_python(tmp_path / "py", {"teleop_collect.frames.rig_file": ('{"ok": false, "error": "x"}\n', 1)},
                           tmp_path / "calls.jsonl")
    use_python(monkeypatch, py)
    station = Station.load(make_station(pi={"address": "192.0.2.30", "port": 5555}), mock=True)
    launcher = Launcher(station, True)
    assert [p.NAME for p in launcher.processes] == ORDER
    for process in launcher.processes:
        assert "--mock" in process.command()
        results = process.base_checks() + process.checks()
        assert all(r.process == process.NAME for r in results)
    assert no_remote == [], f"mock checks contacted {no_remote}"
    assert "teleop_collect.teleop.pc_probe" not in (tmp_path / "calls.jsonl").read_text()
