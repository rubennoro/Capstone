# adapted from rnoro/deployment:deployment/deploy/tests/test_mock.py @ 91b178b
# changes: the real process classes of processes.py instead of launch.py files; their mock checks never contact
# another machine (the check subprocesses answered by a fake interpreter).
"""--mock reaches every process; under --mock the real process classes contact nothing but this machine."""
from __future__ import annotations

import ipaddress
import socket

import pytest
from launcher_fakes import check_line, make_process, write_fake_python

from policy_deploy.contracts import Station
from policy_deploy.launcher import check_helpers
from policy_deploy.launcher.launcher import Launcher

ORDER = ["fake_pc", "frames", "policy", "manager"]


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


def test_real_process_classes_under_mock(make_station, tmp_path, no_remote, link_ports):
    line = check_line("settings", "pass", "ok") + "\n"
    py = write_fake_python(tmp_path / "py", {"policy_deploy.frames.rig_file": ('{"ok": false, "error": "x"}\n', 1),
                                             "policy_deploy.policy.self_check": (line, 0),
                                             "policy_deploy.manager.self_check": (line, 0)})
    station = Station.load(make_station(python={"light": str(py)}), mock=True)
    launcher = Launcher(station, True)
    assert [p.NAME for p in launcher.processes] == ORDER
    for process in launcher.processes:
        assert "--mock" in process.command()
        results = process.base_checks() + process.checks()
        assert all(r.process == process.NAME for r in results)
    assert no_remote == [], f"mock checks contacted {no_remote}"
