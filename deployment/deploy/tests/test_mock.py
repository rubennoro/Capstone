"""--mock reaches every process; the real launch.py files (once they exist) load and choose their stand-ins."""
from __future__ import annotations

import ipaddress
import socket

import pytest

from deploy import Launcher, Station, check_helpers, load_process_class
from fakes import make_process
from helpers import REAL_ROOT

FOLDERS = ["franka_control", "process_frames", "policy", "message_comms"]


@pytest.mark.parametrize("mock", [False, True])
def test_mock_reaches_every_command(fake_root, mock):
    station = Station.load(fake_root / "station.yaml", mock)
    launcher = Launcher(station, mock, [make_process(name, "serve")(station, mock, None) for name in FOLDERS])
    for process in launcher.processes:
        assert process.mock is mock
        assert ("--mock" in process.command()) is mock


@pytest.fixture
def no_pi_or_arm(monkeypatch):
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


@pytest.mark.parametrize("folder", FOLDERS)
def test_real_launch_py_under_mock(folder, no_pi_or_arm):
    if not (REAL_ROOT / folder / "launch.py").exists():
        pytest.skip(f"{folder}/launch.py does not exist yet")
    cls = load_process_class(REAL_ROOT / folder)
    assert cls.NAME == folder
    station = Station.load(REAL_ROOT / "station.yaml", mock=True)
    process = cls(station, True, None)
    assert "--mock" in process.command()
    results = process.base_checks() + process.checks()
    assert all(r.process == folder for r in results)
    assert no_pi_or_arm == [], f"{folder}'s mock checks contacted {no_pi_or_arm}"
