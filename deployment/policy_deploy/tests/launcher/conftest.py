"""Launcher fixtures: temporary stations (free ports), a temporary link.yaml with free C2/C3 ports."""
from __future__ import annotations

from pathlib import Path

import pytest
from deploy_contracts import LinkSettings
from station_helpers import free_port, write_station

from policy_deploy.contracts import Station

ORDER = ["fake_pc", "frames", "policy", "manager"]
FAST_LAUNCHER = {"poll_s": 0.02, "stop_timeout_s": 2.0, "ready_timeout_s": {name: 5.0 for name in ORDER}}


@pytest.fixture
def make_station(tmp_path):
    """make_station(**sections) -> path of a valid station file in tmp_path/stations/."""
    counter = iter(range(1000))

    def make(**sections) -> Path:
        folder = tmp_path / "stations" / str(next(counter))
        folder.mkdir(parents=True)
        return write_station(folder, **sections)

    return make


@pytest.fixture
def station(make_station) -> Station:
    """A real-run station whose launcher polls fast and allows 5 s per process."""
    return Station.load(make_station(launcher=FAST_LAUNCHER), False)


@pytest.fixture
def link_ports(monkeypatch):
    """LinkSettings.load() returns the real link.yaml with free C2/C3 ports (so tests never bind 5620/5630)."""
    import dataclasses
    real = LinkSettings.load()
    ports = sorted({free_port(), free_port()} | {free_port()})[:2]
    fake = dataclasses.replace(real, state_port=ports[0], commands_port=ports[1])
    monkeypatch.setattr(LinkSettings, "load", classmethod(lambda cls, path=None: fake))
    return fake
