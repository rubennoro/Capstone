# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/conftest.py @ 2ffe641
# changes: the teleop_collect ORDER (fake_pc, frames, teleop, ui); make_station builds on this package's
# write_station(tmp_path, sections=, ports=) (a `ports` keyword updates the free ports, every other keyword replaces a
# top-level key); + browser (no test opens a real browser); imports.
"""Launcher fixtures: temporary stations (free ports), a temporary link.yaml with free C2/C3 ports."""
from __future__ import annotations

from pathlib import Path

import pytest
from deploy_contracts import LinkSettings
from station_helpers import free_ports, write_station

from teleop_collect.contracts import Station

ORDER = ["fake_pc", "frames", "teleop", "ui"]
FAST_LAUNCHER = {"poll_s": 0.02, "stop_timeout_s": 2.0, "ready_timeout_s": {name: 5.0 for name in ORDER}}


@pytest.fixture(autouse=True)
def browser(monkeypatch):
    """No test opens a real browser: webbrowser.open only records the URL (and answers True), and there is a
    display. Returns the list of opened URLs."""
    import teleop_collect.launcher.launcher as launcher_module
    opened: list[str] = []
    monkeypatch.setattr(launcher_module.webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)
    monkeypatch.setattr(launcher_module, "_has_display", lambda: True)
    return opened


@pytest.fixture
def make_station(tmp_path):
    """make_station(**sections) -> path of a valid station file in tmp_path/stations/<n>/."""
    counter = iter(range(1000))

    def make(ports: dict | None = None, **sections) -> Path:
        folder = tmp_path / "stations" / str(next(counter))
        folder.mkdir(parents=True)
        return write_station(folder, sections=sections, ports=ports)

    return make


@pytest.fixture
def station(make_station) -> Station:
    """A real-run station whose launcher polls fast and allows 5 s per process."""
    return Station.load(make_station(launcher=FAST_LAUNCHER), False)


@pytest.fixture
def link_ports(monkeypatch):
    """LinkSettings.load() returns the real link.yaml with free C2/C3 ports (so tests never bind the real ones)."""
    import dataclasses
    real = LinkSettings.load()
    ports = free_ports(2)
    fake = dataclasses.replace(real, state_port=ports[0], commands_port=ports[1])
    monkeypatch.setattr(LinkSettings, "load", classmethod(lambda cls, path=None: fake))
    return fake
