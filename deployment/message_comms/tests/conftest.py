"""Fixtures: settings for tests (a station file in tmp_path) and FakeClock."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fakes import FakeClock
from message_comms.settings import load_settings

ENDPOINTS = {
    "frames": "tcp://127.0.0.1:59610", "state": "tcp://127.0.0.1:59620",
    "commands": "tcp://127.0.0.1:59630", "policy": "tcp://127.0.0.1:59640",
}


def write_station(folder: Path, overrides: dict | None = None, endpoints: dict | None = None,
                  extra: dict | None = None) -> Path:
    doc = {"station_version": 1, "session_root": str(folder / "sessions"), "endpoints": dict(endpoints or ENDPOINTS)}
    if overrides:
        doc["message_comms"] = overrides
    doc.update(extra or {})
    path = folder / "station.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


@pytest.fixture
def make_settings(tmp_path):
    def make(**overrides):
        station = write_station(tmp_path, overrides)
        return load_settings(station, tmp_path / "session", mock=False)
    return make


@pytest.fixture
def settings(make_settings):
    return make_settings()


@pytest.fixture
def clock():
    return FakeClock()
