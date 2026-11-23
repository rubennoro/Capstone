# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/conftest.py @ 2ffe641
# changes: imports teleop_collect.contracts; the helper module names of this package's tests; + the autouse fixture
# keeping the repo's storage.local.yaml out of every test (2026-11-22).
"""Shared pytest setup for teleop_collect's tests (STRUCTURE.md 2, 12).

Helper modules shared across test folders live in tests/ (on the pytest pythonpath) with a unique name
(station_helpers.py, <part>_helpers.py), because test folders have no __init__.py.
"""
from __future__ import annotations

import pytest

import teleop_collect.contracts  # noqa: F401  (registers the laptop schemas, as every process does)


@pytest.fixture
def station_file(tmp_path):
    """A valid temporary station file (free ports, session_root and data_root under tmp_path); returns its path."""
    from station_helpers import write_station
    return write_station(tmp_path)


@pytest.fixture(autouse=True)
def no_repo_storage_file(tmp_path_factory, monkeypatch):
    """The repo's own storage.local.yaml (the operator's remembered storage folder) is never read or written by a test:
    for the default station file, storage_file points into a fresh temporary folder instead."""
    from teleop_collect.contracts import paths, station
    real = station.storage_file
    default = paths.default_station().parent
    folder = tmp_path_factory.mktemp("storage")

    def storage_file(station_path):
        path = real(station_path)
        return folder / path.name if path.parent == default else path
    monkeypatch.setattr(station, "storage_file", storage_file)
