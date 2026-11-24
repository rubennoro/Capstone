"""Shared pytest setup for policy_deploy's tests (STRUCTURE.md 12).

Helper modules shared across test folders live in tests/ (on the pytest pythonpath) with a unique name
(station_helpers.py, contract_helpers.py, <subpackage>_*.py), because test folders have no __init__.py.
"""
from __future__ import annotations

import pytest

import policy_deploy.contracts  # noqa: F401  (registers the laptop schemas, as every process does)


@pytest.fixture
def station_file(tmp_path):
    """A valid temporary station file (free ports, session_root under tmp_path); returns its path."""
    from station_helpers import write_station
    return write_station(tmp_path)
