# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/conftest.py @ 2ffe641
# changes: imports teleop_collect.contracts; the helper module names of this package's tests.
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
