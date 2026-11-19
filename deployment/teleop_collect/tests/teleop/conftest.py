# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/manager/conftest.py @ 2ffe641
# changes: teleop: overrides (teleop_helpers.load_settings), imports.
"""Fixtures for the teleop tests: settings from a temporary station (teleop: overrides) and a FakeClock."""
from __future__ import annotations

import pytest
from teleop_helpers import FakeClock, load_settings


@pytest.fixture
def make_settings(tmp_path):
    def make(**overrides):
        return load_settings(tmp_path, **overrides)
    return make


@pytest.fixture
def settings(make_settings):
    return make_settings()


@pytest.fixture
def clock():
    return FakeClock()
