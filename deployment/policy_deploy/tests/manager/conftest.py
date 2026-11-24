"""Fixtures for the manager tests: settings from a temporary station (manager: overrides) and a FakeClock."""
from __future__ import annotations

import pytest
from manager_fakes import FakeClock
from manager_harness import load_settings


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
