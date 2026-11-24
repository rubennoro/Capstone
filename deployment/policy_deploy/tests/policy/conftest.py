# adapted from rnoro/deployment:deployment/policy/tests/conftest.py @ 91b178b
# changes: the shape_meta dicts live in tests/policy_fakes.py; policy_station writes the new station schema (through
# tests/station_helpers.py: free ports, session_root under tmp_path).
"""Fixtures of the policy tests: shape_meta dicts, a fake clock, station files."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import pytest
from policy_fakes import RUN_001_SHAPE_META, FakeClock, plain_diffusion_shape_meta
from station_helpers import write_station


@pytest.fixture
def run_001_shape_meta() -> dict[str, Any]:
    return copy.deepcopy(RUN_001_SHAPE_META)


@pytest.fixture
def plain_shape_meta() -> dict[str, Any]:
    return plain_diffusion_shape_meta()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def policy_station(tmp_path: Path) -> Callable[..., Path]:
    """policy_station(policy_section, **other_sections) -> a station file in tmp_path. policy_section None means
    an empty policy: section (kind unset)."""
    def make(policy_section: dict[str, Any] | None = None, **sections: Any) -> Path:
        return write_station(tmp_path, policy=policy_section or {}, **sections)
    return make
