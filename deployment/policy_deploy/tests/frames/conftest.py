# adapted from rnoro/deployment:deployment/process_frames/tests/conftest.py @ 91b178b (the fixtures)
# changes: the helpers moved to tests/frames_helpers.py (importable); the v2 rig is rig_pi_v2_640 (640x480 both).
"""Fixtures for the frames tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from frames_helpers import SMALL_RIG, V2_MAIN_SIZE, V2_RIG, V2_UW_SIZE, FakeClock, wait_for, write_pi_episode

from policy_deploy.frames.rig_file import load_live_rig


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def waiter():
    return wait_for


@pytest.fixture(scope="session")
def small_rig():
    return load_live_rig(SMALL_RIG)


@pytest.fixture(scope="session")
def v2_rig():
    return load_live_rig(V2_RIG)


@pytest.fixture(scope="session")
def pi_episode(tmp_path_factory) -> Path:
    """30 frames per camera (1 s), with the exposure_ns column."""
    return write_pi_episode(tmp_path_factory.mktemp("pi") / "ep_0001")


@pytest.fixture(scope="session")
def pi_episode_no_exposure(tmp_path_factory) -> Path:
    return write_pi_episode(tmp_path_factory.mktemp("pi") / "ep_0002", n_frames=12, with_exposure=False)


@pytest.fixture(scope="session")
def v2_episode(tmp_path_factory) -> Path:
    """15 frames per camera at the real rig file's sizes (both 640x480)."""
    return write_pi_episode(tmp_path_factory.mktemp("pi") / "ep_v2", n_frames=15, main_size=V2_MAIN_SIZE,
                            uw_size=V2_UW_SIZE)


@pytest.fixture
def make_pi_episode(tmp_path) -> Callable[..., Path]:
    counter = iter(range(1000))

    def make(**kw) -> Path:
        return write_pi_episode(tmp_path / f"ep_{next(counter):04d}", **kw)
    return make


@pytest.fixture
def running(tmp_path, pi_episode):
    from frames_pipeline import Running
    runs = []

    def start(episode=pi_episode, **overrides):
        r = Running(tmp_path / f"run{len(runs)}", episode, **overrides)
        runs.append(r)
        return r
    yield start
    for r in runs:
        r.stop()
