# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/conftest.py @ ea9f7fb
# changes: imports (teleop_collect.frames); the running fixture uses frames_helpers.Running (R1 and U2 clients).
"""Fixtures for the frames tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from frames_helpers import SMALL_RIG, LIVE_MAIN_SIZE, LIVE_RIG, LIVE_UW_SIZE, FakeClock, Running, wait_for, write_pi_episode

from teleop_collect.frames.rig_file import load_live_rig


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
def live_rig():
    return load_live_rig(LIVE_RIG)


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
    return write_pi_episode(tmp_path_factory.mktemp("pi") / "ep_v2", n_frames=15, main_size=LIVE_MAIN_SIZE,
                            uw_size=LIVE_UW_SIZE)


@pytest.fixture
def make_pi_episode(tmp_path) -> Callable[..., Path]:
    counter = iter(range(1000))

    def make(**kw) -> Path:
        return write_pi_episode(tmp_path / f"ep_{next(counter):04d}", **kw)
    return make


@pytest.fixture
def running(tmp_path, pi_episode):
    runs = []

    def start(episode=pi_episode, **overrides):
        r = Running(tmp_path / f"run{len(runs)}", episode, **overrides)
        runs.append(r)
        return r
    yield start
    for r in runs:
        r.stop()
