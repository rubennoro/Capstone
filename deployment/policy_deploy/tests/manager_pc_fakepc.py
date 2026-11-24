"""Helpers for tests that run an in-process FakePc on tcp://127.0.0.1:* (fake_pc and the PC-facing manager tests)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from deploy_contracts import LinkSettings

from policy_deploy.contracts import load_config
from policy_deploy.fake_pc.fake_pc import FakePc
from policy_deploy.fake_pc.settings import FakePcSettings

ANY = "tcp://127.0.0.1:*"


class FakeLaptopClock:
    """A laptop CLOCK_MONOTONIC driven by the test (ns)."""

    def __init__(self, start_ns: int = 1_000_000_000_000) -> None:
        self.now = start_ns

    def __call__(self) -> int:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += int(round(seconds * 1e9))


def fake_pc_settings(tmp_path: Path, link: LinkSettings | None = None, **overrides) -> FakePcSettings:
    config = load_config("fake_pc", None)
    config.update(overrides)
    return FakePcSettings.from_config(config, Path(tmp_path), link=link, state_endpoint=ANY, commands_endpoint=ANY)


def make_fake_pc(tmp_path: Path, link: LinkSettings | None = None, laptop_clock: Callable[[], int] | None = None,
                 session_dir: Path | None = None, **overrides) -> FakePc:
    settings = fake_pc_settings(tmp_path, link=link, **overrides)
    kw = {} if laptop_clock is None else {"laptop_clock": laptop_clock}
    return FakePc(settings, session_dir=session_dir, commit="0123456789abcdef0123456789abcdef01234567", **kw)


def wait_for(condition: Callable[[], bool], timeout_s: float = 5.0, poll_s: float = 0.005) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(poll_s)
    return condition()
