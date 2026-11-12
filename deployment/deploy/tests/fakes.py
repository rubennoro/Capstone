"""Test-only ManagedProcess subclasses that run tests/data/fake_proc.py."""
from __future__ import annotations

import sys
from pathlib import Path

from deploy import CheckResult, ManagedProcess

DATA = Path(__file__).parent / "data"
FAKE_PROC = DATA / "fake_proc.py"


def make_process(name: str, mode: str, *args: str, terminal: bool = False, checks: tuple[CheckResult, ...] = (),
                 events: list | None = None) -> type[ManagedProcess]:
    """A subclass named `name` that runs fake_proc.py in `mode`; start/ready/stop are recorded in `events`."""
    log = events if events is not None else []

    class FakeProcess(ManagedProcess):
        NAME = name
        READY_HINT = f"the {name} hint; check its fake"
        USES_TERMINAL = terminal

        def program(self) -> list[str]:
            return [sys.executable, str(FAKE_PROC), mode, *args]

        def checks(self) -> list[CheckResult]:
            return list(checks)

        def start(self) -> None:
            log.append(("start", name))
            super().start()

        def wait_ready(self, timeout_s: float, poll_s: float) -> None:
            super().wait_ready(timeout_s, poll_s)
            log.append(("ready", name))

        def stop(self, timeout_s: float) -> int | None:
            log.append(("stop", name))
            return super().stop(timeout_s)

    return FakeProcess
