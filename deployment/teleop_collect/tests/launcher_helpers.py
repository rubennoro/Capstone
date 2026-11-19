# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher_fakes.py and
# tests/manager_pc_fakepc.py @ 2ffe641
# changes: one helper module for E6's tests (STRUCTURE.md 2): the launcher fakes (ManagedProcess from
# teleop_collect.launcher; no USES_TERMINAL, no process uses the terminal) and the in-process fake PC helpers
# (teleop_collect.contracts, teleop_collect.fake_pc); fake_proc.py is in tests/launcher/fake_bin/ (the repo's
# .gitignore ignores every data/ folder); + use_python (the interpreter of every process is
# sys.executable, so a test swaps it here instead of a station python: section).
"""Test-only helpers for the launcher and fake PC tests: ManagedProcess subclasses that run
tests/launcher/fake_bin/fake_proc.py, a fake interpreter for the check subprocesses, and an in-process FakePc on
tcp://127.0.0.1:*."""
from __future__ import annotations

import json
import stat
import sys
import time
from pathlib import Path
from typing import Callable

from deploy_contracts import LinkSettings

from teleop_collect.contracts import load_config
from teleop_collect.fake_pc.fake_pc import FakePc
from teleop_collect.fake_pc.settings import FakePcSettings
from teleop_collect.launcher.check_result import CheckResult
from teleop_collect.launcher.managed_process import ManagedProcess

FAKE_BIN = Path(__file__).parent / "launcher" / "fake_bin"
FAKE_PROC = FAKE_BIN / "fake_proc.py"
ANY = "tcp://127.0.0.1:*"


# ------------------------------------------------------------------------------------------------ launcher fakes

def make_process(name: str, mode: str, *args: str, checks: tuple[CheckResult, ...] = (),
                 events: list | None = None) -> type[ManagedProcess]:
    """A subclass named `name` that runs fake_proc.py in `mode`; start/ready/stop are recorded in `events`."""
    log = events if events is not None else []

    class FakeProcess(ManagedProcess):
        NAME = name
        READY_HINT = f"the {name} hint; check its fake"

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


def check_line(name: str, status: str, detail: str) -> str:
    return json.dumps({"name": name, "status": status, "detail": detail})


def write_fake_python(path: Path, answers: dict[str, tuple[str, int]], calls: Path | None = None) -> Path:
    """An executable standing in for a Python. `python -m <module> ...`: if the module is in `answers`, prints its
    text and exits with its code; anything else (e.g. `-c` for the import check) runs the real interpreter. Every
    argument list is appended to `calls` as one JSON line."""
    body = f"""#!{sys.executable}
import json, os, sys
ANSWERS = {repr({k: list(v) for k, v in answers.items()})}
CALLS = {repr(str(calls) if calls else None)}
if CALLS:
    with open(CALLS, "a") as f:
        f.write(json.dumps(sys.argv[1:]) + "\\n")
if len(sys.argv) > 2 and sys.argv[1] == "-m" and sys.argv[2] in ANSWERS:
    text, code = ANSWERS[sys.argv[2]]
    sys.stdout.write(text)
    sys.exit(code)
os.execv({json.dumps(sys.executable)}, [{json.dumps(sys.executable)}] + sys.argv[1:])
"""
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def use_python(monkeypatch, python: str | Path) -> None:
    """Every ManagedProcess runs (and checks with) this interpreter instead of sys.executable."""
    monkeypatch.setattr(ManagedProcess, "python", lambda self: str(python))


# ------------------------------------------------------------------------------------------------ in-process fake PC

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
