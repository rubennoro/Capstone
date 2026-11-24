# adapted from rnoro/deployment:deployment/deploy/tests/fakes.py @ 91b178b
# changes: ManagedProcess from policy_deploy.launcher; fake_proc.py is tests/launcher/data/fake_proc.py; adds
# write_fake_python (a stand-in interpreter that answers the --json self-check subprocesses).
"""Test-only helpers for the launcher tests: ManagedProcess subclasses that run tests/launcher/data/fake_proc.py, and
a fake interpreter for the check subprocesses."""
from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

from policy_deploy.launcher.check_result import CheckResult
from policy_deploy.launcher.managed_process import ManagedProcess

DATA = Path(__file__).parent / "launcher" / "data"
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
