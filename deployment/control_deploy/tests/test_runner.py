"""Runner (`control_deploy run`) with a fake binary (control_deploy/STRUCTURE.md 6.4, 9.2)."""
from __future__ import annotations

import io
import os
import pty
import signal
import termios
import threading

import pytest
from helpers import write_fake_binary, write_station

from control_deploy import runner as runner_mod
from control_deploy.check_result import CheckResult
from control_deploy.runner import READY_HINT, Runner


class StubChecks:
    rows = [CheckResult("control_deploy", "env", "pass", "ok")]

    def __init__(self, station, mock, local) -> None:
        self.version = None

    def run(self):
        return list(self.rows)


class FailingChecks(StubChecks):
    rows = [CheckResult("control_deploy", "binary", "fail", "not found: build it")]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    fake = write_fake_binary(tmp_path)
    station = write_station(tmp_path, franka_control={"binary": str(fake)})
    monkeypatch.setenv("FAKE_ARGS_FILE", str(tmp_path / "args.txt"))
    monkeypatch.setenv("FAKE_RUN_S", "0.5")
    return tmp_path, station


def make(station, out, **kw) -> Runner:
    return Runner(station, True, True, checks=kw.pop("checks", StubChecks), out=out, status_line=False, **kw)


def session_dir(tmp_path):
    (folder,) = list((tmp_path / "sessions").iterdir())
    return folder


def later(seconds: float, sig: int = signal.SIGTERM) -> threading.Timer:
    t = threading.Timer(seconds, lambda: os.kill(os.getpid(), sig))
    t.start()
    return t


def test_the_exact_arguments_and_a_clean_exit(setup):
    tmp_path, station = setup
    out = io.StringIO()
    assert make(station, out).run() == 0
    args = (tmp_path / "args.txt").read_text().split("\n")
    folder = session_dir(tmp_path)
    assert folder.name.endswith("_control_deploy_mock")
    assert args == ["--station", str(station.resolve()), "--session", str(folder), "--ready-file", str(folder / "ready"),
                    "--keyboard", "--mock", "--local"]
    assert "stopped cleanly" in out.getvalue()


def test_log_lines_reach_the_log_file_and_the_terminal_and_the_ready_banner(setup):
    tmp_path, station = setup
    out = io.StringIO()
    make(station, out).run()
    folder = session_dir(tmp_path)
    log = (folder / "logs" / "franka_control.log").read_text()
    assert "fake: starting" in log and "fake: ready" in log
    text = out.getvalue()
    assert "fake: starting" in text
    assert ("control_deploy: ready: C2 tcp://127.0.0.1:5620, C3 tcp://127.0.0.1:5630; start policy_deploy on the "
            "laptop. Keys: h home arm, g home hand, s stop, q quit") in text
    assert "ready:" in (folder / "logs" / "control_deploy.log").read_text()
    for name in ("run.yaml", "station.yaml", "contracts/limits.yaml", "contracts/frames.yaml", "contracts/link.yaml"):
        assert (folder / name).is_file(), name


def test_not_ready_in_time_gets_sigterm(setup, monkeypatch):
    tmp_path, station = setup
    monkeypatch.setenv("FAKE_NO_READY", "1")
    monkeypatch.setenv("FAKE_RUN_S", "30")
    out = io.StringIO()
    assert make(station, out, ready_timeout_s=0.5).run() == 1
    assert READY_HINT in out.getvalue()
    assert "fake: SIGTERM" in (session_dir(tmp_path) / "logs" / "franka_control.log").read_text()


@pytest.mark.parametrize("code", [1, 3])
def test_exit_codes_are_passed_through(setup, monkeypatch, code):
    tmp_path, station = setup
    monkeypatch.setenv("FAKE_EXIT_CODE", str(code))
    out = io.StringIO()
    assert make(station, out).run() == code
    assert "franka_control failed: see" in out.getvalue()


def test_sigterm_is_forwarded(setup, monkeypatch):
    tmp_path, station = setup
    monkeypatch.setenv("FAKE_RUN_S", "30")
    out = io.StringIO()
    timer = later(1.0)
    rc = make(station, out).run()
    timer.join()
    assert rc == 0
    assert "received SIGTERM: forwarding SIGTERM to franka_control" in out.getvalue()
    assert "fake: SIGTERM" in out.getvalue()


def test_sighup_is_forwarded_as_sigterm(setup, monkeypatch):
    tmp_path, station = setup
    monkeypatch.setenv("FAKE_RUN_S", "30")
    out = io.StringIO()
    timer = later(1.0, signal.SIGHUP)
    assert make(station, out).run() == 0
    timer.join()
    assert "received SIGHUP" in out.getvalue() and "fake: SIGTERM" in out.getvalue()


def test_sigkill_after_the_stop_timeout(setup, monkeypatch):
    assert runner_mod.STOP_TIMEOUT_S == 5.0
    tmp_path, station = setup
    monkeypatch.setenv("FAKE_RUN_S", "30")
    monkeypatch.setenv("FAKE_IGNORE_TERM", "1")
    out = io.StringIO()
    timer = later(1.0)
    rc = make(station, out, stop_timeout_s=0.5).run()
    timer.join()
    assert rc == 1
    assert "SIGKILL" in out.getvalue()


def test_termios_is_restored_on_a_pty_after_the_child_is_killed(setup, monkeypatch):
    tmp_path, station = setup
    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        monkeypatch.setenv("FAKE_RUN_S", "30")
        monkeypatch.setenv("FAKE_IGNORE_TERM", "1")
        monkeypatch.setenv("FAKE_RAW_TTY", "1")
        out = io.StringIO()
        timer = later(1.0)
        rc = make(station, out, stdin_fd=slave, stop_timeout_s=0.5).run()
        timer.join()
        assert rc == 1
        assert termios.tcgetattr(slave) == before
    finally:
        os.close(slave)
        os.close(master)


def test_a_failing_check_starts_nothing(setup):
    tmp_path, station = setup
    out = io.StringIO()
    assert make(station, out, checks=FailingChecks).run() == 2
    assert "a check failed: nothing started" in out.getvalue()
    assert "FAIL" in out.getvalue()
    assert not (tmp_path / "args.txt").exists()
    assert not (tmp_path / "sessions").exists()
