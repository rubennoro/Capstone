# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/test_managed_process.py @ 2ffe641
# changes: the teleop_collect process names; cwd is deployment/teleop_collect/; the default program is
# sys.executable -m teleop_collect.<name>; no USES_TERMINAL process (that test is gone, every process gets its own
# group); the merged config is config/teleop.yaml's; imports. The source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/tests/test_managed_process.py @ 91b178b
# changes: laptop process names; cwd is deployment/policy_deploy/; config from config/<name>.yaml; the ready timeout
# message says "during start-up" (STRUCTURE.md 9.3); + the default program.
from __future__ import annotations

import os
import shlex
import signal
import time
from pathlib import Path

import pytest

from launcher_helpers import make_process

from teleop_collect.contracts import Station, StationError, paths
from teleop_collect.launcher.errors import LaunchError
from teleop_collect.launcher.session import Session


def wait_for(condition, timeout_s: float = 5.0) -> bool:
    """Poll a condition until it holds or timeout_s passes (no fixed sleeps in tests)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def session(station) -> Session:
    return Session.create(station.session_root, False, station.path)


@pytest.fixture
def running():
    """Processes a test started; all stopped at teardown."""
    started = []
    yield started
    for process in started:
        process.stop(0.5)


def started(cls, station, session, running, mock=False):
    process = cls(station, mock, session)
    process.start()
    running.append(process)
    return process


def test_command_is_program_then_common_arguments(station, session):
    cls = make_process("teleop", "serve")
    process = cls(station, False, session)
    program = process.program()
    assert process.command() == [*program, "--station", str(station.path), "--session", str(session.path),
                                 "--ready-file", str(session.ready_file("teleop"))]
    assert cls(station, True, session).command()[-1] == "--mock"
    assert "--mock" not in process.command()
    assert cls(station, True, None).command() == [*program, "--station", str(station.path), "--mock"]


def test_paths_come_from_the_session(station, session):
    process = make_process("teleop", "serve")(station, False, session)
    assert process.folder == paths.folder()
    assert process.log_path == session.log_path("teleop")
    assert process.ready_file == session.ready_file("teleop")
    unattached = make_process("teleop", "serve")(station, False, None)
    assert unattached.log_path is None and unattached.ready_file is None


def test_log_starts_with_command_and_captures_stdout_and_stderr(station, session, running):
    process = started(make_process("frames", "echo"), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert wait_for(lambda: "hello on stderr" in process.log_path.read_text())
    lines = process.log_path.read_text().splitlines()
    assert lines[0] == shlex.join(process.command())
    assert "hello on stdout" in lines and "hello on stderr" in lines


def test_started_in_deployment_teleop_collect(station, session, running, tmp_path, monkeypatch):
    """cwd = deployment/teleop_collect/ for every process, wherever the launcher runs from."""
    monkeypatch.chdir(tmp_path)
    cwd_file = tmp_path / "cwd"
    process = started(make_process("teleop", "pwd", "--marker", str(cwd_file)), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert Path(cwd_file.read_text()) == paths.folder() == process.folder


def test_ready_via_ready_file(station, session, running):
    process = started(make_process("teleop", "serve", "--delay", "0.2"), station, session, running)
    assert not process.is_ready()
    process.wait_ready(5.0, 0.01)
    assert process.ready_file.exists() and process.returncode() is None


def test_ready_timeout_names_process_and_hint(station, session, running):
    process = started(make_process("frames", "never_ready"), station, session, running)
    with pytest.raises(LaunchError) as info:
        process.wait_ready(0.3, 0.01)
    message = str(info.value)
    assert message.startswith("FAILED frames during start-up: not ready after 0.3 s (waiting for: the frames hint")
    assert f"[log: {process.log_path}]" in message


def test_exit_during_startup_quotes_fatal_line(station, session, running):
    fatal = "FATAL teleop: config: teleop.motion.v_max_m_s: expected a number > 0: check station.yaml teleop:"
    process = started(make_process("teleop", "exit", "--fatal", fatal), station, session, running)
    with pytest.raises(LaunchError) as info:
        process.wait_ready(5.0, 0.01)
    assert str(info.value) == f"FAILED teleop during start-up (exit 1): {fatal}  [log: {process.log_path}]"
    assert process.fatal_line() == fatal


def test_exit_without_fatal_line(station, session, running):
    process = started(make_process("fake_pc", "exit", "--code", "3"), station, session, running)
    with pytest.raises(LaunchError, match=r"FAILED fake_pc during start-up \(exit 3, no FATAL line\)"):
        process.wait_ready(5.0, 0.01)
    assert process.fatal_line() is None


def test_stop_sends_sigterm(station, session, running, tmp_path):
    marker = tmp_path / "got_sigterm"
    process = started(make_process("teleop", "serve", "--marker", str(marker)), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert process.stop(5.0) == 0
    assert marker.exists()
    assert process.stop(5.0) == 0                     # already exited: just its code


def test_stop_escalates_to_sigkill(station, session, running):
    process = started(make_process("teleop", "ignore_term"), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert process.stop(0.3) == -signal.SIGKILL


def test_stop_reaches_a_grandchild(station, session, running, tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    process = started(make_process("teleop", "grandchild", "--pid-file", str(pid_file)), station, session, running)
    process.wait_ready(5.0, 0.01)
    grandchild = int(pid_file.read_text())
    assert alive(grandchild)
    process.stop(5.0)
    assert wait_for(lambda: not alive(grandchild))


def alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"          # a zombie waiting to be reaped is not running


def test_other_processes_get_their_own_group(station, session, running):
    process = started(make_process("teleop", "serve"), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert os.getpgid(process._popen.pid) == process._popen.pid != os.getpgrp()


def test_is_ready_raising_fails_at_once(station, session, running):
    class Broken(make_process("fake_pc", "never_ready")):
        def is_ready(self) -> bool:
            raise LaunchError("FAILED fake_pc: arm in fault")

    process = started(Broken, station, session, running)
    begin = time.monotonic()
    with pytest.raises(LaunchError, match="arm in fault"):
        process.wait_ready(30.0, 0.01)
    assert time.monotonic() - begin < 5.0


def test_config_path_by_where_the_value_came_from(make_station):
    cls = make_process("frames", "serve")
    default = cls(Station.load(make_station(), False), False, None)
    assert default.config_path("rig_file") == paths.folder() / "rigs" / "rig_pi_v2_640.yaml"
    station = Station.load(make_station(frames={"rig_file": "my_rigs/r.yaml"}), False)
    assert cls(station, False, None).config_path("rig_file") == station.path.parent / "my_rigs" / "r.yaml"


def test_config_is_merged_with_station_section(make_station):
    station = Station.load(make_station(teleop={"motion": {"v_max_m_s": 0.02}, "history_s": 2.0}), False)
    process = make_process("teleop", "serve")(station, False, None)
    assert process.config["motion"]["v_max_m_s"] == 0.02 and process.config["history_s"] == 2.0
    assert process.config["motion"]["slow_factor"] == 0.25


def test_base_checks(make_station, monkeypatch):
    good = make_process("teleop", "serve")(Station.load(make_station(), False), False, None)
    assert [(r.name, r.status) for r in good.base_checks()] == [("overrides", "pass"), ("program", "pass")]
    assert "override(s) of config/teleop.yaml" in good.base_checks()[0].detail

    import teleop_collect.launcher.managed_process as module

    def broken(name, station):
        raise StationError("station.yaml: teleop.histroy_s: not a key of config/teleop.yaml")

    monkeypatch.setattr(module, "load_config", broken)
    bad = make_process("teleop", "serve")(Station.load(make_station(), False), False, None)
    [result] = bad.base_checks()
    assert (result.name, result.status) == ("overrides", "fail")
    assert "teleop.histroy_s: not a key of config/teleop.yaml" in result.detail
    monkeypatch.undo()

    class NoProgram(make_process("teleop", "serve")):
        def program(self) -> list[str]:
            return ["/no/such/binary"]

    checks = NoProgram(Station.load(make_station(), False), False, None).base_checks()
    assert checks[1].status == "fail" and "/no/such/binary" in checks[1].detail


def test_default_program_is_python_dash_m_teleop_collect(make_station):
    import sys

    from teleop_collect.launcher.processes import FramesProcess
    station = Station.load(make_station(), False)
    assert FramesProcess(station, False, None).program() == [sys.executable, "-m", "teleop_collect.frames"]


def test_stop_before_start_and_start_without_session(station):
    process = make_process("teleop", "serve")(station, False, None)
    assert process.stop(1.0) is None and process.returncode() is None
    with pytest.raises(LaunchError, match="without a session"):
        process.start()
