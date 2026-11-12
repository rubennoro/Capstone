from __future__ import annotations

import os
import shlex
import signal
import time
from pathlib import Path

import pytest

from deploy import LaunchError, Session, Station
from fakes import make_process


def wait_for(condition, timeout_s: float = 5.0) -> bool:
    """Poll a condition until it holds or timeout_s passes (no fixed sleeps in tests)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def station(fake_root) -> Station:
    return Station.load(fake_root / "station.yaml", False)


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
    cls = make_process("policy", "serve")
    process = cls(station, False, session)
    program = process.program()
    assert process.command() == [*program, "--station", str(station.path), "--session", str(session.path),
                                 "--ready-file", str(session.ready_file("policy"))]
    assert cls(station, True, session).command()[-1] == "--mock"
    assert "--mock" not in process.command()
    assert cls(station, True, None).command() == [*program, "--station", str(station.path), "--mock"]


def test_paths_come_from_the_session(station, session, fake_root):
    process = make_process("policy", "serve")(station, False, session)
    assert process.folder == fake_root / "policy"
    assert process.log_path == session.log_path("policy")
    assert process.ready_file == session.ready_file("policy")
    unattached = make_process("policy", "serve")(station, False, None)
    assert unattached.log_path is None and unattached.ready_file is None


def test_log_starts_with_command_and_captures_stdout_and_stderr(station, session, running):
    process = started(make_process("process_frames", "echo"), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert wait_for(lambda: "hello on stderr" in process.log_path.read_text())
    lines = process.log_path.read_text().splitlines()
    assert lines[0] == shlex.join(process.command())
    assert "hello on stdout" in lines and "hello on stderr" in lines


def test_started_in_its_own_folder_not_deployment(station, session, running, fake_root, tmp_path, monkeypatch):
    """cwd = the process folder: from deployment/, `python -m <pkg>` would import the folder deployment/<pkg>/ as an
    empty namespace package instead of the installed package."""
    monkeypatch.chdir(fake_root)                              # the launcher itself runs from deployment/
    cwd_file = tmp_path / "cwd"
    process = started(make_process("policy", "pwd", "--marker", str(cwd_file)), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert Path(cwd_file.read_text()) == fake_root / "policy" == process.folder


def test_ready_via_ready_file(station, session, running):
    process = started(make_process("policy", "serve", "--delay", "0.2"), station, session, running)
    assert not process.is_ready()
    process.wait_ready(5.0, 0.01)
    assert process.ready_file.exists() and process.returncode() is None


def test_ready_timeout_names_process_and_hint(station, session, running):
    process = started(make_process("process_frames", "never_ready"), station, session, running)
    with pytest.raises(LaunchError) as info:
        process.wait_ready(0.3, 0.01)
    message = str(info.value)
    assert message.startswith("FAILED process_frames: not ready after 0.3 s (waiting for: the process_frames hint")
    assert f"[log: {process.log_path}]" in message


def test_exit_during_startup_quotes_fatal_line(station, session, running):
    fatal = "FATAL policy: checkpoint not found: /x.ckpt: check policy.checkpoint in station.yaml"
    process = started(make_process("policy", "exit", "--fatal", fatal), station, session, running)
    with pytest.raises(LaunchError) as info:
        process.wait_ready(5.0, 0.01)
    assert str(info.value) == f"FAILED policy during start-up (exit 1): {fatal}  [log: {process.log_path}]"
    assert process.fatal_line() == fatal


def test_exit_without_fatal_line(station, session, running):
    process = started(make_process("franka_control", "exit", "--code", "3"), station, session, running)
    with pytest.raises(LaunchError, match=r"FAILED franka_control during start-up \(exit 3, no FATAL line\)"):
        process.wait_ready(5.0, 0.01)
    assert process.fatal_line() is None


def test_stop_sends_sigterm(station, session, running, tmp_path):
    marker = tmp_path / "got_sigterm"
    process = started(make_process("policy", "serve", "--marker", str(marker)), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert process.stop(5.0) == 0
    assert marker.exists()
    assert process.stop(5.0) == 0                     # already exited: just its code


def test_stop_escalates_to_sigkill(station, session, running):
    process = started(make_process("policy", "ignore_term"), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert process.stop(0.3) == -signal.SIGKILL


def test_stop_reaches_a_grandchild(station, session, running, tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    process = started(make_process("policy", "grandchild", "--pid-file", str(pid_file)), station, session, running)
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


def test_terminal_process_signalled_by_pid_only(station, session, running, tmp_path, monkeypatch):
    def no_group_signals(pgid, sig):
        raise AssertionError(f"killpg({pgid}, {sig}) called for the USES_TERMINAL process")

    monkeypatch.setattr(os, "killpg", no_group_signals)
    marker = tmp_path / "got_sigterm"
    cls = make_process("message_comms", "serve", "--marker", str(marker), terminal=True)
    process = started(cls, station, session, running)
    process.wait_ready(5.0, 0.01)
    assert os.getpgid(process._popen.pid) == os.getpgrp()     # it shares the launcher's process group
    assert process.stop(5.0) == 0
    assert marker.exists()                                     # and the test process is still here


def test_other_processes_get_their_own_group(station, session, running):
    process = started(make_process("policy", "serve"), station, session, running)
    process.wait_ready(5.0, 0.01)
    assert os.getpgid(process._popen.pid) == process._popen.pid != os.getpgrp()


def test_is_ready_raising_fails_at_once(station, session, running):
    class Broken(make_process("franka_control", "never_ready")):
        def is_ready(self) -> bool:
            raise LaunchError("FAILED franka_control: arm in fault")

    process = started(Broken, station, session, running)
    begin = time.monotonic()
    with pytest.raises(LaunchError, match="arm in fault"):
        process.wait_ready(30.0, 0.01)
    assert time.monotonic() - begin < 5.0


def test_config_path_by_where_the_value_came_from(fake_root, make_station):
    cls = make_process("process_frames", "serve")
    default = cls(Station.load(make_station(), False), False, None)
    assert default.config_path("rig_file") == fake_root / "process_frames" / "rigs" / "rig_pi_v1_standin.yaml"
    station = Station.load(make_station(process_frames={"rig_file": "my_rigs/r.yaml"}), False)
    assert cls(station, False, None).config_path("rig_file") == station.path.parent / "my_rigs" / "r.yaml"


def test_config_is_merged_with_station_section(fake_root, make_station):
    station = Station.load(make_station(policy={"kind": "hold_still"}), False)
    process = make_process("policy", "serve")(station, False, None)
    assert process.config["kind"] == "hold_still" and process.config["device"] == "cuda"


def test_base_checks(fake_root, make_station):
    good = make_process("policy", "serve")(Station.load(make_station(), False), False, None)
    assert [(r.name, r.status) for r in good.base_checks()] == [("overrides", "pass"), ("program", "pass")]
    bad = make_process("policy", "serve")(Station.load(make_station(policy={"chekpoint": 1}), False), False, None)
    [result] = bad.base_checks()
    assert (result.name, result.status) == ("overrides", "fail")
    assert "policy.chekpoint: not a key of policy/config.yaml" in result.detail

    class NoProgram(make_process("policy", "serve")):
        def program(self) -> list[str]:
            return ["/no/such/binary"]

    checks = NoProgram(Station.load(make_station(), False), False, None).base_checks()
    assert checks[1].status == "fail" and "/no/such/binary" in checks[1].detail


def test_stop_before_start_and_start_without_session(station):
    process = make_process("policy", "serve")(station, False, None)
    assert process.stop(1.0) is None and process.returncode() is None
    with pytest.raises(LaunchError, match="without a session"):
        process.start()
