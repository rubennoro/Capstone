from __future__ import annotations

import re
import signal

import pytest

from deploy import CheckResult, Launcher, Station
from fakes import make_process
from helpers import station_data, write_yaml

ORDER = ["franka_control", "process_frames", "policy", "message_comms"]


@pytest.fixture
def station(fake_root) -> Station:
    data = station_data(launcher={"poll_s": 0.02, "stop_timeout_s": 2.0,
                                  "ready_timeout_s": {name: 5.0 for name in ORDER}})
    return Station.load(write_yaml(fake_root / "station_test.yaml", data), False)


def build(station, specs: dict, events: list, mock: bool = False):
    """specs: name -> (mode, args..., ) or a ready process class; message_comms uses the terminal by default."""
    processes = []
    for name, spec in specs.items():
        mode, *args = spec
        cls = make_process(name, mode, *args, terminal=(name == "message_comms"), events=events)
        processes.append(cls(station, mock, None))
    return processes


def serve_all(**changes) -> dict:
    specs = {name: ("serve",) for name in ORDER}
    specs.update(changes)
    return specs


def test_start_order_each_after_the_previous_is_ready(station, capsys):
    events: list = []
    specs = serve_all(message_comms=("exit", "--ready", "--delay", "0.2", "--code", "0"))
    processes = build(station, dict(reversed(list(specs.items()))), events)      # given in the wrong order
    assert Launcher(station, False, processes).run() == 0
    starts = [e for e in events if e[0] in ("start", "ready")]
    assert starts == [(kind, name) for name in ORDER for kind in ("start", "ready")]
    out = capsys.readouterr().out
    assert "deploy: all 4 processes ready (session: " in out
    assert re.search(r"deploy: done \(session: .*\)", out)


def test_message_comms_exit_0_stops_the_rest_in_reverse(station):
    events: list = []
    processes = build(station, serve_all(message_comms=("exit", "--ready", "--delay", "0.2", "--code", "0")),
                      events)
    assert Launcher(station, False, processes).run() == 0
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert all(p.returncode() is not None for p in processes)


def test_ready_timeout_of_the_second_stops_the_first(fake_root, capsys):
    data = station_data(launcher={"poll_s": 0.02, "ready_timeout_s": {"process_frames": 0.3}})
    station = Station.load(write_yaml(fake_root / "s.yaml", data), False)
    events: list = []
    processes = build(station, serve_all(process_frames=("never_ready",)), events)
    assert Launcher(station, False, processes).run() == 1
    assert events == [("start", "franka_control"), ("ready", "franka_control"), ("start", "process_frames"),
                      ("stop", "process_frames"), ("stop", "franka_control")]
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith("deploy: FAILED process_frames: not ready after 0.3 s (waiting for: "
                              "the process_frames hint; check its fake)  [log: ")
    assert all(p.returncode() is not None for p in processes[:2])


def test_one_process_exiting_stops_all_in_reverse(station, capsys):
    events: list = []
    fatal = "FATAL policy: CUDA error: check the GPU"
    processes = build(station, serve_all(policy=("exit", "--ready", "--delay", "0.3", "--fatal", fatal)), events)
    launcher = Launcher(station, False, processes)
    assert launcher.run() == 1
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith(f"deploy: FAILED policy (exit 1): {fatal}  [log: ")
    deploy_log = (processes[0].session.path / "logs" / "deploy.log").read_text()
    assert "FAILED policy (exit 1)" in deploy_log
    for name in ORDER:
        assert f"{name} stopped (exit " in deploy_log


def test_exit_during_startup_never_starts_the_rest(station, capsys):
    events: list = []
    processes = build(station, serve_all(process_frames=("exit", "--fatal", "FATAL process_frames: no Pi")), events)
    assert Launcher(station, False, processes).run() == 1
    assert ("start", "policy") not in events and ("start", "message_comms") not in events
    assert "deploy: FAILED process_frames during start-up (exit 1): FATAL process_frames: no Pi" in \
        capsys.readouterr().out


def test_sigint_stops_in_reverse_waits_for_terminal_process(station, tmp_path, capsys):
    marker = tmp_path / "message_comms_got_sigterm"
    events: list = []
    processes = build(station, serve_all(message_comms=("ctrl_c", "--delay", "0.3", "--marker", str(marker))),
                      events)
    old = signal.getsignal(signal.SIGINT)
    assert Launcher(station, False, processes).run() == 130
    assert signal.getsignal(signal.SIGINT) is old                     # the handler is restored
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert processes[-1].returncode() == 0                            # exited on its own ...
    assert not marker.exists()                                        # ... before any SIGTERM
    assert capsys.readouterr().out.splitlines()[-1] == "deploy: stopped by Ctrl-C"


def test_sigterm_to_the_launcher(station, capsys):
    events: list = []

    class SendsSigterm(make_process("message_comms", "serve", terminal=True, events=events)):
        def wait_ready(self, timeout_s, poll_s):
            super().wait_ready(timeout_s, poll_s)
            signal.raise_signal(signal.SIGTERM)

    processes = build(station, {n: ("serve",) for n in ORDER[:3]}, events) + [SendsSigterm(station, False, None)]
    assert Launcher(station, False, processes).run() == 130
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert processes[-1].returncode() == 0                            # SIGTERM sent to it by the launcher
    assert capsys.readouterr().out.splitlines()[-1] == "deploy: stopped by SIGTERM"


def test_failing_check_refuses_to_start(station, capsys):
    events: list = []
    bad = make_process("policy", "serve", checks=(CheckResult("policy", "kind", "fail", "policy.kind is not set"),),
                       events=events)
    processes = build(station, {n: ("serve",) for n in ORDER if n != "policy"}, events)
    processes.append(bad(station, False, None))
    assert Launcher(station, False, processes).run() == 2
    assert events == []
    out = capsys.readouterr().out
    assert "policy.kind is not set" in out
    assert out.splitlines()[-1] == "deploy: not started: 1 check(s) failed (see above)"
    assert not station.session_root.exists()


def test_two_terminal_processes_refused(station, capsys):
    events: list = []
    processes = build(station, serve_all(), events)
    processes[0] = make_process("franka_control", "serve", terminal=True, events=events)(station, False, None)
    assert Launcher(station, False, processes).run() == 2
    assert events == []
    assert "more than one process uses the terminal: franka_control, message_comms" in capsys.readouterr().out


def test_session_created_and_attached(station):
    processes = build(station, serve_all(message_comms=("exit", "--ready", "--delay", "0.1", "--code", "0")), [])
    Launcher(station, False, processes).run()
    session = processes[0].session
    assert all(p.session is session for p in processes)
    assert session.path.parent == station.session_root
    assert (session.path / "station.yaml").read_text() == station.path.read_text()
    for name in ORDER:
        assert (session.path / "logs" / f"{name}.log").exists()
