# adapted from rnoro/deployment:deployment/deploy/tests/test_launcher.py @ 91b178b
# changes: the laptop ORDER (fake_pc, frames, policy, manager); messages policy_deploy:, `policy_deploy: done`;
# launcher.log; the station rows stubbed (test_check_rows); + contracts/ in the session, manager failure, the
# fake_pc switch, the launcher settings.
from __future__ import annotations

import signal

import pytest

from launcher_fakes import make_process

from policy_deploy.contracts import Station
from policy_deploy.launcher.check_result import CheckResult
from policy_deploy.launcher.launcher import Launcher

ORDER = ["fake_pc", "frames", "policy", "manager"]


@pytest.fixture(autouse=True)
def quiet_station_rows(monkeypatch):
    """The station rows (laptop address, Wi-Fi, ...) are test_check_rows' business: one passing row here."""
    monkeypatch.setattr(Launcher, "station_checks",
                        lambda self: [CheckResult("station", "station.yaml", "pass", str(self.station.path))])


def build(station, specs: dict, events: list, mock: bool = False):
    """specs: name -> (mode, args..., ) or a ready process class; message_comms uses the terminal by default."""
    processes = []
    for name, spec in specs.items():
        mode, *args = spec
        cls = make_process(name, mode, *args, terminal=(name == "manager"), events=events)
        processes.append(cls(station, mock, None))
    return processes


def serve_all(**changes) -> dict:
    specs = {name: ("serve",) for name in ORDER}
    specs.update(changes)
    return specs


def test_start_order_each_after_the_previous_is_ready(station, capsys):
    events: list = []
    specs = serve_all(manager=("exit", "--ready", "--delay", "0.2", "--code", "0"))
    processes = build(station, dict(reversed(list(specs.items()))), events)      # given in the wrong order
    assert Launcher(station, False, processes).run() == 0
    starts = [e for e in events if e[0] in ("start", "ready")]
    assert starts == [(kind, name) for name in ORDER for kind in ("start", "ready")]
    out = capsys.readouterr().out
    assert "policy_deploy: all 4 processes ready (session: " in out
    assert out.splitlines()[-1] == "policy_deploy: done"


def test_manager_exit_0_stops_the_rest_in_reverse(station):
    events: list = []
    processes = build(station, serve_all(manager=("exit", "--ready", "--delay", "0.2", "--code", "0")),
                      events)
    assert Launcher(station, False, processes).run() == 0
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert all(p.returncode() is not None for p in processes)


def test_ready_timeout_of_the_second_stops_the_first(make_station, capsys):
    station = Station.load(make_station(launcher={"poll_s": 0.02, "ready_timeout_s": {"frames": 0.3}}), False)
    events: list = []
    processes = build(station, serve_all(frames=("never_ready",)), events)
    assert Launcher(station, False, processes).run() == 1
    assert events == [("start", "fake_pc"), ("ready", "fake_pc"), ("start", "frames"),
                      ("stop", "frames"), ("stop", "fake_pc")]
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith("policy_deploy: FAILED frames during start-up: not ready after 0.3 s (waiting for: "
                              "the frames hint; check its fake)  [log: ")
    assert all(p.returncode() is not None for p in processes[:2])


def test_one_process_exiting_stops_all_in_reverse(station, capsys):
    events: list = []
    fatal = "FATAL policy: CUDA error: check the GPU"
    processes = build(station, serve_all(policy=("exit", "--ready", "--delay", "0.3", "--fatal", fatal)), events)
    launcher = Launcher(station, False, processes)
    assert launcher.run() == 1
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith(f"policy_deploy: FAILED policy (exit 1): {fatal}  [log: ")
    launcher_log = (processes[0].session.path / "logs" / "launcher.log").read_text()
    assert "FAILED policy (exit 1)" in launcher_log
    for name in ORDER:
        assert f"{name} stopped (exit " in launcher_log


def test_exit_during_startup_never_starts_the_rest(station, capsys):
    events: list = []
    processes = build(station, serve_all(frames=("exit", "--fatal", "FATAL frames: no Pi")), events)
    assert Launcher(station, False, processes).run() == 1
    assert ("start", "policy") not in events and ("start", "manager") not in events
    assert "policy_deploy: FAILED frames during start-up (exit 1): FATAL frames: no Pi" in \
        capsys.readouterr().out


def test_sigint_stops_in_reverse_waits_for_terminal_process(station, tmp_path, capsys):
    marker = tmp_path / "manager_got_sigterm"
    events: list = []
    processes = build(station, serve_all(manager=("ctrl_c", "--delay", "0.3", "--marker", str(marker))),
                      events)
    old = signal.getsignal(signal.SIGINT)
    assert Launcher(station, False, processes).run() == 130
    assert signal.getsignal(signal.SIGINT) is old                     # the handler is restored
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert processes[-1].returncode() == 0                            # exited on its own ...
    assert not marker.exists()                                        # ... before any SIGTERM
    assert capsys.readouterr().out.splitlines()[-1] == "policy_deploy: stopped by Ctrl-C"


def test_sigterm_to_the_launcher(station, capsys):
    events: list = []

    class SendsSigterm(make_process("manager", "serve", terminal=True, events=events)):
        def wait_ready(self, timeout_s, poll_s):
            super().wait_ready(timeout_s, poll_s)
            signal.raise_signal(signal.SIGTERM)

    processes = build(station, {n: ("serve",) for n in ORDER[:3]}, events) + [SendsSigterm(station, False, None)]
    assert Launcher(station, False, processes).run() == 130
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert processes[-1].returncode() == 0                            # SIGTERM sent to it by the launcher
    assert capsys.readouterr().out.splitlines()[-1] == "policy_deploy: stopped by SIGTERM"


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
    assert out.splitlines()[-1] == "policy_deploy: not started: 1 check(s) failed (see above)"
    assert not station.session_root.exists()


def test_two_terminal_processes_refused(station, capsys):
    events: list = []
    processes = build(station, serve_all(), events)
    processes[0] = make_process("fake_pc", "serve", terminal=True, events=events)(station, False, None)
    assert Launcher(station, False, processes).run() == 2
    assert events == []
    assert "more than one process uses the terminal: fake_pc, manager" in capsys.readouterr().out


def test_session_created_and_attached(station):
    processes = build(station, serve_all(manager=("exit", "--ready", "--delay", "0.1", "--code", "0")), [])
    Launcher(station, False, processes).run()
    session = processes[0].session
    assert all(p.session is session for p in processes)
    assert session.path.parent == station.session_root
    assert (session.path / "station.yaml").read_text() == station.path.read_text()
    for name in ORDER:
        assert (session.path / "logs" / f"{name}.log").exists()


def test_session_has_contracts_and_launcher_log(station):
    processes = build(station, serve_all(manager=("exit", "--ready", "--delay", "0.1", "--code", "0")), [])
    Launcher(station, False, processes).run()
    session = processes[0].session
    assert sorted(p.name for p in (session.path / "contracts").iterdir()) == ["frames.yaml", "limits.yaml",
                                                                             "link.yaml"]
    assert "manager exited 0" in (session.path / "logs" / "launcher.log").read_text()
    assert all((session.path / "ready" / f"{n}.ready").exists() for n in ORDER)


def test_manager_exit_1_is_a_failure_and_names_it(station, capsys):
    fatal = "FATAL manager: PC not answering: no reply to c3.heartbeat/v1 within 0.3 s: check franka_control"
    processes = build(station, serve_all(manager=("exit", "--ready", "--delay", "0.2", "--fatal", fatal)), [])
    assert Launcher(station, False, processes).run() == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith(f"policy_deploy: FAILED manager (exit 1): {fatal}  [log: ")
    assert last.endswith("logs/manager.log]")


@pytest.mark.parametrize("mock,fake_pc,names", [
    (False, True, ["frames", "policy", "manager"]),
    (True, True, ["fake_pc", "frames", "policy", "manager"]),
    (True, False, ["frames", "policy", "manager"]),
    (False, False, ["frames", "policy", "manager"]),
])
def test_the_fake_pc_only_with_mock_and_its_switch(make_station, mock, fake_pc, names):
    station = Station.load(make_station(launcher={"fake_pc": fake_pc}), mock)
    launcher = Launcher(station, mock)
    assert [p.NAME for p in launcher.processes] == names
    manager = launcher.processes[-1]
    assert manager.USES_TERMINAL and manager.fake_pc is fake_pc


def test_settings_from_config_and_station(make_station):
    launcher = Launcher(Station.load(make_station(launcher={"ready_timeout_s": {"policy": 12.0}}), True), True)
    s = launcher.settings
    assert s.ready_timeout_s == {"fake_pc": 10.0, "frames": 30.0, "policy": 12.0, "manager": 130.0}
    assert (s.stop_timeout_s, s.poll_s, s.fake_pc) == (5.0, 0.1, True)
