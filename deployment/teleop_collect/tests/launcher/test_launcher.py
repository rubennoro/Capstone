# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/test_launcher.py @ 2ffe641
# changes: the teleop_collect ORDER (fake_pc, frames, teleop, ui); messages teleop_collect:; teleop exiting 0 ends
# the run as pd's manager did (exit 0, `teleop stopped (exit 0): done`); no terminal process (the Ctrl-C comes from
# ui's fake, no two-terminal refusal); Ctrl-C and SIGTERM exit 0 (were 130); + SIGHUP like SIGTERM (also with the
# terminal gone), a second signal during the stop
# (130), ui exiting 0 is a failure, the ready line and the browser (--no-browser, ui.open_browser, no display, no
# browser), the teleop start note; imports.
from __future__ import annotations

import signal

import pytest

from launcher_helpers import make_process

import teleop_collect.launcher.launcher as launcher_module
from teleop_collect.contracts import Station
from teleop_collect.launcher.check_result import CheckResult
from teleop_collect.launcher.launcher import Launcher
from teleop_collect.launcher.processes import TeleopProcess

ORDER = ["fake_pc", "frames", "teleop", "ui"]
TELEOP_DONE = ("exit", "--ready", "--delay", "0.2", "--code", "0")
FAST = {"poll_s": 0.02, "stop_timeout_s": 2.0, "ready_timeout_s": {name: 5.0 for name in ORDER}}    # conftest's


@pytest.fixture(autouse=True)
def quiet_station_rows(monkeypatch):
    """The station rows (laptop address, Wi-Fi, ...) are test_check_rows' business: one passing row here."""
    monkeypatch.setattr(Launcher, "station_checks",
                        lambda self: [CheckResult("station", "station.yaml", "pass", str(self.station.path))])


def build(station, specs: dict, events: list, mock: bool = False):
    """specs: name -> (mode, args...)."""
    processes = []
    for name, spec in specs.items():
        mode, *args = spec
        processes.append(make_process(name, mode, *args, events=events)(station, mock, None))
    return processes


def serve_all(**changes) -> dict:
    specs = {name: ("serve",) for name in ORDER}
    specs.update(changes)
    return specs


def test_start_order_each_after_the_previous_is_ready(station, capsys, browser):
    events: list = []
    specs = serve_all(teleop=("exit", "--ready", "--delay", "0.4", "--code", "0"))
    processes = build(station, dict(reversed(list(specs.items()))), events)      # given in the wrong order
    assert Launcher(station, False, processes).run() == 0
    starts = [e for e in events if e[0] in ("start", "ready")]
    assert starts == [(kind, name) for name in ORDER for kind in ("start", "ready")]
    out = capsys.readouterr().out
    assert "teleop_collect: starting fake_pc, frames, teleop, ui (session: " in out
    assert "teleop_collect: all 4 processes ready (session: " in out
    assert f"teleop_collect: ready: open {station.http_url()}\n" in out
    assert browser == [station.http_url()]
    assert out.splitlines()[-1] == "teleop_collect: teleop stopped (exit 0): done"


def test_teleop_exit_0_stops_the_rest_in_reverse(station):
    events: list = []
    processes = build(station, serve_all(teleop=TELEOP_DONE), events)
    assert Launcher(station, False, processes).run() == 0
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert all(p.returncode() is not None for p in processes)


def test_ui_exit_0_is_a_failure(station, capsys):
    events: list = []
    processes = build(station, serve_all(ui=("exit", "--ready", "--delay", "0.2", "--code", "0")), events)
    assert Launcher(station, False, processes).run() == 1
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert capsys.readouterr().out.splitlines()[-1].startswith(
        "teleop_collect: FAILED ui (exit 0, no FATAL line): see the last lines of its log  [log: ")


def test_ready_timeout_of_the_second_stops_the_first(make_station, capsys, browser):
    station = Station.load(make_station(launcher={"poll_s": 0.02, "ready_timeout_s": {"frames": 0.3}}), False)
    events: list = []
    processes = build(station, serve_all(frames=("never_ready",)), events)
    assert Launcher(station, False, processes).run() == 1
    assert events == [("start", "fake_pc"), ("ready", "fake_pc"), ("start", "frames"),
                      ("stop", "frames"), ("stop", "fake_pc")]
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith(
        "teleop_collect: FAILED frames during start-up: not ready after 0.3 s (waiting for: the frames hint; check "
        "its fake)  [log: ")
    assert "ready: open" not in out and browser == []
    assert all(p.returncode() is not None for p in processes[:2])


def test_one_process_exiting_stops_all_in_reverse(station, capsys):
    events: list = []
    fatal = "FATAL frames: main camera down: check the Pi and ASUS_5g"
    processes = build(station, serve_all(frames=("exit", "--ready", "--delay", "0.3", "--fatal", fatal)), events)
    launcher = Launcher(station, False, processes)
    assert launcher.run() == 1
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith(f"teleop_collect: FAILED frames (exit 1): {fatal}  [log: ")
    assert last.endswith("logs/frames.log]")
    launcher_log = (processes[0].session.path / "logs" / "launcher.log").read_text()
    assert "FAILED frames (exit 1)" in launcher_log
    for name in ORDER:
        assert f"{name} stopped (exit " in launcher_log


def test_exit_during_startup_never_starts_the_rest(station, capsys):
    events: list = []
    processes = build(station, serve_all(frames=("exit", "--fatal", "FATAL frames: no Pi")), events)
    assert Launcher(station, False, processes).run() == 1
    assert ("start", "teleop") not in events and ("start", "ui") not in events
    assert "teleop_collect: FAILED frames during start-up (exit 1): FATAL frames: no Pi" in \
        capsys.readouterr().out


def test_sigint_stops_in_reverse_and_exits_0(station, tmp_path, capsys):
    marker = tmp_path / "ui_got_sigterm"
    events: list = []
    processes = build(station, serve_all(ui=("ctrl_c", "--delay", "5", "--marker", str(marker))), events)
    old = signal.getsignal(signal.SIGINT)
    assert Launcher(station, False, processes).run() == 0
    assert signal.getsignal(signal.SIGINT) is old                     # the handler is restored
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert marker.exists()                                            # the launcher stopped it (SIGTERM)
    assert capsys.readouterr().out.splitlines()[-1] == "teleop_collect: stopped by Ctrl-C"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"])
def test_sigterm_or_sighup_to_the_launcher(station, capsys, sig):
    """SIGHUP (the terminal closed) stops everything like SIGTERM: the processes run in their own sessions and would
    otherwise outlive the launcher."""
    events: list = []

    class SendsSignal(make_process("ui", "serve", events=events)):
        def wait_ready(self, timeout_s, poll_s):
            super().wait_ready(timeout_s, poll_s)
            signal.raise_signal(sig)

    old = signal.getsignal(sig)
    processes = build(station, {n: ("serve",) for n in ORDER[:3]}, events) + [SendsSignal(station, False, None)]
    assert Launcher(station, False, processes).run() == 0
    assert signal.getsignal(sig) is old                               # the handler is restored
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert all(p.returncode() is not None for p in processes)
    assert processes[-1].returncode() == 0                            # SIGTERM sent to it by the launcher
    assert capsys.readouterr().out.splitlines()[-1] == f"teleop_collect: stopped by {sig.name}"


def test_sighup_with_the_terminal_gone_still_stops_everything(station, monkeypatch):
    """After a hangup, printing to the closed terminal fails: the stop has happened and the run still exits 0."""
    events: list = []

    class Hangs(make_process("ui", "serve", events=events)):
        def wait_ready(self, timeout_s, poll_s):
            super().wait_ready(timeout_s, poll_s)
            signal.raise_signal(signal.SIGHUP)

    processes = build(station, {n: ("serve",) for n in ORDER[:3]}, events) + [Hangs(station, False, None)]
    launcher = Launcher(station, False, processes)
    import builtins
    real_print = builtins.print

    def print_to_a_dead_terminal(*args, **kwargs):
        if args and str(args[0]).startswith("teleop_collect: stopped by"):
            raise OSError(5, "Input/output error")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", print_to_a_dead_terminal)
    assert launcher.run() == 0
    assert [name for kind, name in events if kind == "stop"] == list(reversed(ORDER))
    assert all(p.returncode() is not None for p in processes)
    log = (processes[0].session.path / "logs" / "launcher.log").read_text()
    assert "teleop_collect: stopped by SIGHUP" in log and "exit 0" in log


def test_a_second_signal_during_the_stop_kills_everything(station, capsys):
    events: list = []

    class InterruptedStop(make_process("teleop", "ignore_term", events=events)):
        def stop(self, timeout_s):
            if timeout_s > 0:                                         # the normal stop: a second Ctrl-C arrives
                signal.raise_signal(signal.SIGINT)
            return super().stop(timeout_s)

    class SendsSigint(make_process("ui", "serve", events=events)):
        def wait_ready(self, timeout_s, poll_s):
            super().wait_ready(timeout_s, poll_s)
            signal.raise_signal(signal.SIGINT)

    processes = build(station, {n: ("serve",) for n in ORDER[:2]}, events) + [
        InterruptedStop(station, False, None), SendsSigint(station, False, None)]
    old = signal.getsignal(signal.SIGINT)
    assert Launcher(station, False, processes).run() == 130
    assert signal.getsignal(signal.SIGINT) is old
    assert all(p.returncode() is not None for p in processes)
    assert capsys.readouterr().out.splitlines()[-1] == "teleop_collect: killed by a second Ctrl-C during the stop"


def test_failing_check_refuses_to_start(station, capsys):
    events: list = []
    bad = make_process("teleop", "serve", checks=(CheckResult("teleop", "U1 port", "fail", "cannot bind"),),
                       events=events)
    processes = build(station, {n: ("serve",) for n in ORDER if n != "teleop"}, events)
    processes.append(bad(station, False, None))
    assert Launcher(station, False, processes).run() == 2
    assert events == []
    out = capsys.readouterr().out
    assert "cannot bind" in out
    assert out.splitlines()[-1] == "teleop_collect: not started: 1 check(s) failed (see above)"
    assert not station.session_root.exists()


def test_session_created_and_attached(station):
    processes = build(station, serve_all(teleop=TELEOP_DONE), [])
    Launcher(station, False, processes).run()
    session = processes[0].session
    assert all(p.session is session for p in processes)
    assert session.path.parent == station.session_root
    assert (session.path / "station.yaml").read_text() == station.path.read_text()
    for name in ORDER:
        assert (session.path / "logs" / f"{name}.log").exists()


def test_session_has_contracts_and_launcher_log(station):
    processes = build(station, serve_all(teleop=TELEOP_DONE), [])
    Launcher(station, False, processes).run()
    session = processes[0].session
    assert sorted(p.name for p in (session.path / "contracts").iterdir()) == ["frames.yaml", "limits.yaml",
                                                                             "link.yaml"]
    log = (session.path / "logs" / "launcher.log").read_text()
    assert "teleop exited 0" in log and f"ready: open {station.http_url()}" in log
    assert all((session.path / "ready" / f"{n}.ready").exists() for n in ORDER)


def test_teleop_exit_1_is_a_failure_and_names_it(station, capsys):
    fatal = "FATAL teleop: PC not answering: no reply to c3.heartbeat/v1 within 0.3 s: check franka_control"
    processes = build(station, serve_all(teleop=("exit", "--ready", "--delay", "0.2", "--fatal", fatal)), [])
    assert Launcher(station, False, processes).run() == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith(f"teleop_collect: FAILED teleop (exit 1): {fatal}  [log: ")
    assert last.endswith("logs/teleop.log]")


# --------------------------------------------------------------------------------------------------- the page


@pytest.mark.parametrize("open_browser,ui,opened", [
    (True, {}, True),
    (False, {}, False),                                               # --no-browser
    (True, {"open_browser": False}, False),                           # ui.open_browser: false
])
def test_the_browser_opens_unless_told_not_to(make_station, capsys, browser, open_browser, ui, opened):
    station = Station.load(make_station(launcher=FAST, ui=ui), False)
    processes = build(station, serve_all(teleop=TELEOP_DONE), [])
    assert Launcher(station, False, processes, open_browser=open_browser).run() == 0
    assert f"teleop_collect: ready: open {station.http_url()}" in capsys.readouterr().out
    assert browser == ([station.http_url()] if opened else [])


@pytest.mark.parametrize("display,answer,line", [
    (False, True, "teleop_collect: no display: open the page by hand"),
    (True, False, "teleop_collect: could not open a browser: open the page by hand"),
    (True, "raise", "teleop_collect: could not open a browser: open the page by hand"),
])
def test_no_browser_is_one_line_and_the_run_goes_on(station, capsys, monkeypatch, display, answer, line):
    calls: list = []

    def fake_open(url, *a, **k):
        calls.append(url)
        if answer == "raise":
            raise launcher_module.webbrowser.Error("no runnable browser")
        return answer

    monkeypatch.setattr(launcher_module.webbrowser, "open", fake_open)
    monkeypatch.setattr(launcher_module, "_has_display", lambda: display)
    processes = build(station, serve_all(teleop=TELEOP_DONE), [])
    assert Launcher(station, False, processes).run() == 0
    assert line in capsys.readouterr().out.splitlines()
    assert calls == ([] if not display else [station.http_url()])


def test_has_display_from_the_environment(monkeypatch):
    monkeypatch.undo()                                                # the real function (autouse patched it)
    monkeypatch.setattr(launcher_module.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert launcher_module._has_display() is False
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert launcher_module._has_display() is True


@pytest.mark.parametrize("mock,on_start,note", [
    (False, True, "teleop_collect: teleop: start-up checks, then the arm homes once (a slow move of all 7 joints): "
                  "keep a hand on the user stop"),
    (True, True, "teleop_collect: teleop: start-up checks, then the fake PC homes once"),
    (False, False, None),
])
def test_the_teleop_start_note(make_station, mock, on_start, note):
    station = Station.load(make_station(teleop={"home": {"on_start": on_start}}), mock)
    launcher = Launcher(station, mock)
    teleop = next(p for p in launcher.processes if p.NAME == "teleop")
    assert isinstance(teleop, TeleopProcess)
    assert launcher._start_note(teleop) == note
    assert all(launcher._start_note(p) is None for p in launcher.processes if p is not teleop)


# --------------------------------------------------------------------------------------------- the processes


@pytest.mark.parametrize("mock,fake_pc,names", [
    (False, True, ["frames", "teleop", "ui"]),
    (True, True, ["fake_pc", "frames", "teleop", "ui"]),
    (True, False, ["frames", "teleop", "ui"]),
    (False, False, ["frames", "teleop", "ui"]),
])
def test_the_fake_pc_only_with_mock_and_its_switch(make_station, mock, fake_pc, names):
    station = Station.load(make_station(launcher={"fake_pc": fake_pc}), mock)
    launcher = Launcher(station, mock)
    assert [p.NAME for p in launcher.processes] == names
    teleop = launcher.processes[-2]
    assert isinstance(teleop, TeleopProcess) and teleop.fake_pc is fake_pc


def test_settings_from_config_and_station(make_station):
    launcher = Launcher(Station.load(make_station(launcher={"ready_timeout_s": {"ui": 12.0}}), True), True)
    s = launcher.settings
    assert s.ready_timeout_s == {"fake_pc": 10.0, "frames": 30.0, "teleop": 150.0, "ui": 12.0}
    assert (s.stop_timeout_s, s.poll_s, s.fake_pc) == (5.0, 0.1, True)
