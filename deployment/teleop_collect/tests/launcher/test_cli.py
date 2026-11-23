# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/test_cli.py @ 2ffe641
# changes: command name teleop_collect; the default station is deployment/teleop_collect/station.yaml; + run
# --no-browser (open_browser=False; not an option of check); imports. The source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/tests/test_cli.py @ 91b178b
# changes: command name policy_deploy; the default station is deployment/policy_deploy/station.yaml; a station error
# is the failing station.yaml row(s) of the table; the namespace-shadowing cases are gone.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import teleop_collect.launcher.cli as cli
from teleop_collect.contracts import paths
from teleop_collect.contracts.station import read_storage


@pytest.mark.parametrize("command", ["check", "run"])
def test_commands_parse(command):
    args = cli.parse_args([command, "--station", "s.yaml", "--mock"])
    assert (args.command, args.station, args.mock) == (command, Path("s.yaml"), True)
    args = cli.parse_args([command])
    assert (args.station, args.mock, args.data_root) == (None, False, None)
    assert cli.parse_args([command, "--data-root", "~/x"]).data_root == Path("~/x")


def test_run_no_browser():
    assert cli.parse_args(["run", "--mock", "--no-browser"]).no_browser is True
    assert cli.parse_args(["run"]).no_browser is False


@pytest.mark.parametrize("argv", [[], ["start"], ["check", "--verbose"], ["run", "--session", "x"],
                                  ["check", "extra"], ["check", "--no-browser"]])
def test_only_check_run_station_mock_and_no_browser(argv):
    with pytest.raises(SystemExit) as info:
        cli.parse_args(argv)
    assert info.value.code == 2


def test_station_defaults_to_teleop_collect_station_yaml(monkeypatch):
    seen = {}

    class FakeLauncher:
        def __init__(self, station, mock, **kwargs):
            seen["path"], seen["mock"] = station.path, mock

        def check(self):
            return 0

    monkeypatch.setattr(cli, "Launcher", FakeLauncher)
    assert cli.main(["check", "--mock"]) == 0
    assert seen == {"path": paths.default_station(), "mock": True}


@pytest.mark.parametrize("command,last", [("run", "teleop_collect: not started: 2 check(s) failed (see above)"),
                                          ("check", "teleop_collect check: 2 check(s) failed")])
def test_station_error_is_the_failing_station_row(make_station, capsys, command, last):
    path = make_station(station_version=3, bogus=1)
    assert cli.main([command, "--station", str(path)]) == 2
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == last
    rows = [line for line in out[2:-1]]
    assert all(line.split()[:3] == ["station", "station.yaml", "FAIL"] for line in rows)
    text = "\n".join(rows)
    assert "station_version: expected 1, got 3" in text and "bogus: unknown key" in text


@pytest.mark.parametrize("argv,open_browser", [(["run", "--mock"], True), (["run", "--mock", "--no-browser"], False)])
def test_run_dispatches(monkeypatch, argv, open_browser):
    seen = {}

    class FakeLauncher:
        def __init__(self, station, mock, open_browser=True):
            assert mock
            seen["open_browser"] = open_browser

        def run(self):
            return 1

    monkeypatch.setattr(cli, "Launcher", FakeLauncher)
    assert cli.main(argv) == 1
    assert seen == {"open_browser": open_browser}


def test_python_dash_m_teleop_collect(tmp_path):
    done = subprocess.run([sys.executable, "-m", "teleop_collect", "check", "--station", str(tmp_path / "nope.yaml")],
                          capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert done.returncode == 2 and "nope.yaml: not found" in done.stdout


def test_console_script(tmp_path):
    script = Path(sys.executable).parent / "teleop_collect"
    if not script.exists():
        pytest.skip("the teleop_collect console script is not installed in this environment")
    done = subprocess.run([str(script), "check", "--station", str(tmp_path / "nope.yaml")], cwd=paths.folder(),
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "nope.yaml: not found" in done.stdout


class _Seen:
    """A Launcher stand-in that keeps the station it was given."""

    def __init__(self):
        self.stations = []

    def __call__(self, station, mock, **kwargs):
        self.stations.append(station)
        return self

    def check(self):
        return 0

    def run(self):
        return 0


def test_run_data_root_is_used_and_remembered(make_station, monkeypatch, tmp_path):
    seen = _Seen()
    monkeypatch.setattr(cli, "Launcher", seen)
    path = make_station()
    assert cli.main(["run", "--station", str(path), "--mock", "--data-root", str(tmp_path / "eps")]) == 0
    assert seen.stations[-1].data_root == (tmp_path / "eps").resolve()
    assert read_storage(path) == {"mock_data_root": str((tmp_path / "eps").resolve())}
    for command in ("run", "check"):                                               # later runs: remembered
        assert cli.main([command, "--station", str(path), "--mock"]) == 0
        assert seen.stations[-1].data_root == (tmp_path / "eps").resolve()
        assert seen.stations[-1].data_root_source == "storage.local.yaml"
    assert cli.main(["run", "--station", str(path)]) == 0                           # the real runs keep theirs
    assert seen.stations[-1].data_root == path.parent / "data"


def test_check_data_root_is_used_but_not_remembered(make_station, monkeypatch, tmp_path):
    seen = _Seen()
    monkeypatch.setattr(cli, "Launcher", seen)
    path = make_station()
    assert cli.main(["check", "--station", str(path), "--data-root", str(tmp_path / "eps")]) == 0
    assert seen.stations[-1].data_root == (tmp_path / "eps").resolve()
    assert seen.stations[-1].data_root_source == "--data-root"
    assert read_storage(path) == {}
    assert cli.main(["check", "--station", str(path)]) == 0
    assert seen.stations[-1].data_root == path.parent / "data"


def test_data_root_relative_is_from_the_current_folder(make_station, monkeypatch, tmp_path):
    seen = _Seen()
    monkeypatch.setattr(cli, "Launcher", seen)
    monkeypatch.chdir(tmp_path)
    path = make_station()
    assert cli.main(["check", "--station", str(path), "--data-root", "rel"]) == 0
    assert seen.stations[-1].data_root == (tmp_path / "rel").resolve()


def test_data_root_equal_to_the_other_mode_is_a_station_row_and_not_remembered(make_station, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Launcher", _Seen())
    path = make_station()
    assert cli.main(["run", "--station", str(path), "--mock", "--data-root", str(path.parent / "data")]) == 2
    out = capsys.readouterr().out
    assert "mock_data_root: must differ from data_root" in out and "FAIL" in out
    assert read_storage(path) == {}

