# adapted from rnoro/deployment:deployment/deploy/tests/test_cli.py @ 91b178b
# changes: command name policy_deploy; the default station is deployment/policy_deploy/station.yaml; a station error
# is the failing station.yaml row(s) of the table; the namespace-shadowing cases are gone.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import policy_deploy.launcher.cli as cli
from policy_deploy.contracts import paths


@pytest.mark.parametrize("command", ["check", "run"])
def test_commands_parse(command):
    args = cli.parse_args([command, "--station", "s.yaml", "--mock"])
    assert (args.command, args.station, args.mock) == (command, Path("s.yaml"), True)
    args = cli.parse_args([command])
    assert (args.station, args.mock) == (None, False)


@pytest.mark.parametrize("argv", [[], ["start"], ["check", "--verbose"], ["run", "--session", "x"],
                                  ["check", "extra"]])
def test_only_check_run_station_and_mock(argv):
    with pytest.raises(SystemExit) as info:
        cli.parse_args(argv)
    assert info.value.code == 2


def test_station_defaults_to_policy_deploy_station_yaml(monkeypatch):
    seen = {}

    class FakeLauncher:
        def __init__(self, station, mock):
            seen["path"], seen["mock"] = station.path, mock

        def check(self):
            return 0

    monkeypatch.setattr(cli, "Launcher", FakeLauncher)
    assert cli.main(["check", "--mock"]) == 0
    assert seen == {"path": paths.default_station(), "mock": True}


@pytest.mark.parametrize("command,last", [("run", "policy_deploy: not started: 2 check(s) failed (see above)"),
                                          ("check", "policy_deploy check: 2 check(s) failed")])
def test_station_error_is_the_failing_station_row(make_station, capsys, command, last):
    path = make_station(station_version=3, bogus=1)
    assert cli.main([command, "--station", str(path)]) == 2
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == last
    rows = [line for line in out[2:-1]]
    assert all(line.split()[:3] == ["station", "station.yaml", "FAIL"] for line in rows)
    text = "\n".join(rows)
    assert "station_version: expected 1, got 3" in text and "bogus: unknown key" in text


def test_run_dispatches(monkeypatch):
    class FakeLauncher:
        def __init__(self, station, mock):
            assert mock

        def run(self):
            return 130

    monkeypatch.setattr(cli, "Launcher", FakeLauncher)
    assert cli.main(["run", "--mock"]) == 130


def test_python_dash_m_policy_deploy(tmp_path):
    done = subprocess.run([sys.executable, "-m", "policy_deploy", "check", "--station", str(tmp_path / "nope.yaml")],
                          capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert done.returncode == 2 and "nope.yaml: not found" in done.stdout


def test_console_script(tmp_path):
    script = Path(sys.executable).parent / "policy_deploy"
    if not script.exists():
        pytest.skip("the policy_deploy console script is not installed in this environment")
    done = subprocess.run([str(script), "check", "--station", str(tmp_path / "nope.yaml")], cwd=paths.folder(),
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "nope.yaml: not found" in done.stdout
