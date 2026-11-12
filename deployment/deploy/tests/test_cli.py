from __future__ import annotations

from pathlib import Path

import pytest

import deploy.cli as cli
from helpers import REAL_ROOT


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


def test_station_defaults_to_deployment_station_yaml(fake_root, monkeypatch):
    seen = {}

    class FakeLauncher:
        def __init__(self, station, mock):
            seen["path"], seen["mock"] = station.path, mock

        def check(self):
            return 0

    monkeypatch.setattr(cli, "Launcher", FakeLauncher)
    assert cli.main(["check"]) == 0
    assert seen == {"path": fake_root / "station.yaml", "mock": False}


def test_station_error_exits_2_with_its_list(make_station, capsys):
    path = make_station(station_version=3, bogus=1)
    assert cli.main(["run", "--station", str(path)]) == 2
    err = capsys.readouterr().err
    assert "station_version: expected 1, got 3" in err and "bogus: unknown key" in err


def test_run_dispatches(fake_root, monkeypatch):
    class FakeLauncher:
        def __init__(self, station, mock):
            assert mock

        def run(self):
            return 130

    monkeypatch.setattr(cli, "Launcher", FakeLauncher)
    assert cli.main(["run", "--mock"]) == 130


def test_python_dash_m_deploy(fake_root):
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-m", "deploy", "check", "--station", str(fake_root / "nope.yaml")],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "nope.yaml: not found" in done.stderr


def test_deploy_command_works_from_deployment_folder(fake_root):
    """The console script does not put the working folder on the path, so it works from deployment/ too."""
    import subprocess
    import sys
    script = Path(sys.executable).parent / "deploy"
    if not script.exists():
        pytest.skip("the deploy console script is not installed in this environment")
    done = subprocess.run([str(script), "check", "--station", str(fake_root / "nope.yaml")], cwd=REAL_ROOT,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "nope.yaml: not found" in done.stderr


@pytest.mark.parametrize("argv,code", [(["deploy", "check"], 2), (["deploy.geometry", "--dry-run"], 1)])
def test_python_dash_m_from_deployment_folder_explains(argv, code):
    """From deployment/, `python -m deploy` sees the folder deployment/deploy/ first: it says so instead of crashing."""
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-m", *argv], cwd=REAL_ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == code
    assert "run from inside deployment/ finds the folder deployment/deploy/" in done.stderr
