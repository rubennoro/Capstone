"""``dataproc`` argument parsing and dispatch (dataproc/cli.py): each subcommand calls its folder's ``cli_*``
function with the parsed arguments; usage errors exit 2 with the subcommand's usage line."""

import subprocess
import sys
from pathlib import Path

import pytest

from dataproc import cli

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path))


def _usage_error(argv, capsys) -> str:
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    return capsys.readouterr().err


def test_subcommand_required(capsys):
    assert "required" in _usage_error([], capsys)


@pytest.mark.parametrize("old", ["intake", "process"])
def test_removed_subcommands(old, capsys):
    assert "invalid choice" in _usage_error([old, "--task", "block_raise"], capsys)


@pytest.mark.parametrize("argv, usage", [
    (["calibrate"], "usage: dataproc calibrate"),
    (["calibrate", "block_raise"], "usage: dataproc calibrate"),
    (["run", "block_raise", "episode_000001"], "usage: dataproc run"),
    (["package", "practice"], "usage: dataproc package"),
    (["status"], "usage: dataproc status"),
    (["sim"], "usage: dataproc sim"),
])
def test_required_positionals(argv, usage, capsys):
    err = _usage_error(argv, capsys)
    assert usage in err and "the following arguments are required" in err


@pytest.mark.parametrize("argv, msg", [
    (["run", "block_raise", "a", "b", "--slam", "gps"], "invalid choice"),
    (["run", "block_raise", "a", "b", "--task", "x"], "unrecognized arguments"),
    (["run", "--task", "block_raise", "a", "b"], "unrecognized arguments"),
    (["status", "--task", "block_raise"], "unrecognized arguments"),
    (["calibrate", "block_raise", "e", "--calibration-episode", "x"], "unrecognized arguments"),
    (["sim", "--out", "x", "--episodes", "-1"], "must be >= 0"),
    (["sim", "--out", "x", "--duration", "0"], "must be > 0"),
    (["sim", "--out", "x", "--imu-gap", "nan?"], "expected a number"),
    (["sim", "--out", "x", "--unverified"], "--unverified needs --contract"),
])
def test_usage_errors(argv, msg, capsys):
    assert msg in _usage_error(argv, capsys)


def _capture(monkeypatch, module: str, name: str) -> dict:
    seen = {}

    def fake(root, *args, **kw):
        seen.update(root=root.path, args=args, kw=kw)
        return 7

    monkeypatch.setattr(f"{module}.{name}", fake)
    return seen


def test_calibrate_dispatch(monkeypatch, tmp_path):
    seen = _capture(monkeypatch, "dataproc.calibrate.cli", "cli_calibrate")
    assert cli.main(["calibrate", "block_raise", "episode_000001"]) == 7
    assert seen == {"root": tmp_path, "args": ("block_raise", "episode_000001"),
                    "kw": {"source": "/data/incoming", "rig_id": "rig_pi_v1"}}
    assert cli.main(["calibrate", "block_raise", "e2", "--source", "/x", "--rig", "rig_v1_fast"]) == 7
    assert seen["kw"] == {"source": "/x", "rig_id": "rig_v1_fast"}


def test_run_dispatch(monkeypatch, tmp_path):
    seen = _capture(monkeypatch, "dataproc.run.cli", "cli_run")
    assert cli.main(["run", "block_raise_2", "episode_000002", "episode_000009"]) == 7
    assert seen == {"root": tmp_path, "args": ("block_raise_2", "episode_000002", "episode_000009"),
                    "kw": {"source": "/data/incoming", "slam": "auto", "retry_failed": False, "force": False,
                           "dry_run": False}}
    assert cli.main(["run", "f", "a", "b", "--source", "/s", "--slam", "oracle", "--retry-failed", "--force",
                     "--dry-run"]) == 7
    assert seen["kw"] == {"source": "/s", "slam": "oracle", "retry_failed": True, "force": True, "dry_run": True}


def test_package_dispatch(monkeypatch, tmp_path):
    seen = _capture(monkeypatch, "dataproc.package.cli", "cli_package")
    assert cli.main(["package", "practice", "block_raise"]) == 7
    assert seen == {"root": tmp_path, "args": ("practice", ["block_raise"]), "kw": {"overwrite": False}}
    assert cli.main(["package", "mix", "a", "b_2", "c", "--overwrite"]) == 7
    assert seen["args"] == ("mix", ["a", "b_2", "c"]) and seen["kw"] == {"overwrite": True}


def test_status_dispatch(monkeypatch, tmp_path):
    seen = _capture(monkeypatch, "dataproc.status", "cli_status")
    assert cli.main(["status", "block_raise_2"]) == 7
    assert seen == {"root": tmp_path, "args": ("block_raise_2",), "kw": {}}


def test_sim_dispatch(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("dataproc.sim.episode.run_sim", lambda **kw: seen.update(kw))
    assert cli.main(["sim", "--out", "o", "--episodes", "3", "--fast", "--imu-gap", "0.5"]) == 0
    assert seen["pi_format"] is True and seen["imu_gap_s"] == 0.5 and seen["episodes"] == 3 and seen["fast"]
    assert seen["rigs_dir"] == tmp_path / "rigs" and seen["verified"] is True
    assert cli.main(["sim", "--out", "o", "--contract", "--unverified", "--rigs-dir", "r"]) == 0
    assert seen["pi_format"] is False and seen["verified"] is False and seen["rigs_dir"] == "r"


def test_missing_task_folder(capsys):
    assert cli.main(["status", "block_raise"]) == 1
    assert "dataproc calibrate first" in capsys.readouterr().err
    assert cli.main(["run", "nope", "a", "b"]) == 1
    assert "does not exist" in (err := capsys.readouterr().err) and "dataproc calibrate <task>" in err


def test_help_has_no_old_wording(capsys):
    for argv in ([], ["calibrate"], ["run"], ["package"], ["status"], ["sim"]):
        with pytest.raises(SystemExit):
            cli.main([*argv, "--help"])
        out = capsys.readouterr().out
        assert "_1" not in out and "intake" not in out and "--task" not in out, argv


def test_python_m_entry_point():
    r = subprocess.run([sys.executable, "-m", "dataproc", "--version"], cwd=PKG, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("dataproc ")
    r = subprocess.run([sys.executable, "-m", "dataproc", "--help"], cwd=PKG, capture_output=True, text=True)
    assert r.returncode == 0 and "{calibrate,run,package,status,sim}" in r.stdout
