import subprocess
import sys
from pathlib import Path

import pytest

from dataproc import cli

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path))


@pytest.mark.parametrize("cmd", ["intake", "process", "status"])
def test_task_is_required(cmd, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([cmd])
    assert e.value.code == 2
    assert "--task" in capsys.readouterr().err


def test_subcommand_required(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2


@pytest.mark.parametrize("argv", [
    ["intake", "--task", "bad name"],
    ["intake", "--task", "block_raise/episode_2"],
    ["status", "--task", "block_raise/episode_2"],
    ["process", "--task", "/elsewhere/block_raise"],
    ["process", "--task", "block_raise/episode_2", "--episode", "3"],
    ["process", "--task", "block_raise", "--episode", "-1"],
])
def test_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("argv, wave", [
    (["intake", "--task", "block_raise", "--new", "--calibration"], "wave 1"),
    (["process", "--task", "block_raise", "--episode", "2", "--force"], "wave 2"),
    (["status", "--task", "block_raise"], "wave 2"),
    (["sim", "--out", "x", "--episodes", "2", "--calibration"], "wave 1"),
])
def test_stubs(argv, wave, capsys):
    assert cli.main(argv) == 1
    assert wave in capsys.readouterr().err


def test_process_episode_from_task_spec(monkeypatch, tmp_path):
    seen = {}

    def fake(args, root, task):
        seen.update(episode=args.episode, folder=task.folder, root=root.path)
        return 0

    monkeypatch.setattr(cli, "cmd_process", fake)
    assert cli.main(["process", "--task", "block_raise/episode_4"]) == 0
    assert seen == {"episode": 4, "folder": "block_raise", "root": tmp_path}
    assert cli.main(["process", "--task", str(tmp_path / "tasks" / "block_raise_1"), "--episode", "0"]) == 0
    assert seen["episode"] == 0 and seen["folder"] == "block_raise_1"


def test_python_m_entry_point():
    r = subprocess.run([sys.executable, "-m", "dataproc", "--version"], cwd=PKG, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("dataproc ")


@pytest.mark.parametrize("argv, usage", [
    (["intake", "--task", "br/episode_2"], "usage: dataproc intake"),
    (["process", "--task", "br/episode_01"], "usage: dataproc process"),
    (["status", "--task", "/elsewhere/x"], "usage: dataproc status"),
])
def test_usage_errors_use_the_subcommand_parser(argv, usage, capsys):
    with pytest.raises(SystemExit):
        cli.main(argv)
    err = capsys.readouterr().err
    assert err.startswith(usage) and f"dataproc {argv[0]}: error:" in err
