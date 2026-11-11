"""The README practice run through the CLI: default (Pi-format) ``dataproc sim`` recordings go through
``calibrate`` -> ``run --slam oracle`` -> ``package``; an IMU-dropout recording ends in ``failed/``; a second
calibration makes ``block_raise_2``; ``run`` on an unknown folder says to calibrate first."""

from __future__ import annotations

import json

import pytest

from dataproc import cli
from dataproc.paths import data_root

RIG = "rig_v1_fast"


@pytest.fixture(scope="module")
def practice(tmp_path_factory):
    base = tmp_path_factory.mktemp("practice")
    mp = pytest.MonkeyPatch()
    mp.setenv("DATAPROC_ROOT", str(base / "root"))
    src = str(base / "src")
    try:
        assert cli.main(["sim", "--out", src, "--episodes", "1", "--calibration", "--fast"]) == 0   # 000000
        assert cli.main(["sim", "--out", src, "--episodes", "2", "--fast"]) == 0                     # 000001-2
        assert cli.main(["sim", "--out", src, "--episodes", "1", "--fast", "--imu-gap", "0.5"]) == 0  # 000003
        assert cli.main(["sim", "--out", src, "--episodes", "1", "--calibration", "--fast"]) == 0   # 000004
        codes = {
            "calibrate": cli.main(["calibrate", "block_raise", "episode_000000", "--source", src, "--rig", RIG]),
            "run": cli.main(["run", "block_raise", "episode_000001", "episode_000003", "--source", src,
                             "--slam", "oracle"]),
            "package": cli.main(["package", "practice", "block_raise"]),
            "calibrate_2": cli.main(["calibrate", "block_raise", "episode_000004", "--source", src, "--rig", RIG]),
            "rerun": cli.main(["run", "block_raise", "episode_000001", "episode_000003", "--source", src,
                               "--slam", "oracle"]),
        }
        yield data_root(), codes
    finally:
        mp.undo()


def test_codes(practice):
    _, codes = practice
    assert codes == {"calibrate": 0, "run": 0, "package": 0, "calibrate_2": 0, "rerun": 0}


def test_folders(practice):
    root, _ = practice
    t = root.tasks / "block_raise"
    assert sorted(p.name for p in root.tasks.iterdir()) == ["block_raise", "block_raise_2"]
    assert sorted(p.name for p in t.glob("episode_*")) == ["episode_1", "episode_2"]
    for k in (1, 2):
        assert (t / f"episode_{k}" / "processed" / "episode.zarr").is_dir()
    failed = [p.name for p in (t / "failed").iterdir()]
    assert len(failed) == 1 and failed[0].endswith("_episode_000003")
    reason = json.loads((t / "failed" / failed[0] / "reason.json").read_text())
    assert reason["stage"] == "validate" and "imu_gap" in reason["reasons"]
    assert not (t / ".staging").exists() or not any((t / ".staging").iterdir())
    assert len(list((root.tasks / "block_raise_2" / "calibration").iterdir())) == 1
    assert not list((root.tasks / "block_raise_2").glob("episode_*"))


def test_package(practice):
    root, _ = practice
    d = root.path / "datasets" / "block_raise" / "practice"
    assert (d / "practice.zarr.zip").is_file()
    rec = json.loads((d / "package.json").read_text())
    assert rec["totals"]["episodes"] == 2 and [f["folder"] for f in rec["folders"]] == ["block_raise"]


def test_unknown_folder(practice, monkeypatch, capsys):
    root, _ = practice
    monkeypatch.setenv("DATAPROC_ROOT", str(root.path))
    assert cli.main(["run", "nope", "a", "b"]) == 1
    assert "dataproc calibrate" in capsys.readouterr().err
