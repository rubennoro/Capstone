"""Shared fixtures: a station-file factory, and a temporary deployment/ root holding small config.yaml files."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import deploy
from helpers import CONFIGS, REAL_GRIPPER, REAL_RIG, station_data, write_yaml


@pytest.fixture
def make_station(tmp_path):
    """make_station(**sections) -> path of a station file in tmp_path (see station_data)."""
    counter = iter(range(1000))

    def make(**sections) -> Path:
        return write_yaml(tmp_path / "stations" / f"station_{next(counter)}.yaml", station_data(**sections))

    return make


@pytest.fixture
def fake_root(tmp_path, monkeypatch) -> Path:
    """A temporary deployment/ root (deploy.DEPLOYMENT_ROOT patched to it) with a config.yaml per process, copies of
    the real gripper.yaml and stand-in rig file, and station.yaml."""
    root = tmp_path / "deployment"
    for name, config in CONFIGS.items():
        write_yaml(root / name / "config.yaml", config)
    (root / "process_frames" / "rigs").mkdir(parents=True)
    shutil.copyfile(REAL_RIG, root / "process_frames" / "rigs" / "rig_pi_v1_standin.yaml")
    shutil.copyfile(REAL_GRIPPER, root / "gripper.yaml")
    write_yaml(root / "station.yaml", station_data(launcher={"poll_s": 0.02}))
    monkeypatch.setattr(deploy, "DEPLOYMENT_ROOT", root)
    return root
