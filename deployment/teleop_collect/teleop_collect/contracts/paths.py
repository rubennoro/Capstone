# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/contracts/paths.py @ 2ffe641
# changes: teleop_collect (folder deployment/teleop_collect/, its install hint); CONFIG_NAMES are this package's
# frames, teleop, ui, fake_pc and launcher.
"""Where teleop_collect's files are (STRUCTURE.md 2, 4): config/, rigs/ and station.yaml next to the package.

They are found from the editable install: folder() = deployment/teleop_collect/. A non-editable install has no
config/ next to the package, and folder() says how to fix it.
"""
from __future__ import annotations

from pathlib import Path

import teleop_collect

INSTALL_HINT = ('teleop_collect must be installed editable from deployment/teleop_collect: '
                '~/envs/teleop_collect/bin/pip install -e "deployment/teleop_collect[test]"')
CONFIG_NAMES = ("frames", "teleop", "ui", "fake_pc", "launcher")


def _package_file() -> Path:
    """teleop_collect/__init__.py; a function so a test can point it somewhere else."""
    return Path(teleop_collect.__file__)


def folder() -> Path:
    """deployment/teleop_collect/: holds config/, rigs/, station.yaml and the package. FileNotFoundError with the
    install hint if config/ is not there (not an editable install)."""
    root = _package_file().resolve().parents[1]
    if not (root / "config").is_dir():
        raise FileNotFoundError(f"{INSTALL_HINT} (no config/ in {root})")
    return root


def config_file(name: str) -> Path:
    """config/<name>.yaml; the frames process's file is config/frames_process.yaml (never confused with
    deployment/contracts/frames.yaml, the TCP)."""
    if name not in CONFIG_NAMES:
        raise ValueError(f"unknown config {name!r}; known: {', '.join(CONFIG_NAMES)}")
    return folder() / "config" / ("frames_process.yaml" if name == "frames" else f"{name}.yaml")


def rigs_dir() -> Path:
    return folder() / "rigs"


def default_station() -> Path:
    return folder() / "station.yaml"


def deployment_dir() -> Path:
    """deployment/ (the git checkout's folder of both machines), for `git rev-parse`."""
    return folder().parent
