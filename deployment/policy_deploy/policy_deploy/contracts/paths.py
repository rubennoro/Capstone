"""Where policy_deploy's files are (STRUCTURE.md 2.1, 2.3): config/, rigs/ and station.yaml next to the package.

They are found from the editable install: folder() = deployment/policy_deploy/. A non-editable install has no config/
next to the package, and folder() says how to fix it.
"""
from __future__ import annotations

from pathlib import Path

import policy_deploy

INSTALL_HINT = ('policy_deploy must be installed editable from deployment/policy_deploy: '
                'uv pip install -e "deployment/policy_deploy[light,test]"')
CONFIG_NAMES = ("frames", "policy", "manager", "fake_pc", "launcher")


def _package_file() -> Path:
    """policy_deploy/__init__.py; a function so a test can point it somewhere else."""
    return Path(policy_deploy.__file__)


def folder() -> Path:
    """deployment/policy_deploy/: holds config/, rigs/, station.yaml and the package. FileNotFoundError with the
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
