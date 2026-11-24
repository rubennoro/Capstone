"""Where things are (control_deploy/STRUCTURE.md 2): this folder, franka_control/, the binary, config.yaml, the repo.

Found from the package itself, so control_deploy must be installed editable from deployment/control_deploy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import control_deploy


def deploy_dir() -> Path:
    """deployment/control_deploy/ (the folder holding pyproject.toml, station.yaml and franka_control/)."""
    folder = Path(control_deploy.__file__).resolve().parents[1]
    if not (folder / "franka_control").is_dir():
        raise FileNotFoundError(f"control_deploy must be installed editable from deployment/control_deploy "
                                f"(no franka_control/ next to {folder}): uv pip install -e deployment/control_deploy")
    return folder


def franka_control_dir() -> Path:
    return deploy_dir() / "franka_control"


def config_file() -> Path:
    """franka_control/config.yaml: every arm setting."""
    return franka_control_dir() / "config.yaml"


def default_station() -> Path:
    """control_deploy/station.yaml: this PC's settings."""
    return deploy_dir() / "station.yaml"


def deployment_dir() -> Path:
    return deploy_dir().parent


def contracts_dir() -> Path:
    """deployment/contracts/ of this checkout (what deploy_contracts must be installed from)."""
    return deployment_dir() / "contracts"


def repo_root() -> Path:
    return deployment_dir().parent


def load_config() -> dict[str, Any]:
    with open(config_file(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def binary(station_overrides: dict[str, Any] | None = None) -> Path:
    """The built franka_control: config.yaml `binary` (or the station's franka_control.binary), relative to
    franka_control/."""
    name = (station_overrides or {}).get("binary") or load_config()["binary"]
    path = Path(str(name)).expanduser()
    return path if path.is_absolute() else franka_control_dir() / path
