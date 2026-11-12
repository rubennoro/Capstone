"""A process's config.yaml merged with its station.yaml section (STRUCTURE.md section 6.3)."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

import deploy
from deploy.errors import StationError

if TYPE_CHECKING:
    from deploy.station import Station


def load_process_config(folder: Path, overrides: Mapping[str, Any]) -> dict:
    """folder/config.yaml with `overrides` merged over it: maps key by key, everything else replaced.

    Every override key must already exist in config.yaml; all unknown keys are listed in one StationError.
    """
    name = folder.name
    path = folder / "config.yaml"
    try:
        base = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise StationError(f"{name}/config.yaml: not found: {path}") from None
    except (OSError, yaml.YAMLError) as exc:
        raise StationError(f"{name}/config.yaml: cannot read {path}: {exc}") from None
    if base is None:
        base = {}
    if not isinstance(base, dict):
        raise StationError(f"{name}/config.yaml: expected a map of settings at the top level")
    problems: list[str] = []
    merged = _merge(copy.deepcopy(base), overrides or {}, name, name, problems)
    if problems:
        raise StationError(problems)
    return merged


def _merge(base: dict, overrides: Mapping[str, Any], key_path: str, name: str, problems: list[str]) -> dict:
    for key, value in overrides.items():
        path = f"{key_path}.{key}"
        if key not in base:
            problems.append(f"station.yaml: {path}: not a key of {name}/config.yaml")
        elif isinstance(base[key], dict):
            if isinstance(value, dict):
                _merge(base[key], value, path, name, problems)
            else:
                problems.append(f"station.yaml: {path}: expected a map (as in {name}/config.yaml), got {value!r}")
        else:
            base[key] = copy.deepcopy(value)
    return base


def resolve_path(station: "Station", process: str, config: Mapping[str, Any], key: str) -> Path:
    """The path setting `key`: a station-set value against the station file's folder, a config.yaml default against
    DEPLOYMENT_ROOT / process. Raises StationError if the value is not set."""
    value = config.get(key)
    if value is None:
        raise StationError(f"{process}.{key} is not set (set it in station.yaml or {process}/config.yaml)")
    if key in station.section(process):
        return station.resolve(value)
    p = Path(value).expanduser()
    return p if p.is_absolute() else deploy.DEPLOYMENT_ROOT / process / p
