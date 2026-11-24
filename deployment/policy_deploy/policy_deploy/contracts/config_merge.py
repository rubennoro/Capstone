# adapted from rnoro/deployment:deployment/deploy/deploy/config_merge.py @ 91b178b
# changes: one function for every laptop process and the launcher, load_config(name, station), reading
# config/<name>.yaml (config/frames_process.yaml for frames); the contract key names are refused in the merged
# result; resolve_path takes a dotted key; relative defaults are taken from deployment/policy_deploy/.
"""A process's config/<name>.yaml merged with its station.yaml section (STRUCTURE.md 2.5).

Merge rule: maps merge key by key, every other value is replaced, and an override key that is not in the config file
is an error with its key path. Relative paths: a station value against the station file's folder, a default against
deployment/policy_deploy/.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from policy_deploy.contracts import paths
from policy_deploy.contracts.station import StationError, refused_keys

if TYPE_CHECKING:
    from policy_deploy.contracts.station import Station


def config_label(name: str) -> str:
    """e.g. "config/frames_process.yaml"."""
    return f"config/{paths.config_file(name).name}"


def read_config(name: str, path: Path | None = None) -> dict:
    """config/<name>.yaml as a dict (path: a test's replacement file)."""
    path = paths.config_file(name) if path is None else Path(path)
    label = config_label(name)
    try:
        base = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StationError(f"{label}: not found: {path}") from None
    except (OSError, yaml.YAMLError) as exc:
        raise StationError(f"{label}: cannot read {path}: {exc}") from None
    if base is None:
        base = {}
    if not isinstance(base, dict):
        raise StationError(f"{label}: expected a map of settings at the top level")
    return base


def load_config(name: str, station: Station | None, path: Path | None = None) -> dict:
    """config/<name>.yaml with station.section(name) merged over it. Every problem (an override key not in the file,
    a map replaced by a value, a contract key name anywhere) is listed in one StationError."""
    base = read_config(name, path)
    label = config_label(name)
    problems: list[str] = []
    for key_path, file in refused_keys(base):
        problems.append(f"{label}: {key_path}: belongs to {file}")
    overrides = station.section(name) if station is not None else {}
    station_name = station.path.name if station is not None else "station.yaml"
    merged = _merge(copy.deepcopy(base), overrides or {}, name, label, station_name, problems)
    if problems:
        raise StationError(problems)
    return merged


def _merge(base: dict, overrides: Mapping[str, Any], key_path: str, label: str, station_name: str,
           problems: list[str]) -> dict:
    for key, value in overrides.items():
        path = f"{key_path}.{key}"
        if key not in base:
            problems.append(f"{station_name}: {path}: not a key of {label}")
        elif isinstance(base[key], dict):
            if isinstance(value, dict):
                _merge(base[key], value, path, label, station_name, problems)
            else:
                problems.append(f"{station_name}: {path}: expected a map (as in {label}), got {value!r}")
        else:
            base[key] = copy.deepcopy(value)
    return base


def get_path(config: Mapping[str, Any], key: str) -> Any:
    """config["a"]["b"] for key "a.b"; KeyError if absent."""
    node: Any = config
    for part in key.split("."):
        node = node[part]
    return node


def from_station(station: Station | None, name: str, key: str) -> bool:
    """True if station.yaml's <name>: section sets the dotted key."""
    if station is None:
        return False
    node: Any = station.section(name)
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
    return True


def resolve_path(station: Station | None, name: str, config: Mapping[str, Any], key: str) -> Path | None:
    """The path setting `key` (dotted): a station-set value against the station file's folder, a config default
    against deployment/policy_deploy/; `~` expanded. None if the value is null."""
    value = get_path(config, key)
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    if from_station(station, name, key):
        return station.path.parent / p                    # type: ignore[union-attr]
    return paths.folder() / p
