"""Settings: config.yaml + the station's `policy:` section, validated (STRUCTURE.md 4.9).

The merge rule is deploy/STRUCTURE.md 6.3's, restated here because this package cannot import `deploy`: maps merge
key by key, every other value is replaced, and an override key that is not in config.yaml is an error with its path.
"""

from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

import policy
from policy.errors import SettingsError
from policy.policy_factory import KINDS

POLICY_DIR = Path(policy.__file__).resolve().parents[1]     # deployment/policy/
CONFIG_FILE = POLICY_DIR / "config.yaml"
SECTION = "policy"
_SESSION_STAMP = "%Y-%m-%d_%H-%M-%S"


def merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any], prefix: str,
          problems: list[str], source: str) -> dict[str, Any]:
    """Merge `overrides` over `defaults`: maps key by key, everything else replaced; unknown keys become problems."""
    merged = copy.deepcopy(dict(defaults))
    for key, value in overrides.items():
        path = f"{prefix}.{key}"
        if key not in defaults:
            problems.append(f"{source}: {path}: not a key of policy/config.yaml")
        elif isinstance(defaults[key], Mapping) and isinstance(value, Mapping):
            merged[key] = merge(defaults[key], value, path, problems, source)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_path(value: str, from_station: bool, station_dir: Path, policy_dir: Path = POLICY_DIR) -> Path:
    """`~` expanded; a relative path from the station file is taken from its folder, a default from policy/."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (station_dir if from_station else policy_dir) / path
    return path.resolve()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Settings:
    kind: str | None
    mock: bool
    active_kind: str                       # "hold_still" if mock else kind
    endpoint: str                          # endpoints.policy from the station file
    session_dir: Path                      # --session DIR / "policy", or <session_root>/<ts>_policy_alone/policy
    station_file: Path
    checkpoint: Path | None
    device: str
    diffusion_steps: int | None
    warmup_inferences: int
    warmup_gripper_width_m: float
    reset_max_s: float
    poll_s: float
    self_check_timeout_s: float
    training_timing: Mapping[str, int]
    key_map: Mapping[str, Mapping[str, str]]

    @classmethod
    def load(cls, station_file: Path, session: Path | None, mock: bool) -> "Settings":
        station_file = Path(station_file).expanduser().resolve()
        source = station_file.name
        station = _read_yaml(station_file, source)
        defaults = _read_yaml(CONFIG_FILE, "policy/config.yaml")
        problems: list[str] = []

        section = station.get(SECTION)
        if section is None:
            section = {}
        if not isinstance(section, Mapping):
            problems.append(f"{source}: {SECTION}: expected a map, got {type(section).__name__}")
            section = {}
        cfg = merge(defaults, section, SECTION, problems, source)

        endpoint = _station_endpoint(station, source, problems)
        session_dir = _session_dir(station, station_file, session, source, problems)

        kind = cfg.get("kind")
        active_kind = "hold_still" if mock else kind
        kind_names = " or ".join(sorted(KINDS))
        if kind is None:
            if not mock:
                problems.append(f"{SECTION}.kind is not set: choose {kind_names} in {source} ({SECTION}.kind)")
        elif not isinstance(kind, str) or kind not in KINDS:
            problems.append(f"{source}: {SECTION}.kind: unknown kind {kind!r} (choose {kind_names})")

        checkpoint = cfg.get("checkpoint")
        checkpoint_path: Path | None = None
        if checkpoint is not None:
            if isinstance(checkpoint, str) and checkpoint:
                checkpoint_path = resolve_path(checkpoint, "checkpoint" in section, station_file.parent)
            else:
                problems.append(f"{source}: {SECTION}.checkpoint: expected a path or null, got {checkpoint!r}")
        elif active_kind == "diffusion":
            problems.append(f"{source}: {SECTION}.checkpoint: required for kind diffusion "
                            f"(the trained .ckpt file)")

        _check(problems, source, "device", cfg.get("device"), cfg.get("device") in ("cuda", "cpu"), "cuda or cpu")
        steps = cfg.get("diffusion_steps")
        _check(problems, source, "diffusion_steps", steps, steps is None or (_is_int(steps) and steps >= 1),
               "an integer >= 1 or null")
        warmups = cfg.get("warmup_inferences")
        _check(problems, source, "warmup_inferences", warmups, _is_int(warmups) and warmups >= 1, "an integer >= 1")
        width = cfg.get("warmup_gripper_width_m")
        _check(problems, source, "warmup_gripper_width_m", width, _is_number(width) and width >= 0, "a number >= 0")
        for key in ("reset_max_s", "poll_s", "self_check_timeout_s"):
            value = cfg.get(key)
            _check(problems, source, key, value, _is_number(value) and value > 0, "a number > 0")

        timing = cfg.get("training_timing")
        if isinstance(timing, Mapping):
            for key, value in timing.items():
                _check(problems, source, f"training_timing.{key}", value, _is_int(value), "an integer")
        else:
            _check(problems, source, "training_timing", timing, False, "a map")
        key_map = cfg.get("key_map")
        if isinstance(key_map, Mapping):
            for part in ("obs", "action"):
                names = key_map.get(part)
                if not isinstance(names, Mapping):
                    _check(problems, source, f"key_map.{part}", names, False, "a map")
                    continue
                for field, model_name in names.items():
                    _check(problems, source, f"key_map.{part}.{field}", model_name,
                           isinstance(model_name, str) and bool(model_name), "a non-empty string")
        else:
            _check(problems, source, "key_map", key_map, False, "a map")

        if problems:
            raise SettingsError(problems)
        return cls(
            kind=kind, mock=mock, active_kind=active_kind, endpoint=endpoint, session_dir=session_dir,
            station_file=station_file, checkpoint=checkpoint_path, device=cfg["device"], diffusion_steps=steps,
            warmup_inferences=warmups, warmup_gripper_width_m=float(width), reset_max_s=float(cfg["reset_max_s"]),
            poll_s=float(cfg["poll_s"]), self_check_timeout_s=float(cfg["self_check_timeout_s"]),
            training_timing=dict(timing), key_map={part: dict(key_map[part]) for part in ("obs", "action")},
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy (paths as strings), for startup.json."""
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, Mapping):
                value = copy.deepcopy(dict(value))
            out[name] = value
        return out


def _check(problems: list[str], source: str, key: str, value: Any, ok: bool, expected: str) -> None:
    if not ok:
        problems.append(f"{source}: {SECTION}.{key}: expected {expected}, got {value!r}")


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise SettingsError([f"{label}: file not found: {path}"]) from None
    except (OSError, yaml.YAMLError) as e:
        raise SettingsError([f"{label}: cannot read {path}: {e}"]) from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SettingsError([f"{label}: expected a map at the top level, got {type(data).__name__}"])
    return data


def _station_endpoint(station: Mapping[str, Any], source: str, problems: list[str]) -> str:
    endpoints = station.get("endpoints")
    value = endpoints.get("policy") if isinstance(endpoints, Mapping) else None
    if value is None:
        problems.append(f"{source}: endpoints.policy: missing")
        return ""
    if not isinstance(value, str) or not value.startswith("tcp://"):
        problems.append(f"{source}: endpoints.policy: expected \"tcp://host:port\", got {value!r}")
        return ""
    return value


def _session_dir(station: Mapping[str, Any], station_file: Path, session: Path | None, source: str,
                 problems: list[str]) -> Path:
    if session is not None:
        return Path(session).expanduser().resolve() / SECTION
    root = station.get("session_root")
    if root is None:
        problems.append(f"{source}: session_root: missing (needed when policy is started without --session)")
        return Path()
    if not isinstance(root, str) or not root:
        problems.append(f"{source}: session_root: expected a path, got {root!r}")
        return Path()
    stamp = datetime.datetime.now().strftime(_SESSION_STAMP)
    return resolve_path(root, True, station_file.parent) / f"{stamp}_{SECTION}_alone" / SECTION
