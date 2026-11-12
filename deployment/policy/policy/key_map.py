"""KeyMap: C4 field names <-> the model's own input/output names (STRUCTURE.md 4.7). numpy only."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from policy.errors import PolicyLoadError

# The C4 field names and per-step sizes (fixed by the contract; contracts/STRUCTURE.md 4.6).
LOW_DIM_SIZES: Mapping[str, int] = {"eef_pos": 3, "eef_rot6d": 6, "eef_rot6d_wrt_start": 6, "gripper_width_m": 1}
IMAGE_FIELDS: tuple[str, ...] = ("main", "ultrawide")
OBS_FIELDS: tuple[str, ...] = (*LOW_DIM_SIZES, *IMAGE_FIELDS)        # Observation field order
ACTION_SIZES: Mapping[str, int] = {"eef_pos": 3, "eef_rot6d": 6, "gripper_width_m": 1}
ACTION_FIELDS: tuple[str, ...] = tuple(ACTION_SIZES)                   # ActionChunk array fields

_CHECK_HINT = "fix policy.key_map in station.yaml (or the defaults in policy/config.yaml)"


class KeyMap:
    def __init__(self, obs: Mapping[str, str], action: Mapping[str, str]) -> None:
        """Both maps are contract field -> model name (obs key / action component type)."""
        self._obs = dict(obs)
        self._action = dict(action)
        self._obs_rev = {model: field for field, model in self._obs.items()}
        self._action_rev = {model: field for field, model in self._action.items()}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "KeyMap":
        """The key_map section of config.yaml: {"obs": {...}, "action": {...}}."""
        return cls(cfg["obs"], cfg["action"])

    def model_key(self, field: str, action: bool = False) -> str:
        """The model's name for a C4 field (an obs key, or with action=True an action component type)."""
        names = self._action if action else self._obs
        if field not in names:
            raise KeyError(f"key_map.{'action' if action else 'obs'}: no entry for C4 field {field!r}")
        return names[field]

    def contract_field(self, model_key: str, action: bool = False) -> str:
        """The C4 field for a model obs key (or with action=True an action component type)."""
        names = self._action_rev if action else self._obs_rev
        if model_key not in names:
            raise KeyError(f"key_map.{'action' if action else 'obs'}: no C4 field maps to {model_key!r}")
        return names[model_key]

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {"obs": dict(self._obs), "action": dict(self._action)}

    def check(self, shape_meta: Mapping) -> None:
        """Raise PolicyLoadError listing every mismatch between this map and the checkpoint's shape_meta."""
        problems: list[str] = []
        model_obs = shape_meta.get("obs", {}) or {}
        for part, names, known in (("obs", self._obs, OBS_FIELDS), ("action", self._action, ACTION_FIELDS)):
            for field in names:
                if field not in known:
                    problems.append(f"key_map.{part}.{field}: unknown C4 field (C4 has {', '.join(known)})")
            seen: dict[str, str] = {}
            for field, model in names.items():
                if model in seen:
                    problems.append(f"key_map.{part}: {seen[model]!r} and {field!r} both map to {model!r} "
                                    f"(the map must be one-to-one)")
                seen.setdefault(model, field)
        for field, model in self._obs.items():
            if model not in model_obs:
                problems.append(f"key_map.obs.{field}: model key {model!r} is not in the checkpoint's shape_meta.obs")
        for model, attr in model_obs.items():
            if not _ignored(attr) and model not in self._obs_rev:
                problems.append(f"shape_meta.obs.{model}: used by the model but no C4 field maps to it "
                                f"(add it to key_map.obs)")

        components = (shape_meta.get("action", {}) or {}).get("components", []) or []
        types = [c.get("type") for c in components]
        for field in ACTION_FIELDS:
            if field not in self._action:
                problems.append(f"key_map.action.{field}: missing")
        for field, model in self._action.items():
            if model not in types:
                problems.append(f"key_map.action.{field}: component {model!r} is not in the checkpoint's "
                                f"shape_meta.action.components")
        for comp in components:
            ctype, size = comp.get("type"), comp.get("size")
            field = self._action_rev.get(ctype)
            if field is None:
                problems.append(f"shape_meta.action.components: {ctype!r} has no C4 field (add it to key_map.action)")
            elif field in ACTION_SIZES and size != ACTION_SIZES[field]:
                problems.append(f"shape_meta.action.components: {ctype!r} has size {size}, "
                                f"C4 {field} has {ACTION_SIZES[field]}")
        if problems:
            raise PolicyLoadError("key map does not fit the checkpoint: " + "; ".join(problems), _CHECK_HINT)

    def used_fields(self, shape_meta: Mapping) -> tuple[str, ...]:
        """The C4 observation fields whose model key the model uses (ignore_by_policy false), in C4 order."""
        model_obs = shape_meta.get("obs", {}) or {}
        return tuple(f for f in OBS_FIELDS
                     if f in self._obs and self._obs[f] in model_obs and not _ignored(model_obs[self._obs[f]]))

    def action_slices(self, shape_meta: Mapping) -> dict[str, slice]:
        """C4 chunk field -> its columns of the model's action vector, from shape_meta.action.components."""
        slices: dict[str, slice] = {}
        start = 0
        for comp in shape_meta["action"]["components"]:
            size = int(comp["size"])
            field = self._action_rev.get(comp["type"])
            if field is not None:
                slices[field] = slice(start, start + size)
            start += size
        return slices

    def split_action(self, action: np.ndarray, slices: Mapping[str, slice]) -> dict[str, np.ndarray]:
        """(..., D) action vectors -> {C4 field: C-contiguous (..., size) array}."""
        return {field: np.ascontiguousarray(action[..., s]) for field, s in slices.items()}

    def join_action(self, parts: Mapping[str, np.ndarray], slices: Mapping[str, slice]) -> np.ndarray:
        """The inverse of split_action (for tests)."""
        width = max(s.stop for s in slices.values())
        first = next(iter(parts.values()))
        out = np.zeros(first.shape[:-1] + (width,), dtype=first.dtype)
        for field, s in slices.items():
            out[..., s] = parts[field]
        return out


def _ignored(attr: Any) -> bool:
    return bool(attr.get("ignore_by_policy", False)) if isinstance(attr, Mapping) else False
