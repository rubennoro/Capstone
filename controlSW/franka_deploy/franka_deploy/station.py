"""Station config loader: one home for every physical and run constant (controlSW/configs/**).

Why: robot, tooling, sensor, scene, host and run settings used to be scattered over bridge.yaml, franka_cartesian.yaml,
deploy.yaml and literals in run_task.py. Here each value lives once, with a unit in its key name and a provenance
entry (measured | placeholder | hw | datasheet | derived | decision), and the legacy config fragments are derived from it.
Imports only the stdlib, numpy and yaml, so it works on the Jetson (no mujoco, no robotmq).

Usage: python -m franka_deploy.station validate [--root R] [--all | --spec station.robot=panda --spec run.link=ideal ...]
       st = load_station({"station": {"robot": "panda", ...}, "run": {"policy": "bp_run001", ...}}, sets=["link.seed=3"])
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path

import numpy as np
import yaml

from .paths import REPO_ROOT

STATION_ROOT = REPO_ROOT / "controlSW" / "configs"

STATION_GROUPS = ("robot", "tooling", "sensor", "scene", "host")
RUN_GROUPS = ("policy", "link", "criteria", "sim", "sweeps")
KINDS = {"station": STATION_GROUPS, "run": RUN_GROUPS}
KIND_OF = {g: k for k, gs in KINDS.items() for g in gs}

STATUSES = ("measured", "placeholder", "hw", "datasheet", "derived", "decision")
PLACEHOLDER_STATUSES = ("placeholder", "hw")
META_KEYS = ("schema", "id", "provenance", "critical")  # not data leaves

# A numeric leaf needs a unit suffix on its own key or on any ancestor key (after stripping _min/_max/_p95 ...).
UNIT_SUFFIXES = (
    "_m", "_rad", "_deg", "_s", "_hz", "_kg", "_n", "_nm", "_m_s", "_rad_s", "_px", "_ppm",
    "_m_s2", "_m_s3", "_rad_s2", "_rad_s3", "_nm_s", "_nm_rad", "_n_m", "_kgm2", "_mbit_s", "_ms", "_mm", "_bytes",
    "_gb", "_fraction", "_ratio", "_scale", "_steps", "_errors",
)
STRIP_TOKENS = ("_min", "_max", "_p95", "_p99", "_p999")
# Numeric leaves allowed without a unit: matched (fnmatch) against every path component and against the dotted path
# (list indices removed).
NON_SUFFIXED_ALLOW = (
    "K", "T_*", "F_T_*", "C_T_*", "*_T_*", "width_map.a", "width_map.b", "inertia_kgm2",   # matrices, dimensionless map
    "n_*", "seed", "version", "*_version", "*_index",                                       # counts, ids, versions
    "port", "ports", "exec_h", "jpeg_quality", "friction", "max_queue", "max_runs",
    "nullspace_stiffness", "filter_coeff", "impratio", "cone_axis",                         # dimensionless
    "collisions", "reflex", "reflexes", "min_success_rate",                                 # criteria counts / rates
)

_REF = re.compile(r"\$\{([^}]+)\}")
_MISSING = object()


class StationError(ValueError):
    """All problems found while loading or validating, one per line."""

    def __init__(self, problems):
        self.problems = list(dict.fromkeys(problems))
        super().__init__("station config errors:\n  " + "\n  ".join(self.problems))


@dataclass(frozen=True)
class Prov:
    path: str       # provenance key that matched ("" = the whole file)
    status: str
    source: str
    note: str = ""


# ---------------------------------------------------------------------------------------------------- tree helpers


def _freeze(v):
    """Deep read-only view: dicts -> MappingProxyType, lists -> tuples."""
    if isinstance(v, Mapping):
        return MappingProxyType({k: _freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def _thaw(v):
    """Deep mutable copy of a (possibly frozen) tree: mappings -> dict, tuples/lists -> list."""
    if isinstance(v, Mapping):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_thaw(x) for x in v]
    return v


def _leaves(d, prefix=()):
    """Yield (path tuple, value) for every leaf. Dicts and lists of dicts recurse; other lists are leaves."""
    if isinstance(d, Mapping) and d:
        for k, v in d.items():
            if not prefix and k in META_KEYS:
                continue
            yield from _leaves(v, prefix + (str(k),))
    elif isinstance(d, (list, tuple)) and d and any(isinstance(x, Mapping) for x in d):
        for i, v in enumerate(d):
            yield from _leaves(v, prefix + (str(i),))
    else:
        yield prefix, d


def _step(node, key):
    if isinstance(node, Mapping):
        if key in node:
            return node[key]
        raise KeyError(key)
    if isinstance(node, (list, tuple)) and key.isdigit() and int(key) < len(node):
        return node[int(key)]
    raise KeyError(key)


def _get_in(d, dotted):
    node = d
    for key in dotted.split(".") if dotted else []:
        node = _step(node, key)
    return node


def _set_in(d, dotted, value):
    keys = dotted.split(".")
    node = _get_in(d, ".".join(keys[:-1])) if len(keys) > 1 else d
    _get_in(d, dotted)  # must exist
    if isinstance(node, list):
        node[int(keys[-1])] = value
    else:
        node[keys[-1]] = value


def _has_number(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_number(x) for x in v)
    return False


def _strip(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for t in STRIP_TOKENS:
            if key.endswith(t) and len(key) > len(t):
                key, changed = key[: -len(t)], True
    return key


def _unit_ok(path: tuple) -> bool:
    comps = [c for c in path if not c.isdigit()]
    dotted = ".".join(comps)
    if any(fnmatch.fnmatchcase(dotted, p) for p in NON_SUFFIXED_ALLOW):
        return True
    for c in comps:
        for cand in (c, _strip(c)):
            if cand.endswith(UNIT_SUFFIXES) or any(fnmatch.fnmatchcase(cand, p) for p in NON_SUFFIXED_ALLOW):
                return True
    return False


def _prov_lookup(prov: dict, dotted: str):
    parts = dotted.split(".") if dotted else []
    for n in range(len(parts), -1, -1):
        key = ".".join(parts[:n])
        if key in prov:
            return key, prov[key]
    return None


# ------------------------------------------------------------------------------------------------------ validation


def validate_group(group: str, d: dict, expect_id: str | None = None) -> list[str]:
    """Problems of one group file (empty list = valid). Cross-group checks happen in load_station."""
    errs: list[str] = []
    tag = f"{group}" + (f"/{expect_id}" if expect_id else "")
    if not isinstance(d, dict):
        return [f"{tag}: file is not a mapping"]
    kind = KIND_OF.get(group)
    if kind is None:
        return [f"{tag}: unknown group '{group}'"]
    schema = d.get("schema")
    if not isinstance(schema, str) or not re.fullmatch(rf"{kind}\.{group}/v\d+", schema):
        errs.append(f"{tag}: schema must be '{kind}.{group}/v<N>', got {schema!r}")
    if not d.get("id"):
        errs.append(f"{tag}: id missing")
    elif expect_id is not None and d["id"] != expect_id:
        errs.append(f"{tag}: id '{d['id']}' does not match the file name '{expect_id}'")
    prov = d.get("provenance")
    if not isinstance(prov, dict):
        errs.append(f"{tag}: provenance map missing")
        prov = {}
    prov = {str(k): v for k, v in prov.items()}
    for k, v in prov.items():
        if not isinstance(v, dict) or v.get("status") not in STATUSES:
            errs.append(f"{tag}: provenance '{k}': status must be one of {list(STATUSES)}, got "
                        f"{v.get('status') if isinstance(v, dict) else v!r}")
        elif not str(v.get("source", "")).strip():
            errs.append(f"{tag}: provenance '{k}': source missing")
    leaves = list(_leaves(d))
    dotted_leaves = [".".join(p) for p, _ in leaves]
    for k in prov:
        if k and not any(dl == k or dl.startswith(k + ".") for dl in dotted_leaves):
            errs.append(f"{tag}: provenance '{k}' matches no value")
    for path, value in leaves:
        dotted = ".".join(path)
        if not path:
            continue
        if _has_number(value) and not _unit_ok(path):
            errs.append(f"{tag}: '{dotted}' is numeric without a unit suffix (see NON_SUFFIXED_ALLOW)")
        if _prov_lookup(prov, dotted) is None:
            errs.append(f"{tag}: '{dotted}' has no provenance")
    crit = d.get("critical", [])
    if not isinstance(crit, list):
        errs.append(f"{tag}: critical must be a list")
        crit = []
    for c in crit:
        try:
            _get_in(d, str(c))
        except KeyError:
            errs.append(f"{tag}: critical path '{c}' does not exist")
    if group == "tooling" and not (isinstance(d.get("sensor_compat"), list) and d["sensor_compat"]):
        errs.append(f"{tag}: tooling needs a non-empty sensor_compat list")
    return errs


# ------------------------------------------------------------------------------------------------------- Station


@dataclass(frozen=True)
class Station:
    groups: dict          # group -> data (schema/id kept, provenance/critical removed), interpolated and overridden;
                          # deep read-only (MappingProxyType / tuples). get() and the fragments return mutable copies.
    ids: dict             # group -> id
    provenance: dict      # group -> {path: {status, source, note?}}
    critical: tuple       # "group.path" strings
    overrides: tuple      # the applied "group.path=value" strings

    def __post_init__(self):
        object.__setattr__(self, "groups", _freeze(self.groups))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    def get(self, path: str, default=_MISSING):
        group, _, rest = path.partition(".")
        try:
            return _thaw(_get_in(self.groups[group], rest))
        except KeyError:
            if default is _MISSING:
                raise KeyError(f"station has no value '{path}'") from None
            return default

    def prov(self, path: str) -> Prov:
        group, _, rest = path.partition(".")
        hit = _prov_lookup(self.provenance.get(group, {}), rest)
        if hit is None:
            raise StationError([f"no provenance for '{path}'"])
        key, e = hit
        return Prov(key, e["status"], str(e["source"]), str(e.get("note", "")))

    def leaf_paths(self) -> list[str]:
        return [f"{g}.{'.'.join(p)}" for g, d in self.groups.items() for p, _ in _leaves(d) if p]

    def placeholders(self, critical_only: bool = False) -> list:
        out = []
        for path in self.leaf_paths():
            p = self.prov(path)
            if p.status not in PLACEHOLDER_STATUSES:
                continue
            if critical_only and not any(path == c or path.startswith(c + ".") for c in self.critical):
                continue
            out.append((path, p))
        return out

    def fingerprint(self) -> str:
        blob = json.dumps(_thaw(self.groups), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_yaml(self) -> str:
        doc = {"ids": self.ids, "overrides": list(self.overrides), "critical": list(self.critical),
               "fingerprint": self.fingerprint(), "groups": _thaw(self.groups), "provenance": _thaw(self.provenance)}
        return yaml.safe_dump(doc, sort_keys=False)

    # ------------------------------------------------------------------------------------ derived legacy fragments

    def bridge_fragment(self) -> dict:
        """What configs/bridge.yaml holds today (merged under the deploy.yaml overrides)."""
        out: dict = {}
        rb = self.get("robot.bridge", None)
        if rb:
            for k in ("watchdog_hold_s", "watchdog_stop_s", "home_O_T_EE_xyz_wxyz"):
                if k in rb:
                    out[k] = _thaw(rb[k])
        safety: dict = {}
        ws = self.get("scene.workspace", None)
        if ws:
            safety.update(workspace_min=list(ws["min_m"]), workspace_max=list(ws["max_m"]),
                          cone_axis=list(ws["cone_axis"]), cone_max_deg=ws["cone_max_deg"])
        lim = self.get("criteria.limits", None)
        if lim:
            safety.update(v_max=lim["v_max_m_s"], w_max=lim["w_max_rad_s"])
        if rb and "slack_s" in rb:
            safety["slack_s"] = rb["slack_s"]
        if safety:
            out["safety"] = safety
        robot = self.groups.get("robot")
        if robot:
            imp, fci = robot["impedance"], robot["fci"]
            t, r = imp["translational_n_m"], imp["rotational_nm_rad"]
            pp = {"hostname": fci["hostname"], "impedance": np.diag([t, t, t, r, r, r]).tolist(),
                  "damping_ratio": imp["damping_ratio"], "nullspace_stiffness": imp["nullspace_stiffness"],
                  "filter_coeff": imp["filter_coeff"], "collision": _thaw(imp["collision_thresholds"]),
                  "NE_T_EE": None if robot["desk_end_effector"] == "none" else _thaw(robot["NE_T_EE"])}
            out["backend"] = {"panda_py_version": fci["panda_py_version"],
                              "libfranka_version": fci["libfranka_version"], "panda_py": pp}
        return out

    def controller_fragment(self) -> dict:
        """What configs/franka_cartesian.yaml holds today: F_T_TCP (EE_T_TCP when Desk has an end effector) + gripper."""
        out: dict = {}
        tool = self.groups.get("tooling")
        if not tool:
            return out
        F_T_TCP = np.array(tool["F_T_TCP"], float)
        robot = self.groups.get("robot")
        if robot and robot["desk_end_effector"] != "none":
            F_T_EE = np.array(robot["F_T_NE"], float) @ np.array(robot["NE_T_EE"], float)
            F_T_TCP = np.linalg.inv(F_T_EE) @ F_T_TCP
        out["F_T_TCP"] = F_T_TCP.tolist()
        hand = tool.get("hand", {})
        if hand.get("type") == "franka_hand":
            wm = tool["width_map"]
            out["gripper"] = {
                "type": "franka_hand", "endpoint": hand["endpoint"],
                "width_map": {k: wm[k] for k in ("a", "b", "gap_min_m", "gap_max_m")},
                "speed_m_s": hand["speed_m_s"], "force_n": hand["force_n"], "force_max_n": hand["force_max_n"],
                "grasp_epsilon_inner_m": hand["grasp_epsilon_inner_m"],
                "grasp_epsilon_outer_m": hand["grasp_epsilon_outer_m"],
            }
        return out

    def task_args(self) -> dict:
        """Values run_task.py hardcodes today (lookahead, home gripper, camera/policy latency) + the image-age guard."""
        out = {"lookahead_s": self.get("policy.task.lookahead_s", 0.1)}
        hw = self.get("tooling.hand.home_width_m", None)
        if hw is not None:
            out["home_gripper_m"] = hw
        cam = self.get("sensor.task", None)
        if cam:
            out["camera_latency_s"] = cam["camera_latency_s"]
        out["policy_latency_s"] = self.get("policy.task.policy_latency_s", 0.1)
        if cam:
            out["max_image_age_s"] = cam["max_image_age_s"]
        return out

    def camera_config(self, endpoint: str, logger_endpoint: str) -> dict:
        """camera.yaml for peripherals.wireless_camera (receiver side), from the sensor group. This dict IS the
        receiver config: `python -m franka_deploy.peripherals.wireless_camera --config camera.yaml` reads it as is.

        streams.<s>.fps_hz is the publish rate (BaseCamera camera_configs fps); physical_fps_hz and K are the device's
        (dejitter grid, v2 hello check). camera_latency_s is BaseCamera's and always 0: the receiver's ClockMap
        applies clock.latency_s exactly once (the task's sensor.task.camera_latency_s is a different value, the pause
        wait, see task_args())."""
        s = self.groups["sensor"]
        tr = s["transport"]
        fps = s["publish"]["fps_hz"]
        return {
            "endpoint": endpoint, "logger_endpoint": logger_endpoint, "name": s["camera_server"]["name"],
            "video_codec": s["camera_server"]["video_codec"], "protocol": tr["protocol"], "bind_host": tr["bind_host"],
            "ports": _thaw(tr["ports"]), "validate_ack": tr["validate_ack"],
            "require_websocket": tr["require_websocket"], "max_http_buffer_bytes": tr["max_http_buffer_bytes"],
            "streams": {n: {"width_px": v["width_px"], "height_px": v["height_px"], "fps_hz": fps,
                            "physical_fps_hz": v["physical_fps_hz"], "K": _thaw(v["K"])}
                        for n, v in s["streams"].items()},
            "publish": _thaw(s["publish"]), "on_size_mismatch": s["on_size_mismatch"],
            "clock": _thaw(s["clock"]), "health": _thaw(s["health"]),
            "camera_latency_s": 0.0,
        }

    def sender_config(self) -> dict:
        """sender.yaml for the sim/replay senders: imitate the sensor's wire behaviour over the run's link."""
        s = self.groups["sensor"]
        tr = s["transport"]
        out = {"protocol": tr["protocol"], "ports": _thaw(tr["ports"]),
               "streams": _thaw(s["streams"]), "publish": _thaw(s["publish"]),
               "clock": _thaw(s["clock"]), "sender": _thaw(s.get("sender")),
               "training_intrinsics_variants": _thaw(s.get("training_intrinsics_variants")),
               "link": None, "sim": None}
        if "link" in self.groups:
            out["link"] = {k: _thaw(v) for k, v in self.groups["link"].items() if k != "schema"}
        if "sim" in self.groups:
            out["sim"] = {k: _thaw(v) for k, v in self.groups["sim"].items() if k != "schema"}
        return out


# -------------------------------------------------------------------------------------------------------- loading


def _read(root: Path, kind: str, group: str, gid: str):
    p = Path(root) / kind / group / f"{gid}.yaml"
    if not p.is_file():
        return None, [f"{group}/{gid}: file not found: {p}"]
    try:
        d = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        return None, [f"{group}/{gid}: invalid YAML: {e}"]
    return d, []


def _resolve(groups: dict) -> list[str]:
    """Replace ${group.path} in place (after all groups loaded). Cycles and unknown targets are errors."""
    raw = copy.deepcopy(groups)
    errs: list[str] = []

    def lookup(ref, stack):
        if ref in stack:
            errs.append("interpolation cycle: " + " -> ".join(list(stack) + [ref]))
            return None
        group, _, rest = ref.partition(".")
        try:
            return value(_get_in(raw[group], rest), stack + (ref,))
        except KeyError:
            errs.append(f"interpolation ${{{ref}}}: no such value (is group '{group}' loaded?)")
            return None

    def value(v, stack):
        if isinstance(v, str) and "${" in v:
            m = _REF.fullmatch(v)
            if m:
                return lookup(m.group(1), stack)
            return _REF.sub(lambda mm: str(lookup(mm.group(1), stack)), v)
        if isinstance(v, list):
            return [value(x, stack) for x in v]
        if isinstance(v, dict):
            return {k: value(x, stack) for k, x in v.items()}
        return v

    for g in groups:
        for path, _ in _leaves(raw[g]):
            if not path:
                continue
            dotted = ".".join(path)
            try:
                cur = _get_in(raw[g], dotted)
            except KeyError:
                continue
            _set_in(groups[g], dotted, value(cur, (f"{g}.{dotted}",)))
    return errs


def load_station(spec: dict, root: Path = STATION_ROOT, sets=()) -> Station:
    """spec = {"station": {group: id}, "run": {group: id}} (any subset). Raises StationError with every problem."""
    errs: list[str] = []
    raw: dict = {}
    ids: dict = {}
    for kind in ("station", "run"):
        for group, gid in (spec.get(kind) or {}).items():
            if group not in KINDS[kind]:
                errs.append(f"unknown {kind} group '{group}'")
                continue
            d, e = _read(root, kind, group, gid)
            errs += e
            if d is None:
                continue
            errs += validate_group(group, d, expect_id=gid)
            raw[group], ids[group] = d, gid
    tool, sens = raw.get("tooling"), raw.get("sensor")
    if tool and sens and isinstance(tool.get("sensor_compat"), list) and sens.get("id") not in tool["sensor_compat"]:
        errs.append(f"tooling '{tool.get('id')}' is not compatible with sensor '{sens.get('id')}' "
                    f"(sensor_compat: {tool['sensor_compat']})")
    if errs:
        raise StationError(errs)
    provenance = {g: {str(k): v for k, v in d["provenance"].items()} for g, d in raw.items()}
    critical = tuple(f"{g}.{c}" for g, d in raw.items() for c in d.get("critical", []))
    groups = {g: {k: copy.deepcopy(v) for k, v in d.items() if k not in ("provenance", "critical")}
              for g, d in raw.items()}
    applied = []
    for s in sets:   # overrides first, so that ${} references to an overridden value follow it
        path, eq, text = str(s).partition("=")
        group, _, rest = path.partition(".")
        if not eq or group not in groups or not rest:
            errs.append(f"override '{s}': expected 'group.path=value' with a loaded group")
            continue
        try:
            _set_in(groups[group], rest, yaml.safe_load(text))
            applied.append(str(s))
        except (KeyError, yaml.YAMLError):
            errs.append(f"override '{s}': path '{path}' does not exist or value is not valid YAML")
    errs += _resolve(groups)
    if errs:
        raise StationError(errs)
    return Station(groups=groups, ids=ids, provenance=provenance, critical=critical, overrides=tuple(applied))


def station_from_deploy_cfg(cfg: dict):
    """Station for a (merged) deploy.yaml dict, or None when it has no `station` key."""
    if "station" not in cfg:
        return None
    return load_station({"station": cfg["station"], "run": cfg.get("run", {})}, sets=cfg.get("station_set", []))


# ------------------------------------------------------------------------------------------------------------ CLI

_SIM_RUN = {"policy": "bp_run001", "link": "wifi_good", "criteria": "block_raise", "sim": "render"}
_REAL_RUN = {"policy": "bp_run001", "criteria": "block_raise"}
_IPHONE = {"robot": "panda", "tooling": "franka_hand_iphumi_iphone16pro", "sensor": "iphone16pro_wifi",
           "scene": "block_raise_table"}
_RIG = {"robot": "panda", "tooling": "franka_hand_iphumi_rig1", "sensor": "rig1_wifi", "scene": "block_raise_table"}
PROFILES = {  # the station combinations of the design profiles (deploy.yaml gets the same ids in B2/B6)
    "sim-iphone": {"station": {**_IPHONE, "host": "pc_laptop"}, "run": _SIM_RUN},
    "sim-rig": {"station": {**_RIG, "host": "pc_laptop"}, "run": _SIM_RUN},
    "sim-iphone-fake": {"station": {**_IPHONE, "host": "pc_laptop"}, "run": {**_SIM_RUN, "policy": "fake"}},
    "sim-rig-fake": {"station": {**_RIG, "host": "pc_laptop"}, "run": {**_SIM_RUN, "policy": "fake"}},
    "real-iphone": {"station": {**_IPHONE, "host": "jetson_orin_nano"}, "run": _REAL_RUN},
    "real-rig": {"station": {**_RIG, "host": "jetson_orin_nano"}, "run": _REAL_RUN},
}


def _validate_all(root: Path) -> list[str]:
    errs: list[str] = []
    n = 0
    for kind, groups in KINDS.items():
        base = Path(root) / kind
        if base.is_dir():
            for sub in sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in groups):
                errs.append(f"{kind}/{sub}: unknown group directory")
        for group in groups:
            for f in sorted((base / group).glob("*.yaml")):
                n += 1
                try:
                    d = yaml.safe_load(f.read_text())
                except yaml.YAMLError as e:
                    errs.append(f"{group}/{f.stem}: invalid YAML: {e}")
                    continue
                errs += validate_group(group, d, expect_id=f.stem)
    print(f"validated {n} files under {root}")
    return errs


def _report(label: str, st: Station, verbose: bool = False) -> None:
    print(f"[{label}] ids={st.ids} fingerprint={st.fingerprint()}")
    crit = {p for p, _ in st.placeholders(critical_only=True)}
    for path, p in st.placeholders():
        if verbose or path in crit:
            print(f"  {'CRITICAL ' if path in crit else '         '}{p.status:11s} {path}")
    print(f"  {len(crit)} critical placeholder/hw value(s), {len(st.placeholders())} placeholder/hw in total")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_deploy.station", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate every file (--all) or a combination (--spec)")
    v.add_argument("--root", default=str(STATION_ROOT))
    v.add_argument("--all", action="store_true", help="every file plus the design profile combinations")
    v.add_argument("--spec", action="append", default=[], metavar="station.robot=panda",
                   help="repeatable; kind.group=id")
    v.add_argument("-v", "--verbose", action="store_true", help="also list non-critical placeholders")
    v.add_argument("--set", action="append", default=[], dest="sets", metavar="group.path=value")
    a = ap.parse_args(argv)
    root = Path(a.root)
    problems: list[str] = []
    if a.all or not a.spec:
        problems += _validate_all(root)
        for name, spec in PROFILES.items():
            try:
                _report(name, load_station(spec, root), a.verbose)
            except StationError as e:
                problems += [f"profile {name}: {p}" for p in e.problems]
    if a.spec:
        spec: dict = {"station": {}, "run": {}}
        for s in a.spec:
            key, eq, gid = s.partition("=")
            kind, _, group = key.partition(".")
            if not eq or kind not in spec or not group:
                problems.append(f"bad --spec '{s}': expected kind.group=id")
            else:
                spec[kind][group] = gid
        try:
            _report("spec", load_station(spec, root, sets=a.sets), a.verbose)
        except StationError as e:
            problems += e.problems
    for p in problems:
        print("ERROR", p, file=sys.stderr)
    print("FAIL" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
