# adapted from rnoro/deployment:deployment/contracts/contracts/examples.py @ 91b178b
# changes: only C2/C3 (C0/C1/C4 examples move to policy_deploy); + state_stopped and the hello examples
# (STRUCTURE.md 5.2); the helpers are public (good_example, broken_example, raw_example, set_key, drop_key);
# examples_dir comes from paths.py; STEP_MS (sizes.py, moved to policy_deploy) is the literal 50 ms here.
"""The shared C2/C3 example messages, good ones and deliberately broken ones (STRUCTURE.md section 5).

Good examples are built from the message classes; broken ones take a good message's dict and change one thing, then
pack it with pack_obj (the classes would refuse to build them). Every example here is C2 or C3 (C3 including its
set_waypoints.reply, hello.reply, ack and error replies) and every one is written to deployment/contracts/examples/ by
`python -m deploy_contracts.make_examples`, for the Python tests on both machines and the C++ tests of franka_control.
policy_deploy builds its C0/C1/C4 examples in memory with the public helpers below; they are never stored.

Stored numbers use only + - * / on literals (no sin/cos, no matmul), so the bytes are the same with any numpy. Hello
examples use literal values, never values read from the yaml files, so tunables never change example bytes (except
set_waypoints_too_many, which is max_waypoints + 1 by definition).

Value paths (Example.values keys, Example.field): dotted names into sub-messages ("gate.pulled_back"), [i] for a
list item or array index ("rejected[0].reason", "O_T_TCP[7][0][3]").
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from deploy_contracts.c2_state import ArmState, GateCounters
from deploy_contracts.c3_commands import (GetState, Heartbeat, Hello, HelloReply, Home, SetWaypoints,
                                          SetWaypointsReply, Start, Stop, WaypointIssue)
from deploy_contracts.codec import encode, pack_array, pack_obj
from deploy_contracts.common import Ack, ErrorReply
from deploy_contracts.limits import ContractLimits
from deploy_contracts.message import Message
from deploy_contracts.paths import examples_dir
from deploy_contracts.shared import SHARED_SCHEMAS

STORED_CONTRACTS = ("c2", "c3")
INDEX_FILE = "index.yaml"
INDEX_FORMAT = 1


@dataclass(frozen=True)
class Example:
    """One example message: `data` is the exact bytes that travel on the wire."""

    name: str                       # "<contract>/<name>", e.g. "c2/state_running"
    schema: str                     # the schema it is (or pretends to be); "?" when unreadable
    valid: bool
    data: bytes
    field: str | None = None        # broken: the field at fault ("" if none)
    problem: str | None = None      # broken: the problem code
    values: dict[str, Any] = dataclasses.field(default_factory=dict)   # good: a few scalars a decoder test can compare
    note: str = ""

    @property
    def contract(self) -> str:
        return self.name.split("/", 1)[0]

    @property
    def stored(self) -> bool:
        return self.contract in STORED_CONTRACTS

    @property
    def file(self) -> str:
        return f"{self.name}.msgpack"


_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def resolve_path(obj: Any, path: str) -> Any:
    """Follow a value path into a to_dict() result; array elements come back as Python scalars."""
    for name, index in _PATH_TOKEN.findall(path):
        obj = obj[name] if name else obj[int(index)]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_index(directory: Path | str | None = None) -> list[Example]:
    """Read index.yaml and the files it lists back into Examples. Default: deployment/contracts/examples/."""
    root = examples_dir() if directory is None else Path(directory)
    with open(root / INDEX_FILE, encoding="utf-8") as f:
        index = yaml.safe_load(f)
    if index.get("format") != INDEX_FORMAT:
        raise ValueError(f"{root / INDEX_FILE}: format {index.get('format')!r}, expected {INDEX_FORMAT}")
    out = []
    for entry in index["examples"]:
        name = entry["file"][: -len(".msgpack")]
        out.append(Example(name=name, schema=entry["schema"], valid=entry["valid"],
                           data=(root / entry["file"]).read_bytes(), field=entry.get("field"),
                           problem=entry.get("problem"), values=entry.get("values", {}) or {},
                           note=entry.get("note", "")))
    return out


# ----------------------------------------------------------------------------------------------------------------
# helpers


def good_example(name: str, msg: Message, values: dict[str, Any] | None = None, note: str = "") -> Example:
    """A valid example: the message's canonical bytes; `values` are value paths a decoder test compares."""
    return Example(name=name, schema=msg.SCHEMA, valid=True, data=encode(msg), values=values or {}, note=note)


def broken_example(name: str, base: Message, change: Callable[[dict], Any], field: str, problem: str, note: str,
                   schema: str | None = None) -> Example:
    """base.to_dict(), changed by `change` (which edits it in place, or returns a replacement), then packed."""
    d = base.to_dict()
    replaced = change(d)
    obj = d if replaced is None else replaced
    return Example(name=name, schema=schema or base.SCHEMA, valid=False, data=pack_obj(obj), field=field,
                   problem=problem, note=note)


def raw_example(name: str, data: bytes, schema: str, field: str, problem: str, note: str) -> Example:
    """A broken example given as raw bytes (not msgpack, not a map, trailing bytes, ...)."""
    return Example(name=name, schema=schema, valid=False, data=data, field=field, problem=problem, note=note)


def set_key(key: str, value: Any) -> Callable[[dict], None]:
    """A change for broken_example: d[key] = value."""
    def change(d: dict) -> None:
        d[key] = value
    return change


def drop_key(*path: str) -> Callable[[dict], None]:
    """A change for broken_example: delete d[path[0]][path[1]]...; drop_key("gate", "pulled_back")."""
    def change(d: dict) -> None:
        for key in path[:-1]:
            d = d[key]
        del d[path[-1]]
    return change


# short names for the tables below
_good, _broken, _raw, _set, _drop = good_example, broken_example, raw_example, set_key, drop_key


def _pose(x: float, y: float, z: float) -> np.ndarray:
    """A TCP pointing down, turned about z by (cos, sin) = (0.6, 0.8): exact literals, no trigonometry."""
    return np.array([[0.6, 0.8, 0.0, x],
                     [0.8, -0.6, 0.0, y],
                     [0.0, 0.0, -1.0, z],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
_T0 = 123_456_789_000_000          # a PC monotonic time, ns
_STEP_NS = 50_000_000             # 50 ms between waypoints (policy_deploy's STEP_MS)


# ----------------------------------------------------------------------------------------------------------------
# C2


def _state(seq: int, mode: str, width: float, gate: GateCounters, loop_late: int, error: str = "",
           t_ns: int = _T0, dq: tuple[float, ...] = (0.0,) * 7) -> ArmState:
    return ArmState(t_ns=t_ns, seq=seq, O_T_TCP=_pose(0.45, -0.05, 0.3), O_T_TCP_target=_pose(0.455, -0.05, 0.3),
                    q=_Q.copy(), dq=np.array(dq, dtype=np.float64), gripper_width_m=width, mode=mode, error=error,
                    gate=gate, loop_late=loop_late)


def _c2() -> list[Example]:
    running = _state(42, "running", 0.05, GateCounters(accepted=120, shortened=4, pulled_back=1, rejected=0), 3,
                     dq=(0.01, -0.02, 0.0, 0.03, 0.0, -0.01, 0.0))
    holding = _state(43, "holding", 0.05, GateCounters(accepted=128, shortened=4, pulled_back=2, rejected=0), 3,
                     t_ns=_T0 + 10_000_000)
    homing = _state(1000, "homing", 0.08, GateCounters(accepted=300, shortened=9, pulled_back=2, rejected=0), 7,
                    t_ns=_T0 + 9_580_000_000, dq=(0.0, 0.1, 0.0, -0.1, 0.0, 0.05, 0.0))
    fault = _state(1001, "fault", 0.05, GateCounters(accepted=300, shortened=9, pulled_back=2, rejected=1), 7,
                   error="waypoint rejected: outside the workspace box: x 0.95 > 0.81", t_ns=_T0 + 9_590_000_000)
    idle = ArmState(t_ns=0, seq=0, O_T_TCP=_pose(0.3, 0.0, 0.5), O_T_TCP_target=_pose(0.3, 0.0, 0.5), q=_Q.copy(),
                    dq=np.zeros(7), gripper_width_m=0.0, mode="idle", error="",
                    gate=GateCounters(accepted=0, shortened=0, pulled_back=0, rejected=0), loop_late=0)
    stopped = _state(1500, "stopped", 0.04, GateCounters(accepted=410, shortened=12, pulled_back=3, rejected=0), 7,
                     t_ns=_T0 + 14_590_000_000)

    def transposed(d: dict) -> None:
        d["O_T_TCP"] = np.ascontiguousarray(d["O_T_TCP"].T)

    def not_orthonormal(d: dict) -> None:
        m = d["O_T_TCP"].copy()
        m[0, 0] = 0.61
        d["O_T_TCP"] = m

    def nan_dq(d: dict) -> None:
        a = d["dq"].copy()
        a[2] = float("nan")
        d["dq"] = a

    def short_bytes(d: dict) -> dict:
        d["O_T_TCP"] = pack_array(d["O_T_TCP"])
        d["O_T_TCP"]["data"] = d["O_T_TCP"]["data"][:-8]
        return d

    def gate_shortened(d: dict) -> None:
        d["gate"]["shortened"] = 200

    good = [
        _good("c2/state_running", running, {"seq": 42, "mode": "running", "gripper_width_m": 0.05, "loop_late": 3,
                                            "gate.pulled_back": 1, "gate.accepted": 120, "t_ns": _T0,
                                            "O_T_TCP[0][3]": 0.45, "O_T_TCP[1][0]": 0.8, "q[3]": -2.356,
                                            "dq[1]": -0.02, "error": ""}),
        _good("c2/state_holding", holding, {"seq": 43, "mode": "holding", "loop_late": 3, "gate.pulled_back": 2,
                                            "O_T_TCP_target[0][3]": 0.455}),
        _good("c2/state_homing", homing, {"seq": 1000, "mode": "homing", "gripper_width_m": 0.08, "loop_late": 7,
                                          "gate.pulled_back": 2, "gate.shortened": 9}),
        _good("c2/state_fault", fault, {"seq": 1001, "mode": "fault", "gate.rejected": 1,
                                        "error": "waypoint rejected: outside the workspace box: x 0.95 > 0.81"}),
        _good("c2/state_idle_zero", idle, {"seq": 0, "t_ns": 0, "mode": "idle", "gripper_width_m": 0.0,
                                           "loop_late": 0, "gate.accepted": 0}),
        _good("c2/state_stopped", stopped, {"seq": 1500, "mode": "stopped", "gripper_width_m": 0.04, "error": "",
                                            "gate.accepted": 410, "t_ns": _T0 + 14_590_000_000},
              note="the watchdog case: no C3 request for watchdog_limit_s while holding/running -> stopped"),
    ]
    broken = [
        _broken("c2/bad_pose_transposed", running, transposed, "O_T_TCP", "value",
                "column-major matrix sent without transposing"),
        _broken("c2/bad_pose_not_orthonormal", running, not_orthonormal, "O_T_TCP", "value",
                "rotation part not orthonormal"),
        _broken("c2/bad_mode", running, _set("mode", "moving"), "mode", "value", "unknown mode"),
        _broken("c2/missing_loop_late", running, _drop("loop_late"), "loop_late", "missing", "loop_late left out"),
        _broken("c2/missing_gate_pulled_back", running, _drop("gate", "pulled_back"), "gate.pulled_back", "missing",
                "gate.pulled_back left out"),
        _broken("c2/wrong_shape_q", running, lambda d: d.update(q=d["q"][:6].copy()), "q", "shape", "6 joints"),
        _broken("c2/wrong_dtype_q", running, lambda d: d.update(q=d["q"].astype(np.float32)), "q", "dtype",
                "float32 joints"),
        _broken("c2/bad_width_negative", running, _set("gripper_width_m", -0.01), "gripper_width_m", "value",
                "negative opening"),
        _broken("c2/bad_width_int", running, _set("gripper_width_m", 0), "gripper_width_m", "type",
                "an integer where float64 is required (float64 fields are always msgpack float64)"),
        _broken("c2/bad_gate_shortened", running, gate_shortened, "gate.shortened", "value",
                "shortened > accepted"),
        _broken("c2/bad_dq_nan", running, nan_dq, "dq", "value", "NaN joint speed"),
        _broken("c2/bad_type_seq", running, _set("seq", "42"), "seq", "type", "seq as a string"),
        _broken("c2/bad_seq_negative", running, _set("seq", -1), "seq", "value", "negative uint64"),
        _broken("c2/bad_array_byte_count", running, short_bytes, "O_T_TCP", "shape",
                "array data 8 bytes short of its shape"),
        _broken("c2/unknown_field", running, _set("temperature", 30.0), "temperature", "unknown_field",
                "a field the contract doesn't have"),
        _broken("c2/schema_version", running, _set("schema", "c2.state/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c2.state/v2"),
    ]
    return good + broken


# ----------------------------------------------------------------------------------------------------------------
# C3 (with the set_waypoints.reply, ack and error replies)


def _waypoints(n: int, chunk_id: int, t0: int = _T0 + 100_000_000) -> SetWaypoints:
    t = np.array([t0 + i * _STEP_NS for i in range(n)], dtype=np.int64)
    poses = np.stack([_pose(0.45 + 0.005 * i, -0.05, 0.3 - 0.002 * i) for i in range(n)]) if n else \
        np.zeros((0, 4, 4))
    widths = np.array([0.05 - 0.001 * i for i in range(n)], dtype=np.float64)
    return SetWaypoints(chunk_id=chunk_id, t_ns=t, O_T_TCP=poses, gripper_width_m=widths)


def _c3() -> list[Example]:
    wp8 = _waypoints(8, 7)
    wp1 = _waypoints(1, 0)
    reply_ok = SetWaypointsReply(chunk_id=7, accepted=6, shortened=1, skipped=2, rejected=())
    reply_rejected = SetWaypointsReply(chunk_id=8, accepted=0, shortened=0, skipped=3, rejected=(
        WaypointIssue(index=3, reason="outside the workspace box: x 0.95 > 0.81 (max 0.80 + margin 0.01)"),
        WaypointIssue(index=4, reason="not checked: another waypoint of this request was rejected"),
    ))
    ack_ok = Ack(ok=True, reason="")
    ack_refused = Ack(ok=False, reason="refused: mode is fault")
    err_state = ErrorReply(request="c3.set_waypoints/v1", error="set_waypoints refused: mode is idle", field="")
    err_contract = ErrorReply(request="c3.set_waypoints/v1",
                              error="c3.set_waypoints/v1: field 'O_T_TCP': not a rigid transform at [3]: bottom row "
                                    "[0.45, -0.05, 0.3, 1.0], expected [0, 0, 0, 1]", field="O_T_TCP")

    def times(d: dict) -> None:
        t = d["t_ns"].copy()
        t[4] = t[3]
        d["t_ns"] = t

    def pose_at(i: int, fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[dict], None]:
        def change(d: dict) -> None:
            p = d["O_T_TCP"].copy()
            p[i] = fn(p[i])
            d["O_T_TCP"] = p
        return change

    def nan_entry(m: np.ndarray) -> np.ndarray:
        m = m.copy()
        m[1, 3] = float("nan")
        return m

    def width_at(i: int, w: float) -> Callable[[dict], None]:
        def change(d: dict) -> None:
            a = d["gripper_width_m"].copy()
            a[i] = w
            d["gripper_width_m"] = a
        return change

    too_many = ContractLimits.load().max_waypoints + 1

    def many(d: dict) -> dict:
        t = np.array([_T0 + i * 1_000_000 for i in range(too_many)], dtype=np.int64)
        poses = np.stack([_pose(0.45, -0.05, 0.3)] * too_many)
        widths = np.full(too_many, 0.05)
        d.update(t_ns=t, O_T_TCP=poses, gripper_width_m=widths)
        return d

    def empty(d: dict) -> None:
        d.update(t_ns=np.zeros(0, dtype=np.int64), O_T_TCP=np.zeros((0, 4, 4)), gripper_width_m=np.zeros(0))

    def bad_array_map(d: dict) -> dict:
        d["O_T_TCP"] = pack_array(d["O_T_TCP"])
        d["O_T_TCP"]["__nd__"] = False
        return d

    def duplicate_index(d: dict) -> None:
        d["rejected"][1]["index"] = 3

    def drop_reason(d: dict) -> None:
        del d["rejected"][0]["reason"]

    wp8_values = {"chunk_id": 7, "t_ns[0]": _T0 + 100_000_000, "t_ns[7]": _T0 + 450_000_000,
                  "O_T_TCP[7][0][3]": 0.45 + 0.005 * 7, "O_T_TCP[7][2][3]": 0.3 - 0.002 * 7,
                  "O_T_TCP[7][1][0]": 0.8, "O_T_TCP[0][3][3]": 1.0, "gripper_width_m[7]": 0.05 - 0.001 * 7}
    good = [
        _good("c3/set_waypoints_8", wp8, wp8_values),
        _good("c3/set_waypoints_1", wp1, {"chunk_id": 0, "t_ns[0]": _T0 + 100_000_000, "gripper_width_m[0]": 0.05}),
        _good("c3/heartbeat", Heartbeat()),
        _good("c3/start", Start()),
        _good("c3/stop", Stop()),
        _good("c3/home", Home()),
        _good("c3/get_state", GetState()),
        _good("c3/set_waypoints_reply_ok", reply_ok, {"chunk_id": 7, "accepted": 6, "shortened": 1, "skipped": 2}),
        _good("c3/set_waypoints_reply_rejected", reply_rejected,
              {"chunk_id": 8, "accepted": 0, "skipped": 3, "rejected[0].index": 3, "rejected[1].index": 4,
               "rejected[0].reason": "outside the workspace box: x 0.95 > 0.81 (max 0.80 + margin 0.01)"}),
        _good("c3/ack_ok", ack_ok, {"ok": True, "reason": ""}),
        _good("c3/ack_refused", ack_refused, {"ok": False, "reason": "refused: mode is fault"}),
        _good("c3/error_set_waypoints_refused", err_state,
              {"request": "c3.set_waypoints/v1", "error": "set_waypoints refused: mode is idle", "field": ""},
              note="set_waypoints refused because of the state (not a fault)"),
        _good("c3/error_contract", err_contract, {"request": "c3.set_waypoints/v1", "field": "O_T_TCP"},
              note="the reply to a request that broke its contract"),
    ]
    broken = [
        _broken("c3/set_waypoints_times_not_increasing", wp8, times, "t_ns", "value", "t_ns[4] == t_ns[3]"),
        _broken("c3/set_waypoints_lengths_differ", wp8, lambda d: d.update(gripper_width_m=d["gripper_width_m"][:7]
                                                                           .copy()),
                "gripper_width_m", "value", "7 widths for 8 times"),
        _broken("c3/set_waypoints_pose_not_rigid", wp8, pose_at(3, lambda m: np.ascontiguousarray(m.T)), "O_T_TCP",
                "value", "pose [3] column-major (transposed)"),
        _broken("c3/set_waypoints_pose_nan", wp8, pose_at(2, nan_entry), "O_T_TCP", "value", "NaN in pose [2]"),
        _broken("c3/set_waypoints_too_many", wp8, many, "t_ns", "value", f"{too_many} waypoints > max_waypoints"),
        _broken("c3/set_waypoints_empty", wp8, empty, "t_ns", "value", "no waypoints"),
        _broken("c3/set_waypoints_width_negative", wp8, width_at(5, -0.001), "gripper_width_m", "value",
                "negative opening"),
        _broken("c3/set_waypoints_width_nan", wp8, width_at(5, float("nan")), "gripper_width_m", "value",
                "NaN opening"),
        _broken("c3/set_waypoints_wrong_shape", wp8, lambda d: d.update(O_T_TCP=d["O_T_TCP"][:, :3, :].copy()),
                "O_T_TCP", "shape", "(8, 3, 4) poses"),
        _broken("c3/set_waypoints_wrong_dtype_t", wp8, lambda d: d.update(t_ns=d["t_ns"].astype(np.float64)),
                "t_ns", "dtype", "float64 times"),
        _broken("c3/set_waypoints_wrong_dtype_width", wp8,
                lambda d: d.update(gripper_width_m=d["gripper_width_m"].astype(np.float32)), "gripper_width_m",
                "dtype", "float32 widths"),
        _broken("c3/set_waypoints_bad_array_map", wp8, bad_array_map, "O_T_TCP", "type", "__nd__ is false"),
        _broken("c3/set_waypoints_missing_width", wp8, _drop("gripper_width_m"), "gripper_width_m", "missing",
                "gripper_width_m left out"),
        _broken("c3/set_waypoints_unknown_field", wp8, _set("speed", 0.1), "speed", "unknown_field",
                "a field the contract doesn't have"),
        _broken("c3/set_waypoints_type_chunk_id", wp8, _set("chunk_id", "7"), "chunk_id", "type",
                "chunk_id as a string"),
        _broken("c3/set_waypoints_chunk_id_negative", wp8, _set("chunk_id", -1), "chunk_id", "value",
                "negative uint64"),
        _broken("c3/set_waypoints_schema_version", wp8, _set("schema", "c3.set_waypoints/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c3.set_waypoints/v2"),
        _broken("c3/heartbeat_unknown_field", Heartbeat(), _set("extra", 1), "extra", "unknown_field",
                "heartbeat has no fields"),
        _broken("c3/unknown_schema", Heartbeat(), _set("schema", "c3.jump/v1"), "", "unknown_schema",
                "a message name nobody knows", schema="c3.jump/v1"),
        _broken("c3/no_schema", Heartbeat(), lambda d: {"ok": True}, "", "no_schema", "a map without schema",
                schema="?"),
        _raw("c3/not_a_map", pack_obj([1, 2, 3]), "?", "", "not_a_map", "a msgpack array, not a map"),
        _raw("c3/not_msgpack", b"\xc1\x00\x01", "?", "", "not_msgpack", "0xc1 is never used by msgpack"),
        _raw("c3/trailing_bytes", encode(Heartbeat()) + b"\x00", "?", "", "not_msgpack",
             "a valid message followed by an extra byte"),
        _broken("c3/set_waypoints_reply_shortened_gt_accepted", reply_ok, _set("shortened", 7), "shortened",
                "value", "shortened > accepted"),
        _broken("c3/set_waypoints_reply_duplicate_index", reply_rejected, duplicate_index, "rejected[1].index",
                "value", "the same waypoint rejected twice"),
        _broken("c3/set_waypoints_reply_missing_skipped", reply_ok, _drop("skipped"), "skipped", "missing",
                "skipped left out"),
        _broken("c3/set_waypoints_reply_missing_reason", reply_rejected, drop_reason, "rejected[0].reason",
                "missing", "a rejection without its reason"),
        _broken("c3/ack_ok_not_bool", ack_ok, _set("ok", 1), "ok", "type", "ok as an integer"),
        _broken("c3/ack_missing_reason", ack_ok, _drop("reason"), "reason", "missing", "reason left out"),
        _broken("c3/ack_schema_version", ack_ok, _set("schema", "ack/v2"), "", "schema_version",
                "a version this side doesn't know", schema="ack/v2"),
        _broken("c3/error_missing_field", err_state, _drop("field"), "field", "missing", "field left out"),
    ]
    return good + broken


# ----------------------------------------------------------------------------------------------------------------
# C3 hello (STRUCTURE.md 3.8, 5.2): literal values only, never read from the yaml files


_HELLO_F_T_TCP = np.array([[0.6, 0.8, 0.0, 0.0],
                           [-0.8, 0.6, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.15],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _hello_reply(mock: bool = False, commit: str = _COMMIT) -> HelloReply:
    return HelloReply(t_pc_ns=123_461_789_000_000, commit=commit, mock=mock, schemas=SHARED_SCHEMAS,
                      F_T_TCP=_HELLO_F_T_TCP.copy(), rotation_tol=1.0e-6, gripper_width_max_m=0.08, max_waypoints=64,
                      watchdog_limit_s=1.0, homing_timeout_s=30.0, max_lead_s=1.5)


def _hello() -> list[Example]:
    reply = _hello_reply()
    mock = _hello_reply(mock=True, commit="unknown")

    def schemas_swapped(d: dict) -> None:
        s = list(d["schemas"])
        s[3], s[4] = s[4], s[3]
        d["schemas"] = s

    def schemas_item_int(d: dict) -> None:
        s = list(d["schemas"])
        s[3] = 7
        d["schemas"] = s

    values = {"t_pc_ns": 123_461_789_000_000, "commit": _COMMIT, "mock": False, "rotation_tol": 1.0e-6,
              "gripper_width_max_m": 0.08, "max_waypoints": 64, "watchdog_limit_s": 1.0, "homing_timeout_s": 30.0,
              "max_lead_s": 1.5, "F_T_TCP[0][1]": 0.8, "F_T_TCP[2][3]": 0.15}
    good = [
        _good("c3/hello", Hello()),
        _good("c3/hello_reply", reply, values, note="schemas is SHARED_SCHEMAS"),
        _good("c3/hello_reply_mock", mock, {"mock": True, "commit": "unknown"}, note="franka_control --mock"),
    ]
    broken = [
        _broken("c3/hello_unknown_field", Hello(), _set("commit", "x"), "commit", "unknown_field",
                "hello has no fields"),
        _broken("c3/hello_schema_version", Hello(), _set("schema", "c3.hello/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c3.hello/v2"),
        _broken("c3/hello_reply_missing_max_lead_s", reply, _drop("max_lead_s"), "max_lead_s", "missing",
                "max_lead_s left out"),
        _broken("c3/hello_reply_t_pc_ns_float", reply, _set("t_pc_ns", 123461789000000.0), "t_pc_ns", "type",
                "float time"),
        _broken("c3/hello_reply_mock_int", reply, _set("mock", 1), "mock", "type", "bool as an integer"),
        _broken("c3/hello_reply_commit_short", reply, _set("commit", "0123456"), "commit", "value",
                "a short hash (the full 40 characters are required)"),
        _broken("c3/hello_reply_schemas_unsorted", reply, schemas_swapped, "schemas", "value",
                "two entries swapped"),
        _broken("c3/hello_reply_schemas_item_int", reply, schemas_item_int, "schemas[3]", "type",
                "a schema as an integer"),
        _broken("c3/hello_reply_F_T_TCP_transposed", reply, lambda d: d.update(F_T_TCP=np.ascontiguousarray(
            d["F_T_TCP"].T)), "F_T_TCP", "value", "column-major matrix sent without transposing"),
        _broken("c3/hello_reply_F_T_TCP_shape", reply, lambda d: d.update(F_T_TCP=d["F_T_TCP"][:3].copy()),
                "F_T_TCP", "shape", "(3, 4)"),
        _broken("c3/hello_reply_watchdog_zero", reply, _set("watchdog_limit_s", 0.0), "watchdog_limit_s", "value",
                "a zero watchdog"),
        _broken("c3/hello_reply_max_waypoints_zero", reply, _set("max_waypoints", 0), "max_waypoints", "value",
                "max_waypoints below 1"),
        _broken("c3/hello_reply_schema_version", reply, _set("schema", "c3.hello.reply/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c3.hello.reply/v2"),
    ]
    return good + broken


def build_examples() -> list[Example]:
    """Every shared (C2/C3) example, good and broken, in a fixed order. All of them are stored."""
    out = _c2() + _c3() + _hello()
    names = [e.name for e in out]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate example names")
    return out
