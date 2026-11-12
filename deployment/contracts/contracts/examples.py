"""Example messages for every contract, good ones and deliberately broken ones (STRUCTURE.md section 8).

Good examples are built from the message classes; broken ones take a good message's dict and change one thing, then
pack it with pack_obj (the classes would refuse to build them). The C2 and C3 examples (C3 including its
set_waypoints.reply, ack and error replies) are written to examples/ by `python -m contracts.make_examples`, for the
C++ tests of franka_control. C0, C1 and C4 are only built in memory.

Stored numbers use only + - * / on literals (no sin/cos, no matmul), so the bytes are the same with any numpy.

Value paths (Example.values keys, Example.field): dotted names into sub-messages ("gate.pulled_back"), [i] for a
list item or array index ("rejected[0].reason", "O_T_TCP[7][0][3]").
"""
from __future__ import annotations

import math
import re
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

import contracts
from contracts.c0_frames import Frame, TimeSyncReply, TimeSyncRequest
from contracts.c1_frames import (CameraHealthReport, FramesReply, GetFramesRequest, GetHealthRequest, HealthReply,
                                 HealthReport)
from contracts.c2_state import ArmState, GateCounters
from contracts.c3_commands import (GetState, Heartbeat, Home, SetWaypoints, SetWaypointsReply, Start, Stop,
                                   WaypointIssue)
from contracts.c4_policy import (OBSERVATION_INPUTS, ActionChunk, Observation, PolicyConfig, PolicyConfigRequest,
                                 Prompt, Reset)
from contracts.codec import encode, pack_array, pack_obj
from contracts.common import Ack, ErrorReply
from contracts.limits import ContractLimits
from contracts.message import Message
from contracts.sizes import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS

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


def examples_dir() -> Path:
    """The committed examples folder, found from the installed (editable) package."""
    return Path(contracts.__file__).parents[1] / "examples"


def load_index(directory: Path | None = None) -> list[Example]:
    """Read index.yaml and the files it lists back into Examples."""
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


def _good(name: str, msg: Message, values: dict[str, Any] | None = None, note: str = "") -> Example:
    return Example(name=name, schema=msg.SCHEMA, valid=True, data=encode(msg), values=values or {}, note=note)


def _broken(name: str, base: Message, change: Callable[[dict], Any], field: str, problem: str, note: str,
            schema: str | None = None) -> Example:
    """base.to_dict(), changed by `change` (which edits it in place, or returns a replacement), then packed."""
    d = base.to_dict()
    replaced = change(d)
    obj = d if replaced is None else replaced
    return Example(name=name, schema=schema or base.SCHEMA, valid=False, data=pack_obj(obj), field=field,
                   problem=problem, note=note)


def _raw(name: str, data: bytes, schema: str, field: str, problem: str, note: str) -> Example:
    return Example(name=name, schema=schema, valid=False, data=data, field=field, problem=problem, note=note)


def _set(key: str, value: Any) -> Callable[[dict], None]:
    def change(d: dict) -> None:
        d[key] = value
    return change


def _drop(*path: str) -> Callable[[dict], None]:
    def change(d: dict) -> None:
        for key in path[:-1]:
            d = d[key]
        del d[path[-1]]
    return change


def _pose(x: float, y: float, z: float) -> np.ndarray:
    """A TCP pointing down, turned about z by (cos, sin) = (0.6, 0.8): exact literals, no trigonometry."""
    return np.array([[0.6, 0.8, 0.0, x],
                     [0.8, -0.6, 0.0, y],
                     [0.0, 0.0, -1.0, z],
                     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
_T0 = 123_456_789_000_000          # a PC monotonic time, ns


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
    t = np.array([t0 + i * STEP_MS * 1_000_000 for i in range(n)], dtype=np.int64)
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
# C0 (in memory)

_H264_KEY = (b"\x00\x00\x00\x01\x67\x64\x00\x28\xac" + b"\x00\x00\x00\x01\x68\xee\x3c\x80"
             + b"\x00\x00\x00\x01\x65\x88\x84\x00\x33\xff")       # SPS, PPS, IDR (stand-in bytes)
_H264_DELTA = b"\x00\x00\x00\x01\x41\x9a\x24\x6c\x41"            # one non-IDR slice (stand-in bytes)
_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 16 + b"\xff\xd9"


def _c0() -> list[Example]:
    key = Frame(camera="main", sensor="imx519", seq=300, t_pi_ns=5_000_000_000, exposure_ns=8_000_000, width=1600,
                height=900, encoding="h264", keyframe=True, data=_H264_KEY)
    delta = Frame(camera="ultrawide", sensor="imx219", seq=301, t_pi_ns=5_033_333_333, exposure_ns=0, width=1640,
                  height=1232, encoding="h264", keyframe=False, data=_H264_DELTA)
    jpeg = Frame(camera="main", sensor="imx519", seq=7, t_pi_ns=6_000_000_000, exposure_ns=10_000_000, width=1600,
                 height=900, encoding="jpeg", keyframe=True, data=_JPEG)
    reply = TimeSyncReply(t_pi_ns=5_100_000_000)
    good = [
        _good("c0/frame_h264_keyframe", key, {"camera": "main", "sensor": "imx519", "seq": 300, "keyframe": True,
                                              "width": 1600, "height": 900, "exposure_ns": 8_000_000}),
        _good("c0/frame_h264_delta", delta, {"camera": "ultrawide", "keyframe": False, "exposure_ns": 0}),
        _good("c0/frame_jpeg", jpeg, {"encoding": "jpeg", "keyframe": True}),
        _good("c0/time_sync_request", TimeSyncRequest()),
        _good("c0/time_sync_reply", reply, {"t_pi_ns": 5_100_000_000}),
    ]
    broken = [
        _broken("c0/frame_bad_camera", key, _set("camera", "left"), "camera", "value", "unknown camera"),
        _broken("c0/frame_bad_sensor", key, _set("sensor", "imx477"), "sensor", "value", "unknown sensor"),
        _broken("c0/frame_bad_encoding", key, _set("encoding", "h265"), "encoding", "value", "unknown encoding"),
        _broken("c0/frame_jpeg_not_keyframe", jpeg, _set("keyframe", False), "keyframe", "value",
                "every jpeg frame decodes on its own"),
        _broken("c0/frame_empty_data", key, _set("data", b""), "data", "value", "empty frame"),
        _broken("c0/frame_data_as_str", key, _set("data", "frame"), "data", "type", "str instead of bin"),
        _broken("c0/frame_negative_exposure", key, _set("exposure_ns", -1), "exposure_ns", "value",
                "negative exposure"),
        _broken("c0/frame_zero_width", key, _set("width", 0), "width", "value", "width 0"),
        _broken("c0/frame_seq_negative", key, _set("seq", -5), "seq", "value", "negative uint64"),
        _broken("c0/frame_keyframe_int", key, _set("keyframe", 1), "keyframe", "type", "bool as an integer"),
        _broken("c0/frame_missing_data", key, _drop("data"), "data", "missing", "data left out"),
        _broken("c0/frame_unknown_field", key, _set("gain", 2.0), "gain", "unknown_field",
                "a field the contract doesn't have"),
        _broken("c0/frame_schema_version", key, _set("schema", "c0.frame/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c0.frame/v2"),
        _broken("c0/time_sync_reply_float", reply, _set("t_pi_ns", 5.1e9), "t_pi_ns", "type", "float time"),
    ]
    return good + broken


# ----------------------------------------------------------------------------------------------------------------
# C1 (in memory)


def _images(seed: int) -> np.ndarray:
    n = OBS_STEPS * IMAGE_SIZE * IMAGE_SIZE * 3
    return ((np.arange(n, dtype=np.int64) * 7 + seed) % 256).astype(np.uint8).reshape(
        (OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3))


def _health(main: str = "ok", ultrawide: str = "ok") -> HealthReport:
    def cam(status: str, age: float) -> CameraHealthReport:
        return CameraHealthReport(status=status, last_frame_age_ms=age, fps=30.0 if status == "ok" else 0.0,
                                  dropped=2)
    age = {"ok": 20.0, "stale": 500.0, "down": 2500.0}
    return HealthReport(main=cam(main, age[main]), ultrawide=cam(ultrawide, age[ultrawide]))


def _c1() -> list[Example]:
    t_main = np.array([_T0, _T0 + 50_000_000], dtype=np.int64)
    t_uw = np.array([_T0 - 50_000_000, _T0 + 50_000_000], dtype=np.int64)
    ok = FramesReply(main=_images(1), main_t_ns=t_main, ultrawide=_images(2), ultrawide_t_ns=t_uw, health=_health())
    stale = FramesReply(main=None, main_t_ns=None, ultrawide=None, ultrawide_t_ns=None,
                        health=_health(main="stale"))
    down = FramesReply(main=None, main_t_ns=None, ultrawide=None, ultrawide_t_ns=None,
                       health=_health(ultrawide="down"))
    health = HealthReply(health=_health())

    def images_while_down(d: dict) -> None:
        d.update(ok.to_dict())
        d["health"] = _health(ultrawide="down").to_dict()

    def times_back(d: dict) -> None:
        d["main_t_ns"] = d["main_t_ns"][::-1].copy()

    def status(cam: str, value: str) -> Callable[[dict], None]:
        def change(d: dict) -> None:
            d["health"][cam]["status"] = value
        return change

    def fps(d: dict) -> None:
        d["health"]["ultrawide"]["fps"] = -1.0

    good = [
        _good("c1/get_frames", GetFramesRequest()),
        _good("c1/get_health", GetHealthRequest()),
        _good("c1/frames_ok", ok, {"main_t_ns[1]": _T0 + 50_000_000, "health.main.status": "ok",
                                   "health.ultrawide.fps": 30.0, "main[0][0][0][1]": 8}),
        _good("c1/frames_main_stale", stale, {"health.main.status": "stale", "health.ultrawide.status": "ok"},
              note="one camera stale: no images at all, only health"),
        _good("c1/frames_ultrawide_down", down, {"health.ultrawide.status": "down"}),
        _good("c1/health", health, {"health.main.dropped": 2, "health.main.last_frame_age_ms": 20.0}),
    ]
    broken = [
        _broken("c1/frames_images_while_down", down, images_while_down, "main", "value",
                "images while a camera is down"),
        _broken("c1/frames_missing_images_while_ok", ok, _set("ultrawide", None), "ultrawide", "value",
                "both cameras ok but no ultrawide images"),
        _broken("c1/frames_wrong_shape", ok, lambda d: d.update(main=d["main"][:, :, 0, :].copy()), "main", "shape",
                "(2, 224, 3) images"),
        _broken("c1/frames_wrong_dtype", ok, lambda d: d.update(ultrawide=d["ultrawide"].astype(np.float32)),
                "ultrawide", "dtype", "float32 images"),
        _broken("c1/frames_times_not_increasing", ok, times_back, "main_t_ns", "value", "newest first"),
        _broken("c1/health_bad_status", health, status("main", "dead"), "health.main.status", "value",
                "unknown status"),
        _broken("c1/health_negative_fps", health, fps, "health.ultrawide.fps", "value", "negative frame rate"),
        _broken("c1/health_missing_dropped", health, _drop("health", "main", "dropped"), "health.main.dropped",
                "missing", "dropped left out"),
        _broken("c1/frames_times_as_list", ok, lambda d: d.update(main_t_ns=d["main_t_ns"].tolist()), "main_t_ns",
                "type", "a plain list instead of an array map"),
        _broken("c1/frames_missing_health", stale, _drop("health"), "health", "missing", "health left out"),
        _broken("c1/frames_unknown_field", stale, _set("depth", None), "depth", "unknown_field",
                "a field the contract doesn't have"),
        _broken("c1/frames_schema_version", stale, _set("schema", "c1.frames/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c1.frames/v2"),
    ]
    return good + broken


# ----------------------------------------------------------------------------------------------------------------
# C4 (in memory)


def _rot6d_about_z(angle: float) -> list[float]:
    c, s = math.cos(angle), math.sin(angle)
    return [c, -s, 0.0, s, c, 0.0]


def observation() -> Observation:
    """A valid observation: the hand moved 2 mm and turned 0.02 rad over the last step."""
    pos = np.array([[0.002, -0.001, 0.0005], [0.0, 0.0, 0.0]], dtype=np.float32)
    rot = np.array([_rot6d_about_z(0.02), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    wrt_start = np.array([_rot6d_about_z(0.1), _rot6d_about_z(0.12)], dtype=np.float32)
    width = np.array([[0.05], [0.049]], dtype=np.float32)
    return Observation(eef_pos=pos, eef_rot6d=rot, eef_rot6d_wrt_start=wrt_start, gripper_width_m=width,
                       main=_images(3), ultrawide=_images(4))


def action_chunk(step0_m: float = 0.0, width_clipped: int = 0) -> ActionChunk:
    """A valid chunk moving 1 mm per step along x; step 0 starts step0_m from the anchor."""
    pos = np.array([[step0_m + 0.001 * k, 0.0, 0.0] for k in range(ACTION_STEPS)], dtype=np.float32)
    rot = np.array([_rot6d_about_z(0.001 * k) for k in range(ACTION_STEPS)], dtype=np.float32)
    width = np.full((ACTION_STEPS, 1), 0.04, dtype=np.float32)
    return ActionChunk(eef_pos=pos, eef_rot6d=rot, gripper_width_m=width, width_clipped=width_clipped)


def _c4() -> list[Example]:
    obs = observation()
    chunk = action_chunk()
    config = PolicyConfig(obs_steps=OBS_STEPS, action_steps=ACTION_STEPS, step_ms=STEP_MS, image_size=IMAGE_SIZE,
                          inputs=OBSERVATION_INPUTS, uses_prompt=False)
    clipped_width = np.full((ACTION_STEPS, 1), 0.04, dtype=np.float32)
    clipped_width[3, 0] = np.float32(ContractLimits.load().gripper_width_max_m)
    clipped_width[9, 0] = 0.0
    clipped = ActionChunk(eef_pos=chunk.eef_pos, eef_rot6d=chunk.eef_rot6d, gripper_width_m=clipped_width,
                          width_clipped=2)

    def at(key: str, row: int, col: int, value: float) -> Callable[[dict], None]:
        def change(d: dict) -> None:
            a = d[key].copy()
            a[row, col] = value
            d[key] = a
        return change

    def not_perpendicular(d: dict) -> None:
        a = d["eef_rot6d"].copy()
        a[2] = [1.0, 0.0, 0.0, 0.6, 0.8, 0.0]     # unit vectors, |cos| = 0.6
        d["eef_rot6d"] = a

    def bad_input(d: dict) -> None:
        d["inputs"] = ["eef_pos", "depth"]

    max_m = ContractLimits.load().gripper_width_max_m
    good = [
        _good("c4/config_request", PolicyConfigRequest()),
        _good("c4/config", config, {"obs_steps": OBS_STEPS, "action_steps": ACTION_STEPS, "step_ms": STEP_MS,
                                    "image_size": IMAGE_SIZE, "uses_prompt": False, "inputs[0]": "eef_pos"}),
        _good("c4/reset", Reset()),
        _good("c4/prompt", Prompt(task_name="block_raise", prompt_index=0), {"task_name": "block_raise"}),
        _good("c4/observation", obs, {"eef_pos[1][0]": 0.0, "eef_rot6d[1][0]": 1.0}),
        _good("c4/chunk", chunk, {"width_clipped": 0, "eef_pos[1][0]": float(np.float32(0.001))}),
        _good("c4/chunk_odd_step0_5cm", action_chunk(step0_m=0.05), {"eef_pos[0][0]": float(np.float32(0.05))},
              note="step 0 is 5 cm from the anchor: odd for message_comms, still valid for the contract"),
        _good("c4/chunk_clipped", clipped, {"width_clipped": 2, "gripper_width_m[9][0]": 0.0},
              note="the policy clipped two widths into range and counted them"),
    ]
    broken = [
        _broken("c4/chunk_nan", chunk, at("eef_pos", 5, 1, float("nan")), "eef_pos", "value", "NaN position"),
        _broken("c4/chunk_inf_width", chunk, at("gripper_width_m", 2, 0, float("inf")), "gripper_width_m", "value",
                "infinite width"),
        _broken("c4/chunk_rot6d_not_unit", chunk, at("eef_rot6d", 4, 0, 1.5), "eef_rot6d", "value",
                "first rot6d vector of length 1.5"),
        _broken("c4/chunk_rot6d_not_perpendicular", chunk, not_perpendicular, "eef_rot6d", "value",
                "rot6d vectors at |cos| 0.6"),
        _broken("c4/chunk_wrong_shape", chunk, lambda d: d.update(eef_pos=d["eef_pos"][:15].copy()), "eef_pos",
                "shape", "15 steps"),
        _broken("c4/chunk_wrong_dtype", chunk, lambda d: d.update(eef_rot6d=d["eef_rot6d"].astype(np.float64)),
                "eef_rot6d", "dtype", "float64 instead of float32"),
        _broken("c4/chunk_width_clipped_missing", chunk, _drop("width_clipped"), "width_clipped", "missing",
                "width_clipped left out"),
        _broken("c4/chunk_width_clipped_too_big", chunk, _set("width_clipped", ACTION_STEPS + 1), "width_clipped",
                "value", "more clipped widths than steps"),
        _broken("c4/chunk_width_above_max", chunk, at("gripper_width_m", 0, 0, max_m + 0.001), "gripper_width_m",
                "value", "max opening + 1 mm (the validator does not clip)"),
        _broken("c4/chunk_width_in_mm", chunk, at("gripper_width_m", 0, 0, 40.0), "gripper_width_m", "value",
                "width in mm instead of m"),
        _broken("c4/chunk_width_clipped_float", chunk, _set("width_clipped", 0.0), "width_clipped", "type",
                "a float count"),
        _broken("c4/chunk_unknown_field", chunk, _set("confidence", 0.9), "confidence", "unknown_field",
                "a field the contract doesn't have"),
        _broken("c4/chunk_schema_version", chunk, _set("schema", "c4.chunk/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c4.chunk/v2"),
        _broken("c4/observation_anchor_row", obs, at("eef_pos", 1, 2, 0.001), "eef_pos", "value",
                "last position row not the anchor"),
        _broken("c4/observation_anchor_rot6d", obs, at("eef_rot6d", 1, 1, 0.01), "eef_rot6d", "value",
                "last rot6d row not identity"),
        _broken("c4/observation_wrt_start_not_unit", obs, at("eef_rot6d_wrt_start", 0, 3, 2.0),
                "eef_rot6d_wrt_start", "value", "second rot6d vector too long"),
        _broken("c4/observation_negative_width", obs, at("gripper_width_m", 0, 0, -0.01), "gripper_width_m",
                "value", "negative opening"),
        _broken("c4/observation_wrong_image_shape", obs, lambda d: d.update(main=d["main"][:, :112].copy()),
                "main", "shape", "112 rows"),
        _broken("c4/observation_missing_ultrawide", obs, _drop("ultrawide"), "ultrawide", "missing",
                "ultrawide left out"),
        _broken("c4/observation_schema_version", obs, _set("schema", "c4.infer/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c4.infer/v2"),
        _broken("c4/config_unknown_input", config, bad_input, "inputs[1]", "value", "an input that isn't a field"),
        _broken("c4/config_zero_steps", config, _set("obs_steps", 0), "obs_steps", "value", "obs_steps 0"),
        _broken("c4/prompt_negative_index", Prompt(task_name="block_raise", prompt_index=0), _set("prompt_index", -1),
                "prompt_index", "value", "negative prompt index"),
    ]
    return good + broken


def build_examples() -> list[Example]:
    """Every example of every contract, good and broken, in a fixed order."""
    out = _c0() + _c1() + _c2() + _c3() + _c4()
    names = [e.name for e in out]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate example names")
    return out
