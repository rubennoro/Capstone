# adapted from rnoro/deployment:deployment/contracts/contracts/examples.py @ 91b178b (lines 402-694, the C0/C1/C4 part)
# changes: built with deploy_contracts' public helpers (good_example, broken_example, set_key, drop_key); the time-sync
# examples dropped; + the c4.config.reply/v2 examples (kind missing, kind empty, the old v1); + the pi_stream/v1 wire
# examples (PiStreamExample: good, wrong part count, header not JSON, missing key, boolean packet_no, version 2, empty
# JPEG). Built in memory, never stored (STRUCTURE.md 3.5).
"""The laptop-only examples: C0 (the Pi's pi_stream/v1 messages and the internal c0.packet/v1 form), C1 and C4.

Good ones are built from the message classes; broken ones take a good message's dict and change one thing, then pack
it (deploy_contracts.broken_example). Value paths as in deploy_contracts.examples (dotted, [i] for an index).
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from deploy_contracts import ContractLimits, Example, broken_example, drop_key, good_example, set_key

from policy_deploy.contracts.c0_packet import FrameInfo, PacketHeader
from policy_deploy.contracts.c1_frames import (CameraHealthReport, FramesReply, GetFramesRequest, GetHealthRequest,
                                               HealthReply, HealthReport)
from policy_deploy.contracts.c4_policy import (OBSERVATION_INPUTS, ActionChunk, Observation, PolicyConfig,
                                               PolicyConfigRequest, Prompt, Reset)
from policy_deploy.contracts.sizes import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS

_T0 = 123_456_789_000_000          # a laptop monotonic time, ns

# ----------------------------------------------------------------------------------------------------------------
# C0, the Pi's wire format pi_stream/v1 (in memory; PI_PAYLOAD.md section 0)

#: A header as the live Pi sent it on 2026-11-12 (PI_PAYLOAD.md 0.6). Its skew_ns has the live Pi's sign
#: (ultrawide - main), the opposite of 0.3's definition: nothing may use it.
PI_HEADER: dict[str, Any] = {
    "version": 1, "packet_no": 800, "t_send_ns": 1269437832529, "clock_offset_ns": 1794539164607668815,
    "main": {"frame_no": 800, "t_sensor_ns": 1269392013000, "exposure_us": 25640, "t_mid_ns": 1269379193000},
    "ultrawide": {"frame_no": 799, "t_sensor_ns": 1269412129000, "exposure_us": 33287, "t_mid_ns": 1269395485500},
    "skew_ns": 16292500, "uw_reused": False}


@dataclass(frozen=True)
class PiStreamExample:
    """One pi_stream/v1 message: `parts` is the exact multipart message (header JSON, main JPEG, ultrawide JPEG)."""

    name: str
    valid: bool
    parts: tuple[bytes, ...]
    field: str | None = None
    problem: str | None = None
    note: str = ""


def tiny_jpeg(w: int, h: int) -> bytes:
    """A minimal byte string with SOI, an APP0 segment and a baseline SOF0 header (enough for jpeg_size)."""
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0\x00\x11\x08" + h.to_bytes(2, "big") + w.to_bytes(2, "big") + b"\x03" + b"\x01\x22\x00" * 3
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xda\x00\x02" + b"\x00" * 8 + b"\xff\xd9"


def pi_parts(header: dict[str, Any] | None = None, main: bytes | None = None,
             ultrawide: bytes | None = None) -> tuple[bytes, ...]:
    """A pi_stream/v1 message: the header as JSON and two JPEGs (tiny 640x480 ones by default)."""
    h = PI_HEADER if header is None else header
    return (json.dumps(h).encode(), tiny_jpeg(640, 480) if main is None else main,
            tiny_jpeg(640, 480) if ultrawide is None else ultrawide)


def _pi_header(change: Callable[[dict], None]) -> dict[str, Any]:
    h = copy.deepcopy(PI_HEADER)
    change(h)
    return h


def pi_stream_examples() -> list[PiStreamExample]:
    good = pi_parts()
    return [
        PiStreamExample("c0/pi_stream_live", True, good, note="the live Pi's packet of PI_PAYLOAD.md 0.6"),
        PiStreamExample("c0/pi_stream_two_parts", False, good[:2], "parts", "value", "the ultrawide JPEG left out"),
        PiStreamExample("c0/pi_stream_four_parts", False, good + (b"x",), "parts", "value", "one part too many"),
        PiStreamExample("c0/pi_stream_header_not_json", False, (b"\x82\xa1a\x01", good[1], good[2]), "parts[0]",
                        "value", "a msgpack header instead of JSON"),
        PiStreamExample("c0/pi_stream_missing_key", False, pi_parts(_pi_header(lambda h: h.pop("t_send_ns"))),
                        "t_send_ns", "missing", "no send stamp: the clock map needs it"),
        PiStreamExample("c0/pi_stream_packet_no_bool", False,
                        pi_parts(_pi_header(lambda h: h.update(packet_no=True))), "packet_no", "type",
                        "a boolean is not an integer"),
        PiStreamExample("c0/pi_stream_version_2", False, pi_parts(_pi_header(lambda h: h.update(version=2))),
                        "version", "schema_version", "a version this receiver doesn't read"),
        PiStreamExample("c0/pi_stream_empty_main", False, pi_parts(main=b""), "parts[1]", "value", "empty main JPEG"),
        PiStreamExample("c0/pi_stream_empty_ultrawide", False, pi_parts(ultrawide=b""), "parts[2]", "value",
                        "empty ultrawide JPEG"),
    ]


def _main(seq: int, t_pi_ns: int, **change: Any) -> FrameInfo:
    d = dict(sensor="imx519", frame_seq=seq, t_pi_ns=t_pi_ns, exposure_ns=8_000_000, width=1600, height=1200,
             encoding="h264", keyframe=False)
    d.update(change)
    return FrameInfo(**d)


def _uw(seq: int, t_pi_ns: int, **change: Any) -> FrameInfo:
    d = dict(sensor="imx219", frame_seq=seq, t_pi_ns=t_pi_ns, exposure_ns=6_000_000, width=1280, height=960,
             encoding="h264", keyframe=False)
    d.update(change)
    return FrameInfo(**d)


def _in(*path: Any) -> Callable[[Callable[[dict], Any]], Callable[[dict], None]]:
    """Apply a change to a sub-map: _in("ultrawide", 0)(set_key("sensor", "imx519"))."""
    def wrap(change: Callable[[dict], Any]) -> Callable[[dict], None]:
        def apply(d: dict) -> None:
            node: Any = d
            for key in path:
                node = node[key]
            change(node)
        return apply
    return wrap


def _c0() -> list[Example]:
    # main 1201 with its pair, ultrawide 1188, captured 5.3 ms before it (the original PI_PAYLOAD.md section 8)
    one = PacketHeader(packet_seq=1201, main=_main(1201, 932_118_204_000),
                       ultrawide=(_uw(1188, 932_112_937_000),), pair_seq=1188)
    none_in = PacketHeader(packet_seq=1202, main=_main(1202, 932_151_537_000), ultrawide=(), pair_seq=1188)
    two = PacketHeader(packet_seq=1203, main=_main(1203, 932_184_870_000, keyframe=True),
                       ultrawide=(_uw(1189, 932_146_270_000), _uw(1190, 932_179_603_000, keyframe=True)),
                       pair_seq=1190)
    unpaired = PacketHeader(packet_seq=7, main=_main(7, 6_000_000_000, exposure_ns=0), ultrawide=(), pair_seq=-1)
    jpeg = PacketHeader(packet_seq=3, main=_main(3, 6_100_000_000, encoding="jpeg", keyframe=True),
                        ultrawide=(_uw(2, 6_095_000_000, encoding="jpeg", keyframe=True),), pair_seq=2)
    good = [
        good_example("c0/packet_one_ultrawide", one, {"packet_seq": 1201, "main.sensor": "imx519", "main.frame_seq": 1201,
                                               "main.width": 1600, "main.height": 1200, "ultrawide[0].frame_seq": 1188,
                                               "ultrawide[0].width": 1280, "pair_seq": 1188},
              note="the original PI_PAYLOAD.md section 8"),
        good_example("c0/packet_pair_from_earlier_packet", none_in, {"pair_seq": 1188}, note="no ultrawide frame in it"),
        good_example("c0/packet_two_ultrawide", two, {"ultrawide[1].frame_seq": 1190, "ultrawide[1].keyframe": True,
                                               "main.keyframe": True}),
        good_example("c0/packet_no_pair", unpaired, {"pair_seq": -1, "main.exposure_ns": 0}),
        good_example("c0/packet_jpeg", jpeg, {"main.encoding": "jpeg", "ultrawide[0].keyframe": True}),
    ]
    broken = [
        broken_example("c0/packet_main_sensor_imx219", one, _in("main")(set_key("sensor", "imx219")), "main.sensor", "value",
                "cameras swapped: the main frame says imx219"),
        broken_example("c0/packet_ultrawide_sensor_imx519", one, _in("ultrawide", 0)(set_key("sensor", "imx519")),
                "ultrawide[0].sensor", "value", "cameras swapped: an ultrawide frame says imx519"),
        broken_example("c0/packet_unknown_sensor", one, _in("main")(set_key("sensor", "imx477")), "main.sensor", "value",
                "unknown sensor"),
        broken_example("c0/packet_bad_encoding", one, _in("main")(set_key("encoding", "h265")), "main.encoding", "value",
                "unknown encoding"),
        broken_example("c0/packet_jpeg_not_keyframe", jpeg, _in("ultrawide", 0)(set_key("keyframe", False)),
                "ultrawide[0].keyframe", "value", "every jpeg frame decodes on its own"),
        broken_example("c0/packet_mixed_ultrawide_encodings", two, _in("ultrawide", 1)(set_key("encoding", "jpeg")),
                "ultrawide[1].encoding", "value", "one camera, two encodings"),
        broken_example("c0/packet_negative_exposure", one, _in("main")(set_key("exposure_ns", -1)), "main.exposure_ns",
                "value", "negative exposure"),
        broken_example("c0/packet_width_negative", one, _in("main")(set_key("width", -1600)), "main.width", "value",
                "width is uint32"),
        broken_example("c0/packet_height_too_big", one, _in("ultrawide", 0)(set_key("height", 2**32)), "ultrawide[0].height",
                "value", "height beyond uint32"),
        broken_example("c0/packet_seq_negative", one, set_key("packet_seq", -5), "packet_seq", "value", "negative uint64"),
        broken_example("c0/packet_frame_seq_negative", one, _in("ultrawide", 0)(set_key("frame_seq", -1)),
                "ultrawide[0].frame_seq", "value", "negative uint64"),
        broken_example("c0/packet_keyframe_int", one, _in("main")(set_key("keyframe", 1)), "main.keyframe", "type",
                "bool as an integer"),
        broken_example("c0/packet_t_pi_float", one, _in("main")(set_key("t_pi_ns", 9.3e11)), "main.t_pi_ns", "type",
                "float time"),
        broken_example("c0/packet_pair_seq_below_minus_one", one, set_key("pair_seq", -2), "pair_seq", "value",
                "pair_seq is -1 or a frame_seq"),
        broken_example("c0/packet_pair_seq_newer_than_packet", one, set_key("pair_seq", 1189), "pair_seq", "value",
                "a pair newer than every ultrawide frame sent"),
        broken_example("c0/packet_ultrawide_out_of_order", two, _in("ultrawide", 1)(set_key("frame_seq", 1189)),
                "ultrawide[1].frame_seq", "value", "not in capture order"),
        broken_example("c0/packet_ultrawide_not_a_list", one, lambda d: d.update(ultrawide=d["ultrawide"][0]), "ultrawide",
                "type", "a single map instead of a list"),
        broken_example("c0/packet_missing_pair_seq", one, drop_key("pair_seq"), "pair_seq", "missing", "pair_seq left out"),
        broken_example("c0/packet_missing_keyframe", one, drop_key("main", "keyframe"), "main.keyframe", "missing",
                "a frame map without keyframe"),
        broken_example("c0/packet_unknown_field", one, set_key("gain", 2), "gain", "unknown_field",
                "a field the contract doesn't have"),
        broken_example("c0/packet_unknown_frame_field", one, _in("main")(set_key("gain", 2)), "main.gain", "unknown_field",
                "a frame-map key the contract doesn't have"),
        broken_example("c0/packet_schema_version", one, set_key("schema", "c0.packet/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c0.packet/v2"),
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
        good_example("c1/get_frames", GetFramesRequest()),
        good_example("c1/get_health", GetHealthRequest()),
        good_example("c1/frames_ok", ok, {"main_t_ns[1]": _T0 + 50_000_000, "health.main.status": "ok",
                                   "health.ultrawide.fps": 30.0, "main[0][0][0][1]": 8}),
        good_example("c1/frames_main_stale", stale, {"health.main.status": "stale", "health.ultrawide.status": "ok"},
              note="one camera stale: no images at all, only health"),
        good_example("c1/frames_ultrawide_down", down, {"health.ultrawide.status": "down"}),
        good_example("c1/health", health, {"health.main.dropped": 2, "health.main.last_frame_age_ms": 20.0}),
    ]
    broken = [
        broken_example("c1/frames_images_while_down", down, images_while_down, "main", "value",
                "images while a camera is down"),
        broken_example("c1/frames_missing_images_while_ok", ok, set_key("ultrawide", None), "ultrawide", "value",
                "both cameras ok but no ultrawide images"),
        broken_example("c1/frames_wrong_shape", ok, lambda d: d.update(main=d["main"][:, :, 0, :].copy()), "main", "shape",
                "(2, 224, 3) images"),
        broken_example("c1/frames_wrong_dtype", ok, lambda d: d.update(ultrawide=d["ultrawide"].astype(np.float32)),
                "ultrawide", "dtype", "float32 images"),
        broken_example("c1/frames_times_not_increasing", ok, times_back, "main_t_ns", "value", "newest first"),
        broken_example("c1/health_bad_status", health, status("main", "dead"), "health.main.status", "value",
                "unknown status"),
        broken_example("c1/health_negative_fps", health, fps, "health.ultrawide.fps", "value", "negative frame rate"),
        broken_example("c1/health_missing_dropped", health, drop_key("health", "main", "dropped"), "health.main.dropped",
                "missing", "dropped left out"),
        broken_example("c1/frames_times_as_list", ok, lambda d: d.update(main_t_ns=d["main_t_ns"].tolist()), "main_t_ns",
                "type", "a plain list instead of an array map"),
        broken_example("c1/frames_missing_health", stale, drop_key("health"), "health", "missing", "health left out"),
        broken_example("c1/frames_unknown_field", stale, set_key("depth", None), "depth", "unknown_field",
                "a field the contract doesn't have"),
        broken_example("c1/frames_schema_version", stale, set_key("schema", "c1.frames/v2"), "", "schema_version",
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
                          inputs=OBSERVATION_INPUTS, uses_prompt=False, kind="hold_still")
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
        good_example("c4/config_request", PolicyConfigRequest()),
        good_example("c4/config", config, {"obs_steps": OBS_STEPS, "action_steps": ACTION_STEPS, "step_ms": STEP_MS,
                                    "image_size": IMAGE_SIZE, "uses_prompt": False, "inputs[0]": "eef_pos",
                                    "kind": "hold_still"}),
        good_example("c4/reset", Reset()),
        good_example("c4/prompt", Prompt(task_name="block_raise", prompt_index=0), {"task_name": "block_raise"}),
        good_example("c4/observation", obs, {"eef_pos[1][0]": 0.0, "eef_rot6d[1][0]": 1.0}),
        good_example("c4/chunk", chunk, {"width_clipped": 0, "eef_pos[1][0]": float(np.float32(0.001))}),
        good_example("c4/chunk_odd_step0_5cm", action_chunk(step0_m=0.05), {"eef_pos[0][0]": float(np.float32(0.05))},
              note="step 0 is 5 cm from the anchor: odd for message_comms, still valid for the contract"),
        good_example("c4/chunk_clipped", clipped, {"width_clipped": 2, "gripper_width_m[9][0]": 0.0},
              note="the policy clipped two widths into range and counted them"),
    ]
    broken = [
        broken_example("c4/chunk_nan", chunk, at("eef_pos", 5, 1, float("nan")), "eef_pos", "value", "NaN position"),
        broken_example("c4/chunk_inf_width", chunk, at("gripper_width_m", 2, 0, float("inf")), "gripper_width_m", "value",
                "infinite width"),
        broken_example("c4/chunk_rot6d_not_unit", chunk, at("eef_rot6d", 4, 0, 1.5), "eef_rot6d", "value",
                "first rot6d vector of length 1.5"),
        broken_example("c4/chunk_rot6d_not_perpendicular", chunk, not_perpendicular, "eef_rot6d", "value",
                "rot6d vectors at |cos| 0.6"),
        broken_example("c4/chunk_wrong_shape", chunk, lambda d: d.update(eef_pos=d["eef_pos"][:15].copy()), "eef_pos",
                "shape", "15 steps"),
        broken_example("c4/chunk_wrong_dtype", chunk, lambda d: d.update(eef_rot6d=d["eef_rot6d"].astype(np.float64)),
                "eef_rot6d", "dtype", "float64 instead of float32"),
        broken_example("c4/chunk_width_clipped_missing", chunk, drop_key("width_clipped"), "width_clipped", "missing",
                "width_clipped left out"),
        broken_example("c4/chunk_width_clipped_too_big", chunk, set_key("width_clipped", ACTION_STEPS + 1), "width_clipped",
                "value", "more clipped widths than steps"),
        broken_example("c4/chunk_width_above_max", chunk, at("gripper_width_m", 0, 0, max_m + 0.001), "gripper_width_m",
                "value", "max opening + 1 mm (the validator does not clip)"),
        broken_example("c4/chunk_width_in_mm", chunk, at("gripper_width_m", 0, 0, 40.0), "gripper_width_m", "value",
                "width in mm instead of m"),
        broken_example("c4/chunk_width_clipped_float", chunk, set_key("width_clipped", 0.0), "width_clipped", "type",
                "a float count"),
        broken_example("c4/chunk_unknown_field", chunk, set_key("confidence", 0.9), "confidence", "unknown_field",
                "a field the contract doesn't have"),
        broken_example("c4/chunk_schema_version", chunk, set_key("schema", "c4.chunk/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c4.chunk/v2"),
        broken_example("c4/observation_anchor_row", obs, at("eef_pos", 1, 2, 0.001), "eef_pos", "value",
                "last position row not the anchor"),
        broken_example("c4/observation_anchor_rot6d", obs, at("eef_rot6d", 1, 1, 0.01), "eef_rot6d", "value",
                "last rot6d row not identity"),
        broken_example("c4/observation_wrt_start_not_unit", obs, at("eef_rot6d_wrt_start", 0, 3, 2.0),
                "eef_rot6d_wrt_start", "value", "second rot6d vector too long"),
        broken_example("c4/observation_negative_width", obs, at("gripper_width_m", 0, 0, -0.01), "gripper_width_m",
                "value", "negative opening"),
        broken_example("c4/observation_wrong_image_shape", obs, lambda d: d.update(main=d["main"][:, :112].copy()),
                "main", "shape", "112 rows"),
        broken_example("c4/observation_missing_ultrawide", obs, drop_key("ultrawide"), "ultrawide", "missing",
                "ultrawide left out"),
        broken_example("c4/observation_schema_version", obs, set_key("schema", "c4.infer/v2"), "", "schema_version",
                "a version this side doesn't know", schema="c4.infer/v2"),
        broken_example("c4/config_unknown_input", config, bad_input, "inputs[1]", "value", "an input that isn't a field"),
        broken_example("c4/config_zero_steps", config, set_key("obs_steps", 0), "obs_steps", "value", "obs_steps 0"),
        broken_example("c4/config_kind_missing", config, drop_key("kind"), "kind", "missing",
                       "a v2 config reply without kind"),
        broken_example("c4/config_kind_empty", config, set_key("kind", ""), "kind", "value", "kind empty"),
        broken_example("c4/config_v1", config, lambda d: {k: v for k, v in d.items() if k != "kind"} | {"schema": "c4.config.reply/v1"},
                       "", "schema_version", "the old v1 config reply (no kind)", schema="c4.config.reply/v1"),
        broken_example("c4/prompt_negative_index", Prompt(task_name="block_raise", prompt_index=0), set_key("prompt_index", -1),
                "prompt_index", "value", "negative prompt index"),
    ]
    return good + broken


def build_examples() -> list[Example]:
    """Every msgpack example of C0 (internal packet form), C1 and C4, good and broken, in a fixed order."""
    out = _c0() + _c1() + _c4()
    names = [e.name for e in out]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate example names")
    return out
