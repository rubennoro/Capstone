"""The iPhone episode JSON (R1 §2): all 32 keys, Swift JSONEncoder-like compact output with float32 shortest repr."""

from __future__ import annotations

import math

import numpy as np

# key -> expected Python type of the parsed value (R1 §2.2)
KEY_TYPES: dict[str, type] = {
    "poseTimes": list, "poseTransforms": list, "rgbTimes": list, "ultrawideRGBTimes": list,
    "depthTimes": list, "depthFrameIsRepeat": list, "recordingStartTime": str, "side": str, "type": str,
    "gripperCalibrationRunName": str, "sessionName": str, "gripperID": str, "hasRGB": bool,
    "hasUltrawideRGB": bool, "hasDepth": bool, "frameCount": int, "taskStartTimestamps": list,
    "taskEndTimestamps": list, "taskNames": list, "narrationStartTimestamps": list, "narrationEndTimestamps": list,
    "narrationTexts": list, "labelType": str, "hasAudio": bool, "isVoiceHost": bool, "sidesPresent": list,
    "formatVersion": str, "deviceName": str, "deviceIdentifier": str, "mainCameraIntrinsics": list,
    "ultrawideCameraIntrinsics": list, "isErrorCorrection": bool,
}
KEYS = tuple(KEY_TYPES)
assert len(KEYS) == 32


def _f32_list(M) -> list:
    return np.asarray(M, dtype=np.float32).tolist()


def build_json(*, kind: str, frame_count: int, pose_times: list[str], pose_transforms, uw_times: list[str],
               recording_start_time: str, side: str, session: str, gripper_id: str, calibration_run: str,
               sides_present: list[str], device_name: str, device_identifier: str, K_main, K_uw,
               task_label: str | None = None) -> dict:
    """Assemble the JSON dict. For calibrations ``pose_times``/``pose_transforms`` must be empty."""
    demo = kind == "demonstration"
    if kind not in ("demonstration", "gripper_calibration"):
        raise ValueError(f"unknown kind {kind!r}")
    if len(uw_times) != frame_count:
        raise ValueError("ultrawideRGBTimes must have frameCount entries")
    if demo and not (len(pose_times) == len(pose_transforms) == frame_count):
        raise ValueError("poseTimes/poseTransforms must have frameCount entries")
    if not demo and (pose_times or len(pose_transforms)):
        raise ValueError("calibrations carry no poses")
    tasks = ([task_label], [pose_times[0]], [pose_times[-1]]) if (demo and task_label) else ([], [], [])
    poses = np.asarray(pose_transforms, dtype=np.float32).reshape(-1, 4, 4) if demo else np.zeros((0, 4, 4))
    return {
        "poseTimes": list(pose_times),
        "poseTransforms": poses.astype(np.float32).tolist() if demo else [],
        "rgbTimes": list(pose_times),
        "ultrawideRGBTimes": list(uw_times),
        "depthTimes": [],
        "depthFrameIsRepeat": [],
        "recordingStartTime": recording_start_time,
        "side": side,
        "type": kind,
        "gripperCalibrationRunName": calibration_run if demo else "",
        "sessionName": session,
        "gripperID": gripper_id,
        "hasRGB": demo,
        "hasUltrawideRGB": True,
        "hasDepth": False,
        "frameCount": int(frame_count),
        "taskStartTimestamps": tasks[1],
        "taskEndTimestamps": tasks[2],
        "taskNames": tasks[0],
        "narrationStartTimestamps": [],
        "narrationEndTimestamps": [],
        "narrationTexts": [],
        "labelType": "Predefined" if tasks[0] else "None",
        "hasAudio": False,
        "isVoiceHost": False,
        "sidesPresent": list(sides_present),
        "formatVersion": "v1",
        "deviceName": device_name,
        "deviceIdentifier": device_identifier,
        "mainCameraIntrinsics": _f32_list(K_main),
        "ultrawideCameraIntrinsics": _f32_list(K_uw),
        "isErrorCorrection": False,
    }


def format_float32(x: float) -> str:
    """Swift Float shortest repr: str(np.float32(x)) without a trailing '.0' (R1 §2.1)."""
    if not math.isfinite(x):
        raise ValueError(f"non-finite float {x!r} cannot be written to the episode JSON")
    s = str(np.float32(x))
    return s[:-2] if s.endswith(".0") else s


def _escape(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) > 0x7E:
            if ord(ch) > 0xFFFF:
                v = ord(ch) - 0x10000
                out.append(f"\\u{0xD800 + (v >> 10):04x}\\u{0xDC00 + (v & 0x3FF):04x}")
            else:
                out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def dumps_swift(obj) -> str:
    """Compact JSON (no spaces, no trailing newline), floats as float32 shortest repr, ASCII only."""
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if obj is None:
        return "null"
    if isinstance(obj, (int, np.integer)):
        return str(int(obj))
    if isinstance(obj, (float, np.floating)):
        return format_float32(float(obj))
    if isinstance(obj, str):
        return _escape(obj)
    if isinstance(obj, dict):
        return "{" + ",".join(_escape(str(k)) + ":" + dumps_swift(v) for k, v in obj.items()) + "}"
    if isinstance(obj, (list, tuple, np.ndarray)):
        return "[" + ",".join(dumps_swift(v) for v in obj) + "]"
    raise TypeError(f"cannot serialise {type(obj)}")
