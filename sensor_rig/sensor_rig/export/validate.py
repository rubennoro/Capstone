"""Validator for iPhone-format episodes (R1 §0 invariants, P1 §5.1-T1). Accepts real iPhone exports (test T1b).

``validate_iphone_episode`` returns a list of error strings (empty = OK). ``validate_export_root`` also checks that
every demonstration's ``gripperCalibrationRunName`` resolves to a calibration episode of the same side.
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import numpy as np

from sensor_rig.export.iphone_json import KEY_TYPES, KEYS
from sensor_rig.timefmt import ISO_RE, parse_episode_name, parse_iso

DEPTH_FRAME_BYTES = 256 * 192 * 2
PTS_TOL_S = 0.002  # ms time strings vs 1/600 s ticks
MAX_SPEED = 7.0  # m/s, app abort rule (VC:306)


def _type_ok(v, t: type) -> bool:
    if t is int:
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, t)


def _strictly_increasing(times: list[str]) -> bool:
    return all(a < b for a, b in zip(times, times[1:]))


def _check_K(K, name: str, size: tuple[int, int] | None, errors: list[str]) -> None:
    try:
        K = np.asarray(K, dtype=np.float64)
    except (TypeError, ValueError):
        errors.append(f"{name}: not a numeric 3x3 matrix")
        return
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        errors.append(f"{name}: not a finite 3x3 matrix")
        return
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or K[1, 0] != 0 or K[0, 1] != 0:
        errors.append(f"{name}: not of the form [[fx,0,cx],[0,fy,cy],[0,0,1]]")
    if size is not None:
        w, h = size
        if abs(K[0, 2] - w / 2) > 0.05 * w or abs(K[1, 2] - h / 2) > 0.05 * h:
            errors.append(f"{name}: principal point ({K[0, 2]:.1f},{K[1, 2]:.1f}) far from the centre of the "
                          f"{w}x{h} video (intrinsics/resolution mismatch)")


def _check_pts(pts_s: np.ndarray, times: list[str], label: str, errors: list[str]) -> None:
    if len(pts_s) != len(times):
        errors.append(f"{label}: {len(pts_s)} packets vs {len(times)} time strings")
        return
    if len(pts_s) == 0:
        return
    if abs(pts_s[0]) > 1e-9:
        errors.append(f"{label}: first PTS is {pts_s[0]:.4f} s, expected 0")
    t = np.array([parse_iso(s) for s in times], dtype=np.int64)
    rel = (t - t[0]) * 1e-9
    err = np.abs(pts_s - rel)
    if err.max() > PTS_TOL_S:
        k = int(err.argmax())
        errors.append(f"{label}: PTS {pts_s[k]:.4f} s vs time string offset {rel[k]:.4f} s at frame {k} "
                      f"(max error {err.max() * 1e3:.1f} ms > {PTS_TOL_S * 1e3:.0f} ms)")


def _decode_count(path: Path) -> int:
    import av

    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(c.streams.video[0]))


def _check_poses(P, times: list[str], errors: list[str]) -> None:
    try:
        T = np.asarray(P, dtype=np.float64)
    except (TypeError, ValueError):
        errors.append("poseTransforms: not a numeric array")
        return
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        errors.append(f"poseTransforms: shape {T.shape}, expected (N,4,4)")
        return
    if not np.all(np.isfinite(T)):
        errors.append("poseTransforms: NaN/inf values")
        return
    if len(T) == 0:
        return
    if np.abs(T[:, 3, :] - [0, 0, 0, 1]).max() > 1e-5:
        errors.append("poseTransforms: last row is not (0,0,0,1)")
    R = T[:, :3, :3]
    det = np.linalg.det(R)
    if np.abs(det - 1).max() > 1e-4:
        errors.append(f"poseTransforms: det(R) off by {np.abs(det - 1).max():.2e}")
    orth = np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3)).max()
    if orth > 1e-4:
        errors.append(f"poseTransforms: rotations not orthonormal ({orth:.2e})")
    # rotation angle from the trace; identity rotations give NaN downstream (iphumi pose_6d_to_4x4)
    cos = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    angle = np.arccos(cos)
    if np.any(angle < 1e-6):
        errors.append(f"poseTransforms: {int((angle < 1e-6).sum())} identity rotation(s) (NaN hazard)")
    if len(times) == len(T) and len(T) > 1:
        t = np.array([parse_iso(s) for s in times], dtype=np.int64) * 1e-9
        dt = np.diff(t)
        if np.all(dt > 0):
            speed = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) / dt
            if speed.max() > MAX_SPEED:
                errors.append(f"poseTransforms: speed {speed.max():.1f} m/s > {MAX_SPEED} m/s")


def validate_iphone_episode(folder, gripper_calibration_yaml=None, decode: bool = False,
                            name: str | None = None) -> list[str]:
    """Check one episode folder ``<NAME>/`` against every R1 invariant. ``name`` overrides the folder name."""
    from sensor_rig.export.video import probe_video

    folder = Path(folder)
    name = name or folder.name
    errors: list[str] = []

    try:
        parts = parse_episode_name(name)
    except ValueError as e:
        return [f"name: {e}"]
    if "_" in parts["session"]:
        errors.append("name: session contains '_'")

    json_path = folder / f"{name}.json"
    if not json_path.is_file():
        return errors + [f"files: {json_path.name} missing"]
    try:
        raw_text = json_path.read_text()
        j = json.loads(raw_text)
    except (OSError, ValueError) as e:
        return errors + [f"json: cannot parse ({e})"]
    if not isinstance(j, dict):
        return errors + ["json: top level is not an object"]

    # ---- keys and types
    keys = set(j)
    if keys != set(KEYS):
        missing, extra = sorted(set(KEYS) - keys), sorted(keys - set(KEYS))
        errors.append(f"json: key set differs (missing {missing}, extra {extra})")
    for k, t in KEY_TYPES.items():
        if k in j and not _type_ok(j[k], t):
            errors.append(f"json: {k} has type {type(j[k]).__name__}, expected {t.__name__}")
    if errors and any(e.startswith("json:") for e in errors):
        return errors
    for k in ("poseTimes", "rgbTimes", "ultrawideRGBTimes", "depthTimes", "taskStartTimestamps",
              "taskEndTimestamps", "narrationStartTimestamps", "narrationEndTimestamps", "narrationTexts",
              "taskNames", "sidesPresent"):
        if not all(isinstance(s, str) for s in j[k]):
            errors.append(f"json: {k} must hold strings only")
    if not all(isinstance(b, bool) for b in j["depthFrameIsRepeat"]):
        errors.append("json: depthFrameIsRepeat must hold bools only")
    if errors:
        return errors

    # ---- identity fields vs the name
    kind = j["type"]
    demo = kind == "demonstration"
    if kind != parts["kind"]:
        errors.append(f"json: type {kind!r} does not match the name ({parts['kind_name']})")
    if j["side"] != parts["side"]:
        errors.append(f"json: side {j['side']!r} does not match the name")
    if j["sessionName"] != parts["session"]:
        errors.append(f"json: sessionName {j['sessionName']!r} does not match the name")
    if j["side"] not in j["sidesPresent"]:
        errors.append("json: sidesPresent does not contain side")
    if j["labelType"] not in ("None", "Predefined", "Narration"):
        errors.append(f"json: labelType {j['labelType']!r}")
    if not (len(j["taskNames"]) == len(j["taskStartTimestamps"]) == len(j["taskEndTimestamps"])):
        errors.append("json: task lists differ in length")
    if not ISO_RE.match(j["recordingStartTime"]):
        errors.append("json: recordingStartTime is not an ISO-ms string")

    # ---- expected file set
    has_depth = j["hasDepth"]
    expected = {f"{name}.json", f"{name}_ultrawidergb.mp4"}
    if demo:
        expected.add(f"{name}_rgb.mp4")
    if has_depth:
        expected |= {f"{name}_depth.raw", f"{name}_depthpreview.mp4"}
    present = {p.name for p in folder.iterdir() if p.name.startswith(name)}
    if present - expected:
        errors.append(f"files: unexpected {sorted(present - expected)} (group.py copies every '<NAME>*')")
    if expected - present:
        errors.append(f"files: missing {sorted(expected - present)}")
        return errors

    n = j["frameCount"]
    uw = j["ultrawideRGBTimes"]
    pose_times = j["poseTimes"]

    # ---- per-kind counts
    if demo:
        if not j["hasRGB"]:
            errors.append("json: hasRGB must be true for demonstrations")
        if not (len(pose_times) == len(j["poseTransforms"]) == len(j["rgbTimes"]) == n):
            errors.append(f"counts: poseTimes {len(pose_times)}, poseTransforms {len(j['poseTransforms'])}, "
                          f"rgbTimes {len(j['rgbTimes'])}, frameCount {n} differ (R1 inv. 1)")
        if pose_times != j["rgbTimes"]:
            errors.append("counts: poseTimes != rgbTimes")
        if not j["gripperCalibrationRunName"] and j["side"] != "head":
            errors.append("json: gripperCalibrationRunName is empty")
    else:
        for k in ("poseTimes", "poseTransforms", "rgbTimes", "depthTimes", "depthFrameIsRepeat"):
            if j[k]:
                errors.append(f"json: {k} must be [] for a gripper calibration")
        if j["hasRGB"] or j["hasDepth"]:
            errors.append("json: hasRGB/hasDepth must be false for a gripper calibration")
        if j["gripperCalibrationRunName"]:
            errors.append("json: gripperCalibrationRunName must be '' for a gripper calibration")
    if not j["hasUltrawideRGB"]:
        errors.append("json: hasUltrawideRGB must be true")
    if len(uw) != n:
        errors.append(f"counts: len(ultrawideRGBTimes) {len(uw)} != frameCount {n} (R1 inv. 2)")
    uw_idx = [i for i, s in enumerate(uw) if s != ""]
    uw_strings = [uw[i] for i in uw_idx]
    if not uw_idx:
        errors.append("ultrawide: no non-empty ultrawideRGBTimes slot (detect.py would crash)")

    # ---- time strings
    for label, lst in (("poseTimes", pose_times), ("ultrawideRGBTimes", uw_strings),
                       ("taskStartTimestamps", j["taskStartTimestamps"]),
                       ("taskEndTimestamps", j["taskEndTimestamps"])):
        bad = [s for s in lst if not ISO_RE.match(s)]
        if bad:
            errors.append(f"times: {label} has non ISO-ms strings, e.g. {bad[0]!r}")
    if not _strictly_increasing(pose_times):
        errors.append("times: poseTimes not strictly increasing")
    if not _strictly_increasing(uw_strings):
        errors.append("times: non-empty ultrawideRGBTimes not strictly increasing")
    if demo and len(pose_times) == len(uw):
        if any(s not in ("", pose_times[i]) for i, s in enumerate(uw)):
            errors.append("times: ultrawideRGBTimes[i] must be '' or poseTimes[i]")

    # ---- depth (R1 inv. 3)
    if has_depth:
        dt = j["depthTimes"]
        if not (len(dt) == len(j["depthFrameIsRepeat"]) == n):
            errors.append("depth: depthTimes/depthFrameIsRepeat length != frameCount")
        size = (folder / f"{name}_depth.raw").stat().st_size
        if size != n * DEPTH_FRAME_BYTES:
            errors.append(f"depth: {name}_depth.raw has {size} bytes, expected {n} x {DEPTH_FRAME_BYTES}")
    elif j["depthTimes"] or j["depthFrameIsRepeat"]:
        errors.append("depth: hasDepth false but depthTimes/depthFrameIsRepeat not empty")

    # ---- intrinsics
    _check_K(j["mainCameraIntrinsics"], "mainCameraIntrinsics", None, errors)

    # ---- videos
    uw_path = folder / f"{name}_ultrawidergb.mp4"
    try:
        p = probe_video(uw_path)
    except Exception as e:  # noqa: BLE001 (any demux failure is a validation error)
        errors.append(f"ultrawide video: cannot demux ({e})")
        p = None
    if p is not None:
        if p["frames_header"] <= 0:
            errors.append("ultrawide video: container reports no frame count")
        if p["frames_header"] != len(uw_idx):
            errors.append(f"counts: ultrawide video has {p['frames_header']} frames, "
                          f"{len(uw_idx)} non-empty ultrawideRGBTimes (R1 inv. 2)")
        _check_K(j["ultrawideCameraIntrinsics"], "ultrawideCameraIntrinsics", (p["width"], p["height"]), errors)
        pts_s = np.array([float(x * p["time_base"]) for x in p["pts_list"]])
        _check_pts(pts_s, uw_strings, "ultrawide PTS (R1 inv. 4)", errors)
        if decode and _decode_count(uw_path) != p["frames_header"]:
            errors.append("ultrawide video: decoded frame count != header count")
    if demo:
        rgb_path = folder / f"{name}_rgb.mp4"
        try:
            q = probe_video(rgb_path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"main video: cannot demux ({e})")
            q = None
        if q is not None:
            if q["frames_header"] != n:
                errors.append(f"counts: main video has {q['frames_header']} frames, frameCount {n} (R1 inv. 1)")
            _check_K(j["mainCameraIntrinsics"], "mainCameraIntrinsics (vs video)", (q["width"], q["height"]),
                     errors)
            pts_s = np.array([float(x * q["time_base"]) for x in q["pts_list"]])
            _check_pts(pts_s, pose_times, "main PTS", errors)
            if decode and _decode_count(rgb_path) != q["frames_header"]:
                errors.append("main video: decoded frame count != header count")
        _check_poses(j["poseTransforms"], pose_times, errors)

    # ---- gripper calibration lookup (R1 inv. 6)
    if gripper_calibration_yaml is not None:
        import yaml

        with open(gripper_calibration_yaml) as f:
            cal = yaml.safe_load(f) or {}
        if j["deviceIdentifier"] not in (cal.get(j["gripperID"]) or {}):
            errors.append(f"calibration: ({j['gripperID']!r}, {j['deviceIdentifier']!r}) not in "
                          f"{Path(gripper_calibration_yaml).name}")
    return errors


def find_episode_jsons(root) -> list[Path]:
    """Episode JSONs exactly as group.py finds them: ``glob('<root>/**/*.json', recursive=True)`` (skips hidden)."""
    root = Path(root)
    if (root / f"{root.name}.json").is_file():
        return [root / f"{root.name}.json"]
    return sorted(Path(p) for p in glob(str(root / "**" / "*.json"), recursive=True))


def validate_export_root(root, gripper_calibration_yaml=None, decode: bool = False) -> dict[str, list[str]]:
    """Validate every episode under ``root``; returns {NAME: errors} (only names with errors are non-empty)."""
    out: dict[str, list[str]] = {}
    infos: dict[str, dict] = {}
    for jp in find_episode_jsons(root):
        name = jp.stem
        if jp.parent.name != name:
            out[name] = [f"layout: {jp} is not inside a folder named after the episode"]
            continue
        out[name] = validate_iphone_episode(jp.parent, gripper_calibration_yaml, decode)
        try:
            j = json.loads(jp.read_text())
            infos[name] = {"type": j.get("type"), "side": j.get("side"),
                           "run": j.get("gripperCalibrationRunName", ""), "gripperID": j.get("gripperID")}
        except ValueError:
            pass
    for name, info in infos.items():
        if info["type"] != "demonstration" or not info["run"]:
            continue
        cal = infos.get(info["run"])
        if cal is None:
            out[name].append(f"calibration: {info['run']} is not under {root}")
        elif cal["type"] != "gripper_calibration" or cal["side"] != info["side"]:
            out[name].append(f"calibration: {info['run']} is not a {info['side']} gripper calibration")
        elif cal["gripperID"] != info["gripperID"]:
            out[name].append(f"calibration: {info['run']} has gripperID {cal['gripperID']!r}")
    return out
