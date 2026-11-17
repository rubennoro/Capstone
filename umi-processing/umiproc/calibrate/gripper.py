"""``umiproc calibrate``: the gripper opening range of a session folder, from its gripper calibration recording
(docs/DESIGN.md "The session" step 2; DIRECTIONS.md section 4).

The copied dataproc method (``umiproc.dp.stages.calibration.fit_calibration``: contract check, rig consistency, the
finger tags ``aruco.left_id``/``right_id`` (DICT_4X4_50 ids 0/1) on every ultrawide frame remapped to the virtual
pinhole **with the effective rig** (the lens file's model), iPhUMI's range fit, the opening and closed-width
warnings), **plus UMI's rule** (``scripts/calibrate_gripper_range.py:22, 28-35, 41-50, 59-61``): each finger tag's
detection rate (frames it was seen in / all frames) is computed, the gripper's rate is the smaller of the two, and
the calibration fails below 0.8 (:func:`finger_tag_rule`). UMI's ``tag_det_threshold`` 0.8 replaces nothing in the
copy: the copied fit keeps its own 0.2 floor, and the 80 % rule is checked on the same detections afterwards.

Steps, under ``<root>/.run.lock``:

1. The folder (``tasks/<folder>/task.yaml``) and its map: complete and passed (a calibration only makes sense in a
   session whose map passed). A folder that already has a gripper calibration **and** episodes is refused (they were
   made with it; one calibration per session); without episodes a new calibration replaces the old one (the newest
   is the one used).
2. The effective sensor values of the folder's rig (:func:`umiproc.sensors.load_sensors`).
3. The recording: ``inspect_recording`` (read only), ``stage`` (converted into ``incoming/``).
4. The fit and the 80 % rule. **Failure**: the staged copy is removed and nothing is recorded (no
   ``pi_sources.csv`` row, nothing in the folder): it can be run again.
5. **Pass**: ``take(CALIBRATION, folder=...)`` (``calibration/<pi_name>/raw``, both ledgers), then the copied
   ``write_calibration_result`` (``calibration/<pi_name>/processed/{gripper_range.json, status.json}`` and the
   folder's ``gripper_range.json``, in the form ``run``'s calibration step reuses without fitting again) and
   ``calibration/<pi_name>/processed/umiproc_calibrate.json`` (the rule, the warnings, the lens).

Exit codes: 0 passed; 1 the fit or the 80 % rule failed (nothing recorded); 2 the command could not run on this input
(folder, map, recording missing / still syncing / already taken, rig files, another command running).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = ["EXIT_OK", "EXIT_FAILED", "EXIT_INPUT", "TAG_DET_THRESHOLD", "REPORT_NAME", "CalibrateInputError",
           "CalibrateOutcome", "finger_tag_rule", "calibrate_folder"]

EXIT_OK, EXIT_FAILED, EXIT_INPUT = 0, 1, 2
#: UMI scripts/calibrate_gripper_range.py:22 --tag_det_threshold 0.8
TAG_DET_THRESHOLD = 0.8
REPORT_NAME = "umiproc_calibrate.json"


class CalibrateInputError(Exception):
    """The command cannot run (exit 2). Nothing was created or recorded. ``code``: a short reason."""

    def __init__(self, message: str, code: str = "usage"):
        super().__init__(message)
        self.code = code


@dataclass
class CalibrateOutcome:
    """``passed``: ``gripper_range.json`` was written for ``folder``. Otherwise ``reasons`` (the copied fit's codes,
    ``finger_tags_below_80``, ``intake``) and ``message``; ``fit`` is the copied ``CalibrationFit`` once it ran,
    ``rule`` the 80 % rule."""

    passed: bool
    exit_code: int
    folder: str
    source_name: str
    pi_name: str | None = None
    slot: str | None = None
    fit: object = None
    rule: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    message: str = ""
    replaced: list[str] = field(default_factory=list)
    lens: dict = field(default_factory=dict)


def finger_tag_rule(diagnostics: dict, threshold: float = TAG_DET_THRESHOLD) -> dict:
    """UMI's rule on the copied ``calibration_diagnostics``: ``min(left_rate, right_rate) >= threshold``. Returns
    ``{ok, rate, left_rate, right_rate, both_rate, threshold, message}`` (``ok`` False without detections)."""
    d = diagnostics or {}
    n = d.get("n_frames") or 0
    left, right = float(d.get("left_rate") or 0.0), float(d.get("right_rate") or 0.0)
    rate = min(left, right) if n else 0.0
    ok = bool(n) and rate >= threshold
    msg = (f"tag {d.get('left_id', 0)} seen in {100 * left:.0f} % and tag {d.get('right_id', 1)} in {100 * right:.0f} %"
           f" of {n} frames (both at once: {100 * float(d.get('both_rate') or 0.0):.0f} %); UMI needs each tag in "
           f">= {100 * threshold:.0f} %")
    return {"ok": ok, "rate": rate, "left_rate": left, "right_rate": right,
            "both_rate": float(d.get("both_rate") or 0.0), "threshold": threshold, "n_frames": int(n),
            "message": msg}


def _folder(r, folder: str):
    from umiproc.dp.config import ConfigError, load_task_config
    from umiproc.dp.paths import validate_name

    try:
        validate_name(folder, "task folder")
    except ValueError as e:
        raise CalibrateInputError(str(e)) from None
    task_dir = r.tasks / folder
    if not (task_dir / "task.yaml").is_file():
        raise CalibrateInputError(f"task folder {task_dir} does not exist (or has no task.yaml): `umiproc map` makes "
                                  "the session's folder first (DIRECTIONS.md section 3)", "folder")
    try:
        return task_dir, load_task_config(task_dir / "task.yaml")
    except ConfigError as e:
        raise CalibrateInputError(str(e), "folder") from None


def _check_map(task_dir: Path) -> None:
    from umiproc.map.stored import read_map

    maps = read_map(task_dir)
    if not maps:
        raise CalibrateInputError(f"task folder {task_dir.name} has no map: calibrate the gripper in a session folder "
                                  "`umiproc map` made", "map")
    m = maps[-1]
    if not m["complete"]:
        raise CalibrateInputError(f"the map of {task_dir.name} (map/{m['pi_name']}) is INCOMPLETE (an interrupted "
                                  "`umiproc map`): record a new mapping clip (a new session folder) and calibrate "
                                  "there", "map")
    if m["passed"] is not True:
        raise CalibrateInputError(f"the map of {task_dir.name} did not pass its checks", "map")


def _existing(task_dir: Path) -> tuple[list[str], list[str]]:
    """``(calibration recordings with a result, episode_N slots)`` already in the folder."""
    from umiproc.dp.files import load_status
    from umiproc.dp.paths import episode_dirs
    from umiproc.recordings import CALIBRATION_DIR

    cal = task_dir / CALIBRATION_DIR
    done = []
    if cal.is_dir():
        done = sorted(p.name for p in cal.iterdir() if not p.name.startswith(".") and (p / "raw").exists()
                      and (load_status(p / "processed").get("gripper_range") or {}).get("status") == "done")
    eps = [d.name for d in episode_dirs(task_dir) if (d / "raw").exists()]
    return done, eps


def calibrate_folder(root, folder: str, cal_dir: str, *, source: Path = Path("/data/incoming"),
                     log: Callable = print) -> CalibrateOutcome:
    """``umiproc calibrate`` (module docstring). Raises :class:`CalibrateInputError` (exit 2); returns the outcome
    otherwise (exit 0 / 1)."""
    from umiproc.dp.imageprep import ImagePrep
    from umiproc.dp.paths import DataRoot, validate_episode_dir_name
    from umiproc.dp.stages.calibration import fit_calibration, write_calibration_result
    from umiproc.recordings import (CALIBRATION, RecordingError, discard, inspect_recording, session_lock, stage,
                                    take)
    from umiproc.sensors import RigFilesError, SensorError, load_sensors

    try:
        validate_episode_dir_name(cal_dir)
    except ValueError as e:
        raise CalibrateInputError(str(e)) from None
    r = root if isinstance(root, DataRoot) else DataRoot(Path(root))
    try:
        lock = session_lock(r)
        lock.__enter__()
    except RecordingError as e:
        raise CalibrateInputError(str(e), e.code) from None
    try:
        task_dir, task = _folder(r, folder)
        _check_map(task_dir)
        done, eps = _existing(task_dir)
        if done and eps:
            raise CalibrateInputError(
                f"task folder {folder} already has a gripper calibration (calibration/{done[-1]}) and "
                f"{len(eps)} episode(s) made with it: one calibration per session. Record a new session (mapping "
                "clip, gripper calibration, demos) to calibrate again", "already_calibrated")
        try:
            sensors = load_sensors(r.path, task.rig, log)
        except (RigFilesError, SensorError) as e:
            raise CalibrateInputError(str(e), "rig_files") from None
        log(f"calibrate: inspecting {Path(source) / cal_dir}")
        try:
            src = inspect_recording(r, Path(source), cal_dir)
            staged = stage(r, src, task.rig)
        except RecordingError as e:
            raise CalibrateInputError(str(e), e.code) from None
        out = CalibrateOutcome(False, EXIT_FAILED, folder, cal_dir, staged.pi_name, replaced=done,
                               lens=sensors.lens_record())
        try:
            rig = sensors.rig                                 # the effective rig: the lens file's model
            log(f"calibrate: finding the finger tags in {staged.pi_name} (lens {sensors.lens_provenance})")
            out.fit = fit = fit_calibration(staged.path, rig, ImagePrep(rig), task.rig)
            out.rule = rule = finger_tag_rule(fit.diagnostics)
            reasons = list(fit.reasons) if not fit.ok else []
            if fit.diagnostics and not rule["ok"]:
                reasons.append("finger_tags_below_80")
            if reasons:
                out.reasons = reasons
                out.message = "; ".join(([fit.reason] if fit.reason else []) +
                                        (["UMI's finger-tag rule: " + rule["message"]]
                                         if "finger_tags_below_80" in reasons else []))
                return out
            taken = take(r, staged, CALIBRATION, folder=folder, log=log)
            if taken.status != "ingested":
                out.reasons, out.exit_code = ["intake"], EXIT_INPUT
                out.message = (f"intake quarantined {staged.pi_name} ({', '.join(taken.reasons)}): "
                               f"{taken.quarantine_dir}")
                return out
            write_calibration_result(taken.slot_dir, task_dir, fit, rig, task.rig)
            from umiproc.dp.files import iso_utc, write_json

            write_json(taken.slot_dir / "processed" / REPORT_NAME, {
                "format": "umiproc.calibrate.v1", "folder": folder, "pi_name": taken.pi_name, "source": str(src.path),
                "finger_tag_rule": rule, "warnings": list(fit.warnings), "opening": fit.details.get("opening"),
                "lens": sensors.lens_record(), "replaces": done, "created_utc": iso_utc(time.time())})
            out.passed, out.exit_code, out.slot = True, EXIT_OK, taken.slot
            return out
        finally:
            discard(staged)                      # gone already when intake took it
    finally:
        lock.__exit__(None, None, None)
