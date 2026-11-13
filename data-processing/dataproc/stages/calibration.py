"""The task-level **calibration** step: the newest ``calibration/<name>/raw`` -> the gripper range.

Reads the calibration episode (contract check, rig consistency, every ultrawide frame remapped to the virtual
pinhole, ArUco tags), fits iPhUMI's gripper range, compares the measured opening and closed width with the rig yaml
``gripper`` section (warnings only: like iPhUMI, only the fit itself can fail), and writes ``calibration/<name>/processed/{gripper_range.json, status.json}``, plus a copy
``tasks/<folder>/gripper_range.json``. That copy always mirrors the current result: it is removed when there is
no calibration or the newest one failed. Skipped (the saved range is reused) when its ``inputs_hash`` (manifest,
``aruco``, ``virtual_ultrawide``, ``cameras.ultrawide``, ``gripper``, task rig) is unchanged since a done or failed
run, unless ``--force``.

The fit itself is :func:`fit_calibration` (pure: reads the episode, writes nothing) and the files are written by
:func:`write_calibration_result`, so ``dataproc calibrate`` fits before any task folder exists and saves the result
in exactly the form :class:`CalibrationStep` later reuses without fitting again.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from dataproc import __version__
from dataproc.files import ERROR_FILE, hash_of, load_status, save_status, sha256_hex
from dataproc.fsutil import write_atomic
from dataproc.stages.base import Calibration, ProcessContext, StageFailed, rig_sections

__all__ = ["GRIPPER_RANGE_FILE", "CALIBRATION_VERSION", "MIN_CYCLES", "CalibrationFit", "newest_calibration",
           "calibration_inputs_hash", "fit_calibration", "write_calibration_result", "CalibrationStep"]

GRIPPER_RANGE_FILE = "gripper_range.json"
CALIBRATION_VERSION = 3          # 2: opening check against rig.gripper; 3: that check (and closed width) only warns
_KEY = "gripper_range"           # the record name in calibration/<name>/processed/status.json
#: Fewer open/close cycles than this in a calibration episode is a warning (the fit still runs).
MIN_CYCLES = 5


@dataclass
class CalibrationFit:
    """Result of :func:`fit_calibration`. ``status``: ``done`` (``gripper_range`` set) or ``failed`` (``reason``
    and the ``reasons`` codes: validator ``invalid_episode``, ``rig_mismatch`` / ``rig_resolution_mismatch`` /
    ``rig_fps_mismatch``, ``gripper_range_fit`` or ``exception``). ``diagnostics`` is
    :func:`~dataproc.stages.width.calibration_diagnostics` (empty when the tags were never detected because an
    earlier check failed); ``details`` holds per-reason data (``opening``, ``rig``, ``contract_errors``);
    ``warnings`` are non-fatal notes; ``error_text`` is the traceback of an ``exception``."""

    status: str = "failed"
    gripper_range: object = None
    reason: str | None = None
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_text: str | None = None
    t_start: float = 0.0
    t_end: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "done"

    @property
    def opening_m(self) -> float | None:
        """``max_width - min_width`` of the fit, else None."""
        o = self.details.get("opening")
        return None if o is None else o["measured_m"]


def newest_calibration(task_dir: Path, ledger_rows: list[dict]) -> Path | None:
    """The ``calibration/<name>`` dir with a ``raw/``, newest by ledger ``intake_unix_ns`` (unlisted ones count as
    oldest; ties by name). Dot-entries are ignored."""
    cdir = Path(task_dir) / "calibration"
    if not cdir.is_dir():
        return None
    when = {}
    for r in ledger_rows:
        slot = r.get("slot", "")
        if r.get("kind") == "calibration" and slot.startswith("calibration/"):
            try:
                when[slot.split("/", 1)[1]] = int(r.get("intake_unix_ns") or -1)
            except ValueError:
                when[slot.split("/", 1)[1]] = -1
    cands = [p for p in cdir.iterdir() if not p.name.startswith(".") and (p / "raw").is_dir()]
    return max(cands, key=lambda p: (when.get(p.name, -1), p.name)) if cands else None


def calibration_inputs_hash(raw_dir: Path, rig, task_rig: str) -> str:
    """The ``inputs_hash`` of a calibration result: stage version, dataproc version, the ``MANIFEST.sha256``
    bytes of ``raw_dir`` and the rig sections the fit reads. Path independent (the dir may be renamed)."""
    from dataproc.raw_episode import MANIFEST

    try:
        manifest = (Path(raw_dir) / MANIFEST).read_bytes()
    except OSError:
        manifest = b""
    s = rig_sections(rig)
    return hash_of({"stage": _KEY, "stage_version": CALIBRATION_VERSION, "dataproc": __version__,
                    "manifest_sha256": sha256_hex(manifest),
                    "inputs": {"aruco": s["aruco"], "virtual_ultrawide": s["virtual_ultrawide"],
                               "cameras.ultrawide": s["cameras.ultrawide"], "gripper": s["gripper"],
                               "task_rig": task_rig}})


def _rig_message(details: dict) -> str:
    parts = []
    for code in ("rig_resolution_mismatch", "rig_fps_mismatch"):
        for d in details.get(code, []):
            what = "resolution" if code == "rig_resolution_mismatch" else "fps"
            parts.append(f"{d['role']} camera {what} {d['meta']} != rig yaml {d['rig']}")
    if "rig_mismatch" in details:
        d = details["rig_mismatch"]
        parts.append(f"calibration rig {d['episode_rig']!r} != task rig {d['task_rig']!r}")
    return "; ".join(parts)


def _check_opening(fit: CalibrationFit, gr, rig) -> None:
    """Record the measured opening and closed width, and add a warning for each that is outside the rig's
    ``gripper`` tolerance. Never fails: iPhUMI's calibration has no expected opening, so dataproc only points out a
    range that looks wrong (a tag hidden when the gripper closes, a gripper not fully opened or closed)."""
    opening, closed = float(gr.max_width - gr.min_width), float(gr.min_width)
    g = rig.gripper
    fit.details["opening"] = {"measured_m": opening, "min_width": closed,
                              "max_width": float(gr.max_width),
                              "expected_m": None if g is None else g.max_opening_m,
                              "tolerance_m": None if g is None else g.opening_tolerance_m,
                              "expected_closed_m": None if g is None else g.closed_width_m}
    if g is None:
        fit.warnings.append(f"rig {rig.rig_id} has no gripper section (max_opening_m, opening_tolerance_m): "
                            "the opening is not compared with anything")
        return
    tol = g.opening_tolerance_m
    if g.closed_width_m is not None and abs(closed - g.closed_width_m) > tol:
        fit.warnings.append(f"closed width {closed:.3f} m, expected {g.closed_width_m:.3f} m ± {tol:.3f} m: "
                            + ("a finger tag may be hidden when the gripper closes, or it was never fully closed; "
                               if closed > g.closed_width_m else "")
                            + "every demonstration's width is measured from this closed width")
    if abs(opening - g.max_opening_m) > tol:
        fit.warnings.append(f"measured opening {opening:.3f} m, expected {g.max_opening_m:.3f} m ± {tol:.3f} m: "
                            "check that the gripper was fully opened and fully closed, and that both tags stayed "
                            "in view")


def fit_calibration(raw_dir: Path, rig, prep, task_rig: str) -> CalibrationFit:
    """Fit and check one calibration episode (a contract episode dir: ``calibration/<name>/raw`` or a converted
    episode in ``incoming/``). Reads only; never raises for episode problems (they are the result's ``reasons``).

    Steps: contract check with videos decoded and manifest hashed; rig consistency (``meta.rig_id`` == ``task_rig``,
    camera resolution / fps == the rig yaml); ArUco tags on every virtual ultrawide frame (``prep.to_virtual``);
    :func:`~dataproc.stages.width.calibration_diagnostics`; :func:`~dataproc.stages.width.fit_gripper_range`. Only
    those steps can fail it, as in iPhUMI. An opening or closed width outside ``rig.gripper`` (see
    :func:`_check_opening`) and fewer than :data:`MIN_CYCLES` open/close cycles are warnings."""
    from dataproc.raw_episode import RawEpisode
    from dataproc.stages.check import rig_consistency
    from dataproc.stages.width import (GripperRangeError, calibration_diagnostics, detect_episode_tags,
                                       fit_gripper_range)

    fit = CalibrationFit(t_start=time.time())
    try:
        raw = RawEpisode.open(Path(raw_dir))
        report = raw.validate(decode_videos=True, check_manifest=True)
        if not report.ok:
            codes = list(dict.fromkeys(p.code for p in report.errors))
            fit.details["contract_errors"] = [p.to_dict() for p in report.errors]
            raise StageFailed("failed", ["invalid_episode"], "contract errors: " + ", ".join(codes))
        reasons, details = rig_consistency(raw.meta, rig, task_rig)
        if reasons:
            fit.details["rig"] = details
            raise StageFailed("failed", reasons, _rig_message(details))
        _, dets = detect_episode_tags(raw, rig, prep.to_virtual)
        fit.diagnostics = calibration_diagnostics(dets, rig)
        try:
            gr = fit_gripper_range(dets, rig)
        except GripperRangeError as e:
            raise StageFailed("failed", ["gripper_range_fit"], str(e)) from None
        _check_opening(fit, gr, rig)
        if fit.diagnostics["n_cycles"] < MIN_CYCLES:
            fit.warnings.append(f"only {fit.diagnostics['n_cycles']} open/close cycle(s) detected (record at least "
                                f"{MIN_CYCLES} for a reliable range)")
        fit.status, fit.gripper_range = "done", gr
    except StageFailed as e:
        fit.reasons, fit.reason = list(e.reasons), str(e)
    except Exception as e:  # noqa: BLE001 - recorded, never fatal
        fit.reasons, fit.reason = ["exception"], f"{type(e).__name__}: {e}"
        fit.error_text = traceback.format_exc()
    fit.t_end = time.time()
    return fit


def write_calibration_result(cal_dir: Path, task_dir: Path | None, fit: CalibrationFit, rig=None,
                             task_rig: str | None = None, inputs_hash: str | None = None) -> Calibration:
    """Save ``fit`` for ``cal_dir`` (``calibration/<name>`` with its ``raw/``): ``processed/gripper_range.json``
    (when done), ``processed/status.json`` (the ``gripper_range`` record with ``inputs_hash``), ``error.txt`` for an
    exception; and, with ``task_dir``, the task copy ``task_dir/gripper_range.json`` (removed when failed).

    ``inputs_hash`` defaults to :func:`calibration_inputs_hash` of ``cal_dir/raw`` with ``rig`` and ``task_rig``,
    the hash :class:`CalibrationStep` compares, so a later ``process`` reuses the result without fitting again."""
    cal_dir = Path(cal_dir)
    if inputs_hash is None:
        if rig is None or task_rig is None:
            raise ValueError("write_calibration_result needs rig and task_rig (or inputs_hash)")
        inputs_hash = calibration_inputs_hash(cal_dir / "raw", rig, task_rig)
    processed = cal_dir / "processed"
    processed.mkdir(exist_ok=True)
    gr_path = processed / GRIPPER_RANGE_FILE
    rec = {"status": fit.status, "reason": fit.reason, "reasons": list(fit.reasons), "inputs_hash": inputs_hash,
           "outputs": {}, "t_start": fit.t_start, "version": CALIBRATION_VERSION, "dataproc_version": __version__,
           "warnings": list(fit.warnings), "diagnostics": dict(fit.diagnostics), "details": dict(fit.details)}
    if fit.ok:
        gr = fit.gripper_range
        gr.save(gr_path)
        rec["outputs"] = {"gripper_range": GRIPPER_RANGE_FILE, "min_width": gr.min_width,
                          "max_width": gr.max_width, **gr.fit_stats}
        (processed / ERROR_FILE).unlink(missing_ok=True)       # from an earlier failed fit
        cal = Calibration("done", cal_dir.name, gripper_range=gr)
    else:
        gr_path.unlink(missing_ok=True)
        if fit.error_text:
            write_atomic(processed / ERROR_FILE, fit.error_text)
            rec["error_file"] = ERROR_FILE
        cal = Calibration("calibration_failed", cal_dir.name, reason=fit.reason)
    rec["t_end"] = fit.t_end
    save_status(processed, {_KEY: rec})
    if task_dir is not None:
        task_copy = Path(task_dir) / GRIPPER_RANGE_FILE
        if fit.ok:
            write_atomic(task_copy, gr_path.read_text(encoding="utf-8"))
        else:
            task_copy.unlink(missing_ok=True)
    cal.ran = True
    return cal


class CalibrationStep:
    """See the module docstring. ``CalibrationStep(ctx).run()`` returns a :class:`Calibration`."""

    def __init__(self, ctx: ProcessContext):
        self.ctx = ctx
        self.task_copy = ctx.task_dir / GRIPPER_RANGE_FILE

    def run(self) -> Calibration:
        cal_dir = newest_calibration(self.ctx.task_dir, self.ctx.ledger)
        if cal_dir is None:
            self.task_copy.unlink(missing_ok=True)
            return Calibration("no_calibration", reason="no calibration/<name>/raw in the task folder",
                               token="no_calibration")
        processed = cal_dir / "processed"
        processed.mkdir(exist_ok=True)
        h = self._inputs_hash(cal_dir / "raw")
        cal = self._reuse(cal_dir, h) or self._fit(cal_dir, h)
        return self._publish(cal, processed / GRIPPER_RANGE_FILE, h)

    # ---- steps
    def _inputs_hash(self, raw_dir: Path) -> str:
        return calibration_inputs_hash(raw_dir, self.ctx.rig, self.ctx.task.rig)

    def _reuse(self, cal_dir: Path, h: str) -> Calibration | None:
        """The saved result if its hash is unchanged (and not --force); None means fit again."""
        from dataproc.stages.width import GripperRange

        prev = load_status(cal_dir / "processed").get(_KEY)
        if self.ctx.force or not prev or prev.get("inputs_hash") != h:
            return None
        gr_path = cal_dir / "processed" / GRIPPER_RANGE_FILE
        if prev.get("status") == "done" and gr_path.is_file():
            try:
                return Calibration("done", cal_dir.name, gripper_range=GripperRange.load(gr_path))
            except Exception as e:  # noqa: BLE001 - any unreadable / malformed saved range: fit again
                self.ctx.log(f"calibration/{cal_dir.name}: cannot read {gr_path} ({e}); fitting again")
                return None
        if prev.get("status") == "failed":
            return Calibration("calibration_failed", cal_dir.name, reason=prev.get("reason"))
        return None

    def _fit(self, cal_dir: Path, h: str) -> Calibration:
        fit = fit_calibration(cal_dir / "raw", self.ctx.rig, self.ctx.prep, self.ctx.task.rig)
        for w in fit.warnings:
            self.ctx.log(f"calibration/{cal_dir.name}: warning: {w}")
        return write_calibration_result(cal_dir, None, fit, inputs_hash=h)

    def _publish(self, cal: Calibration, gr_path: Path, h: str) -> Calibration:
        """Mirror the result into ``tasks/<folder>/gripper_range.json`` and set the token the width stage hashes."""
        if cal.state == "done":
            text = gr_path.read_text(encoding="utf-8")
            if not self.task_copy.is_file() or self.task_copy.read_text(encoding="utf-8") != text:
                write_atomic(self.task_copy, text)
            cal.token = "gripper_range:" + sha256_hex(text.encode("utf-8"))
        else:
            self.task_copy.unlink(missing_ok=True)
            cal.token = f"calibration_failed:{h}"
        return cal
