"""The task-level **calibration** step: the newest ``calibration/<name>/raw`` -> the gripper range.

Reads the calibration episode (contract check, every ultrawide frame remapped to the virtual pinhole, ArUco tags),
fits iPhUMI's gripper range, and writes ``calibration/<name>/processed/{gripper_range.json, status.json}``, plus a
copy ``tasks/<folder>/gripper_range.json``. That copy always mirrors the current result: it is removed when there is
no calibration or the newest one failed. Skipped (the saved range is reused) when its ``inputs_hash`` (manifest,
``aruco``, ``virtual_ultrawide``, ``cameras.ultrawide``, task rig) is unchanged since a done or failed run, unless
``--force``.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from dataproc import __version__
from dataproc.files import ERROR_FILE, hash_of, load_status, save_status, sha256_hex
from dataproc.fsutil import write_atomic
from dataproc.stages.base import Calibration, ProcessContext, StageFailed

__all__ = ["GRIPPER_RANGE_FILE", "CALIBRATION_VERSION", "newest_calibration", "CalibrationStep"]

GRIPPER_RANGE_FILE = "gripper_range.json"
CALIBRATION_VERSION = 1
_KEY = "gripper_range"           # the record name in calibration/<name>/processed/status.json


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
        from dataproc.raw_episode import MANIFEST

        try:
            manifest = (raw_dir / MANIFEST).read_bytes()
        except OSError:
            manifest = b""
        s = self.ctx.sections
        return hash_of({"stage": _KEY, "stage_version": CALIBRATION_VERSION, "dataproc": __version__,
                        "manifest_sha256": sha256_hex(manifest),
                        "inputs": {"aruco": s["aruco"], "virtual_ultrawide": s["virtual_ultrawide"],
                                   "cameras.ultrawide": s["cameras.ultrawide"], "task_rig": self.ctx.task.rig}})

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
        from dataproc.raw_episode import RawEpisode
        from dataproc.stages.width import GripperRangeError, detect_episode_tags, fit_gripper_range

        processed = cal_dir / "processed"
        gr_path = processed / GRIPPER_RANGE_FILE
        rec = {"status": "failed", "reason": None, "reasons": [], "inputs_hash": h, "outputs": {},
               "t_start": time.time(), "version": CALIBRATION_VERSION, "dataproc_version": __version__}
        try:
            raw = RawEpisode.open(cal_dir / "raw")
            report = raw.validate(decode_videos=True, check_manifest=True)
            if not report.ok:
                raise StageFailed("failed", ["invalid_episode"], "contract errors: " +
                                  ", ".join(dict.fromkeys(p.code for p in report.errors)))
            if raw.meta.rig_id != self.ctx.task.rig:
                raise StageFailed("failed", ["rig_mismatch"], f"calibration rig {raw.meta.rig_id!r} != task rig "
                                  f"{self.ctx.task.rig!r}")
            _, dets = detect_episode_tags(raw, self.ctx.rig, self.ctx.prep.to_virtual)
            try:
                gr = fit_gripper_range(dets, self.ctx.rig)
            except GripperRangeError as e:
                raise StageFailed("failed", ["gripper_range_fit"], str(e)) from None
            gr.save(gr_path)
            rec.update(status="done", outputs={"gripper_range": GRIPPER_RANGE_FILE, "min_width": gr.min_width,
                                               "max_width": gr.max_width, **gr.fit_stats})
            cal = Calibration("done", cal_dir.name, gripper_range=gr)
            (processed / ERROR_FILE).unlink(missing_ok=True)       # from an earlier failed fit
        except StageFailed as e:
            rec.update(reasons=e.reasons, reason=str(e))
            cal = Calibration("calibration_failed", cal_dir.name, reason=str(e))
        except Exception as e:  # noqa: BLE001 - recorded, never fatal
            write_atomic(processed / ERROR_FILE, traceback.format_exc())
            rec.update(reasons=["exception"], reason=f"{type(e).__name__}: {e}", error_file=ERROR_FILE)
            cal = Calibration("calibration_failed", cal_dir.name, reason=rec["reason"])
        rec["t_end"] = time.time()
        save_status(processed, {_KEY: rec})
        if cal.state != "done":
            gr_path.unlink(missing_ok=True)
        cal.ran = True
        return cal

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
