"""One demonstration of ``umiproc run``: take it, localize it against the folder's map, apply UMI's rules, check
every piece, then finish it (widths, export) into ``episode_N`` folders, wait for the gripper calibration, or fail
it (docs/DESIGN.md "The session", "Checks -> run", "Episodes").

Where things are while it is processed, in ``tasks/<folder>/.staging/<pi_name>/``::

    raw/                         the recording (intake put it here)
    processed/status.json        the copied stage records (check, slam, grid, pose, width, validate, images,
                                 export), so the copied status / reason.json / packaging read it as before
    processed/check.json         the copied check stage
    processed/grid.npy           the copied 60 Hz grid (the button span)
    processed/localize/          the localization run: trajectory.csv (+ .t0_ns), umiproc_slam.log, times.txt
    processed/localize.json      umiproc's record: map, fingerprint, the run, UMI's rules and the pieces, every
                                 piece's pose checks, the demo's (``demo_checks``) and the orientation check,
                                 ``phase`` (``localized`` = every pose check done, waiting for or ready for the
                                 widths; ``finished`` = exported, being moved into ``episode_N``)
    processed/pieces/<k>/        while finishing: piece k's episode.zarr, status.json, piece.json

Steps (``status.json`` record in brackets):

1. ``check`` [check]: the copied contract + rig consistency check (effective rig).
2. ``grid`` [grid]: the copied 60 Hz grid over the button span.
3. **localize** [slam]: the copied ``write_euroc`` (native size, unmasked frames; removed afterwards), then
   ``umiproc_slam --load_map <map_atlas.osa> --mask_img --max_lost_frames 30`` in real time (UMI's
   ``03_batch_slam.py``); a run that crashed, timed out or left no readable trajectory fails (``slam_error``); a run
   that started a second map is rejected (``new_map``: its poses would not be in the map's coordinates); then UMI's
   lost-frame rules (:mod:`umiproc.run.rules`), which give the pieces or drop the demo.
4. **pose checks** [pose]: gravity, scale and orientation once on the whole demo (a failure drops the demo; an
   unobservable scale is only reported), then each piece's speed (:mod:`umiproc.run.pieces`). ``localize.json`` is
   written with ``phase: localized``.
5. Without ``gripper_range.json``: the recording stays in ``.staging/`` **waiting for the gripper calibration**. A
   later run (after ``umiproc calibrate``) resumes here: steps 1-4 are not redone.
6. **finish** [width, validate, images, export] every piece that passed its pose checks.
7. k pieces exported -> k consecutive ``episode_N``: the first takes the ``.staging/<pi_name>`` folder (its
   ``raw/``); every other one gets ``processed/`` with its own ``episode.zarr``, ``raw`` -> ``../episode_<first>/raw``
   (a relative symlink: the raw copy is stored once) and its own ``episodes.csv`` row. Every episode has
   ``processed/piece.json`` (which piece of which recording, its frames, what was trimmed). No piece exported ->
   ``failed/<pi_name>/`` with ``reason.json``. ``localize.json`` says ``finished`` before the moves start, so a
   crash during them is finished by the next run (:meth:`RecordingProcessor._promotion_interrupted` for a recording
   still in ``.staging/``, :func:`finish_interrupted_pieces` for pieces left in ``episode_N/processed/pieces/``)
   without a new localization or export.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from umiproc.dp.files import iso_utc, load_status, read_json, save_status, write_json
from umiproc.dp.paths import STAGING_DIR

__all__ = ["LOCALIZE_DIR", "LOCALIZE_FILE", "LOCALIZE_FORMAT", "PIECES_DIR", "PIECE_FILE", "EPISODE", "FAILED",
           "WAITING", "SKIPPED", "LOCALIZED", "FINISHED", "Session", "Outcome", "RecordingProcessor",
           "finish_interrupted_pieces"]

#: The renames and links of a promotion (module level, so the tests can interrupt them at any step).
_replace, _symlink = os.replace, os.symlink

LOCALIZE_DIR = "localize"
LOCALIZE_FILE = "localize.json"
LOCALIZE_FORMAT = "umiproc.localize.v1"
PIECES_DIR = "pieces"
PIECE_FILE = "piece.json"
PIECE_FORMAT = "umiproc.piece.v1"
EPISODE, FAILED, WAITING, SKIPPED = "episode", "failed", "waiting", "skipped"
LOCALIZED, FINISHED = "localized", "finished"
#: The record keys of the copied status.json, in order (``umiproc.dp.report.STAGE_NAMES``).
RECORDING_STAGES = ("check", "slam", "grid", "pose")
PIECE_STAGES = ("width", "validate", "images", "export")
STAGE_VERSION = 1


@dataclass
class Session:
    """What every recording of one ``umiproc run`` shares (:func:`umiproc.run.command.run_range` builds it)."""

    root: object                    # DataRoot
    folder: str
    task_dir: Path
    label: str
    rig_id: str
    sensors: object                 # umiproc.sensors.SensorValues
    inputs: object                  # umiproc.slam.inputs.SlamInputs (settings, mask, fingerprint)
    map_pi_name: str
    atlas: Path
    tx_slam_tag: np.ndarray
    ctx: object                     # the copied ProcessContext (effective rig, calibration loaded)
    calibrated: bool                # the folder has a usable gripper range
    work: Path                      # <root>/.work/run
    log: Callable = print
    runner: object = None
    binary: Path | None = None
    vocabulary: Path | None = None
    timeout_s: float | None = None

    @property
    def rig(self):
        return self.sensors.rig

    @property
    def fingerprint_hash(self) -> str:
        return self.inputs.fingerprint_hash


@dataclass
class Outcome:
    """What happened to one recording. ``state``: ``episode`` (``slots``: its ``episode_N``, one per kept piece),
    ``failed`` (``stage``, ``reasons``, ``message``; slot ``failed/<pi_name>``), ``waiting`` (for the gripper
    calibration; slot ``.staging/<pi_name>``) or ``skipped`` (``message``). ``pieces``: per piece found by UMI's
    rules, ``{index, frames, outcome, slot | stage + reasons}``."""

    name: str
    pi_name: str | None
    state: str
    slots: list[str] = field(default_factory=list)
    stage: str | None = None
    reasons: list[str] = field(default_factory=list)
    message: str = ""
    action: str = ""
    pieces: list[dict] = field(default_factory=list)
    quiet: bool = False
    seconds: float = 0.0

    def describe(self) -> str:
        if self.state == EPISODE:
            dropped = [p for p in self.pieces if p.get("outcome") != EPISODE]
            s = ", ".join(self.slots)
            if len(self.pieces) > 1 or dropped:
                s += f" ({len(self.slots)} of {len(self.pieces)} pieces"
                s += ("; " + "; ".join(f"piece {p['index']} dropped: {', '.join(p.get('reasons') or [])}"
                                       for p in dropped) if dropped else "") + ")"
            trims = [f"piece {p['index']} {_trim_words(p['trimmed'])}" for p in self.pieces
                     if p.get("outcome") == EPISODE and (p.get("trimmed") or {}).get("rows")]
            return s + ("; " + "; ".join(trims) if trims else "")
        if self.state == FAILED:
            return f"failed ({self.stage}: {'; '.join(self.reasons) or self.message})"
        if self.state == WAITING:
            n = len([p for p in self.pieces if p.get("outcome") == WAITING])
            dropped = [p for p in self.pieces if p.get("outcome") != WAITING]
            return (f"waiting for the gripper calibration ({n} piece(s) passed every pose check"
                    + "".join(f"; piece {p['index']} dropped: {', '.join(p.get('reasons') or [])}" for p in dropped)
                    + ")")
        return f"skipped ({self.message})"

    def to_dict(self) -> dict:
        return {"source_name": self.name, "pi_name": self.pi_name, "outcome": self.state, "slots": list(self.slots),
                "stage": self.stage, "reasons": list(self.reasons), "message": self.message, "action": self.action,
                "pieces": list(self.pieces), "seconds": round(self.seconds, 3)}


def _trim_words(trim: dict) -> str:
    """``"trimmed 0.42 s (no finger width)"`` for the one-line outcome."""
    from umiproc.run.pieces import _CAUSE_WORDS

    why = ", ".join(_CAUSE_WORDS.get(k, k) for k, v in (trim.get("causes") or {}).items() if v)
    return f"trimmed {trim.get('seconds', 0):.2f} s" + (f" ({why})" if why else "")


def _rec(status: str, *, reasons=(), reason=None, outputs=None, details=None) -> dict:
    """One record of the copied ``status.json`` format (``umiproc.dp.process.EpisodeProcessor._record``)."""
    from umiproc.dp import __version__

    now = time.time()
    out = {"status": status, "reason": reason, "reasons": list(reasons), "inputs_hash": "", "outputs": outputs or {},
           "t_start": now, "t_end": now, "version": STAGE_VERSION, "umiproc.dp_version": __version__,
           "by": "umiproc run"}
    if details:
        out["details"] = details
    return out


class _Stop(Exception):
    """A recording stops at ``stage`` (``status``: rejected | failed) with ``reasons``."""

    def __init__(self, stage: str, status: str, reasons: list[str], message: str):
        super().__init__(message)
        self.stage, self.status, self.reasons, self.message = stage, status, list(reasons), message


class RecordingProcessor:
    """Processes the recordings of one run, one at a time (module docstring). Raises for problems that stop the
    whole run (intake, filesystem); every problem of the recording itself is its :class:`Outcome`."""

    def __init__(self, session: Session):
        self.s = session
        self.task_dir = session.task_dir

    # ----------------------------------------------------------------------------------------- entry
    def run(self, rec) -> Outcome:
        """``rec``: a ``umiproc.dp.run.plan.Recording`` with action take / resume / retry."""
        from umiproc.dp.run import staging
        from umiproc.dp.run.plan import RESUME, RETRY, TAKE

        t0 = time.perf_counter()
        if rec.action == TAKE:
            o = self._take(rec)
            if o is not None:
                o.action, o.seconds = rec.action, time.perf_counter() - t0
                return o
        elif rec.action == RESUME:
            self._record_source(rec)
        elif rec.action == RETRY:
            staging.unfail(self.task_dir, rec.pi_name)
            shutil.rmtree(self.task_dir / STAGING_DIR / rec.pi_name / "processed", ignore_errors=True)
            staging.sync_sources(self.s.root, self.task_dir)
        else:
            raise ValueError(f"nothing to do for {rec.name} ({rec.action})")
        o = self.process(rec.name, rec.pi_name, rec.source_name)
        o.action, o.seconds = rec.action, time.perf_counter() - t0
        staging.sync_sources(self.s.root, self.task_dir)
        return o

    def _take(self, rec) -> Outcome | None:
        """Convert + intake into ``.staging/<pi_name>``. An :class:`Outcome` when it did not get there."""
        from umiproc.recordings import DEMONSTRATION, RecordingError, discard, stage, take
        from umiproc.sensors import recording_mismatches

        bad = recording_mismatches(self.s.sensors, rec.source.meta or {})
        if bad:
            return Outcome(rec.name, rec.pi_name, SKIPPED, message="does not fit rig "
                           f"{self.s.rig_id}'s settings: " + "; ".join(bad) + " (not taken)")
        try:
            staged = stage(self.s.root, rec.source, self.s.rig_id)
        except RecordingError as e:
            return Outcome(rec.name, rec.pi_name, SKIPPED, message=f"{e} (not taken)")
        try:
            taken = take(self.s.root, staged, DEMONSTRATION, folder=self.s.folder, log=lambda *a, **k: None)
        finally:
            discard(staged)
        rec.pi_name = taken.pi_name
        if taken.status != "ingested":
            where = taken.quarantine_dir.relative_to(self.s.root.path) if taken.quarantine_dir else "quarantine/"
            return Outcome(rec.name, taken.pi_name, FAILED, slots=[str(where)], stage="intake",
                           reasons=list(taken.reasons), message=f"quarantined by intake (see {where}/reason.json)")
        return None

    def _record_source(self, rec) -> None:
        """A ``.staging/`` recording whose ``pi_sources.csv`` row was lost: add it (as the copied run does)."""
        from umiproc.dp.pi_adapter import SOURCES_FILE, append_sources

        src = rec.source
        if src is None or rec.taken is not None:
            return
        append_sources(self.s.root.path / SOURCES_FILE, [{
            "source_name": src.name, "source_manifest_sha256": src.manifest_sha256, "converted_name": rec.pi_name,
            "task": self.s.folder, "kind": "demonstration", "status": "ingested", "adapted_unix_ns": time.time_ns()}])
        self.s.log(f"reconciled: {rec.name} was in .staging/{rec.pi_name} but not in {SOURCES_FILE}; recorded it")

    # ----------------------------------------------------------------------------------------- one recording
    def process(self, name: str, pi_name: str, source_name: str | None = None) -> Outcome:
        """Process ``.staging/<pi_name>`` (module docstring): localize + pose checks unless ``localize.json`` says
        they are done for this map and fingerprint, then finish it or leave it waiting."""
        from umiproc.dp.stages.base import Episode

        d = self.task_dir / STAGING_DIR / pi_name
        processed = d / "processed"
        loc = read_json(processed / LOCALIZE_FILE)
        ep = Episode(d, pi_name)
        if self._promotion_interrupted(loc, processed):
            self.s.log(f"  {pi_name}: exported earlier, its move into episode_N was interrupted: finishing the move "
                       "(no new localization, no new export)")
            return self._promote_outcome(name, pi_name, loc)
        try:
            if not self._localized(loc):
                shutil.rmtree(processed, ignore_errors=True)
                processed.mkdir(parents=True)
                try:
                    loc = self._localize_phase(ep)
                except _Stop:
                    raise
                except Exception as e:  # noqa: BLE001 - one bad recording never stops the run (as the copy)
                    raise self._exception(ep, e) from None
            else:
                self.s.log(f"  {pi_name}: localized earlier (pose checks done), only the widths and export run now")
            if not self.s.calibrated:
                return self._waiting(name, pi_name, loc)
            return self._finish_phase(name, pi_name, source_name, ep, loc)
        except _Stop as e:
            st = load_status(processed)
            if (st.get(e.stage) or {}).get("status") != e.status:     # check / grid saved their own record
                st[e.stage] = _rec(e.status, reasons=e.reasons, reason=e.message)
                save_status(processed, st)
            return self._fail(name, pi_name, source_name, e.stage, read_json(processed / LOCALIZE_FILE) or {})

    def _exception(self, ep, e: Exception) -> _Stop:
        """An unexpected error in a recording's steps: the traceback in ``processed/error.txt``, the first stage
        not done fails with ``exception`` (as the copied ``EpisodeProcessor``)."""
        import traceback

        from umiproc.dp.files import ERROR_FILE
        from umiproc.dp.fsutil import write_atomic

        write_atomic(ep.processed / ERROR_FILE, traceback.format_exc())
        st = load_status(ep.processed)
        stage = next((x for x in RECORDING_STAGES + PIECE_STAGES if (st.get(x) or {}).get("status") != "done"),
                     "export")
        return _Stop(stage, "failed", ["exception"], f"{type(e).__name__}: {e} (traceback in processed/{ERROR_FILE})")

    def _promotion_interrupted(self, loc, processed: Path) -> bool:
        """``localize.json`` says ``finished`` while the recording is still in ``.staging/``: every exported piece
        is in ``processed/pieces/<k>/`` (or, for the first one, already moved into ``processed/``) and only the move
        into ``episode_N`` was interrupted."""
        if not (isinstance(loc, dict) and loc.get("format") == LOCALIZE_FORMAT and loc.get("phase") == FINISHED):
            return False
        exported = [p for p in loc.get("pieces") or [] if p.get("outcome") == EPISODE]
        if not exported:
            return False
        first = processed / PIECES_DIR / str(exported[0]["index"])
        return first.is_dir() or (processed / "episode.zarr").is_dir()

    def _localized(self, loc) -> bool:
        return (isinstance(loc, dict) and loc.get("format") == LOCALIZE_FORMAT and loc.get("phase") == LOCALIZED
                and loc.get("fingerprint_hash") == self.s.fingerprint_hash
                and loc.get("map", {}).get("pi_name") == self.s.map_pi_name)

    def _save(self, ep, status: dict) -> None:
        save_status(ep.processed, status)

    # ----------------------------------------------------------------------------------------- steps 1-4
    def _localize_phase(self, ep) -> dict:
        from umiproc.dp.stages.base import StageFailed
        from umiproc.dp.stages.check import CheckStage
        from umiproc.dp.stages.geometry import GridStage

        ctx, log = self.s.ctx, self.s.log
        status: dict = {}
        for stage in (CheckStage(), GridStage()):
            try:
                status[stage.name] = _rec("done", outputs=stage.run(ctx, ep))
            except StageFailed as e:
                status[stage.name] = _rec(e.status, reasons=e.reasons, reason=str(e), details=e.details)
                self._save(ep, status)
                raise _Stop(stage.name, e.status, e.reasons, str(e)) from None
            self._save(ep, status)
        log(f"  {ep.pi_name}: localizing against map/{self.s.map_pi_name} (frames fed in real time)")
        loc = {"format": LOCALIZE_FORMAT, "pi_name": ep.pi_name, "map": {"pi_name": self.s.map_pi_name},
               "fingerprint_hash": self.s.fingerprint_hash, "phase": None, "created_utc": iso_utc(time.time())}
        traj, run_info, err = self._localize(ep)
        loc["run"] = run_info
        if err is not None:
            loc["phase"] = "failed"
            write_json(ep.processed / LOCALIZE_FILE, loc)
            raise _Stop("slam", "failed", ["slam_error"], err)
        rules, reject = self._rules(traj, run_info)
        loc["rules"] = rules.to_json()
        if reject is not None:
            loc["phase"] = "failed"
            write_json(ep.processed / LOCALIZE_FILE, loc)
            raise _Stop("slam", "rejected", reject[0], reject[1])
        status["slam"] = _rec("done", outputs={"dir": LOCALIZE_DIR, "n_frames": rules.n_frames,
                                               "n_tracked": rules.n_tracked, "n_lost": rules.n_lost,
                                               "pieces": len(rules.pieces), "map": self.s.map_pi_name})
        self._save(ep, status)
        log(f"  {ep.pi_name}: {rules.message}")
        loc["pieces"], loc["demo_checks"], loc["orientation"] = self._pose_phase(ep, traj, rules)
        passed = [p for p in loc["pieces"] if p["pose"]["ok"]]
        if not passed:
            loc["phase"] = "failed"
            write_json(ep.processed / LOCALIZE_FILE, loc)
            demo = loc["demo_checks"]
            if demo["reasons"]:
                raise _Stop("pose", "rejected", demo["reasons"], "the demo failed its pose checks: "
                            + ", ".join(demo["reasons"]) + " (" + demo["summary"] + ")")
            reasons = _piece_reasons(loc["pieces"], "pose")
            raise _Stop("pose", "rejected", reasons, "no piece passed its pose checks: " + "; ".join(reasons))
        status["pose"] = _rec("done", outputs={"pieces_passed": [p["index"] for p in passed],
                                               "pieces": len(loc["pieces"]), "world": "table_tag",
                                               "world_offset": "identity",
                                               "demo_checks": {k: loc["demo_checks"][k]
                                                               for k in ("reasons", "reported", "summary")}})
        self._save(ep, status)
        loc["phase"] = LOCALIZED
        write_json(ep.processed / LOCALIZE_FILE, loc)
        return loc

    def _localize(self, ep):
        """The localization run. Returns ``(ForkTrajectory | None, run info, error message | None)``."""
        from umiproc.dp.slam.base import SlamError
        from umiproc.dp.slam.orbslam3 import write_euroc
        from umiproc.map.build import _run_info
        from umiproc.run.rules import MAX_LOST_FRAMES
        from umiproc.slam.runner import SlamRunError, run_umi_slam
        from umiproc.slam.trajectory_csv import TrajectoryCsvError, read_fork_csv

        out = ep.processed / LOCALIZE_DIR
        work = self.s.work / ep.pi_name
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            try:
                prep = write_euroc(ep.raw, self.s.rig, work / "euroc")
            except SlamError as e:
                return None, {}, f"the frames could not be prepared for SLAM: {e}"
            out.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "euroc" / "times.txt", out / "times.txt")
            try:
                res = run_umi_slam(work / "euroc", self.s.inputs.settings, out, load_map=self.s.atlas,
                                   mask_img=self.s.inputs.mask, max_lost_frames=MAX_LOST_FRAMES, realtime=True,
                                   binary=self.s.binary, vocabulary=self.s.vocabulary, timeout_s=self.s.timeout_s,
                                   runner=self.s.runner)
            except SlamRunError as e:
                return None, {}, str(e)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        info = _run_info(res)
        info["euroc"] = {k: v for k, v in prep.items() if k not in ("t_cam_ns", "out_dir")}
        s = res.summary
        if res.timed_out or s is None or res.returncode not in (0, 3):
            why = ("timed out" if res.timed_out else "died before its summary line" if s is None
                   else f"exited with {res.returncode}")
            return None, info, f"the localization run {why} (read {LOCALIZE_DIR}/{res.log_path.name})"
        if s.stop_reason not in (None, "max_lost_frames"):
            return None, info, f"the localization run stopped: {s.stop_reason} ({s.error or 'see its log'})"
        try:
            traj = read_fork_csv(res.trajectory_csv, out / "times.txt", frames_fed=s.frames_fed)
        except TrajectoryCsvError as e:
            return None, info, f"the localization's trajectory could not be read: {e}"
        info["maps_with_keyframes"] = len(s.maps_with_keyframes)
        return traj, info, None

    def _rules(self, traj, info):
        """UMI's rules, after the new-map guard. Returns ``(RulesResult, None | (reasons, message))``."""
        from umiproc.run.rules import REASONS, apply_rules

        stopped = (info.get("stop_reason") == "max_lost_frames")
        rules = apply_rules(traj.tracked, traj.t_ns if traj.n_frames == traj.n_frames_total
                            else _all_times(traj), stopped=stopped, n_fed=traj.n_frames)
        if not stopped and info.get("maps_with_keyframes", 1) > 1:
            return rules, (["new_map"], f"the localization started a new map ({info['maps_with_keyframes']} maps "
                                        "with keyframes): its poses would not be in the map's coordinates")
        if not rules.ok:
            return rules, (rules.reasons, rules.message + ": " + "; ".join(REASONS[r] for r in rules.reasons))
        return rules, None

    def _pose_phase(self, ep, traj, rules) -> tuple[list[dict], dict, dict]:
        """The demo's pose checks once on all its tracked frames (gravity, scale, orientation), then each piece's
        own (speed). A demo failure fails every piece. Returns ``(pieces, demo checks, orientation)``."""
        from umiproc.run.pieces import demo_pose_checks, describe_demo_checks, orientation_check, pose_checks

        grid = ep.grid()
        demo = demo_pose_checks(traj, ep.raw, self.s.rig)
        orient = orientation_check(traj, ep.raw, self.s.rig)
        if not orient["ok"]:
            demo = {**demo, "ok": False, "reasons": demo["reasons"] + ["orientation"]}
        demo["summary"] = describe_demo_checks(demo, orient)
        self.s.log(f"  {ep.pi_name}: demo pose checks " + ("passed" if demo["ok"] else "FAILED: "
                                                           + ", ".join(demo["reasons"])) + f", {demo['summary']}")
        pieces = []
        for p in rules.pieces:
            chk = pose_checks(p, traj, self.s.rig, self.s.tx_slam_tag, grid)
            reasons = chk["reasons"] + demo["reasons"]
            chk = {**chk, "ok": not reasons, "reasons": reasons, "piece_reasons": chk["reasons"]}
            pieces.append({**p.to_json(), "pose": chk, "outcome": WAITING if chk["ok"] else FAILED,
                           "stage": None if chk["ok"] else "pose", "reasons": [] if chk["ok"] else reasons})
            self.s.log(f"  {ep.pi_name}: piece {p.index} ({p.n_frames} frames): pose checks "
                       + ("passed" if chk["ok"] else "FAILED: " + ", ".join(reasons)))
        return pieces, demo, orient

    # ----------------------------------------------------------------------------------------- 5: waiting
    def _waiting(self, name, pi_name, loc) -> Outcome:
        return Outcome(name, pi_name, WAITING, slots=[f"{STAGING_DIR}/{pi_name}"],
                       message="waiting for the gripper calibration", pieces=_piece_summary(loc["pieces"]))

    # ----------------------------------------------------------------------------------------- 6-7: finish
    def _finish_phase(self, name, pi_name, source_name, ep, loc) -> Outcome:
        from umiproc.dp.export.images import camera_times
        from umiproc.dp.slam.base import imu_samples
        from umiproc.dp.stages.width import detect_episode_tags, episode_widths
        from umiproc.run.pieces import describe_trim, finish_piece, piece_zarr_name
        from umiproc.run.rules import Piece
        from umiproc.slam.trajectory_csv import read_fork_csv

        ctx, rig, log = self.s.ctx, self.s.rig, self.s.log
        cal = ctx.calibration
        pieces_dir = ep.processed / PIECES_DIR
        shutil.rmtree(pieces_dir, ignore_errors=True)
        loc_dir = ep.processed / LOCALIZE_DIR
        traj = read_fork_csv(loc_dir / "trajectory.csv", loc_dir / "times.txt", frames_fed=loc["run"]["frames_fed"])
        grid = ep.grid()
        log(f"  {pi_name}: finger widths (gripper calibration {cal.name})")
        try:
            t_frames, dets = detect_episode_tags(ep.raw, rig, ctx.prep.to_virtual)
            t_w, w, _lp, _rp = episode_widths(dets, t_frames, cal.gripper_range, rig)
            widths = (t_w, w, t_frames, np.isin(t_frames, t_w))
            uw_t = camera_times(ep.raw, rig, "ultrawide")
            imu_t = imu_samples(ep.raw, rig)[0]
        except Exception as e:  # noqa: BLE001 - one bad recording never stops the run (as the copy)
            raise self._exception(ep, e) from None
        candidates = [p for p in loc["pieces"] if p["pose"]["ok"]]
        n_all = len(loc["pieces"])
        for p in candidates:
            piece = Piece(p["index"], p["frames"][0], p["frames"][1], p["t_first_ns"], p["t_last_ns"])
            pdir = pieces_dir / str(piece.index)
            pdir.mkdir(parents=True)
            try:
                res = finish_piece(piece, traj, ep.raw, rig, ctx.prep, self.s.tx_slam_tag, grid, widths,
                                   uw_t_ns=uw_t, imu_t_ns=imu_t, label=self.s.label,
                                   name=piece_zarr_name(pi_name, piece.index, n_all), out_zarr=pdir / "episode.zarr")
            except Exception as e:  # noqa: BLE001 - this piece fails, the others go on (as the copy's stages)
                import traceback

                (ep.processed / f"error_piece{piece.index}.txt").write_text(traceback.format_exc())
                res = {"ok": False, "stage": "export", "reasons": ["exception"], "values": {},
                       "message": f"{type(e).__name__}: {e} (traceback in processed/error_piece{piece.index}.txt)"}
            p["finish"] = {k: res[k] for k in ("ok", "stage", "reasons", "message")}
            p["finish"]["values"] = _short_values(res["values"])
            trim = describe_trim(res["values"].get("trimmed"))
            if res["ok"]:
                p["outcome"], p["stage"], p["reasons"] = EPISODE, None, []
                save_status(pdir, self._piece_status(ep, p, res, cal))
                write_json(pdir / PIECE_FILE, self._piece_record(pi_name, source_name, p, n_all))
                log(f"  {pi_name}: piece {piece.index}: exported, {res['values']['n_steps']} steps"
                    + (f"; {trim}" if trim else ""))
            else:
                p["outcome"], p["stage"], p["reasons"] = FAILED, res["stage"], res["reasons"]
                shutil.rmtree(pdir, ignore_errors=True)
                log(f"  {pi_name}: piece {piece.index}: FAILED at {res['stage']}: {', '.join(res['reasons'])}"
                    + (f"; {trim}" if trim else ""))
        exported = [p for p in loc["pieces"] if p.get("outcome") == EPISODE]
        status = load_status(ep.processed)
        if not exported:
            loc["phase"] = "failed"
            write_json(ep.processed / LOCALIZE_FILE, loc)
            stage = _furthest([p.get("stage") for p in candidates]) or "validate"
            reasons = _piece_reasons(loc["pieces"], None)
            status[stage] = _rec("rejected" if stage == "validate" else "failed", reasons=reasons,
                                 reason="no piece was exported: " + "; ".join(reasons))
            save_status(ep.processed, status)
            shutil.rmtree(pieces_dir, ignore_errors=True)
            return self._fail(name, pi_name, source_name, stage, loc)
        loc["phase"] = FINISHED          # from here on a crash is finished by the next run (_promotion_interrupted)
        write_json(ep.processed / LOCALIZE_FILE, loc)
        return self._promote_outcome(name, pi_name, loc)

    def _promote_outcome(self, name, pi_name, loc: dict) -> Outcome:
        exported = [p for p in loc["pieces"] if p.get("outcome") == EPISODE]
        slots = self._promote(pi_name, exported)
        for p, slot in zip(exported, slots):
            p["slot"] = slot
        write_json(self.task_dir / slots[0] / "processed" / LOCALIZE_FILE, loc)
        return Outcome(name, pi_name, EPISODE, slots=slots, pieces=_piece_summary(loc["pieces"]))

    def _piece_status(self, ep, p: dict, res: dict, cal) -> dict:
        """The copied status.json of one exported piece: the recording's records, then the piece's own."""
        st = {k: v for k, v in load_status(ep.processed).items() if k in RECORDING_STAGES}
        v = res["values"]
        st["width"] = _rec("done", outputs={"calibration": cal.name, "n": v["grid_rows"]})
        st["validate"] = _rec("done", outputs={"kept": v["validate"]["kept"], "metrics": v["validate"]["metrics"]})
        st["images"] = _rec("done", outputs={"n_steps": v["n_steps"], "n_uw": v["n_uw"]})
        st["export"] = _rec("done", outputs={"dir": "episode.zarr", "n_steps": v["n_steps"], "n_uw": v["n_uw"],
                                             "name": v["name"], "task_label": v["task_label"]})
        return st

    def _piece_record(self, pi_name, source_name, p: dict, n_all: int) -> dict:
        return {"format": PIECE_FORMAT, "pi_name": pi_name, "source_name": source_name, "piece": p["index"],
                "pieces": n_all, "frames": p["frames"], "n_frames": p["n_frames"], "t_first_ns": p["t_first_ns"],
                "t_last_ns": p["t_last_ns"], "name": p["finish"]["values"].get("name"), "world": "table_tag",
                "world_offset": "identity", "trimmed": p["finish"]["values"].get("trimmed"),
                "map": self.s.map_pi_name, "fingerprint_hash": self.s.fingerprint_hash, "raw_of": None}

    def _promote(self, pi_name: str, exported: list[dict]) -> list[str]:
        """``.staging/<pi_name>`` -> ``episode_N`` with the first exported piece; one more ``episode_N`` per other
        piece (raw -> the first one's). Returns the slots in piece order. Every step is a rename and each can be
        repeated after a crash: the first piece's files that are still in ``pieces/<k>/`` are moved, then the
        folder; the other pieces are :func:`finish_interrupted_pieces`' (also run at the start of every run)."""
        from umiproc.dp.run import staging

        d = self.task_dir / STAGING_DIR / pi_name
        pieces_dir = d / "processed" / PIECES_DIR
        first = pieces_dir / str(exported[0]["index"])
        if first.is_dir():
            for f in ("episode.zarr", "status.json", PIECE_FILE):
                if os.path.lexists(first / f):
                    _replace(first / f, d / "processed" / f)
            first.rmdir()
        slot = staging.promote(self.task_dir, pi_name)
        return [slot] + finish_interrupted_pieces(self.task_dir, only=slot)

    def _fail(self, name, pi_name, source_name, stage: str, loc: dict) -> Outcome:
        from umiproc.dp.run import staging

        reason = staging.fail(self.task_dir, f"{STAGING_DIR}/{pi_name}", pi_name, source_name)
        reason["pieces"] = _piece_summary(loc.get("pieces") or [])
        if loc.get("rules"):
            reason["rules"] = {k: loc["rules"][k] for k in ("message", "reasons", "n_frames", "n_tracked", "n_lost",
                                                             "first_tracked_s")}
        write_json(self.task_dir / "failed" / pi_name / staging.REASON_FILE, reason)
        return Outcome(name, pi_name, FAILED, slots=[f"failed/{pi_name}"], stage=reason["stage"],
                       reasons=reason["reasons"], message=reason["message"], pieces=reason["pieces"])


# --------------------------------------------------------------------------------------------- helpers

def _all_times(traj) -> np.ndarray:
    """times.txt of a run that stopped early (``read_fork_csv`` keeps only the fed frames' times)."""
    from umiproc.slam.trajectory_csv import read_times

    return read_times(Path(traj.path).parent / "times.txt")


_ORDER = ("pose", "width", "validate", "images", "export")


def _furthest(stages) -> str | None:
    known = [s for s in stages if s in _ORDER]
    return max(known, key=_ORDER.index) if known else None


def _piece_reasons(pieces: list[dict], stage: str | None) -> list[str]:
    out = []
    for p in pieces:
        if p.get("outcome") == EPISODE or (stage is not None and p.get("stage") != stage):
            continue
        out.append(f"piece {p['index']}: " + ", ".join(p.get("reasons") or ["?"]))
    return out


def _piece_summary(pieces: list[dict]) -> list[dict]:
    out = []
    for p in pieces:
        trim = ((p.get("finish") or {}).get("values") or {}).get("trimmed")
        out.append({"index": p["index"], "frames": p["frames"], "outcome": p.get("outcome"), "stage": p.get("stage"),
                    "reasons": list(p.get("reasons") or []), **({"slot": p["slot"]} if p.get("slot") else {}),
                    **({"trimmed": trim} if trim is not None else {})})
    return out


def _short_values(v: dict) -> dict:
    """What localize.json keeps of a finish result (the validate record without its thresholds)."""
    out = {k: x for k, x in v.items() if k != "validate"}
    if "validate" in v:
        out["validate"] = {k: v["validate"][k] for k in ("status", "reasons", "kept", "metrics")}
    return out


def finish_interrupted_pieces(task_dir: Path, only: str | None = None) -> list[str]:
    """Give every piece left in ``episode_N/processed/pieces/<k>/`` (an interrupted promotion, or the one just
    promoted with ``only``) its own ``episode_M``: ``processed/`` moved in, its ``episodes.csv`` row (a copy of
    ``episode_N``'s), then ``raw`` -> ``../episode_N/raw``; ``episode_N``'s ``localize.json`` records the slot.
    Without ``only`` also links a piece slot whose ``raw`` link is missing (and adds its missing ledger row).

    Every step can be interrupted and repeated: ``piece.json`` names the slot before it is made, so an
    ``episode_M`` left empty by a crash is reused instead of leaving a gap; a piece moved but not yet linked is
    linked by the second pass; a ledger row is only added when the slot has none. Returns the new slots."""
    from umiproc.dp.intake import Ledger
    from umiproc.dp.paths import episode_dirs, next_episode_index

    task_dir = Path(task_dir)
    made = []
    ledger = Ledger(task_dir)
    for d in episode_dirs(task_dir):
        if only is not None and d.name != only:
            continue
        pdir = d / "processed" / PIECES_DIR
        if pdir.is_dir():
            rows = [r for r in ledger.rows() if r["slot"] == d.name]
            for k in sorted((p for p in pdir.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
                if not (k / "status.json").is_file() or not (k / "episode.zarr").is_dir():
                    shutil.rmtree(k, ignore_errors=True)
                    continue
                rec = read_json(k / PIECE_FILE, {}) or {}
                slot = rec.get("slot")
                new = task_dir / slot if slot else None
                if new is None or not new.is_dir() or any(new.iterdir()) or os.path.islink(new):
                    slot = f"episode_{max(1, next_episode_index(task_dir))}"
                    new = task_dir / slot
                    rec.update(raw_of=d.name, slot=slot)
                    write_json(k / PIECE_FILE, rec)
                    new.mkdir()
                _replace(k, new / "processed")
                if rows and slot not in ledger.slots():
                    ledger.append({**rows[-1], "slot": slot, "intake_unix_ns": str(time.time_ns())})
                _symlink(os.path.join("..", d.name, "raw"), new / "raw")
                _record_slot(d, rec.get("piece"), slot)
                made.append(slot)
            shutil.rmtree(pdir, ignore_errors=True)
    if only is None:
        for d in episode_dirs(task_dir):
            rec = read_json(d / "processed" / PIECE_FILE, {}) or {}
            if rec.get("raw_of") and not os.path.lexists(d / "raw"):
                if d.name not in ledger.slots():
                    rows = [r for r in ledger.rows() if r["slot"] == rec["raw_of"]]
                    if rows:
                        ledger.append({**rows[-1], "slot": d.name, "intake_unix_ns": str(time.time_ns())})
                _symlink(os.path.join("..", rec["raw_of"], "raw"), d / "raw")
                _record_slot(task_dir / rec["raw_of"], rec.get("piece"), d.name)
    return made


def _record_slot(first: Path, piece: int | None, slot: str) -> None:
    """Note in the first piece's ``processed/localize.json`` that piece ``piece`` is ``slot`` (and that the first
    piece, the one in ``first/processed/piece.json``, is ``first`` itself)."""
    first = Path(first)
    path = first / "processed" / LOCALIZE_FILE
    loc = read_json(path)
    if not isinstance(loc, dict):
        return
    want = {(read_json(first / "processed" / PIECE_FILE, {}) or {}).get("piece"): first.name, piece: slot}
    changed = False
    for p in loc.get("pieces") or []:
        k = p.get("index")
        if k is not None and k in want and p.get("slot") != want[k]:
            p["slot"], changed = want[k], True
    if changed:
        write_json(path, loc)
