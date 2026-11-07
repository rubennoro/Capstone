"""``dataproc process``: run every stage on a task's episodes and rebuild ``output/<folder>.zarr.zip``.

Read top to bottom: :class:`TaskProcessor` (the command) runs :class:`EpisodeProcessor` (one episode, the stage
classes of :mod:`dataproc.stages`) for each episode. Behaviour: docs/PROCESS.md (same step names). Python API::

    res = run_process(root, "block_raise", episode=None, force=False, slam="orbslam3", log=print)
    res.exit_code, res.zip_path, res.report          # report == output/report.json
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dataproc import __version__
from dataproc.files import ERROR_FILE, hash_of, iso_utc, load_status, save_status, sha256_hex, write_json
from dataproc.fsutil import write_atomic
from dataproc.paths import DataRoot, episode_dirs, episode_index
from dataproc.report import episode_summary, format_table
from dataproc.stages.base import Episode, ProcessContext, StageFailed
from dataproc.stages.calibration import CalibrationStep
from dataproc.stages.check import CheckStage
from dataproc.stages.geometry import GridStage, PoseStage, SlamStage, WidthStage
from dataproc.stages.output import ExportStage, ImagesStage
from dataproc.stages.validate import ValidateStage

__all__ = ["REPORT_FILE", "ProcessError", "ProcessResult", "TaskProcessor", "EpisodeProcessor", "run_process",
           "cli_process", "zip_path"]

REPORT_FILE = "report.json"


class ProcessError(Exception):
    """Fatal for the whole run (exit 1): task folder, task.yaml, rig yaml or ledger problems, bad arguments."""


@dataclass
class ProcessResult:
    exit_code: int
    zip_path: Path | None = None
    report: dict | None = None
    ran: dict = field(default_factory=dict)        # slot -> names of the stages that ran in this call
    error: str | None = None


def zip_path(task_dir: Path, folder: str) -> Path:
    return Path(task_dir) / "output" / f"{folder}.zarr.zip"


# --------------------------------------------------------------------------------------------- the command

class TaskProcessor:
    """``dataproc process --task FOLDER [--episode N] [--force] [--slam orbslam3|oracle]``."""

    def __init__(self, root: DataRoot, folder: str, episode: int | None = None, force: bool = False,
                 slam: str = "orbslam3", backend=None, log: Callable = print):
        self.root, self.folder, self.episode, self.force = root, folder, episode, force
        self.slam, self.backend, self.log = slam, backend, log
        self.task_dir = root.tasks / folder
        self.context: ProcessContext | None = None
        self.all_episodes: list[Path] = []
        self.ran: dict[str, list[str]] = {}
        self.zip_info: dict | None = None
        self.zip_error: str | None = None
        self.t_start = time.time()

    def run(self) -> ProcessResult:
        try:
            self.load_task()
        except ProcessError as e:
            return ProcessResult(1, error=str(e))
        self.process_calibration()
        for ep_dir in self.episodes_to_process():
            self.process_episode(ep_dir)
        self.build_output()
        return self.write_report()

    def load_task(self) -> None:
        """Reads ``task.yaml``, ``episodes.csv`` and ``rigs/<task.yaml rig>.yaml``; resolves the SLAM backend and
        the episodes (``episode_N`` dirs with a ``raw/``; empty ones and dot-entries are ignored). Writes nothing.
        Raises :class:`ProcessError` (exit 1, nothing created) when any of them is missing or invalid, or when
        ``--episode N`` names no such episode."""
        from dataproc.config import ConfigError, load_rig, load_task_config
        from dataproc.intake import IntakeError, read_ledger
        from dataproc.slam import BACKENDS, get_backend

        from dataproc.slam import resolve_backend_name

        self.slam = resolve_backend_name(self.slam)
        if self.slam not in BACKENDS:
            raise ProcessError(f"unknown --slam {self.slam!r} (expected one of {sorted(BACKENDS)})")
        if not self.task_dir.is_dir():
            raise ProcessError(f"no task folder {self.task_dir} (run intake first)")
        ty = self.task_dir / "task.yaml"
        if not ty.is_file():
            raise ProcessError(f"{ty} is missing (run intake first)")
        try:
            task = load_task_config(ty)
            rig_file = self.root.rig_file(task.rig)
            if not rig_file.is_file():
                raise ProcessError(f"rig yaml {rig_file} is missing (task.yaml rig: {task.rig}); copy the rig "
                                   "description there")
            rig = load_rig(rig_file)
            ledger = read_ledger(self.task_dir / "episodes.csv")
        except (ConfigError, IntakeError, ValueError) as e:
            raise ProcessError(str(e)) from None
        self.all_episodes = [d for d in episode_dirs(self.task_dir) if (d / "raw").is_dir()]
        if self.episode is not None and not any(episode_index(d) == self.episode for d in self.all_episodes):
            raise ProcessError(f"{self.task_dir / f'episode_{self.episode}'} does not exist or has no raw/")
        self.context = ProcessContext(root=self.root, folder=self.folder, task_dir=self.task_dir, task=task, rig=rig,
                                      ledger=ledger, backend=self.backend or get_backend(self.slam),
                                      backend_name=self.slam, force=self.force, log=self.log)

    def process_calibration(self) -> None:
        """Runs :class:`~dataproc.stages.calibration.CalibrationStep` (reads the newest ``calibration/<name>/raw``;
        writes its ``processed/`` and ``tasks/<folder>/gripper_range.json``; skipped when unchanged). Its result
        feeds every episode's width stage."""
        c = self.context.calibration = CalibrationStep(self.context).run()
        self.log(f"calibration: {c.state}" + (f" ({c.name})" if c.name else "")
                 + (f": {c.reason}" if c.reason else ""))

    def episodes_to_process(self) -> list[Path]:
        """All episodes, or only ``--episode N``; in ``episode_N`` order."""
        if self.episode is None:
            return list(self.all_episodes)
        return [d for d in self.all_episodes if episode_index(d) == self.episode]

    def process_episode(self, ep_dir: Path) -> None:
        """Runs :class:`EpisodeProcessor` on one episode (sequentially; it never raises for episode problems)."""
        t0 = time.perf_counter()
        self.ran[ep_dir.name] = EpisodeProcessor(self.context, ep_dir).run()
        state = episode_summary(load_status(ep_dir / "processed"))["status"]
        self.log(f"{ep_dir.name}: {state}  [ran: {','.join(self.ran[ep_dir.name]) or 'nothing (up to date)'}; "
                 f"{time.perf_counter() - t0:.1f} s]")

    def exported_episodes(self) -> list[Path]:
        """``processed/episode.zarr`` of every episode whose export is ``done`` (also those not processed in this
        call), in ``episode_N`` order."""
        return [d / "processed" / "episode.zarr" for d in self.all_episodes
                if episode_summary(load_status(d / "processed"))["status"] == "exported"
                and (d / "processed" / "episode.zarr").is_dir()]

    def build_output(self) -> None:
        """Reads every exported ``episode.zarr``; writes ``output/<folder>.zarr.zip`` (always rebuilt; temp file +
        rename, so a failed build leaves the previous zip as it was). An existing zip is **never deleted**: with no
        exported episode, or when the build fails, it is kept (and is stale) and ``zip_error`` says so. After a
        successful build a leftover ``output/error.txt`` from an earlier failed build is removed."""
        from dataproc.export.zarr_writer import build_task_zarr_zip

        out_dir = self.task_dir / "output"
        out_dir.mkdir(exist_ok=True)
        zp = zip_path(self.task_dir, self.folder)
        kept = "; previous zip kept and is stale" if zp.exists() else ""
        exported = self.exported_episodes()
        if not exported:
            self.zip_error = "no exportable episodes" + kept
            return
        try:
            self.zip_info = build_task_zarr_zip(exported, zp, tmp_dir=out_dir)
        except Exception as e:  # noqa: BLE001
            self.zip_error = f"building {zp.name} failed ({type(e).__name__}: {e}){kept}"
            write_atomic(out_dir / ERROR_FILE, traceback.format_exc())
        else:
            (out_dir / ERROR_FILE).unlink(missing_ok=True)

    def write_report(self) -> ProcessResult:
        """Writes ``output/report.json`` (per episode, calibration, totals, rig id + nominal sections, version,
        time; ``zip: null`` + ``zip_note`` when no zip was built in this call) and prints the summary table.
        Returns the result: exit code 0 when the zip was built, else 1."""
        ctx, zp = self.context, zip_path(self.task_dir, self.folder)
        episodes = []
        for d in self.all_episodes:
            s = episode_summary(load_status(d / "processed"))
            episodes.append({"slot": d.name, "pi_name": ctx.pi_name(d), **s, "ran_this_call": self.ran.get(d.name)})
        totals = {k: sum(e["status"] == k for e in episodes)
                  for k in ("exported", "rejected", "failed", "incomplete", "pending")}
        totals.update(episodes=len(episodes), n_steps=sum(e["n_steps"] or 0 for e in episodes))
        report = {
            "task": self.folder, "label": ctx.task.label, "rig_id": ctx.rig.rig_id,
            "rig_nominal_sections": ctx.rig.nominal_sections(), "slam_backend": ctx.backend_name,
            "dataproc_version": __version__, "created": iso_utc(time.time()),
            "duration_s": time.time() - self.t_start, "calibration": ctx.calibration.to_report(),
            "episodes": episodes, "totals": totals,
            "zip": ({"path": str(zp.relative_to(self.task_dir)),
                     **{k: v for k, v in self.zip_info.items() if k != "path"}} if self.zip_info else None),
            "zip_note": self.zip_error,
        }
        report_path = self.task_dir / "output" / REPORT_FILE
        write_json(report_path, report)

        rows = [[e["slot"], e["pi_name"], e["status"], "-" if e["n_steps"] is None else e["n_steps"],
                 "; ".join(e["reasons"]) or "-"] for e in episodes]
        self.log(format_table(rows, ["slot", "pi_name", "status", "n_steps", "reasons"]))
        if ctx.rig.nominal_sections():
            self.log(f"note: rig {ctx.rig.rig_id} still has nominal sections: {', '.join(ctx.rig.nominal_sections())}")
        if self.zip_info:
            self.log(f"wrote {zp} ({self.zip_info['n_episodes']} episodes, {self.zip_info['n_steps']} steps) and "
                     f"{report_path}")
            return ProcessResult(0, zp, report, self.ran)
        self.log(f"no zip built: {self.zip_error}; wrote {report_path}")
        return ProcessResult(1, None, report, self.ran, error=self.zip_error)


# --------------------------------------------------------------------------------------------- one episode

class EpisodeProcessor:
    """Runs the stages of one ``episode_N`` in order and records each in ``processed/status.json``.

    For each stage: compute its ``inputs_hash`` -> skip it if it is up to date -> run it -> record it. The
    episode stops at the first stage that is ``failed`` or ``rejected`` (newly or from an earlier run); the stages
    after it are recorded as ``skipped``. A stage is up to date when there is no ``--force``, none of its ``deps``
    ran in this call, its record has the same ``inputs_hash`` and its status is ``done`` / ``failed`` /
    ``rejected``; a stage that ``feeds`` another (images -> export) also needs that one to be up to date. An
    unexpected exception fails the stage (``exception``, traceback in ``processed/error.txt``) and never escapes."""

    STAGES = (CheckStage, SlamStage, GridStage, PoseStage, WidthStage, ValidateStage, ImagesStage, ExportStage)

    def __init__(self, ctx: ProcessContext, ep_dir: Path):
        self.ctx = ctx
        self.ep = Episode(ep_dir, ctx.pi_name(ep_dir))
        self.stages = [cls() for cls in self.STAGES]
        self.status: dict = {}
        self.hashes: dict[str, str] = {}
        self.ran: list[str] = []
        self.stopped_by: str | None = None          # "<stage> <status>" once the episode stopped

    def run(self) -> list[str]:
        """Process the episode; returns the names of the stages that ran."""
        self.ep.processed.mkdir(exist_ok=True)
        self.status = load_status(self.ep.processed)
        self.hashes = self.inputs_hashes()
        for stage in self.stages:
            if self.stopped_by:
                self.record_skipped(stage)
                continue
            if not self.is_up_to_date(stage):
                self.run_stage(stage)
            state = self.status[stage.name]["status"]
            if state in ("failed", "rejected"):
                self.stopped_by = f"{stage.name} {state}"
        return self.ran

    def inputs_hashes(self) -> dict[str, str]:
        """Every stage's ``inputs_hash``: sha256 over the stage name and version, the dataproc version, the
        episode's MANIFEST.sha256, the stage's ``inputs(ctx, ep)`` and the hashes of its ``deps``."""
        base = {"dataproc": __version__, "manifest_sha256": sha256_hex(self.ep.manifest_bytes())}
        out: dict[str, str] = {}
        for st in self.stages:
            out[st.name] = hash_of({"stage": st.name, "stage_version": st.version, **base,
                                    "inputs": st.inputs(self.ctx, self.ep),
                                    "upstream": {d: out[d] for d in st.deps}})
        return out

    def is_up_to_date(self, stage) -> bool:
        if not self._own_up_to_date(stage.name) or any(d in self.ran for d in stage.deps):
            return False
        feeds = getattr(stage, "feeds", None)
        return feeds is None or self._own_up_to_date(feeds)

    def _own_up_to_date(self, name: str) -> bool:
        prev = self.status.get(name)
        return (not self.ctx.force and prev is not None and prev.get("inputs_hash") == self.hashes[name]
                and prev.get("status") in ("done", "failed", "rejected"))

    def run_stage(self, stage) -> None:
        prev = self.status.get(stage.name) or {}
        if prev.get("error_file"):
            (self.ep.processed / ERROR_FILE).unlink(missing_ok=True)
        t0 = time.time()
        try:
            outputs = stage.run(self.ctx, self.ep)
            rec = self._record(stage, "done", t0, outputs=outputs)
        except StageFailed as e:
            rec = self._record(stage, e.status, t0, reasons=e.reasons, reason=str(e), details=e.details)
        except Exception as e:  # noqa: BLE001 - one bad episode never stops the run
            write_atomic(self.ep.processed / ERROR_FILE, f"stage {stage.name}\n" + traceback.format_exc())
            rec = self._record(stage, "failed", t0, reasons=["exception"], reason=f"{type(e).__name__}: {e}",
                               error_file=ERROR_FILE)
        self.status[stage.name] = rec
        self.ran.append(stage.name)
        save_status(self.ep.processed, self.status)

    def record_skipped(self, stage) -> None:
        prev = self.status.get(stage.name) or {}
        if (prev.get("status") == "skipped" and prev.get("reason") == self.stopped_by
                and prev.get("inputs_hash") == self.hashes[stage.name]):
            return                                   # unchanged: keep its timestamps
        self.status[stage.name] = self._record(stage, "skipped", time.time(), reason=self.stopped_by)
        save_status(self.ep.processed, self.status)

    def _record(self, stage, status: str, t0: float, *, reasons=(), reason=None, outputs=None, details=None,
                error_file=None) -> dict:
        rec = {"status": status, "reason": reason, "reasons": list(reasons), "inputs_hash": self.hashes[stage.name],
               "outputs": outputs or {}, "t_start": t0, "t_end": time.time(), "version": stage.version,
               "dataproc_version": __version__}
        if details:
            rec["details"] = details
        if error_file:
            rec["error_file"] = error_file
        return rec


# --------------------------------------------------------------------------------------------- entry points

def run_process(root: DataRoot, folder: str, episode: int | None = None, force: bool = False,
                slam: str = "orbslam3", backend=None, log: Callable = print) -> ProcessResult:
    """``TaskProcessor(...).run()``. ``backend``: a SLAM backend instance to use instead of ``get_backend(slam)``
    (tests; only the name ``slam`` is part of the inputs hash, not the instance's options)."""
    return TaskProcessor(root, folder, episode, force, slam, backend, log).run()


def cli_process(root: DataRoot, folder: str, episode: int | None = None, force: bool = False,
                slam: str = "orbslam3") -> int:
    """``dataproc process``; returns the exit code. Fatal errors go to stderr, the summary to stdout."""
    try:
        res = run_process(root, folder, episode=episode, force=force, slam=slam, log=print)
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return 1
    if res.report is None and res.error:
        print(f"dataproc process: {res.error}", file=sys.stderr)
    return res.exit_code
